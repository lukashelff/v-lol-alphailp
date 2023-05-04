import argparse
import os
import warnings
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from numpy import arange
from sklearn.metrics import accuracy_score, recall_score, roc_curve

import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from rtpt import RTPT

from michalski_trains.dataset import get_datasets
from nsfr_utils import denormalize_kandinsky, get_data_loader, get_data_pos_loader, get_data_neg_loader, get_prob, \
    get_nsfr_model, update_initial_clauses
from nsfr_utils import save_images_with_captions, to_plot_images_kandinsky, generate_captions
from logic_utils import get_lang, get_searched_clauses
from mode_declaration import get_mode_declarations

from clause_generator import ClauseGenerator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=24, help="Batch size to infer with")
    parser.add_argument("--batch-size-bs", type=int, default=1, help="Batch size in beam search")
    parser.add_argument("--e", type=int, default=6,
                        help="The maximum number of objects in one image")
    parser.add_argument("--dataset", choices=["twopairs", "threepairs", "red-triangle", "closeby",
                                              "online", "online-pair", "nine-circles",
                                              "clevr-hans0", "clevr-hans1", "clevr-hans2",
                                              "theoryx", "complex", "numerical"], help="Use kandinsky patterns dataset")
    parser.add_argument("--dataset-type", default="kandinsky", help="kandinsky, clevr, michalski")
    parser.add_argument('--device', default='cpu',
                        help='cuda device, i.e. 0 or cpu')
    parser.add_argument("--no-cuda", action="store_true",
                        help="Run on CPU instead of GPU (not recommended)")
    parser.add_argument("--small-data", action="store_true", help="Use small training data.")
    parser.add_argument("--no-xil", action="store_true", help="Do not use confounding labels for clevr-hans.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of threads for data loader")
    parser.add_argument('--gamma', default=0.01, type=float,
                        help='Smooth parameter in the softor function')
    parser.add_argument("--plot", action="store_true",
                        help="Plot images with captions.")
    parser.add_argument("--t-beam", type=int, default=4, help="Number of rule expantion of clause generation.")
    parser.add_argument("--n-beam", type=int, default=5, help="The size of the beam.")
    parser.add_argument("--n-max", type=int, default=50, help="The maximum number of clauses.")
    parser.add_argument("--m", type=int, default=1, help="The size of the logic program.")
    parser.add_argument("--n-obj", type=int, default=2, help="The number of objects to be focused.")
    parser.add_argument("--epochs", type=int, default=101, help="The number of epochs.")
    parser.add_argument("--lr", type=float, default=1e-2, help="The learning rate.")
    parser.add_argument("--n-data", type=float, default=200, help="The number of data to be used.")
    parser.add_argument("--pre-searched", action="store_true", help="Using pre searched clauses.")
    args = parser.parse_args()
    return args


# def get_nsfr_model(args, lang, clauses, atoms, bk, bk_clauses, device, train=False):
def discretise_NSFR(NSFR, args, device):
    lark_path = 'src/lark/exp.lark'
    lang_base_path = 'data/lang/'
    lang, clauses_, bk_clauses, bk, atoms = get_lang(
        lark_path, lang_base_path, args.dataset_type, args.dataset)
    # Discretise NSFR rules
    clauses = NSFR.get_clauses()
    return get_nsfr_model(args, lang, clauses, atoms, bk, bk_clauses, device, train=False)


def predict(NSFR, loader, args, device, th=None, split='train'):
    predicted_list = []
    target_list = []
    count = 0
    ###NSFR = discretise_NSFR(NSFR, args, device)
    # NSFR.print_program()

    for i, sample in tqdm(enumerate(loader, start=0)):
        # to cuda
        imgs, target_set = map(lambda x: x.to(device), sample)

        # infer and predict the target probability
        V_T = NSFR(imgs)
        predicted = get_prob(V_T, NSFR, args)
        predicted_list.append(predicted.detach())
        target_list.append(target_set.detach())
        if args.plot:
            imgs = to_plot_images_kandinsky(imgs)
            captions = generate_captions(
                V_T, NSFR.atoms, NSFR.pm.e, th=0.3)
            save_images_with_captions(
                imgs, captions, folder='result/kandinsky/' + args.dataset + '/' + split + '/', img_id_start=count,
                dataset=args.dataset)
        count += V_T.size(0)  # batch size

    predicted = torch.cat(predicted_list, dim=0).detach().cpu().numpy()
    target_set = torch.cat(target_list, dim=0).to(
        torch.int64).detach().cpu().numpy()

    if th == None:
        fpr, tpr, thresholds = roc_curve(target_set, predicted, pos_label=1)
        accuracy_scores = []
        print('ths', thresholds)
        for thresh in thresholds:
            accuracy_scores.append(accuracy_score(
                target_set, [m > thresh for m in predicted]))

        accuracies = np.array(accuracy_scores)
        max_accuracy = accuracies.max()
        max_accuracy_threshold = thresholds[accuracies.argmax()]
        rec_score = recall_score(
            target_set, [m > thresh for m in predicted], average=None)

        print('target_set: ', target_set, target_set.shape)
        print('predicted: ', predicted, predicted.shape)
        print('accuracy: ', max_accuracy)
        print('threshold: ', max_accuracy_threshold)
        print('recall: ', rec_score)

        return max_accuracy, rec_score, max_accuracy_threshold
    else:
        accuracy = accuracy_score(target_set, [m > th for m in predicted])
        rec_score = recall_score(
            target_set, [m > th for m in predicted], average=None)
        return accuracy, rec_score, th


def setup_ds(full_ds, tr_idx=None, val_idx=None, test_ds=None, batch_size=10, num_worker=4, shuffle=True):
    if len(tr_idx) > 0:
        set_up_txt = f'split ds into training ds with {len(tr_idx)} images and validation ds with {len(val_idx)} images'
        if test_ds is not None:
            set_up_txt += f' and test ds with {test_ds.__len__()} images and train length of {test_ds.min_car}'
        print(set_up_txt)

    labels = [full_ds[i][1] for i in range(len(full_ds))]
    pos_tr_idx, neg_tr_idx = [], []
    pos_val_idx, neg_val_idx = [], []
    pos_test_idx, neg_test_idx = [], []
    for i in tr_idx:
        if labels[i] == 1:
            pos_tr_idx.append(i)
        else:
            neg_tr_idx.append(i)
    for i in val_idx:
        if labels[i] == 1:
            pos_val_idx.append(i)
        else:
            neg_val_idx.append(i)
    if test_ds is not None:
        labels_test = [test_ds[i][1] for i in range(len(test_ds))]
        for i in labels_test:
            if labels[i] == 1:
                pos_test_idx.append(i)
            else:
                neg_test_idx.append(i)

    ds = {
        'train': Subset(full_ds, tr_idx),
        'pos_train': Subset(test_ds, pos_tr_idx),
        'neg_train': Subset(test_ds, neg_tr_idx),
        'val': Subset(full_ds, val_idx),
        'pos_val': Subset(full_ds, pos_val_idx),
        'neg_val': Subset(full_ds, neg_val_idx),
        'test': test_ds,
        'pos_test': Subset(test_ds, pos_test_idx) if test_ds is not None else test_ds,
        'neg_test': Subset(test_ds, neg_test_idx) if test_ds is not None else test_ds
    }
    dl = {'train': DataLoader(ds['train'], batch_size=batch_size, num_workers=num_worker, shuffle=shuffle),
          'pos_train': DataLoader(ds['pos_train'], batch_size=batch_size, num_workers=num_worker, shuffle=shuffle),
          'neg_train': DataLoader(ds['neg_train'], batch_size=batch_size, num_workers=num_worker, shuffle=shuffle),
          'val': DataLoader(ds['val'], batch_size=batch_size, num_workers=num_worker),
          'pos_val': DataLoader(ds['pos_val'], batch_size=batch_size, num_workers=num_worker),
          'neg_val': DataLoader(ds['neg_val'], batch_size=batch_size, num_workers=num_worker),
          'test': DataLoader(ds['test'], batch_size=batch_size,
                             num_workers=num_worker) if test_ds is not None else None,
          'pos_test': DataLoader(ds['pos_test'], batch_size=batch_size,
                                 num_workers=num_worker) if test_ds is not None else None,
          'neg_test': DataLoader(ds['neg_test'], batch_size=batch_size,
                                 num_workers=num_worker) if test_ds is not None else None}
    return ds, dl


def train_nsfr(args, NSFR, optimizer, train_loader, val_loader, test_loader, device, writer, rtpt):
    bce = torch.nn.BCELoss()
    loss_list = []
    for epoch in range(args.epochs):
        loss_i = 0
        for i, sample in tqdm(enumerate(train_loader, start=0)):
            # to cuda
            imgs, target_set = map(lambda x: x.to(device), sample)

            # infer and predict the target probability
            V_T = NSFR(imgs)
            ##NSFR.print_valuation_batch(V_T)
            predicted = get_prob(V_T, NSFR, args)
            # raise Exception(f'predicted with size: {predicted.size()}, target with size: {target_set.size()}')
            # print(predicted.size(), target_set.size())
            # print(f'predicted: {predicted}, target: {target_set}')
            loss = bce(predicted.unsqueeze(1).to(torch.float), target_set.to(torch.float))
            loss_i += loss.item()
            loss.backward()
            optimizer.step()

            # if i % 20 == 0:
            #    NSFR.print_valuation_batch(V_T)
            #    print("predicted: ", np.round(predicted.detach().cpu().numpy(), 2))
            #    print("target: ", target_set.detach().cpu().numpy())
            #    NSFR.print_program()
            #    print("loss: ", loss.item())

            # print("Predicting on validation data set...")
            # acc_val, rec_val, th_val = predict(
            #    NSFR, val_loader, args, device, writer, th=0.33, split='val')
            # print("val acc: ", acc_val, "threashold: ", th_val, "recall: ", rec_val)
        loss_list.append(loss_i)
        rtpt.step(subtitle=f"loss={loss_i:2.2f}")
        writer.add_scalar("metric/train_loss", loss_i, global_step=epoch)
        print("loss: ", loss_i)
        # NSFR.print_program()sdfsdf
        if epoch % 20 == 0:
            NSFR.print_program()
            print("Predicting on validation data set...")
            acc_val, rec_val, th_val = predict(NSFR, val_loader, args, device, th=0.33, split='val')
            writer.add_scalar("metric/val_acc", acc_val, global_step=epoch)
            print("acc_val: ", acc_val)

            print("Predicting on training data set...")
            acc, rec, th = predict(NSFR, train_loader, args, device, th=th_val, split='train')
            writer.add_scalar("metric/train_acc", acc, global_step=epoch)
            print("acc_train: ", acc)

            if test_loader is not None:
                print("Predicting on test data set...")
                acc, rec, th = predict(NSFR, test_loader, args, device, th=th_val, split='train')
                writer.add_scalar("metric/test_acc", acc, global_step=epoch)
                print("acc_test: ", acc)

    return loss


def cross_validation(ds_path: str, label_noise: list, image_noise: list, rules: list, visualizations: list,
                     scenes: list,
                     car_length: list, train_size: list, n_splits=5,
                     replace=False, start_it=0, batch_size=10, raw_trains='MichalskiTrains',
                     ds_size=12000, resize=False):
    '''
    Run cross validation on the given dataset with the given parameters. The cross validation is done on the train set

    Args:
        ds_path: str, path to the dataset
        label_noise: list, list of label noise to add to the dataset
        image_noise: list, list of image noise to add to the dataset
        rules: list, list of rules to add to the dataset
        visualizations: list, list of visualizations to add to the dataset
        rules:
        visualizations:
        scenes:
        car_length:
        train_size:
        n_splits:
        replace:
        start_it:
        batch_size:
        raw_trains:
        ds_size:
        resize:

    Returns:

    '''
    random_state = 0
    test_size = 2000
    tr_it, tr_b = 0, 0
    n_batches = sum(train_size) // batch_size
    tr_b_total = n_splits * n_batches * len(label_noise) * len(image_noise) * len(rules) * len(
        visualizations) * len(scenes) * len(car_length)
    tr_it_total = n_splits * len(train_size) * len(label_noise) * len(image_noise) * len(rules) * len(
        visualizations) * len(scenes) * len(car_length)
    output_dir = f'runs/alphailp_michalski/stats/'
    os.makedirs(output_dir, exist_ok=True)

    for label_noise, image_noise, class_rule, train_vis, base_scene in product(label_noise, image_noise, rules, visualizations, scenes):
        full_ds = get_datasets(base_scene, raw_trains, train_vis, class_rule=class_rule, ds_size=ds_size, max_car=4,
                               min_car=2, label_noise=label_noise, image_noise=image_noise, ds_path=ds_path,
                               resize=resize)
        try:
            if (7, 7) in car_length:
                test_ds = get_datasets(base_scene, raw_trains, train_vis, class_rule=class_rule,
                                       ds_size=2000, max_car=7, min_car=7,
                                       label_noise=label_noise, image_noise=image_noise,
                                       ds_path=ds_path, resize=resize)
                print(f'Using test dataset with train length of 7 for test evaluation')
            else:
                raise Exception(f'No test dataset with train length of 7 for test evaluation found skipping test evaluation')
        except:
            print(f'No test dataset found or selected. Skipping test evaluation with train length of 7 ')
            test_ds = None
        for t_size in train_size:
            full_ds.predictions_im_count = t_size
            cv = StratifiedShuffleSplit(train_size=t_size, test_size=test_size, random_state=random_state,
                                        n_splits=n_splits)
            y = np.concatenate([full_ds.get_direction(item) for item in range(full_ds.__len__())])
            settings = f'{train_vis}_{class_rule}_{t_size}samples_inoise_{image_noise}_lnoise_{label_noise}'
            for fold, (tr_idx, val_idx) in enumerate(cv.split(np.zeros(len(y)), y)):
                o_path = f'{output_dir}{settings}/fold_{fold}.csv'
                if tr_it >= start_it and (not os.path.isfile(o_path) or replace):
                    print('====' * 10)
                    print(f'training iteration {tr_it + 1}/{tr_it_total} with {t_size // batch_size} '
                          f'training batches, already completed: {tr_b}/{tr_b_total} batches. ')
                    ds, dl = setup_ds(full_ds, tr_idx, val_idx, test_ds,
                                      batch_size=batch_size, shuffle=True)
                    ex_it = f'aILP:Michalski_it({tr_it}/{tr_it_total})_batch({tr_b}/{tr_b_total})'
                    ex_it = f'aILP:M_{class_rule[0]}_it({tr_it}/{tr_it_total})'
                    ex_it = f'aILP:M_{class_rule[0]}'
                    setting = f'aILP:Michalski_{settings}_fold_{fold}'
                    remaining_epochs = args.epochs * (tr_it_total - tr_it)
                    stats = train(dl, ex_it, setting, remaining_epochs)
                    frame = [['aILP', t_size, class_rule, train_vis, base_scene, fold, label_noise, image_noise,
                              stats['theory'],
                              stats['train_acc'], stats['val_acc'], stats['test_acc'],
                              stats['train_rec'], stats['val_rec'], stats['test_rec'],
                              stats['train_th'], stats['val_th'], stats['test_th']
                              ]]
                    data = pd.DataFrame(frame, columns=['Methods', 'training samples', 'rule', 'visualization', 'scene',
                                                        'cv iteration', 'label noise', 'image noise',
                                                        'theory',
                                                        'Validation acc', 'Train acc', 'Generalization acc',
                                                        'Validation rec', 'Train rec', 'Generalization rec',
                                                        'Validation th', 'Train th', 'Generalization th']
                                        )
                    os.makedirs(os.path.dirname(o_path), exist_ok=True)
                    data.to_csv(o_path)

                tr_b += t_size // batch_size
                tr_it += 1


def train(dl, ex_it, setting, remaining_epochs):
    # torch.autograd.set_detect_anomaly(True)
    args = get_args()

    print('args ', args)
    if args.no_cuda:
        device = torch.device('cpu')
    elif len(args.device.split(',')) > 1:
        # multi gpu
        device = torch.device('cuda')
    else:
        device = torch.device('cuda:' + args.device)
    print('device: ', device)
    # run_name = 'predict/' + args.dataset
    writer = SummaryWriter(f"runs/{setting}", purge_step=0)

    # Create RTPT object
    rtpt = RTPT(name_initials='LH', experiment_name=ex_it, max_iterations=remaining_epochs)
    # Start the RTPT tracking
    rtpt.start()
    train_loader, val_loader, test_loader = dl['train'], dl['val'], dl['test']
    train_pos_loader, val_pos_loader, test_pos_loader = dl['pos_train'], dl['pos_val'], dl['pos_test']
    train_neg_loader, val_neg_loader, test_neg_loader = dl['neg_train'], dl['neg_val'], dl['neg_test']

    # load logical representations
    lark_path = 'src/lark/exp.lark'
    lang_base_path = 'data/lang/'
    lang, clauses, bk_clauses, bk, atoms = get_lang(
        lark_path, lang_base_path, args.dataset_type, args.dataset)
    clauses = update_initial_clauses(clauses, args.n_obj)
    print("clauses: ", clauses)

    # Neuro-Symbolic Forward Reasoner for clause generation
    NSFR_cgen = get_nsfr_model(args, lang, clauses, atoms, bk, bk_clauses, device=device)  # torch.device('cpu'))
    mode_declarations = get_mode_declarations(args, lang, args.n_obj)
    cgen = ClauseGenerator(args, NSFR_cgen, lang, val_pos_loader, val_neg_loader, mode_declarations, bk_clauses,
                           device=device, no_xil=args.no_xil)  # torch.device('cpu'))
    # generate clauses
    if args.pre_searched:
        clauses = get_searched_clauses(lark_path, lang_base_path, args.dataset_type, args.dataset)
    else:
        clauses = cgen.generate(clauses, T_beam=args.t_beam, N_beam=args.n_beam, N_max=args.n_max)
    print("====== ", len(clauses), " clauses are generated!! ======")
    # update
    NSFR = get_nsfr_model(args, lang, clauses, atoms, bk, bk_clauses, device, train=True)
    # inferred_clauses = NSFR.clauses


    params = NSFR.get_params()
    optimizer = torch.optim.RMSprop(params, lr=args.lr)
    ##optimizer = torch.optim.Adam(params, lr=args.lr)

    loss_list = train_nsfr(args, NSFR, optimizer, train_loader, val_loader, test_loader, device, writer, rtpt)

    # validation split
    print("Predicting on validation data set...")
    acc_val, rec_val, th_val = predict(
        NSFR, val_loader, args, device, th=0.33, split='val')

    print("Predicting on training data set...")
    # training split
    acc, rec, th = predict(
        NSFR, train_loader, args, device, th=th_val, split='train')

    acc_test, rec_test, th_test = None, None, None
    if test_loader is not None:
        print("Predicting on test data set...")
        # test split
        acc_test, rec_test, th_test = predict(
            NSFR, test_loader, args, device, th=th_val, split='test')
    print("training acc: ", acc, "threashold: ", th, "recall: ", rec)
    print("val acc: ", acc_val, "threashold: ", th_val, "recall: ", rec_val)
    print("test acc: ", acc_test, "threashold: ", th_test, "recall: ", rec_test)
    NSFR.print_program()
    prog = NSFR.get_program()
    stats = {'train_acc': acc, 'val_acc': acc_val, 'test_acc': acc_test,
             'train_rec': rec, 'val_rec': rec_val, 'test_rec': rec_test,
             'train_th': th, 'val_th': th_val, 'test_th': th_test,
             'theory': prog}
    return stats


if __name__ == "__main__":
    '''
    docker run command:
    docker run --gpus device=12 --shm-size='20gb' --memory="700g" -v $(pwd)/alphailp:/NSFR -v $(pwd)/MichalskiTrainProblem/TrainGenerator/output/image_generator:/NSFR/data/michalski/all alpha-ilp python3 src/train_michalski.py --dataset-type michalski --dataset theoryx --batch-size 10 --n-beam 50 --t-beam 5 --m 2 --device 0
    '''

    # get arguments
    args = get_args()
    ds_path_local = f'{Path.home()}/Documents/projects/MichalskiTrainProblem/TrainGenerator/output/image_generator'
    ds_path_mac = f'{Path.home()}/Documents/projects/Michalski/Neuro-Symbolic-Relational-Learner/TrainGenerator/output/image_generator'
    ds_path_remote = 'data/michalski/all'
    if args.no_cuda:
        ds_path = ds_path_mac
    else:
        ds_path = ds_path_remote if torch.cuda.get_device_properties(0).total_memory > 8352890880 else ds_path_local
    scenes = ['base_scene']
    n_splits = 1
    batch_size = args.batch_size
    raw_trains = 'MichalskiTrains'
    ds_size = 12000
    resize = False

    label_noise = [0, .1, .3][:1]
    image_noise = [0, .1, .3][:1]
    # rules = ['theoryx', 'numerical', 'complex'][2:]
    rules = [args.dataset]
    visualizations = ['Trains', 'SimpleObjects'][:1]
    car_length = [(2, 4), (7, 7)][:1]
    train_size = [100, 1000, 10000][:1]
    replace = True
    start_it = 0
    cross_validation(ds_path, label_noise, image_noise, rules, visualizations, scenes, car_length, train_size, n_splits,
                     replace, start_it, batch_size)
