import glob
from datetime import datetime
import json
import os
import shutil

import jsonpickle
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def load_images_and_labels(dataset='theoryx', split='train'):
    """Load image paths and labels for clevr-hans dataset.
    """
    image_paths = []
    labels = []
    folder = 'data/michalski/' + dataset + '/' + split + '/'
    true_folder = folder + 'true/'
    false_folder = folder + 'false/'

    filenames = sorted(os.listdir(true_folder))
    for filename in filenames:
        if filename != '.DS_Store':
            image_paths.append(os.path.join(true_folder, filename))
            labels.append(1)

    filenames = sorted(os.listdir(false_folder))
    for filename in filenames:
        if filename != '.DS_Store':
            image_paths.append(os.path.join(false_folder, filename))
            labels.append(0)
    return image_paths, labels


def load_images_and_labels_positive(dataset='theoryx', split='train'):
    """Load image paths and labels for clevr-hans dataset.
    """
    image_paths = []
    labels = []
    folder = 'data/michalski/' + dataset + '/' + split + '/'
    true_folder = folder + 'true/'
    false_folder = folder + 'false/'

    filenames = sorted(os.listdir(true_folder))[:500]
    # n = 500  # int(len(filenames)/10)
    # filenames = random.sample(filenames, n)
    for filename in filenames:
        if filename != '.DS_Store':
            image_paths.append(os.path.join(true_folder, filename))
            labels.append(1)
    return image_paths, labels


def __load_images_and_labels(dataset='theoryx', split='train', base=None):
    """Load image paths and labels for clevr-hans dataset.
    """
    image_paths = []
    labels = []
    if base == None:
        base_folder = 'data/michalski/' + dataset + '/' + split + '/'
    else:
        base_folder = base + '/data/michalski/' + dataset + '/' + split + '/'
    if dataset == 'clevr-hans3':
        for i, cl in enumerate(['class0', 'class1', 'class2']):
            folder = base_folder + cl + '/'
            filenames = sorted(os.listdir(folder))
            for filename in filenames:
                if filename != '.DS_Store':
                    image_paths.append(os.path.join(folder, filename))
                    labels.append(i)
    elif dataset == 'clevr-hans7':
        for i, cl in enumerate(['class0', 'class1', 'class2', 'class3', 'class4', 'class5', 'class6']):
            folder = base_folder + cl + '/'
            filenames = sorted(os.listdir(folder))
            for filename in filenames:
                if filename != '.DS_Store':
                    image_paths.append(os.path.join(folder, filename))
                    labels.append(i)
    return image_paths, labels


class MICHALSKI(torch.utils.data.Dataset):
    """three-dimensional Michalski train dataset.
    """

    def __init__(self, dataset, split, img_size=128, base=None, small_data=False):
        super().__init__()
        self.img_size = img_size
        self.small_data = small_data
        self.dataset = dataset
        assert split in {
            "train",
            "val",
            "test",
        }  # note: test isn't very useful since it doesn't have ground-truth scene information
        self.split = split
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize((img_size, img_size)),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
             ])
        self.image_paths, self.labels = load_images_and_labels(
            dataset=dataset, split=split)

    def __getitem__(self, item):
        path = self.image_paths[item]
        image = Image.open(path).convert("RGB")
        # image = transforms.ToTensor()(image)[:3, :, :]
        image = self.transform(image)
        # image = (image - 0.5) * 2.0  # Rescale to [-1, 1].
        label = torch.tensor(self.labels[item], dtype=torch.float32)
        return image, label

    def __len__(self):
        return len(self.labels)


class MICHALSKI_POSITIVE(torch.utils.data.Dataset):
    """three-dimensional Michalski train dataset.
    """

    def __init__(self, dataset, split, img_size=128, base=None, small_data=False):
        super().__init__()
        self.img_size = img_size
        self.small_data = small_data
        self.dataset = dataset
        assert split in {
            "train",
            "val",
            "test",
        }  # note: test isn't very useful since it doesn't have ground-truth scene information
        self.split = split
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize((img_size, img_size)),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]), ]
        )
        self.image_paths, self.labels = load_images_and_labels_positive(
            dataset=dataset, split=split)

    def __getitem__(self, item):
        path = self.image_paths[item]
        image = Image.open(path).convert("RGB")
        # image = transforms.ToTensor()(image)[:3, :, :]
        image = self.transform(image)
        # image = (image - 0.5) * 2.0  # Rescale to [-1, 1].
        label = torch.tensor(self.labels[item], dtype=torch.float32)
        return image, label

    def __len__(self):
        return len(self.labels)


class MichalskiTrainDataset(Dataset):
    def __init__(self, class_rule, base_scene, raw_trains, train_vis, train_count=10000, img_size=None,
                 ds_path='output/image_generator',
                 ):
        """ MichalskiTrainDataset
            Args:
                val: bool if model is used for vaildation
                resize: bool if true images are resized to 224x224
                :param:  raw_trains (string): typ of train descriptions 'RandomTrains' or 'MichalskiTrains'
                :param:  train_vis (string): visualization of the train description either 'MichalskiTrains' or
                'SimpleObjects'
            @return:
                X_val: X value output for training data returned in __getitem__()
                ['image', 'predicted_attributes', 'gt_attributes', 'gt_attributes_individual_class', 'predicted_mask', gt_mask]
                        image (torch): image of michalski train

                y_val: ['direction','attribute','mask'] y label output for training data returned in __getitem__()

            """
        self.images = []
        self.trains = []
        self.masks = []
        self.img_size = img_size
        self.train_count = train_count
        ds_typ = f'{train_vis}_{class_rule}_{raw_trains}_{base_scene}'
        self.base_scene = base_scene
        self.image_base_path = f'{ds_path}/{ds_typ}/images'
        self.all_scenes_path = f'{ds_path}/{ds_typ}/all_scenes'

        self.labels = ['direction']
        self.label_classes = ['west', 'east']
        self.class_dim = len(self.label_classes)
        self.output_dim = len(self.labels)
        # train with class specific labels
        if not os.path.isfile(self.all_scenes_path + '/all_scenes.json'):
            raise AssertionError('json scene file missing. Not all images were generated')
        if len(os.listdir(self.image_base_path)) < self.train_count:
            raise AssertionError(f'Missing images in dataset. Expected size {self.train_count}.'
                                 f'Available images: {len(os.listdir(self.image_base_path))}')

        path = self.all_scenes_path + '/all_scenes.json'
        with open(path, 'r') as f:
            all_scenes = json.load(f)
            for scene in all_scenes['scenes'][:train_count]:
                self.images.append(scene['image_filename'])
                # self.depths.append(scene['depth_map_filename'])
                train = jsonpickle.decode(scene['m_train'])
                self.trains.append(train)
                self.masks.append(scene['car_masks'])

        trans = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
        if img_size is not None:
            print('resize true')
            trans.append(transforms.Resize((self.img_size, self.img_size), interpolation=InterpolationMode.BICUBIC))
        self.norm = transforms.Compose(trans)
        self.normalize_mask = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])

    def __getitem__(self, item):
        image = self.get_pil_image(item)
        X = self.norm(image)
        y = self.get_direction(item)
        return X, y

    def __len__(self):
        return self.train_count

    def get_direction(self, item):
        lab = self.trains[item].get_label()
        if lab == 'none':
            # return torch.tensor(0).unsqueeze(dim=0)
            raise AssertionError(f'There is no direction label for a RandomTrains. Use MichalskiTrain DS.')
        label_binary = self.label_classes.index(lab)
        label = torch.tensor(label_binary).unsqueeze(dim=0)
        return label

    def get_m_train(self, item):
        return self.trains[item]

    def get_mask(self, item):
        return self.masks[item]

    def get_pil_image(self, item):
        im_path = self.get_image_path(item)
        return Image.open(im_path).convert('RGB')

    def get_image_path(self, item):
        return self.image_base_path + '/' + self.images[item]

    def get_label_for_id(self, item):
        return self.trains[item].get_label()

    def get_trains(self):
        return self.trains

    def get_ds_labels(self):
        return self.labels


def get_datasets(base_scene, raw_trains, train_vis, ds_size, ds_path, class_rule='theoryx', resize=False):
    path_ori = f'{ds_path}/{train_vis}_{class_rule}_{raw_trains}_{base_scene}'
    if not os.path.isfile(path_ori + '/all_scenes/all_scenes.json'):
        combine_json(base_scene, raw_trains, train_vis, class_rule, ds_size=ds_size)
        raise Warning(f'Dataloader did not find JSON ground truth information.'
                      f'Might be caused by interruptions during process of image generation.'
                      f'Generating new JSON file at: {path_ori + "/all_scenes/all_scenes.json"}')
    im_path = path_ori + '/images'
    if not os.path.isdir(im_path):
        raise AssertionError('dataset not found, please generate images first')

    files = os.listdir(im_path)
    # total image count equals 10.000 adjust if not all images need to be generated
    if len(files) < ds_size:
        raise AssertionError(
            f'not enough images in dataset: expected {ds_size}, present: {len(files)}'
            f' please generate the missing images')
    elif len(files) > ds_size:
        raise Warning(
            f' dataloader did not select all images of the dataset, number of selected images:  {ds_size},'
            f' available images in dataset: {len(files)}')

    # merge json files to one if it does not already exist
    if not os.path.isfile(path_ori + '/all_scenes/all_scenes.json'):
        raise AssertionError(
            f'no JSON found')
    # image_count = None for standard image count
    full_ds = MichalskiTrainDataset(class_rule=class_rule, base_scene=base_scene, raw_trains=raw_trains,
                                    train_vis=train_vis,
                                    train_count=ds_size, resize=resize, ds_path=ds_path)
    return full_ds


def merge_json_files(path):
    """
    merging all ground truth json files of the dataset
    :param:  path (string)        : path to the dataset information
    """
    all_scenes = []
    for p in glob.glob(path + '/scenes/*_m_train.json'):
        with open(p, 'r') as f:
            all_scenes.append(json.load(f))
    output = {
        'info': {
            'date': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            'version': '0.1',
            'license': None,
        },
        'scenes': all_scenes
    }
    json_pth = path + '/all_scenes/all_scenes.json'
    os.makedirs(path + '/all_scenes/', exist_ok=True)
    # args.output_scene_file.split('.json')[0]+'_classid_'+str(args.img_class_id)+'.json'
    with open(json_pth, 'w+') as f:
        json.dump(output, f, indent=2)


def combine_json(base_scene, raw_trains, train_vis, class_rule, out_dir='output/image_generator', ds_size=10000):
    path_settings = f'{train_vis}_{class_rule}_{raw_trains}_{base_scene}'
    path_ori = f'tmp/image_generator/{path_settings}'
    path_dest = f'{out_dir}/{path_settings}'
    im_path = path_ori + '/images'
    if os.path.isdir(im_path):
        files = os.listdir(im_path)
        if len(files) == ds_size:
            merge_json_files(path_ori)
            shutil.rmtree(path_ori + '/scenes')
            try:
                shutil.rmtree(path_dest)
            except:
                pass
            shutil.move(path_ori, path_dest)
