import torch
import torch.nn as nn
from torchvision import models

from rcnn.model.mask_rcnn import multi_label_maskrcnn_resnet50_fpn_v2
from rcnn.template_matcher import prediction_to_symbolic_v2


class MichalskiPerceptionModuleResNet(nn.Module):
    """A perception module using Michalski.

    Attrs:
        e (int): The maximum number of entities.
        d (int): The dimension of the object-centric vector.
        device (device): The device where the model and tensors are loaded.
        train (bool): The flag if the parameters are trained.
        preprocess (tensor->tensor): Reshape the yolo output into the unified format of the perceptiom module.
    """

    # shape output, 5 different shaps + absence of car
    # length output, 2 different lengths + absence of car
    # wall output, 2 different walls + absence of car represented as index 0
    # roof output, 4 different roof shapes + absence of car
    # wheels output, 2 different wheel counts + absence of car
    # load number output, max 3 payloads min 0
    # load shape output, 6 different shape + absence of car
    # if dim_out == 28:
    #     self.label_num_classes = [6, 3, 3, 5, 3, 4, 7]
    # else:
    # all labels can obtain all classes

    def __init__(self, device, e=4, d=32, train=False):
        super().__init__()
        self.e = e  # num of entities
        self.d = d  # num of dimension
        self.device = device
        model = models.resnet18()
        self.model = MultiLabelNN(backbone=model)
        weights = torch.load(f='src/weights/michalski/resnet/model.pth', map_location=device)
        self.model.load_state_dict(weights['model_state_dict'])
        self.model.eval()
        self.to(device)
        self.preprocess = MichalskiPreprocess(device)

    def forward(self, x):
        activations = self.model(x)
        post = self.preprocess(activations)

        # print activations and image to check if the model is working
        # print_pred(activations)
        # print_proccessed(post)
        # show_torch_im(x)
        # raise Exception('stop')

        return post


class MichalskiPerceptionModuleRCNN(nn.Module):
    """A perception module using Michalski.

    Attrs:
        e (int): The maximum number of entities.
        d (int): The dimension of the object-centric vector.
        device (device): The device where the model and tensors are loaded.
        train (bool): The flag if the parameters are trained.
        preprocess (tensor->tensor): Reshape the yolo output into the unified format of the perceptiom module.
    """

    # shape output, 5 different shaps + absence of car
    # length output, 2 different lengths + absence of car
    # wall output, 2 different walls + absence of car represented as index 0
    # roof output, 4 different roof shapes + absence of car
    # wheels output, 2 different wheel counts + absence of car
    # load number output, max 3 payloads min 0
    # load shape output, 6 different shape + absence of car
    # if dim_out == 28:
    #     self.label_num_classes = [6, 3, 3, 5, 3, 4, 7]
    # else:
    # all labels can obtain all classes

    def __init__(self, device, e=4, d=32, train=False):
        super().__init__()
        self.e = e  # num of entities
        self.d = d  # num of dimension
        self.device = device
        checkpoint = torch.load(f='src/weights/michalski/rcnn/Trains/model.pth', map_location=device)
        rcnn_labels_per_segment = 3
        self.model = multi_label_maskrcnn_resnet50_fpn_v2(weights=None,
                                                          image_mean=[0.485, 0.456, 0.406],
                                                          image_std=[0.229, 0.224, 0.225],
                                                          # num_classes=22 + 20,
                                                          num_labels=rcnn_labels_per_segment,
                                                          rpn_batch_size_per_image=256,
                                                          box_nms_thresh=0.8,
                                                          # box_score_thresh=0.9
                                                          )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.to(device)
        self.preprocess = MichalskiPreprocess(device)

    def forward(self, x):
        activations = self.model(x)
        activations = [{k: v.to(self.device) for k, v in t.items()} for t in activations]
        concept_padded = torch.zeros((len(activations), 32, 22))
        for i, activation in enumerate(activations):
            concept, issues = prediction_to_symbolic_v2(activation)
            # concept to one hot
            one_hot = torch.nn.functional.one_hot(concept.to(torch.int64), num_classes=22)
            concept_padded[i, :one_hot.shape[0], :] = one_hot

        post = self.preprocess(concept_padded)

        # print activations and image to check if the model is working
        # print_pred(activations)
        # print_proccessed(post)
        # show_torch_im(x)
        # raise Exception('stop')

        return post


class MultiLabelNN(nn.Module):
    """A perception module for Michalski-3D.

    Attrs:
        device (device): The device where the model and tensors are loaded.
        pth (str): The path of the pretrained model.
    Returns:
        Z (tensor): The attribute representation of the train Z of size (batch_size, e, d).
                    e=32 (4*8), number of cars = 4 and numer attributes for each car = 8.
                            Attributes: [color, length, wall, roof, wheels, load1, load2, load3].
                    d=22, number of classes for each attribute. The format is:
                            [none (0), yellow (1), green(2), grey(3), red(4), blue(5),
                            short(6), long(7), braced_wall(8), solid_wall(9),
                            roof_foundation(10), solid_roof(11), braced_roof(12), peaked_roof(13),2_wheels(14), 3_wheels(15),
                            box(16), golden_vase(17), barrel(18), diamond(19), metal_pot(20), oval_vase(21)].
    """

    def __init__(self, backbone):
        super(MultiLabelNN, self).__init__()
        layers = list(backbone.children())[:9]
        self.fc = backbone.fc
        self.features1 = nn.Sequential(*layers[:6])
        self.features2 = nn.Sequential(*layers[6:])
        self.bb = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))
        self.classifier = nn.ModuleList()
        self.label_num_classes = [22] * 8
        all_classes = sum(self.label_num_classes)
        in_features = backbone.inplanes
        for _ in range(4):
            self.classifier.append(nn.Sequential(nn.Linear(in_features=in_features, out_features=all_classes)))

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = torch.flatten(x, 1)
        class_output = [classifier(x) for classifier in self.classifier]
        activations = torch.cat(class_output, dim=1).view(-1, 32, 22)
        return activations


class MichalskiPreprocess(nn.Module):
    """A perception module using Slot Attention.

    Attrs:
        device (device): The device where the model to be loaded.
        img_size (int): The size of the (resized) image to normalize the xy-coordinates.
        classes (list(str)): The classes of objects.
        colors (tensor(int)): The one-hot encodings of the colors (repeated 3 times).
        shapes (tensor(int)): The one-hot encodings of the shapes (repeated 3 times).
    """

    def __init__(self, device, img_size=128, normalize='softmax'):
        super().__init__()
        self.device = device
        self.img_size = img_size
        self.car_nums = ['1', '2', '3', '4']
        self.colors = ["yellow", "green", "grey", "red", "blue"]
        self.lengths = ["short", "long"]
        self.walls = ["full", "braced"]
        self.roofs = ["none", "foundation", "solid_roof", "braced_roof", "peaked_roof"]
        self.wheels = ['2', '3']
        self.loads = ["blue_box", "golden_vase", "barrel", "diamond", "metal_box"]
        self.load_nums = ['0', '1', '2', '3']
        self.normalize = normalize

    def forward(self, x):
        """A preprocess funciton for the YOLO model. The format is: [x1, y1, x2, y2, prob, class].

        Args:
            x (tensor): The attribute representation of the train Z of size (batch_size, e, d).
                        e=32 (4*8), number of cars = 4 and numer attributes for each car = 8.
                            Attributes: [color, length, wall, roof, wheels, load1, load2, load3].
                        d=22, number of classes for each attribute. The format is:
                            [none, yellow, green, grey, red, blue,
                            short, long, braced_wall, solid_wall,
                            roof_foundation, solid_roof, braced_roof, peaked_roof,
                            2_wheels, 3_wheels, box, golden_vase, barrel, diamond, metal_pot, oval_vase].
        Returns:
            Z (tensor): The preprocessed object-centric representation Z of the cars, size (batch_size, e, d).
                    e=4 (number of cars),
                    d=42 (1+4+5+2+2+5+2+7+7+7) symbolic representation of each car
                        [obj_prob(1) + car_number(4) + color(5) + length(2) + wall(2) + roof(5) + wheels(2) + load1(7) +
                         load2(7) + load3(7)].
                        The format is: [objectness, 1, 2, 3, 4, yellow, green, grey, red, blue,
                                        short, long, braced_wall, solid_wall,
                                        no_roof, roof_foundation, solid_roof, braced_roof, peaked_roof,
                                        2_wheels, 3_wheels,
                                        no_load1, box1, golden_vase1, barrel1, diamond1, metal_pot1, oval_vase1,
                                        no_load2, box2, golden_vase2, barrel2, diamond2, metal_pot2, oval_vase2,
                                        no_load3, box3, golden_vase3, barrel3, diamond3, metal_pot3, oval_vase3].
                        The probability for each attribute is obtained by copying the probability of the classification of the perception model.
        """
        # shift min to 0
        soft = nn.Softmax(dim=-1)
        softmin = nn.Softmin(dim=-1)
        batch, e, d = x.size()
        cars = int(e / 8)
        train = torch.zeros(int(batch), cars, 42).to(self.device)
        for i in range(cars):
            # preprocess to object-centric car representation with accuracies for each attribute
            # Attributes: [prob, pos, color, length, wall, roof, wheels, load1, load2, load3].
            #           d=42 (1+4+5+2+2+5+2+7+7+7)

            # car number
            train[:, i, i + 1] = 1
            # # roof
            # train[:, i, 14] = x[:, 3 + 8 * i, 0]
            # train[:, i, 15:19] = x[:, 3 + 8 * i, 10:14]
            # # load 1
            # train[:, i, 21] = x[:, 5 + 8 * i, 0]
            # train[:, i, 22:28] = x[:, 5 + 8 * i, 16:22]
            # # load 2
            # train[:, i, 28] = x[:, 6 + 8 * i, 0]
            # train[:, i, 29:35] = x[:, 6 + 8 * i, 16:22]
            # # load 3
            # train[:, i, 35] = x[:, 7 + 8 * i, 0]
            # train[:, i, 36:42] = x[:, 7 + 8 * i, 16:22]
            if self.normalize == 'softmax':
                # color
                train[:, i, 0] = 1 - soft(x[:, 8 * i, 0:6])[:, 0]
                # color
                train[:, i, 5:10] = soft(x[:, 8 * i, 1:6])
                # length
                train[:, i, 10:12] = soft(x[:, 1 + 8 * i, 6:8])
                # wall
                train[:, i, 12:14] = soft(x[:, 2 + 8 * i, 8:10])
                # roof
                train[:, i, 14:19] = soft(torch.cat([x[:, 3 + 8 * i, 0:1], x[:, 3 + 8 * i, 10:14]], dim=-1))
                # wheel count
                train[:, i, 19:21] = soft(x[:, 4 + 8 * i, 14:16])
                # load 1
                train[:, i, 21:28] = soft(torch.cat([x[:, 5 + 8 * i, 0:1], x[:, 5 + 8 * i, 16:22]], dim=-1))
                # load 2
                train[:, i, 28:35] = soft(torch.cat([x[:, 6 + 8 * i, 0:1], x[:, 6 + 8 * i, 16:22]], dim=-1))
                # load 3
                train[:, i, 35:42] = soft(torch.cat([x[:, 7 + 8 * i, 0:1], x[:, 7 + 8 * i, 16:22]], dim=-1))
            elif self.normalize == 'norm':
                train[:, i, 0] = softmin(x[:, 8 * i, 0:6])[:, 0]
                train[:, i, 5:10] = shift_normalize(x[:, 8 * i, 1:6])
                train[:, i, 10:12] = shift_normalize(x[:, 1 + 8 * i, 6:8])
                train[:, i, 12:14] = shift_normalize(x[:, 2 + 8 * i, 8:10])
                train[:, i, 14:19] = shift_normalize(torch.cat([x[:, 3 + 8 * i, 0:1], x[:, 3 + 8 * i, 10:14]], dim=-1))
                train[:, i, 19:21] = shift_normalize(x[:, 4 + 8 * i, 14:16])
                train[:, i, 21:28] = shift_normalize(torch.cat([x[:, 5 + 8 * i, 0:1], x[:, 5 + 8 * i, 16:22]], dim=-1))
                train[:, i, 28:35] = shift_normalize(torch.cat([x[:, 6 + 8 * i, 0:1], x[:, 6 + 8 * i, 16:22]], dim=-1))
                train[:, i, 35:42] = shift_normalize(torch.cat([x[:, 7 + 8 * i, 0:1], x[:, 7 + 8 * i, 16:22]], dim=-1))
            else:
                raise ValueError("Unknown normalization method: {}".format(self.normalize))
        return train


def shift_normalize(x):
    ''' Shifts the minimum value to 0 and normalizes the values to sum to 1.
    Args: x (tensor): The tensor to normalize.
    Returns: (tensor): The normalized tensor.
    '''
    shifted = x - x.min(dim=-1)[0].unsqueeze(-1)
    return shifted / shifted.sum(dim=-1).unsqueeze(-1)


def show_torch_im(x, name="image"):
    from torchvision.transforms import transforms
    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                   transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                        std=[1., 1., 1.]),
                                   ])
    from matplotlib import pyplot as plt
    t = invTrans(x)
    for i in range(t.size(0)):
        plt.imshow(t[i].permute(1, 2, 0).detach().cpu().numpy())
        plt.savefig(f'{name}_{i}.png')
        plt.show()


def print_pred(outputs):
    color = ['yellow', 'green', 'grey', 'red', 'blue']
    length = ['short', 'long']
    walls = ["braced_wall", 'solid_wall']
    roofs = ["roof_foundation", 'solid_roof', 'braced_roof', 'peaked_roof']
    wheel_count = ['2_wheels', '3_wheels']
    load_obj = ["box", "golden_vase", 'barrel', 'diamond', 'metal_pot', 'oval_vase']
    attribute_classes = ['none'] + color + length + walls + roofs + wheel_count + load_obj
    attributes = ['color', 'length', 'wall', 'roof', 'wheels', 'load1', 'load2', 'load3']
    preds = torch.max(outputs, dim=2)[1]
    preds = preds.T if preds.shape[0] == 32 else preds
    for i in range(preds.shape[0]):
        print("Train", i)
        for j in range(preds.shape[1] // 8):
            car = 'Car' + str(j) + ': '
            for k in range(8):
                car += attributes[k] + '(' + attribute_classes[preds[i, j * 8 + k]] + f'{preds[i, j * 8 + k]})'
                car += ', ' if k < 7 else ''
            print(car)


def print_proccessed(x):
    ''' x (B*E*D)
                e=4 (number of cars),
                d=42 (1+4+5+2+2+5+2+7+7+7) symbolic representation of each car
                    [obj_prob(1) + car_number(4) + color(5) + length(2) + wall(2) + roof(5) + wheels(2) + load1(7) +
                     load2(7) + load3(7)].
                    The format is: [objectness, 1, 2, 3, 4, yellow, green, grey, red, blue,
                                    short, long, braced_wall, solid_wall,
                                    no_roof, roof_foundation, solid_roof, braced_roof, peaked_roof,
                                    2_wheels, 3_wheels,
                                    no_load1, box1, golden_vase1, barrel1, diamond1, metal_pot1, oval_vase1,
                                    no_load2, box2, golden_vase2, barrel2, diamond2, metal_pot2, oval_vase2,
                                    no_load3, box3, golden_vase3, barrel3, diamond3, metal_pot3, oval_vase3].
    '''
    car_num = ['1', '2', '3', '4']
    color = ['yellow', 'green', 'grey', 'red', 'blue']
    length = ['short', 'long']
    walls = ["braced_wall", 'solid_wall']
    roofs = ["none", "roof_foundation", 'solid_roof', 'braced_roof', 'peaked_roof']
    wheel_count = ['2_wheels', '3_wheels']
    load_obj = ['none', "box", "golden_vase", 'barrel', 'diamond', 'metal_pot', 'oval_vase']
    for i in range(x.shape[0]):
        print("Train", i)
        for j in range(x.shape[1]):
            if x[i, j, 0] < 0.5:
                print("Car", j, "No car")
            else:
                car = f'car(Car{str(j)}), '
                car += f'car_number(Car{str(j)}, ' + car_num[
                    torch.max(x[i, j, 1:5], dim=-1)[1].detach().cpu().numpy()] + '), '
                car += f'color(Car{str(j)}, ' + color[
                    torch.max(x[i, j, 5:10], dim=-1)[1].detach().cpu().numpy()] + '), '
                car += f'length(Car{str(j)}, ' + length[
                    torch.max(x[i, j, 10:12], dim=-1)[1].detach().cpu().numpy()] + '), '
                car += f'wall(Car{str(j)}, ' + walls[
                    torch.max(x[i, j, 12:14], dim=-1)[1].detach().cpu().numpy()] + '), '
                car += f'roof(Car{str(j)}, ' + roofs[
                    torch.max(x[i, j, 14:19], dim=-1)[1].detach().cpu().numpy()] + '), '
                car += f'wheels(Car{str(j)}, ' + wheel_count[
                    torch.max(x[i, j, 19:21], dim=-1)[1].detach().cpu().numpy()] + '), '
                car += f'load1(Car{str(j)}, ' + load_obj[
                    torch.max(x[i, j, 21:28], dim=-1)[1].detach().cpu().numpy()] + '), '
                car += f'load2(Car{str(j)}, ' + load_obj[
                    torch.max(x[i, j, 28:35], dim=-1)[1].detach().cpu().numpy()] + '), '
                car += f'load3(Car{str(j)}, ' + load_obj[
                    torch.max(x[i, j, 35:42], dim=-1)[1].detach().cpu().numpy()] + ')'
                print(car)