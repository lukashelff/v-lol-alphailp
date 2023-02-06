import sys

import torch
import torch.nn as nn
from torchvision import models

from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression

from slot_attention.model import SlotAttention_model
import sys

sys.path.insert(0, 'src/yolov5')


class YOLOPerceptionModule(nn.Module):
    """A perception module using YOLO.

    Attrs:
        e (int): The maximum number of entities.
        d (int): The dimension of the object-centric vector.
        device (device): The device where the model and tensors are loaded.
        train (bool): The flag if the parameters are trained.
        preprocess (tensor->tensor): Reshape the yolo output into the unified format of the perceptiom module.
    """

    def __init__(self, e, d, device, train=False, ds_type='kandinsky'):
        super().__init__()
        self.e = e  # num of entities
        self.d = d  # num of dimension
        self.device = device
        self.train_ = train  # the parameters should be trained or not
        if ds_type == 'michalski':
            self.model = self.load_model(
                path='src/weights/michalski/best.pt', device=device)

        else:
            self.model = self.load_model(
                path='src/weights/yolov5/best.pt', device=device)
        # function to transform e * d shape, YOLO returns class labels,
        # it should be decomposed into attributes and the probabilities.
        self.preprocess = YOLOPreprocess(device)

    def load_model(self, path, device):
        print("Loading YOLO model...")
        yolo_net = attempt_load(weights=path)
        yolo_net.to(device)
        if not self.train_:
            for param in yolo_net.parameters():
                param.requires_grad = False
        return yolo_net

    def forward(self, imgs):
        pred = self.model(imgs)[0]  # yolo model returns tuple
        # yolov5.utils.general.non_max_supression returns List[tensors]
        # with lengh of batch size
        # the number of objects can vary image to iamge
        sup = non_max_suppression(pred, max_det=self.e)
        tmp = sup[0]
        yolo_output = self.pad_result(sup)
        post = self.preprocess(yolo_output)
        return post

    def pad_result(self, output):
        """Padding the result by zeros.
            (batch, n_obj, 6) -> (batch, n_max_obj, 6)
        """
        padded_list = []
        for objs in output:
            if objs.size(0) < self.e:
                diff = self.e - objs.size(0)
                zero_tensor = torch.zeros((diff, 6)).to(self.device)
                padded = torch.cat([objs, zero_tensor], dim=0)
                padded_list.append(padded)
            else:
                padded_list.append(objs)
        return torch.stack(padded_list)


class SlotAttentionPerceptionModule(nn.Module):
    """A perception module using Slot Attention.

    Attrs:
        e (int): The maximum number of entities.
        d (int): The dimension of the object-centric vector.
        device (device): The device where the model and tensors are loaded.
        train (bool): The flag if the parameters are trained.
        preprocess (tensor->tensor): Reshape the yolo output into the unified format of the perceptiom module.
        model: The slot attention model.
    """

    def __init__(self, e, d, device, train=False, pretrained=True):
        super().__init__()
        self.e = e  # num of entities -> n_slots=10
        self.d = d  # num of dimension -> encoder_hidden_channels=64
        self.device = device
        self.train_ = train  # the parameters should be trained or not
        self.pretrained = pretrained
        self.model = self.load_model()

    def load_model(self):
        """Load slot attention network.
        """
        if self.device == torch.device('cpu'):
            sa_net = SlotAttention_model(n_slots=self.e, n_iters=3, n_attr=self.d - 1,
                                         encoder_hidden_channels=64,
                                         attention_hidden_channels=128, device=self.device)
            log = torch.load(
                "src/weights/slot_attention/best.pt", map_location=torch.device(self.device))
            if self.pretrained:
                sa_net.load_state_dict(log['weights'], strict=True)
                sa_net.to(self.device)
            if not self.train_:
                for param in sa_net.parameters():
                    param.requires_grad = False
            return sa_net
        else:
            sa_net = SlotAttention_model(n_slots=self.e, n_iters=3, n_attr=self.d - 1,
                                         encoder_hidden_channels=64,
                                         attention_hidden_channels=128, device=self.device)
            log = torch.load("src/weights/slot_attention/best.pt")
            if self.pretrained:
                sa_net.load_state_dict(log['weights'], strict=True)
                sa_net.to(self.device)
            if not self.train_:
                for param in sa_net.parameters():
                    param.requires_grad = False
            return sa_net

    def forward(self, imgs):
        return self.model(imgs)


class YOLOPreprocess(nn.Module):
    """A perception module using Slot Attention.

    Attrs:
        device (device): The device where the model to be loaded.
        img_size (int): The size of the (resized) image to normalize the xy-coordinates.
        classes (list(str)): The classes of objects.
        colors (tensor(int)): The one-hot encodings of the colors (repeated 3 times).
        shapes (tensor(int)): The one-hot encodings of the shapes (repeated 3 times).
    """

    def __init__(self, device, img_size=128):
        super().__init__()
        self.device = device
        self.img_size = img_size
        self.classes = ['red square', 'red circle', 'red triangle',
                        'yellow square', 'yellow circle', 'yellow triangle',
                        'blue square', 'blue circle', 'blue triangle']
        self.colors = torch.stack([
            torch.tensor([1, 0, 0]).to(device),
            torch.tensor([1, 0, 0]).to(device),
            torch.tensor([1, 0, 0]).to(device),
            torch.tensor([0, 1, 0]).to(device),
            torch.tensor([0, 1, 0]).to(device),
            torch.tensor([0, 1, 0]).to(device),
            torch.tensor([0, 0, 1]).to(device),
            torch.tensor([0, 0, 1]).to(device),
            torch.tensor([0, 0, 1]).to(device)
        ])
        self.shapes = torch.stack([
            torch.tensor([1, 0, 0]).to(device),
            torch.tensor([0, 1, 0]).to(device),
            torch.tensor([0, 0, 1]).to(device),
            torch.tensor([1, 0, 0]).to(device),
            torch.tensor([0, 1, 0]).to(device),
            torch.tensor([0, 0, 1]).to(device),
            torch.tensor([1, 0, 0]).to(device),
            torch.tensor([0, 1, 0]).to(device),
            torch.tensor([0, 0, 1]).to(device)
        ])

    def forward(self, x):
        """A preprocess funciton for the YOLO model. The format is: [x1, y1, x2, y2, prob, class].

        Args:
            x (tensor): The output of the YOLO model. The format is:

        Returns:
            Z (tensor): The preprocessed object-centric representation Z. The format is: [x1, y1, x2, y2, color1, color2, color3, shape1, shape2, shape3, objectness].
            x1,x2,y1,y2 are normalized to [0-1].
            The probability for each attribute is obtained by copying the probability of the classification of the YOLO model.
        """
        batch_size = x.size(0)
        obj_num = x.size(1)
        object_list = []
        for i in range(obj_num):
            zi = x[:, i]
            class_id = zi[:, -1].to(torch.int64)
            color = self.colors[class_id] * zi[:, -2].unsqueeze(-1)
            shape = self.shapes[class_id] * zi[:, -2].unsqueeze(-1)
            xyxy = zi[:, 0:4] / self.img_size
            prob = zi[:, -2].unsqueeze(-1)
            obj = torch.cat([xyxy, color, shape, prob], dim=-1)
            object_list.append(obj)
        return torch.stack(object_list, dim=1).to(self.device)


class MichalskiPerceptionModule(nn.Module):
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
        self.model = PerceptioModel(device=device, pth='src/weights/michalski/model.pth')
        self.preprocess = MichalskiPreprocess(device)

    def forward(self, x):
        from torchvision.transforms import transforms



        activations = self.model(x)

        post = self.preprocess(activations)
        # a = torch.max(activations, dim=-1)[1].detach().cpu().numpy()
        # b = post.detach().cpu().numpy()
        invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                       transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                            std=[1., 1., 1.]),
                                       ])
        from matplotlib import pyplot as plt
        plt.imshow(invTrans(x)[0].permute(1, 2, 0))
        plt.savefig("im1.png")
        self.predict_train(x)
        self.print_train(post)


        return post

    def predict_train(self, x):
        color = ['yellow', 'green', 'grey', 'red', 'blue']
        length = ['short', 'long']
        walls = ["braced_wall", 'solid_wall']
        roofs = ["roof_foundation", 'solid_roof', 'braced_roof', 'peaked_roof']
        wheel_count = ['2_wheels', '3_wheels']
        load_obj = ["box", "golden_vase", 'barrel', 'diamond', 'metal_pot', 'oval_vase']
        attribute_classes = ['none'] + color + length + walls + roofs + wheel_count + load_obj
        attributes = ['color', 'length', 'wall', 'roof', 'wheels', 'load1', 'load2', 'load3']

        activations = self.model(x)
        preds = torch.max(activations, dim=-1)[1].detach().cpu().numpy()
        for i in range(preds.shape[0]):
            print("Train", i)
            for j in range(preds.shape[1] // 8):
                car = 'Car' + str(j) + ': '
                for k in range(8):
                    car += attributes[k] + '(' + attribute_classes[preds[i, j * 8 + k]] + ')'
                    car += ', ' if k < 7 else ''
                print(car)


    def print_train(self, x):
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
                    car = 'Car' + str(j) + ': '
                    car += 'car_number(' + car_num[torch.max(x[i, j, 1:5], dim=-1)[1].detach().cpu().numpy()] + '), '
                    car += 'color(' + color[torch.max(x[i, j, 5:10], dim=-1)[1].detach().cpu().numpy()] + '), '
                    car += 'length(' + length[torch.max(x[i, j, 10:12], dim=-1)[1].detach().cpu().numpy()] + '), '
                    car += 'wall(' + walls[torch.max(x[i, j, 12:14], dim=-1)[1].detach().cpu().numpy()] + '), '
                    car += 'roof(' + roofs[torch.max(x[i, j, 14:19], dim=-1)[1].detach().cpu().numpy()] + '), '
                    car += 'wheels(' + wheel_count[torch.max(x[i, j, 19:21], dim=-1)[1].detach().cpu().numpy()] + '), '
                    car += 'load1(' + load_obj[torch.max(x[i, j, 21:28], dim=-1)[1].detach().cpu().numpy()] + '), '
                    car += 'load2(' + load_obj[torch.max(x[i, j, 28:35], dim=-1)[1].detach().cpu().numpy()] + '), '
                    car += 'load3(' + load_obj[torch.max(x[i, j, 35:42], dim=-1)[1].detach().cpu().numpy()] + ')'
                    print(car)


class PerceptioModel(nn.Module):
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

    def __init__(self, device, pth):
        super().__init__()
        resnet = models.resnet18()
        layers = list(resnet.children())[:9]
        self.fc = resnet.fc
        self.features1 = nn.Sequential(*layers[:6])
        self.features2 = nn.Sequential(*layers[6:])
        self.bb = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))
        self.classifier = nn.ModuleList()
        self.label_num_classes = [22] * 8
        all_classes = sum(self.label_num_classes)
        in_features = resnet.inplanes
        for _ in range(4):
            self.classifier.append(nn.Sequential(nn.Linear(in_features=in_features, out_features=all_classes)))
        self.load_state_dict(torch.load(pth, map_location=device)['model_state_dict'])
        self.to(device)

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = torch.flatten(x, 1)
        soft = nn.Softmax(dim=1)
        class_output = [classifier(x) for classifier in self.classifier]
        activations = torch.cat(class_output, dim=1).view(-1, 32, 22)
        # preds = []
        # for output in class_output:
        #     for i, num_classes in enumerate(self.label_num_classes):
        #         ind_start = sum(self.label_num_classes[:i])
        #         ind_end = ind_start + num_classes
        #         pred = soft(output[:, ind_start:ind_end])
        #         preds.append(pred)
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

    def __init__(self, device, img_size=128):
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

        train = torch.zeros(x.size(0), 4, 42).to(self.device)
        for i in range(4):
            # preprocess to object-centric car representation with accuracies for each attribute
            # Attributes: [prob, pos, color, length, wall, roof, wheels, load1, load2, load3].
            #           d=42 (1+4+5+2+2+5+2+7+7+7)

            # car number
            train[:, i, i + 1] = 1
            # color
            train[:, i, 5:10] = x[:, 8 * i, 1:6]
            train[:, i, 5:10] -= train[:, i, 5:10].min(dim=-1)[0].unsqueeze(-1)
            train[:, i, 5:10] /= train[:, i, 5:10].sum(dim=-1).unsqueeze(-1)
            # objectness
            vals = x[:, 8 * i, 0:6]
            vals -= vals.min(dim=-1)[0].unsqueeze(-1)
            vals /= vals[:, 0].unsqueeze(-1) + vals[:, 1:].max(dim=-1)[0].unsqueeze(-1)
            train[:, i, 0] = 1 - vals[:, 0]
            # length
            train[:, i, 10:12] = x[:, 1 + 8 * i, 6:8]
            train[:, i, 10:12] -= train[:, i, 10:12].min(dim=-1)[0].unsqueeze(-1)
            train[:, i, 10:12] /= train[:, i, 10:12].sum(dim=-1).unsqueeze(-1)
            # wall
            train[:, i, 12:14] = x[:, 2 + 8 * i, 8:10]
            train[:, i, 12:14] -= train[:, i, 12:14].min(dim=-1)[0].unsqueeze(-1)
            train[:, i, 12:14] /= train[:, i, 12:14].sum(dim=-1).unsqueeze(-1)
            # roof
            train[:, i, 14] = x[:, 3 + 8 * i, 0]
            train[:, i, 15:19] = x[:, 3 + 8 * i, 10:14]
            train[:, i, 14:19] -= train[:, i, 14:19].min(dim=-1)[0].unsqueeze(-1)
            train[:, i, 14:19] /= train[:, i, 14:19].sum(dim=-1).unsqueeze(-1)
            # wheel count
            train[:, i, 19:21] = x[:, 4 + 8 * i, 14:16]
            train[:, i, 19:21] -= train[:, i, 19:21].min(dim=-1)[0].unsqueeze(-1)
            train[:, i, 19:21] /= train[:, i, 19:21].sum(dim=-1).unsqueeze(-1)
            # load 1
            train[:, i, 21] = x[:, 5 + 8 * i, 0]
            train[:, i, 22:28] = x[:, 5 + 8 * i, 16:22]
            train[:, i, 21:28] -= train[:, i, 21:28].min(dim=-1)[0].unsqueeze(-1)
            train[:, i, 21:28] /= train[:, i, 21:28].sum(dim=-1).unsqueeze(-1)
            # load 2
            train[:, i, 28] = x[:, 6 + 8 * i, 0]
            train[:, i, 29:35] = x[:, 6 + 8 * i, 16:22]
            train[:, i, 28:35] -= train[:, i, 29:35].min(dim=-1)[0].unsqueeze(-1)
            train[:, i, 28:35] /= train[:, i, 29:35].sum(dim=-1).unsqueeze(-1)
            # load 3
            train[:, i, 35] = x[:, 7 + 8 * i, 0]
            train[:, i, 36:42] = x[:, 7 + 8 * i, 16:22]
            train[:, i, 35:42] -= train[:, i, 36:42].min(dim=-1)[0].unsqueeze(-1)
            train[:, i, 35:42] /= train[:, i, 36:42].sum(dim=-1).unsqueeze(-1)
        return train
