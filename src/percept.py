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



class MichalskiPerceptionModule(nn.Module):
    """A perception module using Michalski.

    Attrs:
        e (int): The maximum number of entities.
        d (int): The dimension of the object-centric vector.
        device (device): The device where the model and tensors are loaded.
        train (bool): The flag if the parameters are trained.
        preprocess (tensor->tensor): Reshape the yolo output into the unified format of the perceptiom module.
    """

    # def __init__(self, e, d, device, train=False):
    #     super().__init__()
    #     self.e = e  # num of entities
    #     self.d = d  # num of dimension
    #     self.device = device
    #     self.train_ = train  # the parameters should be trained or not
    #     self.model = self.load_model()
    #     # function to transform e * d shape, YOLO returns class labels,
    #     # it should be decomposed into attributes and the probabilities.
    #     self.preprocess = MichalskiPreprocess(device)
    #
    # def load_model(self):
    #     print("Loading Michalski model...")
    #     michalski_net = attempt_load(weights='src/weights/michalski/model.pth')
    #     michalski_net.to(self.device)
    #     if not self.train_:
    #         for param in michalski_net.parameters():
    #             param.requires_grad = False
    #     return michalski_net

    def __init__(self, device, train=False):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        checkpoint = torch.load('src/weights/michalski/model.pth' , map_location=device)
        layers = list(resnet.children())[:9]
        self.fc = resnet.fc
        self.features1 = nn.Sequential(*layers[:6])
        self.features2 = nn.Sequential(*layers[6:])
        self.bb = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))
        self.classifier = nn.ModuleList()
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
        color = ['yellow', 'green', 'grey', 'red', 'blue']
        length = ['short', 'long']
        walls = ["braced_wall", 'solid_wall']
        roofs = ["roof_foundation", 'solid_roof', 'braced_roof', 'peaked_roof']
        wheel_count = ['2_wheels', '3_wheels']
        load_obj = ["box", "golden_vase", 'barrel', 'diamond', 'metal_pot', 'oval_vase']  # correct order
        # color, length, walls, roofs, wheel_count, load_obj1, load_obj2, load_obj3
        self.label_num_classes = [22] * 8
        all_classes = sum(self.label_num_classes)
        in_features = resnet.inplanes
        for _ in range(4):
            self.classifier.append(nn.Sequential(nn.Linear(in_features=in_features, out_features=all_classes)))
        self.load_state_dict(checkpoint['model_state_dict'])

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = torch.flatten(x, 1)

        soft = nn.Softmax(dim=1)

        class_output = [classifier(x) for classifier in self.classifier]
        preds = torch.cat(class_output, dim=1).view(-1, 32, 22)
        # preds = []
        # for output in class_output:
        #     for i, num_classes in enumerate(self.label_num_classes):
        #         ind_start = sum(self.label_num_classes[:i])
        #         ind_end = ind_start + num_classes
        #         pred = soft(output[:, ind_start:ind_end])
        #         preds.append(pred)
        return preds



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
            classes: obj_prob + car_number + color + length + wall + roof + load + load_number
                [objectness, 1, 2, 3, 4, yellow, green, grey, red, blue, short, long, full, braced,
                 none, foundation, solid_roof, braced_roof, peaked roof, 2,3,
                 blue_box, golden_vase, barrel, diamond, metal_box, 0,1,2,3]

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


            self.car_nums = ['1', '2', '3', '4']
            self.colors = ["yellow", "green", "grey", "red", "blue"]
            self.lengths = ["short", "long"]
            self.walls = ["full", "braced"]
            self.roofs = ["none", "foundation", "solid_roof", "braced_roof", "peaked_roof"]
            self.wheels = ['2', '3']
            self.loads = ["blue_box", "golden_vase", "barrel", "diamond", "metal_box"]
            self.load_nums = ['0', '1', '2', '3']




            xyxy = zi[:, 0:4] / self.img_size
            prob = zi[:, -2].unsqueeze(-1)
            obj = torch.cat([xyxy, color, shape, prob], dim=-1)
            object_list.append(obj)
        return torch.stack(object_list, dim=1).to(self.device)

