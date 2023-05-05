import torch
import torch.nn as nn
import torch.nn.functional as F
from valuation_func import *


class YOLOValuationModule(nn.Module):
    """A module to call valuation functions.
        Attrs:
            lang (language): The language.
            device (device): The device.
            layers (list(nn.Module)): The list of valuation functions.
            vfs (dic(str->nn.Module)): The dictionaty that maps a predicate name to the corresponding valuation function.
            attrs (dic(term->tensor)): The dictionary that maps an attribute term to the corresponding one-hot encoding.
            dataset (str): The dataset.
    """

    def __init__(self, lang, device, dataset):
        super().__init__()
        self.lang = lang
        self.device = device
        self.layers, self.vfs = self.init_valuation_functions(device, dataset)
        # attr_term -> vector representation dic
        self.attrs = self.init_attr_encodings(device)
        self.dataset = dataset

    def init_valuation_functions(self, device, dataset=None):
        """
            Args:
                device (device): The device.
                dataset (str): The dataset.

            Retunrs:
                layers (list(nn.Module)): The list of valuation functions.
                vfs (dic(str->nn.Module)): The dictionaty that maps a predicate name to the corresponding valuation function.
        """
        layers = []
        vfs = {}  # a dictionary: pred_name -> valuation function
        v_color = YOLOColorValuationFunction()
        vfs['color'] = v_color
        layers.append(v_color)
        v_shape = YOLOShapeValuationFunction()
        vfs['shape'] = v_shape
        v_in = YOLOInValuationFunction()
        vfs['in'] = v_in
        layers.append(v_in)
        v_closeby = YOLOClosebyValuationFunction(device)
        # if dataset in ['closeby', 'red-triangle']:
        vfs['closeby'] = v_closeby
        vfs['closeby'].load_state_dict(torch.load(
            'src/weights/neural_predicates/closeby_pretrain.pt', map_location=device))
        vfs['closeby'].eval()
        layers.append(v_closeby)
        # print('Pretrained  neural predicate closeby have been loaded!')
        # elif dataset == 'online-pair':
        v_online = YOLOOnlineValuationFunction(device)
        vfs['online'] = v_online
        vfs['online'].load_state_dict(torch.load(
            'src/weights/neural_predicates/online_pretrain.pt', map_location=device))
        vfs['online'].eval()
        layers.append(v_online)
        #    print('Pretrained  neural predicate online have been loaded!')
        return nn.ModuleList(layers), vfs

    def init_attr_encodings(self, device):
        """Encode color and shape into one-hot encoding.

            Args:
                device (device): The device.

            Returns:
                attrs (dic(term->tensor)): The dictionary that maps an attribute term to the corresponding one-hot encoding.
        """
        attr_names = ['color', 'shape']
        attrs = {}
        for dtype_name in attr_names:
            for term in self.lang.get_by_dtype_name(dtype_name):
                term_index = self.lang.term_index(term)
                num_classes = len(self.lang.get_by_dtype_name(dtype_name))
                one_hot = F.one_hot(torch.tensor(
                    term_index).to(device), num_classes=num_classes)
                one_hot.to(device)
                attrs[term] = one_hot
        return attrs

    def forward(self, zs, atom):
        """Convert the object-centric representation to a valuation tensor.

            Args:
                zs (tensor): The object-centric representaion (the output of the YOLO model).
                atom (atom): The target atom to compute its proability.

            Returns:
                A batch of the probabilities of the target atom.
        """
        if atom.pred.name in self.vfs:
            args = [self.ground_to_tensor(term, zs) for term in atom.terms]
            # call valuation function
            return self.vfs[atom.pred.name](*args)
        else:
            return torch.zeros((zs.size(0),)).to(
                torch.float32).to(self.device)

    def ground_to_tensor(self, term, zs):
        """Ground terms into tensor representations.

            Args:
                term (term): The term to be grounded.
                zs (tensor): The object-centric representation.

            Return:
                The tensor representation of the input term.
        """
        term_index = self.lang.term_index(term)
        if term.dtype.name == 'object':
            return zs[:, term_index]
        elif term.dtype.name == 'color' or term.dtype.name == 'shape':
            return self.attrs[term]
        elif term.dtype.name == 'image':
            return None
        else:
            assert 0, "Invalid datatype of the given term: " + \
                      str(term) + ':' + term.dtype.name


class SlotAttentionValuationModule(nn.Module):
    """A module to call valuation functions.
        Attrs:
            lang (language): The language.
            device (device): The device.
            layers (list(nn.Module)): The list of valuation functions.
            vfs (dic(str->nn.Module)): The dictionary that maps a predicate name to the corresponding valuation function.
            attrs (dic(term->tensor)): The dictionary that maps an attribute term to the corresponding one-hot encoding.
            dataset (str): The dataset.
    """

    def __init__(self, lang, device, pretrained=True):
        super().__init__()
        self.lang = lang
        self.device = device
        self.colors = ["cyan", "blue", "yellow",
                       "purple", "red", "green", "gray", "brown"]
        self.shapes = ["sphere", "cube", "cylinder"]
        self.sizes = ["large", "small"]
        self.materials = ["rubber", "metal"]
        self.sides = ["left", "right"]

        self.layers, self.vfs = self.init_valuation_functions(
            device, pretrained)

    def init_valuation_functions(self, device, pretrained):
        """
            Args:
                device (device): The device.
                pretrained (bool): The flag if the neural predicates are pretrained or not.

            Retunrs:
                layers (list(nn.Module)): The list of valuation functions.
                vfs (dic(str->nn.Module)): The dictionaty that maps a predicate name to the corresponding valuation function.
        """
        layers = []
        vfs = {}  # pred name -> valuation function
        v_color = SlotAttentionColorValuationFunction(device)
        vfs['color'] = v_color
        v_shape = SlotAttentionShapeValuationFunction(device)
        vfs['shape'] = v_shape
        v_in = SlotAttentionInValuationFunction(device)
        vfs['in'] = v_in
        v_size = SlotAttentionSizeValuationFunction(device)
        vfs['size'] = v_size
        v_material = SlotAttentionMaterialValuationFunction(device)
        vfs['material'] = v_material
        v_rightside = SlotAttentionRightSideValuationFunction(device)
        vfs['rightside'] = v_rightside
        v_leftside = SlotAttentionLeftSideValuationFunction(device)
        vfs['leftside'] = v_leftside
        v_front = SlotAttentionFrontValuationFunction(device)
        vfs['front'] = v_front

        if pretrained:
            vfs['rightside'].load_state_dict(torch.load(
                'src/weights/neural_predicates/rightside_pretrain.pt', map_location=device))
            vfs['rightside'].eval()
            vfs['leftside'].load_state_dict(torch.load(
                'src/weights/neural_predicates/leftside_pretrain.pt', map_location=device))
            vfs['leftside'].eval()
            vfs['front'].load_state_dict(torch.load(
                'src/weights/neural_predicates/front_pretrain.pt', map_location=device))
            vfs['front'].eval()
            print('Pretrained  neural predicates have been loaded!')
        return nn.ModuleList([v_color, v_shape, v_in, v_size, v_material, v_rightside, v_leftside, v_front]), vfs

    def forward(self, zs, atom):
        """Convert the object-centric representation to a valuation tensor.

            Args:
                zs (tensor): The object-centric representaion (the output of the YOLO model).
                atom (atom): The target atom to compute its proability.

            Returns:
                A batch of the probabilities of the target atom.
        """
        # term: logical term
        # arg: vector representation of the term
        # zs = self.preprocess(zs)
        args = [self.ground_to_tensor(term, zs) for term in atom.terms]
        # call valuation function
        return self.vfs[atom.pred.name](*args)

    def ground_to_tensor(self, term, zs):
        """Ground terms into tensor representations.

            Args:
                term (term): The term to be grounded.
                zs (tensor): The object-centric representation.
        """
        term_index = self.lang.term_index(term)
        if term.dtype.name == 'object':
            return zs[:, term_index]
        elif term.dtype.name == 'image':
            return None
        else:
            # other attributes
            return self.term_to_onehot(term, batch_size=zs.size(0))

    def term_to_onehot(self, term, batch_size):
        """Ground terms into tensor representations.

            Args:
                term (term): The term to be grounded.
                zs (tensor): The object-centric representation.

            Return:
                The tensor representation of the input term.
        """
        if term.dtype.name == 'color':
            return self.to_onehot_batch(self.colors.index(term.name), len(self.colors), batch_size)
        elif term.dtype.name == 'shape':
            return self.to_onehot_batch(self.shapes.index(term.name), len(self.shapes), batch_size)
        elif term.dtype.name == 'material':
            return self.to_onehot_batch(self.materials.index(term.name), len(self.materials), batch_size)
        elif term.dtype.name == 'size':
            return self.to_onehot_batch(self.sizes.index(term.name), len(self.sizes), batch_size)
        elif term.dtype.name == 'side':
            return self.to_onehot_batch(self.sides.index(term.name), len(self.sides), batch_size)
        else:
            assert True, 'Invalid term: ' + str(term)

    def to_onehot_batch(self, i, length, batch_size):
        """Compute the one-hot encoding that is expanded to the batch size.
        """
        onehot = torch.zeros(batch_size, length, ).to(self.device)
        onehot[:, i] = 1.0
        return onehot


class MichalskiValuationModule(nn.Module):
    """A module to call valuation functions.
        Attrs:
            lang (language): The language.
            device (device): The device.
            layers (list(nn.Module)): The list of valuation functions.
            vfs (dic(str->nn.Module)): The dictionary that maps a predicate name to the corresponding valuation function.
            attrs (dic(term->tensor)): The dictionary that maps an attribute term to the corresponding one-hot encoding.
            dataset (str): The dataset.
    """

    def __init__(self, lang, device, pretrained=True):
        super().__init__()
        self.lang = lang
        self.device = device
        self.car_nums = ['1', '2', '3', '4']
        self.colors = ["yellow", "green", "grey", "red", "blue"]
        self.lengths = ["short", "long"]
        self.walls = ["full", "braced"]
        self.roofs = ["none", "foundation", "solid_roof", "braced_roof", "peaked_roof"]
        self.wheels = ['2', '3']
        self.loads = ["none", "blue_box", "golden_vase", "barrel", "diamond", "metal_box", "oval_vase"]
        self.load_nums = ['0', '1', '2', '3']
        self.int = ['0', '1', '2', '3', '4', '5', '6', '7', ]
        # self.int = ['1', '2', '3', '4']
        self.obj_desc = {
            'car_num': self.car_nums,
            'color': self.colors,
            'length': self.lengths,
            'wall': self.walls,
            'roof': self.roofs,
            'wheel': self.wheels,
            'int': self.int,
            'load': self.loads,
        }

        self.layers, self.vfs = self.init_valuation_functions(
            device, pretrained)

    def init_valuation_functions(self, device, pretrained):
        """
            Args:
                device (device): The device.
                pretrained (bool): The flag if the neural predicates are pretrained or not.

            Retunrs:
                layers (list(nn.Module)): The list of valuation functions.
                vfs (dic(str->nn.Module)): The dictionaty that maps a predicate name to the corresponding valuation function.
        """
        layers = []
        vfs = {}  # pred name -> valuation function
        v_in = MichalskiInValuationFunction(device)
        v_car_num = MichalskiCarNumValuationFunction(device)
        v_color = MichalskiColorValuationFunction(device)
        v_length = MichalskiLengthValuationFunction(device)
        v_wall = MichalskiWallValuationFunction(device)
        v_roof = MichalskiRoofValuationFunction(device)
        v_wheel = MichalskiWheelValuationFunction(device)
        v_load1 = MichalskiLoad1ValuationFunction(device)
        v_load2 = MichalskiLoad2ValuationFunction(device)
        v_load3 = MichalskiLoad3ValuationFunction(device)
        # v_load_num = MichalskiLoadNumValuationFunction(device)
        vfs['in'] = v_in
        vfs['car_num'] = v_car_num
        vfs['color'] = v_color
        vfs['length'] = v_length
        vfs['wall'] = v_wall
        vfs['roof'] = v_roof
        vfs['wheel'] = v_wheel
        vfs['load1'] = v_load1
        vfs['load2'] = v_load2
        vfs['load3'] = v_load3
        # vfs['load_num'] = v_load_num
        return nn.ModuleList([v_in, v_car_num, v_color, v_length, v_wall, v_roof, v_load1, v_load2, v_load3]), vfs

    def forward(self, zs, atom):
        """Convert the object-centric representation to a valuation tensor.

            Args:
                zs (tensor): The object-centric representaion (the output of the YOLO model).
                atom (atom): The target atom to compute its proability.

            Returns:
                A batch of the probabilities of the target atom.
        """
        # term: logical term
        # arg: vector representation of the term
        # zs = self.preprocess(zs)
        args = [self.ground_to_tensor(term, zs) for term in atom.terms]
        # call valuation function
        return self.vfs[atom.pred.name](*args)

    def ground_to_tensor(self, term, zs):
        """Ground terms into tensor representations.

            Args:
                term (term): The term to be grounded.
                zs (tensor): The object-centric representation.
        """
        term_index = self.lang.term_index(term)
        if term.dtype.name == 'car':
            # try:
            #     return zs[:, term_index]
            # except:
            #     return torch.zeros(zs.size(0), zs.size(2)).to(self.device)

            return zs[:, term_index]
        elif term.dtype.name == 'image':
            return None
        else:
            # other attributes
            return self.term_to_onehot(term, batch_size=zs.size(0))

    def term_to_onehot(self, term, batch_size):
        """Ground terms into tensor representations.

            Args:
                term (term): The term to be grounded.
                zs (tensor): The object-centric representation.

            Return:
                The tensor representation of the input term.
        """
        within = term.dtype.name in self.obj_desc
        if term.dtype.name in self.obj_desc:
            values = self.obj_desc[term.dtype.name]
            val_name = term.name
            try:
                val_idx = values.index(val_name)
            except:
                raise AttributeError(f'value {val_name} is not an attribute of {term.dtype.name}: {values}')
            return self.to_onehot_batch(val_idx, len(values), batch_size)
        else:
            assert True, 'Invalid term: ' + str(term)

    def to_onehot_batch(self, i, length, batch_size):
        """Compute the one-hot encoding that is expanded to the batch size.
        """
        onehot = torch.zeros(batch_size, length, ).to(self.device)
        onehot[:, i] = 1.0
        return onehot
