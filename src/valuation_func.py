import torch
import torch.nn as nn
from neural_utils import MLP, LogisticRegression


################################
# Valuation functions for YOLO #
################################

class YOLOColorValuationFunction(nn.Module):
    """The function v_color.
    """

    def __init__(self):
        super(YOLOColorValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor B * d of object-centric representation.
                [x1, y1, x2, y2, color1, color2, color3,
                    shape1, shape2, shape3, objectness]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_color = z[:, 4:7]
        return (a * z_color).sum(dim=1)


class YOLOShapeValuationFunction(nn.Module):
    """The function v_shape.
    """

    def __init__(self):
        super(YOLOShapeValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                [x1, y1, x2, y2, color1, color2, color3,
                    shape1, shape2, shape3, objectness]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_shape = z[:, 7:10]
        # a_batch = a.repeat((z.size(0), 1))  # one-hot encoding for batch
        return (a * z_shape).sum(dim=1)


class YOLOInValuationFunction(nn.Module):
    """The function v_in.
    """

    def __init__(self):
        super(YOLOInValuationFunction, self).__init__()

    def forward(self, z, x):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                [x1, y1, x2, y2, color1, color2, color3,
                    shape1, shape2, shape3, objectness]
            x (none): A dummy argment to represent the input constant.

        Returns:
            A batch of probabilities.
        """
        return z[:, -1]


class YOLOClosebyValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self, device):
        super(YOLOClosebyValuationFunction, self).__init__()
        self.device = device
        self.logi = LogisticRegression(input_dim=1)
        self.logi.to(device)

    def forward(self, z_1, z_2):
        """
        Args:
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
                [x1, y1, x2, y2, color1, color2, color3,
                    shape1, shape2, shape3, objectness]
            z_2 (tensor): 2-d tensor (B * D), the object-centric representation.
                [x1, y1, x2, y2, color1, color2, color3,
                    shape1, shape2, shape3, objectness]

        Returns:
            A batch of probabilities.
        """
        c_1 = self.to_center(z_1)
        c_2 = self.to_center(z_2)
        dist = torch.norm(c_1 - c_2, dim=0).unsqueeze(-1)
        return self.logi(dist).squeeze()

    def to_center(self, z):
        x = (z[:, 0] + z[:, 2]) / 2
        y = (z[:, 1] + z[:, 3]) / 2
        return torch.stack((x, y))


class YOLOOnlineValuationFunction(nn.Module):
    """The function v_online.
    """

    def __init__(self, device):
        super(YOLOOnlineValuationFunction, self).__init__()
        self.logi = LogisticRegression(input_dim=1)
        self.logi.to(device)

    def forward(self, z_1, z_2, z_3, z_4, z_5):
        """The function to compute the probability of the online predicate.

        The closed form of the linear regression is computed.
        The error value is fed into the 1-d logistic regression function.

        Args:
            z_i (tensor): 2-d tensor (B * D), the object-centric representation.
                [x1, y1, x2, y2, color1, color2, color3,
                    shape1, shape2, shape3, objectness]

        Returns:
            A batch of probabilities.
        """
        X = torch.stack([self.to_center_x(z)
                         for z in [z_1, z_2, z_3, z_4, z_5]], dim=1).unsqueeze(-1)
        Y = torch.stack([self.to_center_y(z)
                         for z in [z_1, z_2, z_3, z_4, z_5]], dim=1).unsqueeze(-1)
        # add bias term
        X = torch.cat([torch.ones_like(X), X], dim=2)
        X_T = torch.transpose(X, 1, 2)
        # the optimal weights from the closed form solution
        W = torch.matmul(torch.matmul(
            torch.inverse(torch.matmul(X_T, X)), X_T), Y)
        diff = torch.norm(Y - torch.sum(torch.transpose(W, 1, 2)
                                        * X, dim=2).unsqueeze(-1), dim=1)
        self.diff = diff
        return self.logi(diff).squeeze()

    def to_center_x(self, z):
        x = (z[:, 0] + z[:, 2]) / 2
        return x

    def to_center_y(self, z):
        y = (z[:, 1] + z[:, 3]) / 2
        return y


##########################################
# Valuation functions for slot attention #
##########################################


class SlotAttentionInValuationFunction(nn.Module):
    """The function v_in.
    """

    def __init__(self, device):
        super(SlotAttentionInValuationFunction, self).__init__()

    def forward(self, z, x):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + coords + shape + size + material + color
                [objectness, x, y, z, sphere, cube, cylinder, large, small, rubber,
                    metal, cyan, blue, yellow, purple, red, green, gray, brown]
            x (none): A dummy argment to represent the input constant.

        Returns:
            A batch of probabilities.
        """
        # return the objectness
        return z[:, 0]


class SlotAttentionShapeValuationFunction(nn.Module):
    """The function v_shape.
    """

    def __init__(self, device):
        super(SlotAttentionShapeValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + coords + shape + size + material + color
                [objectness, x, y, z, sphere, cube, cylinder, large, small, rubber,
                    metal, cyan, blue, yellow, purple, red, green, gray, brown]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_shape = z[:, 4:7]
        return (a * z_shape).sum(dim=1)


class SlotAttentionSizeValuationFunction(nn.Module):
    """The function v_size.
    """

    def __init__(self, device):
        super(SlotAttentionSizeValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + coords + shape + size + material + color
                [objectness, x, y, z, sphere, cube, cylinder, large, small, rubber,
                    metal, cyan, blue, yellow, purple, red, green, gray, brown]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_size = z[:, 7:9]
        return (a * z_size).sum(dim=1)


class SlotAttentionMaterialValuationFunction(nn.Module):
    """The function v_material.
    """

    def __init__(self, device):
        super(SlotAttentionMaterialValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + coords + shape + size + material + color
                [objectness, x, y, z, sphere, cube, cylinder, large, small, rubber,
                    metal, cyan, blue, yellow, purple, red, green, gray, brown]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_material = z[:, 9:11]
        return (a * z_material).sum(dim=1)


class SlotAttentionColorValuationFunction(nn.Module):
    """The function v_color.
    """

    def __init__(self, device):
        super(SlotAttentionColorValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + coords + shape + size + material + color
                [objectness, x, y, z, sphere, cube, cylinder, large, small, rubber,
                    metal, cyan, blue, yellow, purple, red, green, gray, brown]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_color = z[:, 11:19]
        return (a * z_color).sum(dim=1)


class SlotAttentionRightSideValuationFunction(nn.Module):
    """The function v_rightside.
    """

    def __init__(self, device):
        super(SlotAttentionRightSideValuationFunction, self).__init__()
        self.logi = LogisticRegression(input_dim=1, output_dim=1)
        self.logi.to(device)

    def forward(self, z):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + coords + shape + size + material + color
                [objectness, x, y, z, sphere, cube, cylinder, large, small, rubber,
                    metal, cyan, blue, yellow, purple, red, green, gray, brown]
        Returns:
            A batch of probabilities.
        """
        z_x = z[:, 1].unsqueeze(-1)  # (B, )
        prob = self.logi(z_x).squeeze()  # (B, )
        objectness = z[:, 0]  # (B, )
        return prob * objectness


class SlotAttentionLeftSideValuationFunction(nn.Module):
    """The function v_leftside.
    """

    def __init__(self, device):
        super(SlotAttentionLeftSideValuationFunction, self).__init__()
        self.logi = LogisticRegression(input_dim=1, output_dim=1)
        self.logi.to(device)

    def forward(self, z):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + coords + shape + size + material + color
                [objectness, x, y, z, sphere, cube, cylinder, large, small, rubber,
                    metal, cyan, blue, yellow, purple, red, green, gray, brown]
        Returns:
            A batch of probabilities.
        """
        z_x = z[:, 1].unsqueeze(-1)  # (B, )
        prob = self.logi(z_x).squeeze()  # (B, )
        objectness = z[:, 0]  # (B, )
        return prob * objectness


class SlotAttentionFrontValuationFunction(nn.Module):
    """The function v_infront.
    """

    def __init__(self, device):
        super(SlotAttentionFrontValuationFunction, self).__init__()
        self.logi = LogisticRegression(input_dim=6, output_dim=1)
        self.logi.to(device)

    def forward(self, z_1, z_2):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + coords + shape + size + material + color
                [objectness, x, y, z, sphere, cube, cylinder, large, small, rubber,
                    metal, cyan, blue, yellow, purple, red, green, gray, brown]
        Returns:
            A batch of probabilities.
        """
        xyz_1 = z_1[:, 1:4]
        xyz_2 = z_2[:, 1:4]
        xyzxyz = torch.cat([xyz_1, xyz_2], dim=1)
        prob = self.logi(xyzxyz).squeeze()  # (B,)
        objectness = z_1[:, 0] * z_2[:, 0]  # (B,)
        return prob * objectness


##########################################
# Michalski valuation functions for slot attention #
##########################################


class MichalskiInValuationFunction(nn.Module):
    """The function v_in.
    """

    def __init__(self, device):
        super(MichalskiInValuationFunction, self).__init__()

    def forward(self, z, x):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
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
            x (none): A dummy argument to represent the input constant.

        Returns:
            A batch of probabilities.
        """
        # return the objectness
        return z[:, 0]


class MichalskiCarNumValuationFunction(nn.Module):
    """The function v__car_num.
    """

    def __init__(self, device):
        super(MichalskiCarNumValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
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
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_shape = z[:, 1:5] * z[:, 0].unsqueeze(-1)  # (B, 4)
        z_shape_padded = torch.cat(
            [torch.zeros(z_shape.shape[0], 1), z_shape, torch.zeros(z_shape.shape[0], 3).to(z.device)], dim=1)  # (B, 5)
        return (a * z_shape_padded).sum(dim=1)


class MichalskiColorValuationFunction(nn.Module):
    """The function v_size.
    """

    def __init__(self, device):
        super(MichalskiColorValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
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
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_size = z[:, 5:10]
        return (a * z_size).sum(dim=1)


class MichalskiLengthValuationFunction(nn.Module):
    """The function v_material.
    """

    def __init__(self, device):
        super(MichalskiLengthValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
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
        Returns:
            A batch of probabilities.
        """
        z_material = z[:, 10:12]
        return (a * z_material).sum(dim=1)


class MichalskiWallValuationFunction(nn.Module):
    """The function v_color.
    """

    def __init__(self, device):
        super(MichalskiWallValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
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
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_color = z[:, 12:14]
        return (a * z_color).sum(dim=1)


class MichalskiRoofValuationFunction(nn.Module):
    """The function v_color.
    """

    def __init__(self, device):
        super(MichalskiRoofValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
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
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_color = z[:, 14:19]
        return (a * z_color).sum(dim=1)


class MichalskiWheelValuationFunction(nn.Module):
    """The function v_color.
    """

    def __init__(self, device):
        super(MichalskiWheelValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
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
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_color = z[:, 19:21]
        z_color_padded = torch.cat([torch.zeros(z_color.shape[0], 2).to(z_color.device), z_color,
                                    torch.zeros(z_color.shape[0], 4).to(z_color.device)], dim=1)
        return (a * z_color_padded).sum(dim=1)


class MichalskiLoad1ValuationFunction(nn.Module):
    """The function v_color.
    """

    def __init__(self, device):
        super(MichalskiLoad1ValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
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
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_color = z[:, 21:28]
        return (a * z_color).sum(dim=1)


class MichalskiLoad2ValuationFunction(nn.Module):
    """The function v_color.
    """

    def __init__(self, device):
        super(MichalskiLoad2ValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
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

        Returns:
            A batch of probabilities.
        """
        z_color = z[:, 28:35]
        return (a * z_color).sum(dim=1)


class MichalskiLoad3ValuationFunction(nn.Module):
    """The function v_color.
    """

    def __init__(self, device):
        super(MichalskiLoad3ValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
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
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_color = z[:, 35:42]
        return (a * z_color).sum(dim=1)

# class MichalskiLoadNumValuationFunction(nn.Module):
#     """The function v_color.
#     """
#
#     def __init__(self, device):
#         super(MichalskiLoadNumValuationFunction, self).__init__()
#
#     def forward(self, z, a):
#         """
#         Args:
#         Args:
#             z (tensor): 2-d tensor (B * D), the object-centric representation.
#                 obj_prob + car_number + color + length + wall + roof + load + load_number
#                 [1,2,3,4, none,
#                  yellow, green, grey, red, blue,
#                  short, long, braced_wall, solid_wall,
#                  roof_foundation, solid_roof, braced_roof, peaked_roof,
#                  2_wheels, 3_wheels,
#                  box, golden_vase, barrel, diamond, metal_pot, oval_vase]
#             a (tensor): The one-hot tensor that is expanded to the batch size.
#
#         Returns:
#             A batch of probabilities.
#         """
#         z_color = z[:,21:26]
#         return (a * z_color).sum(dim=1)
