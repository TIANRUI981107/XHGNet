from thop import profile, clever_format

import torch
from torchvision.models import resnet152
from torchvision.models import resnet101
from torchvision.models import resnet50
from torchvision.models import mobilenet_v3_small
from torchvision.models import mobilenet_v3_large
from torchvision.models import alexnet
from torchvision.models import vgg16_bn
from torchvision.models import vgg19_bn
from torchvision.models import resnext50_32x4d
from torchvision.models import resnext101_32x8d
from torchvision.models import resnext101_64x4d
from torchvision.models import densenet121
from torchvision.models import densenet161
from torchvision.models import densenet169
from torchvision.models import densenet201
from torchvision.models import regnet_y_400mf
from torchvision.models import regnet_y_800mf
from torchvision.models import regnet_y_1_6gf
from torchvision.models import regnet_y_3_2gf
from torchvision.models import regnet_y_8gf
from torchvision.models import regnet_y_16gf
from torchvision.models import regnet_y_32gf
from torchvision.models import regnet_y_128gf
from torchvision.models import regnet_x_400mf
from torchvision.models import regnet_x_800mf
from torchvision.models import regnet_x_1_6gf
from torchvision.models import regnet_x_3_2gf
from torchvision.models import regnet_x_8gf
from torchvision.models import regnet_x_16gf
from torchvision.models import regnet_x_32gf
from torchvision.models import googlenet
from torchvision.models import shufflenet_v2_x0_5
from torchvision.models import shufflenet_v2_x1_0
from torchvision.models import shufflenet_v2_x1_5
from torchvision.models import shufflenet_v2_x2_0
from torchvision.models import squeezenet1_0
from torchvision.models import squeezenet1_1
from torchvision.models import efficientnet_v2_s
from torchvision.models import efficientnet_v2_m
from torchvision.models import efficientnet_v2_l

from mobilevit import mobile_vit_xx_small
from mobilevit import mobile_vit_x_small
from mobilevit import mobile_vit_small
from convnext_impl import convnext_tiny
from convnext_impl import convnext_small
from convnext_impl import convnext_base
from convnext_impl import convnext_large
from convnext_impl import convnext_xlarge
from swin_transformer_imple import swin_base_patch4_window12_384


def create_torch_model(model_name: str, num_classes: int = 68):
    model_dic = {
        "alexnet_p": alexnet(num_classes=num_classes, weights=None),
        "alexnet_np": alexnet(num_classes=num_classes, weights=None),
        "vgg16_bn": vgg16_bn(num_classes=num_classes, weights=None),
        "vgg19_bn": vgg19_bn(num_classes=num_classes, weights=None),
        "resnet50": resnet50(num_classes=num_classes, weights=None),
        "resnet101": resnet101(num_classes=num_classes, weights=None),
        "resnet152": resnet152(num_classes=num_classes, weights=None),
        "resnext50_32x4d": resnext50_32x4d(num_classes=num_classes, weights=None),
        "resnext101_32x8d": resnext101_32x8d(num_classes=num_classes, weights=None),
        "resnext101_64x4d": resnext101_64x4d(num_classes=num_classes, weights=None),
        "densenet121": densenet121(num_classes=num_classes, weights=None),
        "densenet161": densenet161(num_classes=num_classes, weights=None),
        "densenet169": densenet169(num_classes=num_classes, weights=None),
        "densenet201": densenet201(num_classes=num_classes, weights=None),
        "mobilenet_v3_small": mobilenet_v3_small(num_classes=num_classes, weights=None),
        "mobilenet_v3_large": mobilenet_v3_large(num_classes=num_classes, weights=None),
        "shufflenet_v2_x0_5": shufflenet_v2_x0_5(num_classes=num_classes, weights=None),
        "shufflenet_v2_x1_0": shufflenet_v2_x1_0(num_classes=num_classes, weights=None),
        "shufflenet_v2_x1_5": shufflenet_v2_x1_5(num_classes=num_classes, weights=None),
        "shufflenet_v2_x2_0": shufflenet_v2_x2_0(num_classes=num_classes, weights=None),
        "squeezenet1_0": squeezenet1_0(num_classes=num_classes, weights=None),
        "squeezenet1_1": squeezenet1_1(num_classes=num_classes, weights=None),
        "efficientnet_v2_s": efficientnet_v2_s(num_classes=num_classes, weights=None),
        "efficientnet_v2_m": efficientnet_v2_m(num_classes=num_classes, weights=None),
        "efficientnet_v2_l": efficientnet_v2_l(num_classes=num_classes, weights=None),
        "regnet_y_400mf": regnet_y_400mf(num_classes=num_classes, weights=None),
        "regnet_y_800mf": regnet_y_800mf(num_classes=num_classes, weights=None),
        "regnet_y_1_6gf": regnet_y_1_6gf(num_classes=num_classes, weights=None),
        "regnet_y_3_2gf": regnet_y_3_2gf(num_classes=num_classes, weights=None),
        "regnet_y_8gf": regnet_y_8gf(num_classes=num_classes, weights=None),
        "regnet_y_16gf": regnet_y_16gf(num_classes=num_classes, weights=None),
        "regnet_y_32gf": regnet_y_32gf(num_classes=num_classes, weights=None),
        "regnet_y_128gf": regnet_y_128gf(num_classes=num_classes, weights=None),
        "regnet_x_400mf": regnet_x_400mf(num_classes=num_classes, weights=None),
        "regnet_x_800mf": regnet_x_800mf(num_classes=num_classes, weights=None),
        "regnet_x_1_6gf": regnet_x_1_6gf(num_classes=num_classes, weights=None),
        "regnet_x_3_2gf": regnet_x_3_2gf(num_classes=num_classes, weights=None),
        "regnet_x_8gf": regnet_x_8gf(num_classes=num_classes, weights=None),
        "regnet_x_16gf": regnet_x_16gf(num_classes=num_classes, weights=None),
        "regnet_x_32gf": regnet_x_32gf(num_classes=num_classes, weights=None),
        "mobile_vit_xx_small": mobile_vit_xx_small(num_classes=num_classes),
        "mobile_vit_x_small": mobile_vit_x_small(num_classes=num_classes),
        "mobile_vit_small": mobile_vit_small(num_classes=num_classes),
        "convnext_tiny": convnext_tiny(num_classes=num_classes),
        "convnext_small": convnext_small(num_classes=num_classes),
        "convnext_base": convnext_base(num_classes=num_classes),
        "convnext_large": convnext_large(num_classes=num_classes),
        "convnext_xlarge": convnext_xlarge(num_classes=num_classes),
        "swin_base_patch4_window12_384": swin_base_patch4_window12_384(num_classes=num_classes),
    }
    net = model_dic[model_name]
    return model_name, net


if __name__ == "__main__":

    # config

    # device = torch.device('cpu')
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    num_classes = 68
    batch_size = 1
    model_zoo = [
        "alexnet_p",
        "alexnet_np",
        "vgg16_bn",
        "vgg19_bn",
        "resnet50",
        "resnet101",
        "resnet152",
        "resnext50_32x4d",
        "resnext101_32x8d",
        "resnext101_64x4d",
        "densenet121",
        "densenet161",
        "densenet169",
        "densenet201",
        "mobilenet_v3_small",
        "mobilenet_v3_large",
        "shufflenet_v2_x0_5",
        "shufflenet_v2_x1_0",
        "shufflenet_v2_x1_5",
        "shufflenet_v2_x2_0",
        "squeezenet1_0",
        "squeezenet1_1",
        "efficientnet_v2_s",
        "efficientnet_v2_m",
        "efficientnet_v2_l",
        "regnet_y_400mf", 
        "regnet_y_800mf", 
        "regnet_y_1_6gf", 
        "regnet_y_3_2gf", 
        "regnet_y_8gf", 
        "regnet_y_16gf", 
        "regnet_y_32gf", 
        "regnet_y_128gf", 
        "regnet_x_400mf", 
        "regnet_x_800mf", 
        "regnet_x_1_6gf", 
        "regnet_x_3_2gf", 
        "regnet_x_8gf", 
        "regnet_x_16gf", 
        "regnet_x_32gf", 
        "mobile_vit_xx_small", 
        "mobile_vit_x_small", 
        "mobile_vit_small", 
        "convnext_tiny", 
        "convnext_small", 
        "convnext_base", 
        "convnext_large", 
        "convnext_xlarge", 
        "swin_base_patch4_window12_384", 
    ]

    # torchhub models
    for _ in model_zoo:
        m, model = create_torch_model(model_name=_)
        # print(model)

        # Calculate the complexity of models
        my_input = torch.zeros((batch_size, 3, 224, 224)).to(device)
        flops, params = profile(model.to(device), inputs=(my_input,))
        flops, params = clever_format([flops, params], "%.2f")
        msg = f"{m}: num_classes={num_classes}, Params: {params}, FLOPs: {flops}"
        print(msg)
        with open("model_profile.txt", "a+") as af:
            af.write(f"{msg}\n")

    # our imple. models

    # in_features = model.fc.in_features
    # model.fc = torch.nn.Linear(in_features, num_classes)

    # in_features = model.classifier[-1].in_features
    # model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    # print(model)
