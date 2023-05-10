from thop import profile, clever_format

import torch

from timm.models.resnet import (
    resnet50,
    resnet101,
    resnet152,
    resnext50_32x4d,
    resnext101_32x4d,
    resnext101_32x8d,
    resnext101_64x4d,
)
from timm.models.densenet import densenet121, densenet161, densenet169, densenet201
from timm.models.efficientnet import efficientnetv2_s, efficientnetv2_m
from timm.models.regnet import (
    regnety_016,
    regnety_032,
    regnety_040,
    regnety_064,
    regnety_080,
    regnety_120,
)
from timm.models.regnet import (
    regnetx_016,
    regnetx_032,
    regnetx_040,
    regnetx_064,
    regnetx_080,
    regnetx_120,
)
from timm.models.convnext import convnext_tiny, convnext_small, convnext_base
from timm.models.mobilenetv3 import mobilenetv3_large_075, mobilenetv3_large_100
from timm.models.resnet import (
    seresnet50,
    eseresnet50,
    ecaresnet50,
    ecamresnet50,
    cecaresnet50,
    geresnet50,
    gcresnet50,
    gcaresnet50,
    cbamresnet50,
    lcbamresnet50,
    seresnet101,
    eseresnet101,
    ecaresnet101,
    ecamresnet101,
    cecaresnet101,
    geresnet101,
    gcresnet101,
    gcaresnet101,
    cbamresnet101,
    lcbamresnet101,
    seresnet152,
    eseresnet152,
    ecaresnet152,
    ecamresnet152,
    cecaresnet152,
    geresnet152,
    gcresnet152,
    gcaresnet152,
    cbamresnet152,
    lcbamresnet152,
)
from timm.models.resnet import sse_rd116_ada_resnet101dd, sse_rd116_ada_resnet152dd


def create_torch_model(model_name: str, num_classes: int = 11):
    model_dic = {
        "resnet50": resnet50(num_classes=num_classes, pretrained=False),
        "resnet101": resnet101(num_classes=num_classes, pretrained=False),
        "resnet152": resnet152(num_classes=num_classes, pretrained=False),
        "resnext50_32x4d": resnext50_32x4d(num_classes=num_classes, pretrained=False),
        "resnext101_32x4d": resnext101_32x4d(num_classes=num_classes, pretrained=False),
        "resnext101_32x8d": resnext101_32x8d(num_classes=num_classes, pretrained=False),
        "resnext101_64x4d": resnext101_64x4d(num_classes=num_classes, pretrained=False),
        "densenet121": densenet121(num_classes=num_classes, pretrained=False),
        "densenet161": densenet161(num_classes=num_classes, pretrained=False),
        "densenet169": densenet169(num_classes=num_classes, pretrained=False),
        "densenet201": densenet201(num_classes=num_classes, pretrained=False),
        "efficientnetv2_s": efficientnetv2_s(num_classes=num_classes, pretrained=False),
        "efficientnetv2_m": efficientnetv2_m(num_classes=num_classes, pretrained=False),
        "regnety_016": regnety_016(num_classes=num_classes, pretrained=False),
        "regnety_032": regnety_032(num_classes=num_classes, pretrained=False),
        "regnety_040": regnety_040(num_classes=num_classes, pretrained=False),
        "regnety_064": regnety_064(num_classes=num_classes, pretrained=False),
        "regnety_080": regnety_080(num_classes=num_classes, pretrained=False),
        "regnety_120": regnety_120(num_classes=num_classes, pretrained=False),
        "regnetx_016": regnetx_016(num_classes=num_classes, pretrained=False),
        "regnetx_032": regnetx_032(num_classes=num_classes, pretrained=False),
        "regnetx_040": regnetx_040(num_classes=num_classes, pretrained=False),
        "regnetx_064": regnetx_064(num_classes=num_classes, pretrained=False),
        "regnetx_080": regnetx_080(num_classes=num_classes, pretrained=False),
        "regnetx_120": regnetx_120(num_classes=num_classes, pretrained=False),
        "convnext_tiny": convnext_tiny(num_classes=num_classes, pretrained=False),
        "convnext_small": convnext_small(num_classes=num_classes, pretrained=False),
        "convnext_base": convnext_base(num_classes=num_classes, pretrained=False),
        "mobilenetv3_large_075": mobilenetv3_large_075(
            num_classes=num_classes, pretrained=False
        ),
        "mobilenetv3_large_100": mobilenetv3_large_100(
            num_classes=num_classes, pretrained=False
        ),
        "seresnet50": seresnet50(num_classes=num_classes, pretrained=False),
        "eseresnet50": eseresnet50(num_classes=num_classes, pretrained=False),
        "ecaresnet50": ecaresnet50(num_classes=num_classes, pretrained=False),
        "ecamresnet50": ecamresnet50(num_classes=num_classes, pretrained=False),
        "cecaresnet50": cecaresnet50(num_classes=num_classes, pretrained=False),
        "geresnet50": geresnet50(num_classes=num_classes, pretrained=False),
        "gcresnet50": gcresnet50(num_classes=num_classes, pretrained=False),
        "gcaresnet50": gcaresnet50(num_classes=num_classes, pretrained=False),
        "cbamresnet50": cbamresnet50(num_classes=num_classes, pretrained=False),
        "lcbamresnet50": lcbamresnet50(num_classes=num_classes, pretrained=False),
        "seresnet101": seresnet101(num_classes=num_classes, pretrained=False),
        "eseresnet101": eseresnet101(num_classes=num_classes, pretrained=False),
        "ecaresnet101": ecaresnet101(num_classes=num_classes, pretrained=False),
        "ecamresnet101": ecamresnet101(num_classes=num_classes, pretrained=False),
        "cecaresnet101": cecaresnet101(num_classes=num_classes, pretrained=False),
        "geresnet101": geresnet101(num_classes=num_classes, pretrained=False),
        "gcresnet101": gcresnet101(num_classes=num_classes, pretrained=False),
        "gcaresnet101": gcaresnet101(num_classes=num_classes, pretrained=False),
        "cbamresnet101": cbamresnet101(num_classes=num_classes, pretrained=False),
        "lcbamresnet101": lcbamresnet101(num_classes=num_classes, pretrained=False),
        "seresnet152": seresnet152(num_classes=num_classes, pretrained=False),
        "eseresnet152": eseresnet152(num_classes=num_classes, pretrained=False),
        "ecaresnet152": ecaresnet152(num_classes=num_classes, pretrained=False),
        "ecamresnet152": ecamresnet152(num_classes=num_classes, pretrained=False),
        "cecaresnet152": cecaresnet152(num_classes=num_classes, pretrained=False),
        "geresnet152": geresnet152(num_classes=num_classes, pretrained=False),
        "gcresnet152": gcresnet152(num_classes=num_classes, pretrained=False),
        "gcaresnet152": gcaresnet152(num_classes=num_classes, pretrained=False),
        "cbamresnet152": cbamresnet152(num_classes=num_classes, pretrained=False),
        "lcbamresnet152": lcbamresnet152(num_classes=num_classes, pretrained=False),
        "sse_rd116_ada_resnet101dd": sse_rd116_ada_resnet101dd(
            num_classes=num_classes, pretrained=False
        ),
        "sse_rd116_ada_resnet152dd": sse_rd116_ada_resnet152dd(
            num_classes=num_classes, pretrained=False
        ),
    }
    net = model_dic[model_name]
    return model_name, net


if __name__ == "__main__":

    # config

    # device = torch.device('cpu')
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    num_classes = 11
    batch_size = 1
    model_zoo = [
        "mobilenetv3_large_075",
        "mobilenetv3_large_100",
    ]

    attn_zoo = [
        "seresnet50",
        "eseresnet50",
        "ecaresnet50",
        "ecamresnet50",
        "cecaresnet50",
        "geresnet50",
        "gcresnet50",
        "gcaresnet50",
        "cbamresnet50",
        "lcbamresnet50",
        "seresnet101",
        "eseresnet101",
        "ecaresnet101",
        "ecamresnet101",
        "cecaresnet101",
        "geresnet101",
        "gcresnet101",
        "gcaresnet101",
        "cbamresnet101",
        "lcbamresnet101",
        "seresnet152",
        "eseresnet152",
        "ecaresnet152",
        "ecamresnet152",
        "cecaresnet152",
        "geresnet152",
        "gcresnet152",
        "gcaresnet152",
        "cbamresnet152",
        "lcbamresnet152",
    ]

    modified_zoo = [
        "resnext101_32x4d",
    ]

    addition_zoo = [
        "sseresnet50",
        "sseresnet101",
        "sseresnet152",
    ]

    ablation = ["sse_rd116_ada_resnet101dd", "sse_rd116_ada_resnet152dd"]

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
        with open("ap-mobile.txt", "a+") as af:
            af.write(f"{msg}\n")

    # our imple. models

    # in_features = model.fc.in_features
    # model.fc = torch.nn.Linear(in_features, num_classes)

    # in_features = model.classifier[-1].in_features
    # model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    # print(model)
