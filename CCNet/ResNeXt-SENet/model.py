import torch.nn as nn
import torch


class BasicBlock(nn.Module):
    """
    Basic block for ResNet18 and ResNet34.
    """

    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    BottleNeck block for ResNet50, ResNet101 and ResNet152.

    Bottleneck in torchvision places the stride=2 for downsampling at 3x3 convolution(self.conv2)
    while original implementation places the stride=2 at the first 1x1 convolution(self.conv1)
    according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    This variant is also known as ResNet V1.5 and improves accuracy(~0.5%) according to
    https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    """

    expansion = 4  # channel expansion factor

    def __init__(
        self,
        in_channel,
        out_channel,
        stride=1,
        downsample=None,
        groups=1,
        width_per_group=64,
    ):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.0)) * groups

        self.conv1 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=width,
            kernel_size=1,
            stride=1,
            bias=False,
        )  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(
            in_channels=width,
            out_channels=width,
            groups=groups,
            kernel_size=3,
            stride=stride,
            bias=False,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(
            in_channels=width,
            out_channels=out_channel * self.expansion,
            kernel_size=1,
            stride=1,
            bias=False,
        )  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet Network.

    args:
        block: [nn.module, ], Basicblock OR Bottleneck
        block_num: [list, ], number of blocks at every downsampling stage
        num_classes: [int, ], number of classes
        include_top: [bool, ], xx
        groups: [int, ], groups of Group Conv, e.g. 32, 64
        width_per_group: [int, ], "neck channel" of bottleneck
    """

    def __init__(
        self,
        block,
        blocks_num,
        num_classes=1000,
        include_top=True,
        groups=1,
        width_per_group=64,
        zero_init_residual: bool = True,
    ):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(
            3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, channel, block_num, stride=1):
        """
        make conv_x_x layers in ResNet.

        args:
            block: [nn.module,], type of blocks, e.g. basicblock, bottleneck
            channel: [int,], number of channels in conv1
            block_num: number of blocks
            stride:
        """
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channel,
                    channel * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(channel * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.in_channel,
                channel,
                downsample=downsample,
                stride=stride,
                groups=self.groups,
                width_per_group=self.width_per_group,
            )
        )
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(
                block(
                    self.in_channel,
                    channel,
                    groups=self.groups,
                    width_per_group=self.width_per_group,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet18(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet18-f37072fd.pth
    return ResNet(
        BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top
    )


def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-b627a593.pth
    return ResNet(
        BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top
    )


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-0676ba61.pth
    return ResNet(
        Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top
    )


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-63fe2227.pth
    return ResNet(
        Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top
    )


def resnet152(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet152-394f9c45.pth
    return ResNet(
        Bottleneck, [3, 8, 36, 3], num_classes=num_classes, include_top=include_top
    )


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        num_classes=num_classes,
        include_top=include_top,
        groups=groups,
        width_per_group=width_per_group,
    )


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(
        Bottleneck,
        [3, 4, 23, 3],
        num_classes=num_classes,
        include_top=include_top,
        groups=groups,
        width_per_group=width_per_group,
    )


def resnext101_64x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_64x4d-173b62eb.pth
    groups = 64
    width_per_group = 4
    return ResNet(
        Bottleneck,
        [3, 4, 23, 3],
        num_classes=num_classes,
        include_top=include_top,
        groups=groups,
        width_per_group=width_per_group,
    )


if __name__ == "__main__":

    import torch as t

    from torchinfo import summary

    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    inputs = torch.zeros((8, 3, 224, 224), device=device)

    model = resnext101_64x4d(num_classes=1000)
    model.to(device=device)

    outputs = model(inputs)

    summary(model, input_size=(1, 3, 224, 224), verbose=1)

    print("end")
