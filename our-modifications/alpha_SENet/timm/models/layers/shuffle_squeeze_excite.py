"""
SSE module.

ShuffleSEModule is a modified version of SEModule and ECA-Net.

According to ECA-Net, block matrix, implemented by group convolutions, is not as
efficient as conv1d because it only involves channels interactiion within groups.

Inspired by ShuffleNet, this module hereby shuffles the channels among different
groups to facilatate channel interactions at the expense of increase the number of
parameters compared with ECA-Net.

However, extensive experiments show that SSEModule is more effective than ECA-Net,
with the help of channels interaction operation.

This script includes abalation studies:
* SEModule
* GconvSEModule with best rd_ratio
* ShuffleSEModule with best rd_ratio

and trade-off between rd_ratio and groups:
"""
from torch import nn
import torch.nn.functional as F

from .create_act import create_act_layer
from .helpers import make_divisible
from .channel_shuffle import channel_shuffle


class ShuffleSEModule(nn.Module):
    """Constructs a SSE module."""

    def __init__(
        self,
        channels,
        rd_channels=True,
        rd_ratio=1.0 / 16,  # reduce ratio: # 1/1; 1/2; 1/4; 1/8; 1/16; 1/32
        groups=None,  # groups {08, 16, 32, 64, None}
        use_channel_shuffle=True,
        alpha=4,
        rd_divisor=8,
        add_maxpool=True,  # default set True
        bias=False,  # according to SE original paper, bias set to False
        act_layer=None,
        norm_layer=nn.BatchNorm2d,  # use BN layer instead
        gate_layer="hard_sigmoid",
    ):
        super(ShuffleSEModule, self).__init__()

        self.add_maxpool = add_maxpool
        self.use_channel_shuffle = use_channel_shuffle

        # groups
        if channels is not None:
            if groups is None:
                t = int(abs(channels / (8 * alpha)))
                groups = min(int(make_divisible(t, rd_divisor, round_limit=0)), 64)
            else:
                assert isinstance(groups, int)
                groups = groups

        # reduce channels
        if rd_channels:
            assert channels is not None
            rd_channels = make_divisible(
                channels * rd_ratio, rd_divisor, round_limit=0.0
            )
            act_layer = act_layer or nn.ReLU
            self.fc1 = nn.Conv2d(
                channels, rd_channels, kernel_size=1, groups=groups, bias=bias
            )
            self.bn = norm_layer(rd_channels) if norm_layer else nn.Identity()
            self.act = create_act_layer(act_layer)
            self.fc2 = nn.Conv2d(
                rd_channels, channels, kernel_size=1, groups=groups, bias=bias
            )
        else:
            act_layer = act_layer or nn.ReLU
            self.fc1 = nn.Conv2d(
                channels, channels, kernel_size=1, groups=groups, bias=bias
            )
            self.bn = norm_layer(channels) if norm_layer else nn.Identity()
            self.act = create_act_layer(act_layer)
            self.fc2 = nn.Conv2d(
                channels, channels, kernel_size=1, groups=groups, bias=bias
            )

        # channel shffle
        if use_channel_shuffle:
            self.channel_shuffle = channel_shuffle
            self.groups = groups
        else:
            self.channel_shuffle = None
            self.groups = None

        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        if self.use_channel_shuffle:
            x_se = self.channel_shuffle(x=x_se, groups=self.groups)
        x_se = self.act(self.bn(x_se))
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)
