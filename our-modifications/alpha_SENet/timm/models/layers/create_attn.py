""" Attention Factory

Hacked together by / Copyright 2021 Ross Wightman
"""
import torch
from functools import partial

from .bottleneck_attn import BottleneckAttn
from .cbam import CbamModule, LightCbamModule
from .eca import EcaModule, CecaModule
from .gather_excite import GatherExcite
from .global_context import GlobalContext
from .halo_attn import HaloAttn
from .lambda_layer import LambdaLayer
from .non_local_attn import NonLocalAttn, BatNonLocalAttn
from .selective_kernel import SelectiveKernel
from .split_attn import SplitAttn
from .squeeze_excite import SEModule, EffectiveSEModule
from .shuffle_squeeze_excite import ShuffleSEModule


def get_attn(attn_type):
    if isinstance(attn_type, torch.nn.Module):
        return attn_type
    module_cls = None
    if attn_type:
        if isinstance(attn_type, str):
            attn_type = attn_type.lower()
            # Lightweight attention modules (channel and/or coarse spatial).
            # Typically added to existing network architecture blocks in addition to existing convolutions.
            if attn_type == "se":
                module_cls = SEModule
            elif attn_type == "ese":
                module_cls = EffectiveSEModule
            elif attn_type == "eca":
                module_cls = EcaModule
            elif attn_type == "ecam":
                module_cls = partial(EcaModule, use_mlp=True)
            elif attn_type == "ceca":
                module_cls = CecaModule
            elif attn_type == "ge":
                module_cls = GatherExcite
            elif attn_type == "gc":
                module_cls = GlobalContext
            elif attn_type == "gca":
                module_cls = partial(GlobalContext, fuse_add=True, fuse_scale=False)
            elif attn_type == "cbam":
                module_cls = CbamModule
            elif attn_type == "lcbam":
                module_cls = LightCbamModule

            # SSEModule with abalation studies
            elif attn_type == "sse_rd101_g08":
                module_cls = partial(ShuffleSEModule, rd_channels=False, groups=8)
            elif attn_type == "sse_rd101_g16":
                module_cls = partial(ShuffleSEModule, rd_channels=False, groups=16)
            elif attn_type == "sse_rd101_g32":
                module_cls = partial(ShuffleSEModule, rd_channels=False, groups=32)
            elif attn_type == "sse_rd101_g64":
                module_cls = partial(ShuffleSEModule, rd_channels=False, groups=64)
            elif attn_type == "sse_rd101_ada":
                module_cls = partial(ShuffleSEModule, rd_channels=False, groups=None)
            # SSE_rd102
            elif attn_type == "sse_rd102_g08":
                module_cls = partial(
                    ShuffleSEModule, rd_channels=True, rd_ratio=1.0 / 2, groups=8
                )
            elif attn_type == "sse_rd102_g16":
                module_cls = partial(
                    ShuffleSEModule, rd_channels=True, rd_ratio=1.0 / 2, groups=16
                )
            elif attn_type == "sse_rd102_g32":
                module_cls = partial(
                    ShuffleSEModule, rd_channels=True, rd_ratio=1.0 / 2, groups=32
                )
            elif attn_type == "sse_rd102_g64":
                module_cls = partial(
                    ShuffleSEModule, rd_channels=True, rd_ratio=1.0 / 2, groups=64
                )
            elif attn_type == "sse_rd102_ada":
                module_cls = partial(
                    ShuffleSEModule, rd_channels=True, rd_ratio=1.0 / 2, groups=None
                )
            # SSE_rd104
            elif attn_type == "sse_rd104_g08":
                module_cls = partial(
                    ShuffleSEModule, rd_channels=True, rd_ratio=1.0 / 4, groups=8
                )
            elif attn_type == "sse_rd104_g16":
                module_cls = partial(
                    ShuffleSEModule, rd_channels=True, rd_ratio=1.0 / 4, groups=16
                )
            elif attn_type == "sse_rd104_g32":
                module_cls = partial(
                    ShuffleSEModule, rd_channels=True, rd_ratio=1.0 / 4, groups=32
                )
            elif attn_type == "sse_rd104_g64":
                module_cls = partial(
                    ShuffleSEModule, rd_channels=True, rd_ratio=1.0 / 4, groups=64
                )
            elif attn_type == "sse_rd104_ada":
                module_cls = partial(
                    ShuffleSEModule, rd_channels=True, rd_ratio=1.0 / 4, groups=None
                )
            # SSE_rd108
            elif attn_type == "sse_rd108_g08":
                module_cls = partial(
                    ShuffleSEModule, rd_channels=True, rd_ratio=1.0 / 8, groups=8
                )
            elif attn_type == "sse_rd108_g16":
                module_cls = partial(
                    ShuffleSEModule, rd_channels=True, rd_ratio=1.0 / 8, groups=16
                )
            elif attn_type == "sse_rd108_g32":
                module_cls = partial(
                    ShuffleSEModule, rd_channels=True, rd_ratio=1.0 / 8, groups=32
                )
            elif attn_type == "sse_rd108_g64":
                module_cls = partial(
                    ShuffleSEModule, rd_channels=True, rd_ratio=1.0 / 8, groups=64
                )
            elif attn_type == "sse_rd108_ada":
                module_cls = partial(
                    ShuffleSEModule, rd_channels=True, rd_ratio=1.0 / 8, groups=None
                )
            # SSE_rd116
            elif attn_type == "sse_rd116_g08":
                module_cls = partial(
                    ShuffleSEModule, rd_channels=True, rd_ratio=1.0 / 16, groups=8
                )
            elif attn_type == "sse_rd116_g16":
                module_cls = partial(
                    ShuffleSEModule, rd_channels=True, rd_ratio=1.0 / 16, groups=16
                )
            elif attn_type == "sse_rd116_g32":
                module_cls = partial(
                    ShuffleSEModule, rd_channels=True, rd_ratio=1.0 / 16, groups=32
                )
            elif attn_type == "sse_rd116_g64":
                module_cls = partial(
                    ShuffleSEModule, rd_channels=True, rd_ratio=1.0 / 16, groups=64
                )
            elif attn_type == "sse_rd116_ada":
                module_cls = partial(
                    ShuffleSEModule, rd_channels=True, rd_ratio=1.0 / 16, groups=None
                )
            # SSE_rd132
            elif attn_type == "sse_rd132_g08":
                module_cls = partial(
                    ShuffleSEModule, rd_channels=True, rd_ratio=1.0 / 32, groups=8
                )
            elif attn_type == "sse_rd132_g16":
                module_cls = partial(
                    ShuffleSEModule, rd_channels=True, rd_ratio=1.0 / 32, groups=16
                )
            elif attn_type == "sse_rd132_g32":
                module_cls = partial(
                    ShuffleSEModule, rd_channels=True, rd_ratio=1.0 / 32, groups=32
                )
            elif attn_type == "sse_rd132_g64":
                module_cls = partial(
                    ShuffleSEModule, rd_channels=True, rd_ratio=1.0 / 32, groups=64
                )
            elif attn_type == "sse_rd132_ada":
                module_cls = partial(
                    ShuffleSEModule, rd_channels=True, rd_ratio=1.0 / 32, groups=None
                )
            # SE_vanilla
            elif attn_type == "sse_rd101_g01":
                module_cls = partial(
                    ShuffleSEModule,
                    rd_channels=False,
                    groups=1,
                    use_channel_shuffle=False,
                )
            elif attn_type == "sse_rd102_g01":
                module_cls = partial(
                    ShuffleSEModule,
                    rd_channels=True,
                    rd_ratio=1.0 / 2,
                    groups=1,
                    use_channel_shuffle=False,
                )
            elif attn_type == "sse_rd104_g01":
                module_cls = partial(
                    ShuffleSEModule,
                    rd_channels=True,
                    rd_ratio=1.0 / 4,
                    groups=1,
                    use_channel_shuffle=False,
                )
            elif attn_type == "sse_rd108_g01":
                module_cls = partial(
                    ShuffleSEModule,
                    rd_channels=True,
                    rd_ratio=1.0 / 8,
                    groups=1,
                    use_channel_shuffle=False,
                )
            elif attn_type == "sse_rd116_g01":
                module_cls = partial(
                    ShuffleSEModule,
                    rd_channels=True,
                    rd_ratio=1.0 / 16,
                    groups=1,
                    use_channel_shuffle=False,
                )
            elif attn_type == "sse_rd132_g01":
                module_cls = partial(
                    ShuffleSEModule,
                    rd_channels=True,
                    rd_ratio=1.0 / 32,
                    groups=1,
                    use_channel_shuffle=False,
                )

            # Nogrouping shuffle
            elif attn_type == "sse_rd101_nog":
                module_cls = partial(
                    ShuffleSEModule,
                    rd_channels=False,
                    groups=None,
                    use_channel_shuffle=True,
                )
            elif attn_type == "sse_rd102_nog":
                module_cls = partial(
                    ShuffleSEModule,
                    rd_channels=True,
                    rd_ratio=1.0 / 2,
                    groups=None,
                    use_channel_shuffle=True,
                )
            elif attn_type == "sse_rd104_nog":
                module_cls = partial(
                    ShuffleSEModule,
                    rd_channels=True,
                    rd_ratio=1.0 / 4,
                    groups=None,
                    use_channel_shuffle=True,
                )
            elif attn_type == "sse_rd108_nog":
                module_cls = partial(
                    ShuffleSEModule,
                    rd_channels=True,
                    rd_ratio=1.0 / 8,
                    groups=None,
                    use_channel_shuffle=True,
                )
            elif attn_type == "sse_rd116_nog":
                module_cls = partial(
                    ShuffleSEModule,
                    rd_channels=True,
                    rd_ratio=1.0 / 16,
                    groups=None,
                    use_channel_shuffle=True,
                )
            elif attn_type == "sse_rd132_nog":
                module_cls = partial(
                    ShuffleSEModule,
                    rd_channels=True,
                    rd_ratio=1.0 / 32,
                    groups=None,
                    use_channel_shuffle=True,
                )

            # Noshuffle GConv: set groups=1 in the source code
            elif attn_type == "sse_rd101_nos":
                module_cls = partial(
                    ShuffleSEModule,
                    rd_channels=False,
                    groups=None,
                    use_channel_shuffle=False,
                )
            elif attn_type == "sse_rd102_nos":
                module_cls = partial(
                    ShuffleSEModule,
                    rd_channels=True,
                    rd_ratio=1.0 / 2,
                    groups=None,
                    use_channel_shuffle=False,
                )
            elif attn_type == "sse_rd104_nos":
                module_cls = partial(
                    ShuffleSEModule,
                    rd_channels=True,
                    rd_ratio=1.0 / 4,
                    groups=None,
                    use_channel_shuffle=False,
                )
            elif attn_type == "sse_rd108_nos":
                module_cls = partial(
                    ShuffleSEModule,
                    rd_channels=True,
                    rd_ratio=1.0 / 8,
                    groups=None,
                    use_channel_shuffle=False,
                )
            elif attn_type == "sse_rd116_nos":
                module_cls = partial(
                    ShuffleSEModule,
                    rd_channels=True,
                    rd_ratio=1.0 / 16,
                    groups=None,
                    use_channel_shuffle=False,
                )
            elif attn_type == "sse_rd132_nos":
                module_cls = partial(
                    ShuffleSEModule,
                    rd_channels=True,
                    rd_ratio=1.0 / 32,
                    groups=None,
                    use_channel_shuffle=False,
                )

            # Attention / attention-like modules w/ significant params
            # Typically replace some of the existing workhorse convs in a network architecture.
            # All of these accept a stride argument and can spatially downsample the input.
            elif attn_type == "sk":
                module_cls = SelectiveKernel
            elif attn_type == "splat":
                module_cls = SplitAttn

            # Self-attention / attention-like modules w/ significant compute and/or params
            # Typically replace some of the existing workhorse convs in a network architecture.
            # All of these accept a stride argument and can spatially downsample the input.
            elif attn_type == "lambda":
                return LambdaLayer
            elif attn_type == "bottleneck":
                return BottleneckAttn
            elif attn_type == "halo":
                return HaloAttn
            elif attn_type == "nl":
                module_cls = NonLocalAttn
            elif attn_type == "bat":
                module_cls = BatNonLocalAttn

            # Woops!
            else:
                assert False, "Invalid attn module (%s)" % attn_type
        elif isinstance(attn_type, bool):
            if attn_type:
                module_cls = SEModule
        else:
            module_cls = attn_type
    return module_cls


def create_attn(attn_type, channels, **kwargs):
    module_cls = get_attn(attn_type)
    if module_cls is not None:
        # NOTE: it's expected the first (positional) argument of all attention layers is the # input channels
        return module_cls(channels, **kwargs)
    return None
