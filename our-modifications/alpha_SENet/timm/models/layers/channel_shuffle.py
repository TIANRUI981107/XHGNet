import torch as t


def channel_shuffle(x: t.Tensor, groups: int) -> t.Tensor:

    batch_size, num_channels, height, width = x.size()
    channel_per_group = t.div(num_channels, groups, rounding_mode='floor')

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, group, channels_per_group, height, width]
    x = x.view(batch_size, groups, channel_per_group, height, width)

    x = t.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x
