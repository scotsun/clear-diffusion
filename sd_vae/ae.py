import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ResidualBlock, AttentionBlock, Downsample


def build_downblock(
    in_channels: int,
    out_channels: int,
    n_resnet_blocks: int,
    downsample: bool,
    norm_channels: int,
) -> nn.Module:
    downblock = nn.Module()
    resnet_blocks = nn.ModuleList()
    for _ in range(n_resnet_blocks):
        resnet_blocks.append(ResidualBlock(in_channels, out_channels, norm_channels))
        in_channels = out_channels

    downblock.resnet_blocks = resnet_blocks
    if downsample:
        downblock.downsample = Downsample(out_channels)
    else:
        downblock.downsample = nn.Identity()

    return downblock


def build_midblock(channels: int, norm_channels: int, n_heads: int) -> nn.Module:
    midblock = nn.Module()
    midblock.resnet_block1 = ResidualBlock(channels, channels, norm_channels)
    midblock.attention_block = AttentionBlock(channels, norm_channels, n_heads)
    midblock.resnet_block2 = ResidualBlock(channels, channels, norm_channels)

    return midblock


class Encoder(nn.Module):
    def __init__(
        self,
        channels,
        channel_multipliers: list[int],
        n_resnet_blocks: int,
        in_channels: int,
        z_channels: int,
        norm_channels: int,
        n_heads: int,
    ):
        super().__init__()

        self.channels = channels
        self.channel_multipliers = channel_multipliers
        self.n_resolutions = len(channel_multipliers)

        self.conv_in = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        channels_list = [channels * m for m in [1] + channel_multipliers]

        self.down = nn.ModuleList()
        for i in range(self.n_resolutions):
            _down = build_downblock(
                in_channels=channels,
                out_channels=channels_list[i + 1],
                n_resnet_blocks=n_resnet_blocks,
                downsample=i < self.n_resolutions - 1,
                norm_channels=norm_channels,
            )
            channels = channels_list[i + 1]
            self.down.append(_down)

        self.mid = build_midblock(channels, norm_channels, n_heads)
        self.norm = nn.GroupNorm(norm_channels, channels)
        self.conv_out = nn.Conv2d(channels, 2 * z_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        h = self.conv_in(x)
        for down in self.down:
            for resnet_block in down.resnet_blocks:
                h = resnet_block(h)
            h = down.downsample(h)
            print(h.shape)
        h = self.mid.resnet_block1(h)
        h = self.mid.attention_block(h)
        h = self.mid.resnet_block2(h)
        h = self.norm(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h
