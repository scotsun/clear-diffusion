import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.distribution import IsotropicNormalDistribution
from .blocks import (
    build_downblock,
    build_upblock,
    build_midblock,
)


class Encoder(nn.Module):
    def __init__(
        self,
        channels,
        channel_multipliers: list[int],
        n_resnet_blocks: int,
        x_channels: int,
        z_channels: int,
        norm_channels: int,
        n_heads: int,
    ):
        super().__init__()

        self.channel_multipliers = channel_multipliers
        self.n_resolutions = len(channel_multipliers)

        self.conv_in = nn.Conv2d(x_channels, channels, kernel_size=3, padding=1)
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
        for _down in self.down:
            for resnet_block in _down.resnet_blocks:
                h = resnet_block(h)
            h = _down.downsample(h)
        h = self.mid.resnet_block1(h)
        h = self.mid.attention_block(h)
        h = self.mid.resnet_block2(h)
        h = self.norm(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        channels,
        channel_multipliers: list[int],
        n_resnet_blocks: int,
        x_channels: int,
        z_channels: int,
        norm_channels: int,
        n_heads: int,
    ):
        super().__init__()
        self.channel_multipliers = channel_multipliers
        self.n_resolutions = len(channel_multipliers)

        channels_list = [channels * m for m in channel_multipliers]
        channels = channels_list[-1]

        self.conv_in = nn.Conv2d(z_channels, channels, kernel_size=3, padding=1)

        self.mid = build_midblock(channels, norm_channels, n_heads)

        self.up = nn.ModuleList()
        for i in reversed(range(self.n_resolutions)):
            _up = build_upblock(
                in_channels=channels,
                out_channels=channels_list[i],
                n_resnet_blocks=n_resnet_blocks + 1,
                downsample=i > 0,
                norm_channels=norm_channels,
            )
            channels = channels_list[i]
            self.up.append(_up)

        self.norm = nn.GroupNorm(norm_channels, channels)
        self.conv_out = nn.Conv2d(channels, x_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor):
        z /= 0.18215
        h = self.conv_in(z)
        h = self.mid.resnet_block1(h)
        h = self.mid.attention_block(h)
        h = self.mid.resnet_block2(h)
        for _up in self.up:
            for resnet_block in _up.resnet_blocks:
                h = resnet_block(h)
            h = _up.upsample(h)
        h = self.norm(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h


class VAE(nn.Module):
    def __init__(
        self,
        channels: int,
        channel_multipliers: list[int],
        n_resnet_blocks: int,
        x_channels: int,
        z_channels: int,
        norm_channels: int,
        n_heads: int,
    ):
        super().__init__()
        self.encoder = Encoder(
            channels,
            channel_multipliers,
            n_resnet_blocks,
            x_channels,
            z_channels,
            norm_channels,
            n_heads,
        )
        self.decoder = Decoder(
            channels,
            channel_multipliers,
            n_resnet_blocks,
            x_channels,
            z_channels,
            norm_channels,
            n_heads,
        )

    def forward(self, x: torch.Tensor):
        moments = self.encoder(x)
        posterior = IsotropicNormalDistribution(moments)
        z = posterior.sample()
        z *= 0.18215
        # TODO: diffusion in z
        xhat = self.decoder(z)
        return xhat, posterior
