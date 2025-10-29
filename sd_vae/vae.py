import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ResidualBlock, AttentionBlock


class Encoder(nn.Sequential):
    def __init__(self, norm_channels, num_heads):
        super().__init__(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            # (batch_size, 3, height, width) -> (batch_size, 128, height, width)
            ResidualBlock(128, 128, norm_channels),
            ResidualBlock(128, 128, norm_channels),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            # (batch_size, 128, height, width) -> (batch_size, 128, height // 2, width // 2)
            ResidualBlock(128, 256, norm_channels),
            ResidualBlock(256, 256, norm_channels),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            # (batch_size, 256, height // 2, width // 2) -> (batch_size, 256, height // 4, width // 4)
            ResidualBlock(256, 512, norm_channels),
            ResidualBlock(512, 512, norm_channels),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            # (batch_size, 512, height // 4, width // 4) -> (batch_size, 512, height // 8, width // 8)
            ResidualBlock(512, 512, norm_channels),
            ResidualBlock(512, 512, norm_channels),
            # (batch_size, 512, height // 8, width // 8) -> (batch_size, 512, height // 8, width // 8)
            ResidualBlock(512, 512, norm_channels),
            AttentionBlock(512, norm_channels, num_heads),
            ResidualBlock(512, 512, norm_channels),
            nn.GroupNorm(norm_channels, 512),
            nn.SiLU(),
            # (batch_size, 512, height // 8, width // 8) -> (batch_size, 8, height // 8, width // 8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x):
        for module in self:
            if getattr(module, "stride", None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        # hidden state x: (batch_size, 8, height // 8, width // 8)
        mu, log_var = torch.chunk(x, 2, dim=1)
        log_var = torch.clamp(log_var, -30.0, 20.0)
        z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
        z *= 0.18215  # scale factor s.t. flattened z ~ N(0, I)
        return z


class Decoder(nn.Sequential):
    def __init__(self, norm_channels, num_heads):
        super().__init__(
            # (batch_size, 4, height // 8, width // 8) -> (batch_size, 512, height // 8, width // 8)
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 512, norm_channels),
            AttentionBlock(512, norm_channels, num_heads),
            ResidualBlock(512, 512, norm_channels),
            ResidualBlock(512, 512, norm_channels),
            ResidualBlock(512, 512, norm_channels),
            ResidualBlock(512, 512, norm_channels),
            # (batch_size, 512, height // 8, width // 8) -> (batch_size, 512, height // 4, width // 4)
            nn.Upsample(scale_factor=2.0),
            # (batch_size, 512, height // 4, width // 4) -> (batch_size, 512, height // 4, width // 4)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 512, norm_channels),
            ResidualBlock(512, 512, norm_channels),
            ResidualBlock(512, 512, norm_channels),
            # (batch_size, 512, height // 4, width // 4) -> (batch_size, 512, height // 2, width // 2)
            nn.Upsample(scale_factor=2.0),
            # (batch_size, 512, height // 2, width // 2) -> (batch_size, 512, height // 2, width // 2)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 256, norm_channels),
            ResidualBlock(256, 256, norm_channels),
            ResidualBlock(256, 256, norm_channels),
            # (batch_size, 256, height // 2, width // 2) -> (batch_size, 256, height, width)
            nn.Upsample(scale_factor=2.0),
            # (batch_size, 256, height, width) -> (batch_size, 256, height, width)
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            ResidualBlock(256, 128, norm_channels),
            ResidualBlock(128, 128, norm_channels),
            ResidualBlock(128, 128, norm_channels),
            # (batch_size, 128, height, width) -> (batch_size, 128, height, width)
            nn.GroupNorm(norm_channels, 128),
            nn.SiLU(),
            # (batch_size, 128, height, width) -> (batch_size, 3, height, width)
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, z):
        # z: (batch_size, 4, height // 8, width // 8)
        x = z
        for module in self:
            x = module(x)
        # (batch_size, 3, height, width)
        return x
