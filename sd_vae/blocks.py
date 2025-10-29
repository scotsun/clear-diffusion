import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_model, proj_bias):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.in_proj = nn.Linear(d_model, d_model * 3, bias=proj_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=proj_bias)
        self.n_heads = n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_head = d_model // n_heads

    def forward(self, x):
        # x: (batch_size, seq_len = h*w, channel)
        # q, k, v: (batch_size, seq_len, channel)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        out, _ = self.mha(q, k, v)
        out = self.out_proj(out)
        return out


class AttentionBlock(nn.Module):
    def __init__(self, channels, norm_channels, n_heads):
        super().__init__()
        self.groupnorm = nn.GroupNorm(norm_channels, channels)
        self.attention = SelfAttention(n_heads, channels, proj_bias=True)

    def forward(self, x):
        # x: (batch_size, channels, h, w)
        residue = x
        x = self.groupnorm(x)
        n, c, h, w = x.shape
        x = x.view(n, c, h * w)  # (batch_size, channels, hw)
        x = x.transpose(-1, -2)  # (batch_size, hw, channels)
        x = self.attention(x)  # (batch_size, hw, channels)
        x = x.transpose(-1, -2)  # (batch_size, channels, hw)

        x = x.view(n, c, h, w)  # (batch_size, channels, h, w)
        return x + residue


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(norm_channels, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm_2 = nn.GroupNorm(norm_channels, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0
            )

    def forward(self, x):
        # x: (batch_size, in_channels, h, w)
        residue = x
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        return x + self.residual_layer(residue)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        x = F.pad(x, (0, 1, 0, 1), value=0)
        return self.conv(x)
