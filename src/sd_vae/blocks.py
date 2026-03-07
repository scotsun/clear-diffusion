import torch
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
    def __init__(self, in_channels, out_channels, norm_channels, t_emb_dim=None):
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
            

        self.t_emb_dim = t_emb_dim
        if t_emb_dim is not None:
            self.time_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels),
            )

    def forward(self, x, t_emb=None):
        # x: (batch_size, in_channels, h, w)
        residue = x
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        if getattr(self, 't_emb_dim', None) is not None and t_emb is not None: 
            x = x + self.time_proj(t_emb)[:, :, None, None]
        x = F.silu(x)
        x = self.conv_2(x)
        return x + self.residual_layer(residue)



class CrossAttentionBlock(nn.Module):
    """Image features (Q) attend to text context (K, V)."""
    def __init__(self, channels, norm_channels, n_heads, context_dim):
        super().__init__()
        self.norm = nn.GroupNorm(norm_channels, channels)
        self.q_proj = nn.Linear(channels, channels)
        self.kv_proj = nn.Linear(context_dim, channels * 2)
        self.mha = nn.MultiheadAttention(channels, n_heads, batch_first=True)
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, x, context):
        # x: (B, C, H, W),  context: (B, seq_len, context_dim)
        residue = x
        B, C, H, W = x.shape
        x = self.norm(x).view(B, C, H * W).transpose(1, 2)   # (B, HW, C)
        q = self.q_proj(x)
        k, v = self.kv_proj(context).chunk(2, dim=-1)         # (B, seq_len, C) each
        out, _ = self.mha(q, k, v)
        out = self.out_proj(out)
        out = out.transpose(1, 2).view(B, C, H, W)
        return out + residue
    


class Downsample(nn.Module):
    """(batch_size, channels, h, w) -> (batch_size, channels, h//2, w//2)"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        x = F.pad(x, (0, 1, 0, 1), value=0)
        return self.conv(x)


class Upsample(nn.Module):
    """(batch_size, channels, h, w) -> (batch_size, channels, h*2 , w*2)"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


def build_downblock(
    in_channels    : int,
    out_channels   : int,
    n_resnet_blocks: int,
    downsample     : bool,
    norm_channels  : int,
    t_emb_dim      : int  = None,  
    attn           : bool = False,
    n_heads        : int  = 4,
    context_dim    : int  = None,
) -> nn.Module:
    """
    downblock = [(resnet_block + downsample)] * n - 1 + [(resnet_block)]
    """
    downblock = nn.Module()
    resnet_blocks = nn.ModuleList()
    for _ in range(n_resnet_blocks):
        resnet_blocks.append(ResidualBlock(in_channels, out_channels, norm_channels, t_emb_dim))
        in_channels = out_channels
    downblock.resnet_blocks = resnet_blocks
    if attn:
        downblock.attn_blocks = nn.ModuleList(
        [AttentionBlock(out_channels, norm_channels, n_heads) for _ in range(n_resnet_blocks)]
    )
    else:
        downblock.attn_blocks = None
    if context_dim is not None:
        downblock.cross_attn_blocks = nn.ModuleList(
            [CrossAttentionBlock(out_channels, norm_channels, n_heads, context_dim)
            for _ in range(n_resnet_blocks)]
        ) 
    else:
        downblock.cross_attn_blocks = None

    if downsample:
        downblock.downsample = Downsample(out_channels)
    else:
        downblock.downsample = nn.Identity()

    return downblock


def build_midblock(
    in_channels  : int,
    out_channels : int,
    norm_channels: int,
    t_emb_dim    : int = None,    
    n_heads      : int = 4,
    num_layers   : int = 1,
    context_dim  : int = None,
) -> nn.Module:
    """
    midblock = (resnet_block, attention_block, resnet_block)
    """
    midblock = nn.Module()
    midblock.resnet_block1 = ResidualBlock(in_channels, out_channels, norm_channels, t_emb_dim)
    midblock.attention_block = AttentionBlock(out_channels, norm_channels, n_heads)
    if context_dim is not None:
        midblock.cross_attn_block = CrossAttentionBlock(out_channels, norm_channels, n_heads, context_dim)
    else:
        midblock.cross_attn_block = None
    midblock.resnet_block2 = ResidualBlock(out_channels, out_channels, norm_channels, t_emb_dim)

    return midblock


def build_upblock(
    in_channels    : int,
    out_channels   : int,
    n_resnet_blocks: int,
    upsample       : bool,
    norm_channels  : int,
    t_emb_dim      : int  = None,  
    n_heads        : int  = 4,
    context_dim    : int  = None,
) -> nn.Module:
    """
    upblock = [(resnet_block + upsample)] * n - 1 + [(resnet_block)]
    """
    upblock = nn.Module()
    upsample_channels = in_channels // 2
    resnet_blocks = nn.ModuleList()
    for _ in range(n_resnet_blocks):
        resnet_blocks.append(ResidualBlock(in_channels, out_channels, norm_channels, t_emb_dim))
        in_channels = out_channels
    upblock.resnet_blocks = resnet_blocks
    if context_dim is not None:
        upblock.cross_attn_blocks = nn.ModuleList(
            [CrossAttentionBlock(out_channels, norm_channels, n_heads, context_dim)
             for _ in range(n_resnet_blocks)]
        )
    else:
        upblock.cross_attn_blocks = None
    if upsample:
        upblock.upsample = Upsample(upsample_channels)
    else:
        upblock.upsample = nn.Identity()
    return upblock



def get_time_embedding(time_steps: torch.Tensor, temb_dim: int) -> torch.Tensor:
    """
    Convert integer timesteps to sinusoidal embeddings.
    Args:
        time_steps: (B,) integer tensor
        temb_dim  : embedding dimension (must be even)
    Returns:
        (B, temb_dim) float tensor
    """
    assert temb_dim % 2 == 0, "temb_dim must be even"

    # freq = 10000^(2i / temb_dim),  i = 0..temb_dim//2
    half = temb_dim // 2
    factor = 10000 ** (torch.arange(0, half, dtype=torch.float32, device=time_steps.device) / temb_dim)

    # (B, half)
    t_emb = time_steps[:, None].float() / factor[None, :]
    return torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)  # (B, temb_dim)