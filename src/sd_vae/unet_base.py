import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    build_downblock,
    build_upblock,
    build_midblock,
    get_time_embedding,
)


class UNetBase(nn.Module):
    """
    UNet with sinusoidal time embedding for diffusion denoising.

    Architecture:
        conv_in
        -> DownBlock x N  (ResBlock + optional Attention + Downsample)
        -> MidBlock x M   (ResBlock + Attention + ResBlock)
        -> UpBlock x N    (Upsample + concat skip + ResBlock)
        -> norm + conv_out

    Channel contract (critical):
        Skip is saved BEFORE each DownBlock processes features, so skip[i]
        has down_channels[i] channels. build_upblock sets its internal
        Upsample to (in_channels // 2 = down_channels[i]) channels, which
        exactly matches the pre-concat tensor fed into it.

    Expected model_config keys:
        down_channels    : list[int]  e.g. [64, 128, 256, 512]
        mid_channels     : list[int]  e.g. [512, 512, 256]
        time_emb_dim     : int
        down_sample      : list[bool] one entry per DownBlock
        num_down_layers  : int        ResBlocks per DownBlock
        num_mid_layers   : int
        num_up_layers    : int        ResBlocks per UpBlock
        attn_down        : list[bool] whether each DownBlock uses self-attention
        norm_channels    : int        GroupNorm groups
        num_heads        : int        attention heads
        conv_out_channels: int        channels before final conv_out
    """

    def __init__(self, im_channels: int, model_config: dict):
        super().__init__()

        # ── Unpack config ─────────────────────────────────────────
        self.down_channels     = model_config['down_channels']
        self.mid_channels      = model_config['mid_channels']
        self.t_emb_dim         = model_config['time_emb_dim']
        self.down_sample       = model_config['down_sample']
        self.num_down_layers   = model_config['num_down_layers']
        self.num_mid_layers    = model_config['num_mid_layers']
        self.num_up_layers     = model_config['num_up_layers']
        self.attns             = model_config['attn_down']
        self.norm_channels     = model_config['norm_channels']
        self.num_heads         = model_config['num_heads']
        self.conv_out_channels = model_config['conv_out_channels']

        # ── Sanity checks ─────────────────────────────────────────
        assert self.mid_channels[0] == self.down_channels[-1],  \
            "mid_channels[0] must equal down_channels[-1]"
        assert self.mid_channels[-1] == self.down_channels[-2], \
            "mid_channels[-1] must equal down_channels[-2]"
        assert len(self.down_sample) == len(self.down_channels) - 1, \
            "down_sample must have len(down_channels) - 1 entries"
        assert len(self.attns) == len(self.down_channels) - 1, \
            "attn_down must have len(down_channels) - 1 entries"

        # ── Time embedding MLP ────────────────────────────────────
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
        )

        # ── Initial projection ────────────────────────────────────
        self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=1)

        # ── Encoder ───────────────────────────────────────────────
        self.downs = nn.ModuleList()
        for i in range(len(self.down_channels) - 1):
            self.downs.append(
                build_downblock(
                    in_channels     = self.down_channels[i],
                    out_channels    = self.down_channels[i + 1],
                    t_emb_dim       = self.t_emb_dim,
                    n_resnet_blocks = self.num_down_layers,
                    downsample      = self.down_sample[i],
                    norm_channels   = self.norm_channels,
                    attn            = self.attns[i],
                    n_heads         = self.num_heads,
                )
            )

        # ── Bottleneck ────────────────────────────────────────────
        self.mids = nn.ModuleList()
        for i in range(len(self.mid_channels) - 1):
            self.mids.append(
                build_midblock(
                    in_channels   = self.mid_channels[i],
                    out_channels  = self.mid_channels[i + 1],
                    t_emb_dim     = self.t_emb_dim,
                    norm_channels = self.norm_channels,
                    n_heads       = self.num_heads,
                    num_layers    = self.num_mid_layers,
                )
            )

        # ── Decoder ───────────────────────────────────────────────
        # in_channels = down_channels[i] * 2  because:
        #   upsample output : down_channels[i]  (Upsample preserves channels)
        #   skip (pre-proc) : down_channels[i]
        #   concat          : down_channels[i] * 2
        # build_upblock internally sets Upsample(in_channels // 2) = Upsample(down_channels[i]) ✓
        self.ups = nn.ModuleList()
        for i in reversed(range(len(self.down_channels) - 1)):
            self.ups.append(
                build_upblock(
                    in_channels     = self.down_channels[i] * 2,
                    out_channels    = self.down_channels[i - 1] if i != 0 else self.conv_out_channels,
                    t_emb_dim       = self.t_emb_dim,
                    n_resnet_blocks = self.num_up_layers,
                    upsample        = self.down_sample[i],
                    norm_channels   = self.norm_channels,
                    n_heads         = self.num_heads,
                )
            )

        # ── Output projection ─────────────────────────────────────
        self.out_norm = nn.GroupNorm(self.norm_channels, self.conv_out_channels)
        self.conv_out = nn.Conv2d(self.conv_out_channels, im_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (B, im_channels, H, W)
        # t: (B,) integer timesteps

        out = self.conv_in(x)

        # ── Time embedding ────────────────────────────────────────
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb)                              # (B, t_emb_dim)

        # ── Encoder ───────────────────────────────────────────────
        # Skip is saved BEFORE the downblock processes features.
        # At loop iteration i, out has down_channels[i] channels, so
        # skip[i] has down_channels[i] channels — matching Upsample(in_channels//2).
        down_outs = []
        for down in self.downs:
            down_outs.append(out)                               # (B, down_channels[i], H, W)
            for j, res in enumerate(down.resnet_blocks):
                out = res(out, t_emb)
                if down.attn_blocks is not None:
                    out = down.attn_blocks[j](out)
            out = down.downsample(out)

        # ── Bottleneck ────────────────────────────────────────────
        for mid in self.mids:
            out = mid.resnet_block1(out, t_emb)
            out = mid.attention_block(out)
            out = mid.resnet_block2(out, t_emb)

        # ── Decoder ───────────────────────────────────────────────
        # Manual step-by-step (build_upblock returns a bare nn.Module container
        # with no forward(), so it cannot be called directly):
        #   1. up.upsample(out) : (B, down_channels[i], H, W)    -- channels unchanged
        #   2. cat with skip    : (B, down_channels[i]*2, H, W)  -- = in_channels
        #   3. resnet_blocks    : process in_channels -> out_channels
        for up in self.ups:
            skip = down_outs.pop()                              # LIFO: reverses encoder order
            out  = up.upsample(out)                             # (B, C, H*2, W*2) or identity
            out  = torch.cat([out, skip], dim=1)                # (B, C*2, H*2, W*2)
            for j, res in enumerate(up.resnet_blocks):
                out = res(out, t_emb)

        # ── Output ────────────────────────────────────────────────
        out = F.silu(self.out_norm(out))
        return self.conv_out(out)                               # (B, im_channels, H, W)


class UNetConditional(UNetBase):
    """
    Conditional UNet with Classifier-Free Guidance (CFG) support.

    Conditioning types:
        class : integer class index -> embedded and added to t_emb
        text  : text token sequence -> injected via cross-attention
        image : guide image         -> concatenated to input channels

    CFG training:
        Class and text conditioning are randomly dropped with probability
        cond_drop_prob, replaced by null embeddings. This teaches the model
        both conditional and unconditional denoising in one training run.

    CFG inference:
        pred_cond   = model(x, t, cond_input)
        pred_uncond = model(x, t, cond_input, force_drop_cond=True)
        noise_pred  = pred_uncond + scale * (pred_cond - pred_uncond)

    Additional model_config keys under 'condition_config':
        condition_types        : list[str]  subset of ['class', 'text', 'image']
        cond_drop_prob         : float      CFG dropout probability (default 0.1)
        class_condition_config:
            num_classes        : int
        text_condition_config:
            text_embed_dim     : int
        image_condition_config:
            image_condition_input_channels  : int
            image_condition_output_channels : int
    """

    def __init__(self, im_channels: int, model_config: dict):

        # ── Parse conditioning config before super().__init__ ─────
        self.class_cond        = False
        self.text_cond         = False
        self.image_cond        = False
        self.text_embed_dim    = None
        self.im_cond_input_ch  = None
        self.im_cond_output_ch = None
        self.cond_drop_prob    = 0.0

        condition_config = model_config.get('condition_config', None)
        if condition_config is not None:
            self.cond_drop_prob = condition_config.get('cond_drop_prob', 0.1)
            condition_types     = condition_config.get('condition_types', [])
            if 'class' in condition_types:
                self.class_cond  = True
                self.num_classes = condition_config['class_condition_config']['num_classes']
            if 'text' in condition_types:
                self.text_cond      = True
                self.text_embed_dim = condition_config['text_condition_config']['text_embed_dim']
            if 'image' in condition_types:
                self.image_cond        = True
                self.im_cond_input_ch  = condition_config['image_condition_config']['image_condition_input_channels']
                self.im_cond_output_ch = condition_config['image_condition_config']['image_condition_output_channels']

        # ── Build base UNet ────────────────────────────────────────
        super().__init__(im_channels, model_config)

        # ── Class conditioning ────────────────────────────────────
        if self.class_cond:
            self.class_emb = nn.Embedding(self.num_classes, self.t_emb_dim)
            # Learnable null embedding for CFG unconditional pass
            self.null_class_emb = nn.Parameter(torch.zeros(1, self.t_emb_dim))

        # ── Image conditioning ────────────────────────────────────
        if self.image_cond:
            self.cond_conv_in = nn.Conv2d(
                self.im_cond_input_ch, self.im_cond_output_ch, kernel_size=1, bias=False
            )
            # Replace conv_in to accept the extra guide image channels
            self.conv_in = nn.Conv2d(
                im_channels + self.im_cond_output_ch,
                self.down_channels[0], kernel_size=3, padding=1,
            )

        # ── Text conditioning: rebuild all blocks with cross-attention ─
        if self.text_cond:
            # Fixed zero null token used during CFG unconditional text pass
            self.register_buffer(
                'null_text_emb',
                torch.zeros(1, 1, self.text_embed_dim)
            )
            self.downs = nn.ModuleList()
            for i in range(len(self.down_channels) - 1):
                self.downs.append(
                    build_downblock(
                        in_channels     = self.down_channels[i],
                        out_channels    = self.down_channels[i + 1],
                        t_emb_dim       = self.t_emb_dim,
                        n_resnet_blocks = self.num_down_layers,
                        downsample      = self.down_sample[i],
                        norm_channels   = self.norm_channels,
                        attn            = self.attns[i],
                        n_heads         = self.num_heads,
                        context_dim     = self.text_embed_dim,
                    )
                )
            self.mids = nn.ModuleList()
            for i in range(len(self.mid_channels) - 1):
                self.mids.append(
                    build_midblock(
                        in_channels   = self.mid_channels[i],
                        out_channels  = self.mid_channels[i + 1],
                        t_emb_dim     = self.t_emb_dim,
                        norm_channels = self.norm_channels,
                        n_heads       = self.num_heads,
                        num_layers    = self.num_mid_layers,
                        context_dim   = self.text_embed_dim,
                    )
                )
            self.ups = nn.ModuleList()
            for i in reversed(range(len(self.down_channels) - 1)):
                self.ups.append(
                    build_upblock(
                        in_channels     = self.down_channels[i] * 2,
                        out_channels    = self.down_channels[i - 1] if i != 0 else self.conv_out_channels,
                        t_emb_dim       = self.t_emb_dim,
                        n_resnet_blocks = self.num_up_layers,
                        upsample        = self.down_sample[i],
                        norm_channels   = self.norm_channels,
                        n_heads         = self.num_heads,
                        context_dim     = self.text_embed_dim,
                    )
                )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond_input: dict = None,
        force_drop_cond: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x              : (B, C, H, W) noisy image
            t              : (B,) integer timesteps
            cond_input     : dict with any subset of:
                               'class' -> (B,) int indices OR (B, num_classes) one-hot
                               'text'  -> (B, seq_len, text_embed_dim)
                               'image' -> (B, C_cond, H, W)
            force_drop_cond: True forces unconditional pass for CFG inference
        """
        B = x.shape[0]

        # ── Image conditioning ────────────────────────────────────
        # Not dropped for CFG — image context is always structural (depth, seg map).
        if self.image_cond:
            assert cond_input is not None and 'image' in cond_input, \
                "image conditioning requires cond_input['image']"
            im_cond = F.interpolate(cond_input['image'], size=x.shape[-2:])
            im_cond = self.cond_conv_in(im_cond)                # (B, C_out, H, W)
            x = torch.cat([x, im_cond], dim=1)                  # (B, C+C_out, H, W)

        out = self.conv_in(x)

        # ── Time embedding ────────────────────────────────────────
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb)                              # (B, t_emb_dim)

        # ── Class conditioning with CFG dropout ───────────────────
        if self.class_cond:
            assert cond_input is not None and 'class' in cond_input, \
                "class conditioning requires cond_input['class']"
            cls = cond_input['class']
            # Accept both (B,) integer index and (B, num_classes) one-hot
            if cls.dim() == 2:
                cls = cls.argmax(dim=-1)
            class_emb = self.class_emb(cls.long())              # (B, t_emb_dim)

            if self.training:
                # Randomly replace with null embedding during training (CFG dropout)
                drop_mask = torch.rand(B, device=x.device) < self.cond_drop_prob
                null_emb  = self.null_class_emb.expand(B, -1)  # (B, t_emb_dim)
                class_emb = torch.where(drop_mask.unsqueeze(-1), null_emb, class_emb)
            elif force_drop_cond:
                class_emb = self.null_class_emb.expand(B, -1)

            t_emb = t_emb + class_emb                           # (B, t_emb_dim)

        # ── Text conditioning with CFG dropout ────────────────────
        context = None
        if self.text_cond:
            assert cond_input is not None and 'text' in cond_input, \
                "text conditioning requires cond_input['text']"
            context = cond_input['text']                        # (B, S, D)

            if self.training and self.cond_drop_prob > 0:
                # Per-sample text dropout: replace entire sequence with null token
                drop_mask = torch.rand(B, device=x.device) < self.cond_drop_prob
                null_ctx  = self.null_text_emb.expand(B, context.shape[1], -1)
                context   = torch.where(drop_mask[:, None, None], null_ctx, context)
            elif force_drop_cond:
                context = self.null_text_emb.expand(B, context.shape[1], -1)

        # ── Encoder ───────────────────────────────────────────────
        down_outs = []
        for down in self.downs:
            down_outs.append(out)                               # save BEFORE processing
            for j, res in enumerate(down.resnet_blocks):
                out = res(out, t_emb)
                if down.attn_blocks is not None:
                    out = down.attn_blocks[j](out)
                if down.cross_attn_blocks is not None:
                    out = down.cross_attn_blocks[j](out, context)
            out = down.downsample(out)

        # ── Bottleneck ────────────────────────────────────────────
        for mid in self.mids:
            out = mid.resnet_block1(out, t_emb)
            out = mid.attention_block(out)
            if mid.cross_attn_block is not None:
                out = mid.cross_attn_block(out, context)
            out = mid.resnet_block2(out, t_emb)

        # ── Decoder ───────────────────────────────────────────────
        for up in self.ups:
            skip = down_outs.pop()                              # LIFO
            out  = up.upsample(out)                             # (B, C, H*2, W*2)
            out  = torch.cat([out, skip], dim=1)                # (B, C*2, H*2, W*2)
            for j, res in enumerate(up.resnet_blocks):
                out = res(out, t_emb)
                if up.cross_attn_blocks is not None:
                    out = up.cross_attn_blocks[j](out, context)

        # ── Output ────────────────────────────────────────────────
        out = F.silu(self.out_norm(out))
        return self.conv_out(out)                               # (B, im_channels, H, W)
