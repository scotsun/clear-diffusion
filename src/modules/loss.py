import torch
import torch.nn as nn
import torch.nn.functional as F


def d_hinge_loss(real_logits, fake_logits):
    return F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean()


def g_hinge_loss(fake_logits):
    return -fake_logits.mean()


def pairwise_cosine(z: torch.Tensor):
    # z: (batch_size, z_channel, h_fea, w_fea)
    return F.cosine_similarity(z[:, None, :], z[None, :, :], dim=-1)


@torch.jit.script
def logsumexp(inputs: torch.Tensor, dim: int = -1):
    # cite: https://github.com/pytorch/pytorch/issues/31829
    m, _ = inputs.max(dim=dim)
    mask = m == float("-inf")
    s = (inputs - m.masked_fill_(mask, 0).unsqueeze(dim=dim)).exp().sum(dim=dim)
    return s.masked_fill_(mask, 1).log() + m.masked_fill_(mask, float("-inf"))


class SupCon(nn.Module):
    def __init__(
        self,
        temperature: float,
        learnable_temp=False,
        pool: str = "gap",
        use_proj: bool = False,
    ):
        super().__init__()
        self.log_tau = torch.tensor(temperature).log()
        if learnable_temp:
            self.log_tau = nn.Parameter(torch.tensor(temperature).log())
        self.pool = pool
        self.use_proj = use_proj
        if self.use_proj:
            self.proj = nn.Sequential(
                nn.LazyLinear(128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.LazyLinear(128),
            )

    def _proj(self, z: torch.Tensor):
        if self.pool == "gap":
            z = nn.AdaptiveAvgPool2d(output_size=1)(z).squeeze()
        elif self.pool == "flatten":
            z = z.flatten(start_dim=1)

        if self.use_proj:
            z = self.proj(z)

        return z

    def forward(self, z: torch.Tensor, y: torch.Tensor, ps: bool = False):
        n = z.size(0)
        device = z.device
        z = self._proj(z)

        z = F.normalize(z, dim=-1)
        sim = torch.einsum("nc,mc->nm", z, z) / self.log_tau.exp()
        eye = torch.eye(n, dtype=torch.bool, device=device)
        sim = sim.masked_fill(eye, float("-inf"))

        log_q = F.log_softmax(sim, dim=-1)
        log_q = torch.where(eye, torch.zeros_like(log_q), log_q)

        if ps:
            p = (y[None, :] != y[:, None]).float()
        else:
            p = (y[None, :] == y[:, None]).float()
        p = torch.where(eye, torch.zeros_like(p), p)
        p /= p.sum(dim=-1, keepdim=True).clamp_min(1)  # avoid divide-by-zero

        loss = -(p * log_q).sum(dim=-1)
        return loss


class SNN(SupCon):
    def __init__(
        self,
        temperature: float,
        learnable_temp=False,
        pool: str = "gap",
        use_proj: bool = False,
    ):
        super().__init__(temperature, learnable_temp, pool, use_proj)

    def forward(self, z: torch.Tensor, y: torch.Tensor, ps: bool = False):
        n = z.size(0)
        device = z.device
        z = self._proj(z)

        z = F.normalize(z, dim=-1)
        sim = torch.einsum("nc,mc->nm", z, z) / self.log_tau.exp()
        eye = torch.eye(n, dtype=torch.bool, device=device)
        sim = sim.masked_fill(eye, float("-inf"))

        if ps:
            p = (y[None, :] != y[:, None]).float()
        else:
            p = (y[None, :] == y[:, None]).float()

        unselect = p == 0
        select_sim = p * sim
        select_sim = select_sim.masked_fill(unselect, float("-inf"))
        loss = -logsumexp(inputs=select_sim, dim=1) + logsumexp(inputs=sim, dim=1)
        return loss[torch.isfinite(loss)]


class DenseSupCon(nn.Module):
    def __init__(self, temperature, learnable_temp=False, use_proj: bool = False):
        super().__init__()
        self.temperature = temperature
        if learnable_temp:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        self.use_proj = use_proj
        if self.use_proj:
            raise ValueError("TODO! have not implemented yet")

    def forward(self, z: torch.Tensor, y: torch.Tensor, ps: bool = False):
        b, c, w_f, h_f = z.shape
        s = w_f * h_f
        device = z.device

        z = z.view(b, c, s).transpose(1, 2).contiguous()  # (b, s, c)
        z = F.normalize(z, dim=-1)
        z_flat = z.reshape(b * s, c)  # (b*s, c)

        # Allocate pooled similarity output: (b*s, b)
        sim_max = torch.empty((b * s, b), device=device)
        sim_mean = torch.empty((b * s, b), device=device)

        # Compute patch → sample similarities without ever creating (b*s, b*s)
        for j in range(b):
            # patches of sample j: (s, c)
            z_j = z[j]  # (s, c)
            # compute similarity between all patches and sample j’s patches:
            # einsum: (b*s, c) · (c, s) = (b*s, s)
            sim_j = torch.einsum("nc,sc->ns", z_flat, z_j)
            # store reductions
            sim_max[:, j] = sim_j.max(dim=-1).values
            sim_mean[:, j] = sim_j.mean(dim=-1)

        # choose max or mean depending on label match
        eye = torch.eye(b, dtype=torch.bool, device=device)
        self_mask = eye.repeat_interleave(s, dim=0)

        label_mask = (y[None, :] == y[:, None]) & (~eye)
        label_mask = label_mask.repeat_interleave(s, dim=0)

        sim = torch.where(label_mask, sim_max, sim_mean)
        sim = sim / self.temperature

        # mask self sample
        sim = sim.masked_fill(self_mask, float("-inf"))

        log_q = F.log_softmax(sim, dim=-1)
        log_q = torch.where(self_mask, torch.zeros_like(log_q), log_q)

        # build target distribution p
        if ps:  # negative sampling & sim_mean on the numerator (MI min)
            p = (y[None, :] != y[:, None]).float()
        else:  # positive sampling & sim_max on the numerator (MI max)
            p = (y[None, :] == y[:, None]).float()

        p = p.repeat_interleave(s, dim=0)
        p = torch.where(self_mask, torch.zeros_like(p), p)
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1)

        loss = -(p * log_q).sum(dim=-1)
        return loss


class PathologyPerceptualLoss(nn.Module):
    """
    LPIPS-style perceptual loss for pathology images using PLIP (ViT-B/32).

    Extracts intermediate ViT block features from the PLIP vision encoder
    and computes normalized L2 distances across selected layers.

    Usage:
        loss_fn = PathologyPerceptualLoss(device=device)
        loss = loss_fn(img1, img2)  # img1, img2: [B, 3, H, W] in [0, 1]
    """

    PLIP_HF_ID = "vinid/plip"

    # ViT-B/32 has 12 transformer blocks (0–11); pick 4 evenly spaced ones
    FEATURE_LAYERS = [2, 5, 8, 11]

    def __init__(
        self,
        device: torch.device,
        hf_model_id: str = PLIP_HF_ID,
        cache_dir: str | None = None,
    ):
        super().__init__()

        # ── Load PLIP from HuggingFace ──────────────────────────────────────
        model = CLIPModel.from_pretrained(hf_model_id, cache_dir=cache_dir)
        vision_model = model.vision_model

        # ── Decompose ViT into addressable stages ───────────────────────────
        self.embeddings = vision_model.embeddings

        # Defensive: HuggingFace CLIP has a typo in some versions ('pre_layrnorm')
        self.pre_layernorm = getattr(
            vision_model,
            "pre_layrnorm",
            getattr(vision_model, "pre_layernorm", nn.Identity()),
        )

        encoder_layers = vision_model.encoder.layers  # ModuleList of 12 blocks
        self.post_layernorm = vision_model.post_layernorm

        # Register only the blocks up to the deepest required layer
        self.feature_blocks = nn.ModuleList(
            [encoder_layers[i] for i in range(max(self.FEATURE_LAYERS) + 1)]
        )
        self.feature_layer_ids = set(self.FEATURE_LAYERS)

        # Freeze all parameters — fixed feature extractor
        for p in self.parameters():
            p.requires_grad = False

        # ── PLIP / OpenAI CLIP normalization constants ───────────────────────
        self.register_buffer(
            "mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
        )

        self.to(device)
        self.eval()
        print(f"PathologyPerceptualLoss: loaded PLIP from '{hf_model_id}'")

    # ── Preprocessing ────────────────────────────────────────────────────────

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Resize to 224x224 (ViT-B/32 native input), clamp, and normalize
        to CLIP pixel space.
        """
        if x.shape[-2] != 224 or x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
            x = torch.clamp(x, 0.0, 1.0)
        return (x - self.mean) / self.std

    # ── Feature extraction ───────────────────────────────────────────────────

    def _extract_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Run the PLIP ViT encoder up to the deepest required layer,
        collecting hidden states at FEATURE_LAYERS.

        Returns a list of [B, num_patches, C] tensors (CLS token excluded).
        """
        hidden = self.embeddings(x)
        hidden = self.pre_layernorm(hidden)

        features = []
        for idx, block in enumerate(self.feature_blocks):
            # Defensive: some HF versions return a tuple, others return a tensor
            out = block(hidden, attention_mask=None, causal_attention_mask=None)
            hidden = out[0] if isinstance(out, (tuple, list)) else out

            if idx in self.feature_layer_ids:
                # Drop CLS token; keep spatial patch tokens → [B, N, C]
                features.append(hidden[:, 1:, :])

        return features

    # ── Channel-wise L2 normalization (LPIPS-style) ──────────────────────────

    @staticmethod
    def _channel_norm(f: torch.Tensor) -> torch.Tensor:
        """
        L2-normalize along the channel (feature) dimension.
        f: [B, N, C]  →  output: [B, N, C]
        """
        return f / f.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img1, img2 : [B, 3, H, W] float tensors in [0, 1]
        Returns:
            Scalar tensor — mean perceptual distance (↓ = more similar)
        """
        x1 = self._preprocess(img1)
        x2 = self._preprocess(img2)

        feats1 = self._extract_features(x1)
        feats2 = self._extract_features(x2)

        loss = sum(
            ((self._channel_norm(f1) - self._channel_norm(f2)) ** 2)
            .mean(dim=[1, 2])  # mean over patches and channels → [B]
            .mean()            # mean over batch → scalar
            for f1, f2 in zip(feats1, feats2)
        )
        return loss



if __name__ == "__main__":
    supcon = SupCon(temperature=0.5, learnable_temp=True)
    print(list(supcon.parameters()))
