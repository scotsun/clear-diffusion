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
        temperature,
        learnable_temp=False,
        pool: str = "gap",
        use_proj: bool = False,
    ):
        super().__init__()
        self.temperature = temperature
        if learnable_temp:
            self.temperature = nn.Parameter(torch.tensor(temperature).exp())
        self.pool = pool
        self.use_proj = use_proj
        if self.use_proj:
            self.proj = nn.Sequential(
                nn.LazyLinear(128),
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

        sim = pairwise_cosine(z) / self.temperature
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
        temperature,
        learnable_temp=False,
        pool: str = "gap",
        use_proj: bool = False,
    ):
        super().__init__(temperature, learnable_temp, pool, use_proj)

    def forward(self, z: torch.Tensor, y: torch.Tensor, ps: bool = False):
        n = z.size(0)
        device = z.device
        z = self._proj(z)

        sim = pairwise_cosine(z)
        eye = torch.eye(n, dtype=torch.bool, device=device)
        sim = sim.masked_fill(eye, float("-inf"))

        if ps:
            p = (y[None, :] != y[:, None]).float()
        else:
            p = (y[None, :] == y[:, None]).float()

        unselect = p == 0
        select_sim = p * sim
        select_sim = select_sim.masked_fill_(unselect, float("-inf"))
        loss = -logsumexp(inputs=select_sim / self.temperature, dim=1) + logsumexp(
            inputs=sim / self.temperature, dim=1
        )
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


if __name__ == "__main__":
    supcon = SupCon(temperature=0.5, learnable_temp=True)
    print(list(supcon.parameters()))
