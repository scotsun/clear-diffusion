import torch
import torch.nn as nn
import torch.nn.functional as F


def d_hinge_loss(real_logits, fake_logits):
    return F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean()


def g_hinge_loss(fake_logits):
    return -fake_logits.mean()


def pairwise_cosine(z: torch.Tensor):
    # z: (batch_size, z_channel, h_fea, w_fea)
    z_gap = nn.AdaptiveAvgPool2d(output_size=1)(z).squeeze()
    return F.cosine_similarity(z_gap[:, None, :], z_gap[None, :, :], dim=-1)


class SupCon(nn.Module):
    def __init__(self, temperature, learnable_temp=False):
        super().__init__()
        self.temperature = temperature
        if learnable_temp:
            self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, z: torch.Tensor, y: torch.Tensor, ps: bool = False):
        n = z.size(0)
        device = z.device

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
