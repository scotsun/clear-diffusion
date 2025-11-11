import torch
import torch.nn as nn
import torch.nn.functional as F


def d_hinge_loss(real_logits, fake_logits):
    return F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean()


def g_hinge_loss(fake_logits):
    return -fake_logits.mean()


def pairwise_cosine(z: torch.Tensor):
    # z: (batch_size, z_channel, h_fea, w_fea)
    z = nn.AdaptiveAvgPool2d(output_size=1)(z).squeeze()
    # z = z.view(z.shape[0], -1)
    return F.cosine_similarity(z[:, None, :], z[None, :, :], dim=-1)


@torch.jit.script
def logsumexp(inputs: torch.Tensor, dim: int = -1):
    # cite: https://github.com/pytorch/pytorch/issues/31829
    m, _ = inputs.max(dim=dim)
    mask = m == float("-inf")
    s = (inputs - m.masked_fill_(mask, 0).unsqueeze(dim=dim)).exp().sum(dim=dim)
    return s.masked_fill_(mask, 1).log() + m.masked_fill_(mask, float("-inf"))


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


class SNN(nn.Module):
    def __init__(self, temperature, learnable_temp=False):
        super().__init__()
        self.temperature = temperature
        if learnable_temp:
            self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, z: torch.Tensor, y: torch.Tensor, ps: bool = False):
        n = z.size(0)
        device = z.device

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
        return loss.mean()


if __name__ == "__main__":
    supcon = SupCon(temperature=0.5, learnable_temp=True)
    print(list(supcon.parameters()))
