import torch


class IsotropicNormalDistribution:
    """
    Isotropic normal distribution.

    https://github.com/CompVis/stable-diffusion
    """

    def __init__(self, moments: torch.Tensor, deterministic: bool = False):
        # moments: (batch_size, 2 * z_channels, h, w)
        self.deterministic = deterministic
        self.mu, self.logvar = torch.chunk(moments, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        if self.deterministic:
            self.logvar = torch.zeros_like(self.logvar).to(device=moments.device)

    def sample(self):
        std = (0.5 * self.logvar).exp()
        x = self.mu + std * torch.randn_like(self.mu).to(device=self.mu.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        if not other:
            return (
                0.5 * torch.pow(self.mu, 2) + self.logvar.exp() - 1.0 - self.logvar
            ).sum(dim=(1, 2, 3))
        else:
            return normal_kl(self.mu, self.logvar, other.mu, other.logvar).sum(
                dim=(1, 2, 3)
            )

    def nll(self, sample: torch.Tensor):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            return 0.5 * (
                torch.pow(sample - self.mu, 2) / self.logvar.exp()
                + self.logvar
                + torch.log(torch.Tensor([2.0 * torch.pi]))
            ).sum(dim=(1, 2, 3))


def normal_kl(
    mu1: torch.Tensor, logvar1: torch.Tensor, mu2: torch.Tensor, logvar2: torch.Tensor
):
    """
    Compute the KL divergence between two normal distributions.

    https://github.com/openai/guided-diffusion

    """
    return 0.5 * (logvar2 - logvar1 + ((mu1 - mu2) ** 2) * logvar2.exp() - 1.0)
