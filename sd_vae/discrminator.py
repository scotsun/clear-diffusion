"""Patch-based GAN discriminator
(http://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
"""

import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_channels, num_filters_last=64, n_layers=3):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels, num_filters_last, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        num_filters_mult = 1
        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2**i, 8)
            layers += [
                nn.Conv2d(
                    num_filters_last * num_filters_mult_last,
                    num_filters_last * num_filters_mult,
                    kernel_size=4,
                    stride=2 if i < n_layers else 1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        layers += [
            nn.Conv2d(
                num_filters_last * num_filters_mult,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
            )
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    # TODO: add weight init


if __name__ == "__main__":
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    disc = Discriminator(3, n_layers=0).to(device)

    print(disc)

    # params example:
    # (32, 32): n_layers=0
    # (256, 256): n_layers=3
    # (512, 512): n_layers=4

    x = torch.randn(1, 3, 32, 32).to(device)
    print(disc(x).shape)
