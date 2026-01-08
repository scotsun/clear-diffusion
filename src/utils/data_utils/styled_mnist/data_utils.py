"""Data utility functions."""

import numpy as np
from PIL import ImageOps
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from . import corruptions


def random_style_distribution(
    styles: list = [
        corruptions.identity,
        corruptions.stripe,
        corruptions.zigzag,
        corruptions.canny_edges,
    ],
) -> dict:
    probs = np.random.dirichlet([10] * len(styles))
    output = dict()
    for i, fn in enumerate(styles):
        output[fn] = probs[i]
    return output


class StyledMNISTGenerator:
    """A Helper class to fix the random style assignment to each MNIST image."""

    def __init__(
        self, dataset: torchvision.datasets.MNIST, corruption_fns: None | dict
    ) -> None:
        self.dataset = dataset
        self.corruption_fns = list(corruption_fns.keys())
        self.corruption_fns_p = list(corruption_fns.values())

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.corruption_fns is not None:
            cfn_idx = np.random.choice(
                len(self.corruption_fns), p=self.corruption_fns_p
            )
            img = ImageOps.expand(img, border=2, fill="black")
            img = self.corruption_fns[cfn_idx](img)
            return img, label, cfn_idx
        else:
            return img, label, 0

    @property
    def size(self):
        return len(self.dataset)


class StyledMNIST(Dataset):
    def __init__(self, generator, transform) -> None:
        super().__init__()
        self.generator = generator
        self.transform = transform
        self.N = generator.size
        self.dataset = [None] * self.N
        with tqdm(range(self.N), unit="item") as bar:
            bar.set_description("Generating dataset")
            for i in bar:
                self.dataset[i] = self.generator[i]

    def __len__(self):
        return self.N

    def __getitem__(self, idx) -> tuple:
        img, label, style = self.dataset[idx]
        img = self.transform(img)
        return {"image": img, "label": label, "style": style}

    def display(self, idx):
        img = self.__getitem__(idx)["image"]
        return transforms.ToPILImage()(img)


def build_dataloaders(generator: StyledMNISTGenerator, batch_size: int = 256):
    dataset = StyledMNIST(
        generator,
        transforms.Compose(
            [
                transforms.ToTensor(),
                lambda img: img / 255.0,
            ]
        ),
    )
    train, test, valid = random_split(dataset, [40000, 10000, 10000])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return {"train": train_loader, "valid": valid_loader, "test": test_loader}
