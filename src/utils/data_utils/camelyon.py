import torch
from torch.utils.data import DataLoader
from wilds import get_dataset
from torchvision import transforms


def build_dataloader(data_root: str, batch_size: int, download: bool, num_workers: int):
    dataset = get_dataset(dataset="camelyon17", root_dir=data_root, download=download)
    train_data = dataset.get_subset("train", transform=transforms.ToTensor())
    valid_data = dataset.get_subset("val", transform=transforms.ToTensor())
    test_data = dataset.get_subset("test", transform=transforms.ToTensor())

    def collate_fn(batch):
        imgs, ys, metadata = zip(*batch)
        imgs = torch.stack(imgs)
        ys = torch.stack(ys)
        metadata = torch.stack(metadata)
        return {"image": imgs, "label": ys, "style": metadata[:, 0]}

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=num_workers,
    )

    return {"train": train_loader, "valid": valid_loader, "test": test_loader}
