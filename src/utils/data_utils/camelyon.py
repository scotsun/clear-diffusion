import torch
from torch.utils.data import DataLoader, DistributedSampler
from wilds import get_dataset
from torchvision import transforms


def build_dataloader(
    data_root: str,
    batch_size: int,
    download: bool,
    num_workers: int,
    is_distributed: bool,
):
    dataset = get_dataset(dataset="camelyon17", root_dir=data_root, download=download)
    train_data = dataset.get_subset("train", transform=transforms.ToTensor())
    valid_data = dataset.get_subset("val", transform=transforms.ToTensor())
    test_data = dataset.get_subset("test", transform=transforms.ToTensor())

    def _collate_fn(batch):
        imgs, ys, metadata = zip(*batch)
        imgs = torch.stack(imgs)
        ys = torch.stack(ys)
        metadata = torch.stack(metadata)
        return {"image": imgs, "label": ys, "style": metadata[:, 0]}

    def _make_loader(dataset, shuffle, sampler=None):
        # shuffle: True/False is only used when sampler is None
        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=_collate_fn,
            num_workers=num_workers,
            shuffle=shuffle if sampler is None else None,
            drop_last=False if sampler is None else True,
            sampler=sampler,
        )

    if is_distributed:
        samplers = {
            split: DistributedSampler(ds, drop_last=True)
            for split, ds in zip(
                ["train", "valid", "test"],
                [train_data, valid_data, test_data],
            )
        }
        train_loader = _make_loader(train_data, False, samplers["train"])
        valid_loader = _make_loader(valid_data, False, samplers["valid"])
        test_loader = _make_loader(test_data, False, samplers["test"])

    else:
        train_loader = _make_loader(train_data, True)
        valid_loader = _make_loader(valid_data, False)
        test_loader = _make_loader(test_data, False)

    return {"train": train_loader, "valid": valid_loader, "test": test_loader}
