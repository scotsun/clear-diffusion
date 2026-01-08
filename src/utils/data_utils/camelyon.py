from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from torchvision import transforms


def build_dataloader(data_root: str, batch_size: int):
    dataset = get_dataset(dataset="camelyon17", root_dir=data_root, download=True)
    train_data = dataset.get_subset("train", transform=transforms.ToTensor())
    valid_data = dataset.get_subset("val", transform=transforms.ToTensor())
    test_data = dataset.get_subset(
        "test",
        transform=transforms.Compose(
            [transforms.Resize((64, 64)), transforms.ToTensor()]
        ),
    )
    train_loader = get_train_loader("standard", train_data, batch_size=batch_size)
    valid_loader = get_eval_loader("standard", valid_data, batch_size=batch_size)
    test_loader = get_eval_loader("standard", test_data, batch_size=batch_size)

    return {"train": train_loader, "valid": valid_loader, "test": test_loader}
