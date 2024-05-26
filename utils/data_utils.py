import numpy as np
import sys
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def load_cifar10():
    """Loads CIFAR-10 dataset."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    return (
        datasets.CIFAR10("data", train=True, download=True, transform=transform_train),
        datasets.CIFAR10("data", train=False, download=True, transform=transform_test),
    )


def load_cifar100():
    """Loads CIFAR-100 dataset."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5070746, 0.4865490, 0.4409179), (0.2673342, 0.2564385, 0.2761506)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5070746, 0.4865490, 0.4409179), (0.2673342, 0.2564385, 0.2761506)),
    ])

    return (
        datasets.CIFAR100("data", train=True, download=True, transform=transform_train),
        datasets.CIFAR100("data", train=False, download=True, transform=transform_test),
    )


def load_dataset(
    dataset: str,
    subsample: int = 0,
):
    """Loads dataset specified by provided string.

    Args:
        dataset (str): Dataset name.
        subsample (int, optional): How much to subsample data by. Defaults to 0 (no subsampling).
    """
    out_dim = 10
    n_channels = 3  # Number of channels in input image.
    if dataset == "CIFAR10":
        train_data, test_data = load_cifar10()
    elif dataset == "CIFAR100":
        out_dim = 100
        train_data, test_data = load_cifar100()
    else:
        sys.exit(f"Dataset {dataset} is an invalid dataset.")

    # Subsample as necessary.
    if subsample > 0:
        train_data = torch.utils.data.Subset(
            train_data,
            np.random.choice(
                list(range(len(train_data))), size=subsample, replace=False
            ),
        )
        test_data = torch.utils.data.Subset(
            test_data,
            np.random.choice(
                list(range(len(test_data))), size=int(0.2 * subsample), replace=False
            ),
        )

    return train_data, test_data, n_channels, out_dim


def split_train_into_val(train_data, val_prop: float = 0.1):
    """Splits training dataset into train and val.

    Args:
        train_data: Training dataset.
        val_prop: Proportion of data to use for validation.
    """
    val_len = int(val_prop * len(train_data))
    train_subset, val_subset = torch.utils.data.random_split(
        train_data, [len(train_data) - val_len, val_len]
    )
    return train_subset, val_subset
