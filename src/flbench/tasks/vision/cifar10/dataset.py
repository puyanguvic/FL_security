from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def _default_transforms():
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    return train_tf, test_tf


def create_datasets(
    site_name: str,
    train_idx_root: str,
    val_fraction: float = 0.1,
    seed: int = 0,
    data_root: str | None = None,
):
    train_tf, test_tf = _default_transforms()
    root = data_root or os.path.expanduser("~/.torch/data")
    base_train = datasets.CIFAR10(root=root, train=True, download=True, transform=train_tf)
    base_train_noaug = datasets.CIFAR10(root=root, train=True, download=False, transform=test_tf)

    idx_path = os.path.join(train_idx_root, site_name, "train_idx.npy")
    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"Missing split index for {site_name}: {idx_path}")

    idx = np.load(idx_path).astype(np.int64)
    if len(idx) == 0:
        raise ValueError(f"Site {site_name} has 0 samples. Try different alpha/seed.")

    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_val = max(1, int(len(idx) * val_fraction))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    train_dataset = Subset(base_train, train_idx.tolist())
    valid_dataset = Subset(base_train_noaug, val_idx.tolist())
    return train_dataset, valid_dataset


def create_data_loaders(train_dataset, valid_dataset, batch_size: int = 64, num_workers: int = 2):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, valid_loader
