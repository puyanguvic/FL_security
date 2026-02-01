from __future__ import annotations

import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DEFAULT_ROOT = os.path.expanduser("~/.torch/data/tiny-imagenet-200")
DEFAULT_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def _ensure_tiny_imagenet(root: str) -> None:
    train_dir = _train_dir(root)
    if os.path.isdir(train_dir):
        return

    root_path = Path(root)
    data_root = root_path.parent
    data_root.mkdir(parents=True, exist_ok=True)
    zip_path = data_root / "tiny-imagenet-200.zip"

    if not zip_path.exists():
        print(f"Downloading Tiny-ImageNet to {zip_path} ...")
        urlretrieve(DEFAULT_URL, zip_path)

    print(f"Extracting Tiny-ImageNet into {data_root} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_root)

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Tiny-ImageNet train dir not found after extraction: {train_dir}. Please verify the dataset archive."
        )


def _default_transforms():
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
        ]
    )
    return train_tf, test_tf


def _train_dir(root: str) -> str:
    return os.path.join(root, "train")


def create_datasets(
    site_name: str,
    train_idx_root: str,
    val_fraction: float = 0.1,
    seed: int = 0,
    data_root: str | None = None,
):
    root = data_root or DEFAULT_ROOT
    _ensure_tiny_imagenet(root)
    train_dir = _train_dir(root)

    train_tf, test_tf = _default_transforms()
    base_train = datasets.ImageFolder(root=train_dir, transform=train_tf)
    base_train_noaug = datasets.ImageFolder(root=train_dir, transform=test_tf)

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
