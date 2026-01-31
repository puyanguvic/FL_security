from __future__ import annotations

import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

DEFAULT_ROOT = os.path.expanduser("~/.torch/data/har/UCI HAR Dataset")
DEFAULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"


def _ensure_har_dataset(root: str) -> None:
    x_path = os.path.join(root, "train", "X_train.txt")
    y_path = os.path.join(root, "train", "y_train.txt")
    if os.path.exists(x_path) and os.path.exists(y_path):
        return

    root_path = Path(root)
    data_root = root_path.parent
    data_root.mkdir(parents=True, exist_ok=True)
    zip_path = data_root / "UCI_HAR_Dataset.zip"

    if not zip_path.exists():
        print(f"Downloading UCI HAR Dataset to {zip_path} ...")
        urlretrieve(DEFAULT_URL, zip_path)

    print(f"Extracting UCI HAR Dataset into {data_root} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_root)

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(
            f"UCI HAR train files not found after extraction: {x_path} and {y_path}. "
            "Please verify the dataset archive."
        )


def _load_har_train(root: str):
    _ensure_har_dataset(root)
    x_path = os.path.join(root, "train", "X_train.txt")
    y_path = os.path.join(root, "train", "y_train.txt")
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(
            "UCI HAR train files not found. Expected: "
            f"{x_path} and {y_path}. Please download and extract the UCI HAR Dataset."
        )
    x = np.loadtxt(x_path, dtype=np.float32)
    y = np.loadtxt(y_path, dtype=np.int64).reshape(-1)
    y = y - 1  # labels are 1-6 -> 0-5
    return x, y


def create_datasets(
    site_name: str,
    train_idx_root: str,
    val_fraction: float = 0.1,
    seed: int = 0,
    data_root: str | None = None,
):
    root = data_root or DEFAULT_ROOT
    x, y = _load_har_train(root)

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

    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)
    base_dataset = TensorDataset(x_tensor, y_tensor)

    train_dataset = Subset(base_dataset, train_idx.tolist())
    valid_dataset = Subset(base_dataset, val_idx.tolist())
    return train_dataset, valid_dataset


def create_data_loaders(train_dataset, valid_dataset, batch_size: int = 64, num_workers: int = 0):
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
