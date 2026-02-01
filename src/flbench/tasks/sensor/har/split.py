from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List
from urllib.request import urlretrieve

import numpy as np

DEFAULT_ROOT = os.path.expanduser("~/.torch/data/har/UCI HAR Dataset")
DEFAULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"


@dataclass
class SplitPaths:
    root: str

    def site_dir(self, site_name: str) -> str:
        return os.path.join(self.root, site_name)

    def site_train_idx(self, site_name: str) -> str:
        return os.path.join(self.site_dir(site_name), "train_idx.npy")


def _load_labels(root: str) -> np.ndarray:
    x_path = os.path.join(root, "train", "X_train.txt")
    y_path = os.path.join(root, "train", "y_train.txt")
    if not os.path.exists(x_path) or not os.path.exists(y_path):
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
    y = np.loadtxt(y_path, dtype=np.int64).reshape(-1)
    return y - 1


def _dirichlet_split_indices(labels: np.ndarray, num_sites: int, alpha: float, seed: int = 0) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    n_classes = int(labels.max() + 1)

    idx_by_class = [np.where(labels == c)[0] for c in range(n_classes)]
    for c in range(n_classes):
        rng.shuffle(idx_by_class[c])

    site_indices: List[List[int]] = [[] for _ in range(num_sites)]

    for c in range(n_classes):
        idx_c = idx_by_class[c]
        if len(idx_c) == 0:
            continue

        proportions = rng.dirichlet(alpha * np.ones(num_sites))
        counts = (proportions * len(idx_c)).astype(int)

        diff = len(idx_c) - counts.sum()
        if diff > 0:
            frac = proportions * len(idx_c) - counts
            for j in np.argsort(-frac)[:diff]:
                counts[j] += 1
        elif diff < 0:
            for j in np.argsort(-counts)[:(-diff)]:
                if counts[j] > 0:
                    counts[j] -= 1

        start = 0
        for s in range(num_sites):
            end = start + counts[s]
            site_indices[s].extend(idx_c[start:end].tolist())
            start = end

    out = []
    for s in range(num_sites):
        arr = np.array(site_indices[s], dtype=np.int64)
        rng.shuffle(arr)
        out.append(arr)
    return out


def split_and_save(
    num_sites: int,
    alpha: float,
    split_dir_prefix: str,
    seed: int = 0,
    data_root: str | None = None,
) -> str:
    if alpha <= 0:
        raise ValueError("alpha must be > 0")

    root = data_root or DEFAULT_ROOT
    labels = _load_labels(root)

    train_idx_root = f"{split_dir_prefix}_alpha{alpha}_sites{num_sites}_seed{seed}"
    paths = SplitPaths(root=train_idx_root)
    os.makedirs(train_idx_root, exist_ok=True)

    site_idxs = _dirichlet_split_indices(labels, num_sites=num_sites, alpha=alpha, seed=seed)

    for i in range(num_sites):
        site_name = f"site-{i + 1}"
        os.makedirs(paths.site_dir(site_name), exist_ok=True)
        np.save(paths.site_train_idx(site_name), site_idxs[i])

    return train_idx_root
