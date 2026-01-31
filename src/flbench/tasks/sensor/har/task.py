from __future__ import annotations

import torch.nn as nn

from flbench.tasks.sensor.har import dataset as _dataset
from flbench.tasks.sensor.har.split import split_and_save as _split_and_save

default_split_root = "/tmp/flbench_splits/har"
DEFAULT_NUM_CLASSES = 6
DEFAULT_INPUT_DIM = 561
default_model = "mlp/har"

create_datasets = _dataset.create_datasets
create_data_loaders = _dataset.create_data_loaders


class HARMLP(nn.Module):
    def __init__(self, input_dim: int = DEFAULT_INPUT_DIM, num_classes: int = DEFAULT_NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def build_model(model_key: str):
    if not model_key:
        model_key = default_model
    if model_key in {"mlp/har", "mlp", "cnn/moderate"}:
        return HARMLP()
    raise ValueError(f"Unsupported model '{model_key}' for task sensor/har")


def split_and_save(*, num_sites: int, split_dir_prefix: str, seed: int = 0, **kwargs) -> str:
    alpha = kwargs.get("alpha", None)
    if alpha is None:
        raise ValueError("Missing split parameter: alpha")
    data_root = kwargs.get("data_root", None)
    return _split_and_save(
        num_sites=num_sites,
        alpha=float(alpha),
        split_dir_prefix=split_dir_prefix,
        seed=seed,
        data_root=data_root,
    )
