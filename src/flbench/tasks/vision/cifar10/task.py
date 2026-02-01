from __future__ import annotations

from flbench.models.cnn import ModerateCNN
from flbench.models.vision import AlexNetSmall, VGG11Small
from flbench.tasks.vision.cifar10 import dataset as _dataset
from flbench.tasks.vision.cifar10.split import split_and_save as _split_and_save

default_split_root = "experiments/splits/cifar10"
default_model = "cnn/moderate"

create_datasets = _dataset.create_datasets
create_data_loaders = _dataset.create_data_loaders


def build_model(model_key: str):
    if not model_key:
        model_key = default_model
    if model_key == "cnn/moderate":
        return ModerateCNN(in_channels=3, input_size=32)
    if model_key == "vgg11":
        return VGG11Small(num_classes=10, in_channels=3, min_input_size=32)
    if model_key == "alexnet":
        return AlexNetSmall(num_classes=10, in_channels=3)
    raise ValueError(f"Unsupported model '{model_key}' for task vision/cifar10")


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
