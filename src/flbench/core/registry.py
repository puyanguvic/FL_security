from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple


@dataclass(frozen=True)
class AlgoSpec:
    name: str
    run: Callable[..., None]  # run(args) -> None


@dataclass(frozen=True)
class TaskSpec:
    name: str
    build_model: Callable[[str], Any]
    split_and_save: Callable[..., str]
    create_datasets: Callable[..., Any]
    create_data_loaders: Callable[..., Any]
    default_split_root: str
    default_model: str = "cnn/moderate"


# -------------------------
# Registry
# -------------------------
_ALGOS: Dict[str, AlgoSpec] = {}
_TASKS: Dict[str, TaskSpec] = {}


def register_algo(spec: AlgoSpec) -> None:
    _ALGOS[spec.name] = spec


def register_task(spec: TaskSpec) -> None:
    _TASKS[spec.name] = spec


def get_algo(name: str) -> AlgoSpec:
    if name not in _ALGOS:
        raise KeyError(f"Unknown algo '{name}'. Available: {sorted(_ALGOS)}")
    return _ALGOS[name]


def get_task(name: str) -> TaskSpec:
    if name not in _TASKS:
        raise KeyError(f"Unknown task '{name}'. Available: {sorted(_TASKS)}")
    return _TASKS[name]


def list_algos() -> Tuple[str, ...]:
    return tuple(sorted(_ALGOS.keys()))


def list_tasks() -> Tuple[str, ...]:
    return tuple(sorted(_TASKS.keys()))


# -------------------------
# Built-in registrations
# -------------------------
def _register_builtin() -> None:
    from flbench.algorithms.fedavg.job import run_fedavg
    from flbench.tasks.vision.cifar10 import task as cifar10_task
    from flbench.tasks.vision.fashionmnist import task as fashion_task

    register_algo(AlgoSpec(name="fedavg", run=run_fedavg))
    register_task(
        TaskSpec(
            name="vision/cifar10",
            build_model=cifar10_task.build_model,
            split_and_save=cifar10_task.split_and_save,
            create_datasets=cifar10_task.create_datasets,
            create_data_loaders=cifar10_task.create_data_loaders,
            default_split_root=cifar10_task.default_split_root,
            default_model=cifar10_task.default_model,
        )
    )
    register_task(
        TaskSpec(
            name="vision/fashionmnist",
            build_model=fashion_task.build_model,
            split_and_save=fashion_task.split_and_save,
            create_datasets=fashion_task.create_datasets,
            create_data_loaders=fashion_task.create_data_loaders,
            default_split_root=fashion_task.default_split_root,
            default_model=fashion_task.default_model,
        )
    )


_register_builtin()
