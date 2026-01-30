from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple


@dataclass(frozen=True)
class AlgoSpec:
    name: str
    run: Callable[..., None]  # run(args) -> None


@dataclass(frozen=True)
class TaskSpec:
    name: str


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

    register_algo(AlgoSpec(name="fedavg", run=run_fedavg))
    register_task(TaskSpec(name="vision/cifar10"))


_register_builtin()
