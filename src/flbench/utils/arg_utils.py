from __future__ import annotations

import numbers
import re
from typing import Any

_DEVICE_RE = re.compile(r"^cuda:(\d+)$")
_OPTIMIZERS = {"sgd", "adam", "momentum"}


def _ensure_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be int")
    return int(value)


def _ensure_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be float")
    return float(value)


def normalize_common_args(args: Any) -> None:
    if hasattr(args, "seed"):
        args.seed = _ensure_int("seed", args.seed)

    if hasattr(args, "global_rounds"):
        args.global_rounds = _ensure_int("global_rounds", args.global_rounds)
        if args.global_rounds <= 0:
            raise ValueError("global_rounds must be > 0")
        args.num_rounds = args.global_rounds

    if hasattr(args, "num_clients"):
        args.num_clients = _ensure_int("num_clients", args.num_clients)
        if args.num_clients < 1:
            raise ValueError("num_clients must be >= 1")
        args.n_clients = args.num_clients

    if hasattr(args, "client_fraction"):
        args.client_fraction = _ensure_float("client_fraction", args.client_fraction)
        if args.client_fraction <= 0.0 or args.client_fraction > 1.0:
            raise ValueError("client_fraction must be in (0, 1]")

    if hasattr(args, "local_epochs"):
        args.local_epochs = _ensure_int("local_epochs", args.local_epochs)
        if args.local_epochs < 1:
            raise ValueError("local_epochs must be >= 1")
        args.aggregation_epochs = args.local_epochs

    if hasattr(args, "batch_size"):
        args.batch_size = _ensure_int("batch_size", args.batch_size)
        if args.batch_size < 1:
            raise ValueError("batch_size must be >= 1")

    if hasattr(args, "optimizer"):
        opt = str(args.optimizer).lower().strip()
        if opt not in _OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {sorted(_OPTIMIZERS)}")
        args.optimizer = opt

    if hasattr(args, "device"):
        if args.device in (None, ""):
            device = "cuda:0"
        else:
            device = str(args.device).lower().strip()
        if device == "cuda":
            device = "cuda:0"
        if device != "cpu" and _DEVICE_RE.match(device) is None:
            raise ValueError("device must be 'cpu' or 'cuda:{idx}'")
        args.device = device
