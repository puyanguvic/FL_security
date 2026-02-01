from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple

from flbench.utils.cli_kv import merge_kv_and_file

_DEFENSE_REGISTRY: Dict[str, Any] = {}
_IMPORT_ERRORS: Dict[str, Exception] = {}


def _register(name: str, module_path: str, class_name: str) -> None:
    try:
        module = importlib.import_module(module_path)
        _DEFENSE_REGISTRY[name] = getattr(module, class_name)
    except Exception as exc:  # optional defense dependencies
        _IMPORT_ERRORS[name] = exc


_register("mean", "flbench.defenses.aggregation.mean", "MeanDefense")
_register("multikrum", "flbench.defenses.aggregation.multikrum", "MultiKrumDefense")
_register("wbc", "flbench.defenses.aggregation.wbc", "WBCDefense")
_register("fgnv", "flbench.defenses.detection.fgnv", "FGNVDefense")
_register("fldetector", "flbench.defenses.detection.fldetector", "FLDetectorDefense")
_register("beta_reputation", "flbench.defenses.reputation.beta", "BetaReputationDefense")


def list_defenses() -> Tuple[str, ...]:
    return tuple(sorted(_DEFENSE_REGISTRY.keys()))


def build_defense_from_args(args):
    name = str(getattr(args, "defense", "none") or "none").lower()
    if name in {"none", "null", ""}:
        return None

    if name not in _DEFENSE_REGISTRY:
        raise KeyError(f"Unknown defense '{name}'. Available: {list_defenses()}")

    cfg = merge_kv_and_file(kv=getattr(args, "defense_kv", None), config_path=getattr(args, "defense_config", None))
    cls = _DEFENSE_REGISTRY[name]

    if name == "mean":
        return cls()

    if name == "multikrum":
        f = cfg.get("f", cfg.get("num_attackers", None))
        if f is None:
            f = getattr(args, "defense_krum_f", -1)
        if f is None or int(f) < 0:
            f = getattr(args, "n_malicious", 0)
        m = cfg.get("m", cfg.get("num_benign_to_select", None))
        if m is None:
            m = getattr(args, "defense_krum_m", 1)
        lr = float(cfg.get("lr", getattr(args, "lr", 1.0)))
        return cls(num_attackers=int(f), num_benign_to_select=int(m), learning_rate=lr)

    if name == "fgnv":
        lr = float(cfg.get("lr", getattr(args, "lr", 1.0)))
        return cls(learning_rate=lr)

    if name == "beta_reputation":
        discount = float(cfg.get("discount", 0.9))
        return cls(discount=discount)

    if name == "wbc":
        lr = float(cfg.get("lr", getattr(args, "lr", 1.0)))
        device = cfg.get("device", None)
        return cls(lr=lr, device=device)

    if name == "fldetector":
        detector = cfg.get("detector")
        if detector is None:
            raise ValueError("fldetector requires 'detector' in defense_config/defense_kv")
        return cls(detector=detector)

    return cls()


__all__ = ["list_defenses", "build_defense_from_args"]
