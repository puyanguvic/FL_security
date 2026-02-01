from __future__ import annotations

from typing import Dict, List, Type

from flbench.attacks.base import Attack

_ATTACKS: Dict[str, Type[Attack]] = {}
_ALIASES: Dict[str, str] = {
    "sign-flip": "sign_flip",
    "signflip": "sign_flip",
    "noise": "gaussian",
    "gauss": "gaussian",
    "pgd": "pgd_minmax",
    "minmax": "pgd_minmax",
}


def register_attack(name: str, aliases: List[str] | None = None):
    def _decorator(cls: Type[Attack]):
        key = name.lower()
        _ATTACKS[key] = cls
        cls.name = key
        if aliases:
            for alias in aliases:
                _ALIASES[alias.lower()] = key
        return cls

    return _decorator


def get_attack(name: str) -> Type[Attack]:
    key = name.lower()
    key = _ALIASES.get(key, key)
    if key not in _ATTACKS:
        raise KeyError(f"Unknown attack '{name}'. Available: {', '.join(list_attacks())}")
    return _ATTACKS[key]


def list_attacks() -> List[str]:
    return sorted(_ATTACKS.keys())


def build_attack(name: str, **kwargs) -> Attack:
    return get_attack(name)(**kwargs)


def build_attack_from_args(args):
    name = getattr(args, "attack", "none")
    if name is None:
        return None
    key = str(name).lower()
    if key in {"none", "off", "null", "no"}:
        return None
    key = _ALIASES.get(key, key)
    if key not in _ATTACKS:
        raise ValueError(f"Unknown attack '{name}'. Available: {', '.join(list_attacks())}")

    # New (recommended): pass attack params via --attack_config (YAML/JSON) and/or repeated --attack_kv k=v.
    # This makes adding new attacks NOT require touching any algorithm/client argparse.
    from flbench.utils.cli_kv import merge_kv_and_file

    kwargs = merge_kv_and_file(
        kv=getattr(args, "attack_kv", None),
        config_path=getattr(args, "attack_config", None),
    )

    # Backward-compatible legacy flags (only used if no config/kv provided).
    if not kwargs:
        if key == "scale":
            kwargs = {"factor": getattr(args, "attack_scale", 1.0)}
        elif key == "gaussian":
            kwargs = {"sigma": getattr(args, "attack_noise_std", 0.0)}
        elif key == "pgd_minmax":
            kwargs = {
                "steps": getattr(args, "attack_pgd_steps", 1),
                "step_size": getattr(args, "attack_pgd_step_size", 0.1),
                "eps": getattr(args, "attack_pgd_eps", 0.0),
                "eps_factor": getattr(args, "attack_pgd_eps_factor", 1.0),
                "max_batches": getattr(args, "attack_pgd_max_batches", 1),
                "init": getattr(args, "attack_pgd_init", "zero"),
            }

    return _ATTACKS[key](**kwargs)
