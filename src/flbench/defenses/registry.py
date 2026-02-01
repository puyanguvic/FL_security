from __future__ import annotations

from typing import Dict, List, Type

from flbench.defenses.base import Defense

_DEFENSES: Dict[str, Type[Defense]] = {}
_ALIASES: Dict[str, str] = {
    "trimmed-mean": "trimmed_mean",
    "trimmedmean": "trimmed_mean",
    "clip": "norm_clip",
    "krum": "krum",
    "multi-krum": "multi_krum",
    "multikrum": "multi_krum",
}


def register_defense(name: str, aliases: List[str] | None = None):
    def _decorator(cls: Type[Defense]):
        key = name.lower()
        _DEFENSES[key] = cls
        cls.name = key
        if aliases:
            for alias in aliases:
                _ALIASES[alias.lower()] = key
        return cls

    return _decorator


def get_defense(name: str) -> Type[Defense]:
    key = name.lower()
    key = _ALIASES.get(key, key)
    if key not in _DEFENSES:
        raise KeyError(f"Unknown defense '{name}'. Available: {', '.join(list_defenses())}")
    return _DEFENSES[key]


def list_defenses() -> List[str]:
    return sorted(_DEFENSES.keys())


def build_defense(name: str, **kwargs) -> Defense:
    return get_defense(name)(**kwargs)


def build_defense_from_args(args):
    name = getattr(args, "defense", "none")
    if name is None:
        return None
    key = str(name).lower()
    if key in {"none", "off", "null", "no"}:
        return None
    key = _ALIASES.get(key, key)
    if key not in _DEFENSES:
        raise ValueError(f"Unknown defense '{name}'. Available: {', '.join(list_defenses())}")

    # New (recommended): pass defense params via --defense_config (YAML/JSON) and/or repeated --defense_kv k=v.
    from flbench.utils.cli_kv import merge_kv_and_file

    kwargs = merge_kv_and_file(
        kv=getattr(args, "defense_kv", None),
        config_path=getattr(args, "defense_config", None),
    )

    # Backward-compatible legacy flags (only used if no config/kv provided).
    if not kwargs:
        if key == "trimmed_mean":
            kwargs = {"trim_ratio": getattr(args, "defense_trim_ratio", 0.0)}
        elif key == "norm_clip":
            kwargs = {"clip_norm": getattr(args, "defense_clip_norm", 0.0)}
        elif key in {"krum", "multi_krum"}:
            kwargs = {
                "f": getattr(args, "defense_krum_f", None),
                "m": getattr(args, "defense_krum_m", 1),
                "n_malicious": getattr(args, "n_malicious", 0),
            }

    return _DEFENSES[key](**kwargs)
