from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from flbench.utils.cli_kv import merge_kv_and_file

BYZANTINE_ATTACKS: Dict[str, type] = {}
_BYZANTINE_IMPORT_ERRORS: Dict[str, Exception] = {}


def _register(name: str, importer) -> None:
    try:
        BYZANTINE_ATTACKS[name] = importer()
    except Exception as exc:
        _BYZANTINE_IMPORT_ERRORS[name] = exc


def _import_gaussian():
    from .noise.gaussian import GaussianAttack

    return GaussianAttack


def _import_lie():
    from .statistical.lie import LIEAttack

    return LIEAttack


def _import_fang():
    from .model_poisoning.fang import FangAttack

    return FangAttack


def _import_sme():
    from .model_poisoning.sme import SMEAttack

    return SMEAttack


def _import_backdoor():
    from .model_poisoning.backdoor_layers import BackdoorCriticalLayerAttack

    return BackdoorCriticalLayerAttack


def _import_minmax():
    from .optimization.minmax import MinMaxAttack

    return MinMaxAttack


def _import_minsum():
    from .optimization.minsum import MinSumAttack

    return MinSumAttack


_register("gaussian", _import_gaussian)
_register("lie", _import_lie)
_register("fang", _import_fang)
_register("sme", _import_sme)
_register("backdoor", _import_backdoor)
_register("minmax", _import_minmax)
_register("minsum", _import_minsum)


_ALIAS_TO_CANONICAL = {
    # Avoid name collision with update-level gaussian attack.
    "byz_gaussian": "gaussian",
}

_CANONICAL_TO_ALIAS = {v: k for k, v in _ALIAS_TO_CANONICAL.items()}


def _canonicalize(name: str) -> str:
    n = str(name).lower().strip()
    return _ALIAS_TO_CANONICAL.get(n, n)


def _get(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in cfg:
            return cfg[k]
    return default


def list_byzantine_attacks() -> Tuple[str, ...]:
    names = []
    for name in BYZANTINE_ATTACKS:
        names.append(_CANONICAL_TO_ALIAS.get(name, name))
    return tuple(sorted(names))


def is_byzantine_attack_name(name: str | None) -> bool:
    if name is None:
        return False
    raw = str(name).lower().strip()
    if raw in _ALIAS_TO_CANONICAL:
        return True
    if raw in _CANONICAL_TO_ALIAS:
        return False
    return raw in BYZANTINE_ATTACKS or raw in _BYZANTINE_IMPORT_ERRORS


def _resolve_byzantine_attack(name: str) -> str | None:
    raw = str(name).lower().strip()
    if raw in _CANONICAL_TO_ALIAS:
        return None
    canonical = _canonicalize(raw)
    if canonical in BYZANTINE_ATTACKS:
        return canonical
    if canonical in _BYZANTINE_IMPORT_ERRORS:
        err = _BYZANTINE_IMPORT_ERRORS[canonical]
        raise RuntimeError(f"Byzantine attack '{name}' unavailable: {err}") from err
    return None


def _normalize_ability(value: Any, default: str) -> str:
    if value is None:
        return default
    s = str(value).strip()
    if not s:
        return default
    lower = s.lower()
    if lower in {"part", "partial"}:
        return "Part"
    if lower in {"full"}:
        return "Full"
    return s


def _normalize_layers(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    s = str(value).strip()
    if not s:
        return None
    if "," in s:
        return [v.strip() for v in s.split(",") if v.strip()]
    return [s]


def build_byzantine_attack_from_args(args) -> Any | None:
    name = str(getattr(args, "attack", "none") or "none").lower()
    if name in {"none", "null", ""}:
        return None

    canonical = _resolve_byzantine_attack(name)
    if canonical is None:
        return None

    cfg = merge_kv_and_file(kv=getattr(args, "attack_kv", None), config_path=getattr(args, "attack_config", None))

    if canonical == "gaussian":
        scale = float(
            _get(
                cfg,
                "attack_scale",
                "scale",
                "noise_scale",
                "std",
                default=getattr(args, "attack_scale", 1.0),
            )
        )
        device = torch.device("cpu")
        return BYZANTINE_ATTACKS[canonical](attack_scale=scale, device=device)

    if canonical == "lie":
        return BYZANTINE_ATTACKS[canonical]()

    if canonical == "fang":
        return BYZANTINE_ATTACKS[canonical]()

    if canonical == "sme":
        lr = float(_get(cfg, "learning_rate", "lr", "attack_lr", default=getattr(args, "lr", 0.0)))
        scale = float(_get(cfg, "surrogate_scale", "scale", default=1.0))
        ability = _normalize_ability(_get(cfg, "attacker_ability", "ability", default="Full"), "Full")
        return BYZANTINE_ATTACKS[canonical](learning_rate=lr, surrogate_scale=scale, attacker_ability=ability)

    if canonical == "backdoor":
        layers = _normalize_layers(_get(cfg, "critical_layer_names", "layers", "layer_names", default=None))
        if not layers:
            raise ValueError("backdoor attack requires critical_layer_names (or layers) in attack_kv/attack_config")
        scale = float(_get(cfg, "poison_scale", "scale", default=1.0))
        ability = _normalize_ability(_get(cfg, "attacker_ability", "ability", default="Part"), "Part")
        return BYZANTINE_ATTACKS[canonical](critical_layer_names=layers, poison_scale=scale, attacker_ability=ability)

    if canonical == "minmax":
        lr = float(_get(cfg, "learning_rate", "lr", "attack_lr", default=getattr(args, "lr", 0.0)))
        ability = _normalize_ability(_get(cfg, "attacker_ability", "ability", default="Full"), "Full")
        return BYZANTINE_ATTACKS[canonical](learning_rate=lr, attacker_ability=ability)

    if canonical == "minsum":
        lr = float(_get(cfg, "learning_rate", "lr", "attack_lr", default=getattr(args, "lr", 0.0)))
        return BYZANTINE_ATTACKS[canonical](learning_rate=lr)

    raise KeyError(f"Unhandled byzantine attack '{canonical}'")
