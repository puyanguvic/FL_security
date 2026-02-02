
from .byzantine_registry import (
    BYZANTINE_ATTACKS,
    build_byzantine_attack_from_args,
    is_byzantine_attack_name,
    list_byzantine_attacks,
)
from .registry import AttackContext, build_attack_from_args, diff_l2_norm, list_attacks as list_update_attacks


def list_attacks():
    update = set(list_update_attacks())
    byz = set(list_byzantine_attacks())
    return tuple(sorted(update | byz))


ATTACK_REGISTRY = BYZANTINE_ATTACKS

__all__ = [
    "AttackContext",
    "build_attack_from_args",
    "build_byzantine_attack_from_args",
    "diff_l2_norm",
    "list_attacks",
    "list_byzantine_attacks",
    "list_update_attacks",
    "is_byzantine_attack_name",
    "ATTACK_REGISTRY",
]
