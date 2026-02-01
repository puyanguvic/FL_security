from __future__ import annotations

from flbench.attacks.base import Attack, AttackContext, diff_l2_norm
from flbench.attacks.registry import (
    build_attack,
    build_attack_from_args,
    get_attack,
    list_attacks,
    register_attack,
)

# Register built-in attacks
from . import gaussian as _gaussian  # noqa: F401
from . import alie as _alie  # noqa: F401
from . import bounded_sign_flip as _bounded_sign_flip  # noqa: F401
from . import pgd_minmax as _pgd_minmax  # noqa: F401
from . import random_direction as _random_direction  # noqa: F401
from . import scale as _scale  # noqa: F401
from . import sign_flip as _sign_flip  # noqa: F401
from . import topk_flip as _topk_flip  # noqa: F401
from . import topk_flip as _topk_flip  # noqa: F401

__all__ = [
    "Attack",
    "AttackContext",
    "build_attack",
    "build_attack_from_args",
    "diff_l2_norm",
    "get_attack",
    "list_attacks",
    "register_attack",
]
