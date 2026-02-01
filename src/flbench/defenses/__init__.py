from __future__ import annotations

from flbench.defenses.base import Defense, DefenseContext
from flbench.defenses.registry import (
    build_defense,
    build_defense_from_args,
    get_defense,
    list_defenses,
    register_defense,
)

# Register built-in defenses
from . import krum as _krum  # noqa: F401
from . import median as _median  # noqa: F401
from . import norm_clip as _norm_clip  # noqa: F401
from . import trimmed_mean as _trimmed_mean  # noqa: F401

__all__ = [
    "Defense",
    "DefenseContext",
    "build_defense",
    "build_defense_from_args",
    "get_defense",
    "list_defenses",
    "register_defense",
]
