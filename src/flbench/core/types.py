from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ModelState:
    params: Dict[str, Any]
    params_type: str | None = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Update:
    diff: Dict[str, Any]
    weight: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Metrics:
    values: Dict[str, float] = field(default_factory=dict)
    step: int | None = None
