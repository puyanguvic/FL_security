from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class RunContext:
    """Lightweight context passed through training/aggregation hooks."""

    site: str | None = None
    round: int | None = None
    device: Any | None = None
    logger: Any | None = None
    extra: Dict[str, Any] = field(default_factory=dict)
