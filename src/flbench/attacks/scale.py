from __future__ import annotations

from flbench.attacks.base import Attack, AttackContext, apply_to_diff
from flbench.attacks.registry import register_attack


@register_attack("scale")
class ScaleAttack(Attack):
    def __init__(self, factor: float = 1.0):
        self.factor = float(factor)

    def apply(self, diff, ctx: AttackContext):
        return apply_to_diff(diff, lambda t: t * self.factor), {"attack": "scale", "factor": self.factor}
