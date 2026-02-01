from __future__ import annotations

import torch

from flbench.attacks.base import Attack, AttackContext, flatten_diff, project_l2, unflatten_diff
from flbench.attacks.registry import register_attack


@register_attack("bounded_sign_flip", aliases=["sign_flip_eps", "eps_signflip"])
class BoundedSignFlipAttack(Attack):
    """Sign-flip (multiply by -1) followed by L2 projection.

    This is a stronger & more controllable variant than plain sign_flip, useful
    when you want to compare robust aggregators/defenses under a fixed attacker
    budget.

    Parameters
    - radius: explicit L2 radius (optional)
    - radius_factor: if radius not set, uses radius_factor * base_norm
    """

    def __init__(self, radius: float | None = None, radius_factor: float = 1.0):
        self.radius = radius
        self.radius_factor = float(radius_factor)

    def apply(self, diff, ctx: AttackContext):
        vec, spec = flatten_diff(diff)
        if vec.numel() == 0:
            return diff, {"attack": "bounded_sign_flip", "skipped": 1, "reason": "empty_update"}

        attacked = -vec
        target = float(self.radius) if (self.radius is not None and self.radius > 0) else self.radius_factor * float(ctx.base_norm)
        if target > 0:
            attacked = project_l2(attacked, target)

        out = unflatten_diff(attacked, spec, diff)
        return out, {"attack": "bounded_sign_flip", "radius": target}
