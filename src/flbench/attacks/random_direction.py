from __future__ import annotations

import torch

from flbench.attacks.base import Attack, AttackContext, flatten_diff, project_l2, unflatten_diff
from flbench.attacks.registry import register_attack


@register_attack("random_direction", aliases=["rand_dir", "random"])
class RandomDirectionAttack(Attack):
    """Replace the client update with a random direction, keeping a target L2 norm.

    This is useful as a sanity-check Byzantine attack and to study robustness of
    aggregators/defenses without requiring model/data access.
    """

    def __init__(self, radius: float | None = None, radius_factor: float = 1.0):
        # If radius is None or <=0, uses radius_factor * base_norm.
        self.radius = radius
        self.radius_factor = float(radius_factor)

    def apply(self, diff, ctx: AttackContext):
        vec, spec = flatten_diff(diff)
        if vec.numel() == 0:
            return diff, {"attack": "random_direction", "skipped": 1, "reason": "empty_update"}

        target = float(self.radius) if (self.radius is not None and self.radius > 0) else self.radius_factor * float(ctx.base_norm)
        if target <= 0:
            return diff, {"attack": "random_direction", "skipped": 1, "reason": "zero_radius"}

        g = ctx.rng
        noise = torch.randn(vec.shape, generator=g, dtype=torch.float32)
        noise = project_l2(noise, target)
        attacked = unflatten_diff(noise, spec, diff)
        return attacked, {"attack": "random_direction", "radius": target}
