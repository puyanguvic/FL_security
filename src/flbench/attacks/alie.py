from __future__ import annotations

import torch

from flbench.attacks.base import Attack, AttackContext, flatten_diff, project_l2, unflatten_diff
from flbench.attacks.registry import register_attack


@register_attack("alie", aliases=["alie_local", "adaptive_lie"])
class ALIEAttack(Attack):
    """A lightweight "LIE"-style Byzantine update.

    Classic ALIE/LIE attacks require estimating the mean and std of benign updates
    across clients. In pure client-side simulation, the attacker typically does not
    have access to other clients' updates. This implementation uses a pragmatic
    proxy: estimate per-coordinate mean/std from the attacker update itself
    (treating it as a sample distribution) and constructs:

        u = mu + z * sigma

    Then we project to an L2 radius to control detectability.

    Parameters
    - z: how many stds to shift (larger => more harmful but easier to detect)
    - radius: explicit L2 radius (optional)
    - radius_factor: if radius not set, uses radius_factor * base_norm
    """

    def __init__(self, z: float = 2.5, radius: float | None = None, radius_factor: float = 1.0):
        self.z = float(z)
        self.radius = radius
        self.radius_factor = float(radius_factor)

    def apply(self, diff, ctx: AttackContext):
        vec, spec = flatten_diff(diff)
        if vec.numel() == 0:
            return diff, {"attack": "alie", "skipped": 1, "reason": "empty_update"}

        mu = vec.mean()
        sigma = vec.std(unbiased=False)
        # Avoid degenerate sigma.
        if float(sigma.item()) <= 1e-12:
            sigma = sigma + 1e-6

        u = mu + self.z * sigma
        attacked = torch.full_like(vec, float(u.item()))

        target = float(self.radius) if (self.radius is not None and self.radius > 0) else self.radius_factor * float(ctx.base_norm)
        if target > 0:
            attacked = project_l2(attacked, target)

        out = unflatten_diff(attacked, spec, diff)
        return out, {"attack": "alie", "z": self.z, "radius": target, "mu": float(mu.item()), "sigma": float(sigma.item())}
