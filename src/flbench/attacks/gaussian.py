from __future__ import annotations

import math

import torch

from flbench.attacks.base import Attack, AttackContext, apply_to_diff, diff_total_numel
from flbench.attacks.registry import register_attack


@register_attack("gaussian", aliases=["noise", "gauss"])
class GaussianAttack(Attack):
    def __init__(self, sigma: float = 0.0):
        self.sigma = float(sigma)

    def apply(self, diff, ctx: AttackContext):
        total_numel = diff_total_numel(diff)
        if self.sigma <= 0.0 or total_numel == 0:
            return diff, {"attack": "gaussian", "sigma": self.sigma}

        base_norm = float(ctx.base_norm)
        if base_norm > 0.0:
            std = self.sigma * base_norm / math.sqrt(total_numel)
        else:
            std = self.sigma

        generator = ctx.rng if ctx.rng is not None else None

        def _add_noise(t: torch.Tensor):
            noise = torch.randn_like(t, generator=generator) * std
            return t + noise

        return apply_to_diff(diff, _add_noise), {"attack": "gaussian", "sigma": self.sigma, "noise_std": std}
