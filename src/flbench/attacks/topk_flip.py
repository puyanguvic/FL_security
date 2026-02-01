from __future__ import annotations

import torch

from flbench.attacks.base import Attack, AttackContext, flatten_diff, unflatten_diff
from flbench.attacks.registry import register_attack


@register_attack("topk_flip", aliases=["topk_signflip", "partial_flip"])
class TopKFlipAttack(Attack):
    """Flip the sign of the largest-magnitude coordinates.

    Parameters
    - k: number of coordinates to flip (if >1) OR
    - frac: fraction of coordinates to flip (0..1)

    This is a simple evasion-style attack: it keeps most coordinates unchanged,
    but sabotages the dominant directions.
    """

    def __init__(self, k: int = 0, frac: float = 0.01):
        self.k = int(k)
        self.frac = float(frac)

    def apply(self, diff, ctx: AttackContext):
        vec, spec = flatten_diff(diff)
        n = int(vec.numel())
        if n == 0:
            return diff, {"attack": "topk_flip", "skipped": 1, "reason": "empty_update"}

        k = self.k if self.k > 0 else max(1, int(self.frac * n))
        k = min(k, n)

        # Flip signs for top-k |vec|.
        idx = torch.topk(vec.abs(), k=k, largest=True).indices
        vec2 = vec.clone()
        vec2[idx] = -vec2[idx]

        attacked = unflatten_diff(vec2, spec, diff)
        return attacked, {"attack": "topk_flip", "k": k, "n": n}
