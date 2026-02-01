from __future__ import annotations

import torch

from flbench.defenses.base import Defense, DefenseContext
from flbench.defenses.registry import register_defense


@register_defense("median")
class MedianDefense(Defense):
    def apply(self, updates, ctx: DefenseContext):
        if not updates:
            return {}, {"defense": "median", "skipped": 1}

        keys = updates[0].keys()
        out = {}
        for k in keys:
            stacked = torch.stack([u[k].float() for u in updates], dim=0)
            out[k] = stacked.median(dim=0).values
        return out, {"defense": "median"}
