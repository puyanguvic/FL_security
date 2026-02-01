from __future__ import annotations

import torch

from flbench.defenses.base import Defense, DefenseContext, unweighted_mean
from flbench.defenses.registry import register_defense


@register_defense("trimmed_mean", aliases=["trimmed-mean", "trimmedmean"])
class TrimmedMeanDefense(Defense):
    def __init__(self, trim_ratio: float = 0.0):
        self.trim_ratio = float(trim_ratio)

    def apply(self, updates, ctx: DefenseContext):
        if not updates:
            return {}, {"defense": "trimmed_mean", "skipped": 1}

        n = len(updates)
        trim_ratio = max(0.0, min(0.49, self.trim_ratio))
        k = int(trim_ratio * n)
        if k == 0:
            return unweighted_mean(updates), {"defense": "trimmed_mean", "trim_ratio": trim_ratio, "k": k}

        keys = updates[0].keys()
        out = {}
        for key in keys:
            stacked = torch.stack([u[key].float() for u in updates], dim=0)
            sorted_vals, _ = stacked.sort(dim=0)
            trimmed = sorted_vals[k : n - k]
            if trimmed.numel() == 0:
                out[key] = stacked.mean(dim=0)
            else:
                out[key] = trimmed.mean(dim=0)
        return out, {"defense": "trimmed_mean", "trim_ratio": trim_ratio, "k": k}
