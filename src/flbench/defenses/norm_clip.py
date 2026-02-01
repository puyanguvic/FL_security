from __future__ import annotations

import torch

from flbench.defenses.base import Defense, DefenseContext, weighted_mean
from flbench.defenses.registry import register_defense


def _flatten_update(update):
    flat = []
    for _, v in update.items():
        flat.append(v.float().reshape(-1))
    if not flat:
        return torch.zeros(1)
    return torch.cat(flat, dim=0)


@register_defense("norm_clip", aliases=["clip"])
class NormClipDefense(Defense):
    def __init__(self, clip_norm: float = 0.0):
        self.clip_norm = float(clip_norm)

    def apply(self, updates, ctx: DefenseContext):
        if not updates:
            return {}, {"defense": "norm_clip", "skipped": 1}
        if self.clip_norm <= 0.0:
            return weighted_mean(updates, ctx.weights), {"defense": "norm_clip", "clip_norm": self.clip_norm}

        clipped = []
        for update in updates:
            vec = _flatten_update(update)
            norm = float(torch.norm(vec).item())
            if norm <= 0.0:
                clipped.append(update)
                continue
            scale = min(1.0, self.clip_norm / (norm + 1e-12))
            clipped_update = {k: v.float() * scale for k, v in update.items()}
            clipped.append(clipped_update)
        return weighted_mean(clipped, ctx.weights), {"defense": "norm_clip", "clip_norm": self.clip_norm}
