from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

Diff = Dict[str, torch.Tensor]


@dataclass
class DefenseContext:
    weights: List[float]
    contributors: List[str]
    current_round: int | None = None
    n_clients: int | None = None
    n_malicious: int | None = None


class Defense:
    name = "base"

    def apply(self, updates: List[Diff], ctx: DefenseContext) -> Tuple[Diff, dict]:
        return weighted_mean(updates, ctx.weights), {}


def weighted_mean(updates: List[Diff], weights: List[float]) -> Diff:
    if not updates:
        return {}
    total_weight = float(sum(weights)) if weights else 0.0
    if total_weight <= 0.0:
        weights = [1.0 for _ in updates]
        total_weight = float(len(updates))

    out: Diff = {}
    for update, w in zip(updates, weights):
        for k, v in update.items():
            if not torch.is_tensor(v):
                v = torch.tensor(v)
            v = v.float()
            if k not in out:
                out[k] = v * w
            else:
                out[k] = out[k] + v * w
    for k in list(out.keys()):
        out[k] = out[k] / total_weight
    return out


def unweighted_mean(updates: List[Diff]) -> Diff:
    return weighted_mean(updates, [1.0 for _ in updates])
