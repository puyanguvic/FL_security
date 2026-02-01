from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn

Diff = Dict[str, torch.Tensor]


@dataclass
class AttackContext:
    base_diff: Diff
    base_norm: float
    global_model: nn.Module
    local_model: nn.Module
    train_loader: object
    criterion: nn.Module
    device: torch.device
    current_round: int | None = None
    rng: torch.Generator | None = None


class Attack:
    name = "base"

    def apply(self, diff: Diff, ctx: AttackContext) -> Tuple[Diff, dict]:
        return diff, {}


def _iter_float_tensors(diff: Diff):
    for k, t in diff.items():
        if torch.is_tensor(t) and t.is_floating_point():
            yield k, t


def apply_to_diff(diff: Diff, fn) -> Diff:
    out: Diff = {}
    for k, t in diff.items():
        if torch.is_tensor(t) and t.is_floating_point():
            out[k] = fn(t)
        else:
            out[k] = t
    return out


def diff_l2_norm(diff: Diff) -> float:
    sq_sum = 0.0
    for _, t in _iter_float_tensors(diff):
        sq_sum += (t.float().pow(2).sum()).item()
    return float(torch.tensor(sq_sum).sqrt().item())


def diff_total_numel(diff: Diff) -> int:
    total = 0
    for _, t in _iter_float_tensors(diff):
        total += int(t.numel())
    return total
