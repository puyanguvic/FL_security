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


def flatten_diff(diff: Diff) -> tuple[torch.Tensor, list[tuple[str, torch.Size, torch.dtype]]]:
    """Flatten floating tensors in `diff` into a single 1-D CPU tensor.

    Returns (vec, spec) where spec stores (key, shape, dtype) to reconstruct.
    Non-floating tensors/metadata are not included in vec/spec.
    """

    flat_parts = []
    spec: list[tuple[str, torch.Size, torch.dtype]] = []
    for k, t in _iter_float_tensors(diff):
        tt = t.detach().cpu().contiguous().view(-1)
        flat_parts.append(tt)
        spec.append((k, t.shape, t.dtype))

    if not flat_parts:
        return torch.empty(0, dtype=torch.float32), []
    vec = torch.cat([p.float() for p in flat_parts], dim=0)
    return vec, spec


def unflatten_diff(
    vec: torch.Tensor,
    spec: list[tuple[str, torch.Size, torch.dtype]],
    template: Diff,
) -> Diff:
    """Reconstruct a diff dict from a flattened vector.

    Tensors listed in spec are populated from vec in order; all other keys from
    `template` are copied through unchanged.
    """

    out: Diff = dict(template)
    if not spec:
        return out
    if vec.dim() != 1:
        vec = vec.view(-1)

    offset = 0
    for k, shape, dtype in spec:
        n = int(torch.tensor(shape).prod().item()) if len(shape) > 0 else 1
        chunk = vec[offset : offset + n]
        offset += n
        out[k] = chunk.view(shape).to(dtype)
    return out


def project_l2(vec: torch.Tensor, radius: float) -> torch.Tensor:
    """Project vec onto L2 ball of given radius (CPU)."""

    if radius <= 0:
        return torch.zeros_like(vec)
    nrm = float(torch.linalg.vector_norm(vec).item()) if vec.numel() else 0.0
    if nrm <= radius:
        return vec
    return vec * (radius / (nrm + 1e-12))
