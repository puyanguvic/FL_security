from __future__ import annotations

from typing import Dict, Iterable, List

import torch

StateDict = Dict[str, torch.Tensor]


def average_gradients(grads: Iterable[StateDict]) -> StateDict:
    grads = list(grads)
    if not grads:
        return {}

    avg: StateDict = {}
    keys = grads[0].keys()
    for k in keys:
        tensors = [g[k] for g in grads]
        ref = tensors[0]
        if not torch.is_floating_point(ref):
            avg[k] = ref.clone()
            continue
        stacked = torch.stack([t.float() for t in tensors], dim=0)
        avg[k] = stacked.mean(dim=0).to(ref.dtype)
    return avg


def calculate_gradient_std_dev(grads: Iterable[StateDict]) -> StateDict:
    grads = list(grads)
    if not grads:
        return {}

    std: StateDict = {}
    keys = grads[0].keys()
    for k in keys:
        tensors = [g[k] for g in grads]
        ref = tensors[0]
        if not torch.is_floating_point(ref):
            std[k] = torch.zeros_like(ref)
            continue
        stacked = torch.stack([t.float() for t in tensors], dim=0)
        std[k] = stacked.std(dim=0, unbiased=False).to(ref.dtype)
    return std


def calculate_gradients(global_weights: StateDict, client_updates: Iterable[StateDict], lr: float) -> List[StateDict]:
    """Estimate gradients given global weights and client weights.

    Assumes client_updates are full model weights. Gradient is (global - local) / lr.
    """

    grads: List[StateDict] = []
    if lr == 0:
        lr = 1e-12
    for update in client_updates:
        grad: StateDict = {}
        for k, gw in global_weights.items():
            if k not in update:
                continue
            u = update[k]
            if torch.is_floating_point(u):
                grad[k] = (gw - u) / lr
            else:
                grad[k] = torch.zeros_like(u)
        grads.append(grad)
    return grads


def calculate_l2_norm(a: StateDict, b: StateDict | None = None) -> float:
    """Compute L2 norm of a (or a-b if b is provided)."""

    sq_sum = 0.0
    for k, v in a.items():
        if b is not None and k in b:
            diff = v - b[k]
        else:
            diff = v
        sq_sum += diff.float().pow(2).sum().item()
    return float(torch.tensor(sq_sum).sqrt().item())


def distance(grads: List[StateDict], target: StateDict) -> Dict[int, float]:
    return {i: calculate_l2_norm(g, target) for i, g in enumerate(grads)}
