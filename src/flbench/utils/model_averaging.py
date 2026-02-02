from __future__ import annotations

from typing import Dict, List

import torch

StateDict = Dict[str, torch.Tensor]


def average_weights(weights: List[StateDict]) -> StateDict:
    """Compute element-wise mean of a list of state dicts.

    Non-floating tensors (e.g., batch norm counters) are copied from the first entry.
    """

    if not weights:
        return {}

    avg: StateDict = {}
    keys = weights[0].keys()
    for k in keys:
        tensors = [w[k] for w in weights]
        ref = tensors[0]
        if not torch.is_floating_point(ref):
            avg[k] = ref.clone()
            continue
        stacked = torch.stack([t.float() for t in tensors], dim=0)
        avg[k] = stacked.mean(dim=0).to(ref.dtype)
    return avg


def weighted_average_weights(weights: List[StateDict], weights_per_client: List[float]) -> StateDict:
    """Compute element-wise weighted mean of state dicts.

    Non-floating tensors (e.g., batch norm counters) are copied from the first entry.
    """

    if not weights:
        return {}
    if len(weights) != len(weights_per_client):
        return average_weights(weights)

    total_weight = float(sum(weights_per_client))
    if total_weight == 0:
        return average_weights(weights)

    avg: StateDict = {}
    keys = weights[0].keys()
    for k in keys:
        tensors = [w[k] for w in weights]
        ref = tensors[0]
        if not torch.is_floating_point(ref):
            avg[k] = ref.clone()
            continue
        weighted_sum = None
        for t, w in zip(tensors, weights_per_client, strict=False):
            contrib = t.float() * float(w)
            weighted_sum = contrib if weighted_sum is None else (weighted_sum + contrib)
        avg[k] = (weighted_sum / total_weight).to(ref.dtype)
    return avg
