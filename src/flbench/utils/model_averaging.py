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
