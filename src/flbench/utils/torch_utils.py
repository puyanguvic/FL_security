from __future__ import annotations

from typing import Dict, List, Tuple
import torch
import torch.nn as nn


@torch.no_grad()
def evaluate(model: nn.Module, dataloader) -> float:
    model.eval()
    device = next(model.parameters()).device
    correct = 0
    total = 0
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return float(correct) / float(total) if total > 0 else 0.0


@torch.no_grad()
def evaluate_with_loss(model: nn.Module, dataloader, criterion: nn.Module) -> Tuple[float, float]:
    model.eval()
    device = next(model.parameters()).device
    correct = 0
    total = 0
    total_loss = 0.0
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        batch_size = y.numel()
        total_loss += loss.item() * batch_size
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += batch_size
    avg_loss = float(total_loss) / float(total) if total > 0 else 0.0
    acc = float(correct) / float(total) if total > 0 else 0.0
    return avg_loss, acc


def get_lr_values(optimizer) -> List[float]:
    return [group["lr"] for group in optimizer.param_groups]


@torch.no_grad()
def compute_model_diff(local_model: nn.Module, global_model: nn.Module) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """diff = local - global; returns (diff_state_dict_cpu, l2_norm)."""
    local_sd = local_model.state_dict()
    global_sd = global_model.state_dict()

    diff: Dict[str, torch.Tensor] = {}
    sq_sum = 0.0
    for k in local_sd.keys():
        d = local_sd[k] - global_sd[k]
        diff[k] = d.cpu()
        sq_sum += (d.float().pow(2).sum()).item()

    diff_norm = torch.tensor(sq_sum).sqrt()
    return diff, diff_norm
