from __future__ import annotations

import copy

import torch

from flbench.attacks.base import Attack, AttackContext
from flbench.attacks.registry import register_attack


@register_attack("pgd_minmax", aliases=["pgd", "minmax"])
class PGDMinMaxAttack(Attack):
    def __init__(
        self,
        steps: int = 1,
        step_size: float = 0.1,
        eps: float = 0.0,
        eps_factor: float = 1.0,
        max_batches: int = 1,
        init: str = "zero",
    ):
        self.steps = int(steps)
        self.step_size = float(step_size)
        self.eps = float(eps)
        self.eps_factor = float(eps_factor)
        self.max_batches = int(max_batches)
        self.init = str(init).lower()

    def apply(self, diff, ctx: AttackContext):
        if ctx.train_loader is None or ctx.criterion is None:
            return diff, {"attack": "pgd_minmax", "skipped": 1, "reason": "missing_data"}

        base_norm = float(ctx.base_norm)
        eps = self.eps if self.eps > 0.0 else self.eps_factor * base_norm
        if eps <= 0.0:
            return diff, {"attack": "pgd_minmax", "skipped": 1, "reason": "zero_eps"}

        steps = max(1, self.steps)
        max_batches = max(1, self.max_batches)
        step_size = self.step_size if self.step_size > 0.0 else eps / steps

        device = ctx.device
        base_model = ctx.global_model
        base_state = {k: v.detach().clone().to(device) for k, v in base_model.state_dict().items()}

        work_model = copy.deepcopy(base_model)
        work_model.to(device)
        work_model.eval()
        for param in work_model.parameters():
            param.requires_grad_(True)

        with torch.no_grad():
            if self.init == "local":
                for name, param in work_model.named_parameters():
                    if name in diff and diff[name].is_floating_point():
                        param.copy_(base_state[name] + diff[name].to(device))
            elif self.init == "sign":
                for name, param in work_model.named_parameters():
                    if name in diff and diff[name].is_floating_point():
                        param.copy_(base_state[name] - diff[name].to(device))
            else:
                work_model.load_state_dict(base_state, strict=True)

        for _ in range(steps):
            work_model.zero_grad(set_to_none=True)
            loss_total = None
            batch_count = 0
            for batch_idx, batch in enumerate(ctx.train_loader):
                if batch_idx >= max_batches:
                    break
                inputs, labels = batch[0].to(device), batch[1].to(device)
                outputs = work_model(inputs)
                loss = ctx.criterion(outputs, labels)
                loss_total = loss if loss_total is None else loss_total + loss
                batch_count += 1
            if batch_count == 0 or loss_total is None:
                break

            loss = loss_total / batch_count
            loss.backward()

            with torch.no_grad():
                for name, param in work_model.named_parameters():
                    if param.grad is None:
                        continue
                    param.add_(param.grad, alpha=step_size)

                total_norm_sq = torch.tensor(0.0, device=device)
                for name, param in work_model.named_parameters():
                    delta = param - base_state[name]
                    total_norm_sq = total_norm_sq + delta.float().pow(2).sum()
                total_norm = float(torch.sqrt(total_norm_sq).item())
                if total_norm > eps:
                    scale = eps / (total_norm + 1e-12)
                    for name, param in work_model.named_parameters():
                        delta = param - base_state[name]
                        param.copy_(base_state[name] + delta * scale)

        param_names = {name for name, _ in work_model.named_parameters()}
        work_state = work_model.state_dict()
        attacked = {}
        for k, v in diff.items():
            if k in param_names and torch.is_tensor(v) and v.is_floating_point():
                attacked[k] = (work_state[k] - base_state[k]).detach().cpu()
            else:
                attacked[k] = v

        return attacked, {
            "attack": "pgd_minmax",
            "eps": eps,
            "steps": steps,
            "step_size": step_size,
            "max_batches": max_batches,
            "init": self.init,
        }
