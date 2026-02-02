from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from flbench.utils.cli_kv import merge_kv_and_file
from .byzantine_registry import is_byzantine_attack_name

StateDict = Dict[str, torch.Tensor]


@dataclass(frozen=True)
class AttackContext:
    base_diff: StateDict
    base_norm: float
    global_model: nn.Module
    local_model: nn.Module
    train_loader: Any
    criterion: nn.Module
    device: torch.device
    current_round: int
    rng: Optional[torch.Generator] = None


class UpdateAttack:
    name: str = ""

    def apply(self, diff: StateDict, ctx: AttackContext) -> Tuple[StateDict, Dict[str, Any] | None]:
        raise NotImplementedError


class ScaleAttack(UpdateAttack):
    name = "scale"

    def __init__(self, scale: float) -> None:
        self.scale = float(scale)

    def apply(self, diff: StateDict, ctx: AttackContext):
        attacked = {k: v * self.scale for k, v in diff.items()}
        return attacked, {"scale": self.scale}


class SignFlipAttack(UpdateAttack):
    name = "sign_flip"

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = float(scale)

    def apply(self, diff: StateDict, ctx: AttackContext):
        attacked = {k: -self.scale * v for k, v in diff.items()}
        return attacked, {"scale": self.scale}


class GaussianNoiseAttack(UpdateAttack):
    name = "gaussian"

    def __init__(self, noise_std: float) -> None:
        self.noise_std = float(noise_std)

    def apply(self, diff: StateDict, ctx: AttackContext):
        if self.noise_std <= 0:
            return diff, {"noise_std": self.noise_std}

        numel = 0
        for v in diff.values():
            if torch.is_floating_point(v):
                numel += v.numel()
        if numel == 0:
            return diff, {"noise_std": self.noise_std}

        base_norm = float(ctx.base_norm)
        target_norm = self.noise_std * base_norm if base_norm > 0 else self.noise_std
        per_elem_std = target_norm / math.sqrt(numel)

        attacked: StateDict = {}
        for k, v in diff.items():
            if torch.is_floating_point(v):
                noise = torch.randn_like(v, generator=ctx.rng) * per_elem_std
                attacked[k] = v + noise
            else:
                attacked[k] = v
        return attacked, {"noise_std": self.noise_std, "per_elem_std": per_elem_std}


class PGDMinMaxAttack(UpdateAttack):
    name = "pgd_minmax"

    def __init__(
        self,
        steps: int,
        step_size: float,
        eps: float,
        eps_factor: float,
        max_batches: int,
        init: str,
    ) -> None:
        self.steps = int(steps)
        self.step_size = float(step_size)
        self.eps = float(eps)
        self.eps_factor = float(eps_factor)
        self.max_batches = int(max_batches)
        self.init = str(init)

    def apply(self, diff: StateDict, ctx: AttackContext):
        if self.steps <= 0 or ctx.train_loader is None:
            return diff, {"steps": 0}

        radius = self.eps if self.eps > 0 else self.eps_factor * float(ctx.base_norm)
        if radius <= 0:
            return diff, {"steps": 0, "eps": radius}

        device = ctx.device
        global_model = ctx.global_model

        global_state = {k: v.detach().to(device) for k, v in global_model.state_dict().items()}

        def _project(d: StateDict) -> StateDict:
            norm = diff_l2_norm(d)
            if norm <= radius:
                return d
            scale = radius / (norm + 1e-12)
            return {k: v * scale for k, v in d.items()}

        def _init_diff() -> StateDict:
            if self.init == "zero":
                return {k: torch.zeros_like(v, device=device) for k, v in diff.items()}
            if self.init == "sign":
                d = {k: torch.sign(v.to(device)) for k, v in diff.items()}
                return _project(d)
            # default: local (use base diff)
            return {k: v.to(device) for k, v in diff.items()}

        def _load_model(m: nn.Module, d: StateDict) -> None:
            with torch.no_grad():
                sd = m.state_dict()
                for k in sd.keys():
                    if k in d:
                        sd[k].copy_(global_state[k] + d[k])
                    else:
                        sd[k].copy_(global_state[k])
            m.load_state_dict(sd)

        work_diff = _init_diff()
        work_diff = _project(work_diff)

        model = copy.deepcopy(global_model).to(device)
        model.train()
        _load_model(model, work_diff)

        for _ in range(self.steps):
            model.zero_grad(set_to_none=True)
            batches = 0
            total_loss = 0.0
            for batch in ctx.train_loader:
                if batches >= self.max_batches:
                    break
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = ctx.criterion(logits, y)
                total_loss += float(loss.item())
                loss.backward()
                batches += 1
            if batches == 0:
                break

            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is None:
                        continue
                    p.add_(p.grad, alpha=self.step_size)

            # Recompute diff from model params and project
            new_diff: StateDict = {}
            model_state = model.state_dict()
            for k, gv in global_state.items():
                if k in model_state:
                    new_diff[k] = model_state[k] - gv
            work_diff = _project(new_diff)
            _load_model(model, work_diff)

        # Return CPU tensors for compatibility with NVFLARE shareables
        attacked = {k: v.detach().cpu() for k, v in work_diff.items()}
        return attacked, {
            "steps": self.steps,
            "step_size": self.step_size,
            "eps": radius,
            "init": self.init,
        }


_UPDATE_ATTACKS = {
    "scale": ScaleAttack,
    "sign_flip": SignFlipAttack,
    "gaussian": GaussianNoiseAttack,
    "pgd_minmax": PGDMinMaxAttack,
}


def list_attacks() -> Tuple[str, ...]:
    return tuple(sorted(_UPDATE_ATTACKS.keys()))


def list_update_attacks() -> Tuple[str, ...]:
    return list_attacks()


def diff_l2_norm(diff: StateDict) -> float:
    sq_sum = 0.0
    for v in diff.values():
        sq_sum += v.float().pow(2).sum().item()
    return float(torch.tensor(sq_sum).sqrt().item())


def _get(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in cfg:
            return cfg[k]
    return default


def build_attack_from_args(args) -> UpdateAttack | None:
    name = str(getattr(args, "attack", "none") or "none").lower()
    if name in {"none", "null", ""}:
        return None
    if is_byzantine_attack_name(name):
        # Server-side byzantine attacks are handled in the aggregator.
        return None

    cfg = merge_kv_and_file(kv=getattr(args, "attack_kv", None), config_path=getattr(args, "attack_config", None))

    if name == "scale":
        scale = float(_get(cfg, "scale", default=getattr(args, "attack_scale", 1.0)))
        return ScaleAttack(scale=scale)

    if name == "sign_flip":
        scale = float(_get(cfg, "scale", default=1.0))
        return SignFlipAttack(scale=scale)

    if name == "gaussian":
        noise_std = float(_get(cfg, "noise_std", "std", default=getattr(args, "attack_noise_std", 0.0)))
        return GaussianNoiseAttack(noise_std=noise_std)

    if name == "pgd_minmax":
        steps = int(_get(cfg, "steps", "pgd_steps", default=getattr(args, "attack_pgd_steps", 1)))
        step_size = float(_get(cfg, "step_size", "pgd_step_size", default=getattr(args, "attack_pgd_step_size", 0.1)))
        eps = float(_get(cfg, "eps", "pgd_eps", default=getattr(args, "attack_pgd_eps", 0.0)))
        eps_factor = float(
            _get(cfg, "eps_factor", "pgd_eps_factor", default=getattr(args, "attack_pgd_eps_factor", 1.0))
        )
        max_batches = int(
            _get(cfg, "max_batches", "pgd_max_batches", default=getattr(args, "attack_pgd_max_batches", 1))
        )
        init = str(_get(cfg, "init", "pgd_init", default=getattr(args, "attack_pgd_init", "zero")))
        return PGDMinMaxAttack(
            steps=steps,
            step_size=step_size,
            eps=eps,
            eps_factor=eps_factor,
            max_batches=max_batches,
            init=init,
        )

    raise KeyError(f"Unknown update attack '{name}'. Available: {list_attacks()}")
