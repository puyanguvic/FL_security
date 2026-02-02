from __future__ import annotations

import copy
import logging
import random
import re

import nvflare.client as flare
import torch
import torch.nn as nn
import torch.optim as optim
from nvflare.apis.fl_constant import FLMetaKey
from nvflare.app_common.abstract.fl_model import ParamsType
from nvflare.client.tracking import SummaryWriter

from flbench.attacks import AttackContext, build_attack_from_args, diff_l2_norm, is_byzantine_attack_name
from flbench.core.logging import silence_nvflare
from flbench.utils.reproducibility import set_seed
from flbench.utils.torch_utils import compute_model_diff, evaluate_with_loss, get_lr_values


def _resolve_device(device_spec: str | torch.device | None) -> torch.device:
    if isinstance(device_spec, torch.device):
        device = device_spec
    else:
        spec = "cuda:0" if device_spec in (None, "") else str(device_spec).lower().strip()
        if spec == "cuda":
            spec = "cuda:0"
        if spec == "cpu":
            device = torch.device("cpu")
        elif spec.startswith("cuda:"):
            if not torch.cuda.is_available():
                logging.warning("CUDA requested but not available; falling back to CPU")
                return torch.device("cpu")
            try:
                idx = int(spec.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError("device must be 'cpu' or 'cuda:{idx}'") from exc
            if idx < 0:
                raise ValueError("device must be 'cpu' or 'cuda:{idx}'")
            if idx >= torch.cuda.device_count():
                raise ValueError(f"cuda device index {idx} is out of range")
            device = torch.device(spec)
        else:
            raise ValueError("device must be 'cpu' or 'cuda:{idx}'")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    return device

silence_nvflare()


class _NullSummaryWriter:
    def add_scalar(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None


class BaseClient:
    def __init__(self, args):
        self.args = args
        set_seed(args.seed)
        self.device = _resolve_device(getattr(args, "device", None))

        from flbench.core.registry import get_task

        self.task_spec = get_task(args.task)
        self.model = self.build_model(self.task_spec)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._build_optimizer()
        self.scheduler = None
        self.summary_writer = _NullSummaryWriter()

        self.site_name = None
        self.train_loader = None
        self.valid_loader = None

        self.is_malicious = False
        self.attack = None
        self._malicious_set = None

    def build_model(self, task_spec):
        return task_spec.build_model(self.args.model)

    def _build_optimizer(self):
        opt_name = str(getattr(self.args, "optimizer", "sgd") or "sgd").lower()
        if opt_name == "sgd":
            return optim.SGD(self.model.parameters(), lr=self.args.lr)
        if opt_name == "momentum":
            return optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9)
        if opt_name == "adam":
            return optim.Adam(self.model.parameters(), lr=self.args.lr)
        raise ValueError(f"Unknown optimizer '{opt_name}'")

    def compute_loss(self, outputs, labels, model, global_model):
        return self.criterion(outputs, labels)

    def after_backward(self, model, global_model):
        return None

    def before_round(self, input_model, global_model):
        return None

    def after_round(self, global_model, steps):
        return None

    def local_steps(self):
        return self.args.local_epochs * len(self.train_loader)

    def _site_index(self, site_name: str) -> int | None:
        match = re.search(r"(\d+)$", site_name)
        if not match:
            return None
        return int(match.group(1))

    def _is_malicious_site(self, site_name: str) -> bool:
        n_malicious = int(getattr(self.args, "n_malicious", 0) or 0)
        if n_malicious <= 0:
            return False
        site_idx = self._site_index(site_name)
        if site_idx is None:
            return False
        malicious_set = self._get_malicious_set()
        return site_idx in malicious_set

    def _get_malicious_set(self) -> set[int]:
        if self._malicious_set is not None:
            return self._malicious_set

        n_malicious = int(getattr(self.args, "n_malicious", 0) or 0)
        n_clients = int(getattr(self.args, "num_clients", 0) or 0)
        mode = str(getattr(self.args, "malicious_mode", "first")).lower()

        if n_malicious <= 0:
            self._malicious_set = set()
            return self._malicious_set

        if n_clients <= 0:
            if mode == "random":
                print(f"{self.site_name}: random malicious selection requires num_clients; falling back to first-k")
            self._malicious_set = set(range(1, n_malicious + 1))
            return self._malicious_set

        if n_malicious > n_clients:
            self._malicious_set = set(range(1, n_malicious + 1))
            return self._malicious_set

        if mode == "random":
            seed = getattr(self.args, "malicious_seed", None)
            if seed is None:
                seed = self.args.seed
            rng = random.Random(int(seed))
            self._malicious_set = set(rng.sample(range(1, n_clients + 1), k=n_malicious))
        else:
            self._malicious_set = set(range(1, n_malicious + 1))

        return self._malicious_set

    def _init_attack(self):
        self.is_malicious = self._is_malicious_site(self.site_name)
        attack_name = str(getattr(self.args, "attack", "none") or "none").lower()
        if is_byzantine_attack_name(attack_name):
            if self.is_malicious:
                print(f"{self.site_name}: marked as malicious (first {self.args.n_malicious})")
                print(f"{self.site_name}: server-side byzantine attack '{attack_name}' configured")
            else:
                print(f"{self.site_name}: byzantine attack '{attack_name}' configured but client is benign")
            self.attack = None
            return

        configured_attack = build_attack_from_args(self.args)
        if self.is_malicious:
            print(f"{self.site_name}: marked as malicious (first {self.args.n_malicious})")
            self.attack = configured_attack
            if self.attack is not None:
                print(f"Attack enabled: {self.attack.name}")
            else:
                print(f"{self.site_name}: attack disabled (attack=none)")
        else:
            self.attack = None
            if configured_attack is not None:
                print(f"{self.site_name}: attack configured but client is benign (n_malicious=0 or not selected)")

    def _init_scheduler(self, input_model):
        if self.scheduler is None and not self.args.no_lr_scheduler:
            total_rounds = input_model.total_rounds
            eta_min = self.args.lr * self.args.cosine_lr_eta_min_factor
            T_max = total_rounds * self.args.local_epochs
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max, eta_min=eta_min)
            print(f"{self.site_name}: CosineAnnealingLR: initial_lr={self.args.lr}, eta_min={eta_min}, T_max={T_max}")

    def _load_global_model(self, input_model):
        try:
            self.model.load_state_dict(input_model.params, strict=True)
        except RuntimeError as e:
            msg = str(e)
            if "size mismatch" in msg:
                raise RuntimeError(
                    "Global model shape mismatch between server and client.\n"
                    f"- client: task={self.args.task} model={self.args.model}\n"
                    "This usually happens when an old NVFLARE simulation workspace is reused (stale checkpoint) "
                    "or when server/client tasks differ.\n"
                    "Fix: rerun with a fresh job name or without `--resume` (default clears the sim workspace)."
                ) from e
            raise

    def run(self):
        flare.init()
        tracking = str(getattr(self.args, "tracking", "tensorboard")).lower()
        if tracking == "none":
            self.summary_writer = _NullSummaryWriter()
        else:
            try:
                self.summary_writer = SummaryWriter()
            except RuntimeError:
                self.summary_writer = _NullSummaryWriter()
        try:
            self.site_name = flare.get_site_name()
            print(f"Create datasets for site {self.site_name}")

            train_dataset, valid_dataset = self.task_spec.create_datasets(
                self.site_name,
                train_idx_root=self.args.train_idx_root,
                seed=self.args.seed,
                data_root=self.args.data_root,
            )
            self.train_loader, self.valid_loader = self.task_spec.create_data_loaders(
                train_dataset, valid_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers
            )

            self._init_attack()

            while True:
                try:
                    if not flare.is_running():
                        break
                except RuntimeError as exc:
                    msg = str(exc).lower()
                    if "abort" in msg:
                        print(f"{self.site_name}: server aborted job; stopping client loop.")
                        break
                    raise

                input_model = flare.receive()
                print(f"\\n[Current Round={input_model.current_round}, Site={self.site_name}]\\n")

                self._init_scheduler(input_model)
                self._load_global_model(input_model)

                global_model = copy.deepcopy(self.model)
                for p in global_model.parameters():
                    p.requires_grad = False

                self.model.to(self.device)
                global_model.to(self.device)

                self.before_round(input_model, global_model)

                val_loss_global_model, val_acc_global_model = evaluate_with_loss(
                    global_model, self.valid_loader, self.criterion
                )
                print(
                    "Global model validation - "
                    f"acc: {100 * val_acc_global_model:.2f}% "
                    f"loss: {val_loss_global_model:.4f}"
                )
                self.summary_writer.add_scalar("val_acc_global_model", val_acc_global_model, input_model.current_round)
                self.summary_writer.add_scalar(
                    "val_loss_global_model", val_loss_global_model, input_model.current_round
                )

                steps = self.local_steps()
                for epoch in range(self.args.local_epochs):
                    self.model.train()
                    running_loss = 0.0
                    running_correct = 0
                    running_total = 0
                    for data in self.train_loader:
                        inputs, labels = data[0].to(self.device), data[1].to(self.device)

                        self.optimizer.zero_grad()
                        outputs = self.model(inputs)
                        loss = self.compute_loss(outputs, labels, self.model, global_model)
                        loss.backward()
                        self.after_backward(self.model, global_model)
                        self.optimizer.step()

                        running_loss += loss.item()
                        preds = outputs.argmax(dim=1)
                        running_correct += (preds == labels).sum().item()
                        running_total += labels.numel()

                    avg_loss = running_loss / max(1, len(self.train_loader))
                    train_acc = float(running_correct) / float(running_total) if running_total > 0 else 0.0
                    global_epoch = input_model.current_round * self.args.local_epochs + epoch
                    curr_lr = get_lr_values(self.optimizer)[0]

                    self.summary_writer.add_scalar("global_round", input_model.current_round, global_epoch)
                    self.summary_writer.add_scalar("global_epoch", global_epoch, global_epoch)
                    self.summary_writer.add_scalar("train_loss", avg_loss, global_epoch)
                    self.summary_writer.add_scalar("train_acc", train_acc, global_epoch)
                    self.summary_writer.add_scalar("learning_rate", curr_lr, global_epoch)

                    print(
                        f"{self.site_name}: Epoch [{epoch + 1}/{self.args.local_epochs}] "
                        f"- Loss: {avg_loss:.4f} - LR: {curr_lr:.6f}"
                    )

                    val_loss_local_model, val_acc_local_model = evaluate_with_loss(
                        self.model, self.valid_loader, self.criterion
                    )
                    self.summary_writer.add_scalar("val_acc_local_model", val_acc_local_model, global_epoch)
                    self.summary_writer.add_scalar("val_loss_local_model", val_loss_local_model, global_epoch)
                    if self.args.evaluate_local:
                        print(
                            "Local model validation - "
                            f"acc: {100 * val_acc_local_model:.2f}% "
                            f"loss: {val_loss_local_model:.4f}"
                        )

                    if self.scheduler is not None:
                        self.scheduler.step()

                print(f"Finished training for current round {input_model.current_round}")
                self.after_round(global_model, steps)

                model_diff, diff_norm = compute_model_diff(self.model, global_model)
                base_norm = float(diff_norm)
                self.summary_writer.add_scalar("diff_norm", base_norm, input_model.current_round)

                attack_info = None
                if self.attack is not None:
                    rng = None
                    if self.args.attack_seed is not None:
                        rng = torch.Generator(device="cpu")
                        rng.manual_seed(int(self.args.attack_seed) + int(input_model.current_round))
                    ctx = AttackContext(
                        base_diff=model_diff,
                        base_norm=base_norm,
                        global_model=global_model,
                        local_model=self.model,
                        train_loader=self.train_loader,
                        criterion=self.criterion,
                        device=self.device,
                        current_round=int(input_model.current_round),
                        rng=rng,
                    )
                    model_diff, attack_info = self.attack.apply(model_diff, ctx)
                    attacked_norm = diff_l2_norm(model_diff)
                    self.summary_writer.add_scalar("diff_norm_attacked", attacked_norm, input_model.current_round)
                    self.summary_writer.add_scalar(
                        "diff_norm_ratio",
                        attacked_norm / (base_norm + 1e-12),
                        input_model.current_round,
                    )
                    print(f"{self.site_name}: attack={self.attack.name} info={attack_info}")

                num_examples = len(self.train_loader.dataset)
                meta = {
                    # NVFlare's default FedAvg recipe uses NUM_STEPS_CURRENT_ROUND as the aggregation weight.
                    FLMetaKey.NUM_STEPS_CURRENT_ROUND: float(num_examples),
                    "local_steps_current_round": int(steps),
                }
                if self.attack is not None:
                    meta["attack_name"] = self.attack.name
                    if attack_info is not None:
                        meta["attack_info"] = attack_info

                output_model = flare.FLModel(
                    params=model_diff,
                    params_type=ParamsType.DIFF,
                    metrics={"accuracy": val_acc_global_model, "loss": val_loss_global_model},
                    meta=meta,
                )

                try:
                    flare.send(output_model)
                except RuntimeError as exc:
                    msg = str(exc).lower()
                    if "abort" in msg:
                        print(f"{self.site_name}: server aborted job; stopping client loop.")
                        break
                    raise
        finally:
            close_fn = getattr(self.summary_writer, "close", None)
            if callable(close_fn):
                close_fn()
