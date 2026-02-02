from __future__ import annotations

import logging
import math
import os
import shlex
from pathlib import Path

from nvflare.apis.dxo import DataKind
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking

from flbench.core.aggregation import build_nvflare_aggregator
from flbench.core.logging import silence_nvflare
from flbench.core.server_base import BaseServer
from flbench.utils.results_utils import write_client_metrics_csv, write_global_metrics_csv, write_global_metrics_summary

silence_nvflare()


class FedAvgServer(BaseServer):
    def _build_run_meta(self) -> dict:
        return {
            "task": self.args.task,
            "model": self.args.model,
            "alpha": self.args.alpha,
            "num_clients": self.args.num_clients,
            "n_malicious": getattr(self.args, "n_malicious", 0),
            "malicious_mode": getattr(self.args, "malicious_mode", "random"),
            "malicious_seed": getattr(self.args, "malicious_seed", None),
            "global_rounds": self.args.global_rounds,
            "local_epochs": self.args.local_epochs,
            "client_fraction": getattr(self.args, "client_fraction", None),
            "batch_size": self.args.batch_size,
            "lr": self.args.lr,
            "seed": self.args.seed,
            "optimizer": getattr(self.args, "optimizer", None),
            "device": getattr(self.args, "device", None),
            "data_root": getattr(self.args, "data_root", None),
            "attack": getattr(self.args, "attack", "none"),
            "attack_kv": getattr(self.args, "attack_kv", None),
            "attack_config": getattr(self.args, "attack_config", None),
            "attack_seed": getattr(self.args, "attack_seed", None),
            "defense": getattr(self.args, "defense", "none"),
            "defense_kv": getattr(self.args, "defense_kv", None),
            "defense_config": getattr(self.args, "defense_config", None),
        }

    def run(self) -> None:
        from flbench.core.registry import get_task

        task_spec = get_task(self.args.task)

        num_clients = self.args.num_clients
        n_malicious = getattr(self.args, "n_malicious", 0)
        malicious_mode = getattr(self.args, "malicious_mode", "random")
        global_rounds = self.args.global_rounds
        alpha = self.args.alpha
        num_workers = self.args.num_workers
        lr = self.args.lr
        batch_size = self.args.batch_size
        local_epochs = self.args.local_epochs
        client_fraction = float(getattr(self.args, "client_fraction", 1.0))
        task_short = self.args.task.replace("/", "__")
        job_name = self.args.name if self.args.name else f"fedavg__{task_short}__alpha{alpha}"

        print(
            f"Running FedAvg ({global_rounds} rounds) task={self.args.task} alpha={alpha} "
            f"clients={num_clients} frac={client_fraction} malicious={n_malicious} mode={malicious_mode}"
        )

        if n_malicious < 0 or n_malicious > num_clients:
            raise ValueError("n_malicious must be between 0 and num_clients")

        if alpha <= 0.0:
            raise ValueError("alpha must be > 0 for Dirichlet non-iid split")

        split_root = task_spec.default_split_root
        train_idx_root = task_spec.split_and_save(
            num_sites=num_clients,
            split_dir_prefix=os.path.join(split_root, "dirichlet"),
            seed=self.args.seed,
            alpha=alpha,
            data_root=getattr(self.args, "data_root", None),
        )
        train_idx_root = os.path.abspath(train_idx_root)

        train_script = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "cli", "client_runner.py")
        )
        algo_key = getattr(self.args, "algo", "fedavg")
        train_args = (
            f"--algo {algo_key} "
            f"--task {self.args.task} "
            f"--model {self.args.model} "
            f"--train_idx_root {train_idx_root} "
            f"--num_workers {num_workers} "
            f"--lr {lr} "
            f"--batch_size {batch_size} "
            f"--local_epochs {local_epochs} "
            f"--seed {self.args.seed}"
        )
        train_args = (
            f"{train_args} --num_clients {num_clients} --n_malicious {n_malicious} --malicious_mode {malicious_mode}"
        )
        train_args = f"{train_args} --client_fraction {client_fraction}"
        if getattr(self.args, "malicious_seed", None) is not None:
            train_args = f"{train_args} --malicious_seed {self.args.malicious_seed}"

        # Attack plumbing: pass generic params so new attacks do not require touching any algorithm code.
        train_args = f"{train_args} --attack {self.args.attack}"
        for kv in (getattr(self.args, "attack_kv", None) or []):
            train_args = f"{train_args} --attack_kv {shlex.quote(str(kv))}"
        if getattr(self.args, "attack_config", None):
            train_args = f"{train_args} --attack_config {shlex.quote(str(self.args.attack_config))}"
        if getattr(self.args, "tracking", None) is not None:
            train_args = f"{train_args} --tracking {self.args.tracking}"
        if getattr(self.args, "data_root", None):
            train_args = f"{train_args} --data_root {self.args.data_root}"
        if getattr(self.args, "attack_seed", None) is not None:
            train_args = f"{train_args} --attack_seed {self.args.attack_seed}"
        if getattr(self.args, "optimizer", None):
            train_args = f"{train_args} --optimizer {self.args.optimizer}"
        if getattr(self.args, "device", None):
            train_args = f"{train_args} --device {self.args.device}"

        aggregator = build_nvflare_aggregator(
            args=self.args, num_clients=num_clients, expected_data_kind=DataKind.WEIGHT_DIFF
        )
        min_clients = max(1, math.ceil(num_clients * client_fraction))

        recipe = FedAvgRecipe(
            name=job_name,
            min_clients=min_clients,
            num_rounds=global_rounds,
            initial_model=task_spec.build_model(self.args.model),
            train_script=train_script,
            train_args=train_args,
            aggregator=aggregator,
            aggregator_data_kind=DataKind.WEIGHT_DIFF,
        )

        if self.args.tracking != "none":
            add_experiment_tracking(recipe, tracking_type=self.args.tracking)

        sim_workspace_root = getattr(self.args, "sim_workspace_root", "experiments/nvflare/simulation")
        job_workspace = os.path.join(sim_workspace_root, job_name)
        meta_path = os.path.join(job_workspace, "flbench_run_meta.json")
        if getattr(self.args, "resume", False) and os.path.exists(job_workspace):
            existing_meta = self._load_run_meta(meta_path)
            if existing_meta is not None:
                if existing_meta.get("task") != self.args.task or existing_meta.get("model") != self.args.model:
                    raise RuntimeError(
                        "Refusing to resume: task/model mismatch with existing workspace.\n"
                        f"- existing: task={existing_meta.get('task')} model={existing_meta.get('model')}\n"
                        f"- current:  task={self.args.task} model={self.args.model}\n"
                        "Fix: use a new job name, or delete the existing workspace."
                    )
            else:
                print(
                    "Warning: resume requested but existing workspace metadata is missing or unreadable. "
                    "If you see model shape mismatch errors, use a new job name or delete the workspace."
                )
        elif os.path.exists(job_workspace):
            from shutil import rmtree

            rmtree(job_workspace, ignore_errors=True)

        log_config = Path(__file__).resolve().parents[4] / "configs" / "nvflare_log_config.json"
        env = SimEnv(num_clients=num_clients, workspace_root=sim_workspace_root, log_config=str(log_config))
        run = recipe.execute(env)

        status = run.get_status()
        run_result = run.get_result()
        print("\\nJob Status is:", status)
        print("Result can be found in:", run_result)

        metrics_paths = write_global_metrics_csv(run_result)
        if metrics_paths is not None:
            print("Global metrics CSV saved to:", str(metrics_paths[0]))
            print("Global metrics summary CSV saved to:", str(metrics_paths[1]))
        client_metrics = write_client_metrics_csv(run_result)
        if client_metrics is not None:
            print("Client metrics CSV saved to:", str(client_metrics[0]))
        metrics_path = write_global_metrics_summary(run_result)
        if metrics_path is not None:
            print("Global metrics JSON saved to:", str(metrics_path))

        copied_to = self._copy_run_result_to_results_dir(
            run_result=run_result,
            results_dir=getattr(self.args, "results_dir", "experiments/runs"),
            job_name=job_name,
        )
        if copied_to is not None:
            print("Results copied to:", str(copied_to))
        print()

        if os.path.exists(job_workspace):
            self._write_run_meta(meta_path, self._build_run_meta())


def run_fedavg(args) -> None:
    FedAvgServer(args).run()
