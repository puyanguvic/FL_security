from __future__ import annotations

import logging
import os

from nvflare.apis.dxo import DataKind
from nvflare.app_common.aggregators import InTimeAccumulateWeightedAggregator
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking

from flbench.core.server_base import BaseServer
from flbench.defenses import build_defense_from_args
from flbench.defenses.aggregator import DefenseAggregator
from flbench.utils.results_utils import write_client_metrics_csv, write_global_metrics_csv, write_global_metrics_summary

logging.getLogger("nvflare").setLevel(logging.ERROR)


class FedAvgServer(BaseServer):
    def _build_run_meta(self) -> dict:
        return {
            "task": self.args.task,
            "model": self.args.model,
            "alpha": self.args.alpha,
            "n_clients": self.args.n_clients,
            "n_malicious": getattr(self.args, "n_malicious", 0),
            "malicious_mode": getattr(self.args, "malicious_mode", "random"),
            "malicious_seed": getattr(self.args, "malicious_seed", None),
            "num_rounds": self.args.num_rounds,
            "aggregation_epochs": self.args.aggregation_epochs,
            "batch_size": self.args.batch_size,
            "lr": self.args.lr,
            "seed": self.args.seed,
            "data_root": getattr(self.args, "data_root", None),
            "attack": getattr(self.args, "attack", "none"),
            "attack_scale": getattr(self.args, "attack_scale", None),
            "attack_noise_std": getattr(self.args, "attack_noise_std", None),
            "attack_pgd_steps": getattr(self.args, "attack_pgd_steps", None),
            "attack_pgd_step_size": getattr(self.args, "attack_pgd_step_size", None),
            "attack_pgd_eps": getattr(self.args, "attack_pgd_eps", None),
            "attack_pgd_eps_factor": getattr(self.args, "attack_pgd_eps_factor", None),
            "attack_pgd_max_batches": getattr(self.args, "attack_pgd_max_batches", None),
            "attack_pgd_init": getattr(self.args, "attack_pgd_init", None),
            "attack_seed": getattr(self.args, "attack_seed", None),
            "defense": getattr(self.args, "defense", "none"),
            "defense_trim_ratio": getattr(self.args, "defense_trim_ratio", None),
            "defense_clip_norm": getattr(self.args, "defense_clip_norm", None),
            "defense_krum_f": getattr(self.args, "defense_krum_f", None),
            "defense_krum_m": getattr(self.args, "defense_krum_m", None),
        }

    def run(self) -> None:
        from flbench.core.registry import get_task

        task_spec = get_task(self.args.task)

        n_clients = self.args.n_clients
        n_malicious = getattr(self.args, "n_malicious", 0)
        malicious_mode = getattr(self.args, "malicious_mode", "random")
        num_rounds = self.args.num_rounds
        alpha = self.args.alpha
        num_workers = self.args.num_workers
        lr = self.args.lr
        batch_size = self.args.batch_size
        aggregation_epochs = self.args.aggregation_epochs
        task_short = self.args.task.replace("/", "__")
        job_name = self.args.name if self.args.name else f"fedavg__{task_short}__alpha{alpha}"

        print(
            f"Running FedAvg ({num_rounds} rounds) task={self.args.task} alpha={alpha} "
            f"clients={n_clients} malicious={n_malicious} mode={malicious_mode}"
        )

        if n_malicious < 0 or n_malicious > n_clients:
            raise ValueError("n_malicious must be between 0 and n_clients")

        if alpha <= 0.0:
            raise ValueError("alpha must be > 0 for Dirichlet non-iid split")

        split_root = task_spec.default_split_root
        train_idx_root = task_spec.split_and_save(
            num_sites=n_clients,
            split_dir_prefix=os.path.join(split_root, "dirichlet"),
            seed=self.args.seed,
            alpha=alpha,
            data_root=getattr(self.args, "data_root", None),
        )
        train_idx_root = os.path.abspath(train_idx_root)

        train_script = os.path.join(os.path.dirname(__file__), "client.py")
        train_args = (
            f"--task {self.args.task} "
            f"--model {self.args.model} "
            f"--train_idx_root {train_idx_root} "
            f"--num_workers {num_workers} "
            f"--lr {lr} "
            f"--batch_size {batch_size} "
            f"--aggregation_epochs {aggregation_epochs} "
            f"--seed {self.args.seed}"
        )
        train_args = (
            f"{train_args} --n_clients {n_clients} --n_malicious {n_malicious} --malicious_mode {malicious_mode}"
        )
        if getattr(self.args, "malicious_seed", None) is not None:
            train_args = f"{train_args} --malicious_seed {self.args.malicious_seed}"
        train_args = (
            f"{train_args} "
            f"--attack {self.args.attack} "
            f"--attack_scale {self.args.attack_scale} "
            f"--attack_noise_std {self.args.attack_noise_std} "
            f"--attack_pgd_steps {self.args.attack_pgd_steps} "
            f"--attack_pgd_step_size {self.args.attack_pgd_step_size} "
            f"--attack_pgd_eps {self.args.attack_pgd_eps} "
            f"--attack_pgd_eps_factor {self.args.attack_pgd_eps_factor} "
            f"--attack_pgd_max_batches {self.args.attack_pgd_max_batches} "
            f"--attack_pgd_init {self.args.attack_pgd_init}"
        )
        if getattr(self.args, "tracking", None) is not None:
            train_args = f"{train_args} --tracking {self.args.tracking}"
        if getattr(self.args, "data_root", None):
            train_args = f"{train_args} --data_root {self.args.data_root}"
        if getattr(self.args, "attack_seed", None) is not None:
            train_args = f"{train_args} --attack_seed {self.args.attack_seed}"

        defense = build_defense_from_args(self.args)
        aggregation_weights = {f"site-{i}": 1.0 for i in range(1, n_clients + 1)}
        if defense is None:
            aggregator = InTimeAccumulateWeightedAggregator(
                expected_data_kind=DataKind.WEIGHT_DIFF,
                # Avoid repeated NVFLARE warnings about missing per-site "Aggregation_weight".
                # Actual FedAvg weighting is driven by MetaKey.NUM_STEPS_CURRENT_ROUND sent by clients.
                aggregation_weights=aggregation_weights,
            )
        else:
            aggregator = DefenseAggregator(
                defense=defense,
                expected_data_kind=DataKind.WEIGHT_DIFF,
                aggregation_weights=aggregation_weights,
                weigh_by_local_iter=True,
            )

        recipe = FedAvgRecipe(
            name=job_name,
            min_clients=n_clients,
            num_rounds=num_rounds,
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

        env = SimEnv(num_clients=n_clients, workspace_root=sim_workspace_root)
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
