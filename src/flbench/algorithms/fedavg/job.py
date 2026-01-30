"""FedAvg algorithm runner (NVFLARE recipe) for simulation."""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import shutil
from pathlib import Path

from nvflare.apis.dxo import DataKind
from nvflare.app_common.aggregators import InTimeAccumulateWeightedAggregator
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking

logging.getLogger("nvflare").setLevel(logging.ERROR)


def _copy_run_result_to_results_dir(*, run_result: str | None, results_dir: str, job_name: str) -> Path | None:
    if not results_dir:
        return None
    if not run_result:
        return None

    src = Path(run_result)
    if not src.exists():
        return None

    results_root = Path(results_dir).expanduser()
    results_root.mkdir(parents=True, exist_ok=True)

    dest = results_root / job_name
    if dest.exists():
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = results_root / f"{job_name}__{ts}"

    if src.is_dir():
        shutil.copytree(src, dest)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    return dest


def _build_run_meta(args) -> dict:
    return {
        "task": args.task,
        "model": args.model,
        "alpha": args.alpha,
        "n_clients": args.n_clients,
        "num_rounds": args.num_rounds,
        "aggregation_epochs": args.aggregation_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
    }


def _load_run_meta(meta_path: str) -> dict | None:
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        return None


def run_fedavg(args) -> None:
    from flbench.core.registry import get_task

    task_spec = get_task(args.task)

    n_clients = args.n_clients
    num_rounds = args.num_rounds
    alpha = args.alpha
    num_workers = args.num_workers
    lr = args.lr
    batch_size = args.batch_size
    aggregation_epochs = args.aggregation_epochs
    task_short = args.task.replace("/", "__")
    job_name = args.name if args.name else f"fedavg__{task_short}__alpha{alpha}"

    print(f"Running FedAvg ({num_rounds} rounds) task={args.task} alpha={alpha} clients={n_clients}")

    if alpha <= 0.0:
        raise ValueError("alpha must be > 0 for Dirichlet non-iid split")

    split_root = task_spec.default_split_root
    train_idx_root = task_spec.split_and_save(
        num_sites=n_clients,
        split_dir_prefix=os.path.join(split_root, "dirichlet"),
        seed=args.seed,
        alpha=alpha,
    )

    train_script = os.path.join(os.path.dirname(__file__), "client.py")
    train_args = (
        f"--task {args.task} "
        f"--model {args.model} "
        f"--train_idx_root {train_idx_root} "
        f"--num_workers {num_workers} "
        f"--lr {lr} "
        f"--batch_size {batch_size} "
        f"--aggregation_epochs {aggregation_epochs} "
        f"--seed {args.seed}"
    )

    recipe = FedAvgRecipe(
        name=job_name,
        min_clients=n_clients,
        num_rounds=num_rounds,
        initial_model=task_spec.build_model(args.model),
        train_script=train_script,
        train_args=train_args,
        aggregator=InTimeAccumulateWeightedAggregator(
            expected_data_kind=DataKind.WEIGHT_DIFF,
            # Avoid repeated NVFLARE warnings about missing per-site "Aggregation_weight".
            # Actual FedAvg weighting is driven by MetaKey.NUM_STEPS_CURRENT_ROUND sent by clients.
            aggregation_weights={f"site-{i}": 1.0 for i in range(1, n_clients + 1)},
        ),
        aggregator_data_kind=DataKind.WEIGHT_DIFF,
    )

    if args.tracking != "none":
        add_experiment_tracking(recipe, tracking_type=args.tracking)

    sim_workspace_root = getattr(args, "sim_workspace_root", "/tmp/nvflare/simulation")
    job_workspace = os.path.join(sim_workspace_root, job_name)
    meta_path = os.path.join(job_workspace, "flbench_run_meta.json")
    if getattr(args, "resume", False) and os.path.exists(job_workspace):
        existing_meta = _load_run_meta(meta_path)
        if existing_meta is not None:
            if (
                existing_meta.get("task") != args.task
                or existing_meta.get("model") != args.model
            ):
                raise RuntimeError(
                    "Refusing to resume: task/model mismatch with existing workspace.\n"
                    f"- existing: task={existing_meta.get('task')} model={existing_meta.get('model')}\n"
                    f"- current:  task={args.task} model={args.model}\n"
                    "Fix: use a new job name, or delete the existing workspace."
                )
        else:
            print(
                "Warning: resume requested but existing workspace metadata is missing or unreadable. "
                "If you see model shape mismatch errors, use a new job name or delete the workspace."
            )
    elif os.path.exists(job_workspace):
        shutil.rmtree(job_workspace, ignore_errors=True)

    env = SimEnv(num_clients=n_clients, workspace_root=sim_workspace_root)
    run = recipe.execute(env)

    status = run.get_status()
    run_result = run.get_result()
    print("\nJob Status is:", status)
    print("Result can be found in:", run_result)

    copied_to = _copy_run_result_to_results_dir(
        run_result=run_result,
        results_dir=getattr(args, "results_dir", "results"),
        job_name=job_name,
    )
    if copied_to is not None:
        print("Results copied to:", str(copied_to))
    print()

    if os.path.exists(job_workspace):
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(_build_run_meta(args), f, indent=2, sort_keys=True)
        except OSError:
            pass
