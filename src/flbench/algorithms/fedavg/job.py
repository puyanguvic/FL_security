"""FedAvg algorithm runner (NVFLARE recipe) for simulation."""
from __future__ import annotations

import os
import logging
from nvflare.apis.dxo import DataKind
from nvflare.app_common.aggregators import InTimeAccumulateWeightedAggregator
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking

from flbench.models.cnn import ModerateCNN
from flbench.tasks.vision.cifar10.split import split_and_save

logging.getLogger("nvflare").setLevel(logging.ERROR)


def run_fedavg(args) -> None:
    if args.task != "vision/cifar10":
        raise ValueError(f"fedavg runner currently supports task=vision/cifar10 only (got {args.task}).")

    n_clients = args.n_clients
    num_rounds = args.num_rounds
    alpha = args.alpha
    num_workers = args.num_workers
    lr = args.lr
    batch_size = args.batch_size
    aggregation_epochs = args.aggregation_epochs
    job_name = args.name if args.name else f"fedavg__cifar10__alpha{alpha}"

    print(f"Running FedAvg ({num_rounds} rounds) task={args.task} alpha={alpha} clients={n_clients}")

    if alpha <= 0.0:
        raise ValueError("alpha must be > 0 for Dirichlet non-iid split")

    split_root = "/tmp/flbench_splits/cifar10"
    train_idx_root = split_and_save(
        num_sites=n_clients,
        alpha=alpha,
        split_dir_prefix=os.path.join(split_root, "dirichlet"),
        seed=args.seed,
    )

    train_script = os.path.join(os.path.dirname(__file__), "client.py")
    train_args = (
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
        initial_model=ModerateCNN(),
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

    env = SimEnv(num_clients=n_clients)
    run = recipe.execute(env)

    print("\nJob Status is:", run.get_status())
    print("Result can be found in:", run.get_result())
    print()
