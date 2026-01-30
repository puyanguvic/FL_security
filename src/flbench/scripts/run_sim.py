from __future__ import annotations

import argparse

from flbench.core.registry import get_algo, list_algos, list_tasks


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FLBench (NVFLARE-Lab) simulation runner")
    p.add_argument("--algo", type=str, required=True, help=f"Algorithm. Available: {', '.join(list_algos())}")
    p.add_argument("--task", type=str, required=True, help=f"Task. Available: {', '.join(list_tasks())}")
    p.add_argument("--model", type=str, default="cnn/moderate", help="Model key (task may ignore it).")
    p.add_argument("--name", type=str, default=None, help="Optional run/job name override")
    p.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to copy NVFLARE simulation outputs into (default: ./results).",
    )

    # Common FL knobs
    p.add_argument("--n_clients", type=int, default=8)
    p.add_argument("--num_rounds", type=int, default=50)
    p.add_argument("--aggregation_epochs", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--prox_mu", type=float, default=0.0, help="FedProx mu coefficient (only for fedprox).")

    # Non-iid split knobs (task-dependent; CIFAR-10 uses alpha)
    p.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha (for non-iid splits).")

    # Repro
    p.add_argument("--seed", type=int, default=0)

    # Simulation workspace behavior
    p.add_argument(
        "--sim_workspace_root",
        type=str,
        default="/tmp/nvflare/simulation",
        help="NVFLARE simulation workspace root (default: /tmp/nvflare/simulation).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing NVFLARE simulation workspace if present (default: start fresh).",
    )

    # Tracking
    p.add_argument("--tracking", type=str, default="tensorboard", choices=["tensorboard", "none"])

    return p


def main():
    args = build_parser().parse_args()

    # validate
    _ = get_algo(args.algo)
    # task existence check
    # (currently task is used inside algo runner; we validate here for clearer errors)
    from flbench.core.registry import get_task

    _ = get_task(args.task)

    algo = get_algo(args.algo)
    algo.run(args)


if __name__ == "__main__":
    main()
