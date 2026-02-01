from __future__ import annotations

import argparse

from flbench.attacks import list_attacks
from flbench.core.registry import get_algo, list_algos, list_tasks
from flbench.defenses import list_defenses


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FLBench (NVFLARE-Lab) simulation runner")
    p.add_argument("--algo", type=str, required=True, help=f"Algorithm. Available: {', '.join(list_algos())}")
    p.add_argument("--task", type=str, required=True, help=f"Task. Available: {', '.join(list_tasks())}")
    p.add_argument("--model", type=str, default="cnn/moderate", help="Model key (task may ignore it).")
    p.add_argument("--name", type=str, default=None, help="Optional run/job name override")
    p.add_argument(
        "--results_dir",
        type=str,
        default="experiments/runs",
        help="Directory to copy NVFLARE simulation outputs into (default: ./experiments/runs).",
    )

    # Common FL knobs
    p.add_argument("--n_clients", type=int, default=8)
    p.add_argument("--n_malicious", type=int, default=0, help="Number of malicious clients")
    p.add_argument(
        "--malicious_mode",
        type=str,
        default="random",
        choices=["first", "random"],
        help="How to choose malicious clients",
    )
    p.add_argument("--malicious_seed", type=int, default=None, help="Seed for random malicious selection")
    p.add_argument("--num_rounds", type=int, default=50)
    p.add_argument("--aggregation_epochs", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--prox_mu", type=float, default=0.0, help="FedProx mu coefficient (only for fedprox).")
    p.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Dataset root (task-dependent). Passed to task split/dataset loaders.",
    )

    # Non-iid split knobs (task-dependent; CIFAR-10 uses alpha)
    p.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha (for non-iid splits).")

    # Repro
    p.add_argument("--seed", type=int, default=0)

    # Simulation workspace behavior
    p.add_argument(
        "--sim_workspace_root",
        type=str,
        default="experiments/nvflare/simulation",
        help="NVFLARE simulation workspace root (default: ./experiments/nvflare/simulation).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing NVFLARE simulation workspace if present (default: start fresh).",
    )

    # Tracking
    p.add_argument("--tracking", type=str, default="tensorboard", choices=["tensorboard", "none"])

    # Client-side update attacks
    attack_choices = ["none"] + list(list_attacks())
    p.add_argument(
        "--attack",
        type=str,
        default="none",
        choices=attack_choices,
        help=f"Client update attack. Available: {', '.join(attack_choices)}",
    )
    p.add_argument(
        "--attack_kv",
        action="append",
        default=[],
        help="Attack params as key=value (repeatable). Example: --attack_kv eps=5.0 --attack_kv steps=3",
    )
    p.add_argument(
        "--attack_config",
        type=str,
        default=None,
        help="Path to YAML/JSON file providing attack params (can be combined with --attack_kv overrides)",
    )
    p.add_argument("--attack_scale", type=float, default=1.0, help="Scale factor for scale attack")
    p.add_argument("--attack_noise_std", type=float, default=0.0, help="Gaussian noise std factor (relative)")
    p.add_argument("--attack_pgd_steps", type=int, default=1, help="PGD steps for pgd_minmax")
    p.add_argument("--attack_pgd_step_size", type=float, default=0.1, help="PGD step size for pgd_minmax")
    p.add_argument(
        "--attack_pgd_eps",
        type=float,
        default=0.0,
        help="PGD l2 epsilon (absolute). Overrides eps_factor if > 0",
    )
    p.add_argument(
        "--attack_pgd_eps_factor",
        type=float,
        default=1.0,
        help="PGD l2 epsilon factor (relative to base diff norm)",
    )
    p.add_argument("--attack_pgd_max_batches", type=int, default=1, help="Max batches per PGD step")
    p.add_argument(
        "--attack_pgd_init",
        type=str,
        default="zero",
        choices=["zero", "local", "sign"],
        help="PGD init: zero, local (use base diff), or sign (negated base diff)",
    )
    p.add_argument("--attack_seed", type=int, default=None, help="Optional seed for stochastic attacks")

    # Server-side defenses
    defense_choices = ["none"] + list(list_defenses())
    p.add_argument(
        "--defense",
        type=str,
        default="none",
        choices=defense_choices,
        help=f"Server defense. Available: {', '.join(defense_choices)}",
    )
    p.add_argument(
        "--defense_kv",
        action="append",
        default=[],
        help="Defense params as key=value (repeatable). Example: --defense_kv trim_ratio=0.2",
    )
    p.add_argument(
        "--defense_config",
        type=str,
        default=None,
        help="Path to YAML/JSON file providing defense params (can be combined with --defense_kv overrides)",
    )
    p.add_argument("--defense_trim_ratio", type=float, default=0.0, help="Trim ratio for trimmed_mean defense")
    p.add_argument("--defense_clip_norm", type=float, default=0.0, help="Clip norm for norm_clip defense")
    p.add_argument("--defense_krum_f", type=int, default=-1, help="Byzantine count f for (multi-)krum")
    p.add_argument("--defense_krum_m", type=int, default=1, help="Selected count m for multi-krum")

    return p


def main():
    args = build_parser().parse_args()

    # validate
    _ = get_algo(args.algo)
    if args.n_malicious < 0 or args.n_malicious > args.n_clients:
        raise ValueError("n_malicious must be between 0 and n_clients")
    # task existence check
    # (currently task is used inside algo runner; we validate here for clearer errors)
    from flbench.core.registry import get_task

    _ = get_task(args.task)

    algo = get_algo(args.algo)
    algo.run(args)


if __name__ == "__main__":
    main()
