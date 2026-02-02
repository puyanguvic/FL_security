from __future__ import annotations

import argparse
import sys

from flbench.attacks import list_attacks
from flbench.core.registry import get_algo, list_algos, list_tasks
from flbench.defenses import list_defenses
from flbench.utils.arg_utils import normalize_common_args
from flbench.utils.cli_kv import load_config_files, normalize_unified_config


def _has_value(cfg: dict | None, key: str) -> bool:
    if not cfg:
        return False
    if key not in cfg:
        return False
    v = cfg.get(key)
    return v is not None and v != ""


def _load_config_defaults(argv: list[str]) -> dict:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Path to YAML/JSON config (repeatable; later files override earlier).",
    )
    args, _ = config_parser.parse_known_args(argv)
    if not args.config:
        return {}
    raw = load_config_files(args.config)
    return normalize_unified_config(raw)


def _filter_defaults(parser: argparse.ArgumentParser, cfg: dict) -> dict:
    dests = {action.dest for action in parser._actions}
    return {k: v for k, v in cfg.items() if k in dests and k != "config"}


def build_parser(config_defaults: dict | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FLBench (NVFLARE-Lab) simulation runner")
    p.add_argument(
        "--config",
        action="append",
        default=[],
        help="Path to YAML/JSON config (repeatable; later files override earlier).",
    )
    p.add_argument(
        "--algo",
        type=str,
        required=not _has_value(config_defaults, "algo"),
        help=f"Algorithm. Available: {', '.join(list_algos())}",
    )
    p.add_argument(
        "--task",
        type=str,
        required=not _has_value(config_defaults, "task"),
        help=f"Task. Available: {', '.join(list_tasks())}",
    )
    p.add_argument("--model", type=str, default="cnn/moderate", help="Model key (task may ignore it).")
    p.add_argument("--name", type=str, default=None, help="Optional run/job name override")
    p.add_argument(
        "--results_dir",
        type=str,
        default="experiments/runs",
        help="Directory to copy NVFLARE simulation outputs into (default: ./experiments/runs).",
    )

    # Common FL knobs
    p.add_argument("--num_clients", "--n_clients", dest="num_clients", type=int, default=10)
    p.add_argument("--n_malicious", type=int, default=0, help="Number of malicious clients")
    p.add_argument(
        "--malicious_mode",
        type=str,
        default="random",
        choices=["first", "random"],
        help="How to choose malicious clients",
    )
    p.add_argument("--malicious_seed", type=int, default=None, help="Seed for random malicious selection")
    p.add_argument("--global_rounds", "--num_rounds", dest="global_rounds", type=int, default=20)
    p.add_argument("--local_epochs", "--aggregation_epochs", dest="local_epochs", type=int, default=5)
    p.add_argument("--client_fraction", type=float, default=0.5, help="Fraction of clients per round (0 < f <= 1)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam", "momentum"])
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda:0", help="Device: cpu or cuda:{idx}")
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
    p.add_argument("--seed", type=int, default=42)

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
    config_defaults = _load_config_defaults(sys.argv[1:])
    parser = build_parser(config_defaults)
    if config_defaults:
        parser.set_defaults(**_filter_defaults(parser, config_defaults))
    args = parser.parse_args()
    normalize_common_args(args)

    # validate
    _ = get_algo(args.algo)
    if args.n_malicious < 0 or args.n_malicious > args.num_clients:
        raise ValueError("n_malicious must be between 0 and num_clients")
    # task existence check
    # (currently task is used inside algo runner; we validate here for clearer errors)
    from flbench.core.registry import get_task

    _ = get_task(args.task)

    algo = get_algo(args.algo)
    algo.run(args)


if __name__ == "__main__":
    main()
