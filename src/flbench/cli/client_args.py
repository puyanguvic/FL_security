from __future__ import annotations

import argparse
import sys

from flbench.utils.arg_utils import normalize_common_args
from flbench.utils.cli_kv import load_config_files, normalize_unified_config


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


def build_client_parser(config_defaults: dict | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        action="append",
        default=[],
        help="Path to YAML/JSON config (repeatable; later files override earlier).",
    )
    p.add_argument("--algo", type=str, default="fedavg", help="Algorithm key (e.g., fedavg, fedprox, scaffold).")
    p.add_argument("--task", type=str, default="cifar10", help="Task key (e.g., cifar10).")
    p.add_argument("--model", type=str, default="cnn/moderate", help="Model key (task resolves it).")
    p.add_argument("--train_idx_root", type=str, default="/tmp/flbench_splits", help="Split index root dir")
    p.add_argument("--local_epochs", "--aggregation_epochs", dest="local_epochs", type=int, default=5, help="Local epochs per FL round")
    p.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    p.add_argument("--prox_mu", type=float, default=0.0, help="FedProx mu coefficient")
    p.add_argument("--no_lr_scheduler", action="store_true", help="Disable LR scheduler")
    p.add_argument("--cosine_lr_eta_min_factor", type=float, default=0.01, help="eta_min factor for cosine LR")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size")
    p.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    p.add_argument("--evaluate_local", action="store_true", help="Evaluate local model each epoch")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--client_fraction", type=float, default=0.5, help="Fraction of clients per round (0 < f <= 1)")
    p.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam", "momentum"])
    p.add_argument("--device", type=str, default="cuda:0", help="Device: cpu or cuda:{idx}")
    p.add_argument("--tracking", type=str, default="tensorboard", choices=["tensorboard", "none"])
    p.add_argument("--data_root", type=str, default=None, help="Dataset root (task-dependent)")
    p.add_argument(
        "--num_clients",
        "--n_clients",
        dest="num_clients",
        type=int,
        default=10,
        help="Total number of clients",
    )
    p.add_argument("--n_malicious", type=int, default=0, help="Number of malicious clients")
    p.add_argument(
        "--malicious_mode",
        type=str,
        default="random",
        choices=["first", "random"],
        help="How to choose malicious clients",
    )
    p.add_argument("--malicious_seed", type=int, default=None, help="Seed for random malicious selection")
    p.add_argument(
        "--attack",
        type=str,
        default="none",
        help="Attack name (e.g., sign_flip, scale, gaussian, pgd_minmax)",
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
    return p


def parse_client_args(argv: list[str] | None = None):
    argv = sys.argv[1:] if argv is None else argv
    config_defaults = _load_config_defaults(argv)
    parser = build_client_parser(config_defaults)
    if config_defaults:
        parser.set_defaults(**_filter_defaults(parser, config_defaults))
    args = parser.parse_args(argv)
    normalize_common_args(args)
    return args
