from __future__ import annotations

import argparse

from flbench.core.client_base import BaseClient


class FedAvgClient(BaseClient):
    pass


def main(args):
    client = FedAvgClient(args)
    client.run()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="cifar10", help="Task key (e.g., cifar10).")
    p.add_argument("--model", type=str, default="cnn/moderate", help="Model key (task resolves it).")
    p.add_argument("--train_idx_root", type=str, default="/tmp/flbench_splits", help="Split index root dir")
    p.add_argument("--aggregation_epochs", type=int, default=4, help="Local epochs per FL round")
    p.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    p.add_argument("--no_lr_scheduler", action="store_true", help="Disable LR scheduler")
    p.add_argument("--cosine_lr_eta_min_factor", type=float, default=0.01, help="eta_min factor for cosine LR")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size")
    p.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    p.add_argument("--evaluate_local", action="store_true", help="Evaluate local model each epoch")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--tracking", type=str, default="tensorboard", choices=["tensorboard", "none"])
    p.add_argument("--data_root", type=str, default=None, help="Dataset root (task-dependent)")
    p.add_argument("--n_clients", type=int, default=0, help="Total number of clients")
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
    args = p.parse_args()

    main(args)
