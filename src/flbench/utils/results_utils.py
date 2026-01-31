from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _load_scalar_events(site_dir: Path, tag: str):
    try:
        accumulator = EventAccumulator(str(site_dir), size_guidance={"scalars": 0})
        accumulator.Reload()
    except Exception:
        return []
    try:
        return accumulator.Scalars(tag)
    except KeyError:
        return []


def _aggregate_scalars(site_dirs: List[Path], tag: str) -> Dict[int, List[float]]:
    values_by_round: Dict[int, List[float]] = defaultdict(list)
    for site_dir in site_dirs:
        for scalar in _load_scalar_events(site_dir, tag):
            values_by_round[int(scalar.step)].append(float(scalar.value))
    return values_by_round


def _collect_site_epoch_rows(site_dir: Path, tags: List[str]) -> Dict[int, Dict[str, float]]:
    rows: Dict[int, Dict[str, float]] = {}
    for tag in tags:
        for scalar in _load_scalar_events(site_dir, tag):
            epoch = int(scalar.step)
            row = rows.setdefault(epoch, {"epoch": epoch})
            row[tag] = float(scalar.value)
    return rows


def _resolve_tb_root(run_root: Path) -> Path | None:
    candidates = [
        run_root / "tb_events",
        run_root / "simulate_job" / "tb_events",
        run_root / "server" / "simulate_job" / "tb_events",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_global_metrics_summary(run_result: str | Path) -> dict | None:
    run_root = Path(run_result)
    tb_root = _resolve_tb_root(run_root)
    if tb_root is None:
        return None

    site_dirs = sorted(p for p in tb_root.iterdir() if p.is_dir() and p.name.startswith("site-"))
    if not site_dirs:
        return None

    acc_by_round = _aggregate_scalars(site_dirs, "val_acc_global_model")
    loss_by_round = _aggregate_scalars(site_dirs, "val_loss_global_model")

    if not acc_by_round and not loss_by_round:
        return None

    all_rounds = sorted(set(acc_by_round.keys()) | set(loss_by_round.keys()))
    acc_series = []
    loss_series = []
    for round_idx in all_rounds:
        acc_vals = acc_by_round.get(round_idx, [])
        loss_vals = loss_by_round.get(round_idx, [])
        if acc_vals:
            acc_series.append(
                {"round": round_idx, "value": sum(acc_vals) / len(acc_vals), "n_sites": len(acc_vals)}
            )
        if loss_vals:
            loss_series.append(
                {"round": round_idx, "value": sum(loss_vals) / len(loss_vals), "n_sites": len(loss_vals)}
            )

    best_acc = None
    final_acc = None
    if acc_series:
        best_entry = max(acc_series, key=lambda item: item["value"])
        best_acc = {"value": best_entry["value"], "round": best_entry["round"]}
        final_entry = max(acc_series, key=lambda item: item["round"])
        final_acc = {"value": final_entry["value"], "round": final_entry["round"]}

    return {
        "global_validation_accuracy": acc_series,
        "global_validation_loss": loss_series,
        "best_global_accuracy": best_acc,
        "final_global_accuracy": final_acc,
        "source_tags": {
            "global_validation_accuracy": "val_acc_global_model",
            "global_validation_loss": "val_loss_global_model",
        },
    }


def write_global_metrics_summary(run_result: str | Path, filename: str = "flbench_global_metrics.json") -> Path | None:
    summary = build_global_metrics_summary(run_result)
    if summary is None:
        return None
    run_root = Path(run_result)
    tb_root = _resolve_tb_root(run_root)
    if tb_root is None:
        return None
    base_dir = tb_root.parent
    output_path = base_dir / filename
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
    except OSError:
        return None
    return output_path


def write_global_metrics_csv(
    run_result: str | Path,
    metrics_dirname: str = "metrics",
    round_filename: str = "global_round_metrics.csv",
    summary_filename: str = "global_summary_metrics.csv",
) -> Tuple[Path, Path] | None:
    summary = build_global_metrics_summary(run_result)
    if summary is None:
        return None

    run_root = Path(run_result)
    tb_root = _resolve_tb_root(run_root)
    if tb_root is None:
        return None
    base_dir = tb_root.parent
    metrics_dir = base_dir / metrics_dirname
    try:
        metrics_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None

    round_path = metrics_dir / round_filename
    summary_path = metrics_dir / summary_filename

    round_rows = {}
    for entry in summary.get("global_validation_accuracy", []):
        round_rows[int(entry["round"])] = {
            "round": int(entry["round"]),
            "global_validation_accuracy": float(entry["value"]),
        }
    for entry in summary.get("global_validation_loss", []):
        row = round_rows.setdefault(int(entry["round"]), {"round": int(entry["round"])})
        row["global_validation_loss"] = float(entry["value"])

    try:
        with open(round_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["round", "global_validation_accuracy", "global_validation_loss"],
            )
            writer.writeheader()
            for round_idx in sorted(round_rows.keys()):
                writer.writerow(round_rows[round_idx])
    except OSError:
        return None

    try:
        with open(summary_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["metric", "value", "round"],
            )
            writer.writeheader()
            best_acc = summary.get("best_global_accuracy")
            final_acc = summary.get("final_global_accuracy")
            if best_acc is not None:
                writer.writerow(
                    {
                        "metric": "best_global_accuracy",
                        "value": best_acc["value"],
                        "round": best_acc["round"],
                    }
                )
            if final_acc is not None:
                writer.writerow(
                    {
                        "metric": "final_global_accuracy",
                        "value": final_acc["value"],
                        "round": final_acc["round"],
                    }
                )
    except OSError:
        return None

    return round_path, summary_path


def write_client_metrics_csv(
    run_result: str | Path,
    metrics_dirname: str = "metrics/clients",
    per_site_filename: str = "client_epoch_metrics.csv",
    combined_filename: str = "all_sites_epoch_metrics.csv",
) -> Tuple[Path, List[Path]] | None:
    run_root = Path(run_result)
    tb_root = _resolve_tb_root(run_root)
    if tb_root is None:
        return None

    site_dirs = sorted(p for p in tb_root.iterdir() if p.is_dir() and p.name.startswith("site-"))
    if not site_dirs:
        return None

    metrics_dir = tb_root.parent / metrics_dirname
    try:
        metrics_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None

    tags = [
        "train_loss",
        "train_acc",
        "val_loss_local_model",
        "val_acc_local_model",
    ]

    per_site_paths: List[Path] = []
    combined_path = metrics_dir / combined_filename
    try:
        with open(combined_path, "w", encoding="utf-8", newline="") as f_combined:
            combined_writer = csv.DictWriter(
                f_combined,
                fieldnames=["site", "epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
            )
            combined_writer.writeheader()

            for site_dir in site_dirs:
                site_name = site_dir.name
                rows = _collect_site_epoch_rows(site_dir, tags)
                if not rows:
                    continue

                site_metrics_dir = metrics_dir / site_name
                site_metrics_dir.mkdir(parents=True, exist_ok=True)
                site_path = site_metrics_dir / per_site_filename
                per_site_paths.append(site_path)

                with open(site_path, "w", encoding="utf-8", newline="") as f_site:
                    site_writer = csv.DictWriter(
                        f_site,
                        fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
                    )
                    site_writer.writeheader()
                    for epoch in sorted(rows.keys()):
                        row = rows[epoch]
                        out = {
                            "epoch": epoch,
                            "train_loss": row.get("train_loss"),
                            "train_acc": row.get("train_acc"),
                            "val_loss": row.get("val_loss_local_model"),
                            "val_acc": row.get("val_acc_local_model"),
                        }
                        site_writer.writerow(out)
                        combined_writer.writerow({"site": site_name, **out})
    except OSError:
        return None

    return combined_path, per_site_paths
