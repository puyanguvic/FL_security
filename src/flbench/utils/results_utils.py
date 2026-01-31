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
