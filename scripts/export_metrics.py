from __future__ import annotations

import argparse
from pathlib import Path

from flbench.utils.results_utils import (
    write_client_metrics_csv,
    write_global_metrics_csv,
    write_global_metrics_summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export CSV/JSON metrics from an NVFLARE run directory.")
    parser.add_argument(
        "run_result",
        type=str,
        help="Path to NVFLARE run result directory (e.g., .../simulate_job)",
    )
    parser.add_argument(
        "--skip-clients",
        action="store_true",
        help="Skip per-client CSV exports.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_result = Path(args.run_result).expanduser()

    if not run_result.exists():
        raise FileNotFoundError(f"Run result path does not exist: {run_result}")

    global_paths = write_global_metrics_csv(run_result)
    if global_paths is not None:
        print("Global metrics CSV saved to:", str(global_paths[0]))
        print("Global metrics summary CSV saved to:", str(global_paths[1]))
    else:
        print("Global metrics CSV not written (no events found).")

    summary_path = write_global_metrics_summary(run_result)
    if summary_path is not None:
        print("Global metrics JSON saved to:", str(summary_path))
    else:
        print("Global metrics JSON not written (no events found).")

    if not args.skip_clients:
        client_metrics = write_client_metrics_csv(run_result)
        if client_metrics is not None:
            print("Client metrics CSV saved to:", str(client_metrics[0]))
        else:
            print("Client metrics CSV not written (no events found).")


if __name__ == "__main__":
    main()
