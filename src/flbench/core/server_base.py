from __future__ import annotations

import datetime as _dt
import json
import shutil
from pathlib import Path


class BaseServer:
    def __init__(self, args):
        self.args = args

    def run(self) -> None:
        raise NotImplementedError

    def _copy_run_result_to_results_dir(
        self, *, run_result: str | None, results_dir: str, job_name: str
    ) -> Path | None:
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

    def _load_run_meta(self, meta_path: str) -> dict | None:
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError):
            return None

    def _write_run_meta(self, meta_path: str, meta: dict) -> None:
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, sort_keys=True)
        except OSError:
            pass
