from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from uuid import uuid4


def build_run_id(prefix: str | None = None) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid4().hex[:8]
    if prefix:
        return f"{prefix}__{ts}__{suffix}"
    return f"{ts}__{suffix}"


def write_json(path: str | Path, payload: dict) -> Path | None:
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
    except OSError:
        return None
    return path


def write_git_info(path: str | Path) -> Path | None:
    path = Path(path)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = "unknown"
    payload = {
        "commit": commit,
        "dirty": _is_git_dirty(),
    }
    return write_json(path, payload)


def _is_git_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
        )
    except Exception:
        return False
    return bool(result.stdout.strip())
