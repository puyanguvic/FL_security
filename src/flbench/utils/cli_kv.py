from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


def _coerce_scalar(v: str) -> Any:
    """Best-effort type coercion for CLI key=value pairs."""

    s = str(v).strip()
    if s.lower() in {"true", "false"}:
        return s.lower() == "true"
    if s.lower() in {"none", "null"}:
        return None
    # int
    try:
        if s.startswith("0") and s not in {"0", "0.0"}:
            # avoid surprising octal-like casts; keep as string
            raise ValueError
        return int(s)
    except Exception:
        pass
    # float
    try:
        return float(s)
    except Exception:
        pass
    # json inline (arrays/objects)
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            return json.loads(s)
        except Exception:
            return s
    return s


def parse_kv_list(items: Iterable[str] | None) -> Dict[str, Any]:
    """Parse repeated --*_kv key=value into a dict."""

    out: Dict[str, Any] = {}
    if not items:
        return out
    for raw in items:
        if raw is None:
            continue
        s = str(raw).strip()
        if not s:
            continue
        if "=" not in s:
            raise ValueError(f"Invalid kv '{raw}': expected key=value")
        k, v = s.split("=", 1)
        k = k.strip()
        if not k:
            raise ValueError(f"Invalid kv '{raw}': empty key")
        out[k] = _coerce_scalar(v)
    return out


def load_config_file(path: str | None) -> Dict[str, Any]:
    """Load YAML/JSON config file (returns empty dict if path is falsy)."""

    if not path:
        return {}
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(str(p))
    text = p.read_text(encoding="utf-8")
    suffix = p.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        # try yaml as a default
        data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a mapping/dict: {p}")
    return dict(data)


def merge_kv_and_file(*, kv: Iterable[str] | None, config_path: str | None) -> Dict[str, Any]:
    """Merge config file dict with kv overrides (kv wins)."""

    cfg = load_config_file(config_path)
    cfg.update(parse_kv_list(kv))
    return cfg
