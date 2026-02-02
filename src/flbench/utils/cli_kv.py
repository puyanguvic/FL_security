from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

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


def load_config_files(paths: Iterable[str] | None) -> Dict[str, Any]:
    """Load and merge multiple YAML/JSON config files (later files override earlier)."""

    merged: Dict[str, Any] = {}
    if not paths:
        return merged
    for path in paths:
        merged.update(load_config_file(path))
    return merged


def _kv_list_from_dict(values: Mapping[str, Any]) -> List[str]:
    out: List[str] = []
    for k, v in values.items():
        if isinstance(v, (dict, list, tuple)):
            v_str = json.dumps(v)
        elif isinstance(v, bool):
            v_str = "true" if v else "false"
        elif v is None:
            v_str = "null"
        else:
            v_str = str(v)
        out.append(f"{k}={v_str}")
    return out


def _coerce_kv_list(value: Any, label: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return _kv_list_from_dict(value)
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError(f"{label} must be a list of key=value strings or a dict")
            out.append(item)
        return out
    raise ValueError(f"{label} must be a list of key=value strings or a dict")


def normalize_unified_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a unified run config into flat CLI-compatible keys."""

    out: Dict[str, Any] = dict(cfg)

    # Backward-compatible aliases for renamed FL parameters.
    aliases = {
        "n_clients": "num_clients",
        "num_rounds": "global_rounds",
        "aggregation_epochs": "local_epochs",
    }
    for old_key, new_key in aliases.items():
        if old_key in out and new_key not in out:
            out[new_key] = out[old_key]

    def _normalize_section(name: str) -> None:
        section = out.get(name)
        params: Dict[str, Any] | None = None
        if isinstance(section, Mapping):
            name_val = None
            for key in ("name", "type", "id"):
                if key in section:
                    name_val = section.get(key)
                    break
            if isinstance(name_val, str) and name_val != "":
                out[name] = name_val
            else:
                out.pop(name, None)

            rest = {k: v for k, v in section.items() if k not in {"name", "type", "id", "params", "config"}}
            if "params" in section:
                params = section.get("params")
                if rest:
                    if not isinstance(params, Mapping):
                        raise ValueError(f"{name}.params must be a dict/mapping")
                    params = {**params, **rest}
            elif "config" in section:
                cfg_value = section.get("config")
                if isinstance(cfg_value, str):
                    out[f"{name}_config"] = cfg_value
                    params = rest or None
                else:
                    params = cfg_value
                    if rest:
                        if not isinstance(params, Mapping):
                            raise ValueError(f"{name}.config must be a dict/mapping when merging params")
                        params = {**params, **rest}
            else:
                params = rest or None
        elif section is not None and not isinstance(section, str):
            raise ValueError(f"{name} must be a string or mapping")

        extra_params = out.pop(f"{name}_params", None)
        if extra_params is not None:
            if params is None:
                params = extra_params
            elif isinstance(params, Mapping) and isinstance(extra_params, Mapping):
                params = {**params, **extra_params}
            else:
                raise ValueError(f"{name}_params must be a dict/mapping")

        if params is not None:
            if not isinstance(params, Mapping):
                raise ValueError(f"{name} params must be a dict/mapping")
            kv_key = f"{name}_kv"
            kv_list = _kv_list_from_dict(params)
            if kv_key in out:
                kv_list.extend(_coerce_kv_list(out[kv_key], kv_key))
            out[kv_key] = kv_list

    _normalize_section("attack")
    _normalize_section("defense")

    for key in ("attack_kv", "defense_kv"):
        if key in out:
            out[key] = _coerce_kv_list(out[key], key)

    return out
