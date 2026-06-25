from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively update a nested dictionary.
    """
    result = dict(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value

    return result


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")

    return data


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load default.yaml first, then override with the selected config.
    """
    config_path = Path(config_path)
    default_path = config_path.parent / "default.yaml"

    if default_path.exists() and config_path.name != "default.yaml":
        default_cfg = load_yaml(default_path)
        user_cfg = load_yaml(config_path)
        return deep_update(default_cfg, user_cfg)

    return load_yaml(config_path)