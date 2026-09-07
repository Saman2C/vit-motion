from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg


def resolve_from_config(value: str, config_path: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config_path).parent / path
    return path.resolve()

