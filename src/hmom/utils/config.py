"""Load YAML configs and resolve project paths.

Usage:
    from hmom.utils.config import load_config, project_root
    cfg = load_config()
    raw_dir = cfg.path("raw_dir")
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Return the repo root (the folder that contains `configs/`)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs" / "config.yaml").exists():
            return parent
    raise FileNotFoundError("Could not find configs/config.yaml above " + str(here))


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass(frozen=True)
class Config:
    """Thin wrapper around the config dict with path helpers."""

    raw: dict[str, Any]
    root: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def path(self, name: str, create: bool = False) -> Path:
        """Absolute path for an entry under `paths:`; optionally create it."""
        p = self.root / self.raw["paths"][name]
        if create:
            p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache(maxsize=4)
def load_config(name: str = "config.yaml") -> Config:
    """Load a YAML file from `configs/` (cached)."""
    root = project_root()
    return Config(raw=_read_yaml(root / "configs" / name), root=root)


def load_constraints() -> dict[str, Any]:
    """Load the threat-model constraint file as a plain dict."""
    return load_config("constraints.yaml").raw
