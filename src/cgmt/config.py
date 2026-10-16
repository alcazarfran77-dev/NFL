"""Configuration loading and global reproducibility controls."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    def path(self, key: str) -> Path:
        """Absolute path for an entry under `paths`, created if it is a directory."""
        p = PROJECT_ROOT / self.raw["paths"][key]
        (p if p.suffix == "" else p.parent).mkdir(parents=True, exist_ok=True)
        return p


def load_config(path: str | os.PathLike | None = None) -> Config:
    path = Path(path) if path else PROJECT_ROOT / "configs" / "config.yaml"
    with open(path, encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh))


def set_global_seed(seed: int) -> None:
    """Seed every RNG the pipeline touches. Model objects also receive `random_state`."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
