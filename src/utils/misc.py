# utils/misc.py
"""
Utility helpers: seeding, YAML I/O, checkpoint helpers.
"""

from __future__ import annotations
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


# --------------------------------------------------------------------- #
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------- #
def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(obj: Dict[str, Any], path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(obj, f)


# --------------------------------------------------------------------- #
def save_ckpt(state: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_ckpt(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)
