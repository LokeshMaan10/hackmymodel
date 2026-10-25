"""Reproducibility helpers."""
from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> np.random.Generator:
    """Seed Python, NumPy and hashing; return a NumPy Generator to pass around.

    Model libraries (LightGBM, XGBoost) take their own `random_state` /
    `seed` argument — always pass the same seed there too.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
