"""Load the raw IEEE-CIS CSVs and the cleaned Parquet file."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hmom.data.memory import KEEP_FLOAT64, optimize
from hmom.utils.config import Config, load_config

INTERIM_FILE = "train_merged.parquet"

# Text columns in the raw files; everything else is numeric.
STRING_COLS: frozenset[str] = frozenset(
    ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo"]
    + [f"M{i}" for i in range(1, 10)]
    + [f"id_{i}" for i in (12, 15, 16, 23, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38)]
)


def _dtype_map(csv_path: Path) -> dict[str, object]:
    """Read only the header and choose a compact dtype per column.

    Reading numerics as float32 up front keeps peak RAM low (no float64 copy
    first). If a text column is missing from STRING_COLS, read_csv fails loudly.
    """
    cols = pd.read_csv(csv_path, nrows=0).columns
    dtypes: dict[str, object] = {}
    for c in cols:
        if c in ("TransactionID", "TransactionDT"):
            dtypes[c] = np.int32
        elif c == "isFraud":
            dtypes[c] = np.int8
        elif c in STRING_COLS:
            dtypes[c] = "category"
        elif c in KEEP_FLOAT64:
            dtypes[c] = np.float64
        else:
            dtypes[c] = np.float32
    return dtypes


def read_raw_csv(csv_path: Path, sample_frac: float = 1.0, seed: int = 42) -> pd.DataFrame:
    """Read one raw CSV with compact dtypes; optionally subsample rows."""
    df = pd.read_csv(csv_path, dtype=_dtype_map(csv_path), low_memory=False)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=seed)
    return optimize(df)


def load_raw(cfg: Config | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (transactions, identity) DataFrames from data/raw."""
    cfg = cfg or load_config()
    raw = cfg.path("raw_dir")
    d = cfg["data"]
    tx = read_raw_csv(raw / d["transaction_file"], d["sample_frac"], cfg["project"]["default_seed"])
    idt = read_raw_csv(raw / d["identity_file"])
    return tx, idt


def interim_path(cfg: Config | None = None) -> Path:
    cfg = cfg or load_config()
    return cfg.path("interim_dir", create=True) / INTERIM_FILE


def load_interim(cfg: Config | None = None, columns: list[str] | None = None) -> pd.DataFrame:
    """Load the cleaned, merged dataset written by scripts/02_prepare_data.py."""
    path = interim_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run: python scripts/02_prepare_data.py")
    return pd.read_parquet(path, columns=columns)
