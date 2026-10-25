"""Memory helpers: shrink DataFrames so the full dataset fits comfortably in RAM.

Why: pandas defaults to float64 / object. IEEE-CIS has ~430 columns x 590k rows,
so float64 everywhere is ~2 GB before any feature engineering. float32 and
`category` dtypes cut that by more than half with no practical loss.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Columns where float32 rounding would hurt (cents matter for amount patterns).
KEEP_FLOAT64: frozenset[str] = frozenset({"TransactionAmt"})


def memory_mb(df: pd.DataFrame) -> float:
    """Deep memory usage of a DataFrame in MB."""
    return float(df.memory_usage(deep=True).sum() / 1e6)


def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast ints to the smallest safe int type and floats to float32 (in place)."""
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_integer_dtype(s):
            df[col] = pd.to_numeric(s, downcast="integer")
        elif pd.api.types.is_float_dtype(s) and col not in KEEP_FLOAT64:
            df[col] = s.astype(np.float32)
    return df


def objects_to_category(df: pd.DataFrame) -> pd.DataFrame:
    """Convert string (object) columns to `category` (in place)."""
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category")
    return df


def optimize(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all memory optimizations and return the same DataFrame."""
    return objects_to_category(downcast_numeric(df))
