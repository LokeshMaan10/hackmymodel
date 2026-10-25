"""Merge and clean the raw tables.

Cleaning here is deliberately light and label-free (nothing uses `isFraud`),
so it cannot leak information into the time-based split made in Module 3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SECONDS_PER_DAY = 86_400


def merge_tables(tx: pd.DataFrame, idt: pd.DataFrame) -> pd.DataFrame:
    """Left-join identity onto transactions; add a `has_identity` flag.

    Only ~24% of transactions have identity rows. Missing identity is itself
    information (and limits what the attacker's "device" features can touch).
    """
    df = tx.merge(idt, on="TransactionID", how="left", validate="one_to_one")
    df["has_identity"] = df["TransactionID"].isin(idt["TransactionID"]).astype(np.int8)
    return df


def normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case and strip categorical text so 'Gmail.com ' == 'gmail.com'."""
    for col in df.select_dtypes(include="category").columns:
        cats = df[col].cat.categories
        cleaned = pd.Index(cats.astype(str).str.strip().str.lower())
        if not cleaned.equals(pd.Index(cats.astype(str))):
            df[col] = df[col].astype(str).str.strip().str.lower().replace("nan", np.nan).astype("category")
    return df


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add `day` and `hour` derived from TransactionDT (seconds from an unknown start).

    The true start time is unknown, so `hour` is relative - but the daily cycle
    in the data shows it is close to a real clock (see EDA).
    """
    dt = df["TransactionDT"].astype(np.int64)
    df["day"] = (dt // SECONDS_PER_DAY).astype(np.int16)
    df["hour"] = ((dt // 3600) % 24).astype(np.int8)
    return df


def drop_constant_columns(df: pd.DataFrame, protect: set[str]) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns with a single unique value (incl. all-NaN); they carry no signal."""
    const = [c for c in df.columns if c not in protect and df[c].nunique(dropna=False) <= 1]
    return df.drop(columns=const), const


def clean(tx: pd.DataFrame, idt: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Full cleaning pipeline. Returns (clean_df, report)."""
    df = merge_tables(tx, idt)
    df = normalize_strings(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = add_time_columns(df)
    df, dropped = drop_constant_columns(df, protect={"isFraud", "TransactionID"})
    n_dupes = int(df.duplicated(subset="TransactionID").sum())
    df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    report = {"rows": len(df), "cols": df.shape[1], "dropped_constant": dropped, "duplicate_ids": n_dupes}
    return df, report
