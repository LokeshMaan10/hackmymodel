"""Exploratory analysis: small, reusable functions that return tables.

Plotting lives in scripts/02_eda.py so these stay easy to test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TARGET = "isFraud"


def class_balance(df: pd.DataFrame) -> dict[str, float]:
    """Counts and fraud rate."""
    n = len(df)
    n_fraud = int(df[TARGET].sum())
    return {"rows": n, "frauds": n_fraud, "legit": n - n_fraud, "fraud_rate": n_fraud / n,
            "legit_per_fraud": (n - n_fraud) / max(n_fraud, 1)}


def rate_by(df: pd.DataFrame, col: str, top: int | None = None, min_count: int = 1) -> pd.DataFrame:
    """Transaction count, fraud count and fraud rate per value of `col` (NaN kept as 'missing')."""
    key = df[col].astype(object).where(df[col].notna(), "missing")
    g = df.groupby(key, observed=True)[TARGET].agg(count="size", frauds="sum")
    g["fraud_rate"] = g["frauds"] / g["count"]
    g = g[g["count"] >= min_count].sort_values("count", ascending=False)
    return g.head(top) if top else g


def amount_buckets(df: pd.DataFrame, edges: list[float] | None = None) -> pd.DataFrame:
    """Fraud rate by transaction-amount bucket (USD)."""
    edges = edges or [0, 25, 50, 100, 200, 500, 1000, 5000, np.inf]
    b = pd.cut(df["TransactionAmt"], bins=edges, right=False)
    g = df.groupby(b, observed=True)[TARGET].agg(count="size", frauds="sum")
    g["fraud_rate"] = g["frauds"] / g["count"]
    return g


def cents_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """Whole-dollar vs fractional amounts - fractional amounts often hint at currency conversion."""
    frac = (df["TransactionAmt"] * 100).round() % 100
    kind = np.where(frac == 0, "whole_dollar", "has_cents")
    g = df.groupby(kind)[TARGET].agg(count="size", frauds="sum")
    g["fraud_rate"] = g["frauds"] / g["count"]
    return g


def missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Share of missing values per column, overall and by class."""
    is_f = df[TARGET] == 1
    out = pd.DataFrame({
        "missing_all": df.isna().mean(),
        "missing_fraud": df[is_f].isna().mean(),
        "missing_legit": df[~is_f].isna().mean(),
    })
    return out.drop(index=TARGET).sort_values("missing_all", ascending=False)


def univariate_auc(df: pd.DataFrame, exclude: set[str], sample: int = 200_000, seed: int = 42) -> pd.DataFrame:
    """Rank numeric features by how well each one *alone* separates fraud.

    `strength` = |AUC - 0.5| * 2 (0 = useless, 1 = perfect). NaN is filled
    with a sentinel, like a tree would treat it. A quick screening tool only -
    it ignores interactions, which is where tree models get most of their power.
    """
    sub = df.sample(n=min(sample, len(df)), random_state=seed)
    y = sub[TARGET].to_numpy()
    rows = []
    for col in sub.select_dtypes(include="number").columns:
        if col in exclude or col == TARGET:
            continue
        x = sub[col].astype(np.float64).fillna(-999.0).to_numpy()
        if np.unique(x).size < 2:
            continue
        auc = roc_auc_score(y, x)
        rows.append({"feature": col, "auc": auc, "strength": abs(auc - 0.5) * 2,
                     "direction": "higher=fraudier" if auc >= 0.5 else "lower=fraudier"})
    return pd.DataFrame(rows).sort_values("strength", ascending=False).reset_index(drop=True)


def split_drift(df: pd.DataFrame, train_end_day: int, val_end_day: int) -> pd.DataFrame:
    """Rows, fraud rate and identity coverage for each planned time split."""
    part = np.select([df["day"] < train_end_day, df["day"] < val_end_day], ["train", "val"], "test")
    g = df.groupby(part).agg(rows=(TARGET, "size"), fraud_rate=(TARGET, "mean"),
                             identity_coverage=("has_identity", "mean"),
                             first_day=("day", "min"), last_day=("day", "max"))
    return g.loc[["train", "val", "test"]]


def attack_surface(df: pd.DataFrame, easy_cols: list[str]) -> pd.DataFrame:
    """How often each 'easy-to-change' feature is present, for frauds vs legit.

    If an easy feature is missing on most frauds, the attacker has little to play with there.
    """
    is_f = df[TARGET] == 1
    cols = [c for c in easy_cols if c in df.columns]
    return pd.DataFrame({
        "present_fraud": df.loc[is_f, cols].notna().mean(),
        "present_legit": df.loc[~is_f, cols].notna().mean(),
        "n_unique": df[cols].nunique(),
    })
