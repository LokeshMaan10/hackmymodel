"""Module 2b: exploratory data analysis.

Run:   python scripts/02_eda.py
Output: reports/figures/eda_*.png  and  results/eda_summary.json
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")  # no GUI needed; just save PNGs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hmom.data import eda
from hmom.data.load import load_interim
from hmom.utils.config import load_config, load_constraints

FRAUD_C, LEGIT_C = "#d1495b", "#2e86ab"


def _save(fig: plt.Figure, name: str) -> None:
    out = load_config().path("figures_dir", create=True) / f"eda_{name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  saved {out.name}")


def plot_time(df: pd.DataFrame, train_end: int, val_end: int) -> None:
    daily = df.groupby("day")["isFraud"].agg(["size", "mean"])
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(daily.index, daily["size"], color=LEGIT_C)
    ax[0].set_ylabel("transactions / day")
    ax[1].plot(daily.index, daily["mean"] * 100, color=FRAUD_C)
    ax[1].set_ylabel("fraud rate %")
    ax[1].set_xlabel("day")
    for a in ax:
        for d, lbl in ((train_end, "val starts"), (val_end, "test starts")):
            a.axvline(d, ls="--", color="grey")
            a.text(d + 1, a.get_ylim()[1] * 0.92, lbl, fontsize=8, color="grey")
    ax[0].set_title("Volume and fraud rate over time (dashed = planned split)")
    _save(fig, "time_daily")

    hourly = df.groupby("hour")["isFraud"].agg(["size", "mean"])
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(hourly.index, hourly["size"], color=LEGIT_C, alpha=0.5, label="volume")
    ax1.set_xlabel("hour (relative)")
    ax1.set_ylabel("transactions")
    ax2 = ax1.twinx()
    ax2.plot(hourly.index, hourly["mean"] * 100, color=FRAUD_C, marker="o", label="fraud rate")
    ax2.set_ylabel("fraud rate %")
    ax1.set_title("Hour of day: fraud rate peaks when legit volume is low")
    _save(fig, "time_hourly")


def plot_amount(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    bins = np.logspace(0, np.log10(df["TransactionAmt"].max() + 1), 60)
    for lbl, color, name in ((0, LEGIT_C, "legit"), (1, FRAUD_C, "fraud")):
        ax.hist(df.loc[df["isFraud"] == lbl, "TransactionAmt"], bins=bins, density=True,
                alpha=0.55, color=color, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("TransactionAmt (USD, log scale)")
    ax.set_ylabel("density")
    ax.legend()
    ax.set_title("Amount distribution by class")
    _save(fig, "amount")


def plot_rates(table: pd.DataFrame, title: str, name: str, overall: float) -> None:
    fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(table))))
    t = table.iloc[::-1]
    ax.barh([str(i) for i in t.index], t["fraud_rate"] * 100, color=FRAUD_C)
    ax.axvline(overall * 100, ls="--", color="grey", label=f"overall {overall:.1%}")
    for y, (rate, n) in enumerate(zip(t["fraud_rate"], t["count"])):
        ax.text(rate * 100, y, f"  n={n:,}", va="center", fontsize=8)
    ax.set_xlabel("fraud rate %")
    ax.set_title(title)
    ax.legend(loc="lower right")
    _save(fig, name)


def plot_top_signals(auc: pd.DataFrame, top: int = 25) -> None:
    t = auc.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.3 * top + 1))
    ax.barh(t["feature"], t["strength"], color=LEGIT_C)
    ax.set_xlabel("single-feature strength  |AUC - 0.5| x 2")
    ax.set_title(f"Top {top} individual fraud signals")
    _save(fig, "top_signals")


def _records(t: pd.DataFrame) -> dict:
    return json.loads(t.reset_index().astype({t.index.name or "index": str}).to_json(orient="records"))


def main() -> None:
    cfg = load_config()
    split = cfg["split"]
    df = load_interim(cfg)
    print(f"Loaded {df.shape[0]:,} x {df.shape[1]}\n")

    bal = eda.class_balance(df)
    print(f"[1] Class balance: {bal['frauds']:,} frauds / {bal['rows']:,} = {bal['fraud_rate']:.2%} "
          f"(1 fraud per {bal['legit_per_fraud']:.0f} legit)")
    print("    -> accuracy is useless here: predicting 'never fraud' scores "
          f"{1 - bal['fraud_rate']:.1%}. We use PR-AUC, recall, precision and Rs. cost.\n")

    drift = eda.split_drift(df, split["train_end_day"], split["val_end_day"])
    print("[2] Planned time split:\n", drift.round(4).to_string(), "\n")

    print("[3] Amount")
    amt = eda.amount_buckets(df)
    print(amt.round(4).to_string())
    cents = eda.cents_pattern(df)
    print(cents.round(4).to_string(), "\n")

    print("[4] Categorical signals")
    cat_tables = {
        "ProductCD": eda.rate_by(df, "ProductCD"),
        "card4": eda.rate_by(df, "card4"),
        "card6": eda.rate_by(df, "card6"),
        "P_emaildomain": eda.rate_by(df, "P_emaildomain", top=15),
        "DeviceType": eda.rate_by(df, "DeviceType"),
        "has_identity": eda.rate_by(df, "has_identity"),
    }
    for col, t in cat_tables.items():
        print(f"  {col}:\n{t.round(4).to_string()}\n")

    print("[5] Missingness (top 10 most-missing columns)")
    miss = eda.missingness(df)
    print(miss.head(10).round(3).to_string())
    groups = {p: miss.loc[miss.index.str.match(rf"^{p}\d+$"), "missing_all"].mean() for p in "CDMV"}
    print("  mean missing share by group:", {k: round(v, 3) for k, v in groups.items()}, "\n")

    print("[6] Top single-feature fraud signals (screening only)")
    auc = eda.univariate_auc(df, exclude={"TransactionID", "TransactionDT", "day"})
    print(auc.head(15).round(3).to_string(), "\n")

    print("[7] Attack surface: presence of 'easy' features")
    easy = [k for k, v in load_constraints()["features"].items()
            if isinstance(v, dict) and v.get("tier") == "easy"]
    surface = eda.attack_surface(df, easy)
    print(surface.round(3).to_string(), "\n")

    print("Saving figures...")
    plot_time(df, split["train_end_day"], split["val_end_day"])
    plot_amount(df)
    plot_rates(cat_tables["P_emaildomain"], "Fraud rate by purchaser email domain (top 15 by volume)",
               "email", bal["fraud_rate"])
    plot_rates(eda.rate_by(df, "ProductCD"), "Fraud rate by ProductCD", "product", bal["fraud_rate"])
    plot_top_signals(auc)

    summary = {
        "class_balance": bal,
        "split": _records(drift),
        "amount_buckets": _records(amt.set_axis(amt.index.astype(str))),
        "cents": _records(cents),
        "categorical": {k: _records(v) for k, v in cat_tables.items()},
        "missing_by_group": groups,
        "top_signals": auc.head(30).to_dict(orient="records"),
        "attack_surface": _records(surface),
    }
    out = cfg.path("results_dir", create=True) / "eda_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
