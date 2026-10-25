"""Module 2a: raw CSVs -> cleaned, memory-optimized Parquet.

Run:  python scripts/02_prepare_data.py
Output: data/interim/train_merged.parquet
"""
from __future__ import annotations

import time

from hmom.data.clean import clean
from hmom.data.load import interim_path, load_raw
from hmom.data.memory import memory_mb
from hmom.utils.config import load_config


def main() -> None:
    cfg = load_config()
    t0 = time.perf_counter()

    print("Reading raw CSVs (about 1 minute)...")
    tx, idt = load_raw(cfg)
    print(f"  transactions: {tx.shape}  {memory_mb(tx):,.0f} MB")
    print(f"  identity    : {idt.shape}  {memory_mb(idt):,.0f} MB")

    df, report = clean(tx, idt)
    del tx, idt
    print(f"\nCleaned: {report['rows']:,} rows x {report['cols']} cols, {memory_mb(df):,.0f} MB in RAM")
    print(f"  duplicate TransactionIDs : {report['duplicate_ids']}")
    print(f"  dropped constant columns : {report['dropped_constant'] or 'none'}")

    out = interim_path(cfg)
    df.to_parquet(out, index=False)
    print(f"\nSaved {out}  ({out.stat().st_size / 1e6:,.0f} MB on disk)  in {time.perf_counter() - t0:,.0f}s")


if __name__ == "__main__":
    main()
