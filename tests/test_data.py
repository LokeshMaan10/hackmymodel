"""Module 2 tests: memory optimization, cleaning and EDA helpers on tiny fake data."""
import numpy as np
import pandas as pd

from hmom.data import eda
from hmom.data.clean import clean
from hmom.data.memory import memory_mb, optimize


def _fake() -> tuple[pd.DataFrame, pd.DataFrame]:
    tx = pd.DataFrame({
        "TransactionID": [3, 1, 2, 4],
        "isFraud": [0, 1, 0, 1],
        "TransactionDT": [90_000, 86_400, 86_400 + 3_600 * 5, 200_000],
        "TransactionAmt": [10.0, 59.99, 100.0, 7.5],
        "P_emaildomain": [" Gmail.com", "gmail.com", None, "yahoo.com"],
        "const": [1.0, 1.0, 1.0, 1.0],
        "V1": [np.inf, 1.0, 2.0, 3.0],
    })
    idt = pd.DataFrame({"TransactionID": [1, 4], "DeviceType": ["mobile", "desktop"]})
    return optimize(tx), optimize(idt)


def test_optimize_shrinks_and_preserves_values() -> None:
    df = pd.DataFrame({"a": np.arange(1000, dtype=np.int64), "b": np.random.rand(1000), "c": ["x", "y"] * 500})
    before = memory_mb(df)
    small = optimize(df.copy())
    assert memory_mb(small) < before
    assert (small["a"].to_numpy() == df["a"].to_numpy()).all()
    assert np.allclose(small["b"], df["b"], rtol=1e-6)
    assert small["c"].dtype.name == "category"


def test_clean_merges_normalizes_and_sorts() -> None:
    tx, idt = _fake()
    df, report = clean(tx, idt)
    assert report["rows"] == 4 and report["duplicate_ids"] == 0
    assert "const" in report["dropped_constant"]
    assert df["TransactionDT"].is_monotonic_increasing
    assert df["has_identity"].sum() == 2
    assert set(df["P_emaildomain"].dropna()) == {"gmail.com", "yahoo.com"}
    assert not np.isinf(df["V1"].astype(float)).any()
    assert df.loc[df["TransactionID"] == 2, "hour"].item() == 5


def test_eda_helpers() -> None:
    tx, idt = _fake()
    df, _ = clean(tx, idt)
    assert eda.class_balance(df)["fraud_rate"] == 0.5
    rates = eda.rate_by(df, "P_emaildomain")
    assert rates.loc["gmail.com", "count"] == 2 and "missing" in rates.index
    auc = eda.univariate_auc(df, exclude={"TransactionID"})
    assert auc["strength"].between(0, 1).all()
