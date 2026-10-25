"""Module 1 sanity check: libraries import, configs load, dataset present.

Run:  python scripts/00_check_setup.py
"""
from __future__ import annotations

import importlib
import sys

from hmom.utils.config import load_config, load_constraints
from hmom.utils.seed import set_global_seed

LIBS = [
    "pandas", "numpy", "sklearn", "lightgbm", "xgboost", "art", "deap",
    "shap", "mlflow", "streamlit", "pyarrow", "yaml", "matplotlib", "pytest",
]


def check_libraries() -> bool:
    ok = True
    for name in LIBS:
        try:
            mod = importlib.import_module(name)
            print(f"  [ok]   {name:<12} {getattr(mod, '__version__', '')}")
        except Exception as exc:  # noqa: BLE001 - report any import failure
            print(f"  [FAIL] {name:<12} {exc}")
            ok = False
    return ok


def check_data() -> bool:
    cfg = load_config()
    raw = cfg.path("raw_dir")
    ok = True
    for key in ("transaction_file", "identity_file"):
        f = raw / cfg["data"][key]
        if f.exists():
            print(f"  [ok]   {f.name:<24} {f.stat().st_size / 1e6:,.0f} MB")
        else:
            print(f"  [MISS] {f.name:<24} -> put it in {raw}")
            ok = False
    return ok


def main() -> None:
    print(f"Python {sys.version.split()[0]}")
    print("\nLibraries:")
    libs_ok = check_libraries()

    print("\nConfig:")
    cfg = load_config()
    n_feat = len(load_constraints()["features"])
    print(f"  project root : {cfg.root}")
    print(f"  seeds        : {cfg['project']['seeds']}")
    print(f"  constraints  : {n_feat} feature entries loaded")
    set_global_seed(cfg["project"]["default_seed"])

    print("\nDataset:")
    data_ok = check_data()

    print("\nSummary:", "libraries OK" if libs_ok else "fix library errors above",
          "|", "data OK" if data_ok else "dataset not downloaded yet")


if __name__ == "__main__":
    main()
