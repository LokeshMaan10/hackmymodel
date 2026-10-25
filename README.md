# Hack My Own Model — Red-Teaming a Fraud Detection System

> Fraudsters adapt. Static accuracy doesn't measure that. This project builds a
> strong fraud model, attacks it with a constrained AI adversary, then hardens it
> and measures how much more it costs a fraudster to get through — without
> hurting honest customers.

**Status:** 🚧 in progress (Module 1 of 15)

## Why this matters
_TODO (Module 14): business problem, who cares, headline result._

## Results at a glance
_TODO (Module 12): defense comparison table + robustness curve._

| Model | Clean PR-AUC | Recall @ thr | FPR | Attack success | ₹ through |
|---|---|---|---|---|---|
| Baseline | – | – | – | – | – |

## Threat model
Features are split into **easy / costly / impossible / derived** to change
(see [`configs/constraints.yaml`](configs/constraints.yaml)). The attacker
maximises `money stolen − cost of changes` subject to being approved, staying
valid, and stealing a meaningful amount.

Attacker knowledge levels: white-box · gray-box (scores) · black-box (approve/block) · surrogate/transfer.

## Methods
- **Blue team:** LightGBM / XGBoost, cost-based threshold, SHAP
- **Red team:** greedy search · genetic algorithm (DEAP) · ART HopSkipJump / ZOO · surrogate model stealing
- **Defenses:** adversarial training (arms race) · behavioral features · monotonic constraints · ensemble · probe detection

## Data
[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) (Vesta).
Raw data is **not** included in this repo (competition rules). Download it
yourself into `data/raw/` — see Quickstart.

## Quickstart (Windows / PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python scripts/00_check_setup.py
pytest
```

## Project structure
```
configs/      config.yaml (paths, seeds, ₹ costs), constraints.yaml (threat model)
src/hmom/     data · features · models · threat_model · attacks · defenses · evaluation · utils
scripts/      one runnable script per module
app/          Streamlit dashboard (reads precomputed results)
tests/        pytest
```

## Assumptions
- USD→INR fixed at 84 (config). False-alarm cost ₹250 (config).
- Attacker change-costs are abstract units; a sensitivity analysis is included.
- V1–V339 are treated as immutable (their dependencies are unknown).

## Limitations
_TODO (Module 14)._

## Ethics
This is **defensive** research on public data. No real system is probed. The
goal is to show fraud teams where models are weak so they can be strengthened.
Results are reported as aggregate vulnerability classes, not evasion recipes.
