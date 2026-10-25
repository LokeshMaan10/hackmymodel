"""Module 1 smoke tests: configs load and are internally consistent."""
from hmom.utils.config import load_config, load_constraints
from hmom.utils.seed import set_global_seed


def test_config_loads() -> None:
    cfg = load_config()
    assert cfg["data"]["target"] == "isFraud"
    assert cfg["split"]["train_end_day"] < cfg["split"]["val_end_day"]


def test_constraint_tiers_valid() -> None:
    feats = load_constraints()["features"]
    allowed = {"easy", "costly", "impossible", "derived"}
    for name, spec in feats.items():
        if isinstance(spec, dict):
            assert spec["tier"] in allowed, name


def test_seed_reproducible() -> None:
    a = set_global_seed(1).random(3)
    b = set_global_seed(1).random(3)
    assert (a == b).all()
