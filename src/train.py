"""Training (spec section 4).

Chronological split (mandatory, spec 4.1):
    train: 2018 -> (LATEST_YEAR - 2)
    val:   LATEST_YEAR - 1
    test:  LATEST_YEAR          (never touched here; evaluate.py only)

Run: python -m src.train
"""
import json
import logging
import shutil
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd

from .columns import FEATURE_COLS, TARGET

log = logging.getLogger("train")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

RANDOM_STATE = 42

PARAMS = dict(
    objective="binary",
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    n_estimators=2000,
    # top-10 is roughly balanced (~45% positive): no is_unbalance (spec 4.2).
    random_state=RANDOM_STATE,
)


def chronological_split(df: pd.DataFrame):
    latest = int(df["year"].max())
    train = df[df["year"] <= latest - 2]
    val = df[df["year"] == latest - 1]
    test = df[df["year"] == latest]
    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError(
            f"Chronological split produced an empty set "
            f"(latest={latest}, sizes: train={len(train)}, val={len(val)}, "
            f"test={len(test)}). Need at least 3 seasons of data."
        )
    return train, val, test, latest


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    df = pd.read_parquet(DATA_DIR / "features.parquet")
    train, val, test, latest = chronological_split(df)
    log.info("Split (latest season = %s): train=%s rows (<=%s), val=%s rows (%s), "
             "test=%s rows (%s, HELD OUT)",
             latest, len(train), latest - 2, len(val), latest - 1, len(test), latest)

    X_tr, y_tr = train[FEATURE_COLS], train[TARGET]
    X_val, y_val = val[FEATURE_COLS], val[TARGET]

    clf = lgb.LGBMClassifier(**PARAMS)
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric=["auc", "binary_logloss"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)],
    )

    log.info("Best iteration: %s", clf.best_iteration_)
    log.info("Best val scores: %s", dict(clf.best_score_.get("valid_0", {})))

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(clf, MODELS_DIR / "model.joblib")
    clf.booster_.save_model(str(MODELS_DIR / "model.txt"))
    with open(MODELS_DIR / "feature_cols.json", "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)
    # fill-values dict travels with the model (needed at inference, spec 4.3/6.4)
    shutil.copy(DATA_DIR / "fill_values.json", MODELS_DIR / "fill_values.json")

    log.info("Saved model.joblib, model.txt, feature_cols.json, fill_values.json -> %s",
             MODELS_DIR)
    print("\nNext: python -m src.evaluate   (VALIDATION GATE 3 -- beat the grid baseline)")


if __name__ == "__main__":
    main()
