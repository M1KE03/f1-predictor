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
# Re-exported so existing callers (train_rank, blend_rank) keep working.
from .splits import chronological_split  # noqa: F401

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


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DATA_DIR / "features.parquet")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args()
    models_dir = args.models_dir

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    df = pd.read_parquet(args.features)
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

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, models_dir / "model.joblib")
    clf.booster_.save_model(str(models_dir / "model.txt"))
    with open(models_dir / "feature_cols.json", "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)
    # fill-values dict travels with the model (needed at inference, spec 4.3/6.4)
    shutil.copy(args.features.parent / "fill_values.json",
                models_dir / "fill_values.json")

    log.info("Saved model.joblib, model.txt, feature_cols.json, fill_values.json -> %s",
             models_dir)
    print("\nNext: python -m src.evaluate   (VALIDATION GATE 3 -- beat the grid baseline)")


if __name__ == "__main__":
    main()
