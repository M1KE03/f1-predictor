"""Learning-to-rank training: predict each race's finishing ORDER, not an
independent top-10 flag per driver (that's what train.py / model.joblib do,
and per evaluate.py it doesn't beat the grid-order baseline on rank
correlation -- the wrong tool for "who finishes where").

LightGBM's lambdarank objective optimizes ordering *within each race group*
directly, which is what "predict the finishing order / podium" actually
needs. Same leakage-safe features as the classifier (FEATURE_COLS, audited
by src.audit_leakage); only the label, objective, and grouping differ.

Label (relevance, higher = finished better): officially classified drivers get
(field_size - result_order + 1); drivers who never started, were disqualified
or withdrew get 0. Groups = one race each, sizes passed to LightGBM in row
order.

Run:
    python -m src.train_rank                      # writes models/
    python -m src.train_rank --models-dir models/v2
"""
import json
import logging
import shutil
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd

from .columns import FEATURE_COLS
from .metrics import ensure_labels
from .train import chronological_split

log = logging.getLogger("train_rank")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

RANDOM_STATE = 42

PARAMS = dict(
    objective="lambdarank",
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
    random_state=RANDOM_STATE,
)


def add_relevance_label(df: pd.DataFrame) -> pd.DataFrame:
    """Relevance = better official finish scores higher; ineligible rows get 0.

    Eligibility is `officially_classified`, not the deprecated `classified`
    flag. The two differ for exactly the rows that should score 0: drivers who
    never started, were disqualified, or withdrew. A retirement that still
    holds a place in the published classification keeps its real relevance,
    per FIX_PLAN.md section 4 -- retirements are not collapsed to last.

    NOTE the grade scheme itself is unchanged: relevance is still
    `field_size - result_order + 1`, so it depends on field size and, under
    LightGBM's exponential default `label_gain`, weights P1 enormously more
    than P2. FIX_PLAN.md section 6 proposes a bounded grade scheme as an
    experiment; that is a modelling change and is deliberately not made here.
    """
    df = ensure_labels(df).copy()
    field_size = df.groupby(["year", "round"])["driver"].transform("size")
    df["relevance"] = 0
    eligible = df["officially_classified"] == 1
    df.loc[eligible, "relevance"] = (field_size - df["result_order"] + 1).clip(lower=0)
    df["relevance"] = df["relevance"].fillna(0).astype(int)
    return df


def group_sizes(df: pd.DataFrame) -> list[int]:
    """Row order within each (year, round) must be contiguous -- true here
    since features.parquet is date-sorted and a race's rows are never split
    by another race's rows."""
    return df.groupby(["year", "round"], sort=False).size().tolist()


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DATA_DIR / "features.parquet")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR,
                        help="where to write the ranker (default: models/). Use a "
                             "separate directory to keep a frozen model for A/B.")
    args = parser.parse_args()
    models_dir = args.models_dir

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    df = pd.read_parquet(args.features)
    df = add_relevance_label(df)
    train, val, test, latest = chronological_split(df)
    log.info("Split (latest season = %s): train=%s rows (<=%s), val=%s rows (%s), "
             "test=%s rows (%s, HELD OUT)",
             latest, len(train), latest - 2, len(val), latest - 1, len(test), latest)

    X_tr, y_tr = train[FEATURE_COLS], train["relevance"]
    X_val, y_val = val[FEATURE_COLS], val["relevance"]

    ranker = lgb.LGBMRanker(**PARAMS)
    ranker.fit(
        X_tr, y_tr, group=group_sizes(train),
        eval_set=[(X_val, y_val)], eval_group=[group_sizes(val)],
        eval_at=[3, 10],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)],
    )

    log.info("Best iteration: %s", ranker.best_iteration_)
    log.info("Best val scores: %s", dict(ranker.best_score_.get("valid_0", {})))

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(ranker, models_dir / "rank_model.joblib")
    with open(models_dir / "feature_cols.json", "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)
    shutil.copy(args.features.parent / "fill_values.json",
                models_dir / "fill_values.json")

    log.info("Saved rank_model.joblib -> %s", models_dir)
    print("\nNext: python -m src.evaluate_rank   (order/podium metrics vs grid baseline)")


if __name__ == "__main__":
    main()
