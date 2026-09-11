"""Evaluation for the finishing-order ranker (src.train_rank), on the same
held-out test season convention as evaluate.py.

Metrics that match "predict the finishing order / podium":
  - winner accuracy: the top-ranked driver actually won
  - podium overlap: predicted top-3 against the actual podium
  - Spearman against the published result order, reported twice -- over all
    entries and over finishers only, each with its own denominator
Compared against the grid-position baseline (predict order = starting grid).

All metric logic lives in src.metrics, which is shared with evaluate.py and
applies the deterministic tie policy. This module only loads, scores and
prints.

Run: python -m src.evaluate_rank
"""
import logging
from pathlib import Path

import joblib
import pandas as pd

from .columns import FEATURE_COLS
from .metrics import HEADLINE_COLS, race_metrics

log = logging.getLogger("evaluate_rank")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


def order_metrics(test: pd.DataFrame, score_col: str, ascending: bool) -> dict:
    """Backwards-compatible alias for src.metrics.race_metrics.

    Retained so existing callers keep working. NOTE the returned keys changed
    in increment 1.2: `podium_precision` is now `podium_overlap`, and the
    single `spearman` split into `spearman_all` / `spearman_finishers`.
    """
    return race_metrics(test, score_col, ascending)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR,
                        help="directory holding rank_model.joblib (default: models/)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    log.info("Ranker: %s", args.models_dir / "rank_model.joblib")
    ranker = joblib.load(args.models_dir / "rank_model.joblib")
    df = pd.read_parquet(DATA_DIR / "features.parquet")
    latest = int(df["year"].max())
    test = df[df["year"] == latest].copy()
    log.info("Test season: %s (%s rows, %s races)",
             latest, len(test), test.groupby("round").ngroups)

    test["rank_score"] = ranker.predict(test[FEATURE_COLS])

    print("\n=================== RANKER EVALUATION ===================")
    print(f"Held-out test season: {latest}\n")

    model_m = race_metrics(test, "rank_score", ascending=False)
    grid_m = race_metrics(test, "grid_position", ascending=True)

    table = pd.DataFrame([model_m, grid_m], index=["rank_model", "grid_baseline"])
    print(f"Order metrics per race, over {model_m['n_races']} races:")
    print(table[list(HEADLINE_COLS)].astype(float).round(4).to_string())
    print("\nDenominators (races contributing to each metric):")
    print(table[["n_races_winner", "n_races_podium",
                 "n_races_spearman_finishers"]].to_string())

    beats_spearman = model_m["spearman_all"] > grid_m["spearman_all"]
    beats_podium = model_m["podium_overlap"] > grid_m["podium_overlap"]
    beats_winner = model_m["winner_accuracy"] > grid_m["winner_accuracy"]
    print("\nModel vs grid baseline:")
    print(f"  winner_accuracy : {'BEATS' if beats_winner else 'DOES NOT BEAT'} baseline "
          f"({model_m['winner_accuracy']:.4f} vs {grid_m['winner_accuracy']:.4f})")
    print(f"  podium_overlap  : {'BEATS' if beats_podium else 'DOES NOT BEAT'} baseline "
          f"({model_m['podium_overlap']:.4f} vs {grid_m['podium_overlap']:.4f})")
    print(f"  spearman_all    : {'BEATS' if beats_spearman else 'DOES NOT BEAT'} baseline "
          f"({model_m['spearman_all']:.4f} vs {grid_m['spearman_all']:.4f})")

    if beats_spearman and beats_podium:
        print("\nPASS -- ranker adds value over the grid order.")
    else:
        print("\nFAIL / NULL RESULT -- ranker does not beat the grid baseline on both "
              "metrics. Do not trust its predicted order/podium over grid order yet.")
    print("\nNOTE: winner accuracy is the project's stated objective but is NOT yet "
          "part of the pass condition. The promotion gate is defined in "
          "FIX_PLAN.md section 8 and lands with the milestone 2 harness.")
    print("==========================================================")


if __name__ == "__main__":
    main()
