"""Evaluation for the finishing-order ranker (src.train_rank), on the same
held-out test season convention as evaluate.py.

Metrics that actually match "predict the finishing order / podium":
  - Spearman correlation between predicted order and actual finishing position
  - podium precision: overlap between predicted top-3 and actual top-3
  - winner accuracy: predicted P1 actually finished P1
Compared against the grid-position baseline (predict order = starting grid).

Run: python -m src.evaluate_rank
"""
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .columns import FEATURE_COLS

log = logging.getLogger("evaluate_rank")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


def order_metrics(test: pd.DataFrame, score_col: str, ascending: bool) -> dict:
    podium_overlaps, spearmans, winner_hits = [], [], []
    for (_, _), g in test.groupby(["year", "round"], sort=False):
        g = g.copy()
        g["pred_rank"] = g[score_col].rank(ascending=ascending, method="first")

        order = g.sort_values("pred_rank")
        pred_podium = set(order.head(3)["driver"])
        actual_podium = set(g.loc[(g["classified"] == 1) & (g["position"] <= 3), "driver"])
        if actual_podium:
            podium_overlaps.append(len(pred_podium & actual_podium) / 3.0)

        actual_winner = g.loc[(g["classified"] == 1) & (g["position"] == 1), "driver"]
        if len(actual_winner):
            winner_hits.append(int(order.iloc[0]["driver"] == actual_winner.iloc[0]))

        cls = g[g["classified"] == 1]
        if len(cls) >= 3 and cls["pred_rank"].nunique() > 1:
            spearmans.append(cls["pred_rank"].corr(cls["position"], method="spearman"))

    return dict(
        spearman=float(np.mean(spearmans)),
        podium_precision=float(np.mean(podium_overlaps)),
        winner_accuracy=float(np.mean(winner_hits)),
        n_races=len(spearmans),
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    ranker = joblib.load(MODELS_DIR / "rank_model.joblib")
    df = pd.read_parquet(DATA_DIR / "features.parquet")
    latest = int(df["year"].max())
    test = df[df["year"] == latest].copy()
    log.info("Test season: %s (%s rows, %s races)",
             latest, len(test), test.groupby("round").ngroups)

    test["rank_score"] = ranker.predict(test[FEATURE_COLS])

    print("\n=================== RANKER EVALUATION ===================")
    print(f"Held-out test season: {latest}\n")

    model_m = order_metrics(test, "rank_score", ascending=False)
    grid_m = order_metrics(test, "grid_position", ascending=True)

    table = pd.DataFrame([model_m, grid_m], index=["rank_model", "grid_baseline"]).round(4)
    print(f"Order metrics per race, averaged over {model_m['n_races']} races:")
    print(table[["spearman", "podium_precision", "winner_accuracy"]].to_string())

    beats_spearman = model_m["spearman"] > grid_m["spearman"]
    beats_podium = model_m["podium_precision"] > grid_m["podium_precision"]
    print("\nModel vs grid baseline:")
    print(f"  spearman         : {'BEATS' if beats_spearman else 'DOES NOT BEAT'} baseline "
          f"({model_m['spearman']:.4f} vs {grid_m['spearman']:.4f})")
    print(f"  podium_precision : {'BEATS' if beats_podium else 'DOES NOT BEAT'} baseline "
          f"({model_m['podium_precision']:.4f} vs {grid_m['podium_precision']:.4f})")

    if beats_spearman and beats_podium:
        print("\nPASS -- ranker adds value over the grid order.")
    else:
        print("\nFAIL / NULL RESULT -- ranker does not beat the grid baseline on both "
              "metrics. Do not trust its predicted order/podium over grid order yet.")
    print("==========================================================")


if __name__ == "__main__":
    main()
