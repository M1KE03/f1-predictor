"""Blend the finishing-order ranker (src.train_rank) with the grid-position
baseline, since neither evaluate_rank.py run showed the ranker alone beating
grid order on this data's small test season.

Per race, both signals are converted to a rank (1 = predicted best) and
combined as:

    blended_rank = alpha * grid_rank + (1 - alpha) * model_rank

alpha is swept on the VALIDATION season only (never the held-out test
season) and the value that maximizes Spearman correlation with actual
finishing position is kept. That alpha is then reported -- honestly -- on
the test season, and saved for src.predict to reuse.

Run: python -m src.blend_rank
"""
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .columns import FEATURE_COLS
from .evaluate_rank import order_metrics
from .train import chronological_split

log = logging.getLogger("blend_rank")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


def add_blended_score(df: pd.DataFrame, model_score_col: str, alpha: float) -> pd.Series:
    """Lower blended score = predicted to finish better (so use ascending=True
    downstream, same convention as raw grid_position)."""
    grid_rank = df.groupby(["year", "round"])["grid_position"].rank(ascending=True, method="first")
    model_rank = df.groupby(["year", "round"])[model_score_col].rank(ascending=False, method="first")
    return alpha * grid_rank + (1 - alpha) * model_rank


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    ranker = joblib.load(MODELS_DIR / "rank_model.joblib")
    df = pd.read_parquet(DATA_DIR / "features.parquet")
    df["rank_score"] = ranker.predict(df[FEATURE_COLS])
    _, val, test, latest = chronological_split(df)

    print("\n=================== BLEND TUNING (validation season) ===================")
    best_alpha, best_spearman = None, -np.inf
    rows = []
    for alpha in np.round(np.arange(0.0, 1.01, 0.1), 2):
        val = val.copy()
        val["blend_score"] = add_blended_score(val, "rank_score", alpha)
        m = order_metrics(val, "blend_score", ascending=True)
        rows.append(dict(alpha=alpha, **m))
        if m["spearman"] > best_spearman:
            best_spearman, best_alpha = m["spearman"], alpha
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(f"\nChosen alpha (best validation-season Spearman): {best_alpha}")

    test = test.copy()
    test["blend_score"] = add_blended_score(test, "rank_score", best_alpha)
    blend_m = order_metrics(test, "blend_score", ascending=True)
    grid_m = order_metrics(test, "grid_position", ascending=True)
    model_m = order_metrics(test, "rank_score", ascending=False)

    print(f"\n=================== TEST SEASON ({latest}) RESULT ===================")
    table = pd.DataFrame([blend_m, grid_m, model_m],
                         index=[f"blend(alpha={best_alpha})", "grid_baseline", "rank_model_alone"]).round(4)
    print(table[["spearman", "podium_precision", "winner_accuracy"]].to_string())

    beats_spearman = blend_m["spearman"] > grid_m["spearman"]
    beats_podium = blend_m["podium_precision"] >= grid_m["podium_precision"]
    print("\nBlend vs grid baseline:")
    print(f"  spearman         : {'BEATS' if beats_spearman else 'DOES NOT BEAT'} baseline "
          f"({blend_m['spearman']:.4f} vs {grid_m['spearman']:.4f})")
    print(f"  podium_precision : {'>=' if beats_podium else '<'} baseline "
          f"({blend_m['podium_precision']:.4f} vs {grid_m['podium_precision']:.4f})")

    with open(MODELS_DIR / "blend_alpha.json", "w") as f:
        json.dump({"alpha": float(best_alpha)}, f, indent=2)
    print(f"\nSaved alpha -> {MODELS_DIR / 'blend_alpha.json'}")
    print("==========================================================")


if __name__ == "__main__":
    main()
