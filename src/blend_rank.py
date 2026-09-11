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
from .metrics import HEADLINE_COLS, pred_rank_by_race, race_metrics
from .train import chronological_split

log = logging.getLogger("blend_rank")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


def add_blended_score(df: pd.DataFrame, model_score_col: str, alpha: float) -> pd.Series:
    """Lower blended score = predicted to finish better (so use ascending=True
    downstream, same convention as raw grid_position).

    Both component ranks use the deterministic tie policy (src.metrics). The
    inherited rank(method='first') broke ties by row order, which made the
    blended score -- and therefore the published forecast, not merely its
    evaluation -- depend on how the rows happened to be arranged.
    """
    grid_rank = pred_rank_by_race(df, "grid_position", ascending=True)
    model_rank = pred_rank_by_race(df, model_score_col, ascending=False)
    return alpha * grid_rank + (1 - alpha) * model_rank


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DATA_DIR / "features.parquet")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--write", action="store_true",
                        help="overwrite models/blend_alpha.json with the chosen alpha")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    ranker = joblib.load(args.models_dir / "rank_model.joblib")
    df = pd.read_parquet(args.features)
    df["rank_score"] = ranker.predict(df[FEATURE_COLS])
    _, val, test, latest = chronological_split(df)

    print("\n=================== BLEND TUNING (validation season) ===================")
    best_alpha, best_spearman = None, -np.inf
    rows = []
    for alpha in np.round(np.arange(0.0, 1.01, 0.1), 2):
        val = val.copy()
        val["blend_score"] = add_blended_score(val, "rank_score", alpha)
        m = race_metrics(val, "blend_score", ascending=True)
        rows.append(dict(alpha=alpha, **{k: m[k] for k in HEADLINE_COLS}))
        if m["spearman_all"] > best_spearman:
            best_spearman, best_alpha = m["spearman_all"], alpha
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(f"\nChosen alpha (best validation-season Spearman): {best_alpha}")

    test = test.copy()
    test["blend_score"] = add_blended_score(test, "rank_score", best_alpha)
    blend_m = race_metrics(test, "blend_score", ascending=True)
    grid_m = race_metrics(test, "grid_position", ascending=True)
    model_m = race_metrics(test, "rank_score", ascending=False)

    print(f"\n=================== TEST SEASON ({latest}) RESULT ===================")
    table = pd.DataFrame([blend_m, grid_m, model_m],
                         index=[f"blend(alpha={best_alpha})", "grid_baseline", "rank_model_alone"])
    print(table[list(HEADLINE_COLS)].astype(float).round(4).to_string())

    beats_spearman = blend_m["spearman_all"] > grid_m["spearman_all"]
    beats_podium = blend_m["podium_overlap"] >= grid_m["podium_overlap"]
    print("\nBlend vs grid baseline:")
    print(f"  spearman         : {'BEATS' if beats_spearman else 'DOES NOT BEAT'} baseline "
          f"({blend_m['spearman_all']:.4f} vs {grid_m['spearman_all']:.4f})")
    print(f"  podium_overlap   : {'>=' if beats_podium else '<'} baseline "
          f"({blend_m['podium_overlap']:.4f} vs {grid_m['podium_overlap']:.4f})")
    print(f"  winner_accuracy  : {blend_m['winner_accuracy']:.4f} vs "
          f"{grid_m['winner_accuracy']:.4f}  (NOT the selection criterion)")
    print("\nNOTE: alpha is still selected by validation Spearman, which is not "
          "the winner/podium objective this project is for. Changing the "
          "selection criterion is a separate decision (FIX_PLAN.md section 2, "
          "P1) and is deliberately not bundled into this correctness pass.")

    alpha_path = args.models_dir / "blend_alpha.json"
    if args.write:
        with open(alpha_path, "w") as f:
            json.dump({"alpha": float(best_alpha)}, f, indent=2)
        print(f"\nSaved alpha -> {alpha_path}")
    else:
        # The saved artifact is hashed in reports/baseline.json, so overwriting
        # it is now an explicit act rather than a side effect of inspecting the
        # sweep.
        print(f"\nalpha NOT saved. Pass --write to overwrite {alpha_path}.")
    print("==========================================================")


if __name__ == "__main__":
    main()
