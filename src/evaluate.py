"""Evaluation on the held-out test season only (spec section 5).

Reports classification metrics, per-race ranking metrics, and the mandatory
side-by-side comparison against the grid-position baseline (VALIDATION
GATE 3). A model that ties the grid baseline is a null result -- this
script says so explicitly rather than hiding behind AUC.

Run: python -m src.evaluate
"""
import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, log_loss, roc_auc_score)

from .columns import FEATURE_COLS, TARGET
from .metrics import HEADLINE_COLS, race_metrics

log = logging.getLogger("evaluate")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


# ---------------------------------------------------------------------------
# 5.2 Ranking metrics -- now delegated to src.metrics (increment 1.2), which
# applies the deterministic tie policy, takes winner/podium from explicit
# labels, and reports separate Spearman denominators.
# ---------------------------------------------------------------------------
def ranking_metrics(test: pd.DataFrame, score_col: str, ascending: bool) -> dict:
    """Backwards-compatible alias for src.metrics.race_metrics.

    NOTE the returned keys changed in increment 1.2: `set_overlap` is now
    `top10_overlap`, the misleading `top1_hit_rate` is now
    `top_pick_finished_top10`, and `spearman` split into `spearman_all` /
    `spearman_finishers`.
    """
    return race_metrics(test, score_col, ascending)


def feature_importance_plot(clf) -> Path:
    booster = clf.booster_
    imp = pd.Series(booster.feature_importance(importance_type="gain"),
                    index=booster.feature_name()).sort_values()
    REPORTS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, max(6, 0.32 * len(imp))))
    imp.plot.barh(ax=ax)
    ax.set_title("LightGBM feature importance (gain) -- test model")
    ax.set_xlabel("gain")
    fig.tight_layout()
    out = REPORTS_DIR / "feature_importance.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    clf = joblib.load(MODELS_DIR / "model.joblib")
    df = pd.read_parquet(DATA_DIR / "features.parquet")
    latest = int(df["year"].max())
    test = df[df["year"] == latest].copy()
    log.info("Test season: %s (%s rows, %s races)",
             latest, len(test), test.groupby('round').ngroups)

    test["p_top10"] = clf.predict_proba(test[FEATURE_COLS])[:, 1]
    y, p = test[TARGET].values, test["p_top10"].values

    # ---- 5.1 classification metrics --------------------------------------
    print("\n=================== VALIDATION GATE 3 ===================")
    print(f"Held-out test season: {latest}\n")
    print("Classification metrics (test):")
    print(f"  ROC-AUC     : {roc_auc_score(y, p):.4f}")
    print(f"  PR-AUC      : {average_precision_score(y, p):.4f}")
    print(f"  log-loss    : {log_loss(y, p):.4f}")
    print(f"  Brier score : {brier_score_loss(y, p):.4f}")
    tn, fp, fn, tp = confusion_matrix(y, (p >= 0.5).astype(int)).ravel()
    print(f"  Confusion @0.5:  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    # ---- 5.2 / 5.3 ranking metrics vs grid baseline ----------------------
    model_m = ranking_metrics(test, "p_top10", ascending=False)
    grid_m = ranking_metrics(test, "grid_position", ascending=True)

    table = pd.DataFrame([model_m, grid_m], index=["model", "grid_baseline"])
    print(f"\nRanking metrics per race, over {model_m['n_races']} races:")
    print(table[list(HEADLINE_COLS)].astype(float).round(4).to_string())
    print("\nDenominators (races contributing to each metric):")
    print(table[["n_races_winner", "n_races_podium",
                 "n_races_spearman_finishers"]].to_string())

    beats_overlap = model_m["top10_overlap"] > grid_m["top10_overlap"]
    beats_spearman = model_m["spearman_all"] > grid_m["spearman_all"]
    print("\nModel vs grid baseline:")
    print(f"  top10_overlap : {'BEATS' if beats_overlap else 'DOES NOT BEAT'} baseline "
          f"({model_m['top10_overlap']:.4f} vs {grid_m['top10_overlap']:.4f})")
    print(f"  spearman_all  : {'BEATS' if beats_spearman else 'DOES NOT BEAT'} baseline "
          f"({model_m['spearman_all']:.4f} vs {grid_m['spearman_all']:.4f})")
    print(f"  winner_acc    : {model_m['winner_accuracy']:.4f} vs "
          f"{grid_m['winner_accuracy']:.4f}  (reported, not yet gated)")

    if beats_overlap and beats_spearman:
        print("\nGATE 3: PASS -- model adds value over the grid order.")
    else:
        print("\nGATE 3: FAIL / NULL RESULT -- the model does not beat the grid "
              "baseline on both ranking metrics. Iterate on features before "
              "doing anything else (spec 5.3). Do not report AUC alone.")

    # ---- 5.4 interpretability ---------------------------------------------
    out = feature_importance_plot(clf)
    booster = clf.booster_
    imp = pd.Series(booster.feature_importance(importance_type="gain"),
                    index=booster.feature_name()).sort_values(ascending=False)
    print(f"\nTop 10 features by gain (full plot: {out}):")
    print(imp.head(10).round(1).to_string())
    grid_rank = int(imp.rank(ascending=False)["grid_position"])
    print(f"\nSanity: grid_position importance rank = {grid_rank} "
          f"(expected to dominate, spec 5.4)")
    print("==========================================================")


if __name__ == "__main__":
    main()
