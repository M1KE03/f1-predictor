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

log = logging.getLogger("evaluate")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


# ---------------------------------------------------------------------------
# 5.2 Ranking metrics, computed per race then averaged
# ---------------------------------------------------------------------------
def ranking_metrics(test: pd.DataFrame, score_col: str, ascending: bool) -> dict:
    overlaps, spearmans, top1_hits = [], [], []
    for (_, _), g in test.groupby(["year", "round"], sort=False):
        g = g.copy()
        # rank 1 = strongest predicted driver
        g["pred_rank"] = g[score_col].rank(ascending=ascending, method="first")

        order = g.sort_values("pred_rank")
        pred_top10 = set(order.head(10)["driver"])
        actual_top10 = set(g.loc[g[TARGET] == 1, "driver"])
        overlaps.append(len(pred_top10 & actual_top10) / 10.0)

        top1_hits.append(int(order.iloc[0][TARGET] == 1))

        # Spearman between predicted rank and actual finishing position,
        # on classified drivers (DNFs have no finishing position).
        cls = g[g["classified"] == 1]
        if len(cls) >= 3 and cls["pred_rank"].nunique() > 1:
            spearmans.append(cls["pred_rank"].corr(cls["position"], method="spearman"))

    return dict(
        set_overlap=float(np.mean(overlaps)),
        spearman=float(np.mean(spearmans)),
        top1_hit_rate=float(np.mean(top1_hits)),
        n_races=len(overlaps),
    )


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

    table = pd.DataFrame([model_m, grid_m],
                         index=["model", "grid_baseline"]).round(4)
    print(f"\nRanking metrics per race, averaged over {model_m['n_races']} races:")
    print(table[["set_overlap", "spearman", "top1_hit_rate"]].to_string())

    beats_overlap = model_m["set_overlap"] > grid_m["set_overlap"]
    beats_spearman = model_m["spearman"] > grid_m["spearman"]
    print("\nModel vs grid baseline:")
    print(f"  set-overlap : {'BEATS' if beats_overlap else 'DOES NOT BEAT'} baseline "
          f"({model_m['set_overlap']:.4f} vs {grid_m['set_overlap']:.4f})")
    print(f"  spearman    : {'BEATS' if beats_spearman else 'DOES NOT BEAT'} baseline "
          f"({model_m['spearman']:.4f} vs {grid_m['spearman']:.4f})")

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
