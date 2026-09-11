"""Milestone 0 -- freeze the legacy baseline (FIX_PLAN.md sections 5.A.1 and 9).

Writes reports/baseline.json: a provenance record binding the current dataset
and model artifacts (by SHA-256) to the metrics they produce, so that every
later change can be measured against a known starting point.

This module deliberately reproduces LEGACY behaviour, flaws included. It calls
the existing evaluate.ranking_metrics / evaluate_rank.order_metrics rather than
reimplementing them, because the point is to record what the project scores
TODAY, not what it should score. The recorded numbers are NOT a clean estimate
of live forecasting quality -- they are measured on features that include
race-session weather and globally-fitted imputation, i.e. information that is
not available before lights-out. See FIX_PLAN.md section 2.

Consequently the P0 fixes are expected to move these numbers DOWN. That is the
leak being removed, not a regression.

Run: python -m src.baseline
"""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .blend_rank import add_blended_score
from .columns import FEATURE_COLS, TARGET
from .evaluate import ranking_metrics
from .evaluate_rank import order_metrics

log = logging.getLogger("baseline")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
BASELINE_PATH = REPORTS_DIR / "baseline.json"

# Artifacts whose exact bytes define this baseline.
TRACKED_ARTIFACTS = (
    "data/raw_results.parquet",
    "data/features.parquet",
    "data/fill_values.json",
    "models/model.joblib",
    "models/rank_model.joblib",
    "models/blend_alpha.json",
    "models/feature_cols.json",
)

# Recorded in FIX_PLAN.md's appendix at review time. Used to prove the
# artifacts have not drifted since the diagnosis was written.
REVIEW_HASHES = {
    "data/raw_results.parquet": "5d32b7fabb60f77b6357081827db936677d87a0e38729eb7318acf86523895ba",
    "data/features.parquet": "686890d50b49608d7391e18a76cc671cd8e3f3a55b4441f11a9f1476bc56c0e6",
    "models/model.joblib": "2d6ae5dce28033a96672b4706365e678842ab0879a5d6cc2d4a9575bc80ca2ab",
    "models/rank_model.joblib": "ae67fdc89b2e7cb14ca8a6b83580ab7246790b8094ecfc4dffd0e8376e55a625",
    "models/blend_alpha.json": "c0f916c7c40c0cdf72d928d94102684af8722ed812b4346ddfe1fcc4ac4555f6",
}

PINNED_PACKAGES = ("fastf1", "lightgbm", "pandas", "numpy", "scikit-learn",
                   "pyarrow", "joblib", "matplotlib")

SHUFFLE_SEEDS = (0, 1, 2, 3, 4)


def sha256(path: Path) -> str | None:
    """SHA-256 of a file, or None if it does not exist."""
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_manifest() -> dict[str, Any]:
    """Hash every tracked artifact and flag drift from the review snapshot."""
    files: dict[str, Any] = {}
    for rel in TRACKED_ARTIFACTS:
        path = PROJECT_ROOT / rel
        digest = sha256(path)
        entry: dict[str, Any] = {
            "sha256": digest,
            "bytes": path.stat().st_size if path.exists() else None,
        }
        if rel in REVIEW_HASHES:
            entry["matches_review_hash"] = digest == REVIEW_HASHES[rel]
        files[rel] = entry

    checked = [v for v in files.values() if "matches_review_hash" in v]
    return {
        "files": files,
        "all_review_hashes_match": all(v["matches_review_hash"] for v in checked),
        "n_review_hashes_checked": len(checked),
    }


def environment() -> dict[str, Any]:
    """Interpreter and the package versions that affect model behaviour."""
    versions: dict[str, str | None] = {}
    for name in PINNED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
    }


def data_coverage(df: pd.DataFrame) -> dict[str, Any]:
    """Row/race counts per season, so a later dataset change is visible."""
    per_season = (df.groupby("year")
                    .agg(rows=("driver", "size"), races=("round", "nunique"))
                    .sort_index())
    return {
        "n_rows": int(len(df)),
        "n_races": int(df.groupby(["year", "round"]).ngroups),
        "n_features": len(FEATURE_COLS),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
        "seasons": {str(year): {"rows": int(row.rows), "races": int(row.races)}
                    for year, row in per_season.iterrows()},
        # Evidence for the label-semantics defect (FIX_PLAN section 2, P0):
        # these rows are simultaneously flagged as retired and as classified.
        "rows_dnf_and_classified": int(
            ((df["is_dnf"] == 1) & (df["classified"] == 1)).sum()),
    }


def classifier_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Legacy top-10 classification metrics on the test season."""
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 brier_score_loss, log_loss, roc_auc_score)
    return {
        "accuracy_at_0.5": float(accuracy_score(y, (p >= 0.5).astype(int))),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def scored_test_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, int, float]:
    """Attach every legacy score column to the held-out test season."""
    latest = int(df["year"].max())
    test = df[df["year"] == latest].copy()

    clf = joblib.load(MODELS_DIR / "model.joblib")
    ranker = joblib.load(MODELS_DIR / "rank_model.joblib")
    with (MODELS_DIR / "blend_alpha.json").open() as fh:
        alpha = float(json.load(fh)["alpha"])

    test["p_top10"] = clf.predict_proba(test[FEATURE_COLS])[:, 1]
    test["rank_score"] = ranker.predict(test[FEATURE_COLS])
    test["blend_score"] = add_blended_score(test, "rank_score", alpha)
    return test, latest, alpha


# (score column, ascending) for each ordering method, under legacy conventions.
ORDERINGS: dict[str, tuple[str, bool]] = {
    "grid_baseline": ("grid_position", True),
    "top10_classifier": ("p_top10", False),
    "rank_model": ("rank_score", False),
    "blend": ("blend_score", True),
}


def ordering_metrics(test: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Legacy per-race metrics for every ordering method.

    NOTE `top1_hit_rate` keeps its misleading legacy name: it is true when the
    top-scored driver finishes anywhere in the top 10, NOT when they win.
    FIX_PLAN.md flags the rename; it is recorded here unchanged so the frozen
    baseline matches what the code actually reported.
    """
    out: dict[str, dict[str, float]] = {}
    for name, (col, ascending) in ORDERINGS.items():
        out[name] = {**order_metrics(test, col, ascending),
                     **ranking_metrics(test, col, ascending)}
    return out


def shuffle_sensitivity(test: pd.DataFrame,
                        seeds: tuple[int, ...] = SHUFFLE_SEEDS) -> dict[str, Any]:
    """Quantify the row-order dependence caused by rank(method='first').

    Identical rows in a different order must not change a forecast. They
    currently do, so the spread is recorded both as the defect's magnitude and
    as the regression target for the deterministic tie policy (FIX_PLAN
    section 2, P0).
    """
    per_method: dict[str, Any] = {}
    for name, (col, ascending) in ORDERINGS.items():
        spearmans = []
        for seed in seeds:
            shuffled = test.sample(frac=1.0, random_state=seed)
            spearmans.append(order_metrics(shuffled, col, ascending)["spearman"])
        per_method[name] = {
            "seeds": list(seeds),
            "spearman_per_seed": [float(s) for s in spearmans],
            "spearman_min": float(np.min(spearmans)),
            "spearman_max": float(np.max(spearmans)),
            "spearman_spread": float(np.max(spearmans) - np.min(spearmans)),
        }
    return {
        "row_order_sensitivity": per_method,
        "order_dependent": any(v["spearman_spread"] > 1e-12
                               for v in per_method.values()),
    }


def git_commit() -> str | None:
    """Current HEAD, read-only. Returns None outside a git checkout."""
    try:
        done = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, check=True)
        return done.stdout.strip()
    except Exception:
        return None


def build_baseline() -> dict[str, Any]:
    df = pd.read_parquet(DATA_DIR / "features.parquet")
    df["date"] = pd.to_datetime(df["date"])
    test, latest, alpha = scored_test_frame(df)

    return {
        "schema_version": 1,
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "description": (
            "Frozen legacy baseline captured before the FIX_PLAN.md corrections. "
            "Measured with race-session weather and globally-fitted imputation in "
            "the feature matrix, so these numbers overstate live forecasting "
            "quality and are expected to fall once the P0 fixes land."
        ),
        "git_commit": git_commit(),
        "environment": environment(),
        "artifacts": artifact_manifest(),
        "data": data_coverage(df),
        "split": {
            "convention": "train: year <= latest-2, val: latest-1, test: latest",
            "latest_season": latest,
            "test_rows": int(len(test)),
            "test_races": int(test.groupby("round").ngroups),
            "blend_alpha": alpha,
        },
        "metrics": {
            "classifier": classifier_metrics(test[TARGET].to_numpy(),
                                             test["p_top10"].to_numpy()),
            "orderings": ordering_metrics(test),
        },
        "known_defects": shuffle_sensitivity(test),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    baseline = build_baseline()

    REPORTS_DIR.mkdir(exist_ok=True)
    with BASELINE_PATH.open("w", encoding="utf-8") as fh:
        json.dump(baseline, fh, indent=2)

    art = baseline["artifacts"]
    cov = baseline["data"]
    print("\n=================== BASELINE FROZEN ===================")
    print(f"artifacts hashed     : {len(art['files'])}")
    print(f"match review hashes  : {art['all_review_hashes_match']} "
          f"({art['n_review_hashes_checked']} checked)")
    print(f"dataset              : {cov['n_rows']} rows, {cov['n_races']} races, "
          f"{cov['date_min'][:10]} -> {cov['date_max'][:10]}")
    print(f"test season          : {baseline['split']['latest_season']} "
          f"({baseline['split']['test_races']} races)")

    print("\nLegacy ordering metrics (test season):")
    table = pd.DataFrame(baseline["metrics"]["orderings"]).T
    print(table[["winner_accuracy", "podium_precision", "set_overlap",
                 "spearman"]].round(4).to_string())

    sensitivity = baseline["known_defects"]["row_order_sensitivity"]
    spread = max(v["spearman_spread"] for v in sensitivity.values())
    print(f"\nRow-order dependence : max Spearman spread over "
          f"{len(SHUFFLE_SEEDS)} shuffles = {spread:.4f}")
    print("  (must become 0.0000 once the deterministic tie policy lands)")
    print(f"\nWrote {BASELINE_PATH}")
    print("=======================================================")


if __name__ == "__main__":
    main()
