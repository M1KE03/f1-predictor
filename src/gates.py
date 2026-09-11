"""Promotion gates and paired uncertainty intervals (FIX_PLAN.md section 8).

Reads the backtest's exported predictions and answers one question: does a
candidate beat the strongest simple baseline by enough, with enough evidence,
to be promoted?

Three rules this module exists to enforce:

1. **Resample races, not driver rows.** Twenty-two entries in one race are not
   twenty-two independent observations -- they share a track, a weather
   window and a safety-car history. A driver-row bootstrap would report
   intervals several times too narrow.
2. **Compare paired, not marginal.** The candidate and the baseline are scored
   on the same races, so the interval belongs on the per-race DIFFERENCE. Two
   overlapping marginal intervals say nothing about whether the difference is
   real.
3. **Fail loudly.** `main()` exits non-zero when a gate fails. The inherited
   `evaluate.py` printed "FAIL / NULL RESULT" and exited 0, so nothing
   automated could ever notice.

The probability gates in FIX_PLAN.md section 8 (winner log loss, podium Brier)
are NOT implemented here: they need calibrated race-level win probabilities,
which arrive with milestone 4. `summary()` reports them as unavailable rather
than silently scoring a gate that was never checked.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .blend_rank import add_blended_score
from .metrics import HEADLINE_COLS, race_metric_rows

log = logging.getLogger("gates")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
BACKTEST_DIR = REPORTS_DIR / "backtest"

N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20260911
CONFIDENCE = 0.95

# FIX_PLAN.md section 8. Proposals to be frozen before experiments, not
# guaranteed gains -- they are thresholds, not predictions.
MIN_WINNER_GAIN = 0.05        # +5 percentage points
MIN_PODIUM_GAIN = 0.03
MAX_TOP10_LOSS = 0.02
MAX_SPEARMAN_LOSS = 0.02
MIN_FOLDS = 3
MIN_RACES = 60

ORDERINGS: dict[str, tuple[str, bool]] = {
    "grid_baseline": ("grid_position", True),
    "top10_classifier": ("p_top10", False),
    "rank_model": ("rank_score", False),
    "blend": ("blend_score", True),
}
BASELINES = ("grid_baseline",)


def paired_difference(candidate: pd.Series, baseline: pd.Series,
                      n_bootstrap: int = N_BOOTSTRAP,
                      seed: int = BOOTSTRAP_SEED,
                      confidence: float = CONFIDENCE) -> dict[str, Any]:
    """Mean per-race difference with a bootstrap interval, resampled by race."""
    paired = pd.DataFrame({"candidate": candidate, "baseline": baseline}).dropna()
    n = len(paired)
    if n == 0:
        return {"n_races": 0, "mean_difference": None, "ci_low": None,
                "ci_high": None, "resolves": False}

    differences = (paired["candidate"] - paired["baseline"]).to_numpy()
    rng = np.random.default_rng(seed)
    # One draw per race, with replacement -- the unit of resampling is the race.
    draws = rng.integers(0, n, size=(n_bootstrap, n))
    means = differences[draws].mean(axis=1)

    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return {
        "n_races": int(n),
        "mean_difference": float(differences.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        # True when the interval excludes zero, i.e. the sign is resolved.
        "resolves": bool(low > 0 or high < 0),
    }


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str


def evaluate_gates(candidate: pd.DataFrame, baseline: pd.DataFrame,
                   n_folds: int) -> list[GateResult]:
    """Apply the FIX_PLAN.md section 8 promotion rules."""
    results: list[GateResult] = []

    def difference(metric: str) -> dict[str, Any]:
        return paired_difference(candidate[metric], baseline[metric])

    winner = difference("winner_accuracy")
    results.append(GateResult(
        "winner accuracy +0.05",
        bool(winner["mean_difference"] is not None
             and winner["mean_difference"] >= MIN_WINNER_GAIN),
        f"{winner['mean_difference']:+.4f} "
        f"[{winner['ci_low']:+.4f}, {winner['ci_high']:+.4f}]"
        if winner["mean_difference"] is not None else "no races"))

    podium = difference("podium_overlap")
    results.append(GateResult(
        "podium overlap +0.03",
        bool(podium["mean_difference"] is not None
             and podium["mean_difference"] >= MIN_PODIUM_GAIN),
        f"{podium['mean_difference']:+.4f} "
        f"[{podium['ci_low']:+.4f}, {podium['ci_high']:+.4f}]"
        if podium["mean_difference"] is not None else "no races"))

    for metric, limit, label in (("top10_overlap", MAX_TOP10_LOSS, "top-10 overlap"),
                                 ("spearman_all", MAX_SPEARMAN_LOSS, "spearman")):
        guard = difference(metric)
        results.append(GateResult(
            f"{label} loses <= {limit}",
            bool(guard["mean_difference"] is not None
                 and guard["mean_difference"] >= -limit),
            f"{guard['mean_difference']:+.4f} "
            f"[{guard['ci_low']:+.4f}, {guard['ci_high']:+.4f}]"
            if guard["mean_difference"] is not None else "no races"))

    results.append(GateResult(
        f"at least {MIN_FOLDS} folds", n_folds >= MIN_FOLDS, f"{n_folds} folds"))
    results.append(GateResult(
        f"at least {MIN_RACES} races", winner["n_races"] >= MIN_RACES,
        f"{winner['n_races']} races"))
    results.append(GateResult(
        "winner gain resolves (CI excludes 0)", bool(winner["resolves"]),
        "interval excludes 0" if winner["resolves"]
        else "interval spans 0 -- inconclusive, keep collecting"))
    return results


def score_predictions(predictions: pd.DataFrame,
                      alpha: float = 0.5) -> dict[str, pd.DataFrame]:
    """Per-race metric rows for every ordering method."""
    data = predictions.copy()
    data["blend_score"] = add_blended_score(data, "rank_score", alpha)
    return {name: race_metric_rows(data, col, ascending).set_index(["year", "round"])
            for name, (col, ascending) in ORDERINGS.items()}


def summary(per_race: dict[str, pd.DataFrame], n_folds: int) -> dict[str, Any]:
    baseline_name = max(
        BASELINES, key=lambda b: per_race[b]["winner_accuracy"].mean())
    baseline = per_race[baseline_name]

    report: dict[str, Any] = {
        "baseline": baseline_name,
        "n_folds": n_folds,
        "pooled": {name: {m: (float(rows[m].mean()) if rows[m].notna().any() else None)
                          for m in HEADLINE_COLS}
                   for name, rows in per_race.items()},
        "paired_vs_baseline": {
            name: {m: paired_difference(rows[m], baseline[m]) for m in HEADLINE_COLS}
            for name, rows in per_race.items() if name != baseline_name},
        "gates": {},
        "unavailable_gates": [
            "winner log loss and podium Brier need calibrated race-level win "
            "probabilities (milestone 4); not scored here.",
        ],
    }
    for name, rows in per_race.items():
        if name == baseline_name:
            continue
        gates = evaluate_gates(rows, baseline, n_folds)
        report["gates"][name] = {
            "passed": all(g.passed for g in gates),
            "checks": [{"name": g.name, "passed": g.passed, "detail": g.detail}
                       for g in gates],
        }
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backtest-dir", type=Path, default=BACKTEST_DIR)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="blend weight on grid rank")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    predictions = pd.read_parquet(args.backtest_dir / "predictions.parquet")
    with (args.backtest_dir / "folds.json").open(encoding="utf-8") as fh:
        folds = json.load(fh)
    n_folds = len(folds["folds"])

    per_race = score_predictions(predictions, alpha=args.alpha)
    report = summary(per_race, n_folds)

    out = args.out or (args.backtest_dir / "gates.json")
    with out.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(f"\n=================== PROMOTION GATES ===================")
    print(f"scheme folds : {n_folds}   baseline: {report['baseline']}")
    print(f"races        : {len(per_race[report['baseline']])}\n")

    table = pd.DataFrame(report["pooled"]).T[list(HEADLINE_COLS)]
    print("Pooled metrics (mean over races):")
    print(table.astype(float).round(4).to_string())

    for name, gate in report["gates"].items():
        print(f"\n--- {name} vs {report['baseline']} ---")
        for check in gate["checks"]:
            print(f"  {'PASS' if check['passed'] else 'FAIL'}  "
                  f"{check['name']:<38} {check['detail']}")
        print(f"  => {'PROMOTE' if gate['passed'] else 'DO NOT PROMOTE'}")

    for note in report["unavailable_gates"]:
        print(f"\nNOT SCORED: {note}")
    print(f"\nWrote {out}")
    print("=======================================================")

    if not any(gate["passed"] for gate in report["gates"].values()):
        print("\nNo candidate cleared the promotion gates. Exiting non-zero so "
              "this cannot pass unnoticed in an automated run.")
        sys.exit(1)


if __name__ == "__main__":
    main()
