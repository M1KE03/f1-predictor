"""VALIDATION GATE 2 -- leakage audit (spec end of section 2).

Two checks, both against data/features_prefill.parquet (the snapshot taken
BEFORE the missing-value policy):

1. Random-row recompute: sample rows across different seasons and
   independently recompute a set of historical features from the raw table
   using only strictly-earlier dates -- naive filter-and-aggregate loops,
   sharing no code with the pipeline's groupby/shift implementation. Values
   must match exactly (atol 1e-9; NaN==NaN counts as a match).

2. Global first-race check: for every driver's first-ever race, all
   driver-history features must be NaN pre-fill (never a real computed
   stat), and prior-counts must be 0. Same per (driver, circuit) for the
   circuit features.

Exit code is non-zero on any failure. Do not proceed to training until this
passes on REAL data.

Run: python -m src.audit_leakage [--n-rows 3] [--seed 0]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


# ---------------------------------------------------------------------------
# Independent (naive) recomputations -- deliberately loop/filter based,
# sharing nothing with src/leakage.py.
# ---------------------------------------------------------------------------
def naive_form_avg_finish_3(raw, driver, date):
    h = raw[(raw["driver"] == driver) & (raw["date"] < date)].sort_values("date").tail(3)
    return h["position"].mean() if len(h) else np.nan


def naive_form_dnf_rate_5(raw, driver, date):
    h = raw[(raw["driver"] == driver) & (raw["date"] < date)].sort_values("date").tail(5)
    return h["is_dnf"].mean() if len(h) else np.nan


def naive_driver_circuit_avg(raw, driver, circuit_id, date):
    h = raw[(raw["driver"] == driver) & (raw["circuit_id"] == circuit_id)
            & (raw["date"] < date)]
    return h["position"].mean() if len(h) else np.nan


def naive_driver_dnf_rate(raw, driver, date):
    h = raw[(raw["driver"] == driver) & (raw["date"] < date)]
    return h["is_dnf"].mean() if len(h) else np.nan


def naive_driver_overall_avg(raw, driver, date):
    h = raw[(raw["driver"] == driver) & (raw["date"] < date)]
    return h["position"].mean() if len(h) else np.nan


def naive_team_dnf_rate(raw, team, date):
    h = raw[(raw["team"] == team) & (raw["date"] < date)]
    return h["is_dnf"].mean() if len(h) else np.nan


def naive_team_circuit_avg(raw, team, circuit_id, date):
    h = raw[(raw["team"] == team) & (raw["circuit_id"] == circuit_id)
            & (raw["date"] < date)]
    return h["position"].mean() if len(h) else np.nan


CHECKS = [
    ("form_avg_finish_3", lambda raw, r: naive_form_avg_finish_3(raw, r["driver"], r["date"])),
    ("form_dnf_rate_5", lambda raw, r: naive_form_dnf_rate_5(raw, r["driver"], r["date"])),
    ("driver_circuit_avg_finish", lambda raw, r: naive_driver_circuit_avg(raw, r["driver"], r["circuit_id"], r["date"])),
    ("driver_dnf_rate", lambda raw, r: naive_driver_dnf_rate(raw, r["driver"], r["date"])),
    ("driver_overall_avg_finish", lambda raw, r: naive_driver_overall_avg(raw, r["driver"], r["date"])),
    ("team_dnf_rate", lambda raw, r: naive_team_dnf_rate(raw, r["team"], r["date"])),
    ("team_circuit_avg_finish", lambda raw, r: naive_team_circuit_avg(raw, r["team"], r["circuit_id"], r["date"])),
]

# Driver-history features that MUST be NaN (pre-fill) at a driver's debut.
DEBUT_NAN_FEATURES = [
    "form_avg_finish_3", "form_avg_points_3", "form_avg_quali_3",
    "form_dnf_rate_5", "season_avg_finish", "momentum",
    "driver_overall_avg_finish", "driver_wet_delta", "driver_temp_bin_avg",
    "driver_circuit_avg_finish", "driver_circuit_best_finish",
    "driver_circuit_podium_rate", "driver_dnf_rate",
]
DEBUT_ZERO_COUNTS = ["driver_wet_n", "driver_temp_bin_n", "driver_races_at_circuit"]
CIRCUIT_NAN_FEATURES = ["driver_circuit_avg_finish", "driver_circuit_best_finish",
                        "driver_circuit_podium_rate"]


def _match(a, b, atol=1e-9):
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return abs(float(a) - float(b)) <= atol


def audit(n_rows: int = 3, seed: int = 0) -> bool:
    raw = pd.read_parquet(DATA_DIR / "raw_results.parquet")
    pre = pd.read_parquet(DATA_DIR / "features_prefill.parquet")
    raw["date"] = pd.to_datetime(raw["date"])
    pre["date"] = pd.to_datetime(pre["date"])

    ok = True

    # ---- Check 1: random-row independent recompute -----------------------
    rng = np.random.default_rng(seed)
    years = pre["year"].unique()
    pick_years = rng.choice(years, size=min(n_rows, len(years)), replace=False)
    print("=================== VALIDATION GATE 2 ===================")
    print(f"[1] Independent recompute on {len(pick_years)} random rows "
          f"(one per sampled season):")
    for y in pick_years:
        pool = pre[pre["year"] == y]
        row = pool.iloc[int(rng.integers(len(pool)))]
        print(f"    row: {row['year']} R{row['round']:>2} {row['driver']:>4} "
              f"@ {row['circuit_id']}")
        for feat, fn in CHECKS:
            expected = fn(raw, row)
            got = row[feat]
            good = _match(expected, got)
            ok &= good
            flag = "OK " if good else "FAIL"
            print(f"      {flag} {feat:<28} pipeline={got!r:<22} naive={expected!r}")

    # ---- Check 2: global first-race check --------------------------------
    print("[2] Global first-race check (pre-fill values at every debut):")
    pre_sorted = pre.sort_values("date", kind="mergesort")
    debut = pre_sorted.groupby("driver", sort=False).head(1)

    for feat in DEBUT_NAN_FEATURES:
        n_bad = debut[feat].notna().sum()
        good = n_bad == 0
        ok &= good
        print(f"      {'OK ' if good else 'FAIL'} {feat:<28} "
              f"non-NaN at debut: {n_bad}/{len(debut)}")
    for feat in DEBUT_ZERO_COUNTS:
        n_bad = (debut[feat].fillna(0) != 0).sum()
        good = n_bad == 0
        ok &= good
        print(f"      {'OK ' if good else 'FAIL'} {feat:<28} "
              f"non-zero at debut: {n_bad}/{len(debut)}")

    first_at_circuit = pre_sorted.groupby(["driver", "circuit_id"], sort=False).head(1)
    for feat in CIRCUIT_NAN_FEATURES:
        n_bad = first_at_circuit[feat].notna().sum()
        good = n_bad == 0
        ok &= good
        print(f"      {'OK ' if good else 'FAIL'} {feat:<28} "
              f"non-NaN at first circuit visit: {n_bad}/{len(first_at_circuit)}")

    # teammate deltas at debut must be neutral (0) and flagged unavailable
    final = pd.read_parquet(DATA_DIR / "features.parquet")
    final["date"] = pd.to_datetime(final["date"])
    fdebut = final.sort_values("date", kind="mergesort").groupby("driver", sort=False).head(1)
    n_bad = ((fdebut["quali_gap_to_teammate"] != 0)
             | (fdebut["form_finish_vs_teammate"] != 0)
             | (fdebut["teammate_available"] != 0)).sum()
    good = n_bad == 0
    ok &= good
    print(f"      {'OK ' if good else 'FAIL'} teammate deltas neutral at debut: "
          f"violations {n_bad}/{len(fdebut)}")

    print("----------------------------------------------------------")
    print("GATE 2:", "PASS -- leakage disproven for audited features."
          if ok else "FAIL -- DO NOT TRAIN. Fix the pipeline first.")
    print("==========================================================")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-rows", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sys.exit(0 if audit(args.n_rows, args.seed) else 1)


if __name__ == "__main__":
    main()
