"""Feature building orchestrator (spec sections 2 and 3).

Order: sort -> recent form (2.1) -> weather affinity (2.2) -> circuit
history (2.3) -> reliability/constructor -> teammate deltas (2.4, needs
form) -> missing-value policy (2.5).

Outputs:
  data/features_prefill.parquet  (pre-fill snapshot, used by the Gate 2 audit)
  data/features.parquet          (final matrix, zero NaNs in FEATURE_COLS)
  data/fill_values.json          (exact fills used; required at inference)

Run: python -m src.build_features
Then: python -m src.audit_leakage   (VALIDATION GATE 2 -- do not skip)
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .circuit import add_circuit_history
from .columns import FEATURE_COLS, ID_COLS, TARGET
from .leakage import (past_expanding_mean, past_mean_excluding_current_race,
                      past_rolling_mean, sort_frame)
from .teammate import add_teammate_features
from .weather import add_weather_affinity

log = logging.getLogger("build_features")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_PATH = DATA_DIR / "raw_results.parquet"
PREFILL_PATH = DATA_DIR / "features_prefill.parquet"
FEATURES_PATH = DATA_DIR / "features.parquet"
FILLS_PATH = DATA_DIR / "fill_values.json"


# ---------------------------------------------------------------------------
# 2.1 Recent form
# ---------------------------------------------------------------------------
def add_recent_form(df: pd.DataFrame) -> pd.DataFrame:
    df["form_avg_finish_3"] = past_rolling_mean(df, "driver", "position", 3)
    df["form_avg_points_3"] = past_rolling_mean(df, "driver", "points", 3)
    df["form_avg_quali_3"] = past_rolling_mean(df, "driver", "grid_position", 3)
    df["form_dnf_rate_5"] = past_rolling_mean(df, "driver", "is_dnf", 5)
    df["season_avg_finish"] = past_expanding_mean(df, ["driver", "year"], "position")
    # negative = currently overperforming the season baseline
    df["momentum"] = df["form_avg_finish_3"] - df["season_avg_finish"]
    return df


# ---------------------------------------------------------------------------
# Reliability / car (spec 3 feature list)
# ---------------------------------------------------------------------------
def add_reliability(df: pd.DataFrame) -> pd.DataFrame:
    df["driver_dnf_rate"] = past_expanding_mean(df, "driver", "is_dnf")
    # team has two rows per race -> exclude the whole current race, not just
    # the current row (plain shift(1) would leak the teammate's outcome)
    df["team_dnf_rate"] = past_mean_excluding_current_race(df, "team", "is_dnf")

    # Constructor championship points strictly BEFORE this race, per season.
    tp = (df.groupby(["year", "round", "team"], as_index=False)
            .agg(date=("date", "first"), team_pts=("points", "sum"))
            .sort_values("date", kind="mergesort"))
    tp["constructor_standing_prior"] = (
        tp.groupby(["team", "year"])["team_pts"]
          .transform(lambda s: s.cumsum().shift(1))
    )
    df = df.merge(tp[["year", "round", "team", "constructor_standing_prior"]],
                  on=["year", "round", "team"], how="left")
    return df


# ---------------------------------------------------------------------------
# 2.5 Missing-value policy (uniform, last step)
# ---------------------------------------------------------------------------
# Finishing-position-scaled features: never filled with 0 (0 would mean
# "won"). Fallback chain: driver_overall_avg_finish -> global mean finish.
POSITION_SCALED = [
    "form_avg_finish_3", "form_avg_quali_3", "season_avg_finish",
    "driver_circuit_avg_finish",
    "driver_circuit_best_finish", "team_circuit_avg_finish",
]
# Rate features on a 0-1 scale: filled with the global mean of that rate.
RATE_FEATURES = ["form_dnf_rate_5", "driver_dnf_rate", "team_dnf_rate",
                 "driver_circuit_podium_rate"]
# Deltas and sums where 0 is the neutral value.
ZERO_FILL = ["form_avg_points_3", "momentum", "driver_wet_delta",
             "quali_gap_to_teammate", "form_finish_vs_teammate",
             "driver_wet_n", "driver_races_at_circuit",
             "constructor_standing_prior"]
# Flags with sensible defaults (spec 2.5).
FLAG_DEFAULTS = {"is_rookie_here": 1, "teammate_available": 0, "pit_start": 0}
# Race-condition weather is NO LONGER a model input (increment 1.4); these
# columns are retained in features.parquet for auditing only. They are still
# filled so the stored frame has no stray NaNs, but nothing reads the fills.
WEATHER_MEAN_FILL = ["air_temp", "track_temp", "humidity", "wind_speed"]
WEATHER_ZERO_FILL = ["rainfall", "is_wet"]


def apply_fill_policy(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    fills: dict[str, float] = {}

    global_mean_finish = float(df["position"].mean())  # ~10.5
    podium_rate = float((df["position"] <= 3).mean())
    dnf_rate = float(df["is_dnf"].mean())

    for c in POSITION_SCALED:
        df[c] = df[c].fillna(df["driver_overall_avg_finish"])
        df[c] = df[c].fillna(global_mean_finish)
        fills[c] = global_mean_finish

    for c in RATE_FEATURES:
        v = podium_rate if c == "driver_circuit_podium_rate" else dnf_rate
        df[c] = df[c].fillna(v)
        fills[c] = v

    for c in ZERO_FILL:
        df[c] = df[c].fillna(0.0)
        fills[c] = 0.0

    for c, v in FLAG_DEFAULTS.items():
        df[c] = df[c].fillna(v).astype(int)
        fills[c] = float(v)

    for c in WEATHER_MEAN_FILL:
        v = float(df[c].mean())
        df[c] = df[c].fillna(v)
        fills[c] = v
    for c in WEATHER_ZERO_FILL:
        df[c] = df[c].fillna(0.0)
        fills[c] = 0.0

    df["grid_position"] = df["grid_position"].fillna(20.0)
    fills["grid_position"] = 20.0

    # helper column used for fills above; also give it a value everywhere so
    # inference-time per-driver fallback behaves identically
    df["driver_overall_avg_finish"] = df["driver_overall_avg_finish"].fillna(global_mean_finish)
    fills["driver_overall_avg_finish"] = global_mean_finish

    return df, fills


def build(raw_path: Path = RAW_PATH, out_dir: Path = DATA_DIR) -> pd.DataFrame:
    df = pd.read_parquet(raw_path)
    df["date"] = pd.to_datetime(df["date"])
    df = sort_frame(df)

    df = add_recent_form(df)
    df = add_weather_affinity(df)   # resets index via merge_asof; stays date-sorted
    df = add_circuit_history(df)
    df = add_reliability(df)
    df = add_teammate_features(df)  # needs form_avg_* columns

    keep = [c for c in ID_COLS if c in df.columns] + FEATURE_COLS + [TARGET]
    # keep the raw quali columns for future v2 work if present
    extra = [c for c in ("quali_best_s", "gap_to_pole_s") if c in df.columns]
    df = df[keep + extra].copy()

    # Pre-fill snapshot for the Gate 2 audit (first-race values must be NaN
    # here, never a real computed stat).
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / PREFILL_PATH.name, index=False)

    df, fills = apply_fill_policy(df)

    # spec 2.5: assert zero NaNs in the feature matrix
    nan_counts = df[FEATURE_COLS].isna().sum()
    bad = nan_counts[nan_counts > 0]
    if len(bad):
        raise AssertionError(f"NaNs remain in feature matrix after fill:\n{bad}")

    df.to_parquet(out_dir / FEATURES_PATH.name, index=False)
    with open(out_dir / FILLS_PATH.name, "w") as f:
        json.dump(fills, f, indent=2)

    log.info("Wrote %s (%s rows, %s features)",
             out_dir / FEATURES_PATH.name, len(df), len(FEATURE_COLS))
    log.info("Wrote %s", out_dir / FILLS_PATH.name)
    return df


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_PATH)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR,
                        help="where features/prefill/fill_values go. Point this at "
                             "a new directory to avoid overwriting artifacts hashed "
                             "in reports/baseline.json.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    df = build(args.raw, args.out_dir)
    print(f"\nfeatures.parquet: {len(df)} rows x {len(FEATURE_COLS)} features; "
          f"target mean = {df[TARGET].mean():.4f}")
    print("Next: python -m src.audit_leakage   (VALIDATION GATE 2 -- mandatory)")


if __name__ == "__main__":
    main()
