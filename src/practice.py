"""Practice long-run pace (FIX_PLAN.md section 5.D).

The one source of genuinely new information left. Every other feature either
describes history or re-expresses qualifying order:

- `grid_position` IS the qualifying result, and increment 3.1 showed qualifying
  pace adds nothing beyond it.
- Race-pace ratings measure how fast a car WAS, aggregated over prior races.

Practice pace measures how fast a car is THIS WEEKEND over a long run, which is
the closest observable proxy for race pace and is available before the forecast
cutoff. Nothing in the feature set currently carries it.

What is measured
----------------
The best long run: the stint with the lowest median lap time among stints of at
least four clean consecutive laps on one compound. Expressed as a percentage of
the field's median long run, so it is comparable across circuits.

Also kept: stint length, consistency, tyre-degradation slope (seconds per lap of
tyre age, from a fit within the stint), and how many qualifying long runs the
driver managed.

What cannot be measured
-----------------------
**Fuel load and run plans are not observable.** A team doing a heavy-fuel race
simulation looks slow; one doing a low-fuel qualifying simulation looks quick.
FIX_PLAN.md section 5.D is explicit that these are noisy pace estimates and
should be kept only if they improve chronological validation. Taking the BEST
long run rather than the average reduces but does not remove the effect.

Green-flag, accurate, non-pit, non-deleted laps only, as for race pace.
"""
from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

# A "long run" needs enough laps to be about pace rather than a single lap.
MIN_STINT_LAPS: Final = 4

# Preference order. FP2 is traditionally the long-run session; sprint weekends
# have only FP1, and FP3 is the fallback when FP2 is disrupted.
SESSION_PREFERENCE: Final = ("FP2", "FP3", "FP1")

PRACTICE_COLS: Final = (
    "practice_pace_pct", "practice_laps", "practice_sd_pct",
    "practice_deg_slope", "practice_runs",
)


def clean_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Green-flag, accurate, non-pit, non-deleted laps with a usable time."""
    frame = laps.copy()
    seconds = pd.to_timedelta(frame["LapTime"], errors="coerce").dt.total_seconds()
    keep = (
        (frame["TrackStatus"].astype(str) == "1")
        & (frame["IsAccurate"] == True)          # noqa: E712 -- FastF1 flag
        & seconds.notna()
        & frame["PitInTime"].isna()
        & frame["PitOutTime"].isna()
    )
    if "Deleted" in frame.columns:
        keep &= frame["Deleted"] != True         # noqa: E712
    return frame[keep].assign(_secs=seconds[keep])


def _degradation_slope(stint: pd.DataFrame) -> float:
    """Seconds gained per lap of tyre age, fitted within one stint.

    Positive means the car slows as the tyre ages. Requires TyreLife to vary,
    which it does within a real stint.
    """
    if "TyreLife" not in stint.columns:
        return float("nan")
    age = pd.to_numeric(stint["TyreLife"], errors="coerce")
    valid = age.notna() & stint["_secs"].notna()
    if valid.sum() < MIN_STINT_LAPS or age[valid].nunique() < 2:
        return float("nan")
    slope, _ = np.polyfit(age[valid].to_numpy(float), stint.loc[valid, "_secs"].to_numpy(float), 1)
    return float(slope)


def practice_pace_summary(session) -> pd.DataFrame:
    """Per-driver best long run from one practice session."""
    empty = pd.DataFrame(columns=["driver", *PRACTICE_COLS])
    laps = getattr(session, "laps", None)
    if laps is None or len(laps) == 0:
        return empty

    good = clean_laps(laps)
    if good.empty:
        return empty

    rows = []
    for (driver, _stint, _compound), stint in good.groupby(
            ["Driver", "Stint", "Compound"], dropna=False):
        if len(stint) < MIN_STINT_LAPS:
            continue
        rows.append({
            "driver": str(driver),
            "median_s": float(stint["_secs"].median()),
            "laps": int(len(stint)),
            "sd_s": float(stint["_secs"].std()),
            "deg": _degradation_slope(stint),
        })
    if not rows:
        return empty

    stints = pd.DataFrame(rows)
    runs_per_driver = stints.groupby("driver").size()
    # The BEST long run, not the average: a heavy-fuel race simulation and a
    # low-fuel run are both in here, and the quickest is the less contaminated
    # estimate of underlying pace.
    best = stints.sort_values("median_s").groupby("driver", as_index=False).head(1)

    field_median = float(best["median_s"].median())
    if not np.isfinite(field_median) or field_median <= 0:
        return empty

    return pd.DataFrame({
        "driver": best["driver"].to_numpy(),
        "practice_pace_pct": 100.0 * (best["median_s"] / field_median - 1.0),
        "practice_laps": best["laps"].astype(float).to_numpy(),
        "practice_sd_pct": (100.0 * best["sd_s"] / best["median_s"]).to_numpy(),
        "practice_deg_slope": best["deg"].to_numpy(),
        "practice_runs": best["driver"].map(runs_per_driver).astype(float).to_numpy(),
    })


def add_practice_pace(df: pd.DataFrame, practice: pd.DataFrame | None) -> pd.DataFrame:
    """Merge per-weekend practice pace onto the feature frame.

    Current-weekend measurement, so it needs no historical aggregation -- and,
    like the qualifying features, its absence is informative: a driver with no
    long run did not complete one.
    """
    out = df.copy()
    if practice is None or practice.empty:
        for column in PRACTICE_COLS:
            out[column] = np.nan
        return out

    merged = out.merge(practice, on=["year", "round", "driver"], how="left",
                       suffixes=("", "_practice"))
    for column in PRACTICE_COLS:
        out[column] = merged[column].to_numpy()
    return out


NATIVE_MISSING: Final = PRACTICE_COLS
