"""Recency-weighted driver and car form (FIX_PLAN.md section 5.C).

Two gaps this closes.

**Nothing measured race pace.** The feature set knew where a car started
(`grid_position`) and where it historically finished, but not how fast it
actually was. Increment 3.1 established that qualifying pace adds nothing
beyond qualifying order, because `grid_position` already IS the qualifying
order. Race pace is a different measurement: it reflects tyre degradation,
long-run balance and true car performance rather than one low-fuel lap.

**Nothing was recency-weighted.** Circuit and reliability statistics use every
race ever, equally. For driver identity that is defensible; for car performance
it is not -- a team slow in March may be quick by September, and an expanding
mean dilutes the recent evidence with stale seasons. Every rating here is
exponentially weighted with a half-life in races.

All aggregation routes through `src.leakage`, so the current race is excluded
by construction. `race_pace_pct` describes a COMPLETED race and is only ever
read from PRIOR ones.
"""
from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from .leakage import (assert_sorted, past_count, past_ewm_mean,
                      past_ewm_mean_excluding_current_race)

# Half-lives in races. FIX_PLAN.md section 5.C suggests trying 3, 6 and 12;
# a short and a medium window are kept so the model can pick between reacting
# to an upgrade and averaging out a single bad weekend.
SHORT_HALF_LIFE: Final = 3
LONG_HALF_LIFE: Final = 12

PACE = "race_pace_pct"


def add_pace_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Recency-weighted race-pace ratings for driver, team, and their gap."""
    assert_sorted(df)
    out = df

    if PACE not in out.columns:
        # Dataset ingested before race pace existed. Emit the columns as NaN
        # rather than silently omitting them, so a schema mismatch surfaces at
        # the feature contract instead of deep inside LightGBM.
        for column in PACE_FEATURES:
            out[column] = np.nan
        out["driver_pace_races"] = 0.0
        return out

    # --- driver ---------------------------------------------------------
    out["driver_pace_ewm_3"] = past_ewm_mean(out, "driver", PACE, SHORT_HALF_LIFE)
    out["driver_pace_ewm_12"] = past_ewm_mean(out, "driver", PACE, LONG_HALF_LIFE)
    # Short minus long: negative means the driver is currently quicker than
    # their own medium-term level, i.e. genuine form rather than standing.
    out["driver_pace_trend"] = out["driver_pace_ewm_3"] - out["driver_pace_ewm_12"]

    # --- team -----------------------------------------------------------
    # A team has two rows per race, so the whole current race must be excluded,
    # not merely the current row.
    out["team_pace_ewm_3"] = past_ewm_mean_excluding_current_race(
        out, "team", PACE, SHORT_HALF_LIFE)
    out["team_pace_ewm_12"] = past_ewm_mean_excluding_current_race(
        out, "team", PACE, LONG_HALF_LIFE)
    out["team_pace_trend"] = out["team_pace_ewm_3"] - out["team_pace_ewm_12"]

    # --- driver relative to their own car --------------------------------
    # Separates the driver from the machinery: how much of the car's pace is
    # this particular driver extracting? A transfer should not carry the old
    # car's speed with it, but it may carry this.
    out["driver_pace_vs_team"] = out["driver_pace_ewm_3"] - out["team_pace_ewm_3"]

    # --- consistency and sample size -------------------------------------
    if "race_pace_sd_pct" in out.columns:
        out["driver_pace_sd_ewm"] = past_ewm_mean(
            out, "driver", "race_pace_sd_pct", SHORT_HALF_LIFE)
    else:
        out["driver_pace_sd_ewm"] = np.nan

    # Prior races with a usable pace reading, so the model can discount a
    # rating built from one weekend.
    # Cast to float before shifting: shifting a bool column yields object dtype,
    # and filling that triggers pandas' deprecated downcasting.
    has_pace = out[PACE].notna().astype(float)
    out["driver_pace_races"] = (
        has_pace.groupby(out["driver"]).transform(
            lambda s: s.shift(1).fillna(0.0).cumsum()))
    return out


# Ratings are left NaN where there is no history: unlike a finishing position,
# there is no neutral pace value, and 0.0 would read as "exactly average car".
PACE_FEATURES: Final = (
    "driver_pace_ewm_3", "driver_pace_ewm_12", "driver_pace_trend",
    "team_pace_ewm_3", "team_pace_ewm_12", "team_pace_trend",
    "driver_pace_vs_team", "driver_pace_sd_ewm",
)

NATIVE_MISSING: Final = PACE_FEATURES
