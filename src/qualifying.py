"""Current-weekend qualifying pace (FIX_PLAN.md section 5.B).

`quali_best_s` and `gap_to_pole_s` have been ingested since the beginning, are
~99% populated, and were never in `FEATURE_COLS`. For a forecast made after
qualifying that is the most obviously informative input in the dataset, and its
absence is why the models have nothing to say about pace that `grid_position`
does not already say.

Everything here is measured at the qualifying session, which finishes before
the forecast cutoff, so none of it is leakage.

Why per-segment normalisation
-----------------------------
The minimum across Q1/Q2/Q3 is not comparable between drivers. Q3 runs on
fresher tyres, a clearer track and less fuel than Q1, so the fastest lap of a
Q1-eliminated driver is being compared against another driver's Q3 lap. Every
gap here is therefore computed WITHIN a segment:

    gap_pct = 100 * (driver_time / segment_best - 1)

which is also circuit-independent: 0.5% is 0.5% at Monaco and at Spa, whereas
half a second means very different things at the two.

Missing is informative
----------------------
A driver with no Q3 time did not reach Q3. That is a fact about their pace, not
an absent measurement, and imputing it would erase the signal. These columns are
declared native-missing so LightGBM can learn a split on the absence itself.
`quali_stage_reached` encodes the same fact ordinally for models that cannot.
"""
from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

SEGMENTS: Final = ("q1", "q2", "q3")
RACE_KEYS: Final = ["year", "round"]

# Features whose NaN is meaningful and must reach the model unimputed.
NATIVE_MISSING: Final = (
    "q1_gap_pct", "q2_gap_pct", "q3_gap_pct",
    "quali_gap_pct", "quali_gap_to_median_pct", "quali_pace_vs_teammate_pct",
)


def _segment_gap_pct(df: pd.DataFrame, segment: str) -> pd.Series:
    """Percentage off the best time SET IN THAT SEGMENT, per race."""
    column = f"{segment}_s"
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index)
    times = pd.to_numeric(df[column], errors="coerce")
    best = times.groupby([df["year"], df["round"]]).transform("min")
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100.0 * (times / best - 1.0)


def add_qualifying_pace(df: pd.DataFrame) -> pd.DataFrame:
    """Add the current-weekend qualifying features.

    Requires `q1_s` / `q2_s` / `q3_s` from ingest. When they are absent (a
    dataset ingested before this existed) every feature is NaN, which the
    native-missing policy passes through rather than silently filling.
    """
    out = df.copy()

    for segment in SEGMENTS:
        out[f"{segment}_gap_pct"] = _segment_gap_pct(out, segment)

    have = {s: f"{s}_s" in out.columns for s in SEGMENTS}
    set_time = {s: (pd.to_numeric(out[f"{s}_s"], errors="coerce").notna()
                    if have[s] else pd.Series(False, index=out.index))
                for s in SEGMENTS}

    # 3 = reached Q3, 2 = eliminated in Q2, 1 = eliminated in Q1, 0 = no time.
    out["quali_stage_reached"] = (
        np.where(set_time["q3"], 3,
                 np.where(set_time["q2"], 2,
                          np.where(set_time["q1"], 1, 0))).astype(float))

    # Headline pace: the gap in the DEEPEST segment the driver reached, so a
    # Q3 runner is judged on their Q3 lap and a Q1 casualty on their Q1 lap --
    # each against the drivers they actually shared track conditions with.
    deepest = pd.Series(np.nan, index=out.index)
    for segment in SEGMENTS:                       # q1, then q2, then q3 wins
        gap = out[f"{segment}_gap_pct"]
        deepest = gap.where(gap.notna(), deepest)
    out["quali_gap_pct"] = deepest

    # Gap to the MEDIAN of the same segment. Robust where gap-to-best is
    # distorted by one exceptional lap, and it rescales when the whole field is
    # close together.
    median_gap = pd.Series(np.nan, index=out.index)
    for segment in SEGMENTS:
        gap = out[f"{segment}_gap_pct"]
        median = gap.groupby([out["year"], out["round"]]).transform("median")
        median_gap = (gap - median).where(gap.notna(), median_gap)
    out["quali_gap_to_median_pct"] = median_gap

    # Teammate pace on a SHARED segment. Unlike `quali_gap_to_teammate` -- which
    # compares averages of prior STARTING GRIDS -- this is current lap time
    # against the one driver with the same car.
    out["quali_pace_vs_teammate_pct"] = _teammate_gap(out)

    # Wet or disrupted qualifying: the field spreads far more than usual, which
    # tells the model these gaps mean less. Computed from the gaps themselves,
    # so it needs no weather input.
    spread = out.groupby(RACE_KEYS)["q1_gap_pct"].transform(
        lambda s: s.quantile(0.9) - s.quantile(0.1))
    out["quali_field_spread_pct"] = spread.fillna(0.0)

    out["quali_no_time"] = (out["quali_stage_reached"] == 0).astype(int)
    return out


def _teammate_gap(df: pd.DataFrame) -> pd.Series:
    """Percentage gap to the teammate, on the deepest segment BOTH reached.

    A comparison across different segments would measure track evolution rather
    than the drivers, so a pair with no shared segment gets NaN.
    """
    result = pd.Series(np.nan, index=df.index)
    if "team" not in df.columns:
        return result

    for segment in SEGMENTS:
        column = f"{segment}_s"
        if column not in df.columns:
            continue
        times = pd.to_numeric(df[column], errors="coerce")
        group = df.groupby(RACE_KEYS + ["team"])[column]
        size = group.transform("size")
        total = times.groupby([df["year"], df["round"], df["team"]]).transform(
            lambda s: s.sum(skipna=False))
        # Mean of the other same-team drivers; NaN if any of them has no time.
        others = (total - times) / (size - 1)
        gap = 100.0 * (times / others - 1.0)
        gap = gap.replace([np.inf, -np.inf], np.nan)
        result = gap.where(gap.notna(), result)     # deeper segment wins
    return result


QUALIFYING_FEATURES: Final = (
    "quali_gap_pct", "quali_gap_to_median_pct", "quali_pace_vs_teammate_pct",
    "quali_stage_reached", "quali_field_spread_pct", "quali_no_time",
    "q1_gap_pct",
)
