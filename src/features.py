"""The single as-of feature path (FIX_PLAN.md section 2, P0-3 and section 5.A.8).

`build_features.build()` and `predict.predict()` previously each assembled a
feature vector, and they disagreed. Any two-path design drifts, and this one
already had: the trainer used a driver-prior fallback the server did not.

Both now call `build_asof_features()`. Training passes history alone; inference
passes history plus placeholder rows for a race that has not run. The
computation is identical either way, which is what makes the replay test in
`tests/test_parity.py` meaningful -- it asserts that hiding a race's outcome
does not change the feature vector the model sees for it.

`add_recent_form` and `add_reliability` live here rather than in
`build_features.py` so that the inference path does not have to import the
training orchestrator to reach them.
"""
from __future__ import annotations

import pandas as pd

from .circuit import add_circuit_history
from .leakage import (past_expanding_mean, past_mean_excluding_current_race,
                      past_rolling_mean, sort_frame)
from .practice import add_practice_pace
from .preprocessing import FillPolicy
from .qualifying import add_qualifying_pace
from .ratings import add_pace_ratings
from .teammate import add_teammate_features
from .weather import add_weather_affinity

# Outcome columns a not-yet-run race cannot have. Placeholder rows carry these
# as NaN/neutral; the leakage-safe helpers never read a row's own outcome, so
# their presence or absence must not change that row's features.
OUTCOME_COLS = ("position", "status", "points", "is_dnf", "classified",
                "finished_top10", "result_order", "officially_classified",
                "started", "finished", "is_winner", "is_podium")


def add_recent_form(df: pd.DataFrame) -> pd.DataFrame:
    """Spec 2.1 -- rolling driver form over PRIOR races only."""
    df["form_avg_finish_3"] = past_rolling_mean(df, "driver", "position", 3)
    df["form_avg_points_3"] = past_rolling_mean(df, "driver", "points", 3)
    # NOTE misleading name (FIX_PLAN.md section 2, P1): this averages prior
    # STARTING GRID positions, not qualifying pace. Renaming it is a separate
    # increment because it changes the stored schema.
    df["form_avg_quali_3"] = past_rolling_mean(df, "driver", "grid_position", 3)
    df["form_dnf_rate_5"] = past_rolling_mean(df, "driver", "is_dnf", 5)
    df["season_avg_finish"] = past_expanding_mean(df, ["driver", "year"], "position")
    # negative = currently overperforming the season baseline
    df["momentum"] = df["form_avg_finish_3"] - df["season_avg_finish"]
    return df


def add_reliability(df: pd.DataFrame) -> pd.DataFrame:
    """Reliability and constructor-points prior, PRIOR races only."""
    df["driver_dnf_rate"] = past_expanding_mean(df, "driver", "is_dnf")
    # A team has two rows per race, so plain shift(1) would leak the teammate's
    # current-race outcome. See leakage.past_mean_excluding_current_race.
    df["team_dnf_rate"] = past_mean_excluding_current_race(df, "team", "is_dnf")

    # Constructor championship points strictly BEFORE this race, per season.
    # NOTE this is accumulated points, not a standings rank, and omits sprint
    # points (FIX_PLAN.md section 2, P1).
    team_points = (df.groupby(["year", "round", "team"], as_index=False)
                     .agg(date=("date", "first"), team_pts=("points", "sum"))
                     .sort_values("date", kind="mergesort"))
    team_points["constructor_standing_prior"] = (
        team_points.groupby(["team", "year"])["team_pts"]
                   .transform(lambda s: s.cumsum().shift(1))
    )
    return df.merge(
        team_points[["year", "round", "team", "constructor_standing_prior"]],
        on=["year", "round", "team"], how="left")


def compute_features(df: pd.DataFrame,
                     practice: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every historical feature, no imputation. Shared by both paths.

    Order is load-bearing: teammate deltas are built from the recent-form
    columns, so form must exist first.
    """
    df = sort_frame(df)
    df = add_recent_form(df)
    df = add_weather_affinity(df)   # resets index via merge_asof; stays sorted
    df = add_circuit_history(df)
    df = add_reliability(df)
    df = add_pace_ratings(df)      # recency-weighted race pace, PRIOR races
    df = add_teammate_features(df)
    # Current-weekend pace. Purely within-race, so it neither reads nor
    # affects any historical aggregate above it.
    df = add_qualifying_pace(df)
    # Current-weekend practice long runs, merged per (race, driver).
    df = add_practice_pace(df, practice)
    return df


def build_asof_features(history: pd.DataFrame,
                        upcoming: pd.DataFrame | None = None,
                        policy: FillPolicy | None = None,
                        practice: pd.DataFrame | None = None) -> pd.DataFrame:
    """Features for `history` (+ optional not-yet-run `upcoming` rows).

    Args:
        history: rows with known outcomes, one per (driver, completed race).
        upcoming: placeholder rows for a race that has not run. Their outcome
            columns must be NaN/neutral; nothing reads a row's own outcome.
        policy: a FITTED FillPolicy. Omit to get the pre-imputation frame, which
            is what the leakage audit inspects.

    Returns the combined frame, date-sorted. Callers select the rows they want.
    """
    if upcoming is not None and len(upcoming):
        overlap = history.merge(upcoming[["year", "round"]].drop_duplicates(),
                                on=["year", "round"], how="inner")
        if len(overlap):
            raise ValueError(
                "`upcoming` names a race that already exists in `history`. "
                "Rebuild from the real result instead of forecasting it.")
        combined = pd.concat([history, upcoming], ignore_index=True)
    else:
        combined = history.copy()

    combined = compute_features(combined, practice)
    if policy is not None:
        combined = policy.transform(combined)
    return combined
