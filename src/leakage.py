"""Leakage-safe historical aggregations (spec section 2, core mechanism).

Every helper here implements the canonical pattern:

    sort by date  ->  group  ->  shift(1)  ->  expanding/rolling stat

The shift(1) is non-negotiable: it guarantees the current race's outcome
never enters its own features. All feature modules must go through these
helpers (or the subset-then-asof-merge pattern in weather.py, which achieves
the same exclusion via allow_exact_matches=False).

Precondition: the frame must already be sorted by date ascending. Every
helper asserts this instead of silently re-sorting, so an unsorted frame
fails loudly rather than producing subtly wrong (leaky) features.
"""
import pandas as pd


def sort_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical sort for the whole project: date asc, stable tiebreak."""
    return df.sort_values(["date", "year", "round"], kind="mergesort").reset_index(drop=True)


def assert_sorted(df: pd.DataFrame) -> None:
    if not df["date"].is_monotonic_increasing:
        raise ValueError(
            "Frame is not sorted by date ascending. Call leakage.sort_frame() "
            "before computing historical features."
        )


def past_expanding_mean(df, group_cols, value_col):
    """Expanding mean of value_col over each group's PRIOR races only."""
    assert_sorted(df)
    return (
        df.groupby(group_cols, sort=False)[value_col]
          .transform(lambda s: s.shift(1).expanding().mean())
    )


def past_rolling_mean(df, group_cols, value_col, window):
    """Rolling mean over the last `window` PRIOR races (min_periods=1)."""
    assert_sorted(df)
    return (
        df.groupby(group_cols, sort=False)[value_col]
          .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )


def past_expanding_min(df, group_cols, value_col):
    """Expanding min of value_col over each group's PRIOR races only."""
    assert_sorted(df)
    return (
        df.groupby(group_cols, sort=False)[value_col]
          .transform(lambda s: s.shift(1).expanding().min())
    )


def past_cumsum(df, group_cols, value_col):
    """Sum of value_col over each group's PRIOR races only."""
    assert_sorted(df)
    return (
        df.groupby(group_cols, sort=False)[value_col]
          .transform(lambda s: s.cumsum().shift(1))
    )


def past_count(df, group_cols):
    """Number of PRIOR rows in the group (cumcount = strictly-before count)."""
    assert_sorted(df)
    return df.groupby(group_cols, sort=False).cumcount()


def past_mean_excluding_current_race(df, group_cols, value_col):
    """Expanding mean over all rows from PRIOR RACES in the group.

    Required whenever a group has multiple rows in the same race (e.g. team
    features: a team has two drivers per race). Plain shift(1) only excludes
    the current ROW, so the second same-team row would see its teammate's
    outcome from the CURRENT race -- leakage. This helper aggregates to one
    row per (group, race date) first, then shifts at race level, so the
    entire current race is excluded.
    """
    assert_sorted(df)
    keys = list(group_cols) if isinstance(group_cols, (list, tuple)) else [group_cols]

    per_race = (df.groupby(keys + ["date"], as_index=False)
                  .agg(_s=(value_col, "sum"), _c=(value_col, "count"))
                  .sort_values("date", kind="mergesort"))
    g = per_race.groupby(keys, sort=False)
    per_race["_cs"] = g["_s"].transform(lambda s: s.cumsum().shift(1))
    per_race["_cc"] = g["_c"].transform(lambda s: s.cumsum().shift(1))
    per_race["_val"] = per_race["_cs"] / per_race["_cc"]  # 0/0 -> NaN (no history)

    merged = df[keys + ["date"]].merge(
        per_race[keys + ["date", "_val"]], on=keys + ["date"], how="left")
    return pd.Series(merged["_val"].to_numpy(), index=df.index)
