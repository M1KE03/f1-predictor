"""Weather affinity features (spec 2.2).

Mechanism for subset stats (wet races, temperature bins): compute the
running stat *including* each subset race, then merge back onto the full
frame with merge_asof(direction='backward', allow_exact_matches=False).

Because exact date matches are excluded, a race only ever sees subset stats
from strictly earlier dates:
  - the wet race itself sees the stat as of the PREVIOUS wet race
    (i.e. its own pre-race value -- its own result is excluded), and
  - any later race (wet or dry) sees the stat including that wet race.
This is equivalent to the shift(1) pattern, adapted to sparse subsets, and
avoids the classic bug where a shifted subset stat goes stale for the rows
in between subset races.
"""
import numpy as np
import pandas as pd

from .leakage import assert_sorted, past_expanding_mean

TEMP_BIN_LABELS = ("cool", "medium", "hot")


def temp_bin_series(track_temp: pd.Series) -> pd.Series:
    """cool (<30), medium (30-45 inclusive), hot (>45). NaN stays NaN."""
    out = pd.Series(
        np.where(track_temp < 30, "cool", np.where(track_temp > 45, "hot", "medium")),
        index=track_temp.index, dtype=object,
    )
    out[track_temp.isna()] = np.nan
    return out


def _asof_subset_stats(df: pd.DataFrame, mask: pd.Series, prefix: str) -> pd.DataFrame:
    """Attach prior-only expanding mean finish + prior count for a row subset.

    Adds columns: {prefix}_avg (NaN if no prior subset race for the driver)
                  {prefix}_n   (0 if no prior subset race for the driver)
    """
    assert_sorted(df)
    sub = df.loc[mask, ["driver", "date", "position"]].sort_values(
        "date", kind="mergesort").copy()

    if len(sub) == 0:
        df[f"{prefix}_avg"] = np.nan
        df[f"{prefix}_n"] = 0.0
        return df

    # running stats INCLUDING the subset race itself ...
    sub[f"{prefix}_avg"] = (
        sub.groupby("driver")["position"].transform(lambda s: s.expanding().mean())
    )
    sub[f"{prefix}_n"] = sub.groupby("driver").cumcount() + 1

    # ... exposed to later races only, via strict-backward asof
    merged = pd.merge_asof(
        df,
        sub[["driver", "date", f"{prefix}_avg", f"{prefix}_n"]],
        on="date", by="driver",
        direction="backward",
        allow_exact_matches=False,  # strictly earlier dates only
    )
    merged[f"{prefix}_n"] = merged[f"{prefix}_n"].fillna(0.0)
    return merged


def add_weather_affinity(df: pd.DataFrame) -> pd.DataFrame:
    assert_sorted(df)

    # Reused below and by the fill policy (spec 2.2 / 2.5). Intermediate:
    # deliberately NOT in FEATURE_COLS (spec 3).
    df["driver_overall_avg_finish"] = past_expanding_mean(df, "driver", "position")

    # --- wet/dry affinity -------------------------------------------------
    df = _asof_subset_stats(df, df["is_wet"] == 1, "wet")
    df = df.rename(columns={"wet_avg": "driver_wet_avg", "wet_n": "driver_wet_n"})
    # negative = better in the wet than overall
    df["driver_wet_delta"] = df["driver_wet_avg"] - df["driver_overall_avg_finish"]

    # --- temperature-bin affinity ----------------------------------------
    # Caveat honoured (spec 2.2): confounded with circuit identity; keep the
    # _n count so the model can discount thin samples. No manual weighting.
    df["temp_bin"] = temp_bin_series(df["track_temp"])
    for b in TEMP_BIN_LABELS:
        df = _asof_subset_stats(df, df["temp_bin"] == b, f"tb_{b}")

    cond = [df["temp_bin"] == b for b in TEMP_BIN_LABELS]
    df["driver_temp_bin_avg"] = np.select(
        cond, [df[f"tb_{b}_avg"] for b in TEMP_BIN_LABELS], default=np.nan)
    df["driver_temp_bin_n"] = np.select(
        cond, [df[f"tb_{b}_n"] for b in TEMP_BIN_LABELS], default=0.0)

    df = df.drop(columns=[c for b in TEMP_BIN_LABELS
                          for c in (f"tb_{b}_avg", f"tb_{b}_n")] + ["temp_bin"])
    return df
