"""Teammate-relative form (spec 2.4) -- the headline feature.

For each (driver, race), the teammate value of a metric is the MEAN of the
other same-team drivers' values in that same race (handles N >= 2). Deltas
are computed from each side's PRIOR-form features (form_avg_quali_3,
form_avg_finish_3), never from the current race's result, so nothing about
this race's outcome (or its qualifying) leaks in beyond what grid_position
already carries.

Edge cases (spec 2.4.4): solo entries, or any side lacking prior form ->
delta = 0 and teammate_available = 0. Rows are never dropped.

Requires the recent-form features to exist already (build order: form
features first, then this module).
"""
import numpy as np
import pandas as pd

_PAIRS = {
    # prior-form column        -> output delta column
    "form_avg_quali_3": "quali_gap_to_teammate",
    "form_avg_finish_3": "form_finish_vs_teammate",
}


def add_teammate_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in _PAIRS if c not in df.columns]
    if missing:
        raise KeyError(f"Recent-form features must be built first; missing: {missing}")

    grp = ["year", "round", "team"]
    n = df.groupby(grp)["driver"].transform("size")

    for col, out in _PAIRS.items():
        # sum with skipna=False: if ANY member of the pairing lacks prior
        # form, the delta is undefined for the group -> NaN -> flagged
        # unavailable and neutral-filled below (spec edge-case rule).
        total = df.groupby(grp)[col].transform(lambda s: s.sum(skipna=False))
        teammate_mean = (total - df[col]) / (n - 1)  # solo: 0/0 -> NaN
        df[out] = df[col] - teammate_mean
        df[out] = df[out].replace([np.inf, -np.inf], np.nan)

    available = (
        (n > 1)
        & df["quali_gap_to_teammate"].notna()
        & df["form_finish_vs_teammate"].notna()
    )
    df["teammate_available"] = available.astype(int)
    for out in _PAIRS.values():
        df[out] = df[out].fillna(0.0)
    return df
