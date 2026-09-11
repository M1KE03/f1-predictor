"""Recency-weighted pace ratings (FIX_PLAN.md section 5.C).

`race_pace_pct` describes a COMPLETED race, so every check here is really a
leakage check: the rating for race N must depend only on races 1..N-1, and for
a team on races 1..N-1 entirely -- not on the teammate's row in race N.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.leakage import sort_frame
from src.ratings import PACE_FEATURES, add_pace_ratings


def build(pace_by_race: dict[str, list[float]], teams: dict[str, str] | None = None,
          n: int | None = None) -> pd.DataFrame:
    """One row per (driver, race); `pace_by_race` maps driver -> pace per race."""
    teams = teams or {d: "solo_" + d for d in pace_by_race}
    n = n or len(next(iter(pace_by_race.values())))
    rows = []
    for rnd in range(1, n + 1):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(14 * rnd, unit="D")
        for driver, series in pace_by_race.items():
            rows.append(dict(year=2024, round=rnd, date=date, driver=driver,
                             team=teams[driver], race_pace_pct=series[rnd - 1],
                             race_pace_sd_pct=0.5))
    return sort_frame(pd.DataFrame(rows))


# --- leakage ----------------------------------------------------------------

def test_first_race_has_no_rating():
    """Nothing to learn from yet -- and 0.0 would read as 'exactly average car'."""
    out = add_pace_ratings(build({"A": [-1.0, -1.0, -1.0]}))
    first = out[out["round"] == 1].iloc[0]
    for column in PACE_FEATURES:
        assert pd.isna(first[column]), column


def test_rating_ignores_the_current_race():
    """A driver quick in races 1-2 and catastrophic in race 3 must still carry
    a quick rating INTO race 3."""
    out = add_pace_ratings(build({"A": [-1.0, -1.0, +5.0]})).set_index("round")
    assert out.loc[3, "driver_pace_ewm_3"] == pytest.approx(-1.0)


def test_changing_a_race_never_changes_an_earlier_rating():
    base = build({"A": [-1.0, -0.5, 0.0, 0.5]})
    tampered = base.copy()
    tampered.loc[tampered["round"] == 4, "race_pace_pct"] = -99.0

    a = add_pace_ratings(base).set_index("round")["driver_pace_ewm_3"]
    b = add_pace_ratings(tampered).set_index("round")["driver_pace_ewm_3"]
    pd.testing.assert_series_equal(a, b)


def test_team_rating_excludes_the_whole_current_race():
    """The classic trap: a team has two rows per race, so shift(1) alone would
    let the second driver see their teammate's current-race pace."""
    frame = build({"A": [-1.0, -1.0, -1.0], "B": [-1.0, -1.0, -9.0]},
                  teams={"A": "red", "B": "red"})
    out = add_pace_ratings(frame)
    race3 = out[out["round"] == 3].set_index("driver")
    # B's -9.0 in race 3 must not reach A's race-3 team rating.
    assert race3.loc["A", "team_pace_ewm_3"] == pytest.approx(-1.0)
    assert race3.loc["B", "team_pace_ewm_3"] == pytest.approx(-1.0)


# --- recency ----------------------------------------------------------------

def test_recent_races_dominate_the_short_rating():
    """An upgrade mid-season must show up quickly, which an expanding mean
    would dilute with stale evidence."""
    out = add_pace_ratings(
        build({"A": [2.0, 2.0, 2.0, 2.0, -2.0, -2.0]})).set_index("round")
    expanding_mean = np.mean([2.0, 2.0, 2.0, 2.0, -2.0])
    assert out.loc[6, "driver_pace_ewm_3"] < expanding_mean


def test_short_window_reacts_faster_than_long():
    out = add_pace_ratings(
        build({"A": [2.0] * 6 + [-2.0] * 2})).set_index("round")
    assert out.loc[8, "driver_pace_ewm_3"] < out.loc[8, "driver_pace_ewm_12"]


def test_trend_is_negative_when_improving():
    """Short minus long: negative means quicker than the medium-term level."""
    out = add_pace_ratings(
        build({"A": [2.0] * 6 + [-2.0] * 2})).set_index("round")
    assert out.loc[8, "driver_pace_trend"] < 0


def test_trend_is_zero_for_a_steady_car():
    out = add_pace_ratings(build({"A": [1.0] * 8})).set_index("round")
    assert out.loc[8, "driver_pace_trend"] == pytest.approx(0.0, abs=1e-9)


# --- driver vs car ----------------------------------------------------------

def test_driver_vs_team_isolates_the_driver():
    """Both drivers in the same car; A is consistently quicker. A's
    driver_pace_vs_team must be negative and B's positive."""
    frame = build({"A": [-1.0] * 5, "B": [1.0] * 5}, teams={"A": "red", "B": "red"})
    out = add_pace_ratings(frame)
    last = out[out["round"] == 5].set_index("driver")
    assert last.loc["A", "driver_pace_vs_team"] < 0 < last.loc["B", "driver_pace_vs_team"]


def test_two_equal_drivers_have_no_gap_to_their_car():
    frame = build({"A": [-1.0] * 5, "B": [-1.0] * 5}, teams={"A": "red", "B": "red"})
    out = add_pace_ratings(frame)
    last = out[out["round"] == 5].set_index("driver")
    assert last.loc["A", "driver_pace_vs_team"] == pytest.approx(0.0, abs=1e-9)


# --- sample size ------------------------------------------------------------

def test_prior_race_count_excludes_the_current_race():
    out = add_pace_ratings(build({"A": [-1.0] * 4})).set_index("round")
    assert list(out["driver_pace_races"]) == [0.0, 1.0, 2.0, 3.0]


def test_races_without_a_pace_reading_are_not_counted():
    frame = build({"A": [-1.0, np.nan, -1.0, -1.0]})
    out = add_pace_ratings(frame).set_index("round")
    assert out.loc[4, "driver_pace_races"] == 2.0


def test_missing_pace_does_not_poison_the_rating():
    """A race with no usable lap data must be skipped, not read as 0% pace."""
    frame = build({"A": [-2.0, np.nan, -2.0]})
    out = add_pace_ratings(frame).set_index("round")
    assert out.loc[3, "driver_pace_ewm_3"] == pytest.approx(-2.0)


# --- schema safety ----------------------------------------------------------

def test_a_frame_without_race_pace_gets_nan_columns():
    """A dataset ingested before race pace existed must surface at the feature
    contract, not deep inside LightGBM."""
    frame = build({"A": [-1.0, -1.0]}).drop(columns=["race_pace_pct"])
    out = add_pace_ratings(frame)
    for column in PACE_FEATURES:
        assert column in out.columns
        assert out[column].isna().all()
