"""Practice long-run pace (FIX_PLAN.md section 5.D)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.practice import (MIN_STINT_LAPS, PRACTICE_COLS, add_practice_pace,
                          clean_laps, practice_pace_summary)


class FakeSession:
    def __init__(self, laps: pd.DataFrame):
        self.laps = laps


def lap(driver, stint, compound, secs, n, *, status="1", accurate=True,
        pit_in=pd.NaT, pit_out=pd.NaT, start_age=1):
    return pd.DataFrame({
        "Driver": driver, "Stint": stint, "Compound": compound,
        "LapTime": pd.to_timedelta(secs, unit="s"),
        "TrackStatus": status, "IsAccurate": accurate,
        "PitInTime": pit_in, "PitOutTime": pit_out,
        "TyreLife": np.arange(start_age, start_age + len(secs)),
        "Deleted": False,
    }, index=range(n))


def session_with(*frames) -> FakeSession:
    return FakeSession(pd.concat(frames, ignore_index=True))


FAST = lap("AAA", 1, "SOFT", [90.0, 90.1, 90.0, 90.2, 90.1], 5)
SLOW = lap("BBB", 1, "SOFT", [91.0, 91.1, 91.0, 91.2, 91.1], 5)


# --- filtering --------------------------------------------------------------

def test_safety_car_laps_are_excluded():
    """A lap behind the safety car says more about the safety car than the car."""
    mixed = pd.concat([FAST, lap("AAA", 2, "SOFT", [120.0] * 5, 5, status="4")],
                      ignore_index=True)
    assert (clean_laps(mixed)["TrackStatus"] == "1").all()


def test_pit_laps_are_excluded():
    out_lap = lap("AAA", 2, "SOFT", [95.0] * 5, 5, pit_out=pd.Timestamp("2026-01-01"))
    assert len(clean_laps(pd.concat([FAST, out_lap], ignore_index=True))) == len(FAST)


def test_inaccurate_laps_are_excluded():
    bad = lap("AAA", 2, "SOFT", [89.0] * 5, 5, accurate=False)
    assert len(clean_laps(pd.concat([FAST, bad], ignore_index=True))) == len(FAST)


# --- long runs --------------------------------------------------------------

def test_a_short_stint_is_not_a_long_run():
    """Three laps is a qualifying simulation, not race pace."""
    short = lap("CCC", 1, "SOFT", [88.0] * (MIN_STINT_LAPS - 1), MIN_STINT_LAPS - 1)
    out = practice_pace_summary(session_with(FAST, SLOW, short))
    assert "CCC" not in set(out["driver"])


def test_pace_is_relative_to_the_field_median():
    out = practice_pace_summary(session_with(FAST, SLOW)).set_index("driver")
    assert out.loc["AAA", "practice_pace_pct"] < 0 < out.loc["BBB", "practice_pace_pct"]


def test_the_best_long_run_is_used_not_the_average():
    """A heavy-fuel race simulation and a low-fuel run are both in the data;
    the quickest is the less contaminated estimate (FIX_PLAN section 5.D)."""
    heavy = lap("AAA", 2, "HARD", [95.0] * 6, 6)
    out = practice_pace_summary(session_with(FAST, heavy, SLOW)).set_index("driver")
    reference = practice_pace_summary(session_with(FAST, SLOW)).set_index("driver")
    assert out.loc["AAA", "practice_pace_pct"] == pytest.approx(
        reference.loc["AAA", "practice_pace_pct"])


def test_run_count_records_how_many_long_runs_were_completed():
    heavy = lap("AAA", 2, "HARD", [95.0] * 6, 6)
    out = practice_pace_summary(session_with(FAST, heavy, SLOW)).set_index("driver")
    assert out.loc["AAA", "practice_runs"] == 2
    assert out.loc["BBB", "practice_runs"] == 1


def test_degradation_slope_is_positive_when_the_car_slows():
    degrading = lap("CCC", 1, "SOFT", [90.0, 90.4, 90.8, 91.2, 91.6], 5)
    out = practice_pace_summary(session_with(FAST, SLOW, degrading)).set_index("driver")
    assert out.loc["CCC", "practice_deg_slope"] > 0
    assert out.loc["AAA", "practice_deg_slope"] == pytest.approx(0.0, abs=0.05)


def test_consistency_is_scale_free():
    out = practice_pace_summary(session_with(FAST, SLOW)).set_index("driver")
    assert 0 <= out.loc["AAA", "practice_sd_pct"] < 1


def test_a_session_with_no_long_runs_returns_empty():
    short = lap("AAA", 1, "SOFT", [90.0, 90.1], 2)
    assert practice_pace_summary(FakeSession(short)).empty


def test_an_empty_session_returns_empty():
    assert practice_pace_summary(FakeSession(pd.DataFrame())).empty


# --- merging ----------------------------------------------------------------

def test_missing_practice_leaves_the_columns_nan():
    """Absence is informative -- a driver with no long run did not complete one
    -- so these must reach the model as NaN, not as an invented value."""
    frame = pd.DataFrame({"year": 2026, "round": 1, "driver": ["AAA", "BBB"]})
    out = add_practice_pace(frame, None)
    for column in PRACTICE_COLS:
        assert out[column].isna().all()


def test_practice_merges_on_race_and_driver():
    frame = pd.DataFrame({"year": [2026, 2026], "round": [1, 1],
                          "driver": ["AAA", "BBB"]})
    practice = pd.DataFrame({"year": [2026], "round": [1], "driver": ["AAA"],
                             "practice_pace_pct": [-0.5], "practice_laps": [8.0],
                             "practice_sd_pct": [0.2], "practice_deg_slope": [0.01],
                             "practice_runs": [2.0]})
    out = add_practice_pace(frame, practice).set_index("driver")
    assert out.loc["AAA", "practice_pace_pct"] == -0.5
    assert np.isnan(out.loc["BBB", "practice_pace_pct"])


def test_a_driver_absent_from_practice_is_not_given_another_drivers_pace():
    frame = pd.DataFrame({"year": [2026] * 2, "round": [1, 2], "driver": ["AAA"] * 2})
    practice = pd.DataFrame({"year": [2026], "round": [1], "driver": ["AAA"],
                             "practice_pace_pct": [-0.5], "practice_laps": [8.0],
                             "practice_sd_pct": [0.2], "practice_deg_slope": [0.01],
                             "practice_runs": [2.0]})
    out = add_practice_pace(frame, practice)
    assert out.loc[0, "practice_pace_pct"] == -0.5
    assert np.isnan(out.loc[1, "practice_pace_pct"])
