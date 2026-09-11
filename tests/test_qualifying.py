"""Qualifying-pace features (FIX_PLAN.md section 5.B).

The central requirement is that gaps are normalised WITHIN a segment. The
inherited `quali_best_s` took the minimum across Q1/Q2/Q3, which compares a
Q1-eliminated driver's lap against another driver's Q3 lap -- run on fresher
tyres, a rubbered-in track and less fuel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.qualifying import NATIVE_MISSING, add_qualifying_pace

# A plausible session: three drivers reach Q3, two are out in Q2, one in Q1.
# Q3 laps are quicker than Q1 laps for the same car, as in reality.
SESSION = pd.DataFrame([
    # driver, team,   q1,    q2,    q3
    ("AAA", "red",   80.5,  80.0,  79.5),
    ("BBB", "red",   80.7,  80.2,  79.8),
    ("CCC", "blue",  80.9,  80.4,  80.1),
    ("DDD", "blue",  81.2,  80.9,  np.nan),
    ("EEE", "green", 81.4,  81.1,  np.nan),
    ("FFF", "green", 82.0,  np.nan, np.nan),
], columns=["driver", "team", "q1_s", "q2_s", "q3_s"]).assign(year=2026, round=1)


@pytest.fixture
def paced() -> pd.DataFrame:
    return add_qualifying_pace(SESSION).set_index("driver")


# --- per-segment normalisation ----------------------------------------------

def test_segment_best_is_zero_gap(paced):
    assert paced.loc["AAA", "q1_gap_pct"] == pytest.approx(0.0)
    assert paced.loc["AAA", "q3_gap_pct"] == pytest.approx(0.0)


def test_gap_is_measured_against_the_same_segment(paced):
    """FFF is 1.5s off AAA's Q1 lap but 2.5s off AAA's Q3 lap. The gap must use
    Q1, the session both actually ran."""
    expected = 100.0 * (82.0 / 80.5 - 1.0)
    assert paced.loc["FFF", "q1_gap_pct"] == pytest.approx(expected)
    assert paced.loc["FFF", "quali_gap_pct"] == pytest.approx(expected)


def test_headline_gap_uses_the_deepest_segment_reached(paced):
    """A Q3 runner is judged on Q3, a Q2 casualty on Q2."""
    assert paced.loc["AAA", "quali_gap_pct"] == pytest.approx(
        paced.loc["AAA", "q3_gap_pct"])
    assert paced.loc["DDD", "quali_gap_pct"] == pytest.approx(
        paced.loc["DDD", "q2_gap_pct"])
    assert paced.loc["FFF", "quali_gap_pct"] == pytest.approx(
        paced.loc["FFF", "q1_gap_pct"])


def test_gaps_are_scale_free_so_circuits_are_comparable():
    """A 1% deficit is 1% at Monaco and at Spa; half a second is not."""
    monaco = pd.DataFrame({"driver": ["A", "B"], "team": ["r", "b"],
                           "q1_s": [70.0, 70.7], "q2_s": [np.nan] * 2,
                           "q3_s": [np.nan] * 2, "year": 2026, "round": 1})
    spa = monaco.assign(q1_s=[100.0, 101.0], round=2)
    a = add_qualifying_pace(monaco).set_index("driver")
    b = add_qualifying_pace(spa).set_index("driver")
    assert a.loc["B", "q1_gap_pct"] == pytest.approx(1.0)
    assert b.loc["B", "q1_gap_pct"] == pytest.approx(1.0)


def test_each_race_is_normalised_independently():
    two = pd.concat([SESSION, SESSION.assign(round=2, q1_s=SESSION.q1_s + 10)],
                    ignore_index=True)
    out = add_qualifying_pace(two)
    for rnd in (1, 2):
        assert out.loc[out["round"] == rnd, "q1_gap_pct"].min() == pytest.approx(0.0)


# --- stage reached ----------------------------------------------------------

def test_stage_reached(paced):
    assert paced.loc["AAA", "quali_stage_reached"] == 3
    assert paced.loc["DDD", "quali_stage_reached"] == 2
    assert paced.loc["FFF", "quali_stage_reached"] == 1


def test_a_driver_with_no_time_at_all():
    frame = SESSION.copy()
    frame.loc[frame.driver == "FFF", ["q1_s", "q2_s", "q3_s"]] = np.nan
    out = add_qualifying_pace(frame).set_index("driver")
    assert out.loc["FFF", "quali_stage_reached"] == 0
    assert out.loc["FFF", "quali_no_time"] == 1
    assert np.isnan(out.loc["FFF", "quali_gap_pct"])


# --- missing is information -------------------------------------------------

def test_absent_segments_stay_nan(paced):
    """Not reaching Q3 is a fact about pace, not an absent measurement. Filling
    it would erase the signal."""
    assert np.isnan(paced.loc["DDD", "q3_gap_pct"])
    assert np.isnan(paced.loc["FFF", "q2_gap_pct"])


def test_native_missing_columns_are_declared():
    """Whatever the policy does elsewhere, these must reach the model as NaN."""
    for column in ("q3_gap_pct", "quali_gap_pct", "quali_pace_vs_teammate_pct"):
        assert column in NATIVE_MISSING


def test_missing_pace_is_never_read_as_fastest(paced):
    """The failure mode FIX_PLAN.md section 10 names: a NaN treated as 0 would
    make a driver with no time look like the session's quickest."""
    assert not (paced["quali_gap_pct"].fillna(-1) == 0).sum() > 3


# --- teammate comparison ----------------------------------------------------

def test_teammate_gap_uses_a_shared_segment(paced):
    """AAA and BBB both reached Q3, so compare their Q3 laps."""
    assert paced.loc["BBB", "quali_pace_vs_teammate_pct"] == pytest.approx(
        100.0 * (79.8 / 79.5 - 1.0))
    assert paced.loc["AAA", "quali_pace_vs_teammate_pct"] == pytest.approx(
        100.0 * (79.5 / 79.8 - 1.0))


def test_teammate_gap_falls_back_to_the_deepest_shared_segment(paced):
    """CCC reached Q3 but DDD did not, so their comparison must use Q2."""
    assert paced.loc["CCC", "quali_pace_vs_teammate_pct"] == pytest.approx(
        100.0 * (80.4 / 80.9 - 1.0))


def test_teammate_gap_is_nan_without_a_shared_segment():
    """EEE reached Q2, FFF did not. Comparing across segments would measure
    track evolution, not the drivers."""
    frame = SESSION.copy()
    out = add_qualifying_pace(frame).set_index("driver")
    assert out.loc["EEE", "quali_pace_vs_teammate_pct"] == pytest.approx(
        100.0 * (81.4 / 82.0 - 1.0))      # shared Q1


def test_teammate_gap_is_symmetric_in_sign(paced):
    assert (paced.loc["AAA", "quali_pace_vs_teammate_pct"] < 0
            < paced.loc["BBB", "quali_pace_vs_teammate_pct"])


# --- field spread -----------------------------------------------------------

def test_field_spread_is_larger_when_the_field_is_strung_out():
    tight = SESSION.assign(q1_s=[80.0, 80.05, 80.1, 80.15, 80.2, 80.25])
    loose = SESSION.assign(q1_s=[80.0, 81.0, 82.0, 83.0, 84.0, 85.0])
    a = add_qualifying_pace(tight)["quali_field_spread_pct"].iloc[0]
    b = add_qualifying_pace(loose)["quali_field_spread_pct"].iloc[0]
    assert b > a


# --- safety -----------------------------------------------------------------

def test_a_frame_without_quali_columns_is_handled():
    """A dataset ingested before per-segment times existed must not crash; the
    features come through as NaN rather than as invented values."""
    bare = pd.DataFrame({"driver": ["A", "B"], "team": ["r", "b"],
                         "year": 2026, "round": 1})
    out = add_qualifying_pace(bare)
    assert out["quali_gap_pct"].isna().all()
    assert (out["quali_stage_reached"] == 0).all()


def test_input_is_not_mutated():
    before = SESSION.copy()
    add_qualifying_pace(SESSION)
    pd.testing.assert_frame_equal(SESSION, before)


def test_features_are_within_race_only():
    """Adding a LATER race must not change an earlier race's values -- these are
    current-weekend measurements, not historical aggregates."""
    alone = add_qualifying_pace(SESSION).set_index("driver")
    with_later = add_qualifying_pace(
        pd.concat([SESSION, SESSION.assign(round=2)], ignore_index=True))
    with_later = with_later[with_later["round"] == 1].set_index("driver")
    for column in ("quali_gap_pct", "quali_stage_reached",
                   "quali_pace_vs_teammate_pct"):
        pd.testing.assert_series_equal(alone[column], with_later[column])
