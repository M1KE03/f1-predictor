"""Relevance-label fixtures for the LambdaRank target (FIX_PLAN.md sections 4
and 10).

The inherited label gated on `classified`, which is true for 99.9% of rows, so
non-starters and disqualified drivers were scored as if they had finished where
the classification happened to list them.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.train_rank import add_relevance_label
from tests.test_metrics import make_race


@pytest.fixture
def scored() -> pd.DataFrame:
    return add_relevance_label(make_race()).set_index("driver")


def test_better_finish_scores_higher(scored):
    assert scored.loc["AAA", "relevance"] > scored.loc["BBB", "relevance"]
    assert scored.loc["BBB", "relevance"] > scored.loc["CCC", "relevance"]


def test_winner_gets_the_full_field_size(scored):
    assert scored.loc["AAA", "relevance"] == len(scored)


def test_ineligible_rows_score_zero(scored):
    """DSQ, DNS and withdrawal are the rows the deprecated `classified` flag
    failed to exclude."""
    assert scored.loc["HHH", "relevance"] == 0   # disqualified
    assert scored.loc["III", "relevance"] == 0   # did not start


def test_classified_retirement_keeps_its_relevance(scored):
    """A retirement that holds a place in the published classification is not
    collapsed to last (FIX_PLAN.md section 4)."""
    assert scored.loc["GGG", "relevance"] > 0
    assert scored.loc["GGG", "finished"] == 0
    assert scored.loc["GGG", "relevance"] < scored.loc["FFF", "relevance"]


def test_relevance_differs_from_the_legacy_classified_gate():
    """Regression guard: under the old gate HHH and III scored non-zero purely
    because a result place existed for them."""
    race = make_race()
    field_size = len(race)
    legacy = race.assign(
        legacy_rel=(field_size - race["position"] + 1).clip(lower=0).astype(int))
    new = add_relevance_label(race)

    merged = legacy.merge(new[["driver", "relevance"]], on="driver").set_index("driver")
    assert merged.loc["HHH", "legacy_rel"] > 0
    assert merged.loc["HHH", "relevance"] == 0
    assert merged.loc["III", "legacy_rel"] > 0
    assert merged.loc["III", "relevance"] == 0
    # Everyone actually classified is unaffected.
    for driver in ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]:
        assert merged.loc[driver, "legacy_rel"] == merged.loc[driver, "relevance"], driver


def test_relevance_is_a_non_negative_integer(scored):
    assert scored["relevance"].dtype.kind == "i"
    assert (scored["relevance"] >= 0).all()


def test_relevance_stays_within_lightgbm_default_label_gain(scored):
    """LightGBM's default label_gain covers labels 0..30. A larger field would
    overflow it, so the ceiling is asserted rather than assumed."""
    assert scored["relevance"].max() <= 30


def test_label_is_independent_of_row_order():
    race = make_race()
    reference = add_relevance_label(race).set_index("driver")["relevance"].sort_index()
    for seed in range(3):
        shuffled = race.sample(frac=1.0, random_state=seed)
        got = add_relevance_label(shuffled).set_index("driver")["relevance"].sort_index()
        pd.testing.assert_series_equal(reference, got)


def test_multiple_races_are_scored_independently():
    two = pd.concat([make_race(rnd=1), make_race(rnd=2)], ignore_index=True)
    scored = add_relevance_label(two)
    per_race = scored.groupby("round")["relevance"].max()
    assert per_race.nunique() == 1
