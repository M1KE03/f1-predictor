"""Metric-semantics and determinism fixtures (FIX_PLAN.md section 10).

The permutation tests are the regression guard for the recorded defect: the
frozen baseline measured a 0.0080 Spearman spread across five row shuffles, and
that must now be exactly zero.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import metrics


def make_race(year: int = 2026, rnd: int = 1) -> pd.DataFrame:
    """One synthetic race: 6 finishers, a classified retirement, a DSQ, a DNS."""
    rows = [
        # driver, grid, result_order, status, score
        ("AAA", 1.0, 1.0, "Finished", 0.90),
        ("BBB", 2.0, 2.0, "Finished", 0.80),
        ("CCC", 3.0, 3.0, "Finished", 0.70),
        ("DDD", 4.0, 4.0, "Finished", 0.60),
        ("EEE", 5.0, 5.0, "Lapped", 0.50),
        ("FFF", 6.0, 6.0, "+1 Lap", 0.40),
        ("GGG", 7.0, 7.0, "Retired", 0.30),
        ("HHH", 8.0, 8.0, "Disqualified", 0.20),
        ("III", 9.0, 9.0, "Did not start", 0.10),
    ]
    return pd.DataFrame(
        [dict(year=year, round=rnd, driver=d, grid_position=g,
              position=p, status=s, score=sc) for d, g, p, s, sc in rows])


@pytest.fixture
def race() -> pd.DataFrame:
    return make_race()


# --- deterministic ordering -------------------------------------------------

def test_pred_rank_is_a_total_order(race):
    ranks = metrics.assign_pred_rank(race, "score", ascending=False)
    assert sorted(ranks.tolist()) == list(range(1, len(race) + 1))


def test_pred_rank_is_independent_of_row_order(race):
    reference = (race.assign(r=metrics.assign_pred_rank(race, "score", False))
                     .set_index("driver")["r"])
    for seed in range(5):
        shuffled = race.sample(frac=1.0, random_state=seed)
        got = (shuffled.assign(r=metrics.assign_pred_rank(shuffled, "score", False))
                       .set_index("driver")["r"])
        pd.testing.assert_series_equal(reference.sort_index(), got.sort_index())


def test_tied_scores_break_on_grid_then_driver():
    """All scores equal: order must fall back to grid, then driver id."""
    tied = pd.DataFrame([
        dict(year=2026, round=1, driver="ZZZ", grid_position=1.0,
             position=1.0, status="Finished", score=0.5),
        dict(year=2026, round=1, driver="AAA", grid_position=2.0,
             position=2.0, status="Finished", score=0.5),
        dict(year=2026, round=1, driver="MMM", grid_position=2.0,
             position=3.0, status="Finished", score=0.5),
    ])
    ranks = metrics.assign_pred_rank(tied, "score", ascending=False)
    order = tied.assign(r=ranks).sort_values("r")["driver"].tolist()
    # ZZZ first on grid 1; then grid 2 tie resolved alphabetically.
    assert order == ["ZZZ", "AAA", "MMM"]


def test_metrics_are_invariant_to_row_permutation(race):
    """The frozen baseline recorded a 0.0080 Spearman spread over five
    shuffles. It must now be exactly 0."""
    reference = metrics.race_metrics(race, "score", ascending=False)
    for seed in range(5):
        shuffled = race.sample(frac=1.0, random_state=seed)
        assert metrics.race_metrics(shuffled, "score", ascending=False) == reference


def test_missing_tie_breaker_is_an_error(race):
    with pytest.raises(KeyError, match="grid_position"):
        metrics.assign_pred_rank(race.drop(columns=["grid_position"]), "score", False)


# --- label semantics --------------------------------------------------------

def test_winner_and_podium_come_from_labels(race):
    result = metrics.race_metrics(race, "score", ascending=False)
    assert result["winner_accuracy"] == 1.0
    assert result["podium_overlap"] == 1.0


def test_dsq_and_dns_are_not_credited_as_top10(race):
    """HHH (DSQ) and III (DNS) sit at result_order 8 and 9. They must not count
    toward the actual top-10 set, so a model that ranks them highly is not
    rewarded."""
    result = metrics.race_metrics(race, "score", ascending=False)
    # 7 eligible top-10 drivers out of 10 slots: AAA..GGG (GGG is a classified
    # retirement and keeps its place); HHH and III are excluded.
    assert result["top10_overlap"] == pytest.approx(0.7)


def test_classified_retirement_keeps_its_place(race):
    labelled = metrics.ensure_labels(race).set_index("driver")
    assert labelled.loc["GGG", "officially_classified"] == 1
    assert labelled.loc["GGG", "finished"] == 0
    assert labelled.loc["HHH", "officially_classified"] == 0


# --- denominators -----------------------------------------------------------

def test_spearman_variants_use_different_denominators(race):
    result = metrics.race_metrics(race, "score", ascending=False)
    assert result["n_races_spearman_all"] == 1
    assert result["n_races_spearman_finishers"] == 1
    assert result["spearman_all"] is not None
    assert result["spearman_finishers"] is not None


def test_finishers_only_spearman_ignores_non_finishers():
    """Ordering the six finishers perfectly but the retirements backwards must
    give a perfect finishers-only score and an imperfect all-entries score."""
    race = make_race()
    # Score the three non-finishers as if they were the fastest.
    race.loc[race.driver.isin(["GGG", "HHH", "III"]), "score"] = [3.0, 2.0, 1.0]

    result = metrics.race_metrics(race, "score", ascending=False)
    assert result["spearman_finishers"] == pytest.approx(1.0)
    assert result["spearman_all"] < 1.0


def test_race_without_a_winner_is_excluded_from_winner_denominator():
    race = make_race()
    race["status"] = "Retired"          # nobody reaches the flag
    race["position"] = np.nan           # and nobody holds a place
    result = metrics.race_metrics(race, "score", ascending=False)
    assert result["n_races"] == 1
    assert result["n_races_winner"] == 0
    assert result["winner_accuracy"] is None


def test_metrics_aggregate_over_multiple_races():
    two = pd.concat([make_race(rnd=1), make_race(rnd=2)], ignore_index=True)
    result = metrics.race_metrics(two, "score", ascending=False)
    assert result["n_races"] == 2
    assert result["n_races_winner"] == 2


# --- renamed metric ---------------------------------------------------------

def test_top_pick_metric_is_named_for_what_it_measures(race):
    """The inherited name `top1_hit_rate` read as 'picked the winner' but is
    true whenever the top pick finishes anywhere in the top ten."""
    race = race.copy()
    # Make DDD (finishes 4th, inside the top ten) the top pick.
    race.loc[race.driver == "DDD", "score"] = 99.0

    result = metrics.race_metrics(race, "score", ascending=False)
    assert result["top_pick_finished_top10"] == 1.0
    assert result["winner_accuracy"] == 0.0
    assert "top1_hit_rate" not in result


def test_ascending_scores_are_supported(race):
    """grid_position and blended ranks order ascending; the metrics must agree
    with the descending-score equivalent."""
    descending = metrics.race_metrics(race, "score", ascending=False)
    flipped = race.assign(neg=-race["score"])
    ascending = metrics.race_metrics(flipped, "neg", ascending=True)
    assert ascending["winner_accuracy"] == descending["winner_accuracy"]
    assert ascending["spearman_all"] == descending["spearman_all"]


# --- the blend path (the published forecast, not just its evaluation) -------

def test_blended_score_is_invariant_to_row_permutation():
    """add_blended_score feeds src.predict, so an order-dependent result here
    means the published forecast itself was not reproducible."""
    from src.blend_rank import add_blended_score

    race = make_race()
    reference = (race.assign(b=add_blended_score(race, "score", 0.6))
                     .set_index("driver")["b"].sort_index())
    for seed in range(5):
        shuffled = race.sample(frac=1.0, random_state=seed)
        got = (shuffled.assign(b=add_blended_score(shuffled, "score", 0.6))
                       .set_index("driver")["b"].sort_index())
        pd.testing.assert_series_equal(reference, got)


def test_pred_rank_by_race_is_per_race_not_global():
    two = pd.concat([make_race(rnd=1), make_race(rnd=2)], ignore_index=True)
    ranks = metrics.pred_rank_by_race(two, "score", ascending=False)
    assert ranks.max() == len(make_race())          # not 2x the field
    assert sorted(ranks[two["round"] == 1]) == sorted(ranks[two["round"] == 2])


def test_pred_rank_by_race_preserves_input_index_order():
    two = pd.concat([make_race(rnd=1), make_race(rnd=2)], ignore_index=True)
    ranks = metrics.pred_rank_by_race(two, "score", ascending=False)
    assert list(ranks.index) == list(two.index)
    assert ranks.notna().all()
