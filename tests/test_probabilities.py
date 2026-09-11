"""Race-level probability layer (FIX_PLAN.md section 6).

The coherence tests are the ones that matter: win, podium and top-10
probabilities come from ONE Plackett-Luce distribution, so their totals must
hold by construction. Three independently normalised classifier outputs would
not satisfy them, which is precisely why the plan forbids that shortcut.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import probabilities as prob


def race(scores: list[float], winner_index: int = 0, year: int = 2026,
         rnd: int = 1) -> pd.DataFrame:
    n = len(scores)
    order = np.argsort(-np.asarray(scores))
    result = np.empty(n, dtype=float)
    result[order] = np.arange(1, n + 1)
    frame = pd.DataFrame({
        "year": year, "round": rnd,
        "driver": [f"D{i:02d}" for i in range(n)],
        "rank_score": scores,
        "result_order": result,
    })
    frame["is_winner"] = 0
    frame.loc[winner_index, "is_winner"] = 1
    frame["is_podium"] = (frame["result_order"] <= 3).astype(int)
    return frame


FIELD = race([3.0, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0,
              -2.5, -3.0, -3.5, -4.0, -4.5, -5.0, -5.5, -6.0, -6.5, -7.0])


# --- softmax ----------------------------------------------------------------

def test_win_probabilities_sum_to_one():
    assert prob.softmax(np.array([3.0, 2.0, 1.0]), 1.0).sum() == pytest.approx(1.0)


def test_higher_score_means_higher_win_probability():
    p = prob.softmax(np.array([3.0, 2.0, 1.0]), 1.0)
    assert p[0] > p[1] > p[2]


def test_temperature_changes_confidence_not_order():
    """Calibration cannot fix a wrong pick -- it only changes how sure the model
    sounds about it."""
    sharp = prob.softmax(np.array([3.0, 2.0, 1.0]), 0.2)
    flat = prob.softmax(np.array([3.0, 2.0, 1.0]), 5.0)
    assert sharp[0] > flat[0]
    assert list(np.argsort(-sharp)) == list(np.argsort(-flat))


def test_a_very_flat_temperature_approaches_uniform():
    p = prob.softmax(np.array([3.0, 2.0, 1.0]), 1000.0)
    assert p == pytest.approx(np.full(3, 1 / 3), abs=1e-3)


def test_non_positive_temperature_is_refused():
    with pytest.raises(ValueError, match="positive"):
        prob.softmax(np.array([1.0, 2.0]), 0.0)


def test_softmax_is_stable_for_extreme_scores():
    p = prob.softmax(np.array([1e4, -1e4]), 1.0)
    assert np.isfinite(p).all() and p.sum() == pytest.approx(1.0)


# --- coherence --------------------------------------------------------------

@pytest.fixture(scope="module")
def scored() -> pd.DataFrame:
    return prob.add_probabilities(FIELD, "rank_score", temperature=1.0,
                                  n_draws=20_000)


def test_totals_hold(scored):
    assert scored["p_win"].sum() == pytest.approx(1.0, abs=1e-9)
    assert scored["p_podium"].sum() == pytest.approx(3.0, abs=0.02)
    assert scored["p_top10"].sum() == pytest.approx(10.0, abs=0.02)


def test_probabilities_are_nested(scored):
    """p_win <= p_podium <= p_top10 for every driver. Independent binary models
    do not give this for free."""
    assert (scored["p_win"] <= scored["p_podium"] + 1e-9).all()
    assert (scored["p_podium"] <= scored["p_top10"] + 1e-9).all()


def test_coherence_report_passes_on_a_real_distribution(scored):
    report = prob.coherence_report(scored)
    assert report["coherent"], report["problems"]
    assert report["n_races"] == 1


def test_coherence_report_catches_independently_normalised_outputs():
    """The shortcut FIX_PLAN.md section 6 forbids: three separate models, each
    normalised on its own, do not form a coherent distribution."""
    broken = FIELD.copy()
    broken["p_win"] = 1.0 / len(broken)
    broken["p_podium"] = 1.0 / len(broken)     # should sum to 3, sums to 1
    broken["p_top10"] = 1.0 / len(broken)
    report = prob.coherence_report(broken)
    assert not report["coherent"]
    assert any("p_podium" in p for p in report["problems"])


def test_a_short_field_expects_smaller_totals():
    """A race with fewer than ten entrants cannot have ten top-ten finishers."""
    small = prob.add_probabilities(race([2.0, 1.0, 0.0]), "rank_score", 1.0,
                                   n_draws=5000)
    assert prob.coherence_report(small)["coherent"]
    assert small["p_top10"].sum() == pytest.approx(3.0, abs=1e-9)


# --- sampling ---------------------------------------------------------------

def test_sampling_is_reproducible():
    a = prob.sample_orderings(np.array([2.0, 1.0, 0.0]), 1.0, 500, seed=7)
    b = prob.sample_orderings(np.array([2.0, 1.0, 0.0]), 1.0, 500, seed=7)
    assert (a == b).all()


def test_each_draw_is_a_permutation():
    draws = prob.sample_orderings(np.array([2.0, 1.0, 0.0, -1.0]), 1.0, 200, seed=1)
    for row in draws:
        assert sorted(row) == [0, 1, 2, 3]


def test_sampled_win_rate_matches_the_closed_form():
    """Gumbel-max is exact for Plackett-Luce, so the sampled first-place rate
    must converge on the softmax."""
    scores = np.array([2.0, 1.0, 0.0])
    draws = prob.sample_orderings(scores, 1.0, 40_000, seed=3)
    sampled = np.bincount(draws[:, 0], minlength=3) / 40_000
    assert sampled == pytest.approx(prob.softmax(scores, 1.0), abs=0.01)


def test_monte_carlo_error_shrinks_with_draws():
    few = prob.race_probabilities(np.array([2.0, 1.0, 0.0]), 1.0, 100)["mc_error"]
    many = prob.race_probabilities(np.array([2.0, 1.0, 0.0]), 1.0, 10_000)["mc_error"]
    assert many < few


# --- calibration ------------------------------------------------------------

def test_temperature_fit_prefers_sharpness_when_the_model_is_right():
    """Favourite always wins -> the fit should be confident."""
    races = pd.concat([race([3.0, 1.0, 0.0], winner_index=0, rnd=r)
                       for r in range(1, 11)], ignore_index=True)
    assert prob.fit_temperature(races, "rank_score") < 1.0


def test_temperature_fit_prefers_flatness_when_the_model_is_wrong():
    """Favourite never wins -> confidence is punished, so the fit flattens."""
    races = pd.concat([race([3.0, 1.0, 0.0], winner_index=2, rnd=r)
                       for r in range(1, 11)], ignore_index=True)
    assert prob.fit_temperature(races, "rank_score") > 1.0


def test_winner_log_loss_rewards_probability_on_the_actual_winner():
    good = FIELD.assign(p_win=0.0)
    good.loc[good.is_winner == 1, "p_win"] = 0.9
    bad = FIELD.assign(p_win=0.0)
    bad.loc[bad.is_winner == 1, "p_win"] = 0.05
    assert prob.winner_log_loss(good) < prob.winner_log_loss(bad)


def test_uniform_baseline_is_log_of_field_size():
    assert prob.uniform_winner_log_loss(FIELD) == pytest.approx(np.log(20))


def test_a_calibrated_model_should_beat_the_uniform_floor(scored):
    """The sanity floor: a probability model that cannot beat 'every entrant is
    equally likely' has learned nothing."""
    assert prob.winner_log_loss(scored) < prob.uniform_winner_log_loss(scored)


def test_podium_brier_rewards_correct_podium_probabilities():
    confident = FIELD.assign(p_podium=FIELD["is_podium"].astype(float))
    clueless = FIELD.assign(p_podium=0.5)
    assert prob.podium_brier(confident) < prob.podium_brier(clueless)
