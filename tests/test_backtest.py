"""Backtest folds and promotion gates (FIX_PLAN.md sections 8 and 10)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import backtest, gates


def make_frame(n_seasons: int = 4, races: int = 10, drivers: int = 6) -> pd.DataFrame:
    rows = []
    for s in range(n_seasons):
        year = 2020 + s
        for rnd in range(1, races + 1):
            date = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(14 * rnd, unit="D")
            for d in range(drivers):
                rows.append(dict(year=year, round=rnd, date=date,
                                 driver=f"D{d:02d}", grid_position=float(d + 1),
                                 position=float(d + 1), status="Finished",
                                 score=float(drivers - d)))
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return make_frame()


# --- fold construction ------------------------------------------------------

def test_partitions_never_share_a_row(frame):
    for fold in backtest.make_folds(frame, backtest.ROLLING, min_train_races=10,
                                    val_races=5, block_races=5):
        assert not len(fold.train.intersection(fold.val))
        assert not len(fold.train.intersection(fold.test))
        assert not len(fold.val.intersection(fold.test))


def test_training_races_all_precede_the_test_block(frame):
    """The whole point of an expanding window: nothing from the future is fitted."""
    for fold in backtest.make_folds(frame, backtest.ROLLING, min_train_races=10,
                                    val_races=5, block_races=5):
        assert frame.loc[fold.train, "date"].max() < frame.loc[fold.test, "date"].min()
        assert frame.loc[fold.val, "date"].max() < frame.loc[fold.test, "date"].min()


def test_validation_sits_between_training_and_test(frame):
    """Early stopping must not see the races it will be scored on."""
    for fold in backtest.make_folds(frame, backtest.ROLLING, min_train_races=10,
                                    val_races=5, block_races=5):
        assert frame.loc[fold.train, "date"].max() < frame.loc[fold.val, "date"].min()


def test_a_row_belonging_to_two_partitions_is_rejected(frame):
    with pytest.raises(ValueError, match="exactly one partition"):
        backtest.Fold(name="bad", train=frame.index[:10], val=frame.index[5:15],
                      test=frame.index[20:30])


def test_rolling_blocks_do_not_overlap_each_other(frame):
    folds = backtest.make_folds(frame, backtest.ROLLING, min_train_races=10,
                                val_races=5, block_races=5)
    seen: set[int] = set()
    for fold in folds:
        rows = set(fold.test)
        assert not (rows & seen), "a race was evaluated in two folds"
        seen |= rows


def test_season_folds_hold_out_whole_seasons(frame):
    folds = backtest.make_folds(frame, backtest.SEASON)
    assert folds, "expected at least one season fold"
    for fold in folds:
        assert frame.loc[fold.test, "year"].nunique() == 1


def test_training_window_expands(frame):
    folds = backtest.make_folds(frame, backtest.ROLLING, min_train_races=10,
                                val_races=5, block_races=5)
    sizes = [len(f.train) for f in folds]
    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]


def test_unknown_scheme_is_rejected(frame):
    with pytest.raises(ValueError, match="Unknown fold scheme"):
        backtest.make_folds(frame, "random")


def test_too_little_history_yields_no_folds(frame):
    assert backtest.make_folds(frame, backtest.ROLLING,
                               min_train_races=10_000) == []


# --- manifests --------------------------------------------------------------

def test_manifest_hash_is_stable_across_runs(frame):
    a = backtest.make_folds(frame, backtest.ROLLING, min_train_races=10,
                            val_races=5, block_races=5)
    b = backtest.make_folds(frame, backtest.ROLLING, min_train_races=10,
                            val_races=5, block_races=5)
    assert [f.manifest["sha256"] for f in a] == [f.manifest["sha256"] for f in b]


def test_manifest_hash_changes_when_the_races_change(frame):
    a = backtest.make_folds(frame, backtest.ROLLING, min_train_races=10,
                            val_races=5, block_races=5)[0]
    b = backtest.make_folds(frame, backtest.ROLLING, min_train_races=15,
                            val_races=5, block_races=5)[0]
    assert a.manifest["sha256"] != b.manifest["sha256"]


def test_manifest_records_every_event_id(frame):
    fold = backtest.make_folds(frame, backtest.ROLLING, min_train_races=10,
                               val_races=5, block_races=5)[0]
    events = fold.manifest["events"]
    assert fold.manifest["n_races"]["test"] == len(events["test"])
    assert all(len(e) == 7 and e[4] == "-" for e in events["test"])  # 2020-01


# --- paired intervals -------------------------------------------------------

def test_identical_series_give_a_zero_difference():
    values = pd.Series([1.0, 0.0, 1.0, 1.0, 0.0])
    result = gates.paired_difference(values, values, n_bootstrap=200)
    assert result["mean_difference"] == 0.0
    assert result["ci_low"] == result["ci_high"] == 0.0
    assert not result["resolves"]


def test_a_consistent_gain_resolves():
    baseline = pd.Series([0.0] * 40)
    candidate = pd.Series([1.0] * 40)
    result = gates.paired_difference(candidate, baseline, n_bootstrap=500)
    assert result["mean_difference"] == 1.0
    assert result["resolves"]


def test_a_noisy_difference_does_not_resolve():
    rng = np.random.default_rng(0)
    baseline = pd.Series(rng.normal(0, 1, 40))
    candidate = baseline + rng.normal(0, 1, 40)
    result = gates.paired_difference(candidate, baseline, n_bootstrap=2000)
    assert result["ci_low"] < 0 < result["ci_high"]
    assert not result["resolves"]


def test_races_without_a_value_are_dropped_pairwise():
    baseline = pd.Series([1.0, np.nan, 0.0, 1.0])
    candidate = pd.Series([1.0, 1.0, np.nan, 0.0])
    assert gates.paired_difference(candidate, baseline, n_bootstrap=100)["n_races"] == 2


def test_no_shared_races_is_reported_not_crashed():
    result = gates.paired_difference(pd.Series([np.nan]), pd.Series([np.nan]),
                                     n_bootstrap=100)
    assert result["n_races"] == 0
    assert result["mean_difference"] is None
    assert not result["resolves"]


def test_the_bootstrap_unit_is_the_race():
    """Twenty-two drivers in one race are not 22 independent observations.
    Resampling rows instead of races would shrink this interval."""
    rng = np.random.default_rng(1)
    per_race = pd.Series(rng.normal(0.1, 1.0, 30))
    narrow = gates.paired_difference(per_race, pd.Series(np.zeros(30)),
                                     n_bootstrap=2000)
    # Same signal spread over 10x the rows would look far more certain.
    inflated = pd.Series(np.repeat(per_race.to_numpy(), 10))
    wrong = gates.paired_difference(inflated, pd.Series(np.zeros(300)),
                                    n_bootstrap=2000)
    width = narrow["ci_high"] - narrow["ci_low"]
    assert width > (wrong["ci_high"] - wrong["ci_low"]) * 2


def test_intervals_are_reproducible():
    values = pd.Series([1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
    zeros = pd.Series(np.zeros(6))
    a = gates.paired_difference(values, zeros, n_bootstrap=500)
    b = gates.paired_difference(values, zeros, n_bootstrap=500)
    assert a == b


# --- gates ------------------------------------------------------------------

def _rows(winner: float, podium: float, n: int = 70) -> pd.DataFrame:
    return pd.DataFrame({
        "winner_accuracy": [winner] * n,
        "podium_overlap": [podium] * n,
        "top10_overlap": [0.75] * n,
        "spearman_all": [0.65] * n,
        "spearman_finishers": [0.80] * n,
        "top_pick_finished_top10": [0.9] * n,
    })


def test_a_clear_improvement_passes_every_gate():
    result = gates.evaluate_gates(_rows(0.70, 0.70), _rows(0.60, 0.60), n_folds=3)
    assert all(g.passed for g in result), [g for g in result if not g.passed]


def test_too_small_a_winner_gain_fails():
    result = gates.evaluate_gates(_rows(0.63, 0.70), _rows(0.60, 0.60), n_folds=3)
    failed = [g.name for g in result if not g.passed]
    assert "winner accuracy +0.05" in failed


def test_a_guardrail_loss_fails():
    candidate = _rows(0.70, 0.70)
    candidate["top10_overlap"] = 0.70          # baseline is 0.75
    result = gates.evaluate_gates(candidate, _rows(0.60, 0.60), n_folds=3)
    assert "top-10 overlap loses <= 0.02" in [g.name for g in result if not g.passed]


def test_too_few_folds_fails_even_with_a_large_gain():
    result = gates.evaluate_gates(_rows(0.90, 0.90), _rows(0.60, 0.60), n_folds=2)
    assert "at least 3 folds" in [g.name for g in result if not g.passed]


def test_too_few_races_fails_even_with_a_large_gain():
    result = gates.evaluate_gates(_rows(0.90, 0.90, n=20), _rows(0.60, 0.60, n=20),
                                  n_folds=5)
    assert "at least 60 races" in [g.name for g in result if not g.passed]


def test_probability_gates_are_declared_unavailable_not_silently_skipped():
    per_race = {"grid_baseline": _rows(0.60, 0.60), "blend": _rows(0.70, 0.70)}
    report = gates.summary(per_race, n_folds=3)
    assert report["unavailable_gates"]
    assert any("log loss" in note for note in report["unavailable_gates"])
