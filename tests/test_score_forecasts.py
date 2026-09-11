"""Scoring archived forecasts (FIX_PLAN.md section 8.6).

The archive exists so that a forecast cannot be quietly improved after the
fact. These tests pin that: scoring reads the archived prediction, never a
recomputed one.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src import score_forecasts as sf


def make_record(picked: list[str], p_win: list[float], grid: list[float]) -> dict:
    return {
        "schema_version": 1,
        "event": {"year": 2026, "round": 20, "event_name": "Test GP"},
        "grid": {"status": "confirmed"},
        "bundle": {"training_cutoff_utc": "2026-09-06 00:00:00"},
        "predictions": [
            {"pred_finish_rank": i + 1, "driver": d, "grid_position": g,
             "p_win": p, "p_podium": min(1.0, p * 3), "p_top10": min(1.0, p * 8)}
            for i, (d, p, g) in enumerate(zip(picked, p_win, grid))
        ],
    }


def make_results(order: list[str], year: int = 2026, rnd: int = 20) -> pd.DataFrame:
    return pd.DataFrame({
        "year": year, "round": rnd, "driver": order,
        "position": [float(i + 1) for i in range(len(order))],
        "status": "Finished",
    })


FORECAST = make_record(["AAA", "BBB", "CCC", "DDD"],
                       [0.5, 0.25, 0.15, 0.10],
                       [1.0, 2.0, 3.0, 4.0])


def test_a_correct_winner_pick_scores_one():
    row = sf.score_one(FORECAST, make_results(["AAA", "BBB", "CCC", "DDD"]))
    assert row["winner_hit"] == 1


def test_a_wrong_winner_pick_scores_zero():
    row = sf.score_one(FORECAST, make_results(["DDD", "AAA", "BBB", "CCC"]))
    assert row["winner_hit"] == 0


def test_the_grid_baseline_is_scored_on_the_same_race():
    """Grid position comes from the archived record, so the baseline is exactly
    what was knowable at the time."""
    row = sf.score_one(FORECAST, make_results(["AAA", "BBB", "CCC", "DDD"]))
    assert row["grid_winner_hit"] == 1


def test_a_race_with_no_result_yet_returns_none():
    assert sf.score_one(FORECAST, make_results(["AAA"], rnd=99)) is None


def test_log_loss_uses_the_archived_probability():
    """Not a recomputed one: a forecast that can be re-scored with a newer model
    is not evidence of anything."""
    import numpy as np
    row = sf.score_one(FORECAST, make_results(["AAA", "BBB", "CCC", "DDD"]))
    assert row["winner_log_loss"] == pytest.approx(-np.log(0.5))

    unlucky = sf.score_one(FORECAST, make_results(["DDD", "AAA", "BBB", "CCC"]))
    assert unlucky["winner_log_loss"] == pytest.approx(-np.log(0.10))


def test_podium_overlap_counts_the_selected_three():
    row = sf.score_one(FORECAST, make_results(["CCC", "BBB", "AAA", "DDD"]))
    assert row["podium_overlap"] == pytest.approx(1.0)
    partial = sf.score_one(FORECAST, make_results(["DDD", "AAA", "BBB", "CCC"]))
    assert partial["podium_overlap"] == pytest.approx(2 / 3)


def test_uniform_floor_reflects_the_field_size():
    import numpy as np
    row = sf.score_one(FORECAST, make_results(["AAA", "BBB", "CCC", "DDD"]))
    assert row["uniform_log_loss"] == pytest.approx(np.log(4))


def test_a_driver_missing_from_the_result_is_reported_not_hidden():
    """An entry list that changed after the forecast must be visible, not
    silently dropped."""
    row = sf.score_one(FORECAST, make_results(["AAA", "BBB", "CCC"]))
    assert row["drivers_unmatched"] == 1


def test_summary_averages_over_races():
    rows = [sf.score_one(FORECAST, make_results(["AAA", "BBB", "CCC", "DDD"])),
            sf.score_one(FORECAST, make_results(["DDD", "AAA", "BBB", "CCC"]))]
    total = sf.summarise(rows)
    assert total["n_races"] == 2
    assert total["winner_accuracy"] == pytest.approx(0.5)


def test_archive_round_trips_through_disk(tmp_path):
    (tmp_path / "2026-20.json").write_text(json.dumps(FORECAST))
    loaded = sf.load_forecasts(tmp_path)
    assert len(loaded) == 1
    assert loaded[0]["event"]["round"] == 20
