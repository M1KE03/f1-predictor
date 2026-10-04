import pandas as pd

from src.render_log import finish_label, pct, result_table, score

PREDS = [
    {"pred_finish_rank": 1, "driver": "AAA", "team": "red_bull", "grid_position": 2.0},
    {"pred_finish_rank": 2, "driver": "BBB", "team": "mclaren", "grid_position": 1.0},
    {"pred_finish_rank": 3, "driver": "CCC", "team": "sauber", "grid_position": 3.0},
]


def _results(order, raw):
    return pd.DataFrame({"driver": order, "team": ["x"] * 3, "grid_position": [1.0, 2.0, 3.0],
                         "pit_start": [0] * 3, "classified_position_raw": raw,
                         "result_order": [1, 2, 3]})


def test_result_table_is_blank_before_the_race():
    rows = result_table(PREDS, pd.DataFrame())[2:]
    assert rows == ["| 1 | | | | | |", "| 2 | | | | | |", "| 3 | | | | | |"]


def test_result_table_marks_gains_losses_and_dnf():
    rows = result_table(PREDS, _results(["BBB", "AAA", "CCC"], ["1", "2", "R"]))[2:]
    assert rows[0].endswith("| P2 | ▲1 |")
    assert rows[1].endswith("| P1 | ▼1 |")
    assert rows[2].startswith("| DNF | CCC |") and rows[2].endswith("|  |")


def test_score_compares_model_with_grid_and_skips_unclassified():
    s = score(PREDS, _results(["BBB", "AAA", "CCC"], ["1", "2", "R"]))
    assert s["winner"] == "BBB" and not s["model_win"] and s["grid_win"]
    assert s["model_podium"] == 2 and s["grid_podium"] == 2


def test_formatting_helpers():
    assert pct(1.0) == "100%" and pct(0.99975) == ">99.9%" and pct(0.0001) == "<0.1%"
    assert pct(0.318) == "31.8%"
    assert finish_label("7") == "7" and finish_label("D") == "DSQ" and finish_label("R") == "DNF"
