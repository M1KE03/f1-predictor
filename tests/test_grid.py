"""Grid-contract fixtures (FIX_PLAN.md section 2 P0-4 and section 10).

"Missing grid entries do not create invented pit starts. Handle variable
fields, no prior season roster, transfers, and unknown drivers."
"""
from __future__ import annotations

import numpy as np
import pytest

from src import grid

ROSTER = ["AAA", "BBB", "CCC", "DDD"]
QUALI = {"AAA": 1, "BBB": 2, "CCC": 3, "DDD": 4}
GRID = {"AAA": 1, "BBB": 2, "CCC": 3, "DDD": 4}


# --- back of grid uses the real field size ---------------------------------

@pytest.mark.parametrize("field_size", [19, 20, 22, 24])
def test_back_of_grid_follows_field_size(field_size):
    """The inherited code hard-coded 20. The dataset holds 19-, 20- and 22-car
    races, and 2026 runs 22."""
    assert grid.back_of_grid(field_size) == float(field_size)


def test_back_of_grid_rejects_an_empty_field():
    with pytest.raises(grid.GridError):
        grid.back_of_grid(0)


def test_pit_starter_is_placed_behind_a_22_car_field():
    roster = [f"D{i:02d}" for i in range(22)]
    positions = {d: i for i, d in enumerate(roster[1:], start=1)}
    snapshot = grid.from_grid(roster, positions, pit_starts=[roster[0]])
    entry = snapshot.entries.set_index("driver").loc[roster[0]]
    assert entry["grid_position"] == 22.0
    assert entry["pit_start"] == 1


# --- no invented pit starts -------------------------------------------------

def test_a_driver_missing_from_the_grid_is_an_error_not_a_pit_start():
    """The inherited path silently moved absent drivers to the pit lane, so a
    single mistyped abbreviation could demote a front-runner unnoticed."""
    partial = {"AAA": 1, "BBB": 2, "CCC": 3}
    with pytest.raises(grid.GridError, match="No grid information for"):
        grid.from_grid(ROSTER, partial)


def test_the_error_names_the_missing_drivers():
    with pytest.raises(grid.GridError, match="DDD"):
        grid.from_grid(ROSTER, {"AAA": 1, "BBB": 2, "CCC": 3})


def test_declared_pit_starters_are_accepted():
    snapshot = grid.from_grid(ROSTER, {"AAA": 1, "BBB": 2, "CCC": 3},
                              pit_starts=["DDD"])
    assert snapshot.entries.set_index("driver").loc["DDD", "pit_start"] == 1
    assert snapshot.status == grid.CONFIRMED


def test_a_driver_not_in_the_entry_list_is_rejected():
    with pytest.raises(grid.GridError, match="not in the entry list"):
        grid.from_grid(ROSTER, {**GRID, "ZZZ": 5})


# --- qualifying is not the grid ---------------------------------------------

def test_qualifying_derived_grid_is_labelled_provisional():
    """Penalties are applied after the session, so a quali-order grid must never
    be presented as confirmed."""
    snapshot = grid.from_qualifying(ROSTER, QUALI)
    assert snapshot.status == grid.PROVISIONAL
    assert any("penalties" in note.lower() for note in snapshot.notes)


def test_confirmed_and_provisional_grids_can_disagree():
    """A penalty moves a driver without changing their qualifying position -- the
    exact case the inherited code could not represent."""
    penalised = {"AAA": 4, "BBB": 1, "CCC": 2, "DDD": 3}
    confirmed = grid.from_grid(ROSTER, penalised, qualifying=QUALI)
    provisional = grid.from_qualifying(ROSTER, QUALI)

    c = confirmed.entries.set_index("driver")
    p = provisional.entries.set_index("driver")
    assert c.loc["AAA", "qualifying_position"] == 1
    assert c.loc["AAA", "grid_position"] == 4      # served the penalty
    assert p.loc["AAA", "grid_position"] == 1      # penalty unknown


def test_qualifying_order_closes_up_around_pit_starters():
    snapshot = grid.from_qualifying(ROSTER, QUALI, pit_starts=["BBB"])
    entries = snapshot.entries.set_index("driver")
    assert entries.loc["AAA", "grid_position"] == 1
    assert entries.loc["CCC", "grid_position"] == 2   # promoted, no gap left
    assert entries.loc["DDD", "grid_position"] == 3
    assert entries.loc["BBB", "grid_position"] == 4   # back of grid
    assert entries.loc["BBB", "qualifying_position"] == 2


def test_qualifying_position_is_retained_alongside_the_grid():
    snapshot = grid.from_grid(ROSTER, {"AAA": 2, "BBB": 1, "CCC": 3, "DDD": 4},
                              qualifying=QUALI)
    entries = snapshot.entries.set_index("driver")
    assert entries.loc["AAA", "qualifying_position"] == 1
    assert entries.loc["AAA", "grid_position"] == 2


def test_a_driver_with_no_qualifying_time_sorts_to_the_back():
    snapshot = grid.from_qualifying(ROSTER, {"AAA": 1, "BBB": 2, "CCC": 3})
    entries = snapshot.entries.set_index("driver")
    assert entries.loc["DDD", "grid_position"] == 4
    assert np.isnan(entries.loc["DDD", "qualifying_position"])


# --- structural validation --------------------------------------------------

def test_duplicate_grid_positions_are_rejected():
    with pytest.raises(grid.GridError, match="more than once"):
        grid.from_grid(ROSTER, {"AAA": 1, "BBB": 1, "CCC": 3, "DDD": 4})


def test_out_of_range_grid_positions_are_rejected():
    with pytest.raises(grid.GridError, match="outside"):
        grid.from_grid(ROSTER, {"AAA": 1, "BBB": 2, "CCC": 3, "DDD": 99})


def test_grid_positions_are_valid_for_a_19_car_field():
    roster = [f"D{i:02d}" for i in range(19)]
    snapshot = grid.from_grid(roster, {d: i for i, d in enumerate(roster, start=1)})
    assert snapshot.field_size == 19
    assert snapshot.entries["grid_position"].max() == 19


# --- unavailable ------------------------------------------------------------

def test_unavailable_is_an_explicit_state_not_a_silent_fallback():
    snapshot = grid.unavailable(ROSTER, reason="qualifying has not run")
    assert snapshot.status == grid.UNAVAILABLE
    assert not snapshot.is_usable
    assert "qualifying has not run" in snapshot.notes


def test_usable_states():
    assert grid.from_grid(ROSTER, GRID).is_usable
    assert grid.from_qualifying(ROSTER, QUALI).is_usable


# --- provenance -------------------------------------------------------------

def test_snapshot_serialises_its_provenance():
    snapshot = grid.from_grid(ROSTER, {"AAA": 1, "BBB": 2, "CCC": 3},
                              pit_starts=["DDD"], source="race-control",
                              cutoff_utc="2026-08-01T13:00:00Z")
    record = snapshot.to_dict()
    assert record["status"] == grid.CONFIRMED
    assert record["source"] == "race-control"
    assert record["cutoff_utc"] == "2026-08-01T13:00:00Z"
    assert record["field_size"] == 4
    assert record["n_pit_starts"] == 1
    assert len(record["entries"]) == 4


# --- the CLI layer ----------------------------------------------------------

class _Args:
    """Stand-in for argparse.Namespace, so the CLI contract is testable without
    a network round-trip to FastF1."""

    def __init__(self, **kw):
        self.year, self.round = 2026, 12
        self.grid = self.pit_start = self.cutoff = None
        self.from_quali = False
        self.__dict__.update(kw)


def _roster():
    import pandas as pd
    return pd.DataFrame({"driver": ROSTER})


def test_cli_without_a_grid_returns_none_so_predict_can_label_the_fallback():
    from src.predict import grid_from_cli
    assert grid_from_cli(_Args(), _roster()) is None


def test_cli_grid_json_produces_a_confirmed_snapshot():
    from src.predict import grid_from_cli
    snapshot = grid_from_cli(
        _Args(grid='{"AAA": 2, "BBB": 1, "CCC": 3, "DDD": 4}',
              cutoff="2026-08-01T13:00:00Z"), _roster())
    assert snapshot.status == grid.CONFIRMED
    assert snapshot.cutoff_utc == "2026-08-01T13:00:00Z"
    assert snapshot.entries.set_index("driver").loc["BBB", "grid_position"] == 1


def test_cli_incomplete_grid_raises_rather_than_inventing_pit_starts():
    from src.predict import grid_from_cli
    with pytest.raises(grid.GridError, match="No grid information for"):
        grid_from_cli(_Args(grid='{"AAA": 1, "BBB": 2, "CCC": 3}'), _roster())


def test_cli_pit_start_flag_completes_the_grid():
    from src.predict import grid_from_cli
    snapshot = grid_from_cli(
        _Args(grid='{"AAA": 1, "BBB": 2, "CCC": 3}', pit_start="DDD"), _roster())
    entries = snapshot.entries.set_index("driver")
    assert entries.loc["DDD", "pit_start"] == 1
    assert entries.loc["DDD", "grid_position"] == 4.0   # field size, not 20


def test_cli_pit_start_without_a_grid_is_refused():
    from src.predict import grid_from_cli
    with pytest.raises(SystemExit, match="needs a grid"):
        grid_from_cli(_Args(pit_start="DDD"), _roster())


def test_cli_rejects_an_unknown_abbreviation_in_pit_start():
    from src.predict import grid_from_cli
    with pytest.raises(grid.GridError, match="not in the entry list"):
        grid_from_cli(_Args(grid='{"AAA":1,"BBB":2,"CCC":3,"DDD":4}',
                            pit_start="ZZZ"), _roster())
