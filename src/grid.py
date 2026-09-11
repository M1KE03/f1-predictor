"""The starting-grid contract (FIX_PLAN.md section 2 P0-4, sections 4 and 5.B).

Three separate defects in the inherited inference path:

1. **Qualifying order was used as the grid.** `predict --from-quali` read the Q
   session's `Position`, which does not include grid penalties, pit-lane
   decisions or post-session exclusions. Those are exactly the cases where a
   forecast has most to gain from knowing the real grid.
2. **Missing drivers were silently invented as pit starters.** Any roster driver
   absent from `--grid` was assigned a pit start. A typo in one abbreviation
   quietly moved a front-runner to the back and the forecast still printed.
3. **Back of grid was hard-coded to 20.** The dataset holds 19-, 20- and 22-car
   races; the 2026 season runs 22.

The contract here keeps `qualifying_position` and `grid_position` as distinct
fields, requires pit starts to be stated rather than inferred from absence,
derives back-of-grid from the actual field size, and labels every snapshot
`confirmed` / `provisional` / `unavailable` so a forecast can never silently
claim more certainty about its inputs than it has.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Iterable, Mapping

import numpy as np
import pandas as pd

# A grid taken from the published starting order: penalties already applied.
CONFIRMED: Final = "confirmed"
# Derived from qualifying order. Penalties known after the session are NOT
# reflected, so the order may be wrong even though every position is filled.
PROVISIONAL: Final = "provisional"
# Qualifying has NOT run. The order is guessed from each driver's average of
# prior starting grids. Measured cost against a real grid, over 61 races of
# 2024-2026: winner accuracy 0.5902 -> 0.2787, podium 0.6831 -> 0.4809, and the
# assumed grid is 3.45 places off the real one on average -- all three
# differences resolve. This is a materially weaker forecast and must not be
# pooled with post-qualifying ones.
ASSUMED: Final = "assumed"
# No usable grid. A validated forecast must be declined.
UNAVAILABLE: Final = "unavailable"

GRID_STATUSES: Final = (CONFIRMED, PROVISIONAL, ASSUMED, UNAVAILABLE)

ENTRY_COLS: Final = ("driver", "qualifying_position", "grid_position", "pit_start")


class GridError(ValueError):
    """The grid cannot be established. Never downgrade this to a warning."""


def back_of_grid(field_size: int) -> float:
    """Grid position assigned to a pit-lane start.

    A pit starter is behind every car on the grid, so this is the field size,
    not the constant 20 the inherited code used.
    """
    if field_size < 1:
        raise GridError(f"Field size must be at least 1, got {field_size}.")
    return float(field_size)


@dataclass
class GridSnapshot:
    """The grid as known at the forecast cutoff, plus how much to trust it."""

    entries: pd.DataFrame
    status: str
    source: str
    cutoff_utc: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def field_size(self) -> int:
        return int(len(self.entries))

    @property
    def is_usable(self) -> bool:
        return self.status in (CONFIRMED, PROVISIONAL, ASSUMED)

    @property
    def is_post_qualifying(self) -> bool:
        """Whether a real qualifying session stands behind this grid.

        The single most important quality distinction in a forecast: ordering
        by an assumed grid roughly halves winner accuracy (see ASSUMED).
        """
        return self.status in (CONFIRMED, PROVISIONAL)

    def validate(self, roster: Iterable[str] | None = None) -> "GridSnapshot":
        """Raise GridError unless the grid is complete and internally consistent."""
        if self.status not in GRID_STATUSES:
            raise GridError(f"Unknown grid status {self.status!r}; "
                            f"expected one of {GRID_STATUSES}.")

        missing_cols = [c for c in ENTRY_COLS if c not in self.entries.columns]
        if missing_cols:
            raise GridError(f"Grid entries are missing columns: {missing_cols}")

        drivers = self.entries["driver"]
        duplicated = sorted(drivers[drivers.duplicated()].unique())
        if duplicated:
            raise GridError(f"Drivers appear more than once on the grid: {duplicated}")

        if roster is not None:
            roster_set = set(roster)
            absent = sorted(roster_set - set(drivers))
            if absent:
                raise GridError(
                    f"No grid information for {absent}. Supply their grid "
                    f"positions, or declare them pit starters explicitly. They "
                    f"will NOT be assumed to start from the pit lane.")
            extra = sorted(set(drivers) - roster_set)
            if extra:
                raise GridError(f"Grid names drivers not in the entry list: {extra}")

        on_grid = self.entries[self.entries["pit_start"] == 0]
        positions = on_grid["grid_position"]
        if positions.isna().any():
            blank = sorted(on_grid.loc[positions.isna(), "driver"])
            raise GridError(f"On-grid drivers without a grid position: {blank}")

        repeated = sorted(positions[positions.duplicated()].astype(int).unique())
        if repeated:
            raise GridError(f"Grid positions used more than once: {repeated}")

        out_of_range = sorted(
            positions[(positions < 1) | (positions > self.field_size)]
            .astype(int).unique())
        if out_of_range:
            raise GridError(
                f"Grid positions outside 1..{self.field_size}: {out_of_range}")
        return self

    def to_dict(self) -> dict[str, Any]:
        """Serialisable record of the grid a forecast was made against."""
        return {
            "status": self.status,
            "source": self.source,
            "cutoff_utc": self.cutoff_utc,
            "field_size": self.field_size,
            "n_pit_starts": int((self.entries["pit_start"] == 1).sum()),
            "notes": list(self.notes),
            "entries": self.entries[list(ENTRY_COLS)].to_dict(orient="records"),
        }


def _entry_frame(roster: Iterable[str],
                 qualifying: Mapping[str, float] | None,
                 grid: Mapping[str, float] | None,
                 pit_starts: Iterable[str]) -> pd.DataFrame:
    drivers = list(roster)
    pit = set(pit_starts or ())
    field_size = len(drivers)
    rows = []
    for driver in drivers:
        is_pit = driver in pit
        rows.append({
            "driver": driver,
            "qualifying_position": (float(qualifying[driver])
                                    if qualifying and driver in qualifying
                                    else np.nan),
            "grid_position": (back_of_grid(field_size) if is_pit
                              else (float(grid[driver])
                                    if grid and driver in grid else np.nan)),
            "pit_start": int(is_pit),
        })
    return pd.DataFrame(rows, columns=list(ENTRY_COLS))


def _reject_unknown(roster: Iterable[str], *mappings: Mapping[str, float] | None,
                    label: str) -> None:
    """An abbreviation the entry list does not contain is a mistake, not a hint."""
    known = set(roster)
    unknown = sorted({d for m in mappings if m for d in m} - known)
    if unknown:
        raise GridError(f"{label} names drivers not in the entry list: {unknown}")


def from_grid(roster: Iterable[str], grid: Mapping[str, float],
              qualifying: Mapping[str, float] | None = None,
              pit_starts: Iterable[str] = (), source: str = "manual",
              cutoff_utc: str | None = None) -> GridSnapshot:
    """A confirmed grid: positions are the published starting order."""
    _reject_unknown(roster, grid, qualifying, label="Grid")
    _reject_unknown(roster, {d: 0 for d in pit_starts}, label="Pit-start list")

    # Checked before building so the message names the real problem. Absence
    # from the grid map is NEVER read as "starts from the pit lane" -- that was
    # the inherited behaviour, and it demoted drivers silently.
    absent = sorted(set(roster) - set(grid) - set(pit_starts))
    if absent:
        raise GridError(
            f"No grid information for {absent}. Supply their grid positions, or "
            f"declare them pit starters explicitly. They will NOT be assumed to "
            f"start from the pit lane.")

    entries = _entry_frame(roster, qualifying, grid, pit_starts)
    return GridSnapshot(entries=entries, status=CONFIRMED, source=source,
                        cutoff_utc=cutoff_utc).validate(roster)


def from_qualifying(roster: Iterable[str], qualifying: Mapping[str, float],
                    pit_starts: Iterable[str] = (),
                    source: str = "qualifying-session",
                    cutoff_utc: str | None = None) -> GridSnapshot:
    """A PROVISIONAL grid derived from qualifying order.

    Qualifying classification is not the grid: penalties, exclusions and
    pit-lane starts are applied afterwards. Drivers keep their qualifying order,
    closed up to remove gaps left by declared pit starters, and the snapshot is
    labelled provisional so no caller can mistake it for the real thing.
    """
    _reject_unknown(roster, qualifying, label="Qualifying")
    _reject_unknown(roster, {d: 0 for d in pit_starts}, label="Pit-start list")

    # A driver without a qualifying time sorts to the back rather than failing:
    # not setting a time is a real outcome, unlike a name the entry list does
    # not contain. Driver id breaks ties so the order is deterministic.
    ranked = sorted((d for d in roster if d not in set(pit_starts)),
                    key=lambda d: (qualifying.get(d, float("inf")), d))
    grid = {driver: float(i) for i, driver in enumerate(ranked, start=1)}
    entries = _entry_frame(roster, qualifying, grid, pit_starts)
    snapshot = GridSnapshot(
        entries=entries, status=PROVISIONAL, source=source, cutoff_utc=cutoff_utc,
        notes=["Derived from qualifying order; grid penalties applied after the "
               "session are NOT reflected."])
    return snapshot.validate(roster)


def unavailable(roster: Iterable[str], reason: str) -> GridSnapshot:
    """No usable grid. Kept as an explicit object so callers must handle it."""
    entries = _entry_frame(roster, None, None, ())
    return GridSnapshot(entries=entries, status=UNAVAILABLE,
                        source="none", notes=[reason])
