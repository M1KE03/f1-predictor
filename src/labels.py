"""Result-label vocabulary (FIX_PLAN.md sections 2 P0-6 and 4).

The inherited schema collapsed several distinct concepts into one flag:

    classified = notna(Position)

On the stored dataset that is true for 2078 of 2080 rows (99.90%), so it
carries almost no information. It cannot answer any of the questions the
pipeline actually asks: did the car start? did it reach the flag? was it
officially classified? why did it stop? This module separates those concepts
into independent fields so downstream code can ask for exactly the one it
means.

Field definitions
-----------------
result_order
    The published classification order, 1 = winner. This is the source's
    ordering, NOT "finishing position": retirements, disqualifications and
    non-starters are demoted to the back of the order but keep a real place in
    it. Preserved rather than collapsed, per FIX_PLAN section 4.
classified_position_raw
    FastF1's ClassifiedPosition verbatim -- a number for a classified finisher,
    or a status code ('R' retired, 'D' disqualified, 'E' excluded, 'W'
    withdrawn, 'F' failed to qualify, 'N' not classified). None when the source
    did not supply it.
status_category
    Normalised cause bucket; see STATUS_CATEGORIES.
started
    The car took the start. False for DNS / withdrawn / did-not-qualify.
finished
    The car was running at the chequered flag (including lapped runners).
officially_classified
    The driver received an official classification. Authoritative when
    ClassifiedPosition is available; otherwise derived from status, which
    cannot apply the 90%-distance rule -- see officially_classified_source.

Nothing here reads the target race's outcome for any OTHER row, so these are
pure row-wise transforms and safe to apply at any point in the pipeline.
"""
from __future__ import annotations

import logging
import re
from typing import Final

import numpy as np
import pandas as pd

log = logging.getLogger("labels")

# --- status categories ------------------------------------------------------
FINISHED: Final = "finished"
MECHANICAL: Final = "mechanical"
ACCIDENT: Final = "accident"
DISQUALIFIED: Final = "disqualified"
DID_NOT_START: Final = "did_not_start"
WITHDRAWN: Final = "withdrawn"
NOT_CLASSIFIED: Final = "not_classified"
RETIRED_UNSPECIFIED: Final = "retired_unspecified"
UNKNOWN: Final = "unknown"

STATUS_CATEGORIES: Final = (
    FINISHED, MECHANICAL, ACCIDENT, DISQUALIFIED, DID_NOT_START,
    WITHDRAWN, NOT_CLASSIFIED, RETIRED_UNSPECIFIED, UNKNOWN,
)

# Categories in which the car was never on track at the start.
NON_STARTER_CATEGORIES: Final = frozenset({DID_NOT_START, WITHDRAWN})

# Exact status -> category. Keys are lowercased and whitespace-stripped.
#
# Two deliberate choices:
#
# 1. 'Retired' maps to RETIRED_UNSPECIFIED, not MECHANICAL. It is the source's
#    generic "stopped, cause not given" and accounts for 197 of 305 retirements
#    (64.6%) in the stored data. Binning it as mechanical would invent a cause
#    for two thirds of all retirements and quietly corrupt any reliability
#    feature built on top. See FIX_PLAN section 5.C.
# 2. Bare bodywork causes ('Undertray', 'Front wing') map to MECHANICAL. They
#    are ambiguous between contact damage and component failure, but the source
#    reports contact separately as 'Collision damage', so a bare component name
#    is more likely a failure. 4 rows in the stored data; revisit if that grows.
_STATUS_MAP: Final[dict[str, str]] = {
    # --- reached the flag ---
    "finished": FINISHED,
    "lapped": FINISHED,
    # --- non-starters ---
    "did not start": DID_NOT_START,
    "did not qualify": DID_NOT_START,
    "did not prequalify": DID_NOT_START,
    "withdrew": WITHDRAWN,
    "withdrawn": WITHDRAWN,
    # --- excluded from the result ---
    "disqualified": DISQUALIFIED,
    "excluded": DISQUALIFIED,
    "not classified": NOT_CLASSIFIED,
    # --- stopped, cause neither mechanical nor an on-track incident ---
    # "Retired" is the source's generic no-cause-given. "Illness" states a
    # cause, but a driver-condition retirement is neither a car failure nor a
    # collision, and inventing a third bucket for 2 rows would fragment the
    # vocabulary. Both land here; revisit if driver-condition retirements grow.
    "retired": RETIRED_UNSPECIFIED,
    "illness": RETIRED_UNSPECIFIED,
    "fatigue": RETIRED_UNSPECIFIED,
    # --- incidents ---
    "accident": ACCIDENT,
    "collision": ACCIDENT,
    "collision damage": ACCIDENT,
    "spun off": ACCIDENT,
    "damage": ACCIDENT,
    "debris": ACCIDENT,
    # --- mechanical: power unit and drivetrain ---
    "engine": MECHANICAL, "power unit": MECHANICAL, "power loss": MECHANICAL,
    "turbo": MECHANICAL, "exhaust": MECHANICAL, "alternator": MECHANICAL,
    "battery": MECHANICAL, "electrical": MECHANICAL, "electronics": MECHANICAL,
    "ers": MECHANICAL, "mgu-h": MECHANICAL, "mgu-k": MECHANICAL,
    "gearbox": MECHANICAL, "transmission": MECHANICAL, "clutch": MECHANICAL,
    "driveshaft": MECHANICAL, "differential": MECHANICAL, "halfshaft": MECHANICAL,
    # --- mechanical: fluids and cooling ---
    "hydraulics": MECHANICAL, "oil leak": MECHANICAL, "oil pressure": MECHANICAL,
    "fuel pressure": MECHANICAL, "fuel leak": MECHANICAL, "fuel pump": MECHANICAL,
    "fuel system": MECHANICAL, "out of fuel": MECHANICAL,
    "water pressure": MECHANICAL, "water leak": MECHANICAL, "water pump": MECHANICAL,
    "cooling system": MECHANICAL, "radiator": MECHANICAL, "overheating": MECHANICAL,
    # --- mechanical: chassis and running gear ---
    "suspension": MECHANICAL, "steering": MECHANICAL, "brakes": MECHANICAL,
    "throttle": MECHANICAL, "wheel": MECHANICAL, "wheel nut": MECHANICAL,
    "puncture": MECHANICAL, "tyre": MECHANICAL, "vibrations": MECHANICAL,
    "mechanical": MECHANICAL, "technical": MECHANICAL, "handling": MECHANICAL,
    "undertray": MECHANICAL, "front wing": MECHANICAL, "rear wing": MECHANICAL,
    "bodywork": MECHANICAL, "chassis": MECHANICAL, "seat": MECHANICAL,
}

# "+1 Lap", "+2 Laps", "+6 Laps" -- a lapped runner that reached the flag.
_PLUS_LAPS_RE: Final = re.compile(r"^\+\d+\s+laps?$")

# ClassifiedPosition status codes that mean "no official classification".
_UNCLASSIFIED_CODES: Final = frozenset({"r", "d", "e", "w", "f", "n"})


def normalize_status(status: object) -> str:
    """Map one source status string to a STATUS_CATEGORIES member.

    Unmapped values return UNKNOWN and log a warning, mirroring the deliberate
    TEAM_CANONICAL fallback in ingest.py: the mapping is extended on purpose,
    never guessed at silently.
    """
    if not isinstance(status, str) or not status.strip():
        return UNKNOWN
    key = " ".join(status.strip().lower().split())
    if key in _STATUS_MAP:
        return _STATUS_MAP[key]
    if _PLUS_LAPS_RE.match(key):
        return FINISHED
    log.warning("Unmapped result status %r -> %r. Add it to labels._STATUS_MAP.",
                status, UNKNOWN)
    return UNKNOWN


def officially_classified_from_raw(raw: object) -> bool | None:
    """Interpret FastF1's ClassifiedPosition. None when unavailable.

    A numeric value means the driver was officially classified; a letter code
    means they were not.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    text = str(raw).strip().lower()
    if not text or text in {"nan", "none"}:
        return None
    if text in _UNCLASSIFIED_CODES:
        return False
    try:
        float(text)
    except ValueError:
        return None
    return True


def derive_labels(df: pd.DataFrame, *, status_col: str = "status",
                  result_order_col: str = "position",
                  classified_position_col: str | None = None) -> pd.DataFrame:
    """Add the separated label fields to a copy of `df`.

    `classified_position_col` is the authoritative source for
    officially_classified. When it is absent the value is derived from status,
    which CANNOT apply the 90%-race-distance rule -- a driver who retired late
    but completed enough distance is officially classified, and status alone
    does not reveal that. The provenance is recorded per row in
    officially_classified_source so a consumer can refuse the derived value.
    """
    out = df.copy()

    out["status_category"] = out[status_col].map(normalize_status)
    out["result_order"] = pd.to_numeric(out[result_order_col], errors="coerce")

    category = out["status_category"]
    out["started"] = (~category.isin(NON_STARTER_CATEGORIES)).astype(int)
    out["finished"] = (category == FINISHED).astype(int)

    if classified_position_col is not None and classified_position_col in out.columns:
        out["classified_position_raw"] = out[classified_position_col].astype("object")
        from_raw = out[classified_position_col].map(officially_classified_from_raw)
    else:
        out["classified_position_raw"] = None
        from_raw = pd.Series(None, index=out.index, dtype="object")

    # Fallback: a car that reached the flag is classified; one that never
    # started, was excluded, or has no place in the order is not. Everything
    # else (a retirement) is genuinely unknown without lap-distance data, and
    # is assumed classified because the source still gave it a result place.
    derived = np.where(
        category == FINISHED, True,
        np.where(category.isin(NON_STARTER_CATEGORIES)
                 | (category == DISQUALIFIED)
                 | (category == NOT_CLASSIFIED), False,
                 out["result_order"].notna()))

    # Compared with `== True` rather than filled-then-cast: from_raw is an
    # object column holding True/False/None, and .fillna().astype(bool) on it
    # triggers pandas' deprecated object downcasting.
    have_raw = from_raw.notna().to_numpy()
    out["officially_classified"] = np.where(
        have_raw, (from_raw == True).to_numpy(), derived).astype(int)  # noqa: E712
    out["officially_classified_source"] = np.where(
        have_raw, "classified_position", "derived_from_status")

    # Outcome labels, all defined on the published order.
    order = out["result_order"]
    out["is_winner"] = ((order == 1) & (out["officially_classified"] == 1)).astype(int)
    out["is_podium"] = ((order <= 3) & (out["officially_classified"] == 1)).astype(int)
    out["finished_top10"] = ((order <= 10) & (out["officially_classified"] == 1)).astype(int)
    return out


LABEL_COLS: Final = (
    "result_order", "classified_position_raw", "status_category", "started",
    "finished", "officially_classified", "officially_classified_source",
    "is_winner", "is_podium", "finished_top10",
)


def main() -> None:
    """Backfill the separated labels onto an already-ingested raw table.

    Re-ingesting needs network access to FastF1, so this derives every field
    that the stored columns support and writes a NEW file, leaving the
    hash-frozen data/raw_results.parquet byte-identical (reports/baseline.json
    depends on those bytes).

    What a backfill CANNOT recover: classified_position_raw and laps_completed
    were never stored, so officially_classified falls back to the status
    derivation for every row and is flagged as such. Re-ingest to get the
    authoritative values.
    """
    import argparse
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--raw", type=Path,
                        default=project_root / "data" / "raw_results.parquet")
    parser.add_argument("--out", type=Path,
                        default=project_root / "data" / "raw_results_labeled.parquet")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    df = pd.read_parquet(args.raw)
    out = derive_labels(
        df, classified_position_col=("classified_position_raw"
                                     if "classified_position_raw" in df.columns
                                     else None))

    print(f"\nrows: {len(out)}")
    print("\nstatus_category:")
    print(out["status_category"].value_counts().to_string())
    print("\nofficially_classified_source:")
    print(out["officially_classified_source"].value_counts().to_string())
    print(f"\nstarted={int(out['started'].sum())}  "
          f"finished={int(out['finished'].sum())}  "
          f"officially_classified={int(out['officially_classified'].sum())}")

    if "finished_top10" in df.columns:
        changed = int((df["finished_top10"].fillna(-1)
                       != out["finished_top10"]).sum())
        print(f"\nfinished_top10 rows changed vs legacy: {changed}")

    out.to_parquet(args.out, index=False)
    print(f"\nWrote {args.out}")
    print(f"Left {args.raw} untouched (hash-frozen by reports/baseline.json).")


if __name__ == "__main__":
    main()
