"""Ingestion (spec section 1).

Produces one tidy row per (driver, race) for every race 2018 -> latest,
saved to data/raw_results.parquet, then runs VALIDATION GATE 1.

Run from the project root:
    python -m src.ingest
Options:
    python -m src.ingest --start-year 2018 --end-year 2024 --no-quali

Requires network access to FastF1's data sources (livetiming.formula1.com,
api.jolpi.ca). The FastF1 cache is enabled before any API call (spec 0.4).
"""
import argparse
import datetime as dt
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .grid import back_of_grid
from .labels import derive_labels

log = logging.getLogger("ingest")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "cache"
DATA_DIR = PROJECT_ROOT / "data"
RAW_PATH = DATA_DIR / "raw_results.parquet"

START_YEAR = 2018

# ---------------------------------------------------------------------------
# Team canonicalisation (spec 1.3).
#
# Renamed-but-continuous constructors collapse to one canonical id.
# Documented mappings:
#   Force India -> Racing Point -> Aston Martin              => aston_martin
#   Toro Rosso -> AlphaTauri -> RB -> Racing Bulls           => racing_bulls
#   Sauber -> Alfa Romeo (Racing) -> Kick Sauber -> Audi     => sauber
#   Renault -> Alpine                                        => alpine
# Everything else keeps its obvious slug. Unmapped names fall back to a slug
# with a loud warning so the mapping table can be extended deliberately.
# ---------------------------------------------------------------------------
TEAM_CANONICAL = {
    "red bull racing": "red_bull",
    "red bull": "red_bull",
    "mercedes": "mercedes",
    "ferrari": "ferrari",
    "mclaren": "mclaren",
    "williams": "williams",
    "haas f1 team": "haas",
    "haas": "haas",
    "renault": "alpine",
    "alpine": "alpine",
    "force india": "aston_martin",
    "racing point": "aston_martin",
    "aston martin": "aston_martin",
    "sauber": "sauber",
    "alfa romeo racing": "sauber",
    "alfa romeo": "sauber",
    "kick sauber": "sauber",
    "stake f1 team kick sauber": "sauber",
    "audi": "sauber",
    "scuderia toro rosso": "racing_bulls",
    "toro rosso": "racing_bulls",
    "alphatauri": "racing_bulls",
    "visa cash app rb": "racing_bulls",
    "rb f1 team": "racing_bulls",
    "racing bulls": "racing_bulls",
    "rb": "racing_bulls",
    "cadillac": "cadillac",
}


def canonical_team(name) -> str:
    if not isinstance(name, str) or not name.strip():
        return "unknown"
    key = name.strip().lower()
    if key in TEAM_CANONICAL:
        return TEAM_CANONICAL[key]
    # Substring pass for sponsor-decorated names ("BWT Alpine F1 Team",
    # "Scuderia AlphaTauri Honda"). Longest fragments first; fragments
    # shorter than 4 chars (e.g. "rb") are exact-match only to avoid
    # accidental substring hits.
    for frag in sorted(TEAM_CANONICAL, key=len, reverse=True):
        if len(frag) >= 4 and frag in key:
            return TEAM_CANONICAL[frag]
    slug = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    log.warning("Unmapped team name %r -> slug %r. Add it to TEAM_CANONICAL.", name, slug)
    return slug


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


# ---------------------------------------------------------------------------
# Session loading (spec 1.2)
# ---------------------------------------------------------------------------
def _load_session(year: int, rnd: int, kind: str):
    import fastf1
    session = fastf1.get_session(year, rnd, kind)
    session.load(laps=True, telemetry=False, weather=True, messages=False)
    return session


# ---------------------------------------------------------------------------
# Weather aggregation (spec 1.6)
# ---------------------------------------------------------------------------
def weather_row(session) -> dict:
    empty = dict(air_temp=np.nan, track_temp=np.nan, humidity=np.nan,
                 wind_speed=np.nan, rainfall=np.nan, is_wet=np.nan)
    w = getattr(session, "weather_data", None)
    if w is None or len(w) == 0:
        return empty
    rainfall = float(bool(w["Rainfall"].fillna(False).any()))
    return dict(
        air_temp=float(w["AirTemp"].mean()),
        track_temp=float(w["TrackTemp"].mean()),
        humidity=float(w["Humidity"].mean()),
        wind_speed=float(w["WindSpeed"].mean()),
        rainfall=rainfall,
        is_wet=rainfall,  # v1: treated as the same (spec 1.6)
    )


# ---------------------------------------------------------------------------
# Qualifying pace (spec 1.5) -- ingested as raw columns for later use
# (gap to pole etc.). NOT in FEATURE_COLS for v1; grid features come from
# GridPosition, which already reflects penalties.
# ---------------------------------------------------------------------------
def quali_frame(session_q) -> pd.DataFrame:
    res = session_q.results
    out = res[["Abbreviation"]].copy()
    best = pd.concat(
        [pd.to_timedelta(res[c], errors="coerce").dt.total_seconds() for c in ("Q1", "Q2", "Q3")],
        axis=1,
    ).min(axis=1)
    out["quali_best_s"] = best.values
    pole = out["quali_best_s"].min()
    out["gap_to_pole_s"] = out["quali_best_s"] - pole
    return out.rename(columns={"Abbreviation": "driver"})


# ---------------------------------------------------------------------------
# Per-race extraction (spec 1.3 / 1.4 / 1.5)
# ---------------------------------------------------------------------------
def race_rows(session, event, year: int, rnd: int) -> pd.DataFrame:
    res = session.results
    if res is None or len(res) == 0:
        raise ValueError("empty results")

    wrow = weather_row(session)
    field_size = len(res)
    date = pd.Timestamp(session.date)
    if date.tzinfo is not None:
        date = date.tz_localize(None)
    circuit_id = slugify(event["Location"])

    rows = []
    for _, r in res.iterrows():
        grid = r.get("GridPosition")
        pit_start = int(pd.isna(grid) or float(grid) == 0.0)
        # Back of grid is the actual field size, not a constant 20. The dataset
        # holds 19-, 20- and 22-car races and 2026 runs 22, so the inherited
        # literal placed pit starters ahead of two real cars (FIX_PLAN.md
        # section 2, P0-4). Existing stored rows are unaffected: all 15 of them
        # fall in 20-car races.
        grid_position = back_of_grid(field_size) if pit_start else float(grid)

        pos = r.get("Position")
        pos = float(pos) if pd.notna(pos) else np.nan
        status = str(r.get("Status") or "")
        laps = r.get("Laps")

        rows.append(dict(
            year=year, round=rnd,
            event_name=str(event["EventName"]),
            circuit_id=circuit_id,
            date=date,
            driver=str(r["Abbreviation"]),
            driver_id=str(r.get("DriverId") or ""),
            team=canonical_team(r.get("TeamName")),
            grid_position=grid_position,
            pit_start=pit_start,
            field_size=field_size,
            position=pos,
            status=status,
            # Raw source fields consumed by src.labels below. ClassifiedPosition
            # is the authority on official classification; Laps is kept for the
            # 90%-distance rule and for pace-per-stint work later.
            classified_position_raw=r.get("ClassifiedPosition"),
            laps_completed=float(laps) if pd.notna(laps) else np.nan,
            points=float(r.get("Points") if pd.notna(r.get("Points")) else 0.0),
            **wrow,
        ))

    df = pd.DataFrame(rows)

    # Separate the conflated outcome concepts (FIX_PLAN section 2, P0-6):
    # result_order / officially_classified / started / finished /
    # status_category, plus the is_winner / is_podium / finished_top10 labels.
    df = derive_labels(df, classified_position_col="classified_position_raw")

    # DEPRECATED aliases kept so downstream modules that have not been migrated
    # yet keep working. `classified` is true for 99.9% of rows and means only
    # "a result place exists" -- use officially_classified. `is_dnf` is the
    # complement of `finished`; it is derived from it here rather than
    # recomputed, so the two can no longer disagree.
    df["is_dnf"] = 1 - df["finished"]
    df["classified"] = df["result_order"].notna().astype(int)
    return df


# ---------------------------------------------------------------------------
# Main ingestion loop (spec 1.1)
# ---------------------------------------------------------------------------
def ingest(start_year: int = START_YEAR, end_year: int | None = None,
           with_quali: bool = True) -> pd.DataFrame:
    import fastf1
    fastf1.Cache.enable_cache(str(CACHE_DIR))  # spec 0.4: before any API call

    end_year = end_year or dt.date.today().year
    today = pd.Timestamp.now()

    frames = []
    for year in range(start_year, end_year + 1):
        try:
            schedule = fastf1.get_event_schedule(year)
        except Exception as e:
            log.warning("Could not load schedule for %s: %s", year, e)
            continue

        # Exclude testing (spec 1.1): round 0 and EventFormat == 'testing'
        sched = schedule[(schedule["RoundNumber"] > 0)
                         & (schedule["EventFormat"] != "testing")]

        for _, ev in sched.iterrows():
            rnd = int(ev["RoundNumber"])
            ev_date = pd.Timestamp(ev["EventDate"])
            if pd.notna(ev_date):
                if ev_date.tzinfo is not None:
                    ev_date = ev_date.tz_localize(None)
                if ev_date > today:
                    continue  # race hasn't happened yet

            # Race session -- never let one bad session abort the run (spec 1.2)
            try:
                s = _load_session(year, rnd, "R")
                df = race_rows(s, ev, year, rnd)
            except Exception as e:
                log.warning("SKIP %s round %s (R): %s", year, rnd, e)
                continue

            # Qualifying session (spec 1.5) -- failure must not drop the race
            if with_quali:
                try:
                    q = _load_session(year, rnd, "Q")
                    df = df.merge(quali_frame(q), on="driver", how="left")
                except Exception as e:
                    log.warning("No quali data for %s round %s: %s", year, rnd, e)
                    df["quali_best_s"] = np.nan
                    df["gap_to_pole_s"] = np.nan
            else:
                df["quali_best_s"] = np.nan
                df["gap_to_pole_s"] = np.nan

            frames.append(df)
            log.info("OK %s round %-2s %-30s rows=%s", year, rnd, ev["EventName"], len(df))

    if not frames:
        raise RuntimeError("No races ingested. Check network access / cache.")

    out = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    DATA_DIR.mkdir(exist_ok=True)
    out.to_parquet(RAW_PATH, index=False)
    log.info("Wrote %s (%s rows)", RAW_PATH, len(out))
    return out


# ---------------------------------------------------------------------------
# VALIDATION GATE 1 (spec end of section 1)
# ---------------------------------------------------------------------------
def gate1(df: pd.DataFrame, spot_year: int = 2021,
          spot_event_substr: str = "Abu Dhabi") -> None:
    n_races = df.groupby(["year", "round"]).ngroups
    print("\n=================== VALIDATION GATE 1 ===================")
    print(f"rows                 : {len(df)}")
    print(f"unique races         : {n_races}")
    print(f"date range           : {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"finished_top10 mean  : {df['finished_top10'].mean():.4f}  (expect ~0.42-0.48)")
    print(f"dnf rate             : {df['is_dnf'].mean():.4f}")
    print(f"pit starts           : {int(df['pit_start'].sum())}")

    spot = df[(df["year"] == spot_year)
              & (df["event_name"].str.contains(spot_event_substr, case=False, na=False))]
    if len(spot):
        print(f"\nSpot check -- {spot_year} {spot_event_substr} (verify top-10 labels "
              f"against the real result before proceeding):")
        cols = ["driver", "team", "grid_position", "position", "status", "finished_top10"]
        print(spot.sort_values("position")[cols].head(12).to_string(index=False))
    else:
        print(f"\nSpot-check race not found ({spot_year} / {spot_event_substr}) -- "
              f"pick another race and verify manually.")
    print("==========================================================\n")
    print("Do NOT proceed to feature building until the spot check matches reality.")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start-year", type=int, default=START_YEAR)
    ap.add_argument("--end-year", type=int, default=None)
    ap.add_argument("--no-quali", action="store_true",
                    help="skip loading Q sessions (faster; quali_best_s will be NaN)")
    args = ap.parse_args()
    df = ingest(args.start_year, args.end_year, with_quali=not args.no_quali)
    gate1(df)


if __name__ == "__main__":
    main()
