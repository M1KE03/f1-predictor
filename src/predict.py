"""Pre-race inference for a race that has not been run yet.

Spec section 6's inference entry point was deliberately deferred until after
Gate 3 passed on real data (see README). Gate 3 has NOT passed on this data
(evaluate.py: model beats the grid baseline on set-overlap but not Spearman,
on an 8-race partial 2026 test season) -- so treat this script's output as
exploratory, not validated.

Mechanism: append one placeholder row per driver for the target race to
raw_results.parquet's schema, run it through the *exact same* leakage-safe
feature pipeline used for training (so every historical feature -- form,
circuit history, reliability, teammate deltas -- is computed identically),
then score with the trained model. The placeholder's own outcome columns
(position/points/status/is_dnf) are never read by its own features (the
shift(1) pattern excludes the current row), so leaving them empty is safe.

Two unknowns for a not-yet-run race, both flagged in the output:
  - grid_position: real quali hasn't happened. Falls back to each driver's
    own recent qualifying form (form_avg_quali_3) as an "assumed grid".
  - race-day weather: no forecast ingested. Falls back to the model's saved
    global fill values (same policy as any other missing weather reading).

Run: python -m src.predict --year 2026 --round 9
"""
import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .blend_rank import add_blended_score
from .columns import FEATURE_COLS
from . import grid as grid_module
from .features import build_asof_features
from .metrics import pred_rank_by_race
from .preprocessing import FillPolicy, assert_no_missing

log = logging.getLogger("predict")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


def build_placeholder_rows(raw: pd.DataFrame, year: int, rnd: int,
                            event_name: str, circuit_id: str, date: pd.Timestamp,
                            roster: pd.DataFrame) -> pd.DataFrame:
    """One NaN-outcome row per driver in `roster` (driver, driver_id, team)."""
    n = len(roster)
    return pd.DataFrame(dict(
        year=year, round=rnd, event_name=event_name, circuit_id=circuit_id,
        date=date, driver=roster["driver"].values, driver_id=roster["driver_id"].values,
        team=roster["team"].values,
        grid_position=np.nan, pit_start=0,
        position=np.nan, status=None, is_dnf=0, classified=0, points=0.0,
        finished_top10=np.nan,
        air_temp=np.nan, track_temp=np.nan, humidity=np.nan, wind_speed=np.nan,
        rainfall=np.nan, is_wet=np.nan,
        quali_best_s=np.nan, gap_to_pole_s=np.nan,
    ))


def roster_from_session(year: int, rnd: int, session: str = "Q") -> pd.DataFrame:
    """Entry list for the target race taken from a session that has already run.

    The last-ingested-round fallback goes stale whenever the lineup changes
    mid-season (driver swaps, mid-season team moves). If qualifying for the
    target race is already in the timing feed, it is the authoritative entry
    list *and* carries each driver's current team.
    """
    import fastf1
    from .ingest import canonical_team
    res = fastf1.get_session(year, rnd, session)
    res.load(laps=False, telemetry=False, weather=False, messages=False)
    r = res.results
    if r is None or not len(r):
        raise ValueError(f"{year} round {rnd} session {session!r} has no results yet")
    return pd.DataFrame(dict(
        driver=r["Abbreviation"].astype(str).values,
        driver_id=r["DriverId"].astype(str).values,
        team=[canonical_team(t) for t in r["TeamName"]],
    )).drop_duplicates("driver")


def predict(year: int, rnd: int,
            grid_snapshot: grid_module.GridSnapshot | None = None,
            roster: pd.DataFrame | None = None,
            models_dir: Path = MODELS_DIR) -> pd.DataFrame:
    import fastf1
    fastf1.Cache.enable_cache(str(PROJECT_ROOT / "cache"))

    raw = pd.read_parquet(DATA_DIR / "raw_results.parquet")
    raw["date"] = pd.to_datetime(raw["date"])

    schedule = fastf1.get_event_schedule(year)
    ev = schedule[schedule["RoundNumber"] == rnd].iloc[0]
    event_name, circuit_id_src = str(ev["EventName"]), str(ev["Location"])
    date = pd.Timestamp(ev["EventDate"])
    from .ingest import slugify
    circuit_id = slugify(circuit_id_src)

    if ((raw["year"] == year) & (raw["round"] == rnd)).any():
        raise ValueError(f"{year} round {rnd} already has real results ingested; "
                          f"re-run src.build_features / src.evaluate instead of predicting it.")

    if roster is None:
        last_round = raw[raw["year"] == year]["round"].max()
        roster = (raw[(raw["year"] == year) & (raw["round"] == last_round)]
                  [["driver", "driver_id", "team"]].drop_duplicates("driver"))
        log.info("Using %s round %s roster as the %s entry list (%s drivers) -- "
                 "update if the grid changes before the race.",
                 year, last_round, event_name, len(roster))
    else:
        log.info("Using supplied entry list for %s (%s drivers).",
                 event_name, len(roster))

    placeholder = build_placeholder_rows(raw, year, rnd, event_name, circuit_id, date, roster)
    # THE shared path. Imputation is deferred until the grid is known, so no
    # policy is passed here.
    combined = build_asof_features(raw, placeholder)

    target_mask = (combined["year"] == year) & (combined["round"] == rnd)

    entry_list = list(roster["driver"])
    if grid_snapshot is None:
        # Qualifying has not run and no grid was supplied. Fall back to each
        # driver's recent qualifying form, and say so: this is a shape-of-the-
        # field guess, not a grid. Ordering it through from_qualifying keeps
        # positions unique and within the real field size, rather than the old
        # round().clip(1, 22), which could repeat positions and hard-coded 22.
        assumed = (combined.loc[target_mask]
                   .set_index("driver")["form_avg_quali_3"].to_dict())
        grid_snapshot = grid_module.from_qualifying(
            entry_list, assumed, source="assumed-from-recent-form")
        grid_snapshot.notes.append(
            "No qualifying result and no supplied grid: order assumed from "
            "form_avg_quali_3 (average of PRIOR starting grids).")
    grid_snapshot.validate(entry_list)

    positions = grid_snapshot.entries.set_index("driver")
    combined.loc[target_mask, "grid_position"] = (
        combined.loc[target_mask, "driver"].map(positions["grid_position"]).to_numpy())
    combined.loc[target_mask, "pit_start"] = (
        combined.loc[target_mask, "driver"].map(positions["pit_start"]).to_numpy())

    # The SAME fitted policy the model was trained with, replayed exactly --
    # including the driver-prior fallback the old serving path skipped, which
    # gave 2091 cells a different value here than in training.
    policy = FillPolicy.from_json(models_dir / "fill_values.json")
    combined = policy.transform(combined)
    assert_no_missing(combined.loc[target_mask], FEATURE_COLS)

    clf = joblib.load(models_dir / "model.joblib")
    ranker = joblib.load(models_dir / "rank_model.joblib")
    with open(models_dir / "blend_alpha.json") as f:
        alpha = json.load(f)["alpha"]

    race = combined.loc[target_mask, ["driver", "team", "grid_position"]].copy()
    race["p_top10"] = clf.predict_proba(combined.loc[target_mask, FEATURE_COLS])[:, 1]
    race["rank_score"] = ranker.predict(combined.loc[target_mask, FEATURE_COLS])

    # One race, so the frame needs the race keys for the shared per-race
    # helper. Deterministic ties matter more here than anywhere else: this is
    # the published forecast, and the inherited rank(method='first') made it
    # depend on the order the roster happened to arrive in.
    race["year"], race["round"] = year, rnd
    race["blend_score"] = add_blended_score(race, "rank_score", alpha)

    out = (race.assign(_order=pred_rank_by_race(race, "blend_score", ascending=True))
               .sort_values("_order").reset_index(drop=True))
    out.insert(0, "pred_finish_rank", out.index + 1)
    out["predicted_podium"] = out["pred_finish_rank"] <= 3
    out.attrs["grid"] = grid_snapshot.to_dict()
    return out[["pred_finish_rank", "driver", "team", "grid_position",
                "predicted_podium", "p_top10"]]


def grid_from_cli(args, roster: pd.DataFrame) -> grid_module.GridSnapshot | None:
    """Build the grid snapshot the CLI flags describe.

    Returns None when nothing is known, which lets predict() fall back to an
    explicitly-labelled assumed order.
    """
    entry_list = list(roster["driver"])
    pit_starts = [d.strip() for d in (args.pit_start or "").split(",") if d.strip()]
    cutoff = args.cutoff

    if args.grid:
        return grid_module.from_grid(
            entry_list, {k: float(v) for k, v in json.loads(args.grid).items()},
            pit_starts=pit_starts, source="manual --grid", cutoff_utc=cutoff)

    if args.from_quali:
        import fastf1
        session = fastf1.get_session(args.year, args.round, "Q")
        session.load(laps=False, telemetry=False, weather=False, messages=False)
        qualifying = {str(a): float(pos) for a, pos in
                      zip(session.results["Abbreviation"], session.results["Position"])
                      if pd.notna(pos)}
        # Qualifying classification is NOT the grid: penalties, exclusions and
        # pit-lane starts are applied afterwards. The snapshot is labelled
        # provisional so the output says so.
        return grid_module.from_qualifying(
            entry_list, qualifying, pit_starts=pit_starts,
            source=f"{args.year} R{args.round} qualifying session",
            cutoff_utc=cutoff)

    if pit_starts:
        raise SystemExit("--pit-start needs a grid: pass --grid or --from-quali.")
    return None


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--grid", type=str, default=None,
                    help='JSON mapping driver abbreviation -> CONFIRMED grid '
                         'position, e.g. \'{"ANT":1,"LEC":2}\'. Every driver in '
                         'the entry list must appear, or be named in --pit-start.')
    ap.add_argument("--pit-start", type=str, default=None,
                    help="comma-separated abbreviations starting from the pit "
                         "lane, e.g. HUL,STR. Must be stated explicitly: a "
                         "driver missing from --grid is an error, not a pit start.")
    ap.add_argument("--cutoff", type=str, default=None,
                    help="forecast cutoff in UTC, recorded with the output "
                         "(e.g. 2026-08-01T13:00:00Z)")
    ap.add_argument("--models-dir", type=Path, default=MODELS_DIR,
                    help="model bundle directory (default: models/)")
    ap.add_argument("--from-quali", action="store_true",
                    help="take the entry list and a PROVISIONAL grid from the "
                         "target round's qualifying session (must have run). "
                         "Grid penalties applied after the session are not "
                         "reflected -- pass --grid for the real starting order.")
    args = ap.parse_args()

    roster = None
    if args.from_quali:
        import fastf1
        fastf1.Cache.enable_cache(str(PROJECT_ROOT / "cache"))
        roster = roster_from_session(args.year, args.round, "Q")

    if roster is None:
        raw = pd.read_parquet(DATA_DIR / "raw_results.parquet")
        last_round = raw[raw["year"] == args.year]["round"].max()
        roster = (raw[(raw["year"] == args.year) & (raw["round"] == last_round)]
                  [["driver", "driver_id", "team"]].drop_duplicates("driver"))

    try:
        snapshot = grid_from_cli(args, roster)
    except grid_module.GridError as exc:
        raise SystemExit(f"GRID ERROR: {exc}")

    out = predict(args.year, args.round, snapshot, roster, args.models_dir)
    record = out.attrs["grid"]

    print(f"\nGrid status : {record['status'].upper()}  "
          f"(source: {record['source']}, field size {record['field_size']}, "
          f"{record['n_pit_starts']} pit start(s))")
    if record["cutoff_utc"]:
        print(f"Cutoff (UTC): {record['cutoff_utc']}")
    for note in record["notes"]:
        print(f"  ! {note}")
    if record["status"] != grid_module.CONFIRMED:
        print("  ! This forecast is NOT made against a confirmed starting grid.")

    print("\nNOTE: pred_finish_rank/predicted_podium come from a grid-position + "
          "learning-to-rank blend (src.blend_rank, alpha tuned on the validation "
          "season); p_top10 is the separate top-10 classifier's probability, "
          "shown for reference only -- it is not what the order is sorted by.")
    print("NOTE: no weather input is used. Race-session weather was removed as "
          "unavailable before lights-out (see README).\n")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
