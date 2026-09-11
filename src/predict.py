"""Pre-race inference for a race that has not been run yet.

Appends one placeholder row per driver for the target race and runs it through
`features.build_asof_features` -- the SAME path used to build training data, so
the feature vector is identical to the one the model was fitted on. The
placeholder's outcome columns stay empty; nothing reads a row's own result.

Scoring uses a versioned model bundle (src/bundle.py), which verifies its
feature schema, its imputation-policy version, and that it was NOT trained on
races at or after the target -- the one failure that would look like excellent
accuracy and be pure hindsight.

Output is a utility ORDER plus coherent win / podium / top-10 PROBABILITIES
from one Plackett-Luce distribution.

Honest status: winner accuracy does not beat the starting grid by the
FIX_PLAN.md section 8 margin (+0.0394 [-0.0394, +0.1181] over 127 backtested
races). The calibrated probabilities DO beat a probabilistic grid baseline
(winner log loss 1.15 vs 1.66, interval excludes zero). Read the probabilities
as the useful output and the order as roughly grid-equivalent.

Run:
    python -m src.predict --year 2026 --round 14 --from-quali --archive
"""
import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import grid as grid_module
from .blend_rank import add_blended_score
from .bundle import BundleError, ModelBundle
from .columns import FEATURE_COLS
from .features import build_asof_features
from .metrics import pred_rank_by_race
from .preprocessing import assert_no_missing
from .probabilities import add_probabilities

log = logging.getLogger("predict")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
# The frozen legacy bundle in models/ was fitted on 30 features and a
# pre-1.5 fill format, so it cannot serve predictions with this code.
# The corrected pipeline writes here (see README "Artifact layout").
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models" / "champion"
FORECAST_DIR = PROJECT_ROOT / "reports" / "forecasts"


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
            models_dir: Path = DEFAULT_MODELS_DIR) -> pd.DataFrame:
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

    # The bundle verifies its own feature schema, policy version and -- most
    # importantly -- that it was not trained on races at or after this one.
    bundle = ModelBundle.load(models_dir, FEATURE_COLS)
    bundle.assert_can_predict(date)

    combined = bundle.policy.transform(combined)
    assert_no_missing(combined.loc[target_mask], FEATURE_COLS)

    race = combined.loc[target_mask, ["year", "round", "driver", "team",
                                      "grid_position", "pit_start"]].copy()
    race["rank_score"] = bundle.ranker.predict(combined.loc[target_mask, FEATURE_COLS])
    race["blend_score"] = add_blended_score(race, "rank_score",
                                            bundle.manifest.alpha)

    # Coherent win / podium / top-10 probabilities from one Plackett-Luce
    # distribution, using the temperature fitted when the bundle was built.
    race = add_probabilities(race, "rank_score", bundle.manifest.temperature)

    out = (race.assign(_order=pred_rank_by_race(race, "blend_score", ascending=True))
               .sort_values("_order").reset_index(drop=True))
    out.insert(0, "pred_finish_rank", out.index + 1)
    out["predicted_podium"] = out["pred_finish_rank"] <= 3
    out.attrs["grid"] = grid_snapshot.to_dict()
    out.attrs["bundle"] = {
        "path": str(models_dir),
        "training_cutoff_utc": bundle.manifest.training_cutoff_utc,
        "code_revision": bundle.manifest.code_revision,
        "data_sha256": bundle.manifest.data_sha256,
        "temperature": bundle.manifest.temperature,
        "alpha": bundle.manifest.alpha,
        "n_train_races": bundle.manifest.n_train_races,
    }
    out.attrs["event"] = {"year": year, "round": rnd, "event_name": event_name,
                          "circuit_id": circuit_id, "date": str(date)}
    return out[["pred_finish_rank", "driver", "team", "grid_position",
                "predicted_podium", "p_win", "p_podium", "p_top10"]]


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


def archive_forecast(out: pd.DataFrame, directory: Path = FORECAST_DIR) -> Path:
    """Write an IMMUTABLE forecast record.

    FIX_PLAN.md section 8 point 6: historical backtests are development
    evidence once they have been looked at repeatedly, and only a sequence of
    timestamped forecasts frozen BEFORE their outcomes can validate the
    pipeline prospectively. None was being collected, so none exists -- and
    the only way to have that evidence next season is to start now.

    The record binds the prediction to the grid status it was made against and
    the exact bundle that made it, so it can be scored later without trusting
    anyone's memory of which model was live.
    """
    event = out.attrs["event"]
    record = {
        "schema_version": 1,
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "event": event,
        "grid": out.attrs["grid"],
        "bundle": out.attrs["bundle"],
        "predictions": out.to_dict(orient="records"),
    }
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{event['year']}-{int(event['round']):02d}.json"
    if path.exists():
        # Immutable on purpose: a forecast that can be rewritten after the race
        # is not evidence of anything.
        raise FileExistsError(
            f"{path} already exists. A forecast record is immutable -- delete it "
            f"deliberately if you really mean to replace it.")
    with path.open("w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, default=str)
    return path


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
    ap.add_argument("--archive", action="store_true",
                    help="write an immutable timestamped forecast record to "
                         "reports/forecasts/ for later scoring")
    ap.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR,
                    help="model bundle directory (default: models/v2, the "
                         "corrected pipeline). models/ holds the frozen "
                         "legacy bundle and cannot be served by this code.")
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

    try:
        out = predict(args.year, args.round, snapshot, roster, args.models_dir)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"\nCANNOT PREDICT: {exc}")
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
    bundle_info = out.attrs["bundle"]
    print(f"Bundle      : {bundle_info['path']}")
    print(f"  trained on {bundle_info['n_train_races']} races up to "
          f"{pd.Timestamp(bundle_info['training_cutoff_utc']).date()}, "
          f"T={bundle_info['temperature']:.3f}, alpha={bundle_info['alpha']}")
    print(f"  code {(bundle_info['code_revision'] or 'unknown')[:12]}  "
          f"data {(bundle_info['data_sha256'] or '')[:12]}\n")
    print(out.to_string(index=False))

    print("\nNOTE: winner accuracy does NOT beat the starting grid by the "
          "FIX_PLAN section 8 margin; the calibrated PROBABILITIES do beat a "
          "probabilistic grid baseline. Read p_win as the useful output and the "
          "predicted order as roughly grid-equivalent.")

    if args.archive:
        path = archive_forecast(out)
        print(f"\nArchived immutable forecast -> {path}")


if __name__ == "__main__":
    main()
