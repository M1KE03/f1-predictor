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
from .build_features import add_recent_form, add_reliability
from .circuit import add_circuit_history
from .columns import FEATURE_COLS, TARGET
from .leakage import sort_frame
from .metrics import pred_rank_by_race
from .teammate import add_teammate_features
from .weather import add_weather_affinity

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


def run_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Same feature steps as build_features.build(), minus the file I/O."""
    df = sort_frame(df)
    df = add_recent_form(df)
    df = add_weather_affinity(df)
    df = add_circuit_history(df)
    df = add_reliability(df)
    df = add_teammate_features(df)
    return df


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


def predict(year: int, rnd: int, grid_overrides: dict[str, int] | None = None,
            roster: pd.DataFrame | None = None) -> pd.DataFrame:
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
    combined = pd.concat([raw, placeholder], ignore_index=True)
    combined = run_pipeline(combined)

    target_mask = (combined["year"] == year) & (combined["round"] == rnd)

    # Unknown grid (quali hasn't run): assume each driver's recent qualifying
    # form. form_avg_quali_3 is computed from PRIOR races only, so this is
    # available even for this row.
    combined.loc[target_mask, "grid_position"] = (
        combined.loc[target_mask, "form_avg_quali_3"].round().clip(1, 22)
    )
    if grid_overrides:
        missing_drivers = set(grid_overrides) - set(combined.loc[target_mask, "driver"])
        if missing_drivers:
            raise ValueError(f"--grid has drivers not in the entry list: {missing_drivers}")
        for drv, pos in grid_overrides.items():
            combined.loc[target_mask & (combined["driver"] == drv), "grid_position"] = float(pos)
        pit_lane = set(roster["driver"]) - set(grid_overrides)
        if pit_lane:
            log.warning("Drivers missing from --grid (treated as pit-lane start, grid=20): %s",
                        pit_lane)
            combined.loc[target_mask & combined["driver"].isin(pit_lane), "grid_position"] = 20.0
            combined.loc[target_mask & combined["driver"].isin(pit_lane), "pit_start"] = 1

    with open(MODELS_DIR / "fill_values.json") as f:
        fills = json.load(f)
    for col, val in fills.items():
        if col in combined.columns:
            combined[col] = combined[col].fillna(val)
    combined["pit_start"] = combined["pit_start"].fillna(0).astype(int)

    missing = combined.loc[target_mask, FEATURE_COLS].isna().sum()
    bad = missing[missing > 0]
    if len(bad):
        raise AssertionError(f"NaNs remain in inference features:\n{bad}")

    clf = joblib.load(MODELS_DIR / "model.joblib")
    ranker = joblib.load(MODELS_DIR / "rank_model.joblib")
    with open(MODELS_DIR / "blend_alpha.json") as f:
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
    return out[["pred_finish_rank", "driver", "team", "grid_position",
                "predicted_podium", "p_top10"]]


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--grid", type=str, default=None,
                     help='JSON mapping driver abbreviation -> real grid position, '
                          'e.g. \'{"ANT":1,"LEC":2}\'. Overrides the assumed grid.')
    ap.add_argument("--from-quali", action="store_true",
                    help="pull the entry list AND the real grid from the "
                         "target round's qualifying session (must have run)")
    args = ap.parse_args()

    grid_overrides = json.loads(args.grid) if args.grid else None
    roster = None
    if args.from_quali:
        import fastf1
        fastf1.Cache.enable_cache(str(PROJECT_ROOT / "cache"))
        roster = roster_from_session(args.year, args.round, "Q")
        if grid_overrides is None:
            q = fastf1.get_session(args.year, args.round, "Q")
            q.load(laps=False, telemetry=False, weather=False, messages=False)
            grid_overrides = {str(a): int(pos) for a, pos in
                              zip(q.results["Abbreviation"], q.results["Position"])
                              if pd.notna(pos)}

    print("NOTE: pred_finish_rank/predicted_podium come from a grid-position + "
          "learning-to-rank blend (src.blend_rank, alpha tuned on the validation "
          "season); p_top10 is the separate top-10 classifier's probability, "
          "shown for reference only -- it is not what the order is sorted by.")
    if grid_overrides:
        print("NOTE: using REAL qualifying grid passed via --grid.\n")
    else:
        print("NOTE: grid_position is ASSUMED (real qualifying result not yet "
              "available) and weather is filled with training-set global means "
              "(no forecast ingested).\n")
    out = predict(args.year, args.round, grid_overrides, roster)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
