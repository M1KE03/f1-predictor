"""Synthetic raw_results generator -- TEST HARNESS ONLY.

Produces data/raw_results.parquet with the exact schema of src/ingest.py so
the feature pipeline, leakage audit, training and evaluation can be smoke-
tested in environments without access to FastF1's data endpoints.

THIS IS NOT REAL F1 DATA. Validation Gates 1-3 only count when run on real
ingested data. The generator plants latent structure (car strength, driver
skill, wet-weather skill, team reliability) so that leakage-safe features
carry genuine signal and the pipeline's end-to-end behaviour is meaningful.

Run: python -m src.make_synthetic [--seed 7]
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("make_synthetic")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_PATH = DATA_DIR / "raw_results.parquet"

POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
TEAMS = ["red_bull", "mercedes", "ferrari", "mclaren", "aston_martin",
         "alpine", "racing_bulls", "sauber", "haas", "williams"]
YEARS = list(range(2018, 2025))
ROUNDS = 20
DNF_STATUSES = ["Accident", "Engine", "Gearbox", "Hydraulics", "Collision"]


def generate(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # driver pool: 26 drivers; 20 race each season with occasional rookies
    drivers = [f"D{i:02d}" for i in range(26)]
    skill = {d: rng.normal(0, 0.6) for d in drivers}          # lower = faster
    wet_adj = {d: rng.normal(0, 0.5) for d in drivers}
    circuit_adj = {d: rng.normal(0, 0.3, ROUNDS) for d in drivers}

    rows = []
    for yi, year in enumerate(YEARS):
        car = {t: rng.normal(0, 1.0) for t in TEAMS}          # per-season car
        unrel = {t: np.clip(rng.normal(0.05, 0.04), 0.01, 0.2) for t in TEAMS}
        # rotate a couple of seats per season so debuts/swaps exist
        active = drivers[: 18] + drivers[18 + (yi % 4): 20 + (yi % 4)]
        lineup = {t: (active[2 * i], active[2 * i + 1]) for i, t in enumerate(TEAMS)}

        for rnd in range(1, ROUNDS + 1):
            date = pd.Timestamp(f"{year}-03-01") + pd.Timedelta(days=14 * (rnd - 1))
            circuit_id = f"circuit_{rnd:02d}"
            is_wet = int(rng.random() < 0.15)
            track_temp = float(np.clip(rng.normal(36, 9), 12, 58))
            weather = dict(
                air_temp=track_temp - rng.uniform(6, 12),
                track_temp=track_temp,
                humidity=float(np.clip(rng.normal(55, 15), 10, 100)),
                wind_speed=float(np.clip(rng.normal(3, 1.5), 0, 12)),
                rainfall=float(is_wet), is_wet=float(is_wet),
            )

            entries = []
            for team in TEAMS:
                for d in lineup[team]:
                    pace = (car[team] + skill[d]
                            + circuit_adj[d][rnd - 1]
                            + (wet_adj[d] if is_wet else 0.0))
                    quali = pace + rng.normal(0, 0.7)
                    race = pace + rng.normal(0, 1.0 if not is_wet else 1.6)
                    entries.append((d, team, quali, race))

            ent = pd.DataFrame(entries, columns=["driver", "team", "quali", "race"])
            ent["grid_position"] = ent["quali"].rank(method="first")
            ent["pit_start"] = 0
            # occasional pit-lane start
            if rng.random() < 0.08:
                i = int(rng.integers(len(ent)))
                ent.loc[ent.index[i], ["grid_position", "pit_start"]] = [20.0, 1]

            ent["dnf"] = [int(rng.random() < unrel[t] + 0.02) for t in ent["team"]]
            finishers = ent[ent["dnf"] == 0].copy()
            finishers["position"] = finishers["race"].rank(method="first")

            for _, e in ent.iterrows():
                if e["dnf"]:
                    pos, status = np.nan, str(rng.choice(DNF_STATUSES))
                else:
                    pos = float(finishers.loc[finishers["driver"] == e["driver"],
                                              "position"].iloc[0])
                    status = "Finished" if pos <= 14 else "+1 Lap"
                rows.append(dict(
                    year=year, round=rnd, event_name=f"Synthetic GP {rnd}",
                    circuit_id=circuit_id, date=date,
                    driver=e["driver"], driver_id=e["driver"].lower(),
                    team=e["team"],
                    grid_position=float(e["grid_position"]),
                    pit_start=int(e["pit_start"]),
                    position=pos, status=status,
                    is_dnf=int(e["dnf"]), classified=int(not e["dnf"]),
                    points=float(POINTS.get(pos, 0.0)) if not np.isnan(pos) else 0.0,
                    finished_top10=int((not np.isnan(pos)) and pos <= 10),
                    quali_best_s=float(80 + e["quali"]),
                    gap_to_pole_s=np.nan,  # filled per race below
                    **weather,
                ))

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["gap_to_pole_s"] = df["quali_best_s"] - df.groupby(["year", "round"])["quali_best_s"].transform("min")
    return df


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    df = generate(args.seed)
    DATA_DIR.mkdir(exist_ok=True)
    df.to_parquet(RAW_PATH, index=False)
    log.warning("Wrote SYNTHETIC data to %s (%s rows, %s races). "
                "NOT REAL F1 DATA -- for pipeline smoke tests only.",
                RAW_PATH, len(df), df.groupby(['year', 'round']).ngroups)


if __name__ == "__main__":
    main()
