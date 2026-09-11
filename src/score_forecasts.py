"""Score archived forecasts against results as they arrive (FIX_PLAN.md 8.6, 11).

Backtests are development evidence: they have been inspected, re-run and
iterated against, so they can no longer be treated as a clean estimate. The
only evidence immune to that is a forecast frozen BEFORE the race and scored
afterwards.

`predict.py --archive` writes those records. This scores every one whose result
has since arrived, compares it to the grid baseline recorded in the same file,
and accumulates the total. It deliberately re-derives nothing: the prediction
and the grid it was made against come from the archived JSON, not from a fresh
model run, so no later change to the code or the data can alter a past forecast.

Run:
    python -m src.score_forecasts
    python -m src.score_forecasts --results data/v2/raw_results.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .labels import derive_labels

log = logging.getLogger("score_forecasts")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FORECAST_DIR = PROJECT_ROOT / "reports" / "forecasts"

PODIUM_SIZE = 3
TOP10_SIZE = 10


def load_forecasts(directory: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(directory.glob("*.json")):
        with path.open(encoding="utf-8") as fh:
            record = json.load(fh)
        record["_path"] = path.name
        records.append(record)
    return records


def score_one(record: dict[str, Any], results: pd.DataFrame) -> dict[str, Any] | None:
    """Score one archived forecast, or None if the result has not arrived."""
    event = record["event"]
    year, rnd = int(event["year"]), int(event["round"])
    actual = results[(results["year"] == year) & (results["round"] == rnd)]
    if actual.empty:
        return None

    actual = derive_labels(actual) if "is_winner" not in actual.columns else actual
    predicted = pd.DataFrame(record["predictions"])

    merged = predicted.merge(
        actual[["driver", "result_order", "is_winner", "is_podium",
                "finished_top10"]],
        on="driver", how="left", suffixes=("", "_actual"))

    missing = merged["result_order"].isna().sum()
    winners = set(actual.loc[actual["is_winner"] == 1, "driver"])
    podium = set(actual.loc[actual["is_podium"] == 1, "driver"])
    top10 = set(actual.loc[actual["finished_top10"] == 1, "driver"])
    if not winners:
        return None

    by_rank = merged.sort_values("pred_finish_rank")
    picked_winner = str(by_rank.iloc[0]["driver"])

    # Probability metrics use the archived p_win, never a recomputed one.
    winner_row = merged[merged["driver"].isin(winners)]
    p_winner = float(winner_row["p_win"].iloc[0]) if len(winner_row) else 1e-15

    grid_order = merged.sort_values("grid_position")
    grid_pick = str(grid_order.iloc[0]["driver"])

    finishers = merged.dropna(subset=["result_order"])
    spearman = (float(finishers["pred_finish_rank"].corr(
        finishers["result_order"], method="spearman"))
        if len(finishers) >= 3 else np.nan)

    return {
        "event": f"{year}-{rnd:02d}",
        "name": event.get("event_name", ""),
        "grid_status": record["grid"]["status"],
        "bundle_cutoff": record["bundle"]["training_cutoff_utc"][:10],
        "drivers_unmatched": int(missing),
        "winner_hit": int(picked_winner in winners),
        "grid_winner_hit": int(grid_pick in winners),
        "podium_overlap": len(set(by_rank.head(PODIUM_SIZE)["driver"]) & podium) / PODIUM_SIZE,
        "grid_podium_overlap": len(set(grid_order.head(PODIUM_SIZE)["driver"]) & podium) / PODIUM_SIZE,
        "top10_overlap": len(set(by_rank.head(TOP10_SIZE)["driver"]) & top10) / TOP10_SIZE,
        "winner_log_loss": -float(np.log(max(p_winner, 1e-15))),
        "uniform_log_loss": float(np.log(len(merged))),
        "podium_brier": float(((merged["p_podium"] - merged["is_podium"].fillna(0)) ** 2).mean()),
        "spearman": spearman,
    }


def summarise(scored: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(scored)
    return {
        "n_races": int(len(frame)),
        "winner_accuracy": float(frame["winner_hit"].mean()),
        "grid_winner_accuracy": float(frame["grid_winner_hit"].mean()),
        "podium_overlap": float(frame["podium_overlap"].mean()),
        "grid_podium_overlap": float(frame["grid_podium_overlap"].mean()),
        "top10_overlap": float(frame["top10_overlap"].mean()),
        "winner_log_loss": float(frame["winner_log_loss"].mean()),
        "uniform_log_loss": float(frame["uniform_log_loss"].mean()),
        "podium_brier": float(frame["podium_brier"].mean()),
        "spearman": float(frame["spearman"].mean(skipna=True)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecasts", type=Path, default=FORECAST_DIR)
    parser.add_argument("--results", type=Path,
                        default=DATA_DIR / "v2" / "raw_results.parquet")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    records = load_forecasts(args.forecasts)
    if not records:
        raise SystemExit(
            f"No archived forecasts in {args.forecasts}. Produce them with\n"
            f"  python -m src.predict --year YYYY --round N --from-quali --archive\n"
            f"run AFTER qualifying and BEFORE the race.")

    results = pd.read_parquet(args.results)
    scored, pending = [], []
    for record in records:
        row = score_one(record, results)
        (scored if row else pending).append(row or record["_path"])

    print("\n=================== PROSPECTIVE FORECAST RECORD ===================")
    print(f"archived forecasts : {len(records)}")
    print(f"scored (result in) : {len(scored)}")
    print(f"awaiting result    : {len(pending)}"
          + (f"  {pending}" if pending else ""))

    if not scored:
        print("\nNothing to score yet. This is the evidence that accumulates on its "
              "own -- keep archiving a forecast per race and re-run after each.")
        print("==================================================================")
        return

    frame = pd.DataFrame(scored)
    print("\nPer race:")
    print(frame[["event", "grid_status", "winner_hit", "grid_winner_hit",
                 "podium_overlap", "winner_log_loss"]].to_string(index=False))

    total = summarise(scored)
    print(f"\nCumulative over {total['n_races']} prospectively-forecast races:")
    print(f"  winner accuracy : {total['winner_accuracy']:.4f}   "
          f"(grid baseline {total['grid_winner_accuracy']:.4f})")
    print(f"  podium overlap  : {total['podium_overlap']:.4f}   "
          f"(grid baseline {total['grid_podium_overlap']:.4f})")
    print(f"  winner log loss : {total['winner_log_loss']:.4f}   "
          f"(uniform floor {total['uniform_log_loss']:.4f})")
    print(f"  podium Brier    : {total['podium_brier']:.4f}")
    print(f"  spearman        : {total['spearman']:.4f}")

    if total["n_races"] < 20:
        print(f"\nNOTE: {total['n_races']} races is far too few to conclude "
              f"anything. One race moves winner accuracy by "
              f"{1 / total['n_races']:.1%}. This is a record being built, not a "
              f"result.")

    out = args.out or (args.forecasts / "scored.json")
    with out.open("w", encoding="utf-8") as fh:
        json.dump({"summary": total, "races": scored,
                   "pending": [p for p in pending if isinstance(p, str)]},
                  fh, indent=2, default=str)
    print(f"\nWrote {out}")
    print("==================================================================")


if __name__ == "__main__":
    main()
