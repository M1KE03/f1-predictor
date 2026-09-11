"""Combine separately-ingested season ranges into one raw table.

`ingest.ingest()` writes whatever year range it was asked for, so a partial run
would REPLACE the existing seasons rather than extend them. Recovering
2018-2021 therefore happens as separate runs that this module stitches
together, with the coverage checks FIX_PLAN.md section 5.A.2 asks for:
reconcile stored events against the schedule, and explain every gap rather than
quietly accepting a short season.

Run:
    python -m src.merge_raw data/v2/raw_2018_2021.parquet \\
                            data/v2/raw_2022_2026.parquet \\
                            --out data/v2/raw_results.parquet
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("merge_raw")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def merge(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        frame["date"] = pd.to_datetime(frame["date"])
        log.info("%s: %s rows, %s races, %s-%s", path.name, len(frame),
                 frame.groupby(["year", "round"]).ngroups,
                 int(frame["year"].min()), int(frame["year"].max()))
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)

    # A race appearing in two inputs would silently double every historical
    # aggregate built on top of it.
    duplicated = combined.duplicated(subset=["year", "round", "driver"], keep=False)
    if duplicated.any():
        offenders = (combined.loc[duplicated, ["year", "round"]]
                     .drop_duplicates().to_dict("records"))
        raise ValueError(f"The same (year, round, driver) appears in more than "
                         f"one input. Overlapping races: {offenders}")

    return combined.sort_values("date", kind="mergesort").reset_index(drop=True)


def coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-season row/race counts and field sizes, for the gaps check."""
    rounds = df.groupby("year")["round"]
    report = pd.DataFrame({
        "rows": df.groupby("year")["driver"].size(),
        "races": rounds.nunique(),
        "max_round": rounds.max(),
        "min_field": df.groupby(["year", "round"])["driver"].size().groupby("year").min(),
        "max_field": df.groupby(["year", "round"])["driver"].size().groupby("year").max(),
    })
    # A season whose round numbers are not 1..max has a hole in it.
    report["missing_rounds"] = [
        sorted(set(range(1, int(row.max_round) + 1))
               - set(df.loc[df["year"] == year, "round"].astype(int)))
        for year, row in report.iterrows()]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path,
                        default=DATA_DIR / "v2" / "raw_results.parquet")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    combined = merge(args.inputs)

    report = coverage_report(combined)
    print("\n=================== COVERAGE ===================")
    print(report.to_string())

    gaps = report[report["missing_rounds"].apply(bool)]
    if len(gaps):
        print(f"\nWARNING: {len(gaps)} season(s) have missing rounds. Every gap "
              f"must be explained before these are treated as complete history "
              f"(FIX_PLAN.md section 5.A.2).")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.out, index=False)
    print(f"\ntotal: {len(combined)} rows, "
          f"{combined.groupby(['year', 'round']).ngroups} races, "
          f"{combined['date'].min().date()} -> {combined['date'].max().date()}")
    print(f"Wrote {args.out}")
    print("================================================")


if __name__ == "__main__":
    main()
