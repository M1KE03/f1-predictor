"""Ingest practice long-run pace for every race (FIX_PLAN.md section 5.D).

Separate from `src.ingest` because it is expensive and optional: roughly one
session load per race, each pulling full lap data. The result is written to its
own parquet and merged at feature-build time, so a failed or partial practice
run never damages the main dataset.

Sessions are tried in preference order (FP2, then FP3, then FP1) and the one
actually used is recorded per race -- sprint weekends have only FP1, and a
washed-out FP2 is common.

Like `src.ingest`, this aborts on systematic source failures rather than
skipping every remaining race, and writes a coverage manifest.

Run:
    python -m src.ingest_practice --start-year 2018 --end-year 2026
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from .ingest import _is_systematic
from .practice import SESSION_PREFERENCE, practice_pace_summary

log = logging.getLogger("ingest_practice")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "cache"
DEFAULT_OUT = PROJECT_ROOT / "data" / "v2" / "practice.parquet"


def load_one(year: int, rnd: int) -> tuple[pd.DataFrame, str] | None:
    """Best available practice session for one race, with its name."""
    import fastf1

    for name in SESSION_PREFERENCE:
        try:
            session = fastf1.get_session(year, rnd, name)
            session.load(laps=True, telemetry=False, weather=False, messages=False)
        except Exception as exc:
            if _is_systematic(exc):
                raise
            continue                       # this session does not exist here
        summary = practice_pace_summary(session)
        if not summary.empty:
            return summary, name
    return None


def ingest(start_year: int, end_year: int, out_path: Path) -> pd.DataFrame:
    import fastf1
    fastf1.Cache.enable_cache(str(CACHE_DIR))

    frames, skipped = [], []
    today = pd.Timestamp.now()

    for year in range(start_year, end_year + 1):
        try:
            schedule = fastf1.get_event_schedule(year)
        except Exception as exc:
            if _is_systematic(exc):
                raise RuntimeError(f"Could not load the {year} schedule: {exc}") from exc
            skipped.append({"year": year, "round": None, "reason": str(exc)})
            continue

        events = schedule[(schedule["RoundNumber"] > 0)
                          & (schedule["EventFormat"] != "testing")]
        for _, event in events.iterrows():
            rnd = int(event["RoundNumber"])
            event_date = pd.Timestamp(event["EventDate"])
            if pd.notna(event_date):
                if event_date.tzinfo is not None:
                    event_date = event_date.tz_localize(None)
                if event_date > today:
                    continue

            try:
                loaded = load_one(year, rnd)
            except Exception as exc:
                if _is_systematic(exc):
                    raise RuntimeError(
                        f"Practice ingestion aborted at {year} round {rnd}: {exc}\n"
                        f"{len(frames)} races ingested first. Cached sessions cost "
                        f"no API calls, so a resume is cheap.") from exc
                loaded = None

            if loaded is None:
                log.warning("No usable practice long runs for %s round %s", year, rnd)
                skipped.append({"year": year, "round": rnd,
                                "reason": "no session with long runs"})
                continue

            summary, session_name = loaded
            summary.insert(0, "year", year)
            summary.insert(1, "round", rnd)
            summary["practice_session"] = session_name
            frames.append(summary)
            log.info("OK %s round %-2s %-6s drivers=%s", year, rnd, session_name,
                     len(summary))

    if not frames:
        raise RuntimeError("No practice data ingested.")

    out = pd.concat(frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    manifest = {
        "requested": {"start_year": start_year, "end_year": end_year},
        "ingested": {"rows": int(len(out)),
                     "races": int(out.groupby(["year", "round"]).ngroups),
                     "sessions_used": out.groupby("practice_session")["round"]
                                         .count().to_dict()},
        "skipped": skipped,
        "complete": not skipped,
    }
    with out_path.with_suffix(".coverage.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    log.info("Wrote %s (%s rows, %s races); %s skipped",
             out_path, len(out), manifest["ingested"]["races"], len(skipped))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out = ingest(args.start_year, args.end_year, args.out)
    print(f"\npractice pace: {len(out)} rows, "
          f"{out.groupby(['year', 'round']).ngroups} races")
    print(out.groupby("practice_session").size().to_string())


if __name__ == "__main__":
    main()
