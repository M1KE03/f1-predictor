"""Render PREDICTIONS_LOG.md from the archived forecasts and ingested results.

The log is generated, never hand-edited: each race gets a prediction board built
from its `reports/forecasts/<year>-<round>.json` record and a result table that
stays empty until that round appears in the raw results. Re-running after a
race is ingested fills the result table and the season board.

Predictions come only from the archived JSON, so re-rendering cannot change a
past forecast (same rule as `score_forecasts`).

Run:
    python -m src.render_log
    python -m src.render_log --results data/v2/raw_results.parquet --out PREDICTIONS_LOG.md
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("render_log")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORECAST_DIR = PROJECT_ROOT / "reports" / "forecasts"
DEFAULT_RESULTS = PROJECT_ROOT / "data" / "v2" / "raw_results.parquet"
DEFAULT_OUT = PROJECT_ROOT / "PREDICTIONS_LOG.md"

TEAM_NAMES = {
    "red_bull": "Red Bull", "mclaren": "McLaren", "aston_martin": "Aston Martin",
    "racing_bulls": "Racing Bulls", "sauber": "Audi",
}
MEDALS = {1: "🥇", 2: "🥈", 3: "🥉"}
STATUS_CODES = {"R": "DNF", "D": "DSQ", "W": "DNS", "N": "NC", "E": "EXC", "F": "DNQ"}
AFTER_START = "⚠️"


def team_name(team: str) -> str:
    return TEAM_NAMES.get(team, team.replace("_", " ").title())


def pct(p: float) -> str:
    if p >= 1.0:
        return "100%"
    if p >= 0.9995:
        return ">99.9%"
    if p < 0.0005:
        return "<0.1%"
    return f"{p * 100:.1f}%"


def finish_label(raw: str) -> str:
    raw = str(raw)
    return raw if raw.isdigit() else STATUS_CODES.get(raw, "DNF")


def load_schedule(year: int) -> pd.DataFrame | None:
    """Event schedule for locations and race start times; None if unavailable."""
    try:
        import fastf1
        fastf1.Cache.enable_cache(str(PROJECT_ROOT / "cache"))
        fastf1.set_log_level("ERROR")
        return fastf1.get_event_schedule(year, include_testing=False)
    except Exception as exc:  # offline: render without location / start-time flags
        log.warning("schedule unavailable for %s (%s)", year, exc)
        return None


def race_start_utc(event: pd.Series) -> pd.Timestamp | None:
    for i in range(1, 6):
        if event.get(f"Session{i}") == "Race":
            return pd.Timestamp(event[f"Session{i}DateUtc"]).tz_localize("UTC")
    return None


def race_results(results: pd.DataFrame, year: int, rnd: int) -> pd.DataFrame:
    return (results[(results["year"] == year) & (results["round"] == rnd)]
            .sort_values("result_order"))


def score(preds: list[dict], res: pd.DataFrame) -> dict:
    model = [p["driver"] for p in sorted(preds, key=lambda p: p["pred_finish_rank"])]
    grid = [p["driver"] for p in sorted(preds, key=lambda p: p["grid_position"])]
    classified = res[res["classified_position_raw"].astype(str).str.isdigit()]
    finish = list(classified["driver"])
    top3, top10 = set(finish[:3]), set(finish[:10])
    return {
        "pick": model[0], "winner": finish[0],
        "model_win": model[0] == finish[0], "grid_win": grid[0] == finish[0],
        "model_podium": len(set(model[:3]) & top3), "grid_podium": len(set(grid[:3]) & top3),
        "model_top10": len(set(model[:10]) & top10), "grid_top10": len(set(grid[:10]) & top10),
    }


def prediction_table(preds: list[dict]) -> list[str]:
    lines = ["| | Driver | Team | Grid | Win | Podium | Top 10 |",
             "| :---: | --- | --- | :---: | ---: | ---: | ---: |"]
    for p in sorted(preds, key=lambda p: p["pred_finish_rank"]):
        rank = p["pred_finish_rank"]
        drv = f"**{p['driver']}**" if rank <= 3 else p["driver"]
        lines.append(f"| {MEDALS.get(rank, rank)} | {drv} | {team_name(p['team'])} "
                     f"| P{int(p['grid_position'])} | {pct(p['p_win'])} "
                     f"| {pct(p['p_podium'])} | {pct(p['p_top10'])} |")
    return lines


def result_table(preds: list[dict], res: pd.DataFrame) -> list[str]:
    lines = ["| Pos | Driver | Team | Grid | Predicted | Δ |",
             "| :---: | --- | --- | :---: | :---: | :---: |"]
    if res.empty:
        lines += [f"| {i} | | | | | |" for i in range(1, len(preds) + 1)]
        return lines
    predicted = {p["driver"]: p["pred_finish_rank"] for p in preds}
    for _, row in res.iterrows():
        pos = finish_label(row["classified_position_raw"])
        want = predicted.get(row["driver"])
        delta = ""
        if pos.isdigit() and want is not None:
            gained = want - int(pos)
            delta = "=" if gained == 0 else (f"▲{gained}" if gained > 0 else f"▼{-gained}")
        grid = "Pit" if row.get("pit_start") == 1 else (
            f"P{int(row['grid_position'])}" if pd.notna(row["grid_position"]) else "—")
        lines.append(f"| {pos} | {row['driver']} | {team_name(row['team'])} | {grid} "
                     f"| {f'P{want}' if want else '—'} | {delta} |")
    return lines


def render(forecast_dir: Path, results_path: Path) -> str:
    records = sorted((json.loads(p.read_text(encoding="utf-8"))
                      for p in forecast_dir.glob("*.json")),
                     key=lambda r: (r["event"]["year"], r["event"]["round"]))
    if not records:
        raise SystemExit(f"no forecasts in {forecast_dir}")
    year = records[-1]["event"]["year"]
    records = [r for r in records if r["event"]["year"] == year]
    results = pd.read_parquet(results_path)
    schedule = load_schedule(year)
    events = ({int(e["RoundNumber"]): e for _, e in schedule.iterrows()}
              if schedule is not None else {})

    board, sections, scored = [], [], []
    by_round = {r["event"]["round"]: r for r in records}
    rounds = sorted(set(by_round) | {n for n in events if n >= min(by_round)})
    for rnd in rounds:
        ev = events.get(rnd)
        rec = by_round.get(rnd)
        name = (rec["event"]["event_name"] if rec else ev["EventName"]).replace(" Grand Prix", " GP")
        place = f" ({ev['Location']})" if ev is not None else ""
        date = (pd.Timestamp(rec["event"]["date"]) if rec else ev["EventDate"]).strftime("%d %b")
        if rec is None:
            board.append(f"| {rnd} | {name}{place} | {date} | | | | | |")
            continue
        start = race_start_utc(ev) if ev is not None else None
        late = start is not None and pd.Timestamp(rec["created_utc"]) > start
        flag = f" {AFTER_START}" if late else ""
        res = race_results(results, year, rnd)
        pick = next(p["driver"] for p in rec["predictions"] if p["pred_finish_rank"] == 1)
        if res.empty:
            board.append(f"| {rnd} | {name}{place}{flag} | {date} | {pick} | | | | |")
        else:
            s = score(rec["predictions"], res)
            scored.append(s)
            board.append(
                f"| {rnd} | {name}{place}{flag} | {date} | {pick} | {s['winner']} "
                f"| {'✅' if s['model_win'] else '❌'} / {'✅' if s['grid_win'] else '❌'} "
                f"| {s['model_podium']}/3 · {s['grid_podium']}/3 "
                f"| {s['model_top10']}/10 · {s['grid_top10']}/10 |")

        grid = rec["grid"]
        meta = [pd.Timestamp(rec["event"]["date"]).strftime("%d %b %Y"),
                f"grid {grid['status']}"]
        if late:
            meta.append(f"{AFTER_START} forecast made after race start")
        sections += [
            "---", "",
            f"## Round {rnd} · {rec['event']['event_name']}{place}", "",
            " · ".join(meta), "",
            "### Prediction", "", *prediction_table(rec["predictions"]), "",
            "### Result", "", *result_table(rec["predictions"], res), "",
        ]

    n = len(scored)
    if n:
        tot = lambda k: sum(s[k] for s in scored)
        board.append(
            f"| | **Total** | | | | **{tot('model_win')}/{n} · {tot('grid_win')}/{n}** "
            f"| **{tot('model_podium')}/{3 * n} · {tot('grid_podium')}/{3 * n}** "
            f"| **{tot('model_top10')}/{10 * n} · {tot('grid_top10')}/{10 * n}** |")

    out = [
        f"# {year} Prediction Log", "",
        "## Season", "",
        "| Rd | Grand Prix | Date | Pick | Winner | Winner (model / grid) "
        "| Podium (model · grid) | Top 10 (model · grid) |",
        "| :---: | --- | --- | :---: | :---: | :---: | :---: | :---: |",
        *board, "",
        f"{AFTER_START} forecast made after race start",
        "",
        *[line for record in reversed(_split(sections)) for line in record],
        "---", "",
        "<sub>Generated by `python -m src.render_log` — do not edit by hand. "
        "Archived forecasts: `reports/forecasts/`. Detailed notes: `PREDICTIONS_NOTES.md`.</sub>",
        "",
    ]
    return "\n".join(out)


def _split(sections: list[str]) -> list[list[str]]:
    """Split the flat section list into one block per race (to order newest first)."""
    blocks: list[list[str]] = []
    for line in sections:
        if line == "---":
            blocks.append([])
        blocks[-1].append(line)
    return blocks


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--forecasts", type=Path, default=FORECAST_DIR)
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out.write_text(render(args.forecasts, args.results), encoding="utf-8")
    log.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
