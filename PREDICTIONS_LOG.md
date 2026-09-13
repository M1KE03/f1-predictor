# Prediction log book

A race-by-race record of what the model forecast **before lights-out** and what
actually happened. One entry per race weekend, newest first.

**Why this file exists.** `FIX_PLAN.md` §8.6 makes the point that a backtest you
have looked at repeatedly is development evidence, not proof. Only forecasts
frozen *before* their outcomes validate the pipeline prospectively. This is the
human-readable half of that record; `reports/forecasts/*.json` is the machine
half.

> **Note on tracking.** `.gitignore` excludes `reports/*` and re-includes only
> `reports/baseline.json`, so the archived forecast JSONs are **not** in git
> today. Until that is changed, this file is the only version-controlled copy of
> the prospective record. See [Open items](#open-items).

---

## Weekly routine

1. **After qualifying, before the race** — archive the forecast:
   ```bash
   python -m src.predict --year 2026 --round <N> --from-quali --archive
   ```
   `--from-quali` matters. Against an assumed grid the measured cost over 61
   races is winner accuracy 0.5902 → 0.2787 and podium overlap 0.6831 → 0.4809.
   Never log an assumed-grid forecast as if it were a real one.
2. **Add an entry below** from the archived JSON, leaving `Actual` and `Δ` blank.
3. **After the race** — fill in `Actual` / `Δ`, complete the scorecard, and
   update the [season tally](#season-tally-2026).
4. **Once results are ingested** — get the machine-scored version:
   ```bash
   python -m src.ingest --start-year 2026 --end-year 2026 --out data/v2/raw_2026.parquet
   python -m src.score_forecasts
   ```

### Scorecard definitions

| Metric | Meaning |
| --- | --- |
| **Winner hit** | Did `pred_finish_rank == 1` win? (yes / no) |
| **Podium overlap** | How many of the 3 predicted podium drivers finished top 3, out of 3 |
| **Top-10 overlap** | How many of the 10 predicted top-10 drivers finished top 10, out of 10 |
| **Grid comparison** | Same three numbers for "just sort by the starting grid" — the baseline that actually has to be beaten |
| **Favourite's p_win** | The model's stated probability for its own top pick, for calibration tracking |

The grid comparison is the whole point. The champion's *order* is roughly
grid-equivalent; its *probabilities* are what beat the baseline. An entry
without the grid column cannot tell you whether the model added anything.

---

## Season tally 2026

| Round | Race | Model winner | Grid winner | Model podium | Grid podium | Model top-10 | Grid top-10 |
| ---: | --- | :---: | :---: | ---: | ---: | ---: | ---: |
| 14 | Spanish GP (Madrid) | _pending_ | _pending_ | _/3 | _/3 | _/10 | _/10 |
| | **Season** | **0/0** | **0/0** | **0/0** | **0/0** | **0/0** | **0/0** |

Reference points from the 127-race backtest — what "normal" looks like:

| Method | Winner | Podium | Top-10 |
| --- | ---: | ---: | ---: |
| Starting grid | 0.5591 | 0.6693 | 0.7701 |
| Ranker (champion) | 0.6063 | 0.6719 | 0.7811 |

A handful of races is far too small a sample to conclude anything against those.
Log the numbers; resist reading a trend into them before ~20 races.

---

## Entries

### 2026 Round 14 — Spanish Grand Prix (Madrid) — 2026-09-13

| Field | Value |
| --- | --- |
| Forecast made (UTC) | 2026-09-13T13:09:37 |
| Grid status | **PROVISIONAL** — 2026 R14 qualifying session |
| Field size | 22 (0 pit start(s)) |
| Bundle | trained on 186 races to 2026-09-06, T=0.344, alpha=0.3 |
| Bundle fingerprint | code `fe6e7fa7d6e4` · data `6ddcffdd865c` |
| Archived record | `reports/forecasts/2026-14.json` |

**Headline call:** ANT P1 (27.3% win) · NOR P2 (24.2% win) · VER P3 (21.7% win)

#### Prediction vs result

| Pred | Driver | Team | Grid | p_win | p_podium | p_top10 | **Actual** | **Δ** |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | ANT | mercedes | 2 | 27.3% | 71.2% | >99.9% |  |  |
| 2 | NOR | mclaren | 1 | 24.2% | 67.2% | >99.9% |  |  |
| 3 | VER | red_bull | 3 | 21.7% | 62.7% | >99.9% |  |  |
| 4 | HAM | ferrari | 4 | 10.8% | 37.0% | 99.6% |  |  |
| 5 | LEC | ferrari | 5 | 4.3% | 16.6% | 93.6% |  |  |
| 6 | RUS | mercedes | 6 | 3.7% | 14.2% | 90.7% |  |  |
| 7 | PIA | mclaren | 7 | 2.0% | 7.5% | 74.9% |  |  |
| 8 | LAW | red_bull | 8 | 1.4% | 5.4% | 62.7% |  |  |
| 9 | LIN | racing_bulls | 10 | 0.7% | 2.6% | 39.3% |  |  |
| 10 | COL | alpine | 9 | 0.5% | 2.1% | 31.4% |  |  |
| 11 | HUL | sauber | 11 | 0.4% | 1.8% | 26.5% |  |  |
| 12 | BOR | sauber | 12 | 0.4% | 1.6% | 23.5% |  |  |
| 13 | SAI | williams | 17 | 0.3% | 1.1% | 16.7% |  |  |
| 14 | OCO | haas | 13 | 0.3% | 1.0% | 16.5% |  |  |
| 15 | ALB | williams | 16 | 0.3% | 1.2% | 15.7% |  |  |
| 16 | STR | aston_martin | 22 | 0.3% | 1.1% | 16.8% |  |  |
| 17 | PER | cadillac | 19 | 0.3% | 1.0% | 15.9% |  |  |
| 18 | BOT | cadillac | 20 | 0.3% | 1.0% | 15.9% |  |  |
| 19 | ALO | aston_martin | 18 | 0.2% | 0.9% | 15.2% |  |  |
| 20 | TSU | racing_bulls | 15 | 0.2% | 0.9% | 15.2% |  |  |
| 21 | GAS | alpine | 14 | 0.2% | 1.0% | 14.7% |  |  |
| 22 | BEA | haas | 21 | 0.2% | 1.1% | 15.4% |  |  |

_Fill **Actual** with the classified finishing position (`DNF` if retired) and **Δ** with actual − predicted._

#### Scorecard — fill after the race

| Metric | Model | Grid baseline |
| --- | --- | --- |
| Winner hit | _pending_ | _pending_ |
| Podium overlap | _ / 3 | _ / 3 |
| Top-10 overlap | _ / 10 | _ / 10 |
| Favourite's p_win | 27.3% (ANT) | — |
| Actual winner | _pending_ | |
| Actual podium | _pending_ | |

#### Notes

- **This is Madrid, not Barcelona.** In 2026 the *Spanish Grand Prix* runs at the
  Madring; Barcelona is a separate event that was round 7 on 2026-06-14. If you
  came here looking for Barcelona, that race is already in the dataset.
- **Grid is PROVISIONAL,** derived from the qualifying session. Penalties applied
  after qualifying are not reflected. If the published starting grid differs,
  the forecast was made against the wrong grid — record that here rather than
  quietly re-running it.
- **The model disagrees with pole.** It puts ANT (P2 on the grid) ahead of NOR
  (pole), and the top three are unusually tight: 27.3% / 24.2% / 21.7%. That is
  close to a three-way coin toss, and the honest reading is that this race has
  no strong favourite rather than that ANT is backed with conviction.
- **A superseded forecast exists.** An earlier assumed-grid forecast for this
  round, made 2026-09-11 before qualifying, is preserved at
  `reports/forecasts/superseded/2026-14.assumed-grid.json`. It is out of the
  scorer's non-recursive glob deliberately, so it cannot contaminate the
  prospective record. It called ANT / HAM / LEC and had a different roster
  (HAD rather than TSU at Racing Bulls), which is a fair illustration of why
  assumed grids are not worth scoring.

---

## Entry template

Copy this block for each new race.

```markdown
### YYYY Round N — Race Name (Circuit) — YYYY-MM-DD

| Field | Value |
| --- | --- |
| Forecast made (UTC) | |
| Grid status | |
| Field size | |
| Bundle | |
| Bundle fingerprint | |
| Archived record | `reports/forecasts/YYYY-NN.json` |

**Headline call:**

#### Prediction vs result

| Pred | Driver | Team | Grid | p_win | p_podium | p_top10 | **Actual** | **Δ** |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |

#### Scorecard — fill after the race

| Metric | Model | Grid baseline |
| --- | --- | --- |
| Winner hit | | |
| Podium overlap | _ / 3 | _ / 3 |
| Top-10 overlap | _ / 10 | _ / 10 |
| Favourite's p_win | | — |
| Actual winner | | |
| Actual podium | | |

#### Notes
```

---

## Open items

- [ ] **Track the forecast records in git.** `.gitignore` re-includes only
      `reports/baseline.json`, so `reports/forecasts/*.json` is untracked despite
      the file's own comment saying the JSON provenance records should be kept.
      One line fixes it: `!reports/forecasts/` plus `!reports/forecasts/*.json`.
- [ ] **Score round 14** once 2026 results are ingested (`src.score_forecasts`).
