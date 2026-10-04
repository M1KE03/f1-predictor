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
| 14 | Spanish GP (Madrid) | ✓ | ✗ | 3/3 | 3/3 | 9/10 | 9/10 |
| 15 | Azerbaijan GP (Baku) | ✗ | ✓ | 1/3 | 1/3 | 7/10 | 6/10 |
| 16 | Bahrain GP (Kuala Lumpur) | _pending_ | _pending_ | _/3 | _/3 | _/10 | _/10 |
| | **Season (scored: 2)** | **1/2** | **1/2** | **4/6** | **4/6** | **16/20** | **15/20** |

**Prospective only:** none yet. All three forecasts were made after lights-out:
round 14 at 13:09 UTC against a 13:00 UTC start (9 minutes late, missed when
the entry was written), round 15 about 1h32m late, round 16 about 2h25m late.
`src.score_forecasts` does not know this and counts them anyway.

Reference points from the 127-race backtest — what "normal" looks like:

| Method | Winner | Podium | Top-10 |
| --- | ---: | ---: | ---: |
| Starting grid | 0.5591 | 0.6693 | 0.7701 |
| Ranker (champion) | 0.6063 | 0.6719 | 0.7811 |

A handful of races is far too small a sample to conclude anything against those.
Log the numbers; resist reading a trend into them before ~20 races.

---

## Entries

### 2026 Round 16 — Bahrain Grand Prix (Kuala Lumpur) — 2026-10-04

| Field | Value |
| --- | --- |
| Forecast made (UTC) | 2026-10-04T09:25:43 |
| Grid status | **PROVISIONAL** — 2026 R16 qualifying session |
| Field size | 22 (0 pit start(s)) |
| Bundle | trained on 188 races to 2026-09-26, T=0.169, alpha=0.3 |
| Bundle fingerprint | code `a7366603c19e` · data `6b0be9b95e64` |
| Archived record | `reports/forecasts/2026-16.json` |

**Headline call:** HAM P1 (31.8% win) · VER P2 (24.0% win) · HAD P3 (17.8% win)

> **⚠ NOT A PROSPECTIVE FORECAST — made after lights-out.** Scheduled race start
> was 2026-10-04 07:00 UTC; this forecast was generated at 09:25 UTC, about 2h25m
> after the start. Nothing from the race entered the inputs: `src.ingest` skipped
> round 16 with "empty results" (the timing feed had not published it), so the
> data and the bundle stop at round 15. But FIX_PLAN.md §8.6 counts only
> forecasts frozen *before* their outcome as prospective evidence, and this one
> was not. **Exclude this round when tallying prospective accuracy.**

#### Prediction vs result

| Pred | Driver | Team | Grid | p_win | p_podium | p_top10 | **Actual** | **Δ** |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | HAM | ferrari | 2 | 31.8% | 77.6% | 100.0% |  |  |
| 2 | VER | red_bull | 1 | 24.0% | 68.9% | >99.9% |  |  |
| 3 | HAD | red_bull | 3 | 17.8% | 57.2% | >99.9% |  |  |
| 4 | ANT | mercedes | 4 | 14.9% | 49.7% | 99.9% |  |  |
| 5 | LEC | ferrari | 5 | 2.8% | 11.3% | 88.8% |  |  |
| 6 | NOR | mclaren | 6 | 1.9% | 7.8% | 78.6% |  |  |
| 7 | PIA | mclaren | 7 | 1.3% | 5.4% | 66.7% |  |  |
| 8 | RUS | mercedes | 8 | 1.1% | 4.4% | 60.1% |  |  |
| 9 | LAW | racing_bulls | 11 | 0.6% | 2.2% | 36.8% |  |  |
| 10 | GAS | alpine | 9 | 0.5% | 2.0% | 33.5% |  |  |
| 11 | BOR | sauber | 10 | 0.5% | 1.8% | 31.1% |  |  |
| 12 | ALO | aston_martin | 12 | 0.3% | 1.2% | 20.9% |  |  |
| 13 | SAI | williams | 13 | 0.3% | 1.0% | 18.3% |  |  |
| 14 | LIN | racing_bulls | 16 | 0.3% | 1.3% | 20.1% |  |  |
| 15 | ALB | williams | 20 | 0.3% | 1.0% | 18.6% |  |  |
| 16 | STR | aston_martin | 14 | 0.3% | 1.1% | 18.2% |  |  |
| 17 | BOT | cadillac | 21 | 0.3% | 1.0% | 18.3% |  |  |
| 18 | COL | alpine | 15 | 0.3% | 1.0% | 18.1% |  |  |
| 19 | PER | cadillac | 22 | 0.3% | 1.1% | 18.5% |  |  |
| 20 | HUL | sauber | 17 | 0.3% | 1.0% | 17.8% |  |  |
| 21 | BEA | haas | 18 | 0.3% | 0.9% | 17.8% |  |  |
| 22 | OCO | haas | 19 | 0.3% | 1.0% | 17.9% |  |  |

_Fill **Actual** with the classified finishing position (`DNF` if retired) and **Δ** with actual − predicted._

#### Scorecard — fill after the race

| Metric | Model | Grid baseline |
| --- | --- | --- |
| Winner hit | _pending_ | _pending_ |
| Podium overlap | _ / 3 | _ / 3 |
| Top-10 overlap | _ / 10 | _ / 10 |
| Favourite's p_win | 31.8% (HAM) | — |
| Actual winner | _pending_ | |
| Actual podium | _pending_ | |

#### Notes

- **This is Kuala Lumpur, not Sakhir.** In 2026 the *Bahrain Grand Prix* runs in
  Kuala Lumpur. The event resolves to circuit id `kuala_lumpur`, so Sakhir
  history is correctly *not* used. But F1 last raced in Malaysia in 2017 and the
  data starts in 2018, so every driver is on a first visit: the circuit features
  are empty for the whole field and the call rests on grid, quali pace and form.
- **Round 15 (Baku) was ingested first.** `data/v2` previously ended at round 14.
  2026 was re-ingested (15 races; round 16 skipped as unpublished), merged as
  `raw_2022_2025` + `raw_2026`, features rebuilt, Gate 2 passed, and
  `models/champion` rebuilt with training cutoff 2026-09-26.
- **Grid is PROVISIONAL,** derived from the qualifying session. Penalties applied
  after qualifying are not reflected.
- **The model again disagrees with pole.** It puts HAM (P2 on the grid) ahead of
  VER (pole). The top four hold about 89% of the win probability between them,
  and no one is above a third, so there is no strong favourite.

---

### 2026 Round 15 — Azerbaijan Grand Prix (Baku) — 2026-09-26

| Field | Value |
| --- | --- |
| Forecast made (UTC) | 2026-09-26T12:32:50 |
| Grid status | **PROVISIONAL** — 2026 R15 qualifying session |
| Field size | 22 (0 pit start(s)) |
| Bundle | trained on 187 races to 2026-09-13, T=0.169, alpha=0.3 |
| Bundle fingerprint | code `a7366603c19e` · data `656268efc7f2` |
| Archived record | `reports/forecasts/2026-15.json` |

**Headline call:** LEC P1 (35.9% win) · RUS P2 (22.0% win) · PIA P3 (18.0% win)

> **⚠ NOT A PROSPECTIVE FORECAST — made after lights-out.** Scheduled race start
> was 2026-09-26 11:00 UTC; this forecast was generated at 12:32 UTC, about 1h32m
> into the race. Nothing from the race entered the inputs — the bundle's training
> cutoff is 2026-09-13 and round 15 results were not published at the time
> (`src.ingest` skipped round 15 with "empty results", so the timing feed had
> nothing to leak). But FIX_PLAN.md §8.6 counts only forecasts frozen *before*
> their outcome as prospective evidence, and this one was not.
> **Exclude this round when tallying prospective accuracy.**

#### Prediction vs result

| Pred | Driver | Team | Grid | p_win | p_podium | p_top10 | **Actual** | **Δ** |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LEC | ferrari | 2 | 35.9% | 82.5% | 100.0% | 4 | +3 |
| 2 | RUS | mercedes | 1 | 22.0% | 67.1% | >99.9% | 1 | -1 |
| 3 | PIA | mclaren | 3 | 18.0% | 59.0% | >99.9% | 13 | +10 |
| 4 | HAD | red_bull | 4 | 13.9% | 48.8% | >99.9% | 3 | -1 |
| 5 | NOR | mclaren | 5 | 2.7% | 11.3% | 91.7% | DNF | — |
| 6 | HAM | ferrari | 6 | 1.9% | 7.9% | 83.1% | 6 | 0 |
| 7 | VER | red_bull | 8 | 1.2% | 5.0% | 69.2% | 2 | -5 |
| 8 | GAS | alpine | 7 | 0.4% | 1.8% | 33.4% | DNF | — |
| 9 | SAI | williams | 9 | 0.4% | 1.6% | 30.0% | 10 | +1 |
| 10 | ANT | mercedes | 16 | 0.8% | 3.3% | 53.6% | 5 | -5 |
| 11 | COL | alpine | 10 | 0.3% | 1.2% | 24.7% | DNF | — |
| 12 | BEA | haas | 11 | 0.3% | 1.2% | 24.2% | 9 | -3 |
| 13 | LAW | racing_bulls | 12 | 0.3% | 1.1% | 22.5% | 12 | -1 |
| 14 | LIN | racing_bulls | 15 | 0.3% | 1.1% | 23.2% | 7 | -7 |
| 15 | OCO | haas | 14 | 0.2% | 1.0% | 18.9% | 8 | -7 |
| 16 | BOR | sauber | 17 | 0.2% | 1.0% | 19.2% | 15 | -1 |
| 17 | ALB | williams | 13 | 0.2% | 0.8% | 17.6% | DNF | — |
| 18 | HUL | sauber | 18 | 0.2% | 0.8% | 17.8% | 11 | -7 |
| 19 | ALO | aston_martin | 19 | 0.2% | 0.8% | 17.7% | DNF | — |
| 20 | PER | cadillac | 20 | 0.2% | 0.9% | 18.0% | 14 | -6 |
| 21 | STR | aston_martin | 21 | 0.2% | 0.8% | 17.8% | DNF | — |
| 22 | BOT | cadillac | 22 | 0.2% | 0.9% | 17.5% | 16 | -6 |

_Fill **Actual** with the classified finishing position (`DNF` if retired) and **Δ** with actual − predicted._

#### Scorecard — fill after the race

| Metric | Model | Grid baseline |
| --- | --- | --- |
| Winner hit | no (LEC) | yes (RUS) |
| Podium overlap | 1 / 3 | 1 / 3 |
| Top-10 overlap | 7 / 10 | 6 / 10 |
| Favourite's p_win | 35.9% (LEC) | — |
| Actual winner | RUS | |
| Actual podium | RUS · VER · HAD | |

#### Notes

- **Data was a race stale before this run.** `data/v2` ended at round 13 (Monza)
  even though round 14 had been run. Round 14 was ingested first, so the bundle
  is trained on 187 races to 2026-09-13 rather than 186 to 2026-09-06. Practice
  for rounds 14 and 15 was ingested and spliced into `practice.parquet`
  (186 races now); GATE 2 leakage audit re-run and passed.
- **The model disagrees with pole,** putting LEC (P2) ahead of RUS. Unlike round
  14's near-three-way toss (27.3 / 24.2 / 21.7), this is a clearer favourite at
  35.9% — the most conviction in the log so far.
- **HAD is the interesting call.** P4 on the grid, 13.9% win and 48.8% podium,
  yet the utility order leaves him off the predicted podium.
- **ANT is an order/probability disagreement.** 16th on the grid and 10th in
  utility order, but his 0.8% `p_win` and 53.6% `p_top10` exceed several drivers
  ranked above him. Per the stated policy the probabilities are the calibrated
  quantity where the two disagree.
- **Grid is PROVISIONAL,** derived from qualifying. Penalties applied after the
  session are not reflected. Baku is a street circuit where qualifying incidents
  and post-session penalties are common, so check the published grid before
  scoring.
- **Team naming:** the timing feed lists Audi; the pipeline maps it to `sauber`.
- **Published grid differed from the provisional one** (scored 2026-10-04): SAI P9→P14, COL P10→P9, BEA P11→P10, LAW P12→P11, OCO P14→P13, ALB P13→P12, ALO P19→P21, STR P21→P22, BOT P22→P19. Nothing changed in the top 8, so the winner and podium comparisons are unaffected. The grid baseline is scored against the provisional grid the forecast used.

---

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
| 1 | ANT | mercedes | 2 | 27.3% | 71.2% | >99.9% | 1 | 0 |
| 2 | NOR | mclaren | 1 | 24.2% | 67.2% | >99.9% | 3 | +1 |
| 3 | VER | red_bull | 3 | 21.7% | 62.7% | >99.9% | 2 | -1 |
| 4 | HAM | ferrari | 4 | 10.8% | 37.0% | 99.6% | DNF | — |
| 5 | LEC | ferrari | 5 | 4.3% | 16.6% | 93.6% | 4 | -1 |
| 6 | RUS | mercedes | 6 | 3.7% | 14.2% | 90.7% | 5 | -1 |
| 7 | PIA | mclaren | 7 | 2.0% | 7.5% | 74.9% | 8 | +1 |
| 8 | LAW | red_bull | 8 | 1.4% | 5.4% | 62.7% | 6 | -2 |
| 9 | LIN | racing_bulls | 10 | 0.7% | 2.6% | 39.3% | 9 | 0 |
| 10 | COL | alpine | 9 | 0.5% | 2.1% | 31.4% | 7 | -3 |
| 11 | HUL | sauber | 11 | 0.4% | 1.8% | 26.5% | 10 | -1 |
| 12 | BOR | sauber | 12 | 0.4% | 1.6% | 23.5% | 13 | +1 |
| 13 | SAI | williams | 17 | 0.3% | 1.1% | 16.7% | DNF | — |
| 14 | OCO | haas | 13 | 0.3% | 1.0% | 16.5% | 11 | -3 |
| 15 | ALB | williams | 16 | 0.3% | 1.2% | 15.7% | 15 | 0 |
| 16 | STR | aston_martin | 22 | 0.3% | 1.1% | 16.8% | DNF | — |
| 17 | PER | cadillac | 19 | 0.3% | 1.0% | 15.9% | DNF | — |
| 18 | BOT | cadillac | 20 | 0.3% | 1.0% | 15.9% | 18 | 0 |
| 19 | ALO | aston_martin | 18 | 0.2% | 0.9% | 15.2% | 17 | -2 |
| 20 | TSU | racing_bulls | 15 | 0.2% | 0.9% | 15.2% | 14 | -6 |
| 21 | GAS | alpine | 14 | 0.2% | 1.0% | 14.7% | 12 | -9 |
| 22 | BEA | haas | 21 | 0.2% | 1.1% | 15.4% | 16 | -6 |

_Fill **Actual** with the classified finishing position (`DNF` if retired) and **Δ** with actual − predicted._

#### Scorecard — fill after the race

| Metric | Model | Grid baseline |
| --- | --- | --- |
| Winner hit | yes (ANT) | no (NOR) |
| Podium overlap | 3 / 3 | 3 / 3 |
| Top-10 overlap | 9 / 10 | 9 / 10 |
| Favourite's p_win | 27.3% (ANT) | — |
| Actual winner | ANT | |
| Actual podium | ANT · VER · NOR | |

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
- **Published grid differed from the provisional one** (scored 2026-10-04): SAI P17→P20, STR P22→P21, PER P19→P18, BOT P20→P19, ALO P18→P17, BEA P21→P22. Nothing changed in the top 8, so the winner and podium comparisons are unaffected. The grid baseline is scored against the provisional grid the forecast used.

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
- [x] **Score round 14.** Scored 2026-10-04.
- [x] **Score round 15.** Scored 2026-10-04 (after-lights-out; see its entry).
- [ ] **Score round 16** once the Kuala Lumpur result is published, keeping the
      after-lights-out caveat attached to it.
