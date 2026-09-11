# HANDOVER

**Purpose:** cold-start context. If a session dies, paste this file to a fresh
Claude session and it should be able to continue without re-reading everything.

**Maintained by:** Claude. Update at the end of every working session and after
any milestone lands. Keep it current, not comprehensive — detail lives in
`REASONING.md` and `FIX_PLAN.md`.

**Last updated:** 2026-09-11 — session 1 (milestones 0-2 complete; 3.1 done, NEGATIVE result)

---

## 1. Where things are

| Thing | Path |
| --- | --- |
| Project root (git root) | `C:\Users\micha\Documents\f1-predictor\f1-predictor` |
| Python | `./.venv/Scripts/python.exe` (3.12.10) |
| Run modules as | `python -m src.<name>` from project root |
| The diagnosis/plan | `FIX_PLAN.md` (authoritative; produced by Codex review) |
| Decision log | `REASONING.md` |
| Response rules | `INSTRUCTIONS.md` (NOT auto-loaded — only `CLAUDE.md` is) |
| Frozen legacy baseline | `reports/baseline.json` (regenerate: `python -m src.baseline`) |

Outer folder `C:\Users\micha\Documents\f1-predictor\` is just a wrapper — the
real project is the nested `f1-predictor\` directory. Do not confuse them.

Key deps: fastf1 3.8.3, lightgbm 4.6.0, pandas 2.3.3, scikit-learn 1.9.0,
numpy 2.5.0.

---

## 2. What the project is

Pre-race F1 predictor on FastF1 data. Forecast is made **after qualifying,
before the race**. The goal is winner and podium accuracy, with full-field
ordering retained as a guardrail.

Pipeline (each step a module in `src/`):

```
ingest -> build_features -> audit_leakage (Gate 2) -> train / train_rank
       -> evaluate / evaluate_rank -> blend_rank -> predict
```

Three models, all sharing the 30 features in `src/columns.py`:

- `train.py` — LightGBM binary classifier, P(finish top 10)
- `train_rank.py` — LightGBM LambdaRank over race groups, finishing order
- `blend_rank.py` — `alpha*grid_rank + (1-alpha)*model_rank`, alpha = 0.6

---

## 3. Verified state of the data (checked 2026-09-11)

**`data/v2/raw_results.parquet` is the current dataset: 3,744 rows, 186 races,
2018-2026.** Every season complete, no missing rounds, zero unknown statuses,
and all rows carry `officially_classified` from the authoritative FastF1
`ClassifiedPosition`.

| Season | Rows | Races | | Season | Rows | Races |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| 2018 | 420 | 21 | | 2023 | 440 | 22 |
| 2019 | 420 | 21 | | 2024 | 479 | 24 |
| 2020 | 340 | 17 | | 2025 | 479 | 24 |
| 2021 | 440 | 22 | | 2026 | 286 | 13 |
| 2022 | 440 | 22 | | | | |

DNF rate 0.1536, target mean 0.4968. Field sizes 19-22.

`data/raw_results.parquet` (frozen legacy) is the OLD 103-race 2022-2026
dataset. It is kept only so `reports/baseline.json` stays reproducible; do not
use it for new work.

### Superseded

`reports/baseline.json` describes 103 races of 2022-2026 and is **no longer
comparable** to the corrected pipeline, which now runs on 186 races of
2018-2026. The legacy-vs-corrected comparisons recorded in REASONING entries
[006] and [007] are superseded.

## 4. Faults to fix (condensed from FIX_PLAN.md §2)

**P0 — invalidate every current number:**

1. ~~**Race-weather leakage.**~~ **FIXED in 1.4** - 8 features removed
   (30 -> 22). `driver_wet_*` verified clean and kept.
2. ~~**Imputation fitted on all data.**~~ **FIXED in 1.5** -
   `preprocessing.FillPolicy` fits on the training partition only
   (mean finish 10.4790, not 10.5982).
3. ~~**Train/serve skew.**~~ **FIXED in 1.6** - one shared
   `features.build_asof_features()`. Replay parity test proves it.
4. ~~**Quali position treated as final grid.**~~ **FIXED in 1.7** - `src/grid.py`
   separates qualifying from grid, requires explicit pit starts, uses real field
   size, and labels every snapshot confirmed/provisional/unavailable.
5. ~~**Label semantics.**~~ **FIXED in 1.1** — `src/labels.py`. Note the flag was
   99.90% constant (2078/2080), worse than FIX_PLAN recorded. Consumers not yet
   migrated (increment 1.2).
6. ~~**Order-dependent output.**~~ **FIXED in 1.2** - deterministic tie policy
   (score, grid, driver id) in `src/metrics.py`, applied to metrics AND to the
   blend/inference path. Shuffle spread now exactly 0.0 (was 0.0080).
7. ~~**`top1_hit` is misnamed**~~ **FIXED in 1.2** — now
   `top_pick_finished_top10`.

**P1 — why it does not beat the baseline:**

8. Current qualifying pace ingested then discarded (`quali_best_s`,
   `gap_to_pole_s` not in `FEATURE_COLS`).
9. Misleading names: `form_avg_quali_3` averages prior *grid positions*;
   `quali_gap_to_teammate` compares those averages, not lap times;
   `constructor_standing_prior` is cumulative points, omits sprint points.
10. Blend selects alpha by Spearman, not by winner/podium.
11. All-history circuit/reliability stats, no recency weighting or shrinkage.
12. No model bundle/manifest binding data hash + features + fitted state +
    cutoff + model.

Note: `audit_leakage.py` is good but only covers historical aggregation. Its
"leakage disproven" banner overclaims — it never tested weather availability,
fitted imputation state, or serving parity.

---

## 5. Milestone plan (FIX_PLAN.md §9)

| # | Milestone | Status |
| --- | --- | --- |
| 0 | Preserve & reproduce: baseline.json, manifest, dep lock, README refresh | **0.1 DONE** (README refresh deferred to M1) |
| 1 | Correct contracts & replay: weather, fitted state, labels, shared as-of path, deterministic ties | **DONE** (1.1-1.7). All five P0 defects closed |
| 2 | Evaluation harness: `backtest.py`, `metrics.py`, real winner gates | **DONE** - 8 folds / 47 races; gates exit non-zero |
| 3 | Qualifying & car features: `qualifying.py`, `ratings.py` | **3.1 DONE, no gain** (see below); ratings/recency pending |
| 4 | Model comparison M0-M4 (+ Plackett-Luce, winner/podium heads) | NOT STARTED |
| 5 | Validated forecast output with model bundle + probabilities | NOT STARTED |
| 6 | Optional: practice pace, forecast archive, TabPFN, custom NN | NOT STARTED |

Milestones 1 and 2 go together — correctness first, then the harness to measure
it. No feature or model work before both land.

---

## 6. Standing decisions

- Fix correctness before chasing accuracy. A metric drop after a P0 fix is an
  acceptable outcome, not a regression - but do NOT assume one is coming.
  Increment 1.4 removed the weather leak and test metrics IMPROVED (classifier
  test AUC 0.7908 -> 0.8178) because the leaky features were costing more in
  overfitting than they returned. Measure; never predict the direction.
- No hand-coded driver exceptions, ever.
- A transformer is an experiment, not a prerequisite. 103 race groups is far too
  little to train one from scratch. TabPFN (pretrained) is the first and only
  justified transformer trial, and only after M0-M4.
- Never randomly split driver rows — chronological event blocks only.
- Synthetic data must never be written to the real raw-data path.
- Work in small increments: 1-2 related features each, one clean commit's worth.
  Split anything larger into sub-increments.

---

## 7. Current status / next action

**Session 1 (2026-09-11).** `REASONING.md` carries the detail. Nothing is
committed by Claude - the user runs all git commands.

### Completed increments

| # | What | Effect on the held-out season |
| --- | --- | --- |
| 0.1 | `src/baseline.py` -> `reports/baseline.json`; `.gitignore` fixed; `requirements.lock.txt` | records the starting point |
| 1.1 | `src/labels.py` - separated result concepts | none (`finished_top10` changed on 0 rows) |
| 1.2 | `src/metrics.py` - deterministic ties, explicit labels, split denominators | row-shuffle spread 0.0080 -> **0** |
| 1.3 | Ranker relevance on `officially_classified` | none measurable (1.8e-07) |
| 1.4 | Removed 8 leaking weather features (30 -> 22) | classifier AUC **up** 0.7908 -> 0.8178 |
| 1.5+1.6 | `preprocessing.FillPolicy` fitted on train only; `features.build_asof_features()` as the single path; README rewritten | ranker/blend Spearman **down**; classifier still up |
| 1.7 | `src/grid.py` - qualifying vs grid separated, explicit pit starts, real field size, grid status | none (no retrain needed) |

**Milestones 0, 1 and 2 are complete.**

| # | What | Effect |
| --- | --- | --- |
| 2 | `src/backtest.py` expanding-window folds + `src/gates.py` paired intervals | **changed the headline finding - see below** |

Test suite: **149 tests**, `python -m pytest`.

### Artifact layout (IMPORTANT)

- `data/`, `models/` - **frozen legacy**, hashed in `reports/baseline.json`.
  Never overwrite. `src/baseline.py` reproduces the legacy numbers from these
  and is pinned three ways (private metric copies, private blend copy, and the
  frozen `models/feature_cols.json`).
- `data/v2/`, `models/v2/` - **current corrected pipeline**. A working area, not
  a version history.

Full rebuild from the recovered data:

```
python -m src.merge_raw data/v2/raw_2018_2019.parquet \
                        data/v2/raw_2020_2021.parquet \
                        data/v2/raw_2022_2026.parquet \
                        --out data/v2/raw_results.parquet
python -m src.build_features --raw data/v2/raw_results.parquet --out-dir data/v2
python -m src.audit_leakage  --data-dir data/v2 --raw data/v2/raw_results.parquet
python -m src.backtest --scheme season --out-dir reports/backtest_season
python -m src.gates    --backtest-dir reports/backtest_season
```

NOTE the environment HAS network access to FastF1. Jolpica rate-limits at 500
calls/hour; ingestion now aborts on that rather than skipping, and cached races
cost no calls, so a resume is cheap.

### THE HEADLINE FINDING (corrected on full history)

Backtested over **6 season folds / 127 races (2018-2026)**, paired against the
grid baseline with 95% intervals from 10,000 race-level bootstrap resamples:

| method | winner | podium | top-10 | spearman (all) | spearman (finishers) |
| --- | ---: | ---: | ---: | ---: | ---: |
| grid_baseline | 0.5591 | 0.6693 | 0.7701 | 0.6305 | 0.7637 |
| top10_classifier | 0.3543 | 0.5774 | 0.7866 | 0.6624 | 0.7999 |
| rank_model | 0.5591 | 0.6483 | 0.7764 | 0.6419 | 0.7901 |
| blend | 0.5669 | 0.6667 | 0.7843 | 0.6622 | 0.8029 |

Paired vs grid, winner accuracy:

- blend **+0.0079** [-0.0236, +0.0394] - does NOT resolve
- rank_model **+0.0000** [-0.0945, +0.0945] - does NOT resolve
- top10_classifier **-0.2047** [-0.3150, -0.0945] - resolves, worse

**The blend is statistically indistinguishable from the grid baseline at
picking winners.** It DOES resolve as better on the guardrails: top-10 overlap
+0.0142 [+0.0039, +0.0244] and spearman +0.0317 [+0.0200, +0.0436].

### RETRACTED

An earlier backtest (8 rolling folds, 47 races, 2022-2026 only) reported the
blend as **significantly worse** at picking winners (-0.0851, CI excluding
zero). **That does not replicate on 186 races.** Running the same rolling scheme
on the full data gives +0.0154 [-0.0231, +0.0538], so the cause was the narrow
DATA WINDOW, not the fold scheme.

Note grid winner accuracy is 0.5591 over 127 races against 0.7273 over the 11
races of 2026 alone - that season was unusually grid-predictable. Two confident
conclusions in this project have now come from too-small samples. Treat "the
interval excludes zero" as necessary but not sufficient when the window is
narrow.

`python -m src.gates` still exits 1, but the reason changed: the VOLUME gates
now pass (6 folds, 127 races) and the failures are the winner/podium thresholds
themselves.

Reproduce:

```
python -m src.backtest --scheme season --out-dir reports/backtest_season
python -m src.gates --backtest-dir reports/backtest_season
```

### Guards in place

- `src/baseline.py` compares, and needs `--force` to overwrite.
- `src/blend_rank.py` needs `--write` to save an alpha.
- `tests/test_columns.py` fails if race-session weather re-enters `FEATURE_COLS`.
- `tests/test_parity.py` fails if training and serving features diverge, or if
  any feature reads its own race's outcome.
- `FillPolicy.from_json` refuses an unknown `schema_version`.

### Open questions for the user

1. Re-ingesting 2018-2021 needs network access (Claude has none). It roughly
   triples the data and is a prerequisite for the expanding-window backtests in
   FIX_PLAN.md section 8.
2. Milestone 2 (evaluation harness) before milestone 3 (qualifying features)?
   FIX_PLAN.md says yes - without paired intervals over multiple folds, an
   11-race test season cannot tell whether a new feature helped.

### Increment 3.1 result: qualifying pace does NOT help

Seven per-segment qualifying-pace features added (22 -> 29 features). On the
same 127 races, paired:

- blend winner **+0.0000** [-0.0394, +0.0394]
- rank_model winner **+0.0000** [-0.0551, +0.0551]
- nothing resolves; blend podium and spearman are marginally WORSE
- 2026 specifically: blend 10/13 winners -> 9/13

The model uses them heavily - 38.6% of ranker gain, with
`quali_gap_to_median_pct` second overall at 31.3% - but outcomes do not move.
Diagnosed:

    corr(grid_position, result_order)  = 0.627
    corr(quali_gap_pct, result_order)  = 0.496
    corr(grid_position, quali_gap_pct) = 0.658

`grid_position` IS the qualifying result with penalties applied, it is the
better predictor, and the new features are 0.66-correlated with it. They are
**substitutes, not complements**. The finding: **the MAGNITUDE of a qualifying
gap does not predict race result beyond qualifying ORDER.**

Keep the features (leakage-free, may combine with race-pace features later) but
do not call them an improvement. `quali_stage_reached` and `quali_no_time`
contribute 0.00% gain and are removal candidates.

### Next action

FIX_PLAN.md nominated unused qualifying pace as the most obvious missing
signal. It has now been added properly and is not the answer. The gap between
the models and the grid baseline is NOT a qualifying-information gap.

What remains untried, in order of expected value:

1. **Race pace.** Nothing in the feature set measures how fast a car is over a
   stint, only where it started and where it historically finished. Practice
   long-run stints are the natural source (FIX_PLAN.md section 5.D: robust
   long-run lap pace, stint consistency, tyre-age slope, excluding in/out and
   deleted laps). This needs new ingestion.
2. **Recency-weighted car form** (section 5.C). `constructor_standing_prior` is
   cumulative season points - lagging, badly scaled, and it omits sprint points.
   Exponentially weighted recent team pace is the prescribed replacement.
3. **Specialist winner/podium objectives** (section 6, M4). Every current model
   optimises a top-10 flag or full-field order; none optimises the thing being
   measured. The blend's alpha is still selected by Spearman.

Note the backtest harness makes each of these a one-command measurement, which
is what milestone 2 was for.

## 8. Update protocol

At the end of each session, update: §3 if data changed, §5 milestone statuses,
§7 status/next action, and the "Last updated" line. Log the *why* of every
change in `REASONING.md` — this file records state, that one records reasoning.
