# HANDOVER

**Purpose:** cold-start context. If a session dies, paste this file to a fresh
Claude session and it should be able to continue without re-reading everything.

**Maintained by:** Claude. Update at the end of every working session and after
any milestone lands. Keep it current, not comprehensive — detail lives in
`REASONING.md` and `FIX_PLAN.md`.

**Last updated:** 2026-09-11 — session 1 (milestones 0, 1 and 2 complete)

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

## 3. Verified state of the data (checked 2026-09-11, not just claimed)

- `data/raw_results.parquet` — 2,080 rows x 24 cols
- `data/features.parquet` — 2,080 rows x 47 cols, 30 model features
- Seasons **2022-2026** (NOT 2018+ as README claims), 103 races,
  last race 2026-07-26
- Split: train 2022-24 (1,359 rows / 68 races), val 2025, test 2026 (11 races)
- Real driver codes (LEC, VER, ...) — this is real data, not `make_synthetic`
- `quali_best_s` 98.9% populated **and unused** (not in `FEATURE_COLS`)
- 303 rows have `is_dnf=1` AND `classified=1`, incl. 16 DNS and 10 Disqualified

### The headline problem

The blend picks **exactly the same winners as the raw starting grid** (8/11 on
the 2026 test races). Podium overlap identical. The ML stack currently adds
nothing over "sort by grid position" for winner/podium.

| Ordering method | Winner | Podium overlap | Top-10 overlap | Spearman |
| --- | ---: | ---: | ---: | ---: |
| Starting grid | 8/11 | 57.6% | 73.6% | 0.6432 |
| Top-10 classifier | 3/11 | 48.5% | 75.5% | 0.5672 |
| LambdaRank alone | 6/11 | 57.6% | 69.1% | 0.5005 |
| Saved blend (a=0.6) | 8/11 | 57.6% | 75.5% | 0.6289 |

The "~85%" figure that circulated is the classifier's **validation AUC
(0.8455)**, not accuracy of finishing positions.

---

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
| 3 | Qualifying & car features: `qualifying.py`, `ratings.py` | NOT STARTED |
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

Full rebuild command sequence is in `README.md`.

### THE HEADLINE FINDING (changed by milestone 2)

Backtested over **8 folds / 47 races**, paired against the grid baseline with
95% intervals from 10,000 race-level bootstrap resamples:

| candidate | winner accuracy vs grid | resolves? |
| --- | ---: | --- |
| blend | **-0.0851** [-0.1702, -0.0213] | YES |
| rank_model | **-0.2128** [-0.3404, -0.0851] | YES |
| top10_classifier | **-0.4468** [-0.6383, -0.2553] | YES |

Pooled winner accuracy: grid **0.6170**, blend 0.5319, rank_model 0.4043,
classifier 0.1702.

**The models are not merely no better than the grid - they are significantly
WORSE at picking winners.** The single 11-race season showed the blend tying
grid at 8/11, which read as "adds nothing". Over 47 races the sign resolves and
the interval excludes zero. That tie was a small-sample artefact.

The blend does buy better ordering: spearman +0.0184 [+0.0040, +0.0333]
(resolves), podium +0.0142 [-0.0213, +0.0496] (does not). It trades winner
accuracy for Spearman - exactly the objective mismatch FIX_PLAN.md flags as P1,
since alpha is SELECTED by Spearman.

`python -m src.gates` exits 1. No candidate passes.

Reproduce:

```
python -m src.backtest --scheme rolling
python -m src.gates
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

### Next action

**Re-ingest 2018-2021 first.** This is now the highest-value pending task, and
the environment DOES have network access.

- 47 evaluated races is below FIX_PLAN.md section 8's 60-race target, so the
  race-count gate fails for every candidate and no promotion decision can be
  made on the current data at all.
- It also enables season folds (currently only 2 possible) instead of the
  partial-season rolling blocks.
- Expect it to be slow: roughly 4 seasons x 21 races x 2 sessions of FastF1
  downloads. Run `python -m src.ingest --start-year 2018 --end-year 2021`,
  then rebuild and re-backtest.

**Then milestone 3: qualifying and car features.** `quali_best_s` and
`gap_to_pole_s` are ingested, 98.9% populated and unused - the single most
obvious missing signal for a post-qualifying forecast. The backtest harness now
exists to measure whether they help.

Also worth doing early, given the finding above: alpha is selected by Spearman
while the objective is winner/podium (FIX_PLAN.md P1). The backtest shows this
mismatch is not theoretical - the blend buys Spearman by giving up winner
accuracy.

## 8. Update protocol

At the end of each session, update: §3 if data changed, §5 milestone statuses,
§7 status/next action, and the "Last updated" line. Log the *why* of every
change in `REASONING.md` — this file records state, that one records reasoning.
