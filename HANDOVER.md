# HANDOVER

**Purpose:** cold-start context. If a session dies, paste this file to a fresh
Claude session and it should be able to continue without re-reading everything.

**Maintained by:** Claude. Update at the end of every working session and after
any milestone lands. Keep it current, not comprehensive — detail lives in
`REASONING.md` and `FIX_PLAN.md`.

**Last updated:** 2026-09-11 — session 1 (increments 0.1 and 1.1 done)

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

1. **Race-weather leakage.** `ingest.weather_row()` averages weather across the
   *race session*; those 6 cols are in `FEATURE_COLS`.
   `weather.add_weather_affinity()` also picks the historical temp bucket using
   the *target race's realized* track temp.
2. **Imputation fitted on all data.** `build_features.apply_fill_policy()`
   computes global means before `train.chronological_split()`.
3. **Train/serve skew.** Training fills position features driver-prior-then-
   global; `predict.py` applies only saved globals. 736 rows differ.
4. **Quali position treated as final grid.** `--from-quali` reads Q `Position`,
   ignores penalties; missing drivers silently get a hard-coded pit start at 20.
5. ~~**Label semantics.**~~ **FIXED in 1.1** — `src/labels.py`. Note the flag was
   99.90% constant (2078/2080), worse than FIX_PLAN recorded. Consumers not yet
   migrated (increment 1.2).
6. **Order-dependent output.** `rank(method='first')` ties break on row order;
   shuffling rows moves Spearman 0.5005 -> 0.5187.
7. **`top1_hit` is misnamed** — true when the top pick finishes anywhere in the
   top 10, not P1.

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
| 1 | Correct contracts & replay: weather, fitted state, labels, shared as-of path, deterministic ties | **1.1 DONE** (labels); 1.2-1.5 pending |
| 2 | Evaluation harness: `backtest.py`, `metrics.py`, real winner gates | NOT STARTED |
| 3 | Qualifying & car features: `qualifying.py`, `ratings.py` | NOT STARTED |
| 4 | Model comparison M0-M4 (+ Plackett-Luce, winner/podium heads) | NOT STARTED |
| 5 | Validated forecast output with model bundle + probabilities | NOT STARTED |
| 6 | Optional: practice pace, forecast archive, TabPFN, custom NN | NOT STARTED |

Milestones 1 and 2 go together — correctness first, then the harness to measure
it. No feature or model work before both land.

---

## 6. Standing decisions

- Fix correctness before chasing accuracy. Metrics are **expected to get worse**
  after P0 fixes, because today's numbers lean on information unavailable at
  prediction time. That is the correct outcome, not a regression.
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

**Session 1 (2026-09-11):** read the full repo and `FIX_PLAN.md`, verified the
plan's data claims against the parquet files. **No code changed.** Created
`HANDOVER.md`, `REASONING.md`, `INSTRUCTIONS.md`.

**Open questions put to the user, not yet answered:**
1. Do milestone 0 (preserve artifacts + `reports/baseline.json`) before touching
   anything, so we can prove what changed?

**Resolved:** git workflow — Claude never runs git commands that change history
or remote state. Changes are proposed as discrete increments with a suggested
commit message; the user executes all git operations. See `INSTRUCTIONS.md`.

**Increment 0.1 complete (uncommitted, awaiting user review):**
- `src/baseline.py` (new) -> `reports/baseline.json`
- `.gitignore` rewritten so JSON provenance is trackable (`dir/*` + negation)
- `requirements.lock.txt` (new)
- Reproduces FIX_PLAN section 2's table exactly; all 5 review hashes match
- Row-order defect quantified: max Spearman spread 0.0080 over 5 shuffles

**Increment 1.1 complete (uncommitted):** `src/labels.py`, `tests/test_labels.py`
(53 tests), `pytest.ini`, `requirements-dev.txt`; `ingest.py` wired up.
Backfill: `python -m src.labels` -> `data/raw_results_labeled.parquet`.
`finished_top10` changed on 0 rows; frozen artifact hash unchanged.

**Next action:** increment 1.2 — migrate downstream consumers off the
deprecated `classified` flag (`train_rank.py` relevance, `evaluate.py` and
`evaluate_rank.py` Spearman filters, `columns.py` ID_COLS). This DOES change
model inputs and metric denominators, so `reports/baseline.json` is the
comparison point.

---

## 8. Update protocol

At the end of each session, update: §3 if data changed, §5 milestone statuses,
§7 status/next action, and the "Last updated" line. Log the *why* of every
change in `REASONING.md` — this file records state, that one records reasoning.
