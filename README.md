# F1 race-result predictor

Forecasts a Formula 1 race **after qualifying, before the start**. The goal is
winner and podium accuracy; full-field ordering is kept as a guardrail.

Data comes from [FastF1](https://docs.fastf1.dev/). Models are LightGBM. The
governing rule is that every feature must be computable from information
available at the forecast cutoff — not merely from prior races. Those are
different guarantees, and the gap between them hid a real leak (see
[Corrections](#corrections-applied)).

> **Status.** Milestones 0–5 of [`FIX_PLAN.md`](FIX_PLAN.md) are complete;
> milestone 6 was measured and is mostly negative. The champion model's
> **order** is not distinguishable from sorting by the starting grid at the
> FIX_PLAN §8 margin. Its **probabilities** are: winner log loss 1.15 against
> the grid baseline's 1.66, interval excluding zero. Read the probabilities as
> the useful output and the order as roughly grid-equivalent. `predict.py`
> prints that distinction on every forecast.

Session state and next actions live in [`HANDOVER.md`](HANDOVER.md); the reason
for every decision is in [`REASONING.md`](REASONING.md).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
```

`requirements.lock.txt` pins the exact environment that produced
`reports/baseline.json`. Python 3.12 is what this is developed on.

Run everything from the project root as modules — `src` is a package and uses
relative imports.

`data/`, `models/` and `reports/` are untracked generated artifacts, with the
single exception of `reports/baseline.json`. A fresh clone has no numbers in
it; the commands below regenerate them.

## Artifact layout

Two parallel sets, and the distinction matters:

| Location | What | Rule |
| --- | --- | --- |
| `data/`, `models/` | **Frozen legacy** artifacts, hashed in `reports/baseline.json` | Never overwrite. They are the evidence of what the project scored before the corrections. |
| `data/v2/`, `models/v2/` | **Current corrected pipeline** | Rebuilt by each increment. A working area, not a version history. |
| `models/champion/` | **The deployable bundle** — model, fitted imputation state, calibration, manifest | Produced by `src.build_bundle`. Keeps no rollback copy yet. |

`data/raw_results.parquet` is the **old** 103-race 2022–2026 table, kept only so
`reports/baseline.json` stays reproducible. New work uses
`data/v2/raw_results.parquet`.

`reports/baseline.json` describes that 103-race window and is **no longer
comparable** to the corrected pipeline, which runs on 186 races.

## Data actually present

**2018–2026**, 3,744 driver-race rows, 186 races, last race 2026-09-06. Real
FastF1 data, not synthetic. Every season complete, no missing rounds, no unknown
statuses; `officially_classified` comes from FastF1's `ClassifiedPosition`. DNF
rate 0.1536, field sizes 19–22.

| Season | Rows | Races | | Season | Rows | Races |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| 2018 | 420 | 21 | | 2023 | 440 | 22 |
| 2019 | 420 | 21 | | 2024 | 479 | 24 |
| 2020 | 340 | 17 | | 2025 | 479 | 24 |
| 2021 | 440 | 22 | | 2026 | 286 | 13 |
| 2022 | 440 | 22 | | | | |

For one-shot training the chronological split is train ≤ 2024, validation 2025,
test 2026. **That split is for development only.** Every promotion decision is
made on the backtest below.

## Running the pipeline

Rebuild from the ingested season ranges:

```bash
python -m src.merge_raw data/v2/raw_2018_2019.parquet \
                        data/v2/raw_2020_2021.parquet \
                        data/v2/raw_2022_2026.parquet \
                        --out data/v2/raw_results.parquet
python -m src.build_features --raw data/v2/raw_results.parquet --out-dir data/v2
python -m src.audit_leakage  --data-dir data/v2 --raw data/v2/raw_results.parquet   # GATE 2
```

Evaluate — this, not a single held-out season, is what decides anything:

```bash
python -m src.backtest --scheme season --out-dir reports/backtest_final
python -m src.gates    --backtest-dir reports/backtest_final     # exits non-zero on failure
```

Build the deployable bundle and forecast a race:

```bash
python -m src.build_bundle                                       # -> models/champion
python -m src.predict --year 2026 --round 14 --from-quali --archive
python -m src.score_forecasts                                    # once results are ingested
```

`--archive` writes an immutable record to `reports/forecasts/` binding the
prediction to its grid status and the exact bundle. That is the prospective
evidence FIX_PLAN.md §8.6 requires and nothing else provides.

Single-shot training and the older evaluation entry points still work
(`src.train`, `src.train_rank`, `src.blend_rank --write`, `src.evaluate_rank`),
but `blend_rank.py` still picks alpha by validation Spearman and says so; the
backtest picks it by winner accuracy.

Re-ingesting needs network access to FastF1's endpoints:

```bash
python -m src.ingest --start-year 2018 --end-year 2026 --out data/v2/raw_2018_2026.parquet
python -m src.ingest_practice          # optional, expensive, one session load per race
```

Jolpica rate-limits at 500 calls/hour; ingestion aborts on that rather than
skipping, and cached races cost no calls, so a resume is cheap.

`python -m src.baseline` re-measures the frozen legacy artifacts and *compares*
against `reports/baseline.json` rather than overwriting it. It needs `--force`
to replace the record — deliberately, so the pre-correction reference cannot be
destroyed by a routine re-run.

`python -m pytest` — 267 tests.

## The headline result

Backtested over **6 expanding-window season folds / 127 races** (test seasons
2021–2026), paired against the grid baseline with 95% intervals from 10,000
race-level bootstrap resamples:

| Method | Winner | Podium | Top-10 | Spearman (all) | Spearman (finishers) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Starting grid | 0.5591 | 0.6693 | 0.7701 | 0.6305 | 0.7637 |
| Top-10 classifier | 0.5354 | 0.6719 | 0.7811 | 0.6425 | 0.7890 |
| **Ranker (champion)** | **0.6063** | 0.6719 | 0.7811 | 0.6487 | 0.7941 |
| Blend (alpha per fold) | 0.5984 | **0.6745** | 0.7787 | **0.6570** | **0.8029** |

Paired against the grid, winner accuracy:

| Method | Difference | 95% interval | Resolves? |
| --- | ---: | --- | --- |
| Ranker | +0.0472 | [−0.0315, +0.1260] | no |
| Blend | +0.0394 | [−0.0394, +0.1181] | no |
| Top-10 classifier | −0.0236 | [−0.0630, +0.0079] | no |

What *does* resolve is ordering quality: Spearman over finishers is better for
all three (ranker +0.0304 [+0.0086, +0.0523]), and the blend also resolves on
Spearman over all rows (+0.0265 [+0.0105, +0.0424]).

`src.gates` still exits 1. The reason is no longer volume — 6 folds and 127
races clear those checks — but the winner and podium thresholds themselves
(+0.05 and +0.03).

### Probabilities: the part that passes

The champion adds a Plackett-Luce layer over the ranker scores, temperature-
calibrated per fold. All four probability gates pass:

| Gate | Result |
| --- | --- |
| Winner log loss beats grid | 1.1469 vs 1.6568, paired −0.5099 [−0.8659, −0.2330] — **resolves** |
| Podium Brier worsens ≤ 0.01 | 0.07083 vs 0.07108 |
| Probabilities coherent | 0 problems over 127 races |
| Beats the uniform floor | 1.1469 vs 3.0047 |

Calibration is mildly conservative: the favourite's mean `p_win` is 0.5619
against an actual win rate of 0.6063.

`models/champion` is the ranker (38 features, 65 iterations, temperature 0.344,
blend alpha 0.3) fitted on all 186 races through 2026-09-06, with the data hash,
feature schema, imputation-policy version and training cutoff in its manifest.
The bundle refuses to score a race at or before its own cutoff — the one failure
that would look like excellent accuracy and be pure hindsight. The specialist
winner/podium head ensemble is a **challenger and is deliberately excluded**; it
scored worse.

### Two retractions, same cause

An earlier README reported, from the **11 races of 2026 alone**, that the grid
wins 0.7273 and "the machine learning adds nothing." An earlier backtest
reported, from **47 races of 2022–2026**, that the blend was significantly
*worse* at picking winners (−0.0851, interval excluding zero). Neither
replicates on 186 races. Grid winner accuracy is 0.5591 across 127 backtested
races — 2026 was simply an unusually grid-predictable season — and re-running
that same rolling scheme on the full data gives +0.0154 [−0.0231, +0.0538].

The cause in both cases was the **data window**, not the method. Treat "the
interval excludes zero" as necessary but not sufficient when the window is
narrow.

If you have seen "~85%" quoted for this project, that was the classifier's
validation AUC, not accuracy of finishing positions.

## Where things live

| Module | Role |
| --- | --- |
| `columns.py` | Single source of truth: `FEATURE_COLS` (38), `ID_COLS`, `TARGET`, `WEATHER_REMOVED` |
| `labels.py` | Result vocabulary — `result_order`, `officially_classified`, `started`, `finished`, `status_category` |
| `leakage.py` | The `shift(1)` and past-only EWM helpers every historical aggregation routes through |
| `features.py` | **The single as-of feature path**, shared by training and inference |
| `qualifying.py` | Current-weekend qualifying pace, normalised per segment |
| `ratings.py` | Recency-weighted driver and team race-pace ratings |
| `practice.py`, `ingest_practice.py` | Practice long-run pace. Measured **worse**; data retained, features excluded |
| `preprocessing.py` | `FillPolicy` — imputation fitted on training rows, frozen, replayed at serving |
| `splits.py` | Chronological partitioning |
| `grid.py` | The starting-grid contract: qualifying order vs grid, pit starts, grid status |
| `metrics.py` | Race metrics + the deterministic tie policy |
| `probabilities.py` | Plackett-Luce win/podium/top-10 probabilities and coherence checks |
| `heads.py` | Specialist winner/podium heads — challenger, not promoted |
| `backtest.py` | Expanding-window folds (season and rolling schemes), per-fold alpha and temperature |
| `gates.py` | Promotion gates and paired bootstrap intervals |
| `bundle.py`, `build_bundle.py` | Immutable model bundles; cutoff enforcement |
| `score_forecasts.py` | Scores archived pre-race forecasts once results arrive |
| `weather.py`, `circuit.py`, `teammate.py` | Feature families |
| `ingest.py`, `merge_raw.py` | FastF1 → raw parquet, team canonicalisation, Gate 1, coverage checks |
| `audit_leakage.py` | Gate 2 — independent naive recomputation |
| `baseline.py` | Frozen legacy record (`reports/baseline.json`) |
| `build_features.py`, `train.py`, `train_rank.py`, `blend_rank.py`, `evaluate*.py`, `predict.py` | Pipeline entry points |
| `make_synthetic.py` | Test harness only. **Not real data** |

## Corrections applied

Detailed reasoning for each is in [`REASONING.md`](REASONING.md).

- **Label semantics.** The inherited `classified = notna(Position)` was true for
  2,078 of 2,080 rows, so it could not distinguish a winner from a non-starter.
  Split into independent fields. `result_order` preserves the published
  classification — retirements keep their real places rather than collapsing to
  last.
- **Deterministic ordering.** `rank(method='first')` broke ties by row order, so
  the same entry list in a different order produced a different predicted
  podium. Ties now break on score → grid → driver id, all pre-race information.
  Row-shuffle Spearman spread: 0.0080 → **0**.
- **Metric honesty.** `top1_hit_rate` (true when the top pick finished anywhere
  in the top ten) renamed to `top_pick_finished_top10`. The single ambiguous
  `spearman` split into `spearman_all` and `spearman_finishers`, each with its
  own denominator.
- **Race-weather leakage.** Eight features removed. Six were averaged over the
  *race session*. The other two, `driver_temp_bin_*`, were subtler: their values
  were historical, so Gate 2 passed them, but the *bucket selection* used the
  target race's realized track temperature. `driver_wet_*` was checked and
  **kept** — it counts prior wet races and does not reveal the target's
  conditions. Removing the eight *improved* test AUC (0.7908 → 0.8178).
- **Fitted preprocessing.** Imputation constants were computed over the whole
  frame before the split. Now fitted on the 2,979 training rows only — mean
  finish 10.4872, where a full-data fit would have given 10.5638.
- **One feature path.** Training and inference each built their own feature
  vector and disagreed on 2,091 cells. Both now call
  `features.build_asof_features()`.
- **Grid contract.** `predict --from-quali` read the qualifying session's
  position as the grid, ignoring penalties and pit-lane starts, and hard-coded
  absent drivers to 20. `grid.py` now separates the two, requires pit starts to
  be explicit, derives back-of-grid from the real field size, and labels every
  snapshot confirmed / provisional / assumed.
- **Evaluation.** `backtest.py` + `gates.py` replaced eyeballing a single
  season, and exit non-zero when a gate fails.
- **Model bundles.** `bundle.py` binds data hash, feature schema, imputation
  state, calibration and training cutoff to the model, and refuses to predict a
  race it could have trained on.
- **Qualifying pace is now used.** Seven per-segment features are live — see
  below for what that did and did not buy.
- **Early stopping.** A bug in the fold fitting was worth +0.008 winner accuracy
  on its own — more than every feature added in milestone 3.

## Experiments measured

| Experiment | Result |
| --- | --- |
| Race pace from green-flag laps + recency ratings (9 features) | **kept**, winner 0.5591 → 0.5984 |
| Alpha selected per fold by winner accuracy instead of Spearman | **kept**, Spearman-selected alpha was costing ~4pp |
| Plackett-Luce probabilities + temperature calibration | **kept**, first gates ever passed |
| Early-stopping fix | **kept**, +0.008 winner |
| Per-segment qualifying pace (7 features) | no gain — `grid_position` *is* the qualifying order (corr 0.66); retained, harmless |
| Specialist winner/podium heads | negative; weight selection overfits the validation block |
| Bounded hyperparameter tuning (`--tune`) | no effect; chosen leaf counts vary 7/31/7/31/15/7 across folds |
| Practice long-run pace (5 features, 184 races) | **significantly worse**, winner −0.0787 [−0.1417, −0.0157]. Removed |
| Recent field-normalised places gained | negative, winner 0.6063 → 0.5669 |
| Current teammate grid position | inconclusive; slight probability gain, intervals span zero |
| Same-weekend sprint result and pace (29 events) | negative, winner 0.6063 → 0.5748 |

The pattern is the single most useful thing this project has learned: **on 127
races, anything selected on a ~22-race validation block fits noise.** The
per-fold blend alpha swings 0.6 / 0.2 / 0.0 / 0.1 / 0.3 / 0.3 across the six
folds for exactly that reason. Only genuinely new information or a removed
defect has survived. These negatives are conditional on the sample size, not
permanent; `--tune` and `heads.py` stay in the tree so they can be re-measured
rather than re-argued.

Forecasting *before* qualifying is possible and measurably much worse: over 61
races, winner accuracy 0.5902 → 0.2787 and podium overlap 0.6831 → 0.4809
against a real grid. `predict.py` prints that cost when it has to assume a grid.

## Still not fixed

- `form_avg_quali_3` averages prior *starting grids*, not qualifying pace, and
  `quali_gap_to_teammate` compares those averages. Both names lie.
- `constructor_standing_prior` is cumulative points, not a standings rank, and
  omits sprint points.
- `blend_rank.py` still selects alpha by validation Spearman. The backtest does
  not; the standalone script has not been migrated.
- `models/champion` keeps no previous champion for rollback.
- Displayed order and probabilities can disagree row for row. Stated in the
  output, worth unifying.
- An unexplained `p_top10` shift between two historical backtest runs. The
  pipeline is verified deterministic and the classifier is not the champion;
  see REASONING [013].
- Sprint weekends are not modelled. `circuit_id` is a slug of the event
  location, so the 2020 Sakhir GP outer layout collides with Bahrain (20 rows).

## Tests

`python -m pytest` — 267 tests. They target the specific failure modes in
`FIX_PLAN.md` §10, not coverage for its own sake:

- **Replay parity**: hiding a real race's outcome and feeding it back through
  the inference path reproduces the training feature vector exactly. This
  catches both serving drift and any feature that reads its own race's results.
- **Permutation invariance**: shuffling driver rows changes no prediction,
  ordering or metric.
- **Label fixtures**: normal finish, lapped runner, classified retirement, DNS,
  DSQ, withdrawal.
- **Feature contract**: race-session weather cannot re-enter `FEATURE_COLS`.
- **Fitted state**: changing held-out outcomes cannot move training constants.
- **Grid contract**: qualifying order is never silently used as the grid.
- **Bundle**: a bundle refuses to score a race at or before its training cutoff.
- **Probabilities**: Plackett-Luce outputs are coherent and order-invariant.

Other guards: `src/baseline.py` compares and needs `--force` to overwrite;
`blend_rank.py` needs `--write`; `FillPolicy.from_json` refuses an unknown
`schema_version`.

## Known modelling caveats

Finishing position is mostly the car — teammate-relative, constructor and
team-pace features exist for that reason. All-history circuit stats blur car
changes across seasons. The model partly predicts reliability because
retirements stay in the target, which is intended. Every small-sample affinity
ships with its `_n` count so the model can discount noise itself.

A transformer is an experiment, not a prerequisite, and FIX_PLAN.md §7 argues
127 race groups cannot support one. The repeated negative feature results are
consistent with that. TabPFN, being pretrained, is the only justified trial.

The remaining work is new information and prospective evidence, not further
tuning of the same inputs: archive a forecast for every remaining 2026 race and
score it once the result is in.
