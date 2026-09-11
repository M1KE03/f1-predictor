# F1 race-result predictor

Forecasts a Formula 1 race **after qualifying, before the start**. The goal is
winner and podium accuracy; full-field ordering is kept as a guardrail.

Data comes from [FastF1](https://docs.fastf1.dev/). Models are LightGBM. The
governing rule is that every feature must be computable from information
available at the forecast cutoff — not merely from prior races. Those are
different guarantees, and the gap between them hid a real leak (see
[Corrections](#corrections-applied)).

> **Status: under repair.** The project is working through the diagnosis in
> [`FIX_PLAN.md`](FIX_PLAN.md). Milestone 1 (correctness) is partly done;
> milestones 2–6 (evaluation harness, qualifying features, model comparison)
> have not started. **No model currently beats sorting by starting grid** for
> winner or podium. Do not treat its output as validated.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
```

`requirements.lock.txt` pins the exact environment that produced
`reports/baseline.json`. Python 3.12 is what this is developed on.

Run everything from the project root as modules — `src` is a package and uses
relative imports.

## Artifact layout

Two parallel sets, and the distinction matters:

| Location | What | Rule |
| --- | --- | --- |
| `data/`, `models/` | **Frozen legacy** artifacts, hashed in `reports/baseline.json` | Never overwrite. They are the evidence of what the project scored before the corrections. |
| `data/v2/`, `models/v2/` | **Current corrected pipeline** | Rebuilt by each increment. A working area, not a version history. |

`data/raw_results.parquet` is shared by both: ingestion has not been re-run.

## Running the corrected pipeline

```bash
python -m src.build_features --raw data/raw_results.parquet --out-dir data/v2
python -m src.audit_leakage  --data-dir data/v2 --n-rows 5      # GATE 2
python -m src.train          --features data/v2/features.parquet --models-dir models/v2
python -m src.train_rank     --features data/v2/features.parquet --models-dir models/v2
python -m src.blend_rank     --features data/v2/features.parquet --models-dir models/v2 --write
python -m src.evaluate_rank  --models-dir models/v2
python -m pytest
```

Re-ingesting needs network access to FastF1's endpoints:

```bash
python -m src.ingest            # 2018 -> latest; prints GATE 1
```

`python -m src.baseline` re-measures the frozen legacy artifacts and *compares*
against `reports/baseline.json` rather than overwriting it. It needs `--force`
to replace the record — deliberately, so the pre-correction reference cannot be
destroyed by a routine re-run.

## Data actually present

**2022–2026**, 2,080 driver-race rows, 103 races, last race 2026-07-26. Real
FastF1 data, not synthetic.

| Season | Rows | Races | Role |
| --- | ---: | ---: | --- |
| 2022–2024 | 1,359 | 68 | train |
| 2025 | 479 | 24 | validation |
| 2026 | 242 | 11 | test (held out) |

2018–2021 is **not** ingested. Recovering it roughly triples the data and is a
prerequisite for the expanding-window backtests in `FIX_PLAN.md` §8.

## Where things live

| Module | Role |
| --- | --- |
| `columns.py` | Single source of truth: `FEATURE_COLS`, `ID_COLS`, `WEATHER_REMOVED` |
| `labels.py` | Result vocabulary — `result_order`, `officially_classified`, `started`, `finished`, `status_category` |
| `leakage.py` | The `shift(1)` helpers every historical aggregation routes through |
| `features.py` | **The single as-of feature path**, shared by training and inference |
| `preprocessing.py` | `FillPolicy` — imputation fitted on training rows, frozen, replayed at serving |
| `splits.py` | Chronological partitioning |
| `metrics.py` | Race metrics + the deterministic tie policy |
| `weather.py`, `circuit.py`, `teammate.py` | Feature families |
| `ingest.py` | FastF1 → `raw_results.parquet`, team canonicalisation, Gate 1 |
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
  conditions.
- **Fitted preprocessing.** Imputation constants were computed over the whole
  frame before the split. Now fitted on training rows only (mean finish 10.4790,
  not 10.5982).
- **One feature path.** Training and inference each built their own feature
  vector and disagreed on 2,091 cells. Both now call
  `features.build_asof_features()`.

### Not yet fixed

- Qualifying pace (`quali_best_s`, `gap_to_pole_s`) is ingested, 98.9%
  populated, and **unused**.
- `form_avg_quali_3` averages prior *starting grids*, not qualifying pace.
  `quali_gap_to_teammate` compares those averages. `constructor_standing_prior`
  is cumulative points, not a standings rank, and omits sprint points.
- Blend alpha is selected by Spearman, not by winner/podium.
- `predict.py --from-quali` reads qualifying position as if it were the grid,
  ignoring penalties, and assigns absent drivers a hard-coded pit start at 20.
- No model bundle binding data hash, features, fitted state, cutoff and model.
- Evaluation prints failures but exits 0.

## Results, held-out 2026 (11 races)

| Method | Winner | Podium | Top-10 | Spearman (all) | Spearman (finishers) |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Starting grid** | **0.7273** | 0.5758 | 0.7364 | **0.6432** | **0.8450** |
| Top-10 classifier | 0.1818 | 0.4242 | 0.7636 | 0.5812 | 0.7856 |
| LambdaRank | 0.5455 | **0.6061** | 0.7182 | 0.4949 | 0.7296 |
| Blend (α=0.5) | **0.7273** | 0.5758 | 0.7364 | 0.6055 | 0.8101 |

**Read this carefully.** The blend selects exactly the same winners as the raw
starting grid (8 of 11). The machine learning adds nothing over "sort by grid
position" for the task the project exists to do.

Two caveats that must travel with these numbers:

1. **Eleven races.** One race is 9.1 percentage points of winner accuracy.
   `FIX_PLAN.md` §8 is explicit that this sample cannot rank candidates.
   Nothing in the table is statistically meaningful.
2. The LambdaRank Spearman partly reflects the tie-break, not the model: 32% of
   its rows are tied and fall back to grid order.

If you have seen "~85%" quoted for this project, that was the classifier's
**validation AUC**, not accuracy of finishing positions.

## Tests

`python -m pytest` — 97 tests. They target the specific failure modes in
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

## Known modelling caveats

Finishing position is mostly the car — teammate-relative and constructor
features exist for that reason. All-history circuit stats blur car changes
across seasons; recency weighting and shrinkage are milestone 3. The model
partly predicts reliability because retirements stay in the target, which is
intended. Every small-sample affinity ships with its `_n` count so the model can
discount noise itself.

Circuit id is a slug of the event location. Known blind spot: the 2020 Sakhir GP
outer layout shares `sakhir` with the Bahrain GP.
