# REASONING

**Purpose:** the decision log. Every change gets an entry recording what the
architecture was before, what it is now, and **why**. This is the file that
explains choices a reader could not infer from the diff.

**Maintained by:** Claude. Append a new entry per change — never rewrite or
delete history. If a decision is later reversed, add a new entry that supersedes
it and link back; leave the original in place.

**Entry format:**

```
## [NNN] Short title
**Date:** YYYY-MM-DD · **Milestone:** N · **Files:** a.py, b.py
**Status:** proposed | implemented | superseded by [NNN] | reverted

### Before
What the code did, concretely.

### After
What it does now.

### Why
The reasoning. Include the failure mode being closed and the evidence for it.

### Trade-offs / what this costs
What got worse, what was deliberately not done, known limits.

### Verification
How we know it worked. Commands, metrics before/after, tests added.
```

---

## [000] Baseline architecture as found

**Date:** 2026-09-11 · **Milestone:** — · **Files:** all of `src/`
**Status:** implemented (this is the starting point, not a change)

Recorded before any work begins, so "previous architecture" has a fixed
reference point once edits start.

### Architecture as inherited

Linear pipeline, each stage a module run as `python -m src.<name>`, passing
state through parquet files in `data/` and artifacts in `models/`.

```
ingest.py          FastF1 -> data/raw_results.parquet, Gate 1 (manual spot-check)
build_features.py  raw -> features_prefill.parquet -> features.parquet + fill_values.json
audit_leakage.py   Gate 2, naive recompute vs pipeline, non-zero exit on fail
train.py           LGBMClassifier, binary, P(top 10)     -> models/model.joblib
train_rank.py      LGBMRanker, lambdarank, race groups   -> models/rank_model.joblib
evaluate.py        Gate 3, classifier vs grid baseline
evaluate_rank.py   ranker vs grid baseline (winner/podium/spearman)
blend_rank.py      alpha sweep on validation             -> models/blend_alpha.json
predict.py         appends placeholder rows, reruns pipeline, scores
```

Supporting modules: `columns.py` (single source of truth for `FEATURE_COLS`),
`leakage.py` (the `shift(1)` helpers everything else routes through),
`weather.py`, `circuit.py`, `teammate.py` (feature families),
`make_synthetic.py` (test harness).

### Design decisions worth preserving

These are good and should survive the revamp:

- **`columns.py` as a single source of truth.** One list defines what reaches
  the model; everything else in the parquet is explicitly an identifier, label
  or helper. Keep this property.
- **`leakage.py` as a chokepoint.** All historical aggregation goes through
  shared helpers that assert date-sortedness rather than silently re-sorting, so
  an unsorted frame fails loudly instead of producing leaky features.
- **`past_mean_excluding_current_race`.** A team has two rows per race, so a
  plain `groupby().shift(1)` excludes only the current *row* — the second
  driver's row would still see its teammate's same-race outcome. This helper
  aggregates to one row per (team, race) and shifts at race level. The bug was
  found by the audit, not by inspection, which is the argument for keeping the
  audit in the loop for every new feature.
- **`weather.py`'s strict-backward `merge_asof`.** For sparse subsets (wet
  races, temp bins) a shifted subset stat goes stale for rows between subset
  races. Computing the running stat including each subset race and exposing it
  via `merge_asof(direction='backward', allow_exact_matches=False)` gets the
  same exclusion guarantee without the staleness bug.
- **`audit_leakage.py`'s independent recomputation.** Naive filter-and-loop
  reimplementations that share no code with the pipeline. This is the right
  shape for a leakage test — it just needs to cover more than historical
  aggregation.
- **Honest gates.** `evaluate.py` prints FAIL/NULL RESULT rather than hiding
  behind AUC. Preserve that posture; extend it to exit non-zero.

### Design decisions that must change

Full evidence in `FIX_PLAN.md` §2; condensed in `HANDOVER.md` §4. Summary of
the *reasoning*, which is what belongs here:

1. **The forecast contract was never written down.** Features were audited for
   "computable from prior races" but not for "available at the prediction
   cutoff". Those are different guarantees, and the gap is exactly where the
   race-weather leak lives: weather averaged over the race session passes the
   first test and fails the second. Everything else in P0 follows from the same
   missing contract.
2. **Preprocessing state is fitted globally, not per split.** Fill values are
   computed over the whole frame before the chronological split, so held-out
   outcomes influence training inputs. Standard leakage; the audit did not look
   for it because it only inspected feature *formulas*, not fitted state.
3. **Two code paths compute features.** `build_features.build()` and
   `predict.predict()` both assemble a feature vector, and they disagree — the
   trainer uses a driver-prior fallback the server does not. Any two-path design
   drifts; it needs to become one `build_asof_features(...)` call used by both.
4. **Labels conflate distinct concepts.** `classified = notna(Position)` merges
   "a result position exists" with "officially classified", so DNS and DSQ rows
   claim classification. Downstream code then documents behaviour it does not
   have (`train_rank` says DNFs get zero relevance; most retirements keep
   position-based relevance).
5. **Output is not deterministic.** `rank(method='first')` breaks ties by row
   order, so shuffling input rows changes both predictions and metrics. A
   forecast that depends on row order cannot be reproduced or audited.
6. **The objective does not match the goal.** The blend selects alpha by
   Spearman while the stated aim is winner and podium. `evaluate_rank.py` can
   report PASS without improving winner accuracy at all.
7. **The strongest available signal is discarded.** `quali_best_s` and
   `gap_to_pole_s` are ingested, 98.9% populated, and excluded from
   `FEATURE_COLS`. For a post-qualifying forecast this is the most obviously
   informative input in the dataset.

### Why the ordering is correctness-first

The plan fixes contracts and the measurement harness before touching features or
models. Reason: with race-session weather and globally-fitted fills in place,
every benchmark number is measured on inputs that will not exist at prediction
time. Adding a model to that pipeline compares candidates on unreal inputs and
can reward the wrong behaviour. Correcting the pipeline first is also the only
way to attribute a later gain to the model rather than to the leak.

**Expect the metrics to drop after milestone 1.** That is the leak being removed,
not a regression, and it must not be treated as a reason to revert.

### Verification

`FIX_PLAN.md` claims were checked against the artifacts on 2026-09-11:
2,080 rows / 103 races / 2022-2026 confirmed; 303 `is_dnf & classified` rows
confirmed (16 DNS, 10 Disqualified among them); `quali_best_s` 98.9% populated
confirmed; saved alpha 0.6 confirmed. Recorded SHA-256 hashes are in
`FIX_PLAN.md` appendix.

---

## [001] Freeze the legacy baseline before any correction

**Date:** 2026-09-11 · **Milestone:** 0 · **Files:** `src/baseline.py` (new),
`.gitignore`, `requirements.lock.txt` (new)
**Status:** implemented

### Before

No record bound the dataset to the artifacts to the numbers. `FIX_PLAN.md`
quoted metrics in prose, but nothing in the repo could regenerate them, and
`data/`, `models/` and `reports/` were entirely gitignored, so no provenance
evidence could be committed at all. Any post-fix number would have been
compared against a remembered figure rather than a measured one.

### After

- `src/baseline.py` writes `reports/baseline.json`: SHA-256 of seven artifacts,
  interpreter and package versions, per-season coverage, split convention, and
  the legacy metrics for all four ordering methods.
- `.gitignore` switched from `dir/` to `dir/*` plus negations, so small JSON
  provenance records are tracked while parquet/joblib/png artifacts stay out.
- `requirements.lock.txt` pins the exact environment that produced the record.

### Why

- **A baseline that cannot be regenerated is not a baseline.** The P0 fixes are
  expected to move the metrics down; without a committed record, that drop is
  indistinguishable from a bug introduced during the fix.
- **The baseline records defects, not just scores.** `known_defects` stores the
  row-order Spearman spread across five shuffles. It is the regression target
  for the deterministic tie policy: the number must become exactly 0.
- **Legacy behaviour is reproduced, not corrected.** `baseline.py` calls the
  existing `ranking_metrics` / `order_metrics` rather than reimplementing them,
  and keeps the misleading `top1_hit_rate` name. Recording what the code scores
  today is the entire point; correcting on the way in would destroy the
  comparison.
- **Git could not store the evidence.** Git cannot re-include a file whose
  parent directory is excluded, so `reports/` + `!reports/baseline.json` would
  have silently failed. `reports/*` + negation is the form that works.

### Trade-offs / what this costs

- `baseline.py` imports `evaluate.py`, which imports matplotlib at module level.
  Accepted for now: this module is temporary scaffolding and the coupling
  disappears when the milestone 2 metric harness replaces these functions.
- The metrics are measured on a leaky feature matrix and an 11-race test season.
  One race is 9.1 percentage points of winner accuracy. The file therefore
  carries an explicit `description` saying it is not a clean estimate of live
  forecasting quality, so the numbers cannot later be quoted as a target.
- `requirements.lock.txt` duplicates `requirements.txt`. Deliberate: the loose
  ranges stay for fresh installs, the lock reproduces the frozen result.
- Only the shuffle defect is quantified. Weather leakage and train/serve skew
  are recorded in `FIX_PLAN.md` but not yet measured as numbers; they need the
  milestone 2 harness.

### Verification

- All five hashes recorded in `FIX_PLAN.md`'s appendix still match the artifacts
  on disk, so the diagnosis and this baseline describe the same bytes.
- `python -m src.baseline` reproduces `FIX_PLAN.md` section 2's table exactly:
  grid 0.7273 / 0.5758 / 0.7364 / 0.6432; classifier 0.2727 / 0.4848 / 0.7545 /
  0.5672; ranker 0.5455 / 0.5758 / 0.6909 / 0.5005; blend 0.7273 / 0.5758 /
  0.7545 / 0.6289 (winner / podium / top-10 overlap / Spearman). Independent
  confirmation that the review's diagnostic numbers are correct.
- Row-order dependence confirmed present: max Spearman spread 0.0080 over five
  shuffles.
- `git check-ignore` confirms `reports/baseline.json` and the `.gitkeep` files
  are trackable while `raw_results.parquet`, `model.joblib` and
  `feature_importance.png` remain ignored.

### Not done in this increment

`README.md` is still stale (claims 2018+ data and synthetic-only; the data is
really 2022-2026 and real). Held back so this increment stays one reviewable
concern; it belongs with the milestone 1 changes that alter the documented
pipeline anyway.

---

## [002] Separate the conflated result-label concepts

**Date:** 2026-09-11 · **Milestone:** 1 (increment 1.1) · **Files:**
`src/labels.py` (new), `src/ingest.py`, `tests/test_labels.py` (new),
`pytest.ini` (new), `requirements-dev.txt` (new)
**Status:** implemented

### Before

`ingest.race_rows()` derived every outcome concept from two expressions:

```python
is_dnf     = 0 if status in ("Finished", "Lapped") or status.startswith("+") else 1
classified = int(pd.notna(pos))
```

`classified` therefore meant only "a result place exists".

### After

`src/labels.py` defines the vocabulary and derives seven independent fields —
`result_order`, `classified_position_raw`, `status_category`, `started`,
`finished`, `officially_classified`, `officially_classified_source` — plus
`is_winner` / `is_podium` / `finished_top10`. `ingest.py` calls it and now also
captures FastF1's `ClassifiedPosition` and `Laps`. `is_dnf` and `classified`
survive as deprecated aliases so unmigrated consumers keep working, with
`is_dnf` now *derived* as `1 - finished` rather than separately recomputed, so
the two can no longer disagree.

### Why

Measured on the stored data before writing any code:

- **`classified` is 99.90% constant** — 2078 of 2080 rows. The only two zeros
  are 2022 R2 MSC and 2023 R15 STR, both "Withdrew". It cannot distinguish a
  winner from a non-starter, so every consumer gating on it is gating on
  nothing. This is stronger than `FIX_PLAN.md` records and is the real reason
  the flag has to go.
- **`position` is result order, not finishing position.** FastF1 publishes the
  full classification: in 2026 R1, retirements take places 18-20 and the two
  DNS take 21-22. Preserving that order is correct per `FIX_PLAN.md` section 4
  ("do not blindly collapse every retirement to last") — but nothing in the
  schema said so, so the meaning was only discoverable by inspection.
- **Two documented behaviours were false.** `train_rank.py`'s docstring says
  DNFs receive zero relevance; 303 of 305 retirements actually receive
  position-based relevance. `README.md` says Spearman is computed on finishers
  only "since DNFs have no finishing position"; it is computed over all 2078
  rows, DNFs included. Both follow directly from `classified` being ~constant.
- **Cause coverage does not support a mechanical/incident split.** 197 of 305
  retirements (64.6%) carry the bare status "Retired" with no cause.
  `FIX_PLAN.md` section 5.C proposes separating mechanical failures from
  incidents "where status coverage supports it" — on this dataset it does not,
  for two thirds of retirements. Hence the explicit `retired_unspecified`
  category rather than a guess.

### Trade-offs / what this costs

- **`officially_classified` is derived, not authoritative, on stored data.**
  The 90%-race-distance rule needs lap counts, which were never stored. A
  driver who retired late but completed the distance IS officially classified
  and status alone cannot reveal that. Rather than hide the uncertainty, every
  row carries `officially_classified_source`, and all 2080 stored rows read
  `derived_from_status`. Re-ingestion (needs network) upgrades them to
  `classified_position`.
- **Two bodywork statuses are judgement calls.** "Undertray" (3) and "Front
  wing" (1) map to `mechanical`. They are ambiguous between contact damage and
  failure, but the source reports contact separately as "Collision damage", so
  a bare component name is more likely a failure. 4 rows; revisit if it grows.
- **The deprecated aliases are debt.** `is_dnf` and `classified` stay until
  consumers migrate in a later increment. Leaving them avoids bundling a
  breaking change into a definition change.
- **The backfill writes a second file** (`data/raw_results_labeled.parquet`)
  rather than editing `raw_results.parquet`, because `reports/baseline.json`
  hashes the original. Path sprawl, accepted until the `data/raw`,
  `data/snapshots`, `data/features` restructure in `FIX_PLAN.md` section 5.A.1.
- **pytest is new to the repo.** Bundled here rather than committed separately
  so the first test ships with the code it covers.

### Verification

- 53 tests pass. Fixtures cover a normal finish, a lapped runner, both lapped
  spellings, a classified retirement, a bare retirement, DNS, DSQ and
  withdrawal, plus the ClassifiedPosition-wins-over-status case.
- Backfill over all 2080 stored rows: **every status mapped, zero `unknown`**.
  Categories: finished 1775, retired_unspecified 197, mechanical 42,
  accident 37, did_not_start 16, disqualified 10, withdrawn 3.
- **`finished_top10` changed on 0 rows.** The new definitions are
  behaviour-preserving for the target label — no silent relabelling.
- `is_dnf` and `classified` reproduce their legacy values exactly on all rows.
- Checked and found NOT to be a problem: DSQ and DNS rows cannot acquire a
  top-10 label, because FastF1 demotes them to the back of the order (DSQ rows
  land at 18-21, DNS at 18-22). Recorded as a test so it stays true.
- `data/raw_results.parquet` still matches its frozen hash;
  `python -m src.baseline` reproduces identical metrics.

### Not done in this increment

Downstream consumers (`train_rank.py` relevance, `evaluate.py` /
`evaluate_rank.py` Spearman filters, `columns.py` ID_COLS) still read the
deprecated `classified`. Migrating them changes model inputs and metric
denominators, which is a behaviour change and belongs in its own reviewable
increment.

---

## [003] Protect the frozen baseline from silent overwrite

**Date:** 2026-09-11 · **Milestone:** 0 (follow-up to [001]) · **Files:**
`src/baseline.py`
**Status:** implemented

### Before

`python -m src.baseline` unconditionally rewrote `reports/baseline.json`.

### After

It refuses to overwrite an existing record unless `--force` is passed, and
instead compares the freshly computed values against the frozen ones, reporting
which of `artifacts` / `data` / `split` / `metrics` / `known_defects` differ.

### Why

Found by running into it: re-running the recorder purely to *verify* nothing had
drifted produced a git diff, because `created_utc` and `git_commit` changed
while every metric stayed identical. That is noise at best. The real hazard is
worse — once the P0 fixes land and the metrics legitimately move, an
unconditional write would silently replace the pre-correction reference with the
post-correction one, destroying the only evidence of what the corrections cost.
The guard turns the command into a comparison by default and a replacement only
on request.

### Trade-offs / what this costs

Regenerating after a deliberate change now takes an extra flag. Acceptable: the
whole point of the record is that replacing it should be a conscious act.

### Verification

`python -m src.baseline` on the committed record reports
"Substantive differences vs the frozen record: NONE (metrics identical)" and
leaves the file untouched — `git status` clean for `reports/`. This also
confirms increment 1.1 did not perturb any baseline metric.

---

## [004] Correct evaluation semantics and make ordering deterministic

**Date:** 2026-09-11 - **Milestone:** 1 (increment 1.2) - **Files:**
`src/metrics.py` (new), `tests/test_metrics.py` (new), `src/evaluate.py`,
`src/evaluate_rank.py`, `src/blend_rank.py`, `src/predict.py`, `src/baseline.py`
**Status:** implemented

### Before

`evaluate.ranking_metrics` and `evaluate_rank.order_metrics` were near-duplicate
loops. Both ordered drivers with pandas `rank(method="first")`, took
winner/podium from ad-hoc `(classified == 1) & (position <= k)` tests, computed
one `spearman` over every row holding a result place, and returned a single
`n_races` that did not match each metric's real denominator.
`blend_rank.add_blended_score` and `predict.py` used the same ranking call.

### After

`src/metrics.py` owns all of it: `assign_pred_rank` / `pred_rank_by_race` apply
one deterministic tie policy, and `race_metrics` returns `winner_accuracy`,
`podium_overlap`, `top10_overlap`, `top_pick_finished_top10`, `spearman_all`,
`spearman_finishers` and a per-metric race count. The two evaluate modules keep
their old function names as thin aliases. The blend and the inference path use
the shared deterministic ranker.

### Why

- **The forecast itself was order-dependent, not just its score.** The defect
  was recorded as a metric artefact, but grepping for the ranking call found it
  in `blend_rank.add_blended_score` and `predict.py` too. Those feed the
  published output, so an identical entry list in a different order produced a
  different predicted podium. Fixing only the metrics would have left the
  forecast unreproducible while making the reported numbers look clean.
- **The tie policy uses only pre-race information**: score, then grid position
  known at the cutoff, then driver id. Driver is unique within a race, so this
  is a total order and the result depends solely on which rows are present.
- **Winner and podium now come from explicit labels** ([002]) rather than from
  a flag that is true for 99.9% of rows.
- **One Spearman could not mean two things.** The inherited metric covered all
  entries against the published order while `README.md` described it as
  finishers-only. Both are legitimate, so both are reported with separate
  denominators and neither can be quoted with the wrong meaning.
- **`top1_hit_rate` was renamed** to `top_pick_finished_top10`, which is what it
  measures (FIX_PLAN.md section 2, P0-5).
- **`baseline.py` was pinned to private copies** of the pre-correction metric
  and blend functions. It previously imported them, so correcting those modules
  would have made the provenance tool reproduce corrected numbers while claiming
  to describe the legacy baseline - silently destroying the comparison.

### Trade-offs / what this costs

- **`rank_model`'s Spearman rose 0.5005 -> 0.5329 without the model improving.**
  32% of its test rows (77 of 242) are tied, and the tie-break falls back to
  grid order, so a model that cannot separate two drivers now inherits the grid
  baseline's ordering there. That is the correct policy - grid is legitimate
  pre-race information - but the gain must not be read as a modelling gain. The
  legacy 0.5005 was one arbitrary draw from a 0.5005-0.5187 range.
- **The blend's numbers moved in both directions** (top-10 overlap
  0.7545 -> 0.7455, Spearman 0.6289 -> 0.6416) because the blend itself changed,
  not merely its measurement. The legacy values were partly an artefact of row
  order.
- **Alias functions are debt.** `ranking_metrics` and `order_metrics` survive
  with changed return keys, which is a silent breaking change for any unmigrated
  caller. All in-repo callers were migrated; the aliases exist so the diff stays
  reviewable.
- **`blend_rank` no longer writes by default.** It needs `--write`, because the
  alpha it saves is hashed in `reports/baseline.json` and was previously
  rewritten as a side effect of merely inspecting the sweep.
- `make_synthetic.py` still uses the old ranking call. Left alone: it generates
  fixture data from continuous floats and is not a forecast path.

### Verification

- 70 tests pass (17 new in `tests/test_metrics.py`).
- **The recorded defect is closed.** On the real 2026 test season, Spearman
  spread across five row shuffles is exactly `0.0000000000` for all four
  ordering methods, against the `0.0080` in `reports/baseline.json`.
- Winner accuracy, podium overlap and top-10 overlap are **unchanged** for
  grid_baseline, top10_classifier and rank_model, confirming the label
  corrections did not silently move the headline numbers.
- Only the two ordering methods with tied scores changed their Spearman.
  Verified as the cause: `rank_score` has 77 tied rows of 242 and `blend_score`
  29, while `grid_position` and `p_top10` have zero - and those two are exactly
  the methods whose metrics did not move.
- `python -m src.baseline` still reports "NONE (metrics identical)" against the
  frozen record, proving the pinned legacy copies are faithful.
- Re-running the alpha sweep with corrected metrics still selects **0.6**, so
  `models/blend_alpha.json` needs no change.

### Not done in this increment

- `train_rank.py` still derives relevance from `classified`, so 303 of 305
  retirements keep position-based relevance. Fixing it changes model inputs and
  forces a retrain.
- `columns.py` ID_COLS still lists the deprecated `classified`.
- Evaluation still exits 0 on failure (FIX_PLAN.md section 2, P1); belongs with
  the milestone 2 harness.
- Alpha is still selected by Spearman rather than winner/podium (P1). A
  deliberate omission: changing the objective is a modelling decision, not a
  correctness fix, and bundling it here would have made the metric movements
  impossible to attribute.

---

## [005] Ranker relevance off the deprecated `classified` gate

**Date:** 2026-09-11 - **Milestone:** 1 (increment 1.3) - **Files:**
`src/train_rank.py`, `src/columns.py`, `src/evaluate_rank.py`,
`tests/test_train_rank.py` (new), `models/v2/` (new artifacts)
**Status:** implemented

### Before

```python
classified = df["classified"] == 1
df.loc[classified, "relevance"] = (field_size - df["position"] + 1).clip(lower=0)
```

Because `classified` is true for 99.9% of rows, drivers who never started or
were disqualified received relevance based on wherever the classification
happened to list them. The module docstring claimed DNFs received zero
relevance; 303 of 305 retirements did not.

### After

Eligibility is `officially_classified`. Non-starters, disqualifications and
withdrawals score 0; a retirement that still holds a place in the published
classification keeps its real relevance. `columns.py` ID_COLS carries the
separated label fields and marks `classified` / `is_dnf` deprecated in place.
`train_rank.py` and `evaluate_rank.py` both take `--models-dir`, so a retrain
can be written alongside the frozen artifacts instead of over them.

### Why

- **The label contradicted its own documentation.** Whatever the right grade
  scheme turns out to be, a driver who did not start the race cannot be
  evidence about finishing order. Training on that teaches the ranker to
  predict positions for cars that were never on track.
- **Retirements keep their places deliberately** (FIX_PLAN.md section 4). The
  tempting alternative - collapse every retirement to last - would discard the
  source's real ordering information and is explicitly warned against.
- **The grade scheme was left alone on purpose.** Relevance is still
  `field_size - result_order + 1`, which is field-size dependent and, under
  LightGBM's exponential default `label_gain`, weights P1 far above P2.
  FIX_PLAN.md section 6 proposes a bounded scheme, but calls it a proposed
  experiment rather than a proven setting. Changing eligibility and grades in
  one step would have made the result impossible to attribute.
- **A separate model directory keeps the A/B honest.** `rank_model.joblib` is
  hashed in `reports/baseline.json`; overwriting it would have left no way to
  compare against the frozen artifact.

### Trade-offs / what this costs

- **This fix changes nothing measurable on the current dataset.** Only 27 of
  2080 rows change relevance, 11 of them in the training partition. The
  retrained ranker differs from the frozen one by a maximum of 1.8e-07 in
  predicted score - about 1.2e-07 of the score range - and not one driver's
  predicted position moves. Every headline metric is identical to four decimal
  places. The honest summary is that this closes a semantic hole, not a
  performance gap. It should matter more once 2018-2021 is re-ingested, where
  DNS and DSQ rows are more numerous.
- **`models/v2/` is path sprawl** pending the artifact restructure in
  FIX_PLAN.md section 5.A.1. It is not a versioning scheme, just somewhere safe
  to write while the baseline stays frozen.
- **v2 should not be promoted on performance grounds** - there is no
  difference to promote on. It is the correct code path, not a better model.
- `columns.py` still lists `classified` and `is_dnf`. Removing them is a
  breaking change for anything reading `features.parquet` directly, and the
  stored artifact still has those columns.

### Verification

- 79 tests pass (9 new in `tests/test_train_rank.py`), including a regression
  guard asserting that DSQ and DNS rows scored non-zero under the legacy gate
  and score zero now, while all seven classified drivers are unaffected.
- Impact measured before implementing: 27 of 2080 rows change relevance
  (16 did_not_start, 10 disqualified, 1 withdrawn); train 11, val 9, test 7.
- A/B on the held-out 2026 season, v1 vs v2: winner 0.5455 both, podium 0.5758
  both, top-10 0.6909 both, spearman_all 0.5329 both, spearman_finishers 0.7179
  both. Blend metrics identical too. Zero drivers change predicted position.
- Both models stop at iteration 20. Their serialised structures differ, so the
  retrain is real; the numerical effect is simply negligible.
- `models/rank_model.joblib` still matches its frozen hash, and every artifact
  recorded in `reports/baseline.json` is intact.
- `python -m src.baseline` still reports "NONE (metrics identical)".

### Not done in this increment

- The bounded grade-scheme experiment (FIX_PLAN.md section 6) and the
  field-size dependence of relevance.
- `blend_rank.py` and `predict.py` still load from `models/` with no
  `--models-dir`, so v2 is reachable only from `train_rank` and
  `evaluate_rank`. Wiring the rest belongs with the model-bundle work in
  milestone 5, which replaces ad-hoc directories with a real bundle.
- Removing the deprecated `classified` / `is_dnf` columns.

---

## [006] Remove race-weather leakage from the feature matrix

**Date:** 2026-09-11 - **Milestone:** 1 (increment 1.4) - **Files:**
`src/columns.py`, `src/weather.py`, `src/build_features.py`,
`src/audit_leakage.py`, `src/train.py`, `src/train_rank.py`,
`src/blend_rank.py`, `src/baseline.py`, `tests/test_columns.py` (new),
`data/v2/` + `models/v2/` (rebuilt artifacts)
**Status:** implemented

### Before

`FEATURE_COLS` held 30 features, eight of which required knowledge of the race
being predicted:

- `air_temp`, `track_temp`, `humidity`, `wind_speed`, `rainfall`, `is_wet` came
  from `ingest.weather_row()`, which averages FastF1's weather stream over the
  **race session**.
- `driver_temp_bin_avg` / `driver_temp_bin_n` read historical per-bucket
  averages, but `weather.add_weather_affinity()` chose *which* bucket to read
  from the target race's realized `track_temp`.

### After

22 features. The six raw readings stay in `features.parquet` as ID columns for
auditing but can no longer reach the model; the temperature-bin block is deleted
with its reasoning left in place. `driver_wet_delta` / `driver_wet_n` are
**kept**. Every pipeline module takes `--features` / `--out-dir` / `--models-dir`,
so the corrected artifacts were rebuilt into `data/v2/` and `models/v2/` without
touching the hash-frozen originals.

### Why

- **Two different leaks, only one of them obvious.** The raw readings are
  straightforwardly unavailable before lights-out. The temperature affinity is
  subtler and more instructive: its *values* were historical, so it passed the
  Gate 2 audit, but its *selection* used the target race's outcome-time data.
  Gate 2 tests whether a feature aggregates prior races; it never tested whether
  the inputs choosing it exist at the forecast cutoff. That gap is why
  `tests/test_columns.py` now guards the contract directly.
- **Not all weather features leak, and the distinction was checked rather than
  assumed.** `driver_wet_n` averages 16.3 at wet target races and 15.6 at dry
  ones - it counts a driver's PRIOR wet races and does not reveal the target's
  conditions. Dropping it would have discarded usable pre-cutoff signal in the
  name of caution.
- **Removed rather than imputed.** FIX_PLAN.md section 2 P0-1 says to remove
  unavailable race-weather inputs from the first corrected benchmark. Filling
  them with a global mean would have kept a column whose training values came
  from hindsight and whose serving values are a constant - strictly worse than
  not having it.

### The result contradicted the stated expectation

Every previous entry, `README.md`, and this file predicted the P0 fixes would
move metrics DOWN. **They did not.** On the held-out 2026 season:

| | legacy (30 feat) | corrected (22 feat) |
| --- | ---: | ---: |
| classifier ROC-AUC | 0.7908 | **0.8178** |
| classifier log-loss | 0.5440 | **0.5258** |
| ranker podium overlap | 0.5758 | **0.6061** |
| ranker top-10 overlap | 0.6909 | **0.7273** |
| ranker spearman_all | 0.5005 | **0.5394** |
| blend winner accuracy | 0.7273 | 0.7273 |
| grid baseline (control) | 0.6432 | 0.6432 |

The mechanism is visible in the generalisation gap. Validation AUC fell slightly
(0.8455 -> 0.8424) while test AUC rose (0.7908 -> 0.8178), halving the val-test
gap from +0.0547 to +0.0245. The weather features were helping the model fit the
validation season without generalising to the test season - on 1359 training
rows, eight noisy columns cost more in overfitting than they returned in signal.

**This is not evidence that the model got better.** The test season is 11 races,
where one race is 9.1 percentage points of winner accuracy, and FIX_PLAN.md
section 8 is explicit that this sample cannot rank candidates. The defensible
claim is narrower: removing the leak did not cost accuracy, so there is no
tension between correctness and performance here.

### Trade-offs / what this costs

- **A real signal may have been discarded.** Wet races genuinely change finishing
  order. What is gone is the ability to use *this* race's conditions; it returns
  in milestone 3 only with a forecast whose pre-cutoff availability can be proven
  (FIX_PLAN.md section 5.D), never with observed or reanalysis weather.
- **The blend's alpha moved 0.6 -> 0.5**, refitted on the corrected validation
  season and saved to `models/v2/blend_alpha.json`. The frozen
  `models/blend_alpha.json` is untouched.
- **`models/v2/` now means "current corrected pipeline", superseding the 1.3
  artifacts** written to the same directory. Since [005] measured no difference
  between them, nothing comparable was lost - but the directory is a working
  area, not a version history, and should be replaced by the milestone 5 model
  bundle.
- **`baseline.py` needed a second pin.** It imported `FEATURE_COLS`, so shrinking
  the list to 22 broke it against the 30-feature frozen model. It now reads
  `models/feature_cols.json`. Worth noting the failure mode: it raised a
  LightGBM shape error, but a subtler change could have silently scored the
  legacy model on a different feature set and misreported the historical record.
- The fill policy still fits global means over all partitions (P0-2, unfixed),
  so the corrected numbers above are not yet leak-free.

### Verification

- 87 tests pass (8 new in `tests/test_columns.py`).
- **Gate 2 passes on the rebuilt features**, including all debut and
  first-circuit-visit checks.
- `models/v2/feature_cols.json` contains 22 names and none of the eight removed.
- Every artifact hashed in `reports/baseline.json` is byte-intact, and
  `python -m src.baseline` still reports "NONE (metrics identical)".
- Grid baseline is unchanged at every metric, as it must be - it does not use
  the model - which confirms the comparison is like-for-like.

### Not done in this increment

- Fitted-on-training-only preprocessing (P0-2) and the single shared
  `build_asof_features` path (P0-3). Those are increments 1.5 and 1.6.
- `predict.py` still loads from `models/` and has no `--models-dir`.
- `README.md` remains stale and now understates the drift: it documents 30
  features and race-condition weather.

---

## [007] Fit preprocessing on training rows only, behind one shared feature path

**Date:** 2026-09-11 - **Milestone:** 1 (increments 1.5 + 1.6) - **Files:**
`src/preprocessing.py` (new), `src/features.py` (new), `src/splits.py` (new),
`src/build_features.py`, `src/predict.py`, `src/train.py`,
`tests/test_parity.py` (new), `README.md`
**Status:** implemented

### Before

`build_features.apply_fill_policy()` computed its constants over the whole frame
and then imputed everything, before `train.chronological_split()` ever ran.
Separately, `predict.py` had its own `run_pipeline()` and applied the saved
constants with a flat `fillna` loop.

### After

`FillPolicy.fit()` sees the training partition only; `FillPolicy.transform()` is
the one implementation both paths call; `to_json()` stores the fitted constants
**together with the category lists defining the fallback chain**, so serving
replays the saved logic rather than whatever the current source says.
`features.build_asof_features(history, upcoming, policy)` is the single feature
path. `predict.run_pipeline()` is deleted. `chronological_split` moved to
`src/splits.py` so feature building can reach it without importing LightGBM.

### Why

- **These were one defect, not two.** Fitting on everything (P0-2) and the
  train/serve mismatch (P0-3) are the same missing object seen from opposite
  ends: there was no fitted state, only constants recomputed in one place and
  partially reapplied in another. Splitting them across two increments would
  have meant building the object twice.
- **Measured, not assumed.** Full-data mean finish 10.5982 against 10.4790 on
  training - reproducing FIX_PLAN.md's figure exactly. The serving gap is 2091
  cells across six features: driver_circuit_avg_finish 736,
  driver_circuit_best_finish 736, team_circuit_avg_finish 540,
  season_avg_finish 79.
- **The policy stores its own category lists.** Storing only constants would
  let a later edit to POSITION_SCALED silently change how an existing model's
  inputs are imputed, with no error. `schema_version` makes an incompatible
  policy a loud failure instead.
- **The replay test is the real deliverable.** Taking a real race, hiding its
  outcome, feeding it back as a not-yet-run event and requiring the identical
  feature vector tests serving parity AND leakage in one assertion: if any
  feature for race R reads R's own results - its own, a teammate's, anyone's in
  that race - the vector changes and the test fails. It passes for both the
  final race and a mid-dataset race, the latter also proving that later results
  cannot reach back into an earlier forecast.

### The metrics moved DOWN this time

Unlike [006]. On the held-out 2026 season, against the frozen legacy baseline:

| | legacy | after 1.4 | after 1.5+1.6 |
| --- | ---: | ---: | ---: |
| classifier ROC-AUC | 0.7908 | 0.8178 | **0.8153** |
| classifier log-loss | 0.5440 | 0.5258 | **0.5290** |
| ranker spearman_all | 0.5005 | 0.5394 | **0.4949** |
| ranker podium overlap | 0.5758 | 0.6061 | **0.6061** |
| blend spearman_all | 0.6289 | 0.6240 | **0.6055** |
| blend top-10 overlap | 0.7545 | 0.7545 | **0.7364** |
| winner accuracy (all methods) | - | - | unchanged |

The ranker and blend now sit slightly BELOW the legacy baseline on Spearman; the
classifier stays well above it. This is the expected shape of removing a leak:
the imputed constants no longer carry information from the held-out seasons, so
some of the apparent skill went away with it. The val-test AUC gap stays halved
(+0.0258 against the legacy +0.0547).

Winner accuracy did not move anywhere, which remains the point: 8 of 11 for both
grid and blend, before and after every correction so far.

None of these differences is significant on 11 races. The defensible statement
is that the corrected pipeline is now measuring the right thing, not that it is
better or worse.

### Trade-offs / what this costs

- **`transform` is applied to the full frame, not per partition.** Correct -
  validation and test rows must be transformed with training-fitted constants,
  exactly as an unseen race would be - but it means `features.parquet` is no
  longer a pure function of its own partition. The policy JSON records
  `fitted_on` so this is auditable.
- **Imputation happens after the grid override in `predict.py`**, because
  `grid_position` is only known once qualifying has run. That ordering is now
  load-bearing and undocumented outside the code comment.
- **A schema bump invalidates old policies.** `models/fill_values.json` (the
  frozen legacy flat dict) can no longer be loaded by `FillPolicy.from_json`.
  Deliberate: the legacy models are served by `baseline.py`, which has its own
  pinned copies and does not use this class.
- **`add_recent_form` / `add_reliability` moved** from `build_features.py` to
  `features.py`. Any external caller importing them from the old location
  breaks. Nothing in-repo does.
- The two misleading feature names (`form_avg_quali_3` averages prior grids;
  `constructor_standing_prior` is cumulative points) are now documented at their
  definitions but not renamed - that changes the stored schema.

### Verification

- 97 tests pass (10 new in `tests/test_parity.py`).
- **Replay parity holds exactly** (`atol=1e-12`) for the final race and a
  mid-dataset race.
- Tampering with held-out outcomes provably cannot move the fitted constants.
- `transform` is idempotent; the driver-prior-then-global chain is asserted
  directly on the cells that previously diverged.
- A policy with an unknown `schema_version` is refused rather than
  reinterpreted.
- Gate 2 passes on the rebuilt features.
- The fitted policy records `n_rows: 1359, year_min: 2022, year_max: 2024`,
  i.e. the training partition alone, and `global_mean_finish: 10.479`.
- Every artifact hashed in `reports/baseline.json` is byte-intact.

### Not done in this increment

- `predict.py` still assigns absent drivers a hard-coded pit start at position
  20 and reads qualifying position as the grid (P0-4).
- No model bundle binding data hash, features, policy, cutoff and model (P1).
- Evaluation still exits 0 on failure.

---

<!-- Append new entries above this line, newest last. -->
