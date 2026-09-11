# F1 Top-10 Finish Predictor

Binary classification: for each (driver, race), predict `P(driver finishes in the top 10)`. Data from FastF1 (2018+), model is LightGBM. Built to the accompanying build specification; the golden rule throughout is that every feature is computable strictly before lights-out from prior races only.

## Setup

Python 3.10 or 3.11 recommended (developed and smoke-tested on 3.12; no 3.12-specific syntax is used).

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run everything from the project root as modules (`python -m src.<name>`); `src` is a package and modules use relative imports.

## Build order (spec section 7)

```
python -m src.ingest            # 2018 -> latest; prints VALIDATION GATE 1
python -m src.build_features    # spec sections 2-3
python -m src.audit_leakage     # VALIDATION GATE 2 -- mandatory, non-zero exit on fail
python -m src.train             # chronological split + LightGBM early stopping
python -m src.evaluate          # VALIDATION GATE 3 -- model vs grid baseline
```

Do not skip gates. Gate 1 requires a manual spot-check of one famous race against reality. Gate 3 is only a pass if the model beats the grid-position baseline on **both** set-overlap and Spearman; a tie is a null result and the script says so.

The inference entry point (spec section 6) is deliberately not implemented yet: per the spec's build order it comes only after Gate 3 passes on real data.

## What ships in this repo

- `src/columns.py` — single source of truth for `FEATURE_COLS` / identifiers / target.
- `src/leakage.py` — the leakage-safe aggregation helpers (`shift(1)` pattern), including `past_mean_excluding_current_race` for team-grouped features (see "Leakage note" below).
- `src/ingest.py` — schedule iteration, safe session loading, results/weather/quali extraction, `TEAM_CANONICAL` mapping, Gate 1.
- `src/weather.py`, `src/circuit.py`, `src/teammate.py`, `src/build_features.py` — spec sections 2.1–2.5, plus reliability and constructor-points-prior features and the missing-value policy (fills saved to `data/fill_values.json` for identical treatment at inference).
- `src/audit_leakage.py` — Gate 2: independent naive recomputation of seven features on random rows across seasons, plus a global check that every driver's first-ever race (and first visit to each circuit) carries only NaN-before-fill values, plus neutrality of teammate deltas at debuts.
- `src/train.py`, `src/evaluate.py` — spec sections 4–5, including the grid baseline and the feature-importance plot (`reports/feature_importance.png`).
- `src/make_synthetic.py` — test harness only. Generates `data/raw_results.parquet` with the exact ingestion schema and planted latent structure so the pipeline can be exercised without network access. **Not real data; gates only count on real data.**

## Sandbox verification status

This codebase was built in an environment where FastF1's data endpoints (`livetiming.formula1.com`, `api.jolpi.ca`) are blocked, so real ingestion has not been run yet. What **has** been verified end to end on the synthetic dataset (2,800 rows, 140 races, 7 seasons): feature building, the full Gate 2 audit (all checks pass, two seeds), training with early stopping, and evaluation including the baseline comparison and its honest-failure path. Your first local step is `python -m src.ingest` followed by the Gate 1 spot-check, then re-running everything downstream on the real parquet.

## Leakage note (found and fixed by the Gate 2 audit)

Team-grouped historical features are a trap: a team has two rows per race, so the canonical `groupby(...).shift(1)` excludes only the current *row* — the second driver's row would still see the same-race teammate's outcome. `team_dnf_rate` and `team_circuit_avg_finish` therefore aggregate to one row per (team, race) first and shift at race level, excluding the entire current race. The audit's independent date-based recomputation is what caught this; keep it in the loop for any new feature.

## Deviations from / interpretations of the spec

- **Fill policy scales (2.5):** "fill with `driver_overall_avg_finish`" is applied to finishing-position-scaled features only. `form_avg_points_3` is filled with 0 (points scale), DNF/podium *rates* with their global means, deltas with 0 (neutral), counts with 0, flags with their spec defaults. Everything is recorded in `data/fill_values.json`.
- **Pit-lane starts:** `grid_position = 20` per spec 1.5. With 22-car grids (2026, Cadillac) "back of grid" is 22; noted as a v2 refinement, not changed here.
- **Quali pace (1.5):** `quali_best_s` / `gap_to_pole_s` are ingested and stored as raw columns for future use, but are not in `FEATURE_COLS` — spec section 3 defines the v1 matrix and grid features come from `GridPosition` (which already reflects penalties).
- **Temperature bins:** cool `<30`, hot `>45`, medium otherwise (30 and 45 inclusive in medium).
- **Spearman (5.2):** computed on classified drivers only, since DNFs have no finishing position; set-overlap uses denominator 10 per spec even in races with fewer than 10 classified finishers.
- **`driver_overall_avg_finish`** is computed (spec 2.2) and kept in the parquet as a helper for fills and auditing, but excluded from `FEATURE_COLS`, matching the spec's section-3 list exactly.
- **Circuit id:** slug of `event['Location']`, stable across seasons. Known blind spot: the 2020 Sakhir GP outer layout shares `sakhir` with the Bahrain GP. Acceptable for v1.
- **`TEAM_CANONICAL`:** Force India → Racing Point → Aston Martin; Toro Rosso → AlphaTauri → RB → Racing Bulls; Sauber → Alfa Romeo → Kick Sauber → Audi; Renault → Alpine. Unmapped names fall back to a slug with a loud warning — extend the dict deliberately when that fires.

## Known caveats (spec section 8 — do not hand-tune around these)

Finishing position is mostly the car; teammate-relative and constructor features exist for exactly that reason. All-history circuit stats blur car changes across years (recency-weighting is v2). The model partly predicts reliability because DNFs stay in the target — intended. Every small-sample affinity ships with its `_n` count so the model can discount noise on its own.
