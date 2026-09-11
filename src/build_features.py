"""Feature-matrix orchestrator (spec sections 2 and 3).

Feature computation lives in `src/features.py` and imputation in
`src/preprocessing.py`; this module is the training-side entry point that wires
them together and writes the artifacts.

The imputation state is fitted on the TRAINING partition only and then frozen
(FIX_PLAN.md section 2, P0-2). Validation and test rows are transformed with
training-fitted constants, exactly as an unseen race would be at serving time.

Outputs (into --out-dir):
  features_prefill.parquet  pre-imputation snapshot, inspected by Gate 2
  features.parquet          final matrix, zero NaNs in FEATURE_COLS
  fill_values.json          the fitted FillPolicy, required at inference

Run:
    python -m src.build_features
    python -m src.build_features --raw data/raw_results.parquet --out-dir data/v2

Then: python -m src.audit_leakage   (VALIDATION GATE 2 -- do not skip)
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .columns import FEATURE_COLS, ID_COLS, TARGET
from .features import build_asof_features
from .preprocessing import FillPolicy, assert_no_missing
from .splits import chronological_split

log = logging.getLogger("build_features")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_PATH = DATA_DIR / "raw_results.parquet"
PREFILL_NAME = "features_prefill.parquet"
FEATURES_NAME = "features.parquet"
FILLS_NAME = "fill_values.json"

# Raw columns worth keeping alongside the matrix even though nothing models
# them (qualifying pace is unused in v1 but 98.9% populated -- FIX_PLAN.md
# section 2, P1, earmarks it as the first feature experiment).
EXTRA_COLS = ("quali_best_s", "gap_to_pole_s")


def build(raw_path: Path = RAW_PATH, out_dir: Path = DATA_DIR) -> pd.DataFrame:
    raw = pd.read_parquet(raw_path)
    raw["date"] = pd.to_datetime(raw["date"])

    # Pass no policy: this is the pre-imputation frame.
    df = build_asof_features(raw)

    keep = [c for c in ID_COLS if c in df.columns] + FEATURE_COLS + [TARGET]
    extra = [c for c in EXTRA_COLS if c in df.columns]
    df = df[keep + extra].copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / PREFILL_NAME, index=False)

    # Fit on the training partition ONLY, then apply to everything. Fitting on
    # `df` would let held-out outcomes shape the constants imputed into
    # training rows.
    train, _, _, latest = chronological_split(df)
    policy = FillPolicy.fit(train)
    log.info("Fill policy fitted on %s training rows (%s-%s, %s races); "
             "mean finish %.4f (full-data would have been %.4f)",
             policy.fitted_on["n_rows"], policy.fitted_on["year_min"],
             policy.fitted_on["year_max"], policy.fitted_on["n_races"],
             policy.constants["global_mean_finish"], df["position"].mean())

    df = policy.transform(df)
    assert_no_missing(df, FEATURE_COLS)

    df.to_parquet(out_dir / FEATURES_NAME, index=False)
    policy.to_json(out_dir / FILLS_NAME)

    log.info("Wrote %s (%s rows, %s features)",
             out_dir / FEATURES_NAME, len(df), len(FEATURE_COLS))
    log.info("Wrote %s", out_dir / FILLS_NAME)
    return df


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_PATH)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR,
                        help="where features/prefill/fill_values go. Point this at "
                             "a new directory to avoid overwriting artifacts hashed "
                             "in reports/baseline.json.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    df = build(args.raw, args.out_dir)
    print(f"\n{FEATURES_NAME}: {len(df)} rows x {len(FEATURE_COLS)} features; "
          f"target mean = {df[TARGET].mean():.4f}")
    print("Next: python -m src.audit_leakage   (VALIDATION GATE 2 -- mandatory)")


if __name__ == "__main__":
    main()
