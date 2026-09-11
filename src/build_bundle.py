"""Build a deployable model bundle (FIX_PLAN.md section 11).

Two-stage, because the recipe and the final fit need different data:

1. **Develop.** Split chronologically, fit on the training partition, early-stop
   on validation. This yields the iteration count, the calibration temperature
   and the blend weight -- all chosen without the test season.
2. **Refit on everything.** Retrain with the iteration count frozen, on ALL
   eligible history. FIX_PLAN.md section 11 requires this: "deployment must
   refit the selected recipe using all eligible past results", and the
   evaluation convention otherwise ships a model that never saw the two most
   recent seasons.

The imputation policy is refitted on the full set too, since at deployment
every completed race IS training data. The manifest records the resulting
training cutoff, which is what stops the bundle being used to "predict" a race
it was fitted on.

Run:
    python -m src.build_bundle --features data/v2/features.parquet \\
                               --out models/champion
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from .backtest import select_alpha
from .bundle import ModelBundle, build_manifest
from .columns import FEATURE_COLS
from .metrics import ensure_labels, race_metrics
from .preprocessing import FillPolicy, assert_no_missing
from .probabilities import (fit_temperature, uniform_winner_log_loss,
                            winner_log_loss, add_probabilities)
from .splits import chronological_split
from .train_rank import PARAMS, add_relevance_label, group_sizes

log = logging.getLogger("build_bundle")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def build(features_path: Path, raw_path: Path, out_dir: Path) -> ModelBundle:
    prefill_path = features_path.parent / "features_prefill.parquet"
    source = prefill_path if prefill_path.exists() else features_path
    df = pd.read_parquet(source)
    df["date"] = pd.to_datetime(df["date"])

    # --- stage 1: develop the recipe -------------------------------------
    dev_policy = FillPolicy.fit(chronological_split(df)[0])
    dev = add_relevance_label(ensure_labels(dev_policy.transform(df)))
    train, val, _, latest = chronological_split(dev)

    ranker = lgb.LGBMRanker(**PARAMS)
    ranker.fit(train[FEATURE_COLS], train["relevance"], group=group_sizes(train),
               eval_set=[(val[FEATURE_COLS], val["relevance"])],
               eval_group=[group_sizes(val)], eval_at=[3, 1, 10],
               # NDCG@3 is the PRIMARY early-stopping metric and
               # first_metric_only makes that explicit. Without it LightGBM
               # stops as soon as ANY eval metric stalls, and NDCG@1 on a
               # ~22-race validation block is far too noisy to govern training:
               # it truncated folds to as few as 4 iterations
               # (FIX_PLAN.md section 6).
               callbacks=[lgb.early_stopping(100, first_metric_only=True,
                                             verbose=False),
                          lgb.log_evaluation(0)])
    best_iteration = int(ranker.best_iteration_ or PARAMS["n_estimators"])

    scored_val = val.assign(rank_score=ranker.predict(val[FEATURE_COLS]))
    temperature = fit_temperature(scored_val, "rank_score")
    alpha = select_alpha(scored_val)
    log.info("Recipe from development: %s iterations, T=%.4f, alpha=%.2f",
             best_iteration, temperature, alpha)

    validation = {
        "development_test_season": latest,
        "val_metrics": {k: v for k, v in
                        race_metrics(scored_val, "rank_score", False).items()
                        if isinstance(v, (int, float)) or v is None},
        "val_winner_log_loss": winner_log_loss(
            add_probabilities(scored_val, "rank_score", temperature)),
        "val_uniform_log_loss": uniform_winner_log_loss(scored_val),
    }

    # --- stage 2: refit the frozen recipe on ALL history ------------------
    policy = FillPolicy.fit(df)             # every completed race is training data
    full = add_relevance_label(ensure_labels(policy.transform(df)))
    assert_no_missing(full, FEATURE_COLS)

    final_params = {**PARAMS, "n_estimators": best_iteration}
    champion = lgb.LGBMRanker(**final_params)
    champion.fit(full[FEATURE_COLS], full["relevance"], group=group_sizes(full))

    manifest = build_manifest(
        train=full, feature_cols=FEATURE_COLS, temperature=temperature,
        alpha=alpha, best_iteration=best_iteration, raw_path=raw_path,
        project_root=PROJECT_ROOT, validation=validation,
        notes=[
            "Recipe (iterations, temperature, alpha) chosen on the development "
            "split; the ranker is then refit on ALL history with the iteration "
            "count frozen (FIX_PLAN.md section 11).",
            "Champion is the ranker plus the Plackett-Luce probability layer. "
            "The specialist-head ensemble is a CHALLENGER and is deliberately "
            "not included: it scored worse on the backtest (REASONING [012]).",
            "Winner accuracy does not beat the grid baseline by the FIX_PLAN "
            "section 8 margin. Probability quality does. Treat the ORDER as "
            "comparable to grid and the PROBABILITIES as the useful output.",
        ])

    ModelBundle.save(out_dir, champion, policy, manifest)
    return ModelBundle.load(out_dir, FEATURE_COLS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path,
                        default=DATA_DIR / "v2" / "features.parquet")
    parser.add_argument("--raw", type=Path,
                        default=DATA_DIR / "v2" / "raw_results.parquet")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "models" / "champion")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    bundle = build(args.features, args.raw, args.out)

    m = bundle.manifest
    print(f"\n=================== BUNDLE BUILT ===================")
    print(bundle.describe())
    print(f"  training cutoff : {pd.Timestamp(m.training_cutoff_utc).date()}  "
          f"(a target race must be AFTER this)")
    print(f"  rows / races    : {m.n_train_rows} / {m.n_train_races}")
    print(f"  iterations      : {m.best_iteration}   alpha: {m.alpha}")
    print(f"  data sha256     : {(m.data_sha256 or '')[:16]}")
    print(f"  code revision   : {(m.code_revision or 'unknown')[:12]}")
    print(f"\nWrote {args.out}")
    print("====================================================")


if __name__ == "__main__":
    main()
