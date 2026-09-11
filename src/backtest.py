"""Expanding-window backtesting (FIX_PLAN.md sections 8 and 9, milestone 2).

The held-out 2026 season is 11 races. One race is 9.1 percentage points of
winner accuracy, so it cannot distinguish a real improvement from noise. Every
feature and model decision from milestone 3 onward depends on having more
evaluated races and an interval around the difference.

What this module guarantees:

- **Event-level folds, never random rows.** Races are sequential and
  correlated; a random split lets a model see the future.
- **Nothing from the test block reaches the fit.** Each fold refits the
  imputation policy AND the models on that fold's training rows alone.
- **Manifests are frozen and hashed.** A fold's event list is recorded with a
  SHA-256 so a later run cannot quietly evaluate on a different set of races.
- **Predictions are exported per fold**, so errors can be diagnosed after the
  fact rather than re-derived.

Two fold schemes, because the available data cannot support the preferred one:

`season`
    One outer fold per held-out season. This is what FIX_PLAN.md section 8 asks
    for, but 2022-2026 yields only 2 folds / 35 races against its target of 3+
    folds and 60+ races. Re-ingesting 2018-2021 is what fixes that.
`rolling`
    Expanding training window with fixed-size race blocks as the test set. More
    folds and more evaluated races from the same data, at the cost of fitting
    on partial seasons. FIX_PLAN.md explicitly allows this fallback and
    requires the weaker evidence to be reported as such.

Run:
    python -m src.backtest --scheme rolling --features data/v2/features.parquet
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Iterator

import numpy as np
import pandas as pd

from .blend_rank import add_blended_score
from .columns import FEATURE_COLS, TARGET
from .metrics import ensure_labels
from .preprocessing import FillPolicy
from .probabilities import add_probabilities, fit_temperature

log = logging.getLogger("backtest")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

RACE_KEYS = ["year", "round"]

SEASON = "season"
ROLLING = "rolling"


def race_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per race, in chronological order."""
    return (df.groupby(RACE_KEYS, as_index=False)
              .agg(date=("date", "first"), n_rows=("driver", "size"))
              .sort_values("date", kind="mergesort")
              .reset_index(drop=True))


def _event_ids(races: pd.DataFrame) -> list[str]:
    return [f"{int(r.year)}-{int(r.round):02d}" for r in races.itertuples()]


def _manifest(name: str, train: pd.DataFrame, val: pd.DataFrame,
              test: pd.DataFrame) -> dict[str, Any]:
    """Frozen, hashed description of exactly which races a fold used."""
    events = {"train": _event_ids(train), "val": _event_ids(val),
              "test": _event_ids(test)}
    payload = json.dumps(events, sort_keys=True).encode()
    return {
        "name": name,
        "n_races": {k: len(v) for k, v in events.items()},
        "date_ranges": {
            k: [str(frame["date"].min()), str(frame["date"].max())] if len(frame) else None
            for k, frame in (("train", train), ("val", val), ("test", test))
        },
        "events": events,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


@dataclass
class Fold:
    """One outer fold. Index labels refer to rows of the frame folds were built from."""

    name: str
    train: pd.Index
    val: pd.Index
    test: pd.Index
    manifest: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = getattr(self, a).intersection(getattr(self, b))
            if len(overlap):
                raise ValueError(
                    f"Fold {self.name}: {a} and {b} share {len(overlap)} rows. "
                    f"A row must belong to exactly one partition.")


def _fold_from_races(df: pd.DataFrame, name: str, train_races: pd.DataFrame,
                     val_races: pd.DataFrame, test_races: pd.DataFrame) -> Fold:
    def rows(races: pd.DataFrame) -> pd.Index:
        if not len(races):
            return df.index[:0]
        keys = set(zip(races["year"], races["round"]))
        mask = [(y, r) in keys for y, r in zip(df["year"], df["round"])]
        return df.index[np.asarray(mask)]

    return Fold(name=name, train=rows(train_races), val=rows(val_races),
                test=rows(test_races),
                manifest=_manifest(name, train_races, val_races, test_races))


def season_folds(df: pd.DataFrame, min_train_seasons: int = 2,
                 train_from: int | None = None) -> list[Fold]:
    """One fold per held-out season; the season before it is the inner validation.

    `train_from` truncates the START of the training window, leaving the test
    and validation seasons untouched. That is what makes an era comparison
    fair: both arms are scored on exactly the same races and differ only in how
    much history they were allowed to learn from (FIX_PLAN.md section 5.A.4,
    "evaluate both 2018+ and recent-era windows").
    """
    races = race_table(df)
    seasons = sorted(races["year"].unique())
    folds = []
    for i, year in enumerate(seasons):
        if i < min_train_seasons + 1:
            continue          # need training seasons plus one for inner validation
        val_year = seasons[i - 1]
        train = races[races["year"] < val_year]
        if train_from is not None:
            train = train[train["year"] >= train_from]
        if train["year"].nunique() < min_train_seasons:
            continue          # not enough history left after truncation
        folds.append(_fold_from_races(
            df, name=f"season-{int(year)}",
            train_races=train,
            val_races=races[races["year"] == val_year],
            test_races=races[races["year"] == year]))
    return folds


def rolling_folds(df: pd.DataFrame, min_train_races: int = 44,
                  val_races: int = 12, block_races: int = 6) -> list[Fold]:
    """Expanding training window, fixed-size race blocks held out in turn.

    The inner validation block sits between training and test, so early stopping
    never sees the races it will be scored on.
    """
    races = race_table(df)
    folds = []
    start = min_train_races + val_races
    for i, block_start in enumerate(range(start, len(races), block_races)):
        test = races.iloc[block_start:block_start + block_races]
        if not len(test):
            break
        val = races.iloc[block_start - val_races:block_start]
        train = races.iloc[:block_start - val_races]
        folds.append(_fold_from_races(df, name=f"block-{i + 1:02d}",
                                      train_races=train, val_races=val,
                                      test_races=test))
    return folds


def make_folds(df: pd.DataFrame, scheme: str = ROLLING, **kwargs) -> list[Fold]:
    if scheme == SEASON:
        return season_folds(df, **kwargs)
    if scheme == ROLLING:
        return rolling_folds(df, **kwargs)
    raise ValueError(f"Unknown fold scheme {scheme!r}; expected {SEASON!r} or {ROLLING!r}.")


# Blend weights swept inside each fold. 0 = ranker only, 1 = grid only.
ALPHA_GRID: Final = tuple(round(a, 2) for a in np.arange(0.0, 1.01, 0.1))

# The objective alpha is selected against. Winner accuracy is the project's
# stated goal; podium overlap breaks ties between alphas that pick the same
# number of winners, which is common on a 20-race validation block.
ALPHA_OBJECTIVE: Final = ("winner_accuracy", "podium_overlap")


def select_alpha(val: pd.DataFrame,
                 grid: tuple[float, ...] = ALPHA_GRID) -> float:
    """Choose the blend weight on VALIDATION rows, by winner then podium.

    Never call this on training or test rows. Selecting on test is leakage;
    selecting on training overfits the weight to data the models already saw.
    """
    from .metrics import race_metrics

    best_alpha, best_score = grid[-1], None
    for alpha in grid:
        scored = val.assign(_b=add_blended_score(val, "rank_score", alpha))
        m = race_metrics(scored, "_b", ascending=True)
        key = tuple((m[name] if m[name] is not None else -1.0)
                    for name in ALPHA_OBJECTIVE)
        # `>=` with an ascending grid means ties go to the HIGHER alpha, i.e.
        # to the grid baseline. The model has to demonstrably beat the baseline
        # to earn weight; where nothing separates the options, the simpler one
        # wins. With `>` the sweep silently defaulted to alpha 0 -- full weight
        # on the ranker for free -- whenever no alpha stood out.
        if best_score is None or key >= best_score:
            best_score, best_alpha = key, alpha
    return float(best_alpha)


# ---------------------------------------------------------------------------
# Fitting one fold
# ---------------------------------------------------------------------------
def fit_and_predict(prefill: pd.DataFrame, fold: Fold,
                    params: dict | None = None,
                    rank_params: dict | None = None) -> pd.DataFrame:
    """Refit the policy and both models on `fold.train`, score `fold.test`.

    `prefill` must be the PRE-IMPUTATION frame: the policy is fitted inside the
    fold, so a frame that has already been imputed would carry constants fitted
    on data this fold is not allowed to see.
    """
    import lightgbm as lgb

    from .train import PARAMS as CLF_PARAMS
    from .train_rank import PARAMS as RANK_PARAMS
    from .train_rank import add_relevance_label, group_sizes

    policy = FillPolicy.fit(prefill.loc[fold.train])
    data = policy.transform(prefill)
    data = add_relevance_label(ensure_labels(data))

    train, val, test = data.loc[fold.train], data.loc[fold.val], data.loc[fold.test]

    clf = lgb.LGBMClassifier(**{**CLF_PARAMS, **(params or {})})
    clf.fit(train[FEATURE_COLS], train[TARGET],
            eval_set=[(val[FEATURE_COLS], val[TARGET])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=False),
                       lgb.log_evaluation(0)])

    ranker = lgb.LGBMRanker(**{**RANK_PARAMS, **(rank_params or {})})
    ranker.fit(train[FEATURE_COLS], train["relevance"], group=group_sizes(train),
               eval_set=[(val[FEATURE_COLS], val["relevance"])],
               eval_group=[group_sizes(val)], eval_at=[1, 3, 10],
               callbacks=[lgb.early_stopping(100, verbose=False),
                          lgb.log_evaluation(0)])

    out = test[RACE_KEYS + ["driver", "team", "grid_position", "result_order",
                            "is_winner", "is_podium", TARGET, "finished"]].copy()
    out["fold"] = fold.name
    out["p_top10"] = clf.predict_proba(test[FEATURE_COLS])[:, 1]
    out["rank_score"] = ranker.predict(test[FEATURE_COLS])

    # Blend weight chosen INSIDE this fold's validation block, against the
    # objective the project is actually judged on. Two defects closed at once:
    # alpha was a fixed constant carried in from outside every fold (a small
    # leak), and it was selected by Spearman while winner/podium is the goal
    # (FIX_PLAN.md section 2, P1). The alpha sweep showed the second is not
    # theoretical -- Spearman peaks near 0.5 while winner accuracy peaks at 0.
    alpha = select_alpha(val.assign(
        rank_score=ranker.predict(val[FEATURE_COLS])))
    out["alpha"] = alpha
    out["blend_score"] = add_blended_score(out, "rank_score", alpha)

    # Race-level probabilities. The temperature is fitted on THIS fold's
    # validation block: it changes confidence without changing order, so
    # fitting it on the test block would flatter every probability metric while
    # leaving the ranking metrics untouched -- an easy leak to miss.
    temperature = fit_temperature(
        val.assign(rank_score=ranker.predict(val[FEATURE_COLS])), "rank_score")
    out["temperature"] = temperature
    out = add_probabilities(out, "rank_score", temperature)

    # A probabilistic GRID baseline. A deterministic order has no win
    # probabilities of its own, so "better winner log loss" needs something to
    # be better than (FIX_PLAN.md section 8: "a one-parameter grid-utility
    # distribution fitted on development data"). Utility is minus the grid
    # position, with its own temperature fitted on the same validation block.
    val_grid = val.assign(_u=-val["grid_position"].astype(float))
    grid_temperature = fit_temperature(val_grid, "_u")
    out["_u"] = -out["grid_position"].astype(float)
    grid_probs = add_probabilities(out, "_u", grid_temperature)
    for column in ("p_win", "p_podium", "p_top10"):
        out[f"{column}_grid"] = grid_probs[column].to_numpy()
    out["grid_temperature"] = grid_temperature
    out = out.drop(columns=["_u"])

    out.attrs["policy"] = policy.constants
    out.attrs["alpha"] = alpha
    out.attrs["best_iteration"] = {"classifier": clf.best_iteration_,
                                   "ranker": ranker.best_iteration_}
    return out


def run(prefill: pd.DataFrame, folds: list[Fold]) -> tuple[pd.DataFrame, list[dict]]:
    """Score every fold. Returns pooled predictions and the fold manifests."""
    frames, manifests = [], []
    for fold in folds:
        log.info("fold %s: train=%s races, val=%s, test=%s",
                 fold.name, fold.manifest["n_races"]["train"],
                 fold.manifest["n_races"]["val"], fold.manifest["n_races"]["test"])
        predictions = fit_and_predict(prefill, fold)
        manifests.append({**fold.manifest,
                          "alpha": predictions.attrs["alpha"],
                          "temperature": float(predictions["temperature"].iloc[0]),
                          "best_iteration": predictions.attrs["best_iteration"],
                          "fitted_constants": predictions.attrs["policy"]})
        frames.append(predictions)
    return pd.concat(frames, ignore_index=True), manifests


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill", type=Path,
                        default=DATA_DIR / "v2" / "features_prefill.parquet",
                        help="PRE-imputation frame; the policy is refitted per fold")
    parser.add_argument("--scheme", choices=[SEASON, ROLLING], default=ROLLING)
    parser.add_argument("--train-from", type=int, default=None,
                        help="earliest season allowed in TRAINING. Test and "
                             "validation seasons are unchanged, so two runs "
                             "with different values are directly comparable.")
    parser.add_argument("--min-train-races", type=int, default=44)
    parser.add_argument("--val-races", type=int, default=12)
    parser.add_argument("--block-races", type=int, default=6)
    parser.add_argument("--out-dir", type=Path, default=REPORTS_DIR / "backtest")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    prefill = pd.read_parquet(args.prefill)
    prefill["date"] = pd.to_datetime(prefill["date"])

    if args.scheme == SEASON:
        folds = make_folds(prefill, SEASON, train_from=args.train_from)
    else:
        folds = make_folds(prefill, ROLLING, min_train_races=args.min_train_races,
                           val_races=args.val_races, block_races=args.block_races)
    if not folds:
        raise SystemExit(
            "No folds could be built. There is not enough history for the "
            "requested scheme; lower --min-train-races or re-ingest earlier seasons.")

    predictions, manifests = run(prefill, folds)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(args.out_dir / "predictions.parquet", index=False)
    with (args.out_dir / "folds.json").open("w", encoding="utf-8") as fh:
        json.dump({"scheme": args.scheme, "folds": manifests}, fh, indent=2)

    n_races = predictions.groupby(RACE_KEYS).ngroups
    print(f"\nscheme            : {args.scheme}")
    print(f"folds             : {len(folds)}")
    print(f"races evaluated   : {n_races}")
    print(f"driver-race rows  : {len(predictions)}")
    print(f"\nWrote {args.out_dir / 'predictions.parquet'}")
    print(f"Wrote {args.out_dir / 'folds.json'}")

    # FIX_PLAN.md section 8 asks for at least three outer periods and preferably
    # 60+ evaluated races before an improvement claim is credible.
    if len(folds) < 3 or n_races < 60:
        print(f"\nWARNING: {len(folds)} folds / {n_races} races is below the "
              f"FIX_PLAN.md section 8 target of 3+ folds and 60+ races. "
              f"Evidence from this backtest is WEAK; report it as such. "
              f"Re-ingesting 2018-2021 is what fixes this.")
    print("\nNext: python -m src.gates   (paired intervals and promotion gates)")


if __name__ == "__main__":
    main()
