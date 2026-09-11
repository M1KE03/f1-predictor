"""Specialist winner and podium heads (FIX_PLAN.md section 6, M4).

Every model so far optimises something other than the thing being judged. The
classifier optimises a top-10 flag; the ranker optimises full-field order. The
project is scored on winners and podiums, and increment 3.2 showed that
objective mismatch is not academic -- selecting the blend weight by Spearman
cost four points of winner accuracy.

These heads optimise the target directly: one binary model on `is_winner`, one
on `is_podium`.

How they are combined
---------------------
NOT by normalising their probabilities within a race. FIX_PLAN.md section 6 is
explicit that doing so and calling the result calibrated is wrong. Instead each
model contributes a UTILITY, the utilities are standardised within a race so
their scales are comparable, and the weighted sum feeds the same Plackett-Luce
layer as before. Win, podium and top-10 therefore still come from one
distribution and remain coherent.

Weights are selected on the VALIDATION block, where the base models were not
fitted -- "never train a meta-model on the base model's in-sample predictions".

Why log loss selects the weights
--------------------------------
Winner accuracy on a ~20-race validation block moves in steps of 5 percentage
points, which is far too coarse to tune three weights on. Winner log loss is
smooth and strictly proper, so a genuine improvement in the ranking of
probabilities shows up as a smaller loss. Accuracy is reported, never tuned on.
"""
from __future__ import annotations

from typing import Any, Final

import numpy as np
import pandas as pd

RACE_KEYS: Final = ["year", "round"]

# Winners are ~5% of rows and podiums ~15%, so the heads are deliberately small:
# a deep tree on 125 positive examples memorises drivers rather than learning
# conditions.
HEAD_PARAMS: Final = dict(
    objective="binary",
    learning_rate=0.05,
    num_leaves=7,
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.5,
    n_estimators=800,
    is_unbalance=True,
    random_state=42,
    verbose=-1,
)

# Ensemble weights swept on validation. Kept coarse on purpose: a fine grid on
# ~20 races fits noise.
WEIGHT_GRID: Final = (0.0, 0.25, 0.5, 0.75, 1.0)


def train_head(train: pd.DataFrame, val: pd.DataFrame, feature_cols: list[str],
               target: str):
    """Fit one specialist binary head with early stopping on `val`."""
    import lightgbm as lgb

    model = lgb.LGBMClassifier(**HEAD_PARAMS)
    model.fit(
        train[feature_cols], train[target],
        eval_set=[(val[feature_cols], val[target])],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def head_utility(model, frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Log-odds of the head's probability, which is the natural utility scale.

    Probabilities are bounded and compress differences at the extremes; log-odds
    are unbounded and add sensibly to another model's score.
    """
    p = model.predict_proba(frame[feature_cols])[:, 1]
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def standardise_within_race(values: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    """Z-score within each race so different models' scales are comparable.

    Within-race, not global: the ranking distribution only ever compares
    drivers inside one race, and a global scale would let a race with an
    unusually spread field dominate the weight search.
    """
    series = pd.Series(np.asarray(values, dtype=float), index=frame.index)
    grouped = series.groupby([frame["year"], frame["round"]])
    mean = grouped.transform("mean")
    sd = grouped.transform("std").replace(0.0, np.nan)
    return (series - mean).div(sd).fillna(0.0).to_numpy()


def combine(components: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Weighted sum of standardised utilities."""
    total = np.zeros(len(next(iter(components.values()))))
    for name, values in components.items():
        total = total + weights.get(name, 0.0) * np.asarray(values, dtype=float)
    return total


def select_weights(val: pd.DataFrame, components: dict[str, np.ndarray],
                   grid: tuple[float, ...] = WEIGHT_GRID) -> dict[str, float]:
    """Choose ensemble weights on VALIDATION, minimising winner log loss.

    The ranker keeps weight 1.0 as the reference; only the heads' weights are
    searched, so the ensemble can never be worse-specified than the ranker
    alone with both heads at zero.
    """
    from .probabilities import fit_temperature, softmax, winner_log_loss

    best = {"ranker": 1.0, "winner_head": 0.0, "podium_head": 0.0}
    best_loss = float("inf")

    for w_win in grid:
        for w_pod in grid:
            weights = {"ranker": 1.0, "winner_head": w_win, "podium_head": w_pod}
            scored = val.assign(_u=combine(components, weights))
            # Temperature is refitted for each weight combination: a different
            # utility scale needs a different temperature, and comparing losses
            # at a fixed temperature would reward whichever happened to suit it.
            temperature = fit_temperature(scored, "_u")
            probs = []
            for _, race in scored.groupby(RACE_KEYS, sort=False):
                probs.append(pd.Series(softmax(race["_u"].to_numpy(), temperature),
                                       index=race.index))
            scored["p_win"] = pd.concat(probs)
            loss = winner_log_loss(scored)
            if np.isfinite(loss) and loss < best_loss:
                best_loss, best = loss, weights
    return best


def fit_heads(train: pd.DataFrame, val: pd.DataFrame,
              feature_cols: list[str]) -> dict[str, Any]:
    """Train both heads and return them with their validation utilities."""
    winner = train_head(train, val, feature_cols, "is_winner")
    podium = train_head(train, val, feature_cols, "is_podium")
    return {"winner_head": winner, "podium_head": podium,
            "best_iteration": {"winner": winner.best_iteration_,
                               "podium": podium.best_iteration_}}
