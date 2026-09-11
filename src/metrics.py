"""Race-level evaluation metrics with explicit labels and denominators.

Replaces the near-duplicate `evaluate.ranking_metrics` and
`evaluate_rank.order_metrics`, which computed Spearman from copy-pasted blocks
and reported a single `n_races` that did not match every metric's actual
denominator.

Three corrections over the inherited behaviour (FIX_PLAN.md sections 2 and 8):

1. **Deterministic ordering.** `rank(method='first')` broke ties by row order,
   so shuffling identical rows changed both the forecast and its score
   (measured spread: 0.0080 Spearman over five shuffles). Ordering is now
   score, then grid position known at the cutoff, then driver id -- all
   pre-race information, and a total order because driver is unique per race.

2. **Explicit outcome labels.** Winner and podium come from `is_winner` /
   `is_podium` (src.labels), not from ad-hoc `(classified == 1) & (position <=
   k)` tests against a flag that is true for 99.9% of rows.

3. **Separate denominators.** `spearman_all` covers every entry against the
   published result order; `spearman_finishers` covers only cars that reached
   the flag. The inherited single "spearman" was the former while the README
   described it as the latter. Both are reported, each with its own race count,
   so neither can be quoted with the wrong meaning.
"""
from __future__ import annotations

from typing import Any, Final

import numpy as np
import pandas as pd

from .labels import LABEL_COLS, derive_labels

# Applied after the score, in order. Both are known before lights-out, so the
# displayed order never depends on information the forecast could not have had.
TIE_BREAKERS: Final = ("grid_position", "driver")

# Set-overlap denominators are fixed by the spec even when a race classifies
# fewer finishers, so the metric stays comparable across races.
PODIUM_SIZE: Final = 3
TOP10_SIZE: Final = 10

RACE_KEYS: Final = ("year", "round")


def ensure_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame guaranteed to carry the src.labels fields.

    features.parquet predates those columns but does retain `status` and
    `position`, and derive_labels is a pure row-wise transform, so they can be
    reconstructed without rebuilding any artifact.
    """
    if all(col in df.columns for col in LABEL_COLS):
        return df
    return derive_labels(df)


def assign_pred_rank(race: pd.DataFrame, score_col: str,
                     ascending: bool) -> pd.Series:
    """Deterministic 1..n predicted order for one race.

    Returns a Series aligned to `race.index`. Ties in `score_col` are broken by
    grid position then driver id, so the result depends only on the rows
    present -- never on the order they arrived in.
    """
    missing = [c for c in TIE_BREAKERS if c not in race.columns]
    if missing:
        raise KeyError(f"Tie-breaker columns missing: {missing}. The "
                       f"deterministic order needs {list(TIE_BREAKERS)}.")

    ordered = race.sort_values(
        [score_col, *TIE_BREAKERS],
        ascending=[ascending] + [True] * len(TIE_BREAKERS),
        kind="mergesort",
    )
    return pd.Series(np.arange(1, len(ordered) + 1), index=ordered.index,
                     dtype="int64").reindex(race.index)


def pred_rank_by_race(df: pd.DataFrame, score_col: str,
                      ascending: bool) -> pd.Series:
    """Deterministic predicted rank for every row, computed within each race.

    The grouped equivalent of assign_pred_rank. Use this anywhere a score is
    turned into a displayed order -- including the blend and the inference
    path, not only evaluation, since an order-dependent forecast cannot be
    reproduced or audited.
    """
    parts = [assign_pred_rank(race, score_col, ascending)
             for _, race in df.groupby(list(RACE_KEYS), sort=False)]
    return pd.concat(parts).reindex(df.index)


def _spearman(pred_rank: pd.Series, actual: pd.Series) -> float | None:
    """Spearman correlation, or None when the sample cannot support one."""
    pair = pd.DataFrame({"pred": pred_rank, "actual": actual}).dropna()
    if len(pair) < 3 or pair["pred"].nunique() < 2 or pair["actual"].nunique() < 2:
        return None
    return float(pair["pred"].corr(pair["actual"], method="spearman"))


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def race_metrics(df: pd.DataFrame, score_col: str, ascending: bool) -> dict[str, Any]:
    """Per-race metrics averaged over races, with per-metric denominators.

    `ascending=True` means a lower score predicts a better finish (grid
    position, blended rank); `False` means higher is better (probabilities,
    ranker scores).
    """
    data = ensure_labels(df)

    winner_hits: list[float] = []
    podium_overlaps: list[float] = []
    top10_overlaps: list[float] = []
    top_pick_top10: list[float] = []
    spearman_all: list[float] = []
    spearman_finishers: list[float] = []

    n_races = 0
    for _, race in data.groupby(list(RACE_KEYS), sort=False):
        n_races += 1
        race = race.assign(pred_rank=assign_pred_rank(race, score_col, ascending))
        ordered = race.sort_values("pred_rank")

        actual_winner = set(race.loc[race["is_winner"] == 1, "driver"])
        actual_podium = set(race.loc[race["is_podium"] == 1, "driver"])
        actual_top10 = set(race.loc[race["finished_top10"] == 1, "driver"])

        top_pick = ordered.iloc[0]
        if actual_winner:
            winner_hits.append(float(top_pick["driver"] in actual_winner))
        if actual_podium:
            predicted = set(ordered.head(PODIUM_SIZE)["driver"])
            podium_overlaps.append(len(predicted & actual_podium) / PODIUM_SIZE)
        if actual_top10:
            predicted = set(ordered.head(TOP10_SIZE)["driver"])
            top10_overlaps.append(len(predicted & actual_top10) / TOP10_SIZE)
            top_pick_top10.append(float(top_pick["finished_top10"] == 1))

        value = _spearman(race["pred_rank"], race["result_order"])
        if value is not None:
            spearman_all.append(value)

        finishers = race[race["finished"] == 1]
        value = _spearman(finishers["pred_rank"], finishers["result_order"])
        if value is not None:
            spearman_finishers.append(value)

    return {
        "winner_accuracy": _mean(winner_hits),
        "podium_overlap": _mean(podium_overlaps),
        "top10_overlap": _mean(top10_overlaps),
        # Renamed from the inherited `top1_hit_rate`, which read as "picked the
        # winner" but is true whenever the top pick finishes anywhere in the
        # top ten (FIX_PLAN.md section 2, P0-5).
        "top_pick_finished_top10": _mean(top_pick_top10),
        "spearman_all": _mean(spearman_all),
        "spearman_finishers": _mean(spearman_finishers),
        "n_races": n_races,
        "n_races_winner": len(winner_hits),
        "n_races_podium": len(podium_overlaps),
        "n_races_top10": len(top10_overlaps),
        "n_races_spearman_all": len(spearman_all),
        "n_races_spearman_finishers": len(spearman_finishers),
    }


HEADLINE_COLS: Final = ("winner_accuracy", "podium_overlap", "top10_overlap",
                        "spearman_all", "spearman_finishers")


def comparison_table(df: pd.DataFrame,
                     orderings: dict[str, tuple[str, bool]]) -> pd.DataFrame:
    """Headline metrics for several ordering methods, one row each."""
    return pd.DataFrame(
        {name: race_metrics(df, col, ascending)
         for name, (col, ascending) in orderings.items()}).T
