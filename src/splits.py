"""Chronological partitioning (spec 4.1).

Lives in its own module so that anything needing the split -- feature building,
preprocessing, training, backtesting -- can import it without pulling in
LightGBM. `train.py` re-exports `chronological_split` for existing callers.
"""
from __future__ import annotations

import pandas as pd

TRAIN = "train"
VAL = "val"
TEST = "test"


def chronological_split(df: pd.DataFrame):
    """train: <= latest-2, val: latest-1, test: latest.

    Never split driver rows randomly: races are sequential and correlated, so a
    random split lets a model see the future (FIX_PLAN.md section 8).
    """
    latest = int(df["year"].max())
    train = df[df["year"] <= latest - 2]
    val = df[df["year"] == latest - 1]
    test = df[df["year"] == latest]
    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError(
            f"Chronological split produced an empty set "
            f"(latest={latest}, sizes: train={len(train)}, val={len(val)}, "
            f"test={len(test)}). Need at least 3 seasons of data."
        )
    return train, val, test, latest


def split_labels(df: pd.DataFrame) -> pd.Series:
    """Per-row partition name, for auditing which rows fitted what."""
    latest = int(df["year"].max())
    return pd.Series(
        pd.cut(df["year"], bins=[-1, latest - 2, latest - 1, latest],
               labels=[TRAIN, VAL, TEST]).astype(str),
        index=df.index, name="partition")
