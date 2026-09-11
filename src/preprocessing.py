"""Fitted missing-value policy (FIX_PLAN.md section 2, P0-2 and P0-3).

Two defects are fixed together here because they are one defect seen from two
sides:

**P0-2, fitted on everything.** `build_features.apply_fill_policy()` computed
its constants over the whole frame before `chronological_split()`, so held-out
outcomes shaped the values imputed into training rows. On this dataset the
full-data mean finishing position is 10.5982 against 10.4790 on the training
partition.

**P0-3, train and serve disagreed.** Training filled position-scaled features
with the driver's own prior average and only then fell back to a global
constant. `predict.py` applied the global constants alone, so 2091 cells got a
different value at serving than the identical row got in training.

The fix is one fitted object: `FillPolicy.fit()` sees training rows only,
`FillPolicy.transform()` is the single implementation both paths call, and
`to_json()` stores the fitted constants **together with the category lists that
define the fallback chain**. Serving replays the saved lists, not the current
source code, so editing the categories cannot silently change how an existing
model's inputs are imputed.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

SCHEMA_VERSION = 2

# Finishing-position-scaled: never filled with 0, which would read as "won".
# Fallback chain is driver_overall_avg_finish -> global mean finish.
POSITION_SCALED = [
    "form_avg_finish_3", "form_avg_quali_3", "season_avg_finish",
    "driver_circuit_avg_finish", "driver_circuit_best_finish",
    "team_circuit_avg_finish",
]
# 0-1 rates -> the global mean of that rate, named so the JSON is self-describing.
RATE_FEATURES = {
    "form_dnf_rate_5": "dnf_rate",
    "driver_dnf_rate": "dnf_rate",
    "team_dnf_rate": "dnf_rate",
    "driver_circuit_podium_rate": "podium_rate",
}
# Deltas, counts and sums where 0 is the neutral value.
ZERO_FILL = [
    "form_avg_points_3", "momentum", "driver_wet_delta", "quali_gap_to_teammate",
    "form_finish_vs_teammate", "driver_wet_n", "driver_races_at_circuit",
    "constructor_standing_prior", "driver_pace_races",
]
FLAG_DEFAULTS = {"is_rookie_here": 1.0, "teammate_available": 0.0, "pit_start": 0.0}
# Retained for auditing only -- no longer model inputs (increment 1.4).
WEATHER_MEAN_FILL = ["air_temp", "track_temp", "humidity", "wind_speed"]
WEATHER_ZERO_FILL = ["rainfall", "is_wet"]

DRIVER_PRIOR = "driver_overall_avg_finish"
GRID_DEFAULT = 20.0

# Features whose missingness is INFORMATION, not an absent measurement. A
# driver with no Q3 time did not reach Q3; imputing a value would erase exactly
# the fact the model should learn. LightGBM splits on NaN natively, which
# FIX_PLAN.md section 5.A.7 prefers ("native missing-value handling where
# suitable"). These are excluded from the no-NaN assertion.
from .qualifying import NATIVE_MISSING as _QUALI_MISSING  # noqa: E402
from .practice import NATIVE_MISSING as _PRACTICE_MISSING  # noqa: E402
from .ratings import NATIVE_MISSING as _PACE_MISSING  # noqa: E402

NATIVE_MISSING = _QUALI_MISSING + _PACE_MISSING + _PRACTICE_MISSING


@dataclass
class FillPolicy:
    """Fitted imputation state. Fit once on training rows, then freeze."""

    constants: dict[str, float]
    weather_means: dict[str, float]
    position_scaled: list[str] = field(default_factory=lambda: list(POSITION_SCALED))
    rate_features: dict[str, str] = field(default_factory=lambda: dict(RATE_FEATURES))
    zero_fill: list[str] = field(default_factory=lambda: list(ZERO_FILL))
    flag_defaults: dict[str, float] = field(default_factory=lambda: dict(FLAG_DEFAULTS))
    weather_zero_fill: list[str] = field(default_factory=lambda: list(WEATHER_ZERO_FILL))
    fitted_on: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    # -- fitting ------------------------------------------------------------
    @classmethod
    def fit(cls, train: pd.DataFrame) -> "FillPolicy":
        """Fit on the TRAINING partition only. Never pass the full frame."""
        if train.empty:
            raise ValueError("Cannot fit a fill policy on an empty training set.")
        constants = {
            "global_mean_finish": float(train["position"].mean()),
            "podium_rate": float((train["position"] <= 3).mean()),
            "dnf_rate": float(train["is_dnf"].mean()),
            "grid_position_default": GRID_DEFAULT,
        }
        weather_means = {c: float(train[c].mean())
                         for c in WEATHER_MEAN_FILL if c in train.columns}
        return cls(
            constants=constants,
            weather_means=weather_means,
            fitted_on={
                "n_rows": int(len(train)),
                "year_min": int(train["year"].min()),
                "year_max": int(train["year"].max()),
                "n_races": int(train.groupby(["year", "round"]).ngroups),
            },
        )

    # -- applying -----------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the frozen policy. Identical at training and at inference.

        Order matters: position-scaled features fall back to the driver's own
        prior average FIRST, so `driver_overall_avg_finish` must keep its NaNs
        until that step is done.
        """
        out = df.copy()
        mean_finish = self.constants["global_mean_finish"]

        for col in self.position_scaled:
            if col not in out.columns:
                continue
            if DRIVER_PRIOR in out.columns:
                out[col] = out[col].fillna(out[DRIVER_PRIOR])
            out[col] = out[col].fillna(mean_finish)

        for col, constant_name in self.rate_features.items():
            if col in out.columns:
                out[col] = out[col].fillna(self.constants[constant_name])

        for col in self.zero_fill:
            if col in out.columns:
                out[col] = out[col].fillna(0.0)

        for col, value in self.flag_defaults.items():
            if col in out.columns:
                out[col] = out[col].fillna(value).astype(int)

        for col, value in self.weather_means.items():
            if col in out.columns:
                out[col] = out[col].fillna(value)
        for col in self.weather_zero_fill:
            if col in out.columns:
                out[col] = out[col].fillna(0.0)

        if "grid_position" in out.columns:
            out["grid_position"] = out["grid_position"].fillna(
                self.constants["grid_position_default"])

        # Filled last: it is the fallback source for the block above.
        if DRIVER_PRIOR in out.columns:
            out[DRIVER_PRIOR] = out[DRIVER_PRIOR].fillna(mean_finish)
        return out

    # -- persistence --------------------------------------------------------
    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> "FillPolicy":
        with Path(path).open(encoding="utf-8") as fh:
            data = json.load(fh)
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"{path} has fill-policy schema {version!r}, this code expects "
                f"{SCHEMA_VERSION}. A model must be served with the policy it "
                f"was trained with -- refit rather than reinterpreting it."
            )
        return cls(**data)


def assert_no_missing(df: pd.DataFrame, feature_cols: list[str],
                      allow_missing: tuple[str, ...] = NATIVE_MISSING) -> None:
    """Fail loudly if any model input is UNEXPECTEDLY NaN after transform.

    `allow_missing` names features whose NaN is deliberate and must reach the
    model intact. Everything else must be filled: a NaN there means the policy
    has a gap, and it would arrive at the model as an accident rather than a
    decision.
    """
    checked = [c for c in feature_cols if c not in allow_missing]
    counts = df[checked].isna().sum()
    bad = counts[counts > 0]
    if len(bad):
        raise AssertionError(f"NaNs remain in the feature matrix:\n{bad}")
