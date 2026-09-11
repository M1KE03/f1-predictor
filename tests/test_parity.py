"""Train/serve parity and fitted-state tests (FIX_PLAN.md sections 2 P0-2/P0-3
and 10).

The replay test is the important one. It takes a real race out of the stored
dataset, hides its outcome, feeds it back through the inference path as a
not-yet-run event, and asserts the resulting feature vector is identical to the
one the training path produced for that race.

That single assertion covers two separate defects at once:

- **Serving parity.** Both paths must compute features the same way. They did
  not: training used a driver-prior fallback the server skipped.
- **Leakage.** If any feature for race R reads R's own outcomes -- its own, a
  teammate's, or any other driver's in the same race -- hiding those outcomes
  changes the vector and the test fails.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.columns import FEATURE_COLS
from src.features import OUTCOME_COLS, build_asof_features
from src.preprocessing import FillPolicy
from src.splits import chronological_split

RAW = "data/raw_results.parquet"


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    df = pd.read_parquet(RAW)
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture(scope="module")
def policy(raw: pd.DataFrame) -> FillPolicy:
    built = build_asof_features(raw)
    train, _, _, _ = chronological_split(built)
    return FillPolicy.fit(train)


def hide_outcomes(rows: pd.DataFrame) -> pd.DataFrame:
    """Turn completed rows into the placeholder shape src.predict builds."""
    hidden = rows.copy()
    for col in OUTCOME_COLS:
        if col not in hidden.columns:
            continue
        if col == "status":
            hidden[col] = None
        elif col in ("is_dnf", "classified"):
            hidden[col] = 0
        elif col == "points":
            hidden[col] = 0.0
        else:
            hidden[col] = np.nan
    # Weather is measured during the race and is unknown beforehand.
    for col in ("air_temp", "track_temp", "humidity", "wind_speed",
                "rainfall", "is_wet"):
        if col in hidden.columns:
            hidden[col] = np.nan
    return hidden


def last_race(raw: pd.DataFrame) -> tuple[int, int]:
    final = raw.sort_values("date").iloc[-1]
    return int(final["year"]), int(final["round"])


def test_replay_reproduces_the_training_feature_vector(raw, policy):
    """Hiding a race's outcome must not change the features computed for it."""
    year, rnd = last_race(raw)
    target = (raw["year"] == year) & (raw["round"] == rnd)

    from_training = build_asof_features(raw, policy=policy)
    from_training = (from_training[(from_training.year == year)
                                   & (from_training["round"] == rnd)]
                     .set_index("driver")[FEATURE_COLS].sort_index())

    history = raw[~target]
    upcoming = hide_outcomes(raw[target])
    from_serving = build_asof_features(history, upcoming, policy=policy)
    from_serving = (from_serving[(from_serving.year == year)
                                 & (from_serving["round"] == rnd)]
                    .set_index("driver")[FEATURE_COLS].sort_index())

    assert len(from_training) == len(from_serving) > 0
    pd.testing.assert_frame_equal(from_training, from_serving,
                                  check_like=True, atol=1e-12)


def test_replay_holds_for_an_earlier_race_too(raw, policy):
    """A mid-dataset race also has FUTURE races in `history`, so this additionally
    checks that later results cannot reach back into an earlier forecast."""
    races = raw[["year", "round", "date"]].drop_duplicates().sort_values("date")
    year, rnd = races.iloc[len(races) // 2][["year", "round"]].astype(int)
    target = (raw["year"] == year) & (raw["round"] == rnd)

    from_training = build_asof_features(raw, policy=policy)
    from_training = (from_training[(from_training.year == year)
                                   & (from_training["round"] == rnd)]
                     .set_index("driver")[FEATURE_COLS].sort_index())

    combined = pd.concat([raw[~target], hide_outcomes(raw[target])],
                         ignore_index=True)
    from_serving = build_asof_features(combined, policy=policy)
    from_serving = (from_serving[(from_serving.year == year)
                                 & (from_serving["round"] == rnd)]
                    .set_index("driver")[FEATURE_COLS].sort_index())

    pd.testing.assert_frame_equal(from_training, from_serving,
                                  check_like=True, atol=1e-12)


def test_forecasting_an_already_ingested_race_is_refused(raw):
    year, rnd = last_race(raw)
    with pytest.raises(ValueError, match="already exists"):
        build_asof_features(raw, raw[(raw["year"] == year) & (raw["round"] == rnd)])


# --- fitted state -----------------------------------------------------------

def test_policy_is_fitted_on_training_rows_only(raw):
    built = build_asof_features(raw)
    train, _, _, _ = chronological_split(built)
    fitted = FillPolicy.fit(train)

    assert fitted.constants["global_mean_finish"] == pytest.approx(
        train["position"].mean())
    assert fitted.constants["global_mean_finish"] != pytest.approx(
        built["position"].mean())
    assert fitted.fitted_on["n_rows"] == len(train)
    assert fitted.fitted_on["year_max"] == int(built["year"].max()) - 2


def test_held_out_outcomes_cannot_move_the_fitted_constants(raw):
    """Changing test-season results must not change what is imputed into
    training rows."""
    built = build_asof_features(raw)
    train, _, _, latest = chronological_split(built)
    baseline = FillPolicy.fit(train)

    tampered = built.copy()
    tampered.loc[tampered["year"] == latest, "position"] = 1.0
    train_after, _, _, _ = chronological_split(tampered)

    assert FillPolicy.fit(train_after).constants == baseline.constants


def test_transform_is_idempotent(raw, policy):
    once = policy.transform(build_asof_features(raw))
    twice = policy.transform(once)
    pd.testing.assert_frame_equal(once[FEATURE_COLS], twice[FEATURE_COLS])


def test_transform_uses_the_driver_prior_before_the_global_constant(policy):
    """The fallback the old serving path skipped, on 2091 cells."""
    frame = pd.DataFrame({
        "driver_circuit_avg_finish": [np.nan, np.nan, 4.0],
        "driver_overall_avg_finish": [7.5, np.nan, 9.0],
    })
    out = policy.transform(frame)
    assert out.loc[0, "driver_circuit_avg_finish"] == 7.5            # driver prior
    assert out.loc[1, "driver_circuit_avg_finish"] == pytest.approx(
        policy.constants["global_mean_finish"])                       # global
    assert out.loc[2, "driver_circuit_avg_finish"] == 4.0             # untouched


def test_policy_round_trips_through_json(tmp_path, policy):
    path = tmp_path / "fill_values.json"
    policy.to_json(path)
    assert FillPolicy.from_json(path) == policy


def test_a_policy_from_an_unknown_schema_is_refused(tmp_path, policy):
    """A model must be served with the policy it was trained with."""
    import json
    path = tmp_path / "fill_values.json"
    policy.to_json(path)
    data = json.loads(path.read_text())
    data["schema_version"] = 999
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="schema"):
        FillPolicy.from_json(path)


def test_fitting_on_an_empty_frame_is_refused():
    with pytest.raises(ValueError, match="empty"):
        FillPolicy.fit(pd.DataFrame(columns=["position", "is_dnf", "year", "round"]))
