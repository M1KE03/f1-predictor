"""Guards on the feature contract (FIX_PLAN.md sections 2 P0-1 and 10).

These exist because the leak they prevent was invisible to the Gate 2 audit:
`audit_leakage.py` checks that a feature aggregates only prior races, which
race-session weather trivially satisfies at the row level. Availability at the
forecast cutoff is a separate property and needs its own guard.
"""
from __future__ import annotations

from src import columns


def test_race_session_weather_is_not_a_model_input():
    """The six readings come from ingest.weather_row(), which averages the
    weather stream over the race itself."""
    for name in ["air_temp", "track_temp", "humidity", "wind_speed",
                 "rainfall", "is_wet"]:
        assert name not in columns.FEATURE_COLS, name


def test_target_dependent_temperature_affinity_is_not_a_model_input():
    """driver_temp_bin_* read historical values, but selected the bucket using
    the target race's realized track temperature."""
    assert "driver_temp_bin_avg" not in columns.FEATURE_COLS
    assert "driver_temp_bin_n" not in columns.FEATURE_COLS


def test_prior_race_wet_affinity_is_retained():
    """These are clean: a driver's count and average over PRIOR wet races. The
    value at a target row does not change with that race's own conditions, so
    removing them would discard usable pre-cutoff signal."""
    assert "driver_wet_delta" in columns.FEATURE_COLS
    assert "driver_wet_n" in columns.FEATURE_COLS


def test_removed_weather_never_overlaps_feature_cols():
    assert not set(columns.FEATURE_COLS) & set(columns.WEATHER_REMOVED)


def test_weather_is_retained_for_auditing_but_only_as_an_identifier():
    """Keeping the raw readings in features.parquet is what makes this change
    auditable; ID_COLS membership is what keeps them out of the model."""
    for name in ["air_temp", "track_temp", "humidity", "wind_speed",
                 "rainfall", "is_wet"]:
        assert name in columns.ID_COLS, name


def test_feature_and_id_columns_are_disjoint():
    assert not set(columns.FEATURE_COLS) & set(columns.ID_COLS)


def test_no_duplicate_feature_names():
    assert len(columns.FEATURE_COLS) == len(set(columns.FEATURE_COLS))


def test_target_is_not_a_feature():
    assert columns.TARGET not in columns.FEATURE_COLS
