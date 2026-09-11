"""Model-bundle verification (FIX_PLAN.md section 2 P1-11, section 11).

The test that matters is `assert_can_predict`: a model fitted on races at or
after the target has already seen the answer, and a forecast from it would
report excellent accuracy while being pure hindsight. Nothing else in the
pipeline detects that.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.bundle import BUNDLE_VERSION, BundleError, Manifest, ModelBundle
from src.preprocessing import SCHEMA_VERSION


def make_manifest(**overrides) -> Manifest:
    base = dict(
        bundle_version=BUNDLE_VERSION, created_utc="2026-09-11T00:00:00",
        code_revision="abc123", training_cutoff_utc="2026-09-06 00:00:00",
        n_train_rows=3744, n_train_races=186, train_year_min=2018,
        train_year_max=2026, feature_cols=["a", "b"],
        fill_policy_schema=SCHEMA_VERSION, temperature=0.344, alpha=0.3,
        best_iteration=65, data_sha256="deadbeef", dependencies={},
    )
    base.update(overrides)
    return Manifest(**base)


def bundle_with(manifest: Manifest) -> ModelBundle:
    return ModelBundle(ranker=None, policy=None, manifest=manifest, path=__import__("pathlib").Path("x"))


# --- the cutoff check -------------------------------------------------------

def test_a_race_after_the_cutoff_is_allowed():
    bundle_with(make_manifest()).assert_can_predict("2026-09-20")


def test_a_race_before_the_cutoff_is_refused():
    """The model was fitted on it; the 'forecast' would be hindsight."""
    with pytest.raises(BundleError, match="hindsight"):
        bundle_with(make_manifest()).assert_can_predict("2026-08-01")


def test_a_race_exactly_on_the_cutoff_is_refused():
    """The cutoff race IS in training, so it must not be predicted."""
    with pytest.raises(BundleError, match="hindsight"):
        bundle_with(make_manifest()).assert_can_predict("2026-09-06")


def test_the_error_names_both_dates():
    with pytest.raises(BundleError, match="2026-09-06"):
        bundle_with(make_manifest()).assert_can_predict("2026-01-01")


def test_a_timezone_aware_target_is_handled():
    bundle_with(make_manifest()).assert_can_predict(
        pd.Timestamp("2026-09-20", tz="UTC"))


# --- schema verification ----------------------------------------------------

def write_bundle(tmp_path, manifest: Manifest):
    import joblib
    path = tmp_path / "b"
    path.mkdir()
    joblib.dump({"dummy": True}, path / "rank_model.joblib")
    (path / "fill_values.json").write_text("{}")
    (path / "manifest.json").write_text(json.dumps(manifest.__dict__, default=str))
    return path


def test_a_feature_schema_mismatch_is_refused(tmp_path):
    """Serving a model on features it never saw. This actually happened twice
    in this project when FEATURE_COLS changed."""
    path = write_bundle(tmp_path, make_manifest(feature_cols=["a", "b", "c"]))
    with pytest.raises(BundleError, match="never saw"):
        ModelBundle.load(path, ["a", "b"])


def test_the_mismatch_error_names_the_differing_columns(tmp_path):
    path = write_bundle(tmp_path, make_manifest(feature_cols=["a", "gone"]))
    with pytest.raises(BundleError, match="gone"):
        ModelBundle.load(path, ["a", "added"])


def test_an_old_bundle_version_is_refused(tmp_path):
    path = write_bundle(tmp_path, make_manifest(bundle_version=BUNDLE_VERSION - 1))
    with pytest.raises(BundleError, match="bundle version"):
        ModelBundle.load(path, ["a", "b"])


def test_a_mismatched_policy_schema_is_refused(tmp_path):
    path = write_bundle(tmp_path, make_manifest(fill_policy_schema=999))
    with pytest.raises(BundleError, match="fill-policy schema"):
        ModelBundle.load(path, ["a", "b"])


def test_a_directory_without_a_manifest_is_refused(tmp_path):
    bare = tmp_path / "legacy"
    bare.mkdir()
    with pytest.raises(BundleError, match="manifest"):
        ModelBundle.load(bare, ["a", "b"])


# --- the real champion ------------------------------------------------------

def test_the_built_champion_loads_and_verifies():
    import pathlib
    from src.columns import FEATURE_COLS

    path = pathlib.Path("models/champion")
    if not (path / "manifest.json").exists():
        pytest.skip("champion bundle not built")
    bundle = ModelBundle.load(path, FEATURE_COLS)
    assert bundle.manifest.n_train_races > 0
    assert len(bundle.manifest.feature_cols) == len(FEATURE_COLS)
    # It must refuse its own most recent training race.
    with pytest.raises(BundleError):
        bundle.assert_can_predict(bundle.manifest.training_cutoff_utc)
