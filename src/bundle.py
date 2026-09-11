"""Immutable model bundles (FIX_PLAN.md section 2 P1-11, section 11).

A saved model on its own cannot answer the questions that matter before you
trust a forecast: which data was it fitted on, with which feature schema, which
imputation state, which calibration, and — critically — **could it have seen
the race it is being asked to predict?**

The inherited `models/` directory held a `.joblib`, a feature list and a flat
dict of fill values with nothing binding them together. Nothing stopped a model
being served with a different feature schema than it was fitted on (that
actually happened twice: `src.baseline` broke when `FEATURE_COLS` shrank, and
`predict.py` would have scored a 30-feature model on 22 features). Nothing
recorded a training cutoff, so nothing could detect the worst failure of all —
a model trained on races *after* the one it is forecasting, which would report
excellent accuracy and be pure hindsight.

A bundle is a directory holding the artifacts plus a `manifest.json` that binds
them. `load()` verifies before returning, and `assert_can_predict()` refuses a
target race at or before the training cutoff.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

BUNDLE_VERSION = 1

MANIFEST_NAME = "manifest.json"
RANKER_NAME = "rank_model.joblib"
POLICY_NAME = "fill_values.json"

PINNED_PACKAGES = ("fastf1", "lightgbm", "pandas", "numpy", "scikit-learn")


def _git_revision(root: Path) -> str | None:
    try:
        done = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                              capture_output=True, text=True, check=True)
        return done.stdout.strip()
    except Exception:
        return None


def file_sha256(path: Path) -> str | None:
    if not Path(path).exists():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BundleError(RuntimeError):
    """The bundle cannot be trusted for this prediction. Never downgrade."""


@dataclass
class Manifest:
    """Everything needed to decide whether a forecast from this bundle counts."""

    bundle_version: int
    created_utc: str
    code_revision: str | None
    # The single most important field: the latest race the model was fitted on.
    # A target race at or before this date was potentially seen in training.
    training_cutoff_utc: str
    n_train_rows: int
    n_train_races: int
    train_year_min: int
    train_year_max: int
    feature_cols: list[str]
    fill_policy_schema: int
    temperature: float
    alpha: float
    best_iteration: int
    data_sha256: str | None
    dependencies: dict[str, str | None]
    validation: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class ModelBundle:
    """A loaded bundle: the ranker, its fitted policy, and the manifest."""

    def __init__(self, ranker, policy, manifest: Manifest, path: Path):
        self.ranker = ranker
        self.policy = policy
        self.manifest = manifest
        self.path = Path(path)

    # -- persistence --------------------------------------------------------
    @staticmethod
    def save(path: Path, ranker, policy, manifest: Manifest) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(ranker, path / RANKER_NAME)
        policy.to_json(path / POLICY_NAME)
        with (path / MANIFEST_NAME).open("w", encoding="utf-8") as fh:
            json.dump(asdict(manifest), fh, indent=2)
        return path

    @classmethod
    def load(cls, path: Path, feature_cols: list[str] | None = None) -> "ModelBundle":
        """Load and VERIFY. Raises BundleError rather than returning something
        that cannot be trusted."""
        from .preprocessing import SCHEMA_VERSION, FillPolicy

        path = Path(path)
        manifest_path = path / MANIFEST_NAME
        if not manifest_path.exists():
            raise BundleError(
                f"{path} has no {MANIFEST_NAME}. Pre-bundle model directories "
                f"cannot be served: nothing binds their model, feature schema, "
                f"imputation state and training cutoff together. Rebuild with "
                f"`python -m src.build_bundle`.")

        with manifest_path.open(encoding="utf-8") as fh:
            manifest = Manifest(**json.load(fh))

        if manifest.bundle_version != BUNDLE_VERSION:
            raise BundleError(
                f"{path} is bundle version {manifest.bundle_version}; this code "
                f"expects {BUNDLE_VERSION}.")

        if feature_cols is not None and list(manifest.feature_cols) != list(feature_cols):
            only_bundle = sorted(set(manifest.feature_cols) - set(feature_cols))
            only_code = sorted(set(feature_cols) - set(manifest.feature_cols))
            raise BundleError(
                f"{path} was fitted on {len(manifest.feature_cols)} features; the "
                f"current code defines {len(feature_cols)}. Serving it would score "
                f"the model on inputs it never saw.\n"
                f"  only in bundle: {only_bundle}\n"
                f"  only in code  : {only_code}")

        if manifest.fill_policy_schema != SCHEMA_VERSION:
            raise BundleError(
                f"{path} carries fill-policy schema {manifest.fill_policy_schema}; "
                f"this code expects {SCHEMA_VERSION}.")

        return cls(ranker=joblib.load(path / RANKER_NAME),
                   policy=FillPolicy.from_json(path / POLICY_NAME),
                   manifest=manifest, path=path)

    # -- the check that matters ---------------------------------------------
    def assert_can_predict(self, target_date) -> None:
        """Refuse a race the model could have been trained on.

        A model fitted on races at or after the target has already seen the
        answer. It would report excellent accuracy and be pure hindsight, and
        nothing else in the pipeline would notice.
        """
        target = pd.Timestamp(target_date)
        if target.tzinfo is not None:
            target = target.tz_localize(None)
        cutoff = pd.Timestamp(self.manifest.training_cutoff_utc)
        if target <= cutoff:
            raise BundleError(
                f"Bundle {self.path.name} was trained on races up to "
                f"{cutoff.date()}, but the target race is {target.date()}. The "
                f"model may have been fitted on this race's result, so any "
                f"forecast from it is hindsight, not prediction. Retrain with a "
                f"cutoff before the target, or predict a later race.")

    def describe(self) -> str:
        m = self.manifest
        return (f"bundle {self.path.name} | trained on {m.n_train_races} races "
                f"({m.train_year_min}-{m.train_year_max}) up to "
                f"{pd.Timestamp(m.training_cutoff_utc).date()} | "
                f"{len(m.feature_cols)} features | T={m.temperature:.3f}")


def build_manifest(train: pd.DataFrame, feature_cols: list[str],
                   temperature: float, alpha: float, best_iteration: int,
                   raw_path: Path, project_root: Path,
                   validation: dict[str, Any] | None = None,
                   notes: list[str] | None = None) -> Manifest:
    from .preprocessing import SCHEMA_VERSION

    versions: dict[str, str | None] = {}
    for name in PINNED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None

    return Manifest(
        bundle_version=BUNDLE_VERSION,
        created_utc=pd.Timestamp.utcnow().isoformat(),
        code_revision=_git_revision(project_root),
        training_cutoff_utc=str(pd.Timestamp(train["date"].max())),
        n_train_rows=int(len(train)),
        n_train_races=int(train.groupby(["year", "round"]).ngroups),
        train_year_min=int(train["year"].min()),
        train_year_max=int(train["year"].max()),
        feature_cols=list(feature_cols),
        fill_policy_schema=SCHEMA_VERSION,
        temperature=float(temperature),
        alpha=float(alpha),
        best_iteration=int(best_iteration),
        data_sha256=file_sha256(raw_path),
        dependencies=versions,
        validation=validation or {},
        notes=notes or [],
    )
