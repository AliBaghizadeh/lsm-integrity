"""
Stage 4 required CI check: save(load()) reproduces predictions exactly, and
load() refuses a feature_version / schema_version / library-version mismatch.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from lsm.bundle import (
    BundleVersionMismatchError,
    library_versions,
    load_bundle,
    save_bundle,
    training_feature_summary,
)
from lsm.models.anomaly import MADBaseline


def _fitted_bundle() -> dict:
    X = pd.DataFrame({"r_mag_nt": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]})
    model = MADBaseline().fit(X)
    return {
        "task": "anomaly",
        "model": model,
        "feature_cols": ["r_mag_nt"],
        "threshold": 3.0,
        "feature_version": 1,
        "schema_version": 2,
        "config_sha256": "abc123",
        "git_sha": "deadbeef",
        "data_sha256": "cafef00d",
        "truth_as_of": "2026-07-29T00:00:00+00:00",
    }


def test_bundle_roundtrip_reproduces_predictions_exactly(tmp_path):
    bundle = _fitted_bundle()
    path = tmp_path / "bundle.joblib"
    save_bundle(path, bundle)

    reloaded = load_bundle(path, expected_feature_version=1, expected_schema_version=2)

    X_test = pd.DataFrame({"r_mag_nt": [2.5, 50.0]})
    original_scores = bundle["model"].score(X_test)
    reloaded_scores = reloaded["model"].score(X_test)
    np.testing.assert_array_equal(original_scores, reloaded_scores)


def test_bundle_stamps_library_versions_at_save_time(tmp_path):
    bundle = _fitted_bundle()
    path = tmp_path / "bundle.joblib"
    save_bundle(path, bundle)

    reloaded = load_bundle(path, expected_feature_version=1, expected_schema_version=2)
    assert reloaded["library_versions"] == library_versions()


def test_load_bundle_refuses_a_feature_version_mismatch(tmp_path):
    bundle = _fitted_bundle()
    path = tmp_path / "bundle.joblib"
    save_bundle(path, bundle)

    with pytest.raises(BundleVersionMismatchError):
        load_bundle(path, expected_feature_version=999, expected_schema_version=2)


def test_load_bundle_refuses_a_schema_version_mismatch(tmp_path):
    bundle = _fitted_bundle()
    path = tmp_path / "bundle.joblib"
    save_bundle(path, bundle)

    with pytest.raises(BundleVersionMismatchError):
        load_bundle(path, expected_feature_version=1, expected_schema_version=999)


def test_load_bundle_refuses_a_library_version_mismatch(tmp_path):
    """Simulates a bundle trained under a different numpy/sklearn/lightgbm --
    write the payload directly (bypassing save_bundle's live stamping) with a
    tampered library_versions to prove load_bundle enforces it, not just
    records it.
    """
    bundle = _fitted_bundle()
    bundle["library_versions"] = {**library_versions(), "numpy": "0.0.1-not-installed"}
    path = tmp_path / "bundle.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)

    with pytest.raises(BundleVersionMismatchError):
        load_bundle(path, expected_feature_version=1, expected_schema_version=2)


def test_load_bundle_raises_file_not_found_with_a_helpful_message(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_bundle(tmp_path / "does_not_exist.joblib", expected_feature_version=1, expected_schema_version=2)


def test_training_feature_summary_reports_mean_std_min_max():
    corpus = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    summary = training_feature_summary(corpus, ["a"])
    assert summary["a"]["mean"] == 3.0
    assert summary["a"]["min"] == 1.0
    assert summary["a"]["max"] == 5.0


def test_training_feature_summary_handles_all_nan_column():
    corpus = pd.DataFrame({"a": [np.nan, np.nan]})
    summary = training_feature_summary(corpus, ["a"])
    assert np.isnan(summary["a"]["mean"])
