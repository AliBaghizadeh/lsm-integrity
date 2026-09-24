"""Stage 3: models/anomaly.py -- MAD baseline and IsolationForest scorers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lsm.models.anomaly import (
    IsolationForestAnomalyModel,
    MADBaseline,
    calibrated_threshold,
)


def test_mad_baseline_scores_zero_at_the_median():
    X = pd.DataFrame({"r_mag_nt": [1.0, 2.0, 3.0, 4.0, 5.0]})
    model = MADBaseline().fit(X)
    scores = model.score(pd.DataFrame({"r_mag_nt": [3.0]}))
    assert scores[0] == 0.0


def test_mad_baseline_flags_an_outlier_far_above_the_pack():
    X = pd.DataFrame({"r_mag_nt": np.concatenate([np.full(99, 8.0), [8.0]])})
    model = MADBaseline().fit(X)
    normal_score = model.score(pd.DataFrame({"r_mag_nt": [8.0]}))[0]
    outlier_score = model.score(pd.DataFrame({"r_mag_nt": [80.0]}))[0]
    assert outlier_score > normal_score
    assert normal_score == 0.0


def test_mad_baseline_score_before_fit_raises():
    model = MADBaseline()
    try:
        model.score(pd.DataFrame({"r_mag_nt": [1.0]}))
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_isolation_forest_flags_an_injected_outlier():
    rng = np.random.default_rng(0)
    n = 300
    normal = rng.normal(0, 1, size=(n, 3))
    X = pd.DataFrame(normal, columns=["a", "b", "c"])
    model = IsolationForestAnomalyModel(
        feature_cols=["a", "b", "c"], contamination=0.05, n_estimators=100, seed=0
    ).fit(X)

    outlier = pd.DataFrame({"a": [20.0], "b": [20.0], "c": [20.0]})
    inlier = pd.DataFrame({"a": [0.0], "b": [0.0], "c": [0.0]})
    assert model.score(outlier)[0] > model.score(inlier)[0]


def test_isolation_forest_fills_nan_shape_features_with_zero():
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0, np.nan, 4.0] * 20})
    model = IsolationForestAnomalyModel(
        feature_cols=["a"], contamination=0.1, n_estimators=50, seed=0
    ).fit(X)
    # Must not raise on NaN, at fit or score time.
    scores = model.score(X)
    assert np.isfinite(scores).all()


def test_calibrated_threshold_is_the_upper_quantile():
    train_scores = np.arange(100, dtype=float)  # 0..99
    thresh = calibrated_threshold(train_scores, contamination=0.1)
    assert thresh == np.quantile(train_scores, 0.9)


def test_emphasis_repeats_widens_the_fitted_matrix():
    X = pd.DataFrame({"a": np.arange(50.0), "b": np.arange(50.0) * 2})
    model = IsolationForestAnomalyModel(
        feature_cols=["a", "b"],
        contamination=0.1,
        n_estimators=10,
        seed=0,
        emphasize_features=["a"],
        emphasis_repeats=4,
    )
    matrix = model._matrix(X)
    assert matrix.shape == (50, 2 + 3)  # base (a, b) + 3 extra copies of a


def test_emphasis_repeats_of_one_is_a_no_op():
    X = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0) * 2})
    plain = IsolationForestAnomalyModel(
        feature_cols=["a", "b"],
        contamination=0.1,
        n_estimators=10,
        seed=0,
    )
    emphasised = IsolationForestAnomalyModel(
        feature_cols=["a", "b"],
        contamination=0.1,
        n_estimators=10,
        seed=0,
        emphasize_features=["a"],
        emphasis_repeats=1,
    )
    np.testing.assert_array_equal(plain._matrix(X), emphasised._matrix(X))


def test_emphasize_features_must_be_a_subset_of_feature_cols():
    try:
        IsolationForestAnomalyModel(
            feature_cols=["a", "b"],
            contamination=0.1,
            n_estimators=10,
            seed=0,
            emphasize_features=["c"],
            emphasis_repeats=3,
        )
        assert False, "expected ValueError"
    except ValueError:
        pass
