"""Stage 4: models/severity.py -- LightGBM quantile + split conformal, and the
global-mean baseline it has to beat."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lsm.models.severity import GlobalMeanSeverityBaseline, SeverityModel, conformal_margin

LGBM_CFG = {"deterministic": True, "force_row_wise": True, "num_threads": 1}


def test_conformal_margin_uses_the_finite_sample_corrected_level_not_naive_quantile():
    """The bug this pins: np.quantile(scores, 1 - alpha) directly would give a
    LOWER (under-covering) margin than the correction requires, for n this
    small. n=9, alpha=0.10 -> level = ceil(10*0.9)/9 = 9/9 = 1.0 -> the margin
    must be the MAXIMUM score, not the naive 90th-percentile (which, with
    linear interpolation on 9 sorted points, sits at the 8th value, not the
    9th/max).
    """
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 100.0])  # sorted, max is an outlier
    margin = conformal_margin(scores, alpha=0.10)
    assert margin == 100.0  # must be the max, not np.quantile's naive 0.9-quantile (~44.8)
    naive = np.quantile(scores, 0.9)
    assert margin > naive


def test_conformal_margin_clips_the_level_at_one_for_very_small_n():
    # n=1, alpha=0.10 -> ceil(2*0.9)/1 = 2 -> clipped to 1.0 -> the one score itself.
    assert conformal_margin(np.array([42.0]), alpha=0.10) == 42.0


def test_conformal_margin_on_empty_scores_is_zero():
    assert conformal_margin(np.array([]), alpha=0.10) == 0.0


def test_conformal_margin_matches_plain_quantile_for_large_n():
    """The correction should become negligible as n grows -- large-n behaviour
    must still look like ordinary split conformal, not a permanently inflated
    margin.
    """
    rng = np.random.default_rng(0)
    scores = rng.normal(0, 1, size=5000)
    margin = conformal_margin(scores, alpha=0.10)
    naive = np.quantile(scores, 0.9)
    assert abs(margin - naive) < 0.05


def test_global_mean_baseline_predicts_the_training_mean():
    baseline = GlobalMeanSeverityBaseline(conformal_alpha=0.10).fit(
        y_train=np.array([10.0, 20.0, 30.0]), y_calib=np.array([15.0, 25.0])
    )
    med, lo, hi = baseline.predict(n=4)
    assert (med == 20.0).all()
    assert (lo < med).all() and (hi > med).all()


def test_global_mean_baseline_with_empty_calibration_has_zero_margin():
    baseline = GlobalMeanSeverityBaseline(conformal_alpha=0.10).fit(
        y_train=np.array([10.0, 20.0]), y_calib=np.array([])
    )
    med, lo, hi = baseline.predict(n=1)
    assert lo[0] == med[0] == hi[0]


def _synthetic_severity_data(n=40, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(20, 80, size=n)
    y = x + rng.normal(0, 2, size=n)  # severity ~ feature, small noise
    return pd.DataFrame({"amplitude": x}), y


def test_severity_model_interval_is_always_ordered():
    X, y = _synthetic_severity_data()
    X_train, y_train = X.iloc[:30], y[:30]
    X_calib, y_calib = X.iloc[30:], y[30:]

    model = SeverityModel(
        feature_cols=["amplitude"], quantiles=(0.05, 0.5, 0.95),
        conformal_alpha=0.10, seed=0, lgbm_cfg=LGBM_CFG,
    ).fit(X_train, y_train, X_calib, y_calib)

    med, lo, hi = model.predict(X)
    assert (lo <= med).all()
    assert (med <= hi).all()


def test_severity_model_tracks_the_correlated_feature():
    X, y = _synthetic_severity_data()
    X_train, y_train = X.iloc[:30], y[:30]
    X_calib, y_calib = X.iloc[30:], y[30:]

    model = SeverityModel(
        feature_cols=["amplitude"], quantiles=(0.05, 0.5, 0.95),
        conformal_alpha=0.10, seed=0, lgbm_cfg=LGBM_CFG,
    ).fit(X_train, y_train, X_calib, y_calib)

    med, _, _ = model.predict(X)
    # Not a strict correlation test (tiny-n LightGBM is noisy), just: the model
    # must separate a genuinely low-severity point from a genuinely high one.
    low_idx = X["amplitude"].idxmin()
    high_idx = X["amplitude"].idxmax()
    assert med[high_idx] > med[low_idx]


def test_severity_model_predict_before_fit_raises():
    model = SeverityModel(
        feature_cols=["amplitude"], quantiles=(0.05, 0.5, 0.95),
        conformal_alpha=0.10, seed=0, lgbm_cfg=LGBM_CFG,
    )
    with pytest.raises(RuntimeError):
        model.predict(pd.DataFrame({"amplitude": [1.0]}))


def test_severity_model_fills_nan_features():
    X, y = _synthetic_severity_data()
    X.loc[0, "amplitude"] = np.nan
    X_train, y_train = X.iloc[:30], y[:30]
    X_calib, y_calib = X.iloc[30:], y[30:]

    model = SeverityModel(
        feature_cols=["amplitude"], quantiles=(0.05, 0.5, 0.95),
        conformal_alpha=0.10, seed=0, lgbm_cfg=LGBM_CFG,
    ).fit(X_train, y_train, X_calib, y_calib)
    med, lo, hi = model.predict(X)
    assert np.isfinite(med).all() and np.isfinite(lo).all() and np.isfinite(hi).all()
