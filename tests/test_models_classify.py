"""Stage 5: models/classify.py -- LightGBM multiclass + isotonic calibration,
and the majority-class baseline it has to beat."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lsm.models.classify import ClassifyModel, MajorityClassBaseline

LGBM_CFG = {"deterministic": True, "force_row_wise": True, "num_threads": 1}
CLASSES = ["scc", "weld", "dent", "corrosion", "interference"]


def test_majority_class_baseline_predicts_the_most_frequent_class():
    baseline = MajorityClassBaseline(classes=CLASSES).fit(
        np.array(["scc", "scc", "scc", "weld"])
    )
    pred_type, pred_conf = baseline.predict(n=3)
    assert (pred_type == "scc").all()
    assert (pred_conf == 0.75).all()


def test_majority_class_baseline_proba_sums_to_one_and_matches_class_order():
    baseline = MajorityClassBaseline(classes=CLASSES).fit(np.array(["weld"] * 3 + ["dent"]))
    proba = baseline.predict_proba(n=2)
    assert list(proba.columns) == CLASSES
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba["weld"] == 0.75).all()


def test_majority_class_baseline_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        MajorityClassBaseline(classes=CLASSES).predict_proba(n=1)


def _synthetic_classify_data(n=60, seed=0):
    """`a` cleanly separates scc (high) from weld (low); `b` is noise."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1, 1, size=n)
    y = np.where(a > 0, "scc", "weld")
    X = pd.DataFrame({"a": a, "b": rng.normal(size=n)})
    return X, y


def test_classify_model_proba_columns_match_pinned_class_order_and_sum_to_one():
    X, y = _synthetic_classify_data()
    X_train, y_train = X.iloc[:40], y[:40]
    X_calib, y_calib = X.iloc[40:], y[40:]

    model = ClassifyModel(
        feature_cols=["a", "b"], classes=CLASSES, seed=0, lgbm_cfg=LGBM_CFG,
    ).fit(X_train, y_train, X_calib, y_calib)

    proba = model.predict_proba(X)
    assert list(proba.columns) == CLASSES
    assert np.allclose(proba.sum(axis=1), 1.0)
    # untrained classes (dent/corrosion/interference) must be zero-filled, not missing
    assert (proba["dent"] == 0.0).all()


def test_classify_model_tracks_the_correlated_feature():
    X, y = _synthetic_classify_data()
    X_train, y_train = X.iloc[:40], y[:40]
    X_calib, y_calib = X.iloc[40:], y[40:]

    model = ClassifyModel(
        feature_cols=["a", "b"], classes=CLASSES, seed=0, lgbm_cfg=LGBM_CFG,
    ).fit(X_train, y_train, X_calib, y_calib)

    pred_type, _ = model.predict(X)
    high_a_idx = X["a"].idxmax()
    low_a_idx = X["a"].idxmin()
    assert pred_type[high_a_idx] == "scc"
    assert pred_type[low_a_idx] == "weld"


def test_classify_model_predict_before_fit_raises():
    model = ClassifyModel(feature_cols=["a"], classes=CLASSES, seed=0, lgbm_cfg=LGBM_CFG)
    with pytest.raises(RuntimeError):
        model.predict(pd.DataFrame({"a": [1.0]}))


def test_classify_model_fills_nan_features():
    X, y = _synthetic_classify_data()
    X.loc[0, "a"] = np.nan
    X_train, y_train = X.iloc[:40], y[:40]
    X_calib, y_calib = X.iloc[40:], y[40:]

    model = ClassifyModel(
        feature_cols=["a", "b"], classes=CLASSES, seed=0, lgbm_cfg=LGBM_CFG,
    ).fit(X_train, y_train, X_calib, y_calib)
    pred_type, pred_conf = model.predict(X)
    assert len(pred_type) == len(X)
    assert np.isfinite(pred_conf).all()


def test_classify_model_calibrates_when_every_class_is_in_the_calib_split():
    X, y = _synthetic_classify_data()
    X_train, y_train = X.iloc[:40], y[:40]
    X_calib, y_calib = X.iloc[40:], y[40:]  # both classes present in both halves

    model = ClassifyModel(
        feature_cols=["a", "b"], classes=CLASSES, seed=0, lgbm_cfg=LGBM_CFG,
    ).fit(X_train, y_train, X_calib, y_calib)
    assert model.calibrated_ is True


def test_classify_model_falls_back_to_uncalibrated_softmax_when_calib_missing_a_class(capsys):
    """The real, reproducible sklearn failure mode this project's own small
    per-class counts hit: CalibratedClassifierCV.fit raises IndexError if a
    trained class has zero calibration examples. Confirmed empirically before
    writing the fallback -- this test pins that the fallback actually engages
    (not a crash) and says so out loud.
    """
    X, y = _synthetic_classify_data()
    X_train, y_train = X.iloc[:40], y[:40]
    # calib split deliberately missing "weld" entirely
    calib_mask = y[40:] == "scc"
    X_calib, y_calib = X.iloc[40:][calib_mask], y[40:][calib_mask]
    assert "weld" not in set(y_calib)

    model = ClassifyModel(
        feature_cols=["a", "b"], classes=CLASSES, seed=0, lgbm_cfg=LGBM_CFG,
    ).fit(X_train, y_train, X_calib, y_calib)

    assert model.calibrated_ is False
    proba = model.predict_proba(X)  # must not raise
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert "degenerate calibration split" in capsys.readouterr().out


def test_classify_model_falls_back_to_constant_prediction_with_one_class_in_train(capsys):
    """An even more degenerate fold: only one class present in y_train at all
    -- LightGBM's multiclass objective cannot fit here (confirmed empirically:
    raises a fatal LightGBMError). Falls back to a constant prediction rather
    than force a fake second class into existence.
    """
    X, y = _synthetic_classify_data()
    one_class_mask = y == "scc"
    X_train, y_train = X[one_class_mask], y[one_class_mask]

    model = ClassifyModel(
        feature_cols=["a", "b"], classes=CLASSES, seed=0, lgbm_cfg=LGBM_CFG,
    ).fit(X_train, y_train, X_train.iloc[0:0], y_train[0:0])

    pred_type, pred_conf = model.predict(X.iloc[:3])
    assert (pred_type == "scc").all()
    assert (pred_conf == 1.0).all()
    assert "falling back to a constant prediction" in capsys.readouterr().out
