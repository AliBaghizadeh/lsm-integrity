"""
Stage 4: per-indication severity regression -- LightGBM quantile regression at
(0.05, 0.5, 0.95) wrapped in split conformal (CQR, Romano et al. 2019) for a
distribution-free 90% interval.

Regressed per INDICATION, never per row (SKILL invariant #2: `severity_smys` is
constant across a whole label box, so a row-level regressor would just learn to
reproduce that constant and report a flattering, meaningless MAE). Feature
vectors come from `indications.attach_indication_features` -- the peak row's
own residual/shape features plus `extent_m`.

With ~12 physical defects on today's single line, this training set is tiny --
LightGBM's own defaults (`min_child_samples=20`) would refuse to split at all
and degenerate into predicting a single constant per quantile, which is exactly
what the GLOBAL-MEAN BASELINE already does. `min_child_samples` is set low here
specifically to let *some* feature-driven learning happen on a training set this
small; this is a small-sample accommodation, not a production default (Stage 6's
scale rehearsal would need to revisit it).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


def conformal_margin(scores: np.ndarray, alpha: float) -> float:
    """The finite-sample-corrected (1 - alpha) quantile of nonconformity scores
    (Romano et al. 2019's split conformal / CQR), NOT a plain
    `np.quantile(scores, 1 - alpha)`.

    The naive quantile systematically UNDERSHOOTS the margin the coverage
    guarantee actually requires once the calibration set is small -- exactly
    the regime this project's ~12-defect corpus lives in (calibration sets of
    a handful of rows per fold). Measured causing real undercoverage (~73%
    actual vs 90% nominal) before this correction was added; this is not a
    theoretical nicety, it changed the reported number.
    """
    n = len(scores)
    if n == 0:
        return 0.0
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(scores, level))


class GlobalMeanSeverityBaseline:
    """Baseline: predict the training set's mean severity for everything,
    with a constant interval half-width from the training residual spread.
    What the LightGBM quantile model has to beat (SKILL: "add its baseline
    first").
    """

    def __init__(self, conformal_alpha: float):
        self.conformal_alpha = conformal_alpha
        self.mean_: float | None = None
        self.margin_: float | None = None

    def fit(self, y_train: np.ndarray, y_calib: np.ndarray) -> "GlobalMeanSeverityBaseline":
        self.mean_ = float(np.mean(y_train))
        residuals = np.abs(y_calib - self.mean_)
        self.margin_ = conformal_margin(residuals, self.conformal_alpha)
        return self

    def predict(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        med = np.full(n, self.mean_)
        return med, med - self.margin_, med + self.margin_


class SeverityModel:
    """LightGBM quantile regression (0.05/0.5/0.95) + split conformal (CQR).

    `fit` takes an explicit train/calibration split -- conformal's finite-sample
    coverage guarantee depends on the calibration set being held out of
    training, not reused from it (the same reasoning as any other fitted
    transform: fitting and calibrating on the same rows leaks).
    """

    def __init__(
        self,
        feature_cols: list[str],
        quantiles: tuple[float, float, float],
        conformal_alpha: float,
        seed: int,
        lgbm_cfg: dict,
        min_child_samples: int = 3,
    ):
        self.feature_cols = feature_cols
        self.q_lo, self.q_med, self.q_hi = quantiles
        self.conformal_alpha = conformal_alpha
        self.seed = seed
        self.lgbm_cfg = lgbm_cfg
        self.min_child_samples = min_child_samples
        self.models: dict[float, LGBMRegressor] = {}
        self.conformal_margin_: float | None = None

    def _make_model(self, alpha: float) -> LGBMRegressor:
        return LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=50,
            num_leaves=7,
            min_child_samples=self.min_child_samples,
            deterministic=self.lgbm_cfg.get("deterministic", True),
            force_row_wise=self.lgbm_cfg.get("force_row_wise", True),
            num_threads=self.lgbm_cfg.get("num_threads", 1),
            random_state=self.seed,
            verbosity=-1,
        )

    def fit(
        self, X_train: pd.DataFrame, y_train: np.ndarray, X_calib: pd.DataFrame, y_calib: np.ndarray
    ) -> "SeverityModel":
        X_train_mat = X_train[self.feature_cols].fillna(0.0)
        for q in (self.q_lo, self.q_med, self.q_hi):
            model = self._make_model(q)
            model.fit(X_train_mat, y_train)
            self.models[q] = model

        if len(X_calib) > 0:
            lo_pred = self.models[self.q_lo].predict(X_calib[self.feature_cols].fillna(0.0))
            hi_pred = self.models[self.q_hi].predict(X_calib[self.feature_cols].fillna(0.0))
            # CQR nonconformity score: how far y falls outside the raw [lo, hi]
            # interval, signed so a point INSIDE the interval scores negative.
            scores = np.maximum(lo_pred - y_calib, y_calib - hi_pred)
            self.conformal_margin_ = conformal_margin(scores, self.conformal_alpha)
        else:
            self.conformal_margin_ = 0.0
        return self

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.conformal_margin_ is None:
            raise RuntimeError("SeverityModel.predict() called before fit()")
        X_mat = X[self.feature_cols].fillna(0.0)
        med = self.models[self.q_med].predict(X_mat)
        lo = self.models[self.q_lo].predict(X_mat) - self.conformal_margin_
        hi = self.models[self.q_hi].predict(X_mat) + self.conformal_margin_
        return med, lo, hi
