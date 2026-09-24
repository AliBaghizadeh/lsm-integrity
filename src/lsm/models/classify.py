"""
Stage 5: per-indication classification -- LightGBM multiclass over
{scc, weld, dent, corrosion, interference}, isotonic-calibrated.

Classified per INDICATION, matched to a physical source (defect or
interference) -- the same training universe discipline as severity, just
widened to include interference as an explicit class rather than excluding
it (see train.py's `_build_classify_training_frame`).

With ~60 physical defects spread over 5 classes (as few as a dozen SCC
instances), this training set is tiny -- the same `min_child_samples`
small-sample accommodation severity.py already documents applies here too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator


class MajorityClassBaseline:
    """Baseline: always predict the training set's single most frequent
    class, confidence = that class's training-set frequency, remainder spread
    uniformly over the other classes -- the classification analogue of
    `GlobalMeanSeverityBaseline`'s radical simplicity. What the LightGBM
    classifier has to beat (SKILL: "add its baseline first").
    """

    def __init__(self, classes: list[str]):
        self.classes = list(classes)
        self.majority_: str | None = None
        self.majority_frac_: float | None = None

    def fit(self, y_train: np.ndarray) -> MajorityClassBaseline:
        counts = pd.Series(y_train).value_counts()
        self.majority_ = str(counts.idxmax())
        self.majority_frac_ = float(counts.max() / len(y_train))
        return self

    def predict_proba(self, n: int) -> pd.DataFrame:
        if self.majority_ is None or self.majority_frac_ is None:
            raise RuntimeError(
                "MajorityClassBaseline.predict_proba() called before fit()"
            )
        majority_frac = self.majority_frac_
        remainder = (1.0 - majority_frac) / max(len(self.classes) - 1, 1)
        row = {
            c: (majority_frac if c == self.majority_ else remainder)
            for c in self.classes
        }
        return pd.DataFrame([row] * n, columns=self.classes)

    def predict(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        pred_type = np.full(n, self.majority_)
        pred_conf = np.full(n, self.majority_frac_)
        return pred_type, pred_conf


class ClassifyModel:
    """LightGBM multiclass + isotonic calibration.

    Calibration is "prefit" in spirit but not in sklearn's literal API:
    sklearn removed `CalibratedClassifierCV(cv="prefit")` (confirmed against
    the installed version, not assumed from older docs) in favour of wrapping
    an already-fitted estimator in `sklearn.frozen.FrozenEstimator`, which
    makes `ensemble="auto"` resolve to `False` -- all data passed to
    `CalibratedClassifierCV.fit()` calibrates the frozen model directly, no
    internal re-fitting or internal CV.

    `fit` takes an explicit train/calibration split (same reasoning as
    `SeverityModel`: the calibration guarantee depends on the calibration set
    being held out of training). Isotonic calibration needs every trained
    class represented in the calibration split, or sklearn raises
    (`IndexError`, confirmed empirically, not assumed) -- a real failure mode
    at ~15 SCC instances total in the real corpus, not a theoretical one.
    When the calibration split is empty or missing a trained class, `fit`
    falls back to the RAW (uncalibrated) softmax for that fit, printed
    loudly, not silently swallowed -- the same honesty discipline as
    severity's own small-sample caveats.
    """

    def __init__(
        self,
        feature_cols: list[str],
        classes: list[str],
        seed: int,
        lgbm_cfg: dict,
        class_weight: str | None = "balanced",
        min_child_samples: int = 3,
    ):
        self.feature_cols = feature_cols
        self.classes = list(classes)
        self.seed = seed
        self.lgbm_cfg = lgbm_cfg
        self.class_weight = class_weight
        self.min_child_samples = min_child_samples
        self.model: LGBMClassifier | None = None
        self.calibrated_model: CalibratedClassifierCV | LGBMClassifier | None = None
        self.calibrated_: bool | None = None
        self._fitted = False
        self._is_constant = False  # set only in the single-class-in-fold fallback
        self._constant_class: str | None = None

    def _matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.feature_cols].fillna(0.0)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_calib: pd.DataFrame,
        y_calib: np.ndarray,
    ) -> ClassifyModel:
        train_classes = set(y_train)
        if len(train_classes) < 2:
            # LightGBM's multiclass objective cannot fit at all with fewer
            # than 2 classes present -- a real, reachable state with this
            # project's per-fold class counts (as few as a dozen SCC total),
            # not a hypothetical one. Fall back to a constant prediction
            # rather than force a fake second class into existence.
            print(
                f"ClassifyModel.fit: only {len(train_classes)} class in y_train "
                f"({sorted(train_classes)}) -- falling back to a constant prediction."
            )
            self._constant_class = (
                next(iter(train_classes)) if train_classes else self.classes[0]
            )
            self._is_constant = True
            self.calibrated_ = False
            self._fitted = True
            return self

        # num_class must be the number of classes ACTUALLY in y_train this
        # fit, not len(self.classes) (the full pinned list) -- confirmed
        # empirically, not assumed: LightGBM allocates predict_proba's output
        # width from num_class regardless of how many distinct labels y_train
        # contains, so num_class=5 with only 2 classes present produces a
        # (n, 5) probability array while `classes_` reports only the 2 seen
        # -- a real shape mismatch this project's own small-per-fold classes
        # hits often, not a rare edge case. Reindexing to `self.classes`
        # afterwards (in predict_proba) correctly zero-fills the classes this
        # particular fit never saw.
        self.model = LGBMClassifier(
            objective="multiclass",
            num_class=len(train_classes),
            n_estimators=self.lgbm_cfg.get("n_estimators", 50),
            num_leaves=self.lgbm_cfg.get("num_leaves", 7),
            min_child_samples=self.min_child_samples,
            class_weight=self.class_weight,
            deterministic=self.lgbm_cfg.get("deterministic", True),
            force_row_wise=self.lgbm_cfg.get("force_row_wise", True),
            num_threads=self.lgbm_cfg.get("num_threads", 1),
            random_state=self.seed,
            verbosity=-1,
        )
        self.model.fit(self._matrix(X_train), y_train)

        calib_classes = set(y_calib) if len(y_calib) else set()
        if len(X_calib) > 0 and train_classes <= calib_classes:
            frozen = FrozenEstimator(self.model)
            calibrated = CalibratedClassifierCV(estimator=frozen, method="isotonic")
            calibrated.fit(self._matrix(X_calib), y_calib)
            self.calibrated_model = calibrated
            self.calibrated_ = True
        else:
            print(
                f"ClassifyModel.fit: degenerate calibration split "
                f"(train classes {sorted(train_classes)}, calib classes {sorted(calib_classes)}) "
                "-- falling back to uncalibrated softmax for this fit."
            )
            self.calibrated_model = self.model
            self.calibrated_ = False
        self._fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("ClassifyModel.predict_proba() called before fit()")
        if self._is_constant:
            row = {c: (1.0 if c == self._constant_class else 0.0) for c in self.classes}
            return pd.DataFrame([row] * len(X), columns=self.classes, index=X.index)
        assert (
            self.calibrated_model is not None
        )  # guaranteed once _fitted and not _is_constant
        proba = self.calibrated_model.predict_proba(self._matrix(X))
        classes_ = list(self.calibrated_model.classes_)
        df = pd.DataFrame(proba, columns=classes_, index=X.index)
        # Reindex to the pinned class order -- classes_'s own order depends on
        # whatever happened to be in y_train, never trust it directly.
        return df.reindex(columns=self.classes, fill_value=0.0)

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        proba = self.predict_proba(X)
        pred_type = proba.idxmax(axis=1).to_numpy()
        pred_conf = proba.max(axis=1).to_numpy()
        return pred_type, pred_conf
