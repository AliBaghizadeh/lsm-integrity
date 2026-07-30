"""
Stage 3: two interchangeable row-level anomaly scorers, sharing one interface so
evaluate.py can compare them without special-casing either. "Add its baseline
first" (SKILL invariant): MADBaseline is what IsolationForest has to beat, not a
strawman fit after the fact.

Both are FITTED (population statistics: a median/MAD, a forest), so unlike
features.py's per-survey stateless transforms, these belong in a model bundle
and must be fit on train only -- fitting on the evaluation fold would leak the
very background level the fold's residual is being judged against.

Neither ever sees `defect`, `defect_type` or `severity_smys` -- both fit only on
the label-free feature store, same as `feature_columns()` -- these are
unsupervised detectors, not classifiers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class MADBaseline:
    """Robust z-score on |residual|: (|r| - median) / (1.4826 * MAD).

    Deliberately amplitude-only, no shape information at all -- so whatever
    IsolationForest gains over this baseline has to come from somewhere other
    than "residual is big", which is exactly the interference-rejection
    question Stage 3's gate asks about (validation-and-trust.md: interference
    is separated by shape, not amplitude).
    """

    def __init__(self, residual_col: str = "r_mag_nt"):
        self.residual_col = residual_col
        self.median_: float | None = None
        self.mad_: float | None = None

    def fit(self, X: pd.DataFrame) -> "MADBaseline":
        x = X[self.residual_col].to_numpy(dtype=np.float64)
        self.median_ = float(np.median(x))
        mad = float(np.median(np.abs(x - self.median_)))
        self.mad_ = max(mad, 1e-9) * 1.4826  # guard div-by-zero on a degenerate fold
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        if self.median_ is None:
            raise RuntimeError("MADBaseline.score() called before fit()")
        x = X[self.residual_col].to_numpy(dtype=np.float64)
        return (x - self.median_) / self.mad_


class IsolationForestAnomalyModel:
    """sklearn IsolationForest over the full residual + shape feature set --
    the same ordered list `bundle.py` will pin in Stage 4 (features.feature_columns()).

    Scored via `decision_function`, not `score_samples`: sklearn calibrates
    `decision_function`'s zero-crossing to the configured `contamination` rate
    on the TRAINING data (negative = outlier), so negating it gives "higher =
    more anomalous" with threshold=0 already meaning "the contamination-th
    percentile of training rows" -- a principled cutoff shared with MADBaseline
    (see evaluate.py's caller, which calibrates MAD's threshold to the same
    contamination for a fair comparison), not a second magic number.

    Shape features (fwhm_m, peak_asymmetry, decay_exponent, ...) are NaN on rows
    with no peak nearby -- IsolationForest has no native missing-value support,
    so they are filled with 0.0 ("no local peak structure"), fit and applied
    identically at score time.
    """

    def __init__(self, feature_cols: list[str], contamination: float, n_estimators: int, seed: int):
        self.feature_cols = feature_cols
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=seed,
            n_jobs=-1,
        )

    def _matrix(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.feature_cols].fillna(0.0).to_numpy(dtype=np.float64)

    def fit(self, X: pd.DataFrame) -> "IsolationForestAnomalyModel":
        self.model.fit(self._matrix(X))
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return -self.model.decision_function(self._matrix(X))


def calibrated_threshold(train_scores: np.ndarray, contamination: float) -> float:
    """The (1 - contamination) quantile of a model's scores on its OWN training
    fold. IsolationForest already self-calibrates this to 0 via its
    `contamination` parameter; MADBaseline has no such built-in, so this gives
    it the same nominal flagged-fraction, fit on train only, so both models feed
    indications.py the same starting sensitivity and any difference downstream
    is attributable to WHICH rows get flagged, not to a mismatched threshold.
    """
    return float(np.quantile(train_scores, 1.0 - contamination))
