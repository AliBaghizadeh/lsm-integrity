"""
Background removal, and the shape features that separate an on-pipe defect from
off-pipe interference. This is the stage the project is actually about: the raw
field is ~45000 nT and the defect signature is ~25 nT, so 99.95% of the signal is
something to be removed before any model sees it.

The physics encoded here:
  - A defect is a buried dipole; its field falls off as 1/r^3. At stand-off h the
    along-track profile depends on s/h alone (times 1/h^3), so doubling the depth
    doubles the anomaly's WIDTH and divides its amplitude by eight. That scale
    invariance is a testable property, and tests/test_features.py tests it.
  - Off-pipe interference is farther away AND lateral, so it arrives weaker and
    broader. Width and decay shape separate it from a defect far better than
    amplitude does -- which is why fwhm_m, peak_asymmetry and decay_exponent
    exist and why amplitude alone is a false-positive machine.
  - With a second sensor head at a vertical baseline, the difference between the
    heads cancels the common-mode background (which is spatially uniform at this
    scale) while keeping the near-field defect term (which is not). That is
    gradiometry proper, not the along-track derivative.

Two kinds of transform, distinguished in the TYPE SIGNATURES rather than in a
comment (references/architecture.md, "Feature layer: stateless vs fitted"):

  StatelessTransform  (survey_df, SurveyContext) -> DataFrame
      Takes exactly one survey and that survey's own metadata. There is no
      parameter through which another survey's statistics could enter, so
      point-in-time correctness holds by construction rather than by discipline.
      These are legitimately re-fit at inference: they ARE background removal.

  FittedTransform     .fit(X) once, then .transform(X) many times
      Cross-survey statistics -- scalers, calibration maps, conformal quantiles,
      thresholds. Fit on the training corpus only and shipped inside the model
      bundle. Re-fitting one at inference is training/serving skew, so fit()
      raises on a second call instead of trusting the caller not to.

Storage is float32 (halves Stage-6 memory; ULP at 45000 nT is 0.004 nT, finer
than any magnetometer's resolution); arithmetic is float64 (least-squares
detrend fits and rolling sums accumulate error). See data-contract.md #2.
"""

from __future__ import annotations

import abc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths, savgol_filter

from lsm.config import FeaturesConfig
from lsm.logging_utils import get_logger
from lsm.schemas import FEATURE_KEY_COLUMNS, validate_feature_schema

log = get_logger("lsm.features")

STORAGE_DTYPE = "float32"
AXES = ("x", "y", "z")


class PointInTimeViolation(Exception):
    """A feature computation was handed data from after the survey it is computing.

    Spatial grouping does not catch temporal leakage: a GroupKFold on (line_id,
    block) will happily put survey 0 in train and survey 2 in test while a feature
    computed on survey 0 quietly used a statistic from survey 2.
    """


class NotFittedError(Exception):
    """transform() called on a FittedTransform that has never been fit."""


class AlreadyFittedError(Exception):
    """fit() called twice -- i.e. a train-only transform being re-fit at inference."""


@dataclass(frozen=True)
class SurveyContext:
    """Everything a stateless transform is allowed to know.

    Deliberately narrow: a survey's own identity, geometry and timestamp. There is
    no corpus, no connection and no population statistic here, so a stateless
    transform *cannot* reach across surveys even if someone tries.

    `gradiometer_baseline_m` is sensor metadata. It lives in code config today
    because the generator owns it; on real data it belongs on the `survey` row
    alongside step_m and standoff_m, and would be read from there.
    """

    survey_id: str
    line_id: str
    run_id: int
    step_m: float
    standoff_m: float
    surveyed_at: str
    gradiometer_baseline_m: float | None = None


@runtime_checkable
class StatelessTransform(Protocol):
    """Per-survey background removal. One survey in, features out, no state.

    The absence of a fit() method is the point: there is nothing to leak and
    nothing to keep in sync between training and serving.
    """

    def __call__(self, survey: pd.DataFrame, ctx: SurveyContext) -> pd.DataFrame: ...


class FittedTransform(abc.ABC):
    """Cross-survey statistics, fit on train only and carried in the model bundle.

    The guard rails are here rather than in the subclasses so every fitted
    transform in the project inherits the same two refusals: transform-before-fit,
    and fit-twice. The second is the one that matters -- silently re-fitting a
    scaler on the incoming survey at inference produces no error, no data-quality
    warning, and a model that is quietly wrong.
    """

    def __init__(self) -> None:
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, X: pd.DataFrame) -> FittedTransform:
        if self._fitted:
            raise AlreadyFittedError(
                f"{type(self).__name__} is already fit. Fitted transforms are fit on the "
                "training corpus and shipped in the bundle; re-fitting at inference is "
                "training/serving skew. Load the bundle's instance instead."
            )
        self._fit(X)
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise NotFittedError(f"{type(self).__name__}.transform() before fit()")
        return self._transform(X)

    @abc.abstractmethod
    def _fit(self, X: pd.DataFrame) -> None: ...

    @abc.abstractmethod
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame: ...


class RobustFeatureScaler(FittedTransform):
    """Median/IQR scaling fit on the training corpus. Robust rather than mean/std
    because a defect IS an outlier -- fitting a standard scaler on this data lets
    twelve anomalies set the scale for four thousand background rows.

    Trees do not need it; the conformal and calibration layers in Stage 4 do, and
    it is the concrete case that makes the fitted/stateless split real rather than
    a naming convention.
    """

    def __init__(self, columns: list[str] | None = None) -> None:
        super().__init__()
        self.columns = columns
        self.center_: pd.Series | None = None
        self.scale_: pd.Series | None = None

    def _fit(self, X: pd.DataFrame) -> None:
        cols = self.columns or [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        self.columns = cols
        block = X[cols].astype("float64")
        q1, q3 = block.quantile(0.25), block.quantile(0.75)
        self.center_ = block.median()
        # A constant column has IQR 0; scaling it by 1.0 leaves it constant, which
        # is correct. Substituting a small epsilon would amplify float noise into
        # a feature that looks informative.
        self.scale_ = (q3 - q1).replace(0.0, 1.0)

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        block = X[self.columns].astype("float64")
        out[self.columns] = ((block - self.center_) / self.scale_).astype(STORAGE_DTYPE)
        return out


# ---------------------------------------------------------------------------
# Point-in-time correctness
# ---------------------------------------------------------------------------


def assert_point_in_time(df: pd.DataFrame, as_of: str, what: str = "frame") -> None:
    """Raise if `df` contains any row surveyed after `as_of`.

    Called on the way into a feature computation and again on the way out of the
    corpus loader. Cheap, and it is the only thing standing between Stage 8's
    growth features and a model that has read the future.
    """
    if "surveyed_at" not in df.columns:
        return
    later = df["surveyed_at"].astype(str) > str(as_of)
    if bool(later.any()):
        offenders = sorted(df.loc[later, "surveyed_at"].astype(str).unique())[:5]
        raise PointInTimeViolation(
            f"{what} contains {int(later.sum())} rows surveyed after as_of={as_of}: "
            f"{offenders}. Features for a survey must be computable from data available "
            f"at that survey's surveyed_at."
        )


# ---------------------------------------------------------------------------
# Stateless transforms: background removal
# ---------------------------------------------------------------------------


def robust_poly_baseline(
    s_m: np.ndarray, y: np.ndarray, degree: int, n_iter: int = 4, tukey_c: float = 4.685
) -> np.ndarray:
    """Degree-`degree` polynomial baseline fit by IRLS with a Tukey biweight.

    Plain least squares would let the defects drag the baseline toward themselves
    and then subtract part of the signal we are trying to expose. Reweighting
    downweights exactly the rows we care about, which is the correct thing to do
    when fitting the thing we want to remove.

    s is normalised to [-1, 1] first: a Vandermonde matrix in raw chainage
    (0 to 2000 m, cubed) is badly conditioned and the fit degrades at the ends.
    """
    y = np.asarray(y, dtype=np.float64)
    span = s_m[-1] - s_m[0]
    x = (2.0 * (s_m - s_m[0]) / span - 1.0) if span > 0 else np.zeros_like(s_m)

    w = np.ones_like(y)
    coef = np.polyfit(x, y, degree, w=np.sqrt(w))
    for _ in range(n_iter):
        resid = y - np.polyval(coef, x)
        mad = np.median(np.abs(resid - np.median(resid)))
        scale = 1.4826 * mad
        if scale <= np.finfo(np.float64).eps * max(1.0, float(np.abs(y).max())):
            break  # already an exact fit; further reweighting is 0/0
        u = resid / (tukey_c * scale)
        w = np.where(np.abs(u) < 1.0, (1.0 - u**2) ** 2, 0.0)
        if w.sum() <= degree + 1:
            break  # too few surviving points to constrain the fit
        coef = np.polyfit(x, y, degree, w=np.sqrt(w))
    return np.polyval(coef, x)


def detrend_axis(s_m: np.ndarray, y: np.ndarray, cfg: FeaturesConfig) -> np.ndarray:
    """Raw axis -> residual. Stateless: fit on this survey, at inference too.

    Two passes. The polynomial removes the diurnal-scale drift; the rolling-median
    high-pass then removes long-wavelength structure the polynomial cannot express
    (the generator's sinusoid, and on real data the geomagnetic variation). The
    median is used rather than the mean because a defect occupies a small fraction
    of a 40 m window and a median ignores it, where a mean would subtract part of it.

    min_periods=1 on the high-pass deliberately keeps the edges rather than
    emitting NaN: the detrend must not create holes in a mandatory column. The
    edges ARE biased by the truncated window, which is what dq_flag='edge' records.
    """
    y = np.asarray(y, dtype=np.float64)
    step_m = (s_m[-1] - s_m[0]) / max(1, len(s_m) - 1)

    if cfg.detrend.method == "savgol":
        win = _odd_window(cfg.detrend.window_m, step_m, minimum=cfg.detrend.degree + 2)
        win = min(win, len(y) - 1 if len(y) % 2 == 0 else len(y))
        resid = y - savgol_filter(y, win, cfg.detrend.degree)
    else:
        resid = y - robust_poly_baseline(s_m, y, cfg.detrend.degree)

    if cfg.detrend.window_m > 0 and cfg.detrend.method != "savgol":
        win = _odd_window(cfg.detrend.window_m, step_m, minimum=3)
        baseline = pd.Series(resid).rolling(win, center=True, min_periods=1).median()
        resid = resid - baseline.to_numpy()
    return resid


def _odd_window(window_m: float, step_m: float, minimum: int = 3) -> int:
    """Window length in samples, forced odd so `center=True` is symmetric."""
    n = round(window_m / step_m)
    n = max(minimum, n)
    return n if n % 2 == 1 else n + 1


def _orientation(rx: np.ndarray, ry: np.ndarray, rz: np.ndarray, r_mag: np.ndarray):
    """Inclination/declination of the RESIDUAL vector, in degrees.

    A defect's dipole moment has a direction, and the residual's orientation is
    close to constant across the anomaly while background noise points anywhere.
    Guarded at |r| ~ 0 where the angles are genuinely undefined -> NaN, not 0.
    """
    tiny = r_mag < 1e-9
    incl = np.degrees(np.arcsin(np.divide(rz, r_mag, out=np.zeros_like(rz), where=~tiny)))
    decl = np.degrees(np.arctan2(ry, rx))
    return np.where(tiny, np.nan, incl), np.where(tiny, np.nan, decl)


def _peak_shape(
    r_mag: np.ndarray, step_m: float, cfg: FeaturesConfig
) -> dict[str, np.ndarray]:
    """FWHM, asymmetry and decay exponent -- the interference discriminators.

    Computed per detected peak (a handful per survey) and then broadcast to the
    rows around each peak, rather than recomputed in a window per row: at Stage-6
    row counts a per-row shape fit is the difference between minutes and hours,
    and the shape of a peak is a property of the peak, not of the row.

    Rows with no peak within assign_radius_fwhm get NaN. That is a real "no local
    anomaly here" statement, and LightGBM splits on it natively -- imputing 0
    would claim a zero-width peak exists.
    """
    n = len(r_mag)
    out = {
        k: np.full(n, np.nan)
        for k in ("fwhm_m", "peak_asymmetry", "decay_exponent", "peak_prominence_nt", "peak_distance_m")
    }
    med = np.median(r_mag)
    mad = np.median(np.abs(r_mag - med))
    sigma = 1.4826 * mad
    if sigma <= 0 or n < 8:
        return out

    peaks, props = find_peaks(r_mag, prominence=cfg.peak.prominence_mad * sigma)
    if len(peaks) == 0:
        return out

    widths, _, left_ips, right_ips = peak_widths(r_mag, peaks, rel_height=0.5)
    widths = np.maximum(widths, 1.0)
    fwhm_m = widths * step_m
    # +1 skewed right (a slow trailing flank), -1 skewed left. Normalised by the
    # width so it is a shape, not a size.
    asym = ((right_ips - peaks) - (peaks - left_ips)) / widths

    # Guard each flank fit against the neighbouring anomaly: fit only out to the
    # midpoint between this peak and the next, or the decay exponent measures the
    # rise of the neighbour instead of the fall of this peak.
    bounds_lo = np.empty(len(peaks), dtype=int)
    bounds_hi = np.empty(len(peaks), dtype=int)
    for i, p in enumerate(peaks):
        bounds_lo[i] = 0 if i == 0 else (peaks[i - 1] + p) // 2
        bounds_hi[i] = n - 1 if i == len(peaks) - 1 else (p + peaks[i + 1]) // 2

    decay = np.array(
        [
            _decay_exponent(r_mag, p, widths[i], cfg.peak.flank_fit_span, step_m,
                            bounds_lo[i], bounds_hi[i])
            for i, p in enumerate(peaks)
        ]
    )

    radius = cfg.peak.assign_radius_fwhm * widths
    nearest, dist = _nearest_peak(n, peaks)
    within = dist <= radius[nearest]
    for name, per_peak in (
        ("fwhm_m", fwhm_m),
        ("peak_asymmetry", asym),
        ("decay_exponent", decay),
        ("peak_prominence_nt", props["prominences"]),
    ):
        out[name][within] = per_peak[nearest[within]]
    out["peak_distance_m"][within] = dist[within] * step_m
    return out


def _nearest_peak(n: int, peaks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For every row, the index of the nearest peak and the distance in samples."""
    idx = np.arange(n)
    pos = np.searchsorted(peaks, idx)
    left = np.clip(pos - 1, 0, len(peaks) - 1)
    right = np.clip(pos, 0, len(peaks) - 1)
    d_left = np.abs(idx - peaks[left])
    d_right = np.abs(idx - peaks[right])
    nearest = np.where(d_left <= d_right, left, right)
    return nearest, np.minimum(d_left, d_right)


def _decay_exponent(
    r_mag: np.ndarray,
    peak: int,
    width_samples: float,
    span: tuple[float, float],
    step_m: float,
    lo_bound: int,
    hi_bound: int,
) -> float:
    """Slope of log|r| against log(distance from peak), fit on both flanks.

    The fit span is measured in FWHM, not metres, which makes this exponent
    SCALE-INVARIANT -- and a dipole profile is self-similar in s/h, so it comes
    out near-identical for an on-pipe defect and an off-pipe source. Measured on
    this data: -0.95 vs -0.84. It does not separate them, and fwhm_m (2.2 m vs
    6.2 m) does.

    It is kept because it measures a different thing: how dipole-LIKE the anomaly
    is, independent of size. A line source, an adjacent parallel pipeline or a
    saturation artifact departs from the dipole exponent where a scaled dipole
    does not. Recorded here rather than quietly dropped, because "this feature
    does not do what I first assumed" is worth knowing at Stage 5's SHAP review.
    """
    lo = max(1, int(span[0] * width_samples))
    hi = max(lo + 2, int(span[1] * width_samples))
    offsets = np.arange(lo, hi + 1)
    js = np.concatenate([peak - offsets, peak + offsets])
    ds = np.concatenate([offsets, offsets]).astype(np.float64) * step_m
    keep = (js >= lo_bound) & (js <= hi_bound)
    js, ds = js[keep], ds[keep]
    if len(js) < 4:
        return np.nan
    vals = r_mag[js]
    positive = vals > 0
    if positive.sum() < 4:
        return np.nan
    slope, _ = np.polyfit(np.log(ds[positive]), np.log(vals[positive]), 1)
    return float(slope)


# ---------------------------------------------------------------------------
# The stateless feature computation
# ---------------------------------------------------------------------------


def window_name(window_m: float) -> str:
    return f"w{window_m:g}m".replace(".", "p")


def feature_columns(cfg: FeaturesConfig, with_gradiometer: bool) -> list[str]:
    """The ordered feature list, derived from config.

    A bundle pins this list and refuses to load against a mismatch. Deriving it
    from config in one place -- rather than letting it emerge from whatever
    columns a DataFrame happened to end up with -- is what makes that check
    meaningful, and it also fixes column ORDER, which matters the moment anything
    downstream indexes by position.

    `r_mag_norm_nt_m3` / `peak_prominence_norm_nt_m3` (stand-off-normalised
    amplitude, feature_version 1) were removed entirely in feature_version 2:
    EDA (Stage 2.75, see PLAN.md) measured them as EXACT duplicates (r=1.000)
    of `r_mag_nt` / `peak_prominence_nt`, because `depth_m` is one global
    config value, not measured per row, so dividing by `depth_m**3` is a
    constant scalar multiply, not new information -- two dead dimensions in a
    model that already has too many redundant ones. Reintroduce them (with
    another feature_version bump) once a future stage makes stand-off a
    genuine per-row measurement and they diverge from their un-normalised
    counterparts again.
    """
    cols = [
        "rx_nt", "ry_nt", "rz_nt", "r_mag_nt",
        "r_incl_deg", "r_decl_deg",
        "dr_ds_nt_per_m", "d2r_ds2_nt_per_m2", "drz_ds_nt_per_m",
    ]
    if with_gradiometer:
        cols += ["gx_nt_per_m", "gy_nt_per_m", "gz_nt_per_m", "g_mag_nt_per_m"]
    for w in cfg.windows_m:
        p = window_name(w)
        cols += [f"{p}_mean_nt", f"{p}_std_nt", f"{p}_max_nt", f"{p}_ptp_nt",
                 f"{p}_kurt", f"{p}_zcr", f"{p}_energy_nt2"]
    cols += ["fwhm_m", "peak_asymmetry", "decay_exponent",
             "peak_prominence_nt", "peak_distance_m"]
    return cols


def compute_survey_features(
    df: pd.DataFrame, ctx: SurveyContext, cfg: FeaturesConfig
) -> pd.DataFrame:
    """One raw survey -> one feature frame. The stateless half of the feature layer.

    Signature is the enforcement: a single survey and its own context. Nothing in
    this function can see another survey, so there is no route by which a
    population statistic or a later run's data could enter.
    """
    assert_point_in_time(df, ctx.surveyed_at, what=f"survey {ctx.survey_id}")

    df = df.sort_values("sample_idx").reset_index(drop=True)
    n = len(df)
    step_m = ctx.step_m
    sample_idx = df["sample_idx"].to_numpy()
    s_m = sample_idx.astype(np.float64) * step_m  # derived, never a key

    # -- background removal, per axis, in float64 ---------------------------
    resid = {a: detrend_axis(s_m, df[f"b{a}_nt"].to_numpy(), cfg) for a in AXES}
    rx, ry, rz = resid["x"], resid["y"], resid["z"]
    r_mag = np.sqrt(rx**2 + ry**2 + rz**2)
    incl, decl = _orientation(rx, ry, rz, r_mag)

    out: dict[str, np.ndarray] = {
        "rx_nt": rx, "ry_nt": ry, "rz_nt": rz, "r_mag_nt": r_mag,
        "r_incl_deg": incl, "r_decl_deg": decl,
        # np.gradient uses central differences inside and one-sided at the ends,
        # so the derivative has no NaN of its own -- the window stats below own
        # the edge story, and having one owner for it keeps the flag honest.
        "dr_ds_nt_per_m": np.gradient(r_mag, s_m),
        "drz_ds_nt_per_m": np.gradient(rz, s_m),
    }
    out["d2r_ds2_nt_per_m2"] = np.gradient(out["dr_ds_nt_per_m"], s_m)

    # -- vertical gradiometer, when the second head is fitted ---------------
    has_grad = _has_gradiometer(df)
    if has_grad:
        if not ctx.gradiometer_baseline_m:
            raise ValueError(
                f"{ctx.survey_id} carries second-head columns but the context has no "
                "gradiometer_baseline_m -- a gradient needs its baseline, and guessing "
                "one silently rescales every gradient feature."
            )
        b = ctx.gradiometer_baseline_m
        g = {
            a: (df[f"b{a}2_nt"].to_numpy(dtype=np.float64)
                - df[f"b{a}_nt"].to_numpy(dtype=np.float64)) / b
            for a in AXES
        }
        out |= {f"g{a}_nt_per_m": g[a] for a in AXES}
        out["g_mag_nt_per_m"] = np.sqrt(g["x"] ** 2 + g["y"] ** 2 + g["z"] ** 2)

    # -- sliding-window statistics over |r| ---------------------------------
    r_series = pd.Series(r_mag)
    # Zero crossings are counted on rz rather than |r|: a magnitude is
    # non-negative and never crosses zero, so the same code on |r| would be
    # silently identically zero -- a feature that looks computed and is not.
    sign_change = np.zeros(n, dtype=np.float64)
    if n > 1:
        sign_change[1:] = (np.signbit(rz[1:]) != np.signbit(rz[:-1])).astype(np.float64)
    zc_series = pd.Series(sign_change)
    energy_series = pd.Series(r_mag**2)

    max_win = 0
    for w in cfg.windows_m:
        win = _odd_window(w, step_m, minimum=3)
        max_win = max(max_win, win)
        p = window_name(w)
        roll = r_series.rolling(win, center=True, min_periods=win)
        out[f"{p}_mean_nt"] = roll.mean().to_numpy()
        out[f"{p}_std_nt"] = roll.std().to_numpy()
        out[f"{p}_max_nt"] = roll.max().to_numpy()
        out[f"{p}_ptp_nt"] = (roll.max() - roll.min()).to_numpy()
        # rolling.kurt() needs 4 points; a 3-sample window legitimately yields NaN.
        out[f"{p}_kurt"] = roll.kurt().to_numpy() if win >= 4 else np.full(n, np.nan)
        out[f"{p}_zcr"] = zc_series.rolling(win, center=True, min_periods=win).mean().to_numpy()
        out[f"{p}_energy_nt2"] = (
            energy_series.rolling(win, center=True, min_periods=win).sum().to_numpy()
        )

    out |= _peak_shape(r_mag, step_m, cfg)

    # Stand-off normalisation (1/r^3: multiplying by depth_m^3 puts surveys flown
    # at different heights on one amplitude scale) was removed here (feature_version
    # 1 -> 2): EDA (Stage 2.75, PLAN.md) measured r_mag_norm_nt_m3 / peak_prominence_
    # norm_nt_m3 as EXACT duplicates (r=1.000) of r_mag_nt / peak_prominence_nt,
    # because depth_m is one global config value, not measured per row -- a constant
    # scalar multiply within any survey, confirming what this code already suspected
    # ("buys a tree model exactly nothing"). It earns its place once stand-off is a
    # genuine per-row measurement across surveys/lines, which is Stage 6 -- reintroduce
    # it there, not before.

    # -- assemble, flag edges, cast to storage precision ---------------------
    feat = pd.DataFrame(out, index=df.index)
    expected = feature_columns(cfg, with_gradiometer=has_grad)
    missing = set(expected) - set(feat.columns)
    extra = set(feat.columns) - set(expected)
    if missing or extra:
        raise AssertionError(
            f"feature_columns() and compute_survey_features() disagree for "
            f"{ctx.survey_id}: missing={sorted(missing)} extra={sorted(extra)}. "
            "The ordered list is what a bundle pins -- they cannot drift apart."
        )
    feat = feat[expected].astype(STORAGE_DTYPE)

    # The detrend high-pass window is wider than any statistics window, so it sets
    # the edge width. Using the statistics windows alone would leave the first and
    # last 20 m biased and unflagged.
    detrend_win = _odd_window(cfg.detrend.window_m, step_m, minimum=3)
    edge = max(max_win, detrend_win) // 2
    dq_flag = np.full(n, "clean", dtype=object)
    if edge > 0 and n > 0:
        dq_flag[:edge] = "edge"
        dq_flag[max(0, n - edge):] = "edge"

    keys = pd.DataFrame(
        {
            "survey_id": ctx.survey_id,
            "line_id": ctx.line_id,
            "run_id": np.int64(ctx.run_id),
            "sample_idx": sample_idx.astype(np.int64),
            "chainage_m": s_m,
        },
        index=df.index,
    )
    result = pd.concat(
        [keys, pd.DataFrame({"feature_version": np.int64(cfg.version), "dq_flag": dq_flag},
                            index=df.index), feat],
        axis=1,
    )

    if cfg.edge_policy == "drop":
        result = result[result["dq_flag"] == "clean"].reset_index(drop=True)

    validate_feature_schema(result)
    log.info(
        "features computed",
        extra={
            "survey_id": ctx.survey_id,
            "feature_version": cfg.version,
            "n_rows": len(result),
            "n_features": len(expected),
            "n_edge": int((result["dq_flag"] == "edge").sum()),
            "gradiometer": has_grad,
        },
    )
    return result


def _has_gradiometer(df: pd.DataFrame) -> bool:
    """A second head is fitted if its columns exist and carry any real value.

    All-NULL bx2_nt is the documented "no second sensor head" case, not a defect
    in the data -- the bundle must not require gradiometer features to exist.
    """
    cols = [f"b{a}2_nt" for a in AXES]
    if not all(c in df.columns for c in cols):
        return False
    return bool(df[cols].notna().any().any())


# ---------------------------------------------------------------------------
# Feature store
# ---------------------------------------------------------------------------


def feature_store_dir(feature_dir: str | Path, feature_version: int, line_id: str, run_id: int) -> Path:
    """features/fv=<n>/line_id=.../run_id=... -- feature_version is IN THE PATH.

    Not a column, not a convention: a directory. Two feature versions coexist on
    disk, an old bundle keeps reading the features it was trained against, and a
    bumped version cannot silently overwrite them.
    """
    return Path(feature_dir) / f"fv={feature_version}" / f"line_id={line_id}" / f"run_id={run_id}"


def read_feature_meta(dir_path: Path) -> dict | None:
    meta_path = Path(dir_path) / "meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None  # a half-written sidecar is a cache miss, not a crash


def is_cache_hit(dir_path: Path, content_sha256: str, feature_version: int) -> bool:
    """Cache key is content_sha256 + feature_version (data-contract.md #6).

    The content hash alone is not enough: the data is identical after a
    features.py change, which is exactly the training/serving skew no data check
    can see. Both halves, or the cache is a liability.
    """
    meta = read_feature_meta(dir_path)
    if meta is None or not (Path(dir_path) / "features.parquet").exists():
        return False
    return (
        meta.get("content_sha256") == content_sha256
        and meta.get("feature_version") == feature_version
    )


def write_features(
    dir_path: Path,
    features: pd.DataFrame,
    ctx: SurveyContext,
    content_sha256: str,
    feature_version: int,
    config_sha256: str,
) -> Path:
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    out_path = dir_path / "features.parquet"
    features.sort_values("sample_idx").to_parquet(out_path, index=False, compression="zstd")
    (dir_path / "meta.json").write_text(
        json.dumps(
            {
                "survey_id": ctx.survey_id,
                "line_id": ctx.line_id,
                "run_id": ctx.run_id,
                "surveyed_at": ctx.surveyed_at,
                "content_sha256": content_sha256,
                "feature_version": feature_version,
                "config_sha256": config_sha256,
                "n_rows": len(features),
                "n_edge_rows": int((features["dq_flag"] == "edge").sum()),
                "columns": list(features.columns),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_path


def compute_and_store(
    raw_df: pd.DataFrame,
    ctx: SurveyContext,
    cfg: FeaturesConfig,
    feature_dir: str | Path,
    content_sha256: str,
    config_sha256: str,
    force: bool = False,
) -> tuple[str, Path]:
    """Returns ('hit'|'computed', dir). The cache check is the only reason a
    Stage-6 re-run over 10^7 rows is tolerable."""
    dir_path = feature_store_dir(feature_dir, cfg.version, ctx.line_id, ctx.run_id)
    if not force and is_cache_hit(dir_path, content_sha256, cfg.version):
        return "hit", dir_path
    features = compute_survey_features(raw_df, ctx, cfg)
    write_features(dir_path, features, ctx, content_sha256, cfg.version, config_sha256)
    return "computed", dir_path


def load_feature_corpus(
    feature_dir: str | Path,
    feature_version: int,
    as_of: str,
    line_ids: list[str] | None = None,
) -> pd.DataFrame:
    """The as-of join: assemble a training corpus from the feature store using only
    surveys whose `surveyed_at` is at or before `as_of`.

    Filtering here and asserting afterwards is deliberate belt-and-braces -- the
    filter is the contract, the assertion is what catches a future refactor that
    reads the store some other way.
    """
    root = Path(feature_dir) / f"fv={feature_version}"
    frames = []
    for meta_path in sorted(root.glob("line_id=*/run_id=*/meta.json")):
        meta = read_feature_meta(meta_path.parent)
        if meta is None:
            continue
        if line_ids is not None and meta["line_id"] not in line_ids:
            continue
        if str(meta["surveyed_at"]) > str(as_of):
            continue
        frame = pd.read_parquet(meta_path.parent / "features.parquet")
        frame["surveyed_at"] = meta["surveyed_at"]
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=FEATURE_KEY_COLUMNS + ["surveyed_at"])
    corpus = pd.concat(frames, ignore_index=True)
    assert_point_in_time(corpus, as_of, what=f"feature corpus fv={feature_version}")
    return corpus
