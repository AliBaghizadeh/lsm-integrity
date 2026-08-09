"""
Stage B: registration. The module `LSM_PROJECT.md` has openly flagged as
missing ("Stage 2, register/align, has no counterpart here") since before
this rework -- and now, under Rig-v2, a real gap rather than a nice-to-have:
raw no longer carries a physically-final `chainage_m` at all (see
schemas.py). A human walker at irregular speed with GPS that drops out means
along-track position has to be RECONSTRUCTED, not read off a column.

Input: one survey's raw DataFrame, straight off the Parquet file (whatever a
real acquisition would have -- sample_idx, t_s, lat/lon (possibly NaN),
b_lo/mid/hi_nt). Output: a per-row registered chainage_m, a per-row distance
to the nearest detected girth weld, and a quality report.

THE ALGORITHM NEVER READS `girth_weld` OR `chainage_true_m`. Both are
truth-tier columns the generator can afford to know and this module cannot:
a real survey never has them. They exist only so a caller (tests, Stage D's
evaluation) can SCORE this module's output after the fact, exactly the way
`defect`/`interference` truth labels are used everywhere else in this
project -- never as an input signal. `registration_error_m` below is the one
function in this module that touches truth, and it is a pure scoring
utility: it takes two plain arrays, not a DataFrame, so there is no column
name that could accidentally leak truth into `register_survey` itself.

Two stages, in order:

1. Dead reckoning (`_dead_reckon_chainage`): locked GPS fixes are projected
   onto the survey's fixed bearing (the same north/east rotation
   generate.py::_gps_track uses, inverted) to get an along-track position
   wherever GPS is locked. Through a dropout, position is reconstructed by
   integrating a LOCALLY estimated walking rate (the last/next known GPS
   ground speed either side of the gap, blended across it) rather than a
   flat straight-line interpolation between the two anchoring fixes --
   physically, a walker's pace right before losing lock is a much better
   predictor of pace during the gap than the gap's own average secant slope
   would be once the gap is more than a few strides long. The result is then
   nudged so it still lands exactly on every locked GPS fix (anchoring) and
   forced non-decreasing (belt-and-braces; the physics already guarantees
   this if the rate estimate stays positive, but a `np.maximum.accumulate`
   costs nothing and removes any doubt).

2. Weld-comb registration (`_detect_weld_comb` + `_monotonic_warp`): girth
   welds are a periodic dipole train (WeldConfig.pitch_m) with ARBITRARY,
   isotropic orientation -- the developer's point that there is no fixed
   amplitude/polarity template for a joint. So this detects the comb by
   PERIODICITY alone: autocorrelate the DETRENDED |b_mid| ENVELOPE (not the
   raw signed residual -- verified in
   tests/test_generate.py::test_weld_pitch_is_recoverable_by_autocorrelation
   that autocorrelating the signed residual lets adjacent welds' random
   polarity partially cancel at lag=pitch, while the envelope's autocorrelation
   recovers the pitch cleanly). `cfg.weld.pitch_m` is accepted as a SEARCH-
   RANGE PRIOR only (a physical constant of the pipe class, same as knowing
   a joint spacing spec sheet -- not leaked ground truth): the actual pitch
   estimate is read off the genuine autocorrelation peak within a window
   around that prior, never trusted blindly. Individual weld positions are
   then found by a matched comb / peak-picking pass at the recovered pitch,
   not a fixed-shape template. Detected weld positions are snapped onto the
   evenly-spaced lattice the autocorrelation implies, and a monotonic
   piecewise-linear warp from dead-reckoned chainage onto that lattice is
   applied to every row -- between two welds (~12 m apart) the residual
   error is bounded by a small fraction of the pitch, which is how a
   centimetre-level position becomes reachable from metre-level raw GPS.

Every step degrades gracefully instead of raising: 0% GPS lock still
produces a monotonic (if wider-uncertainty) chainage; a survey with no welds
in view (too short, or `weld.pitch_m` bigger than the surveyed length) falls
back to GPS/dead-reckoning-only chainage with `n_welds_detected=0`.

Dependency-light on purpose (numpy/pandas/scipy only, no import from
features.py): Stage C's features.py will need to import FROM this module, so
the reverse import would be a circular dependency. The local detrend used
for weld detection is therefore its own small rolling-median high-pass, not
a reuse of features.py's `robust_poly_baseline`/`detrend_axis`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from lsm.config import DataConfig

# Resampling grid step for the weld-comb autocorrelation/matched-filter pass.
# Independent of the survey's own (irregular) row spacing -- see module
# docstring point 2. 10 cm resolves a ~12 m pitch comfortably (>100 samples/
# period) without the grid getting so fine it amplifies interpolation noise.
_GRID_DX_M = 0.10

# Rolling-median baseline (in metres of along-track travel) used to smooth
# raw per-sample GPS fixes before they anchor dead reckoning -- see
# _dead_reckon_chainage's comment on why raw per-sample fixes are too noisy
# to anchor directly at this rig's sample rate.
_GPS_SMOOTH_M = 2.0


@dataclass(frozen=True)
class RegistrationQuality:
    """What `register_survey` actually managed, and how much to trust it.

    `note` records WHY weld detection fell back (0% GPS is not recorded here
    -- that is `gps_locked_fraction == 0.0`, self-explanatory); `note` is for
    the weld-comb-specific degradations (survey too short, no periodicity
    found, etc).
    """

    n_welds_detected: int
    pitch_estimate_m: float | None
    gps_locked_fraction: float
    max_interpolated_gap_m: float
    note: str = ""


@dataclass(frozen=True)
class RegistrationResult:
    """`chainage_m`, `dist_to_weld_m` and `weld_chainage_m` are aligned to
    the INPUT DataFrame's row order (same length as `df`, `chainage_m[i]`
    corresponds to `df.iloc[i]`) -- never resorted, never reindexed, so a
    caller can assign this straight back onto the original frame.

    `dist_to_weld_m` is NaN, not +inf, when `n_welds_detected == 0`: there is
    genuinely no weld reference to measure a distance to, and a fabricated
    "very far" sentinel would silently look like real information to a
    downstream consumer that doesn't check `quality.n_welds_detected` first.

    `weld_chainage_m` holds the detected welds' own positions, already on
    the corrected (registered) axis -- useful for anything downstream that
    wants the weld lattice itself (e.g. a nuisance mask window), not just
    each row's distance to it.
    """

    chainage_m: np.ndarray
    dist_to_weld_m: np.ndarray
    weld_chainage_m: np.ndarray
    quality: RegistrationQuality


def register_survey(df: pd.DataFrame, cfg: DataConfig) -> RegistrationResult:
    """Register one survey's raw DataFrame to a chainage_m axis.

    `df` must have `t_s`, `lat`, `lon`, `b_mid_nt` (whatever a real
    acquisition has -- see module docstring). `cfg` supplies the fixed
    survey bearing/origin (`cfg.geo`, needed to project GPS) and the weld-
    pitch PRIOR (`cfg.weld.pitch_m`/`pitch_jitter_m`, a search-range hint,
    never trusted blindly -- see module docstring point 2). Never raises on
    degenerate input (no GPS, no welds in view); see `RegistrationQuality`.
    """
    n = len(df)
    t_s = df["t_s"].to_numpy(dtype=float)
    lat = df["lat"].to_numpy(dtype=float)
    lon = df["lon"].to_numpy(dtype=float)
    b_mid = df["b_mid_nt"].to_numpy(dtype=float)

    dr_chainage, gps_locked_fraction, max_gap_m = _dead_reckon_chainage(t_s, lat, lon, cfg)

    detected_dr, ideal_dr, pitch_estimate, note = _detect_weld_comb(
        dr_chainage, b_mid, cfg.weld.pitch_m
    )
    n_welds = len(detected_dr)

    if n_welds >= 2:
        chainage_m = _monotonic_warp(dr_chainage, detected_dr, ideal_dr)
        weld_chainage_m = ideal_dr
    else:
        chainage_m = dr_chainage
        weld_chainage_m = np.array([], dtype=float)

    # Belt-and-braces: both dead reckoning and the warp are monotonic by
    # construction if their inputs behave, but this costs nothing and
    # guarantees the invariant regardless.
    chainage_m = np.maximum.accumulate(chainage_m)

    if n_welds:
        dist_to_weld_m = _nearest_distance(chainage_m, weld_chainage_m)
    else:
        dist_to_weld_m = np.full(n, np.nan)

    quality = RegistrationQuality(
        n_welds_detected=n_welds,
        pitch_estimate_m=pitch_estimate,
        gps_locked_fraction=gps_locked_fraction,
        max_interpolated_gap_m=max_gap_m,
        note=note,
    )
    return RegistrationResult(chainage_m, dist_to_weld_m, weld_chainage_m, quality)


def registration_error_m(chainage_m: np.ndarray, chainage_true_m: np.ndarray) -> dict:
    """SCORING ONLY. Takes two plain arrays, not a DataFrame or column name,
    so there is no way to accidentally wire truth into `register_survey`
    itself through this function. A real survey never has `chainage_true_m`
    -- this exists purely for tests/evaluation to quantify accuracy against
    the generator's own truth column (data-contract.md's truth tier, same
    status as `defect`/`interference`).
    """
    err = np.abs(np.asarray(chainage_m, dtype=float) - np.asarray(chainage_true_m, dtype=float))
    return {
        "median_abs_error_m": float(np.median(err)),
        "p90_abs_error_m": float(np.percentile(err, 90)),
        "max_abs_error_m": float(np.max(err)),
        "median_abs_error_cm": float(np.median(err) * 100.0),
    }


# ---------------------------------------------------------------------------
# Stage 1: dead reckoning
# ---------------------------------------------------------------------------


def _project_gps_chainage(lat: np.ndarray, lon: np.ndarray, cfg: DataConfig) -> np.ndarray:
    """Inverse of generate.py::_gps_track's north/east rotation: given a
    fixed survey origin/bearing (a property of how the line was surveyed,
    known to any real registration step -- not leaked truth), recover
    along-track chainage from lat/lon. NaN in, NaN out.
    """
    brg = np.radians(cfg.geo.bearing_deg)
    lat0, lon0 = cfg.geo.lat0, cfg.geo.lon0
    north = (lat - lat0) * 111_320.0
    east = (lon - lon0) * 111_320.0 * np.cos(np.radians(lat0))
    return north * np.cos(brg) + east * np.sin(brg)


def _dead_reckon_chainage(
    t_s: np.ndarray, lat: np.ndarray, lon: np.ndarray, cfg: DataConfig
) -> tuple[np.ndarray, float, float]:
    """Provisional along-track chainage: locked GPS wherever available,
    integrated local rate through gaps. Returns (chainage_m, gps_locked_
    fraction, max_interpolated_gap_m). Always monotonic non-decreasing.
    """
    n = len(t_s)
    if n == 0:
        return np.array([]), 0.0, 0.0

    locked = ~(np.isnan(lat) | np.isnan(lon))
    gps_locked_fraction = float(locked.mean())

    if not locked.any():
        # Pure dead reckoning: no fix anywhere in this survey. The nominal
        # configured walking pace is the best available prior with nothing
        # to correct it against -- report the whole span as one big
        # "interpolated" gap so the quality report is honest about it.
        chainage = cfg.walk.speed_m_per_s * (t_s - t_s[0])
        return chainage, 0.0, float(chainage[-1] - chainage[0]) if n > 1 else 0.0

    s_gps = _project_gps_chainage(lat, lon, cfg)
    idx_locked = np.flatnonzero(locked)
    t_locked = t_s[idx_locked]
    s_locked = s_gps[idx_locked]

    # A per-sample GPS fix has sigma_m=1.5 m of IID noise (GpsConfig), but at
    # the rig's sample rate real motion between consecutive samples is only
    # ~walk.speed_m_per_s / sample_rate_hz -- a few CM -- so the raw fixes
    # bounce around by tens of their own real spacing and are NOT individually
    # trustworthy anchors (measured: anchoring to them directly and forcing
    # monotonicity afterwards introduced a ~2 m systematic upward bias, from
    # the monotonic clamp repeatedly ratcheting through downward noise
    # excursions). A short rolling median over the locked subsequence -- a
    # baseline much shorter than the weld pitch, long enough to average
    # several independent fixes -- removes that noise while still tracking
    # genuine speed changes (OU correlation length 10 m, WalkConfig).
    if len(t_locked) > 1:
        dt_nominal = float(np.median(np.diff(t_s))) if n > 1 else 1.0
        smooth_samples = max(1, round(_GPS_SMOOTH_M / max(cfg.walk.speed_m_per_s * dt_nominal, 1e-6)))
        if smooth_samples % 2 == 0:
            smooth_samples += 1
        smooth_samples = min(smooth_samples, len(s_locked) if len(s_locked) % 2 else len(s_locked) - 1)
        smooth_samples = max(smooth_samples, 1)
        if smooth_samples > 1:
            s_locked = median_filter(s_locked, size=smooth_samples, mode="nearest")

    if len(idx_locked) == 1:
        # One fix, no local rate observable -- integrate at nominal speed
        # and shift so the single fix is matched exactly.
        chainage = cfg.walk.speed_m_per_s * (t_s - t_s[0])
        chainage += s_locked[0] - chainage[idx_locked[0]]
        return chainage, gps_locked_fraction, float(t_s[-1] - t_s[0])

    # Local ground speed at each locked sample (gradient handles the locked
    # subsequence's own -- possibly irregular -- spacing correctly).
    v_locked = np.gradient(s_locked, t_locked)
    v_locked = np.clip(v_locked, cfg.walk.speed_min_m_per_s, cfg.walk.speed_max_m_per_s)

    # Extend that local-rate estimate to every sample (locked or not) by
    # linear interpolation in TIME, clamped flat beyond the first/last fix
    # (best available estimate at the edges is "whatever the nearest fix's
    # rate was"). This -- not a flat secant across the whole gap -- is what
    # carries "the local dead-reckoning rate" through a dropout.
    v_full = np.interp(t_s, t_locked, v_locked)

    dt = np.diff(t_s)
    v_mid = 0.5 * (v_full[:-1] + v_full[1:])
    raw = np.concatenate([[0.0], np.cumsum(v_mid * dt)])

    # raw() drifts away from the true anchors between fixes (v_full is only
    # ever an estimate). Correct it with a piecewise-linear residual fit
    # through the locked anchors, so the final chainage matches GPS exactly
    # wherever GPS is trusted and blends smoothly (not flatly) in between.
    residual_at_locked = s_locked - raw[idx_locked]
    residual_full = np.interp(t_s, t_locked, residual_at_locked)
    chainage = raw + residual_full
    chainage = np.maximum.accumulate(chainage)

    # Longest along-track span bridged without a GPS fix -- the plan's
    # "max_interpolated_gap_m" quality figure.
    gap_runs = _false_run_lengths(locked)
    if gap_runs:
        max_gap_m = max(
            float(chainage[j - 1] - chainage[i]) if i > 0 else float(chainage[j - 1] - chainage[0])
            for i, j in gap_runs
        )
    else:
        max_gap_m = 0.0

    return chainage, gps_locked_fraction, max_gap_m


def _false_run_lengths(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of each run of consecutive False in `mask`."""
    runs = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            i += 1
            continue
        j = i
        while j < n and not mask[j]:
            j += 1
        runs.append((i, j))
        i = j
    return runs


# ---------------------------------------------------------------------------
# Stage 2: weld-comb detection and registration
# ---------------------------------------------------------------------------


def _detect_weld_comb(
    dr_chainage: np.ndarray, b_mid_nt: np.ndarray, pitch_prior_m: float
) -> tuple[np.ndarray, np.ndarray, float | None, str]:
    """Returns (detected_positions_m, ideal_lattice_positions_m,
    pitch_estimate_m, note) on the SAME axis as `dr_chainage` (dead-reckoned
    chainage) -- both empty arrays and `pitch_estimate_m=None` mean "no
    reliable weld comb found here", never an exception.
    """
    span = float(dr_chainage[-1] - dr_chainage[0]) if len(dr_chainage) > 1 else 0.0
    if span < 1.5 * pitch_prior_m or len(dr_chainage) < 20:
        return (
            np.array([]),
            np.array([]),
            None,
            "survey shorter than ~1.5 weld pitches -- no periodicity to detect",
        )

    grid = np.arange(dr_chainage[0], dr_chainage[-1], _GRID_DX_M)
    if len(grid) < 20:
        return np.array([]), np.array([]), None, "too few resampled points for autocorrelation"

    b_grid = np.interp(grid, dr_chainage, b_mid_nt)

    # Local high-pass: rolling MEDIAN (robust to a weld's own spike, unlike a
    # mean) over a window several pitches wide -- wide enough to leave the
    # periodic comb itself alone while removing slower background drift.
    window_m = max(5.0 * pitch_prior_m, 10 * _GRID_DX_M)
    window_samples = round(window_m / _GRID_DX_M)
    window_samples = min(window_samples, len(b_grid) - 1)
    window_samples = max(window_samples, 3)
    if window_samples % 2 == 0:
        window_samples += 1
    baseline = median_filter(b_grid, size=window_samples, mode="nearest")
    resid = b_grid - baseline

    # Envelope, not the signed residual: each weld's isotropic orientation
    # means adjacent welds' anomalies have uncorrelated SIGN as well as
    # shape, so the signed autocorrelation partially cancels at lag=pitch.
    # See module docstring / test_generate.py's verified reference test.
    envelope = np.abs(resid)
    envelope = envelope - envelope.mean()

    ac = np.correlate(envelope, envelope, mode="full")
    ac = ac[len(ac) // 2 :]
    lags_m = np.arange(len(ac)) * _GRID_DX_M

    lo = max(3 * _GRID_DX_M, 0.5 * pitch_prior_m)
    hi = min(1.5 * pitch_prior_m, lags_m[-1])
    search = (lags_m >= lo) & (lags_m <= hi)
    if not search.any():
        return np.array([]), np.array([]), None, "search range around the pitch prior is empty"

    # Degenerate-signal guard: a genuinely flat/constant residual (e.g. a
    # window so wide relative to the grid that the median filter reproduces
    # b_grid almost exactly) leaves no envelope variance to autocorrelate,
    # and an argmax over near-zero noise would just be a coin flip.
    if envelope.std() < 1e-9:
        return np.array([]), np.array([]), None, "no residual variance after detrending"

    ac_search, lag_search = ac[search], lags_m[search]
    peak_i = int(np.argmax(ac_search))
    peak_val = ac_search[peak_i]
    pitch_estimate = float(lag_search[peak_i])

    # No amplitude/significance gate beyond that: measured against real
    # generated surveys, a genuine weld comb's autocorrelation peak clears
    # its own search-window median by as little as ~1.3x (gain-mismatch and
    # background-gradient noise dilute it a lot more than the clean-signal
    # reference test in test_generate.py sees), while a pure-noise window
    # can spike to ~3x from chance alone -- an amplitude threshold rejects
    # real detections without reliably rejecting noise. `cfg.weld.pitch_m`
    # is a physical fact about how the pipe was built (not a maybe), so
    # "are there welds in view at all" is answered by the STRUCTURAL checks
    # above (is the surveyed span even long enough to contain one) rather
    # than by how prominent the autocorrelation peak looks.
    if peak_val <= 0:
        return np.array([]), np.array([]), None, "no positive periodicity found in range"

    n_periods = int(span // pitch_estimate)
    if n_periods < 2:
        return np.array([]), np.array([]), pitch_estimate, "fewer than 2 periods fit in the survey"

    # Phase: matched filter for a periodic impulse comb at the recovered
    # pitch (spacing only, no amplitude/shape template -- welds have none).
    best_phase, best_score = 0.0, -np.inf
    for phase in np.arange(0.0, pitch_estimate, _GRID_DX_M):
        positions = grid[0] + phase + np.arange(n_periods + 1) * pitch_estimate
        positions = positions[(positions >= grid[0]) & (positions <= grid[-1])]
        if len(positions) == 0:
            continue
        idxs = np.clip(np.searchsorted(grid, positions), 0, len(envelope) - 1)
        score = float(envelope[idxs].sum())
        if score > best_score:
            best_score, best_phase = score, phase

    candidates = grid[0] + best_phase + np.arange(n_periods + 1) * pitch_estimate
    half_window = pitch_estimate / 4.0
    # median_filter's boundary replication (mode="nearest") biases `baseline`
    # over roughly the first/last half of `window_m` -- a candidate in that
    # zone gets a corrupted residual and its "peak" is really an artefact of
    # the boundary, not the weld (measured: this pulled several early-survey
    # detections to the very edge of their search window). Skip that margin
    # rather than detect against a baseline known to be wrong there.
    edge_margin_m = window_m / 2.0

    detected_list: list[float] = []
    k_index_list: list[int] = []
    for k, c in enumerate(candidates):
        if c - half_window < grid[0] or c + half_window > grid[-1]:
            continue  # partial period at the very edge -- skip rather than guess
        if (c - grid[0]) < edge_margin_m or (grid[-1] - c) < edge_margin_m:
            continue
        lo_i = np.searchsorted(grid, c - half_window)
        hi_i = np.searchsorted(grid, c + half_window)
        if hi_i <= lo_i:
            continue
        local = envelope[lo_i:hi_i]
        detected_list.append(float(grid[lo_i + int(np.argmax(local))]))
        k_index_list.append(k)

    if len(detected_list) < 2:
        return np.array([]), np.array([]), pitch_estimate, "fewer than 2 welds survived edge trimming"

    # Refine pitch/phase by least-squares fitting detected position ~ index,
    # rather than trusting the coarse autocorrelation lag (quantised to
    # _GRID_DX_M, ~0.1 m) directly as the lattice step: extrapolated over
    # dozens of periods, a 0.1-0.3 m pitch bias compounds into METRES of
    # drift by the far end of the survey (measured: using the raw
    # autocorrelation pitch as the lattice step alone made registration
    # WORSE than the naive baseline on a 500 m/41-weld survey). Averaging
    # over every detected weld cancels that bias the same way any regression
    # beats a single two-point slope estimate.
    detected = np.array(detected_list)
    k_index = np.array(k_index_list, dtype=float)
    refined_pitch, refined_phase = np.polyfit(k_index, detected, 1)

    # One robust re-fit pass: an isolated bad detection (a defect or
    # interference source is deliberately allowed to overlap a weld window,
    # generate.py::_build_welds's docstring) can pull an OLS fit noticeably
    # even though it is a single point -- drop anything beyond ~3 robust
    # sigma from the first fit and refit. A no-op when nothing is actually
    # an outlier (kept-fraction stays 1.0), so this costs nothing on the
    # common case.
    resid0 = detected - (refined_phase + k_index * refined_pitch)
    mad = float(np.median(np.abs(resid0 - np.median(resid0)))) * 1.4826 + 1e-9
    keep = np.abs(resid0) < max(3.0 * mad, 0.5)
    if keep.sum() >= 2 and keep.sum() < len(detected):
        # Drop the outlier from the returned welds entirely, not just from the
        # fit -- keeping its (bad) detected position as a warp knot would
        # locally distort the piecewise-linear warp right around it even with
        # a correctly refit global pitch/phase.
        detected, k_index = detected[keep], k_index[keep]
        refined_pitch, refined_phase = np.polyfit(k_index, detected, 1)

    ideal = refined_phase + k_index * refined_pitch

    return detected, ideal, float(refined_pitch), "ok"


def _monotonic_warp(x_all: np.ndarray, knot_x: np.ndarray, knot_y: np.ndarray) -> np.ndarray:
    """Piecewise-linear monotonic warp through (knot_x, knot_y), linearly
    EXTRAPOLATED (not flat-clamped) beyond the first/last knot using the
    nearest segment's own slope -- np.interp's default flat clamp would
    freeze chainage before the first / after the last detected weld, which
    is wrong (the walk keeps moving) even though it stays non-decreasing.
    """
    order = np.argsort(knot_x)
    kx, ky = knot_x[order], knot_y[order]
    warped = np.interp(x_all, kx, ky)

    below = x_all < kx[0]
    if below.any():
        slope = (ky[1] - ky[0]) / (kx[1] - kx[0])
        warped[below] = ky[0] + slope * (x_all[below] - kx[0])
    above = x_all > kx[-1]
    if above.any():
        slope = (ky[-1] - ky[-2]) / (kx[-1] - kx[-2])
        warped[above] = ky[-1] + slope * (x_all[above] - kx[-1])
    return warped


def _nearest_distance(values: np.ndarray, ref_sorted: np.ndarray) -> np.ndarray:
    """|values[i] - nearest ref_sorted| via searchsorted -- O(n log w), not
    the O(n*w) full pairwise difference (matters once welds number in the
    hundreds on a full-length survey).
    """
    if len(ref_sorted) == 1:
        return np.abs(values - ref_sorted[0])
    idx = np.clip(np.searchsorted(ref_sorted, values), 1, len(ref_sorted) - 1)
    left, right = ref_sorted[idx - 1], ref_sorted[idx]
    return np.minimum(np.abs(values - left), np.abs(values - right))
