"""
Synthetic LSM survey generator -- Rig-v2. Physics-inspired, not physically
validated: a buried pipeline modelled as a line of magnetic dipoles
("defects"), sensed by a rod carrying three total-field magnetometers (middle
+ two, 50 cm apart) at varying stand-off height, carried by a human walker at
irregular speed with GPS that drops out. This replaces the original single
3-axis-vector-magnetometer-on-a-rail model after a developer interview
revealed the real instrument (see the Rig-v2 plan's "Context" and "The
physics that drives everything else" sections).

THE CENTRAL PHYSICS: each head reports only |B|, not x/y/z. With the anomaly
(~25 nT) tiny against the ambient field (~48,800 nT), that reading is the
TOTAL-FIELD anomaly:

    |B0 + dB| - |B0| ~= dB . B_hat0        (error < 0.01 nT at these amplitudes)

i.e. only the projection of a defect's field onto the ambient field direction
is visible. This module computes the exact norm (never the linear projection)
so that approximation becomes a TESTABLE PROPERTY of the generated data
(tests/test_generate.py), not a modelling shortcut baked into it. Three
consequences fall out and are asserted in tests, not just claimed: (1) a
moment near-perpendicular to B_hat0 is nearly invisible -- probability of
detection genuinely varies with defect orientation; (2) |B| is
rotation-invariant, so rod sway/tilt moves head POSITIONS but never corrupts
a reading by itself; (3) three heads give both a first difference (common-mode
background rejection) and a second difference (linear-gradient rejection,
`b_hi + b_lo - 2*b_mid` -- the third head's specific value over a two-head
gradiometer).

`data.rig` selects between two structurally different generators:
  - "scalar" (default): Rig-v2, everything above.
  - "vector": the ORIGINAL pre-Rig-v2 model, preserved byte-for-byte in
    behaviour (uniform grid, bx/by/bz [+bx2/by2/bz2] output, no walk/GPS/weld/
    sensor physics). Not a maintained second product -- it exists solely as
    the reference arm of the Stage D ablation ("what would a hardware upgrade
    to full vector output buy over the real scalar rod"). It does NOT produce
    RawReadingSchema-conformant output and is not run by the default config.

Two structural invariants carried over unchanged from the original generator
(references/data-contract.md):
  1. The physical key is the integer `sample_idx`, never a float. Under the
     scalar rig it is now a dense TIME-sample counter (the walk is sampled at
     a fixed rate in time, `walk.sample_rate_hz`, and irregularly spaced in
     distance because speed varies) rather than a distance-grid index, but it
     stays dense, monotonic, and the only key -- nothing about joins or
     content_sha256 changes.
  2. Output is one immutable Parquet file per survey (zstd, sorted by
     sample_idx), not a single combined CSV.

Run: python -m lsm generate
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from lsm.config import (
    PROJECT_ROOT,
    ArrayConfig,
    DataConfig,
    GpsConfig,
    ObservatoryBackgroundConfig,
    SensorConfig,
    WalkConfig,
)
from lsm.hashing import content_sha256, file_sha256

COUPLING = 3.0

RAW_COLUMNS_SCALAR = [
    "sample_idx",
    "t_s",
    "chainage_true_m",
    "chainage_provisional_m",
    "lat",
    "lon",
    "b_lo_nt",
    "b_mid_nt",
    "b_hi_nt",
    "defect",
    "defect_type",
    "severity_smys",
    "interference",
    "girth_weld",
    "line_id",
    "run_id",
]

RAW_COLUMNS_VECTOR = [
    "sample_idx",
    "chainage_m",
    "lat",
    "lon",
    "bx_nt",
    "by_nt",
    "bz_nt",
    "bx2_nt",
    "by2_nt",
    "bz2_nt",
    "defect",
    "defect_type",
    "severity_smys",
    "interference",
    "line_id",
    "run_id",
]


def _load_observatory_background(cfg: ObservatoryBackgroundConfig, s: np.ndarray) -> np.ndarray:
    """Resample a real geomagnetic observatory trace (data/reference/*.csv, fetched
    from the USGS Geomagnetism Program) onto the survey's along-track chainage,
    replacing the synthetic drift+wave terms with genuine measured field variation.

    The trace's native 1 Hz cadence is stretched onto chainage assuming a constant
    `survey_speed_m_per_s` -- a modelling simplification, not a claim that the real
    station was surveyed at that speed. What the detrend-honesty test in
    PLAN.md Stage 2.5 item 3 needs is the trace's genuine broadband SHAPE, which
    survives this stretch; only the real-world speed correspondence is invented.
    Unchanged by Rig-v2: `s` is now the walk's true (irregularly-sampled) along-
    track chainage rather than a uniform grid, but this function only ever
    treats `s` as an array of positions, so nothing here depends on the grid.
    """
    path = PROJECT_ROOT / cfg.csv_path
    df = pd.read_csv(path, comment="#")
    t_s = df["elapsed_s"].to_numpy(dtype=float)
    survey_t_s = s / cfg.survey_speed_m_per_s
    if survey_t_s[-1] > t_s[-1]:
        raise ValueError(
            f"{path} covers only {t_s[-1]:.0f} s but the survey needs {survey_t_s[-1]:.0f} s "
            f"at {cfg.survey_speed_m_per_s} m/s -- fetch a longer trace or slow the survey down"
        )
    background = np.column_stack(
        [np.interp(survey_t_s, t_s, df[col].to_numpy()) for col in ("x_nt", "y_nt", "z_nt")]
    )
    return background - background.mean(axis=0)


def dipole_field(r_obs: np.ndarray, r_src: np.ndarray, moment) -> np.ndarray:
    """3-axis field of a point dipole `moment` at `r_src`, observed at `r_obs` (N,3).

    Always returns the full vector, for both rig modes: the scalar rig takes
    ||.|| of the SUM of these (see module docstring on why the exact norm,
    never the linear projection, is what generates the data).
    """
    d = r_obs - r_src
    dist = np.linalg.norm(d, axis=-1, keepdims=True)
    dist = np.clip(dist, 0.05, None)
    rhat = d / dist
    m = np.asarray(moment, dtype=float)
    term = 3.0 * np.sum(rhat * m, axis=-1, keepdims=True) * rhat - m
    return COUPLING * term / dist**3


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


@dataclass
class SurveyResult:
    survey_id: str
    line_id: str
    run_id: int
    path: Path
    file_sha256: str
    content_sha256: str
    step_m: float
    n_samples: int
    chainage_start_m: float
    chainage_end_m: float
    standoff_m: float
    surveyed_at: str


def _sample_spaced_chainage(
    rng: np.random.Generator,
    low: float,
    high: float,
    half_width: float,
    placed: list[tuple[float, float]],
    margin: float = 2.0,
    max_tries: int = 2000,
) -> float:
    """Reject-sample a chainage so this feature's label window never touches an
    already-placed feature's window (plus a small margin).

    Without this, unconstrained `rng.uniform()` placement produces overlapping
    label windows once feature density rises (measured: 25% of features
    overlap at n_defects=40 on this line length, unconstrained). A SAME-KIND
    overlap (defect-defect or interference-interference) is silent data
    corruption, not cosmetic: `truth.build_truth_registry` finds physical
    sources by looking for contiguous `defect==1` (or `interference==1`) runs,
    so two overlapping defects merge into what it counts as ONE physical
    defect -- undercounting exactly the thing a denser corpus is meant to
    increase. A cross-kind overlap (defect touching interference) doesn't
    corrupt the count, but still creates row-level assignment ambiguity at the
    boundary; excluding it too keeps the corpus unambiguous everywhere.

    Girth welds do NOT go through this: they are a periodic train (_build_welds),
    not rejection-sampled, and are deliberately ALLOWED to land near/on a defect
    or interference window -- a defect sitting on a weld is a realistic scenario
    Stage B/C's `dist_to_weld_m` nuisance mask exists to handle, not corruption.
    """
    for _ in range(max_tries):
        candidate = rng.uniform(low, high)
        if all(abs(candidate - c) >= (half_width + hw + margin) for c, hw in placed):
            return candidate
    raise RuntimeError(
        f"Could not place a feature with the required label-window spacing after "
        f"{max_tries} tries -- too many features (or too wide a label window) for "
        f"this line length. Reduce n_defects/n_interference or increase length_m."
    )


# Per-type severity (dipole moment magnitude) ranges -- developer-team
# follow-up feedback: the team does not trust a SHAPE distinction between
# defect types (a point-dipole model has no principled way to fake one
# honestly, since footprint width is governed by depth/offset, and every
# defect sits at y_off_m=0), but each type DOES plausibly carry a different
# INTENSITY. Physics-inspired reasoning, not calibrated to real ROSEN data:
# dent = a sharp mechanical stress concentration, the strongest signature;
# SCC = fine, branching cracks, more diffuse, weakest; corrosion = gradual
# metal loss, moderate; weld (workmanship anomaly at a joint, distinct from
# the periodic girth-weld TRAIN in _build_welds) = the best-controlled
# class per train.py's own consequence-proxy comment, so the smallest/most
# consistent range. Replaces the old single shared 20-80 range every type
# drew from, which carried zero physical signal by construction.
DEFECT_SEVERITY_RANGES: dict[str, tuple[float, float]] = {
    "dent": (40.0, 80.0),
    "corrosion": (20.0, 70.0),
    "scc": (15.0, 50.0),
    "weld": (15.0, 45.0),
}


def _build_features(cfg: DataConfig, rng: np.random.Generator) -> list[dict]:
    """Defects and interference sources -- unchanged by Rig-v2. Girth welds are
    a structurally different (periodic, not rejection-sampled) source train,
    built separately by `_build_welds`.
    """
    types = list(DEFECT_SEVERITY_RANGES)
    feats = []
    placed: list[tuple[float, float]] = []  # (chainage_m, half_width_m), for spacing checks
    # y_off_m=0 for every defect, so r_eff=depth_m and the half-width is the
    # same constant for all of them.
    defect_half_width = cfg.label_window_scale * cfg.depth_m

    for _ in range(cfg.n_defects):
        chainage = _sample_spaced_chainage(rng, 50, cfg.length_m - 50, defect_half_width, placed)
        placed.append((chainage, defect_half_width))
        defect_type = rng.choice(types)
        feats.append(
            {
                "chainage_m": chainage,
                "y_off_m": 0.0,
                "type": defect_type,
                "severity": rng.uniform(*DEFECT_SEVERITY_RANGES[defect_type]),
                "orientation": rng.normal(0, 1, 3),
                "is_defect": True,
            }
        )
    for _ in range(cfg.n_interference):
        # Interference is a LARGE steel object -- a fence post, buried scrap, a
        # casing, an adjacent line -- not a stress-concentration zone, so its
        # magnetic moment is one to two orders of magnitude bigger. Without that
        # scale factor 1/r^3 from 3-8 m lateral puts every interference source
        # below the ~8 nT noise floor (a source at 5.5 m lateral peaks at 0.9 nT
        # for a defect-scale moment), and the "designed false-positive trap"
        # traps nothing. Scaled this way it arrives at roughly DEFECT-COMPARABLE
        # amplitude but visibly BROADER, which is the discrimination the whole
        # project turns on: shape separates it, amplitude does not.
        # The scale itself is drawn per-source, not fixed -- developer-team
        # follow-up: real interference runs "a bit weaker in most cases," and
        # its strength varies independently of distance (y_off_m's own 3-8 m
        # draw already covers distance; this varies the moment on top of it).
        y_off_m = rng.uniform(3, 8) * rng.choice([-1, 1])
        half_width = cfg.label_window_scale * float(np.hypot(cfg.depth_m, y_off_m))
        chainage = _sample_spaced_chainage(rng, 50, cfg.length_m - 50, half_width, placed)
        placed.append((chainage, half_width))
        moment_scale = rng.uniform(*cfg.interference_moment_scale_range)
        feats.append(
            {
                "chainage_m": chainage,
                "y_off_m": y_off_m,
                "type": "interference",
                "severity": rng.uniform(30, 90) * moment_scale,
                "orientation": rng.normal(0, 1, 3),
                "is_defect": False,
            }
        )
    return feats


def _build_welds(cfg: DataConfig, rng: np.random.Generator) -> list[dict]:
    """A periodic girth-weld train, one joint every `weld.pitch_m` (+/- jitter)
    along the WHOLE line -- pipe joints are a fact of construction, not a rare
    event, which is the physical error this corrects (the pre-Rig-v2 model
    drew `weld` as one of a dozen rare point-defect types, i.e. ~12 TIMES on a
    2 km line, not every ~12 m). Isotropic orientation and an inflated moment
    (5-20x a defect's, cfg.weld.moment_scale_range), same severity-draw shape
    as a defect so the two trains are comparable in scale.

    Deliberately independent of `_build_features` / `_sample_spaced_chainage`
    -- see that function's docstring for why overlap with a defect/interference
    window is allowed here, not a bug.
    """
    weld = cfg.weld
    welds = []
    i = 0
    while True:
        # Centred pitch grid (pitch/2, 3*pitch/2, ...) so the first joint isn't
        # jammed against chainage=0.
        chainage = (i + 0.5) * weld.pitch_m
        if chainage >= cfg.length_m:
            break
        jitter = rng.uniform(-weld.pitch_jitter_m, weld.pitch_jitter_m)
        scale = rng.uniform(*weld.moment_scale_range)
        welds.append(
            {
                "chainage_m": chainage + jitter,
                "orientation": rng.normal(0, 1, 3),
                "severity": rng.uniform(20, 80) * scale,
            }
        )
        i += 1
    return welds


# ---------------------------------------------------------------------------
# Rig-v2 walk model (data.rig: scalar)
# ---------------------------------------------------------------------------


def _ou_deviation(
    rng: np.random.Generator, n: int, dt_s: float, tau_s: float, sigma_stat: float
) -> np.ndarray:
    """Ornstein-Uhlenbeck deviation-from-zero process, `n` samples at spacing
    `dt_s`, mean-reversion time `tau_s`, stationary std `sigma_stat`.

    Euler-Maruyama discretisation: dx[t+1] = a*dx[t] + b*eps[t], with
    a = 1 - dt/tau, b = sigma_stat*sqrt(2*dt/tau). That recurrence is LINEAR,
    so the whole trajectory is one `scipy.signal.lfilter` call (a 1-pole IIR
    filter) instead of a Python loop over up to ~2*10^5 samples per survey.
    dx[0] = b*eps[0] (effectively zero initial deviation), so a process
    added to a nominal mean starts close to that mean, not at zero.
    """
    theta = 1.0 / tau_s
    a = 1.0 - theta * dt_s
    b = sigma_stat * np.sqrt(2.0 * theta * dt_s)
    eps = rng.normal(0.0, 1.0, n)
    return lfilter([b], [1.0, -a], eps)


def _walk_trajectory(cfg: DataConfig, rng: np.random.Generator) -> dict:
    """Simulate one human walk down the line: irregular speed -> irregular
    along-track spacing, wandering stand-off and lateral position, and mast
    sway that moves head positions (never the scalar reading itself -- |B| is
    rotation-invariant by construction, see module docstring).

    Returns arrays truncated to the first crossing of `length_m`, plus the
    3D observation positions for all three heads.
    """
    walk: WalkConfig = cfg.walk
    dt_s = 1.0 / walk.sample_rate_hz

    # Speed is clipped >= speed_min_m_per_s every step, so cumulative distance
    # after n_max samples is guaranteed >= length_m -- the walk is guaranteed
    # to reach the end of the line within n_max samples, no retry loop needed.
    n_max = int(np.ceil(cfg.length_m / walk.speed_min_m_per_s / dt_s) * 1.10) + 2

    speed = walk.speed_m_per_s + _ou_deviation(
        rng, n_max, dt_s, walk.speed_correlation_s, walk.speed_sigma_m_per_s
    )
    speed = np.clip(speed, walk.speed_min_m_per_s, walk.speed_max_m_per_s)

    chainage_true = np.cumsum(speed * dt_s)
    n = max(2, int(np.searchsorted(chainage_true, cfg.length_m)) + 1)
    n = min(n, n_max)
    chainage_true = chainage_true[:n]

    t_s = np.arange(n, dtype=np.float64) * dt_s
    # Naive constant-nominal-speed dead reckoning -- deliberately WRONG
    # (ignores the actual OU speed variation above) to stand in for what a
    # system with no odometer/GPS correction would assume. This is what Stage
    # B's registration exists to correct; see schemas.py's chainage_provisional_m.
    chainage_provisional = walk.speed_m_per_s * t_s

    standoff = walk.standoff_m + _ou_deviation(
        rng, n, dt_s, walk.standoff_correlation_m / walk.speed_m_per_s, walk.standoff_sigma_m
    )
    lateral = _ou_deviation(
        rng, n, dt_s, walk.lateral_correlation_m / walk.speed_m_per_s, walk.lateral_sigma_m
    )
    tilt_deg = _ou_deviation(
        rng, n, dt_s, walk.tilt_correlation_m / walk.speed_m_per_s, walk.tilt_sigma_deg
    )
    tilt_rad = np.radians(tilt_deg)

    obs_mid = np.column_stack([chainage_true, lateral, standoff])
    # Unit vector along the (swaying) mast axis: tilt leans it fore/aft along
    # the walk direction. At tilt_sigma_deg=5 the resulting extra vertical
    # separation between heads (spacing_m*(1-cos(tilt))) is negligible -- this
    # is squarely about moving head POSITIONS, not about changing separation.
    axis = np.column_stack([np.sin(tilt_rad), np.zeros(n), np.cos(tilt_rad)])

    return {
        "n": n,
        "t_s": t_s,
        "chainage_true_m": chainage_true,
        "chainage_provisional_m": chainage_provisional,
        "lateral_m": lateral,
        "obs_mid": obs_mid,
        "axis": axis,
    }


def _gps_track(
    cfg: DataConfig, rng: np.random.Generator, chainage_true: np.ndarray, lateral: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Lat/lon from true position, with Markov good/bad lock state (NaN through
    a dropout) and horizontal noise while locked.

    Run-length (not per-sample) simulation: a 2-state Markov chain's
    consecutive-same-state run lengths are geometrically distributed, so
    drawing segment lengths directly is exactly equivalent to a per-sample
    state machine but needs O(n_segments) draws, not O(n_samples) -- a handful
    of dropouts per survey rather than one RNG draw per row.
    """
    gps: GpsConfig = cfg.gps
    n = len(chainage_true)
    fs = cfg.walk.sample_rate_hz

    # p_bg: per-sample probability a dropout ends (bad -> good), set by the
    # mean dropout dwell time. p_gb: per-sample probability a dropout starts
    # (good -> bad), backed out from the steady-state occupancy so that
    # dropout_rate really is "the fraction of the survey spent unlocked".
    p_bg = min(1.0, 1.0 / max(gps.mean_gap_s * fs, 1.0))
    p_gb = min(1.0, gps.dropout_rate / max(1.0 - gps.dropout_rate, 1e-9) * p_bg)
    p_gb = max(p_gb, 1e-9)
    p_bg = max(p_bg, 1e-9)

    bad = np.zeros(n, dtype=bool)
    i = 0
    state_bad = False
    while i < n:
        p = p_bg if state_bad else p_gb
        length = max(1, int(rng.geometric(p)))
        j = min(n, i + length)
        if state_bad:
            bad[i:j] = True
        i = j
        state_bad = not state_bad

    lat0, lon0 = cfg.geo.lat0, cfg.geo.lon0
    brg = np.radians(cfg.geo.bearing_deg)
    # True ground position: along-track chainage plus lateral wander
    # perpendicular to the survey bearing.
    north = chainage_true * np.cos(brg) - lateral * np.sin(brg)
    east = chainage_true * np.sin(brg) + lateral * np.cos(brg)
    north = north + rng.normal(0, gps.sigma_m, n)
    east = east + rng.normal(0, gps.sigma_m, n)

    lat = lat0 + north / 111_320.0
    lon = lon0 + east / (111_320.0 * np.cos(np.radians(lat0)))
    lat = np.where(bad, np.nan, lat)
    lon = np.where(bad, np.nan, lon)
    return lat, lon


def _apply_sensor(rng: np.random.Generator, b_true: np.ndarray, sensor: SensorConfig, n: int) -> np.ndarray:
    """One head's imperfect reading from its exact total-field magnitude:
    fixed-per-survey multiplicative gain error + additive offset, independent
    per-row noise, then ADC quantization. Gain MISMATCH between heads (not
    gain error itself) is the point -- see SensorConfig's docstring.
    """
    gain = 1.0 + rng.normal(0.0, sensor.gain_sigma)
    offset = rng.normal(0.0, sensor.offset_nt)
    reading = b_true * gain + offset
    reading = reading + rng.normal(0.0, sensor.noise_nt, size=n)

    full_scale_nt = sensor.full_scale_ut * 1000.0
    lsb_nt = 2.0 * full_scale_nt / (2**sensor.adc_bits)
    reading = np.round(reading / lsb_nt) * lsb_nt
    # A magnitude reading only ever rails at the TOP of the ADC's +/- range.
    return np.clip(reading, 0.0, full_scale_nt)


def _make_run_scalar(
    line_id: str,
    run_id: int,
    cfg: DataConfig,
    features: list[dict],
    welds: list[dict],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Rig-v2: one survey run for the real 3-head scalar rig. See module
    docstring for the total-field physics this implements exactly (never the
    linear projection).
    """
    array: ArrayConfig = cfg.array
    if array.orientation != "vertical":
        raise NotImplementedError(
            f"array.orientation={array.orientation!r} -- only 'vertical' is implemented. "
            "This is one of the Rig-v2 open questions for ROSEN (mast vertical vs "
            "across-track); picking a physics path for 'horizontal' without confirming "
            "it would silently answer a question that hasn't been asked yet."
        )
    if array.n_heads != 3:
        raise NotImplementedError(
            f"array.n_heads={array.n_heads} -- the scalar rig's schema (b_lo/b_mid/b_hi) "
            "is fixed to the real instrument's 3 heads. Use rig: vector for a "
            "configurable 1/2-head vector output."
        )

    walk = _walk_trajectory(cfg, rng)
    n = walk["n"]
    s_true = walk["chainage_true_m"]
    obs_mid = walk["obs_mid"]
    axis = walk["axis"]
    obs_lo = obs_mid - array.spacing_m * axis
    obs_hi = obs_mid + array.spacing_m * axis

    lat, lon = _gps_track(cfg, rng, s_true, walk["lateral_m"])

    base = np.array(cfg.background_nT, dtype=float)
    b0hat = unit(base)
    if cfg.observatory_background.enabled:
        # Stage 2.5 item 3: a real trace, not a synthetic sinusoid -- see
        # _load_observatory_background's docstring for the along-track mapping.
        background_variation = _load_observatory_background(cfg.observatory_background, s_true)
    else:
        drift = np.linspace(0, 1, n)[:, None] * rng.normal(0, 40, 3)
        wave = (
            np.sin(2 * np.pi * s_true / cfg.length_m * rng.uniform(1, 3))[:, None]
            * rng.normal(0, 25, 3)
        )
        background_variation = drift + wave
    B_bg = base + background_variation

    # Per-head vertical background gradient: without this the background seen
    # by all three heads is identical and common-/gradient-mode rejection is
    # perfect by construction, which is not real (ArrayConfig docstring). Mid
    # is the reference level (no extra term); lo/hi get +/- spacing_m worth.
    geology_dir = rng.normal(0, 1, 3)
    grad_shape = (
        np.linspace(0, 1, n)[:, None]
        + np.sin(2 * np.pi * s_true / cfg.length_m * rng.uniform(1, 3))[:, None]
    )

    def _grad_offset(z_off: float) -> np.ndarray:
        main = array.main_field_gradient_nT_per_m * z_off
        geo = array.geology_gradient_scale_nT_per_m * z_off * grad_shape * geology_dir
        return main + geo

    B_lo = B_bg + _grad_offset(-array.spacing_m)
    B_mid = B_bg.copy()
    B_hi = B_bg + _grad_offset(+array.spacing_m)

    growth = cfg.growth**run_id
    sources: list[tuple[np.ndarray, np.ndarray]] = []
    for f in features:
        src = np.array([f["chainage_m"], f["y_off_m"], -cfg.depth_m])
        m_dir = unit(f["orientation"])
        if f["is_defect"] and cfg.stress_polarity == "positive":
            # Canonicalise so the anomaly always ENHANCES the field: probe the
            # field this moment produces directly above the source and flip if
            # its projection onto B_hat0 (the physically measured quantity --
            # module docstring) would come out negative. `random` (the honest
            # default) leaves the isotropic orientation draw's sign exactly as
            # physics gives it -- enhancing or degrading with equal probability.
            probe_obs = np.array([[f["chainage_m"], f["y_off_m"], 0.0]])
            probe_src = src
            probe_b = dipole_field(probe_obs, probe_src, m_dir)[0]
            if np.dot(probe_b, b0hat) < 0:
                m_dir = -m_dir
        g = growth if f["is_defect"] else 1.0
        sources.append((src, m_dir * f["severity"] * g))
    for w in welds:
        src = np.array([w["chainage_m"], 0.0, -cfg.depth_m])
        sources.append((src, unit(w["orientation"]) * w["severity"]))

    for src, moment in sources:
        B_lo += dipole_field(obs_lo, src, moment)
        B_mid += dipole_field(obs_mid, src, moment)
        B_hi += dipole_field(obs_hi, src, moment)

    sensor = cfg.sensor
    b_lo = _apply_sensor(rng, np.linalg.norm(B_lo, axis=1), sensor, n)
    b_mid = _apply_sensor(rng, np.linalg.norm(B_mid, axis=1), sensor, n)
    b_hi = _apply_sensor(rng, np.linalg.norm(B_hi, axis=1), sensor, n)

    sample_idx = np.arange(n, dtype=np.int64)

    label = np.zeros(n, dtype=int)
    dtype_arr = np.array(["none"] * n, dtype=object)
    sev = np.full(n, np.nan)
    interference = np.zeros(n, dtype=int)
    if features:
        # Half-width scales with r_eff, the actual dipole source-to-sensor
        # distance -- see the pre-Rig-v2 rationale, unchanged. Windows can
        # overlap, so each row goes to its NEAREST feature.
        centers = np.array([f["chainage_m"] for f in features])
        half_widths = np.array(
            [cfg.label_window_scale * np.hypot(cfg.depth_m, f["y_off_m"]) for f in features]
        )
        dist = np.abs(s_true[:, None] - centers[None, :])
        within = dist <= half_widths[None, :]
        nearest = np.argmin(np.where(within, dist, np.inf), axis=1)
        has_any = within.any(axis=1)
        for idx, f in enumerate(features):
            mask = has_any & (nearest == idx)
            if not f["is_defect"]:
                interference[mask] = 1
                continue
            label[mask] = 1
            dtype_arr[mask] = f["type"]
            sev[mask] = f["severity"] * growth

    girth_weld = np.zeros(n, dtype=int)
    if welds:
        # Independent of the defect/interference window above -- a row CAN be
        # both defect==1 and girth_weld==1 (see _build_welds's docstring).
        weld_centers = np.array([w["chainage_m"] for w in welds])
        weld_half_width = cfg.label_window_scale * cfg.depth_m
        weld_dist = np.abs(s_true[:, None] - weld_centers[None, :])
        girth_weld = (weld_dist <= weld_half_width).any(axis=1).astype(int)

    df = pd.DataFrame(
        {
            "sample_idx": sample_idx,
            "t_s": walk["t_s"],
            "chainage_true_m": s_true,
            "chainage_provisional_m": walk["chainage_provisional_m"],
            "lat": lat,
            "lon": lon,
            "b_lo_nt": b_lo.astype(np.float64),
            "b_mid_nt": b_mid.astype(np.float64),
            "b_hi_nt": b_hi.astype(np.float64),
            "defect": label,
            "defect_type": dtype_arr,
            "severity_smys": sev,
            "interference": interference,
            "girth_weld": girth_weld,
        }
    )
    df["line_id"] = line_id
    df["run_id"] = run_id
    return df


# ---------------------------------------------------------------------------
# Legacy vector rig (data.rig: vector) -- preserved unchanged as the Stage D
# ablation's reference arm. See module docstring.
# ---------------------------------------------------------------------------


def _make_run_vector(
    line_id: str, run_id: int, cfg: DataConfig, features: list[dict], rng: np.random.Generator
) -> pd.DataFrame:
    """The ORIGINAL pre-Rig-v2 generator, unchanged in behaviour: a uniform
    step_m grid, one 3-axis vector head (+ an optional second at
    array.spacing_m if array.n_heads==2, replacing the old
    `gradiometer.enabled`/`baseline_m`). No walk, GPS dropout, girth welds or
    sensor imperfections -- those are Rig-v2-only physics with no vector-rig
    analogue in this reference arm.
    """
    if cfg.array.n_heads not in (1, 2):
        raise NotImplementedError(
            f"array.n_heads={cfg.array.n_heads} -- rig: vector only implements the "
            "original single-head (1) or dual-head/gradiometer (2) output. Use "
            "rig: scalar (n_heads=3, fixed) for the real instrument."
        )

    n = int(cfg.length_m / cfg.step_m)
    sample_idx = np.arange(n, dtype=np.int64)
    s = sample_idx * cfg.step_m  # chainage_m, derived
    obs = np.column_stack([s, np.zeros(n), np.zeros(n)])

    base = np.array(cfg.background_nT, dtype=float)
    if cfg.observatory_background.enabled:
        background_variation = _load_observatory_background(cfg.observatory_background, s)
    else:
        drift = np.linspace(0, 1, n)[:, None] * rng.normal(0, 40, 3)
        wave = np.sin(2 * np.pi * s / cfg.length_m * rng.uniform(1, 3))[:, None] * rng.normal(0, 25, 3)
        background_variation = drift + wave
    B = base + background_variation

    for f in features:
        src = np.array([f["chainage_m"], f["y_off_m"], -cfg.depth_m])
        growth = cfg.growth**run_id if f["is_defect"] else 1.0
        moment = unit(f["orientation"]) * f["severity"] * growth
        B += dipole_field(obs, src, moment)

    B += rng.normal(0, cfg.noise_nT, size=B.shape)

    label = np.zeros(n, dtype=int)
    dtype_arr = np.array(["none"] * n, dtype=object)
    sev = np.full(n, np.nan)
    interference = np.zeros(n, dtype=int)

    centers = np.array([f["chainage_m"] for f in features])
    half_widths = np.array(
        [cfg.label_window_scale * np.hypot(cfg.depth_m, f["y_off_m"]) for f in features]
    )
    dist = np.abs(s[:, None] - centers[None, :])
    within = dist <= half_widths[None, :]
    nearest = np.argmin(np.where(within, dist, np.inf), axis=1)
    has_any = within.any(axis=1)

    for idx, f in enumerate(features):
        mask = has_any & (nearest == idx)
        if not f["is_defect"]:
            interference[mask] = 1
            continue
        label[mask] = 1
        dtype_arr[mask] = f["type"]
        sev[mask] = f["severity"] * cfg.growth**run_id

    lat0, lon0 = cfg.geo.lat0, cfg.geo.lon0
    brg = np.radians(cfg.geo.bearing_deg)
    lat = lat0 + (s * np.cos(brg)) / 111_320.0
    lon = lon0 + (s * np.sin(brg)) / (111_320.0 * np.cos(np.radians(lat0)))

    df = pd.DataFrame(
        {
            "sample_idx": sample_idx,
            "chainage_m": s,
            "lat": lat.astype(np.float64),
            "lon": lon.astype(np.float64),
            "bx_nt": B[:, 0].astype(np.float64),
            "by_nt": B[:, 1].astype(np.float64),
            "bz_nt": B[:, 2].astype(np.float64),
            "bx2_nt": np.nan,
            "by2_nt": np.nan,
            "bz2_nt": np.nan,
            "defect": label,
            "defect_type": dtype_arr,
            "severity_smys": sev,
            "interference": interference,
        }
    )

    if cfg.array.n_heads == 2:
        baseline_m = cfg.array.spacing_m
        obs2 = np.column_stack([s, np.zeros(n), np.full(n, baseline_m)])
        # The upper head sees a slightly different background, not the identical
        # one -- otherwise common-mode rejection is perfect by construction. The
        # main field's vertical gradient (~0.02 nT/m) is a near-constant offset;
        # geology contributes more AND varies along the line, so it gets its own
        # spatially-varying term (independent random draw, same drift+wave shape
        # as the base background) scaled by the baseline so a bigger baseline
        # sees a bigger difference, as it physically should.
        main_field_grad = cfg.array.main_field_gradient_nT_per_m * baseline_m
        grad_shape = (
            np.linspace(0, 1, n)[:, None]
            + np.sin(2 * np.pi * s / cfg.length_m * rng.uniform(1, 3))[:, None]
        )
        geology_grad = (
            cfg.array.geology_gradient_scale_nT_per_m
            * baseline_m
            * grad_shape
            * rng.normal(0, 1, 3)
        )
        B2 = base + background_variation + main_field_grad + geology_grad
        for f in features:
            src = np.array([f["chainage_m"], f["y_off_m"], -cfg.depth_m])
            growth = cfg.growth**run_id if f["is_defect"] else 1.0
            moment = unit(f["orientation"]) * f["severity"] * growth
            B2 += dipole_field(obs2, src, moment)
        B2 += rng.normal(0, cfg.noise_nT, size=B2.shape)
        df["bx2_nt"] = B2[:, 0]
        df["by2_nt"] = B2[:, 1]
        df["bz2_nt"] = B2[:, 2]

    df["line_id"] = line_id
    df["run_id"] = run_id
    return df


def _make_run(
    line_id: str,
    run_id: int,
    cfg: DataConfig,
    features: list[dict],
    welds: list[dict],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Dispatch on cfg.rig. See module docstring."""
    if cfg.rig == "vector":
        return _make_run_vector(line_id, run_id, cfg, features, rng)
    return _make_run_scalar(line_id, run_id, cfg, features, welds, rng)


def find_raw_survey_path(raw_dir: Path, survey_id: str) -> Path:
    """survey_id 'LINE000_R2' -> data/raw/line_id=LINE000/run_id=2/survey.parquet"""
    line_id, run_part = survey_id.rsplit("_R", 1)
    return Path(raw_dir) / f"line_id={line_id}" / f"run_id={run_part}" / "survey.parquet"


def load_survey_result(path: Path, step_m: float, standoff_m: float) -> SurveyResult:
    """Reconstruct a SurveyResult (hashes + metadata) from an already-written raw
    Parquet file. Used by both the CLI `ingest` command and the Dagster asset --
    one code path for "what does an already-generated survey look like".

    Reads whichever chainage column the file actually has: `chainage_m` for a
    `rig: vector` file (the legacy grid), `chainage_true_m` for a `rig: scalar`
    file (there is no `chainage_m` in that schema -- see schemas.py).

    `surveyed_at` used to be hardcoded "unknown" here. That string sorts AFTER
    every real ISO-8601 timestamp lexicographically, and
    `features.load_feature_corpus`'s as-of filter is a plain string comparison
    (`str(meta["surveyed_at"]) > str(as_of)`) -- so every survey ingested through
    this path (i.e. every survey ingested via the real CLI or Dagster, not
    `generate_all`'s own in-memory SurveyResult) was silently excluded from every
    training corpus. Point-in-time correctness (SKILL invariant #12) doesn't help
    if the timestamp itself is a lie. There is no acquisition-time column in the
    raw file to recover the true value from, so the file's own mtime is used as
    the least-bad stand-in: raw data is immutable once written (another
    invariant), so mtime is a stable, monotonic proxy for "when this survey was
    produced" in a demonstrator with no separate acquisition-time channel.
    """
    df = pd.read_parquet(path)
    line_id = str(df["line_id"].iloc[0])
    run_id = int(df["run_id"].iloc[0])
    surveyed_at = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.UTC).isoformat()
    chainage_col = "chainage_m" if "chainage_m" in df.columns else "chainage_true_m"
    return SurveyResult(
        survey_id=f"{line_id}_R{run_id}",
        line_id=line_id,
        run_id=run_id,
        path=path,
        file_sha256=file_sha256(path),
        content_sha256=content_sha256(df),
        step_m=step_m,
        n_samples=len(df),
        chainage_start_m=float(df[chainage_col].min()),
        chainage_end_m=float(df[chainage_col].max()),
        standoff_m=standoff_m,
        surveyed_at=surveyed_at,
    )


def generate_all(cfg: DataConfig, raw_dir: Path, seed: int) -> list[SurveyResult]:
    """Generate cfg.n_lines lines x cfg.n_runs surveys, write one Parquet file each."""
    raw_dir = Path(raw_dir)
    results: list[SurveyResult] = []
    surveyed_at = dt.datetime.now(dt.UTC).isoformat()
    write_cols = RAW_COLUMNS_VECTOR if cfg.rig == "vector" else RAW_COLUMNS_SCALAR
    chainage_col = "chainage_m" if cfg.rig == "vector" else "chainage_true_m"
    # Nominal mean spacing, for metadata only under rig: scalar (the actual row
    # spacing is irregular by design -- see DataConfig.step_m's docstring).
    reported_step_m = cfg.step_m if cfg.rig == "vector" else cfg.walk.speed_m_per_s / cfg.walk.sample_rate_hz
    reported_standoff_m = cfg.depth_m if cfg.rig == "vector" else cfg.walk.standoff_m

    for line_idx in range(cfg.n_lines):
        line_id = f"LINE{line_idx:03d}"
        line_rng = np.random.default_rng(seed + line_idx)
        features = _build_features(cfg, line_rng)
        # weld.enabled=False skips _build_welds ENTIRELY (not just its field
        # contribution), so its RNG draws are never consumed and the whole
        # downstream trajectory shifts -- fine, it is a different corpus by
        # construction. The default (True) path is unchanged, byte for byte.
        build_welds = cfg.rig != "vector" and cfg.weld.enabled
        welds = _build_welds(cfg, line_rng) if build_welds else []
        for run_id in range(cfg.n_runs):
            df = _make_run(line_id, run_id, cfg, features, welds, line_rng)
            survey_id = f"{line_id}_R{run_id}"

            out_dir = raw_dir / f"line_id={line_id}" / f"run_id={run_id}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "survey.parquet"

            df[write_cols].sort_values("sample_idx").to_parquet(
                out_path, index=False, compression="zstd"
            )

            results.append(
                SurveyResult(
                    survey_id=survey_id,
                    line_id=line_id,
                    run_id=run_id,
                    path=out_path,
                    file_sha256=file_sha256(out_path),
                    content_sha256=content_sha256(df),
                    step_m=reported_step_m,
                    n_samples=len(df),
                    chainage_start_m=float(df[chainage_col].min()),
                    chainage_end_m=float(df[chainage_col].max()),
                    standoff_m=reported_standoff_m,
                    surveyed_at=surveyed_at,
                )
            )
    return results
