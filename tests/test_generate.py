from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from lsm.config import SensorConfig
from lsm.generate import (
    _apply_sensor,
    _build_features,
    dipole_field,
    generate_all,
    unit,
)


def test_generate_is_deterministic_given_same_seed(tiny_cfg, tmp_path):
    r1 = generate_all(tiny_cfg.base.data, tmp_path / "raw1", seed=42)
    r2 = generate_all(tiny_cfg.base.data, tmp_path / "raw2", seed=42)
    assert [r.content_sha256 for r in r1] == [r.content_sha256 for r in r2]


def test_generate_sample_idx_is_dense_integer_key(tiny_cfg, tmp_path):
    results = generate_all(tiny_cfg.base.data, tmp_path / "raw", seed=42)
    df = pd.read_parquet(results[0].path)
    assert df["sample_idx"].dtype.kind in "iu"
    assert list(df["sample_idx"]) == list(range(len(df)))
    # chainage_m is GONE from raw (Rig-v2): the walk is irregular, so there is
    # no single step_m that derives it. sample_idx stays the dense, monotonic,
    # ONLY key -- t_s (elapsed time) is the regular axis instead, and
    # chainage_true_m (truth) is monotonically non-decreasing because the
    # walker never walks backwards (speed is clipped > 0).
    assert "chainage_m" not in df.columns
    assert (np.diff(df["t_s"].to_numpy()) > 0).all()
    assert (np.diff(df["chainage_true_m"].to_numpy()) >= 0).all()


def test_generate_writes_one_parquet_per_survey(tiny_cfg, tmp_path):
    results = generate_all(tiny_cfg.base.data, tmp_path / "raw", seed=42)
    n_expected = tiny_cfg.base.data.n_lines * tiny_cfg.base.data.n_runs
    assert len(results) == n_expected
    for r in results:
        assert r.path.exists()
        assert r.path.name == "survey.parquet"


def test_defect_signature_is_detectable_against_background(cfg, tmp_path):
    """Cross-check against the physics claim in PLAN.md: defect residual small
    but learnable relative to the raw ~48,800 nT total field. Uses the middle
    head's total-field reading, b_mid_nt -- the scalar rig's direct analogue
    of the pre-Rig-v2 bx_nt.
    """
    cfg.base.data.n_lines = 1
    cfg.base.data.n_runs = 1
    cfg.base.data.walk.sample_rate_hz = 20.0  # length_m/n_defects left at base.yaml's defaults
    results = generate_all(cfg.base.data, tmp_path / "raw", seed=42)
    df = pd.read_parquet(results[0].path)

    defect_rows = df[df["defect"] == 1]
    assert len(defect_rows) > 0

    # crude "residual" proxy: deviation from a rolling median (no real detrend
    # yet). Window sized to ~40 m via the walk's own mean spacing (irregular
    # per-row, but ~1/samples_per_m on average) -- the pre-Rig-v2 test used a
    # fixed sample count because spacing was a uniform grid; that assumption
    # no longer holds, so the window is derived from the walk config instead.
    samples_per_m = cfg.base.data.walk.sample_rate_hz / cfg.base.data.walk.speed_m_per_s
    win = int(40 * samples_per_m)
    win += 1 if win % 2 == 0 else 0
    med = df["b_mid_nt"].rolling(win, center=True, min_periods=1).median()
    resid = (df["b_mid_nt"] - med).abs()
    defect_resid = resid[df["defect"] == 1].median()
    background_resid = resid[df["defect"] == 0].median()

    assert defect_resid > background_resid
    # raw field is ~48,800 nT; defect signal must be a small fraction of it
    assert defect_resid < 0.01 * df["b_mid_nt"].median()


def _label_windows(cfg, features):
    """[(start, end, is_defect)] label windows for a features list, matching
    generate.py's own half-width formula."""
    windows = []
    for f in features:
        r_eff = float(np.hypot(cfg.depth_m, f["y_off_m"]))
        hw = cfg.label_window_scale * r_eff
        windows.append((f["chainage_m"] - hw, f["chainage_m"] + hw, f["is_defect"]))
    return sorted(windows)


def test_build_features_never_places_overlapping_label_windows(cfg):
    """Unconstrained rng.uniform() placement silently corrupted the truth
    registry at higher defect density: two overlapping same-kind windows
    merge into ONE contiguous region, undercounting physical defects. This is
    the regression test for _sample_spaced_chainage. Unaffected by Rig-v2:
    defect/interference placement is unchanged (only girth welds, a separate
    periodic train, are new -- see _build_welds).
    """
    cfg.base.data.n_defects = 60
    cfg.base.data.n_interference = 16
    rng = np.random.default_rng(cfg.seed)
    features = _build_features(cfg.base.data, rng)

    windows = _label_windows(cfg.base.data, features)
    for (_, hi, _), (lo2, _, _) in itertools.pairwise(windows):
        assert hi <= lo2, "two label windows overlap"


def _count_contiguous_runs(flag: np.ndarray) -> int:
    """How many contiguous runs of 1 in a 0/1 array -- the same physical-
    source counting `truth.build_truth_registry` does, reimplemented locally
    here so this test doesn't depend on truth.py's raw-schema assumptions
    (a `chainage_m` column on the raw frame). truth.py is downstream/Stage-C
    territory that Rig-v2's raw contract change deliberately breaks (raw no
    longer carries a physically-final chainage_m -- see schemas.py); fixing
    truth.py itself is out of Stage A's scope.
    """
    edges = np.flatnonzero(np.diff(np.concatenate([[0], flag.astype(int), [0]])))
    return len(edges) // 2


def test_build_features_defect_count_survives_into_generated_labels(cfg, tmp_path):
    """End-to-end proof the spacing fix does its job: the generated raw survey
    has exactly as many contiguous defect/interference regions as were
    requested, not fewer.
    """
    cfg.base.data.n_lines = 1
    cfg.base.data.n_runs = 1
    cfg.base.data.n_defects = 60
    cfg.base.data.n_interference = 16
    cfg.base.data.walk.sample_rate_hz = 20.0
    results = generate_all(cfg.base.data, tmp_path / "raw", seed=cfg.seed)

    df = pd.read_parquet(results[0].path)
    assert _count_contiguous_runs(df["defect"].to_numpy()) == 60
    assert _count_contiguous_runs(df["interference"].to_numpy()) == 16


def test_build_features_raises_loudly_when_too_dense_to_place(cfg):
    """Too many features for the line length must fail loudly, not silently
    corrupt spacing by giving up the constraint.
    """
    cfg.base.data.n_defects = 80
    cfg.base.data.n_interference = 20
    rng = np.random.default_rng(cfg.seed)
    with pytest.raises(RuntimeError):
        _build_features(cfg.base.data, rng)


# ---------------------------------------------------------------------------
# Rig-v2 physics tests (Stage A5): the total-field-anomaly claims the module
# docstring makes are asserted here, not just claimed in prose.
# ---------------------------------------------------------------------------


def test_total_field_projection_matches_exact_norm_to_subnT():
    """|B0+dB| - |B0| ~= dB . B_hat0 to < 0.01 nT at anomaly-scale amplitudes.
    generate.py always computes the EXACT norm (never this linear
    projection) to produce b_lo/b_mid/b_hi -- this test is what makes that
    choice meaningful: the approximation is a property of the generated data,
    not a modelling shortcut baked into how it was generated.
    """
    rng = np.random.default_rng(0)
    B0 = np.array([19000.0, 1000.0, 45000.0])
    b0hat = unit(B0)
    src = np.array([0.0, 0.0, -1.5])
    # 3-6 m sensor-to-defect distance with defect-scale severity (20-80) keeps
    # |dB| in the tens-of-nT regime the approximation is valid for -- the
    # worst case (moment exactly on-axis with the observer) at dist=3 m,
    # severity=80 gives |dB| ~= 6*80/3**3 ~= 17.8 nT, well inside the range
    # where the second-order Taylor term (~|dB|**2/(2*|B0|)) stays sub-0.01 nT.
    # Much closer/stronger encounters exist physically but leave this regime
    # -- that is real physics (the approximation genuinely degrades in the
    # near field), not something this test should paper over.
    for _ in range(50):
        direction = unit(rng.normal(0, 1, 3))
        dist = rng.uniform(3.0, 6.0)
        obs = (src + dist * direction)[None, :]
        moment = unit(rng.normal(0, 1, 3)) * rng.uniform(20, 80)
        dB = dipole_field(obs, src, moment)[0]
        exact = float(np.linalg.norm(B0 + dB) - np.linalg.norm(B0))
        approx = float(np.dot(dB, b0hat))
        assert abs(exact - approx) < 0.01


def test_moment_perpendicular_to_ambient_field_produces_near_null_anomaly():
    """A dipole whose moment is near-perpendicular to B_hat0 is nearly
    invisible to a total-field sensor -- probability of detection genuinely
    varies with defect orientation, not a bug.

    Constructed exactly rather than approximately: on-axis (observation point
    along the moment's own direction), a dipole's field is exactly PARALLEL
    to its moment (dipole_field's `term` reduces to `2*|m|*m_hat` when
    `r_hat == m_hat`), so a moment exactly perpendicular to B_hat0 gives a
    field there with EXACTLY zero projection onto B_hat0 -- the "invisible"
    case is the geometry, not a coincidence of a near-zero draw.
    """
    B0 = np.array([19000.0, 1000.0, 45000.0])
    b0hat = unit(B0)
    arbitrary = np.array([1.0, 0.3, -0.2])
    m_hat = unit(arbitrary - np.dot(arbitrary, b0hat) * b0hat)  # exactly perp to b0hat
    assert abs(np.dot(m_hat, b0hat)) < 1e-12

    src = np.array([0.0, 0.0, -1.5])
    # 2.5 m out keeps |dB| ~19 nT -- realistic defect-encounter scale, where
    # the near-null claim is physically meaningful rather than trivially true
    # of an already-vanishing field.
    obs = (src + 2.5 * m_hat)[None, :]  # on-axis
    moment = m_hat * 50.0
    dB = dipole_field(obs, src, moment)[0]
    assert abs(np.dot(dB, b0hat)) < 1e-9  # exactly perpendicular by construction

    anomaly = float(np.linalg.norm(B0 + dB) - np.linalg.norm(B0))
    assert abs(anomaly) < 0.01  # near-null total-field READING, not just a null projection

    # Contrast: a generic (non-perpendicular) moment at a comparable distance
    # is clearly visible -- the null above is the geometry, not "everything
    # this small is invisible".
    generic = unit(np.array([1.0, 1.0, 1.0])) * 50.0
    dB_generic = dipole_field(np.array([[0.0, 0.0, 0.0]]), src, generic)[0]
    anomaly_generic = abs(float(np.linalg.norm(B0 + dB_generic) - np.linalg.norm(B0)))
    assert anomaly_generic > 1.0


def test_second_difference_cancels_linear_gradient_first_difference_does_not():
    """Inject a background that varies LINEARLY with height across the 3-head
    array (no defect at all) -- g2 = (b_hi + b_lo - 2*b_mid) must be many
    orders of magnitude smaller than g1 = (b_hi - b_lo), which correctly
    RECOVERS the linear gradient's projection instead of cancelling it (that
    is the value of a first-difference gradiometer: it MEASURES a linear
    gradient, it does not remove it).

    Not literally exact machine-precision zero: g2 measures curvature in the
    MEASURED quantity |B(z)|, and |B(z)| has a tiny intrinsic curvature term
    even when the underlying vector field B(z) is exactly linear in z (the
    magnitude of a linear function is not itself linear). That curvature term
    is ~1e-4 nT here -- utterly negligible next to g1's ~9 nT/m and the ~5 nT
    sensor noise floor, which is the honest form of "cancels to near machine
    precision".
    """
    B0 = np.array([19000.0, 1000.0, 45000.0])
    grad = np.array([0.02, -0.01, 10.0])  # nT/m, spanning the configured gradient scales
    spacing = 0.5

    def f(z: float) -> float:
        return float(np.linalg.norm(B0 + grad * z))

    b_lo, b_mid, b_hi = f(-spacing), f(0.0), f(spacing)
    g1 = (b_hi - b_lo) / (2 * spacing)
    g2 = (b_hi + b_lo - 2 * b_mid) / spacing**2

    assert abs(g1 - float(np.dot(grad, unit(B0)))) < 1e-6  # g1 recovers the projected gradient
    assert abs(g2) < 1e-3
    assert abs(g1) > 1.0
    assert abs(g2) < 1e-4 * abs(g1)  # several orders of magnitude smaller


def test_weld_pitch_is_recoverable_by_autocorrelation(tiny_cfg, tmp_path):
    """Girth welds are a periodic dipole train (WeldConfig.pitch_m) -- their
    spacing shows up as a peak in the along-track autocorrelation of the
    DETRENDED |r| ENVELOPE, exactly as the Stage B plan specifies (not the
    raw signed residual): each joint's orientation is isotropic and
    arbitrary, so consecutive welds' anomalies have uncorrelated SIGN as well
    as shape -- autocorrelating the raw signed signal lets positive/negative
    lobes partially cancel at lag=pitch (measured: peak lands at ~8.5 m, not
    ~12.2 m). Autocorrelating the envelope |signal| removes the sign and
    leaves only the (unsigned) periodic ENERGY structure, recovering the
    pitch cleanly -- this is exactly what Stage B's weld-comb detector
    relies on (periodicity, not a fixed amplitude/polarity template).
    """
    cfg = tiny_cfg
    cfg.base.data.length_m = 500.0
    cfg.base.data.n_defects = 0
    cfg.base.data.n_interference = 0
    cfg.base.data.weld.pitch_m = 12.2
    cfg.base.data.weld.pitch_jitter_m = 0.15
    cfg.base.data.walk.sample_rate_hz = 20.0  # dense enough to resolve a 12 m pitch

    results = generate_all(cfg.base.data, tmp_path / "raw", seed=cfg.seed)
    df = pd.read_parquet(results[0].path)
    assert df["girth_weld"].sum() > 0

    # Resample onto a uniform CHAINAGE grid before autocorrelating -- the raw
    # rows are uniform in TIME, not distance, by design (irregular walk speed).
    grid = np.arange(0.0, cfg.base.data.length_m, 0.1)
    b_mid = df["b_mid_nt"].to_numpy()
    resid = np.interp(grid, df["chainage_true_m"].to_numpy(), b_mid - np.median(b_mid))
    envelope = np.abs(resid)
    envelope = envelope - envelope.mean()

    ac = np.correlate(envelope, envelope, mode="full")
    ac = ac[len(ac) // 2 :]  # lag >= 0
    lags_m = np.arange(len(ac)) * 0.1
    search = (lags_m > 6.0) & (lags_m < 20.0)  # skip the trivial lag=0 peak
    peak_lag = float(lags_m[search][np.argmax(ac[search])])

    assert abs(peak_lag - cfg.base.data.weld.pitch_m) < 1.0


def test_gain_mismatch_leaves_the_predicted_common_mode_residual():
    """SensorConfig's headline claim, checked exactly against _apply_sensor's
    own arithmetic: a 0.2% gain MISMATCH between two heads leaves
    ~b_true*0.002 nT of uncancelled common-mode signal -- ~100 nT against a
    ~48,800 nT background, dwarfing a ~25 nT defect anomaly. Offset and noise
    are pinned to zero via a fixed-deviation stub RNG so gain is the only
    thing that differs between the two heads.
    """

    class _FixedDeviationRng:
        """Returns a fixed deviation regardless of the requested sigma, so
        this test checks _apply_sensor's arithmetic exactly rather than the
        statistics of real random draws."""

        def __init__(self, gain_dev: float, offset_val: float, noise_val: float = 0.0):
            self._scalars = [gain_dev, offset_val]
            self._i = 0
            self._noise_val = noise_val

        def normal(self, loc, scale, size=None):
            if size is None:
                v = self._scalars[self._i]
                self._i += 1
                return v
            return np.full(size, self._noise_val)

    sensor = SensorConfig(gain_sigma=0.002, offset_nt=2.0, adc_bits=24, full_scale_ut=100.0, noise_nt=5.0)
    n = 10
    b_true = np.full(n, 48_800.0)

    b_lo = _apply_sensor(_FixedDeviationRng(0.0, 0.0), b_true, sensor, n)
    b_hi = _apply_sensor(_FixedDeviationRng(0.002, 0.0), b_true, sensor, n)

    residual = float(np.mean(b_hi - b_lo))
    expected = 48_800.0 * 0.002  # 97.6 nT

    # ADC quantization (LSB ~0.012 nT at 24-bit) is the only source of
    # deviation from the exact gain arithmetic here.
    assert abs(residual - expected) < 0.1
    assert 90.0 < residual < 110.0  # matches SensorConfig's own "~100 nT" claim
