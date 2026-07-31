"""
Synthetic LSM survey generator. Physics-inspired, not physically validated:
a buried pipeline modelled as a line of magnetic dipoles ("defects"), sensed by
a magnetometer at stand-off height over a slowly drifting geomagnetic background.

Adapted from the original generate_lsm_data.py with two structural changes
required by the data contract (references/data-contract.md):
  1. The physical key is the integer `sample_idx`, not the float `chainage_m`.
     chainage_m = sample_idx * step_m is derived and written for convenience only.
  2. Output is one immutable Parquet file per survey (zstd, sorted by sample_idx),
     not a single combined CSV -- this is what ingest.py treats as "raw".

Run: python -m lsm generate
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lsm.config import PROJECT_ROOT, DataConfig, ObservatoryBackgroundConfig
from lsm.hashing import content_sha256, file_sha256

COUPLING = 3.0


def _load_observatory_background(cfg: ObservatoryBackgroundConfig, s: np.ndarray) -> np.ndarray:
    """Resample a real geomagnetic observatory trace (data/reference/*.csv, fetched
    from the USGS Geomagnetism Program) onto the survey's along-track chainage,
    replacing the synthetic drift+wave terms with genuine measured field variation.

    The trace's native 1 Hz cadence is stretched onto chainage assuming a constant
    `survey_speed_m_per_s` -- a modelling simplification, not a claim that the real
    station was surveyed at that speed. What the detrend-honesty test in
    PLAN.md Stage 2.5 item 3 needs is the trace's genuine broadband SHAPE, which
    survives this stretch; only the real-world speed correspondence is invented.
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
    """3-axis field of a point dipole `moment` at `r_src`, observed at `r_obs` (N,3)."""
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


def _build_features(cfg: DataConfig, rng: np.random.Generator) -> list[dict]:
    types = ["scc", "weld", "dent", "corrosion"]
    feats = []
    placed: list[tuple[float, float]] = []  # (chainage_m, half_width_m), for spacing checks
    # y_off_m=0 for every defect, so r_eff=depth_m and the half-width is the
    # same constant for all of them.
    defect_half_width = cfg.label_window_scale * cfg.depth_m

    for _ in range(cfg.n_defects):
        chainage = _sample_spaced_chainage(rng, 50, cfg.length_m - 50, defect_half_width, placed)
        placed.append((chainage, defect_half_width))
        feats.append(
            {
                "chainage_m": chainage,
                "y_off_m": 0.0,
                "type": rng.choice(types),
                "severity": rng.uniform(20, 80),
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
        # traps nothing. Scaled this way it arrives at DEFECT-COMPARABLE
        # amplitude but visibly BROADER, which is the discrimination the whole
        # project turns on: shape separates it, amplitude does not.
        y_off_m = rng.uniform(3, 8) * rng.choice([-1, 1])
        half_width = cfg.label_window_scale * float(np.hypot(cfg.depth_m, y_off_m))
        chainage = _sample_spaced_chainage(rng, 50, cfg.length_m - 50, half_width, placed)
        placed.append((chainage, half_width))
        feats.append(
            {
                "chainage_m": chainage,
                "y_off_m": y_off_m,
                "type": "interference",
                "severity": rng.uniform(30, 90) * cfg.interference_moment_scale,
                "orientation": rng.normal(0, 1, 3),
                "is_defect": False,
            }
        )
    return feats


def _make_run(
    line_id: str, run_id: int, cfg: DataConfig, features: list[dict], rng: np.random.Generator
) -> pd.DataFrame:
    n = int(cfg.length_m / cfg.step_m)
    sample_idx = np.arange(n, dtype=np.int64)
    s = sample_idx * cfg.step_m  # chainage_m, derived
    obs = np.column_stack([s, np.zeros(n), np.zeros(n)])

    base = np.array(cfg.background_nT, dtype=float)
    if cfg.observatory_background.enabled:
        # Stage 2.5 item 3: a real trace, not a synthetic sinusoid -- see
        # _load_observatory_background's docstring for the along-track mapping.
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
    # Interference ground truth is tracked separately from `defect`/`defect_type`:
    # interference sources perturb the field exactly like a defect does, but must
    # never be labelled as one. Without this, there is no way to later compute
    # "interference precision" -- a metric this project's own invariants require.
    interference = np.zeros(n, dtype=int)

    # Half-width scales with r_eff, the actual dipole source-to-sensor distance --
    # the same 1/r^3 physics that sets amplitude also sets along-track extent. A
    # flat constant undersizes off-pipe interference (farther + lateral, so bigger
    # r_eff) relative to on-pipe defects (y_off_m=0, so r_eff=depth_m). Windows are
    # wide enough that neighbouring features can now overlap, so each row is
    # assigned to its NEAREST feature (by chainage) rather than to every feature
    # whose window contains it -- otherwise an overlap row would be labelled both
    # a defect and interference at once.
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

    if cfg.gradiometer.enabled:
        obs2 = np.column_stack([s, np.zeros(n), np.full(n, cfg.gradiometer.baseline_m)])
        # The upper head sees a slightly different background, not the identical
        # one -- otherwise common-mode rejection is perfect by construction. The
        # main field's vertical gradient (~0.02 nT/m) is a near-constant offset;
        # geology contributes more AND varies along the line, so it gets its own
        # spatially-varying term (independent random draw, same drift+wave shape
        # as the base background) scaled by the baseline so a bigger baseline
        # sees a bigger difference, as it physically should.
        main_field_grad = cfg.gradiometer.main_field_gradient_nT_per_m * cfg.gradiometer.baseline_m
        grad_shape = (
            np.linspace(0, 1, n)[:, None]
            + np.sin(2 * np.pi * s / cfg.length_m * rng.uniform(1, 3))[:, None]
        )
        geology_grad = (
            cfg.gradiometer.geology_gradient_scale_nT_per_m
            * cfg.gradiometer.baseline_m
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


def find_raw_survey_path(raw_dir: Path, survey_id: str) -> Path:
    """survey_id 'LINE000_R2' -> data/raw/line_id=LINE000/run_id=2/survey.parquet"""
    line_id, run_part = survey_id.rsplit("_R", 1)
    return Path(raw_dir) / f"line_id={line_id}" / f"run_id={run_part}" / "survey.parquet"


def load_survey_result(path: Path, step_m: float, standoff_m: float) -> SurveyResult:
    """Reconstruct a SurveyResult (hashes + metadata) from an already-written raw
    Parquet file. Used by both the CLI `ingest` command and the Dagster asset --
    one code path for "what does an already-generated survey look like".

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
    return SurveyResult(
        survey_id=f"{line_id}_R{run_id}",
        line_id=line_id,
        run_id=run_id,
        path=path,
        file_sha256=file_sha256(path),
        content_sha256=content_sha256(df),
        step_m=step_m,
        n_samples=len(df),
        chainage_start_m=float(df["chainage_m"].min()),
        chainage_end_m=float(df["chainage_m"].max()),
        standoff_m=standoff_m,
        surveyed_at=surveyed_at,
    )


def generate_all(cfg: DataConfig, raw_dir: Path, seed: int) -> list[SurveyResult]:
    """Generate cfg.n_lines lines x cfg.n_runs surveys, write one Parquet file each."""
    raw_dir = Path(raw_dir)
    results: list[SurveyResult] = []
    surveyed_at = dt.datetime.now(dt.UTC).isoformat()

    for line_idx in range(cfg.n_lines):
        line_id = f"LINE{line_idx:03d}"
        line_rng = np.random.default_rng(seed + line_idx)
        features = _build_features(cfg, line_rng)
        for run_id in range(cfg.n_runs):
            df = _make_run(line_id, run_id, cfg, features, line_rng)
            survey_id = f"{line_id}_R{run_id}"

            out_dir = raw_dir / f"line_id={line_id}" / f"run_id={run_id}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "survey.parquet"

            write_cols = [
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
                    step_m=cfg.step_m,
                    n_samples=len(df),
                    chainage_start_m=float(df["chainage_m"].min()),
                    chainage_end_m=float(df["chainage_m"].max()),
                    standoff_m=cfg.depth_m,
                    surveyed_at=surveyed_at,
                )
            )
    return results
