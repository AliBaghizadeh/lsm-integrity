"""
Layered configuration.

Two independent layers with different lifecycles, per docs/production-architecture.md:
  - CODE config (config/base.yaml): hyperparameters, feature specs, gate thresholds.
    This is the ONLY layer that feeds config_sha256 -- it is what "reproducible run"
    means. Mixing environment settings into this hash would make the same code+data
    hash differently in dev vs prod, silently voiding the reproducibility guarantee.
  - ENVIRONMENT config (config/{dev,prod}.yaml): bucket, DB path, MLflow URI,
    concurrency. Never hashed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


class ArrayConfig(BaseModel):
    """The physical rod, Rig-v2: `n_heads` scalar magnetometers `spacing_m` apart
    on a vertical mast, replacing the old two-head `GradiometerConfig` now that
    the real instrument has three (middle + two, 50 cm apart) rather than an
    optional second head at a configurable baseline.

    Under `data.rig: scalar` the head count is fixed at 3 (b_lo/b_mid/b_hi are
    hardcoded column names in schemas.py, matching the real rig) -- `n_heads`
    there is documentation of that physical fact, asserted in generate.py, not
    a free dial. Under `data.rig: vector` (the preserved legacy path) `n_heads`
    does its old GradiometerConfig job: 1 = single vector head, 2 = the old
    `gradiometer.enabled` dual-head behaviour, at `spacing_m` apart.

    `main_field_gradient_nT_per_m` / `geology_gradient_scale_nT_per_m` are
    unchanged physics carried over unrenamed from GradiometerConfig: without
    them the background seen by every head is identical and common-mode /
    gradient rejection would be perfect by construction, which is not real.
    `orientation` records the Stage-A decision to model the rod as a vertical
    mast (Context, "Decisions taken") -- "horizontal" is one of the open
    unresolved instrument questions and is deliberately NOT implemented; picking it raises
    rather than silently running vertical-mast physics under a different label.
    """

    n_heads: int = 3
    spacing_m: float = 0.5
    orientation: Literal["vertical", "horizontal"] = "vertical"
    main_field_gradient_nT_per_m: float = 0.02
    geology_gradient_scale_nT_per_m: float = 0.3


class WalkConfig(BaseModel):
    """A human walking the line, not a survey cart on rails: speed, stand-off
    and lateral position all wander as OU (Ornstein-Uhlenbeck, mean-reverting
    random-walk) processes rather than being fixed. Correlation lengths convert
    to correlation TIMES internally via the nominal walking speed, since the
    walk is sampled at a fixed rate in TIME (`sample_rate_hz`), not in distance
    -- that irregularity in along-track spacing is the whole point of Rig-v2.

    Values marked "plan" come directly from the approved Stage-A physics spec.
    Values marked "reasoned default" (the *_correlation_s/_m fields not spelled
    out there) are this implementation's own choice of a physically plausible
    time/length scale, documented here rather than left as a bare magic number.
    """

    speed_m_per_s: float = 1.2  # plan: typical walking pace
    speed_sigma_m_per_s: float = 0.15  # plan
    # reasoned default: a walker's stride-to-stride pace correlates over a few
    # strides, not metres -- a handful of seconds is the natural scale.
    speed_correlation_s: float = 8.0
    speed_min_m_per_s: float = 0.6  # plan: OU clip floor
    speed_max_m_per_s: float = 1.8  # plan: OU clip ceiling
    sample_rate_hz: float = 120.0  # plan: -> ~1 cm mean spacing at nominal speed
    standoff_m: float = 1.5  # plan: nominal rod height above the ground/pipe
    standoff_sigma_m: float = 0.20  # plan
    standoff_correlation_m: float = 10.0  # plan
    lateral_sigma_m: float = 0.30  # plan: walker wander off centreline
    # reasoned default: a walker corrects lateral drift roughly every few
    # strides -- shorter than the stand-off correlation length, which tracks
    # slower postural changes (arm height, fatigue), not footfall.
    lateral_correlation_m: float = 5.0
    tilt_sigma_deg: float = 5.0  # plan: rod sway -- moves head POSITIONS only
    # reasoned default: mast sway follows gait cadence, correlated over a
    # stride or two.
    tilt_correlation_m: float = 3.0


class GpsConfig(BaseModel):
    """Markov good/bad lock state, not "always on": poor sky view drops GPS
    out entirely for stretches, which is what `registration.py` (Stage B) and
    `check_gps_chainage_consistency` exist to survive. `dropout_rate` is the
    steady-state FRACTION of time spent unlocked; `mean_gap_s` is the mean
    dwell time of one dropout, which together fix the two Markov transition
    probabilities (see generate.py's `_gps_track`). Both losing lock (NaN
    lat/lon) and horizontal noise while locked are modelled -- a real receiver
    does both, not one or the other.
    """

    dropout_rate: float = 0.05
    mean_gap_s: float = 15.0  # plan: ~15 s ~= 18 m at nominal walking speed
    sigma_m: float = 1.5  # plan: horizontal noise when locked


class WeldConfig(BaseModel):
    """Girth welds: a periodic train of joints every `pitch_m` (+/- jitter),
    NOT one of the dozen rare point-defect types -- a pipeline seam every
    ~12 m is a fact of how the pipe was built, not a rare event, and this
    corrects a real physical error in the pre-Rig-v2 model (`weld` used to be
    drawn 12 TIMES total on a 2 km line via `rng.choice(types)`, not every
    12 m). Isotropic orientation, same as a defect -- the developer's point
    that the extra metal at a joint has arbitrary rotation, so no fixed
    amplitude/polarity template exists; only the spacing is a reliable prior
    (this is what Stage B's weld-comb detector uses instead of a template).
    Labelled `girth_weld`, never `defect` -- a weld is not damage.
    """

    # False builds NO weld train at all (generate.py's generate_all), which no
    # real buried pipeline can be -- this exists solely for the clean-room
    # counterfactual corpus (scripts/cleanroom_experiment.py): an isolated test
    # spool with defects and nothing else, the in-house controlled experiment
    # this project argues for. Leaving it True is the physical default; setting
    # it False is a deliberate, labelled counterfactual, not a config nicety.
    enabled: bool = True
    pitch_m: float = 12.2
    pitch_jitter_m: float = 0.15
    # A weld's extra steel makes it 5-20x a defect's dipole moment for the
    # same severity-draw distribution (generate.py's _build_welds).
    moment_scale_range: tuple[float, float] = (5.0, 20.0)


class SensorConfig(BaseModel):
    """Per-head sensor imperfections that a real fluxgate head actually has and
    the pre-Rig-v2 model didn't: fixed-per-survey gain error and additive
    offset, ADC quantization from `adc_bits` over +/- `full_scale_ut`, and
    independent per-row noise. `gain_sigma` is the point of this whole class:
    a 0.2% gain MISMATCH between two heads leaves ~100 nT of uncancelled
    common-mode signal against a ~25 nT anomaly -- inter-sensor calibration is
    therefore a modelled first-class problem, not a footnote (see
    tests/test_generate.py's gain-mismatch test). `adc_bits: 16` is a
    deliberate stress test: LSB goes from ~0.012 nT at 24-bit to ~3 nT.
    """

    gain_sigma: float = 0.002
    offset_nt: float = 2.0
    adc_bits: int = 24
    full_scale_ut: float = (
        100.0  # Open question: full-scale range or noise floor?
    )
    noise_nt: float = 5.0


class GeoConfig(BaseModel):
    lat0: float
    lon0: float
    bearing_deg: float


class ObservatoryBackgroundConfig(BaseModel):
    """Stage 2.5 item 3: swap the synthetic drift+wave background for a real
    geomagnetic observatory trace (data/reference/*.csv). Off by default -- it is
    a deliberate realism stress test, not the pipeline's operating default, since
    a disturbed-day trace can push background residual above the defect signal
    and break the Stage 2 detectability gate. See PLAN.md Stage 2.5 item 3.
    """

    enabled: bool = False
    csv_path: str = "data/reference/geomag_bou_2024-05-10_storm.csv"
    survey_speed_m_per_s: float = 1.0


class DataConfig(BaseModel):
    n_lines: int = 1
    length_m: float
    # step_m: mean/nominal along-track spacing. Under `rig: scalar` it is NOT
    # what determines row spacing (the walk is time-sampled at
    # `walk.sample_rate_hz`, irregularly spaced in distance by design) -- it
    # is kept only for the `rig: vector` legacy grid and for metadata
    # (SurveyResult.step_m) that downstream code still expects a value for.
    step_m: float
    depth_m: float
    background_nT: list[float]
    noise_nT: float
    n_defects: int
    n_interference: int
    # A range, not a fixed constant -- the developer team's follow-up point:
    # real external interference is "a bit weaker in most cases" than a
    # single uniform 50x figure implies, and its strength varies source to
    # source independently of distance (distance/geometry is already
    # captured by y_off_m's own 3-8 m draw in _build_features -- this scales
    # the MOMENT on top of that, not instead of it). Drawn once per
    # interference source in _build_features, not once per survey.
    interference_moment_scale_range: tuple[float, float] = (30.0, 50.0)
    label_window_scale: float
    n_runs: int
    growth: float
    # rig: scalar is Rig-v2, the real instrument (3 total-field heads, human
    # walk, GPS dropout) and the new default. rig: vector reproduces the
    # pre-Rig-v2 model UNCHANGED -- kept reachable, not deleted, as the
    # reference arm of the Stage D hardware-upgrade ablation ("what would a
    # full vector-output sensor buy over the real scalar rod").
    rig: Literal["scalar", "vector"] = "scalar"
    # random: each defect's moment sign is whatever its isotropic orientation
    # draw gives (stress can enhance OR degrade the total-field anomaly -- the
    # developer's stated point, and the honest default). positive: canonicalised
    # so the anomaly always enhances the field, i.e. the old implicit assumption.
    # Only affects rig: scalar -- see generate.py's _make_run_scalar docstring
    # for why this has no well-defined meaning for a full vector output.
    stress_polarity: Literal["random", "positive"] = "random"
    array: ArrayConfig = ArrayConfig()
    walk: WalkConfig = WalkConfig()
    gps: GpsConfig = GpsConfig()
    weld: WeldConfig = WeldConfig()
    sensor: SensorConfig = SensorConfig()
    observatory_background: ObservatoryBackgroundConfig = ObservatoryBackgroundConfig()
    geo: GeoConfig


GateAction = Literal["fail", "warn"]


class ValidateConfig(BaseModel):
    gates: dict[str, GateAction]
    max_gap_m: float
    max_gps_jump_m: float
    field_range_nT: tuple[float, float]
    noise_floor_range_nT: tuple[float, float]
    min_coverage_frac: float
    saturation_run_length: int
    overlap_correlation_threshold: float


class DetrendConfig(BaseModel):
    method: Literal["robust_poly", "savgol"]
    degree: int
    window_m: float


class PeakConfig(BaseModel):
    prominence_mad: float = 4.0
    flank_fit_span: tuple[float, float] = (0.5, 3.0)
    assign_radius_fwhm: float = 2.0


class FeaturesConfig(BaseModel):
    version: int
    detrend: DetrendConfig
    windows_m: list[float]
    peak: PeakConfig = PeakConfig()
    edge_policy: Literal["flag", "drop"]


class SplitConfig(BaseModel):
    by: str
    n_folds: int
    fallback_block_m: float


class BootstrapConfig(BaseModel):
    n_resamples: int
    level: float


class LightGBMConfig(BaseModel):
    deterministic: bool = True
    force_row_wise: bool = True
    num_threads: int = 16


class ModelConfig(BaseModel):
    split: SplitConfig
    bootstrap: BootstrapConfig
    lightgbm: LightGBMConfig
    anomaly: dict
    severity: dict
    classify: dict
    growth: dict
    monitor: dict
    dig_budget_per_km: int


class BaseConfig(BaseModel):
    """Everything in config/base.yaml -- the hashed, code-config layer."""

    seed: int
    data: DataConfig
    validate: ValidateConfig  # type: ignore[assignment]  # shadows BaseModel.validate; not worth the call-site churn to rename, see pydantic UserWarning at import time
    features: FeaturesConfig
    model: ModelConfig
    mlflow: dict
    schema_version: int


class S3Config(BaseModel):
    bucket: str | None = None
    prefix: str = "lsm-demo"
    enabled: bool = False


class StorageConfig(BaseModel):
    sqlite_path: str
    raw_dir: str
    quarantine_dir: str
    feature_dir: str
    model_dir: str = "models"
    reports_dir: str = "reports"
    s3: S3Config = S3Config()


class EnvConfig(BaseModel):
    """Everything in config/{env}.yaml -- the unhashed, environment layer."""

    storage: StorageConfig
    mlflow: dict
    dagster: dict


def _canonical_hash(obj: dict) -> str:
    """Deterministic hash over a dict: stable key order, no whitespace variance."""
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class Config(BaseModel):
    env_name: str
    base: BaseConfig
    env: EnvConfig
    config_sha256: str

    @property
    def seed(self) -> int:
        return self.base.seed


def load_config(env_name: str = "dev", config_dir: Path | None = None) -> Config:
    """Load and merge config/base.yaml (hashed) + config/{env_name}.yaml (not hashed).

    The hash is computed from base.yaml ALONE, so the same code config produces the
    same config_sha256 regardless of which environment it is run in. This is asserted
    by tests/test_config.py.
    """
    config_dir = config_dir or CONFIG_DIR
    base_path = config_dir / "base.yaml"
    env_path = config_dir / f"{env_name}.yaml"

    with open(base_path, encoding="utf-8") as f:
        base_raw = yaml.safe_load(f)
    with open(env_path, encoding="utf-8") as f:
        env_raw = yaml.safe_load(f)

    config_sha256 = _canonical_hash(base_raw)

    return Config(
        env_name=env_name,
        base=BaseConfig.model_validate(base_raw),
        env=EnvConfig.model_validate(env_raw),
        config_sha256=config_sha256,
    )
