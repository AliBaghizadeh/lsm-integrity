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


class GradiometerConfig(BaseModel):
    enabled: bool = False
    baseline_m: float = 0.5
    main_field_gradient_nT_per_m: float = 0.02
    geology_gradient_scale_nT_per_m: float = 0.3


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
    step_m: float
    depth_m: float
    background_nT: list[float]
    noise_nT: float
    n_defects: int
    n_interference: int
    interference_moment_scale: float = 1.0
    label_window_scale: float
    n_runs: int
    growth: float
    gradiometer: GradiometerConfig = GradiometerConfig()
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
    standoff_normalise: bool
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
    dig_budget_per_km: int


class BaseConfig(BaseModel):
    """Everything in config/base.yaml -- the hashed, code-config layer."""

    seed: int
    data: DataConfig
    validate: ValidateConfig
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
