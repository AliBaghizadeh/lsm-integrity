from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from lsm.config import CONFIG_DIR, load_config


@pytest.fixture
def cfg(tmp_path: Path):
    """A Config identical to the real base.yaml, but pointed at an isolated tmp_path
    for all storage (sqlite, raw, quarantine) so tests never touch data/.
    """
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    shutil.copyfile(CONFIG_DIR / "base.yaml", cfg_dir / "base.yaml")

    env_raw = {
        "storage": {
            "sqlite_path": str(tmp_path / "lsm.db"),
            "raw_dir": str(tmp_path / "raw"),
            "quarantine_dir": str(tmp_path / "quarantine"),
            "feature_dir": str(tmp_path / "features"),
            "model_dir": str(tmp_path / "models"),
            "reports_dir": str(tmp_path / "reports"),
            "s3": {"enabled": False},
        },
        "mlflow": {"tracking_uri": f"sqlite:///{tmp_path / 'mlruns.db'}"},
        "dagster": {"concurrency": 1},
    }
    with open(cfg_dir / "dev.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(env_raw, f)

    return load_config("dev", config_dir=cfg_dir)


@pytest.fixture
def tiny_cfg(cfg):
    """A small survey (200 samples) so tests run fast. One line, one run.
    length_m must stay > 100: defect/interference placement keeps a 50 m margin
    off each end (rng.uniform(50, length_m - 50)), a real physics assumption in
    generate.py, not something to work around in the fixture.
    """
    cfg.base.data.length_m = 200.0
    cfg.base.data.step_m = 1.0
    cfg.base.data.n_lines = 1
    cfg.base.data.n_runs = 1
    cfg.base.data.n_defects = 2
    cfg.base.data.n_interference = 1
    return cfg


def generate_one_survey(tiny_cfg, tmp_path, mutate=None):
    """Generate a single tiny survey, optionally corrupt its raw Parquet file via
    `mutate(df) -> df`, and return the (possibly rewritten) SurveyResult.
    """
    import pandas as pd

    from lsm.generate import generate_all
    from lsm.hashing import content_sha256, file_sha256

    raw_dir = tmp_path / "raw"
    results = generate_all(tiny_cfg.base.data, raw_dir, seed=42)
    sr = results[0]

    if mutate is not None:
        df = pd.read_parquet(sr.path)
        df = mutate(df)
        df.to_parquet(sr.path, index=False, compression="zstd")
        sr.file_sha256 = file_sha256(sr.path)
        sr.content_sha256 = content_sha256(df)
        sr.n_samples = len(df)

    return sr


def run_pipeline_on(tiny_cfg, sr):
    from lsm.db import connect
    from lsm.pipeline import run_survey_pipeline

    conn = connect(tiny_cfg.env.storage.sqlite_path)
    status, report = run_survey_pipeline(conn, sr, tiny_cfg)
    return conn, status, report
