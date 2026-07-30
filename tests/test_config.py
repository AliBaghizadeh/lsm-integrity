from __future__ import annotations

import shutil

import yaml

from lsm.config import CONFIG_DIR, load_config


def test_config_sha256_is_deterministic():
    a = load_config("dev")
    b = load_config("dev")
    assert a.config_sha256 == b.config_sha256


def test_config_sha256_ignores_environment_layer(tmp_path):
    """Same code config, different environment config -> same hash. This is the
    property that makes the reproducibility guarantee survive a dev->prod move.
    """
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    shutil.copyfile(CONFIG_DIR / "base.yaml", cfg_dir / "base.yaml")

    for env_name, sqlite_path in [("dev", "dev.db"), ("prod", "prod.db")]:
        env_raw = {
            "storage": {
                "sqlite_path": sqlite_path,
                "raw_dir": "raw",
                "quarantine_dir": "quarantine",
                "feature_dir": "features",
                "s3": {"enabled": env_name == "prod"},
            },
            "mlflow": {"tracking_uri": "file:./mlruns"},
            "dagster": {"concurrency": 4 if env_name == "dev" else 16},
        }
        with open(cfg_dir / f"{env_name}.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(env_raw, f)

    dev_cfg = load_config("dev", config_dir=cfg_dir)
    prod_cfg = load_config("prod", config_dir=cfg_dir)

    assert dev_cfg.config_sha256 == prod_cfg.config_sha256
    assert dev_cfg.env.storage.sqlite_path != prod_cfg.env.storage.sqlite_path


def test_config_sha256_changes_when_code_config_changes(tmp_path):
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    shutil.copyfile(CONFIG_DIR / "dev.yaml", cfg_dir / "dev.yaml")

    with open(CONFIG_DIR / "base.yaml", encoding="utf-8") as f:
        base_raw = yaml.safe_load(f)
    with open(cfg_dir / "base.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(base_raw, f)
    original = load_config("dev", config_dir=cfg_dir).config_sha256

    base_raw["seed"] = 999
    with open(cfg_dir / "base.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(base_raw, f)
    changed = load_config("dev", config_dir=cfg_dir).config_sha256

    assert original != changed
