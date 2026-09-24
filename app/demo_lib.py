"""
Stage 4.5: the seam between the demo app and `lsm` -- deliberately Streamlit-
free (no `import streamlit`) so it can be exercised directly by plain pytest,
not just by eyeballing a browser.

DEMO mode reads everything here straight off the committed `serving/`
directory (Parquet + JSON) -- zero SQLite, zero compute, cannot fail. LIVE
mode creates one real, session-private SQLite db per browser session (never
`:memory:` through `db.connect()`'s `Path(...)` wrapping -- ambiguous on
Windows; a real tempfile is simpler) and drives the exact same
`pipeline.run_full_pipeline` the CLI/Dagster use, seeded with the manifest's
own real `pipeline_release`/`model_run` rows so `predict_survey`'s FK reads
resolve. The live connection belongs in `st.session_state`, never
`st.cache_resource` -- that cache is a process-wide singleton, and this
connection is written to (validate quarantines files, predict inserts rows).
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd

from lsm.config import Config
from lsm.db import connect
from lsm.features import feature_store_dir
from lsm.generate import load_survey_result
from lsm.pipeline import run_full_pipeline
from lsm.validate import DQReport

SERVING_DIR = Path(__file__).resolve().parents[1] / "serving"


def load_manifest() -> dict:
    return json.loads((SERVING_DIR / "manifest.json").read_text(encoding="utf-8"))


def demo_scenarios() -> list[dict]:
    return load_manifest()["demo_scenarios"]


def load_model_card() -> str:
    """The auto-generated model card (`model_card.py`) baked alongside the
    bundles -- the real MAD-vs-IsolationForest, severity-vs-baseline and
    classify-vs-baseline numbers from the training run that produced whatever
    this app is currently serving. Same file regardless of scenario or mode
    (demo/live), since it describes the released pipeline, not one survey."""
    return (SERVING_DIR / "model_card.md").read_text(encoding="utf-8")


def load_demo_raw(survey_id: str) -> pd.DataFrame:
    return pd.read_parquet(SERVING_DIR / "demo_surveys" / f"{survey_id}.parquet")


def load_demo_features(survey_id: str) -> pd.DataFrame | None:
    path = SERVING_DIR / "precomputed" / f"{survey_id}_features.parquet"
    return pd.read_parquet(path) if path.exists() else None


def load_demo_indications(survey_id: str) -> pd.DataFrame | None:
    path = SERVING_DIR / "precomputed" / f"{survey_id}_indications.parquet"
    return pd.read_parquet(path) if path.exists() else None


def load_demo_dq_failure(survey_id: str) -> dict | None:
    """The baked DQReport for the one scenario that's supposed to fail --
    None for every clean scenario, which never has one."""
    report = load_manifest().get("corrupted_scenario_dq_report")
    if report is not None and report["survey_id"] == survey_id:
        return report
    return None


def init_live_session(base_cfg: Config) -> tuple[sqlite3.Connection, Config]:
    """One real, session-private SQLite db + a Config pointed at it and at the
    baked (read-only, shared) `serving/bundles/`. Call once per browser
    session (the caller is responsible for stashing the result in
    `st.session_state` and not calling this again for that session).
    `config_sha256` is untouched: only the unhashed storage-path layer moves.
    """
    session_dir = Path(tempfile.mkdtemp(prefix="lsm_demo_session_"))
    cfg = base_cfg.model_copy(deep=True)
    cfg.env.storage.sqlite_path = str(session_dir / "session.db")
    cfg.env.storage.raw_dir = str(session_dir / "raw")
    cfg.env.storage.quarantine_dir = str(session_dir / "quarantine")
    cfg.env.storage.feature_dir = str(session_dir / "features")
    cfg.env.storage.reports_dir = str(session_dir / "reports")
    cfg.env.storage.model_dir = str(SERVING_DIR / "bundles")

    # check_same_thread=False: Streamlit can dispatch a session's consecutive
    # reruns onto different worker threads from its own thread pool, even
    # though only one rerun ever executes at a time for that session -- so the
    # stdlib's same-thread guard fires on a false positive here, not a real
    # concurrent-access hazard. Confirmed by headless testing (streamlit.testing
    # .v1.AppTest): a second live-mode rerun raised sqlite3.ProgrammingError
    # without this flag.
    conn = connect(cfg.env.storage.sqlite_path, check_same_thread=False)

    manifest = load_manifest()
    # model_run before pipeline_release: pipeline_release.anomaly_version/
    # severity_version REFERENCE model_run(model_version), and this session db
    # runs with PRAGMA foreign_keys=ON.
    for mr in manifest["model_run"].values():
        conn.execute(
            "INSERT INTO model_run (model_version, task, mlflow_run_id, git_sha, "
            "config_sha256, data_sha256, feature_version, truth_as_of, trained_at, "
            "metrics_json, artifact_uri, final_test_uses) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                mr["model_version"],
                mr["task"],
                mr["mlflow_run_id"],
                mr["git_sha"],
                mr["config_sha256"],
                mr["data_sha256"],
                mr["feature_version"],
                mr["truth_as_of"],
                mr["trained_at"],
                mr["metrics_json"],
                mr["artifact_uri"],
                mr["final_test_uses"],
            ),
        )
    pr = manifest["pipeline_release"]
    conn.execute(
        "INSERT INTO pipeline_release (pipeline_version, anomaly_version, severity_version, "
        "classify_version, growth_version, feature_version, schema_version, "
        "container_digest, released_at, alias) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            pr["pipeline_version"],
            pr["anomaly_version"],
            pr["severity_version"],
            pr["classify_version"],
            pr["growth_version"],
            pr["feature_version"],
            pr["schema_version"],
            pr["container_digest"],
            pr["released_at"],
            pr["alias"],
        ),
    )
    conn.commit()
    return conn, cfg


def run_live_scenario(
    conn: sqlite3.Connection, cfg: Config, survey_id: str
) -> tuple[DQReport, pd.DataFrame | None]:
    """Load one baked demo survey and run it through the real
    register -> validate -> features -> score path, live."""
    raw_path = SERVING_DIR / "demo_surveys" / f"{survey_id}.parquet"
    sr = load_survey_result(raw_path, cfg.base.data.step_m, cfg.base.data.depth_m)
    return run_full_pipeline(conn, sr, cfg)


def load_live_features(cfg: Config, survey_id: str) -> pd.DataFrame | None:
    """Read back the feature-store parquet a successful `run_live_scenario`
    call just wrote -- `run_full_pipeline` itself only returns indications,
    not the intermediate detrended/gradient features Beat 2 plots."""
    line_id, run_part = survey_id.split("_R")
    path = (
        feature_store_dir(
            cfg.env.storage.feature_dir,
            cfg.base.features.version,
            line_id,
            int(run_part),
        )
        / "features.parquet"
    )
    return pd.read_parquet(path) if path.exists() else None
