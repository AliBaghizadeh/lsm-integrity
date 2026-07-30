"""
Single orchestration path: register -> validate (raw) -> load readings only if
clean -> quarantine on hard fail -> features. Shared by the CLI (__main__.py) and
the Dagster asset graph (dagster_defs.py), so "how the CLI does it" and "how the
scheduled pipeline does it" cannot silently diverge.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from lsm.config import Config
from lsm.features import SurveyContext, compute_and_store
from lsm.generate import SurveyResult
from lsm.ingest import load_readings, register_survey
from lsm.validate import DQReport, validate_raw_survey


class SurveyNotFeaturisableError(Exception):
    """The survey is not registered, or is quarantined. Quarantined data does not
    get features: the whole point of a hard DQ gate is that nothing downstream
    consumes what failed it."""


def run_survey_pipeline(
    conn: sqlite3.Connection, sr: SurveyResult, cfg: Config
) -> tuple[str, DQReport]:
    """Register the survey, validate its raw content, and only load readings
    (the PK-constrained table) if validation found no hard fail. Idempotent:
    re-running on an already-registered survey is safe -- register no-ops,
    validate re-runs cheaply, and load_readings is skipped if rows already exist.
    """
    register_status = register_survey(conn, sr, cfg.base.schema_version)

    df = pd.read_parquet(sr.path)
    report = validate_raw_survey(
        conn,
        sr.survey_id,
        sr.line_id,
        sr.step_m,
        sr.chainage_start_m,
        sr.chainage_end_m,
        sr.content_sha256,
        df,
        cfg.base.validate,
        quarantine_dir=Path(cfg.env.storage.quarantine_dir),
        source_path=sr.path,
    )

    if not report.has_fail:
        already_loaded = conn.execute(
            "SELECT 1 FROM reading WHERE survey_id=? LIMIT 1", (sr.survey_id,)
        ).fetchone()
        if not already_loaded:
            load_readings(conn, sr.survey_id, df)

    return register_status, report


def survey_context(conn: sqlite3.Connection, survey_id: str, cfg: Config) -> SurveyContext:
    """Build a SurveyContext from the registry. Geometry comes from the `survey`
    row rather than from config, because on real data step_m and standoff_m are
    properties of the acquisition, not of our code.
    """
    row = conn.execute(
        "SELECT line_id, run_id, step_m, standoff_m, surveyed_at, status FROM survey "
        "WHERE survey_id=?",
        (survey_id,),
    ).fetchone()
    if row is None:
        raise SurveyNotFeaturisableError(f"{survey_id} is not registered")
    line_id, run_id, step_m, standoff_m, surveyed_at, status = row
    if status != "accepted":
        raise SurveyNotFeaturisableError(f"{survey_id} has status={status!r}")
    return SurveyContext(
        survey_id=survey_id,
        line_id=line_id,
        run_id=int(run_id),
        step_m=float(step_m),
        standoff_m=float(standoff_m),
        surveyed_at=str(surveyed_at),
        gradiometer_baseline_m=cfg.base.data.gradiometer.baseline_m,
    )


def run_feature_pipeline(
    conn: sqlite3.Connection, survey_id: str, cfg: Config, force: bool = False
) -> tuple[str, Path]:
    """Registered, accepted survey -> feature store. Returns ('hit'|'computed', dir).

    Reads the RAW parquet, not the `reading` table: features must be computed from
    the same bytes validation ran against, and `reading` has already dropped the
    convenience columns. Same reasoning as validate.py.
    """
    ctx = survey_context(conn, survey_id, cfg)
    source_uri, content_hash = conn.execute(
        "SELECT source_uri, content_sha256 FROM survey WHERE survey_id=?", (survey_id,)
    ).fetchone()
    df = pd.read_parquet(source_uri)
    return compute_and_store(
        df,
        ctx,
        cfg.base.features,
        feature_dir=cfg.env.storage.feature_dir,
        content_sha256=content_hash,
        config_sha256=cfg.config_sha256,
        force=force,
    )
