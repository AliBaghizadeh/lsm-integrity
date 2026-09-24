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
from lsm.predict import predict_survey
from lsm.registration import register_survey as register_chainage
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


def survey_context(
    conn: sqlite3.Connection, survey_id: str, cfg: Config
) -> SurveyContext:
    """Build a SurveyContext from the registry. `standoff_m` comes from the
    `survey` row rather than from config, because on real data it is a
    property of the acquisition, not of our code -- `step_m` no longer does
    (dropped from SurveyContext at Rig-v2: chainage is registration's output,
    not `sample_idx * step_m`, so no reader was left for a fixed nominal step;
    see features.SurveyContext's docstring).
    """
    row = conn.execute(
        "SELECT line_id, run_id, standoff_m, surveyed_at, status FROM survey "
        "WHERE survey_id=?",
        (survey_id,),
    ).fetchone()
    if row is None:
        raise SurveyNotFeaturisableError(f"{survey_id} is not registered")
    line_id, run_id, standoff_m, surveyed_at, status = row
    if status != "accepted":
        raise SurveyNotFeaturisableError(f"{survey_id} has status={status!r}")
    return SurveyContext(
        survey_id=survey_id,
        line_id=line_id,
        run_id=int(run_id),
        surveyed_at=str(surveyed_at),
        standoff_m=float(standoff_m),
        # cfg.base.data.gradiometer was absorbed into ArrayConfig at Rig-v2
        # (config.py) -- array.spacing_m is its replacement (the fixed 0.5 m
        # head separation, same physical role the old gradiometer baseline
        # played, now mandatory rather than optional: rig: scalar always has
        # 3 heads).
        array_spacing_m=cfg.base.data.array.spacing_m,
    )


def run_feature_pipeline(
    conn: sqlite3.Connection, survey_id: str, cfg: Config, force: bool = False
) -> tuple[str, Path]:
    """Registered, accepted survey -> feature store. Returns ('hit'|'computed', dir).

    Reads the RAW parquet, not the `reading` table: features must be computed from
    the same bytes validation ran against, and `reading` has already dropped the
    convenience columns. Same reasoning as validate.py.

    Stage B's registration runs HERE, once per survey, rather than inside
    features.py: `register_survey` needs the full `DataConfig` (geo/weld/walk),
    which `compute_survey_features` deliberately does not receive (see its
    docstring) -- this function already has both the full `Config` and the raw
    `df`, so it is the natural (and only) place both are in scope together.
    `registration.register_survey` is aliased on import: `ingest.register_survey`
    (the survey-METADATA DB registration this function's sibling calls) is an
    unrelated function that happens to share the name.
    """
    ctx = survey_context(conn, survey_id, cfg)
    source_uri, content_hash = conn.execute(
        "SELECT source_uri, content_sha256 FROM survey WHERE survey_id=?", (survey_id,)
    ).fetchone()
    df = pd.read_parquet(source_uri)
    reg = register_chainage(df, cfg.base.data)
    return compute_and_store(
        df,
        ctx,
        cfg.base.features,
        feature_dir=cfg.env.storage.feature_dir,
        content_sha256=content_hash,
        config_sha256=cfg.config_sha256,
        chainage_m=reg.chainage_m,
        dist_to_weld_m=reg.dist_to_weld_m,
        force=force,
    )


def run_full_pipeline(
    conn: sqlite3.Connection, sr: SurveyResult, cfg: Config
) -> tuple[DQReport, pd.DataFrame | None]:
    """Register -> validate -> features -> score, in one call. `indications` is
    None iff `report.has_fail` -- a quarantined survey is never featurised or
    scored, by design (the same guarantee `run_survey_pipeline` already gives
    register vs `load_readings`). This never raises on a bad survey: a hard DQ
    fail degrades to "here's why it failed" (the returned report), not an
    exception -- that degradation is the whole point for a demo app's live
    mode, but it is equally usable by anything else that wants "just get me
    indications for this raw file" in one call.
    """
    _, report = run_survey_pipeline(conn, sr, cfg)
    if report.has_fail:
        return report, None
    run_feature_pipeline(conn, sr.survey_id, cfg)
    indications = predict_survey(conn, sr.survey_id, cfg)
    return report, indications
