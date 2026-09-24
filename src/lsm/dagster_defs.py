"""
Dagster asset graph -- the orchestration INTERFACE, per docs/production-architecture.md
Section 3. The Typer CLI (__main__.py) is the implementation; this is what makes
processing the archive a partition backfill instead of a bespoke script.

    registered_survey --deps--> dq_report --deps--> survey_features

partitioned by survey_id (dynamic partitions). Retries: 3x on registration
(genuine IO/transport failures); NONE on dq_report -- a data-quality failure is
never an exception here (validate_raw_survey returns a report, it doesn't
raise), so "no retry on DQ failure" holds by construction, not by a
retry-policy carve-out. dq_report also loads the `reading` rows, but only
after confirming there is no hard fail -- see pipeline.py / validate.py for why
readings must never be loaded before validation.

survey_features carries `code_version = feature_version`. That correspondence is
the reason Dagster is the orchestrator here rather than Airflow: bumping
features.version in config marks exactly the affected partitions stale, so the
backfill re-materialises what actually changed instead of everything or nothing.
"""

from pathlib import Path

# NOTE: deliberately no `from __future__ import annotations` here. Dagster's
# @asset decorator checks the `context` parameter's annotation by identity
# against the real AssetExecutionContext class; PEP 563 postponed evaluation
# would turn that annotation into the plain string "AssetExecutionContext",
# which fails the identity check with a confusing error.
import dagster as dg
import pandas as pd
from dagster import AssetExecutionContext  # Dagster's context type-check inspects the

# literal annotation text, so `dg.AssetExecutionContext` on a function signature fails
# even though it is the same class -- the unaliased import is required here.
from lsm.config import load_config
from lsm.db import connect
from lsm.generate import find_raw_survey_path, load_survey_result
from lsm.ingest import load_readings, register_survey
from lsm.pipeline import SurveyNotFeaturisableError, run_feature_pipeline
from lsm.validate import validate_raw_survey

survey_partitions = dg.DynamicPartitionsDefinition(name="survey_id")


class LsmResource(dg.ConfigurableResource):
    """Holds only config -- never an open sqlite3.Connection (not pydantic-safe,
    and SQLite is single-writer so each asset invocation opens its own short-lived
    connection rather than sharing one across the run). `config_dir` is override-
    able so tests can point the asset graph at an isolated config/ directory
    instead of the real project one.
    """

    env_name: str = "dev"
    config_dir: str | None = None

    def load(self):
        return load_config(
            self.env_name, config_dir=Path(self.config_dir) if self.config_dir else None
        )


@dg.asset(
    partitions_def=survey_partitions,
    retry_policy=dg.RetryPolicy(
        max_retries=3, delay=0.2, backoff=dg.Backoff.EXPONENTIAL
    ),
    group_name="lsm",
)
def registered_survey(
    context: AssetExecutionContext, lsm: LsmResource
) -> dg.Output[str]:
    """raw parquet -> `survey` registry row. Idempotent: re-materializing an
    already-registered partition is a no-op, which is what makes a backfill
    resumable without reprocessing or duplicating rows.
    """
    survey_id = context.partition_key
    cfg = lsm.load()
    conn = connect(cfg.env.storage.sqlite_path)
    try:
        path = find_raw_survey_path(Path(cfg.env.storage.raw_dir), survey_id)
        sr = load_survey_result(path, cfg.base.data.step_m, cfg.base.data.depth_m)
        status = register_survey(conn, sr, cfg.base.schema_version)
    finally:
        conn.close()
    context.log.info(f"registered {survey_id}: {status}")
    return dg.Output(survey_id, metadata={"status": status})


@dg.asset(
    partitions_def=survey_partitions,
    deps=["registered_survey"],
    group_name="lsm",
)
def dq_report(context: AssetExecutionContext, lsm: LsmResource) -> dg.Output[str]:
    """Runs the 14 DQ checks against the raw file; quarantines on any hard fail;
    loads `reading` rows only if clean. No retry_policy -- a DQ result is a
    returned status, never a raised exception, so there is nothing here for
    Dagster's retry mechanism to act on.
    """
    survey_id = context.partition_key
    cfg = lsm.load()
    conn = connect(cfg.env.storage.sqlite_path)
    try:
        row = conn.execute(
            "SELECT line_id, step_m, chainage_start_m, chainage_end_m, "
            "content_sha256, source_uri FROM survey WHERE survey_id=?",
            (survey_id,),
        ).fetchone()
        line_id, step_m, c_start, c_end, content_hash, source_uri = row
        df = pd.read_parquet(source_uri)
        report = validate_raw_survey(
            conn,
            survey_id,
            line_id,
            step_m,
            c_start,
            c_end,
            content_hash,
            df,
            cfg.base.validate,
            quarantine_dir=Path(cfg.env.storage.quarantine_dir),
            source_path=Path(source_uri),
        )
        if not report.has_fail:
            already = conn.execute(
                "SELECT 1 FROM reading WHERE survey_id=? LIMIT 1", (survey_id,)
            ).fetchone()
            if not already:
                load_readings(conn, survey_id, df)
    finally:
        conn.close()
    summary = report.summary()
    context.log.info(f"validated {survey_id}: {summary}")
    return dg.Output(
        survey_id,
        metadata={
            "pass": summary["pass"],
            "warn": summary["warn"],
            "fail": summary["fail"],
            "quarantined": report.has_fail,
        },
    )


@dg.asset(
    partitions_def=survey_partitions,
    deps=["dq_report"],
    code_version=str(load_config("dev").base.features.version),
    group_name="lsm",
)
def survey_features(context: AssetExecutionContext, lsm: LsmResource) -> dg.Output[str]:
    """Accepted survey -> feature store at fv=<feature_version>.

    A quarantined partition yields rather than raises: the DQ gate already
    reported it, and turning "we correctly refused this survey" into a run
    failure is how a nightly pipeline gets switched off by its operators.

    No retry_policy: the only failures reachable here are a missing raw file or a
    genuine bug in features.py, and retrying either three times just delays the
    ticket. The IO retries belong on registered_survey, where the transport is.
    """
    survey_id = context.partition_key
    cfg = lsm.load()
    conn = connect(cfg.env.storage.sqlite_path)
    try:
        try:
            outcome, dir_path = run_feature_pipeline(conn, survey_id, cfg)
        except SurveyNotFeaturisableError as exc:
            context.log.info(f"skipped {survey_id}: {exc}")
            return dg.Output(
                survey_id, metadata={"outcome": "skipped", "reason": str(exc)}
            )
    finally:
        conn.close()
    context.log.info(f"features {survey_id}: {outcome} -> {dir_path}")
    return dg.Output(
        survey_id,
        metadata={
            "outcome": outcome,
            "feature_version": cfg.base.features.version,
            "path": str(dir_path),
        },
    )


defs = dg.Definitions(
    assets=[registered_survey, dq_report, survey_features],
    resources={"lsm": LsmResource(env_name="dev")},
)
