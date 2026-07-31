"""
Raw Parquet -> SQLite, split into two phases so a corrupted raw file can be
validated and quarantined instead of crashing on the reading table's primary
key constraint:

  1. register_survey  -- always inserts the `survey` metadata row (idempotent on
                          content_sha256). Runs regardless of data quality.
  2. load_readings     -- inserts into the PK-constrained `reading` table. Only
                          called by pipeline.py AFTER validation confirms there
                          is no duplicate_sample_idx / monotonicity violation, so
                          it can never hit the PK constraint on real data.

Validating straight off the SQLite `reading` table would be pointless for
monotonicity/duplicate checks in particular: SQLite's WITHOUT ROWID storage
clusters rows by primary key, so a SELECT * always comes back sorted regardless
of whether the raw file was actually out of order. validate.py therefore checks
the RAW DataFrame, read directly from the source Parquet file.
"""

from __future__ import annotations

import datetime as dt
import sqlite3

import pandas as pd

from lsm.generate import SurveyResult
from lsm.logging_utils import get_logger

log = get_logger("lsm.ingest")


class IngestConflictError(Exception):
    """A survey with this (line_id, run_id) already exists with different content."""


def register_survey(conn: sqlite3.Connection, sr: SurveyResult, schema_version: int) -> str:
    """Insert the `survey` row. Returns 'accepted', 'noop' (idempotent re-register),
    or raises IngestConflictError. Never touches `reading` -- safe to call on data
    of unknown quality.
    """
    cur = conn.execute(
        "SELECT survey_id, content_sha256 FROM survey WHERE line_id=? AND run_id=?",
        (sr.line_id, sr.run_id),
    )
    row = cur.fetchone()
    if row is not None:
        existing_survey_id, existing_hash = row
        if existing_hash == sr.content_sha256:
            log.info(
                "register no-op: identical content already present",
                extra={"survey_id": existing_survey_id},
            )
            return "noop"
        raise IngestConflictError(
            f"{sr.line_id}/{sr.run_id} already registered as {existing_survey_id} "
            f"with content_sha256={existing_hash[:12]}..., but this file hashes to "
            f"{sr.content_sha256[:12]}... -- raw data is immutable, refusing to overwrite"
        )

    # No UNIQUE(content_sha256) guard here by design -- an exact-content
    # duplicate under a different (line_id, run_id) is deliberately allowed to
    # register, so that the `duplicate_content` DQ check can quarantine it
    # gracefully instead of this function raising an uncaught exception.
    conn.execute(
        """
        INSERT INTO survey (
            survey_id, line_id, run_id, surveyed_at, step_m, n_samples,
            chainage_start_m, chainage_end_m, standoff_m, schema_version,
            source_uri, file_sha256, content_sha256, status, ingested_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'accepted', ?)
        """,
        (
            sr.survey_id,
            sr.line_id,
            sr.run_id,
            sr.surveyed_at,
            sr.step_m,
            sr.n_samples,
            sr.chainage_start_m,
            sr.chainage_end_m,
            sr.standoff_m,
            schema_version,
            str(sr.path),
            sr.file_sha256,
            sr.content_sha256,
            dt.datetime.now(dt.timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    log.info("survey registered", extra={"survey_id": sr.survey_id})
    return "accepted"


def load_readings(conn: sqlite3.Connection, survey_id: str, df: pd.DataFrame) -> None:
    """Insert signal rows into `reading`. Caller's responsibility to have already
    confirmed there is no duplicate_sample_idx violation -- this will raise
    sqlite3.IntegrityError otherwise, by design (it should never be called on
    data that validation has flagged).

    Vectorized NaN->None + dtype-cast, not a per-row/per-cell Python loop --
    measured 235,982 rows/sec on the old itertuples()+float()-per-cell version
    at 80,000 rows (Stage 6 scale rehearsal), the real bottleneck this project's
    row-by-row conversion was; see docs/stage6-scale-rehearsal.md for the
    measured before/after.
    """
    cols = ["sample_idx", "lat", "lon", "bx_nt", "by_nt", "bz_nt", "bx2_nt", "by2_nt", "bz2_nt"]
    grad_cols = ["bx2_nt", "by2_nt", "bz2_nt"]
    prepared = df[cols].copy()
    # numpy.int64 is not a Python `int` subclass (unlike numpy.float64/`float`),
    # so sample_idx needs an explicit vectorized cast -- sqlite3 rejects it
    # silently-wrong otherwise on some driver/dtype combinations.
    prepared["sample_idx"] = prepared["sample_idx"].astype(int)
    prepared[grad_cols] = prepared[grad_cols].astype(object).where(prepared[grad_cols].notna(), None)
    rows = [(survey_id, *row) for row in prepared.to_numpy(dtype=object).tolist()]

    conn.executemany(
        """
        INSERT INTO reading (survey_id, sample_idx, lat, lon, bx_nt, by_nt, bz_nt,
                              bx2_nt, by2_nt, bz2_nt)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    log.info("readings loaded", extra={"survey_id": survey_id})
