"""
SQLite schema and connection helpers. Matches .claude/skills/lsm-integrity/
references/architecture.md -- read that before changing this DDL.

Concurrency: WAL mode, PRAGMA foreign_keys=ON. SQLite allows exactly one writer
at a time -- registry writes are serialised through a single connection in the
Dagster resource (see dagster_defs.py); feature computation fans out separately.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS survey (
  survey_id        TEXT PRIMARY KEY,
  line_id          TEXT NOT NULL,
  run_id           INTEGER NOT NULL,
  surveyed_at      TEXT NOT NULL,
  step_m           REAL NOT NULL,
  n_samples        INTEGER NOT NULL,
  chainage_start_m REAL NOT NULL,
  chainage_end_m   REAL NOT NULL,
  standoff_m       REAL NOT NULL,
  schema_version   INTEGER NOT NULL,
  source_uri       TEXT NOT NULL,
  file_sha256      TEXT NOT NULL,
  content_sha256   TEXT NOT NULL,
  status           TEXT NOT NULL DEFAULT 'accepted',
  ingested_at      TEXT NOT NULL,
  UNIQUE (line_id, run_id)
);
-- Deliberately NO UNIQUE(content_sha256): an exact-content duplicate under a
-- different (line_id, run_id) must be quarantined gracefully by the
-- `duplicate_content` DQ check (validate.py), not crash register_survey with
-- an uncaught IntegrityError -- "quarantine, not crash" applies here too.

-- Rig-v2 (schema_version 3): bx/by/bz(+bx2/by2/bz2) are gone -- the real rig
-- has three total-field HEADS (b_lo/b_mid/b_hi), never a vector reading (see
-- schemas.py). lat/lon are nullable now: GPS dropout is a real, expected
-- acquisition state (GpsConfig), not corrupt data. t_s, girth_weld and
-- chainage_true_m are new -- chainage_true_m is TRUTH TIER, stored directly
-- on `reading` the same way severity_smys already is (nullable there, NOT
-- NULL here since the generator always knows it for every row). There is
-- deliberately no registered-chainage column here: chainage_m is a
-- FEATURE-layer output (registration.py, Stage B), never written to raw.
CREATE TABLE IF NOT EXISTS reading (
  survey_id  TEXT    NOT NULL REFERENCES survey(survey_id),
  sample_idx INTEGER NOT NULL,
  t_s REAL NOT NULL,
  lat REAL, lon REAL,
  b_lo_nt REAL NOT NULL, b_mid_nt REAL NOT NULL, b_hi_nt REAL NOT NULL,
  girth_weld INTEGER NOT NULL,
  chainage_true_m REAL NOT NULL,
  PRIMARY KEY (survey_id, sample_idx)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS defect (
  defect_id TEXT PRIMARY KEY,
  line_id   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS truth_defect (
  defect_id   TEXT NOT NULL REFERENCES defect(defect_id),
  revision    INTEGER NOT NULL,
  chainage_m  REAL NOT NULL,
  defect_type TEXT NOT NULL,
  source      TEXT NOT NULL,
  verified_at TEXT,
  valid_from  TEXT NOT NULL,
  valid_to    TEXT,
  PRIMARY KEY (defect_id, revision)
);

CREATE TABLE IF NOT EXISTS truth_observation (
  defect_id     TEXT NOT NULL REFERENCES defect(defect_id),
  survey_id     TEXT NOT NULL REFERENCES survey(survey_id),
  severity_smys REAL NOT NULL,
  PRIMARY KEY (defect_id, survey_id)
);

CREATE TABLE IF NOT EXISTS dq_report (
  dq_id      INTEGER PRIMARY KEY AUTOINCREMENT,
  survey_id  TEXT NOT NULL,
  checked_at TEXT NOT NULL,
  check_name TEXT NOT NULL,
  status     TEXT NOT NULL,
  n_affected INTEGER NOT NULL,
  detail_json TEXT
);

CREATE TABLE IF NOT EXISTS model_run (
  model_version   TEXT PRIMARY KEY,
  task            TEXT NOT NULL,
  mlflow_run_id   TEXT NOT NULL,
  git_sha         TEXT NOT NULL,
  config_sha256   TEXT NOT NULL,
  data_sha256     TEXT NOT NULL,
  feature_version INTEGER NOT NULL,
  truth_as_of     TEXT NOT NULL,
  trained_at      TEXT NOT NULL,
  metrics_json    TEXT NOT NULL,
  artifact_uri    TEXT NOT NULL,
  final_test_uses INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS pipeline_release (
  pipeline_version  TEXT PRIMARY KEY,
  anomaly_version   TEXT REFERENCES model_run(model_version),
  severity_version  TEXT REFERENCES model_run(model_version),
  classify_version  TEXT REFERENCES model_run(model_version),
  growth_version    TEXT REFERENCES model_run(model_version),
  feature_version   INTEGER NOT NULL,
  schema_version    INTEGER NOT NULL,
  container_digest  TEXT NOT NULL,
  released_at       TEXT NOT NULL,
  alias             TEXT
);

CREATE TABLE IF NOT EXISTS indication (
  indication_id    TEXT PRIMARY KEY,
  survey_id        TEXT NOT NULL REFERENCES survey(survey_id),
  pipeline_version TEXT NOT NULL REFERENCES pipeline_release(pipeline_version),
  is_shadow        INTEGER NOT NULL DEFAULT 0,
  chainage_peak_m  REAL NOT NULL,
  chainage_start_m REAL NOT NULL,
  chainage_end_m   REAL NOT NULL,
  lat REAL, lon REAL,
  anomaly_score    REAL NOT NULL,
  p_defect_cal     REAL NOT NULL,
  pred_type TEXT, pred_type_conf REAL,
  sev_pred REAL, sev_lo REAL, sev_hi REAL, interval_nominal REAL,
  risk_score       REAL,
  dq_flag          TEXT NOT NULL,
  created_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_indication_survey
  ON indication(survey_id, pipeline_version, risk_score DESC);
CREATE INDEX IF NOT EXISTS ix_dq_survey ON dq_report(survey_id, status);
"""


def connect(sqlite_path: str | Path, check_same_thread: bool = True) -> sqlite3.Connection:
    """`check_same_thread=False` is for callers that cache this connection
    across a threaded reuse they don't control -- Streamlit reruns a cached
    resource on whatever thread its runtime picks, and sqlite3 refuses cross-
    thread use by default. Safe here because this project already serialises
    writes at the application level (WAL + a single Dagster resource, per
    architecture.md) -- the thread guard was never the thing preventing
    concurrent-write corruption.
    """
    path = Path(sqlite_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=check_same_thread)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.executescript(DDL)
    conn.commit()
    return conn
