"""
Stage 1.5 gate: a partitioned backfill runs, can be interrupted mid-flight and
resumed without reprocessing completed partitions or duplicating rows, and one
deliberately corrupt survey quarantines itself while the others succeed.

Scaled to 4 partitions for test runtime; the mechanism (partition backfill,
resumability via idempotent register/load, per-partition quarantine) is
identical at the 120-partition scale described in PLAN.md Stage 1.5 -- only
the count differs. That larger run belongs in Stage 6's scale rehearsal.
"""

from __future__ import annotations

import sqlite3

import dagster as dg
import pandas as pd
import pytest

from lsm.dagster_defs import LsmResource, dq_report, registered_survey, survey_features
from lsm.features import feature_store_dir
from lsm.generate import generate_all


@pytest.fixture
def backfill_setup(cfg, tmp_path):
    """4 independent lines (1 run each) = 4 survey partitions. Corrupt one."""
    cfg.base.data.length_m = 200.0
    cfg.base.data.step_m = 1.0
    cfg.base.data.n_lines = 4
    cfg.base.data.n_runs = 1
    cfg.base.data.n_defects = 2
    cfg.base.data.n_interference = 1

    raw_dir = tmp_path / "raw"
    results = generate_all(cfg.base.data, raw_dir, seed=42)
    survey_ids = [r.survey_id for r in results]

    # Corrupt one survey: an out-of-range field value -> hard fail on `range`.
    corrupt_sr = results[2]
    df = pd.read_parquet(corrupt_sr.path)
    df.loc[5, "bx_nt"] = 999_999.0
    df.to_parquet(corrupt_sr.path, index=False, compression="zstd")

    instance = dg.DagsterInstance.ephemeral()
    instance.add_dynamic_partitions("survey_id", survey_ids)

    lsm_resource = LsmResource(env_name="dev", config_dir=str(tmp_path / "config"))

    return {
        "instance": instance,
        "lsm_resource": lsm_resource,
        "survey_ids": survey_ids,
        "corrupt_survey_id": corrupt_sr.survey_id,
        "sqlite_path": cfg.env.storage.sqlite_path,
        "n_samples": results[0].n_samples,
    }


def _materialize_partition(setup, survey_id: str):
    return dg.materialize(
        [registered_survey, dq_report, survey_features],
        partition_key=survey_id,
        instance=setup["instance"],
        resources={"lsm": setup["lsm_resource"]},
    )


def test_backfill_kill_and_resume_no_duplication(backfill_setup):
    survey_ids = backfill_setup["survey_ids"]

    # "Kill mid-flight": only process the first 2 of 4 partitions.
    for sid in survey_ids[:2]:
        result = _materialize_partition(backfill_setup, sid)
        assert result.success

    # "Resume": process the remaining partitions.
    for sid in survey_ids[2:]:
        result = _materialize_partition(backfill_setup, sid)
        assert result.success

    conn = sqlite3.connect(backfill_setup["sqlite_path"])
    n_surveys = conn.execute("SELECT COUNT(*) FROM survey").fetchone()[0]
    assert n_surveys == 4

    # One corrupted, three clean.
    statuses = dict(conn.execute("SELECT survey_id, status FROM survey").fetchall())
    corrupt_id = backfill_setup["corrupt_survey_id"]
    assert statuses[corrupt_id] == "quarantined"
    clean_ids = [sid for sid in survey_ids if sid != corrupt_id]
    for sid in clean_ids:
        assert statuses[sid] == "accepted"

    # Corrupted survey's readings were never loaded; clean ones were, exactly once.
    n_readings_corrupt = conn.execute(
        "SELECT COUNT(*) FROM reading WHERE survey_id=?", (corrupt_id,)
    ).fetchone()[0]
    assert n_readings_corrupt == 0
    for sid in clean_ids:
        n = conn.execute(
            "SELECT COUNT(*) FROM reading WHERE survey_id=?", (sid,)
        ).fetchone()[0]
        assert n == backfill_setup["n_samples"]

    # Re-materialize an already-clean partition ("backfill touches a done
    # partition again") -- must not duplicate rows.
    already_done = clean_ids[0]
    result = _materialize_partition(backfill_setup, already_done)
    assert result.success
    n_after = conn.execute(
        "SELECT COUNT(*) FROM reading WHERE survey_id=?", (already_done,)
    ).fetchone()[0]
    assert n_after == backfill_setup["n_samples"]  # unchanged, not doubled

    conn.close()


def test_quarantine_files_written_for_corrupt_partition(backfill_setup, tmp_path):
    for sid in backfill_setup["survey_ids"]:
        result = _materialize_partition(backfill_setup, sid)
        assert result.success

    corrupt_id = backfill_setup["corrupt_survey_id"]
    quarantine_dir = tmp_path / "quarantine" / corrupt_id
    assert (quarantine_dir / "survey.parquet").exists()
    assert (quarantine_dir / "dq.json").exists()

    import json

    dq = json.loads((quarantine_dir / "dq.json").read_text())
    fail_checks = [r["check_name"] for r in dq["results"] if r["status"] == "fail"]
    assert "range" in fail_checks


def test_features_materialise_for_clean_partitions_only(backfill_setup, cfg, tmp_path):
    """The quarantine boundary holds all the way downstream. A survey that failed
    a hard DQ gate must produce no features -- and the partition must still
    SUCCEED, because "we correctly refused this survey" is not a pipeline failure
    and treating it as one is how a nightly run gets switched off.
    """
    for sid in backfill_setup["survey_ids"]:
        assert _materialize_partition(backfill_setup, sid).success

    fv = cfg.base.features.version
    corrupt_id = backfill_setup["corrupt_survey_id"]
    for sid in backfill_setup["survey_ids"]:
        line_id, run_id = sid.rsplit("_R", 1)
        path = feature_store_dir(tmp_path / "features", fv, line_id, int(run_id)) / "features.parquet"
        assert path.exists() is (sid != corrupt_id), sid

    clean_id = next(s for s in backfill_setup["survey_ids"] if s != corrupt_id)
    line_id, run_id = clean_id.rsplit("_R", 1)
    feats = pd.read_parquet(
        feature_store_dir(tmp_path / "features", fv, line_id, int(run_id)) / "features.parquet"
    )
    assert len(feats) == backfill_setup["n_samples"]
    assert (feats["feature_version"] == fv).all()
