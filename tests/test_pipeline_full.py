"""Stage 4.5: pipeline.run_full_pipeline -- the live-mode glue the demo app's
APP_MODE=live calls. register -> validate -> features -> score in one call,
degrading to "here's why it failed" (no exception) on a bad survey.
"""

from __future__ import annotations

from lsm.db import connect
from lsm.generate import generate_all
from lsm.pipeline import run_feature_pipeline, run_full_pipeline, run_survey_pipeline
from lsm.train import run_train


def _trained_conn_and_a_fresh_unregistered_survey(tiny_cfg):
    """Train on line 0, leave line 1 unregistered -- a genuinely fresh survey
    for run_full_pipeline to exercise end to end (same content_sha256 cannot
    be re-registered under a different line_id/run_id: raw data is immutable,
    keyed on (line_id, run_id), so a distinct line -- not just a new seed --
    is what "not yet registered" actually requires here).
    """
    tiny_cfg.base.data.n_lines = 2
    results = generate_all(tiny_cfg.base.data, tiny_cfg.env.storage.raw_dir, seed=tiny_cfg.seed)
    line0, line1 = results[0], results[1]

    conn = connect(tiny_cfg.env.storage.sqlite_path)
    _status, report = run_survey_pipeline(conn, line0, tiny_cfg)
    assert not report.has_fail
    run_feature_pipeline(conn, line0.survey_id, tiny_cfg)
    run_train(tiny_cfg, conn)  # real pipeline_release + bundles predict_survey needs

    return conn, line1


def test_run_full_pipeline_scores_a_clean_survey(tiny_cfg):
    conn, fresh = _trained_conn_and_a_fresh_unregistered_survey(tiny_cfg)

    report, indications = run_full_pipeline(conn, fresh, tiny_cfg)

    assert not report.has_fail
    assert indications is not None
    n_rows = conn.execute(
        "SELECT COUNT(*) FROM indication WHERE survey_id=?", (fresh.survey_id,)
    ).fetchone()[0]
    assert n_rows == len(indications)


def test_run_full_pipeline_refuses_a_corrupted_survey_without_raising(tiny_cfg, tmp_path):
    import pandas as pd

    from lsm.hashing import content_sha256, file_sha256

    conn = connect(tiny_cfg.env.storage.sqlite_path)
    sr = generate_all(tiny_cfg.base.data, tmp_path / "raw", seed=tiny_cfg.seed)[0]

    df = pd.read_parquet(sr.path)
    df.loc[3, "b_lo_nt"] = 999_999.0  # outside field_range_nT, trips check_range
    df.to_parquet(sr.path, index=False, compression="zstd")
    sr.file_sha256 = file_sha256(sr.path)
    sr.content_sha256 = content_sha256(df)

    report, indications = run_full_pipeline(conn, sr, tiny_cfg)

    assert report.has_fail
    assert indications is None
    assert any(r.check_name == "range" and r.status == "fail" for r in report.results)
    # features must never have been computed for a quarantined survey.
    assert not (
        conn.execute("SELECT 1 FROM reading WHERE survey_id=?", (sr.survey_id,)).fetchone()
    )


def test_run_full_pipeline_is_idempotent_on_rerun(tiny_cfg):
    conn, fresh = _trained_conn_and_a_fresh_unregistered_survey(tiny_cfg)

    _, first = run_full_pipeline(conn, fresh, tiny_cfg)
    _, second = run_full_pipeline(conn, fresh, tiny_cfg)

    n_rows = conn.execute(
        "SELECT COUNT(*) FROM indication WHERE survey_id=?", (fresh.survey_id,)
    ).fetchone()[0]
    assert n_rows == len(first) == len(second)
