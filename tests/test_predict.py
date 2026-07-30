"""Stage 3/4: predict.py -- score a survey against the latest released model(s)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lsm.db import connect
from lsm.generate import generate_all
from lsm.pipeline import run_feature_pipeline, run_survey_pipeline
from lsm.predict import (
    NoReleasedPipelineError,
    SurveyNotScorableError,
    latest_pipeline_release,
    predict_survey,
)
from lsm.train import run_train


def _train_tiny(tiny_cfg):
    results = generate_all(tiny_cfg.base.data, tiny_cfg.env.storage.raw_dir, seed=tiny_cfg.seed)
    conn = connect(tiny_cfg.env.storage.sqlite_path)
    for sr in results:
        run_survey_pipeline(conn, sr, tiny_cfg)
    for sr in results:
        run_feature_pipeline(conn, sr.survey_id, tiny_cfg)
    run_train(tiny_cfg, conn)
    return conn, results


def test_predict_survey_writes_indications_referencing_the_latest_release(tiny_cfg):
    conn, results = _train_tiny(tiny_cfg)
    survey_id = results[0].survey_id

    indications = predict_survey(conn, survey_id, tiny_cfg)

    pipeline_version, _, _ = latest_pipeline_release(conn)
    if len(indications) > 0:
        assert (indications["pipeline_version"] == pipeline_version).all()
        assert (indications["survey_id"] == survey_id).all()

    stored = conn.execute(
        "SELECT COUNT(*) FROM indication WHERE survey_id=?", (survey_id,)
    ).fetchone()[0]
    assert stored == len(indications)


def test_predict_survey_is_idempotent_on_rerun(tiny_cfg):
    conn, results = _train_tiny(tiny_cfg)
    survey_id = results[0].survey_id

    predict_survey(conn, survey_id, tiny_cfg)
    n_first = conn.execute(
        "SELECT COUNT(*) FROM indication WHERE survey_id=?", (survey_id,)
    ).fetchone()[0]

    predict_survey(conn, survey_id, tiny_cfg)  # re-run: INSERT OR IGNORE, same indication_ids
    n_second = conn.execute(
        "SELECT COUNT(*) FROM indication WHERE survey_id=?", (survey_id,)
    ).fetchone()[0]

    assert n_first == n_second


def test_predict_survey_raises_before_any_training(tiny_cfg):
    results = generate_all(tiny_cfg.base.data, tiny_cfg.env.storage.raw_dir, seed=tiny_cfg.seed)
    conn = connect(tiny_cfg.env.storage.sqlite_path)
    for sr in results:
        run_survey_pipeline(conn, sr, tiny_cfg)
    for sr in results:
        run_feature_pipeline(conn, sr.survey_id, tiny_cfg)

    with pytest.raises(NoReleasedPipelineError):
        predict_survey(conn, results[0].survey_id, tiny_cfg)


def test_predict_survey_raises_on_unregistered_survey(tiny_cfg):
    conn, _ = _train_tiny(tiny_cfg)
    with pytest.raises(SurveyNotScorableError):
        predict_survey(conn, "NOT_A_REAL_SURVEY", tiny_cfg)


def test_predict_survey_attaches_severity_and_writes_geojson(cfg):
    """tiny_cfg never produces enough matched indications for a severity model
    to exist at all (see test_train.py's _severity_sized_cfg) -- this exercises
    the actual attach_severity + GeoJSON export code paths, not just their
    absence.
    """
    cfg.base.data.length_m = 500.0
    cfg.base.data.step_m = 1.0
    cfg.base.data.n_lines = 1
    cfg.base.data.n_runs = 3
    cfg.base.data.n_defects = 10
    cfg.base.data.n_interference = 3

    results = generate_all(cfg.base.data, cfg.env.storage.raw_dir, seed=cfg.seed)
    conn = connect(cfg.env.storage.sqlite_path)
    for sr in results:
        run_survey_pipeline(conn, sr, cfg)
    for sr in results:
        run_feature_pipeline(conn, sr.survey_id, cfg)
    run_train(cfg, conn)

    pipeline_version, _, severity_version = latest_pipeline_release(conn)
    assert severity_version is not None, "test fixture sized wrong -- no severity model trained"

    survey_id = results[0].survey_id
    indications = predict_survey(conn, survey_id, cfg)

    if len(indications) > 0:
        assert indications["interval_nominal"].notna().any()
        assert indications["sev_pred"].notna().any()

    geojson_path = Path(cfg.env.storage.reports_dir) / survey_id / "indications.geojson"
    assert geojson_path.exists()
    geojson = json.loads(geojson_path.read_text(encoding="utf-8"))
    assert geojson["type"] == "FeatureCollection"
    if len(indications) > 0:
        assert len(geojson["features"]) > 0
        feature = geojson["features"][0]
        assert feature["geometry"]["type"] == "Point"
        assert "sev_pred" in feature["properties"]
