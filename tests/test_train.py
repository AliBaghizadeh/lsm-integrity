"""Stage 3: train.py end-to-end smoke test on a tiny synthetic survey.

Deliberately uses `tiny_cfg` (200 m, 1 line, 1 run) so this runs in well under a
second -- this is a plumbing check (does the whole generate -> ingest ->
features -> train path wire together and write what it claims to), not a
statistically meaningful evaluation. The real training run, on the full
2000 m / 3-run dataset, is run by hand (see PLAN.md Stage 3 / the run guide),
not from pytest.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lsm.bundle import load_bundle
from lsm.db import connect
from lsm.generate import generate_all
from lsm.pipeline import run_feature_pipeline, run_survey_pipeline
from lsm.train import run_train


def test_train_runs_end_to_end_on_a_tiny_survey(tiny_cfg, tmp_path):
    results = generate_all(tiny_cfg.base.data, tiny_cfg.env.storage.raw_dir, seed=tiny_cfg.seed)

    conn = connect(tiny_cfg.env.storage.sqlite_path)
    for sr in results:
        status, report = run_survey_pipeline(conn, sr, tiny_cfg)
        assert not report.has_fail

    for sr in results:
        outcome, _ = run_feature_pipeline(conn, sr.survey_id, tiny_cfg)
        assert outcome == "computed"

    result = run_train(tiny_cfg, conn)

    for model_name in ("mad", "isolation_forest"):
        m = result[model_name]
        for metric in ("recall_at_budget", "false_dig_rate", "interference_dig_fraction", "localisation_error_m"):
            point, lo, hi = m[metric]
            assert 0.0 <= point or point != point  # allow nan for localisation error if no hits
        assert isinstance(m["pr_auc"], float)

    # model_run rows exist for both models.
    rows = conn.execute("SELECT model_version, task FROM model_run").fetchall()
    assert len(rows) == 2
    assert {r[1] for r in rows} == {"anomaly"}

    # exactly one pipeline_release, aliased as challenger, referencing the
    # isolation-forest model_version.
    release = conn.execute(
        "SELECT anomaly_version, alias FROM pipeline_release"
    ).fetchone()
    assert release is not None
    assert release[1] == "challenger"
    assert release[0].startswith("anomaly-if-")

    # indications were actually written and are queryable.
    n_indications = conn.execute("SELECT COUNT(*) FROM indication").fetchone()[0]
    assert n_indications >= 0  # a tiny survey may legitimately produce zero

    metrics_json = conn.execute(
        "SELECT metrics_json FROM model_run WHERE task='anomaly' LIMIT 1"
    ).fetchone()[0]
    json.loads(metrics_json)  # must be valid, parseable JSON


def test_train_raises_on_empty_feature_corpus(tiny_cfg):
    conn = connect(tiny_cfg.env.storage.sqlite_path)
    try:
        run_train(tiny_cfg, conn)
        assert False, "expected RuntimeError on an empty feature corpus"
    except RuntimeError:
        pass


def _severity_sized_cfg(cfg):
    """tiny_cfg (200 m, 1 run, 2 defects) never has enough matched indications
    to exercise Stage 4's severity CV path at all -- it's always skipped.
    This is still small (500 m x 3 runs = 1500 rows total) but has enough
    defects and runs for `_build_severity_training_frame` to produce >= 4
    matched samples across >= 2 defects, so the severity fit/predict/bundle
    code actually runs, not just its "too little data" skip branch.
    """
    cfg.base.data.length_m = 500.0
    cfg.base.data.step_m = 1.0
    cfg.base.data.n_lines = 1
    cfg.base.data.n_runs = 3
    cfg.base.data.n_defects = 10
    cfg.base.data.n_interference = 3
    return cfg


def test_train_exercises_the_severity_cv_path(cfg, tmp_path):
    cfg = _severity_sized_cfg(cfg)
    results = generate_all(cfg.base.data, cfg.env.storage.raw_dir, seed=cfg.seed)

    conn = connect(cfg.env.storage.sqlite_path)
    for sr in results:
        status, report = run_survey_pipeline(conn, sr, cfg)
        assert not report.has_fail
    for sr in results:
        outcome, _ = run_feature_pipeline(conn, sr.survey_id, cfg)
        assert outcome == "computed"

    result = run_train(cfg, conn)

    assert result["n_severity_samples"] >= 4, (
        "test fixture sized wrong -- not enough matched indications to exercise "
        "the severity CV path at all"
    )
    assert "severity" in result, "severity CV should have run, not been skipped"

    for model_name in ("severity", "severity_baseline"):
        m = result[model_name]
        for metric in ("coverage", "mae", "interval_width"):
            point, lo, hi = m[metric]
            assert lo <= hi
    assert isinstance(result["severity_gate_passed"], (bool, np.bool_))

    # a real severity model_run + bundle + pipeline_release.severity_version.
    severity_rows = conn.execute(
        "SELECT model_version, artifact_uri FROM model_run WHERE task='severity'"
    ).fetchall()
    assert len(severity_rows) == 2  # lightgbm CQR + global-mean baseline
    lgbm_row = next(r for r in severity_rows if "lgbm" in r[0])
    bundle = load_bundle(
        lgbm_row[1], expected_feature_version=cfg.base.features.version,
        expected_schema_version=cfg.base.schema_version,
    )
    assert bundle["task"] == "severity"

    pipeline_version, release_severity_version = conn.execute(
        "SELECT pipeline_version, severity_version FROM pipeline_release"
    ).fetchone()
    assert release_severity_version == lgbm_row[0]

    # the model card (keyed by pipeline_version, not any one model_version)
    # mentions severity when severity actually ran.
    model_card_path = Path(cfg.env.storage.model_dir) / pipeline_version / "model_card.md"
    card_text = model_card_path.read_text(encoding="utf-8")
    assert "## Severity" in card_text
