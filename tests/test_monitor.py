"""Stage 8: monitor.py -- PSI/KS feature drift, prediction drift, background
regime shift, and coverage tracking, against a released pipeline's stored
bundle reference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lsm.db import connect
from lsm.generate import _build_features, _make_run, generate_all, load_survey_result
from lsm.monitor import monitor_report_to_text, monitor_survey
from lsm.pipeline import run_feature_pipeline, run_survey_pipeline
from lsm.predict import predict_survey
from lsm.train import run_train


def _growth_sized_cfg(cfg):
    cfg.base.data.length_m = 500.0
    cfg.base.data.step_m = 1.0
    cfg.base.data.n_lines = 1
    cfg.base.data.n_runs = 3
    cfg.base.data.n_defects = 10
    cfg.base.data.n_interference = 3
    return cfg


def _train_a_pipeline(cfg):
    results = generate_all(cfg.base.data, cfg.env.storage.raw_dir, seed=cfg.seed)
    conn = connect(cfg.env.storage.sqlite_path)
    for sr in results:
        status, report = run_survey_pipeline(conn, sr, cfg)
        assert not report.has_fail
    for sr in results:
        outcome, _ = run_feature_pipeline(conn, sr.survey_id, cfg)
        assert outcome == "computed"
    run_train(cfg, conn)
    return conn


def _write_and_predict_shifted_survey(cfg, conn, line_id="LINE000", run_id=3, bx_offset=300.0):
    """A 4th run of the ALREADY-TRAINED line, with a deliberately shifted DC
    background on bx_nt -- a background-regime check needs same-line HISTORY
    to compare against (background_regime_shift, same as validate.py's own
    check_background_regime, "passes" on insufficient history), so this must
    extend the trained line, not introduce a brand-new one with no history.
    Bypasses generate_all (which always starts line_idx from 0, so a second
    call would collide with LINE000) by calling the lower-level generation
    functions directly with the SAME per-line seed generate_all itself uses
    for line_idx=0, exactly the way tests/conftest.py's own
    `generate_one_survey` helper post-processes an already-generated file.
    """
    data_cfg = cfg.base.data.model_copy(deep=True)
    data_cfg.background_nT = [data_cfg.background_nT[0] + bx_offset, *data_cfg.background_nT[1:]]

    rng = np.random.default_rng(cfg.seed)  # matches generate_all's line_idx=0 seed
    features = _build_features(data_cfg, rng)
    # Advance the rng past the same draws generate_all's own 3-run training
    # loop already consumed, so this "4th run" gets a genuinely fresh
    # background draw -- not a replay of run 0's exact background (which
    # would trip check_survey_overlap on the shared defect signal PLUS a
    # near-identical background, an artifact of this test's own rng reuse,
    # not the deliberate shift being tested).
    for prior_run_id in range(cfg.base.data.n_runs):
        _make_run(line_id, prior_run_id, data_cfg, features, rng)
    df = _make_run(line_id, run_id, data_cfg, features, rng)

    write_cols = [
        "sample_idx", "chainage_m", "lat", "lon", "bx_nt", "by_nt", "bz_nt",
        "bx2_nt", "by2_nt", "bz2_nt", "defect", "defect_type", "severity_smys",
        "interference", "line_id", "run_id",
    ]
    out_dir = Path(cfg.env.storage.raw_dir) / f"line_id={line_id}" / f"run_id={run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "survey.parquet"
    df[write_cols].sort_values("sample_idx").to_parquet(out_path, index=False, compression="zstd")

    sr = load_survey_result(out_path, data_cfg.step_m, data_cfg.depth_m)
    status, report = run_survey_pipeline(conn, sr, cfg)
    outcome, _ = run_feature_pipeline(conn, sr.survey_id, cfg)
    assert outcome == "computed"
    predict_survey(conn, sr.survey_id, cfg)
    return sr.survey_id


def test_drift_monitor_fires_on_a_survey_with_a_shifted_background(cfg):
    cfg = _growth_sized_cfg(cfg)
    conn = _train_a_pipeline(cfg)

    survey_id = _write_and_predict_shifted_survey(cfg, conn, bx_offset=300.0)

    report = monitor_survey(conn, survey_id, cfg)

    assert report["status"] in ("warn", "block")
    assert report["background_regime"]["n_bad"] > 0
    text = monitor_report_to_text(report)
    assert survey_id in text


def test_monitor_report_to_text_renders_without_error(cfg):
    cfg = _growth_sized_cfg(cfg)
    conn = _train_a_pipeline(cfg)
    survey_id = _write_and_predict_shifted_survey(cfg, conn, bx_offset=0.0)

    report = monitor_survey(conn, survey_id, cfg)
    text = monitor_report_to_text(report)
    assert "Stage 8: monitor" in text
    assert "Overall status" in text
