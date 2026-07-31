"""Stage 8: growth.py -- partially-pooled log-linear growth rate,
remaining-life projection, and the 'no growth' baseline. Never `tiny_cfg`
(n_runs=1) -- growth needs multiple runs of the same line to have anything
to fit at all.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd

from lsm.bundle import load_bundle
from lsm.db import connect
from lsm.generate import generate_all
from lsm.growth import (
    build_growth_frame,
    evaluate_growth_gate,
    fit_defect_rates,
    fit_population_log_rate,
    no_growth_baseline_predict,
    project_remaining_life,
    run_forecast,
)
from lsm.pipeline import run_feature_pipeline, run_survey_pipeline
from lsm.train import run_train

_BOOT_CFG = SimpleNamespace(n_resamples=500, level=0.95)


def _growth_sized_cfg(cfg):
    """Same shape as test_train.py's _severity_sized_cfg -- 1 line x 3 runs
    x 10 defects, confirmed to produce enough matched, multi-run severity
    observations for growth fitting, not just severity's own CV path.
    """
    cfg.base.data.length_m = 500.0
    cfg.base.data.step_m = 1.0
    cfg.base.data.n_lines = 1
    cfg.base.data.n_runs = 3
    cfg.base.data.n_defects = 10
    cfg.base.data.n_interference = 3
    return cfg


def _build_and_train(cfg):
    results = generate_all(cfg.base.data, cfg.env.storage.raw_dir, seed=cfg.seed)
    conn = connect(cfg.env.storage.sqlite_path)
    for sr in results:
        _status, report = run_survey_pipeline(conn, sr, cfg)
        assert not report.has_fail
    for sr in results:
        outcome, _ = run_feature_pipeline(conn, sr.survey_id, cfg)
        assert outcome == "computed"
    run_train(cfg, conn)
    return conn


def test_population_log_rate_recovers_growth_law(cfg):
    cfg = _growth_sized_cfg(cfg)
    conn = _build_and_train(cfg)

    result = run_forecast(cfg, conn)

    assert abs(result["growth_population_log_rate"] - math.log(cfg.base.data.growth)) < 0.1


def test_single_observation_defects_are_fully_shrunk():
    growth_frame = pd.DataFrame([
        {"matched_source_id": "D0", "survey_id": "S0", "run_id": 0, "y_true": 30.0, "distance_m": 0.5},
        {"matched_source_id": "D0", "survey_id": "S1", "run_id": 1, "y_true": 34.5, "distance_m": 0.5},
        {"matched_source_id": "D0", "survey_id": "S2", "run_id": 2, "y_true": 39.675, "distance_m": 0.5},
        {"matched_source_id": "D1", "survey_id": "S0", "run_id": 0, "y_true": 50.0, "distance_m": 0.5},
    ])
    pop_rate, pop_var = fit_population_log_rate(growth_frame)
    rates = fit_defect_rates(growth_frame, pop_rate, pop_var, min_observations_for_own_rate=2)

    single = rates[rates["matched_source_id"] == "D1"].iloc[0]
    assert single["n_obs"] == 1
    assert single["shrinkage_weight"] == 0.0
    assert single["shrunk_rate"] == pop_rate
    assert np.isnan(single["own_rate"])

    multi = rates[rates["matched_source_id"] == "D0"].iloc[0]
    assert multi["n_obs"] == 3
    assert multi["shrinkage_weight"] > 0.0


def test_no_growth_baseline_gate(cfg):
    cfg = _growth_sized_cfg(cfg)
    conn = _build_and_train(cfg)
    result = run_forecast(cfg, conn)

    assert result["growth_gate_passed"] is True
    _point, lo, _hi = result["growth_gap_baseline_minus_model"]
    assert lo > 0.0  # the CI lower bound, not just the point estimate


def test_evaluate_growth_gate_on_synthetic_data_matching_the_generator_law():
    """A fast, direct unit test of the gate logic (no real pipeline run):
    synthetic defects following the EXACT generator law with a touch of
    noise -- the model should clearly beat the no-growth baseline on the
    held-out run.
    """
    rng = np.random.default_rng(1)
    rows = []
    for i in range(20):
        base = rng.uniform(20, 60)
        for run_id in range(3):
            sev = base * 1.15**run_id + rng.normal(0, 0.5)
            rows.append({
                "matched_source_id": f"D{i}", "survey_id": f"S_R{run_id}", "run_id": run_id,
                "y_true": sev, "distance_m": 0.5,
            })
    growth_frame = pd.DataFrame(rows)

    gate = evaluate_growth_gate(growth_frame, frozenset({0, 1}), 2, 2, _BOOT_CFG, seed=0)

    assert gate["gate_passed"] is True
    assert gate["gap_baseline_minus_model"][1] > 0.0
    assert abs(gate["population_log_rate"] - math.log(1.15)) < 0.05
    assert gate["n_defects_evaluated"] == 20


def test_no_growth_baseline_predicts_unchanged_severity():
    last = np.array([10.0, 20.0, 30.0])
    predicted = no_growth_baseline_predict(last)
    np.testing.assert_array_equal(predicted, last)


def test_match_residual_is_recorded_and_finite(cfg):
    cfg = _growth_sized_cfg(cfg)
    conn = _build_and_train(cfg)

    import datetime as dt
    from pathlib import Path

    from lsm import train as train_module
    from lsm.bundle import load_bundle as _load_bundle
    from lsm.features import load_feature_corpus
    from lsm.predict import latest_pipeline_release

    as_of = dt.datetime.now(dt.UTC).isoformat()
    corpus = load_feature_corpus(cfg.env.storage.feature_dir, cfg.base.features.version, as_of=as_of)
    survey_ids = sorted(corpus["survey_id"].unique())
    truth = train_module._load_truth_and_geometry(conn, survey_ids)
    corpus = corpus.merge(truth, on=["survey_id", "sample_idx"], how="left", validate="one_to_one")
    line_ids = sorted(corpus["line_id"].unique())
    registries = []
    for line_id in line_ids:
        ref_survey_id = min(corpus.loc[corpus["line_id"] == line_id, "survey_id"].unique())
        ref_rows = corpus[corpus["survey_id"] == ref_survey_id]
        registries.append(train_module.build_truth_registry(ref_rows, line_id))
    registry = pd.concat(registries, ignore_index=True)

    # Reuse the actually-released anomaly bundle to score+cluster, matching
    # run_forecast's own path exactly (score_if only exists inside the CV loop).
    _, anomaly_version, _, _ = latest_pipeline_release(conn)
    anomaly_bundle = _load_bundle(
        Path(cfg.env.storage.model_dir) / anomaly_version / "bundle.joblib",
        expected_feature_version=cfg.base.features.version, expected_schema_version=cfg.base.schema_version,
    )
    corpus["_score"] = anomaly_bundle["model"].score(corpus)
    all_indications = train_module._cluster_all_indications(corpus, "_score", anomaly_bundle["threshold"])
    matched_all = train_module._match_all_indications(all_indications, corpus, registry)

    growth_frame = build_growth_frame(matched_all, corpus)
    assert len(growth_frame) > 0
    assert np.isfinite(growth_frame["distance_m"]).all()
    assert (growth_frame["distance_m"] >= 0).all()


def test_remaining_life_interval_is_monotonic():
    life_lo, life_med, life_hi = project_remaining_life((40.0, 50.0, 60.0), 0.1, 100.0)
    assert life_lo <= life_med <= life_hi


def test_remaining_life_is_zero_at_or_past_the_limit_state():
    life_lo, life_med, life_hi = project_remaining_life((100.0, 110.0, 120.0), 0.1, 100.0)
    assert life_lo == life_med == life_hi == 0.0


def test_remaining_life_is_infinite_at_zero_or_negative_rate():
    life_lo, life_med, life_hi = project_remaining_life((40.0, 50.0, 60.0), 0.0, 100.0)
    assert life_lo == life_med == life_hi == float("inf")


def test_forecast_persists_growth_bundle_and_release(cfg):
    cfg = _growth_sized_cfg(cfg)
    conn = _build_and_train(cfg)
    run_forecast(cfg, conn)

    rows = conn.execute("SELECT model_version, artifact_uri FROM model_run WHERE task='growth'").fetchall()
    assert len(rows) == 1
    growth_version, artifact_uri = rows[0]

    bundle = load_bundle(
        artifact_uri, expected_feature_version=cfg.base.features.version, expected_schema_version=cfg.base.schema_version,
    )
    assert bundle["task"] == "growth"
    assert "population_log_rate" in bundle
    assert "defect_rates" in bundle

    pipeline_version, growth_version_released = conn.execute(
        "SELECT pipeline_version, growth_version FROM pipeline_release ORDER BY released_at DESC LIMIT 1"
    ).fetchone()
    assert growth_version_released == growth_version

    from pathlib import Path
    report_path = Path(cfg.env.storage.reports_dir) / pipeline_version / "growth_forecast.parquet"
    assert report_path.exists()

    model_card_path = Path(cfg.env.storage.model_dir) / pipeline_version / "model_card.md"
    card_text = model_card_path.read_text(encoding="utf-8")
    assert "## Growth" in card_text


def test_as_of_flag_restricts_which_runs_are_used(cfg):
    cfg = _growth_sized_cfg(cfg)
    conn = _build_and_train(cfg)

    # Backdate surveyed_at so run 2 is clearly "in the future" relative to a
    # chosen as_of cut -- generate_all stamps every run with the SAME
    # wall-clock timestamp by default, so this must be done explicitly.
    conn.execute("UPDATE survey SET surveyed_at='2024-01-01T00:00:00+00:00' WHERE run_id=0")
    conn.execute("UPDATE survey SET surveyed_at='2025-01-01T00:00:00+00:00' WHERE run_id=1")
    conn.execute("UPDATE survey SET surveyed_at='2026-01-01T00:00:00+00:00' WHERE run_id=2")
    conn.commit()
    # meta.json (read by load_feature_corpus) is keyed off the survey row at
    # feature-compute time -- recompute features so the backdated
    # surveyed_at actually propagates into the feature store's own metadata.
    from lsm.pipeline import run_feature_pipeline as _run_feature_pipeline
    survey_ids = [r[0] for r in conn.execute("SELECT survey_id FROM survey").fetchall()]
    for survey_id in survey_ids:
        _run_feature_pipeline(conn, survey_id, cfg, force=True)

    restricted = run_forecast(cfg, conn, as_of="2025-06-01T00:00:00+00:00")
    unrestricted = run_forecast(cfg, conn, as_of=None)

    assert restricted["n_growth_samples"] < unrestricted["n_growth_samples"]
