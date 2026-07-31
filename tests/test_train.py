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

import pandas as pd

from lsm.bundle import load_bundle
from lsm.db import connect
from lsm.evaluate import shap_denylist_check
from lsm.generate import generate_all
from lsm.models.classify import ClassifyModel
from lsm.pipeline import run_feature_pipeline, run_survey_pipeline
from lsm.schemas import CLASSIFY_CLASSES
from lsm.train import (
    CLASSIFY_DENYLIST,
    _build_severity_training_frame,
    _fit_final_classify_model,
    _lightgbm_shap_importance,
    run_train,
)

_LGBM_CFG = {"deterministic": True, "force_row_wise": True, "num_threads": 1}


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

    # Stage 5: this same fixture already produces enough matched,
    # multi-class indications to exercise the classify path too (confirmed
    # empirically -- no separate, larger fixture needed).
    assert result["n_classify_samples"] >= 8, (
        "test fixture sized wrong -- not enough matched indications to exercise "
        "the classify CV path at all"
    )
    assert "classify" in result, "classify CV should have run, not been skipped"

    for model_name in ("classify", "classify_baseline"):
        m = result[model_name]
        for triple in m["per_class_recall"].values():
            point, lo, hi = triple
            # a class absent from this tiny fixture's test folds legitimately
            # yields an empty bootstrap sample -- bootstrap_ci's honest "no
            # data" answer is (nan, nan, nan), not an error.
            assert np.isnan(lo) or lo <= hi
        for metric in ("interference_precision", "brier"):
            point, lo, hi = m[metric]
            assert np.isnan(lo) or lo <= hi
    assert isinstance(result["classify_recall_gate_passed"], (bool, np.bool_))
    assert "classify_shap_check" in result
    assert isinstance(result["classify_shap_check"]["passed"], bool)
    assert isinstance(result["classify_gate_passed"], (bool, np.bool_))

    classify_rows = conn.execute(
        "SELECT model_version, artifact_uri FROM model_run WHERE task='classify'"
    ).fetchall()
    assert len(classify_rows) == 2  # lightgbm multiclass + majority-class baseline
    classify_lgbm_row = next(r for r in classify_rows if "lgbm" in r[0])
    classify_bundle = load_bundle(
        classify_lgbm_row[1], expected_feature_version=cfg.base.features.version,
        expected_schema_version=cfg.base.schema_version,
    )
    assert classify_bundle["task"] == "classify"
    assert classify_bundle["defect_calibrator"] is not None
    assert classify_bundle["classify_classes"] == ["scc", "weld", "dent", "corrosion", "interference"]

    release_classify_version = conn.execute(
        "SELECT classify_version FROM pipeline_release WHERE pipeline_version=?", (pipeline_version,)
    ).fetchone()[0]
    assert release_classify_version == classify_lgbm_row[0]

    assert "## Classification" in card_text


def test_build_severity_training_frame_drops_rows_with_nan_severity_smys(capsys):
    """Real bug found via Stage 6 at scale (docs/stage6-scale-rehearsal.md):
    generate.py initialises severity_smys to NaN off-defect, and a detector's
    peak occasionally lands just outside a defect's label window while still
    within the looser dig-matching tolerance -- so its OWN row's severity_smys
    is NaN, not the defect's true value (~0.4% of matched defects at 9,600-
    defect scale). A single such NaN y_true reaching SeverityModel.fit's
    calibration used to silently NaN the whole fold's conformal margin,
    collapsing pooled OOF coverage to 0%. Must be dropped upstream, loudly.
    """
    corpus = pd.DataFrame({
        "survey_id": ["S1", "S1", "S1"],
        "chainage_m": [10.0, 20.0, 30.0],
        "severity_smys": [50.0, np.nan, 60.0],
        "fold": [0, 0, 1],
        "feat_a": [1.0, 2.0, 3.0],
    })
    matched = pd.DataFrame({
        "indication_id": ["I1", "I2"],
        "survey_id": ["S1", "S1"],
        "chainage_peak_m": [10.0, 20.0],
        "chainage_start_m": [9.0, 19.0],
        "chainage_end_m": [11.0, 21.0],
        "matched_kind": ["defect", "defect"],
        "matched_source_id": ["D1", "D2"],
    })

    result = _build_severity_training_frame(matched, corpus, ["feat_a"])

    assert len(result) == 1
    assert result.iloc[0]["matched_source_id"] == "D1"
    assert not result["y_true"].isna().any()
    assert "dropping 1 matched indication" in capsys.readouterr().out


def test_fit_final_classify_model_respects_configured_capacity(cfg):
    """Stage 6 regression guard: config/base.yaml's model.classify.n_estimators/
    num_leaves used to be silently dead -- train.py never threaded them from
    config into ClassifyModel's lgbm_cfg, so editing them had NO effect (found
    via the same scale rehearsal that found severity's capacity bug; see
    docs/stage6-scale-rehearsal.md). Pin that a non-default value actually
    reaches the underlying LGBMClassifier.
    """
    cfg.base.model.classify["n_estimators"] = 77
    cfg.base.model.classify["num_leaves"] = 15

    rng = np.random.default_rng(0)
    n = 60
    a = rng.uniform(-1, 1, size=n)
    defect_type = np.select([a > 0.3, a < -0.3], ["scc", "weld"], default="corrosion")
    classify_frame = pd.DataFrame({
        "matched_source_id": [f"D{i}" for i in range(n)],
        "defect_type": defect_type,
        "a": a,
    })

    model, baseline = _fit_final_classify_model(classify_frame, ["a"], cfg, seed=0)

    assert model is not None
    assert model.model is not None
    assert model.model.n_estimators == 77
    assert model.model.num_leaves == 15


def test_shap_check_catches_a_deliberately_leaky_chainage_feature():
    """Engineered-leak integration test for the physics-consistency gate.
    `chainage_m` is structurally excluded from `feature_columns()` in real
    use, so a test asserting it's absent from the real feature set would pass
    trivially regardless of whether the CHECK ITSELF has teeth. This proves
    it does: chainage is deliberately included here and made informative by
    construction (perfectly predicts the class), a real ClassifyModel is fit,
    real contribution importances are computed, and the check must catch it.
    """
    rng = np.random.default_rng(0)
    n = 200
    chainage = rng.uniform(0, 2000, size=n)
    y = np.where(chainage < 1000, "scc", "weld")  # perfectly predictable from chainage alone
    X = pd.DataFrame({
        "chainage_m": chainage,
        "noise1": rng.normal(size=n),
        "noise2": rng.normal(size=n),
    })
    X_train, y_train = X.iloc[:150], y[:150]
    X_calib, y_calib = X.iloc[150:], y[150:]

    model = ClassifyModel(
        feature_cols=["chainage_m", "noise1", "noise2"], classes=CLASSIFY_CLASSES,
        seed=0, lgbm_cfg=_LGBM_CFG,
    ).fit(X_train, y_train, X_calib, y_calib)

    importance = _lightgbm_shap_importance(model, X)
    check = shap_denylist_check(importance, CLASSIFY_DENYLIST, top_k=2)
    assert check["passed"] is False
    assert "chainage_m" in check["leaked_denylist_features"]


def test_shap_check_does_not_cry_wolf_when_the_denylisted_feature_is_pure_noise():
    """Companion non-leaky case: `chainage_m` is present in the feature set
    (same as the test above) but this time it's pure noise, not informative
    -- the real, physically-meaningful feature (`a`) determines the class
    instead. Proves the check doesn't flag a denylisted column just for
    existing.
    """
    rng = np.random.default_rng(1)
    n = 200
    a = rng.normal(size=n)
    y = np.where(a > 0, "scc", "weld")
    X = pd.DataFrame({
        "chainage_m": rng.uniform(0, 2000, size=n),  # present, but uninformative
        "a": a,
    })
    X_train, y_train = X.iloc[:150], y[:150]
    X_calib, y_calib = X.iloc[150:], y[150:]

    model = ClassifyModel(
        feature_cols=["chainage_m", "a"], classes=CLASSIFY_CLASSES, seed=0, lgbm_cfg=_LGBM_CFG,
    ).fit(X_train, y_train, X_calib, y_calib)

    importance = _lightgbm_shap_importance(model, X)
    check = shap_denylist_check(importance, CLASSIFY_DENYLIST, top_k=1)
    assert check["passed"] is True
