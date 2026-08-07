"""
Stage 8: PSI/KS feature drift, prediction drift, background regime shift, and
coverage-vs-nominal, against a released pipeline's stored bundle reference.

A CONSUMER of `bundle.py`'s stored training-time reference -- never re-derives
its own "what training looked like" snapshot, per validation-and-trust.md
Layer 4 ("PSI/KS of each production feature against the bundle's stored
training reference").

Run: python -m lsm monitor <survey_id>
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from lsm.bundle import load_bundle
from lsm.config import Config
from lsm.dig_feedback import recompute_coverage_from_verifications
from lsm.evaluate import (
    background_regime_shift,
    feature_drift_report,
    prediction_drift_report,
)
from lsm.features import feature_store_dir
from lsm.predict import latest_pipeline_release


def monitor_survey(conn, survey_id: str, cfg: Config) -> dict:
    """Loads the latest released pipeline's anomaly bundle (feature/
    prediction-drift reference), the target (already-`lsm predict`-ed)
    survey's features and indications, and the line's other already-
    ingested surveys' raw-reading medians. Filters every query on
    `pipeline_version` (SKILL invariant #11). Returns `{"status":
    "ok"|"warn"|"block", "feature_drift", "prediction_drift",
    "background_regime", "coverage"}` -- status is the worst of any
    individual check. Only feature PSI crossing `psi_block` blocks
    (validation-and-trust.md Layer 4 states this explicitly for feature
    drift only); prediction drift, background regime, and stale coverage
    are "warn" signals, not automatic blocks.
    """
    row = conn.execute(
        "SELECT line_id, run_id, source_uri, status FROM survey WHERE survey_id=?", (survey_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"{survey_id} is not registered -- run `lsm ingest` first")
    line_id, run_id, _source_uri, status = row
    if status != "accepted":
        raise ValueError(f"{survey_id} has status={status!r}, cannot monitor it")

    feat_path = (
        feature_store_dir(cfg.env.storage.feature_dir, cfg.base.features.version, line_id, run_id)
        / "features.parquet"
    )
    if not feat_path.exists():
        raise FileNotFoundError(f"no features at {feat_path} -- run `lsm features` first")
    features = pd.read_parquet(feat_path)

    monitor_cfg = cfg.base.model.monitor
    pipeline_version, anomaly_version, _severity_version, _classify_version = latest_pipeline_release(conn)
    model_dir = Path(cfg.env.storage.model_dir)

    worst = "ok"

    def _bump(level: str) -> None:
        nonlocal worst
        order = {"ok": 0, "warn": 1, "block": 2}
        if order[level] > order[worst]:
            worst = level

    feature_drift = pd.DataFrame(columns=["feature", "psi", "ks_stat", "ks_pvalue", "status"])
    prediction_drift: dict = {}
    anomaly_bundle = load_bundle(
        model_dir / anomaly_version / "bundle.joblib",
        expected_feature_version=cfg.base.features.version, expected_schema_version=cfg.base.schema_version,
    )
    feature_cols = anomaly_bundle["feature_cols"]
    feature_drift = feature_drift_report(
        anomaly_bundle.get("training_feature_summary", {}), features, feature_cols,
        monitor_cfg["psi_warn"], monitor_cfg["psi_block"],
    )
    if (feature_drift["status"] == "block").any():
        _bump("block")
    elif (feature_drift["status"] == "warn").any():
        _bump("warn")

    indications = pd.read_sql_query(
        "SELECT chainage_peak_m, p_defect_cal FROM indication WHERE survey_id=? AND pipeline_version=?",
        conn, params=(survey_id, pipeline_version),
    )
    length_km = (features["chainage_m"].max() - features["chainage_m"].min()) / 1000.0
    current_indications_per_km = len(indications) / length_km if length_km > 0 else 0.0
    prediction_drift = prediction_drift_report(
        anomaly_bundle.get("training_prediction_reference", {}),
        indications["p_defect_cal"].to_numpy() if len(indications) else pd.Series([], dtype=float).to_numpy(),
        current_indications_per_km, monitor_cfg["indications_per_km_ratio_warn"],
    )
    if prediction_drift.get("flagged"):
        _bump("warn")

    prior_ids = [
        r[0] for r in conn.execute(
            "SELECT survey_id FROM survey WHERE line_id=? AND survey_id != ? AND status='accepted'",
            (line_id, survey_id),
        ).fetchall()
    ]
    # Rig-v2: compare all three heads' medians (b_lo/mid/hi_nt), matching
    # validate.py::check_background_regime's choice -- same per-column
    # z-score math (background_regime_shift), three heads instead of the
    # old bx/by/bz vector axes.
    heads = ["b_lo_nt", "b_mid_nt", "b_hi_nt"]
    medians = []
    for pid in prior_ids:
        prior = pd.read_sql_query(
            f"SELECT {', '.join(heads)} FROM reading WHERE survey_id=?", conn, params=(pid,)
        )
        if not prior.empty:
            medians.append(prior[heads].median())
    cur_reading = pd.read_sql_query(
        f"SELECT {', '.join(heads)} FROM reading WHERE survey_id=?", conn, params=(survey_id,)
    )
    background_regime: dict = {"n_bad": 0, "z_scores": {}}
    if len(medians) >= 2 and not cur_reading.empty:
        hist = pd.DataFrame(medians)
        cur_median = cur_reading[heads].median()
        background_regime = background_regime_shift(cur_median, hist, monitor_cfg["background_regime_z_threshold"])
    if background_regime["n_bad"] > 0:
        _bump("warn")

    coverage: dict = {"n": 0}
    if pipeline_version is not None:
        nominal = 1.0 - cfg.base.model.severity.get("conformal_alpha", 0.1)
        coverage = recompute_coverage_from_verifications(conn, pipeline_version, nominal)
    if coverage.get("below_nominal"):
        _bump("warn")

    return {
        "pipeline_version": pipeline_version,
        "survey_id": survey_id,
        "status": worst,
        "feature_drift": feature_drift.to_dict(orient="records"),
        "prediction_drift": prediction_drift,
        "background_regime": background_regime,
        "coverage": coverage,
    }


def monitor_report_to_text(report: dict) -> str:
    """One line per check, PASS/WARN/BLOCK -- mirrors `train._print_report`'s style."""
    lines = [
        f"\n=== Stage 8: monitor -- {report['survey_id']} (pipeline {report['pipeline_version']}) ===",
        f"Overall status: {report['status'].upper()}",
    ]
    blocked_features = [r["feature"] for r in report["feature_drift"] if r["status"] == "block"]
    warned_features = [r["feature"] for r in report["feature_drift"] if r["status"] == "warn"]
    lines.append(
        f"Feature drift: {len(blocked_features)} blocked, {len(warned_features)} warned "
        f"(of {len(report['feature_drift'])} features)"
    )
    if blocked_features:
        lines.append(f"  blocked: {blocked_features}")
    pd_report = report["prediction_drift"]
    if pd_report:
        lines.append(
            f"Prediction drift: indications/km {pd_report['indications_per_km_current']:.2f} "
            f"(training rate {pd_report['indications_per_km_reference']:.2f}, "
            f"ratio {pd_report['indications_per_km_ratio']:.2f}) "
            f"-- {'FLAGGED' if pd_report['flagged'] else 'ok'}"
        )
    br = report["background_regime"]
    lines.append(f"Background regime: {br['n_bad']} axis/axes exceeded z-threshold ({br['z_scores']})")
    cov = report["coverage"]
    if cov.get("n", 0) > 0:
        lines.append(
            f"Coverage (n={cov['n']} verified): empirical {cov['empirical']:.3f} vs nominal "
            f"{cov['nominal']:.3f} -- {'BELOW NOMINAL' if cov['below_nominal'] else 'ok'}"
        )
    else:
        lines.append("Coverage: n/a -- no excavations recorded yet")
    return "\n".join(lines)
