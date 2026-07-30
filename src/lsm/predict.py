"""
Stage 3/4: batch inference on a single already-featurised survey -> `indication`
rows + `indications.geojson`.

Loads the latest released bundle(s) via `bundle.py` (the real versioned
artifact contract: hard-fails on a feature_version/schema_version/library
mismatch, not just an ad hoc joblib dict). Scores the anomaly detector always;
scores severity too if a `severity_version` has been released -- an indication
is a valid detection with no severity estimate before Stage 4 exists, not an
error.

Run: python -m lsm predict <survey_id>
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from lsm.bundle import load_bundle
from lsm.config import Config
from lsm.features import feature_store_dir
from lsm.indications import attach_severity, cluster_indications, write_indications


class NoReleasedPipelineError(Exception):
    """No `pipeline_release` row exists yet -- run `lsm train` first."""


class SurveyNotScorableError(Exception):
    """The survey is not registered, not accepted, or has no computed features."""


def latest_pipeline_release(conn: sqlite3.Connection) -> tuple[str, str, str | None]:
    """(pipeline_version, anomaly_model_version, severity_model_version) for
    the most recent release.

    A real deployment reads whichever release carries `alias='champion'`;
    nothing in this demonstrator has ever been promoted past the freshly
    trained `challenger` (that promotion decision is explicitly Stage 7 CI
    scope), so "most recent by released_at" is the honest stand-in.
    """
    row = conn.execute(
        "SELECT pipeline_version, anomaly_version, severity_version FROM pipeline_release "
        "ORDER BY released_at DESC LIMIT 1"
    ).fetchone()
    if row is None:
        raise NoReleasedPipelineError("no pipeline_release found -- run `lsm train` first")
    return row


def _write_geojson(path: str | Path, indications: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    features = []
    for _, row in indications.iterrows():
        if pd.isna(row.get("lon")) or pd.isna(row.get("lat")):
            continue
        properties = {
            k: (None if pd.isna(v) else v)
            for k, v in row.items()
            if k not in ("lat", "lon")
        }
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(row["lon"]), float(row["lat"])]},
                "properties": properties,
            }
        )
    geojson = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(geojson, default=str), encoding="utf-8")


def predict_survey(conn: sqlite3.Connection, survey_id: str, cfg: Config) -> pd.DataFrame:
    """Score one survey against the latest released pipeline and persist its
    indications (+ severity, if a severity model has been released). Reuses
    the already-computed, already-(re)validated feature store rather than
    recomputing -- `lsm features` already re-runs `validate` on every call
    (Layer 1 "runs on ingest AND again inside predict" is satisfied by that
    symmetry, not duplicated here).
    """
    row = conn.execute(
        "SELECT line_id, run_id, source_uri, status FROM survey WHERE survey_id=?", (survey_id,)
    ).fetchone()
    if row is None:
        raise SurveyNotScorableError(f"{survey_id} is not registered -- run `lsm ingest` first")
    line_id, run_id, source_uri, status = row
    if status != "accepted":
        raise SurveyNotScorableError(f"{survey_id} has status={status!r}, cannot score it")

    feat_path = (
        feature_store_dir(cfg.env.storage.feature_dir, cfg.base.features.version, line_id, run_id)
        / "features.parquet"
    )
    if not feat_path.exists():
        raise SurveyNotScorableError(f"no features at {feat_path} -- run `lsm features` first")
    features = pd.read_parquet(feat_path)

    # lat/lon are restricted columns, excluded from the feature store -- joined
    # back from the raw survey on the integer key, same as everywhere else.
    raw = pd.read_parquet(source_uri, columns=["sample_idx", "lat", "lon"])
    df = features.merge(raw, on="sample_idx", how="left", validate="one_to_one")

    pipeline_version, anomaly_version, severity_version = latest_pipeline_release(conn)
    anomaly_bundle = load_bundle(
        Path(cfg.env.storage.model_dir) / anomaly_version / "bundle.joblib",
        expected_feature_version=cfg.base.features.version,
        expected_schema_version=cfg.base.schema_version,
    )

    df["_score"] = anomaly_bundle["model"].score(df)
    indications = cluster_indications(
        df, "_score", anomaly_bundle["threshold"], survey_id=survey_id, pipeline_version=pipeline_version
    )

    if severity_version is not None:
        severity_bundle = load_bundle(
            Path(cfg.env.storage.model_dir) / severity_version / "bundle.joblib",
            expected_feature_version=cfg.base.features.version,
            expected_schema_version=cfg.base.schema_version,
        )
        # severity_model.feature_cols ends with "extent_m", which
        # attach_severity/attach_indication_features compute themselves --
        # strip it back off for the "which peak-row columns to join" argument.
        base_feature_cols = [c for c in severity_bundle["feature_cols"] if c != "extent_m"]
        sev_cfg = cfg.base.model.severity
        indications = attach_severity(
            indications, df, severity_bundle["model"], base_feature_cols,
            nominal_coverage=1.0 - sev_cfg["conformal_alpha"],
        )

    write_indications(conn, indications)

    geojson_path = Path(cfg.env.storage.reports_dir) / survey_id / "indications.geojson"
    _write_geojson(geojson_path, indications)

    return indications
