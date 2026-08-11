"""
Stage 4.5: bake the `serving/` artifact the demo app reads without importing
training code or touching `data/lsm.db` -- 3 real, already-trained clean
scenarios (genuine model output, not fabricated) plus one freshly-corrupted
scenario, model bundles copied verbatim, and a manifest tying it all together.

architecture.md's serving/ layout:
    serving/bundles/<model_version>/bundle.joblib
    serving/demo_surveys/<survey_id>.parquet
    serving/precomputed/<survey_id>_features.parquet
    serving/precomputed/<survey_id>_indications.parquet
    serving/model_card.md
    serving/manifest.json

Re-run this whenever `data/lsm.db` gets a new real `lsm train` run, or whenever
feature_version/schema_version changes -- `load_bundle` hard-fails on a
mismatch, by design, so a stale `serving/` fails loudly rather than silently.

Usage:
    python scripts/bake_demo_assets.py
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lsm.config import load_config
from lsm.db import connect
from lsm.features import feature_store_dir
from lsm.hashing import content_sha256, file_sha256
from lsm.pipeline import run_survey_pipeline
from lsm.predict import latest_pipeline_release

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVING_DIR = PROJECT_ROOT / "serving"

# Three distinct line_ids -- never 3 runs of the same line, so check_survey_overlap /
# check_background_regime (which compare against other accepted surveys sharing a
# line_id) never produce order-dependent surprises across scenario picks in one
# live-mode session.
CLEAN_SCENARIOS = [
    ("LINE000_R0", "Clean survey A"),
    ("LINE001_R0", "Clean survey B"),
    ("LINE002_R0", "Clean survey C"),
]
CORRUPTED_SOURCE_SURVEY_ID = "LINE004_R0"  # bytes borrowed, then relabelled + mutated
CORRUPTED_SURVEY_ID = "LINEDEMO_R0"


def _raw_path(cfg, survey_id: str) -> Path:
    line_id, run_id = survey_id.split("_R")
    return Path(cfg.env.storage.raw_dir) / f"line_id={line_id}" / f"run_id={run_id}" / "survey.parquet"


def _bake_clean_scenario(conn, cfg, survey_id: str, pipeline_version: str) -> None:
    line_id, run_id = survey_id.split("_R")
    run_id = int(run_id)

    raw_src = _raw_path(cfg, survey_id)
    (SERVING_DIR / "demo_surveys").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(raw_src, SERVING_DIR / "demo_surveys" / f"{survey_id}.parquet")

    feat_src = feature_store_dir(cfg.env.storage.feature_dir, cfg.base.features.version, line_id, run_id) / "features.parquet"
    (SERVING_DIR / "precomputed").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(feat_src, SERVING_DIR / "precomputed" / f"{survey_id}_features.parquet")

    indications = pd.read_sql_query(
        "SELECT * FROM indication WHERE survey_id=? AND pipeline_version=? AND is_shadow=0",
        conn, params=(survey_id, pipeline_version),
    )
    indications.to_parquet(SERVING_DIR / "precomputed" / f"{survey_id}_indications.parquet", index=False)
    print(f"  {survey_id}: {len(indications)} precomputed indications")


def _bake_corrupted_scenario(cfg) -> dict:
    src = _raw_path(cfg, CORRUPTED_SOURCE_SURVEY_ID)
    df = pd.read_parquet(src)
    df["line_id"] = "LINEDEMO"
    df["run_id"] = 0
    df.loc[3, "b_lo_nt"] = 999_999.0  # outside field_range_nT -- trips check_range, same
    # recipe tests/test_validate.py already proves, reused rather than invented.

    (SERVING_DIR / "demo_surveys").mkdir(parents=True, exist_ok=True)
    out_path = SERVING_DIR / "demo_surveys" / f"{CORRUPTED_SURVEY_ID}.parquet"
    df.to_parquet(out_path, index=False, compression="zstd")

    from lsm.generate import SurveyResult

    sr = SurveyResult(
        survey_id=CORRUPTED_SURVEY_ID,
        line_id="LINEDEMO",
        run_id=0,
        path=out_path,
        file_sha256=file_sha256(out_path),
        content_sha256=content_sha256(df),
        step_m=cfg.base.data.step_m,
        n_samples=len(df),
        chainage_start_m=float(df["chainage_true_m"].min()),
        chainage_end_m=float(df["chainage_true_m"].max()),
        standoff_m=cfg.base.data.walk.standoff_m,
        surveyed_at="2026-01-01T00:00:00+00:00",
    )

    # A throwaway db, purely to capture a genuine DQReport -- never data/lsm.db.
    # Windows keeps the sqlite file locked until the connection is explicitly
    # closed, so TemporaryDirectory's own cleanup would otherwise fail.
    with tempfile.TemporaryDirectory() as tmp:
        throwaway_conn = connect(Path(tmp) / "throwaway.db")
        try:
            _, report = run_survey_pipeline(throwaway_conn, sr, cfg)
        finally:
            throwaway_conn.close()

    if not report.has_fail:
        raise AssertionError(
            "the corrupted demo scenario no longer trips any DQ gate -- the "
            "generator or validator changed underneath this fixture; fix the "
            "corruption recipe before shipping a 'corrupted' scenario that isn't."
        )
    print(f"  {CORRUPTED_SURVEY_ID}: DQ report has_fail={report.has_fail}, "
          f"failed checks={[r.check_name for r in report.results if r.status == 'fail']}")

    return {
        "survey_id": report.survey_id,
        "checked_at": report.checked_at,
        "results": [dataclasses.asdict(r) for r in report.results],
    }


def main() -> None:
    cfg = load_config("dev")
    conn = connect(cfg.env.storage.sqlite_path)

    pipeline_version, anomaly_version, severity_version, classify_version = latest_pipeline_release(conn)
    print(f"Baking against pipeline_version={pipeline_version}")

    # latest_pipeline_release() doesn't carry growth_version -- growth is UPDATEd
    # onto an already-released pipeline (never a new release row, see growth.py),
    # so pull it directly rather than widening a function three other call sites
    # already depend on for a value only this script needs before the pr_row
    # query below.
    growth_version = conn.execute(
        "SELECT growth_version FROM pipeline_release WHERE pipeline_version=?",
        (pipeline_version,),
    ).fetchone()[0]

    if SERVING_DIR.exists():
        shutil.rmtree(SERVING_DIR)
    SERVING_DIR.mkdir(parents=True)

    print("Clean scenarios:")
    for survey_id, _label in CLEAN_SCENARIOS:
        _bake_clean_scenario(conn, cfg, survey_id, pipeline_version)

    print("Corrupted scenario:")
    corrupted_dq_report = _bake_corrupted_scenario(cfg)

    print("Bundles + model card:")
    (SERVING_DIR / "bundles").mkdir(parents=True, exist_ok=True)
    model_dir = Path(cfg.env.storage.model_dir)
    model_run_rows: dict[str, dict] = {}
    for model_version in (anomaly_version, severity_version, classify_version, growth_version):
        if model_version is None:
            continue
        src_bundle = model_dir / model_version / "bundle.joblib"
        dst_dir = SERVING_DIR / "bundles" / model_version
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_bundle, dst_dir / "bundle.joblib")
        print(f"  {model_version}: bundle copied")

        row = conn.execute(
            "SELECT model_version, task, mlflow_run_id, git_sha, config_sha256, data_sha256, "
            "feature_version, truth_as_of, trained_at, metrics_json, artifact_uri, "
            "final_test_uses FROM model_run WHERE model_version=?",
            (model_version,),
        ).fetchone()
        cols = ["model_version", "task", "mlflow_run_id", "git_sha", "config_sha256",
                "data_sha256", "feature_version", "truth_as_of", "trained_at",
                "metrics_json", "artifact_uri", "final_test_uses"]
        model_run_rows[model_version] = dict(zip(cols, row))

    model_card_src = model_dir / pipeline_version / "model_card.md"
    shutil.copyfile(model_card_src, SERVING_DIR / "model_card.md")

    pr_row = conn.execute(
        "SELECT pipeline_version, anomaly_version, severity_version, classify_version, "
        "growth_version, feature_version, schema_version, container_digest, released_at, "
        "alias FROM pipeline_release WHERE pipeline_version=?",
        (pipeline_version,),
    ).fetchone()
    pr_cols = ["pipeline_version", "anomaly_version", "severity_version", "classify_version",
               "growth_version", "feature_version", "schema_version", "container_digest",
               "released_at", "alias"]

    manifest = {
        "baked_from_pipeline_version": pipeline_version,
        "feature_version": cfg.base.features.version,
        "schema_version": cfg.base.schema_version,
        "config_sha256": cfg.config_sha256,
        "pipeline_release": dict(zip(pr_cols, pr_row)),
        "model_run": model_run_rows,
        "demo_scenarios": [
            {"survey_id": sid, "label": label, "kind": "clean"} for sid, label in CLEAN_SCENARIOS
        ] + [
            {"survey_id": CORRUPTED_SURVEY_ID, "label": "Corrupted survey (bad sensor reading)", "kind": "corrupted"}
        ],
        "corrupted_scenario_dq_report": corrupted_dq_report,
    }
    (SERVING_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {SERVING_DIR}/manifest.json")
    print(f"serving/ is ready: {len(CLEAN_SCENARIOS)} clean + 1 corrupted scenario, "
          f"feature_version={manifest['feature_version']}, schema_version={manifest['schema_version']}")


if __name__ == "__main__":
    main()
