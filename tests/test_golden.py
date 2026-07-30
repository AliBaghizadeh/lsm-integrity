"""
Golden regression test: a frozen tiny survey (tests/golden/tiny_survey.parquet,
generated once with seed=1234, length_m=200, step_m=1.0) with recorded expected
DQ outcomes. If a refactor to validate.py silently changes behaviour, this is
what catches it -- these numbers should never change without a deliberate,
reviewed reason.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from lsm.generate import SurveyResult
from lsm.hashing import content_sha256, file_sha256
from lsm.ingest import register_survey
from lsm.validate import validate_raw_survey

GOLDEN_PATH = Path(__file__).parent / "golden" / "tiny_survey.parquet"
EXPECTED_CONTENT_SHA256 = "bd7704617446223db6b2e89a13c18c177351e3254a565b7f2c267e2e405f5d91"
EXPECTED_N_SAMPLES = 200
EXPECTED_STATUSES = {
    "schema": "pass",
    "range": "pass",
    "saturation": "pass",
    "sample_idx_monotonic": "pass",
    "sample_idx_gap": "pass",
    "duplicate_sample_idx": "pass",
    "duplicate_content": "pass",
    "survey_overlap": "pass",
    "gps_jump": "pass",
    "gps_chainage_consistency": "pass",
    "noise_floor": "pass",
    "background_regime": "pass",
    "interference_density": "pass",
    "coverage": "pass",
}


def test_golden_survey_content_hash_is_stable():
    """If this fails, either the fixture changed or content_sha256's
    canonicalisation changed -- both are things a refactor must not do silently.
    """
    df = pd.read_parquet(GOLDEN_PATH)
    assert len(df) == EXPECTED_N_SAMPLES
    assert content_sha256(df) == EXPECTED_CONTENT_SHA256


def test_golden_survey_validates_clean(cfg):
    df = pd.read_parquet(GOLDEN_PATH)
    sr = SurveyResult(
        survey_id="LINE000_R0",
        line_id="LINE000",
        run_id=0,
        path=GOLDEN_PATH,
        file_sha256=file_sha256(GOLDEN_PATH),
        content_sha256=content_sha256(df),
        step_m=1.0,
        n_samples=len(df),
        chainage_start_m=float(df["chainage_m"].min()),
        chainage_end_m=float(df["chainage_m"].max()),
        standoff_m=1.5,
        surveyed_at="unknown",
    )

    from lsm.db import connect

    conn = connect(cfg.env.storage.sqlite_path)
    register_survey(conn, sr, schema_version=cfg.base.schema_version)
    report = validate_raw_survey(
        conn, sr.survey_id, sr.line_id, sr.step_m, sr.chainage_start_m,
        sr.chainage_end_m, sr.content_sha256, df, cfg.base.validate,
    )

    actual_statuses = {r.check_name: r.status for r in report.results}
    assert actual_statuses == EXPECTED_STATUSES
    assert not report.has_fail
