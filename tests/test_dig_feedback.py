"""Stage 8: dig_feedback.py -- the dig-feedback loop. No schema change:
`record_excavation`/`median_days_to_verification`/
`recompute_coverage_from_verifications` all use existing columns
(`indication.created_at`, `truth_defect.verified_at`) joined by chainage
proximity, not a dedicated link table.
"""

from __future__ import annotations

from lsm.db import connect
from lsm.dig_feedback import (
    median_days_to_verification,
    recompute_coverage_from_verifications,
    record_excavation,
)


def _seed_survey(conn, survey_id="S1", line_id="LINE000"):
    conn.execute(
        "INSERT INTO survey (survey_id, line_id, run_id, surveyed_at, step_m, n_samples, "
        "chainage_start_m, chainage_end_m, standoff_m, schema_version, source_uri, "
        "file_sha256, content_sha256, status, ingested_at) VALUES (?,?,0,?,0.5,1,0.0,1.0,1.5,2,"
        "'dummy','deadbeef','cafef00d','accepted',?)",
        (survey_id, line_id, "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
    )
    conn.commit()


def _seed_pipeline_release(conn, pipeline_version="P1"):
    conn.execute(
        "INSERT INTO pipeline_release (pipeline_version, anomaly_version, severity_version, "
        "classify_version, growth_version, feature_version, schema_version, container_digest, "
        "released_at, alias) VALUES (?,NULL,NULL,NULL,NULL,2,2,'local-dev',?,'challenger')",
        (pipeline_version, "2026-01-01T00:00:00+00:00"),
    )
    conn.commit()


def _seed_indication(
    conn,
    indication_id,
    survey_id,
    pipeline_version,
    chainage_peak_m,
    created_at,
    sev_lo=None,
    sev_hi=None,
):
    conn.execute(
        "INSERT INTO indication (indication_id, survey_id, pipeline_version, is_shadow, "
        "chainage_peak_m, chainage_start_m, chainage_end_m, lat, lon, anomaly_score, "
        "p_defect_cal, pred_type, pred_type_conf, sev_pred, sev_lo, sev_hi, interval_nominal, "
        "risk_score, dq_flag, created_at) "
        "VALUES (?,?,?,0,?,?,?,NULL,NULL,1.0,0.5,NULL,NULL,NULL,?,?,NULL,NULL,'clean',?)",
        (
            indication_id,
            survey_id,
            pipeline_version,
            chainage_peak_m,
            chainage_peak_m - 1.0,
            chainage_peak_m + 1.0,
            sev_lo,
            sev_hi,
            created_at,
        ),
    )
    conn.commit()


def test_record_excavation_creates_a_new_defect_when_unmatched(tmp_path):
    conn = connect(tmp_path / "t.db")
    _seed_survey(conn)
    _seed_pipeline_release(conn)
    _seed_indication(
        conn,
        "I1",
        "S1",
        "P1",
        chainage_peak_m=100.0,
        created_at="2026-01-01T00:00:00+00:00",
    )

    defect_id = record_excavation(
        conn, "I1", verified_severity_smys=55.0, verified_at="2026-02-01T00:00:00+00:00"
    )

    defect_row = conn.execute(
        "SELECT line_id FROM defect WHERE defect_id=?", (defect_id,)
    ).fetchone()
    assert defect_row is not None and defect_row[0] == "LINE000"

    truth_row = conn.execute(
        "SELECT revision, source, valid_to FROM truth_defect WHERE defect_id=?",
        (defect_id,),
    ).fetchone()
    assert truth_row == (0, "excavation", None)

    obs_row = conn.execute(
        "SELECT severity_smys FROM truth_observation WHERE defect_id=? AND survey_id='S1'",
        (defect_id,),
    ).fetchone()
    assert obs_row == (55.0,)


def test_record_excavation_matches_an_existing_defect_and_revises(tmp_path):
    conn = connect(tmp_path / "t.db")
    _seed_survey(conn)
    _seed_pipeline_release(conn)
    _seed_indication(
        conn,
        "I1",
        "S1",
        "P1",
        chainage_peak_m=100.0,
        created_at="2026-01-01T00:00:00+00:00",
    )

    # Manually seed a prior defect + revision 0 near the indication's chainage
    # -- ingest.py doesn't populate these yet, so this mirrors the only way
    # a "known" defect would exist today.
    conn.execute("INSERT INTO defect (defect_id, line_id) VALUES ('D1', 'LINE000')")
    conn.execute(
        "INSERT INTO truth_defect (defect_id, revision, chainage_m, defect_type, source, "
        "verified_at, valid_from, valid_to) VALUES ('D1', 0, 99.5, 'scc', 'synthetic', NULL, "
        "'2025-01-01T00:00:00+00:00', NULL)"
    )
    conn.commit()

    defect_id = record_excavation(
        conn, "I1", verified_severity_smys=60.0, verified_at="2026-02-01T00:00:00+00:00"
    )

    assert defect_id == "D1"
    revisions = conn.execute(
        "SELECT revision, source, valid_to FROM truth_defect WHERE defect_id='D1' ORDER BY revision"
    ).fetchall()
    assert revisions == [
        (0, "synthetic", "2026-02-01T00:00:00+00:00"),
        (1, "excavation", None),
    ]


def test_record_excavation_raises_on_unknown_indication(tmp_path):
    conn = connect(tmp_path / "t.db")
    try:
        record_excavation(conn, "does-not-exist", verified_severity_smys=50.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_median_days_to_verification_matches_hand_computed_value(tmp_path):
    conn = connect(tmp_path / "t.db")
    _seed_survey(conn)
    _seed_pipeline_release(conn)
    _seed_indication(
        conn,
        "I1",
        "S1",
        "P1",
        chainage_peak_m=100.0,
        created_at="2026-01-01T00:00:00+00:00",
    )
    _seed_indication(
        conn,
        "I2",
        "S1",
        "P1",
        chainage_peak_m=200.0,
        created_at="2026-01-01T00:00:00+00:00",
    )
    _seed_indication(
        conn,
        "I3",
        "S1",
        "P1",
        chainage_peak_m=300.0,
        created_at="2026-01-01T00:00:00+00:00",
    )

    # 10, 20, 30 days to verification -- median 20.
    record_excavation(conn, "I1", 50.0, verified_at="2026-01-11T00:00:00+00:00")
    record_excavation(conn, "I2", 55.0, verified_at="2026-01-21T00:00:00+00:00")
    record_excavation(conn, "I3", 60.0, verified_at="2026-01-31T00:00:00+00:00")

    assert median_days_to_verification(conn) == 20.0


def test_median_days_to_verification_is_none_with_no_excavations(tmp_path):
    conn = connect(tmp_path / "t.db")
    assert median_days_to_verification(conn) is None


def test_recompute_coverage_from_verifications_matches_expectation(tmp_path):
    conn = connect(tmp_path / "t.db")
    _seed_survey(conn)
    _seed_pipeline_release(conn, "P1")
    _seed_indication(
        conn,
        "I1",
        "S1",
        "P1",
        chainage_peak_m=100.0,
        created_at="2026-01-01T00:00:00+00:00",
        sev_lo=40.0,
        sev_hi=60.0,
    )
    _seed_indication(
        conn,
        "I2",
        "S1",
        "P1",
        chainage_peak_m=200.0,
        created_at="2026-01-01T00:00:00+00:00",
        sev_lo=40.0,
        sev_hi=60.0,
    )

    record_excavation(
        conn, "I1", verified_severity_smys=50.0, verified_at="2026-02-01T00:00:00+00:00"
    )  # inside [40,60]
    record_excavation(
        conn, "I2", verified_severity_smys=90.0, verified_at="2026-02-01T00:00:00+00:00"
    )  # outside [40,60]

    result = recompute_coverage_from_verifications(conn, "P1", nominal=0.9)
    assert result["n"] == 2
    assert result["empirical"] == 0.5
    assert result["below_nominal"] is True


def test_recompute_coverage_from_verifications_filters_by_pipeline_version(tmp_path):
    conn = connect(tmp_path / "t.db")
    _seed_survey(conn, "S1", line_id="LINE000")
    _seed_survey(conn, "S2", line_id="LINE001")
    _seed_pipeline_release(conn, "P1")
    _seed_pipeline_release(conn, "P2")
    _seed_indication(
        conn,
        "I1",
        "S1",
        "P1",
        chainage_peak_m=100.0,
        created_at="2026-01-01T00:00:00+00:00",
        sev_lo=40.0,
        sev_hi=60.0,
    )
    _seed_indication(
        conn,
        "I2",
        "S2",
        "P2",
        chainage_peak_m=100.0,
        created_at="2026-01-01T00:00:00+00:00",
        sev_lo=0.0,
        sev_hi=1.0,
    )

    record_excavation(
        conn, "I1", verified_severity_smys=50.0, verified_at="2026-02-01T00:00:00+00:00"
    )
    record_excavation(
        conn, "I2", verified_severity_smys=50.0, verified_at="2026-02-01T00:00:00+00:00"
    )

    result_p1 = recompute_coverage_from_verifications(conn, "P1", nominal=0.9)
    assert result_p1["n"] == 1
    assert result_p1["empirical"] == 1.0  # 50 is inside [40, 60]


def test_recompute_coverage_from_verifications_on_no_excavations(tmp_path):
    conn = connect(tmp_path / "t.db")
    result = recompute_coverage_from_verifications(conn, "P1", nominal=0.9)
    assert result == {"n": 0}
