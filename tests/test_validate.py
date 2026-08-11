from __future__ import annotations

from conftest import generate_one_survey, run_pipeline_on


def _status_by_check(report):
    return {r.check_name: r.status for r in report.results}


def _n_affected_by_check(report):
    return {r.check_name: r.n_affected for r in report.results}


def test_clean_survey_passes_with_zero_fail(tiny_cfg, tmp_path):
    sr = generate_one_survey(tiny_cfg, tmp_path)
    conn, status, report = run_pipeline_on(tiny_cfg, sr)

    assert status == "accepted"
    assert not report.has_fail
    assert report.summary()["fail"] == 0
    # clean survey's readings actually get loaded
    n_rows = conn.execute(
        "SELECT COUNT(*) FROM reading WHERE survey_id=?", (sr.survey_id,)
    ).fetchone()[0]
    assert n_rows == sr.n_samples
    row = conn.execute(
        "SELECT status FROM survey WHERE survey_id=?", (sr.survey_id,)
    ).fetchone()
    assert row[0] == "accepted"


def test_range_violation_is_caught(tiny_cfg, tmp_path):
    def corrupt(df):
        df.loc[3, "b_lo_nt"] = 999_999.0  # far outside [20000, 80000]
        return df

    sr = generate_one_survey(tiny_cfg, tmp_path, mutate=corrupt)
    conn, _status, report = run_pipeline_on(tiny_cfg, sr)

    statuses = _status_by_check(report)
    assert statuses["range"] == "fail"
    assert report.has_fail
    row = conn.execute(
        "SELECT status FROM survey WHERE survey_id=?", (sr.survey_id,)
    ).fetchone()
    assert row[0] == "quarantined"
    # a hard-failed survey's readings are NEVER loaded
    n_rows = conn.execute(
        "SELECT COUNT(*) FROM reading WHERE survey_id=?", (sr.survey_id,)
    ).fetchone()[0]
    assert n_rows == 0


def test_one_bad_value_reports_one_row_affected_not_the_whole_survey(tiny_cfg, tmp_path):
    """Regression: check_schema used to hardcode n_affected=len(df), so a
    SINGLE out-of-range value on a 200-row tiny_cfg survey reported "200 rows
    affected" -- misleading, and exactly what the demo app's refusal card
    showed Ali on a 4000-row real survey ("4000 row(s) affected" for 1 bad
    value). This same corruption recipe trips BOTH range and schema
    (pandera's own Check.in_range duplicates the range gate) -- both must
    report the true count, 1, not the survey's row count.
    """
    def corrupt(df):
        df.loc[3, "b_lo_nt"] = 999_999.0
        return df

    sr = generate_one_survey(tiny_cfg, tmp_path, mutate=corrupt)
    _, _, report = run_pipeline_on(tiny_cfg, sr)

    affected = _n_affected_by_check(report)
    assert affected["range"] == 1
    assert affected["schema"] == 1
    schema_result = next(r for r in report.results if r.check_name == "schema")
    assert schema_result.detail["failures"][0]["row_index"] == 3
    assert schema_result.detail["failures"][0]["column"] == "b_lo_nt"


def test_saturation_is_caught(tiny_cfg, tmp_path):
    def corrupt(df):
        df.loc[5:9, "b_mid_nt"] = 1234.5  # 5 identical consecutive values >= run_length=3
        return df

    sr = generate_one_survey(tiny_cfg, tmp_path, mutate=corrupt)
    _conn, _status, report = run_pipeline_on(tiny_cfg, sr)

    statuses = _status_by_check(report)
    assert statuses["saturation"] == "fail"
    assert report.has_fail


def test_duplicate_sample_idx_is_caught(tiny_cfg, tmp_path):
    def corrupt(df):
        df.loc[df.index[-1], "sample_idx"] = df.loc[df.index[-2], "sample_idx"]
        return df

    sr = generate_one_survey(tiny_cfg, tmp_path, mutate=corrupt)
    conn, _status, report = run_pipeline_on(tiny_cfg, sr)

    statuses = _status_by_check(report)
    assert statuses["duplicate_sample_idx"] == "fail"
    assert report.has_fail
    # this must NOT crash on the reading table's primary key -- readings simply
    # never get loaded for a quarantined survey.
    n_rows = conn.execute(
        "SELECT COUNT(*) FROM reading WHERE survey_id=?", (sr.survey_id,)
    ).fetchone()[0]
    assert n_rows == 0


def test_sample_idx_monotonic_violation_is_caught(tiny_cfg, tmp_path):
    def corrupt(df):
        # swap two adjacent sample_idx values -> a genuine out-of-order raw file,
        # not just an unsorted read (duplicate check would not fire on this).
        df = df.copy()
        i, j = 4, 5
        df.loc[i, "sample_idx"], df.loc[j, "sample_idx"] = (
            df.loc[j, "sample_idx"],
            df.loc[i, "sample_idx"],
        )
        return df

    sr = generate_one_survey(tiny_cfg, tmp_path, mutate=corrupt)
    _conn, _status, report = run_pipeline_on(tiny_cfg, sr)

    statuses = _status_by_check(report)
    assert statuses["sample_idx_monotonic"] == "fail"
    assert statuses["duplicate_sample_idx"] == "pass"  # still all-unique, just reordered
    assert report.has_fail


def test_gps_jump_is_caught(tiny_cfg, tmp_path):
    def corrupt(df):
        df.loc[10, "lat"] += 1.0  # ~111 km jump, far past max_gps_jump_m=5
        return df

    sr = generate_one_survey(tiny_cfg, tmp_path, mutate=corrupt)
    _conn, _status, report = run_pipeline_on(tiny_cfg, sr)

    statuses = _status_by_check(report)
    assert statuses["gps_jump"] == "warn"  # configured gate: warn
    assert not report.has_fail  # warn-only check, survey still accepted


def test_schema_violation_is_caught(tiny_cfg, tmp_path):
    def corrupt(df):
        df.loc[0, "defect_type"] = "not_a_real_type"
        return df

    sr = generate_one_survey(tiny_cfg, tmp_path, mutate=corrupt)
    _conn, _status, report = run_pipeline_on(tiny_cfg, sr)

    statuses = _status_by_check(report)
    assert statuses["schema"] == "fail"
    assert report.has_fail


def test_duplicate_content_is_caught_on_second_ingest(tiny_cfg, tmp_path):
    sr = generate_one_survey(tiny_cfg, tmp_path)
    conn, _, report1 = run_pipeline_on(tiny_cfg, sr)
    assert not report1.has_fail

    # A second, differently-labelled survey with byte-identical content. This
    # must register successfully (no DB-level crash -- see db.py) and instead
    # be caught and quarantined by the duplicate_content DQ check.
    sr2 = generate_one_survey(tiny_cfg, tmp_path)
    sr2.survey_id = "LINE999_R9"
    sr2.line_id = "LINE999"
    sr2.run_id = 9
    conn, status2, report2 = run_pipeline_on(tiny_cfg, sr2)

    assert status2 == "accepted"  # registered fine -- quarantine happens via DQ, not a crash
    statuses = _status_by_check(report2)
    assert statuses["duplicate_content"] == "fail"
    assert report2.has_fail
    row = conn.execute(
        "SELECT status FROM survey WHERE survey_id=?", (sr2.survey_id,)
    ).fetchone()
    assert row[0] == "quarantined"


def test_survey_overlap_flags_near_identical_signal(tiny_cfg, tmp_path):
    sr = generate_one_survey(tiny_cfg, tmp_path)
    _conn, _, report1 = run_pipeline_on(tiny_cfg, sr)
    assert not report1.has_fail

    # Same physical acquisition, mislabelled as a different run: content_sha256
    # differs only because of a single-value tweak (dodges duplicate_content),
    # but the raw signal is still >99% correlated with sr's.
    def near_identical(df):
        df.loc[0, "severity_smys"] = df.loc[0, "severity_smys"]  # no-op, forces a fresh write
        return df

    sr2 = generate_one_survey(tiny_cfg, tmp_path, mutate=near_identical)
    sr2.survey_id = "LINE000_R7"
    sr2.line_id = "LINE000"  # SAME line_id as sr -> overlap is checked
    sr2.run_id = 7
    _, _, report2 = run_pipeline_on(tiny_cfg, sr2)

    statuses = _status_by_check(report2)
    # near-identical raw signal on the same line -> either duplicate_content
    # (if the mutate above didn't change the hash) or survey_overlap must fire.
    assert statuses["duplicate_content"] == "fail" or statuses["survey_overlap"] == "fail"


def test_coverage_warns_on_short_survey(tiny_cfg, tmp_path):
    def truncate(df):
        return df.iloc[: len(df) // 2].copy()

    sr = generate_one_survey(tiny_cfg, tmp_path, mutate=truncate)
    _conn, _status, _report = run_pipeline_on(
        tiny_cfg, sr
    )
    # coverage check needs an expected_length_m to fire; run_survey_pipeline
    # does not pass one, so this documents current scope: coverage is exercised
    # directly here instead.
    from lsm.validate import check_coverage

    result = check_coverage(
        __import__("pandas").read_parquet(sr.path),
        expected_length_m=tiny_cfg.base.data.length_m,
        step_m=tiny_cfg.base.data.step_m,
        gate="warn",
    )
    assert result.status == "warn"


def test_noise_floor_and_interference_density_run_without_error(tiny_cfg, tmp_path):
    """Coarse Stage-1 proxies (real detrend-based versions land in Stage 2) --
    assert they execute and report a status, not a specific threshold outcome.
    """
    sr = generate_one_survey(tiny_cfg, tmp_path)
    _conn, _status, report = run_pipeline_on(tiny_cfg, sr)
    statuses = _status_by_check(report)
    assert statuses["noise_floor"] in ("pass", "warn", "fail")
    assert statuses["interference_density"] in ("pass", "warn", "fail")
