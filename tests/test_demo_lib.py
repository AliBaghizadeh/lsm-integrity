"""Stage 4.5: demo_lib.py against the real, committed `serving/` directory --
the same data the demo app reads, exercised headlessly (no `streamlit`
import anywhere in demo_lib itself, so this needs none either).
"""

from __future__ import annotations

import demo_lib
import pytest

from lsm.bundle import load_bundle
from lsm.config import load_config
from lsm.schemas import validate_reading_schema


def test_manifest_and_scenarios_parse():
    manifest = demo_lib.load_manifest()
    assert manifest["feature_version"] >= 1
    scenarios = demo_lib.demo_scenarios()
    kinds = {s["kind"] for s in scenarios}
    assert kinds == {"clean", "corrupted"}
    assert sum(1 for s in scenarios if s["kind"] == "corrupted") == 1


@pytest.mark.parametrize("survey_id", ["LINE000_R0", "LINE001_R0", "LINE002_R0"])
def test_clean_scenarios_have_precomputed_features_and_indications(survey_id):
    assert demo_lib.load_demo_features(survey_id) is not None
    assert demo_lib.load_demo_indications(survey_id) is not None
    assert demo_lib.load_demo_dq_failure(survey_id) is None


def test_corrupted_scenario_has_no_precomputed_output_but_a_dq_failure():
    survey_id = next(
        s["survey_id"] for s in demo_lib.demo_scenarios() if s["kind"] == "corrupted"
    )
    assert demo_lib.load_demo_features(survey_id) is None
    assert demo_lib.load_demo_indications(survey_id) is None
    failure = demo_lib.load_demo_dq_failure(survey_id)
    assert failure is not None
    failed_checks = {
        r["check_name"] for r in failure["results"] if r["status"] == "fail"
    }
    assert "range" in failed_checks


def test_every_demo_survey_passes_the_raw_schema_contract():
    for scenario in demo_lib.demo_scenarios():
        df = demo_lib.load_demo_raw(scenario["survey_id"])
        if scenario["kind"] == "clean":
            validate_reading_schema(df)
        # the corrupted one is EXPECTED to fail schema validation -- that's
        # the whole point of the scenario, so it is deliberately not asserted
        # here (see test_corrupted_scenario_has_no_precomputed_output... above).


def test_bundles_load_against_the_current_config_versions():
    cfg = load_config("dev")
    manifest = demo_lib.load_manifest()
    for model_version in manifest["model_run"]:
        bundle = load_bundle(
            demo_lib.SERVING_DIR / "bundles" / model_version / "bundle.joblib",
            expected_feature_version=cfg.base.features.version,
            expected_schema_version=cfg.base.schema_version,
        )
        assert bundle["feature_cols"]


def test_live_session_scores_all_three_clean_scenarios():
    cfg = load_config("dev")
    conn, live_cfg = demo_lib.init_live_session(cfg)
    for scenario in demo_lib.demo_scenarios():
        if scenario["kind"] != "clean":
            continue
        report, indications = demo_lib.run_live_scenario(
            conn, live_cfg, scenario["survey_id"]
        )
        assert not report.has_fail
        assert indications is not None
        features = demo_lib.load_live_features(live_cfg, scenario["survey_id"])
        assert features is not None and len(features) > 0


def test_live_session_refuses_the_corrupted_scenario_without_raising():
    cfg = load_config("dev")
    conn, live_cfg = demo_lib.init_live_session(cfg)
    survey_id = next(
        s["survey_id"] for s in demo_lib.demo_scenarios() if s["kind"] == "corrupted"
    )

    report, indications = demo_lib.run_live_scenario(conn, live_cfg, survey_id)

    assert report.has_fail
    assert indications is None


def test_live_session_rerun_on_the_same_scenario_is_idempotent():
    cfg = load_config("dev")
    conn, live_cfg = demo_lib.init_live_session(cfg)
    survey_id = next(
        s["survey_id"] for s in demo_lib.demo_scenarios() if s["kind"] == "clean"
    )

    _, first = demo_lib.run_live_scenario(conn, live_cfg, survey_id)
    _, second = demo_lib.run_live_scenario(conn, live_cfg, survey_id)

    n_rows = conn.execute(
        "SELECT COUNT(*) FROM indication WHERE survey_id=?", (survey_id,)
    ).fetchone()[0]
    assert n_rows == len(first) == len(second)
