"""Stage 4.5: headless verification of app/demo_app.py via Streamlit's own
`streamlit.testing.v1.AppTest` -- no browser, no `claude-in-chrome`, but a
real script execution through Streamlit's actual ScriptRunner, which is what
originally caught two real bugs while building this: `app/`'s own directory
was never on `sys.path` (Streamlit does not add it the way plain `python
script.py` does), and the live-mode SQLite connection needed
`check_same_thread=False` (Streamlit can rerun one session's script on a
different worker thread between reruns).
"""

from __future__ import annotations

from pathlib import Path

import demo_lib
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "app" / "demo_app.py")


def _enter(at: AppTest) -> AppTest:
    """Click through the client-facing landing screen into the actual app --
    every test below exercises the app itself, not the splash, and the splash
    gates the tabs behind `st.stop()` until this button is clicked."""
    at.button[0].click().run()
    return at


def test_landing_page_shows_with_no_exception_and_gates_the_app():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception
    assert (
        len(at.tabs) == 0
    )  # gated behind st.stop() until "Launch the demo" is clicked
    assert len(at.button) == 1


def test_launching_the_demo_reveals_the_tabs():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _enter(at)
    assert not at.exception
    assert len(at.tabs) == 7


def test_demo_mode_corrupted_scenario_shows_the_refusal():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _enter(at)
    at.segmented_control(key="scenario_label").set_value(
        "Corrupted survey (bad sensor reading)"
    ).run()
    assert not at.exception
    assert any("Refused to score" in e.value for e in at.error)


def test_scenario_widget_returning_none_does_not_crash_the_app():
    """Ali's actual crash: `st.segmented_control` returns None if clicked on
    its own already-selected option (the default single-select "toggle off"
    behaviour), and the original code did `next(s for s in scenarios if
    s["label"] == selected_label)` with no fallback -- StopIteration,
    unhandled, killed the whole app. Fixed with `required=True` on the widget
    AND a defensive `next(..., scenarios[0])` fallback; this test forces the
    None case directly (AppTest.set_value doesn't simulate the real
    click-to-deselect gesture) to prove the fallback path itself works.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _enter(at)
    at.session_state["scenario_label"] = None
    at.run()
    assert not at.exception


def test_model_performance_tab_renders_the_real_model_card():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _enter(at)
    assert not at.exception
    card_text = demo_lib.load_model_card()
    rendered = "\n".join(m.value for m in at.markdown)
    # spot-check a real Stage 3 number from the baked model card actually
    # made it into the page, not just that SOME markdown rendered.
    first_metric_line = next(
        line for line in card_text.splitlines() if "recall @ dig budget" in line
    )
    assert first_metric_line in rendered


def test_beat1_deviation_scale_toggle_switches_without_error():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _enter(at)
    assert at.radio(key="beat1_scale").value == "Log"
    at.radio(key="beat1_scale").set_value("Linear").run()
    assert not at.exception


def test_live_mode_survives_two_consecutive_reruns_on_different_scenarios():
    """The exact sequence that broke without check_same_thread=False: switch
    into live mode, run a clean scenario, then rerun on a DIFFERENT scenario
    in the same session -- Streamlit is free to dispatch that second rerun on
    a different worker thread.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=90)
    at.run()
    _enter(at)
    at.segmented_control(key="app_mode").set_value("live").run()
    assert not at.exception

    at.segmented_control(key="scenario_label").set_value("Clean survey B").run()
    assert not at.exception


def test_live_mode_does_not_rerun_the_pipeline_for_an_unrelated_widget_change():
    """Ali's original complaint: the app felt slow. Root cause -- Streamlit
    reruns the whole script on ANY widget interaction, so without per-
    survey_id caching, moving the dig-budget control in Beat 3 (which has
    nothing to do with which scenario is scored) silently re-ran the full
    live validate->features->score pipeline every time. Pin that it doesn't.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=90)
    at.run()
    _enter(at)
    at.segmented_control(key="app_mode").set_value("live").run()
    scenario_label = at.session_state["scenario_label"]
    survey_id = next(
        s["survey_id"]
        for s in demo_lib.demo_scenarios()
        if s["label"] == scenario_label
    )
    cached_before = dict(at.session_state["live_results"])
    assert survey_id in cached_before

    at.segmented_control(key="dig_budget").set_value("10").run()
    assert not at.exception
    cached_after = dict(at.session_state["live_results"])

    assert set(cached_before) == set(cached_after)
    # same object, not just equal content -- proves no recompute happened.
    assert cached_before[survey_id] is cached_after[survey_id]


def test_live_mode_refuses_the_corrupted_scenario_too():
    at = AppTest.from_file(APP_PATH, default_timeout=90)
    at.run()
    _enter(at)
    at.segmented_control(key="app_mode").set_value("live").run()
    at.segmented_control(key="scenario_label").set_value(
        "Corrupted survey (bad sensor reading)"
    ).run()
    assert not at.exception
    assert any("Refused to score" in e.value for e in at.error)


def test_beat3_table_ranks_by_risk_score_once_a_classify_model_is_baked_in():
    """The currently baked `serving/` (Rig-v2, 2026-08-06) includes a real
    Stage 5 classify model, so every indication has a non-NULL `risk_score`
    -- Beat 3 ranks by it. (Previously, before Stage 5 was baked in,
    `risk_score` was all-NULL and Beat 3 correctly fell back to
    `anomaly_score` instead of silently sorting by an all-NULL column --
    that fallback path is still real code, just not exercised by TODAY's
    baked data; see chart_utils.indications_rank_chart's own NULL-fallback
    tests in test_chart_utils.py for direct coverage of it.)
    """
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _enter(at)
    assert not at.exception
    ranked_by = [c.value for c in at.caption if c.value.startswith("Ranked by")]
    assert ranked_by == ["Ranked by **risk_score**."]

    cols = at.dataframe[0].value.columns.tolist()
    for expected in ["pred_type", "pred_type_conf", "risk_score"]:
        assert expected in cols


def test_heatmap_tab_aggregates_across_the_clean_demo_scenarios():
    """Tab 7 is a CONSUMER of every clean scenario's baked indications at
    once, not just the one selected in the segmented control above it --
    proves the cross-line aggregation actually ran in demo mode (zero
    compute, so this must always have data to show)."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _enter(at)
    assert not at.exception
    captions = [c.value for c in at.caption]
    assert any("line(s)" in c and "indication(s) total" in c for c in captions)


def test_heatmap_tab_live_mode_notes_only_run_scenarios_are_included():
    at = AppTest.from_file(APP_PATH, default_timeout=90)
    at.run()
    _enter(at)
    at.segmented_control(key="app_mode").set_value("live").run()
    assert not at.exception
    captions = [c.value for c in at.caption]
    assert any("only includes scenarios you've already run" in c for c in captions)


def test_streamlit_app_stage3_thin_app_still_boots_with_no_exception():
    """Stage 3's own app -- unmodified except for reusing map_utils.build_map,
    which needed the same sys.path fix. Guard against regressing it."""
    stage3_path = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
    at = AppTest.from_file(stage3_path, default_timeout=60)
    at.run()
    assert not at.exception
