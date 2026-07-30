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

from streamlit.testing.v1 import AppTest

import demo_lib

APP_PATH = str(Path(__file__).resolve().parents[1] / "app" / "demo_app.py")


def test_demo_mode_boots_with_no_exception():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception
    assert len(at.tabs) == 4


def test_demo_mode_corrupted_scenario_shows_the_refusal():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
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
    at.session_state["scenario_label"] = None
    at.run()
    assert not at.exception


def test_beat1_deviation_scale_toggle_switches_without_error():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
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
    at.segmented_control(key="app_mode").set_value("live").run()
    scenario_label = at.session_state["scenario_label"]
    survey_id = next(
        s["survey_id"] for s in demo_lib.demo_scenarios() if s["label"] == scenario_label
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
    at.segmented_control(key="app_mode").set_value("live").run()
    at.segmented_control(key="scenario_label").set_value(
        "Corrupted survey (bad sensor reading)"
    ).run()
    assert not at.exception
    assert any("Refused to score" in e.value for e in at.error)


def test_streamlit_app_stage3_thin_app_still_boots_with_no_exception():
    """Stage 3's own app -- unmodified except for reusing map_utils.build_map,
    which needed the same sys.path fix. Guard against regressing it."""
    stage3_path = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
    at = AppTest.from_file(stage3_path, default_timeout=60)
    at.run()
    assert not at.exception
