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
