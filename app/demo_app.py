"""
Stage 4.5: the polished demo app. A CONSUMER of the `serving/` artifact and
of the real pipeline functions -- not the serving layer (SKILL invariant
#14; the batch scoring job in train.py/predict.py is the server).

APP_MODE=demo serves precomputed results straight from `serving/` -- zero
SQLite, zero compute, cannot fail. APP_MODE=live runs the real
`pipeline.run_full_pipeline` (register -> validate -> features -> score) on
the same baked survey files, against a session-private SQLite db seeded from
`serving/manifest.json` -- same code, so the impressive path and the safe
path are the same path.

Data loads are `st.cache_data`-wrapped and live-mode scoring is cached per
survey_id in `st.session_state`: Streamlit reruns this whole script on EVERY
widget interaction, so without caching, clicking the dig-budget control in
Beat 3 would silently re-run the full validate->features->score pipeline in
live mode even though nothing about the scenario changed.

Run: streamlit run app/demo_app.py
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import chart_utils  # noqa: E402
import demo_lib  # noqa: E402
from map_utils import build_map  # noqa: E402

from lsm.config import load_config  # noqa: E402

st.set_page_config(layout="wide", page_title="LSM demo -- Stage 4.5", initial_sidebar_state="collapsed")


@st.cache_data(show_spinner=False)
def _cached_demo_raw(survey_id: str):
    return demo_lib.load_demo_raw(survey_id)


@st.cache_data(show_spinner=False)
def _cached_demo_features(survey_id: str):
    return demo_lib.load_demo_features(survey_id)


@st.cache_data(show_spinner=False)
def _cached_demo_indications(survey_id: str):
    return demo_lib.load_demo_indications(survey_id)


manifest = demo_lib.load_manifest()
scenarios = demo_lib.demo_scenarios()

st.title("LSM Pipeline Integrity -- Live Demo")
st.markdown(
    "A **survey** is one magnetometer pass along a stretch of pipeline -- the raw sensor "
    "reading the whole pipeline runs on. The three **clean** scenarios below are real "
    "surveys the trained model has genuinely scored (not fabricated for this demo); the "
    "**corrupted** one is a real survey with one sensor reading pushed out of range, to show "
    "what happens when the data itself is bad, not the model."
)

top = st.columns([1, 2])
with top[0]:
    mode = st.segmented_control(
        "Mode", ["demo", "live"], default=os.environ.get("APP_MODE", "demo"), key="app_mode",
        help="demo: precomputed results, zero compute. live: re-runs the real pipeline now, ~1s.",
    )
with top[1]:
    labels = [s["label"] for s in scenarios]
    selected_label = st.segmented_control("Scenario", labels, default=labels[0], key="scenario_label")

scenario = next(s for s in scenarios if s["label"] == selected_label)
survey_id = scenario["survey_id"]
st.caption(f"Scenario: **{scenario['label']}** -- survey_id `{survey_id}`, mode `{mode}`")

if mode == "demo":
    raw = _cached_demo_raw(survey_id)
    features = _cached_demo_features(survey_id)
    indications = _cached_demo_indications(survey_id)
    dq_failure = demo_lib.load_demo_dq_failure(survey_id)
else:
    if "live_conn" not in st.session_state:
        base_cfg = load_config("dev")
        conn, live_cfg = demo_lib.init_live_session(base_cfg)
        st.session_state["live_conn"] = conn
        st.session_state["live_cfg"] = live_cfg
        st.session_state["live_results"] = {}
    live_cfg = st.session_state["live_cfg"]
    live_results = st.session_state["live_results"]

    raw = _cached_demo_raw(survey_id)  # raw bytes are identical regardless of mode
    if survey_id not in live_results:
        with st.spinner("Running validate -> features -> score..."):
            report, indications = demo_lib.run_live_scenario(
                st.session_state["live_conn"], live_cfg, survey_id
            )
            if report.has_fail:
                features_live = None
            else:
                features_live = demo_lib.load_live_features(live_cfg, survey_id)
            live_results[survey_id] = (report, indications, features_live)
    report, indications, features = live_results[survey_id]
    dq_failure = (
        {
            "survey_id": report.survey_id,
            "checked_at": report.checked_at,
            "results": [dataclasses.asdict(r) for r in report.results],
        }
        if report.has_fail
        else None
    )

beat1, beat2, beat3, beat4 = st.tabs([
    "1. Raw signal", "2. Detrend + gradient", "3. Ranked indications", "4. Corrupted survey",
])

with beat1:
    st.subheader("The raw field, full range")
    st.caption(
        "bx/by/bz per axis against the ~45,000 nT background. Dashed lines mark where the "
        "true defects (orange) and interference sources (grey) actually are -- if the "
        "anomaly were visible here, background removal wouldn't be the hard part of this "
        "project."
    )
    st.altair_chart(chart_utils.raw_components_chart(raw), width="stretch")

    if st.checkbox(
        "Show |B| deviation from its own median, log scale",
        key="beat1_log",
        help="Proof the anomaly is genuinely there, just buried -- the log-scale deviation "
             "view is where it stops being invisible.",
    ):
        st.altair_chart(chart_utils.deviation_log_chart(raw), width="stretch")

with beat2:
    st.subheader("Background removed")
    if features is None:
        st.warning("No features computed for this scenario -- it was refused at validation. See tab 4.")
    else:
        st.caption(
            "Residual (and the along-track/vertical gradient, if a second sensor head is "
            "present) after detrending. Both true defects AND true interference sources "
            "appear now -- separating them is what Stage 3's model does."
        )
        st.altair_chart(chart_utils.residual_gradient_chart(features, raw), width="stretch")

with beat3:
    st.subheader("Ranked indications, on a dig budget")
    if indications is None or not len(indications):
        st.info("No indications for this scenario.")
    else:
        length_km = max(float(raw["chainage_m"].max()) / 1000.0, 0.001)
        budget_choice = st.segmented_control("Dig budget (per km)", ["3", "5", "10"], default="5", key="dig_budget")
        budget_n = max(1, round(int(budget_choice) * length_km))
        ranked = indications.sort_values("anomaly_score", ascending=False)
        dug = ranked.head(budget_n)

        st.pydeck_chart(build_map(raw, dug))
        st.caption("orange = true defect - grey = true interference (unlabelled to the model) - blue = dug indication")

        show_cols = [c for c in [
            "chainage_peak_m", "anomaly_score", "p_defect_cal", "sev_pred", "sev_lo", "sev_hi", "dq_flag",
        ] if c in dug.columns]
        st.dataframe(dug[show_cols], width="stretch")

with beat4:
    st.subheader("What happens when the data itself is bad")
    if dq_failure is None:
        st.success("This scenario passed every DQ gate -- nothing to refuse.")
    else:
        failed = [r for r in dq_failure["results"] if r["status"] == "fail"]
        st.error(f"Refused to score {dq_failure['survey_id']}: {len(failed)} DQ check(s) failed.")
        for r in failed:
            with st.container(border=True):
                st.markdown(f"**{r['check_name']}** -- {r['n_affected']} row(s) affected")
                if r["detail"]:
                    st.json(r["detail"])

st.divider()
pr = manifest["pipeline_release"]
anomaly_git_sha = next(iter(manifest["model_run"].values()))["git_sha"] if manifest["model_run"] else "unknown"
st.caption(
    f"pipeline_version={pr['pipeline_version']} | feature_version={manifest['feature_version']} | "
    f"schema_version={manifest['schema_version']} | config_sha256={manifest['config_sha256'][:12]}... | "
    f"git_sha={anomaly_git_sha[:12]}... | mode={mode}"
)
