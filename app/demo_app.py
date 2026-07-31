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

import chart_utils
import demo_lib
from map_utils import build_map

from lsm.config import load_config

st.set_page_config(layout="wide", page_title="LSM demo -- Stage 4.5", initial_sidebar_state="collapsed")

# A client-facing splash before the working app -- gated on session_state so a
# widget interaction later in the session (which reruns this whole script,
# same reasoning as the caching comment above) never sends the viewer back to
# it. st.stop() below is what actually prevents the rest of the script
# (data loads, tabs) from running while the splash is showing.
if "entered" not in st.session_state:
    st.session_state["entered"] = False

if not st.session_state["entered"]:
    st.write("")
    hero_text, hero_img = st.columns([1.5, 1], gap="large")
    with hero_text:
        st.markdown("# AI in Action")
        st.markdown("### Inspecting pipelines before they fail")
        st.write(
            "A live walkthrough of an LSM pipeline-integrity pipeline -- from a raw "
            "45,000 nT magnetometer signal to a risk-ranked dig list, scored end to end "
            "by real, trained models."
        )
        if st.button("Launch the demo  →", type="primary"):
            st.session_state["entered"] = True
            st.rerun()
    with hero_img:
        st.image(str(Path(__file__).resolve().parent / "ROSEN_inspection.jpeg"), width="stretch")
    st.stop()


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
    "A **survey** is one pass of the magnetometer sled along a stretch of buried pipe -- one "
    "raw time-series of sensor readings against position, the same physical pipe surveyed "
    "repeatedly over time (like a baseline inspection plus follow-ups) to track change. "
    "**Clean** below means *passed data-quality validation* -- unrelated to whether the model "
    "did well; the three clean scenarios are real surveys the trained model has genuinely "
    "scored (not fabricated for this demo). The **corrupted** scenario is a real survey with "
    "one sensor reading pushed out of range, to show what happens when the DATA itself is "
    "bad, not the model. For the baseline-vs-trained-model numbers (MAD vs. IsolationForest, "
    "boosted severity/classification vs. their baselines), see tab 2."
)

top = st.columns([1, 2])
with top[0]:
    mode = st.segmented_control(
        "Mode", ["demo", "live"], default=os.environ.get("APP_MODE", "demo"), key="app_mode",
        help="demo: precomputed results, zero compute. live: re-runs the real pipeline now, ~1s.",
        required=True,  # a segmented_control click can otherwise DESELECT and return None
    )
with top[1]:
    labels = [s["label"] for s in scenarios]
    selected_label = st.segmented_control(
        "Scenario", labels, default=labels[0], key="scenario_label", required=True,
    )

# Defensive even with required=True: st.segmented_control CAN return None if
# clicked on its already-selected option (single-select "toggle off" is the
# default widget behaviour) -- this crashed with StopIteration on `next(...)`
# before `required=True` was added. Fall back to the first scenario rather
# than ever hard-crash the whole app on a widget edge case.
scenario = next((s for s in scenarios if s["label"] == selected_label), scenarios[0])
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

overview, performance, beat1, beat2, beat3, beat4 = st.tabs([
    "1. How it works", "2. Model performance", "3. Raw signal", "4. Detrend + gradient",
    "5. Ranked indications", "6. Corrupted survey",
])

# Real module names and check names, not paraphrases -- this tab exists so an
# LSM engineer can map what they're looking at straight back to the codebase,
# not just trust a marketing diagram. Kept Streamlit-native (columns +
# bordered containers, no unsafe_allow_html/graphviz) so it can't fail to
# render for a missing system binary on demo day, same reasoning as the rest
# of this app.
_STAGES = [
    ("1. Ingest", "lsm.ingest",
     ("Raw survey Parquet -> SQLite, content-hashed. Same bytes twice is a no-op; a "
     "changed hash under an existing (line_id, run_id) is an error, never a silent "
     "overwrite.")),
    ("2. Validate", "lsm.validate",
     ("14 data-quality checks (below). A hard failure quarantines the survey and "
     "stops the pipeline here -- see tab 6 for what that looks like for real.")),
    ("3. Features", "lsm.features",
     ("Per-survey detrend, along-track/vertical gradient, sliding-window stats, "
     "peak-shape descriptors (FWHM, asymmetry, decay exponent).")),
]
_STAGES_2 = [
    ("4. Detect", "lsm.models.anomaly / lsm.indications",
     ("MAD baseline or IsolationForest scores every row; contiguous flagged rows "
     "cluster into one indication.")),
    ("5. Severity + Classify", "lsm.models.severity / lsm.models.classify",
     ("Two independent models score the SAME indication in parallel: a calibrated "
     "5/50/95% severity interval, and a defect-type class + confidence.")),
    ("6. Risk rank", "lsm.indications.attach_classification",
     ("risk_score = calibrated P(defect) x severity x a stated consequence proxy. "
     "Tab 4's table is sorted by this.")),
]

_DQ_CHECKS = [
    ("schema", "Column presence/dtypes/units/enum values against the declared data contract."),
    ("range", "bx/by/bz stay inside the sensor's physical field range."),
    ("saturation", "No stuck sensor / ADC rail -- N+ consecutive identical raw values on any axis."),
    ("sample_idx_monotonic", "sample_idx strictly increasing, no reordering."),
    ("sample_idx_gap", "No missing samples beyond the allowed spacing tolerance."),
    ("duplicate_sample_idx", "No repeated sample_idx within one survey."),
    ("duplicate_content", "This survey's content hash doesn't already exist under a different run -- a re-export."),
    ("survey_overlap", "Cross-correlation against other accepted runs of the same line -- an overlapping re-run neither hash catches."),
    ("gps_jump", "No physically-impossible lat/lon jump between consecutive samples."),
    ("gps_chainage_consistency", "GPS-derived distance agrees with the recorded chainage_m."),
    ("noise_floor", "Raw-signal noise sits within the expected sensor-floor band."),
    ("background_regime", "This run's background statistics haven't shifted from the line's prior accepted runs."),
    ("interference_density", "The fraction of interference-like readings is within the expected range."),
    ("coverage", "The survey covers its expected chainage length with no large unexplained gaps."),
]

with overview:
    st.subheader("One survey, six stages")
    st.caption(
        "Every scenario below runs this same path, live or precomputed -- this tab is a map "
        "of it, not a beat of its own. A hard validation failure is the one place the path "
        "stops early (tab 6)."
    )

    row1 = st.columns([3, 1, 3, 1, 3])
    for i, (name, module, desc) in enumerate(_STAGES):
        with row1[i * 2], st.container(border=True):
            st.markdown(f"**{name}**")
            st.caption(f"`{module}`")
            st.caption(desc)
    with row1[1]:
        st.markdown("### →")
    with row1[3]:
        st.markdown("### →")

    st.markdown("**↓**")

    row2 = st.columns([3, 1, 3, 1, 3])
    for i, (name, module, desc) in enumerate(_STAGES_2):
        with row2[i * 2], st.container(border=True):
            st.markdown(f"**{name}**")
            st.caption(f"`{module}`")
            st.caption(desc)
    with row2[1]:
        st.markdown("### →")
    with row2[3]:
        st.markdown("### →")

    st.caption(
        "Not shown: growth/remaining-life and drift monitoring (`lsm forecast` / `lsm monitor`) "
        "run downstream of a released pipeline via the CLI, not this app -- Stage 8 backend only."
    )

    with st.expander("The 14 validation checks (lsm.validate)"):
        for check_name, desc in _DQ_CHECKS:
            st.markdown(f"- **`{check_name}`** -- {desc}")

with performance:
    st.subheader("Trained model vs. baseline, on real held-out numbers")
    st.caption(
        "The auto-generated model card for the pipeline this app is currently serving -- "
        "the SAME evaluation `lsm train` printed to the console, not a re-derived or "
        "prettied-up copy. Every number below is grouped cross-validated, out-of-fold, and "
        "reported with a bootstrap confidence interval, not a point estimate picked to look "
        "good. A gate that says DID NOT PASS is reported as such, not hidden."
    )
    st.markdown(demo_lib.load_model_card())

with beat1:
    st.subheader("The raw field, full range")
    st.caption(
        "bx/by/bz per axis against the ~45,000 nT background. Dashed orange lines mark true "
        "defects, dashed grey lines mark true interference sources -- if the anomaly were "
        "visible here, background removal wouldn't be the hard part of this project. "
        "Scroll/drag to zoom and pan."
    )
    st.altair_chart(chart_utils.raw_components_chart(raw), width="stretch")

    st.markdown("**|B| deviation from its own median** -- the same signal, one transform closer to visible")
    scale_choice = st.radio(
        "Y-axis", ["Log", "Linear"], index=0, horizontal=True, key="beat1_scale",
        help="Log scale is the direct answer to 'I need log scale to see anomalies' -- raw "
             "bx/by/bz are signed and can't be log-scaled directly, but this deviation can.",
    )
    st.caption(
        "Some peaks here won't line up with a dashed marker -- expected, not a bug: this is "
        "the RAW, un-detrended view, so slow background drift (not a discrete source, often "
        "worst near the survey's own start/end) can also produce a large deviation from the "
        "median. Beat 2's detrended residual is the fair comparison."
    )
    st.altair_chart(
        chart_utils.deviation_chart(raw, log_scale=(scale_choice == "Log")), width="stretch"
    )

with beat2:
    st.subheader("Background removed")
    if features is None:
        st.warning("No features computed for this scenario -- it was refused at validation. See tab 6.")
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
        budget_choice = st.segmented_control(
            "Dig budget (per km)", ["3", "5", "10"], default="5", key="dig_budget", required=True,
        )
        budget_n = max(1, round(int(budget_choice or 5) * length_km))
        # risk_score (Stage 5: calibrated P(defect) x severity x consequence) is the
        # real dig-priority ranking once a classify model has been released; it's
        # NULL on every row until then (no severity model either), so fall back to
        # anomaly_score rather than silently dig-ranking by an all-NULL column.
        if "risk_score" in indications.columns and indications["risk_score"].notna().any():
            rank_col = "risk_score"
        else:
            rank_col = "anomaly_score"
        ranked = indications.sort_values(rank_col, ascending=False, na_position="last")
        dug = ranked.head(budget_n)

        st.pydeck_chart(build_map(raw, dug))
        st.caption("orange = true defect - grey = true interference (unlabelled to the model) - blue = dug indication")
        st.caption(f"Ranked by **{rank_col}**.")

        st.altair_chart(chart_utils.indications_chart(dug, raw), width="stretch")
        st.caption(
            "Same dug indications as the map and table below, plotted by predicted severity "
            "(with its 90% interval) and type -- the table's numbers, made visible."
        )

        show_cols = [c for c in [
            "chainage_peak_m", "anomaly_score", "p_defect_cal", "pred_type", "pred_type_conf",
            "sev_pred", "sev_lo", "sev_hi", "risk_score", "dq_flag",
        ] if c in dug.columns]
        st.dataframe(dug[show_cols], width="stretch")

        st.altair_chart(chart_utils.indications_rank_chart(dug), width="stretch")
        st.caption(f"The table above, ranked and labeled -- one bar per row, by **{rank_col}**.")

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
                failures = r["detail"].get("failures") if r["detail"] else None
                if failures:
                    for f in failures:
                        where = f"row {f['row_index']}" if f["row_index"] is not None else "the whole survey"
                        column = f"`{f['column']}`" if f["column"] else "a"
                        st.markdown(f"- {column} at {where} failed `{f['check']}` (value: `{f['failure_case']}`)")
                elif r["detail"]:
                    st.json(r["detail"])

st.divider()
pr = manifest["pipeline_release"]
anomaly_git_sha = next(iter(manifest["model_run"].values()))["git_sha"] if manifest["model_run"] else "unknown"
st.caption(
    f"pipeline_version={pr['pipeline_version']} | feature_version={manifest['feature_version']} | "
    f"schema_version={manifest['schema_version']} | config_sha256={manifest['config_sha256'][:12]}... | "
    f"git_sha={anomaly_git_sha[:12]}... | mode={mode}"
)
