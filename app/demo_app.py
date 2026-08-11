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

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import chart_utils
import demo_lib
from map_utils import build_map

from lsm.config import load_config
from lsm.indications import block_risk_heatmap

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
    _, hero_center, _ = st.columns([1, 2, 1])
    with hero_center:
        st.markdown("<h1 style='text-align:center'>AI in Action</h1>", unsafe_allow_html=True)
        st.markdown(
            "<h3 style='text-align:center;font-weight:400'>Inspecting pipelines before they fail</h3>",
            unsafe_allow_html=True,
        )
        st.write(
            "A live walkthrough of an LSM pipeline-integrity pipeline -- from a raw "
            "~48,800 nT scalar magnetometer signal (3 heads on a walked rod) to a "
            "risk-ranked dig list, scored end to end by real, trained models."
        )
        components_img = Path(__file__).resolve().parents[1] / "img" / "project-components.png"
        if components_img.exists():
            st.image(str(components_img), width="stretch")
        _, button_center, _ = st.columns([1, 1, 1])
        with button_center:
            if st.button("Launch the demo  →", type="primary", width="stretch"):
                st.session_state["entered"] = True
                st.rerun()
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

overview, performance, beat1, beat2, beat3, beat4, heatmap_tab = st.tabs([
    "1. How it works", "2. Model performance", "3. Raw signal", "4. Detrend + gradient",
    "5. Ranked indications", "6. Corrupted survey", "7. Risk heatmap",
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
    ("3. Register", "lsm.registration",
     ("GPS dead-reckoning + girth-weld-comb detection -> chainage_m. Raw no longer "
     "carries a usable position column -- the walker's speed is irregular and GPS "
     "drops out, so along-track position has to be reconstructed, not read off.")),
    ("4. Features", "lsm.features",
     ("Per-survey detrend, per-head first/second difference (g1/g2), stand-off "
     "inversion, sliding-window stats, peak-shape descriptors (FWHM, asymmetry, "
     "decay exponent).")),
]
_STAGES_2 = [
    ("5. Detect", "lsm.models.anomaly / lsm.indications",
     ("MAD baseline or IsolationForest scores every row; contiguous flagged rows "
     "cluster into one indication.")),
    ("6. Severity + Classify", "lsm.models.severity / lsm.models.classify",
     ("Two independent models score the SAME indication in parallel: a calibrated "
     "5/50/95% severity interval, and a defect-type class + confidence.")),
    ("7. Risk rank", "lsm.indications.attach_classification",
     ("risk_score = calibrated P(defect) x severity x a stated consequence proxy. "
     "Tab 5's table is sorted by this.")),
]

_DQ_CHECKS = [
    ("schema", "Column presence/dtypes/units/enum values against the declared data contract."),
    ("range", "b_lo/mid/hi (the rod's three scalar heads) each stay inside the sensor's physical field range."),
    ("saturation", "No stuck sensor / ADC rail -- N+ consecutive identical raw values on any head."),
    ("sample_idx_monotonic", "sample_idx strictly increasing, no reordering."),
    ("sample_idx_gap", "No missing samples beyond the allowed spacing tolerance."),
    ("duplicate_sample_idx", "No repeated sample_idx within one survey."),
    ("duplicate_content", "This survey's content hash doesn't already exist under a different run -- a re-export."),
    ("survey_overlap", "Cross-correlation against other accepted runs of the same line -- an overlapping re-run neither hash catches."),
    ("gps_jump", "No physically-impossible lat/lon jump between consecutive locked GPS fixes."),
    ("gps_chainage_consistency", "GPS-derived path length over locked stretches agrees with the true along-track distance over those same rows."),
    ("noise_floor", "Raw-signal noise sits within the expected sensor-floor band."),
    ("background_regime", "This run's background statistics haven't shifted from the line's prior accepted runs."),
    ("interference_density", "The fraction of interference-like readings is within the expected range."),
    ("coverage", "The survey covers its expected chainage length with no large unexplained gaps."),
]

with overview:
    st.subheader("One survey, seven stages")
    st.caption(
        "Every scenario below runs this same path, live or precomputed -- this tab is a map "
        "of it, not a beat of its own. A hard validation failure is the one place the path "
        "stops early (tab 6)."
    )

    row1 = st.columns([3, 1, 3, 1, 3, 1, 3])
    for i, (name, module, desc) in enumerate(_STAGES):
        with row1[i * 2], st.container(border=True):
            st.markdown(f"**{name}**")
            st.caption(f"`{module}`")
            st.caption(desc)
    with row1[1]:
        st.markdown("### →")
    with row1[3]:
        st.markdown("### →")
    with row1[5]:
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

    # Why the pipeline above needs all seven stages, in two pictures. Rendered
    # from committed PNGs (scripts/plot_signal_decomposition.py regenerates
    # them) rather than plotted live: they need the generator's own internal
    # per-source traces, which a served survey does not carry -- the app only
    # ever sees the summed field, which is the entire point being made here.
    # Guarded on .exists() like the landing-page image, so a missing asset
    # degrades to no image instead of crashing the tab on demo day.
    _IMG_DIR = Path(__file__).resolve().parents[1] / "img"
    with st.expander("Why this is hard -- the signal, in pieces"):
        components_png = _IMG_DIR / "signal_components.png"
        decomposition_png = _IMG_DIR / "signal_decomposition.png"
        st.markdown(
            "**Each physical source on its own scale, over one 2 km line.** Girth welds are "
            "the loudest thing in the data by a wide margin, and there are ~163 of them per "
            "line. Interference is rarer but still reaches over 150 nT. The defects -- the "
            "only thing anyone actually wants -- peak around 11 nT. **The thing being looked "
            "for is the smallest signal present**, which is why stages 3 and 4 above exist at "
            "all."
        )
        if components_png.exists():
            st.image(str(components_png), width="stretch")
        st.markdown(
            "**The same survey, from raw field to isolated defect signal.** The raw trace "
            "hides everything under the ~48,800 nT background; two-stage detrending "
            "(`lsm.features`) exposes the residual; masking the weld and interference windows "
            "leaves what the detector is actually asked to find."
        )
        if decomposition_png.exists():
            st.image(str(decomposition_png), width="stretch")

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
        "b_lo/mid/hi -- the rod's three scalar total-field heads -- against the ~48,800 nT "
        "background. Dashed orange lines mark true defects, dashed grey lines mark true "
        "interference sources -- if the anomaly were visible here, background removal "
        "wouldn't be the hard part of this project. Scroll/drag to zoom and pan."
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
        _chainage_col = "chainage_m" if "chainage_m" in raw.columns else "chainage_true_m"
        length_km = max(float(raw[_chainage_col].max()) / 1000.0, 0.001)
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
        st.caption(
            f"**{budget_choice} digs/km** x {length_km:.2f} km surveyed -> top **{budget_n}** "
            f"of {len(indications)} detected indication(s) get dug (ranked by {rank_col}); the "
            f"rest are left unexamined. This is a real inspection-cost tradeoff, not a display "
            f"setting (`lsm.indications.select_dig_budget`) -- a higher budget catches more "
            f"true defects but pays for more digs, including false ones. This is exactly what "
            f"'recall @ dig budget' (tab 2) is measured against."
        )

        show_heatmap = st.checkbox("Show risk heatmap on map", value=False, key="beat3_heatmap")
        st.pydeck_chart(build_map(raw, dug, show_heatmap=show_heatmap))
        st.caption(
            "orange = true defect - grey = true interference (unlabelled to the model) - "
            "blue = dug indication (or a heatmap glow, weighted by risk_score/anomaly_score, "
            "if the box above is checked)"
        )
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

        st.markdown("**Risk by chainage block, this line**")
        st.caption(
            "The same dug indications, aggregated into the 100 m blocks "
            "`lsm.indications.block_risk_heatmap` uses -- WHERE risk clusters along this one "
            "line, complementing the rank order above. Tab 7 does the same aggregation across "
            "every line at once."
        )
        line_dug = dug.assign(line_id=survey_id.split("_R")[0])
        line_blocks = block_risk_heatmap(line_dug)
        if line_blocks.empty:
            st.info("No indications above threshold for this scenario.")
        else:
            st.altair_chart(chart_utils.block_heatmap_chart(line_blocks), width="content")

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

with heatmap_tab:
    st.subheader("Risk heatmap across lines")
    st.caption(
        "Indications from every scenario run so far, aggregated into (line, 100 m block) "
        "cells -- `lsm.indications.block_risk_heatmap`, the same block grouping "
        "`evaluate.assign_group()` uses for CV folds. A CONSUMER of the `indication` table "
        "across surveys, not a new model."
    )

    frames = []
    if mode == "demo":
        for s in scenarios:
            if s["kind"] != "clean":
                continue  # the corrupted scenario is quarantined -- never has indications
            ind = _cached_demo_indications(s["survey_id"])
            if ind is not None and len(ind):
                ind = ind.copy()
                ind["line_id"] = s["survey_id"].split("_R")[0]
                frames.append(ind)
    else:
        for sid, (_report, ind, _feat) in live_results.items():
            if ind is not None and len(ind):
                ind = ind.copy()
                ind["line_id"] = sid.split("_R")[0]
                frames.append(ind)
        n_clean = sum(1 for s in scenarios if s["kind"] == "clean")
        if len(frames) < n_clean:
            st.caption(
                "Live mode only includes scenarios you've already run this session -- pick "
                "each clean scenario above (with Mode = live) to add its line here."
            )

    if not frames:
        st.info("No indications available yet for a cross-line heatmap.")
    else:
        all_indications = pd.concat(frames, ignore_index=True)
        blocks = block_risk_heatmap(all_indications)
        if blocks.empty:
            st.info("No indications above threshold in any scenario yet.")
        else:
            st.altair_chart(chart_utils.block_heatmap_chart(blocks), width="content")
            st.caption(f"{len(frames)} line(s), {len(all_indications)} indication(s) total.")

st.divider()
pr = manifest["pipeline_release"]
anomaly_git_sha = next(iter(manifest["model_run"].values()))["git_sha"] if manifest["model_run"] else "unknown"
st.caption(
    f"pipeline_version={pr['pipeline_version']} | feature_version={manifest['feature_version']} | "
    f"schema_version={manifest['schema_version']} | config_sha256={manifest['config_sha256'][:12]}... | "
    f"git_sha={anomaly_git_sha[:12]}... | mode={mode}"
)
