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

Run: streamlit run app/demo_app.py
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import demo_lib  # noqa: E402
from map_utils import build_map  # noqa: E402

from lsm.config import load_config  # noqa: E402

st.set_page_config(layout="wide", page_title="LSM demo -- Stage 4.5", initial_sidebar_state="collapsed")

manifest = demo_lib.load_manifest()
scenarios = demo_lib.demo_scenarios()

top = st.columns([1, 2])
with top[0]:
    mode = st.segmented_control(
        "Mode", ["demo", "live"], default=os.environ.get("APP_MODE", "demo"), key="app_mode"
    )
with top[1]:
    labels = [s["label"] for s in scenarios]
    selected_label = st.segmented_control("Scenario", labels, default=labels[0], key="scenario_label")

scenario = next(s for s in scenarios if s["label"] == selected_label)
survey_id = scenario["survey_id"]

st.title(f"{scenario['label']} ({survey_id})")

if mode == "demo":
    raw = demo_lib.load_demo_raw(survey_id)
    features = demo_lib.load_demo_features(survey_id)
    indications = demo_lib.load_demo_indications(survey_id)
    dq_failure = demo_lib.load_demo_dq_failure(survey_id)
else:
    if "live_conn" not in st.session_state:
        base_cfg = load_config("dev")
        conn, live_cfg = demo_lib.init_live_session(base_cfg)
        st.session_state["live_conn"] = conn
        st.session_state["live_cfg"] = live_cfg
    live_cfg = st.session_state["live_cfg"]

    raw = demo_lib.load_demo_raw(survey_id)
    with st.spinner("Running validate -> features -> score..."):
        report, indications = demo_lib.run_live_scenario(
            st.session_state["live_conn"], live_cfg, survey_id
        )
    if report.has_fail:
        dq_failure = {
            "survey_id": report.survey_id,
            "checked_at": report.checked_at,
            "results": [dataclasses.asdict(r) for r in report.results],
        }
        features = None
    else:
        dq_failure = None
        features = demo_lib.load_live_features(live_cfg, survey_id)

beat1, beat2, beat3, beat4 = st.tabs([
    "1. Raw signal", "2. Detrend + gradient", "3. Ranked indications", "4. Corrupted survey",
])

with beat1:
    st.caption("The full-range raw field. The defect is invisible against ~45,000 nT background.")
    r_mag = np.sqrt(raw["bx_nt"] ** 2 + raw["by_nt"] ** 2 + raw["bz_nt"] ** 2)
    st.line_chart(pd.DataFrame({"chainage_m": raw["chainage_m"], "|B| (nT)": r_mag}).set_index("chainage_m"))

with beat2:
    if features is None:
        st.warning("No features computed for this scenario (it was refused at validation -- see tab 4).")
    else:
        st.caption("Background removed: the residual and the along-track gradient. Defects AND interference sources both appear.")
        cols = {"chainage_m": features["chainage_m"], "residual |r| (nT)": features["r_mag_nt"]}
        if "g_mag_nt_per_m" in features.columns:
            cols["gradient (nT/m)"] = features["g_mag_nt_per_m"]
        st.line_chart(pd.DataFrame(cols).set_index("chainage_m"))

with beat3:
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
