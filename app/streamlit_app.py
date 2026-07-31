"""
Stage 3's thin app: the GPS track, detected indications overlaid, true defects
and interference sources shown separately, ranked table below the map. A
CONSUMER of the `indication` table, not the serving layer (SKILL invariant #14
-- the batch scoring job in train.py/predict.py is the server; this reads what
it wrote).

This is NOT the Stage 4.5 demo app: no APP_MODE=demo/live toggle, no pre-baked
scenarios, no HF Spaces deploy, no iPad ergonomics. It exists to see Stage 3's
own output, on the real (or tiny) local SQLite database, nothing more.

Run: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from map_utils import build_map

from lsm.config import load_config
from lsm.db import connect

st.set_page_config(layout="wide", page_title="LSM Stage 3 -- indications")


@st.cache_resource
def _get_conn_and_cfg(env: str):
    # check_same_thread=False: st.cache_resource can hand this connection back
    # on a different thread than the one that created it -- read-only usage
    # here, so the default cross-thread guard is just noise, not a real hazard.
    cfg = load_config(env)
    return connect(cfg.env.storage.sqlite_path, check_same_thread=False), cfg


conn, cfg = _get_conn_and_cfg("dev")

survey_ids = [r[0] for r in conn.execute("SELECT survey_id FROM survey ORDER BY survey_id").fetchall()]
if not survey_ids:
    st.error("No surveys registered. Run `lsm generate && lsm ingest` first.")
    st.stop()

survey_id = st.selectbox("Survey", survey_ids)

line_id, run_id, source_uri = conn.execute(
    "SELECT line_id, run_id, source_uri FROM survey WHERE survey_id=?", (survey_id,)
).fetchone()
raw = pd.read_parquet(source_uri)

pipeline_row = conn.execute(
    "SELECT pipeline_version FROM pipeline_release ORDER BY released_at DESC LIMIT 1"
).fetchone()

indications = pd.DataFrame()
pipeline_version = None
if pipeline_row:
    pipeline_version = pipeline_row[0]
    indications = pd.read_sql_query(
        "SELECT * FROM indication WHERE survey_id=? AND pipeline_version=? AND is_shadow=0",
        conn,
        params=(survey_id, pipeline_version),
    )

st.title(f"{survey_id} -- detected indications")
if pipeline_version:
    st.caption(
        f"pipeline_version={pipeline_version} · feature_version={cfg.base.features.version} "
        f"· config_sha256={cfg.config_sha256[:12]}..."
    )
else:
    st.warning("No trained pipeline yet -- run `lsm train` first. Showing ground truth only.")

st.pydeck_chart(build_map(raw, indications))
st.caption(
    "orange = true defect · grey = true interference (unlabelled to the model) · "
    "blue = detected indication"
)

st.subheader("Ranked indications")
if len(indications):
    ranked = indications.sort_values("anomaly_score", ascending=False)[
        ["chainage_peak_m", "anomaly_score", "p_defect_cal", "dq_flag", "chainage_start_m", "chainage_end_m"]
    ]
    st.dataframe(ranked, width="stretch")
else:
    st.info("No indications for this survey yet -- run `lsm predict <survey_id>` first.")
