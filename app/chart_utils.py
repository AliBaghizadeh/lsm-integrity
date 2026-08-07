"""
Signal-plot builders for the demo app's Beat 1/2 -- Streamlit-free (returns
Altair `Chart` objects / plain data), so directly pytest-testable without a
browser, same reasoning as `map_utils.py` and `demo_lib.py`.

Ali's feedback on the first cut: Beat 1 showed only one line (|B| magnitude),
gave no way to see that the anomaly is genuinely there (just buried), and the
app didn't explain what a "survey" even is. This fixes the plots; the
explanatory text lives in `demo_app.py` itself.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

# Written when a raw survey was ~4,000 rows (0.5 m fixed grid). Rig-v2's
# walked rig samples at 120 Hz, so a single survey is now ~200,000 rows --
# melting 3 axes over that is 600,000+ points handed to Vega-Lite as inline
# JSON, which is what made the app "very slow" (seconds of melt/serialize on
# every rerun, then a browser struggling to render that many marks). Still
# disabled, because MAX_PLOT_ROWS below already caps what actually reaches
# Altair -- this just stops the (now-irrelevant) 5000-row warning from firing
# on the pre-downsample frame in any code path that skips _thin().
alt.data_transformers.disable_max_rows()

TRUE_DEFECT_COLOR = "#dc5028"
TRUE_INTERFERENCE_COLOR = "#828282"

# Target point count for the full-survey line charts (Beat 1/2). Chosen the
# same way map_utils.build_map thins the track: way more than a screen has
# pixels for, so downsampling is visually lossless, but far below the point
# count that makes melt()/JSON-serialize/browser-render slow.
MAX_PLOT_ROWS = 3000


def _thin(df: pd.DataFrame) -> pd.DataFrame:
    """Stride-sample a survey-length frame down to ~MAX_PLOT_ROWS for
    plotting only -- never used for anything the model or a metric reads."""
    stride = max(1, len(df) // MAX_PLOT_ROWS)
    return df.iloc[::stride] if stride > 1 else df


def _raw_chainage_col(raw: pd.DataFrame) -> str:
    """Rig-v2: raw no longer carries a plain `chainage_m` (see schemas.py --
    it is a feature-layer output of registration.register_survey now). The
    demo app's baked scenarios are synthetic, so `chainage_true_m` (truth-
    tier, but only ever used here for PLOTTING the demo's own known-good
    survey, never as a model input) is the honest stand-in; fall back to
    `chainage_m` for any caller still passing an old-schema/feature-layer
    frame that already has it."""
    return "chainage_m" if "chainage_m" in raw.columns else "chainage_true_m"


def contiguous_midpoints(raw: pd.DataFrame, flag_col: str) -> list[float]:
    """Chainage midpoint of each contiguous run where `raw[flag_col] == 1` --
    one marker per physical feature, not one per row (a feature spans many
    rows at 0.5 m sampling)."""
    flag = raw[flag_col].to_numpy()
    if not flag.any():
        return []
    padded = np.concatenate(([0], flag, [0]))
    change = np.diff(padded)
    starts = np.where(change == 1)[0]
    ends = np.where(change == -1)[0]  # exclusive
    chainage = raw[_raw_chainage_col(raw)].to_numpy()
    return [float(chainage[s:e].mean()) for s, e in zip(starts, ends)]


def truth_rule_layer(raw: pd.DataFrame) -> alt.Chart:
    """Vertical dashed rules marking true defect / interference locations --
    layered under a signal chart so it's visible whether or not the raw trace
    itself shows anything there. That's the whole point of Beat 1: the rules
    are there, the signal isn't (visibly)."""
    rows = (
        [{"chainage_m": c, "kind": "true defect"} for c in contiguous_midpoints(raw, "defect")]
        + [{"chainage_m": c, "kind": "true interference"} for c in contiguous_midpoints(raw, "interference")]
    )
    df = pd.DataFrame(rows, columns=["chainage_m", "kind"])
    return (
        alt.Chart(df)
        .mark_rule(strokeDash=[4, 3])
        .encode(
            x="chainage_m:Q",
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(
                    domain=["true defect", "true interference"],
                    range=[TRUE_DEFECT_COLOR, TRUE_INTERFERENCE_COLOR],
                ),
                legend=alt.Legend(title=None),
            ),
        )
    )


def raw_components_chart(raw: pd.DataFrame) -> alt.Chart:
    """b_lo/mid/hi vs chainage, full range, three separate lines -- not a
    single |B| magnitude line, which throws away head information and (per
    Ali's feedback) reads as "only one line, nothing to look at". Rig-v2: the
    three lines are the rod's three total-field HEADS, not vector axes --
    there is no x/y/z under the scalar rig (generate.py's module docstring)."""
    chainage_col = _raw_chainage_col(raw)
    long_df = _thin(raw).melt(
        id_vars=[chainage_col], value_vars=["b_lo_nt", "b_mid_nt", "b_hi_nt"],
        var_name="head", value_name="field_nT",
    ).rename(columns={chainage_col: "chainage_m"})
    lines = (
        alt.Chart(long_df)
        .mark_line()
        .encode(
            x=alt.X("chainage_m:Q", title="chainage (m)"),
            y=alt.Y("field_nT:Q", title="field (nT)"),
            color=alt.Color("head:N", legend=alt.Legend(title=None)),
        )
    )
    return alt.layer(lines, truth_rule_layer(raw)).properties(height=320).interactive()


def deviation_chart(raw: pd.DataFrame, log_scale: bool = True) -> alt.Chart:
    """|b_mid - median(b_mid)| vs chainage, switchable between a log and a
    linear y-axis. The raw field is centred on a huge background, so a
    literal log-scale of b_lo/mid/hi doesn't mean anything -- but the
    ABSOLUTE deviation from the survey's own median does, and on a log axis
    it's the one view where the anomaly stops being invisible. This is the
    direct answer to "I need log scale to see anomalies" -- `log_scale` is a
    real switch, not a one-way toggle to show/hide the chart. Uses the middle
    head as the single representative signal (same choice validate.py's
    check_background_regime / monitor.py make for consistency)."""
    # Median computed on the full-resolution signal (robust statistic, cheap
    # even at 200k rows) -- only the plotted deviation trace is thinned.
    median = raw["b_mid_nt"].median()
    thin = _thin(raw)
    dev = (thin["b_mid_nt"] - median).abs().clip(lower=1e-3)
    df = pd.DataFrame({"chainage_m": thin[_raw_chainage_col(thin)], "deviation_nT": dev})
    scale = alt.Scale(type="log") if log_scale else alt.Scale(type="linear")
    y_title = "|deviation| from median |B| (nT" + (", log scale)" if log_scale else ")")
    line = (
        alt.Chart(df)
        .mark_line(color="#2878dc")
        .encode(
            x=alt.X("chainage_m:Q", title="chainage (m)"),
            y=alt.Y("deviation_nT:Q", title=y_title, scale=scale),
        )
    )
    return alt.layer(line, truth_rule_layer(raw)).properties(height=320).interactive()


def residual_gradient_chart(features: pd.DataFrame, raw: pd.DataFrame) -> alt.Chart:
    """Background-removed residual, plus the first-difference gradient across
    the 3 heads -- the Beat 2 "now it appears" view, with the same
    true-location markers so the reveal is visually obvious. Rig-v2:
    `r_mag_nt`/`g_mag_nt_per_m` (vector magnitude columns) are gone --
    `r_mid_nt` (the middle head's own detrended residual) and `g1_nt_per_m`
    (first difference across heads) are their direct successors, see
    features.py's feature_columns() docstring."""
    features = _thin(features)
    resid_df = pd.DataFrame({"chainage_m": features["chainage_m"], "residual_nT": features["r_mid_nt"]})
    residual = (
        alt.Chart(resid_df)
        .mark_line(color=TRUE_DEFECT_COLOR)
        .encode(x=alt.X("chainage_m:Q", title="chainage (m)"), y=alt.Y("residual_nT:Q", title="residual r_mid (nT)"))
    )
    if "g1_nt_per_m" in features.columns:
        grad_df = pd.DataFrame({"chainage_m": features["chainage_m"], "gradient_nT_per_m": features["g1_nt_per_m"]})
        gradient = (
            alt.Chart(grad_df)
            .mark_line(color="#2878dc")
            .encode(x="chainage_m:Q", y=alt.Y("gradient_nT_per_m:Q", title="gradient (nT/m)"))
        )
        combined = alt.layer(residual, gradient).resolve_scale(y="independent")
    else:
        combined = residual
    return alt.layer(combined, truth_rule_layer(raw)).properties(height=320).interactive()


def indications_chart(dug: pd.DataFrame, raw: pd.DataFrame) -> alt.Chart:
    """Beat 4's ranked table, made visible: predicted severity (with its 90%
    conformal interval, if a severity model has been released) vs chainage
    for the currently dug indications, colored by predicted defect type.
    Falls back to `anomaly_score` with no error bars pre-Stage-4/5 (no
    severity model released yet) -- same fallback reasoning as
    `demo_app.py`'s risk_score dig-ranking.
    """
    df = dug.copy()
    has_severity = "sev_pred" in df.columns and df["sev_pred"].notna().any()
    y_col = "sev_pred" if has_severity else "anomaly_score"
    y_title = "predicted severity (%SMYS)" if has_severity else "anomaly score"
    has_type = "pred_type" in df.columns and df["pred_type"].notna().any()

    color = (
        alt.Color("pred_type:N", legend=alt.Legend(title="predicted type"))
        if has_type
        else alt.value("#2878dc")
    )
    tooltip = [
        c for c in [
            "chainage_peak_m", "pred_type", "pred_type_conf", "sev_pred",
            "risk_score", "anomaly_score", "p_defect_cal",
        ] if c in df.columns
    ]

    points = (
        alt.Chart(df)
        .mark_point(filled=True, size=100)
        .encode(
            x=alt.X("chainage_peak_m:Q", title="chainage (m)"),
            y=alt.Y(f"{y_col}:Q", title=y_title),
            color=color,
            tooltip=tooltip,
        )
    )
    layers = [points]
    if has_severity and "sev_lo" in df.columns and "sev_hi" in df.columns:
        layers.append(
            alt.Chart(df)
            .mark_errorbar()
            .encode(x="chainage_peak_m:Q", y=alt.Y("sev_lo:Q", title=y_title), y2="sev_hi:Q", color=color)
        )
    layers.append(truth_rule_layer(raw))
    return alt.layer(*layers).properties(height=320).interactive()


def indications_rank_chart(dug: pd.DataFrame) -> alt.Chart:
    """One labeled, sorted horizontal bar per dug indication -- the table
    below the map read in one glance instead of row by row. Ranked (and
    valued) by `risk_score` if a classify model has scored these indications,
    `anomaly_score` otherwise -- same fallback reasoning as `demo_app.py`'s
    dig-ranking and `indications_chart` above.
    """
    df = dug.copy()
    has_risk = "risk_score" in df.columns and df["risk_score"].notna().any()
    value_col = "risk_score" if has_risk else "anomaly_score"
    value_title = "risk score" if has_risk else "anomaly score"
    has_type = "pred_type" in df.columns and df["pred_type"].notna().any()

    df["label"] = df["chainage_peak_m"].round(0).astype(int).astype(str) + " m"
    if has_type:
        df["label"] = df["label"] + " -- " + df["pred_type"].fillna("unclassified")

    color = (
        alt.Color("pred_type:N", legend=alt.Legend(title="predicted type"))
        if has_type
        else alt.value("#2878dc")
    )
    tooltip = [
        c for c in [
            "chainage_peak_m", "pred_type", "pred_type_conf", "sev_pred",
            "risk_score", "anomaly_score", "p_defect_cal",
        ] if c in df.columns
    ]

    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{value_col}:Q", title=value_title),
            y=alt.Y("label:N", sort="-x", title="indication (chainage -- type)"),
            color=color,
            tooltip=tooltip,
        )
    )
    labels = (
        alt.Chart(df)
        .mark_text(align="left", dx=3)
        .encode(
            x=f"{value_col}:Q",
            y=alt.Y("label:N", sort="-x"),
            text=alt.Text(f"{value_col}:Q", format=".2f"),
        )
    )
    return (bars + labels).properties(height=max(120, 32 * len(df)))


def block_heatmap_chart(blocks: pd.DataFrame) -> alt.Chart:
    """(line, 100 m block) risk matrix -- output of
    `lsm.indications.block_risk_heatmap`, one rect per non-empty cell.
    Sequential single-hue ramp (light -> dark = low -> high), never a
    rainbow, matching this app's other "hotter = more concerning" cues
    (the pydeck heatmap layer, `TRUE_DEFECT_COLOR`'s orange-red family)."""
    if blocks.empty:
        return alt.Chart(pd.DataFrame({"msg": ["no indications yet"]})).mark_text().encode(text="msg:N")

    metric = blocks["metric"].iloc[0]
    value_title = "risk score" if metric == "risk_score" else "anomaly score"

    # Cell size scales with how many blocks/lines there are, targeting a
    # comfortably wide chart (~1100px) rather than a fixed small cell --
    # clamped so a handful of blocks doesn't produce giant cells and a huge
    # scale-rehearsal-sized corpus doesn't produce unreadably thin ones (it
    # scrolls instead).
    n_blocks = blocks["block_start_m"].nunique()
    n_lines = blocks["line_id"].nunique()
    cell_w = min(60, max(28, 1100 // max(n_blocks, 1)))
    cell_h = min(60, max(32, 320 // max(n_lines, 1)))

    return (
        alt.Chart(blocks)
        .mark_rect()
        .encode(
            x=alt.X("block_start_m:O", title="chainage block (100 m)"),
            y=alt.Y("line_id:N", title="line"),
            color=alt.Color("value:Q", title=value_title, scale=alt.Scale(scheme="oranges")),
            tooltip=[
                "line_id", "block_start_m",
                alt.Tooltip("value:Q", title=value_title, format=".3f"),
                alt.Tooltip("n_indications:Q", title="indications in block"),
            ],
        )
        # Both axes are discrete (block, line) -- a fixed total height/width
        # combined with Streamlit's container-stretch resize collapses every
        # row to a sliver (a known Vega-Lite "fit" autosize interaction with
        # two discrete scales). alt.Step gives each cell a fixed per-category
        # size instead of a total-size target, which sidesteps it; the caller
        # must display this with width="content", not "stretch".
        .properties(width=alt.Step(cell_w), height=alt.Step(cell_h))
    )
