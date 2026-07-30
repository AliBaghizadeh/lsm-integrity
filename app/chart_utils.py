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

# These are single-survey demo datasets (thousands of rows, not millions) --
# Altair's default 5000-row cap (aimed at genuinely huge datasets) was
# silently rejecting the 3-line raw components chart (4000 rows x 3 axes =
# 12,000 melted rows), which rendered as an EMPTY chart with only the truth-
# marker rule layer visible, not an error. Disabling it here is safe at this
# bounded scale; a real "millions of rows" dataset (Stage 6) would need
# server-side aggregation before plotting, not this.
alt.data_transformers.disable_max_rows()

TRUE_DEFECT_COLOR = "#dc5028"
TRUE_INTERFERENCE_COLOR = "#828282"


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
    chainage = raw["chainage_m"].to_numpy()
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
    """bx/by/bz vs chainage, full range, three separate lines -- not a single
    |B| magnitude line, which throws away axis information and (per Ali's
    feedback) reads as "only one line, nothing to look at"."""
    long_df = raw.melt(
        id_vars=["chainage_m"], value_vars=["bx_nt", "by_nt", "bz_nt"],
        var_name="axis", value_name="field_nT",
    )
    lines = (
        alt.Chart(long_df)
        .mark_line()
        .encode(
            x=alt.X("chainage_m:Q", title="chainage (m)"),
            y=alt.Y("field_nT:Q", title="field (nT)"),
            color=alt.Color("axis:N", legend=alt.Legend(title=None)),
        )
    )
    return alt.layer(lines, truth_rule_layer(raw)).properties(height=320).interactive()


def deviation_chart(raw: pd.DataFrame, log_scale: bool = True) -> alt.Chart:
    """|r_mag - median(r_mag)| vs chainage, switchable between a log and a
    linear y-axis. The raw field is signed and centred on a huge background,
    so a literal log-scale of bx/by/bz doesn't mean anything -- but the
    ABSOLUTE deviation from the survey's own median does, and on a log axis
    it's the one view where the anomaly stops being invisible. This is the
    direct answer to "I need log scale to see anomalies" -- `log_scale` is a
    real switch, not a one-way toggle to show/hide the chart."""
    r_mag = np.sqrt(raw["bx_nt"] ** 2 + raw["by_nt"] ** 2 + raw["bz_nt"] ** 2)
    dev = (r_mag - r_mag.median()).abs().clip(lower=1e-3)
    df = pd.DataFrame({"chainage_m": raw["chainage_m"], "deviation_nT": dev})
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
    """Background-removed residual, plus the along-track/vertical gradient if
    a second sensor head is present -- the Beat 2 "now it appears" view, with
    the same true-location markers so the reveal is visually obvious."""
    resid_df = pd.DataFrame({"chainage_m": features["chainage_m"], "residual_nT": features["r_mag_nt"]})
    residual = (
        alt.Chart(resid_df)
        .mark_line(color=TRUE_DEFECT_COLOR)
        .encode(x=alt.X("chainage_m:Q", title="chainage (m)"), y=alt.Y("residual_nT:Q", title="residual |r| (nT)"))
    )
    if "g_mag_nt_per_m" in features.columns:
        grad_df = pd.DataFrame({"chainage_m": features["chainage_m"], "gradient_nT_per_m": features["g_mag_nt_per_m"]})
        gradient = (
            alt.Chart(grad_df)
            .mark_line(color="#2878dc")
            .encode(x="chainage_m:Q", y=alt.Y("gradient_nT_per_m:Q", title="gradient (nT/m)"))
        )
        combined = alt.layer(residual, gradient).resolve_scale(y="independent")
    else:
        combined = residual
    return alt.layer(combined, truth_rule_layer(raw)).properties(height=320).interactive()
