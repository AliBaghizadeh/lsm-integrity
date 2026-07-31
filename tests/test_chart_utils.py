"""chart_utils.py -- Streamlit-free, so testable without a browser."""

from __future__ import annotations

import chart_utils
import numpy as np
import pandas as pd


def _toy_raw(n=200):
    chainage = np.arange(n) * 0.5
    defect = np.zeros(n, dtype=int)
    defect[20:26] = 1  # one contiguous run
    defect[100:108] = 1  # a second contiguous run
    interference = np.zeros(n, dtype=int)
    interference[60:65] = 1
    return pd.DataFrame({
        "chainage_m": chainage,
        "bx_nt": np.random.default_rng(0).normal(0, 5, n),
        "by_nt": np.random.default_rng(1).normal(0, 5, n),
        "bz_nt": np.random.default_rng(2).normal(45000, 5, n),
        "defect": defect,
        "interference": interference,
    })


def test_contiguous_midpoints_finds_one_marker_per_run():
    raw = _toy_raw()
    mids = chart_utils.contiguous_midpoints(raw, "defect")
    assert len(mids) == 2
    assert mids[0] < mids[1]
    # each midpoint should land inside its own run's chainage span
    assert 10.0 <= mids[0] <= 13.0
    assert 50.0 <= mids[1] <= 54.0


def test_contiguous_midpoints_empty_when_no_flags_set():
    raw = _toy_raw()
    raw["defect"] = 0
    assert chart_utils.contiguous_midpoints(raw, "defect") == []


def test_raw_components_chart_builds_without_error():
    chart = chart_utils.raw_components_chart(_toy_raw())
    assert chart is not None


def test_raw_components_chart_does_not_hit_altairs_row_limit_on_a_real_size_survey():
    """Regression: a 4000-row survey melted into 3 axes is 12,000 rows,
    over Altair's default 5000-row cap -- previously an EMPTY chart (only
    the truth-marker layer rendered), not a visible error."""
    raw = _toy_raw(n=4000)
    chart = chart_utils.raw_components_chart(raw)
    spec = chart.to_dict()  # raises MaxRowsError if the row cap isn't disabled
    assert spec is not None


def test_deviation_chart_switches_between_log_and_linear_scale():
    log_chart = chart_utils.deviation_chart(_toy_raw(), log_scale=True)
    linear_chart = chart_utils.deviation_chart(_toy_raw(), log_scale=False)
    log_spec = log_chart.to_dict()
    linear_spec = linear_chart.to_dict()
    log_scale_type = log_spec["layer"][0]["encoding"]["y"]["scale"]["type"]
    linear_scale_type = linear_spec["layer"][0]["encoding"]["y"]["scale"]["type"]
    assert log_scale_type == "log"
    assert linear_scale_type == "linear"


def test_residual_gradient_chart_with_and_without_gradient_column():
    raw = _toy_raw()
    features = pd.DataFrame({"chainage_m": raw["chainage_m"], "r_mag_nt": np.random.default_rng(3).normal(0, 2, len(raw))})

    chart_no_grad = chart_utils.residual_gradient_chart(features, raw)
    assert chart_no_grad is not None

    features["g_mag_nt_per_m"] = np.random.default_rng(4).normal(0, 0.5, len(raw))
    chart_with_grad = chart_utils.residual_gradient_chart(features, raw)
    assert chart_with_grad is not None


def _toy_dug():
    return pd.DataFrame({
        "chainage_peak_m": [10.0, 52.0],
        "anomaly_score": [0.8, 0.6],
        "p_defect_cal": [0.9, 0.4],
        "pred_type": ["scc", "interference"],
        "pred_type_conf": [0.7, 0.95],
        "sev_pred": [55.0, np.nan],
        "sev_lo": [40.0, np.nan],
        "sev_hi": [70.0, np.nan],
        "risk_score": [0.5, 0.0],
    })


def test_indications_chart_plots_severity_with_interval_when_present():
    chart = chart_utils.indications_chart(_toy_dug(), _toy_raw())
    spec = chart.to_dict()
    y_fields = {layer["encoding"]["y"]["field"] for layer in spec["layer"] if "y" in layer.get("encoding", {})}
    assert "sev_pred" in y_fields
    assert "sev_lo" in y_fields  # the error-bar layer


def test_indications_chart_falls_back_to_anomaly_score_when_severity_is_all_null():
    dug = _toy_dug()
    dug["sev_pred"] = np.nan
    dug["sev_lo"] = np.nan
    dug["sev_hi"] = np.nan
    chart = chart_utils.indications_chart(dug, _toy_raw())
    spec = chart.to_dict()
    y_fields = {layer["encoding"]["y"]["field"] for layer in spec["layer"] if "y" in layer.get("encoding", {})}
    assert y_fields == {"anomaly_score"}  # no error-bar layer, no sev_pred


def test_indications_rank_chart_ranks_by_risk_score_when_present():
    chart = chart_utils.indications_rank_chart(_toy_dug())
    spec = chart.to_dict()
    x_fields = {layer["encoding"]["x"]["field"] for layer in spec["layer"]}
    assert x_fields == {"risk_score"}


def test_indications_rank_chart_falls_back_to_anomaly_score_when_risk_is_all_null():
    dug = _toy_dug()
    dug["risk_score"] = np.nan
    chart = chart_utils.indications_rank_chart(dug)
    spec = chart.to_dict()
    x_fields = {layer["encoding"]["x"]["field"] for layer in spec["layer"]}
    assert x_fields == {"anomaly_score"}


def test_indications_rank_chart_one_bar_per_row():
    dug = _toy_dug()
    spec = chart_utils.indications_rank_chart(dug).to_dict()
    dataset = next(iter(spec["datasets"].values()))  # Altair hoists inline data by name
    assert len(dataset) == len(dug)
