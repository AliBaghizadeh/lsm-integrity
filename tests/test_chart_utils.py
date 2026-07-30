"""chart_utils.py -- Streamlit-free, so testable without a browser."""

from __future__ import annotations

import numpy as np
import pandas as pd

import chart_utils


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
