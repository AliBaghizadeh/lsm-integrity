"""map_utils.py -- Streamlit-free, so testable without a browser (same
reasoning as test_chart_utils.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from map_utils import build_map


def _toy_raw(n=50):
    defect = np.zeros(n, dtype=int)
    defect[10:14] = 1
    interference = np.zeros(n, dtype=int)
    interference[30:33] = 1
    return pd.DataFrame(
        {
            "lat": 46.0 + np.arange(n) * 1e-5,
            "lon": 8.0 + np.arange(n) * 1e-5,
            "defect": defect,
            "defect_type": [None] * n,
            "interference": interference,
        }
    )


def _toy_indications():
    return pd.DataFrame(
        {
            "lat": [46.0001, 46.0003],
            "lon": [8.0001, 8.0003],
            "anomaly_score": [0.8, 0.6],
            "risk_score": [np.nan, np.nan],
        }
    )


def test_build_map_defaults_to_scatter_layer_for_indications():
    deck = build_map(_toy_raw(), _toy_indications())
    layer_types = [layer.type for layer in deck.layers]
    assert "HeatmapLayer" not in layer_types
    assert (
        layer_types.count("ScatterplotLayer") == 3
    )  # defects, interference, indications


def test_build_map_show_heatmap_swaps_indication_dots_for_a_heatmap_layer():
    deck = build_map(_toy_raw(), _toy_indications(), show_heatmap=True)
    layer_types = [layer.type for layer in deck.layers]
    assert "HeatmapLayer" in layer_types
    assert (
        layer_types.count("ScatterplotLayer") == 2
    )  # ground-truth defects/interference stay dots


def test_build_map_show_heatmap_falls_back_to_anomaly_score_when_risk_is_all_null():
    deck = build_map(_toy_raw(), _toy_indications(), show_heatmap=True)
    heat_layer = next(layer for layer in deck.layers if layer.type == "HeatmapLayer")
    assert heat_layer.get_weight == "@@=anomaly_score"


def test_build_map_show_heatmap_uses_risk_score_when_present():
    indications = _toy_indications()
    indications["risk_score"] = [0.3, 0.9]
    deck = build_map(_toy_raw(), indications, show_heatmap=True)
    heat_layer = next(layer for layer in deck.layers if layer.type == "HeatmapLayer")
    assert heat_layer.get_weight == "@@=risk_score"


def test_build_map_with_no_indications_never_adds_a_heatmap_layer():
    deck = build_map(_toy_raw(), None, show_heatmap=True)
    layer_types = [layer.type for layer in deck.layers]
    assert "HeatmapLayer" not in layer_types


def test_build_map_path_and_dots_have_a_pixel_size_floor():
    """PathLayer.get_width and ScatterplotLayer.get_radius default to METERS,
    not pixels -- at a whole-survey overview zoom a 2 m line / 3 m dot is
    sub-pixel and effectively invisible. Every layer needs an explicit
    *_min_pixels so the pipeline and its markers stay visible regardless of
    zoom (regression: this made the map look empty at the default zoom)."""
    deck = build_map(_toy_raw(), _toy_indications())
    path_layer = next(layer for layer in deck.layers if layer.type == "PathLayer")
    assert path_layer.width_min_pixels >= 2

    scatter_layers = [
        layer for layer in deck.layers if layer.type == "ScatterplotLayer"
    ]
    assert len(scatter_layers) == 3
    for layer in scatter_layers:
        assert layer.radius_min_pixels >= 4
