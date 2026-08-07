"""
Shared pydeck map builder for both Streamlit apps (Stage 3's thin
`streamlit_app.py` and Stage 4.5's `demo_app.py`) -- extracted so the two
never quietly diverge on how a survey's track/defects/interference/
indications get drawn.
"""

from __future__ import annotations

import pandas as pd
import pydeck as pdk


def build_map(
    raw: pd.DataFrame, indications: pd.DataFrame | None, show_heatmap: bool = False
) -> pdk.Deck:
    """GPS track (grey), true defects (orange), true interference (grey dots,
    unlabelled to the model), detected indications if any -- as discrete blue
    dots by default, or as a density heatmap (weighted by risk_score, falling
    back to anomaly_score) when `show_heatmap=True`. Ground truth (defects/
    interference) stays as dots either way -- only the model's own output
    representation toggles."""
    # Rig-v2: raw carries real GPS dropout (`lat`/`lon` are NaN wherever the
    # walker's GPS lock was lost, generate.py::_gps_track) -- a NaN vertex in
    # a pydeck PathLayer's coordinate list corrupts the whole line (WebGL
    # draws stray segments through it, not just a gap), which is what made
    # the map show "random lines" instead of the walked survey. Drop
    # unlocked rows before building every layer, not just the path.
    located = raw.dropna(subset=["lat", "lon"])

    # Thin the track for the map (cheap and plenty to see the line) -- the
    # model never sees fewer points than this, only the plotted path does.
    track = located[["lat", "lon"]].iloc[:: max(1, len(located) // 2000)]
    defects = located.loc[located["defect"] == 1, ["lat", "lon", "defect_type"]]
    interference = located.loc[located["interference"] == 1, ["lat", "lon"]]

    layers = [
        pdk.Layer(
            "PathLayer",
            data=[{"path": track[["lon", "lat"]].to_numpy().tolist()}],
            get_path="path",
            get_width=2,
            width_min_pixels=3,  # get_width is in METERS -- sub-pixel, invisible, at survey-overview zoom
            get_color=[120, 120, 120],
        ),
        pdk.Layer(
            "ScatterplotLayer", data=defects, get_position=["lon", "lat"],
            get_color=[220, 80, 40], get_radius=3, radius_min_pixels=5, pickable=True,
        ),
        pdk.Layer(
            "ScatterplotLayer", data=interference, get_position=["lon", "lat"],
            get_color=[130, 130, 130], get_radius=3, radius_min_pixels=5, pickable=True,
        ),
    ]
    if indications is not None and len(indications) and "lat" in indications and "lon" in indications:
        if show_heatmap:
            weight_col = (
                "risk_score"
                if "risk_score" in indications.columns and indications["risk_score"].notna().any()
                else "anomaly_score"
            )
            heat = indications[["lat", "lon", weight_col]].dropna()
            if len(heat):
                # HeatmapLayer weights must be non-negative -- anomaly_score
                # is a raw detector score, not guaranteed >= 0 unlike a
                # calibrated risk_score.
                heat = heat.assign(**{weight_col: heat[weight_col].clip(lower=0.0)})
                layers.append(
                    pdk.Layer(
                        "HeatmapLayer",
                        data=heat,
                        get_position=["lon", "lat"],
                        get_weight=weight_col,
                        radiusPixels=40,
                    )
                )
        else:
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer", data=indications, get_position=["lon", "lat"],
                    get_color=[40, 120, 220], get_radius=4, radius_min_pixels=6, pickable=True,
                )
            )

    view_state = pdk.ViewState(
        latitude=float(raw["lat"].mean()), longitude=float(raw["lon"].mean()), zoom=13
    )
    return pdk.Deck(layers=layers, initial_view_state=view_state, map_style=None)
