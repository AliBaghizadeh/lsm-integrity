"""
Shared pydeck map builder for both Streamlit apps (Stage 3's thin
`streamlit_app.py` and Stage 4.5's `demo_app.py`) -- extracted so the two
never quietly diverge on how a survey's track/defects/interference/
indications get drawn.
"""

from __future__ import annotations

import pandas as pd
import pydeck as pdk


def build_map(raw: pd.DataFrame, indications: pd.DataFrame | None) -> pdk.Deck:
    """GPS track (grey), true defects (orange), true interference (grey dots,
    unlabelled to the model), detected indications (blue) if any."""
    # Thin the track for the map (cheap and plenty to see the line) -- the
    # model never sees fewer points than this, only the plotted path does.
    track = raw[["lat", "lon"]].iloc[:: max(1, len(raw) // 2000)]
    defects = raw.loc[raw["defect"] == 1, ["lat", "lon", "defect_type"]]
    interference = raw.loc[raw["interference"] == 1, ["lat", "lon"]]

    layers = [
        pdk.Layer(
            "PathLayer",
            data=[{"path": track[["lon", "lat"]].to_numpy().tolist()}],
            get_path="path",
            get_width=2,
            get_color=[120, 120, 120],
        ),
        pdk.Layer(
            "ScatterplotLayer", data=defects, get_position=["lon", "lat"],
            get_color=[220, 80, 40], get_radius=3, pickable=True,
        ),
        pdk.Layer(
            "ScatterplotLayer", data=interference, get_position=["lon", "lat"],
            get_color=[130, 130, 130], get_radius=3, pickable=True,
        ),
    ]
    if indications is not None and len(indications):
        layers.append(
            pdk.Layer(
                "ScatterplotLayer", data=indications, get_position=["lon", "lat"],
                get_color=[40, 120, 220], get_radius=4, pickable=True,
            )
        )

    view_state = pdk.ViewState(
        latitude=float(raw["lat"].mean()), longitude=float(raw["lon"].mean()), zoom=13
    )
    return pdk.Deck(layers=layers, initial_view_state=view_state, map_style=None)
