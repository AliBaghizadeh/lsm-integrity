"""Stage 3: truth.py -- the derived truth-defect/interference registry."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lsm.truth import build_truth_registry, nearest_source


def _raw_frame(n=400, step_m=1.0):
    """Rig-v2: raw no longer carries chainage_m at all (schemas.py) --
    build_truth_registry now takes the (registered) chainage as a separate
    array, aligned to this frame's row order. On this fixture that array is
    just sample_idx * step_m, the same numbers the old inline column had.
    """
    sample_idx = np.arange(n)
    df = pd.DataFrame(
        {
            "sample_idx": sample_idx,
            "defect": 0,
            "defect_type": "none",
            "interference": 0,
        }
    )
    return df, sample_idx.astype(float) * step_m


def test_build_truth_registry_finds_one_defect_and_one_interference_region():
    df, chainage = _raw_frame()
    df.loc[100:105, "defect"] = 1
    df.loc[100:105, "defect_type"] = "weld"
    df.loc[300:310, "interference"] = 1

    registry = build_truth_registry(df, line_id="LINE000", chainage_m=chainage)

    defects = registry[registry["kind"] == "defect"]
    interference = registry[registry["kind"] == "interference"]
    assert len(defects) == 1
    assert len(interference) == 1
    assert defects.iloc[0]["defect_type"] == "weld"
    assert defects.iloc[0]["chainage_start_m"] == 100.0
    assert defects.iloc[0]["chainage_end_m"] == 105.0
    assert interference.iloc[0]["defect_type"] == "interference"


def test_build_truth_registry_separates_multiple_defects_by_type():
    df, chainage = _raw_frame()
    df.loc[50:55, "defect"] = 1
    df.loc[50:55, "defect_type"] = "scc"
    df.loc[200:206, "defect"] = 1
    df.loc[200:206, "defect_type"] = "corrosion"

    registry = build_truth_registry(df, line_id="LINE000", chainage_m=chainage)
    defects = registry[registry["kind"] == "defect"].sort_values("chainage_m")

    assert len(defects) == 2
    assert list(defects["defect_type"]) == ["scc", "corrosion"]
    assert list(defects["source_id"]) == ["LINE000_defect_00", "LINE000_defect_01"]


def test_nearest_source_finds_closest_within_kind():
    registry = pd.DataFrame(
        {
            "source_id": ["d0", "d1", "i0"],
            "line_id": ["L", "L", "L"],
            "kind": ["defect", "defect", "interference"],
            "defect_type": ["weld", "scc", "interference"],
            "chainage_m": [100.0, 500.0, 100.5],
            "chainage_start_m": [98.0, 498.0, 99.0],
            "chainage_end_m": [102.0, 502.0, 102.0],
        }
    )

    source_id, dist = nearest_source(101.0, registry, kind="defect")
    assert source_id == "d0"
    assert dist == 1.0

    # Without the kind filter, the interference source at 100.5 is actually closer.
    source_id_any, dist_any = nearest_source(101.0, registry)
    assert source_id_any == "i0"
    assert dist_any == 0.5


def test_nearest_source_returns_none_on_empty_registry():
    empty = pd.DataFrame(columns=["source_id", "line_id", "kind", "defect_type",
                                   "chainage_m", "chainage_start_m", "chainage_end_m"])
    source_id, dist = nearest_source(50.0, empty)
    assert source_id is None
    assert dist == float("inf")
