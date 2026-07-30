"""
Stage 3: a truth-defect/interference registry, derived directly from raw survey
label columns (`defect`, `defect_type`, `interference`) rather than the SQLite
`truth_defect`/`truth_observation` tables -- `ingest` does not populate those yet
(a known gap, see PLAN.md / memory), and building the full slowly-changing-
dimension registry those tables imply is out of Stage 3's scope.

Defect/interference POSITION is fixed across a line's 3 runs -- only
severity/moment grows per run, see generate.py's `_make_run` -- so one reference
run's labels are enough to build the whole line's registry. Contiguous same-label
chainage runs become one physical source, with a stable `source_id` used for:
  (a) matching a predicted indication to the true source it corresponds to,
  (b) the bootstrap resampling UNIT for recall/false-dig-rate CIs
      (validation-and-trust.md: "resampled over groups (lines / defects)" --
      with only 12 physical defects on this line, resampling rows or indications
      instead would wildly overstate precision).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REGISTRY_COLUMNS = [
    "source_id", "line_id", "kind", "defect_type",
    "chainage_m", "chainage_start_m", "chainage_end_m",
]


def _contiguous_runs(flag: np.ndarray) -> list[tuple[int, int]]:
    """[(start, end)) index pairs, end exclusive, for each contiguous run of flag == 1."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], flag.astype(int), [0]])))
    return list(zip(edges[::2], edges[1::2]))


def build_truth_registry(raw_df: pd.DataFrame, line_id: str) -> pd.DataFrame:
    """One reference survey's raw frame -> a registry of physical truth sources.

    Columns: source_id, line_id, kind ('defect'|'interference'), defect_type,
    chainage_m (region centre -- what indications are matched against),
    chainage_start_m, chainage_end_m (the labelled extent).
    """
    d = raw_df.sort_values("sample_idx").reset_index(drop=True)
    rows = []
    for kind, flag_col in (("defect", "defect"), ("interference", "interference")):
        flag = d[flag_col].to_numpy()
        for i, (a, b) in enumerate(_contiguous_runs(flag)):
            run = d.iloc[a:b]
            defect_type = str(run["defect_type"].iloc[0]) if kind == "defect" else "interference"
            rows.append(
                {
                    "source_id": f"{line_id}_{kind}_{i:02d}",
                    "line_id": line_id,
                    "kind": kind,
                    "defect_type": defect_type,
                    "chainage_m": float(run["chainage_m"].mean()),
                    "chainage_start_m": float(run["chainage_m"].min()),
                    "chainage_end_m": float(run["chainage_m"].max()),
                }
            )
    return pd.DataFrame(rows, columns=REGISTRY_COLUMNS)


def nearest_source(
    chainage_m: float, registry: pd.DataFrame, kind: str | None = None
) -> tuple[str | None, float]:
    """Nearest truth source to a chainage position -> (source_id, distance_m).
    (None, inf) if the registry (filtered to `kind` if given) is empty -- callers
    compare `distance_m` against a tolerance, never assume a match exists.
    """
    reg = registry if kind is None else registry[registry["kind"] == kind]
    if len(reg) == 0:
        return None, float("inf")
    dist = (reg["chainage_m"] - chainage_m).abs()
    idx = dist.idxmin()
    return str(reg.loc[idx, "source_id"]), float(dist.loc[idx])
