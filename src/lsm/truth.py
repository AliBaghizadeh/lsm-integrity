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
    "source_id",
    "line_id",
    "kind",
    "defect_type",
    "chainage_m",
    "chainage_start_m",
    "chainage_end_m",
]


def _contiguous_runs(flag: np.ndarray) -> list[tuple[int, int]]:
    """[(start, end)) index pairs, end exclusive, for each contiguous run of flag == 1."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], flag.astype(int), [0]])))
    return list(zip(edges[::2], edges[1::2]))


def build_truth_registry(
    raw_df: pd.DataFrame, line_id: str, chainage_m: np.ndarray
) -> pd.DataFrame:
    """One reference survey's raw frame -> a registry of physical truth sources.

    Columns: source_id, line_id, kind ('defect'|'interference'), defect_type,
    chainage_m (region centre -- what indications are matched against),
    chainage_start_m, chainage_end_m (the labelled extent).

    `chainage_m` is a plain array, aligned to `raw_df`'s INPUT row order (same
    convention as `registration.RegistrationResult` -- see that module's
    docstring), not a column read off `raw_df` itself: Rig-v2 raw no longer
    carries a physically-final `chainage_m` at all (schemas.py), only
    `chainage_true_m` (truth tier, never a feature) and the registered
    `chainage_m` Stage B's `register_survey` produces. This function accepts
    it as a separate parameter rather than importing `registration.py` and
    calling `register_survey` itself, for two reasons: it avoids a second,
    wasteful (if harmless -- registration is deterministic) recomputation
    when a caller already has the registered chainage from the feature
    pipeline, and it keeps this module's own logic -- contiguous-run
    detection on `defect`/`interference` flags -- genuinely schema-agnostic,
    tested here with a bare synthetic chainage array and not coupled to
    registration's own DataConfig-shaped input.
    """
    d = raw_df.reset_index(drop=True)
    chainage = np.asarray(chainage_m, dtype=np.float64)
    if len(chainage) != len(d):
        raise ValueError(
            f"chainage_m has {len(chainage)} rows but raw_df has {len(d)} -- "
            "they must be aligned to the same (input) row order."
        )
    order = np.argsort(d["sample_idx"].to_numpy(), kind="stable")
    d = d.iloc[order].reset_index(drop=True)
    chainage = chainage[order]

    rows = []
    for kind, flag_col in (("defect", "defect"), ("interference", "interference")):
        flag = d[flag_col].to_numpy()
        for i, (a, b) in enumerate(_contiguous_runs(flag)):
            run_chainage = chainage[a:b]
            defect_type = (
                str(d["defect_type"].iloc[a]) if kind == "defect" else "interference"
            )
            rows.append(
                {
                    "source_id": f"{line_id}_{kind}_{i:02d}",
                    "line_id": line_id,
                    "kind": kind,
                    "defect_type": defect_type,
                    "chainage_m": float(run_chainage.mean()),
                    "chainage_start_m": float(run_chainage.min()),
                    "chainage_end_m": float(run_chainage.max()),
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
