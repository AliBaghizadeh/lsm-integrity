"""
Stage 3: row scores -> indication objects. An indication is a contiguous run of
supra-threshold rows, collapsed to one peak + extent -- the unit an inspection
engineer actually reads (SKILL invariant #2: row-level scoring, then indications,
never the reverse).

`dq_flag` is carried through from the per-row feature flag (clean|edge, set by
features.py at survey boundaries) -- an indication built partly from a truncated
rolling window is not a clean prediction, and this is how that travels downstream
to whoever consumes the `indication` table.

`p_defect_cal` here is an UNCALIBRATED STAND-IN (percentile rank of the peak's raw
anomaly score against the survey's own row-level score distribution) -- real
isotonic calibration is explicitly Stage 5 scope (SKILL: "add its baseline first"
applies to calibration too; a fake-precise probability before there is a
calibration map would be worse than an honest rank).
"""

from __future__ import annotations

import datetime as dt
import hashlib

import numpy as np
import pandas as pd

INDICATION_COLUMNS = [
    "indication_id",
    "survey_id",
    "pipeline_version",
    "is_shadow",
    "chainage_peak_m",
    "chainage_start_m",
    "chainage_end_m",
    "lat",
    "lon",
    "anomaly_score",
    "p_defect_cal",
    "pred_type",
    "pred_type_conf",
    "sev_pred",
    "sev_lo",
    "sev_hi",
    "interval_nominal",
    "risk_score",
    "dq_flag",
    "created_at",
]


def _make_indication_id(
    survey_id: str, pipeline_version: str, chainage_peak_m: float, peak_sample_idx: int
) -> str:
    """Deterministic, so re-running predict on the same survey+pipeline_version
    is idempotent -- required by the indication table's contract (architecture.md:
    "idempotent on (survey_id, pipeline_version)").

    Includes `peak_sample_idx` (the physical, integer, tie-free key -- SKILL
    invariant #8) alongside `chainage_peak_m`, not instead of it. Rig-v2's
    registered `chainage_m` is a dead-reckoned + weld-locked estimate and
    genuinely repeats the same value across many samples (e.g. flat
    extrapolation before the first locked GPS fix, or several samples landing
    in the same interpolation step) -- on the production corpus, over 38,000
    of ~200,000 rows in a single survey share their `chainage_m` with at least
    one other row, some in runs of 90+. Two DIFFERENT indications (different
    extent, different anomaly_score) can therefore land on peak rows that tie
    in chainage to the millisecond it used to be hashed at, and a chainage-only
    key silently collapsed them onto the same `indication_id` -- not a
    non-determinism bug (the collision was 100% reproducible run to run) but a
    genuine collision bug: `write_indications`'s `INSERT OR IGNORE` then
    dropped the second, distinct indication as if it were a re-run duplicate
    of the first. `sample_idx` is dense and unique per row by construction, so
    keying on it as well makes two distinct clusters' peaks unable to collide
    even when their chainage does.
    """
    key = f"{survey_id}|{pipeline_version}|{peak_sample_idx}|{chainage_peak_m:.3f}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def cluster_indications(
    df: pd.DataFrame,
    score_col: str,
    threshold: float,
    survey_id: str,
    pipeline_version: str,
    is_shadow: bool = False,
) -> pd.DataFrame:
    """Cluster contiguous rows where `score_col > threshold` into indications.

    `df` must be sorted by sample_idx and carry chainage_m, lat, lon, dq_flag and
    `score_col` for every row of the survey (not just the supra-threshold ones --
    the full row set is what p_defect_cal's percentile rank is measured against).
    """
    d = df.sort_values("sample_idx").reset_index(drop=True)
    scores = d[score_col].to_numpy(dtype=np.float64)
    above = scores > threshold

    edges = np.flatnonzero(np.diff(np.concatenate([[0], above.astype(int), [0]])))
    starts, ends = (
        edges[::2],
        edges[1::2],
    )  # [start, end) row-index pairs, end exclusive

    now = dt.datetime.now(dt.UTC).isoformat()
    rows = []
    for a, b in zip(starts, ends):
        run = d.iloc[a:b]
        peak_pos = int(run[score_col].to_numpy().argmax())
        peak = run.iloc[peak_pos]
        peak_score = float(peak[score_col])
        rows.append(
            {
                "indication_id": _make_indication_id(
                    survey_id,
                    pipeline_version,
                    float(peak["chainage_m"]),
                    int(peak["sample_idx"]),
                ),
                "survey_id": survey_id,
                "pipeline_version": pipeline_version,
                "is_shadow": int(is_shadow),
                "chainage_peak_m": float(peak["chainage_m"]),
                "chainage_start_m": float(run["chainage_m"].min()),
                "chainage_end_m": float(run["chainage_m"].max()),
                "lat": float(peak["lat"])
                if "lat" in d.columns and pd.notna(peak.get("lat"))
                else None,
                "lon": float(peak["lon"])
                if "lon" in d.columns and pd.notna(peak.get("lon"))
                else None,
                "anomaly_score": peak_score,
                "p_defect_cal": float(np.mean(scores <= peak_score)),
                # Stage 4/5 fields -- filled in later by attach_severity() /
                # a future classify step. NULL here is the documented "not yet
                # scored" state, not a missing-data bug.
                "pred_type": None,
                "pred_type_conf": None,
                "sev_pred": None,
                "sev_lo": None,
                "sev_hi": None,
                "interval_nominal": None,
                "risk_score": None,
                "dq_flag": "edge" if (run["dq_flag"] == "edge").any() else "clean",
                "created_at": now,
            }
        )

    return pd.DataFrame(rows, columns=INDICATION_COLUMNS)


def select_dig_budget(
    indications: pd.DataFrame, survey_length_m: float, digs_per_km: int
) -> pd.DataFrame:
    """Rank indications by anomaly_score and keep the top budget-worth. This top
    slice -- never the full indication list -- is what recall@budget and
    false-dig-rate are measured against; an engineer only ever digs the budget.
    """
    if len(indications) == 0:
        return indications
    budget = max(1, round(digs_per_km * survey_length_m / 1000.0))
    return indications.sort_values("anomaly_score", ascending=False).head(budget)


BLOCK_HEATMAP_COLUMNS = ["line_id", "block_start_m", "value", "n_indications", "metric"]


def block_risk_heatmap(
    indications: pd.DataFrame,
    line_id_col: str = "line_id",
    chainage_col: str = "chainage_peak_m",
    block_m: float = 100.0,
) -> pd.DataFrame:
    """Aggregate indications into (line_id, 100 m block) cells for a
    cross-line "defective areas" heatmap -- one row per non-empty cell, so
    an inspection engineer can read off where risk clusters along the whole
    pipe, not just within one survey.

    Deliberately the SAME block grouping `evaluate.assign_group()` uses for
    CV fold assignment (`int(chainage_m // block_m)`): a heatmap cell and a
    fold's group_key describe the same physical 100 m stretch, so the two
    views of the data never silently disagree about where a "block" starts.

    Falls back to `anomaly_score` when `risk_score` is entirely missing (no
    classify/severity model released yet) -- same fallback `chart_utils.py`
    already uses for the dig-ranking table/chart. `metric` in the output
    records which column was actually used, per call, not assumed by the
    caller.
    """
    if indications is None or indications.empty:
        return pd.DataFrame(columns=BLOCK_HEATMAP_COLUMNS)

    has_risk = (
        "risk_score" in indications.columns and indications["risk_score"].notna().any()
    )
    metric = "risk_score" if has_risk else "anomaly_score"
    if metric not in indications.columns:
        return pd.DataFrame(columns=BLOCK_HEATMAP_COLUMNS)

    df = (
        indications[[line_id_col, chainage_col, metric]]
        .rename(
            columns={
                line_id_col: "line_id",
                chainage_col: "chainage_peak_m",
                metric: "value",
            }
        )
        .dropna(subset=["line_id", "chainage_peak_m", "value"])
    )
    if df.empty:
        return pd.DataFrame(columns=BLOCK_HEATMAP_COLUMNS)

    df["block_start_m"] = (df["chainage_peak_m"] // block_m) * block_m

    out = (
        df.groupby(["line_id", "block_start_m"])["value"]
        .agg(value="max", n_indications="count")
        .reset_index()
    )
    out["metric"] = metric
    return out[BLOCK_HEATMAP_COLUMNS]


def attach_indication_features(
    indications: pd.DataFrame, rows: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    """Stage 4: join each indication to ONE feature vector, for severity
    regression. SKILL invariant #2 is explicit that severity must never be
    regressed row-wise (`severity_smys` is constant across a whole label box,
    so a row-level regressor just learns to reproduce that constant) -- this is
    the join that makes indication-level regression possible.

    Uses the PEAK row's own feature vector (the least-noisy amplitude/shape
    estimate, since peak-finding already located it) plus one indication-only
    feature, `extent_m` -- the physical footprint length, which no single row
    can express. `rows` must carry survey_id, chainage_m and every column in
    `feature_cols`.
    """
    peak_keys = indications[["indication_id", "survey_id", "chainage_peak_m"]]
    peak_rows = rows.merge(
        peak_keys,
        left_on=["survey_id", "chainage_m"],
        right_on=["survey_id", "chainage_peak_m"],
        how="inner",
    )
    # Rig-v2: registration.register_survey's own belt-and-braces
    # np.maximum.accumulate legitimately produces exact ties in `chainage_m`
    # among immediately-adjacent rows (registration.py's module docstring) --
    # this float-equality join (a pre-existing pattern, not introduced here)
    # can therefore match MORE THAN ONE row to a single indication_id where
    # ties land exactly on a peak's chainage, silently exploding attach_
    # severity's row count downstream (measured: 131 indications -> 616
    # merged rows on a real corpus). Keep exactly one match per indication --
    # the tied rows are immediately-adjacent samples with near-identical
    # feature values, so which one survives doesn't matter; a future cleanup
    # could join on sample_idx instead (SKILL invariant #8: never key/join on
    # a float) by threading the peak row's sample_idx through
    # cluster_indications, which would need a schema/DDL change this fix
    # deliberately avoids.
    peak_rows = peak_rows.drop_duplicates(subset="indication_id", keep="first")
    out = indications.merge(
        peak_rows[["indication_id", *feature_cols]], on="indication_id", how="left"
    )
    out["extent_m"] = out["chainage_end_m"] - out["chainage_start_m"]
    return out


def attach_severity(
    indications: pd.DataFrame,
    rows: pd.DataFrame,
    severity_model,
    base_feature_cols: list[str],
    nominal_coverage: float,
) -> pd.DataFrame:
    """Stage 4: fill `sev_pred`/`sev_lo`/`sev_hi`/`interval_nominal` on every
    indication via its peak row's feature vector, scored through the bundle's
    severity model. Severity is estimated for every indication regardless of
    whether a later classify stage would confirm it's a real defect -- an
    engineer sizing a dig wants this even for something that turns out to be
    interference; classify (Stage 5) is what would tell them so.

    `base_feature_cols` excludes `extent_m` -- `attach_indication_features`
    computes it, and `severity_model.feature_cols` (which includes it) is what
    actually gets indexed at predict time, so the two lists don't need to
    match; only every name in `severity_model.feature_cols` needs to end up a
    column somewhere.
    """
    if len(indications) == 0:
        return indications
    with_features = attach_indication_features(indications, rows, base_feature_cols)
    med, lo, hi = severity_model.predict(with_features)
    out = indications.copy()
    out["sev_pred"] = med
    out["sev_lo"] = lo
    out["sev_hi"] = hi
    out["interval_nominal"] = nominal_coverage
    return out


def attach_classification(
    indications: pd.DataFrame,
    rows: pd.DataFrame,
    classify_model,
    defect_calibrator,
    base_feature_cols: list[str],
    consequence_proxy: dict[str, float],
) -> pd.DataFrame:
    """Stage 5: fill `pred_type`/`pred_type_conf` on every indication via its
    peak row's feature vector, scored through the classifier. OVERWRITES
    `p_defect_cal` via `defect_calibrator` -- replacing `cluster_indications`'s
    percentile-rank stand-in with a real isotonic-calibrated P(this is a real
    defect), fit on the out-of-fold clustered indication population (matched-
    defect, matched-interference, AND unmatched false alarms alike -- see
    `train.py`'s `defect_calibrator`). Computes
    `risk_score = p_defect_cal * sev_pred * consequence_proxy[pred_type]`.

    `risk_score` stays `None` wherever `sev_pred` is `None` (no severity model
    released yet, or this indication wasn't scored by one) -- a risk score
    computed from a missing severity term would be a fabricated number, not an
    honestly absent prediction (SKILL invariant #5: "an indication with no
    interval... is not a prediction, it is a rumour").
    """
    if len(indications) == 0:
        return indications
    with_features = attach_indication_features(indications, rows, base_feature_cols)
    pred_type, pred_conf = classify_model.predict(with_features)

    out = indications.copy()
    out["pred_type"] = pred_type
    out["pred_type_conf"] = pred_conf
    out["p_defect_cal"] = defect_calibrator.predict(out["anomaly_score"].to_numpy())

    # sev_pred is Python None until attach_severity() has run (or forever, if
    # no severity model has been released) -- coerce to real NaN so the
    # multiplication below propagates NaN rather than raising on None.
    sev_pred_numeric = pd.to_numeric(out["sev_pred"], errors="coerce")
    consequence = out["pred_type"].map(consequence_proxy)
    risk = out["p_defect_cal"] * sev_pred_numeric * consequence
    out["risk_score"] = risk.where(sev_pred_numeric.notna())
    return out


def write_indications(conn, indications: pd.DataFrame) -> int:
    """Persist to the `indication` table. Idempotent on `indication_id` (itself
    deterministic on survey_id + pipeline_version + peak chainage): re-running
    predict on the same survey and pipeline_version is a no-op, matching the
    table's documented contract ("idempotent on (survey_id, pipeline_version)").

    The table holds the FULL candidate list, not just a dig-budget-selected
    slice -- budget is a serving-time ranking choice
    (validation-and-trust.md: "the model does not decide to dig; it decides
    what to look at first"), not something baked into what gets stored.
    """
    if len(indications) == 0:
        return 0
    rows = indications[INDICATION_COLUMNS].itertuples(index=False, name=None)
    conn.executemany(
        f"""
        INSERT OR IGNORE INTO indication ({", ".join(INDICATION_COLUMNS)})
        VALUES ({", ".join("?" * len(INDICATION_COLUMNS))})
        """,
        rows,
    )
    conn.commit()
    return len(indications)
