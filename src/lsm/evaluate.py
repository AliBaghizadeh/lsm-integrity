"""
Stage 3/4: grouped CV split, indication-level metrics, bootstrap CIs.

Splitting: GroupKFold on (line_id, 100 m block) -- **never** `block` alone, so
two overlapping surveys of the same stretch cannot land in different folds
(SKILL invariant #1). Fold assignment is by sha256(group_key) % n_folds, not
positional -- PLAN.md's reference design says blake3; this project already
decided "stdlib sha256 throughout, no blake3/xxhash needed" (hashing.py), so
that convention wins here too. Either hash gives the same property that matters:
growing the archive does not reshuffle existing folds.

Row-level metrics are diagnostics; indication-level metrics are the result
(validation-and-trust.md). Every headline metric is bootstrapped over PHYSICAL
GROUPS, never rows -- see each function's docstring for which group it uses and
why, since the right resampling unit differs per metric.
"""

from __future__ import annotations

import hashlib
from typing import Callable

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics import average_precision_score


def assign_group(line_id: str, chainage_m: float, block_m: float = 100.0) -> str:
    """(line_id, block) -- never block alone, so the same physical stretch
    surveyed by two different runs always lands in the same fold."""
    return f"{line_id}::{int(chainage_m // block_m)}"


def assign_fold(group_key: str, n_folds: int) -> int:
    """Hash-based, not positional: appending new surveys/lines later does not
    reshuffle folds an existing model was evaluated against."""
    digest = hashlib.sha256(group_key.encode("utf-8")).hexdigest()
    return int(digest, 16) % n_folds


def add_fold_column(
    df: pd.DataFrame, line_id_col: str, chainage_col: str, block_m: float, n_folds: int
) -> pd.DataFrame:
    """Returns a copy of `df` with two new columns: `group_key` and `fold`."""
    out = df.copy()
    out["group_key"] = [
        assign_group(line_id, chainage_m, block_m)
        for line_id, chainage_m in zip(out[line_id_col], out[chainage_col])
    ]
    out["fold"] = [assign_fold(g, n_folds) for g in out["group_key"]]
    return out


def assign_fold_by_line(line_id: str, n_folds: int) -> int:
    """Stage 6: hash on `line_id` ALONE, no chainage block -- an entire
    physical line lands in one fold, the whole-line holdout PLAN.md Stage 6
    asks for ("the real generalisation test": today's default `assign_fold`
    scatters one line's 100 m blocks across ~all folds, so a model can see
    part of a line in training and be scored on another part of the SAME
    line -- this is a stronger, disjoint-line test). Same sha256(...) %
    n_folds scheme as `assign_fold`, for the same "growing the corpus later
    does not reshuffle existing folds" property. Deliberately NOT wired into
    `add_fold_column`/the default Stage 3-5 path -- see `scale_eval.py`.
    """
    digest = hashlib.sha256(line_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % n_folds


def add_fold_column_by_line(df: pd.DataFrame, line_id_col: str, n_folds: int) -> pd.DataFrame:
    """Returns a copy of `df` with `group_key` (== line_id) and `fold` columns
    -- same two-column contract as `add_fold_column`, so downstream consumers
    (`train._run_grouped_cv`, the severity/classify fold merges) don't need to
    know which grouping scheme produced `fold`.
    """
    out = df.copy()
    out["group_key"] = out[line_id_col]
    out["fold"] = [assign_fold_by_line(g, n_folds) for g in out["group_key"]]
    return out


def match_dug_indications(
    dug: pd.DataFrame, registry: pd.DataFrame, tolerance_m: float
) -> pd.DataFrame:
    """For each dug indication, find the nearest truth source (defect or
    interference) within `tolerance_m` of its peak chainage. Adds
    `matched_source_id`, `matched_kind` ('defect'|'interference'|None),
    `distance_m` (inf if unmatched). No truth source may be matched twice --
    ties broken by whichever candidate indication is closer, so a second dig
    near an already-claimed defect is correctly scored as a false dig, not a
    second true positive.
    """
    out = dug.copy()
    out["matched_source_id"] = None
    out["matched_kind"] = None
    out["distance_m"] = np.inf
    if len(out) == 0 or len(registry) == 0:
        return out

    claimed: set[str] = set()
    # Closest indication-to-source pairs first, so the best match wins a
    # contested source.
    candidates = []
    for i, peak in zip(out.index, out["chainage_peak_m"]):
        dist = (registry["chainage_m"] - peak).abs()
        for j in dist.index:
            d = float(dist.loc[j])
            if d <= tolerance_m:
                candidates.append((d, i, j))
    for d, i, j in sorted(candidates, key=lambda t: t[0]):
        source_id = registry.loc[j, "source_id"]
        if source_id in claimed or out.loc[i, "matched_source_id"] is not None:
            continue
        claimed.add(source_id)
        out.loc[i, "matched_source_id"] = source_id
        out.loc[i, "matched_kind"] = registry.loc[j, "kind"]
        out.loc[i, "distance_m"] = d
    return out


def defect_hit_rates(
    matched_by_run: dict[str, pd.DataFrame],
    registry: pd.DataFrame,
    run_line_id: dict[str, str] | None = None,
) -> dict[str, float]:
    """Per-defect DETECTION RATE across runs: fraction of runs in which that
    physical defect was matched by at least one dug indication.

    The bootstrap unit for recall is the ~12 physical defects, not the ~36
    (defect, run) observations -- the 3 runs re-observe the SAME defects, so
    treating them as 36 independent samples would understate the true
    uncertainty (see PLAN.md Stage 2.5 / known limitation: "12 independent
    defects, not 36").

    `matched_by_run` keys are survey_ids (unique across lines). `run_line_id`
    maps each survey_id to its line_id, so with multiple lines a defect's rate
    is only divided by ITS OWN line's run count, not the whole corpus's --
    inactive with today's single line, but wrong (denominator too large) the
    moment a second line is added without it.
    """
    defect_ids = registry.loc[registry["kind"] == "defect", "source_id"].tolist()
    defect_line = dict(zip(registry["source_id"], registry["line_id"]))
    hits = {d: 0 for d in defect_ids}
    n_runs = {d: 0 for d in defect_ids}
    for survey_id, matched in matched_by_run.items():
        found = set(matched.loc[matched["matched_kind"] == "defect", "matched_source_id"])
        for d in defect_ids:
            if run_line_id is not None and run_line_id.get(survey_id) != defect_line[d]:
                continue
            n_runs[d] += 1
            if d in found:
                hits[d] += 1
    return {d: hits[d] / max(n_runs[d], 1) for d in defect_ids}


def false_dig_rate_per_run(matched: pd.DataFrame) -> float:
    """Fraction of a run's dug indications that did NOT land on a real defect
    (landed on interference, or on nothing within tolerance)."""
    if len(matched) == 0:
        return 0.0
    return float((matched["matched_kind"] != "defect").mean())


def interference_dig_fraction_per_run(matched: pd.DataFrame) -> float:
    """Fraction of a run's dug indications specifically wasted on an
    interference source -- the sub-metric the Stage 3 gate needs: "the gap [to
    the baseline] is attributable to interference rejection, shown explicitly".
    """
    if len(matched) == 0:
        return 0.0
    return float((matched["matched_kind"] == "interference").mean())


def localisation_errors_m(matched: pd.DataFrame) -> np.ndarray:
    """Absolute chainage error, in metres, for dug indications that matched a
    real defect -- a false dig has no localisation error to speak of."""
    hits = matched[matched["matched_kind"] == "defect"]
    return hits["distance_m"].to_numpy(dtype=float)


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Row-level PR-AUC -- a diagnostic only (per validation-and-trust.md,
    ROC-AUC is banned outright: positives are ~2.7% of rows and ROC flatters
    that). The Stage 3 gate is on indication-level recall@budget, not this.
    """
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def bootstrap_ci(
    units: list[float] | np.ndarray,
    n_resamples: int,
    level: float,
    seed: int,
    compute_fn: Callable[[np.ndarray], float] = np.mean,
) -> tuple[float, float, float]:
    """Generic case-resampling bootstrap over whatever `units` already IS the
    right group for (a per-defect hit rate, a per-run false-dig-rate, a
    per-match localisation error -- see each metric's own docstring for which).
    Returns (point_estimate, ci_lo, ci_hi).
    """
    arr = np.asarray(units, dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(compute_fn(arr))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boots[i] = compute_fn(sample)
    lo_pct = (1 - level) / 2 * 100
    hi_pct = (1 + level) / 2 * 100
    return point, float(np.percentile(boots, lo_pct)), float(np.percentile(boots, hi_pct))


def paired_bootstrap_ci(
    units_a: list[float] | np.ndarray,
    units_b: list[float] | np.ndarray,
    n_resamples: int,
    level: float,
    seed: int,
) -> tuple[float, float, float]:
    """Bootstrap CI on the DIFFERENCE mean(a) - mean(b), resampling the SAME
    indices for both arrays each iteration.

    This is what the Stage 3 gate ("IsolationForest beats MAD by >= 0.15
    recall @ budget, compared on intervals") actually needs: two separately
    bootstrapped CIs can both be wide and overlapping even when the paired
    difference is consistently positive on every resample, because a's and b's
    per-defect hit rates are correlated (evaluated on the SAME defects) -- a
    paired resample preserves that correlation and gives the tighter, correct
    answer. `units_a`/`units_b` must be the same length and index-aligned (e.g.
    both keyed to the same defect order).
    """
    a = np.asarray(units_a, dtype=float)
    b = np.asarray(units_b, dtype=float)
    if len(a) != len(b):
        raise ValueError("paired_bootstrap_ci requires index-aligned, equal-length arrays")
    if len(a) == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(np.mean(a) - np.mean(b))
    rng = np.random.default_rng(seed)
    n = len(a)
    boots = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boots[i] = np.mean(a[idx]) - np.mean(b[idx])
    lo_pct = (1 - level) / 2 * 100
    hi_pct = (1 + level) / 2 * 100
    return point, float(np.percentile(boots, lo_pct)), float(np.percentile(boots, hi_pct))


# ---------------------------------------------------------------------------
# Stage 4: severity metrics
# ---------------------------------------------------------------------------


def mae_by_severity_decile(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> pd.Series:
    """MAE within each decile of TRUE severity. A single overall MAE hides a
    model that is only accurate on benign defects -- validation-and-trust.md is
    explicit that this is "useless", not just an incomplete report. With today's
    handful of matched defects, `n_bins` collapses automatically (via
    `duplicates='drop'`) rather than producing empty or duplicate-edge bins.
    """
    df = pd.DataFrame({"y_true": y_true, "abs_err": np.abs(np.asarray(y_true) - np.asarray(y_pred))})
    if df["y_true"].nunique() < 2:
        return pd.Series({"all": float(df["abs_err"].mean())})
    bins = min(n_bins, df["y_true"].nunique())
    df["decile"] = pd.qcut(df["y_true"], q=bins, duplicates="drop")
    return df.groupby("decile", observed=True)["abs_err"].mean()


def per_group_severity_metrics(df: pd.DataFrame, group_col: str) -> dict[str, list[float]]:
    """`df` needs `group_col`, `y_true`, `y_pred`, `lo`, `hi` -- one row per
    matched indication. Returns per-GROUP scalar coverage/MAE/interval-width
    lists, the bootstrap unit for severity metrics (same physical-defect
    grouping as detection recall in `defect_hit_rates` -- a defect observed in
    all 3 runs must not count as 3 independent samples).
    """
    coverage, mae, width = [], [], []
    for _, g in df.groupby(group_col):
        coverage.append(float(np.mean((g["y_true"] >= g["lo"]) & (g["y_true"] <= g["hi"]))))
        mae.append(float(np.mean(np.abs(g["y_true"] - g["y_pred"]))))
        width.append(float(np.mean(g["hi"] - g["lo"])))
    return {"coverage": coverage, "mae": mae, "interval_width": width}


# ---------------------------------------------------------------------------
# Stage 5: classification metrics
# ---------------------------------------------------------------------------


def per_class_recall_units(
    df: pd.DataFrame, group_col: str, true_col: str, pred_col: str, classes: list[str]
) -> dict[str, list[float]]:
    """Per TRUE class, per-physical-source correct-classification indicator --
    same grouping discipline as `defect_hit_rates`/`per_group_severity_metrics`
    (a source re-observed across a line's 3 runs is one sample, not three).
    Feed each class's list straight into `bootstrap_ci` -- one CI per class,
    no new bootstrap loop. A class absent from `df[true_col]` gets an empty
    list (`bootstrap_ci` already returns NaNs for that, not an error).
    """
    out: dict[str, list[float]] = {c: [] for c in classes}
    for _, g in df.groupby(group_col):
        true_class = g[true_col].iloc[0]
        if true_class not in out:
            continue
        out[true_class].append(float((g[pred_col] == true_class).mean()))
    return out


def interference_precision_units(df: pd.DataFrame, group_col: str, true_col: str, pred_col: str) -> list[float]:
    """Of indications PREDICTED 'interference', per-source fraction whose TRUE
    class really was interference -- the false-positive-trap converse of
    interference recall (already covered by `per_class_recall_units`)."""
    predicted_interference = df[df[pred_col] == "interference"]
    if len(predicted_interference) == 0:
        return []
    return [
        float((g[true_col] == "interference").mean())
        for _, g in predicted_interference.groupby(group_col)
    ]


def multiclass_brier_score(y_true: np.ndarray, proba: pd.DataFrame, classes: list[str]) -> float:
    """Mean squared error between calibrated probability and one-hot truth,
    summed over classes, averaged over rows -- the standard multiclass Brier
    score generalisation. 0 is a perfect calibrated prediction; a uniform
    1/n_classes guess for every row scores (n_classes - 1) / n_classes.
    """
    one_hot = pd.DataFrame(
        {c: (np.asarray(y_true) == c).astype(float) for c in classes}, index=proba.index
    )
    return float(((proba[classes] - one_hot[classes]) ** 2).sum(axis=1).mean())


def brier_by_group(
    df: pd.DataFrame, group_col: str, true_col: str, proba_cols: list[str], classes: list[str]
) -> list[float]:
    """Per-physical-source mean Brier score -- feed into `bootstrap_ci`, same
    grouping discipline as every other Stage 5 metric."""
    return [
        multiclass_brier_score(g[true_col].to_numpy(), g[proba_cols], classes)
        for _, g in df.groupby(group_col)
    ]


def reliability_curve(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Bin by predicted max-probability confidence; empirical accuracy per bin
    -- a well-calibrated classifier's points fall near the diagonal
    (confidence == accuracy). What train.py plots as the reliability diagram.
    Same `duplicates='drop'` bin-collapsing as `mae_by_severity_decile` for a
    small, low-diversity confidence sample.
    """
    df = pd.DataFrame({"confidence": confidences, "correct": correct})
    n_unique = df["confidence"].nunique()
    if n_unique < 2:
        return pd.DataFrame(
            {"bin_mean_confidence": [df["confidence"].mean()], "bin_accuracy": [df["correct"].mean()],
             "n": [len(df)]}
        )
    bins = min(n_bins, n_unique)
    df["bin"] = pd.qcut(df["confidence"], q=bins, duplicates="drop")
    grouped = df.groupby("bin", observed=True).agg(
        bin_mean_confidence=("confidence", "mean"), bin_accuracy=("correct", "mean"), n=("correct", "size")
    )
    return grouped.reset_index(drop=True)


def shap_denylist_check(shap_importance: pd.Series, denylist: list[str], top_k: int = 10) -> dict:
    """Pure logic, no SHAP/contribution computation itself -- takes an
    already-computed mean-|contribution| `Series` indexed by feature name.
    The physics-consistency gate: the model should key on residual amplitude/
    gradient/peak width, not absolute position (chainage, sample_idx) -- a
    shortcut that would happen to work on this synthetic corpus's fixed
    defect placement but never generalise to a real, different pipeline.
    """
    ranked = shap_importance.sort_values(ascending=False)
    top_features = ranked.index[:top_k].tolist()
    leaked = [f for f in top_features if f in denylist]
    return {"top_features": top_features, "leaked_denylist_features": leaked, "passed": len(leaked) == 0}


# -- Stage 8: drift monitoring (validation-and-trust.md Layer 4) -------------


def psi(reference: np.ndarray, current: np.ndarray, bin_edges: np.ndarray) -> float:
    """Population Stability Index: bins `current` into the REFERENCE's OWN
    bin edges (never re-derived from `current` -- that would compare a
    distribution to a rebinning of itself). Values outside the reference's
    original range are clipped into the extreme bin rather than silently
    dropped by `np.histogram` -- a distribution that has moved entirely past
    the old boundary is exactly the case this check exists to catch, not an
    edge case to discard. 0.0 if either sample is empty (nothing to compare).
    PSI > 0.2 warns, > 0.3 blocks (validation-and-trust.md Layer 4).
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    bin_edges = np.asarray(bin_edges, dtype=float)
    if len(reference) == 0 or len(current) == 0 or len(bin_edges) < 2:
        return 0.0
    ref_clipped = np.clip(reference, bin_edges[0], bin_edges[-1])
    cur_clipped = np.clip(current, bin_edges[0], bin_edges[-1])
    ref_counts, _ = np.histogram(ref_clipped, bins=bin_edges)
    cur_counts, _ = np.histogram(cur_clipped, bins=bin_edges)
    eps = 1e-6
    ref_pct = np.clip(ref_counts / max(1, ref_counts.sum()), eps, None)
    cur_pct = np.clip(cur_counts / max(1, cur_counts.sum()), eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_drift(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test -- (statistic, p_value). (nan, nan)
    if either sample is empty, an honest "no data" answer, not an error.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if len(reference) == 0 or len(current) == 0:
        return float("nan"), float("nan")
    result = scipy.stats.ks_2samp(reference, current)
    return float(result.statistic), float(result.pvalue)


def feature_drift_report(
    reference: dict[str, dict], current: pd.DataFrame, feature_cols: list[str],
    psi_warn: float, psi_block: float,
) -> pd.DataFrame:
    """One row per feature: psi, ks_stat, ks_pvalue, status ('ok'|'warn'|
    'block'). `reference` is a bundle's `training_feature_summary` dict
    (per-feature `bin_edges`/`sample`, plus mean/std/min/max unused here).
    A feature missing from `reference` (e.g. a stale bundle) is skipped, not
    an error -- the caller sees fewer rows, not a crash.
    """
    rows = []
    for col in feature_cols:
        if col not in reference:
            continue
        ref = reference[col]
        ref_sample = np.asarray(ref.get("sample", []), dtype=float)
        bin_edges = np.asarray(ref.get("bin_edges", []), dtype=float)
        cur_values = current[col].dropna().to_numpy(dtype=float) if col in current.columns else np.array([])
        psi_val = psi(ref_sample, cur_values, bin_edges)
        ks_stat, ks_pvalue = ks_drift(ref_sample, cur_values)
        status = "block" if psi_val >= psi_block else "warn" if psi_val >= psi_warn else "ok"
        rows.append({"feature": col, "psi": psi_val, "ks_stat": ks_stat, "ks_pvalue": ks_pvalue, "status": status})
    return pd.DataFrame(rows, columns=["feature", "psi", "ks_stat", "ks_pvalue", "status"])


def prediction_drift_report(
    reference: dict, current_p_defect_cal: np.ndarray, current_indications_per_km: float, ratio_warn: float,
) -> dict:
    """KS of the current survey's calibrated P(defect) against the training
    reference sample, plus the indications-per-km ratio vs the training
    rate. "A survey suddenly producing 3x the usual indications is a sensor
    or background problem until proven otherwise" (Layer 4) -- flagged when
    the ratio falls outside [1/ratio_warn, ratio_warn].
    """
    ref_sample = np.asarray(reference.get("p_defect_cal_sample", []), dtype=float)
    ks_stat, ks_pvalue = ks_drift(ref_sample, np.asarray(current_p_defect_cal, dtype=float))
    ref_rate = reference.get("indications_per_km", 0.0)
    ratio = current_indications_per_km / ref_rate if ref_rate and ref_rate > 0 else float("nan")
    flagged = not np.isnan(ratio) and not (1.0 / ratio_warn <= ratio <= ratio_warn)
    return {
        "p_defect_cal_ks_stat": ks_stat,
        "p_defect_cal_ks_pvalue": ks_pvalue,
        "indications_per_km_current": current_indications_per_km,
        "indications_per_km_reference": ref_rate,
        "indications_per_km_ratio": ratio,
        "flagged": flagged,
    }


def background_regime_shift(
    current_median: pd.Series, history_medians: pd.DataFrame, z_threshold: float = 3.0
) -> dict:
    """Per-axis (bx/by/bz) z-score of `current_median` against
    `history_medians`' own mean/std -- "catches recalibration, a different
    scanner unit, or a seasonal geomagnetic excursion" (Layer 4). Extracted
    from `validate.py::check_background_regime`'s exact math so there is ONE
    implementation, reused by both the ingest-time DQ gate (comparing a new
    survey against other already-ingested surveys of the same line) and the
    Stage 8 monitor (comparing against a released pipeline's history) --
    not two copies that can silently diverge.
    """
    hist_mean, hist_std = history_medians.mean(), history_medians.std().replace(0, np.nan)
    z = ((current_median - hist_mean) / hist_std).abs()
    n_bad = int((z > z_threshold).fillna(False).sum())
    return {"z_scores": z.to_dict(), "n_bad": n_bad}


def coverage_vs_nominal(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray, nominal: float) -> dict:
    """Empirical coverage of `[lo, hi]` vs `nominal`. The Stage 8 coverage-
    tracking check: as `truth_observation` rows accumulate from the
    dig-feedback loop, re-run this; if empirical coverage drifts below
    nominal, re-fit the conformal quantiles before touching the model
    (Layer 4). `n=0` is an honest "nothing verified yet" answer, not a
    failure.
    """
    y_true = np.asarray(y_true, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    n = len(y_true)
    if n == 0:
        return {"empirical": float("nan"), "nominal": nominal, "n": 0, "below_nominal": False}
    empirical = float(np.mean((lo <= y_true) & (y_true <= hi)))
    return {"empirical": empirical, "nominal": nominal, "n": n, "below_nominal": empirical < nominal}
