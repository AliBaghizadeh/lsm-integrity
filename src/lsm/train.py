"""
Stage 3 + Stage 4: MAD-threshold baseline vs IsolationForest (detection), then
per-indication severity regression (LightGBM quantile + split conformal) on
top of whichever indications the anomaly detector actually produces. Grouped
CV throughout, MLflow logging, real `bundle.py` artifacts, one
`pipeline_release` row referencing both, a real `indication` table, and an
auto-generated model card.

`indications.geojson` export is `predict.py`'s job, not this module's.

Run: python -m lsm train
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # never needs a GUI backend -- only ever saves figures to MLflow
import matplotlib.pyplot as plt  # noqa: E402
import mlflow
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_recall_curve

from lsm.bundle import save_bundle, training_feature_summary, training_prediction_reference
from lsm.config import Config
from lsm.evaluate import (
    add_fold_column,
    bootstrap_ci,
    brier_by_group,
    defect_hit_rates,
    false_dig_rate_per_run,
    interference_dig_fraction_per_run,
    interference_precision_units,
    localisation_errors_m,
    mae_by_severity_decile,
    match_dug_indications,
    paired_bootstrap_ci,
    per_class_recall_units,
    per_group_severity_metrics,
    pr_auc,
    reliability_curve,
    shap_denylist_check,
)
from lsm.features import feature_columns, load_feature_corpus
from lsm.hashing import data_sha256
from lsm.indications import (
    attach_indication_features,
    cluster_indications,
    select_dig_budget,
    write_indications,
)
from lsm.logging_utils import get_logger
from lsm.model_card import write_model_card
from lsm.models.anomaly import IsolationForestAnomalyModel, MADBaseline, calibrated_threshold
from lsm.models.classify import ClassifyModel, MajorityClassBaseline
from lsm.models.severity import GlobalMeanSeverityBaseline, SeverityModel
from lsm.schemas import CLASSIFY_CLASSES, DEFECT_TYPES
from lsm.truth import build_truth_registry

log = get_logger("lsm.train")

# A dug indication is credited to the nearest truth source within this many
# metres. Generous enough to cover interference's wider label box (half-width up
# to ~2 * sqrt(depth_m^2 + 8^2) =~ 16 m at the far end of the lateral-offset
# range, see config/base.yaml's label_window_scale comment) without being so
# wide that unrelated background rows start matching by chance.
MATCH_TOLERANCE_M = 15.0

# The Stage 3 gate (PLAN.md): IsolationForest must beat MAD by this much
# recall@budget, on the LOWER bound of the paired bootstrap CI -- a point-
# estimate gap alone is not a passing gate (validation-and-trust.md).
GATE_RECALL_MARGIN = 0.15

# The Stage 4 gate: split-conformal coverage must land in this band on the
# grouped holdout -- too low means the interval is a false promise, too high
# means it's uselessly wide.
GATE_SEVERITY_COVERAGE_RANGE = (0.87, 0.93)

# Fraction of each fold's TRAIN indications held out as the conformal
# calibration set. Conformal's finite-sample coverage guarantee requires the
# calibration set to be disjoint from what fit the quantile models -- fitting
# and calibrating on the same rows leaks, same as any other fitted transform.
CONFORMAL_CALIB_FRACTION = 0.3

# The Stage 5 gate's other half (see CLASSIFY_DENYLIST below): SCC is
# protected specifically because a missed crack-like defect is the sudden-
# failure-mode class this whole project cares most about catching -- checked
# on the CI lower bound, same discipline as every other gate here.
GATE_SCC_RECALL = 0.90

# Physics-consistency denylist for the classifier's SHAP/contribution
# importance ranking. These are absolute-position columns that could make a
# model key on "this class always sits at this chainage" -- true only on
# THIS synthetic corpus's fixed defect placement, and a shortcut that would
# never generalise to a real, different pipeline.
CLASSIFY_DENYLIST = ["chainage_m", "chainage_peak_m", "chainage_start_m", "chainage_end_m", "sample_idx"]

# Same reasoning as CONFORMAL_CALIB_FRACTION, for the classifier's isotonic
# probability calibration -- a defect-grouped split, never by row.
CLASSIFY_CALIB_FRACTION = 0.3


def _git_sha(cwd: str | Path | None = None) -> str:
    """HEAD's SHA, with a `-dirty` suffix if the working tree has uncommitted
    changes. Without this, a run against edited-but-uncommitted code would tag
    itself with the PREVIOUS commit's SHA, silently claiming to be reproducible
    from a commit that doesn't actually match what ran -- a real gap in the
    project's own git_sha + config_sha256 + data_sha256 reproducibility
    invariant (SKILL #6), not a hypothetical one.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, cwd=cwd
        )
        sha = out.stdout.strip()
        if out.returncode != 0 or not sha:
            return "uncommitted"
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5, cwd=cwd
        )
        if status.returncode == 0 and status.stdout.strip():
            return f"{sha}-dirty"
        return sha
    except Exception:
        return "uncommitted"


def _short_sha(git_sha: str) -> str:
    """First 8 hex chars, with `-dirty` preserved if present -- so a dirty-tree
    run is visible in the human-readable model/pipeline version tag, not only
    in the full `git_sha` field logged to MLflow and `model_run`.
    """
    if git_sha.endswith("-dirty"):
        return f"{git_sha[: -len('-dirty')][:8]}-dirty"
    return git_sha[:8]


def _load_truth_and_geometry(conn, survey_ids: list[str]) -> pd.DataFrame:
    """Per (survey_id, sample_idx): lat, lon, defect, defect_type, interference,
    severity_smys -- read from the raw parquet, same reasoning as pipeline.py's
    run_feature_pipeline: this is ground truth + restricted geometry, none of
    which lives in the label-free feature store, and `truth_defect` /
    `truth_observation` are not populated yet (known gap, see PLAN.md).
    """
    placeholders = ",".join("?" * len(survey_ids))
    rows = conn.execute(
        f"SELECT survey_id, source_uri FROM survey WHERE survey_id IN ({placeholders})",
        survey_ids,
    ).fetchall()
    frames = []
    for survey_id, source_uri in rows:
        df = pd.read_parquet(
            source_uri,
            columns=["sample_idx", "lat", "lon", "defect", "defect_type", "interference", "severity_smys"],
        )
        df["survey_id"] = survey_id
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _run_grouped_cv(
    corpus: pd.DataFrame, feature_cols: list[str], cfg: Config, seed: int
) -> pd.DataFrame:
    """Fit both models fold-by-fold, train on the other folds, score the held-out
    one -- returns `corpus` with two new out-of-fold score columns,
    `score_mad` / `score_if`, covering every row exactly once.

    `corpus` must already carry a `fold` column (assigned by the caller via
    `add_fold_column` or, for Stage 6's whole-line holdout, `add_fold_column_
    by_line` -- see `scale_eval.run_whole_line_cv`); `n_folds` is derived from
    the data (`corpus["fold"].max() + 1`), the same pattern `_run_severity_cv`/
    `_run_classify_cv` already use, so this function works identically
    regardless of which grouping scheme assigned `fold`.
    """
    n_folds = int(corpus["fold"].max()) + 1
    corpus["score_mad"] = np.nan
    corpus["score_if"] = np.nan
    mad_thresholds = []

    for k in range(n_folds):
        train_mask = corpus["fold"] != k
        test_mask = corpus["fold"] == k
        if not test_mask.any():
            continue
        train, test = corpus.loc[train_mask], corpus.loc[test_mask]

        mad = MADBaseline().fit(train)
        iso = IsolationForestAnomalyModel(
            feature_cols=feature_cols,
            contamination=cfg.base.model.anomaly["contamination"],
            n_estimators=cfg.base.model.anomaly["n_estimators"],
            seed=seed,
            emphasize_features=cfg.base.model.anomaly.get("emphasize_features"),
            emphasis_repeats=cfg.base.model.anomaly.get("emphasis_repeats", 1),
        ).fit(train)

        mad_thresholds.append(calibrated_threshold(mad.score(train), cfg.base.model.anomaly["contamination"]))
        corpus.loc[test_mask, "score_mad"] = mad.score(test)
        corpus.loc[test_mask, "score_if"] = iso.score(test)

    corpus.attrs["mad_threshold"] = float(np.mean(mad_thresholds))
    return corpus


def _dig_and_match(
    corpus: pd.DataFrame, score_col: str, threshold: float, registry: pd.DataFrame, cfg: Config
) -> dict[str, pd.DataFrame]:
    """Per survey: cluster indications, apply the dig budget, match to truth.
    Returns {survey_id: matched_dug_dataframe}.
    """
    matched_by_run: dict[str, pd.DataFrame] = {}
    for survey_id, survey_rows in corpus.groupby("survey_id"):
        survey_rows = survey_rows.sort_values("sample_idx")
        indications = cluster_indications(
            survey_rows.rename(columns={score_col: "_score"}),
            "_score",
            threshold,
            survey_id=str(survey_id),
            pipeline_version="cv-eval",
        )
        dug = select_dig_budget(
            indications,
            survey_length_m=cfg.base.data.length_m,
            digs_per_km=cfg.base.model.dig_budget_per_km,
        )
        line_registry = registry[registry["line_id"] == survey_rows["line_id"].iloc[0]]
        matched_by_run[str(survey_id)] = match_dug_indications(dug, line_registry, MATCH_TOLERANCE_M)
    return matched_by_run


def _cluster_all_indications(
    corpus: pd.DataFrame, score_col: str, threshold: float, pipeline_version: str = "cv-eval"
) -> pd.DataFrame:
    """Every survey's supra-threshold rows clustered into indications -- the
    FULL candidate list, not a dig-budget-selected slice. Severity applies to
    whatever the detector flagged, not just what fits one run's budget.
    """
    all_indications = []
    for survey_id, survey_rows in corpus.groupby("survey_id"):
        survey_rows = survey_rows.sort_values("sample_idx").rename(columns={score_col: "_score"})
        indications = cluster_indications(
            survey_rows, "_score", threshold, survey_id=str(survey_id), pipeline_version=pipeline_version
        )
        all_indications.append(indications)
    return pd.concat(all_indications, ignore_index=True) if all_indications else pd.DataFrame()


def _match_all_indications(
    indications: pd.DataFrame, corpus: pd.DataFrame, registry: pd.DataFrame
) -> pd.DataFrame:
    """Match every clustered indication -- defect, interference, AND unmatched
    false alarm, the unfiltered superset -- to its nearest truth source, per
    survey's own line registry. Shared by the severity frame (defect-only),
    the Stage 5 classify frame (defect+interference), and the Stage 5
    `p_defect_cal` calibrator, which specifically NEEDS the unmatched rows
    too (it learns what "not a real match" looks like from them).
    """
    if len(indications) == 0:
        return indications
    matched_frames = []
    for survey_id, group in indications.groupby("survey_id"):
        line_id = corpus.loc[corpus["survey_id"] == survey_id, "line_id"].iloc[0]
        line_registry = registry[registry["line_id"] == line_id]
        matched_frames.append(match_dug_indications(group, line_registry, MATCH_TOLERANCE_M))
    return pd.concat(matched_frames, ignore_index=True)


def _build_severity_training_frame(
    matched: pd.DataFrame, corpus: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    """Every already-matched indication that matches a real DEFECT (not
    interference, not unmatched), with its feature vector, true severity, and
    CV fold attached -- the full severity training/evaluation universe.
    `matched` is the shared cluster+match pass (`_cluster_all_indications` +
    `_match_all_indications`), not recomputed here.
    """
    defects_only = matched[matched["matched_kind"] == "defect"].copy()
    if len(defects_only) == 0:
        return defects_only

    with_features = attach_indication_features(defects_only, corpus, feature_cols)

    # True severity at THIS survey's peak row -- severity_smys grows per
    # run_id, so it must come from the matched indication's own survey, not
    # from the registry (which only carries a chainage/type, no value).
    truth = corpus[["survey_id", "chainage_m", "severity_smys"]].rename(
        columns={"chainage_m": "chainage_peak_m", "severity_smys": "y_true"}
    )
    with_features = with_features.merge(truth, on=["survey_id", "chainage_peak_m"], how="left")

    # fold: whichever fold the peak row belongs to (assigned in _run_grouped_cv).
    fold_lookup = corpus[["survey_id", "chainage_m", "fold"]].rename(columns={"chainage_m": "chainage_peak_m"})
    with_features = with_features.merge(fold_lookup, on=["survey_id", "chainage_peak_m"], how="left")

    # generate.py initialises severity_smys to NaN off-defect (row outside
    # every defect's label window), not 0 -- a real, longstanding discrepancy
    # from this project's own documented "0 off-defect" convention, found via
    # Stage 6 at scale (2026-07-31): a detector's PEAK occasionally lands just
    # outside a defect's exact label-window half-width while still within the
    # looser MATCH_TOLERANCE_M dig-matching radius, so match_dug_indications
    # still credits it as matched, but its OWN row's severity_smys is NaN, not
    # the defect's true value. ~0.4% of matched defects at 9,600-defect scale
    # (0 at demo scale, apparently never sampled). A single NaN y_true reaching
    # SeverityModel.fit's conformal calibration silently NaNs the WHOLE fold's
    # margin (np.quantile propagates NaN), collapsing pooled OOF coverage to
    # 0% -- drop these rows here, loudly, rather than let that cascade.
    n_nan_y_true = int(with_features["y_true"].isna().sum())
    if n_nan_y_true > 0:
        print(f"_build_severity_training_frame: dropping {n_nan_y_true} matched indication(s) "
              "whose peak row has a NaN severity_smys (peak landed outside the true label "
              "window while still within the dig-matching tolerance) -- see train.py's comment.")
        with_features = with_features[with_features["y_true"].notna()]
    return with_features


def _build_classify_training_frame(
    matched: pd.DataFrame, corpus: pd.DataFrame, feature_cols: list[str], registry: pd.DataFrame
) -> pd.DataFrame:
    """Every already-matched indication that matches a real DEFECT OR
    INTERFERENCE source -- widened from severity's defect-only restriction,
    since interference is an explicit class to classify here, not to exclude.
    `defect_type` comes straight off the registry join (already carries the
    literal string "interference" for interference sources) -- no new
    label-derivation logic, just one merge.
    """
    labelled = matched[matched["matched_kind"].isin(["defect", "interference"])].copy()
    if len(labelled) == 0:
        return labelled

    with_features = attach_indication_features(labelled, corpus, feature_cols)

    with_features = with_features.merge(
        registry[["source_id", "defect_type"]], left_on="matched_source_id", right_on="source_id", how="left"
    ).drop(columns=["source_id"])

    fold_lookup = corpus[["survey_id", "chainage_m", "fold"]].rename(columns={"chainage_m": "chainage_peak_m"})
    return with_features.merge(fold_lookup, on=["survey_id", "chainage_peak_m"], how="left")


def _run_severity_cv(
    severity_frame: pd.DataFrame, feature_cols: list[str], cfg: Config, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fold-by-fold: hold out one fold's matched indications as test, split the
    REMAINING folds' indications into an inner train/calibration partition --
    by DEFECT, never by row, so a defect never straddles the two -- fit the
    quantile model + conformal margin on train/calib, predict on the held-out
    fold. Returns (oof_model, oof_baseline): pooled out-of-fold rows
    (defect_id, y_true, y_pred, lo, hi) for the SeverityModel and the
    global-mean baseline, evaluated identically.
    """
    n_folds = int(severity_frame["fold"].max()) + 1 if len(severity_frame) else 0
    sev_cfg = cfg.base.model.severity
    quantiles = tuple(sev_cfg["quantiles"])
    conformal_alpha = sev_cfg["conformal_alpha"]
    lgbm_cfg = cfg.base.model.lightgbm.model_dump()
    rng = np.random.default_rng(seed)

    model_rows, baseline_rows = [], []
    for k in range(n_folds):
        test = severity_frame[severity_frame["fold"] == k]
        dev = severity_frame[severity_frame["fold"] != k]
        if len(test) == 0 or len(dev) == 0:
            continue

        dev_defects = dev["matched_source_id"].unique()
        rng.shuffle(dev_defects)
        n_calib = max(1, round(len(dev_defects) * CONFORMAL_CALIB_FRACTION)) if len(dev_defects) > 1 else 0
        calib_defects = set(dev_defects[:n_calib])
        train = dev[~dev["matched_source_id"].isin(calib_defects)]
        calib = dev[dev["matched_source_id"].isin(calib_defects)]
        if len(train) == 0:
            # Too few distinct defects in this fold's dev set to split at all --
            # calibrate on the same rows as training. Coverage from a fold like
            # this is honest-but-noisy, not wrong; the CI over defects is what
            # protects a reader from over-trusting it.
            train, calib = dev, dev.iloc[0:0]

        model = SeverityModel(
            feature_cols=feature_cols, quantiles=quantiles, conformal_alpha=conformal_alpha,
            seed=seed, lgbm_cfg=lgbm_cfg,
            min_child_samples=sev_cfg.get("min_child_samples", 3),
            n_estimators=sev_cfg.get("n_estimators", 50),
            num_leaves=sev_cfg.get("num_leaves", 7),
        ).fit(train, train["y_true"].to_numpy(), calib, calib["y_true"].to_numpy())
        med, lo, hi = model.predict(test)

        baseline = GlobalMeanSeverityBaseline(conformal_alpha=conformal_alpha).fit(
            train["y_true"].to_numpy(), calib["y_true"].to_numpy()
        )
        bmed, blo, bhi = baseline.predict(len(test))

        for i, (_, row) in enumerate(test.iterrows()):
            model_rows.append(
                {"defect_id": row["matched_source_id"], "y_true": row["y_true"],
                 "y_pred": med[i], "lo": lo[i], "hi": hi[i]}
            )
            baseline_rows.append(
                {"defect_id": row["matched_source_id"], "y_true": row["y_true"],
                 "y_pred": bmed[i], "lo": blo[i], "hi": bhi[i]}
            )

    return pd.DataFrame(model_rows), pd.DataFrame(baseline_rows)


def _severity_bootstrap_metrics(oof: pd.DataFrame, cfg: Config, seed: int) -> dict[str, tuple[float, float, float]]:
    boot_cfg = cfg.base.model.bootstrap
    per_group = per_group_severity_metrics(oof, "defect_id")
    return {
        "coverage": bootstrap_ci(per_group["coverage"], boot_cfg.n_resamples, boot_cfg.level, seed),
        "mae": bootstrap_ci(per_group["mae"], boot_cfg.n_resamples, boot_cfg.level, seed + 1),
        "interval_width": bootstrap_ci(per_group["interval_width"], boot_cfg.n_resamples, boot_cfg.level, seed + 2),
    }


def _stratified_defect_calib_split(
    dev: pd.DataFrame, class_col: str, group_col: str, calib_fraction: float, rng: np.random.Generator
) -> tuple[set, set]:
    """Allocate each class's OWN distinct defects to calib proportionally,
    rather than a bare shuffle of all defects -- a plain shuffle risks a rare
    class (as few as a dozen SCC instances total in the real corpus) landing
    with zero calibration examples purely by chance, which sklearn's isotonic
    calibration cannot recover from (confirmed empirically: raises
    `IndexError`, not a theoretical concern). Returns (train_defect_ids,
    calib_defect_ids). A class with too few distinct defects to split at all
    keeps all of that class's defects in train -- same "honest but noisy"
    acceptance as severity's own degenerate-fold path.
    """
    train_ids: set = set()
    calib_ids: set = set()
    for _, class_group in dev.groupby(class_col):
        defects = class_group[group_col].unique()
        rng.shuffle(defects)
        n_calib = max(1, round(len(defects) * calib_fraction)) if len(defects) > 1 else 0
        calib_ids.update(defects[:n_calib])
        train_ids.update(defects[n_calib:])
    return train_ids, calib_ids


def _run_classify_cv(
    classify_frame: pd.DataFrame, feature_cols: list[str], cfg: Config, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fold-by-fold: hold out one fold's matched indications as test, split
    the remaining folds' indications into an inner train/calibration
    partition by DEFECT, class-stratified (`_stratified_defect_calib_split`)
    -- fit `ClassifyModel` + `MajorityClassBaseline` on train/calib, predict
    on the held-out fold. Returns (oof_model, oof_baseline): pooled
    out-of-fold rows (defect_id, true_class, pred_type, pred_conf, and one
    column per class's calibrated probability) for the classifier and the
    baseline, evaluated identically.
    """
    n_folds = int(classify_frame["fold"].max()) + 1 if len(classify_frame) else 0
    classify_cfg = cfg.base.model.classify
    lgbm_cfg = {
        **cfg.base.model.lightgbm.model_dump(),
        "n_estimators": classify_cfg.get("n_estimators", 50),
        "num_leaves": classify_cfg.get("num_leaves", 7),
    }
    rng = np.random.default_rng(seed)

    model_rows, baseline_rows = [], []
    for k in range(n_folds):
        test = classify_frame[classify_frame["fold"] == k].reset_index(drop=True)
        dev = classify_frame[classify_frame["fold"] != k]
        if len(test) == 0 or len(dev) == 0:
            continue

        train_ids, calib_ids = _stratified_defect_calib_split(
            dev, "defect_type", "matched_source_id", CLASSIFY_CALIB_FRACTION, rng
        )
        train = dev[dev["matched_source_id"].isin(train_ids)]
        calib = dev[dev["matched_source_id"].isin(calib_ids)]
        if len(train) == 0:
            train, calib = dev, dev.iloc[0:0]

        model = ClassifyModel(
            feature_cols=feature_cols, classes=CLASSIFY_CLASSES, seed=seed, lgbm_cfg=lgbm_cfg,
            class_weight=classify_cfg.get("class_weight", "balanced"),
            min_child_samples=classify_cfg.get("min_child_samples", 3),
        ).fit(train, train["defect_type"].to_numpy(), calib, calib["defect_type"].to_numpy())
        proba = model.predict_proba(test)
        pred_type, pred_conf = model.predict(test)

        baseline = MajorityClassBaseline(classes=CLASSIFY_CLASSES).fit(train["defect_type"].to_numpy())
        baseline_proba = baseline.predict_proba(len(test))
        base_pred_type, base_pred_conf = baseline.predict(len(test))

        for i in range(len(test)):
            row = test.iloc[i]
            model_row = {
                "defect_id": row["matched_source_id"], "true_class": row["defect_type"],
                "pred_type": pred_type[i], "pred_conf": pred_conf[i],
            }
            model_row.update({c: proba.iloc[i][c] for c in CLASSIFY_CLASSES})
            model_rows.append(model_row)

            baseline_row = {
                "defect_id": row["matched_source_id"], "true_class": row["defect_type"],
                "pred_type": base_pred_type[i], "pred_conf": base_pred_conf[i],
            }
            baseline_row.update({c: baseline_proba.iloc[i][c] for c in CLASSIFY_CLASSES})
            baseline_rows.append(baseline_row)

    return pd.DataFrame(model_rows), pd.DataFrame(baseline_rows)


def _classify_bootstrap_metrics(oof: pd.DataFrame, cfg: Config, seed: int) -> dict:
    boot_cfg = cfg.base.model.bootstrap
    recall_units = per_class_recall_units(oof, "defect_id", "true_class", "pred_type", CLASSIFY_CLASSES)
    per_class_recall = {
        c: bootstrap_ci(recall_units[c], boot_cfg.n_resamples, boot_cfg.level, seed + i)
        for i, c in enumerate(CLASSIFY_CLASSES)
    }
    interference_prec_units = interference_precision_units(oof, "defect_id", "true_class", "pred_type")
    brier_units = brier_by_group(oof, "defect_id", "true_class", CLASSIFY_CLASSES, CLASSIFY_CLASSES)
    return {
        "per_class_recall": per_class_recall,
        "interference_precision": bootstrap_ci(
            interference_prec_units, boot_cfg.n_resamples, boot_cfg.level, seed + 100
        ),
        "brier": bootstrap_ci(brier_units, boot_cfg.n_resamples, boot_cfg.level, seed + 101),
    }


def _lightgbm_shap_importance(model: ClassifyModel, X: pd.DataFrame) -> pd.Series:
    """Mean(|contribution|) per feature, aggregated across rows and classes --
    LightGBM's native `pred_contrib=True` gives genuine TreeSHAP values
    without the external `shap` package, whose `numba` dependency does not
    support the installed numpy in this environment (confirmed empirically:
    `import shap` raises `ImportError: Numba needs NumPy 2.4 or less`, and
    numba's latest release still doesn't support it -- not fixable by
    upgrading). This is what `evaluate.shap_denylist_check` (the physics-
    consistency gate) runs against.
    """
    assert model.model is not None, "_lightgbm_shap_importance requires a real (non-constant-fallback) model"
    X_mat = X[model.feature_cols].fillna(0.0)
    contrib = model.model.predict(X_mat, pred_contrib=True)
    n_features = len(model.feature_cols)
    # The ACTUALLY-fitted model's class count, not len(model.classes) (the
    # full pinned list) -- num_class is set dynamically per fit from the
    # classes really present in y_train (see ClassifyModel.fit), so the two
    # can differ, and pred_contrib's output width follows the fitted model.
    n_classes = len(model.model.classes_)
    reshaped = np.asarray(contrib).reshape(len(X_mat), n_classes, n_features + 1)
    mean_abs = np.abs(reshaped[:, :, :n_features]).mean(axis=(0, 1))
    return pd.Series(mean_abs, index=model.feature_cols)


def _explain_indications_sample(
    model: ClassifyModel, X: pd.DataFrame, pred_type: np.ndarray, n_examples: int = 5
) -> dict:
    """One worked example per predicted class: its own per-feature
    contribution to ITS predicted class, via the same native `pred_contrib`
    mechanism as `_lightgbm_shap_importance`. Logged as an MLflow JSON
    artifact only -- no `indication` table column exists for per-indication
    SHAP, and adding one is a `schema_version` bump this first pass doesn't
    need (no consumer yet); a real productionisation would add one once a
    dashboard panel actually reads it.
    """
    assert model.model is not None, "_explain_indications_sample requires a real (non-constant-fallback) model"
    X_mat = X[model.feature_cols].fillna(0.0)
    contrib = np.asarray(model.model.predict(X_mat, pred_contrib=True))
    n_features = len(model.feature_cols)
    n_classes = len(model.model.classes_)  # see _lightgbm_shap_importance's comment
    reshaped = contrib.reshape(len(X_mat), n_classes, n_features + 1)
    class_index = {c: i for i, c in enumerate(model.model.classes_)}

    examples: dict = {}
    for c in model.classes:
        if c not in class_index:
            continue
        rows_of_class = np.where(pred_type == c)[0]
        if len(rows_of_class) == 0:
            continue
        i, ci = rows_of_class[0], class_index[c]
        examples[c] = dict(zip(model.feature_cols, reshaped[i, ci, :n_features].tolist()))
        if len(examples) >= n_examples:
            break
    return examples


def _bootstrap_metrics(
    matched_by_run: dict[str, pd.DataFrame], registry: pd.DataFrame, run_line_id: dict[str, str], cfg: Config
) -> tuple[dict[str, tuple[float, float, float]], dict[str, float], list[float], list[float]]:
    boot_cfg = cfg.base.model.bootstrap
    hit_rates = defect_hit_rates(matched_by_run, registry, run_line_id=run_line_id)
    false_digs = [false_dig_rate_per_run(m) for m in matched_by_run.values()]
    interference_fracs = [interference_dig_fraction_per_run(m) for m in matched_by_run.values()]
    loc_errors = np.concatenate([localisation_errors_m(m) for m in matched_by_run.values()])

    return {
        "recall_at_budget": bootstrap_ci(list(hit_rates.values()), boot_cfg.n_resamples, boot_cfg.level, cfg.seed),
        "false_dig_rate": bootstrap_ci(false_digs, boot_cfg.n_resamples, boot_cfg.level, cfg.seed + 1),
        "interference_dig_fraction": bootstrap_ci(
            interference_fracs, boot_cfg.n_resamples, boot_cfg.level, cfg.seed + 2
        ),
        "localisation_error_m": bootstrap_ci(loc_errors, boot_cfg.n_resamples, boot_cfg.level, cfg.seed + 3),
    }, hit_rates, false_digs, interference_fracs


def _evaluate_corpus(
    corpus: pd.DataFrame,
    feature_cols: list[str],
    cfg: Config,
    seed: int,
    registry: pd.DataFrame,
    run_line_id: dict[str, str],
    survey_ids: list[str],
) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame, "IsotonicRegression | None"]:
    """Stage 3+4+5 evaluation over an already-fold-assigned `corpus` (caller
    must have already called `add_fold_column`/`add_fold_column_by_line`).
    Returns (corpus_with_scores, result, severity_frame, classify_frame,
    defect_calibrator) -- exactly what `_log_and_persist`/`_print_report`
    need. Factored out of `run_train` so Stage 6's whole-line-holdout
    rehearsal (`scale_eval.run_whole_line_cv`) gets identical metrics
    computation with a differently-assigned `fold` column, without
    duplicating this body. `run_train`'s own default (block) CV path calls
    this too -- see below -- so this is a pure extraction, not new logic.
    """
    corpus = _run_grouped_cv(corpus, feature_cols, cfg, seed=seed)
    mad_threshold = corpus.attrs["mad_threshold"]

    matched_mad = _dig_and_match(corpus, "score_mad", mad_threshold, registry, cfg)
    matched_if = _dig_and_match(corpus, "score_if", 0.0, registry, cfg)

    metrics_mad, hit_rates_mad, false_digs_mad, interference_mad = _bootstrap_metrics(
        matched_mad, registry, run_line_id, cfg
    )
    metrics_if, hit_rates_if, false_digs_if, interference_if = _bootstrap_metrics(
        matched_if, registry, run_line_id, cfg
    )

    pr_auc_mad = pr_auc(corpus["defect"].to_numpy(), corpus["score_mad"].to_numpy())
    pr_auc_if = pr_auc(corpus["defect"].to_numpy(), corpus["score_if"].to_numpy())

    boot_cfg = cfg.base.model.bootstrap
    defect_ids = sorted(hit_rates_mad)
    recall_gap = paired_bootstrap_ci(
        [hit_rates_if[d] for d in defect_ids],
        [hit_rates_mad[d] for d in defect_ids],
        boot_cfg.n_resamples,
        boot_cfg.level,
        seed + 10,
    )
    interference_gap = paired_bootstrap_ci(
        interference_mad, interference_if, boot_cfg.n_resamples, boot_cfg.level, seed + 11
    )

    gate_passed = recall_gap[1] >= GATE_RECALL_MARGIN  # lower CI bound, not the point estimate
    interference_attributable = interference_gap[1] > 0  # MAD wastes MORE digs on interference, at the CI floor

    result = {
        "mad": {"recall_at_budget": metrics_mad["recall_at_budget"], "false_dig_rate": metrics_mad["false_dig_rate"],
                "interference_dig_fraction": metrics_mad["interference_dig_fraction"],
                "localisation_error_m": metrics_mad["localisation_error_m"], "pr_auc": pr_auc_mad},
        "isolation_forest": {"recall_at_budget": metrics_if["recall_at_budget"], "false_dig_rate": metrics_if["false_dig_rate"],
                              "interference_dig_fraction": metrics_if["interference_dig_fraction"],
                              "localisation_error_m": metrics_if["localisation_error_m"], "pr_auc": pr_auc_if},
        "recall_gap_if_minus_mad": recall_gap,
        "interference_gap_mad_minus_if": interference_gap,
        "gate_passed": gate_passed,
        "interference_attributable": interference_attributable,
        "n_defects": len(defect_ids),
        "n_surveys": len(survey_ids),
    }

    # Stage 4 and Stage 5 share one cluster+match pass over IsolationForest's
    # OOF scores -- the unfiltered superset (defect, interference, AND
    # unmatched false alarms), computed once rather than separately per stage.
    all_indications = _cluster_all_indications(corpus, "score_if", 0.0)
    matched_all = _match_all_indications(all_indications, corpus, registry)

    # Stage 4: severity, conditional on the anomaly detector having flagged
    # SOMETHING that matches a real defect.
    severity_frame = _build_severity_training_frame(matched_all, corpus, feature_cols)
    result["n_severity_samples"] = len(severity_frame)
    if len(severity_frame) >= 4 and severity_frame["matched_source_id"].nunique() >= 2:
        oof_model, oof_baseline = _run_severity_cv(
            severity_frame, [*feature_cols, "extent_m"], cfg, seed=seed
        )
        if len(oof_model) > 0:
            sev_metrics = _severity_bootstrap_metrics(oof_model, cfg, seed=seed + 20)
            base_metrics = _severity_bootstrap_metrics(oof_baseline, cfg, seed=seed + 30)
            severity_gate_passed = (
                GATE_SEVERITY_COVERAGE_RANGE[0] <= sev_metrics["coverage"][0] <= GATE_SEVERITY_COVERAGE_RANGE[1]
            )
            result["severity"] = sev_metrics
            result["severity_baseline"] = base_metrics
            result["severity_gate_passed"] = severity_gate_passed
            result["severity_mae_by_decile"] = {
                str(k): v
                for k, v in mae_by_severity_decile(
                    oof_model["y_true"].to_numpy(), oof_model["y_pred"].to_numpy()
                ).to_dict().items()
            }

    # Stage 5: the p_defect_cal calibrator -- fit on the UNFILTERED matched
    # population (defect, interference, unmatched all present), so it learns
    # what "not a real match" looks like, which the classifier itself never
    # sees (its training universe excludes true false alarms -- see
    # _build_classify_training_frame).
    defect_calibrator = None
    if len(matched_all) > 0:
        is_defect = (matched_all["matched_kind"] == "defect").astype(float).to_numpy()
        defect_calibrator = IsotonicRegression(out_of_bounds="clip").fit(
            matched_all["anomaly_score"].to_numpy(), is_defect
        )

    # Stage 8: prediction-drift reference material -- the training corpus's
    # own indications-per-km rate and calibrated P(defect) distribution,
    # captured here (not re-derived at monitor time, which could itself have
    # drifted). Stashed on `result` so _log_and_persist can bundle it.
    total_km = corpus.groupby("survey_id")["chainage_m"].agg(lambda s: s.max() - s.min()).sum() / 1000.0
    result["indications_per_km"] = len(all_indications) / total_km if total_km > 0 else 0.0
    result["p_defect_cal_sample"] = (
        defect_calibrator.predict(matched_all["anomaly_score"].to_numpy())
        if defect_calibrator is not None else np.array([])
    )

    # Stage 5: classification, conditional on enough matched, multi-class
    # data to fit and evaluate at all (mirrors severity's own sample-size
    # guard, widened to also require class diversity).
    classify_frame = _build_classify_training_frame(matched_all, corpus, feature_cols, registry)
    result["n_classify_samples"] = len(classify_frame)
    if len(classify_frame) >= 8 and classify_frame["defect_type"].nunique() >= 3:
        oof_classify, oof_classify_baseline = _run_classify_cv(
            classify_frame, [*feature_cols, "extent_m"], cfg, seed=seed
        )
        if len(oof_classify) > 0:
            classify_metrics = _classify_bootstrap_metrics(oof_classify, cfg, seed=seed + 40)
            classify_baseline_metrics = _classify_bootstrap_metrics(oof_classify_baseline, cfg, seed=seed + 50)
            scc_recall_ci = classify_metrics["per_class_recall"].get("scc", (float("nan"),) * 3)
            scc_gate_passed = scc_recall_ci[1] >= GATE_SCC_RECALL  # lower CI bound, not the point estimate
            result["classify"] = classify_metrics
            result["classify_baseline"] = classify_baseline_metrics
            result["classify_recall_gate_passed"] = scc_gate_passed
            result["classify_reliability"] = reliability_curve(
                oof_classify["pred_conf"].to_numpy(),
                (oof_classify["pred_type"] == oof_classify["true_class"]).to_numpy(),
            ).to_dict(orient="list")

    return corpus, result, severity_frame, classify_frame, defect_calibrator


def run_train(cfg: Config, conn) -> dict:
    """Orchestrates the whole Stage 3 training + evaluation + logging path.
    Returns the metrics dict actually printed/logged, for tests to assert on.
    """
    feature_version = cfg.base.features.version
    as_of = dt.datetime.now(dt.timezone.utc).isoformat()

    corpus = load_feature_corpus(cfg.env.storage.feature_dir, feature_version, as_of=as_of)
    if len(corpus) == 0:
        raise RuntimeError("empty feature corpus -- run `lsm features` first")

    survey_ids = sorted(corpus["survey_id"].unique())
    truth = _load_truth_and_geometry(conn, survey_ids)
    corpus = corpus.merge(truth, on=["survey_id", "sample_idx"], how="left", validate="one_to_one")

    has_grad = "g_mag_nt_per_m" in corpus.columns and corpus["g_mag_nt_per_m"].notna().any()
    feature_cols = feature_columns(cfg.base.features, with_gradiometer=has_grad)

    # One reference run per line builds that line's truth registry -- defect/
    # interference position is fixed across a line's runs (generate.py), so any
    # one run's labels describe the whole line.
    line_ids = sorted(corpus["line_id"].unique())
    registries = []
    for line_id in line_ids:
        ref_survey_id = sorted(corpus.loc[corpus["line_id"] == line_id, "survey_id"].unique())[0]
        ref_rows = corpus[corpus["survey_id"] == ref_survey_id]
        registries.append(build_truth_registry(ref_rows, line_id))
    registry = pd.concat(registries, ignore_index=True)

    run_line_id = corpus.drop_duplicates("survey_id").set_index("survey_id")["line_id"].to_dict()

    split_cfg = cfg.base.model.split
    corpus = add_fold_column(
        corpus, "line_id", "chainage_m", block_m=split_cfg.fallback_block_m, n_folds=split_cfg.n_folds
    )
    corpus, result, severity_frame, classify_frame, defect_calibrator = _evaluate_corpus(
        corpus, feature_cols, cfg, cfg.seed, registry, run_line_id, survey_ids
    )
    mad_threshold = corpus.attrs["mad_threshold"]

    _log_and_persist(
        cfg, conn, corpus, feature_cols, registry, survey_ids, result, mad_threshold,
        severity_frame, classify_frame, defect_calibrator,
    )
    _print_report(result)
    return result


def _fit_final_severity_model(
    severity_frame: pd.DataFrame, feature_cols: list[str], cfg: Config, seed: int
) -> tuple[SeverityModel | None, GlobalMeanSeverityBaseline | None]:
    """The severity model that ships: fit on ALL matched indications, with one
    more train/calibration split (by defect) for its own conformal margin --
    a production bundle needs a calibrated margin too, not just the CV loop's
    honest-but-thrown-away fold models.
    """
    if len(severity_frame) < 4 or severity_frame["matched_source_id"].nunique() < 2:
        return None, None
    rng = np.random.default_rng(seed)
    defects = severity_frame["matched_source_id"].unique()
    rng.shuffle(defects)
    n_calib = max(1, round(len(defects) * CONFORMAL_CALIB_FRACTION)) if len(defects) > 1 else 0
    calib_defects = set(defects[:n_calib])
    train = severity_frame[~severity_frame["matched_source_id"].isin(calib_defects)]
    calib = severity_frame[severity_frame["matched_source_id"].isin(calib_defects)]
    if len(train) == 0:
        train, calib = severity_frame, severity_frame.iloc[0:0]

    sev_cfg = cfg.base.model.severity
    model = SeverityModel(
        feature_cols=feature_cols, quantiles=tuple(sev_cfg["quantiles"]), conformal_alpha=sev_cfg["conformal_alpha"],
        seed=seed, lgbm_cfg=cfg.base.model.lightgbm.model_dump(),
        min_child_samples=sev_cfg.get("min_child_samples", 3),
        n_estimators=sev_cfg.get("n_estimators", 50),
        num_leaves=sev_cfg.get("num_leaves", 7),
    ).fit(train, train["y_true"].to_numpy(), calib, calib["y_true"].to_numpy())
    baseline = GlobalMeanSeverityBaseline(conformal_alpha=sev_cfg["conformal_alpha"]).fit(
        train["y_true"].to_numpy(), calib["y_true"].to_numpy()
    )
    return model, baseline


def _fit_final_classify_model(
    classify_frame: pd.DataFrame, feature_cols: list[str], cfg: Config, seed: int
) -> tuple[ClassifyModel | None, MajorityClassBaseline | None]:
    """The classifier that ships: fit on ALL matched (defect+interference)
    indications, with one more class-stratified train/calibration split (by
    defect) for its own isotonic calibration -- a production bundle needs a
    calibrated model too, not just the CV loop's honest-but-thrown-away fold
    models.
    """
    if len(classify_frame) < 8 or classify_frame["defect_type"].nunique() < 3:
        return None, None
    rng = np.random.default_rng(seed)
    classify_cfg = cfg.base.model.classify
    train_ids, calib_ids = _stratified_defect_calib_split(
        classify_frame, "defect_type", "matched_source_id", CLASSIFY_CALIB_FRACTION, rng
    )
    train = classify_frame[classify_frame["matched_source_id"].isin(train_ids)]
    calib = classify_frame[classify_frame["matched_source_id"].isin(calib_ids)]
    if len(train) == 0:
        train, calib = classify_frame, classify_frame.iloc[0:0]

    model = ClassifyModel(
        feature_cols=feature_cols, classes=CLASSIFY_CLASSES, seed=seed,
        lgbm_cfg={
            **cfg.base.model.lightgbm.model_dump(),
            "n_estimators": classify_cfg.get("n_estimators", 50),
            "num_leaves": classify_cfg.get("num_leaves", 7),
        },
        class_weight=classify_cfg.get("class_weight", "balanced"),
        min_child_samples=classify_cfg.get("min_child_samples", 3),
    ).fit(train, train["defect_type"].to_numpy(), calib, calib["defect_type"].to_numpy())
    baseline = MajorityClassBaseline(classes=CLASSIFY_CLASSES).fit(train["defect_type"].to_numpy())
    return model, baseline


def _log_and_persist(
    cfg: Config,
    conn,
    corpus: pd.DataFrame,
    feature_cols: list[str],
    registry: pd.DataFrame,
    survey_ids: list[str],
    result: dict,
    mad_threshold: float,
    severity_frame: pd.DataFrame,
    classify_frame: pd.DataFrame,
    defect_calibrator: IsotonicRegression | None,
) -> None:
    """MLflow logging + model_run/pipeline_release rows + a real `indication`
    table, populated from models refit on the FULL corpus (the CV-fold models in
    _run_grouped_cv exist only to produce honest out-of-fold evaluation scores;
    what actually gets shipped is trained on everything available).
    """
    now = dt.datetime.now(dt.timezone.utc)
    git_sha = _git_sha()
    content_hashes = [
        row[0]
        for row in conn.execute(
            f"SELECT content_sha256 FROM survey WHERE survey_id IN ({','.join('?' * len(survey_ids))})",
            survey_ids,
        ).fetchall()
    ]
    corpus_data_sha256 = data_sha256(content_hashes)

    mlflow.set_tracking_uri(cfg.env.mlflow["tracking_uri"])
    mlflow.set_experiment(cfg.base.mlflow.get("experiment", "lsm-integrity"))

    with mlflow.start_run(run_name="stage3-anomaly") as run:
        mlflow.log_params(
            {
                "config_sha256": cfg.config_sha256,
                "git_sha": git_sha,
                "data_sha256": corpus_data_sha256,
                "seed": cfg.seed,
                "feature_version": cfg.base.features.version,
                "schema_version": cfg.base.schema_version,
                "n_folds": cfg.base.model.split.n_folds,
                "fallback_block_m": cfg.base.model.split.fallback_block_m,
                "contamination": cfg.base.model.anomaly["contamination"],
                "n_estimators": cfg.base.model.anomaly["n_estimators"],
                "dig_budget_per_km": cfg.base.model.dig_budget_per_km,
                "match_tolerance_m": MATCH_TOLERANCE_M,
                "gate_recall_margin": GATE_RECALL_MARGIN,
            }
        )
        for model_name in ("mad", "isolation_forest"):
            m = result[model_name]
            for metric_name in ("recall_at_budget", "false_dig_rate", "interference_dig_fraction", "localisation_error_m"):
                point, lo, hi = m[metric_name]
                mlflow.log_metric(f"{model_name}_{metric_name}", point)
                mlflow.log_metric(f"{model_name}_{metric_name}_lo", lo)
                mlflow.log_metric(f"{model_name}_{metric_name}_hi", hi)
            mlflow.log_metric(f"{model_name}_pr_auc", m["pr_auc"])
        mlflow.log_metric("recall_gap_if_minus_mad", result["recall_gap_if_minus_mad"][0])
        mlflow.log_metric("recall_gap_lo", result["recall_gap_if_minus_mad"][1])
        mlflow.log_metric("gate_passed", int(result["gate_passed"]))
        mlflow.log_metric("interference_attributable", int(result["interference_attributable"]))
        mlflow.log_metric("n_severity_samples", result["n_severity_samples"])

        if "severity" in result:
            for model_name in ("severity", "severity_baseline"):
                m = result[model_name]
                for metric_name in ("coverage", "mae", "interval_width"):
                    point, lo, hi = m[metric_name]
                    mlflow.log_metric(f"{model_name}_{metric_name}", point)
                    mlflow.log_metric(f"{model_name}_{metric_name}_lo", lo)
                    mlflow.log_metric(f"{model_name}_{metric_name}_hi", hi)
            mlflow.log_metric("severity_gate_passed", int(result["severity_gate_passed"]))
            mlflow.log_dict(result["severity_mae_by_decile"], "severity_mae_by_decile.json")

        mlflow.log_metric("n_classify_samples", result["n_classify_samples"])
        if "classify" in result:
            for model_name in ("classify", "classify_baseline"):
                m = result[model_name]
                for cls, (point, lo, hi) in m["per_class_recall"].items():
                    mlflow.log_metric(f"{model_name}_recall_{cls}", point)
                    mlflow.log_metric(f"{model_name}_recall_{cls}_lo", lo)
                    mlflow.log_metric(f"{model_name}_recall_{cls}_hi", hi)
                for metric_name in ("interference_precision", "brier"):
                    point, lo, hi = m[metric_name]
                    mlflow.log_metric(f"{model_name}_{metric_name}", point)
                    mlflow.log_metric(f"{model_name}_{metric_name}_lo", lo)
                    mlflow.log_metric(f"{model_name}_{metric_name}_hi", hi)
            mlflow.log_metric("classify_recall_gate_passed", int(result["classify_recall_gate_passed"]))
            mlflow.log_dict(result["classify_reliability"], "classify_reliability.json")

        dq_rows = conn.execute(
            f"SELECT survey_id, check_name, status, n_affected FROM dq_report "
            f"WHERE survey_id IN ({','.join('?' * len(survey_ids))})",
            survey_ids,
        ).fetchall()
        mlflow.log_dict(
            {"checks": [dict(zip(("survey_id", "check_name", "status", "n_affected"), r)) for r in dq_rows]},
            "dq_report.json",
        )

        fig, ax = plt.subplots(figsize=(6, 5))
        for model_name, score_col in (("MAD", "score_mad"), ("IsolationForest", "score_if")):
            y_true = corpus["defect"].to_numpy()
            y_score = corpus[score_col].to_numpy()
            if len(np.unique(y_true)) > 1:
                precision, recall, _ = precision_recall_curve(y_true, y_score)
                ax.plot(recall, precision, label=model_name)
        ax.set_xlabel("recall (row-level, diagnostic only)")
        ax.set_ylabel("precision")
        ax.set_title("Stage 3: row-level PR curve (indication-level recall@budget is the gate)")
        ax.legend()
        mlflow.log_figure(fig, "pr_curve.png")
        plt.close(fig)

        # Refit on the FULL corpus -- the model that actually ships.
        mad_final = MADBaseline().fit(corpus)
        mad_final_threshold = calibrated_threshold(mad_final.score(corpus), cfg.base.model.anomaly["contamination"])
        iso_final = IsolationForestAnomalyModel(
            feature_cols=feature_cols,
            contamination=cfg.base.model.anomaly["contamination"],
            n_estimators=cfg.base.model.anomaly["n_estimators"],
            seed=cfg.seed,
            emphasize_features=cfg.base.model.anomaly.get("emphasize_features"),
            emphasis_repeats=cfg.base.model.anomaly.get("emphasis_repeats", 1),
        ).fit(corpus)

        model_dir = Path(cfg.env.storage.model_dir)
        date_tag = now.strftime("%Y.%m.%d")
        short_sha = _short_sha(git_sha)
        mad_version = f"anomaly-mad-{date_tag}-{short_sha}"
        if_version = f"anomaly-if-{date_tag}-{short_sha}"
        truth_as_of = now.isoformat()
        feature_summary = training_feature_summary(corpus, feature_cols, seed=cfg.seed)
        prediction_reference = training_prediction_reference(
            result["indications_per_km"], result["p_defect_cal_sample"], seed=cfg.seed,
        )

        for model_version, model_obj, model_kind, threshold in (
            (mad_version, mad_final, "mad", mad_final_threshold),
            (if_version, iso_final, "isolation_forest", 0.0),
        ):
            artifact_path = model_dir / model_version / "bundle.joblib"
            save_bundle(
                artifact_path,
                {
                    "task": "anomaly",
                    "model_kind": model_kind,
                    "model": model_obj,
                    "feature_cols": feature_cols,
                    "threshold": threshold,
                    "defect_type_categories": DEFECT_TYPES,
                    "feature_version": cfg.base.features.version,
                    "schema_version": cfg.base.schema_version,
                    "config_sha256": cfg.config_sha256,
                    "git_sha": git_sha,
                    "data_sha256": corpus_data_sha256,
                    "truth_as_of": truth_as_of,
                    "training_feature_summary": feature_summary,
                    "training_prediction_reference": prediction_reference,
                },
            )
            metrics_for_model = result["mad" if model_kind == "mad" else "isolation_forest"]
            conn.execute(
                "INSERT OR REPLACE INTO model_run (model_version, task, mlflow_run_id, git_sha, "
                "config_sha256, data_sha256, feature_version, truth_as_of, trained_at, "
                "metrics_json, artifact_uri, final_test_uses) VALUES (?,?,?,?,?,?,?,?,?,?,?,0)",
                (
                    model_version, "anomaly", run.info.run_id, git_sha, cfg.config_sha256,
                    corpus_data_sha256, cfg.base.features.version, truth_as_of, now.isoformat(),
                    json.dumps(metrics_for_model, default=list), str(artifact_path),
                ),
            )
        conn.commit()

        # Stage 4: fit and persist the final severity model, if there was
        # enough matched, severity-labelled data to do so at all.
        severity_version = None
        sev_feature_cols = [*feature_cols, "extent_m"]
        sev_final, sev_baseline_final = _fit_final_severity_model(severity_frame, sev_feature_cols, cfg, cfg.seed)
        if sev_final is not None:
            severity_version = f"severity-lgbm-{date_tag}-{short_sha}"
            baseline_version = f"severity-mean-{date_tag}-{short_sha}"
            sev_artifact_path = model_dir / severity_version / "bundle.joblib"
            save_bundle(
                sev_artifact_path,
                {
                    "task": "severity",
                    "model_kind": "lightgbm_cqr",
                    "model": sev_final,
                    "feature_cols": sev_feature_cols,
                    "defect_type_categories": DEFECT_TYPES,
                    "feature_version": cfg.base.features.version,
                    "schema_version": cfg.base.schema_version,
                    "config_sha256": cfg.config_sha256,
                    "git_sha": git_sha,
                    "data_sha256": corpus_data_sha256,
                    "truth_as_of": truth_as_of,
                    "training_feature_summary": feature_summary,
                    "training_prediction_reference": prediction_reference,
                },
            )
            conn.execute(
                "INSERT OR REPLACE INTO model_run (model_version, task, mlflow_run_id, git_sha, "
                "config_sha256, data_sha256, feature_version, truth_as_of, trained_at, "
                "metrics_json, artifact_uri, final_test_uses) VALUES (?,?,?,?,?,?,?,?,?,?,?,0)",
                (
                    severity_version, "severity", run.info.run_id, git_sha, cfg.config_sha256,
                    corpus_data_sha256, cfg.base.features.version, truth_as_of, now.isoformat(),
                    json.dumps(result.get("severity", {}), default=list), str(sev_artifact_path),
                ),
            )
            # baseline logged for comparison/audit only -- not referenced by any release.
            conn.execute(
                "INSERT OR REPLACE INTO model_run (model_version, task, mlflow_run_id, git_sha, "
                "config_sha256, data_sha256, feature_version, truth_as_of, trained_at, "
                "metrics_json, artifact_uri, final_test_uses) VALUES (?,?,?,?,?,?,?,?,?,?,?,0)",
                (
                    baseline_version, "severity", run.info.run_id, git_sha, cfg.config_sha256,
                    corpus_data_sha256, cfg.base.features.version, truth_as_of, now.isoformat(),
                    json.dumps(result.get("severity_baseline", {}), default=list), "n/a (baseline, no artifact)",
                ),
            )
        conn.commit()

        # Stage 5: fit and persist the final classifier + the p_defect_cal
        # calibrator, if there was enough matched, multi-class data to do so.
        classify_version = None
        classify_feature_cols = [*feature_cols, "extent_m"]
        classify_final, classify_baseline_final = _fit_final_classify_model(
            classify_frame, classify_feature_cols, cfg, cfg.seed
        )
        if classify_final is not None:
            # Physics-consistency check: the classifier should key on residual
            # amplitude/gradient/peak width, not absolute position -- run on
            # the FINAL model, not a CV fold's, since this is the gate on what
            # actually ships.
            shap_importance = _lightgbm_shap_importance(classify_final, classify_frame)
            shap_check = shap_denylist_check(shap_importance, CLASSIFY_DENYLIST, top_k=10)
            result["classify_shap_check"] = shap_check
            result["classify_gate_passed"] = (
                result.get("classify_recall_gate_passed", False) and shap_check["passed"]
            )
            mlflow.log_metric("classify_shap_check_passed", int(shap_check["passed"]))
            mlflow.log_dict(shap_check, "classify_shap_denylist_check.json")

            fig, ax = plt.subplots(figsize=(6, 5))
            ranked = shap_importance.sort_values(ascending=False).head(15)
            ax.barh(ranked.index[::-1], ranked.to_numpy()[::-1])
            ax.set_xlabel("mean |contribution| (LightGBM native SHAP, all classes)")
            ax.set_title("Stage 5: global feature importance")
            fig.tight_layout()
            mlflow.log_figure(fig, "classify_shap_importance.png")
            plt.close(fig)

            pred_type_full, _ = classify_final.predict(classify_frame)
            mlflow.log_dict(
                _explain_indications_sample(classify_final, classify_frame, pred_type_full),
                "classify_shap_examples.json",
            )

            classify_version = f"classify-lgbm-{date_tag}-{short_sha}"
            classify_baseline_version = f"classify-majority-{date_tag}-{short_sha}"
            classify_artifact_path = model_dir / classify_version / "bundle.joblib"
            save_bundle(
                classify_artifact_path,
                {
                    "task": "classify",
                    "model_kind": "lightgbm_multiclass",
                    "model": classify_final,
                    "defect_calibrator": defect_calibrator,
                    "feature_cols": classify_feature_cols,
                    "classify_classes": CLASSIFY_CLASSES,
                    "defect_type_categories": DEFECT_TYPES,
                    "feature_version": cfg.base.features.version,
                    "schema_version": cfg.base.schema_version,
                    "config_sha256": cfg.config_sha256,
                    "git_sha": git_sha,
                    "data_sha256": corpus_data_sha256,
                    "truth_as_of": truth_as_of,
                    "training_feature_summary": feature_summary,
                    "training_prediction_reference": prediction_reference,
                },
            )
            conn.execute(
                "INSERT OR REPLACE INTO model_run (model_version, task, mlflow_run_id, git_sha, "
                "config_sha256, data_sha256, feature_version, truth_as_of, trained_at, "
                "metrics_json, artifact_uri, final_test_uses) VALUES (?,?,?,?,?,?,?,?,?,?,?,0)",
                (
                    classify_version, "classify", run.info.run_id, git_sha, cfg.config_sha256,
                    corpus_data_sha256, cfg.base.features.version, truth_as_of, now.isoformat(),
                    json.dumps(result.get("classify", {}), default=list), str(classify_artifact_path),
                ),
            )
            # baseline logged for comparison/audit only -- not referenced by any release.
            conn.execute(
                "INSERT OR REPLACE INTO model_run (model_version, task, mlflow_run_id, git_sha, "
                "config_sha256, data_sha256, feature_version, truth_as_of, trained_at, "
                "metrics_json, artifact_uri, final_test_uses) VALUES (?,?,?,?,?,?,?,?,?,?,?,0)",
                (
                    classify_baseline_version, "classify", run.info.run_id, git_sha, cfg.config_sha256,
                    corpus_data_sha256, cfg.base.features.version, truth_as_of, now.isoformat(),
                    json.dumps(result.get("classify_baseline", {}), default=list), "n/a (baseline, no artifact)",
                ),
            )
        conn.commit()

        pipeline_version = f"{date_tag}-{short_sha}"
        conn.execute(
            "INSERT OR REPLACE INTO pipeline_release (pipeline_version, anomaly_version, "
            "severity_version, classify_version, growth_version, feature_version, "
            "schema_version, container_digest, released_at, alias) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                pipeline_version, if_version, severity_version, classify_version, None, cfg.base.features.version,
                cfg.base.schema_version, "local-dev", now.isoformat(), "challenger",
            ),
        )
        conn.commit()

        model_card_path = model_dir / pipeline_version / "model_card.md"
        write_model_card(
            model_card_path,
            provenance={
                "pipeline_version": pipeline_version, "config_sha256": cfg.config_sha256,
                "git_sha": git_sha, "data_sha256": corpus_data_sha256,
                "feature_version": cfg.base.features.version, "schema_version": cfg.base.schema_version,
                "n_defects": result["n_defects"], "n_surveys": result["n_surveys"],
                "consequence_proxy": cfg.base.model.classify.get("consequence_proxy", {}),
            },
            result=result,
        )
        mlflow.log_artifact(str(model_card_path))

        # Score with the SHIPPED model (iso_final, fit on the full corpus), not
        # the CV loop's out-of-fold `score_if` column -- those exist purely for
        # an honest generalisation estimate (each fold's model saw only 4/5 of
        # the data) and would silently disagree with what predict.py reloads
        # from the bundle and reproduces on the exact same rows.
        corpus["_final_score"] = iso_final.score(corpus)
        all_indications = []
        for survey_id, survey_rows in corpus.groupby("survey_id"):
            survey_rows = survey_rows.sort_values("sample_idx").rename(columns={"_final_score": "_score"})
            indications = cluster_indications(
                survey_rows, "_score", 0.0, survey_id=str(survey_id), pipeline_version=pipeline_version
            )
            all_indications.append(indications)
        indications_df = pd.concat(all_indications, ignore_index=True) if all_indications else pd.DataFrame()
        n_written = write_indications(conn, indications_df)
        mlflow.log_dict({"indications": indications_df.to_dict(orient="records")}, "indications.json")
        mlflow.log_metric("n_indications_written", n_written)

    log.info(
        "training complete",
        extra={
            "pipeline_version": pipeline_version,
            "model_version": if_version,
            "severity_version": severity_version,
            "gate_passed": result["gate_passed"],
        },
    )


def _print_report(result: dict) -> None:
    def fmt(name: str, triple: tuple[float, float, float]) -> str:
        point, lo, hi = triple
        return f"  {name:28s} {point:6.3f}  [{lo:.3f}, {hi:.3f}]"

    print("\n=== Stage 3: MAD baseline vs IsolationForest (grouped CV, out-of-fold) ===")
    for model_name, label in (("mad", "MAD baseline"), ("isolation_forest", "IsolationForest")):
        m = result[model_name]
        print(f"\n{label}:")
        print(fmt("recall @ dig budget", m["recall_at_budget"]))
        print(fmt("false-dig rate", m["false_dig_rate"]))
        print(fmt("  of which: interference", m["interference_dig_fraction"]))
        print(fmt("localisation error (m)", m["localisation_error_m"]))
        print(f"  {'PR-AUC (diagnostic)':28s} {m['pr_auc']:6.3f}")

    point, lo, hi = result["recall_gap_if_minus_mad"]
    print(f"\nRecall gap (IsolationForest - MAD): {point:.3f}  [{lo:.3f}, {hi:.3f}]")
    verdict = "PASSES" if result["gate_passed"] else "DOES NOT PASS"
    print(f"Stage 3 gate (>= {GATE_RECALL_MARGIN:.2f} recall gap, at the CI lower bound): {verdict}")

    ipoint, ilo, ihi = result["interference_gap_mad_minus_if"]
    print(f"\nInterference-dig-fraction gap (MAD - IsolationForest): {ipoint:.3f}  [{ilo:.3f}, {ihi:.3f}]")
    if result["interference_attributable"]:
        print("-> the recall gap IS attributable to interference rejection: MAD wastes "
              "more of its dig budget on interference than IsolationForest does.")
    else:
        print("-> the recall gap is NOT clearly attributable to interference rejection at this "
              "confidence level -- reporting this honestly, as the gate requires, rather than "
              "claiming a mechanism the data doesn't support.")
    print(f"\n({result['n_defects']} physical defects, {result['n_surveys']} surveys evaluated)")

    print("\n=== Stage 4: severity -- LightGBM quantile + split conformal vs global-mean baseline ===")
    if "severity" not in result:
        print(f"  Skipped: only {result['n_severity_samples']} matched, severity-labelled indications "
              "-- too few (or too few distinct defects) to fit and calibrate at all.")
    else:
        for model_name, label in (("severity_baseline", "Global-mean baseline"), ("severity", "LightGBM CQR")):
            m = result[model_name]
            print(f"\n{label}:")
            print(fmt("coverage @ 90% nominal", m["coverage"]))
            print(fmt("MAE", m["mae"]))
            print(fmt("mean interval width", m["interval_width"]))

        verdict = "PASSES" if result["severity_gate_passed"] else "DOES NOT PASS"
        print(f"\nStage 4 gate (coverage in [{GATE_SEVERITY_COVERAGE_RANGE[0]}, "
              f"{GATE_SEVERITY_COVERAGE_RANGE[1]}]): {verdict}")
        print(f"MAE by severity decile: {result['severity_mae_by_decile']}")
        print(f"(n={result['n_severity_samples']} matched, severity-labelled indications)")

    print("\n=== Stage 5: classification -- LightGBM multiclass + isotonic calibration vs majority-class baseline ===")
    if "classify" not in result:
        print(f"  Skipped: only {result['n_classify_samples']} matched, classifiable indications "
              "-- too few (or too few distinct classes) to fit and calibrate at all.")
        return

    for model_name, label in (("classify_baseline", "Majority-class baseline"), ("classify", "LightGBM multiclass")):
        m = result[model_name]
        print(f"\n{label}:")
        for cls, triple in m["per_class_recall"].items():
            print(fmt(f"  recall: {cls}", triple))
        print(fmt("interference precision", m["interference_precision"]))
        print(fmt("Brier score", m["brier"]))

    scc_point, scc_lo, _ = result["classify"]["per_class_recall"].get("scc", (float("nan"),) * 3)
    verdict = "PASSES" if result["classify_recall_gate_passed"] else "DOES NOT PASS"
    print(f"\nSCC recall: {scc_point:.3f} [CI lower bound {scc_lo:.3f}]")
    print(f"Stage 5 recall gate (SCC recall >= {GATE_SCC_RECALL:.2f} at the CI lower bound): {verdict}")

    if "classify_shap_check" in result:
        shap_verdict = "PASSES" if result["classify_shap_check"]["passed"] else "DOES NOT PASS"
        print(f"Stage 5 physics-consistency gate (no {CLASSIFY_DENYLIST} feature in top-10 "
              f"by contribution): {shap_verdict}")
        if not result["classify_shap_check"]["passed"]:
            print(f"  Leaked features: {result['classify_shap_check']['leaked_denylist_features']}")
        overall_verdict = "PASSES" if result.get("classify_gate_passed") else "DOES NOT PASS"
        print(f"Stage 5 gate overall: {overall_verdict}")
    print(f"(n={result['n_classify_samples']} matched, classifiable indications)")
