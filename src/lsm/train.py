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
from sklearn.metrics import precision_recall_curve

from lsm.bundle import save_bundle, training_feature_summary
from lsm.config import Config
from lsm.evaluate import (
    add_fold_column,
    bootstrap_ci,
    defect_hit_rates,
    false_dig_rate_per_run,
    interference_dig_fraction_per_run,
    localisation_errors_m,
    mae_by_severity_decile,
    match_dug_indications,
    paired_bootstrap_ci,
    per_group_severity_metrics,
    pr_auc,
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
from lsm.models.severity import GlobalMeanSeverityBaseline, SeverityModel
from lsm.schemas import DEFECT_TYPES
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


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        sha = out.stdout.strip()
        return sha if out.returncode == 0 and sha else "uncommitted"
    except Exception:
        return "uncommitted"


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
    """
    split_cfg = cfg.base.model.split
    corpus = add_fold_column(
        corpus, "line_id", "chainage_m", block_m=split_cfg.fallback_block_m, n_folds=split_cfg.n_folds
    )
    corpus["score_mad"] = np.nan
    corpus["score_if"] = np.nan
    mad_thresholds = []

    for k in range(split_cfg.n_folds):
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


def _build_severity_training_frame(
    corpus: pd.DataFrame, feature_cols: list[str], registry: pd.DataFrame
) -> pd.DataFrame:
    """Every indication (clustered from IsolationForest's out-of-fold scores at
    its self-calibrated threshold=0.0) that matches a real defect, with its
    feature vector, true severity, and CV fold attached -- the full severity
    training/evaluation universe.
    """
    indications = _cluster_all_indications(corpus, "score_if", 0.0)
    if len(indications) == 0:
        return indications

    matched_frames = []
    for survey_id, group in indications.groupby("survey_id"):
        line_id = corpus.loc[corpus["survey_id"] == survey_id, "line_id"].iloc[0]
        line_registry = registry[registry["line_id"] == line_id]
        matched_frames.append(match_dug_indications(group, line_registry, MATCH_TOLERANCE_M))
    matched = pd.concat(matched_frames, ignore_index=True)

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

    corpus = _run_grouped_cv(corpus, feature_cols, cfg, seed=cfg.seed)
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
        cfg.seed + 10,
    )
    interference_gap = paired_bootstrap_ci(
        interference_mad, interference_if, boot_cfg.n_resamples, boot_cfg.level, cfg.seed + 11
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

    # Stage 4: severity, conditional on the anomaly detector having flagged
    # SOMETHING that matches a real defect -- see _build_severity_training_frame.
    severity_frame = _build_severity_training_frame(corpus, feature_cols, registry)
    result["n_severity_samples"] = len(severity_frame)
    if len(severity_frame) >= 4 and severity_frame["matched_source_id"].nunique() >= 2:
        oof_model, oof_baseline = _run_severity_cv(
            severity_frame, [*feature_cols, "extent_m"], cfg, seed=cfg.seed
        )
        if len(oof_model) > 0:
            sev_metrics = _severity_bootstrap_metrics(oof_model, cfg, seed=cfg.seed + 20)
            base_metrics = _severity_bootstrap_metrics(oof_baseline, cfg, seed=cfg.seed + 30)
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

    _log_and_persist(cfg, conn, corpus, feature_cols, registry, survey_ids, result, mad_threshold, severity_frame)
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
    ).fit(train, train["y_true"].to_numpy(), calib, calib["y_true"].to_numpy())
    baseline = GlobalMeanSeverityBaseline(conformal_alpha=sev_cfg["conformal_alpha"]).fit(
        train["y_true"].to_numpy(), calib["y_true"].to_numpy()
    )
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

        dq_rows = conn.execute(
            f"SELECT survey_id, check_name, status, n_affected FROM dq_report "
            f"WHERE survey_id IN ({','.join('?' * len(survey_ids))})",
            survey_ids,
        ).fetchall()
        mlflow.log_dict(
            [dict(zip(("survey_id", "check_name", "status", "n_affected"), r)) for r in dq_rows],
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
        ).fit(corpus)

        model_dir = Path(cfg.env.storage.model_dir)
        date_tag = now.strftime("%Y.%m.%d")
        mad_version = f"anomaly-mad-{date_tag}-{git_sha[:8]}"
        if_version = f"anomaly-if-{date_tag}-{git_sha[:8]}"
        truth_as_of = now.isoformat()
        feature_summary = training_feature_summary(corpus, feature_cols)

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
            severity_version = f"severity-lgbm-{date_tag}-{git_sha[:8]}"
            baseline_version = f"severity-mean-{date_tag}-{git_sha[:8]}"
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

        pipeline_version = f"{date_tag}-{git_sha[:8]}"
        conn.execute(
            "INSERT OR REPLACE INTO pipeline_release (pipeline_version, anomaly_version, "
            "severity_version, classify_version, growth_version, feature_version, "
            "schema_version, container_digest, released_at, alias) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                pipeline_version, if_version, severity_version, None, None, cfg.base.features.version,
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
        mlflow.log_dict(indications_df.to_dict(orient="records"), "indications.json")
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
        return

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
