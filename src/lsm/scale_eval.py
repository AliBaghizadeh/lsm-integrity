"""
Stage 6: scale-rehearsal-only evaluation paths -- whole-line-holdout CV, a
temporal holdout (train on early runs, test on the latest), and DuckDB-backed
alternatives to the pandas-concat bulk-read paths in `features.py`/`train.py`.

This module is Stage-6-only. It deliberately imports `train.py`'s
underscore-prefixed CV/fit helpers (the same way `tests/test_train.py`
already does) rather than duplicating them -- it is never imported by
`run_train()`, `pipeline.py`, or the CLI's default path. `run_train()`'s
default 5-fold block-CV path (the real gate) is exercised separately, by
`scripts/stage6_scale_rehearsal.py` calling `lsm.train.run_train(cfg, conn)`
directly, unchanged.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from lsm import train
from lsm.config import Config
from lsm.evaluate import add_fold_column_by_line
from lsm.features import FEATURE_KEY_COLUMNS, assert_point_in_time, read_feature_meta
from lsm.models.anomaly import (
    IsolationForestAnomalyModel,
    MADBaseline,
    calibrated_threshold,
)


def load_feature_corpus_duckdb(
    feature_dir: str | Path,
    feature_version: int,
    as_of: str,
    line_ids: list[str] | None = None,
) -> pd.DataFrame:
    """DuckDB-backed alternative to `features.load_feature_corpus`: one glob
    read over every survey's `features.parquet` (fast, columnar) instead of a
    per-file Python loop + `pd.concat`. `line_id`/`run_id` are already real
    columns inside the parquet payload (`schemas.FEATURE_KEY_COLUMNS`), so no
    per-file join is needed for those; `surveyed_at` lives only in each
    survey's `meta.json` sidecar (never in the parquet payload), so it is
    stitched on via a small metadata-only scan, then `as_of`/`line_ids`
    filtered in pandas exactly as the pandas version does -- same empty-input
    contract, drop-in comparable.
    """
    import duckdb

    root = Path(feature_dir) / f"fv={feature_version}"
    glob_pattern = str(root / "line_id=*" / "run_id=*" / "features.parquet").replace(
        "\\", "/"
    )

    meta_rows = []
    for meta_path in sorted(root.glob("line_id=*/run_id=*/meta.json")):
        meta = read_feature_meta(meta_path.parent)
        if meta is not None:
            meta_rows.append(
                {
                    "line_id": meta["line_id"],
                    "run_id": meta["run_id"],
                    "surveyed_at": meta["surveyed_at"],
                }
            )

    if not meta_rows:
        return pd.DataFrame(columns=FEATURE_KEY_COLUMNS + ["surveyed_at"])

    corpus = duckdb.sql(f"SELECT * FROM read_parquet('{glob_pattern}')").df()
    if len(corpus) == 0:
        return pd.DataFrame(columns=FEATURE_KEY_COLUMNS + ["surveyed_at"])

    meta_df = pd.DataFrame(meta_rows)
    corpus = corpus.merge(
        meta_df, on=["line_id", "run_id"], how="left", validate="many_to_one"
    )
    if line_ids is not None:
        corpus = corpus[corpus["line_id"].isin(line_ids)]
    corpus = corpus[corpus["surveyed_at"].astype(str) <= str(as_of)]
    corpus = corpus.reset_index(drop=True)
    assert_point_in_time(
        corpus, as_of, what=f"feature corpus fv={feature_version} (duckdb)"
    )
    return corpus


def load_truth_and_geometry_duckdb(
    conn: sqlite3.Connection, survey_ids: list[str]
) -> pd.DataFrame:
    """DuckDB-backed alternative to `train._load_truth_and_geometry`: same
    `(survey_id, source_uri)` lookup from SQLite `survey` (cheap, unchanged),
    then ONE DuckDB call reading every survey's raw Parquet at once
    (`read_parquet(<uris>, filename=true)`), instead of a Python loop of
    `pd.read_parquet` + `pd.concat`. The `filename` column DuckDB adds when
    given a list of paths is mapped back to `survey_id` via the same lookup.
    """
    import duckdb

    placeholders = ",".join("?" * len(survey_ids))
    rows = conn.execute(
        f"SELECT survey_id, source_uri FROM survey WHERE survey_id IN ({placeholders})",
        survey_ids,
    ).fetchall()
    if not rows:
        return pd.DataFrame(
            columns=[
                "sample_idx",
                "lat",
                "lon",
                "defect",
                "defect_type",
                "interference",
                "severity_smys",
                "survey_id",
            ]
        )

    uri_to_survey = {uri: sid for sid, uri in rows}
    uris = list(uri_to_survey)
    corpus = duckdb.sql(
        "SELECT sample_idx, lat, lon, defect, defect_type, interference, severity_smys, filename "
        "FROM read_parquet(?, filename=true)",
        params=[uris],
    ).df()
    corpus["survey_id"] = corpus["filename"].map(uri_to_survey)
    return corpus.drop(columns="filename").reset_index(drop=True)


def run_whole_line_cv(
    corpus: pd.DataFrame,
    feature_cols: list[str],
    cfg: Config,
    seed: int,
    registry: pd.DataFrame,
    run_line_id: dict[str, str],
    survey_ids: list[str],
    n_folds: int = 5,
) -> dict:
    """Stage 3-5 evaluation with an entire physical line held out per fold
    (PLAN.md Stage 6: "the real generalisation test"), instead of today's
    default (line, 100 m block) hash grouping, which scatters one line's
    blocks across ~all folds. Reuses `train._evaluate_corpus` verbatim --
    same metrics, same gates, same `result` shape `train._print_report`
    already knows how to render -- with only the fold assignment differing.
    Default `n_folds=5` (not one-fold-per-line): at Stage 6 scale (40 lines)
    this still gives each held-out fold ~8 lines' worth of defects, a real
    statistical-power improvement over the demo corpus's per-fold defect
    counts.
    """
    corpus = add_fold_column_by_line(corpus, "line_id", n_folds=n_folds)
    _corpus, result, _severity_frame, _classify_frame, _defect_calibrator = (
        train._evaluate_corpus(
            corpus, feature_cols, cfg, seed, registry, run_line_id, survey_ids
        )
    )
    return result


def run_temporal_holdout(
    corpus: pd.DataFrame,
    feature_cols: list[str],
    cfg: Config,
    seed: int,
    registry: pd.DataFrame,
    run_line_id: dict[str, str],
    train_run_ids: frozenset[int] = frozenset({0, 1}),
    test_run_id: int = 2,
) -> dict:
    """A single fixed split -- train on early runs, test on the latest -- not
    a k-fold loop (PLAN.md Stage 6: "train surveys 0-1, test survey 2").
    `run_id` is a purely sequential per-line growth index (NOT reflected in
    `surveyed_at`, a different, already-correctly-used point-in-time concept
    elsewhere in this project), so the split filters `run_id` directly.
    Reuses `MADBaseline`/`IsolationForestAnomalyModel` (fit on train only) and
    `train._fit_final_severity_model`/`_fit_final_classify_model` (already
    exist precisely for "fit on all of this frame") -- no new fitting logic.
    Returns the same `result`-dict shape `run_train`/`run_whole_line_cv`
    produce, so `train._print_report` is reusable here too.
    """
    train_corpus = corpus[corpus["run_id"].isin(train_run_ids)].copy()
    test_corpus = corpus[corpus["run_id"] == test_run_id].copy()

    anomaly_cfg = cfg.base.model.anomaly
    mad = MADBaseline(residual_col="r_mid_nt").fit(train_corpus)
    iso = IsolationForestAnomalyModel(
        feature_cols=feature_cols,
        contamination=anomaly_cfg["contamination"],
        n_estimators=anomaly_cfg["n_estimators"],
        seed=seed,
        emphasize_features=anomaly_cfg.get("emphasize_features"),
        emphasis_repeats=anomaly_cfg.get("emphasis_repeats", 1),
    ).fit(train_corpus)
    mad_threshold = calibrated_threshold(
        mad.score(train_corpus), anomaly_cfg["contamination"]
    )

    for c in (train_corpus, test_corpus):
        c["score_mad"] = mad.score(c)
        c["score_if"] = iso.score(c)
        # _build_severity_training_frame/_build_classify_training_frame merge
        # on corpus["fold"] -- unused for a single fixed split, but the column
        # must exist for that merge to find a match.
        c["fold"] = 0

    test_survey_ids = sorted(test_corpus["survey_id"].unique())
    matched_mad = train._dig_and_match(
        test_corpus, "score_mad", mad_threshold, registry, cfg
    )
    matched_if = train._dig_and_match(test_corpus, "score_if", 0.0, registry, cfg)
    metrics_mad, hit_rates_mad, _false_digs_mad, _interference_mad = (
        train._bootstrap_metrics(matched_mad, registry, run_line_id, cfg)
    )
    metrics_if, hit_rates_if, _false_digs_if, _interference_if = (
        train._bootstrap_metrics(matched_if, registry, run_line_id, cfg)
    )
    pr_auc_mad = train.pr_auc(
        test_corpus["defect"].to_numpy(), test_corpus["score_mad"].to_numpy()
    )
    pr_auc_if = train.pr_auc(
        test_corpus["defect"].to_numpy(), test_corpus["score_if"].to_numpy()
    )

    boot_cfg = cfg.base.model.bootstrap
    defect_ids = sorted(hit_rates_mad)
    recall_gap = train.paired_bootstrap_ci(
        [hit_rates_if[d] for d in defect_ids],
        [hit_rates_mad[d] for d in defect_ids],
        boot_cfg.n_resamples,
        boot_cfg.level,
        seed + 10,
    )

    result: dict = {
        # localisation_error_cm is Rig-v2's addition (the ~1 cm dig-marking
        # requirement's own unit) and `train._print_report` requires it. It was
        # added to train.py's own result dicts but not here, so any attempt to
        # print a scale_eval result died with KeyError -- AFTER both holdout CVs
        # had already run to completion. `train._bootstrap_metrics` (called
        # above) has always produced it; this was only ever a pass-through gap.
        "mad": {
            "recall_at_budget": metrics_mad["recall_at_budget"],
            "false_dig_rate": metrics_mad["false_dig_rate"],
            "interference_dig_fraction": metrics_mad["interference_dig_fraction"],
            "localisation_error_m": metrics_mad["localisation_error_m"],
            "localisation_error_cm": metrics_mad["localisation_error_cm"],
            "pr_auc": pr_auc_mad,
        },
        "isolation_forest": {
            "recall_at_budget": metrics_if["recall_at_budget"],
            "false_dig_rate": metrics_if["false_dig_rate"],
            "interference_dig_fraction": metrics_if["interference_dig_fraction"],
            "localisation_error_m": metrics_if["localisation_error_m"],
            "localisation_error_cm": metrics_if["localisation_error_cm"],
            "pr_auc": pr_auc_if,
        },
        "recall_gap_if_minus_mad": recall_gap,
        "interference_gap_mad_minus_if": (float("nan"),) * 3,
        "gate_passed": recall_gap[1] >= train.GATE_RECALL_MARGIN,
        "interference_attributable": False,
        "n_defects": len(defect_ids),
        "n_surveys": len(test_survey_ids),
    }

    all_indications_train = train._cluster_all_indications(
        train_corpus, "score_if", 0.0
    )
    matched_all_train = train._match_all_indications(
        all_indications_train, train_corpus, registry
    )
    all_indications_test = train._cluster_all_indications(test_corpus, "score_if", 0.0)
    matched_all_test = train._match_all_indications(
        all_indications_test, test_corpus, registry
    )

    severity_frame_train = train._build_severity_training_frame(
        matched_all_train, train_corpus, feature_cols
    )
    severity_frame_test = train._build_severity_training_frame(
        matched_all_test, test_corpus, feature_cols
    )
    result["n_severity_samples"] = len(severity_frame_test)
    if (
        len(severity_frame_train) >= 4
        and severity_frame_train["matched_source_id"].nunique() >= 2
        and len(severity_frame_test) > 0
    ):
        sev_feature_cols = [*feature_cols, "extent_m"]
        sev_model, sev_baseline = train._fit_final_severity_model(
            severity_frame_train, sev_feature_cols, cfg, seed
        )
        if sev_model is not None and sev_baseline is not None:
            med, lo, hi = sev_model.predict(severity_frame_test)
            bmed, blo, bhi = sev_baseline.predict(len(severity_frame_test))
            oof_model = pd.DataFrame(
                {
                    "defect_id": severity_frame_test["matched_source_id"].to_numpy(),
                    "y_true": severity_frame_test["y_true"].to_numpy(),
                    "y_pred": med,
                    "lo": lo,
                    "hi": hi,
                }
            )
            oof_baseline = pd.DataFrame(
                {
                    "defect_id": severity_frame_test["matched_source_id"].to_numpy(),
                    "y_true": severity_frame_test["y_true"].to_numpy(),
                    "y_pred": bmed,
                    "lo": blo,
                    "hi": bhi,
                }
            )
            sev_metrics = train._severity_bootstrap_metrics(oof_model, cfg, seed + 20)
            base_metrics = train._severity_bootstrap_metrics(
                oof_baseline, cfg, seed + 30
            )
            result["severity"] = sev_metrics
            result["severity_baseline"] = base_metrics
            result["severity_gate_passed"] = (
                train.GATE_SEVERITY_COVERAGE_RANGE[0]
                <= sev_metrics["coverage"][0]
                <= train.GATE_SEVERITY_COVERAGE_RANGE[1]
            )
            result["severity_mae_by_decile"] = {
                str(k): v
                for k, v in train.mae_by_severity_decile(
                    oof_model["y_true"].to_numpy(), oof_model["y_pred"].to_numpy()
                )
                .to_dict()
                .items()
            }

    classify_frame_train = train._build_classify_training_frame(
        matched_all_train, train_corpus, feature_cols, registry
    )
    classify_frame_test = train._build_classify_training_frame(
        matched_all_test, test_corpus, feature_cols, registry
    )
    result["n_classify_samples"] = len(classify_frame_test)
    if (
        len(classify_frame_train) >= 8
        and classify_frame_train["defect_type"].nunique() >= 3
        and len(classify_frame_test) > 0
    ):
        classify_feature_cols = [*feature_cols, "extent_m"]
        classify_model, classify_baseline = train._fit_final_classify_model(
            classify_frame_train, classify_feature_cols, cfg, seed
        )
        if classify_model is not None and classify_baseline is not None:
            proba = classify_model.predict_proba(classify_frame_test)
            pred_type, pred_conf = classify_model.predict(classify_frame_test)
            base_proba = classify_baseline.predict_proba(len(classify_frame_test))
            base_pred_type, base_pred_conf = classify_baseline.predict(
                len(classify_frame_test)
            )

            oof_classify = pd.DataFrame(
                {
                    "defect_id": classify_frame_test["matched_source_id"].to_numpy(),
                    "true_class": classify_frame_test["defect_type"].to_numpy(),
                    "pred_type": pred_type,
                    "pred_conf": pred_conf,
                }
            )
            for c in train.CLASSIFY_CLASSES:
                oof_classify[c] = proba[c].to_numpy()
            oof_classify_baseline = pd.DataFrame(
                {
                    "defect_id": classify_frame_test["matched_source_id"].to_numpy(),
                    "true_class": classify_frame_test["defect_type"].to_numpy(),
                    "pred_type": base_pred_type,
                    "pred_conf": base_pred_conf,
                }
            )
            for c in train.CLASSIFY_CLASSES:
                oof_classify_baseline[c] = base_proba[c].to_numpy()

            classify_metrics = train._classify_bootstrap_metrics(
                oof_classify, cfg, seed + 40
            )
            classify_baseline_metrics = train._classify_bootstrap_metrics(
                oof_classify_baseline, cfg, seed + 50
            )
            scc_recall_ci = classify_metrics["per_class_recall"].get(
                "scc", (float("nan"),) * 3
            )
            result["classify"] = classify_metrics
            result["classify_baseline"] = classify_baseline_metrics
            result["classify_recall_gate_passed"] = (
                scc_recall_ci[1] >= train.GATE_SCC_RECALL
            )

    return result


def run_scale_evaluation(cfg: Config, conn: sqlite3.Connection) -> dict:
    """Loads the corpus/truth/registry (a small, ~15-line duplication of
    `run_train`'s own preamble, accepted deliberately rather than extracting
    yet another shared helper out of train.py for this one call site) and
    runs both new Stage 6 evaluations. Returns {"whole_line_cv": ...,
    "temporal_holdout": ...}. Does NOT call `run_train()`/`_log_and_persist`
    -- the default 5-fold block-CV path and its MLflow/model_run persistence
    are exercised separately, unchanged.
    """
    feature_version = cfg.base.features.version
    as_of = pd.Timestamp.now(tz="UTC").isoformat()

    corpus = train.load_feature_corpus(
        cfg.env.storage.feature_dir, feature_version, as_of=as_of
    )
    if len(corpus) == 0:
        raise RuntimeError("empty feature corpus -- run `lsm features` first")

    survey_ids = sorted(corpus["survey_id"].unique())
    truth = train._load_truth_and_geometry(conn, survey_ids)
    corpus = corpus.merge(
        truth, on=["survey_id", "sample_idx"], how="left", validate="one_to_one"
    )

    feature_cols = train.feature_columns(cfg.base.features)

    line_ids = sorted(corpus["line_id"].unique())
    registries = []
    for line_id in line_ids:
        ref_survey_id = min(
            corpus.loc[corpus["line_id"] == line_id, "survey_id"].unique()
        )
        ref_rows = corpus[corpus["survey_id"] == ref_survey_id]
        registries.append(
            train.build_truth_registry(
                ref_rows, line_id, ref_rows["chainage_m"].to_numpy()
            )
        )
    registry = pd.concat(registries, ignore_index=True)
    run_line_id = (
        corpus.drop_duplicates("survey_id").set_index("survey_id")["line_id"].to_dict()
    )

    whole_line_result = run_whole_line_cv(
        corpus.copy(), feature_cols, cfg, cfg.seed, registry, run_line_id, survey_ids
    )
    temporal_result = run_temporal_holdout(
        corpus.copy(), feature_cols, cfg, cfg.seed, registry, run_line_id
    )
    return {"whole_line_cv": whole_line_result, "temporal_holdout": temporal_result}
