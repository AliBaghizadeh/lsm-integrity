"""Stage 6: scale_eval.py -- DuckDB read-path equivalence + whole-line-holdout
CV / temporal-holdout smoke tests. All on tiny/small fixtures -- never the
real ~9.6M-row scale config (that's `scripts/stage6_scale_rehearsal.py`'s job,
run once, by hand, outside pytest).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from lsm import scale_eval, train
from lsm.db import connect
from lsm.evaluate import add_fold_column_by_line
from lsm.features import load_feature_corpus
from lsm.generate import generate_all
from lsm.pipeline import run_feature_pipeline, run_survey_pipeline


def _now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _scale_eval_sized_cfg(cfg):
    """3 independent lines x 3 runs -- enough lines to exercise whole-line
    holdout (a fold must be able to hold out at least one whole line) and
    enough runs to exercise the temporal (run 0-1 vs run 2) split, while
    staying small enough to run in well under a second.
    """
    cfg.base.data.length_m = 500.0
    cfg.base.data.step_m = 1.0
    cfg.base.data.n_lines = 3
    cfg.base.data.n_runs = 3
    cfg.base.data.n_defects = 10
    cfg.base.data.n_interference = 3
    return cfg


def _build_corpus(cfg):
    results = generate_all(cfg.base.data, cfg.env.storage.raw_dir, seed=cfg.seed)
    conn = connect(cfg.env.storage.sqlite_path)
    for sr in results:
        _status, report = run_survey_pipeline(conn, sr, cfg)
        assert not report.has_fail
    for sr in results:
        outcome, _ = run_feature_pipeline(conn, sr.survey_id, cfg)
        assert outcome == "computed"
    return conn


def test_load_feature_corpus_duckdb_matches_pandas_version(cfg):
    cfg = _scale_eval_sized_cfg(cfg)
    _build_corpus(cfg)
    as_of = _now_iso()

    pandas_corpus = load_feature_corpus(
        cfg.env.storage.feature_dir, cfg.base.features.version, as_of=as_of
    )
    duckdb_corpus = scale_eval.load_feature_corpus_duckdb(
        cfg.env.storage.feature_dir, cfg.base.features.version, as_of=as_of
    )

    assert len(pandas_corpus) == len(duckdb_corpus)
    common_cols = sorted(set(pandas_corpus.columns) & set(duckdb_corpus.columns))
    p = (
        pandas_corpus[common_cols]
        .sort_values(["survey_id", "sample_idx"])
        .reset_index(drop=True)
    )
    d = (
        duckdb_corpus[common_cols]
        .sort_values(["survey_id", "sample_idx"])
        .reset_index(drop=True)
    )
    for col in common_cols:
        if p[col].dtype.kind in "fc":
            assert np.allclose(
                p[col].to_numpy(dtype=float),
                d[col].to_numpy(dtype=float),
                equal_nan=True,
            )
        else:
            assert (p[col].to_numpy() == d[col].to_numpy()).all()


def test_load_feature_corpus_duckdb_respects_line_ids_filter(cfg):
    cfg = _scale_eval_sized_cfg(cfg)
    _build_corpus(cfg)
    as_of = _now_iso()

    duckdb_corpus = scale_eval.load_feature_corpus_duckdb(
        cfg.env.storage.feature_dir,
        cfg.base.features.version,
        as_of=as_of,
        line_ids=["LINE000"],
    )
    assert set(duckdb_corpus["line_id"].unique()) == {"LINE000"}


def test_load_truth_and_geometry_duckdb_matches_sqlite_pandas_version(cfg):
    cfg = _scale_eval_sized_cfg(cfg)
    conn = _build_corpus(cfg)
    survey_ids = ["LINE000_R0", "LINE001_R0"]

    pandas_truth = train._load_truth_and_geometry(conn, survey_ids)
    duckdb_truth = scale_eval.load_truth_and_geometry_duckdb(conn, survey_ids)

    assert len(pandas_truth) == len(duckdb_truth)
    common_cols = sorted(set(pandas_truth.columns) & set(duckdb_truth.columns))
    p = (
        pandas_truth[common_cols]
        .sort_values(["survey_id", "sample_idx"])
        .reset_index(drop=True)
    )
    d = (
        duckdb_truth[common_cols]
        .sort_values(["survey_id", "sample_idx"])
        .reset_index(drop=True)
    )
    for col in common_cols:
        if p[col].dtype.kind in "fc":
            assert np.allclose(
                p[col].to_numpy(dtype=float),
                d[col].to_numpy(dtype=float),
                equal_nan=True,
            )
        else:
            assert (p[col].to_numpy() == d[col].to_numpy()).all()


def test_run_scale_evaluation_smoke(cfg):
    cfg = _scale_eval_sized_cfg(cfg)
    conn = _build_corpus(cfg)

    result = scale_eval.run_scale_evaluation(cfg, conn)

    assert set(result.keys()) == {"whole_line_cv", "temporal_holdout"}
    for key in ("whole_line_cv", "temporal_holdout"):
        r = result[key]
        assert "mad" in r and "isolation_forest" in r
        assert isinstance(r["gate_passed"], (bool, np.bool_))
        assert "n_defects" in r and "n_surveys" in r
        # severity/classify are optional (may legitimately be skipped on a
        # small fixture) -- if present, must have the right shape.
        if "severity" in r:
            assert "coverage" in r["severity"]
        if "classify" in r:
            assert "per_class_recall" in r["classify"]


def test_run_whole_line_cv_never_splits_a_line_across_folds(cfg):
    cfg = _scale_eval_sized_cfg(cfg)
    _build_corpus(cfg)

    corpus = load_feature_corpus(
        cfg.env.storage.feature_dir, cfg.base.features.version, as_of=_now_iso()
    )

    with_fold = add_fold_column_by_line(corpus, "line_id", n_folds=3)
    assert with_fold.groupby("line_id")["fold"].nunique().max() == 1


def test_run_temporal_holdout_smoke(cfg):
    cfg = _scale_eval_sized_cfg(cfg)
    conn = _build_corpus(cfg)

    feature_version = cfg.base.features.version
    as_of = _now_iso()
    corpus = load_feature_corpus(
        cfg.env.storage.feature_dir, feature_version, as_of=as_of
    )
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

    result = scale_eval.run_temporal_holdout(
        corpus, feature_cols, cfg, cfg.seed, registry, run_line_id
    )

    assert "mad" in result and "isolation_forest" in result
    assert result["n_surveys"] == 3  # one run (run_id==2) across 3 lines
