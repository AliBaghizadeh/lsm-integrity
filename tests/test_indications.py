"""Stage 3: indications.py -- row scores -> indication objects."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import lsm.indications as indications_module
from lsm.indications import (
    attach_classification,
    attach_indication_features,
    attach_severity,
    cluster_indications,
    select_dig_budget,
)


def _survey_frame(n=200, step_m=1.0):
    sample_idx = np.arange(n)
    return pd.DataFrame(
        {
            "sample_idx": sample_idx,
            "chainage_m": sample_idx * step_m,
            "lat": 46.0 + sample_idx * 1e-6,
            "lon": 8.0 + sample_idx * 1e-6,
            "dq_flag": "clean",
            "score": 0.0,
        }
    )


def test_cluster_indications_finds_one_contiguous_run():
    df = _survey_frame()
    df.loc[95:105, "score"] = 5.0
    df.loc[100, "score"] = 9.0  # the peak

    out = cluster_indications(df, "score", threshold=1.0, survey_id="S1", pipeline_version="P1")

    assert len(out) == 1
    row = out.iloc[0]
    assert row["chainage_peak_m"] == 100.0
    assert row["chainage_start_m"] == 95.0
    assert row["chainage_end_m"] == 105.0
    assert row["anomaly_score"] == 9.0


def test_cluster_indications_separates_two_runs():
    df = _survey_frame()
    df.loc[20:25, "score"] = 5.0
    df.loc[150:155, "score"] = 6.0

    out = cluster_indications(df, "score", threshold=1.0, survey_id="S1", pipeline_version="P1")

    assert len(out) == 2
    assert sorted(out["chainage_peak_m"]) != [None]


def test_no_rows_above_threshold_gives_no_indications():
    df = _survey_frame()
    out = cluster_indications(df, "score", threshold=1.0, survey_id="S1", pipeline_version="P1")
    assert len(out) == 0
    assert list(out.columns) == indications_module.INDICATION_COLUMNS


def test_dq_flag_propagates_edge_if_any_row_in_the_run_is_edge():
    df = _survey_frame()
    df.loc[95:105, "score"] = 5.0
    df.loc[105, "dq_flag"] = "edge"  # one edge row inside the run

    out = cluster_indications(df, "score", threshold=1.0, survey_id="S1", pipeline_version="P1")
    assert out.iloc[0]["dq_flag"] == "edge"


def test_indication_id_is_deterministic_given_same_inputs():
    df = _survey_frame()
    df.loc[95:105, "score"] = 5.0

    out1 = cluster_indications(df, "score", threshold=1.0, survey_id="S1", pipeline_version="P1")
    out2 = cluster_indications(df, "score", threshold=1.0, survey_id="S1", pipeline_version="P1")
    assert out1.iloc[0]["indication_id"] == out2.iloc[0]["indication_id"]

    out3 = cluster_indications(df, "score", threshold=1.0, survey_id="S1", pipeline_version="P2")
    assert out3.iloc[0]["indication_id"] != out1.iloc[0]["indication_id"]


def test_p_defect_cal_is_a_percentile_rank_against_the_full_survey():
    df = _survey_frame()
    df.loc[95:105, "score"] = 5.0
    out = cluster_indications(df, "score", threshold=1.0, survey_id="S1", pipeline_version="P1")
    # The peak (5.0) beats every one of the other 189 background rows (score=0.0).
    assert out.iloc[0]["p_defect_cal"] > 0.9


def test_attach_indication_features_joins_the_peak_rows_own_values():
    df = _survey_frame()
    df["survey_id"] = "S1"
    df["feat_a"] = np.arange(len(df), dtype=float)
    df.loc[95:105, "score"] = 5.0
    df.loc[100, "score"] = 9.0  # the peak, chainage_m=100 -> feat_a=100

    indications = cluster_indications(df, "score", threshold=1.0, survey_id="S1", pipeline_version="P1")
    out = attach_indication_features(indications, df, feature_cols=["feat_a"])

    assert out.iloc[0]["feat_a"] == 100.0
    assert out.iloc[0]["extent_m"] == 10.0  # 105 - 95


def test_attach_indication_features_on_empty_indications():
    df = _survey_frame()
    df["survey_id"] = "S1"
    empty = cluster_indications(df, "score", threshold=1.0, survey_id="S1", pipeline_version="P1")
    out = attach_indication_features(empty, df, feature_cols=["score"])
    assert len(out) == 0


def test_select_dig_budget_ranks_by_score_and_respects_budget():
    indications = pd.DataFrame(
        {
            "indication_id": ["a", "b", "c", "d"],
            "anomaly_score": [1.0, 5.0, 3.0, 2.0],
        }
    )
    dug = select_dig_budget(indications, survey_length_m=1000.0, digs_per_km=2)
    assert len(dug) == 2
    assert list(dug["indication_id"]) == ["b", "c"]


def test_select_dig_budget_on_empty_indications_returns_empty():
    empty = pd.DataFrame(columns=["indication_id", "anomaly_score"])
    out = select_dig_budget(empty, survey_length_m=1000.0, digs_per_km=5)
    assert len(out) == 0


class _FakeSeverityModel:
    """Deterministic stand-in: predicts feat_a itself as the median, +-1 as the interval."""

    feature_cols = ["feat_a"]

    def predict(self, X):
        med = X["feat_a"].to_numpy(dtype=float)
        return med, med - 1.0, med + 1.0


def test_attach_severity_fills_in_the_severity_columns():
    df = _survey_frame()
    df["survey_id"] = "S1"
    df["feat_a"] = np.arange(len(df), dtype=float)
    df.loc[95:105, "score"] = 5.0
    df.loc[100, "score"] = 9.0  # peak at chainage_m=100 -> feat_a=100

    indications = cluster_indications(df, "score", threshold=1.0, survey_id="S1", pipeline_version="P1")
    out = attach_severity(indications, df, _FakeSeverityModel(), base_feature_cols=["feat_a"], nominal_coverage=0.9)

    assert out.iloc[0]["sev_pred"] == 100.0
    assert out.iloc[0]["sev_lo"] == 99.0
    assert out.iloc[0]["sev_hi"] == 101.0
    assert out.iloc[0]["interval_nominal"] == 0.9


def test_attach_severity_on_empty_indications():
    df = _survey_frame()
    df["survey_id"] = "S1"
    df["feat_a"] = 0.0
    empty = cluster_indications(df, "score", threshold=1.0, survey_id="S1", pipeline_version="P1")
    out = attach_severity(empty, df, _FakeSeverityModel(), base_feature_cols=["feat_a"], nominal_coverage=0.9)
    assert len(out) == 0


class _FakeClassifyModel:
    """Deterministic stand-in: always predicts 'scc' at confidence 0.8."""

    def predict(self, X):
        n = len(X)
        return np.array(["scc"] * n), np.full(n, 0.8)


class _FakeDefectCalibrator:
    """Deterministic stand-in: P(defect) = anomaly_score / 10, unclipped."""

    def predict(self, scores):
        return np.asarray(scores, dtype=float) / 10.0


_CONSEQUENCE_PROXY = {"scc": 1.0, "weld": 0.4, "dent": 0.5, "corrosion": 0.6, "interference": 0.0}


def test_attach_classification_fills_pred_type_and_overwrites_p_defect_cal():
    df = _survey_frame()
    df["survey_id"] = "S1"
    df["feat_a"] = np.arange(len(df), dtype=float)
    df.loc[95:105, "score"] = 5.0
    df.loc[100, "score"] = 9.0  # peak, anomaly_score=9.0

    indications = cluster_indications(df, "score", threshold=1.0, survey_id="S1", pipeline_version="P1")
    out = attach_classification(
        indications, df, _FakeClassifyModel(), _FakeDefectCalibrator(),
        base_feature_cols=["feat_a"], consequence_proxy=_CONSEQUENCE_PROXY,
    )

    assert out.iloc[0]["pred_type"] == "scc"
    assert out.iloc[0]["pred_type_conf"] == 0.8
    # overwritten from cluster_indications' percentile-rank stand-in (which
    # would have been > 0.9 here, see test_p_defect_cal_is_a_percentile_rank_
    # against_the_full_survey) to the calibrator's own value: 9.0 / 10 = 0.9.
    assert out.iloc[0]["p_defect_cal"] == pytest.approx(0.9)


def test_attach_classification_risk_score_formula():
    df = _survey_frame()
    df["survey_id"] = "S1"
    df["feat_a"] = np.arange(len(df), dtype=float)
    df.loc[95:105, "score"] = 5.0
    df.loc[100, "score"] = 9.0

    indications = cluster_indications(df, "score", threshold=1.0, survey_id="S1", pipeline_version="P1")
    indications = attach_severity(
        indications, df, _FakeSeverityModel(), base_feature_cols=["feat_a"], nominal_coverage=0.9
    )
    out = attach_classification(
        indications, df, _FakeClassifyModel(), _FakeDefectCalibrator(),
        base_feature_cols=["feat_a"], consequence_proxy=_CONSEQUENCE_PROXY,
    )

    # sev_pred = feat_a = 100.0 (peak's own feature value, see _FakeSeverityModel);
    # p_defect_cal = anomaly_score/10 = 0.9; consequence_proxy["scc"] = 1.0.
    expected = 0.9 * 100.0 * 1.0
    assert out.iloc[0]["risk_score"] == pytest.approx(expected)


def test_attach_classification_risk_score_is_none_when_sev_pred_is_none():
    df = _survey_frame()
    df["survey_id"] = "S1"
    df["feat_a"] = np.arange(len(df), dtype=float)
    df.loc[95:105, "score"] = 5.0
    df.loc[100, "score"] = 9.0

    # no attach_severity call -- sev_pred stays whatever cluster_indications
    # initialised it to (None), never a fabricated number.
    indications = cluster_indications(df, "score", threshold=1.0, survey_id="S1", pipeline_version="P1")
    assert indications.iloc[0]["sev_pred"] is None

    out = attach_classification(
        indications, df, _FakeClassifyModel(), _FakeDefectCalibrator(),
        base_feature_cols=["feat_a"], consequence_proxy=_CONSEQUENCE_PROXY,
    )
    assert pd.isna(out.iloc[0]["risk_score"])


def test_attach_classification_on_empty_indications():
    df = _survey_frame()
    df["survey_id"] = "S1"
    df["feat_a"] = 0.0
    empty = cluster_indications(df, "score", threshold=1.0, survey_id="S1", pipeline_version="P1")
    out = attach_classification(
        empty, df, _FakeClassifyModel(), _FakeDefectCalibrator(),
        base_feature_cols=["feat_a"], consequence_proxy=_CONSEQUENCE_PROXY,
    )
    assert len(out) == 0
