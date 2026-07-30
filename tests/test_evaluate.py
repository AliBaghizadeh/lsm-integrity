"""Stage 3: evaluate.py -- matching, indication-level metrics, bootstrap CI."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lsm.evaluate import (
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

REGISTRY = pd.DataFrame(
    {
        "source_id": ["LINE000_defect_00", "LINE000_defect_01", "LINE000_interference_00"],
        "line_id": ["LINE000"] * 3,
        "kind": ["defect", "defect", "interference"],
        "defect_type": ["weld", "scc", "interference"],
        "chainage_m": [100.0, 500.0, 800.0],
        "chainage_start_m": [98.0, 498.0, 795.0],
        "chainage_end_m": [102.0, 502.0, 805.0],
    }
)


def _dug(peaks: list[float], scores: list[float] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "indication_id": [f"i{i}" for i in range(len(peaks))],
            "chainage_peak_m": peaks,
            "anomaly_score": scores if scores is not None else [1.0] * len(peaks),
        }
    )


def test_match_dug_indications_matches_within_tolerance():
    dug = _dug([101.0, 800.5, 950.0])  # hits defect_00, interference_00, nothing
    out = match_dug_indications(dug, REGISTRY, tolerance_m=5.0)

    assert list(out["matched_kind"]) == ["defect", "interference", None]
    assert out.iloc[0]["matched_source_id"] == "LINE000_defect_00"
    assert out.iloc[0]["distance_m"] == 1.0


def test_match_dug_indications_does_not_double_claim_a_source():
    """Two dug indications both near the same defect -- only the closer one
    should get the true-positive match; the other is correctly a false dig.
    """
    dug = _dug([100.5, 101.5])
    out = match_dug_indications(dug, REGISTRY, tolerance_m=5.0)

    matched = out[out["matched_kind"] == "defect"]
    assert len(matched) == 1
    assert matched.iloc[0]["chainage_peak_m"] == 100.5  # the closer one wins


def test_false_dig_rate_and_interference_fraction():
    dug = _dug([101.0, 800.5, 950.0])
    matched = match_dug_indications(dug, REGISTRY, tolerance_m=5.0)

    assert false_dig_rate_per_run(matched) == 2 / 3  # interference + unmatched
    assert interference_dig_fraction_per_run(matched) == 1 / 3


def test_false_dig_rate_on_empty_dug_is_zero():
    empty = _dug([])
    matched = match_dug_indications(empty, REGISTRY, tolerance_m=5.0)
    assert false_dig_rate_per_run(matched) == 0.0
    assert interference_dig_fraction_per_run(matched) == 0.0


def test_localisation_errors_only_counts_defect_matches():
    dug = _dug([101.0, 800.5, 950.0])
    matched = match_dug_indications(dug, REGISTRY, tolerance_m=5.0)
    errors = localisation_errors_m(matched)
    assert list(errors) == [1.0]


def test_defect_hit_rates_averages_across_runs():
    dug_run0 = _dug([101.0])  # finds defect_00 only
    dug_run1 = _dug([101.0, 501.0])  # finds both defects
    dug_run2 = _dug([])  # finds neither

    matched_by_run = {
        0: match_dug_indications(dug_run0, REGISTRY, tolerance_m=5.0),
        1: match_dug_indications(dug_run1, REGISTRY, tolerance_m=5.0),
        2: match_dug_indications(dug_run2, REGISTRY, tolerance_m=5.0),
    }
    rates = defect_hit_rates(matched_by_run, REGISTRY)

    assert rates["LINE000_defect_00"] == 2 / 3
    assert rates["LINE000_defect_01"] == 1 / 3


def test_pr_auc_perfect_separation_is_one():
    y_true = np.array([0, 0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
    assert pr_auc(y_true, y_score) == 1.0


def test_pr_auc_with_a_single_class_is_nan():
    y_true = np.zeros(5)
    y_score = np.arange(5.0)
    assert np.isnan(pr_auc(y_true, y_score))


def test_bootstrap_ci_point_estimate_matches_mean_and_ci_contains_it():
    units = [0.5, 0.6, 0.7, 0.4, 0.55]
    point, lo, hi = bootstrap_ci(units, n_resamples=2000, level=0.95, seed=42)
    assert point == np.mean(units)
    assert lo <= point <= hi


def test_bootstrap_ci_is_wider_with_fewer_units():
    rng = np.random.default_rng(0)
    many = rng.normal(0.7, 0.1, size=200)
    few = many[:5]

    _, lo_many, hi_many = bootstrap_ci(many, n_resamples=2000, level=0.95, seed=1)
    _, lo_few, hi_few = bootstrap_ci(few, n_resamples=2000, level=0.95, seed=1)

    assert (hi_few - lo_few) > (hi_many - lo_many)


def test_bootstrap_ci_on_empty_units_is_nan():
    point, lo, hi = bootstrap_ci([], n_resamples=100, level=0.95, seed=0)
    assert np.isnan(point) and np.isnan(lo) and np.isnan(hi)


def test_defect_hit_rates_with_multiple_lines_divides_by_the_defects_own_line_runs():
    """A second line's runs must not dilute the first line's defect denominator --
    the latent bug this test pins against.
    """
    registry = pd.DataFrame(
        {
            "source_id": ["LINE000_defect_00", "LINE001_defect_00"],
            "line_id": ["LINE000", "LINE001"],
            "kind": ["defect", "defect"],
            "defect_type": ["weld", "weld"],
            "chainage_m": [100.0, 100.0],
            "chainage_start_m": [98.0, 98.0],
            "chainage_end_m": [102.0, 102.0],
        }
    )
    line0_reg = registry[registry["line_id"] == "LINE000"]
    line1_reg = registry[registry["line_id"] == "LINE001"]

    matched_by_run = {
        "LINE000_R0": match_dug_indications(_dug([101.0]), line0_reg, tolerance_m=5.0),
        "LINE001_R0": match_dug_indications(_dug([]), line1_reg, tolerance_m=5.0),
        "LINE001_R1": match_dug_indications(_dug([]), line1_reg, tolerance_m=5.0),
    }
    run_line_id = {"LINE000_R0": "LINE000", "LINE001_R0": "LINE001", "LINE001_R1": "LINE001"}

    rates = defect_hit_rates(matched_by_run, registry, run_line_id=run_line_id)

    # LINE000's defect was found in its only run -> rate 1.0, not diluted by
    # LINE001's two (unrelated) runs.
    assert rates["LINE000_defect_00"] == 1.0
    assert rates["LINE001_defect_00"] == 0.0


def test_paired_bootstrap_ci_matches_mean_difference_and_ci_contains_it():
    a = [1.0, 1.0, 0.0, 1.0]
    b = [0.0, 0.0, 0.0, 1.0]
    point, lo, hi = paired_bootstrap_ci(a, b, n_resamples=2000, level=0.95, seed=1)
    assert point == np.mean(a) - np.mean(b)
    assert lo <= point <= hi


def test_paired_bootstrap_ci_is_tighter_than_two_separate_cis_when_correlated():
    """a and b move together (correlated per-defect performance) -- the paired
    difference should have less spread than what you'd get treating them as
    independent, since resampling preserves the pairing.
    """
    rng = np.random.default_rng(0)
    base = rng.normal(0.6, 0.2, size=30)
    a = np.clip(base + rng.normal(0, 0.02, size=30), 0, 1)
    b = np.clip(base - 0.15 + rng.normal(0, 0.02, size=30), 0, 1)

    _, lo_paired, hi_paired = paired_bootstrap_ci(a, b, n_resamples=2000, level=0.95, seed=2)
    _, lo_a, hi_a = bootstrap_ci(a, n_resamples=2000, level=0.95, seed=2)
    _, lo_b, hi_b = bootstrap_ci(b, n_resamples=2000, level=0.95, seed=3)

    naive_width = (hi_a - lo_a) + (hi_b - lo_b)
    assert (hi_paired - lo_paired) < naive_width


def test_paired_bootstrap_ci_rejects_mismatched_lengths():
    try:
        paired_bootstrap_ci([1.0, 2.0], [1.0], n_resamples=10, level=0.95, seed=0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_mae_by_severity_decile_separates_bins():
    y_true = np.array([10.0, 10.0, 90.0, 90.0])
    y_pred = np.array([12.0, 8.0, 80.0, 100.0])  # abs errors: 2, 2, 10, 10
    result = mae_by_severity_decile(y_true, y_pred, n_bins=2)
    assert len(result) == 2
    assert sorted(result.to_numpy()) == [2.0, 10.0]


def test_mae_by_severity_decile_collapses_on_a_single_unique_value():
    y_true = np.array([50.0, 50.0, 50.0])
    y_pred = np.array([48.0, 52.0, 50.0])
    result = mae_by_severity_decile(y_true, y_pred, n_bins=10)
    assert len(result) == 1
    assert result.iloc[0] == pytest.approx((2.0 + 2.0 + 0.0) / 3.0)


def test_per_group_severity_metrics_computes_coverage_mae_width_per_group():
    df = pd.DataFrame(
        {
            "defect_id": ["d0", "d0", "d1"],
            "y_true": [50.0, 55.0, 20.0],
            "y_pred": [50.0, 50.0, 30.0],
            "lo": [40.0, 40.0, 10.0],
            "hi": [60.0, 60.0, 15.0],  # d1's interval does NOT cover y_true=20
        }
    )
    metrics = per_group_severity_metrics(df, "defect_id")
    assert len(metrics["coverage"]) == 2  # one entry per group (d0, d1)
    assert set(metrics["coverage"]) == {1.0, 0.0}
    assert sorted(metrics["mae"]) == [2.5, 10.0]  # d0: (|50-50|+|55-50|)/2=2.5; d1: |20-30|=10
    assert sorted(metrics["interval_width"]) == [5.0, 20.0]  # d0: hi-lo=20 both rows; d1: hi-lo=5
