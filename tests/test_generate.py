from __future__ import annotations

import numpy as np
import pytest

from lsm.generate import _build_features, generate_all
from lsm.truth import build_truth_registry


def test_generate_is_deterministic_given_same_seed(tiny_cfg, tmp_path):
    r1 = generate_all(tiny_cfg.base.data, tmp_path / "raw1", seed=42)
    r2 = generate_all(tiny_cfg.base.data, tmp_path / "raw2", seed=42)
    assert [r.content_sha256 for r in r1] == [r.content_sha256 for r in r2]


def test_generate_sample_idx_is_dense_integer_key(tiny_cfg, tmp_path):
    import pandas as pd

    results = generate_all(tiny_cfg.base.data, tmp_path / "raw", seed=42)
    df = pd.read_parquet(results[0].path)
    assert df["sample_idx"].dtype.kind in "iu"
    assert list(df["sample_idx"]) == list(range(len(df)))
    # chainage_m is derived, never a key
    assert (df["chainage_m"] == df["sample_idx"] * tiny_cfg.base.data.step_m).all()


def test_generate_writes_one_parquet_per_survey(tiny_cfg, tmp_path):
    results = generate_all(tiny_cfg.base.data, tmp_path / "raw", seed=42)
    n_expected = tiny_cfg.base.data.n_lines * tiny_cfg.base.data.n_runs
    assert len(results) == n_expected
    for r in results:
        assert r.path.exists()
        assert r.path.name == "survey.parquet"


def test_defect_signature_is_detectable_against_background(cfg, tmp_path):
    """Cross-check against the physics claim in PLAN.md: defect residual ~25 nT
    against a ~4 nT background floor, i.e. learnable but small relative to the
    ~45000 nT raw field.
    """
    import pandas as pd

    cfg.base.data.n_lines = 1
    cfg.base.data.n_runs = 1
    results = generate_all(cfg.base.data, tmp_path / "raw", seed=42)
    df = pd.read_parquet(results[0].path)

    defect_rows = df[df["defect"] == 1]
    assert len(defect_rows) > 0

    # crude "residual" proxy: deviation from a rolling median (no real detrend yet)
    med = df["bx_nt"].rolling(101, center=True, min_periods=1).median()
    resid = (df["bx_nt"] - med).abs()
    defect_resid = resid[df["defect"] == 1].median()
    background_resid = resid[df["defect"] == 0].median()

    assert defect_resid > background_resid
    # raw field is ~19000-45000 nT; defect signal must be a small fraction of it
    assert defect_resid < 0.01 * df["bz_nt"].abs().median()


def _label_windows(cfg, features):
    """[(start, end, is_defect)] label windows for a features list, matching
    generate.py's own half-width formula."""
    windows = []
    for f in features:
        r_eff = float(np.hypot(cfg.depth_m, f["y_off_m"]))
        hw = cfg.label_window_scale * r_eff
        windows.append((f["chainage_m"] - hw, f["chainage_m"] + hw, f["is_defect"]))
    return sorted(windows)


def test_build_features_never_places_overlapping_label_windows(cfg):
    """Unconstrained rng.uniform() placement silently corrupted the truth
    registry at higher defect density: two overlapping same-kind windows
    merge into ONE contiguous region, undercounting physical defects. This is
    the regression test for _sample_spaced_chainage.
    """
    cfg.base.data.n_defects = 60
    cfg.base.data.n_interference = 16
    rng = np.random.default_rng(cfg.seed)
    features = _build_features(cfg.base.data, rng)

    windows = _label_windows(cfg.base.data, features)
    for (_, hi, _), (lo2, _, _) in zip(windows, windows[1:]):
        assert hi <= lo2, "two label windows overlap"


def test_build_features_defect_count_survives_into_the_truth_registry(cfg, tmp_path):
    """End-to-end proof the spacing fix does its job: the truth registry finds
    exactly as many physical defects as were requested, not fewer.
    """
    from lsm.generate import generate_all

    cfg.base.data.n_lines = 1
    cfg.base.data.n_runs = 1
    cfg.base.data.n_defects = 60
    cfg.base.data.n_interference = 16
    results = generate_all(cfg.base.data, tmp_path / "raw", seed=cfg.seed)

    import pandas as pd

    df = pd.read_parquet(results[0].path)
    registry = build_truth_registry(df, line_id="LINE000")
    assert (registry["kind"] == "defect").sum() == 60
    assert (registry["kind"] == "interference").sum() == 16


def test_build_features_raises_loudly_when_too_dense_to_place(cfg):
    """Too many features for the line length must fail loudly, not silently
    corrupt spacing by giving up the constraint.
    """
    cfg.base.data.n_defects = 80
    cfg.base.data.n_interference = 20
    rng = np.random.default_rng(cfg.seed)
    with pytest.raises(RuntimeError):
        _build_features(cfg.base.data, rng)
