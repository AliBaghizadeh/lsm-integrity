from __future__ import annotations

from lsm.generate import generate_all


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
    import numpy as np
    import pandas as pd

    cfg.base.data.n_lines = 1
    cfg.base.data.n_runs = 1
    results = generate_all(cfg.base.data, tmp_path / "raw", seed=42)
    df = pd.read_parquet(results[0].path)

    defect_rows = df[df["defect"] == 1]
    background_rows = df[df["defect"] == 0]
    assert len(defect_rows) > 0

    # crude "residual" proxy: deviation from a rolling median (no real detrend yet)
    med = df["bx_nt"].rolling(101, center=True, min_periods=1).median()
    resid = (df["bx_nt"] - med).abs()
    defect_resid = resid[df["defect"] == 1].median()
    background_resid = resid[df["defect"] == 0].median()

    assert defect_resid > background_resid
    # raw field is ~19000-45000 nT; defect signal must be a small fraction of it
    assert defect_resid < 0.01 * df["bz_nt"].abs().median()
