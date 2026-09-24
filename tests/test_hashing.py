from __future__ import annotations

import numpy as np
import pandas as pd

from lsm.hashing import content_sha256, data_sha256, file_sha256


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_idx": [0, 1, 2],
            "lat": [46.958, 46.9581, 46.9582],
            "lon": [8.365, 8.3651, 8.3652],
            "bx_nt": [19000.1, 19000.2, 19000.3],
            "by_nt": [1000.1, 1000.2, 1000.3],
            "bz_nt": [45000.1, 45000.2, 45000.3],
            "bx2_nt": [np.nan, np.nan, np.nan],
            "by2_nt": [np.nan, np.nan, np.nan],
            "bz2_nt": [np.nan, np.nan, np.nan],
            "defect": [0, 0, 1],
            "defect_type": ["none", "none", "scc"],
            "severity_smys": [np.nan, np.nan, 42.0],
        }
    )


def test_content_hash_is_column_order_invariant():
    df = _sample_df()
    reordered = df[df.columns[::-1]]
    assert content_sha256(df) == content_sha256(reordered)


def test_content_hash_is_row_order_invariant():
    df = _sample_df()
    shuffled = df.sample(frac=1, random_state=1).reset_index(drop=True)
    assert content_sha256(df) == content_sha256(shuffled)


def test_content_hash_changes_with_real_data_change():
    df = _sample_df()
    changed = df.copy()
    changed.loc[0, "bx_nt"] = 99999.0
    assert content_sha256(df) != content_sha256(changed)


def test_content_hash_ignores_sub_precision_float_noise():
    """Canonicalisation rounds to CANONICAL_FLOAT_DECIMALS -- re-exporting the same
    data through a different float formatter must not change the content hash.
    """
    df = _sample_df()
    noisy = df.copy()
    noisy["bx_nt"] = noisy["bx_nt"] + 1e-9
    assert content_sha256(df) == content_sha256(noisy)


def test_file_hash_differs_from_content_hash_semantics(tmp_path):
    """Two files with identical content but different on-disk byte formatting
    (here: parquet vs csv-then-reparsed) have different file hashes but the same
    content hash -- the two-hash design is exercised, not just asserted.
    """
    df = _sample_df()
    p1 = tmp_path / "a.parquet"
    p2 = tmp_path / "b.parquet"
    df.to_parquet(p1, index=False)
    df[df.columns[::-1]].to_parquet(
        p2, index=False
    )  # same content, different column order

    assert file_sha256(p1) != file_sha256(p2)
    assert content_sha256(pd.read_parquet(p1)) == content_sha256(pd.read_parquet(p2))


def test_data_sha256_is_order_independent_over_members():
    hashes = ["c" * 64, "a" * 64, "b" * 64]
    assert data_sha256(hashes) == data_sha256(list(reversed(hashes)))


def test_data_sha256_changes_with_membership():
    a = data_sha256(["a" * 64, "b" * 64])
    b = data_sha256(["a" * 64, "b" * 64, "c" * 64])
    assert a != b
