"""Stage 6: ingest.py::load_readings -- correctness of the vectorized bulk-
insert rewrite (replaced the old itertuples()+float()-per-cell loop for
throughput; see docs/stage6-scale-rehearsal.md for the measured before/after).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lsm.db import connect
from lsm.ingest import load_readings


def _naive_load_readings(conn, survey_id, df):
    """The pre-Stage-6 reference implementation, kept ONLY here as an
    equivalence oracle -- never restored to ingest.py itself.
    """
    conn.executemany(
        """
        INSERT INTO reading (survey_id, sample_idx, lat, lon, bx_nt, by_nt, bz_nt,
                              bx2_nt, by2_nt, bz2_nt)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                survey_id,
                int(r.sample_idx),
                float(r.lat),
                float(r.lon),
                float(r.bx_nt),
                float(r.by_nt),
                float(r.bz_nt),
                None if pd.isna(r.bx2_nt) else float(r.bx2_nt),
                None if pd.isna(r.by2_nt) else float(r.by2_nt),
                None if pd.isna(r.bz2_nt) else float(r.bz2_nt),
            )
            for r in df.itertuples(index=False)
        ],
    )
    conn.commit()


def _make_frame(n=20, seed=0, with_gradiometer=True):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "sample_idx": np.arange(n),
        "lat": 46.0 + rng.normal(size=n) * 1e-5,
        "lon": 8.0 + rng.normal(size=n) * 1e-5,
        "bx_nt": rng.normal(19000.0, 5.0, size=n),
        "by_nt": rng.normal(1000.0, 5.0, size=n),
        "bz_nt": rng.normal(45000.0, 5.0, size=n),
    })
    if with_gradiometer:
        df["bx2_nt"] = rng.normal(19000.0, 5.0, size=n)
        df["by2_nt"] = rng.normal(1000.0, 5.0, size=n)
        df["bz2_nt"] = rng.normal(45000.0, 5.0, size=n)
    else:
        df["bx2_nt"] = np.nan
        df["by2_nt"] = np.nan
        df["bz2_nt"] = np.nan
    return df


def _register_dummy_survey(conn, survey_id="S1"):
    conn.execute(
        """
        INSERT INTO survey (survey_id, line_id, run_id, surveyed_at, step_m, n_samples,
                             chainage_start_m, chainage_end_m, standoff_m, schema_version,
                             source_uri, file_sha256, content_sha256, status, ingested_at)
        VALUES (?, 'LINE000', 0, '2026-07-31T00:00:00+00:00', 0.5, 1, 0.0, 1.0, 1.5, 2,
                'dummy', 'deadbeef', 'cafef00d', 'accepted', '2026-07-31T00:00:00+00:00')
        """,
        (survey_id,),
    )
    conn.commit()


def test_load_readings_round_trip(tmp_path):
    conn = connect(tmp_path / "t.db")
    _register_dummy_survey(conn)
    df = _make_frame(n=20, with_gradiometer=True)

    load_readings(conn, "S1", df)

    fetched = pd.read_sql_query(
        "SELECT sample_idx, lat, lon, bx_nt, by_nt, bz_nt, bx2_nt, by2_nt, bz2_nt "
        "FROM reading WHERE survey_id='S1' ORDER BY sample_idx",
        conn,
    )
    assert len(fetched) == 20
    for col in ["sample_idx", "lat", "lon", "bx_nt", "by_nt", "bz_nt", "bx2_nt", "by2_nt", "bz2_nt"]:
        np.testing.assert_allclose(fetched[col].to_numpy(dtype=float), df[col].to_numpy(dtype=float))


def test_load_readings_sample_idx_stored_as_correct_type(tmp_path):
    """Regression guard: numpy.int64 is NOT a Python `int` subclass (unlike
    numpy.float64/`float`) -- a vectorized rewrite that forgets this can
    silently mis-cast or raise sqlite3.InterfaceError.
    """
    conn = connect(tmp_path / "t.db")
    _register_dummy_survey(conn)
    df = _make_frame(n=5, with_gradiometer=True)
    assert df["sample_idx"].dtype == np.int64

    load_readings(conn, "S1", df)  # must not raise

    fetched = pd.read_sql_query("SELECT sample_idx FROM reading WHERE survey_id='S1'", conn)
    assert list(fetched["sample_idx"]) == list(range(5))


def test_load_readings_nulls_gradiometer_columns_when_disabled(tmp_path):
    conn = connect(tmp_path / "t.db")
    _register_dummy_survey(conn)
    df = _make_frame(n=10, with_gradiometer=False)

    load_readings(conn, "S1", df)

    fetched = pd.read_sql_query(
        "SELECT bx2_nt, by2_nt, bz2_nt FROM reading WHERE survey_id='S1'", conn
    )
    assert fetched["bx2_nt"].isna().all()
    assert fetched["by2_nt"].isna().all()
    assert fetched["bz2_nt"].isna().all()


@pytest.mark.parametrize("with_gradiometer", [True, False])
def test_load_readings_matches_naive_reference_implementation(tmp_path, with_gradiometer):
    df = _make_frame(n=30, seed=1, with_gradiometer=with_gradiometer)
    if not with_gradiometer:
        # a NaN mix, not all-or-nothing -- the real gradiometer-enabled case
        # never has partial NaNs, but the conversion logic shouldn't care.
        df.loc[::3, "bx2_nt"] = np.nan

    conn_fast = connect(tmp_path / "fast.db")
    _register_dummy_survey(conn_fast)
    load_readings(conn_fast, "S1", df)

    conn_naive = connect(tmp_path / "naive.db")
    _register_dummy_survey(conn_naive)
    _naive_load_readings(conn_naive, "S1", df)

    cols = "sample_idx, lat, lon, bx_nt, by_nt, bz_nt, bx2_nt, by2_nt, bz2_nt"
    fast_rows = pd.read_sql_query(f"SELECT {cols} FROM reading ORDER BY sample_idx", conn_fast)
    naive_rows = pd.read_sql_query(f"SELECT {cols} FROM reading ORDER BY sample_idx", conn_naive)
    pd.testing.assert_frame_equal(fast_rows, naive_rows)
