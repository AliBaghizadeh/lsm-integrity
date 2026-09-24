"""Stage 6: ingest.py::load_readings -- correctness of the vectorized bulk-
insert rewrite (replaced the old itertuples()+float()-per-cell loop for
throughput; see docs/stage6-scale-rehearsal.md for the measured before/after).

Rig-v2 (2026-08-06, Stage B): rewritten for the new scalar-rig `reading`
schema (t_s, b_lo/mid/hi_nt, girth_weld, chainage_true_m -- see db.py's DDL
and schemas.py). The old bx_nt/by_nt/bz_nt/bx2_nt/by2_nt/bz2_nt vector-axis
columns are gone; lat/lon are the nullable columns now (GPS dropout, not an
optional second gradiometer head).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lsm.db import connect
from lsm.ingest import load_readings


def _naive_load_readings(conn, survey_id, df):
    """The pre-Stage-6 reference implementation, kept ONLY here as an
    equivalence oracle -- never restored to ingest.py itself. Mirrors
    load_readings' current column set (t_s/b_lo/mid/hi_nt/girth_weld/
    chainage_true_m), not the pre-Rig-v2 vector schema this test used to use.
    """
    conn.executemany(
        """
        INSERT INTO reading (survey_id, sample_idx, t_s, lat, lon,
                              b_lo_nt, b_mid_nt, b_hi_nt, girth_weld, chainage_true_m)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                survey_id,
                int(r.sample_idx),
                float(r.t_s),
                None if pd.isna(r.lat) else float(r.lat),
                None if pd.isna(r.lon) else float(r.lon),
                float(r.b_lo_nt),
                float(r.b_mid_nt),
                float(r.b_hi_nt),
                int(r.girth_weld),
                float(r.chainage_true_m),
            )
            for r in df.itertuples(index=False)
        ],
    )
    conn.commit()


def _make_frame(n=20, seed=0, with_gps_dropout=False):
    """A raw-schema-shaped frame: sample_idx/t_s dense and regular (irregular
    real spacing doesn't matter to load_readings -- it just inserts whatever
    chainage_true_m/t_s it's given), b_lo/mid/hi_nt in the real field range,
    lat/lon locked everywhere unless `with_gps_dropout` punches NaN gaps in
    (Rig-v2's actual failure mode -- GpsConfig -- not an all-or-nothing head).
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "sample_idx": np.arange(n),
            "t_s": np.arange(n, dtype=np.float64) * 0.5,
            "lat": 46.0 + rng.normal(size=n) * 1e-5,
            "lon": 8.0 + rng.normal(size=n) * 1e-5,
            "b_lo_nt": rng.normal(19000.0, 5.0, size=n),
            "b_mid_nt": rng.normal(45000.0, 5.0, size=n),
            "b_hi_nt": rng.normal(19000.0, 5.0, size=n),
            "girth_weld": np.zeros(n, dtype=np.int64),
            "chainage_true_m": np.arange(n, dtype=np.float64) * 0.6,
        }
    )
    if with_gps_dropout:
        # A real dropout is a RUN of consecutive NaN rows (Markov good/bad
        # lock state, generate.py::_gps_track), not scattered singletons --
        # but load_readings' NaN->None conversion is per-row regardless, so
        # a simple periodic mask is enough to exercise "some locked, some
        # not" without needing the full Markov simulation here.
        df.loc[::4, "lat"] = np.nan
        df.loc[::4, "lon"] = np.nan
    return df


def _register_dummy_survey(conn, survey_id="S1"):
    conn.execute(
        """
        INSERT INTO survey (survey_id, line_id, run_id, surveyed_at, step_m, n_samples,
                             chainage_start_m, chainage_end_m, standoff_m, schema_version,
                             source_uri, file_sha256, content_sha256, status, ingested_at)
        VALUES (?, 'LINE000', 0, '2026-07-31T00:00:00+00:00', 0.5, 1, 0.0, 1.0, 1.5, 3,
                'dummy', 'deadbeef', 'cafef00d', 'accepted', '2026-07-31T00:00:00+00:00')
        """,
        (survey_id,),
    )
    conn.commit()


def test_load_readings_round_trip(tmp_path):
    conn = connect(tmp_path / "t.db")
    _register_dummy_survey(conn)
    df = _make_frame(n=20)

    load_readings(conn, "S1", df)

    fetched = pd.read_sql_query(
        "SELECT sample_idx, t_s, lat, lon, b_lo_nt, b_mid_nt, b_hi_nt, girth_weld, chainage_true_m "
        "FROM reading WHERE survey_id='S1' ORDER BY sample_idx",
        conn,
    )
    cols = [
        "sample_idx",
        "t_s",
        "lat",
        "lon",
        "b_lo_nt",
        "b_mid_nt",
        "b_hi_nt",
        "girth_weld",
        "chainage_true_m",
    ]
    assert len(fetched) == 20
    for col in cols:
        np.testing.assert_allclose(
            fetched[col].to_numpy(dtype=float), df[col].to_numpy(dtype=float)
        )


def test_load_readings_sample_idx_stored_as_correct_type(tmp_path):
    """Regression guard: numpy.int64 is NOT a Python `int` subclass (unlike
    numpy.float64/`float`) -- a vectorized rewrite that forgets this can
    silently mis-cast or raise sqlite3.InterfaceError. girth_weld is the
    other integer column load_readings casts explicitly -- covered here too.
    """
    conn = connect(tmp_path / "t.db")
    _register_dummy_survey(conn)
    df = _make_frame(n=5)
    df.loc[2, "girth_weld"] = 1
    assert df["sample_idx"].dtype == np.int64
    assert df["girth_weld"].dtype == np.int64

    load_readings(conn, "S1", df)  # must not raise

    fetched = pd.read_sql_query(
        "SELECT sample_idx, girth_weld FROM reading WHERE survey_id='S1' ORDER BY sample_idx",
        conn,
    )
    assert list(fetched["sample_idx"]) == list(range(5))
    assert list(fetched["girth_weld"]) == [0, 0, 1, 0, 0]


def test_load_readings_nulls_lat_lon_through_gps_dropout(tmp_path):
    """Repurposed 2026-08-06 (Stage B): the old version of this test asserted
    bx2_nt/by2_nt/bz2_nt got nulled when a SECOND GRADIOMETER HEAD was
    disabled -- that concept does not exist under Rig-v2's scalar rig (there
    is no "disabled" head; there are always exactly 3 heads reporting
    b_lo/mid/hi_nt, all NOT NULL per RawReadingSchema). The column that IS
    genuinely nullable now, for a genuinely real reason, is lat/lon through a
    GPS dropout (GpsConfig) -- so this test now covers THAT, which is the
    schema's actual nullable-column story under Rig-v2, rather than forcing
    the old test to pass against a schema it no longer describes.
    """
    conn = connect(tmp_path / "t.db")
    _register_dummy_survey(conn)
    df = _make_frame(n=12, with_gps_dropout=True)
    dropped_rows = df.index[df["lat"].isna()]
    locked_rows = df.index[df["lat"].notna()]
    assert (
        len(dropped_rows) > 0 and len(locked_rows) > 0
    )  # fixture actually exercises both

    load_readings(conn, "S1", df)

    fetched = pd.read_sql_query(
        "SELECT sample_idx, lat, lon FROM reading WHERE survey_id='S1' ORDER BY sample_idx",
        conn,
    )
    assert fetched.loc[fetched["sample_idx"].isin(dropped_rows), "lat"].isna().all()
    assert fetched.loc[fetched["sample_idx"].isin(dropped_rows), "lon"].isna().all()
    assert fetched.loc[fetched["sample_idx"].isin(locked_rows), "lat"].notna().all()
    assert fetched.loc[fetched["sample_idx"].isin(locked_rows), "lon"].notna().all()
    # b_lo/mid/hi_nt/girth_weld/chainage_true_m are NOT NULL regardless of GPS lock.
    fetched_fields = pd.read_sql_query(
        "SELECT b_lo_nt, b_mid_nt, b_hi_nt, girth_weld, chainage_true_m "
        "FROM reading WHERE survey_id='S1'",
        conn,
    )
    assert fetched_fields.notna().all().all()


@pytest.mark.parametrize("with_gps_dropout", [True, False])
def test_load_readings_matches_naive_reference_implementation(
    tmp_path, with_gps_dropout
):
    df = _make_frame(n=30, seed=1, with_gps_dropout=with_gps_dropout)

    conn_fast = connect(tmp_path / "fast.db")
    _register_dummy_survey(conn_fast)
    load_readings(conn_fast, "S1", df)

    conn_naive = connect(tmp_path / "naive.db")
    _register_dummy_survey(conn_naive)
    _naive_load_readings(conn_naive, "S1", df)

    cols = "sample_idx, t_s, lat, lon, b_lo_nt, b_mid_nt, b_hi_nt, girth_weld, chainage_true_m"
    fast_rows = pd.read_sql_query(
        f"SELECT {cols} FROM reading ORDER BY sample_idx", conn_fast
    )
    naive_rows = pd.read_sql_query(
        f"SELECT {cols} FROM reading ORDER BY sample_idx", conn_naive
    )
    pd.testing.assert_frame_equal(fast_rows, naive_rows)
