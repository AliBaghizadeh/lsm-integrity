from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lsm.schemas import DEFECT_TYPES, SchemaValidationError, validate_reading_schema


def _clean_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_idx": [0, 1, 2],
            "lat": [46.9, 46.91, 46.92],
            "lon": [8.3, 8.31, 8.32],
            "bx_nt": [19000.0, 19001.0, 19002.0],
            "by_nt": [1000.0, 1001.0, 1002.0],
            "bz_nt": [45000.0, 45001.0, 45002.0],
            "bx2_nt": [np.nan, np.nan, np.nan],
            "by2_nt": [np.nan, np.nan, np.nan],
            "bz2_nt": [np.nan, np.nan, np.nan],
            "defect": [0, 0, 1],
            "defect_type": ["none", "none", "scc"],
            "severity_smys": [np.nan, np.nan, 42.0],
            "interference": [0, 0, 0],
        }
    )


def test_clean_frame_validates():
    validate_reading_schema(_clean_df())  # must not raise


def test_unseen_defect_type_is_rejected():
    df = _clean_df()
    df.loc[0, "defect_type"] = "not_a_real_category"
    with pytest.raises(SchemaValidationError):
        validate_reading_schema(df)


def test_out_of_range_field_is_rejected():
    df = _clean_df()
    df.loc[0, "bx_nt"] = 200_000.0
    with pytest.raises(SchemaValidationError):
        validate_reading_schema(df)


def test_schema_error_reports_affected_rows_not_the_whole_frame():
    """One bad value in a 3-row frame is "1 row affected", not "3 rows
    affected" -- n_affected_rows previously wasn't computed at all; the only
    caller (validate.check_schema) used to hardcode len(df), which reported
    the entire survey as bad for a single out-of-range value.
    """
    df = _clean_df()
    df.loc[0, "bx_nt"] = 200_000.0
    with pytest.raises(SchemaValidationError) as exc_info:
        validate_reading_schema(df)
    exc = exc_info.value
    assert exc.n_affected_rows == 1
    assert len(exc.failures) == 1
    failure = exc.failures[0]
    assert failure["column"] == "bx_nt"
    assert failure["row_index"] == 0
    assert failure["failure_case"] == 200_000.0
    # JSON/st.json-safe: native Python types, not numpy scalars.
    assert isinstance(failure["failure_case"], float)
    assert isinstance(failure["row_index"], int)


def test_null_required_field_is_rejected():
    df = _clean_df()
    df.loc[0, "bx_nt"] = np.nan
    with pytest.raises(SchemaValidationError):
        validate_reading_schema(df)


def test_defect_type_order_is_pinned():
    # the classifier's output columns are keyed to this order -- an accidental
    # reshuffle here is a silent, hard-to-detect correctness bug.
    assert DEFECT_TYPES == ["scc", "weld", "dent", "corrosion", "interference", "none"]
