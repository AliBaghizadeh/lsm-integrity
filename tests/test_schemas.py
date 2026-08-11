from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lsm.schemas import DEFECT_TYPES, SchemaValidationError, validate_reading_schema


def _clean_df() -> pd.DataFrame:
    """Rig-v2 (schema_version 3): a scalar-rig raw frame -- b_lo/b_mid/b_hi_nt
    total-field magnitudes, t_s, chainage_true_m, girth_weld. See schemas.py.
    """
    return pd.DataFrame(
        {
            "sample_idx": [0, 1, 2],
            "t_s": [0.0, 0.5, 1.0],
            "lat": [46.9, 46.91, 46.92],
            "lon": [8.3, 8.31, 8.32],
            "b_lo_nt": [48800.0, 48801.0, 48802.0],
            "b_mid_nt": [48857.0, 48858.0, 48859.0],
            "b_hi_nt": [48900.0, 48901.0, 48902.0],
            "defect": [0, 0, 1],
            "defect_type": ["none", "none", "scc"],
            "severity_smys": [np.nan, np.nan, 42.0],
            "interference": [0, 0, 0],
            "girth_weld": [0, 0, 0],
            "chainage_true_m": [0.0, 1.2, 2.4],
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
    df.loc[0, "b_lo_nt"] = 200_000.0
    with pytest.raises(SchemaValidationError):
        validate_reading_schema(df)


def test_schema_error_reports_affected_rows_not_the_whole_frame():
    """One bad value in a 3-row frame is "1 row affected", not "3 rows
    affected" -- n_affected_rows previously wasn't computed at all; the only
    caller (validate.check_schema) used to hardcode len(df), which reported
    the entire survey as bad for a single out-of-range value.
    """
    df = _clean_df()
    df.loc[0, "b_lo_nt"] = 200_000.0
    with pytest.raises(SchemaValidationError) as exc_info:
        validate_reading_schema(df)
    exc = exc_info.value
    assert exc.n_affected_rows == 1
    assert len(exc.failures) == 1
    failure = exc.failures[0]
    assert failure["column"] == "b_lo_nt"
    assert failure["row_index"] == 0
    assert failure["failure_case"] == 200_000.0
    # JSON/st.json-safe: native Python types, not numpy scalars.
    assert isinstance(failure["failure_case"], float)
    assert isinstance(failure["row_index"], int)


def test_null_required_field_is_rejected():
    df = _clean_df()
    df.loc[0, "b_lo_nt"] = np.nan
    with pytest.raises(SchemaValidationError):
        validate_reading_schema(df)


def test_defect_type_order_is_pinned():
    # the classifier's output columns are keyed to this order -- an accidental
    # reshuffle here is a silent, hard-to-detect correctness bug.
    assert DEFECT_TYPES == ["scc", "weld", "dent", "corrosion", "interference", "none"]
