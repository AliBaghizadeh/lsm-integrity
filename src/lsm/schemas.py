"""
The data contract, declared once (references/data-contract.md) and imported by
generate.py, ingest.py, validate.py, features.py. A boundary that does not import
from here is a boundary where training/serving skew can appear.

Key decisions this module encodes:
  - sample_idx (int) is the physical key. chainage_m is DERIVED and never a key.
  - lat/lon are float64 -- float32 ULP at this latitude (~0.42 m) would quantise
    the survey, since it's comparable to the 0.5 m sample spacing.
  - defect_type is a CLOSED, PINNED-ORDER category. An unseen category is a hard
    fail, never a silent "other". Order matters: it is what a classifier's output
    columns are keyed to.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check

# Pinned category order -- serialised into every model bundle. Never let pandas
# infer this from whatever happened to be in a training frame.
DEFECT_TYPES: list[str] = ["scc", "weld", "dent", "corrosion", "interference", "none"]

FIELD_RANGE_NT = (-80_000.0, 80_000.0)

# Raw reading table, as generated/ingested per survey. This is the schema
# `validate.py`'s `schema` check enforces (Layer 1, gate: fail).
RawReadingSchema = DataFrameSchema(
    {
        "sample_idx": Column(pa.Int64, Check.ge(0), nullable=False),
        "lat": Column(pa.Float64, nullable=False),
        "lon": Column(pa.Float64, nullable=False),
        "bx_nt": Column(pa.Float64, Check.in_range(*FIELD_RANGE_NT), nullable=False),
        "by_nt": Column(pa.Float64, Check.in_range(*FIELD_RANGE_NT), nullable=False),
        "bz_nt": Column(pa.Float64, Check.in_range(*FIELD_RANGE_NT), nullable=False),
        "bx2_nt": Column(pa.Float64, Check.in_range(*FIELD_RANGE_NT), nullable=True),
        "by2_nt": Column(pa.Float64, Check.in_range(*FIELD_RANGE_NT), nullable=True),
        "bz2_nt": Column(pa.Float64, Check.in_range(*FIELD_RANGE_NT), nullable=True),
        "defect": Column(pa.Int64, Check.isin([0, 1]), nullable=False),
        "defect_type": Column(pa.String, Check.isin(DEFECT_TYPES), nullable=False),
        "severity_smys": Column(pa.Float64, nullable=True),
        # Ground truth for off-pipe interference sources -- tracked separately
        "interference": Column(pa.Int64, Check.isin([0, 1]), nullable=False),
    },
    strict=False,  # extra convenience columns (e.g. chainage_m) are allowed but not checked
    coerce=False,
)

SurveyMetaSchema = DataFrameSchema(
    {
        "line_id": Column(pa.String, nullable=False),
        "run_id": Column(pa.Int64, Check.ge(0), nullable=False),
    },
    strict=False,
)


# Per-row data-quality flag on a FEATURE row (distinct from the survey-level DQ
# report). 'edge' means a rolling window was truncated at the survey ends, so the
# row is kept, excluded from metrics, and scored with a widened interval --
# see references/data-contract.md #5. Closed enum, same as defect_type.
DQ_FLAGS: list[str] = ["clean", "edge"]

# Identity columns every feature frame carries. Note what is NOT here: lat/lon are
# restricted columns (data-contract.md #10) and are joined back from `reading` on
# the exact integer key when an indication needs a position. This is the integer-key
# decision paying off -- a float chainage join would silently drop rows.
FEATURE_KEY_COLUMNS: list[str] = [
    "survey_id",
    "line_id",
    "run_id",
    "sample_idx",
    "chainage_m",
]

# The fixed core of a feature frame. Window and peak-shape columns are driven by
# config (features.windows_m), so their names are asserted separately against the
# ordered feature list in features.feature_columns() -- which is what a bundle pins.
FeatureFrameSchema = DataFrameSchema(
    {
        "survey_id": Column(pa.String, nullable=False),
        "line_id": Column(pa.String, nullable=False),
        "run_id": Column(pa.Int64, Check.ge(0), nullable=False),
        "sample_idx": Column(pa.Int64, Check.ge(0), nullable=False, unique=True),
        "chainage_m": Column(pa.Float64, nullable=False),
        "feature_version": Column(pa.Int64, Check.ge(1), nullable=False),
        "dq_flag": Column(pa.String, Check.isin(DQ_FLAGS), nullable=False),
        # Residuals are the whole point of the stage: they must exist and be finite.
        "rx_nt": Column(pa.Float32, nullable=False),
        "ry_nt": Column(pa.Float32, nullable=False),
        "rz_nt": Column(pa.Float32, nullable=False),
        "r_mag_nt": Column(pa.Float32, Check.ge(0), nullable=False),
    },
    strict=False,
    coerce=False,
)


class SchemaValidationError(Exception):
    """Raised when a frame fails RawReadingSchema/FeatureFrameSchema -- always a
    hard `fail` gate.

    Carries pandera's structured `failure_cases` (one row per violated check),
    not just its string repr -- a caller that only had `str(exc)` had no way to
    tell "1 row out of 4000 violated one check" from "the whole frame is bad"
    except by re-parsing a pandas DataFrame's printed representation. Both
    `n_affected_rows` (distinct rows, not distinct checks) and `failures` (JSON-
    safe: native Python types, not numpy scalars) are computed once here so
    every caller gets the same honest count.
    """

    def __init__(self, failure_cases) -> None:
        self.failures: list[dict] = [
            {
                "column": None if pd.isna(row.get("column")) else str(row["column"]),
                "check": str(row["check"]),
                "failure_case": None if pd.isna(row.get("failure_case")) else _to_native(row["failure_case"]),
                "row_index": None if pd.isna(row.get("index")) else int(row["index"]),
            }
            for row in failure_cases.to_dict("records")
        ]
        row_indices = failure_cases["index"].dropna()
        # A dataframe-level check (no per-row index) still affected >=1 row --
        # never report 0 rows affected for a check that failed.
        self.n_affected_rows: int = int(row_indices.nunique()) if len(row_indices) else len(self.failures)
        super().__init__(
            f"{len(self.failures)} check failure(s) across {self.n_affected_rows} row(s): "
            f"{self.failures[:5]}"
        )


def _to_native(value):
    """numpy scalar -> the equivalent native Python type, so `failures` is
    directly JSON/`st.json`-safe without a `default=str` escape hatch."""
    if hasattr(value, "item"):
        return value.item()
    return value


def validate_reading_schema(df) -> None:
    """Raise SchemaValidationError with a readable message on any contract violation."""
    try:
        RawReadingSchema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:  # pragma: no cover - exercised via validate.py
        raise SchemaValidationError(exc.failure_cases) from exc


def validate_feature_schema(df) -> None:
    """The feature-store boundary check. Same failure mode as the raw boundary:
    raise, never coerce -- a boundary that repairs its input is a boundary where
    training/serving skew hides.
    """
    try:
        FeatureFrameSchema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        raise SchemaValidationError(exc.failure_cases) from exc
