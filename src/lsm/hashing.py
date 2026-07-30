"""
Two hashes, two jobs (see references/data-contract.md #6):
  - file_sha256:    over raw file bytes. Provenance / immutability -- "this exact
                     artifact". Different formatting of the same data -> different hash.
  - content_sha256: over CANONICALISED content (contract columns, declared order,
                     sorted by sample_idx, cast to contract dtypes, rounded floats).
                     Semantic identity / dedup -- "this data, however exported".

data_sha256 pins an exact training corpus as a Merkle-style hash over the sorted
content_sha256 list of its member surveys.

Uses stdlib hashlib.sha256 throughout -- fine at this project's row counts, and
avoids an extra dependency (blake3/xxhash) not already installed in ml_gpu.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pyarrow as pa

# Contract precision for canonicalisation -- must match schemas.py.
CANONICAL_FLOAT_DECIMALS = 6

# Column order the canonical hash is computed over. Anything outside this set
# (e.g. an ingestion timestamp) is metadata, not content, and must not affect
# the content hash.
CONTENT_COLUMNS = [
    "sample_idx",
    "lat",
    "lon",
    "bx_nt",
    "by_nt",
    "bz_nt",
    "bx2_nt",
    "by2_nt",
    "bz2_nt",
    "defect",
    "defect_type",
    "severity_smys",
    "interference",
]


def file_sha256(path: Path) -> str:
    """Hash of the raw file bytes -- provenance and immutability."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonicalise(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a reading-table frame to the canonical form used for content hashing.

    Re-exporting the same data with different column order or float formatting
    must produce the SAME canonical form, and therefore the same content_sha256.
    """
    cols = [c for c in CONTENT_COLUMNS if c in df.columns]
    out = df[cols].sort_values("sample_idx").reset_index(drop=True)
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].astype("float64").round(CANONICAL_FLOAT_DECIMALS)
    return out


def content_sha256(df: pd.DataFrame) -> str:
    """Hash over canonicalised content -- identity/dedup, robust to re-export."""
    canon = canonicalise(df)
    table = pa.Table.from_pandas(canon, preserve_index=False)
    h = hashlib.sha256()
    for batch in table.to_batches():
        for column in batch.columns:
            for buf in column.buffers():
                if buf is not None:
                    h.update(buf)
    return h.hexdigest()


def data_sha256(member_content_hashes: list[str]) -> str:
    """Merkle-style hash over a training corpus: sorted member content hashes."""
    h = hashlib.sha256()
    for c in sorted(member_content_hashes):
        h.update(c.encode("ascii"))
    return h.hexdigest()
