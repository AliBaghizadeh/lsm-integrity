"""
Stage 8: the dig-feedback loop -- a verification (an excavation) writes
`source='excavation'` truth rows and lets empirical coverage be re-measured
against real outcomes, not just held-out synthetic data.

No schema change: `indication.created_at` and `truth_defect.verified_at`
already exist. "Median days from indication to verification" is computed by
a chainage-proximity join between them at query time (the same
`MATCH_TOLERANCE_M` pattern used everywhere else in this project), not a new
dedicated link table -- this deliberately avoids a `schema_version` bump,
which would otherwise invalidate every already-trained bundle
(`bundle.load_bundle` hard-fails on any schema mismatch) for a metric that
doesn't need one.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
import uuid

import numpy as np

from lsm.evaluate import coverage_vs_nominal
from lsm.train import MATCH_TOLERANCE_M


def record_excavation(
    conn: sqlite3.Connection,
    indication_id: str,
    verified_severity_smys: float,
    verified_defect_type: str | None = None,
    verified_at: str | None = None,
    match_tolerance_m: float = MATCH_TOLERANCE_M,
) -> str:
    """Looks up the indication's `(survey_id, chainage_peak_m)` and
    `line_id`; finds the nearest EXISTING defect on that line (via each
    defect's LATEST `truth_defect` revision, `valid_to IS NULL`) within
    `match_tolerance_m`. If found: writes a NEW `truth_defect` revision
    (closing the prior revision's `valid_to` -- label corrections are
    revisions, not edits) with `source='excavation'`, plus a
    `truth_observation` row for THIS indication's survey. If not found:
    creates a brand-new `defect` (a dig finding something not predicted
    before), revision 0, `source='excavation'`. Returns `defect_id`.

    Since `ingest.py` currently has ZERO insert sites into
    `truth_defect`/`truth_observation` (a known, documented gap -- `truth.py`'s
    own module docstring says so), the FIRST excavation against any given
    physical defect in this project always takes the "not found -> create
    new" branch. The "match an existing defect" branch is real production
    logic but is only exercised by a test that seeds a prior
    `truth_defect`/`defect` row itself.
    """
    verified_at = verified_at or dt.datetime.now(dt.UTC).isoformat()

    row = conn.execute(
        "SELECT survey_id, chainage_peak_m FROM indication WHERE indication_id=?",
        (indication_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"indication {indication_id!r} not found")
    survey_id, chainage_peak_m = row
    line_row = conn.execute(
        "SELECT line_id FROM survey WHERE survey_id=?", (survey_id,)
    ).fetchone()
    if line_row is None:
        raise ValueError(
            f"survey {survey_id!r} (for indication {indication_id!r}) not found"
        )
    line_id = line_row[0]

    candidates = conn.execute(
        """
        SELECT td.defect_id, td.chainage_m, td.revision
        FROM truth_defect td
        JOIN defect d ON d.defect_id = td.defect_id
        WHERE d.line_id = ? AND td.valid_to IS NULL
        """,
        (line_id,),
    ).fetchall()

    best: tuple[str, int, float] | None = None
    for defect_id, chainage_m, revision in candidates:
        dist = abs(chainage_m - chainage_peak_m)
        if dist <= match_tolerance_m and (best is None or dist < best[2]):
            best = (defect_id, revision, dist)

    if best is not None:
        defect_id, prev_revision, _ = best
        new_revision = prev_revision + 1
        conn.execute(
            "UPDATE truth_defect SET valid_to=? WHERE defect_id=? AND revision=?",
            (verified_at, defect_id, prev_revision),
        )
    else:
        defect_id = f"excavation_{uuid.uuid4().hex[:12]}"
        new_revision = 0
        conn.execute(
            "INSERT INTO defect (defect_id, line_id) VALUES (?, ?)",
            (defect_id, line_id),
        )

    conn.execute(
        "INSERT INTO truth_defect (defect_id, revision, chainage_m, defect_type, source, "
        "verified_at, valid_from, valid_to) VALUES (?,?,?,?,?,?,?,NULL)",
        (
            defect_id,
            new_revision,
            chainage_peak_m,
            verified_defect_type or "unknown",
            "excavation",
            verified_at,
            verified_at,
        ),
    )
    conn.execute(
        "INSERT OR REPLACE INTO truth_observation (defect_id, survey_id, severity_smys) VALUES (?,?,?)",
        (defect_id, survey_id, verified_severity_smys),
    )
    conn.commit()
    return defect_id


def median_days_to_verification(conn: sqlite3.Connection) -> float | None:
    """Median `(verified_at - earliest matching indication's created_at)` in
    days, over every `source='excavation'` `truth_defect` row -- "the
    number to request from the data owner". Computed via a chainage-proximity
    join at query time (see module docstring), not a stored link. `None` if
    no excavation has been recorded yet.
    """
    excavations = conn.execute(
        """
        SELECT td.chainage_m, td.verified_at, d.line_id
        FROM truth_defect td
        JOIN defect d ON d.defect_id = td.defect_id
        WHERE td.source = 'excavation'
        """
    ).fetchall()
    if not excavations:
        return None

    deltas = []
    for chainage_m, verified_at, line_id in excavations:
        candidates = conn.execute(
            """
            SELECT i.chainage_peak_m, i.created_at FROM indication i
            JOIN survey s ON s.survey_id = i.survey_id
            WHERE s.line_id = ? AND i.created_at <= ?
            """,
            (line_id, verified_at),
        ).fetchall()
        matching_created_ats = [
            created_at
            for chainage_peak_m, created_at in candidates
            if abs(chainage_peak_m - chainage_m) <= MATCH_TOLERANCE_M
        ]
        if not matching_created_ats:
            continue
        earliest_created_at = min(matching_created_ats)
        delta_days = (
            dt.datetime.fromisoformat(verified_at)
            - dt.datetime.fromisoformat(earliest_created_at)
        ).total_seconds() / 86400.0
        deltas.append(delta_days)

    if not deltas:
        return None
    return float(np.median(deltas))


def recompute_coverage_from_verifications(
    conn: sqlite3.Connection, pipeline_version: str, nominal: float
) -> dict:
    """Joins verified `truth_observation.severity_smys` (as-verified ground
    truth, for `source='excavation'` defects only) against the NEAREST
    matching `indication.(sev_lo, sev_hi)` in that same survey AND
    `pipeline_version` (SKILL invariant #11 -- never compare verifications
    against indications scored by a different release), within
    `MATCH_TOLERANCE_M` of the defect's own chainage. `{"n": 0}` if nothing
    has been verified yet under this `pipeline_version`.
    """
    excavated = conn.execute(
        """
        SELECT o.survey_id, o.severity_smys, td.chainage_m
        FROM truth_observation o
        JOIN truth_defect td ON td.defect_id = o.defect_id AND td.valid_to IS NULL
        WHERE td.source = 'excavation'
        """
    ).fetchall()
    if not excavated:
        return {"n": 0}

    y_true_list, lo_list, hi_list = [], [], []
    for survey_id, severity_smys, chainage_m in excavated:
        candidates = conn.execute(
            """
            SELECT chainage_peak_m, sev_lo, sev_hi FROM indication
            WHERE survey_id=? AND pipeline_version=? AND sev_lo IS NOT NULL AND sev_hi IS NOT NULL
            """,
            (survey_id, pipeline_version),
        ).fetchall()
        in_range = [
            c for c in candidates if abs(c[0] - chainage_m) <= MATCH_TOLERANCE_M
        ]
        if not in_range:
            continue
        _, lo, hi = min(in_range, key=lambda c: abs(c[0] - chainage_m))
        y_true_list.append(severity_smys)
        lo_list.append(lo)
        hi_list.append(hi)

    if not y_true_list:
        return {"n": 0}
    return coverage_vs_nominal(
        np.array(y_true_list), np.array(lo_list), np.array(hi_list), nominal
    )
