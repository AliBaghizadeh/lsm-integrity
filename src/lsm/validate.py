"""
Layer-1 data validation (references/validation-and-trust.md). Checks run
against the RAW DataFrame (read directly from the source Parquet file), NOT
against SQLite's `reading` table -- a WITHOUT ROWID table clustered on
(survey_id, sample_idx) always returns rows in key order regardless of whether
the raw file was actually out of order, which would make sample_idx_monotonic
and duplicate_sample_idx trivially and silently useless.

Two checks (noise_floor, interference_density) are implemented as coarse
proxies on the RAW signal rather than the detrended residual, because
background removal is Stage 2. They are still discriminating -- corrupted
data changes them meaningfully -- but they are not the final Stage-2 features.
This is called out explicitly rather than silently underclaimed.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from lsm.config import ValidateConfig
from lsm.evaluate import background_regime_shift
from lsm.logging_utils import get_logger
from lsm.schemas import validate_reading_schema, SchemaValidationError

log = get_logger("lsm.validate")

Status = str  # "pass" | "warn" | "fail"


@dataclass
class DQCheckResult:
    check_name: str
    status: Status
    n_affected: int
    detail: dict = field(default_factory=dict)


@dataclass
class DQReport:
    survey_id: str
    checked_at: str
    results: list[DQCheckResult]

    @property
    def has_fail(self) -> bool:
        return any(r.status == "fail" for r in self.results)

    def summary(self) -> dict[str, int]:
        out = {"pass": 0, "warn": 0, "fail": 0}
        for r in self.results:
            out[r.status] += 1
        return out


def _gate_status(is_violation: bool, gate_action: str) -> Status:
    if not is_violation:
        return "pass"
    return gate_action  # "fail" or "warn", as configured


# --------------------------------------------------------------------------
# Individual checks. Pure functions over a raw survey DataFrame (+ context),
# so tests can hit them directly with a clean frame and a corrupted one.
# --------------------------------------------------------------------------


def check_schema(df: pd.DataFrame, gate: str) -> DQCheckResult:
    try:
        validate_reading_schema(df)
        return DQCheckResult("schema", "pass", 0)
    except SchemaValidationError as exc:
        # n_affected is the count of DISTINCT rows that actually violated a
        # check, not len(df) -- one bad value in a 4000-row survey is "1 row
        # affected", not "the whole survey is bad".
        return DQCheckResult("schema", gate, exc.n_affected_rows, {"failures": exc.failures})


def check_range(df: pd.DataFrame, field_range: tuple[float, float], gate: str) -> DQCheckResult:
    lo, hi = field_range
    axes = ["bx_nt", "by_nt", "bz_nt"]
    mask = pd.Series(False, index=df.index)
    for a in axes:
        mask |= (df[a] < lo) | (df[a] > hi)
    n = int(mask.sum())
    return DQCheckResult("range", _gate_status(n > 0, gate), n)


def check_saturation(df: pd.DataFrame, run_length: int, gate: str) -> DQCheckResult:
    """>= run_length consecutive identical raw values on any axis: ADC rail / stuck sensor."""
    axes = ["bx_nt", "by_nt", "bz_nt"]
    n_affected = 0
    axes_hit = []
    for a in axes:
        vals = df[a].to_numpy()
        if len(vals) == 0:
            continue
        same_as_prev = np.concatenate([[False], vals[1:] == vals[:-1]])
        group_id = np.cumsum(~same_as_prev)  # increments each time the value changes
        run_len_per_row = np.bincount(group_id)[group_id]
        hit = run_len_per_row >= run_length
        if hit.any():
            axes_hit.append(a)
        n_affected += int(hit.sum())
    return DQCheckResult(
        "saturation", _gate_status(n_affected > 0, gate), n_affected, {"axes": axes_hit}
    )


def check_sample_idx_monotonic(df: pd.DataFrame, gate: str) -> DQCheckResult:
    idx = df["sample_idx"].to_numpy()
    n = int((np.diff(idx) <= 0).sum()) if len(idx) > 1 else 0
    return DQCheckResult("sample_idx_monotonic", _gate_status(n > 0, gate), n)


def check_sample_idx_gap(df: pd.DataFrame, step_m: float, max_gap_m: float, gate: str) -> DQCheckResult:
    idx = np.sort(df["sample_idx"].to_numpy())
    if len(idx) < 2:
        return DQCheckResult("sample_idx_gap", "pass", 0)
    gaps = np.diff(idx)
    max_gap_steps = max(1, int(round(max_gap_m / step_m)))
    bad = gaps > max_gap_steps
    n = int(bad.sum())
    missing_m = float(((gaps[bad] - 1) * step_m).sum()) if n else 0.0
    return DQCheckResult(
        "sample_idx_gap", _gate_status(n > 0, gate), n, {"missing_length_m": missing_m}
    )


def check_duplicate_sample_idx(df: pd.DataFrame, gate: str) -> DQCheckResult:
    n = int(df["sample_idx"].duplicated().sum())
    return DQCheckResult("duplicate_sample_idx", _gate_status(n > 0, gate), n)


def check_duplicate_content(
    conn: sqlite3.Connection, survey_id: str, content_hash: str, gate: str
) -> DQCheckResult:
    cur = conn.execute(
        "SELECT survey_id FROM survey WHERE content_sha256=? AND survey_id != ?",
        (content_hash, survey_id),
    )
    dupes = [r[0] for r in cur.fetchall()]
    return DQCheckResult(
        "duplicate_content", _gate_status(len(dupes) > 0, gate), len(dupes), {"duplicates_of": dupes}
    )


def check_survey_overlap(
    conn: sqlite3.Connection,
    survey_id: str,
    line_id: str,
    chainage_start_m: float,
    chainage_end_m: float,
    df: pd.DataFrame,
    corr_threshold: float,
    gate: str,
) -> DQCheckResult:
    """Overlap is EXPECTED for legitimate repeat surveys of the same line (that is
    the whole point of the growth-forecasting scenario). What is not expected is
    near-perfect signal correlation in the overlap -- that indicates the same
    physical acquisition was ingested twice under different run_id labels.
    """
    cur = conn.execute(
        """
        SELECT survey_id, chainage_start_m, chainage_end_m FROM survey
        WHERE line_id=? AND survey_id != ? AND status='accepted'
        """,
        (line_id, survey_id),
    )
    max_r = 0.0
    overlap_with = None
    step_m = (chainage_end_m - chainage_start_m) / max(1, len(df) - 1)
    for other_id, o_start, o_end in cur.fetchall():
        lo, hi = max(chainage_start_m, o_start), min(chainage_end_m, o_end)
        if lo >= hi:
            continue
        other = pd.read_sql_query(
            "SELECT sample_idx, bx_nt FROM reading WHERE survey_id=?", conn, params=(other_id,)
        )
        if other.empty:
            continue
        lo_idx, hi_idx = int(lo / step_m), int(hi / step_m)
        mine = df[(df["sample_idx"] >= lo_idx) & (df["sample_idx"] <= hi_idx)]
        theirs = other[(other["sample_idx"] >= lo_idx) & (other["sample_idx"] <= hi_idx)]
        merged = mine[["sample_idx", "bx_nt"]].merge(
            theirs, on="sample_idx", suffixes=("_mine", "_theirs")
        )
        if len(merged) < 10:
            continue
        r = float(np.corrcoef(merged["bx_nt_mine"], merged["bx_nt_theirs"])[0, 1])
        if abs(r) > max_r:
            max_r = abs(r)
            overlap_with = other_id

    if max_r > corr_threshold:
        return DQCheckResult(
            "survey_overlap", gate, 1, {"correlation": max_r, "overlap_with": overlap_with}
        )
    return DQCheckResult("survey_overlap", "pass", 0, {"max_correlation": max_r})


def check_gps_jump(df: pd.DataFrame, max_jump_m: float, gate: str) -> DQCheckResult:
    lat = np.radians(df["lat"].to_numpy())
    lon = np.radians(df["lon"].to_numpy())
    if len(lat) < 2:
        return DQCheckResult("gps_jump", "pass", 0)
    dlat = np.diff(lat)
    dlon = np.diff(lon)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon / 2) ** 2
    dist_m = 2 * 6_371_000.0 * np.arcsin(np.clip(np.sqrt(a), 0, 1))
    n = int((dist_m > max_jump_m).sum())
    return DQCheckResult(
        "gps_jump", _gate_status(n > 0, gate), n, {"max_jump_m": float(dist_m.max()) if len(dist_m) else 0.0}
    )


def check_gps_chainage_consistency(df: pd.DataFrame, step_m: float, gate: str) -> DQCheckResult:
    lat = np.radians(df["lat"].to_numpy())
    lon = np.radians(df["lon"].to_numpy())
    if len(lat) < 2:
        return DQCheckResult("gps_chainage_consistency", "pass", 0)
    dlat = np.diff(lat)
    dlon = np.diff(lon)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon / 2) ** 2
    dist_m = 2 * 6_371_000.0 * np.arcsin(np.clip(np.sqrt(a), 0, 1))
    gps_length = float(dist_m.sum())
    chainage_length = step_m * (len(df) - 1)
    if chainage_length == 0:
        return DQCheckResult("gps_chainage_consistency", "pass", 0)
    rel_err = abs(gps_length - chainage_length) / chainage_length
    return DQCheckResult(
        "gps_chainage_consistency",
        _gate_status(rel_err > 0.02, gate),
        int(rel_err > 0.02),
        {"relative_error": rel_err},
    )


def check_noise_floor(df: pd.DataFrame, noise_range: tuple[float, float], gate: str) -> DQCheckResult:
    """Coarse proxy: std of the first difference per axis, which suppresses slow
    drift and approximates the sensor noise floor without a real detrend (Stage 2).
    """
    lo, hi = noise_range
    bad_axes = []
    detail = {}
    for a in ["bx_nt", "by_nt", "bz_nt"]:
        sigma = float(np.diff(df[a].to_numpy()).std() / np.sqrt(2))
        detail[a] = sigma
        if sigma < lo or sigma > hi:
            bad_axes.append(a)
    return DQCheckResult(
        "noise_floor", _gate_status(len(bad_axes) > 0, gate), len(bad_axes), detail
    )


def check_background_regime(
    conn: sqlite3.Connection, line_id: str, survey_id: str, df: pd.DataFrame, gate: str
) -> DQCheckResult:
    cur = conn.execute(
        """
        SELECT survey_id FROM survey
        WHERE line_id=? AND survey_id != ? AND status='accepted'
        """,
        (line_id, survey_id),
    )
    prior_ids = [r[0] for r in cur.fetchall()]
    if len(prior_ids) < 2:
        return DQCheckResult(
            "background_regime", "pass", 0, {"reason": "insufficient history", "n_prior": len(prior_ids)}
        )
    medians = []
    for pid in prior_ids:
        prior = pd.read_sql_query(
            "SELECT bx_nt, by_nt, bz_nt FROM reading WHERE survey_id=?", conn, params=(pid,)
        )
        if not prior.empty:
            medians.append(prior[["bx_nt", "by_nt", "bz_nt"]].median())
    if len(medians) < 2:
        return DQCheckResult(
            "background_regime", "pass", 0, {"reason": "insufficient loaded history"}
        )
    hist = pd.DataFrame(medians)
    cur_median = df[["bx_nt", "by_nt", "bz_nt"]].median()
    shift = background_regime_shift(cur_median, hist, z_threshold=3.0)
    return DQCheckResult(
        "background_regime", _gate_status(shift["n_bad"] > 0, gate), shift["n_bad"],
        {"z_scores": shift["z_scores"]},
    )


def check_interference_density(df: pd.DataFrame, gate: str, expected_frac: float = 0.05) -> DQCheckResult:
    """Coarse proxy: robust z-score of raw Bx magnitude vs the survey median;
    fraction of |z|>5 rows compared against an expected small baseline.
    """
    vals = df["bx_nt"].to_numpy()
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) or 1e-9
    z = 0.6745 * (vals - med) / mad
    frac = float((np.abs(z) > 5).mean())
    return DQCheckResult(
        "interference_density",
        _gate_status(frac > expected_frac, gate),
        int((np.abs(z) > 5).sum()),
        {"fraction": frac},
    )


def check_coverage(df: pd.DataFrame, expected_length_m: float | None, step_m: float, gate: str) -> DQCheckResult:
    if expected_length_m is None:
        return DQCheckResult("coverage", "pass", 0, {"reason": "no expected length declared"})
    actual = step_m * len(df)
    frac = actual / expected_length_m
    return DQCheckResult("coverage", _gate_status(frac < 0.95, gate), int(frac < 0.95), {"coverage_frac": frac})


# --------------------------------------------------------------------------


def run_checks(
    df: pd.DataFrame,
    survey_id: str,
    line_id: str,
    step_m: float,
    chainage_start_m: float,
    chainage_end_m: float,
    content_hash: str,
    cfg: ValidateConfig,
    conn: sqlite3.Connection,
    expected_length_m: float | None = None,
) -> list[DQCheckResult]:
    g = cfg.gates
    return [
        check_schema(df, g["schema"]),
        check_range(df, cfg.field_range_nT, g["range"]),
        check_saturation(df, cfg.saturation_run_length, g["saturation"]),
        check_sample_idx_monotonic(df, g["sample_idx_monotonic"]),
        check_sample_idx_gap(df, step_m, cfg.max_gap_m, g["sample_idx_gap"]),
        check_duplicate_sample_idx(df, g["duplicate_sample_idx"]),
        check_duplicate_content(conn, survey_id, content_hash, g["duplicate_content"]),
        check_survey_overlap(
            conn, survey_id, line_id, chainage_start_m, chainage_end_m, df,
            cfg.overlap_correlation_threshold, g["survey_overlap"],
        ),
        check_gps_jump(df, cfg.max_gps_jump_m, g["gps_jump"]),
        check_gps_chainage_consistency(df, step_m, g["gps_chainage_consistency"]),
        check_noise_floor(df, cfg.noise_floor_range_nT, g["noise_floor"]),
        check_background_regime(conn, line_id, survey_id, df, g["background_regime"]),
        check_interference_density(df, g["interference_density"]),
        check_coverage(df, expected_length_m, step_m, g["coverage"]),
    ]


def validate_raw_survey(
    conn: sqlite3.Connection,
    survey_id: str,
    line_id: str,
    step_m: float,
    chainage_start_m: float,
    chainage_end_m: float,
    content_hash: str,
    df: pd.DataFrame,
    cfg: ValidateConfig,
    expected_length_m: float | None = None,
    quarantine_dir: Path | None = None,
    source_path: Path | None = None,
) -> DQReport:
    """Run all 14 checks against the raw survey DataFrame, persist the DQ report,
    and quarantine (flip survey.status + copy the raw file) on any hard fail.
    Assumes `survey_id` already has a row in `survey` (register_survey ran first).
    """
    results = run_checks(
        df, survey_id, line_id, step_m, chainage_start_m, chainage_end_m,
        content_hash, cfg, conn, expected_length_m,
    )
    report = DQReport(
        survey_id=survey_id,
        checked_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        results=results,
    )

    for r in results:
        conn.execute(
            "INSERT INTO dq_report (survey_id, checked_at, check_name, status, n_affected, detail_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (survey_id, report.checked_at, r.check_name, r.status, r.n_affected, str(r.detail)),
        )

    if report.has_fail:
        conn.execute("UPDATE survey SET status='quarantined' WHERE survey_id=?", (survey_id,))
        log.info("survey quarantined", extra={"survey_id": survey_id, "status": "quarantined"})
        if quarantine_dir is not None and source_path is not None:
            quarantine_raw_file(survey_id, source_path, quarantine_dir, report)
    else:
        log.info("survey validated clean", extra={"survey_id": survey_id, "status": "accepted"})
    conn.commit()
    return report


def quarantine_raw_file(survey_id: str, source_uri, quarantine_dir: Path, report: DQReport) -> Path:
    dest_dir = Path(quarantine_dir) / survey_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / "survey.parquet"
    shutil.copyfile(source_uri, dest_file)
    dq_json = dest_dir / "dq.json"
    dq_json.write_text(
        json.dumps(
            {
                "survey_id": report.survey_id,
                "checked_at": report.checked_at,
                "results": [
                    {"check_name": r.check_name, "status": r.status, "n_affected": r.n_affected, "detail": r.detail}
                    for r in report.results
                ],
            },
            indent=2,
            default=str,
        )
    )
    return dest_dir
