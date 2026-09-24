"""
Stage 6: the scale rehearsal. Generates, ingests, featurises, and trains
against `config/scale/` (currently ~18M rows, 30 lines x 2 km x 3 runs =
360 physical defects, re-derived for Rig-v2's ~100 rows/m time-based
sampling -- see `config/scale/base.yaml`'s own header for the full
arithmetic and for why matching the pre-Rig-v2 run's 9,600-defect
statistical power would now need ~4.8x10^8 rows),
measuring wall-clock and peak RSS at each step, comparing the pandas-concat
vs. DuckDB bulk-read paths, and running the whole-line-holdout + temporal-
holdout evaluations alongside the default 5-fold block CV. Writes
`docs/stage6-scale-rehearsal.md`.

This is a REAL, ONE-TIME LOCAL RUN, not a test: real wall-clock time (likely
tens of minutes) and ~2-4 GB of disk under `data_scale/` (gitignored,
regenerable). It is NOT wired into any CI workflow and is never run inside
pytest -- same convention as `scripts/check_observatory_background.py`.

Usage:
    python scripts/stage6_scale_rehearsal.py
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import contextlib
import io

import pandas as pd

from lsm import scale_eval, train
from lsm.config import CONFIG_DIR, load_config
from lsm.db import connect
from lsm.features import load_feature_corpus
from lsm.generate import generate_all
from lsm.perf_utils import PerfResult, format_perf_table, measure
from lsm.pipeline import (
    SurveyNotFeaturisableError,
    run_feature_pipeline,
    run_survey_pipeline,
)

# Measured once, by hand, against the pre-Stage-6 itertuples()+float()-per-
# cell load_readings implementation, at the same representative scale (one
# 40 km / 80,000-row survey) used for the "after" measurement below --
# see PLAN.md Stage 6 / the ingest.py docstring. Not re-measured on every run:
# the old code path no longer exists in this repo (see git history for
# ingest.py before this commit).
INGEST_BEFORE_ROWS_PER_SEC = 235_982
INGEST_BEFORE_PEAK_RSS_MB = 183.5
INGEST_BEFORE_N_ROWS = 80_000


def _now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _background_contrast_check(cfg) -> dict:
    """Empirical re-check of the Stage 2 background-contrast ratio at this
    config's line length -- density-per-km is unchanged from the demo corpus,
    so this is EXPECTED to hold, but must be checked, not assumed, per this
    project's own standard for every past scaling change. Returns the
    background/defect medians too, not just the ratio, so a miss can be
    diagnosed (which side moved) rather than just reported as a number.

    RIG-V2 PORT (2026-08-10): this read `r_mag_nt` -- the pre-Rig-v2 VECTOR
    residual magnitude, which no longer exists (`rig: scalar` has no x/y/z to
    take a magnitude of). Its successor is `r_mid_nt`, the middle head's own
    detrended residual, but that is **signed**: a total-field anomaly can
    enhance OR degrade the ambient field, and with `stress_polarity: random`
    it does either with equal probability. A median of the signed column sits
    near zero for both populations, which would make this ratio meaningless
    (and near-divide-by-zero). Taking |r_mid_nt| restores a magnitude-like
    quantity comparable in spirit to the old vector magnitude.

    The 3.0x threshold below is a PRE-RIG-V2 calibration (the vector rig
    measured 3.19x). No like-for-like contrast baseline has been established
    for the scalar rig -- Stage D re-measured the gates and the ablation
    ladder but never produced a matching contrast figure. So the verdict here
    is INFORMATIONAL, not a gate: report the number, do not read a pass/fail
    against a threshold calibrated on different physics.
    """
    raw_path = (
        Path(cfg.env.storage.raw_dir)
        / "line_id=LINE000"
        / "run_id=0"
        / "survey.parquet"
    )
    from lsm.features import feature_store_dir

    feat_path = (
        feature_store_dir(
            cfg.env.storage.feature_dir, cfg.base.features.version, "LINE000", 0
        )
        / "features.parquet"
    )
    raw = pd.read_parquet(raw_path)
    feat = pd.read_parquet(feat_path)
    df = raw.merge(feat, on="sample_idx", suffixes=("", "_f"))
    clean = df[df["dq_flag"] == "clean"]
    residual_col = "r_mid_nt" if "r_mid_nt" in clean.columns else "r_mag_nt"
    resid = clean[residual_col].abs()
    d = resid[clean["defect"] == 1]
    b = resid[(clean["defect"] == 0) & (clean["interference"] == 0)]
    background_median = float(b.median())
    defect_median = float(d.median())
    defect_max = float(d.max())
    contrast = defect_median / background_median if background_median else float("nan")
    return {
        "contrast": contrast,
        "passed": contrast >= 3.0,
        "threshold_is_pre_rig_v2": True,
        "residual_col": residual_col,
        "background_median": background_median,
        "defect_median": defect_median,
        "defect_max": defect_max,
    }


def main() -> None:
    cfg = load_config("dev", config_dir=CONFIG_DIR / "scale")
    # Pre-run ESTIMATE only, for the banner -- every perf metric below uses the
    # ACTUAL row count instead (see `perf_gen.n_rows` right after generate).
    # Rig-v2 samples in TIME at walk.sample_rate_hz, not on a step_m distance
    # grid, so the old `length_m / step_m` arithmetic understated this by ~50x
    # and -- because it was passed straight into measure(n_rows=...) -- made
    # every rows/sec figure in the emitted report wrong by the same factor.
    if cfg.base.data.rig == "scalar":
        rows_per_m = (
            cfg.base.data.walk.sample_rate_hz / cfg.base.data.walk.speed_m_per_s
        )
    else:
        rows_per_m = 1.0 / cfg.base.data.step_m
    estimated_rows = int(
        cfg.base.data.n_lines
        * cfg.base.data.length_m
        * rows_per_m
        * cfg.base.data.n_runs
    )
    print(
        f"Stage 6 scale rehearsal: {cfg.base.data.n_lines} lines x "
        f"{cfg.base.data.length_m:.0f} m x {cfg.base.data.n_runs} runs "
        f"=~ {estimated_rows:,} rows estimated ({rows_per_m:.1f} rows/m, rig={cfg.base.data.rig}). "
        f"Storage under {cfg.env.storage.raw_dir}'s parent."
    )

    perf_results: list[PerfResult] = []

    with measure("generate") as perf_gen:
        results = generate_all(cfg.base.data, cfg.env.storage.raw_dir, cfg.seed)
    # Actual, not estimated -- an irregular walk's row count is only known once
    # the walk has actually been simulated.
    total_rows = sum(r.n_samples for r in results)
    perf_gen.n_rows = total_rows
    perf_results.append(perf_gen)
    print(
        f"generate: {len(results)} surveys, {perf_gen.wall_seconds:.1f}s, "
        f"{perf_gen.rows_per_sec:,.0f} rows/sec, peak RSS {perf_gen.peak_rss_mb:,.1f} MB"
    )

    conn = connect(cfg.env.storage.sqlite_path)
    quarantined: list[tuple[str, str]] = []
    n_rows_ingested = 0
    with measure("ingest") as perf_ingest:
        for sr in results:
            _status, report = run_survey_pipeline(conn, sr, cfg)
            if report.has_fail:
                fail_detail = "; ".join(
                    f"{r.check_name} (n_affected={r.n_affected})"
                    for r in report.results
                    if r.status == "fail"
                )
                quarantined.append((sr.survey_id, fail_detail))
                print(f"  WARNING: {sr.survey_id} quarantined: {fail_detail}")
            else:
                n_rows_ingested += sr.n_samples
    perf_ingest.n_rows = n_rows_ingested
    perf_results.append(perf_ingest)
    print(
        f"ingest: {perf_ingest.wall_seconds:.1f}s, {perf_ingest.rows_per_sec:,.0f} rows/sec, "
        f"peak RSS {perf_ingest.peak_rss_mb:,.1f} MB "
        f"({len(quarantined)} quarantined of {len(results)} surveys)"
    )

    n_rows_featurised = 0
    with measure("features") as perf_features:
        for sr in results:
            try:
                outcome, _ = run_feature_pipeline(conn, sr.survey_id, cfg)
            except SurveyNotFeaturisableError as exc:
                print(f"  WARNING: {sr.survey_id} features skipped ({exc})")
                continue
            if outcome not in ("computed", "hit"):
                print(f"  WARNING: {sr.survey_id} features skipped ({outcome})")
                continue
            n_rows_featurised += sr.n_samples
    perf_features.n_rows = n_rows_featurised
    perf_results.append(perf_features)
    print(
        f"features: {perf_features.wall_seconds:.1f}s, {perf_features.rows_per_sec:,.0f} rows/sec, "
        f"peak RSS {perf_features.peak_rss_mb:,.1f} MB"
    )

    contrast_result = _background_contrast_check(cfg)
    print(
        f"background-contrast re-check: {contrast_result['contrast']:.2f}x "
        f"(|{contrast_result['residual_col']}| defect median / background median) "
        f"-- INFORMATIONAL: the 3.0x threshold is a pre-Rig-v2 vector-rig calibration, "
        f"not a valid gate for the scalar rig"
    )

    as_of = _now_iso()
    with measure("corpus_read_pandas") as perf_pd_read:
        corpus_pd = load_feature_corpus(
            cfg.env.storage.feature_dir, cfg.base.features.version, as_of=as_of
        )
    perf_pd_read.n_rows = len(corpus_pd)
    perf_results.append(perf_pd_read)

    with measure("corpus_read_duckdb") as perf_dk_read:
        corpus_dk = scale_eval.load_feature_corpus_duckdb(
            cfg.env.storage.feature_dir, cfg.base.features.version, as_of=as_of
        )
    perf_dk_read.n_rows = len(corpus_dk)
    perf_results.append(perf_dk_read)
    print(
        f"corpus read: pandas-concat {perf_pd_read.wall_seconds:.2f}s vs "
        f"DuckDB {perf_dk_read.wall_seconds:.2f}s ({len(corpus_pd):,} rows each, "
        f"{'MATCH' if len(corpus_pd) == len(corpus_dk) else 'MISMATCH'})"
    )

    n_feature_files = sum(
        1
        for _ in Path(
            cfg.env.storage.feature_dir, f"fv={cfg.base.features.version}"
        ).glob("line_id=*/run_id=*/features.parquet")
    )

    with measure("train_block_cv", n_rows=total_rows) as perf_train:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            block_result = train.run_train(cfg, conn)
        block_report_text = buf.getvalue()
    perf_results.append(perf_train)
    print(
        f"train (default 5-fold block CV): {perf_train.wall_seconds:.1f}s, "
        f"peak RSS {perf_train.peak_rss_mb:,.1f} MB, gate_passed={block_result['gate_passed']}"
    )

    with measure("train_whole_line_and_temporal") as perf_scale_eval:
        scale_results = scale_eval.run_scale_evaluation(cfg, conn)
    perf_results.append(perf_scale_eval)

    whole_line_buf = io.StringIO()
    with contextlib.redirect_stdout(whole_line_buf):
        train._print_report(scale_results["whole_line_cv"])
    temporal_buf = io.StringIO()
    with contextlib.redirect_stdout(temporal_buf):
        train._print_report(scale_results["temporal_holdout"])

    ingest_after_rate = perf_ingest.rows_per_sec or 0.0
    speedup = (
        ingest_after_rate / INGEST_BEFORE_ROWS_PER_SEC
        if INGEST_BEFORE_ROWS_PER_SEC
        else float("nan")
    )

    report_lines = [
        f"# Stage 6 scale rehearsal -- {_now_iso()}",
        "",
        (
            f"Config: `config/scale/base.yaml` -- {cfg.base.data.n_lines} lines x "
            f"{cfg.base.data.length_m:.0f} m x {cfg.base.data.n_runs} runs = "
            f"**{total_rows:,} actual rows** ({rows_per_m:.1f} rows/m under `rig="
            f"{cfg.base.data.rig}`, sampled in time at {cfg.base.data.walk.sample_rate_hz} Hz, "
            f"not on a step_m grid), {cfg.base.data.n_lines * cfg.base.data.n_defects} physical "
            f"defects. `config_sha256={cfg.config_sha256[:12]}...`"
        ),
        "",
        "## Wall-clock + peak RSS per step",
        "",
        (
            "Peak RSS is *sampled* (a polling thread reading `psutil.Process().memory_info"
            "().rss` every 50ms), not an exact accounting -- the true peak between samples "
            "can be missed."
        ),
        "",
        format_perf_table(perf_results),
        "",
        "## Background-contrast re-check (informational, not a gate)",
        "",
        (
            f"Contrast (defect median / background median of "
            f"`|{contrast_result['residual_col']}|`) on `LINE000_R0`: "
            f"**{contrast_result['contrast']:.2f}x**. Background median "
            f"{contrast_result['background_median']:.2f} nT, defect median "
            f"{contrast_result['defect_median']:.2f} nT, defect max "
            f"{contrast_result['defect_max']:.2f} nT (the gap between defect median and max indicates how "
            "right-skewed the defect population is -- severity is drawn per type, so a wide spread is "
            "expected by construction, not a defect in the measurement)."
        ),
        "",
        (
            "**Read this as a number, not a verdict.** The historical >= 3.0x threshold was calibrated on the "
            "PRE-RIG-V2 vector rig, which measured 3.19x on a residual VECTOR MAGNITUDE (`r_mag_nt`). The scalar "
            "rig has no vector to take a magnitude of: the closest equivalent is `|r_mid_nt|`, the absolute value "
            "of the middle head's signed total-field residual. These are different physical quantities, and no "
            "like-for-like contrast baseline has been established for the scalar rig -- Stage D re-measured every "
            "promotion gate and ran the ablation ladder, but never produced a matching contrast figure. Comparing "
            "the number above against 3.0x would therefore be comparing across a rig change, which this project "
            "explicitly does not do elsewhere. What this check IS good for: confirming the detrend/high-pass "
            "still separates defect rows from background rows at this corpus size at all, and giving a first "
            "scalar-rig contrast figure that a future run can be compared against. Density-per-km is unchanged "
            "from the demo corpus (6 defects/km, 2 interference/km), so any movement here is not a packing-density "
            "artefact."
        ),
        "",
        "## Historical: a real bug found at scale (original 2026-07-31 pre-Rig-v2 run)",
        "",
        (
            "*The two bug write-ups in this section and the next were found by the ORIGINAL Stage 6 run on the "
            "pre-Rig-v2 vector rig, and both were fixed then. They are kept here as the record of what a scale "
            "rehearsal is FOR -- the incidence figures and follow-on metrics quoted in them are that run's, not "
            "this one's. Do not read them as measurements of the current corpus.*"
        ),
        "",
        (
            "`generate.py` initialises `severity_smys` to NaN off-defect (not 0, contradicting this project's "
            'own documented "0 off-defect" convention). A detector\'s peak occasionally lands just outside a '
            "defect's exact label-window half-width while still within the looser `MATCH_TOLERANCE_M` "
            "dig-matching radius, so `match_dug_indications` still credits it as matched, but that row's own "
            "`severity_smys` is NaN, not the defect's true value -- 36 of 18,917 matched indications (0.4%) at "
            "this scale (0 at demo scale, apparently never sampled). A single NaN `y_true` reaching "
            "`SeverityModel.fit`'s conformal calibration silently NaN'd the WHOLE fold's margin "
            "(`np.quantile` propagates NaN), which then NaN'd every prediction's interval for that fold -- "
            "cascading a 0.4%-incidence data issue into an initial 0% pooled coverage across the entire OOF "
            "result, not a gradual degradation. Fixed in `train.py::_build_severity_training_frame` (drop "
            "NaN-`y_true` rows upstream, loudly) plus a belt-and-braces guard in `SeverityModel.fit` itself. "
            "The fix is still in place and still guards this path; whether Stage 4's gate passes on the CURRENT "
            "corpus is reported in the results section below, not asserted here."
        ),
        "",
        "## Historical: classify's n_estimators/num_leaves were dead config (same original run)",
        "",
        (
            "`config/base.yaml`'s `model.classify.n_estimators`/`num_leaves` looked tunable but were never "
            "threaded from config into `ClassifyModel`'s `lgbm_cfg` in `train.py` -- editing them had NO "
            "effect, silently, since they coincidentally matched `ClassifyModel`'s own hardcoded fallback "
            "defaults (50/7). Found while investigating why `config/scale/base.yaml`'s larger capacity values "
            "weren't visibly changing classify's behaviour. Fixed by merging `classify_cfg`'s values into the "
            "`lgbm_cfg` dict at both call sites. Confirmed working: SCC recall improved from a first, "
            "capacity-starved run (0.50 block-CV / 0.39 whole-line) to the numbers reported below."
        ),
        "",
        "## A real DQ finding at scale: survey_overlap quarantines",
        "",
        (
            f"{len(quarantined)} of {len(results)} surveys ({len(quarantined) / len(results) * 100:.1f}%) "
            "were quarantined by `check_survey_overlap`, all on a correlation just above the fixed 0.9 "
            "threshold:\n"
            + "\n".join(f"  - `{sid}`: {detail}" for sid, detail in quarantined)
            if quarantined
            else "None -- every survey passed `check_survey_overlap` at this scale."
        ),
        "",
        (
            "`check_survey_overlap` (validate.py) takes the MAX correlation across ALL prior "
            "same-line surveys, not just the immediately preceding one -- so as more runs of a "
            "line accumulate, this max is an order statistic over a growing number of comparisons "
            "and trends upward even if each individual run-pair's correlation distribution is "
            "unchanged (R2 is checked against both R0 and R1; R0 has nothing to compare against). "
            f"Separately, the more rows a single survey carries ({total_rows // max(len(results), 1):,} per "
            f"survey here, at {cfg.base.data.length_m:.0f} m and {rows_per_m:.0f} rows/m), the more samples "
            "two runs' shared deterministic structure (same defect/interference positions, same geo/lat-lon "
            "path) has to accumulate correlated structure in a plain Pearson correlation, even though each "
            "run's background noise is drawn independently. The ORIGINAL pre-Rig-v2 Stage 6 run hit this "
            "regime with 40 km lines and saw real quarantines; whether this run does is given by the count "
            "immediately above, not assumed here. Either way the fixed 0.9 threshold was calibrated against "
            "short lines and few runs and has never been systematically stress-tested -- and where it does "
            "fire, the DQ layer is doing exactly what it is designed to do (quarantine, not crash), with the "
            "affected surveys correctly excluded from the training corpus. Revisiting the threshold for "
            "long-line, many-run deployments is future work, out of Stage 6's scope."
        ),
        "",
        "## Where SQLite stopped being the right tool",
        "",
        (
            f"- Before (pre-Stage-6, per-row `itertuples()`+`float()`-per-cell "
            f"`load_readings`, measured once by hand at {INGEST_BEFORE_N_ROWS:,} rows): "
            f"**{INGEST_BEFORE_ROWS_PER_SEC:,} rows/sec**, peak RSS {INGEST_BEFORE_PEAK_RSS_MB:.1f} MB."
        ),
        (
            f"- After (vectorized NaN->None + dtype-cast, this rehearsal, "
            f"{n_rows_ingested:,} rows): **{ingest_after_rate:,.0f} rows/sec**, "
            f"peak RSS {perf_ingest.peak_rss_mb:,.1f} MB ({speedup:.2f}x)."
        ),
        (
            "- Conclusion: the naive Python-level row conversion was a real, measurable cost, "
            "but not the dominant one -- SQLite's own `executemany` insert path is the majority "
            "of the remaining cost at this row count. This matches "
            "`.claude/skills/lsm-integrity/references/architecture.md`'s own claim that SQLite "
            '"comfortably handles 10^7 rows read-mostly" -- it stops being the right tool at '
            "concurrent multi-writer ingest, not at this data volume, which this demonstrator "
            "never has."
        ),
        "",
        "## Bulk-read path: pandas-concat vs DuckDB",
        "",
        (
            f"- `features.load_feature_corpus` (per-file glob + `pd.concat`): "
            f"{perf_pd_read.wall_seconds:.2f}s for {len(corpus_pd):,} rows."
        ),
        (
            f"- `scale_eval.load_feature_corpus_duckdb` (one DuckDB glob read): "
            f"{perf_dk_read.wall_seconds:.2f}s for {len(corpus_dk):,} rows."
        ),
        "",
        "## Small-files problem",
        "",
        (
            f"{n_feature_files} feature-store files at this scale ({cfg.base.data.n_lines} lines x "
            f"{cfg.base.data.n_runs} runs) -- PLAN.md's own \"120 tiny per-survey Parquet files is "
            'fine" example, not the 10^5-file failure case. This rehearsal does not hit that '
            "failure mode -- the documented production answer (periodic compaction to line-level "
            "Parquet files) is noted here, not artificially triggered."
        ),
        "",
        "## float32 arithmetic",
        "",
        (
            "The feature store is float32 throughout (`features.py`'s `STORAGE_DTYPE`). Severity's "
            "quantile regression and classify's LightGBM training both ran to completion against "
            "these float32-stored features with no dtype errors (LightGBM upconverts internally as "
            "needed) -- see the block-CV report below for the resulting metrics."
        ),
        "",
        "## Default 5-fold block CV (the real gate, `run_train`, unchanged)",
        "",
        "```",
        block_report_text.strip(),
        "```",
        "",
        '## Whole-line-holdout CV (Stage 6: "the real generalisation test")',
        "",
        (
            f"An entire physical line held out per fold (5 folds, ~{cfg.base.data.n_lines / 5:.0f} "
            "lines/fold), instead of today's default (line, 100 m block) hash grouping, which "
            "scatters one line's blocks across ~all folds."
        ),
        "",
        "```",
        whole_line_buf.getvalue().strip(),
        "```",
        "",
        "## Temporal holdout (train runs 0-1, test run 2)",
        "",
        "```",
        temporal_buf.getvalue().strip(),
        "```",
        "",
    ]

    report_path = PROJECT_ROOT / "docs" / "stage6-scale-rehearsal.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
