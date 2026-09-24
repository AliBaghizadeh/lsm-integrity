"""
Golden regression test: a frozen tiny survey (tests/golden/tiny_survey.parquet,
generated once with seed=1234, length_m=200, step_m=1.0) with recorded expected
DQ outcomes. If a refactor to validate.py silently changes behaviour, this is
what catches it -- these numbers should never change without a deliberate,
reviewed reason.

Rig-v2 (2026-08-06, Stage A): the fixture was REGENERATED against the new
scalar-rig schema (b_lo/b_mid/b_hi_nt, t_s, chainage_true_m, nullable lat/lon,
girth_weld -- see schemas.py) using the same seed=1234/length_m=200/step_m=1.0
convention, plus n_defects=2/n_interference=1/walk.sample_rate_hz=2.0 (a
deliberately slow sample rate so a 200 m survey stays a genuinely tiny
fixture -- see conftest.py's tiny_cfg for the same reasoning). Content hash
and row count below are DELIBERATELY updated, recorded here rather than
silently changed -- see the Rig-v2 plan's Stage A5 note that a silently-
updated golden hash would be the one thing that makes this stage look
dishonest.

EXPECTED_CONTENT_SHA256 was updated a second time (still 2026-08-06, same
fixture file, same row count) after discovering hashing.py's CONTENT_COLUMNS
still listed only the old vector-rig columns (bx_nt/by_nt/bz_nt/...) -- none
present on this fixture, so `canonicalise()`'s `if c in df.columns` filter
was silently hashing ONLY sample_idx/defect/defect_type/severity_smys/
interference, ignoring every field reading, GPS position, chainage and
girth_weld label. Fixed by listing both rigs' content columns in
hashing.py; this is the hash of the SAME fixture bytes under the corrected,
actually-content-sensitive hash function.

`test_golden_survey_validates_clean` is EXPECTED TO FAIL until Stage C
updates validate.py for the new raw schema -- validate.py still reads
bx_nt/by_nt/bz_nt directly (a KeyError on this fixture) and FIELD_RANGE_NT
assumptions that no longer apply. That is deliberate downstream breakage
Stage A does not fix (see generate.py's module docstring and the Rig-v2 plan),
left here as a visible failure rather than papered over.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from lsm.generate import SurveyResult
from lsm.hashing import content_sha256, file_sha256
from lsm.ingest import register_survey
from lsm.validate import validate_raw_survey

GOLDEN_PATH = Path(__file__).parent / "golden" / "tiny_survey.parquet"
EXPECTED_CONTENT_SHA256 = (
    "b2576e3012e91f51a85bf099d7976a9757ad5c1fc597ad215d3e938270cd53cc"
)
# No longer exactly 200: under rig: scalar the walk is TIME-sampled, not a
# step_m distance grid, so row count is an emergent property of the walk
# (speed, sample_rate_hz) rather than length_m/step_m. See the module
# docstring above for the exact fixture-generation config.
EXPECTED_N_SAMPLES = 332
EXPECTED_STATUSES = {
    "schema": "pass",
    "range": "pass",
    "saturation": "pass",
    "sample_idx_monotonic": "pass",
    "sample_idx_gap": "pass",
    "duplicate_sample_idx": "pass",
    "duplicate_content": "pass",
    "survey_overlap": "pass",
    # gps_jump/gps_chainage_consistency/noise_floor: updated pass -> warn
    # 2026-08-06 (Stage B harness fix) after actually exercising this path for
    # the first time -- the KeyError this file's module docstring describes
    # meant these three were never empirically verified against real
    # validate_raw_survey output before now, only guessed. All three are
    # warn-gated (config/base.yaml), so this does not flip report.has_fail.
    # Root cause is the SAME for all three, and it is real, not a bug: this
    # fixture uses walk.sample_rate_hz=2.0 (conftest.py's tiny_cfg
    # convention, ~0.6 m between consecutive samples) instead of the
    # production default of 120 Hz (~1 cm). check_gps_jump/
    # check_gps_chainage_consistency difference consecutive LOCKED GPS fixes
    # without normalising by elapsed time -- fine at 120 Hz where true
    # per-sample motion (~1 cm) is negligible next to GPS noise, but at 2 Hz
    # the ~1.5 m per-fix noise (GpsConfig.sigma_m) is comparable to or larger
    # than the ~0.6 m of real motion per sample, so the noise itself
    # dominates the diffed distance -- verified directly: this fixture's GPS
    # is 100% locked (zero real dropouts), yet the naive summed consecutive-
    # fix distance is ~961 m against a true 200 m span, and a handful of
    # per-step noise excursions exceed the flat 5 m gps_jump threshold purely
    # from that noise, not from any receiver glitch. check_noise_floor's
    # first-difference proxy has the same granularity dependency: at 2 Hz
    # spacing, real background/defect/weld structure changes measurably
    # sample-to-sample, inflating the proxy above its noise_floor_range_nT
    # band calibrated for near-instantaneous (120 Hz) sampling -- exactly the
    # "coarse Stage-1 proxy, not the final Stage-2 feature" caveat this
    # module's own docstring already states.
    "gps_jump": "warn",
    "gps_chainage_consistency": "warn",
    "noise_floor": "warn",
    "background_regime": "pass",
    "interference_density": "pass",
    "coverage": "pass",
}


def test_golden_survey_content_hash_is_stable():
    """If this fails, either the fixture changed or content_sha256's
    canonicalisation changed -- both are things a refactor must not do silently.
    """
    df = pd.read_parquet(GOLDEN_PATH)
    assert len(df) == EXPECTED_N_SAMPLES
    assert content_sha256(df) == EXPECTED_CONTENT_SHA256


def test_golden_survey_validates_clean(cfg):
    df = pd.read_parquet(GOLDEN_PATH)
    sr = SurveyResult(
        survey_id="LINE000_R0",
        line_id="LINE000",
        run_id=0,
        path=GOLDEN_PATH,
        file_sha256=file_sha256(GOLDEN_PATH),
        content_sha256=content_sha256(df),
        step_m=1.0,
        n_samples=len(df),
        chainage_start_m=float(df["chainage_true_m"].min()),
        chainage_end_m=float(df["chainage_true_m"].max()),
        standoff_m=1.5,
        surveyed_at="unknown",
    )

    from lsm.db import connect

    conn = connect(cfg.env.storage.sqlite_path)
    register_survey(conn, sr, schema_version=cfg.base.schema_version)
    report = validate_raw_survey(
        conn,
        sr.survey_id,
        sr.line_id,
        sr.step_m,
        sr.chainage_start_m,
        sr.chainage_end_m,
        sr.content_sha256,
        df,
        cfg.base.validate,
    )

    actual_statuses = {r.check_name: r.status for r in report.results}
    assert actual_statuses == EXPECTED_STATUSES
    assert not report.has_fail
