"""
Stage D headline: the ablation ladder.

The engineering question this experiment addresses is: "can data scientists get more
out of the EXISTING 3-head scalar rod, or do you need new hardware?" Six
arms, isolating one algorithmic capability at a time, evaluated with the
SAME machinery (`evaluate.bootstrap_ci` / `evaluate.paired_bootstrap_ci`) so
every arm-to-arm delta carries a confidence interval, not just a point
estimate (SKILL invariant #10):

  1. middle head only, GPS-only chainage       -- the honest floor
  2. + first difference g1                     -- common-mode rejection
  3. + second difference g2                     -- linear-gradient rejection,
                                                    the THIRD head's specific
                                                    value over a two-head
                                                    gradiometer
  4. + stand-off inversion & normalisation      -- what MEASURING height
                                                    buys over ASSUMING it
  5. + weld-comb registration                    -- metre -> centimetre
                                                    localisation
  6. full 3-axis vector output (rig: vector)     -- what a HARDWARE upgrade
                                                    would buy

Arms 1-5 share ONE real Rig-v2 (`rig: scalar`) corpus -- they differ only in
which already-computed feature COLUMNS the detector is allowed to see and
which chainage axis (GPS-only dead-reckoned vs weld-comb registered) is used,
never in what physically exists on the rod. Arm 6 is structurally different
hardware (`data.rig: vector`, the preserved pre-Rig-v2 generator -- see
generate.py's module docstring) and needs its own minimal, clearly-labelled
feature/scoring path, since `features.py` only supports `rig: scalar` by
design (its own module docstring: "this module does not stay polymorphic
across both raw schemas").

SAFETY: this script NEVER touches `data/raw`, `data/lsm.db` or any other
production path -- every survey it generates goes under an isolated tmp
directory, and truth labels are read directly off the generator's own raw
Parquet columns (bypassing SQLite entirely), not off any shared database.
This matters: an earlier exploratory script in this same effort accidentally
wrote a differently-sized corpus straight into `data/raw` (production) under
the same LINE000 survey_id and corrupted it, caught and fixed by
regenerating from the same seed+config (generation is a pure function of
the two). Never repeat that mistake -- always pass an isolated `raw_dir`.

SCALE: `config/base.yaml`'s literal default is 5 lines x 2000 m x 3 runs
(~3M rows for the scalar rig). Fitting 5 separate IsolationForests (one per
scalar arm, 5-fold grouped CV, n_estimators=300 each) at that size is a
30-60+ minute proposition. This script instead uses ABLATION_N_LINES lines
(default 2) at the SAME per-line length/defect/interference/run density as
production -- a real, non-toy scale-down, not a toy corpus -- documented
here rather than silently run at a different size. Bump ABLATION_N_LINES
back to 5 to reproduce the full-corpus numbers, at the cost of runtime.

Not wired into CI, not a pytest test -- a real, one-time measurement script,
matching scripts/stage6_scale_rehearsal.py's own house style.

Usage:
    python scripts/ablation_ladder.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

from lsm import train as train_module
from lsm.config import load_config
from lsm.evaluate import add_fold_column, paired_bootstrap_ci
from lsm.features import (
    SurveyContext,
    _mean_step_m,
    _odd_window,
    _peak_shape,
    compute_survey_features,
    detrend_axis,
    window_name,
)
from lsm.generate import generate_all
from lsm.registration import _dead_reckon_chainage, register_survey
from lsm.truth import build_truth_registry

# Deliberate scale-down from the production default (5 lines) -- see module
# docstring's "SCALE" paragraph. Density per line (length_m, n_defects,
# n_interference, n_runs) is left at the real production values.
ABLATION_N_LINES = 2


def _isolated_config(rig: str):
    """A Config pointed at a throwaway tmp directory, NEVER `data/raw` or
    `data/lsm.db` -- see module docstring's SAFETY note.
    """
    cfg = load_config("dev")
    tmp = Path(tempfile.mkdtemp(prefix=f"lsm_ablation_{rig}_"))
    cfg.env.storage.raw_dir = str(tmp / "raw")
    cfg.env.storage.sqlite_path = str(
        tmp / "lsm.db"
    )  # unused (no ingest), set for hygiene only
    cfg.env.storage.feature_dir = str(tmp / "features")
    cfg.env.storage.model_dir = str(tmp / "models")
    cfg.env.storage.reports_dir = str(tmp / "reports")
    cfg.base.data.n_lines = ABLATION_N_LINES
    cfg.base.data.rig = rig
    return cfg


def _truth_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Truth labels straight off the raw survey's own columns -- no SQLite
    ingest needed for this evaluation script (see module docstring's SAFETY
    note: `reading`/`survey` tables are production-DB concerns this script
    deliberately never touches).
    """
    return df[
        ["sample_idx", "defect", "defect_type", "interference", "severity_smys"]
    ].copy()


# ---------------------------------------------------------------------------
# Arms 1-5: the real Rig-v2 scalar rig, varying feature columns + chainage
# ---------------------------------------------------------------------------

# Column groups, in the ladder's own order -- see feature_columns()'s
# docstring in features.py for what each one physically is.
_WINDOW_STAT_SUFFIXES = (
    "_mean_nt",
    "_std_nt",
    "_max_nt",
    "_ptp_nt",
    "_kurt",
    "_zcr",
    "_energy_nt2",
)
_PEAK_SHAPE_COLS = [
    "fwhm_m",
    "peak_asymmetry",
    "decay_exponent",
    "peak_prominence_nt",
    "peak_distance_m",
]


def _mid_only_cols(cfg) -> list[str]:
    """Arm 1's floor: everything derived from r_mid_nt alone -- no other
    head's information (not even as a difference), no registration.
    """
    cols = ["dr_ds_nt_per_m", "d2r_ds2_nt_per_m2"]
    for w in cfg.base.features.windows_m:
        p = window_name(w)
        cols += [f"{p}{suffix}" for suffix in _WINDOW_STAT_SUFFIXES]
    cols += _PEAK_SHAPE_COLS
    return cols


def _scalar_corpus_with_gps_chainage(cfg) -> tuple[pd.DataFrame, list, dict]:
    """Generate the scalar-rig corpus once and compute its FULL feature set
    (all fv=3 columns) using GPS-ONLY (dead-reckoned, never weld-locked)
    chainage -- the axis arms 1-4 evaluate against. Returns (corpus,
    survey_results, run_line_id).
    """
    results = generate_all(cfg.base.data, cfg.env.storage.raw_dir, seed=cfg.seed)
    frames = []
    for sr in results:
        df = pd.read_parquet(sr.path)
        n = len(df)
        dr_chainage, _locked_frac, _max_gap = _dead_reckon_chainage(
            df["t_s"].to_numpy(dtype=float),
            df["lat"].to_numpy(dtype=float),
            df["lon"].to_numpy(dtype=float),
            cfg.base.data,
        )
        ctx = SurveyContext(
            survey_id=sr.survey_id,
            line_id=sr.line_id,
            run_id=sr.run_id,
            surveyed_at=sr.surveyed_at,
            standoff_m=sr.standoff_m,
            array_spacing_m=cfg.base.data.array.spacing_m,
        )
        feat = compute_survey_features(
            df,
            ctx,
            cfg.base.features,
            chainage_m=dr_chainage,
            dist_to_weld_m=np.full(n, np.nan),
        )
        feat = feat.merge(
            _truth_columns(df), on="sample_idx", how="left", validate="one_to_one"
        )
        frames.append(feat)
    corpus = pd.concat(frames, ignore_index=True)
    run_line_id = (
        corpus.drop_duplicates("survey_id").set_index("survey_id")["line_id"].to_dict()
    )
    return corpus, results, run_line_id


def _scalar_corpus_with_registration(cfg, results) -> pd.DataFrame:
    """Arm 5: the SAME raw surveys, this time with the real weld-comb-locked
    chainage_m + genuine dist_to_weld_m -- exactly what `lsm features` does
    in production (`pipeline.run_feature_pipeline`), just called directly
    here rather than through the DB-backed pipeline.
    """
    frames = []
    for sr in results:
        df = pd.read_parquet(sr.path)
        reg = register_survey(df, cfg.base.data)
        ctx = SurveyContext(
            survey_id=sr.survey_id,
            line_id=sr.line_id,
            run_id=sr.run_id,
            surveyed_at=sr.surveyed_at,
            standoff_m=sr.standoff_m,
            array_spacing_m=cfg.base.data.array.spacing_m,
        )
        feat = compute_survey_features(
            df,
            ctx,
            cfg.base.features,
            chainage_m=reg.chainage_m,
            dist_to_weld_m=reg.dist_to_weld_m,
        )
        feat = feat.merge(
            _truth_columns(df), on="sample_idx", how="left", validate="one_to_one"
        )
        frames.append(feat)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Arm 6: the preserved pre-Rig-v2 vector rig, its own minimal feature path
# ---------------------------------------------------------------------------


def _vector_corpus(cfg) -> tuple[pd.DataFrame, dict]:
    """Standalone, deliberately lightweight feature computation for
    `rig: vector` raw data -- NOT production-grade, NOT added to
    features.py (whose own module docstring says the vector rig gets "its
    own simpler feature path if it ever needs one" -- this is it, and its
    only job is this one reference arm).

    Reuses features.py's genuinely rig-agnostic pieces directly
    (`detrend_axis`, `_peak_shape`) rather than re-deriving them -- both
    already operate on a bare (s_m, y) array pair, nothing scalar-rig-
    specific about either.

    Feature set is the direct vector analogue of the pre-Rig-v2 columns
    `feature_columns()`'s own docstring says were removed at fv=3 (rx_nt,
    ry_nt, rz_nt, r_incl_deg, r_decl_deg, r_mag_nt, dr_ds, d2r_ds2, window
    stats, peak shape) -- the genuinely NEW information a vector head buys
    over a scalar one is direction (inclination/declination of the residual
    vector), which a total-field magnitude can never recover.
    """
    array_cfg = cfg.base.data.array.model_copy(update={"n_heads": 1})
    cfg.base.data.array = array_cfg
    results = generate_all(cfg.base.data, cfg.env.storage.raw_dir, seed=cfg.seed)

    frames = []
    for sr in results:
        df = pd.read_parquet(sr.path)
        n = len(df)
        s_m = df["chainage_m"].to_numpy(dtype=np.float64)
        step_m = _mean_step_m(s_m)

        rx = detrend_axis(s_m, df["bx_nt"].to_numpy(), cfg.base.features)
        ry = detrend_axis(s_m, df["by_nt"].to_numpy(), cfg.base.features)
        rz = detrend_axis(s_m, df["bz_nt"].to_numpy(), cfg.base.features)
        r_mag = np.sqrt(rx**2 + ry**2 + rz**2)
        # Direction of the residual vector -- information a scalar total-
        # field rig structurally cannot produce (module docstring point 2).
        with np.errstate(invalid="ignore", divide="ignore"):
            r_incl_deg = np.degrees(
                np.arcsin(np.clip(rz / np.where(r_mag > 0, r_mag, np.nan), -1.0, 1.0))
            )
        r_decl_deg = np.degrees(np.arctan2(ry, rx))

        s_grad = (
            s_m + np.arange(n) * 5e-9
        )  # tie-break only, same convention as features.py
        dr_ds = np.gradient(r_mag, s_grad)
        d2r_ds2 = np.gradient(dr_ds, s_grad)

        out: dict[str, np.ndarray] = {
            "rx_nt": rx,
            "ry_nt": ry,
            "rz_nt": rz,
            "r_mag_nt": r_mag,
            "r_incl_deg": np.nan_to_num(r_incl_deg, nan=0.0),
            "r_decl_deg": r_decl_deg,
            "dr_ds_nt_per_m": dr_ds,
            "d2r_ds2_nt_per_m2": d2r_ds2,
        }
        r_series = pd.Series(r_mag)
        sign_change = np.zeros(n, dtype=np.float64)
        if n > 1:
            sign_change[1:] = (np.signbit(rz[1:]) != np.signbit(rz[:-1])).astype(
                np.float64
            )
        zc_series = pd.Series(sign_change)
        energy_series = pd.Series(r_mag**2)
        for w in cfg.base.features.windows_m:
            win = _odd_window(w, step_m, minimum=3)
            p = window_name(w)
            roll = r_series.rolling(win, center=True, min_periods=win)
            out[f"{p}_mean_nt"] = roll.mean().to_numpy()
            out[f"{p}_std_nt"] = roll.std().to_numpy()
            out[f"{p}_max_nt"] = roll.max().to_numpy()
            out[f"{p}_ptp_nt"] = (roll.max() - roll.min()).to_numpy()
            out[f"{p}_kurt"] = (
                roll.kurt().to_numpy() if win >= 4 else np.full(n, np.nan)
            )
            out[f"{p}_zcr"] = (
                zc_series.rolling(win, center=True, min_periods=win).mean().to_numpy()
            )
            out[f"{p}_energy_nt2"] = (
                energy_series.rolling(win, center=True, min_periods=win)
                .sum()
                .to_numpy()
            )

        out |= _peak_shape(r_mag, step_m, cfg.base.features)

        feat = pd.DataFrame(out)
        feat["survey_id"] = sr.survey_id
        feat["line_id"] = sr.line_id
        feat["run_id"] = sr.run_id
        feat["sample_idx"] = df["sample_idx"].to_numpy()
        feat["chainage_m"] = s_m
        # dq_flag: edge rows only (same detrend-window logic as features.py),
        # a data-quality companion, not a physics feature.
        detrend_win = _odd_window(cfg.base.features.detrend.window_m, step_m, minimum=3)
        max_win = max(
            _odd_window(w, step_m, minimum=3) for w in cfg.base.features.windows_m
        )
        edge = max(max_win, detrend_win) // 2
        dq_flag = np.full(n, "clean", dtype=object)
        if edge > 0:
            dq_flag[:edge] = "edge"
            dq_flag[max(0, n - edge) :] = "edge"
        feat["dq_flag"] = dq_flag
        feat = feat.merge(
            _truth_columns(df), on="sample_idx", how="left", validate="one_to_one"
        )
        frames.append(feat)

    corpus = pd.concat(frames, ignore_index=True)
    run_line_id = (
        corpus.drop_duplicates("survey_id").set_index("survey_id")["line_id"].to_dict()
    )
    return corpus, run_line_id


def _vector_feature_cols(cfg) -> list[str]:
    """Arm 6's full feature list: the vector-specific columns (direction
    information a scalar rig cannot produce) plus the same window-stat/
    peak-shape shape features every scalar arm gets, computed over the
    vector rig's own r_mag_nt -- see `_vector_corpus`'s docstring.
    """
    cols = [
        "rx_nt",
        "ry_nt",
        "rz_nt",
        "r_mag_nt",
        "r_incl_deg",
        "r_decl_deg",
        "dr_ds_nt_per_m",
        "d2r_ds2_nt_per_m2",
    ]
    for w in cfg.base.features.windows_m:
        p = window_name(w)
        cols += [f"{p}{suffix}" for suffix in _WINDOW_STAT_SUFFIXES]
    cols += _PEAK_SHAPE_COLS
    return cols


# ---------------------------------------------------------------------------
# Shared arm-evaluation machinery
# ---------------------------------------------------------------------------


def _evaluate_arm(
    corpus: pd.DataFrame,
    feature_cols: list[str],
    cfg,
    registry: pd.DataFrame,
    run_line_id: dict,
    seed: int,
    mad_residual_col: str = "r_mid_nt",
) -> dict:
    """One arm's IsolationForest recall@budget / false-dig-rate /
    localisation-error-cm, with bootstrap CIs, plus the raw per-defect hit
    rates (for paired arm-to-arm deltas). Reuses train.py's own private CV/
    dig/match/bootstrap helpers directly -- the same pattern growth.py and
    scale_eval.py already established for this codebase.
    """
    split_cfg = cfg.base.model.split
    corpus = add_fold_column(
        corpus,
        "line_id",
        "chainage_m",
        block_m=split_cfg.fallback_block_m,
        n_folds=split_cfg.n_folds,
    )
    # _run_grouped_cv hardcodes MADBaseline(residual_col="r_mid_nt") -- alias
    # the vector rig's own r_mag_nt onto that name so the SAME shared helper
    # works unchanged for arm 6 too (r_mag_nt is r_mid_nt's exact analogue:
    # the one channel a scalar OR vector single head both directly provide).
    if mad_residual_col != "r_mid_nt":
        corpus = corpus.copy()
        corpus["r_mid_nt"] = corpus[mad_residual_col]
    corpus = train_module._run_grouped_cv(corpus, feature_cols, cfg, seed=seed)
    matched_if = train_module._dig_and_match(corpus, "score_if", 0.0, registry, cfg)
    metrics, hit_rates, _false_digs, _interference = train_module._bootstrap_metrics(
        matched_if, registry, run_line_id, cfg
    )
    return {"metrics": metrics, "hit_rates": hit_rates}


def _fmt_ci(triple: tuple[float, float, float]) -> str:
    point, lo, hi = triple
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"


def _paired_delta(
    arm_a: dict, arm_b: dict, boot_cfg, seed: int
) -> tuple[float, float, float] | None:
    """recall(b) - recall(a), paired over the SAME defect set -- only valid
    when both arms share the same physical-defect universe (arms 1-5, all
    scored against the same scalar-rig corpus's registry). Returns None if
    the defect sets differ (arm 6 is a structurally different corpus).
    """
    ids_a, ids_b = set(arm_a["hit_rates"]), set(arm_b["hit_rates"])
    if ids_a != ids_b:
        return None
    ids = sorted(ids_a)
    return paired_bootstrap_ci(
        [arm_b["hit_rates"][d] for d in ids],
        [arm_a["hit_rates"][d] for d in ids],
        boot_cfg.n_resamples,
        boot_cfg.level,
        seed,
    )


def main() -> None:
    print(
        f"Ablation ladder -- {ABLATION_N_LINES} lines x production per-line density "
        "(see module docstring's SCALE note for why not the full 5-line default).\n"
    )

    # -- Arms 1-4: one scalar-rig corpus, GPS-only chainage -----------------
    cfg_scalar = _isolated_config("scalar")
    boot_cfg = cfg_scalar.base.model.bootstrap
    corpus_gps, results, run_line_id = _scalar_corpus_with_gps_chainage(cfg_scalar)
    print(
        f"Scalar-rig corpus (arms 1-4, GPS-only chainage): {len(corpus_gps):,} rows, "
        f"{len(results)} surveys."
    )

    line_ids = sorted(corpus_gps["line_id"].unique())
    registries_gps = []
    for line_id in line_ids:
        ref_survey_id = min(
            corpus_gps.loc[corpus_gps["line_id"] == line_id, "survey_id"].unique()
        )
        ref_rows = corpus_gps[corpus_gps["survey_id"] == ref_survey_id]
        registries_gps.append(
            build_truth_registry(ref_rows, line_id, ref_rows["chainage_m"].to_numpy())
        )
    registry_gps = pd.concat(registries_gps, ignore_index=True)

    mid_cols = _mid_only_cols(cfg_scalar)
    arm1_cols = mid_cols
    arm2_cols = mid_cols + ["g1_nt_per_m"]
    arm3_cols = arm2_cols + ["g2_nt_per_m2"]
    arm4_cols = arm3_cols + [
        "standoff_est_m",
        "r_mag_norm_nt_m3",
        "peak_prominence_norm_nt_m3",
    ]

    print("Arm 1 (mid-head only, GPS chainage)...")
    arm1 = _evaluate_arm(
        corpus_gps,
        arm1_cols,
        cfg_scalar,
        registry_gps,
        run_line_id,
        seed=cfg_scalar.seed + 100,
    )
    print("  " + _fmt_ci(arm1["metrics"]["recall_at_budget"]))

    print("Arm 2 (+ g1, GPS chainage)...")
    arm2 = _evaluate_arm(
        corpus_gps,
        arm2_cols,
        cfg_scalar,
        registry_gps,
        run_line_id,
        seed=cfg_scalar.seed + 200,
    )
    print("  " + _fmt_ci(arm2["metrics"]["recall_at_budget"]))

    print("Arm 3 (+ g2, GPS chainage)...")
    arm3 = _evaluate_arm(
        corpus_gps,
        arm3_cols,
        cfg_scalar,
        registry_gps,
        run_line_id,
        seed=cfg_scalar.seed + 300,
    )
    print("  " + _fmt_ci(arm3["metrics"]["recall_at_budget"]))

    print("Arm 4 (+ stand-off inversion/normalisation, GPS chainage)...")
    arm4 = _evaluate_arm(
        corpus_gps,
        arm4_cols,
        cfg_scalar,
        registry_gps,
        run_line_id,
        seed=cfg_scalar.seed + 400,
    )
    print("  " + _fmt_ci(arm4["metrics"]["recall_at_budget"]))

    # -- Arm 5: SAME feature set as arm 4, real weld-comb registration ------
    print("Computing registered-chainage feature set for arm 5...")
    corpus_reg = _scalar_corpus_with_registration(cfg_scalar, results)
    registries_reg = []
    for line_id in line_ids:
        ref_survey_id = min(
            corpus_reg.loc[corpus_reg["line_id"] == line_id, "survey_id"].unique()
        )
        ref_rows = corpus_reg[corpus_reg["survey_id"] == ref_survey_id]
        registries_reg.append(
            build_truth_registry(ref_rows, line_id, ref_rows["chainage_m"].to_numpy())
        )
    registry_reg = pd.concat(registries_reg, ignore_index=True)
    arm5_cols = arm4_cols + ["dist_to_weld_m"]
    print("Arm 5 (+ weld-comb registration)...")
    arm5 = _evaluate_arm(
        corpus_reg,
        arm5_cols,
        cfg_scalar,
        registry_reg,
        run_line_id,
        seed=cfg_scalar.seed + 500,
    )
    print("  " + _fmt_ci(arm5["metrics"]["recall_at_budget"]))

    # -- Arm 6: the preserved vector rig, its own corpus/registry -----------
    print("Generating rig: vector reference corpus for arm 6...")
    cfg_vector = _isolated_config("vector")
    corpus_vec, run_line_id_vec = _vector_corpus(cfg_vector)
    print(f"Vector-rig corpus (arm 6): {len(corpus_vec):,} rows.")
    line_ids_vec = sorted(corpus_vec["line_id"].unique())
    registries_vec = []
    for line_id in line_ids_vec:
        ref_survey_id = min(
            corpus_vec.loc[corpus_vec["line_id"] == line_id, "survey_id"].unique()
        )
        ref_rows = corpus_vec[corpus_vec["survey_id"] == ref_survey_id]
        registries_vec.append(
            build_truth_registry(ref_rows, line_id, ref_rows["chainage_m"].to_numpy())
        )
    registry_vec = pd.concat(registries_vec, ignore_index=True)
    print("Arm 6 (full 3-axis vector output)...")
    arm6 = _evaluate_arm(
        corpus_vec,
        _vector_feature_cols(cfg_vector),
        cfg_vector,
        registry_vec,
        run_line_id_vec,
        seed=cfg_vector.seed + 600,
        mad_residual_col="r_mag_nt",
    )
    print("  " + _fmt_ci(arm6["metrics"]["recall_at_budget"]))

    # -- Report ---------------------------------------------------------------
    arms = [
        ("1. mid-head only, GPS chainage", arm1),
        ("2. + first difference g1", arm2),
        ("3. + second difference g2", arm3),
        ("4. + stand-off inversion/normalisation", arm4),
        ("5. + weld-comb registration", arm5),
        ("6. full 3-axis vector output (hardware)", arm6),
    ]

    lines = [
        "",
        "=" * 100,
        "ABLATION LADDER -- IsolationForest, grouped CV, out-of-fold",
        "=" * 100,
    ]
    for name, arm in arms:
        m = arm["metrics"]
        lines.append(f"\n{name}")
        lines.append(f"  recall @ dig budget      {_fmt_ci(m['recall_at_budget'])}")
        lines.append(f"  false-dig rate           {_fmt_ci(m['false_dig_rate'])}")
        lines.append(
            f"  localisation error (cm)  {_fmt_ci(m['localisation_error_cm'])}"
        )

    lines.append("\n" + "-" * 100)
    lines.append(
        "Arm-to-arm recall deltas (paired bootstrap where the defect universe is shared; "
        "arm 5->6 is NOT paired -- rig:vector is a structurally different corpus/generator, "
        "compare via CI overlap only, not a paired delta):"
    )
    pairs = [
        ("1->2", arm1, arm2),
        ("2->3", arm2, arm3),
        ("3->4", arm3, arm4),
        ("4->5", arm4, arm5),
        ("1->5 (software total)", arm1, arm5),
    ]
    for label, a, b in pairs:
        delta = _paired_delta(a, b, boot_cfg, seed=9001)
        if delta is not None:
            lines.append(f"  {label:28s} {_fmt_ci(delta)}")
        else:
            lines.append(f"  {label:28s} not paired (different defect universe)")

    r5 = arm5["metrics"]["recall_at_budget"]
    r6 = arm6["metrics"]["recall_at_budget"]
    lines.append(
        f"\n  5->6 (hardware headline)     software {_fmt_ci(r5)} vs hardware {_fmt_ci(r6)}"
        f" -- independent CIs, point gap {r6[0] - r5[0]:+.3f} "
        f"(NOT a paired delta -- see note above)"
    )

    report = "\n".join(lines)
    print(report)

    # Stage D deliberately does NOT write into docs/ -- that tree is Stage E's
    # (documentation) territory in this rework. Results land in the reports/
    # dir instead; Stage E can fold the numbers into LSM_PROJECT.md itself.
    out_path = PROJECT_ROOT / "reports" / "ablation-ladder.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        f"# Stage D: the ablation ladder\n\n"
        f"Measured {ABLATION_N_LINES}-line scale-down of `config/base.yaml`'s production "
        f"density (see `scripts/ablation_ladder.py`'s module docstring).\n\n```\n{report}\n```\n",
        encoding="utf-8",
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
