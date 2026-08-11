"""
Stage D: probability of detection (POD) vs. defect-moment angle to B_hat0.

generate.py's own module docstring states the central Rig-v2 physics claim
plainly: a defect whose magnetic moment is near-perpendicular to the
ambient field direction (`B_hat0 = unit(background_nT)`) is nearly
invisible to a scalar (total-field) rig, because only the projection of the
anomaly onto B_hat0 is ever measured. That claim is asserted and unit-tested
at the single-dipole level in tests/test_generate.py, but `orientation` (the
3-vector drawn per defect in `generate._build_features`) is never persisted
downstream of generation -- it cannot be recovered from a normal survey/
feature table. This script measures the claim directly, at population
scale, rather than leaving it as prose.

Method (the LIGHTWEIGHT option the Stage D plan explicitly prefers over a
schema change): regenerate a corpus with an isolated seed, calling
`generate._build_features` directly (bypassing the full ingest/registration/
feature pipeline) to capture each defect's `orientation` alongside its known
TRUE chainage -- deliberately not routed through registration, since the
physics question here ("is this defect's own field visible above the noise
floor at all") does not depend on how well along-track position gets
reconstructed; using `chainage_true_m` to locate each defect's label window
keeps this a clean, isolated measurement of ONE physics effect, not a
second registration-accuracy study (that is `scripts/ablation_ladder.py`'s
job).

Detectability, per (defect, run): whether the detrended middle-head
residual's max |r_mid_nt| inside the defect's true label window exceeds
this project's own established "clearly anomalous" robust threshold
(4 x MAD of that survey's r_mid_nt -- the same bound `config/base.yaml`'s
`label_window_scale` comment already uses for an equivalent purpose). This
is a direct SNR/visibility measure, not a trained model's idiosyncrasies --
the most honest test of the underlying physics claim, and far cheaper than
a full IsolationForest CV fit.

CAVEAT, found while building this script and worth stating up front rather
than burying: `tests/test_generate.py`'s own null-anomaly test constructs a
SPECIAL geometry (observation point placed directly along the moment's own
axis, `r_hat == m_hat`) under which moment-perpendicular-to-B_hat0 implies
EXACTLY zero field projection. A real along-track survey pass does not sit
in that special geometry -- the sensor-to-source direction r_hat sweeps
through many angles as the walker passes over/near a defect, so the dipole
radiation pattern `3(m_hat.r_hat)r_hat - m_hat` further modulates what
`angle_cos` alone predicts. `angle_cos` (this script) is therefore a real,
physically-grounded, honestly-reported PREDICTOR of detectability at
POPULATION scale, not a deterministic per-defect guarantee -- reported
alongside a continuous peak-SNR measure (not just a binary hit/miss), and a
severity-normalised version of it, precisely because random per-defect
severity (a 4x range, 20-80) is a large confound on top of the angle effect
and binary thresholding throws away most of the signal.

SAFETY: isolated tmp storage only -- see ablation_ladder.py's module
docstring for why this matters.

Usage:
    python scripts/pod_by_angle.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from lsm.config import load_config
from lsm.evaluate import bootstrap_ci
from lsm.features import detrend_axis
from lsm.generate import _build_features, _build_welds, _make_run, unit

# ~120 physical defects at config/base.yaml's default 12/line. Much larger
# than ablation_ladder.py's own N_LINES choice, deliberately: this script is
# far cheaper per line (no model fit, just detrending), and a fast 2-line/
# 40-defect check found angle_cos correlating only weakly (and not
# significantly) with detectability even after normalising by severity --
# more lines buys back some of that statistical power. See module
# docstring's CAVEAT for why the correlation is expected to be real but
# noisy, not a clean deterministic relationship.
N_LINES = 10
MAD_Z_THRESHOLD = 4.0  # config/base.yaml's own "clearly anomalous" bound


def main() -> None:
    # This script never writes a raw survey to disk (generation is done
    # in-memory via _build_features/_build_welds/_make_run directly, not
    # generate_all) -- there is no storage path to isolate, and nothing here
    # can ever touch data/raw or data/lsm.db (see ablation_ladder.py's
    # module docstring for why that safety property matters).
    cfg = load_config("dev")
    cfg.base.data.n_lines = N_LINES

    b0hat = unit(cfg.base.data.background_nT)
    print(f"B_hat0 = {b0hat} (from background_nT={cfg.base.data.background_nT})")

    # Regenerate with the SAME per-line rng sequence generate_all itself uses
    # (seed + line_idx), capturing `orientation` straight off _build_features'
    # return value -- see module docstring. Stop consuming the rng right
    # after _build_features/_build_welds so the SAME line_rng state produces
    # the identical _make_run rows generate_all's own call would.
    defect_rows = []  # (line_id, defect_idx, chainage_true_m, angle_cos)
    surveys_by_line: dict[str, list] = {}
    for line_idx in range(cfg.base.data.n_lines):
        line_id = f"LINE{line_idx:03d}"
        line_rng = np.random.default_rng(cfg.seed + line_idx)
        features = _build_features(cfg.base.data, line_rng)
        welds = _build_welds(cfg.base.data, line_rng)
        for i, f in enumerate(features):
            if not f["is_defect"]:
                continue
            angle_cos = float(abs(np.dot(unit(f["orientation"]), b0hat)))
            defect_rows.append({
                "line_id": line_id, "defect_idx": i,
                "chainage_true_m": f["chainage_m"], "angle_cos": angle_cos,
                "severity": f["severity"],
            })
        runs = [_make_run(line_id, run_id, cfg.base.data, features, welds, line_rng)
                for run_id in range(cfg.base.data.n_runs)]
        surveys_by_line[line_id] = runs

    defects = pd.DataFrame(defect_rows)
    print(f"{len(defects)} physical defects across {cfg.base.data.n_lines} lines.")

    half_width = cfg.base.data.label_window_scale * cfg.base.data.depth_m

    detected = []  # one row per (defect, run): defect_idx, line_id, angle_cos, hit, peak_z
    for line_id, runs in surveys_by_line.items():
        for df in runs:
            s_true = df["chainage_true_m"].to_numpy()
            r_mid = detrend_axis(s_true, df["b_mid_nt"].to_numpy(), cfg.base.features)
            med = np.median(r_mid)
            mad = np.median(np.abs(r_mid - med))
            sigma = 1.4826 * mad
            line_defects = defects[defects["line_id"] == line_id]
            for _, d in line_defects.iterrows():
                window = np.abs(s_true - d["chainage_true_m"]) <= half_width
                if not window.any() or sigma <= 0:
                    hit, peak_z = False, 0.0
                else:
                    peak_z = float(np.max(np.abs(r_mid[window] - med)) / sigma)
                    hit = peak_z > MAD_Z_THRESHOLD
                detected.append({
                    "line_id": line_id, "defect_idx": d["defect_idx"],
                    "angle_cos": d["angle_cos"], "severity": d["severity"],
                    "hit": hit, "peak_z": peak_z,
                })

    det_df = pd.DataFrame(detected)
    per_defect = det_df.groupby(["line_id", "defect_idx"]).agg(
        angle_cos=("angle_cos", "first"), severity=("severity", "first"),
        detected_fraction=("hit", "mean"), mean_peak_z=("peak_z", "mean"), n_runs=("hit", "size"),
    ).reset_index()
    # Severity-normalised SNR: peak_z scales roughly linearly with moment
    # magnitude (severity), a large confound (20-80, a 4x range) on top of
    # the angle effect this script is actually trying to isolate -- see
    # module docstring's CAVEAT.
    per_defect["peak_z_per_severity"] = per_defect["mean_peak_z"] / per_defect["severity"]

    print(f"\n{len(per_defect)} physical defects, {len(det_df)} (defect, run) observations.")
    print(per_defect.sort_values("angle_cos").to_string(index=False))

    corr_binary, p_binary = scipy_stats.spearmanr(per_defect["angle_cos"], per_defect["detected_fraction"])
    corr_z, p_z = scipy_stats.spearmanr(per_defect["angle_cos"], per_defect["mean_peak_z"])
    corr_zn, p_zn = scipy_stats.spearmanr(per_defect["angle_cos"], per_defect["peak_z_per_severity"])
    print(f"\nSpearman correlation(angle_cos, detected_fraction)   = {corr_binary:+.3f} (p={p_binary:.4f})")
    print(f"Spearman correlation(angle_cos, mean_peak_z)         = {corr_z:+.3f} (p={p_z:.4f})")
    print(f"Spearman correlation(angle_cos, peak_z_per_severity) = {corr_zn:+.3f} (p={p_zn:.4f}) "
          f"-- severity-normalised, isolates the angle effect from the severity confound")
    print(f"(n={len(per_defect)} physical defects for all three)")

    # Binned view: low/mid/high |cos(angle to B_hat0)|, bootstrap CI per bin
    # over PHYSICAL DEFECTS (same grouping discipline as every other
    # detection metric in this project -- SKILL invariant #10).
    bins = pd.qcut(per_defect["angle_cos"], q=3, duplicates="drop", labels=["near-perpendicular", "mid", "near-parallel"])
    per_defect = per_defect.assign(angle_bin=bins)
    boot_cfg = cfg.base.model.bootstrap
    print("\nDetection rate by angle-to-B_hat0 bin (bootstrap CI over physical defects):")
    for label, group in per_defect.groupby("angle_bin", observed=True):
        point, lo, hi = bootstrap_ci(group["detected_fraction"].to_numpy(), boot_cfg.n_resamples, boot_cfg.level, seed=17)
        print(f"  {label:20s} (n={len(group):2d}, angle_cos range [{group['angle_cos'].min():.2f}, "
              f"{group['angle_cos'].max():.2f}]): {point:.3f} [{lo:.3f}, {hi:.3f}]")
    print("\nSeverity-normalised peak SNR by angle-to-B_hat0 bin (bootstrap CI over physical defects):")
    for label, group in per_defect.groupby("angle_bin", observed=True):
        point, lo, hi = bootstrap_ci(
            group["peak_z_per_severity"].to_numpy(), boot_cfg.n_resamples, boot_cfg.level, seed=18
        )
        print(f"  {label:20s} (n={len(group):2d}): {point:.5f} [{lo:.5f}, {hi:.5f}]")

    out_path = PROJECT_ROOT / "reports" / "pod-by-angle.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage D: POD vs defect-moment angle to B_hat0", "",
        f"B_hat0 = {b0hat.tolist()} (from background_nT={cfg.base.data.background_nT})", "",
        f"{len(per_defect)} physical defects, {N_LINES} lines, {cfg.base.data.n_runs} runs each.", "",
        f"Spearman correlation(angle_cos, detected_fraction)   = {corr_binary:+.3f} (p={p_binary:.4f})",
        f"Spearman correlation(angle_cos, mean_peak_z)         = {corr_z:+.3f} (p={p_z:.4f})",
        (f"Spearman correlation(angle_cos, peak_z_per_severity) = {corr_zn:+.3f} (p={p_zn:.4f}) "
         "-- severity-normalised, isolates the angle effect from the severity confound"),
        "",
        "```", per_defect.sort_values("angle_cos").to_string(index=False), "```", "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
