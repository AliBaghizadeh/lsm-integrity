"""
The Stage 2 picture, and the number that justifies the second sensor head.

Four panels down one chainage axis -- raw field, detrended residual, along-track
gradient, vertical gradient -- with true defects and interference sources shaded
separately, plus a printed table quantifying:

  1. defect residual vs background floor (the Stage 2 gate),
  2. how the width features separate an on-pipe defect from off-pipe interference,
  3. what the gradiometer actually buys, reported honestly including where it loses.

A dev/reporting tool, not a pipeline stage: it reads the raw parquet and the
feature store, so it needs `lsm generate && lsm ingest && lsm features` first.

Usage:
    python scripts/plot_features.py LINE000_R0 --save docs/img/stage2.png
    python scripts/plot_features.py LINE000_R0 --chainage 600 900
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lsm.config import load_config  # noqa: E402
from lsm.features import feature_store_dir  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load(survey_id: str, cfg):
    line_id, run_part = survey_id.rsplit("_R", 1)
    run_id = int(run_part)
    raw_path = (
        PROJECT_ROOT / cfg.env.storage.raw_dir / f"line_id={line_id}" / f"run_id={run_id}"
        / "survey.parquet"
    )
    feat_path = (
        feature_store_dir(
            PROJECT_ROOT / cfg.env.storage.feature_dir,
            cfg.base.features.version, line_id, run_id,
        )
        / "features.parquet"
    )
    for p in (raw_path, feat_path):
        if not p.exists():
            raise SystemExit(f"missing {p}\nRun: lsm generate && lsm ingest && lsm features")
    raw = pd.read_parquet(raw_path)
    feat = pd.read_parquet(feat_path)
    # Integer key join. This is the data-contract decision paying rent: a float
    # chainage join between two frames built in different code paths would drop
    # rows silently, and nothing downstream would notice.
    return raw.merge(feat, on="sample_idx", suffixes=("", "_f"))


def bands(df: pd.DataFrame, flag_col: str) -> list[tuple[float, float]]:
    """Contiguous runs of flag_col == 1 -> [(start_m, end_m), ...]."""
    flag = df[flag_col].to_numpy()
    chainage = df["chainage_m"].to_numpy()
    edges = np.flatnonzero(np.diff(np.concatenate([[0], flag, [0]])))
    return [(chainage[a], chainage[min(b, len(df) - 1)]) for a, b in zip(edges[::2], edges[1::2])]


def report(df: pd.DataFrame, cfg) -> None:
    """The numbers. Printed rather than drawn, because these are the ones that get
    quoted, and a number read off a plot is a number nobody can check."""
    clean = df[df["dq_flag"] == "clean"]
    d = clean[clean["defect"] == 1]
    i = clean[(clean["interference"] == 1) & (clean["defect"] == 0)]
    b = clean[(clean["defect"] == 0) & (clean["interference"] == 0)]

    print("\n=== Stage 2 gate: is the defect signal above the floor after background removal? ===")
    print(f"  raw field |B|                   {np.linalg.norm(df[['bx_nt','by_nt','bz_nt']], axis=1).mean():10.0f} nT")
    print(f"  background residual |r| median  {b['r_mag_nt'].median():10.2f} nT")
    print(f"  defect     residual |r| median  {d['r_mag_nt'].median():10.2f} nT")
    print(f"  defect     residual |r| peak    {d['r_mag_nt'].max():10.2f} nT")
    print(f"  contrast (defect median / background median)  {d['r_mag_nt'].median() / b['r_mag_nt'].median():6.1f}x")
    print(f"  defect signal as a fraction of the raw field  "
          f"{d['r_mag_nt'].median() / np.linalg.norm(df[['bx_nt','by_nt','bz_nt']], axis=1).mean() * 100:6.3f} %")

    print("\n=== Interference discrimination: amplitude does not separate them, shape does ===")
    print(f"  {'':22s}{'defect':>12}{'interference':>14}")
    for col, unit in [("r_mag_nt", "nT"), ("peak_prominence_nt", "nT"),
                      ("fwhm_m", "m"), ("peak_asymmetry", ""), ("decay_exponent", "")]:
        print(f"  {col + ' (' + unit + ')':22s}{d[col].median():12.2f}{i[col].median():14.2f}")
    print(f"  -> amplitude ratio {i['r_mag_nt'].median() / d['r_mag_nt'].median():.2f}x, "
          f"width ratio {i['fwhm_m'].median() / d['fwhm_m'].median():.2f}x. "
          "A threshold on amplitude cannot tell these apart.")

    print("\n=== What the second sensor head buys (and costs) ===")
    if "g_mag_nt_per_m" not in df.columns:
        print("  gradiometer disabled for this survey (bx2_nt all NULL)")
        return
    bg = (df["defect"] == 0) & (df["interference"] == 0)
    noise_floor = cfg.base.data.noise_nT * np.sqrt(2)
    diff_stds = []
    for ax in "xyz":
        raw_std = df.loc[bg, f"b{ax}_nt"].std()
        diff_std = (df.loc[bg, f"b{ax}2_nt"] - df.loc[bg, f"b{ax}_nt"]).std()
        diff_stds.append(diff_std)
        print(f"  B{ax}: background std {raw_std:7.2f} nT -> head-difference std {diff_std:6.2f} nT "
              f"({raw_std / diff_std:5.1f}x, {20 * np.log10(raw_std / diff_std):5.1f} dB common-mode rejection)")
    excess = np.sqrt(max(np.mean(diff_stds) ** 2 - noise_floor ** 2, 0.0))
    print(f"  Doubled sensor noise alone (sigma*sqrt(2)) would be {noise_floor:.2f} nT; the extra "
          f"~{excess:.2f} nT above that is the head-to-head background gradient (Stage 2.5: the two "
          "heads no longer share an identical background, so cancellation is not perfect by "
          "construction). It is still small relative to the drift/wave background std shown above --")
    print("  this generator's vertical gradient is a real, non-zero term, not yet a fully independent")
    print("  per-head geomagnetic realisation. Treat the gradiometer numbers below as directionally")
    print("  honest, not as a claim about a specific real-world common-mode rejection figure.")

    single = d["r_mag_nt"].median() / b["r_mag_nt"].median()
    grad = d["g_mag_nt_per_m"].median() / b["g_mag_nt_per_m"].median()
    print(f"\n  Detection contrast, detrended single head : {single:5.2f}x")
    print(f"  Detection contrast, vertical gradiometer  : {grad:5.2f}x")
    verdict = "WORSE" if grad < single else "better"
    print(f"  -> the gradiometer is {verdict} for detection here. Detrending has already removed the")
    print( "     common mode, so differencing adds sqrt(2) noise and subtracts part of the near-field")
    print( "     signal across a 0.5 m baseline. It earns its place where the background is NOT smooth")
    print( "     along-track -- the assumption the detrend depends on -- and because it is instantaneous:")
    print( "     no fitting window, so no edge rows and nothing to get wrong at a survey boundary.")


def plot(df: pd.DataFrame, survey_id: str, chainage_range) -> plt.Figure:
    if chainage_range:
        lo, hi = chainage_range
        df = df[(df["chainage_m"] >= lo) & (df["chainage_m"] <= hi)]

    defect_bands = bands(df, "defect")
    interf_bands = bands(df, "interference")
    has_grad = "g_mag_nt_per_m" in df.columns and df["g_mag_nt_per_m"].notna().any()

    panels = [
        ("bz_nt", "Bz raw (nT)", "the defect is invisible: it is 0.05% of the field"),
        ("r_mag_nt", "|residual| (nT)", "after robust detrend + high-pass: defects appear, so does interference"),
        ("dr_ds_nt_per_m", "d|r|/ds (nT/m)", "along-track gradient -- one sensor, a derivative"),
    ]
    if has_grad:
        panels.append(("g_mag_nt_per_m", "|vertical gradient| (nT/m)",
                       "two heads at 0.5 m: common mode cancelled without fitting anything"))

    fig, axs = plt.subplots(len(panels), 1, figsize=(12, 2.3 * len(panels)), sharex=True)
    fig.suptitle(
        f"{survey_id} — background removal and the interference discriminators\n"
        "amber = true defect   ·   grey = off-pipe interference (unlabelled to the model)",
        fontsize=11,
    )

    for ax, (col, ylabel, note) in zip(axs, panels):
        ax.plot(df["chainage_m"], df[col], color="#2a7f7a", linewidth=0.7)
        for start, end in interf_bands:
            ax.axvspan(start, end, color="#8a97a1", alpha=0.35, lw=0)
        for start, end in defect_bands:
            ax.axvspan(start, end, color="#e0a24a", alpha=0.40, lw=0)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.2)
        ax.text(0.005, 0.93, note, transform=ax.transAxes, fontsize=7.5,
                va="top", color="#4a5560")

    axs[-1].set_xlabel("chainage (m)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("survey_id", help="e.g. LINE000_R0")
    parser.add_argument("--env", default="dev")
    parser.add_argument("--save", type=Path, help="save PNG here instead of opening a window")
    parser.add_argument("--chainage", nargs=2, type=float, metavar=("START_M", "END_M"))
    parser.add_argument("--no-plot", action="store_true", help="print the table only")
    args = parser.parse_args()

    cfg = load_config(args.env)
    df = load(args.survey_id, cfg)
    report(df, cfg)

    if args.no_plot:
        return
    fig = plot(df, args.survey_id, tuple(args.chainage) if args.chainage else None)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150)
        print(f"\nsaved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
