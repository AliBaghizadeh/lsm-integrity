"""
Quick-look plot of a raw synthetic survey. A dev/debug tool, not a pipeline
stage -- reads straight from data/raw/*.parquet (bit-identical to what's in
SQLite), so it works even before `lsm ingest` has run. The real interactive
map lands in Stage 4.5 (Streamlit); this is for "let me see it right now".

Usage:
    python scripts/plot_survey.py                  # list available surveys
    python scripts/plot_survey.py LINE000_R0        # plot one, opens a window
    python scripts/plot_survey.py LINE000_R0 --save out.png
    python scripts/plot_survey.py LINE000_R0 --chainage 700 900   # zoom in
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def find_surveys() -> list[Path]:
    return sorted(RAW_DIR.glob("line_id=*/run_id=*/survey.parquet"))


def survey_id_of(path: Path) -> str:
    line_id = path.parent.parent.name.split("=", 1)[1]
    run_id = path.parent.name.split("=", 1)[1]
    return f"{line_id}_R{run_id}"


def load(survey_id: str) -> pd.DataFrame:
    for path in find_surveys():
        if survey_id_of(path) == survey_id:
            return pd.read_parquet(path)
    raise SystemExit(
        f"no raw survey found for '{survey_id}'. Available: "
        + ", ".join(survey_id_of(p) for p in find_surveys())
    )


def label_bands(df: pd.DataFrame, flag_col: str, type_col: str | None = None) -> list[tuple[float, float, str]]:
    """Contiguous runs of flag_col==1 -> [(start_m, end_m, label), ...]."""
    bands = []
    in_band = False
    start = label = None
    chainage = df["chainage_m"].to_numpy()
    flag = df[flag_col].to_numpy()
    types = df[type_col].to_numpy() if type_col else None
    for i in range(len(df)):
        if flag[i] == 1 and not in_band:
            in_band, start = True, chainage[i]
            label = types[i] if types is not None else ""
        if flag[i] == 0 and in_band:
            in_band = False
            bands.append((start, chainage[i - 1], label))
    if in_band:
        bands.append((start, chainage[-1], label))
    return bands


def plot(
    df: pd.DataFrame,
    survey_id: str,
    chainage_range: tuple[float, float] | None,
    show_interference: bool,
) -> plt.Figure:
    if chainage_range:
        lo, hi = chainage_range
        df = df[(df["chainage_m"] >= lo) & (df["chainage_m"] <= hi)]

    defect = label_bands(df, "defect", "defect_type")
    interference = label_bands(df, "interference") if show_interference else []
    axes_cols = ["bx_nt", "by_nt", "bz_nt"]
    labels = ["Bx (nT)", "By (nT)", "Bz (nT)"]

    fig, axs = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    title = f"{survey_id}  —  raw signal, pre-detrend  ({len(df):,} samples, {df['chainage_m'].min():.0f}–{df['chainage_m'].max():.0f} m)"
    if show_interference:
        title += "\namber = true defect  ·  grey = interference (unlabeled false-positive trap)"
    fig.suptitle(title, fontsize=11)

    for ax, col, label in zip(axs, axes_cols, labels):
        ax.plot(df["chainage_m"], df[col], color="#2a7f7a", linewidth=0.8)
        for start, end, _ in interference:
            ax.axvspan(start, end, color="#8a97a1", alpha=0.35, lw=0)
        for start, end, _ in defect:
            ax.axvspan(start, end, color="#e0a24a", alpha=0.35, lw=0)
        ax.set_ylabel(label, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.2)

    for start, end, dtype in defect:
        axs[0].text(
            (start + end) / 2, axs[0].get_ylim()[1], dtype,
            fontsize=7, ha="center", va="bottom", color="#8a5a1e", rotation=0,
        )
    for start, end, _ in interference:
        axs[0].text(
            (start + end) / 2, axs[0].get_ylim()[1], "interf.",
            fontsize=7, ha="center", va="bottom", color="#5c6b74", rotation=0,
        )

    axs[-1].set_xlabel("chainage (m)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94 if show_interference else 0.96])
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("survey_id", nargs="?", help="e.g. LINE000_R0 (omit to list available surveys)")
    parser.add_argument("--save", type=Path, help="save PNG here instead of opening a window")
    parser.add_argument("--chainage", nargs=2, type=float, metavar=("START_M", "END_M"), help="zoom to a chainage range")
    parser.add_argument(
        "--interference", action="store_true",
        help="also shade the (unlabeled-to-the-model) interference sources, in grey",
    )
    args = parser.parse_args()

    surveys = find_surveys()
    if not surveys:
        print(f"No raw surveys found under {RAW_DIR}. Run `lsm generate` first.", file=sys.stderr)
        raise SystemExit(1)

    if args.survey_id is None:
        print("Available surveys:")
        for p in surveys:
            print(f"  {survey_id_of(p)}")
        print("\nUsage: python scripts/plot_survey.py <survey_id> [--save out.png] [--chainage START END] [--interference]")
        return

    df = load(args.survey_id)
    if "interference" not in df.columns and args.interference:
        raise SystemExit(
            "This raw file predates the `interference` ground-truth column "
            "(schema_version 1). Run `lsm generate` again to regenerate it."
        )
    fig = plot(df, args.survey_id, tuple(args.chainage) if args.chainage else None, args.interference)

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
