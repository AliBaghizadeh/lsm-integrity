"""Render the four standalone figures proposed for the Medium article.

The values are the final scalar-rig / clean-room results recorded in
docs/experiment-log.md.  This is an editorial rendering utility only; it does
not run or alter the ML pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from plot_signal_decomposition import build

from lsm.config import load_config

OUT = PROJECT_ROOT / "docs" / "img"
INK = "#17212b"
ORANGE = "#dc5028"
BLUE = "#2474a6"
GREY = "#71808c"
PURPLE = "#7856b6"


def save(fig: plt.Figure, filename: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / filename, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def signal_story() -> None:
    """A readable three-panel version of the existing detailed decomposition."""
    result = build(load_config("dev", config_dir=PROJECT_ROOT / "config"), seed=7)
    s = result["s_true"]
    residual = result["r_mid"]
    defects = [f["chainage_m"] for f in result["features"] if f["is_defect"]]
    interference = [f["chainage_m"] for f in result["features"] if not f["is_defect"]]
    welds = [w["chainage_m"] for w in result["welds"]]
    mid = defects[len(defects) // 2]
    window = (mid - 85, mid + 85)
    mask = (s >= window[0]) & (s <= window[1])

    fig, axes = plt.subplots(3, 1, figsize=(12, 8.6), sharex=False, constrained_layout=True)
    fig.suptitle("Why a strong magnetic signal is not automatically a defect", fontsize=17, fontweight="bold")

    axes[0].plot(s, result["b_mid_noisy"], color=INK, lw=0.65)
    axes[0].set(title="1. Raw total field: ambient field and weld spikes dominate", ylabel="total field (nT)")
    axes[0].grid(alpha=0.16)

    axes[1].plot(s, residual, color=ORANGE, lw=0.65)
    for x in defects:
        axes[1].axvline(x, color=ORANGE, alpha=0.48, lw=0.9)
    for x in interference:
        axes[1].axvline(x, color=GREY, alpha=0.55, lw=0.9, ls="--")
    axes[1].set(title="2. After background removal: defects and interference both remain", ylabel="residual (nT)", xlabel="chainage (m)")
    axes[1].grid(alpha=0.16)

    axes[2].plot(s[mask], residual[mask], color=ORANGE, lw=1.0)
    for x in welds:
        if window[0] <= x <= window[1]:
            axes[2].axvline(x, color=PURPLE, alpha=0.40, lw=0.8, ls=":")
    for x in interference:
        if window[0] <= x <= window[1]:
            axes[2].axvline(x, color=GREY, alpha=0.75, lw=1.1, ls="--")
    for x in defects:
        if window[0] <= x <= window[1]:
            axes[2].axvline(x, color=ORANGE, alpha=0.85, lw=1.25)
    axes[2].set(title="3. Zoomed context: shape and repeat behaviour matter, not peak height alone", ylabel="residual (nT)", xlabel="chainage (m)")
    axes[2].grid(alpha=0.16)
    axes[2].text(0.01, -0.30, "orange = simulated defect   ·   grey dashed = external interference   ·   purple dotted = girth weld", transform=axes[2].transAxes, fontsize=9, color="#46535d")
    save(fig, "medium-fig-1-signal-story.png")


def detection_gate() -> None:
    fig, ax = plt.subplots(figsize=(11, 4.7), constrained_layout=True)
    methods = ["Robust threshold baseline", "IsolationForest"]
    recall = [0.106, 0.106]
    y = np.array([1, 0])
    ax.barh(y, recall, color=[GREY, BLUE], height=0.48)
    for yi, value in zip(y, recall):
        ax.text(value + 0.005, yi, f"{value:.3f}", va="center", fontweight="bold", color=INK)
    ax.set(yticks=y, yticklabels=methods, xlim=(0, 0.30), xlabel="recall within the fixed follow-up budget", title="The detector did not earn promotion")
    ax.grid(axis="x", alpha=0.18)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(0.02, -0.35, "Recall gap (IsolationForest − baseline): 0.000   |   95% CI: −0.050 to +0.061", transform=ax.transAxes, fontsize=12, color=INK)
    ax.text(0.02, -0.50, "Promotion required the lower confidence bound to exceed +0.150. It did not.", transform=ax.transAxes, fontsize=11, color=ORANGE, fontweight="bold")
    save(fig, "medium-fig-2-detection-gate.png")


def ablation() -> None:
    labels = ["mid-head", "+ first\ndifference", "+ second\ndifference", "+ stand-off\ncorrection", "+ weld\nregistration", "3-axis\nvector output"]
    recall = np.array([0.153, 0.208, 0.208, 0.222, 0.208, 0.611])
    low = np.array([0.056, 0.083, 0.097, 0.097, 0.097, 0.444])
    high = np.array([0.278, 0.347, 0.347, 0.361, 0.347, 0.764])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    colors = [GREY, BLUE, GREY, GREY, GREY, ORANGE]
    ax.bar(x, recall, color=colors, width=0.66)
    ax.errorbar(x, recall, yerr=[recall - low, high - recall], fmt="none", color=INK, capsize=4, lw=1.4)
    for xi, value in zip(x, recall):
        ax.text(xi, value + 0.025, f"{value:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.axvspan(-0.5, 4.5, color="#eff4f7", zorder=-2)
    ax.text(2, 0.79, "software changes on the existing scalar rig", ha="center", color="#4a5a65", fontsize=10)
    ax.text(5, 0.79, "hardware reference", ha="center", color="#8d381f", fontsize=10)
    ax.set(xticks=x, xticklabels=labels, ylim=(0, 0.84), ylabel="recall at fixed follow-up budget", title="One free software gain; much larger information gain from vector output")
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.01, -0.23, "The first difference improved recall by +0.056 [0.014, 0.111]. Later scalar-rig additions did not show a reliable gain.", transform=ax.transAxes, fontsize=10.5, color=INK)
    save(fig, "medium-fig-3-software-vs-hardware.png")


def clean_room_table() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.1), constrained_layout=True)
    ax.axis("off")
    ax.set_title("The clean-room counterfactual diagnoses the failure — it is not a deployment result", loc="left", fontsize=15, fontweight="bold", pad=15)
    cells = [
        ["Recall gap\n(IsolationForest − baseline)", "−0.019  [−0.051, +0.019]", "+0.199  [+0.139, +0.255]"],
        ["Localisation error", "812 cm", "75 cm"],
        ["Interpretation", "Cluttered field setting:\nno reliable detector advantage", "Confounders removed:\ndetector advantage appears"],
    ]
    table = ax.table(cellText=cells, colLabels=["Metric", "Realistic simulated field", "Clean-room counterfactual"], cellLoc="left", colLoc="left", bbox=[0.01, 0.18, 0.98, 0.66], colWidths=[0.31, 0.33, 0.36])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.PAD = 0.07
        if row == 0:
            cell.set_facecolor(INK)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif col == 2:
            cell.set_facecolor("#fff0e9")
        elif col == 1:
            cell.set_facecolor("#f0f4f6")
        else:
            cell.set_facecolor("#f8fafb")
        if row in (1, 2) and col == 2:
            cell.get_text().set_weight("bold")
    ax.text(0.01, 0.06, "Same simulated rig, walking process, sensor imperfections, defect physics, and defect density. Only interference, welds, and chainage quality change.", transform=ax.transAxes, fontsize=10.5, color="#46535d")
    save(fig, "medium-table-1-clean-room-counterfactual.png")


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold", "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
    signal_story()
    detection_gate()
    ablation()
    clean_room_table()


if __name__ == "__main__":
    main()
