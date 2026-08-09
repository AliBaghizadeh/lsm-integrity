"""
"Where does the signal come from" -- a single figure showing, stage by
stage, how the raw ~48,800 nT field is built up from its physical sources
(background, girth welds, interference, defects) and torn back down by
processing to isolate the true defect signal.

A dev/analysis tool, not a pipeline stage -- reads nothing from disk,
writes nothing to storage. Regenerates ONE line in memory by calling
generate.py's own building blocks directly (_build_features, _build_welds,
_walk_trajectory, dipole_field), the same "reach into the generator
internals" pattern scripts/pod_by_angle.py and scripts/ablation_ladder.py
already use, rather than modifying the production _make_run_scalar to
expose a decomposition it has no other reason to support.

The four-source cumulative build (background -> +welds -> +interference ->
+defects, panel 2) is computed on the CLEAN pre-sensor field (no ADC
quantisation/noise) so each source's own contribution is exact, not
noise-obscured. Panels 1/3/4 use the real post-sensor b_mid_nt (via
_apply_sensor), i.e. what a real survey actually reports.

Usage:
    python scripts/plot_signal_decomposition.py
    python scripts/plot_signal_decomposition.py --save docs/img/signal_decomposition.png
    python scripts/plot_signal_decomposition.py --zoom 0 300   # detail-panel window
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np

from lsm.config import load_config
from lsm.features import detrend_axis
from lsm.generate import (
    _apply_sensor,
    _build_features,
    _build_welds,
    _load_observatory_background,
    _walk_trajectory,
    dipole_field,
    unit,
)

DEFECT_COLOR = "#dc5028"
INTERFERENCE_COLOR = "#828282"
WELD_COLOR = "#7a5fc7"
BG_COLOR = "#5b6b76"
WELD_STAGE_COLOR = "#7a5fc7"
INTERFERENCE_STAGE_COLOR = "#828282"
DEFECT_STAGE_COLOR = "#dc5028"
FULL_STAGE_COLOR = "#1a2733"


def _field_sum(obs: np.ndarray, sources: list[tuple[np.ndarray, np.ndarray]], n: int) -> np.ndarray:
    total = np.zeros((n, 3))
    for src, moment in sources:
        total += dipole_field(obs, src, moment)
    return total


def build(cfg, seed: int) -> dict:
    data_cfg = cfg.base.data
    rng = np.random.default_rng(seed)

    features = _build_features(data_cfg, rng)
    welds = _build_welds(data_cfg, rng)
    walk = _walk_trajectory(data_cfg, rng)
    n = walk["n"]
    s_true = walk["chainage_true_m"]
    obs_mid = walk["obs_mid"]

    base = np.array(data_cfg.background_nT, dtype=float)
    b0hat = unit(base)
    if data_cfg.observatory_background.enabled:
        background_variation = _load_observatory_background(data_cfg.observatory_background, s_true)
    else:
        drift = np.linspace(0, 1, n)[:, None] * rng.normal(0, 40, 3)
        wave = (
            np.sin(2 * np.pi * s_true / data_cfg.length_m * rng.uniform(1, 3))[:, None]
            * rng.normal(0, 25, 3)
        )
        background_variation = drift + wave
    B_bg = base + background_variation

    weld_sources = [
        (np.array([w["chainage_m"], 0.0, -data_cfg.depth_m]), unit(w["orientation"]) * w["severity"])
        for w in welds
    ]
    interference_sources, defect_sources = [], []
    for f in features:
        src = np.array([f["chainage_m"], f["y_off_m"], -data_cfg.depth_m])
        m_dir = unit(f["orientation"])
        if f["is_defect"] and data_cfg.stress_polarity == "positive":
            # Same canonicalisation _make_run_scalar applies: orient so the
            # anomaly enhances B_hat0's projection, not a physics difference.
            probe_b = dipole_field(np.array([[f["chainage_m"], f["y_off_m"], 0.0]]), src, m_dir)[0]
            if np.dot(probe_b, b0hat) < 0:
                m_dir = -m_dir
        moment = m_dir * f["severity"]
        (defect_sources if f["is_defect"] else interference_sources).append((src, moment))

    B_background = B_bg.copy()
    # Full cumulative build (for the raw-signal/residual panels) ...
    B_full = (
        B_background
        + _field_sum(obs_mid, weld_sources, n)
        + _field_sum(obs_mid, interference_sources, n)
        + _field_sum(obs_mid, defect_sources, n)
    )
    # ... and each category ISOLATED against background alone (for panel 2) --
    # NOT cumulative, so welds' sheer numeric density (~163/line) can't drown
    # out interference/defects' much rarer (4, 12 total) but individually
    # comparable-or-larger contributions the way a cumulative build does.
    B_welds_only = B_background + _field_sum(obs_mid, weld_sources, n)
    B_interference_only = B_background + _field_sum(obs_mid, interference_sources, n)
    B_defects_only = B_background + _field_sum(obs_mid, defect_sources, n)

    mag_background = np.linalg.norm(B_background, axis=1)
    mag_welds_only = np.linalg.norm(B_welds_only, axis=1)
    mag_interference_only = np.linalg.norm(B_interference_only, axis=1)
    mag_defects_only = np.linalg.norm(B_defects_only, axis=1)
    mag_full_clean = np.linalg.norm(B_full, axis=1)

    b_mid_noisy = _apply_sensor(rng, mag_full_clean, data_cfg.sensor, n)
    r_mid = detrend_axis(s_true, b_mid_noisy, cfg.base.features)

    return {
        "s_true": s_true,
        "mag_background": mag_background,
        "mag_welds_only": mag_welds_only,
        "mag_interference_only": mag_interference_only,
        "mag_defects_only": mag_defects_only,
        "mag_full_clean": mag_full_clean,
        "b_mid_noisy": b_mid_noisy,
        "r_mid": r_mid,
        "features": features,
        "welds": welds,
        "data_cfg": data_cfg,
    }


def _mask_windows(s_true: np.ndarray, centers: list[float], half_width: float) -> np.ndarray:
    if not centers:
        return np.zeros_like(s_true, dtype=bool)
    dist = np.abs(s_true[:, None] - np.array(centers)[None, :])
    return (dist <= half_width).any(axis=1)


def plot(result: dict, zoom: tuple[float, float], save: Path | None) -> None:
    s = result["s_true"]
    data_cfg = result["data_cfg"]
    defect_chainages = [f["chainage_m"] for f in result["features"] if f["is_defect"]]
    interference_chainages = [f["chainage_m"] for f in result["features"] if not f["is_defect"]]
    weld_chainages = [w["chainage_m"] for w in result["welds"]]

    fig = plt.figure(figsize=(13, 20), constrained_layout=True)
    gs = fig.add_gridspec(8, 1, height_ratios=[2, 1, 1, 1, 1, 2, 2, 1.6])

    # 1. Raw signal -- background dominates, but welds are strong enough (up to
    # 20x a defect's moment) to already poke through here. Worth showing
    # honestly, not claiming "nothing is visible" when something plainly is --
    # the finding is WHICH sources are/aren't visible, not that none are.
    ax = fig.add_subplot(gs[0])
    ax.plot(s, result["b_mid_noisy"], color=FULL_STAGE_COLOR, lw=0.6)
    ax.set_title("1. Raw signal (b_mid) -- background dominates; the sharp spikes poking through "
                 "are girth welds (up to 20x a defect's moment), not defects or interference")
    ax.set_ylabel("field (nT)")

    # 2. Each source ISOLATED against background alone (NOT cumulative), small
    # multiples sharing one y-scale, zoomed to the same window as panel 4.
    # A cumulative build (background -> +welds -> +welds+interference ->
    # +everything) was tried first and was misleading: ~24 welds fall in any
    # 300 m window vs. ~1 interference source and ~1 defect, so a cumulative
    # sum makes the later, rarer additions nearly invisible next to welds'
    # sheer numeric density -- reading as "they're all the same" when what's
    # actually true is "welds are far more NUMEROUS, not indistinguishable."
    # Isolating each category against the same background baseline makes
    # each one's own contribution directly comparable.
    stage_specs = [
        ("background\nonly", result["mag_background"], BG_COLOR),
        ("welds\nonly", result["mag_welds_only"], WELD_STAGE_COLOR),
        ("interference\nonly", result["mag_interference_only"], INTERFERENCE_STAGE_COLOR),
        ("defects\nonly", result["mag_defects_only"], DEFECT_STAGE_COLOR),
    ]
    med_bg = np.median(result["mag_background"])
    zoom_mask = (s >= zoom[0]) & (s <= zoom[1])
    dev_all = [vals[zoom_mask] - med_bg for _, vals, _ in stage_specs]
    ylim = (min(d.min() for d in dev_all) * 1.15, max(d.max() for d in dev_all) * 1.15)
    for i, (label, vals, color) in enumerate(stage_specs):
        ax = fig.add_subplot(gs[1 + i])
        ax.plot(s, vals - med_bg, color=color, lw=0.9)
        ax.set_xlim(*zoom)
        ax.set_ylim(*ylim)
        ax.set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center")
        if i == 0:
            ax.set_title(f"2. Each source ISOLATED against background alone (not cumulative -- "
                         f"welds are far more NUMEROUS per km, not indistinguishable from the "
                         f"others), zoomed to {zoom[0]:.0f}-{zoom[1]:.0f} m, same y-scale throughout")
        if i < len(stage_specs) - 1:
            ax.set_xticklabels([])

    # 3. Two-stage-detrended residual, full line. Weld markers dropped here
    # (163 of them at 12.2 m pitch turns into unreadable clutter at this
    # scale) -- interference/defect markers only; welds get their due in the
    # zoomed panel below where individual markers are actually legible.
    ax = fig.add_subplot(gs[5])
    ax.plot(s, result["r_mid"], color=DEFECT_STAGE_COLOR, lw=0.6)
    for c in interference_chainages:
        ax.axvline(c, color=INTERFERENCE_COLOR, ls="--", lw=1.0, alpha=0.8)
    for c in defect_chainages:
        ax.axvline(c, color=DEFECT_COLOR, ls="--", lw=1.0, alpha=0.8)
    ax.set_title("3. After background removal (2-stage detrend) -- welds, interference, defects all visible")
    ax.set_ylabel("residual (nT)")

    # 4. Same residual, zoomed -- now weld markers earn their place back.
    ax = fig.add_subplot(gs[6])
    ax.plot(s, result["r_mid"], color=DEFECT_STAGE_COLOR, lw=0.6)
    for c in weld_chainages:
        ax.axvline(c, color=WELD_COLOR, ls=":", lw=0.8, alpha=0.6)
    for c in interference_chainages:
        ax.axvline(c, color=INTERFERENCE_COLOR, ls="--", lw=1.2, alpha=0.85)
    for c in defect_chainages:
        ax.axvline(c, color=DEFECT_COLOR, ls="--", lw=1.2, alpha=0.85)
    ax.set_xlim(*zoom)
    ax.set_title("4. Same residual, zoomed -- individual bump SHAPES (width) are the discriminator")
    ax.set_ylabel("residual (nT)")

    # 5. Final isolated defect signal -- weld/interference windows masked out.
    ax = fig.add_subplot(gs[7])
    half_width = data_cfg.label_window_scale * data_cfg.depth_m
    mask = _mask_windows(s, weld_chainages, half_width) | _mask_windows(s, interference_chainages, half_width)
    isolated = np.where(mask, np.nan, result["r_mid"])
    ax.plot(s, isolated, color=DEFECT_COLOR, lw=0.7)
    for c in defect_chainages:
        ax.axvline(c, color=DEFECT_COLOR, ls="--", lw=1.0, alpha=0.8)
    ax.set_title("5. Final isolated signal -- weld + interference windows masked out, defects remain")
    ax.set_ylabel("residual (nT)")
    ax.set_xlabel("chainage (m)")

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=DEFECT_COLOR, ls="--", label="true defect"),
        Line2D([0], [0], color=INTERFERENCE_COLOR, ls="--", label="true interference"),
        Line2D([0], [0], color=WELD_COLOR, ls=":", label="girth weld"),
    ]
    fig.legend(handles=legend_handles, loc="outside lower center", ncol=3, fontsize=9)
    fig.suptitle("From raw field to isolated defect signal -- one synthetic survey (LINE000)", fontsize=13)

    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150)
        print(f"saved {save}")
    else:
        plt.show()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", type=Path, default=Path("docs/img/signal_decomposition.png"))
    ap.add_argument("--zoom", type=float, nargs=2, default=(0.0, 300.0), metavar=("START_M", "END_M"))
    ap.add_argument("--seed", type=int, default=None, help="default: config's own seed")
    args = ap.parse_args()

    cfg = load_config("dev")
    seed = args.seed if args.seed is not None else cfg.seed
    result = build(cfg, seed)
    plot(result, tuple(args.zoom), args.save)


if __name__ == "__main__":
    main()
