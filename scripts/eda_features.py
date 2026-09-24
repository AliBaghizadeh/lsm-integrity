"""
EDA over the real feature store: class balance, per-feature distributions,
correlation/redundancy across the 48 modelling features, and per-feature
separability (defect vs background, interference vs background, defect vs
interference).

This exists to answer a question no DQ gate or model metric answers on its
own: is the feature set itself reasonable going into Stage 3, and does it shed
light on why IsolationForest loses to the MAD baseline specifically at the
tight dig-budget cutoff? Separability uses PR-AUC, not ROC-AUC, for the same
reason the project's own metrics do (positives are a small minority) --
consistent with the modelling gate, not a separate rule for EDA.

A dev/reporting tool, not a pipeline stage: no feature_version, no CLI
subcommand, reads whatever lines/runs already exist under data/raw and
data/features. Needs `lsm generate && lsm ingest && lsm features` first.

Usage:
    python scripts/eda_features.py [--save-dir docs/img] [--top-n 10]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lsm.config import load_config
from lsm.features import feature_columns, feature_store_dir

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_corpus(cfg) -> pd.DataFrame:
    """Every line/run on disk, clean rows only, features joined to raw labels."""
    raw_root = PROJECT_ROOT / cfg.env.storage.raw_dir
    frames = []
    for run_dir in sorted(raw_root.glob("line_id=*/run_id=*")):
        line_id = run_dir.parent.name.split("=", 1)[1]
        run_id = int(run_dir.name.split("=", 1)[1])
        raw = pd.read_parquet(run_dir / "survey.parquet")
        feat_path = (
            feature_store_dir(
                PROJECT_ROOT / cfg.env.storage.feature_dir,
                cfg.base.features.version,
                line_id,
                run_id,
            )
            / "features.parquet"
        )
        if not feat_path.exists():
            continue
        feat = pd.read_parquet(feat_path)
        merged = feat.merge(
            raw[["sample_idx", "defect", "interference"]], on="sample_idx"
        )
        merged = merged[merged["dq_flag"] == "clean"]
        merged["line_id"] = line_id
        merged["run_id"] = run_id
        frames.append(merged)
    if not frames:
        raise SystemExit(
            "no feature store found -- run: lsm generate && lsm ingest && lsm features"
        )
    df = pd.concat(frames, ignore_index=True)
    df["klass"] = np.where(
        df["defect"] == 1,
        "defect",
        np.where(df["interference"] == 1, "interference", "background"),
    )
    return df


def class_balance(df: pd.DataFrame) -> None:
    counts = df["klass"].value_counts()
    frac = (counts / len(df) * 100).round(3)
    print("\n=== Class balance (row-level, clean rows only) ===")
    for k in ("defect", "interference", "background"):
        print(f"  {k:12s} {counts.get(k, 0):7d} rows  ({frac.get(k, 0.0):.3f}%)")
    print(f"  {'total':12s} {len(df):7d} rows")


def feature_summary_stats(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        for k in ("background", "interference", "defect"):
            vals = df.loc[df["klass"] == k, c].dropna()
            if len(vals) == 0:
                continue
            rows.append(
                {
                    "feature": c,
                    "klass": k,
                    "n": len(vals),
                    "mean": vals.mean(),
                    "std": vals.std(),
                    "skew": vals.skew(),
                    "median": vals.median(),
                }
            )
    return pd.DataFrame(rows)


def missingness_by_class(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    nan_cols = [c for c in cols if df[c].isna().any()]
    if not nan_cols:
        return pd.DataFrame()
    out = df.groupby("klass")[nan_cols].apply(lambda g: g.isna().mean() * 100)
    return out.round(1)


def _pr_auc_either_direction(y: np.ndarray, x: np.ndarray) -> float:
    """PR-AUC of x as a score for y, trying both signs since some features
    (e.g. decay_exponent) may separate classes by being LOW, not high."""
    x = np.nan_to_num(x, nan=0.0)
    ap_pos = average_precision_score(y, x)
    ap_neg = average_precision_score(y, -x)
    return max(ap_pos, ap_neg)


def separability_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Per-feature PR-AUC for three binary contrasts. Diagnostic only -- not a
    model, no CV, just "does this one feature carry any signal at all"."""
    is_defect = (df["klass"] == "defect").to_numpy()
    is_interference = (df["klass"] == "interference").to_numpy()
    is_background = (df["klass"] == "background").to_numpy()

    def_bg = is_defect | is_background
    int_bg = is_interference | is_background
    def_int = is_defect | is_interference

    rows = []
    for c in cols:
        vals = df[c].to_numpy(dtype=np.float64)
        rows.append(
            {
                "feature": c,
                "defect_vs_background": _pr_auc_either_direction(
                    is_defect[def_bg], vals[def_bg]
                ),
                "interference_vs_background": _pr_auc_either_direction(
                    is_interference[int_bg], vals[int_bg]
                ),
                "defect_vs_interference": _pr_auc_either_direction(
                    is_defect[def_int], vals[def_int]
                ),
            }
        )
    out = pd.DataFrame(rows).set_index("feature")
    out["base_rate_defect_vs_background"] = is_defect[def_bg].mean()
    out["base_rate_interference_vs_background"] = is_interference[int_bg].mean()
    out["base_rate_defect_vs_interference"] = is_defect[def_int].mean()
    return out


def redundant_pairs(
    corr: pd.DataFrame, threshold: float
) -> list[tuple[str, str, float]]:
    pairs = []
    cols = corr.columns
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            r = corr.loc[a, b]
            if abs(r) >= threshold:
                pairs.append((a, b, float(r)))
    pairs.sort(key=lambda t: -abs(t[2]))
    return pairs


def plot_correlation_heatmap(corr: pd.DataFrame, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(corr.columns, fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Feature correlation matrix (48 modelling features)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_top_distributions(
    df: pd.DataFrame, ranked_features: list[str], save_path: Path
) -> None:
    n = len(ranked_features)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    colors = {"background": "grey", "interference": "orange", "defect": "crimson"}
    for ax, feat in zip(axes, ranked_features):
        for k, color in colors.items():
            vals = df.loc[df["klass"] == k, feat].dropna()
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=40, density=True, histtype="step", color=color, label=k)
        ax.set_title(feat, fontsize=9)
        ax.legend(fontsize=6)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", default="docs/img")
    parser.add_argument(
        "--top-n", type=int, default=8, help="features to plot per rank tier"
    )
    parser.add_argument("--redundancy-threshold", type=float, default=0.9)
    args = parser.parse_args()

    save_dir = PROJECT_ROOT / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config("dev")
    df = load_corpus(cfg)
    cols = feature_columns(cfg.base.features, with_gradiometer=True)

    class_balance(df)

    print("\n=== Missingness by class (%), shape features only ===")
    miss = missingness_by_class(df, cols)
    print(miss.to_string() if not miss.empty else "  none")

    corr = df[cols].corr(method="pearson")
    plot_correlation_heatmap(corr, save_dir / "eda_feature_correlation.png")
    redundant = redundant_pairs(corr, args.redundancy_threshold)
    print(
        f"\n=== Redundant feature pairs (|r| >= {args.redundancy_threshold}), {len(redundant)} found ==="
    )
    for a, b, r in redundant[:30]:
        print(f"  {a:28s} {b:28s} r={r:+.3f}")
    if len(redundant) > 30:
        print(f"  ... and {len(redundant) - 30} more")

    sep = separability_table(df, cols)
    print("\n=== Per-feature PR-AUC (diagnostic; 0.5 = uninformative) ===")
    print(
        sep[
            [
                "defect_vs_background",
                "interference_vs_background",
                "defect_vs_interference",
            ]
        ]
        .sort_values("defect_vs_background", ascending=False)
        .to_string()
    )

    ranked = sep.sort_values("defect_vs_background", ascending=False).index.tolist()
    top = ranked[: args.top_n]
    bottom = ranked[-args.top_n :]
    plot_top_distributions(df, top, save_dir / "eda_top_separating_features.png")
    plot_top_distributions(df, bottom, save_dir / "eda_bottom_separating_features.png")

    stats = feature_summary_stats(df, cols)
    stats_path = save_dir / "eda_feature_stats.csv"
    stats.to_csv(stats_path, index=False)
    sep_path = save_dir / "eda_separability.csv"
    sep.to_csv(sep_path)

    print(f"\nSaved: {save_dir / 'eda_feature_correlation.png'}")
    print(f"Saved: {save_dir / 'eda_top_separating_features.png'}")
    print(f"Saved: {save_dir / 'eda_bottom_separating_features.png'}")
    print(f"Saved: {stats_path}")
    print(f"Saved: {sep_path}")


if __name__ == "__main__":
    main()
