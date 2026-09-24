"""
Stage 2.5 item 3 gate: does the pipeline's ACTUAL two-stage detrend (robust
polynomial + rolling-median high-pass, features.detrend_axis) still reach the
noise floor when the background is a real geomagnetic observatory trace instead
of the synthetic drift+wave sinusoid?

Runs generate -> features once with the synthetic background and once per real
reference trace in data/reference/ (quiet + storm days, both fetched from the
USGS Geomagnetism Program, station BOU/Boulder), all through the exact same code
path the pipeline uses, and prints the background residual next to the Stage 2
noise floor and defect signal so the three are directly comparable.

A dev/reporting tool, not a pipeline stage -- writes to a temp dir, touches
nothing in data/.

Usage: python scripts/check_observatory_background.py
"""

from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lsm.config import load_config
from lsm.features import SurveyContext, compute_survey_features
from lsm.generate import generate_all

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(cfg, label: str) -> None:
    cfg = copy.deepcopy(cfg)
    cfg.base.data.n_lines = 1
    cfg.base.data.n_runs = 1

    with tempfile.TemporaryDirectory() as tmp:
        result = generate_all(cfg.base.data, Path(tmp), seed=cfg.base.seed)[0]
        raw = pd.read_parquet(result.path)
        ctx = SurveyContext(
            result.survey_id,
            result.line_id,
            result.run_id,
            result.step_m,
            result.standoff_m,
            result.surveyed_at,
            cfg.base.data.gradiometer.baseline_m,
        )
        feats = compute_survey_features(raw, ctx, cfg.base.features)

    m = feats.merge(raw[["sample_idx", "defect", "interference"]], on="sample_idx")
    m = m[m["dq_flag"] == "clean"]
    bg = m[(m["defect"] == 0) & (m["interference"] == 0)]["r_mag_nt"]
    d = m[m["defect"] == 1]["r_mag_nt"]
    noise_floor = cfg.base.data.noise_nT

    print(f"\n=== {label} ===")
    print(
        f"  background residual |r|  median={bg.median():7.2f} nT   p95={bg.quantile(0.95):7.2f} nT"
    )
    print(f"  defect     residual |r|  median={d.median():7.2f} nT")
    print(f"  sensor noise floor                {noise_floor:7.2f} nT")
    print(
        f"  detect contrast (defect / background median): {d.median() / bg.median():5.2f}x "
        f"{'-- Stage 2 gate (>3x) HOLDS' if d.median() / bg.median() > 3.0 else '-- Stage 2 gate BREAKS'}"
    )


def main() -> None:
    cfg = load_config("dev")

    run(cfg, "Synthetic background (drift + one sinusoid) -- the current default")

    for csv_name, label in [
        (
            "geomag_bou_2024-03-15_quiet.csv",
            "Real trace: BOU, 2024-03-15 (geomagnetically quiet day)",
        ),
        (
            "geomag_bou_2024-05-10_storm.csv",
            "Real trace: BOU, 2024-05-10 (G5 'Mother's Day' storm)",
        ),
    ]:
        obs_cfg = copy.deepcopy(cfg)
        obs_cfg.base.data.observatory_background.enabled = True
        obs_cfg.base.data.observatory_background.csv_path = f"data/reference/{csv_name}"
        run(obs_cfg, label)


if __name__ == "__main__":
    main()
