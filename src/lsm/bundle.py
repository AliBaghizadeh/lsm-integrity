"""
Stage 4: the versioned model bundle -- the real artifact contract, replacing
Stage 3 train.py's ad hoc joblib dict.

One bundle per model_version (one per task: anomaly, severity, ...), carrying:
fitted model(s), the ordered feature list a bundle PINS (refuses to load
against a mismatched `feature_version`, so a stale feature set or a silent
column reorder cannot change predictions without an error), the pinned
`defect_type` category order (an alphabetical reshuffle would silently permute
a classifier's output columns -- prepared here for Stage 5, harmless before
it), calibration data (conformal margin / anomaly threshold), all three
hashes, `truth_as_of`, installed library versions, and the training-set
feature summary that becomes the Stage 8 drift reference.

**Hard-fails on load** on a `feature_version`/`schema_version` mismatch OR a
library (numpy/sklearn/lightgbm) version mismatch -- recording the versions
without enforcing them would let a silent minor-version difference change
predictions with no error, which architecture.md calls "the worst failure
class there is".
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn


class BundleVersionMismatchError(Exception):
    """A bundle was trained against a different feature_version, schema_version,
    or library version than what's currently installed/configured. Refuses to
    load rather than silently produce predictions the bundle was never tested
    against.
    """


def library_versions() -> dict[str, str]:
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
    }


def training_feature_summary(
    corpus: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
    n_bins: int = 10,
    sample_size: int = 2000,
) -> dict[str, dict]:
    """Per-feature mean/std/min/max PLUS quantile bin edges (n_bins+1 edges)
    and a bounded, seeded raw sample -- the Stage 8 drift reference. PSI
    (evaluate.psi) bins a comparison survey's values into THESE edges, never
    re-derived from the comparison survey; KS (evaluate.ks_drift) needs a
    raw sample, not a summary statistic (`scipy.stats.ks_2samp` takes two
    samples of values, not four numbers).
    """
    rng = np.random.default_rng(seed)
    summary: dict[str, dict] = {}
    for col in feature_cols:
        x = corpus[col].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            summary[col] = {
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "bin_edges": [],
                "sample": [],
            }
            continue
        sample = (
            x
            if len(x) <= sample_size
            else rng.choice(x, size=sample_size, replace=False)
        )
        summary[col] = {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "bin_edges": np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1)).tolist(),
            "sample": sample.tolist(),
        }
    return summary


def training_prediction_reference(
    indications_per_km: float,
    p_defect_cal_sample: np.ndarray,
    seed: int,
    sample_size: int = 2000,
) -> dict:
    """{'indications_per_km': float, 'p_defect_cal_sample': list[float]} --
    the Stage 8 prediction-drift reference (validation-and-trust.md Layer 4:
    "distribution of p_defect_cal and indications-per-km vs the training-set
    rate"). Not per-feature, so it doesn't belong inside
    `training_feature_summary` -- stored once per training run, reused
    across bundles exactly like that function's output already is.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(p_defect_cal_sample, dtype=float)
    x = x[np.isfinite(x)]
    sample = (
        x if len(x) <= sample_size else rng.choice(x, size=sample_size, replace=False)
    )
    return {
        "indications_per_km": float(indications_per_km),
        "p_defect_cal_sample": sample.tolist(),
    }


def save_bundle(path: str | Path, bundle: dict) -> None:
    """`bundle` must at minimum carry `task`, `feature_cols`, `feature_version`,
    `schema_version` -- `load_bundle` checks those. `library_versions` is
    stamped here, not by the caller, so it always reflects what was ACTUALLY
    importable at save time rather than what the caller assumed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(bundle)
    payload["library_versions"] = library_versions()
    joblib.dump(payload, path)


def load_bundle(
    path: str | Path, expected_feature_version: int, expected_schema_version: int
) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist -- run `lsm train` first")
    bundle = joblib.load(path)

    if bundle["feature_version"] != expected_feature_version:
        raise BundleVersionMismatchError(
            f"bundle feature_version={bundle['feature_version']} != current config's "
            f"feature_version={expected_feature_version} -- retrain"
        )
    if bundle["schema_version"] != expected_schema_version:
        raise BundleVersionMismatchError(
            f"bundle schema_version={bundle['schema_version']} != current config's "
            f"schema_version={expected_schema_version} -- retrain"
        )

    current = library_versions()
    stored = bundle.get("library_versions", {})
    mismatches = {
        lib: (stored.get(lib), current[lib])
        for lib in ("numpy", "sklearn", "lightgbm")
        if stored.get(lib) != current[lib]
    }
    if mismatches:
        detail = ", ".join(
            f"{lib}: trained={old} vs installed={new}"
            for lib, (old, new) in mismatches.items()
        )
        raise BundleVersionMismatchError(
            f"library version mismatch ({detail}) -- a silent minor-version "
            "difference can change predictions with no error; retrain or "
            "reinstall the matching versions"
        )
    return bundle
