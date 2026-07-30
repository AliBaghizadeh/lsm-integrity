"""
Stage 4: auto-generated model card per pipeline release -- intended use,
training corpus, metrics with intervals, known limitations, and the explicit
statement that the data is synthetic (validation-and-trust.md Layer 5: "the
artifact an auditor or a customer's integrity engineer actually asks for").
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path


def _fmt_ci(triple: tuple[float, float, float]) -> str:
    point, lo, hi = triple
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"


def write_model_card(path: str | Path, provenance: dict, result: dict) -> None:
    """`provenance` needs: pipeline_version, config_sha256, git_sha,
    data_sha256, feature_version, schema_version, n_defects, n_surveys.
    `result` is exactly what `train.run_train` returns -- `mad`,
    `isolation_forest`, `gate_passed`, `interference_attributable`, and
    (once Stage 4's severity CV has run) `severity`, `severity_baseline`,
    `severity_gate_passed`.
    """
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    mad, iso = result["mad"], result["isolation_forest"]

    lines = [
        f"# Model card -- pipeline {provenance['pipeline_version']}",
        "",
        "**This is a synthetic demonstrator.** All defects, interference sources and "
        "survey data are generated, not from a real pipeline. The metrics below "
        "describe performance on that synthetic data only and are not a claim about "
        "real-world detection performance.",
        "",
        "## Intended use",
        "Detects candidate pipeline defects from magnetometer survey residuals and "
        "estimates their severity, ranked for dig-budget-constrained inspection "
        "planning. Not validated against real inspection outcomes.",
        "",
        "## Provenance",
        f"- generated: {now}",
        f"- git_sha: {provenance['git_sha']}",
        f"- config_sha256: {provenance['config_sha256']}",
        f"- data_sha256: {provenance['data_sha256']}",
        f"- feature_version: {provenance['feature_version']}",
        f"- schema_version: {provenance['schema_version']}",
        "",
        "## Training corpus",
        f"- {provenance['n_surveys']} surveys, {provenance['n_defects']} physical defects "
        "(the same defects are re-observed across a line's multiple surveys, not "
        "independent instances)",
        "",
        "## Detection (Stage 3): IsolationForest vs MAD baseline",
        f"- recall @ dig budget: IsolationForest {_fmt_ci(iso['recall_at_budget'])}, "
        f"MAD {_fmt_ci(mad['recall_at_budget'])}",
        f"- false-dig rate: IsolationForest {_fmt_ci(iso['false_dig_rate'])}, "
        f"MAD {_fmt_ci(mad['false_dig_rate'])}",
        f"- of which interference: IsolationForest {_fmt_ci(iso['interference_dig_fraction'])}, "
        f"MAD {_fmt_ci(mad['interference_dig_fraction'])}",
        f"- localisation error (m): IsolationForest {_fmt_ci(iso['localisation_error_m'])}, "
        f"MAD {_fmt_ci(mad['localisation_error_m'])}",
        f"- recall gap (IsolationForest - MAD): {_fmt_ci(result['recall_gap_if_minus_mad'])}",
        f"- **gate (>= 0.15 recall gap at the CI lower bound): "
        f"{'PASSED' if result['gate_passed'] else 'DID NOT PASS'}**",
        f"- attributable to interference rejection: {'yes' if result['interference_attributable'] else 'no'}",
        "",
    ]

    if "severity" in result:
        sev, base = result["severity"], result["severity_baseline"]
        lines += [
            "## Severity (Stage 4): LightGBM quantile + split conformal vs global-mean baseline",
            f"- coverage @ 90% nominal: model {_fmt_ci(sev['coverage'])}, "
            f"baseline {_fmt_ci(base['coverage'])}",
            f"- MAE: model {_fmt_ci(sev['mae'])}, baseline {_fmt_ci(base['mae'])}",
            f"- mean interval width: model {_fmt_ci(sev['interval_width'])}, "
            f"baseline {_fmt_ci(base['interval_width'])}",
            f"- **gate (coverage in [0.87, 0.93]): "
            f"{'PASSED' if result['severity_gate_passed'] else 'DID NOT PASS'}**",
            f"- n matched, severity-labelled indications: {result['n_severity_samples']}",
            "",
        ]

    lines += [
        "## Known limitations",
        "- 12 physical defects per line is too few for a stable point estimate on any "
        "metric here -- every headline number above carries a bootstrap CI for exactly "
        "this reason; read the interval, not the point.",
        "- The 3 surveys of a line re-observe the SAME 12 defects, not independent "
        "instances -- bootstrap CIs are resampled over defects, never over rows or "
        "(defect, run) pairs, to avoid overstating precision.",
        "- Severity is regressed per indication, never per row (a row-level regressor "
        "would just learn to reproduce the constant `severity_smys` value inside a "
        "label box).",
        "- Labels have a physically-derived but still finite extent (see PLAN.md Stage "
        "2.5), so localisation error below roughly that extent is not meaningfully "
        "measurable.",
    ]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
