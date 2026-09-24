"""
Stage 4: auto-generated model card per pipeline release -- intended use,
training corpus, metrics with intervals, known limitations, and the explicit
statement that the data is synthetic (validation-and-trust.md Layer 5: "the
artifact an auditor or a customer's integrity engineer actually asks for").
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path


def _fmt_ci(triple: tuple[float, float, float]) -> str:
    point, lo, hi = triple
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"


def write_model_card(path: str | Path, provenance: dict, result: dict) -> None:
    """`provenance` needs: pipeline_version, config_sha256, git_sha,
    data_sha256, feature_version, schema_version, n_defects, n_surveys, and
    (once Stage 5's classifier has run) consequence_proxy. `result` is
    exactly what `train.run_train` returns -- `mad`, `isolation_forest`,
    `gate_passed`, `interference_attributable`, and (once Stage 4's severity
    CV has run) `severity`, `severity_baseline`, `severity_gate_passed`, and
    (once Stage 5's classify CV has run) `classify`, `classify_baseline`,
    `classify_recall_gate_passed`, `classify_shap_check`, `classify_gate_passed`.
    """
    now = dt.datetime.now(dt.UTC).isoformat()
    mad, iso = result["mad"], result["isolation_forest"]

    lines = [
        f"# Model card -- pipeline {provenance['pipeline_version']}",
        "",
        (
            "**This is a synthetic demonstrator.** All defects, interference sources and "
            "survey data are generated, not from a real pipeline. The metrics below "
            "describe performance on that synthetic data only and are not a claim about "
            "real-world detection performance."
        ),
        "",
        "## Intended use",
        (
            "Detects candidate pipeline defects from magnetometer survey residuals and "
            "estimates their severity, ranked for dig-budget-constrained inspection "
            "planning. Not validated against real inspection outcomes."
        ),
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
        (
            f"- {provenance['n_surveys']} surveys, {provenance['n_defects']} physical defects "
            "(the same defects are re-observed across a line's multiple surveys, not "
            "independent instances)"
        ),
        "",
        "## Detection (Stage 3): IsolationForest vs MAD baseline",
        (
            f"- recall @ dig budget: IsolationForest {_fmt_ci(iso['recall_at_budget'])}, "
            f"MAD {_fmt_ci(mad['recall_at_budget'])}"
        ),
        (
            f"- false-dig rate: IsolationForest {_fmt_ci(iso['false_dig_rate'])}, "
            f"MAD {_fmt_ci(mad['false_dig_rate'])}"
        ),
        (
            f"- of which interference: IsolationForest {_fmt_ci(iso['interference_dig_fraction'])}, "
            f"MAD {_fmt_ci(mad['interference_dig_fraction'])}"
        ),
        (
            f"- localisation error (m): IsolationForest {_fmt_ci(iso['localisation_error_m'])}, "
            f"MAD {_fmt_ci(mad['localisation_error_m'])}"
        ),
        (
            f"- localisation error (cm): IsolationForest {_fmt_ci(iso['localisation_error_cm'])}, "
            f"MAD {_fmt_ci(mad['localisation_error_cm'])} -- the ~1 cm dig-marking requirement's "
            "own unit (Rig-v2 plan), what Stage B weld-comb registration was built to reach"
        ),
        f"- recall gap (IsolationForest - MAD): {_fmt_ci(result['recall_gap_if_minus_mad'])}",
        (
            f"- **gate (>= 0.15 recall gap at the CI lower bound): "
            f"{'PASSED' if result['gate_passed'] else 'DID NOT PASS'}**"
        ),
        f"- attributable to interference rejection: {'yes' if result['interference_attributable'] else 'no'}",
        "",
    ]

    if "severity" in result:
        sev, base = result["severity"], result["severity_baseline"]
        lines += [
            "## Severity (Stage 4): LightGBM quantile + split conformal vs global-mean baseline",
            (
                f"- coverage @ 90% nominal: model {_fmt_ci(sev['coverage'])}, "
                f"baseline {_fmt_ci(base['coverage'])}"
            ),
            f"- MAE: model {_fmt_ci(sev['mae'])}, baseline {_fmt_ci(base['mae'])}",
            (
                f"- mean interval width: model {_fmt_ci(sev['interval_width'])}, "
                f"baseline {_fmt_ci(base['interval_width'])}"
            ),
            (
                f"- **gate (coverage in [0.87, 0.93]): "
                f"{'PASSED' if result['severity_gate_passed'] else 'DID NOT PASS'}**"
            ),
            f"- n matched, severity-labelled indications: {result['n_severity_samples']}",
            "",
        ]

    if "classify" in result:
        clf, cbase = result["classify"], result["classify_baseline"]
        lines += [
            (
                "## Classification (Stage 5): LightGBM multiclass + isotonic calibration "
                "vs majority-class baseline"
            ),
            "",
            "Per-class recall (model / baseline):",
        ]
        for cls, triple in clf["per_class_recall"].items():
            base_triple = cbase["per_class_recall"].get(cls, (float("nan"),) * 3)
            marker = " **(protected, gate below)**" if cls == "scc" else ""
            lines.append(f"- {cls}{marker}: {_fmt_ci(triple)} / {_fmt_ci(base_triple)}")
        lines += [
            "",
            (
                f"- interference precision: model {_fmt_ci(clf['interference_precision'])}, "
                f"baseline {_fmt_ci(cbase['interference_precision'])}"
            ),
            (
                f"- Brier score (lower is better): model {_fmt_ci(clf['brier'])}, "
                f"baseline {_fmt_ci(cbase['brier'])}"
            ),
            (
                f"- **recall gate (SCC recall >= 0.90 at the CI lower bound): "
                f"{'PASSED' if result.get('classify_recall_gate_passed') else 'DID NOT PASS'}**"
            ),
        ]
        if "classify_shap_check" in result:
            shap_check = result["classify_shap_check"]
            lines += [
                (
                    f"- **physics-consistency gate (no absolute-position feature in the top-10 "
                    f"by contribution): {'PASSED' if shap_check['passed'] else 'DID NOT PASS'}**"
                ),
                f"  - top features by contribution: {shap_check['top_features']}",
            ]
            if not shap_check["passed"]:
                lines.append(
                    f"  - leaked features: {shap_check['leaked_denylist_features']}"
                )
            lines.append(
                f"- **overall Stage 5 gate: "
                f"{'PASSED' if result.get('classify_gate_passed') else 'DID NOT PASS'}**"
            )
        lines += [
            f"- n matched, classifiable indications: {result['n_classify_samples']}",
            "",
        ]

        consequence_proxy = provenance.get("consequence_proxy")
        if consequence_proxy:
            lines += [
                (
                    "**`risk_score` = calibrated P(defect) x predicted severity x consequence proxy.** "
                    "The consequence proxy below is a STATED ENGINEERING-JUDGMENT ranking, not derived "
                    "from real consequence-of-failure data (population density, product type, MAOP) -- "
                    "there is none in this synthetic project. It is meant to be replaced wholesale once "
                    "that data exists, not treated as a calibrated output:"
                ),
                "",
            ]
            for cls, weight in consequence_proxy.items():
                lines.append(f"- {cls}: {weight}")
            lines.append("")

    lines += [
        "## Known limitations",
        (
            "- 12 physical defects per line is too few for a stable point estimate on any "
            "metric here -- every headline number above carries a bootstrap CI for exactly "
            "this reason; read the interval, not the point."
        ),
        (
            "- The 3 surveys of a line re-observe the SAME 12 defects, not independent "
            "instances -- bootstrap CIs are resampled over defects, never over rows or "
            "(defect, run) pairs, to avoid overstating precision."
        ),
        (
            "- Severity is regressed per indication, never per row (a row-level regressor "
            "would just learn to reproduce the constant `severity_smys` value inside a "
            "label box)."
        ),
        (
            "- Labels have a physically-derived but still finite extent (see PLAN.md Stage "
            "2.5), so localisation error below roughly that extent is not meaningfully "
            "measurable."
        ),
    ]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_growth_section(path: str | Path, provenance: dict, result: dict) -> None:
    """Stage 8: appends a growth/remaining-life section to an ALREADY-WRITTEN
    model card (from `write_model_card`, run at `lsm train` time), inserted
    before "## Known limitations" if present, else at the end. `forecast` is
    a separate command run after `train` -- this avoids `run_forecast`
    needing to reconstruct anomaly/severity/classify's own result dicts just
    to call `write_model_card` again. `result` needs: `growth`,
    `growth_baseline` (each `{"mae": (point, lo, hi)}`),
    `growth_gap_baseline_minus_model`, `growth_gate_passed`,
    `growth_population_log_rate`, `n_growth_samples`, `n_defects_evaluated`,
    `median_days_to_verification`. `provenance` needs `limit_state_smys`,
    `assumed_interval_years`, and optionally `synthetic_growth_rate` (the
    generator's own `data.growth`, for the honest ln(rate) comparison).
    """
    growth, base = result["growth"], result["growth_baseline"]
    days = result.get("median_days_to_verification")
    days_text = (
        f"{days:.1f}" if days is not None else "n/a -- no excavations recorded yet"
    )

    rate_comparison = ""
    if "synthetic_growth_rate" in provenance:
        rate_comparison = f" (synthetic ground truth: ln({provenance['synthetic_growth_rate']}) = {math.log(provenance['synthetic_growth_rate']):.4f})"

    lines = [
        "## Growth & remaining life (Stage 8): partially-pooled log-linear growth vs 'no growth' baseline",
        "",
        f"- population log-growth-rate: {result['growth_population_log_rate']:.4f}{rate_comparison}",
        (
            f"- one-step-ahead severity MAE at the held-out run: model {_fmt_ci(growth['mae'])}, "
            f"no-growth baseline {_fmt_ci(base['mae'])}"
        ),
        f"- gap (baseline - model): {_fmt_ci(result['growth_gap_baseline_minus_model'])}",
        (
            f"- **gate (model beats the no-growth baseline at the CI lower bound): "
            f"{'PASSED' if result['growth_gate_passed'] else 'DID NOT PASS'}**"
        ),
        (
            f"- n defects evaluated: {result['n_defects_evaluated']} (of {result['n_growth_samples']} "
            "matched, multi-run severity observations)"
        ),
        (
            f"- assumed limit state: {provenance.get('limit_state_smys')} %SMYS, assumed survey interval: "
            f"{provenance.get('assumed_interval_years')} years -- STATED ENGINEERING JUDGMENT, not derived "
            "from data: `surveyed_at` carries no real elapsed calendar time between a line's runs in this "
            "synthetic generator (see config/base.yaml's model.growth comment)."
        ),
        (
            "- **honest scope statement: with as few as 3 observations per defect, per-defect growth "
            "rates are almost entirely population-shrunk, not independently fitted -- the method is "
            "right for this sample size, but every remaining-life number here is provisional, not a "
            "calibrated forecast.**"
        ),
        (
            "- growth-rate ESTIMATION uncertainty is not propagated into the remaining-life interval -- "
            "only the current severity's own conformal interval is (a stated, deliberate scope limit, "
            "not an oversight)."
        ),
        (
            f"- median days from indication to verification: {days_text} (the dig-feedback loop's own "
            "number -- bounds how fast this system can learn anything at all)."
        ),
        "",
    ]

    path = Path(path)
    existing = path.read_text(encoding="utf-8")
    marker = "## Known limitations"
    if marker in existing:
        head, sep, tail = existing.partition(marker)
        new_text = head + "\n".join(lines) + "\n" + sep + tail
    else:
        new_text = existing.rstrip("\n") + "\n\n" + "\n".join(lines) + "\n"
    path.write_text(new_text, encoding="utf-8")
