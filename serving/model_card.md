# Model card -- pipeline 2026.07.30-62821561-dirty

**This is a synthetic demonstrator.** All defects, interference sources and survey data are generated, not from a real pipeline. The metrics below describe performance on that synthetic data only and are not a claim about real-world detection performance.

## Intended use
Detects candidate pipeline defects from magnetometer survey residuals and estimates their severity, ranked for dig-budget-constrained inspection planning. Not validated against real inspection outcomes.

## Provenance
- generated: 2026-07-30T11:11:47.903090+00:00
- git_sha: 6282156163c0b934d56b3935271eefd00776972b-dirty
- config_sha256: 59ff17e5adbc54048ee31c728a64774d582bcb38076aae4325d7c831b915c282
- data_sha256: 416d99ad9060f47ae062b21ccf886a91ad4bf8ed004401178243ec50c24dd09f
- feature_version: 2
- schema_version: 2

## Training corpus
- 15 surveys, 60 physical defects (the same defects are re-observed across a line's multiple surveys, not independent instances)

## Detection (Stage 3): IsolationForest vs MAD baseline
- recall @ dig budget: IsolationForest 0.639 [0.522, 0.744], MAD 0.644 [0.528, 0.745]
- false-dig rate: IsolationForest 0.216 [0.167, 0.270], MAD 0.227 [0.180, 0.267]
- of which interference: IsolationForest 0.196 [0.150, 0.243], MAD 0.227 [0.180, 0.267]
- localisation error (m): IsolationForest 0.402 [0.354, 0.454], MAD 0.289 [0.267, 0.315]
- recall gap (IsolationForest - MAD): -0.006 [-0.067, 0.050]
- **gate (>= 0.15 recall gap at the CI lower bound): DID NOT PASS**
- attributable to interference rejection: yes

## Severity (Stage 4): LightGBM quantile + split conformal vs global-mean baseline
- coverage @ 90% nominal: model 0.920 [0.867, 0.967], baseline 0.887 [0.820, 0.953]
- MAE: model 7.462 [6.089, 8.856], baseline 15.089 [12.981, 17.410]
- mean interval width: model 55.515 [52.341, 59.033], baseline 67.091 [63.632, 70.622]
- **gate (coverage in [0.87, 0.93]): PASSED**
- n matched, severity-labelled indications: 137

## Known limitations
- 12 physical defects per line is too few for a stable point estimate on any metric here -- every headline number above carries a bootstrap CI for exactly this reason; read the interval, not the point.
- The 3 surveys of a line re-observe the SAME 12 defects, not independent instances -- bootstrap CIs are resampled over defects, never over rows or (defect, run) pairs, to avoid overstating precision.
- Severity is regressed per indication, never per row (a row-level regressor would just learn to reproduce the constant `severity_smys` value inside a label box).
- Labels have a physically-derived but still finite extent (see PLAN.md Stage 2.5), so localisation error below roughly that extent is not meaningfully measurable.
