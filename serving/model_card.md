# Model card -- pipeline 2026.08.06-519f7a95-dirty

**This is a synthetic demonstrator.** All defects, interference sources and survey data are generated, not from a real pipeline. The metrics below describe performance on that synthetic data only and are not a claim about real-world detection performance.

## Intended use
Detects candidate pipeline defects from magnetometer survey residuals and estimates their severity, ranked for dig-budget-constrained inspection planning. Not validated against real inspection outcomes.

## Provenance
- generated: 2026-08-06T17:13:45.073307+00:00
- git_sha: 519f7a956e59541b24cbba7a09e1a6e55c81b38d-dirty
- config_sha256: 54d903ccb8c1b6ad48250fd3f09b99460395b186a4ccea40791bd1aac0678087
- data_sha256: 77583fabceaf7e4cb42005b273771469ae7d8fd0bc908190da600f4b6bf4427f
- feature_version: 3
- schema_version: 3

## Training corpus
- 15 surveys, 60 physical defects (the same defects are re-observed across a line's multiple surveys, not independent instances)

## Detection (Stage 3): IsolationForest vs MAD baseline
- recall @ dig budget: IsolationForest 0.139 [0.067, 0.222], MAD 0.144 [0.067, 0.233]
- false-dig rate: IsolationForest 0.833 [0.767, 0.893], MAD 0.827 [0.773, 0.874]
- of which interference: IsolationForest 0.053 [0.020, 0.080], MAD 0.067 [0.027, 0.113]
- localisation error (m): IsolationForest 5.644 [4.013, 7.484], MAD 6.359 [4.358, 8.228]
- recall gap (IsolationForest - MAD): -0.006 [-0.061, 0.056]
- **gate (>= 0.15 recall gap at the CI lower bound): DID NOT PASS**
- attributable to interference rejection: no

## Severity (Stage 4): LightGBM quantile + split conformal vs global-mean baseline
- coverage @ 90% nominal: model 0.689 [0.420, 0.945], baseline 0.589 [0.313, 0.865]
- MAE: model 28.952 [20.548, 38.243], baseline 29.087 [19.006, 38.649]
- mean interval width: model 72.867 [62.829, 82.105], baseline 69.910 [61.160, 78.399]
- **gate (coverage in [0.87, 0.93]): DID NOT PASS**
- n matched, severity-labelled indications: 293

## Classification (Stage 5): LightGBM multiclass + isotonic calibration vs majority-class baseline

Per-class recall (model / baseline):
- scc **(protected, gate below)**: 0.042 [0.000, 0.125] / 0.000 [0.000, 0.000]
- weld: 0.000 [0.000, 0.000] / 0.000 [0.000, 0.000]
- dent: 0.000 [0.000, 0.000] / 0.000 [0.000, 0.000]
- corrosion: 0.304 [0.054, 0.625] / 0.000 [0.000, 0.000]
- interference: 0.326 [0.122, 0.537] / 1.000 [1.000, 1.000]

- interference precision: model 0.389 [0.167, 0.611], baseline 0.318 [0.182, 0.477]
- Brier score (lower is better): model 0.993 [0.900, 1.074], baseline 0.805 [0.732, 0.872]
- **recall gate (SCC recall >= 0.90 at the CI lower bound): DID NOT PASS**
- **physics-consistency gate (no absolute-position feature in the top-10 by contribution): PASSED**
  - top features by contribution: ['w5m_kurt', 'w10m_mean_nt', 'w25m_mean_nt', 'w10m_energy_nt2', 'r_mid_nt', 'w10m_zcr', 'w5m_ptp_nt', 'w25m_kurt', 'w25m_energy_nt2', 'd2r_ds2_nt_per_m2']
- **overall Stage 5 gate: DID NOT PASS**
- n matched, classifiable indications: 376

**`risk_score` = calibrated P(defect) x predicted severity x consequence proxy.** The consequence proxy below is a STATED ENGINEERING-JUDGMENT ranking, not derived from real consequence-of-failure data (population density, product type, MAOP) -- there is none in this synthetic project. It is meant to be replaced wholesale once that data exists, not treated as a calibrated output:

- scc: 1.0
- corrosion: 0.6
- dent: 0.5
- weld: 0.4
- interference: 0.0

## Growth & remaining life (Stage 8): partially-pooled log-linear growth vs 'no growth' baseline

- population log-growth-rate: 0.1398 (synthetic ground truth: ln(1.15) = 0.1398)
- one-step-ahead severity MAE at the held-out run: model 0.000 [0.000, 0.000], no-growth baseline 10.034 [7.747, 12.929]
- gap (baseline - model): 10.034 [7.729, 12.568]
- **gate (model beats the no-growth baseline at the CI lower bound): PASSED**
- n defects evaluated: 14 (of 69 matched, multi-run severity observations)
- assumed limit state: 100.0 %SMYS, assumed survey interval: 5.0 years -- STATED ENGINEERING JUDGMENT, not derived from data: `surveyed_at` carries no real elapsed calendar time between a line's runs in this synthetic generator (see config/base.yaml's model.growth comment).
- **honest scope statement: with as few as 3 observations per defect, per-defect growth rates are almost entirely population-shrunk, not independently fitted -- the method is right for this sample size, but every remaining-life number here is provisional, not a calibrated forecast.**
- growth-rate ESTIMATION uncertainty is not propagated into the remaining-life interval -- only the current severity's own conformal interval is (a stated, deliberate scope limit, not an oversight).
- median days from indication to verification: n/a -- no excavations recorded yet (the dig-feedback loop's own number -- bounds how fast this system can learn anything at all).

## Known limitations
- 12 physical defects per line is too few for a stable point estimate on any metric here -- every headline number above carries a bootstrap CI for exactly this reason; read the interval, not the point.
- The 3 surveys of a line re-observe the SAME 12 defects, not independent instances -- bootstrap CIs are resampled over defects, never over rows or (defect, run) pairs, to avoid overstating precision.
- Severity is regressed per indication, never per row (a row-level regressor would just learn to reproduce the constant `severity_smys` value inside a label box).
- Labels have a physically-derived but still finite extent (see PLAN.md Stage 2.5), so localisation error below roughly that extent is not meaningfully measurable.
