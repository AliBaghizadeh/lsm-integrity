# Model card -- pipeline 2026.07.31-79453eae-dirty

**This is a synthetic demonstrator.** All defects, interference sources and survey data are generated, not from a real pipeline. The metrics below describe performance on that synthetic data only and are not a claim about real-world detection performance.

## Intended use
Detects candidate pipeline defects from magnetometer survey residuals and estimates their severity, ranked for dig-budget-constrained inspection planning. Not validated against real inspection outcomes.

## Provenance
- generated: 2026-07-31T13:11:18.314730+00:00
- git_sha: 79453eae855ddd6e9d58d40c01cc0d97e258e817-dirty
- config_sha256: ed59fc0b784008707668d9cfe3fdf73936c0be0b85c55f2d67e93cb3299438ce
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

## Classification (Stage 5): LightGBM multiclass + isotonic calibration vs majority-class baseline

Per-class recall (model / baseline):
- scc **(protected, gate below)**: 0.125 [0.000, 0.264] / 0.000 [0.000, 0.000]
- weld: 0.111 [0.000, 0.278] / 0.083 [0.000, 0.250]
- dent: 0.646 [0.479, 0.812] / 0.750 [0.562, 0.938]
- corrosion: 0.383 [0.167, 0.617] / 0.000 [0.000, 0.000]
- interference: 0.778 [0.528, 1.000] / 0.000 [0.000, 0.000]

- interference precision: model 0.909 [0.727, 1.000], baseline nan [nan, nan]
- Brier score (lower is better): model 0.738 [0.636, 0.845], baseline 0.804 [0.788, 0.819]
- **recall gate (SCC recall >= 0.90 at the CI lower bound): DID NOT PASS**
- **physics-consistency gate (no absolute-position feature in the top-10 by contribution): PASSED**
  - top features by contribution: ['w25m_kurt', 'r_decl_deg', 'w25m_mean_nt', 'w25m_std_nt', 'w25m_energy_nt2', 'rx_nt', 'w10m_mean_nt', 'w25m_zcr', 'w5m_kurt', 'r_incl_deg']
- **overall Stage 5 gate: DID NOT PASS**
- n matched, classifiable indications: 170

**`risk_score` = calibrated P(defect) x predicted severity x consequence proxy.** The consequence proxy below is a STATED ENGINEERING-JUDGMENT ranking, not derived from real consequence-of-failure data (population density, product type, MAOP) -- there is none in this synthetic project. It is meant to be replaced wholesale once that data exists, not treated as a calibrated output:

- scc: 1.0
- corrosion: 0.6
- dent: 0.5
- weld: 0.4
- interference: 0.0

## Growth & remaining life (Stage 8): partially-pooled log-linear growth vs 'no growth' baseline

- population log-growth-rate: 0.1398 (synthetic ground truth: ln(1.15) = 0.1398)
- one-step-ahead severity MAE at the held-out run: model 0.000 [0.000, 0.000], no-growth baseline 9.799 [9.012, 10.758]
- gap (baseline - model): 9.799 [8.989, 10.707]
- **gate (model beats the no-growth baseline at the CI lower bound): PASSED**
- n defects evaluated: 44 (of 132 matched, multi-run severity observations)
- assumed limit state: 100.0 %SMYS, assumed survey interval: 5.0 years -- STATED ENGINEERING JUDGMENT, not derived from data: `surveyed_at` carries no real elapsed calendar time between a line's runs in this synthetic generator (see config/base.yaml's model.growth comment).
- **honest scope statement: with as few as 3 observations per defect, per-defect growth rates are almost entirely population-shrunk, not independently fitted -- the method is right for this sample size, but every remaining-life number here is provisional, not a calibrated forecast.**
- growth-rate ESTIMATION uncertainty is not propagated into the remaining-life interval -- only the current severity's own conformal interval is (a stated, deliberate scope limit, not an oversight).
- median days from indication to verification: n/a -- no excavations recorded yet (the dig-feedback loop's own number -- bounds how fast this system can learn anything at all).

## Known limitations
- 12 physical defects per line is too few for a stable point estimate on any metric here -- every headline number above carries a bootstrap CI for exactly this reason; read the interval, not the point.
- The 3 surveys of a line re-observe the SAME 12 defects, not independent instances -- bootstrap CIs are resampled over defects, never over rows or (defect, run) pairs, to avoid overstating precision.
- Severity is regressed per indication, never per row (a row-level regressor would just learn to reproduce the constant `severity_smys` value inside a label box).
- Labels have a physically-derived but still finite extent (see PLAN.md Stage 2.5), so localisation error below roughly that extent is not meaningfully measurable.
