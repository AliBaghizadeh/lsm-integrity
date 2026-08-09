# Model card -- pipeline 2026.08.09-2661562e

**This is a synthetic demonstrator.** All defects, interference sources and survey data are generated, not from a real pipeline. The metrics below describe performance on that synthetic data only and are not a claim about real-world detection performance.

## Intended use
Detects candidate pipeline defects from magnetometer survey residuals and estimates their severity, ranked for dig-budget-constrained inspection planning. Not validated against real inspection outcomes.

## Provenance
- generated: 2026-08-09T18:26:45.604226+00:00
- git_sha: 2661562e481338799b30c3977286e0280feb5398
- config_sha256: 3b57dd5cb61a17740d0e6d944651941a457ddc3d434b76459bc414ba8ba21509
- data_sha256: f6bcaf15108757d1cb493379fc189e544dcee4157c43692b665482f16080985f
- feature_version: 3
- schema_version: 3

## Training corpus
- 15 surveys, 60 physical defects (the same defects are re-observed across a line's multiple surveys, not independent instances)

## Detection (Stage 3): IsolationForest vs MAD baseline
- recall @ dig budget: IsolationForest 0.106 [0.044, 0.183], MAD 0.106 [0.044, 0.183]
- false-dig rate: IsolationForest 0.873 [0.807, 0.927], MAD 0.873 [0.800, 0.933]
- of which interference: IsolationForest 0.087 [0.047, 0.127], MAD 0.053 [0.020, 0.087]
- localisation error (m): IsolationForest 9.216 [7.808, 10.599], MAD 6.758 [4.930, 8.569]
- localisation error (cm): IsolationForest 921.551 [780.808, 1059.907], MAD 675.770 [493.038, 856.926] -- the ~1 cm dig-marking requirement's own unit (Rig-v2 plan), what Stage B weld-comb registration was built to reach
- recall gap (IsolationForest - MAD): 0.000 [-0.050, 0.061]
- **gate (>= 0.15 recall gap at the CI lower bound): DID NOT PASS**
- attributable to interference rejection: no

## Severity (Stage 4): LightGBM quantile + split conformal vs global-mean baseline
- coverage @ 90% nominal: model 0.618 [0.386, 0.834], baseline 0.667 [0.400, 0.867]
- MAE: model 14.170 [9.797, 18.851], baseline 12.942 [8.801, 17.215]
- mean interval width: model 37.581 [31.323, 43.623], baseline 38.872 [34.250, 43.917]
- **gate (coverage in [0.87, 0.93]): DID NOT PASS**
- n matched, severity-labelled indications: 555

## Classification (Stage 5): LightGBM multiclass + isotonic calibration vs majority-class baseline

Per-class recall (model / baseline):
- scc **(protected, gate below)**: 0.023 [0.000, 0.068] / 0.273 [0.000, 0.545]
- weld: 0.258 [0.041, 0.525] / 0.000 [0.000, 0.000]
- dent: 0.144 [0.000, 0.344] / 0.000 [0.000, 0.000]
- corrosion: 0.171 [0.000, 0.402] / 0.000 [0.000, 0.000]
- interference: 0.511 [0.258, 0.755] / 0.455 [0.182, 0.727]

- interference precision: model 0.333 [0.167, 0.500], baseline 0.263 [0.105, 0.474]
- Brier score (lower is better): model 0.928 [0.819, 1.038], baseline 0.832 [0.778, 0.877]
- **recall gate (SCC recall >= 0.90 at the CI lower bound): DID NOT PASS**
- **physics-consistency gate (no absolute-position feature in the top-10 by contribution): PASSED**
  - top features by contribution: ['w25m_zcr', 'w25m_std_nt', 'w25m_max_nt', 'w25m_energy_nt2', 'w25m_mean_nt', 'peak_distance_m', 'w10m_ptp_nt', 'w10m_mean_nt', 'r_hi_nt', 'w10m_max_nt']
- **overall Stage 5 gate: DID NOT PASS**
- n matched, classifiable indications: 330

**`risk_score` = calibrated P(defect) x predicted severity x consequence proxy.** The consequence proxy below is a STATED ENGINEERING-JUDGMENT ranking, not derived from real consequence-of-failure data (population density, product type, MAOP) -- there is none in this synthetic project. It is meant to be replaced wholesale once that data exists, not treated as a calibrated output:

- scc: 1.0
- corrosion: 0.6
- dent: 0.5
- weld: 0.4
- interference: 0.0

## Growth & remaining life (Stage 8): partially-pooled log-linear growth vs 'no growth' baseline

- population log-growth-rate: 0.1398 (synthetic ground truth: ln(1.15) = 0.1398)
- one-step-ahead severity MAE at the held-out run: model 0.000 [0.000, 0.000], no-growth baseline 9.461 [8.014, 10.942]
- gap (baseline - model): 9.461 [8.017, 10.843]
- **gate (model beats the no-growth baseline at the CI lower bound): PASSED**
- n defects evaluated: 15 (of 87 matched, multi-run severity observations)
- assumed limit state: 100.0 %SMYS, assumed survey interval: 5.0 years -- STATED ENGINEERING JUDGMENT, not derived from data: `surveyed_at` carries no real elapsed calendar time between a line's runs in this synthetic generator (see config/base.yaml's model.growth comment).
- **honest scope statement: with as few as 3 observations per defect, per-defect growth rates are almost entirely population-shrunk, not independently fitted -- the method is right for this sample size, but every remaining-life number here is provisional, not a calibrated forecast.**
- growth-rate ESTIMATION uncertainty is not propagated into the remaining-life interval -- only the current severity's own conformal interval is (a stated, deliberate scope limit, not an oversight).
- median days from indication to verification: n/a -- no excavations recorded yet (the dig-feedback loop's own number -- bounds how fast this system can learn anything at all).

## Known limitations
- 12 physical defects per line is too few for a stable point estimate on any metric here -- every headline number above carries a bootstrap CI for exactly this reason; read the interval, not the point.
- The 3 surveys of a line re-observe the SAME 12 defects, not independent instances -- bootstrap CIs are resampled over defects, never over rows or (defect, run) pairs, to avoid overstating precision.
- Severity is regressed per indication, never per row (a row-level regressor would just learn to reproduce the constant `severity_smys` value inside a label box).
- Labels have a physically-derived but still finite extent (see PLAN.md Stage 2.5), so localisation error below roughly that extent is not meaningfully measurable.
