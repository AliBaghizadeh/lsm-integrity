# Stage 6 scale rehearsal -- 2026-08-10T09:41:03.159484+00:00

Config: `config/scale/base.yaml` -- 30 lines x 2000 m x 3 runs = **18,001,953 actual rows** (100.0 rows/m under `rig=scalar`, sampled in time at 120.0 Hz, not on a step_m grid), 360 physical defects. `config_sha256=ad597156ae1a...`

## Wall-clock + peak RSS per step

Peak RSS is *sampled* (a polling thread reading `psutil.Process().memory_info().rss` every 50ms), not an exact accounting -- the true peak between samples can be missed.

| step | rows | wall (s) | rows/sec | peak RSS (MB) |
|---|---|---|---|---|
| generate | 18,001,953 | 1179.23 | 15,266 | 935.8 |
| ingest | 18,001,953 | 87.68 | 205,312 | 757.3 |
| features | 18,001,953 | 367.01 | 49,050 | 767.9 |
| corpus_read_pandas | 18,001,953 | 5.22 | 3,449,323 | 9,226.0 |
| corpus_read_duckdb | 18,001,953 | 19.50 | 922,947 | 20,887.1 |
| train_block_cv | 18,001,953 | 1433.08 | 12,562 | 43,453.5 |
| train_whole_line_and_temporal | - | 1505.58 | - | 49,443.0 |

## Background-contrast re-check (informational, not a gate)

Contrast (defect median / background median of `|r_mid_nt|`) on `LINE000_R0`: **1.62x**. Background median 13.52 nT, defect median 21.94 nT, defect max 219.87 nT (the gap between defect median and max indicates how right-skewed the defect population is -- severity is drawn per type, so a wide spread is expected by construction, not a defect in the measurement).

**Read this as a number, not a verdict.** The historical >= 3.0x threshold was calibrated on the PRE-RIG-V2 vector rig, which measured 3.19x on a residual VECTOR MAGNITUDE (`r_mag_nt`). The scalar rig has no vector to take a magnitude of: the closest equivalent is `|r_mid_nt|`, the absolute value of the middle head's signed total-field residual. These are different physical quantities, and no like-for-like contrast baseline has been established for the scalar rig -- Stage D re-measured every promotion gate and ran the ablation ladder, but never produced a matching contrast figure. Comparing the number above against 3.0x would therefore be comparing across a rig change, which this project explicitly does not do elsewhere. What this check IS good for: confirming the detrend/high-pass still separates defect rows from background rows at this corpus size at all, and giving a first scalar-rig contrast figure that a future run can be compared against. Density-per-km is unchanged from the demo corpus (6 defects/km, 2 interference/km), so any movement here is not a packing-density artefact.

## Historical: a real bug found at scale (original 2026-07-31 pre-Rig-v2 run)

*The two bug write-ups in this section and the next were found by the ORIGINAL Stage 6 run on the pre-Rig-v2 vector rig, and both were fixed then. They are kept here as the record of what a scale rehearsal is FOR -- the incidence figures and follow-on metrics quoted in them are that run's, not this one's. Do not read them as measurements of the current corpus.*

`generate.py` initialises `severity_smys` to NaN off-defect (not 0, contradicting this project's own documented "0 off-defect" convention). A detector's peak occasionally lands just outside a defect's exact label-window half-width while still within the looser `MATCH_TOLERANCE_M` dig-matching radius, so `match_dug_indications` still credits it as matched, but that row's own `severity_smys` is NaN, not the defect's true value -- 36 of 18,917 matched indications (0.4%) at this scale (0 at demo scale, apparently never sampled). A single NaN `y_true` reaching `SeverityModel.fit`'s conformal calibration silently NaN'd the WHOLE fold's margin (`np.quantile` propagates NaN), which then NaN'd every prediction's interval for that fold -- cascading a 0.4%-incidence data issue into an initial 0% pooled coverage across the entire OOF result, not a gradual degradation. Fixed in `train.py::_build_severity_training_frame` (drop NaN-`y_true` rows upstream, loudly) plus a belt-and-braces guard in `SeverityModel.fit` itself. The fix is still in place and still guards this path; whether Stage 4's gate passes on the CURRENT corpus is reported in the results section below, not asserted here.

## Historical: classify's n_estimators/num_leaves were dead config (same original run)

`config/base.yaml`'s `model.classify.n_estimators`/`num_leaves` looked tunable but were never threaded from config into `ClassifyModel`'s `lgbm_cfg` in `train.py` -- editing them had NO effect, silently, since they coincidentally matched `ClassifyModel`'s own hardcoded fallback defaults (50/7). Found while investigating why `config/scale/base.yaml`'s larger capacity values weren't visibly changing classify's behaviour. Fixed by merging `classify_cfg`'s values into the `lgbm_cfg` dict at both call sites. Confirmed working: SCC recall improved from a first, capacity-starved run (0.50 block-CV / 0.39 whole-line) to the numbers reported below.

## A real DQ finding at scale: survey_overlap quarantines

None -- every survey passed `check_survey_overlap` at this scale.

`check_survey_overlap` (validate.py) takes the MAX correlation across ALL prior same-line surveys, not just the immediately preceding one -- so as more runs of a line accumulate, this max is an order statistic over a growing number of comparisons and trends upward even if each individual run-pair's correlation distribution is unchanged (R2 is checked against both R0 and R1; R0 has nothing to compare against). Separately, the more rows a single survey carries (200,021 per survey here, at 2000 m and 100 rows/m), the more samples two runs' shared deterministic structure (same defect/interference positions, same geo/lat-lon path) has to accumulate correlated structure in a plain Pearson correlation, even though each run's background noise is drawn independently. The ORIGINAL pre-Rig-v2 Stage 6 run hit this regime with 40 km lines and saw real quarantines; whether this run does is given by the count immediately above, not assumed here. Either way the fixed 0.9 threshold was calibrated against short lines and few runs and has never been systematically stress-tested -- and where it does fire, the DQ layer is doing exactly what it is designed to do (quarantine, not crash), with the affected surveys correctly excluded from the training corpus. Revisiting the threshold for long-line, many-run deployments is future work, out of Stage 6's scope.

## Where SQLite stopped being the right tool

- Before (pre-Stage-6, per-row `itertuples()`+`float()`-per-cell `load_readings`, measured once by hand at 80,000 rows): **235,982 rows/sec**, peak RSS 183.5 MB.
- After (vectorized NaN->None + dtype-cast, this rehearsal, 18,001,953 rows): **205,312 rows/sec**, peak RSS 757.3 MB (0.87x).
- Conclusion: the naive Python-level row conversion was a real, measurable cost, but not the dominant one -- SQLite's own `executemany` insert path is the majority of the remaining cost at this row count. This matches `.claude/skills/lsm-integrity/references/architecture.md`'s own claim that SQLite "comfortably handles 10^7 rows read-mostly" -- it stops being the right tool at concurrent multi-writer ingest, not at this data volume, which this demonstrator never has.

## Bulk-read path: pandas-concat vs DuckDB

- `features.load_feature_corpus` (per-file glob + `pd.concat`): 5.22s for 18,001,953 rows.
- `scale_eval.load_feature_corpus_duckdb` (one DuckDB glob read): 19.50s for 18,001,953 rows.

## Small-files problem

90 feature-store files at this scale (30 lines x 3 runs) -- PLAN.md's own "120 tiny per-survey Parquet files is fine" example, not the 10^5-file failure case. This rehearsal does not hit that failure mode -- the documented production answer (periodic compaction to line-level Parquet files) is noted here, not artificially triggered.

## float32 arithmetic

The feature store is float32 throughout (`features.py`'s `STORAGE_DTYPE`). Severity's quantile regression and classify's LightGBM training both ran to completion against these float32-stored features with no dtype errors (LightGBM upconverts internally as needed) -- see the block-CV report below for the resulting metrics.

## Default 5-fold block CV (the real gate, `run_train`, unchanged)

```
_build_severity_training_frame: dropping 6314 matched indication(s) whose peak row has a NaN severity_smys (peak landed outside the true label window while still within the dig-matching tolerance) -- see train.py's comment.

=== Stage 3: MAD baseline vs IsolationForest (grouped CV, out-of-fold) ===

MAD baseline:
  recall @ dig budget           0.110  [0.086, 0.135]
  false-dig rate                0.868  [0.846, 0.891]
    of which: interference      0.081  [0.068, 0.096]
  localisation error (m)        7.549  [6.707, 8.344]
  localisation error (cm)      754.851  [670.715, 834.385]
  PR-AUC (diagnostic)           0.035

IsolationForest:
  recall @ dig budget           0.103  [0.080, 0.128]
  false-dig rate                0.877  [0.856, 0.899]
    of which: interference      0.069  [0.056, 0.083]
  localisation error (m)        7.873  [7.085, 8.612]
  localisation error (cm)      787.298  [708.457, 861.247]
  PR-AUC (diagnostic)           0.033

Recall gap (IsolationForest - MAD): -0.007  [-0.030, 0.017]
Stage 3 gate (>= 0.15 recall gap, at the CI lower bound): DOES NOT PASS

Interference-dig-fraction gap (MAD - IsolationForest): 0.012  [-0.002, 0.028]
-> the recall gap is NOT clearly attributable to interference rejection at this confidence level -- reporting this honestly, as the gate requires, rather than claiming a mechanism the data doesn't support.

(360 physical defects, 90 surveys evaluated)

=== Stage 4: severity -- LightGBM quantile + split conformal vs global-mean baseline ===

Global-mean baseline:
  coverage @ 90% nominal        0.838  [0.756, 0.904]
  MAE                          16.959  [14.600, 19.421]
  mean interval width          69.416  [65.377, 73.138]

LightGBM CQR:
  coverage @ 90% nominal        0.872  [0.805, 0.928]
  MAE                          21.263  [18.616, 23.898]
  mean interval width          69.179  [66.669, 71.690]

Stage 4 gate (coverage in [0.87, 0.93]): PASSES
MAE by severity decile: {'(19.596, 25.251]': 35.988660327946334, '(25.251, 32.415]': 31.58966521526455, '(32.415, 35.421]': 29.608164039941812, '(35.421, 41.764]': 15.3944770294914, '(41.764, 45.051]': 11.285570526258839, '(45.051, 51.809]': 10.419416634798262, '(51.809, 57.387]': 10.742070407376042, '(57.387, 58.538]': 7.299805152585589, '(58.538, 81.195]': 23.33452493255931, '(81.195, 102.497]': 34.841123003139074}
(n=3741 matched, severity-labelled indications)

=== Stage 5: classification -- LightGBM multiclass + isotonic calibration vs majority-class baseline ===

Majority-class baseline:
    recall: scc                 0.000  [0.000, 0.000]
    recall: weld                0.000  [0.000, 0.000]
    recall: dent                0.000  [0.000, 0.000]
    recall: corrosion           0.000  [0.000, 0.000]
    recall: interference        1.000  [1.000, 1.000]
  interference precision        0.281  [0.229, 0.332]
  Brier score                   0.792  [0.781, 0.804]

LightGBM multiclass:
    recall: scc                 0.129  [0.062, 0.207]
    recall: weld                0.071  [0.024, 0.131]
    recall: dent                0.152  [0.081, 0.238]
    recall: corrosion           0.048  [0.004, 0.111]
    recall: interference        0.490  [0.398, 0.583]
  interference precision        0.271  [0.216, 0.342]
  Brier score                   0.862  [0.838, 0.887]

SCC recall: 0.129 [CI lower bound 0.062]
Stage 5 recall gate (SCC recall >= 0.90 at the CI lower bound): DOES NOT PASS
Stage 5 physics-consistency gate (no ['chainage_m', 'chainage_peak_m', 'chainage_start_m', 'chainage_end_m', 'sample_idx'] feature in top-10 by contribution): PASSES
Stage 5 gate overall: DOES NOT PASS
(n=2278 matched, classifiable indications)
```

## Whole-line-holdout CV (Stage 6: "the real generalisation test")

An entire physical line held out per fold (5 folds, ~6 lines/fold), instead of today's default (line, 100 m block) hash grouping, which scatters one line's blocks across ~all folds.

```
=== Stage 3: MAD baseline vs IsolationForest (grouped CV, out-of-fold) ===

MAD baseline:
  recall @ dig budget           0.111  [0.086, 0.136]
  false-dig rate                0.867  [0.844, 0.889]
    of which: interference      0.080  [0.066, 0.093]
  localisation error (m)        7.558  [6.705, 8.326]
  localisation error (cm)      755.759  [670.492, 832.633]
  PR-AUC (diagnostic)           0.035

IsolationForest:
  recall @ dig budget           0.098  [0.075, 0.123]
  false-dig rate                0.882  [0.861, 0.903]
    of which: interference      0.070  [0.056, 0.086]
  localisation error (m)        7.847  [7.061, 8.640]
  localisation error (cm)      784.670  [706.052, 864.029]
  PR-AUC (diagnostic)           0.033

Recall gap (IsolationForest - MAD): -0.013  [-0.037, 0.012]
Stage 3 gate (>= 0.15 recall gap, at the CI lower bound): DOES NOT PASS

Interference-dig-fraction gap (MAD - IsolationForest): 0.010  [-0.003, 0.023]
-> the recall gap is NOT clearly attributable to interference rejection at this confidence level -- reporting this honestly, as the gate requires, rather than claiming a mechanism the data doesn't support.

(360 physical defects, 90 surveys evaluated)

=== Stage 4: severity -- LightGBM quantile + split conformal vs global-mean baseline ===

Global-mean baseline:
  coverage @ 90% nominal        0.840  [0.764, 0.912]
  MAE                          16.463  [14.219, 18.707]
  mean interval width          56.182  [55.288, 57.233]

LightGBM CQR:
  coverage @ 90% nominal        0.845  [0.772, 0.911]
  MAE                          17.129  [14.648, 19.828]
  mean interval width          64.308  [60.918, 67.607]

Stage 4 gate (coverage in [0.87, 0.93]): DOES NOT PASS
MAE by severity decile: {'(18.046999999999997, 25.251]': 10.401786756240336, '(25.251, 29.51]': 20.20979372289579, '(29.51, 37.015]': 16.658800600068282, '(37.015, 41.764]': 7.714719698292241, '(41.764, 48.785]': 5.103661464516056, '(48.785, 54.716]': 12.897671319283631, '(54.716, 57.387]': 11.153032183528166, '(57.387, 60.425]': 9.402386884179778, '(60.425, 78.429]': 23.351661877179698, '(78.429, 102.497]': 37.71973126888667}
(n=2938 matched, severity-labelled indications)

=== Stage 5: classification -- LightGBM multiclass + isotonic calibration vs majority-class baseline ===

Majority-class baseline:
    recall: scc                 0.000  [0.000, 0.000]
    recall: weld                0.000  [0.000, 0.000]
    recall: dent                0.000  [0.000, 0.000]
    recall: corrosion           0.000  [0.000, 0.000]
    recall: interference        1.000  [1.000, 1.000]
  interference precision        0.271  [0.216, 0.326]
  Brier score                   0.795  [0.785, 0.804]

LightGBM multiclass:
    recall: scc                 0.100  [0.045, 0.168]
    recall: weld                0.030  [0.000, 0.073]
    recall: dent                0.061  [0.017, 0.117]
    recall: corrosion           0.028  [0.000, 0.064]
    recall: interference        0.710  [0.624, 0.791]
  interference precision        0.265  [0.214, 0.315]
  Brier score                   0.816  [0.797, 0.835]

SCC recall: 0.100 [CI lower bound 0.045]
Stage 5 recall gate (SCC recall >= 0.90 at the CI lower bound): DOES NOT PASS
(n=2213 matched, classifiable indications)
```

## Temporal holdout (train runs 0-1, test run 2)

```
=== Stage 3: MAD baseline vs IsolationForest (grouped CV, out-of-fold) ===

MAD baseline:
  recall @ dig budget           0.122  [0.092, 0.156]
  false-dig rate                0.853  [0.817, 0.887]
    of which: interference      0.083  [0.060, 0.107]
  localisation error (m)        7.073  [5.928, 8.317]
  localisation error (cm)      707.267  [592.775, 831.727]
  PR-AUC (diagnostic)           0.035

IsolationForest:
  recall @ dig budget           0.108  [0.081, 0.142]
  false-dig rate                0.870  [0.840, 0.903]
    of which: interference      0.070  [0.047, 0.097]
  localisation error (m)        7.409  [6.090, 8.783]
  localisation error (cm)      740.902  [609.009, 878.286]
  PR-AUC (diagnostic)           0.033

Recall gap (IsolationForest - MAD): -0.014  [-0.047, 0.022]
Stage 3 gate (>= 0.15 recall gap, at the CI lower bound): DOES NOT PASS

Interference-dig-fraction gap (MAD - IsolationForest): nan  [nan, nan]
-> the recall gap is NOT clearly attributable to interference rejection at this confidence level -- reporting this honestly, as the gate requires, rather than claiming a mechanism the data doesn't support.

(360 physical defects, 30 surveys evaluated)

=== Stage 4: severity -- LightGBM quantile + split conformal vs global-mean baseline ===

Global-mean baseline:
  coverage @ 90% nominal        0.825  [0.719, 0.912]
  MAE                          18.351  [15.059, 22.302]
  mean interval width          65.799  [65.799, 65.799]

LightGBM CQR:
  coverage @ 90% nominal        0.860  [0.772, 0.947]
  MAE                          20.726  [16.967, 24.846]
  mean interval width          74.341  [72.166, 76.581]

Stage 4 gate (coverage in [0.87, 0.93]): DOES NOT PASS
MAE by severity decile: {'(25.25, 29.951]': 16.563381129002547, '(29.951, 40.651]': 9.748368657670538, '(40.651, 51.649]': 9.432971417245495, '(51.649, 57.439]': 10.206926043018633, '(57.439, 59.827]': 11.918683278497568, '(59.827, 61.597]': 22.473849984313965, '(61.597, 76.77]': 28.31315197662098, '(76.77, 82.43]': 41.08429769337296, '(82.43, 102.497]': 45.030348699635354}
(n=1192 matched, severity-labelled indications)

=== Stage 5: classification -- LightGBM multiclass + isotonic calibration vs majority-class baseline ===

Majority-class baseline:
    recall: scc                 0.000  [0.000, 0.000]
    recall: weld                0.000  [0.000, 0.000]
    recall: dent                0.000  [0.000, 0.000]
    recall: corrosion           0.000  [0.000, 0.000]
    recall: interference        1.000  [1.000, 1.000]
  interference precision        0.288  [0.228, 0.347]
  Brier score                   0.791  [0.780, 0.802]

LightGBM multiclass:
    recall: scc                 0.000  [0.000, 0.000]
    recall: weld                0.128  [0.026, 0.231]
    recall: dent                0.054  [0.000, 0.135]
    recall: corrosion           0.000  [0.000, 0.000]
    recall: interference        0.794  [0.698, 0.889]
  interference precision        0.298  [0.226, 0.363]
  Brier score                   0.809  [0.778, 0.842]

SCC recall: 0.000 [CI lower bound 0.000]
Stage 5 recall gate (SCC recall >= 0.90 at the CI lower bound): DOES NOT PASS
(n=835 matched, classifiable indications)
```

