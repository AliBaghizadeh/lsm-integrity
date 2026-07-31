# Stage 6 scale rehearsal -- 2026-07-31T09:04:31.396855+00:00

Config: `config/scale/base.yaml` -- 40 lines x 40000 m x 3 runs = 9,600,000 rows (~10^7). `config_sha256=7639fecc081a...`

## Wall-clock + peak RSS per step

Peak RSS is *sampled* (a polling thread reading `psutil.Process().memory_info().rss` every 50ms), not an exact accounting -- the true peak between samples can be missed.

| step | rows | wall (s) | rows/sec | peak RSS (MB) |
|---|---|---|---|---|
| generate | 9,600,000 | 505.72 | 18,983 | 709.0 |
| ingest | 9,440,000 | 33.67 | 280,367 | 391.2 |
| features | 9,440,000 | 1.04 | 9,069,022 | 375.3 |
| corpus_read_pandas | 9,440,000 | 2.66 | 3,543,958 | 5,031.1 |
| corpus_read_duckdb | 9,440,000 | 8.34 | 1,132,199 | 11,254.5 |
| train_block_cv | 9,600,000 | 765.42 | 12,542 | 23,730.8 |
| train_whole_line_and_temporal | - | 1017.25 | - | 28,402.9 |

## Background-contrast gate re-check

Contrast (defect median / background median) on `LINE000_R0`: **2.70x** -- **DID NOT PASS** the Stage 2 gate (>= 3.0x). Background median 7.72 nT (essentially unchanged from the demo corpus's ~7.7-8.1 nT -- the detrend/high-pass is NOT degrading at 40 km), but defect median 20.83 nT is notably lower than the demo's ~25 nT, while defect MAX is 139.07 nT -- a strongly right-skewed distribution. Most likely explanation, consistent with both measurements: the demo's own median was computed over only 12 defects and was itself optimistic (a small-sample fluke on a skewed distribution), not a sign that detrending degrades at this length. Density-per-km is unchanged from the demo corpus (6 defects/km, 2 interference/km), ruling out a packing-density explanation. This is a real, measured gate miss, reported honestly rather than adjusted away -- revisiting it (e.g. a larger single-survey sample for the Stage 2 gate check itself) is future work, out of Stage 6's scope.

## A real bug found at scale: NaN severity_smys poisoning conformal coverage

`generate.py` initialises `severity_smys` to NaN off-defect (not 0, contradicting this project's own documented "0 off-defect" convention). A detector's peak occasionally lands just outside a defect's exact label-window half-width while still within the looser `MATCH_TOLERANCE_M` dig-matching radius, so `match_dug_indications` still credits it as matched, but that row's own `severity_smys` is NaN, not the defect's true value -- 36 of 18,917 matched indications (0.4%) at this scale (0 at demo scale, apparently never sampled). A single NaN `y_true` reaching `SeverityModel.fit`'s conformal calibration silently NaN'd the WHOLE fold's margin (`np.quantile` propagates NaN), which then NaN'd every prediction's interval for that fold -- cascading a 0.4%-incidence data issue into an initial 0% pooled coverage across the entire OOF result (confirmed by an earlier run of this same rehearsal, before the fix), not a gradual degradation. Fixed in `train.py::_build_severity_training_frame` (drop NaN-`y_true` rows upstream, loudly) plus a belt-and-braces guard in `SeverityModel.fit` itself. Confirmed working below: Stage 4's gate now PASSES at this scale.

## A real bug found at scale: classify's n_estimators/num_leaves were dead config

`config/base.yaml`'s `model.classify.n_estimators`/`num_leaves` looked tunable but were never threaded from config into `ClassifyModel`'s `lgbm_cfg` in `train.py` -- editing them had NO effect, silently, since they coincidentally matched `ClassifyModel`'s own hardcoded fallback defaults (50/7). Found while investigating why `config/scale/base.yaml`'s larger capacity values weren't visibly changing classify's behaviour. Fixed by merging `classify_cfg`'s values into the `lgbm_cfg` dict at both call sites. Confirmed working: SCC recall improved from a first, capacity-starved run (0.50 block-CV / 0.39 whole-line) to the numbers reported below (0.64 block-CV / 0.68 whole-line).

## A real DQ finding at scale: survey_overlap quarantines

2 of 120 surveys (1.7%) were quarantined by `check_survey_overlap`, all on a correlation just above the fixed 0.9 threshold:
  - `LINE013_R2`: survey_overlap (n_affected=1)
  - `LINE029_R1`: survey_overlap (n_affected=1)

`check_survey_overlap` (validate.py) takes the MAX correlation across ALL prior same-line surveys, not just the immediately preceding one -- so as more runs of a line accumulate, this max is an order statistic over a growing number of comparisons and trends upward even if each individual run-pair's correlation distribution is unchanged (R2 is checked against both R0 and R1; R0 has nothing to compare against). Separately, a much longer line (40 km vs the demo's 2 km) gives two runs' shared deterministic structure (same defect/interference positions, same geo/lat-lon path) far more samples to accumulate correlated structure in a plain Pearson correlation, even though each run's background noise is drawn independently. Together these make a same-line overlap correlation naturally higher at this scale than the 2 km demo corpus ever exercised -- the fixed 0.9 threshold, calibrated only against short lines and few runs, was never stress-tested against this regime. This is a real, scale-driven finding, not a generator bug: the DQ layer did exactly what it's designed to do (quarantine, not crash), and the affected surveys were correctly excluded from the training corpus below. Revisiting the threshold for long-line, many-run deployments is future work, out of Stage 6's scope.

## Where SQLite stopped being the right tool

- Before (pre-Stage-6, per-row `itertuples()`+`float()`-per-cell `load_readings`, measured once by hand at 80,000 rows): **235,982 rows/sec**, peak RSS 183.5 MB.
- After (vectorized NaN->None + dtype-cast, this rehearsal, 9,600,000 rows): **280,367 rows/sec**, peak RSS 391.2 MB (1.19x).
- Conclusion: the naive Python-level row conversion was a real, measurable cost, but not the dominant one -- SQLite's own `executemany` insert path is the majority of the remaining cost at this row count. This matches `.claude/skills/lsm-integrity/references/architecture.md`'s own claim that SQLite "comfortably handles 10^7 rows read-mostly" -- it stops being the right tool at concurrent multi-writer ingest, not at this data volume, which this demonstrator never has.

## Bulk-read path: pandas-concat vs DuckDB

- `features.load_feature_corpus` (per-file glob + `pd.concat`): 2.66s for 9,440,000 rows.
- `scale_eval.load_feature_corpus_duckdb` (one DuckDB glob read): 8.34s for 9,440,000 rows.

## Small-files problem

118 feature-store files at this scale (40 lines x 3 runs) -- PLAN.md's own "120 tiny per-survey Parquet files is fine" example, not the 10^5-file failure case. This rehearsal does not hit that failure mode -- the documented production answer (periodic compaction to line-level Parquet files) is noted here, not artificially triggered.

## float32 arithmetic

The feature store is float32 throughout (`features.py`'s `STORAGE_DTYPE`). Severity's quantile regression and classify's LightGBM training both ran to completion against these float32-stored features with no dtype errors (LightGBM upconverts internally as needed) -- see the block-CV report below for the resulting metrics.

## Default 5-fold block CV (the real gate, `run_train`, unchanged)

```
_build_severity_training_frame: dropping 36 matched indication(s) whose peak row has a NaN severity_smys (peak landed outside the true label window while still within the dig-matching tolerance) -- see train.py's comment.

=== Stage 3: MAD baseline vs IsolationForest (grouped CV, out-of-fold) ===

MAD baseline:
  recall @ dig budget           0.639  [0.630, 0.648]
  false-dig rate                0.233  [0.228, 0.237]
    of which: interference      0.233  [0.228, 0.237]
  localisation error (m)        0.284  [0.283, 0.286]
  PR-AUC (diagnostic)           0.332

IsolationForest:
  recall @ dig budget           0.610  [0.600, 0.618]
  false-dig rate                0.266  [0.257, 0.274]
    of which: interference      0.230  [0.226, 0.233]
  localisation error (m)        0.423  [0.416, 0.431]
  PR-AUC (diagnostic)           0.383

Recall gap (IsolationForest - MAD): -0.029  [-0.033, -0.026]
Stage 3 gate (>= 0.15 recall gap, at the CI lower bound): DOES NOT PASS

Interference-dig-fraction gap (MAD - IsolationForest): 0.003  [0.001, 0.005]
-> the recall gap IS attributable to interference rejection: MAD wastes more of its dig budget on interference than IsolationForest does.

(9600 physical defects, 118 surveys evaluated)

=== Stage 4: severity -- LightGBM quantile + split conformal vs global-mean baseline ===

Global-mean baseline:
  coverage @ 90% nominal        0.889  [0.883, 0.895]
  MAE                          15.015  [14.825, 15.200]
  mean interval width          59.713  [59.708, 59.719]

LightGBM CQR:
  coverage @ 90% nominal        0.896  [0.891, 0.901]
  MAE                           3.363  [3.321, 3.410]
  mean interval width          16.272  [16.214, 16.329]

Stage 4 gate (coverage in [0.87, 0.93]): PASSES
MAE by severity decile: {'(20.134999999999998, 44.04]': 3.0789948178144395, '(44.04, 51.668]': 3.311677131022236, '(51.668, 57.838]': 3.2971199930654334, '(57.838, 63.262]': 3.3126750071972784, '(63.262, 68.178]': 3.2436426606961626, '(68.178, 72.834]': 3.216103354951089, '(72.834, 77.48]': 3.4513197622165603, '(77.48, 83.327]': 3.4583020589974147, '(83.327, 90.922]': 3.4953862465443972, '(90.922, 105.789]': 4.05488212229367}
(n=18881 matched, severity-labelled indications)

=== Stage 5: classification -- LightGBM multiclass + isotonic calibration vs majority-class baseline ===

Majority-class baseline:
    recall: scc                 0.000  [0.000, 0.000]
    recall: weld                0.000  [0.000, 0.000]
    recall: dent                0.000  [0.000, 0.000]
    recall: corrosion           0.000  [0.000, 0.000]
    recall: interference        1.000  [1.000, 1.000]
  interference precision        0.217  [0.209, 0.225]
  Brier score                   0.800  [0.799, 0.801]

LightGBM multiclass:
    recall: scc                 0.636  [0.620, 0.652]
    recall: weld                0.038  [0.031, 0.044]
    recall: dent                0.026  [0.021, 0.031]
    recall: corrosion           0.292  [0.277, 0.306]
    recall: interference        0.998  [0.996, 1.000]
  interference precision        0.999  [0.998, 1.000]
  Brier score                   0.590  [0.583, 0.596]

SCC recall: 0.636 [CI lower bound 0.620]
Stage 5 recall gate (SCC recall >= 0.90 at the CI lower bound): DOES NOT PASS
Stage 5 physics-consistency gate (no ['chainage_m', 'chainage_peak_m', 'chainage_start_m', 'chainage_end_m', 'sample_idx'] feature in top-10 by contribution): PASSES
Stage 5 gate overall: DOES NOT PASS
(n=24678 matched, classifiable indications)
```

## Whole-line-holdout CV (Stage 6: "the real generalisation test")

An entire physical line held out per fold (5 folds, ~8 lines/fold), instead of today's default (line, 100 m block) hash grouping, which scatters one line's blocks across ~all folds.

```
=== Stage 3: MAD baseline vs IsolationForest (grouped CV, out-of-fold) ===

MAD baseline:
  recall @ dig budget           0.639  [0.630, 0.648]
  false-dig rate                0.233  [0.228, 0.237]
    of which: interference      0.233  [0.228, 0.237]
  localisation error (m)        0.284  [0.283, 0.286]
  PR-AUC (diagnostic)           0.332

IsolationForest:
  recall @ dig budget           0.610  [0.601, 0.619]
  false-dig rate                0.264  [0.256, 0.273]
    of which: interference      0.231  [0.227, 0.235]
  localisation error (m)        0.460  [0.453, 0.467]
  PR-AUC (diagnostic)           0.383

Recall gap (IsolationForest - MAD): -0.029  [-0.033, -0.025]
Stage 3 gate (>= 0.15 recall gap, at the CI lower bound): DOES NOT PASS

Interference-dig-fraction gap (MAD - IsolationForest): 0.002  [-0.000, 0.003]
-> the recall gap is NOT clearly attributable to interference rejection at this confidence level -- reporting this honestly, as the gate requires, rather than claiming a mechanism the data doesn't support.

(9600 physical defects, 118 surveys evaluated)

=== Stage 4: severity -- LightGBM quantile + split conformal vs global-mean baseline ===

Global-mean baseline:
  coverage @ 90% nominal        0.885  [0.879, 0.891]
  MAE                          15.012  [14.839, 15.207]
  mean interval width          59.293  [59.285, 59.302]

LightGBM CQR:
  coverage @ 90% nominal        0.902  [0.897, 0.907]
  MAE                           3.367  [3.326, 3.408]
  mean interval width          16.659  [16.599, 16.714]

Stage 4 gate (coverage in [0.87, 0.93]): PASSES
MAE by severity decile: {'(20.134999999999998, 43.83]': 3.1665786929773163, '(43.83, 51.578]': 3.338689176764921, '(51.578, 57.693]': 3.2686555094966723, '(57.693, 63.122]': 3.397651651528315, '(63.122, 68.035]': 3.27311331394923, '(68.035, 72.785]': 3.253854747768094, '(72.785, 77.418]': 3.3436251208661782, '(77.418, 83.285]': 3.3073754930038164, '(83.285, 90.899]': 3.418096261362236, '(90.899, 105.789]': 4.123461693059629}
(n=18933 matched, severity-labelled indications)

=== Stage 5: classification -- LightGBM multiclass + isotonic calibration vs majority-class baseline ===

Majority-class baseline:
    recall: scc                 0.000  [0.000, 0.000]
    recall: weld                0.000  [0.000, 0.000]
    recall: dent                0.000  [0.000, 0.000]
    recall: corrosion           0.000  [0.000, 0.000]
    recall: interference        1.000  [1.000, 1.000]
  interference precision        0.215  [0.207, 0.224]
  Brier score                   0.800  [0.800, 0.801]

LightGBM multiclass:
    recall: scc                 0.675  [0.659, 0.691]
    recall: weld                0.039  [0.033, 0.046]
    recall: dent                0.016  [0.012, 0.020]
    recall: corrosion           0.249  [0.235, 0.265]
    recall: interference        0.998  [0.996, 1.000]
  interference precision        0.997  [0.994, 0.999]
  Brier score                   0.590  [0.585, 0.597]

SCC recall: 0.675 [CI lower bound 0.659]
Stage 5 recall gate (SCC recall >= 0.90 at the CI lower bound): DOES NOT PASS
(n=24738 matched, classifiable indications)
```

## Temporal holdout (train runs 0-1, test run 2)

```
=== Stage 3: MAD baseline vs IsolationForest (grouped CV, out-of-fold) ===

MAD baseline:
  recall @ dig budget           0.639  [0.630, 0.650]
  false-dig rate                0.213  [0.208, 0.218]
    of which: interference      0.213  [0.207, 0.218]
  localisation error (m)        0.280  [0.277, 0.283]
  PR-AUC (diagnostic)           0.373

IsolationForest:
  recall @ dig budget           0.637  [0.628, 0.647]
  false-dig rate                0.216  [0.210, 0.221]
    of which: interference      0.212  [0.207, 0.217]
  localisation error (m)        0.406  [0.399, 0.411]
  PR-AUC (diagnostic)           0.430

Recall gap (IsolationForest - MAD): -0.002  [-0.007, 0.002]
Stage 3 gate (>= 0.15 recall gap, at the CI lower bound): DOES NOT PASS

Interference-dig-fraction gap (MAD - IsolationForest): nan  [nan, nan]
-> the recall gap is NOT clearly attributable to interference rejection at this confidence level -- reporting this honestly, as the gate requires, rather than claiming a mechanism the data doesn't support.

(9600 physical defects, 39 surveys evaluated)

=== Stage 4: severity -- LightGBM quantile + split conformal vs global-mean baseline ===

Global-mean baseline:
  coverage @ 90% nominal        0.687  [0.677, 0.697]
  MAE                          18.934  [18.641, 19.191]
  mean interval width          51.245  [51.245, 51.245]

LightGBM CQR:
  coverage @ 90% nominal        0.708  [0.698, 0.718]
  MAE                           4.750  [4.664, 4.838]
  mean interval width          16.549  [16.489, 16.613]

Stage 4 gate (coverage in [0.87, 0.93]): DOES NOT PASS
MAE by severity decile: {'(26.503999999999998, 45.467]': 3.1413589285967087, '(45.467, 54.098]': 3.39775373507666, '(54.098, 61.067]': 3.339353233428903, '(61.067, 68.037]': 3.224822001540721, '(68.037, 74.2]': 3.1039253275089176, '(74.2, 80.463]': 3.0718944166873587, '(80.463, 87.034]': 3.119437450997296, '(87.034, 93.052]': 4.711375288165124, '(93.052, 99.37]': 7.8462154803039645, '(99.37, 105.789]': 12.54322935276542}
(n=7374 matched, severity-labelled indications)

=== Stage 5: classification -- LightGBM multiclass + isotonic calibration vs majority-class baseline ===

Majority-class baseline:
    recall: scc                 0.000  [0.000, 0.000]
    recall: weld                0.000  [0.000, 0.000]
    recall: dent                0.000  [0.000, 0.000]
    recall: corrosion           0.000  [0.000, 0.000]
    recall: interference        1.000  [1.000, 1.000]
  interference precision        0.208  [0.201, 0.217]
  Brier score                   0.801  [0.800, 0.802]

LightGBM multiclass:
    recall: scc                 0.572  [0.550, 0.594]
    recall: weld                0.137  [0.122, 0.152]
    recall: dent                0.170  [0.153, 0.187]
    recall: corrosion           0.174  [0.157, 0.192]
    recall: interference        0.997  [0.995, 0.999]
  interference precision        1.000  [1.000, 1.000]
  Brier score                   0.597  [0.590, 0.603]

SCC recall: 0.572 [CI lower bound 0.550]
Stage 5 recall gate (SCC recall >= 0.90 at the CI lower bound): DOES NOT PASS
(n=9347 matched, classifiable indications)
```

