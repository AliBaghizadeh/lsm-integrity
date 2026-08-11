# The clean-room experiment: does in-house controlled data help the field model?

Measured at `CLEANROOM_N_LINES=12` lines per domain, production per-line density (see `scripts/cleanroom_experiment.py`'s module docstring for the scale, safety and what-this-cannot-show notes).

```

====================================================================================================
E1 -- CEILING (within-domain, diagnostic only: NOT evidence for the campaign)
====================================================================================================

[field]  7,194,643 rows, 144 physical defects, 36 surveys
  Stage 3 detect
    MAD  recall @ budget      0.134 [0.093, 0.183]
    IF   recall @ budget      0.116 [0.076, 0.162]
    recall gap (IF - MAD)     -0.019 [-0.051, 0.019]   gate(>=0.15 @ CI lo): FAIL
    IF   false-dig rate       0.861 [0.822, 0.897]
    IF   localisation (cm)    812.236 [691.004, 930.586]
  Stage 4 severity
    coverage                  0.917 [0.835, 0.998]   (baseline 0.890 [0.782, 0.973])   gate[0.87,0.93]: PASS
    MAE                       16.535 [13.682, 19.581]   (baseline 14.844 [12.033, 17.848])
  Stage 5 classify (per-class recall)
    scc                       0.140 [0.033, 0.268]
    weld                      0.120 [0.000, 0.280]
    dent                      0.232 [0.096, 0.404]
    corrosion                 0.000 [0.000, 0.000]
    interference              0.374 [0.251, 0.509]
    interference precision    0.277 [0.169, 0.385]

[cleanroom]  7,194,199 rows, 144 physical defects, 36 surveys
  Stage 3 detect
    MAD  recall @ budget      0.215 [0.167, 0.264]
    IF   recall @ budget      0.414 [0.343, 0.481]
    recall gap (IF - MAD)     0.199 [0.139, 0.255]   gate(>=0.15 @ CI lo): FAIL
    IF   false-dig rate       0.503 [0.458, 0.544]
    IF   localisation (cm)    74.935 [63.434, 93.185]
  Stage 4 severity
    coverage                  0.885 [0.836, 0.927]   (baseline 0.899 [0.860, 0.938])   gate[0.87,0.93]: PASS
    MAE                       11.606 [10.320, 13.036]   (baseline 15.751 [14.118, 17.459])
  Stage 5 classify (per-class recall)
    scc                       0.359 [0.242, 0.495]
    weld                      0.253 [0.145, 0.382]
    dent                      0.567 [0.458, 0.667]
    corrosion                 0.109 [0.046, 0.184]
    interference              n/a
    interference precision    n/a

NOTE: the two domains have DIFFERENT physical defect universes (different
corpora, different RNG trajectories), so these are independent CIs -- read them
via overlap, never as a paired delta. Same convention as ablation_ladder.py's
arm 5 -> 6 comparison.

====================================================================================================
E2 -- TRANSFER (train on one domain, test on the other's held-out lines)
====================================================================================================

Indication counts per cell (train lines ['LINE000', 'LINE001', 'LINE002', 'LINE003', 'LINE004', 'LINE005'] / test lines ['LINE006', 'LINE007', 'LINE008', 'LINE009', 'LINE010', 'LINE011']):
  train field     -> test field      severity  368/ 20src train,  434/ 17src test | classify  389/ 49src,  406/ 56src
  train field     -> test cleanroom  severity  368/ 20src train,  174/ 65src test | classify  389/ 49src,  183/ 67src
  train cleanroom -> test field      severity  159/ 62src train,  434/ 17src test | classify  166/ 66src,  406/ 56src
  train cleanroom -> test cleanroom  severity  159/ 62src train,  174/ 65src test | classify  166/ 66src,  183/ 67src

CAVEAT on the test sets: an indication only exists because the DETECTOR flagged
it, and that detector was cross-validated within its own domain across all lines
-- so a test-line indication was produced by a model that saw other folds of the
same domain. That is common-mode across every cell sharing a test domain (the two
`-> field` rows are scored on the identical indication set, likewise the two
`-> cleanroom` rows), so it cannot favour one SOURCE domain over the other, which
is the comparison being made. It does mean an absolute number here is not a
deployment estimate.

-- Stage 5 classify: per-class recall on the TEST domain's held-out lines --
  train -> test                                  scc                  weld                  dent             corrosion          interference
  field -> field                0.000 [0.000, 0.000]  0.514 [0.143, 0.857]  0.000 [0.000, 0.000]  0.037 [0.000, 0.113]  0.444 [0.233, 0.669]
  field -> cleanroom            0.000 [0.000, 0.000]  0.000 [0.000, 0.000]  0.000 [0.000, 0.000]  0.000 [0.000, 0.000]                   n/a
  cleanroom -> field            0.000 [0.000, 0.000]  0.143 [0.000, 0.429]  0.000 [0.000, 0.000]  0.875 [0.700, 1.000]  0.000 [0.000, 0.000]
  cleanroom -> cleanroom        0.333 [0.177, 0.500]  0.345 [0.167, 0.548]  0.273 [0.152, 0.424]  0.200 [0.067, 0.378]                   n/a

  train -> test                   interference precision                     brier
  field -> field                    0.273 [0.121, 0.424]      0.925 [0.850, 1.014]
  field -> cleanroom                0.000 [0.000, 0.000]      1.500 [1.414, 1.585]
  cleanroom -> field                                 n/a      0.965 [0.872, 1.055]
  cleanroom -> cleanroom                             n/a      0.841 [0.769, 0.916]

-- Stage 4 severity: conformal coverage / MAE on the TEST domain's held-out lines --
  train -> test                                 coverage            interval width                       MAE              baseline MAE
  field -> field                    0.600 [0.365, 0.835]   32.871 [29.536, 36.546]   15.832 [11.998, 20.024]   15.915 [10.713, 21.219]
  field -> cleanroom                0.718 [0.613, 0.808]   46.716 [46.646, 46.759]   17.946 [14.741, 21.504]   17.361 [14.149, 20.737]
  cleanroom -> field                0.882 [0.706, 1.000]   75.513 [70.652, 79.880]   19.362 [14.003, 24.294]   16.765 [13.692, 19.743]
  cleanroom -> cleanroom            0.933 [0.897, 0.964]   68.639 [67.457, 69.785]   13.420 [11.109, 15.975]   16.441 [13.951, 18.961]

HOW TO READ E2: the decisive comparison is the two rows ending in `-> field`.
`cleanroom -> field` is what an in-house campaign actually buys; `field -> field`
is today's status quo on the same held-out lines. If the clean-room-trained model
does not at least match the field-trained one ON FIELD DATA, the campaign does not
pay for the model, whatever the `-> cleanroom` column says.
```
