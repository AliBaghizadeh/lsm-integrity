# Stage D: the ablation ladder

Measured 2-line scale-down of `config/base.yaml`'s production density (see `scripts/ablation_ladder.py`'s module docstring).

```

====================================================================================================
ABLATION LADDER -- IsolationForest, grouped CV, out-of-fold
====================================================================================================

1. mid-head only, GPS chainage
  recall @ dig budget      0.194 [0.069, 0.333]
  false-dig rate           0.767 [0.717, 0.833]
  localisation error (cm)  617.069 [378.531, 858.758]

2. + first difference g1
  recall @ dig budget      0.208 [0.083, 0.375]
  false-dig rate           0.750 [0.717, 0.783]
  localisation error (cm)  424.585 [243.952, 632.201]

3. + second difference g2
  recall @ dig budget      0.194 [0.069, 0.361]
  false-dig rate           0.767 [0.733, 0.800]
  localisation error (cm)  306.191 [157.895, 489.903]

4. + stand-off inversion/normalisation
  recall @ dig budget      0.222 [0.083, 0.389]
  false-dig rate           0.733 [0.700, 0.767]
  localisation error (cm)  439.233 [235.166, 669.837]

5. + weld-comb registration
  recall @ dig budget      0.208 [0.069, 0.361]
  false-dig rate           0.750 [0.717, 0.783]
  localisation error (cm)  535.702 [320.796, 744.704]

6. full 3-axis vector output (hardware)
  recall @ dig budget      0.597 [0.430, 0.764]
  false-dig rate           0.272 [0.233, 0.311]
  localisation error (cm)  48.256 [40.116, 57.558]

----------------------------------------------------------------------------------------------------
Arm-to-arm recall deltas (paired bootstrap where the defect universe is shared; arm 5->6 is NOT paired -- rig:vector is a structurally different corpus/generator, compare via CI overlap only, not a paired delta):
  1->2                         0.014 [-0.042, 0.083]
  2->3                         -0.014 [-0.056, 0.028]
  3->4                         0.028 [-0.028, 0.097]
  4->5                         -0.014 [-0.042, 0.000]
  1->5 (software total)        0.014 [-0.028, 0.056]

  5->6 (hardware headline)     software 0.208 [0.069, 0.361] vs hardware 0.597 [0.430, 0.764] -- independent CIs, point gap +0.389 (NOT a paired delta -- see note above)
```
