# Stage D: the ablation ladder

Measured 2-line scale-down of `config/base.yaml`'s production density (see `scripts/ablation_ladder.py`'s module docstring).

```

====================================================================================================
ABLATION LADDER -- IsolationForest, grouped CV, out-of-fold
====================================================================================================

1. mid-head only, GPS chainage
  recall @ dig budget      0.153 [0.056, 0.278]
  false-dig rate           0.817 [0.767, 0.867]
  localisation error (cm)  854.041 [616.818, 1071.638]

2. + first difference g1
  recall @ dig budget      0.208 [0.083, 0.347]
  false-dig rate           0.750 [0.683, 0.817]
  localisation error (cm)  950.933 [773.046, 1123.970]

3. + second difference g2
  recall @ dig budget      0.208 [0.097, 0.347]
  false-dig rate           0.750 [0.650, 0.800]
  localisation error (cm)  913.105 [720.401, 1111.684]

4. + stand-off inversion/normalisation
  recall @ dig budget      0.222 [0.097, 0.361]
  false-dig rate           0.733 [0.667, 0.783]
  localisation error (cm)  886.996 [718.933, 1049.934]

5. + weld-comb registration
  recall @ dig budget      0.208 [0.097, 0.347]
  false-dig rate           0.750 [0.683, 0.800]
  localisation error (cm)  828.953 [659.269, 988.585]

6. full 3-axis vector output (hardware)
  recall @ dig budget      0.611 [0.444, 0.764]
  false-dig rate           0.211 [0.128, 0.300]
  localisation error (cm)  46.591 [38.636, 55.682]

----------------------------------------------------------------------------------------------------
Arm-to-arm recall deltas (paired bootstrap where the defect universe is shared; arm 5->6 is NOT paired -- rig:vector is a structurally different corpus/generator, compare via CI overlap only, not a paired delta):
  1->2                         0.056 [0.014, 0.111]
  2->3                         0.000 [-0.056, 0.056]
  3->4                         0.014 [-0.056, 0.083]
  4->5                         -0.014 [-0.042, 0.000]
  1->5 (software total)        0.056 [0.014, 0.111]

  5->6 (hardware headline)     software 0.208 [0.097, 0.347] vs hardware 0.611 [0.444, 0.764] -- independent CIs, point gap +0.403 (NOT a paired delta -- see note above)
```
