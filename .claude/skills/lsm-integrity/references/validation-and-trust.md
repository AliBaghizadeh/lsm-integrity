# Validation and trust

ROSEN's stated concern is **validation and trust**. An LSM prediction can authorise a dig
that costs tens of thousands of euros, or fail to authorise one and leave a stress-corrosion
crack in the ground. Trust is engineered in five layers, each with an owner and a gate.

| Layer | Question it answers | Gate |
|---|---|---|
| 1 Data | Is this survey fit to score at all? | `validate` exits non-zero |
| 2 Model | Does it generalise to an unseen pipeline? | grouped CV thresholds |
| 3 Release | Is this version better than what is deployed? | CI promotion gates |
| 4 Runtime | Is today's data like training data? | drift alarms |
| 5 Human | Can an engineer see *why*? | explanations + intervals |

---

## Layer 1 — Data validation (`validate.py`)

Runs on ingest **and again inside `predict`**, same code path. Emits one row per check into
`dq_report` and a JSON artifact to MLflow/S3. `fail` blocks the pipeline; `warn` propagates
to `indication.dq_flag` so downstream users see it.

| Check | Definition | Default gate |
|---|---|---|
| `schema` | the declared `schemas.py` contract: columns, dtypes, units, enum membership, no nulls in required fields | fail |
| `range` | each axis within `field_range_nT` (±80 000 nT) | fail |
| `duplicate_content` | `content_sha256` already present under another `survey_id` | fail |
| `survey_overlap` | chainage interval intersects another survey on the same `line_id`; if it does, cross-correlate the detrended residual in the overlap | fail if r > 0.9, else warn |
| `saturation` | ≥3 consecutive identical raw values on any axis (ADC rail / stuck sensor) | fail |
| `sample_idx_monotonic` | strictly increasing within a survey | fail |
| `sample_idx_gap` | no gap > `max_gap_m` / `step_m`; report total missing length | warn |
| `duplicate_sample_idx` | no repeated `sample_idx` (enforced by the PK; surfaced by name so the message is useful) | fail |
| `gps_jump` | haversine step ≤ `max_gps_jump_m` (5 m); catches GPS dropout/multipath | warn |
| `gps_chainage_consistency` | cumulative GPS distance vs chainage within 2% | warn |
| `noise_floor` | high-pass residual σ within [1, 15] nT per axis; too low = dead channel, too high = interference-rich survey | warn |
| `background_regime` | median field per axis within 3σ of the historical median for that line | warn |
| `interference_density` | fraction of rows with |residual| > 5σ above the expected rate | warn |
| `coverage` | surveyed length ≥ 95% of the registered line length | warn |

**Rejection is normal operations, not a crash.** A survey failing a hard gate is written to
`quarantine/` with its DQ report attached and recorded as `status='quarantined'`; it is
never deleted and never silently skipped. `validate` exits non-zero so CI fails, but the
production path raises a ticket and moves on to the next survey. A pipeline that halts the
whole nightly run because one survey had a stuck channel will be switched off by its
operators within a month.

Design notes worth saying aloud: gates are **configurable per line**, because a survey over
a rail crossing legitimately has more interference than one over farmland; and a `warn`
never silently disappears — it is attached to every indication produced from that survey.
The full dtype/unit/null/enum contract these checks enforce lives in `data-contract.md`;
it is declared once and validated at every boundary, not re-checked ad hoc per module.

## Layer 2 — Model validation (`evaluate.py`)

**Splitting.** `GroupKFold` on `line_id`; when only one line exists, on 100 m chainage
blocks — with the group key **`(line_id, block)`, never `block` alone**, so that two
partially overlapping surveys of the same stretch cannot land in different folds. Never
random. Never row-level. The same physical defect appears in all three surveys, so the
group key must be *segment identity*, not row identity. `tests/test_leakage.py` asserts
this and is a required CI check. Fold assignment is by `blake3(group_key) % n_folds`, not
positional, so growing the archive does not reshuffle existing folds and invalidate
historical comparisons.

**Baselines, always.** Robust-MAD threshold on |residual| for detection; global-mean
severity for regression; majority class for classification; "no growth" for forecasting.
A model that does not beat these by a stated margin is not reported as a success.

**Metrics.** Row-level metrics are diagnostics; **indication-level metrics are the result.**

| Task | Primary | Supporting | Gate |
|---|---|---|---|
| Detection | recall @ dig budget (top-5 indications/km) | PR-AUC, false-dig rate, localisation error (m) | recall ≥ 0.80 @ budget, ≥ 0.15 above MAD baseline |
| Severity | conformal coverage @ 90% nominal | MAE, mean interval width, MAE by severity decile | coverage ∈ [0.87, 0.93] |
| Classification | macro-F1 | **SCC recall**, interference precision, reliability diagram, Brier | SCC recall ≥ 0.90 |
| Growth | MAE on held-out survey 2 | remaining-life error in years, interval coverage | beats "no growth" baseline |

`ROC-AUC` is banned in reporting: positives are ~2.7% of rows and ROC flatters that.

**Every headline metric is reported with a bootstrap confidence interval, resampled over
*groups* (lines / defects), never over rows.** With ~12 defects per line, "recall@budget =
0.83" is noise wearing a decimal point, and a gate comparing two point estimates will
promote models on sampling variation. Report `0.83 [0.61, 0.94]`, and make the promotion
gate compare intervals, not points. This is the same intellectual honesty as the conformal
intervals, applied to your own evaluation — and it pre-empts "how sure are you about that
number?", which is the obvious follow-up question.

**Two evaluation axes, both required.**
1. *Spatial holdout* — unseen line: does it generalise to a new pipeline?
2. *Temporal holdout* — train on surveys 0–1, test on survey 2: does it survive a new
   survey with a different background drift realisation? This is the one that matters in
   production, where you always score data acquired after training.

**Honest limitations to state, not hide:** 12 defects per line is too few for a stable
supervised estimate — hence the scale rehearsal; three surveys is too few to fit growth
curves independently — hence partial pooling; and the labels are a ±2 m box, so
localisation error below ~2 m is not measurable.

## Layer 3 — Release gates (CI)

`train.yml` cuts a `pipeline_release` and sets the **`@challenger` alias** only if **all**
hold (MLflow 3 removed model stages — promotion is an alias move, not a state transition):
1. Every Layer-1 `fail` gate passed on the training data.
2. `tests/test_leakage.py`, `tests/test_determinism.py` and the golden-dataset regression
   test are green.
3. Primary metric beats the current `production` model on the same holdout at the same
   `truth_as_of`, or beats the baseline by the margin above if no production model exists —
   **compared on bootstrap intervals, not point estimates.**
4. Conformal coverage within the stated band — an interval that does not cover is worse
   than no interval.
5. No feature in the top-10 SHAP list is a known leak (`defect`, `severity_smys`,
   `defect_type`, anything derived from them). Enforced by an explicit denylist test.
6. Bundle round-trips: `load(save(model))` reproduces predictions exactly, and the bundle's
   `feature_version` / `schema_version` match the current code.

Moving **`@challenger` → `@champion`** is a human decision recorded on the registered
version, with the reason. Deployment pins a `pipeline_version`; rollback repoints
`@champion`, and **CI drills that path** — the last N = 3 feature versions must stay
loadable, or gate 6 blocks the rollback it was meant to protect.

**Before promotion, the challenger runs in shadow.** Both releases score every live survey,
the challenger's rows flagged `is_shadow=1`. Since outcomes are months away, the decision
evidence is a **disagreement report**: change in indications/km, rank churn at the dig
budget, and whether disagreements concentrate in DQ-warned surveys. This is the only
pre-label validation this domain allows.

**Holdout-reuse discipline.** Comparing every candidate against the incumbent on one fixed
holdout selects on holdout noise after enough attempts — multiple comparisons applied to the
release process itself. Keep a final test set touched rarely, count its uses in
`model_run.final_test_uses`, and rotate it as new lines arrive.

## Layer 4 — Runtime monitoring

- **Feature drift**: PSI and two-sample KS of each production feature against the bundle's
  stored training reference. PSI > 0.2 warns, > 0.3 blocks scoring pending review.
- **Prediction drift**: distribution of `p_defect_cal` and indications-per-km vs the
  training-set rate. A survey suddenly producing 3× the usual indications is a sensor or
  background problem until proven otherwise.
- **Background regime shift**: median and drift slope per axis vs the line's history —
  catches recalibration, a different scanner unit, or a seasonal geomagnetic excursion.
- **Coverage tracking**: measured empirical coverage of the severity intervals against
  nominal, recomputed as verification results arrive. If real coverage drifts below
  nominal, re-fit the conformal quantiles before touching the model.

## Layer 5 — Human trust

- **Uncertainty is a first-class output.** Split conformal gives a distribution-free 90%
  interval with a finite-sample guarantee under exchangeability — and exchangeability is
  exactly what a new line or a new scanner breaks, which is why Layer 4 exists. State both
  halves of that sentence; the guarantee without the caveat is overselling.
- **Explanations**: global SHAP for the model, local SHAP for each indication above the
  dig threshold. Plus a **physics consistency check** — the model should key on residual
  amplitude, along-track gradient and peak width. If it keys on absolute chainage or raw
  field magnitude, it has found a shortcut, and that is a finding, not a nuisance.
- **Model card per version**, auto-generated at train time: intended use, training corpus
  (`data_sha256`, line and defect counts), metrics *with intervals*, known limitations,
  and the explicit statement that the data is synthetic. Cheap to generate, and it is the
  artifact an auditor or a customer's integrity engineer actually asks for.
- **The dig-feedback loop**: an excavation writes a row into `truth_defect` /
  `truth_observation` with `source='excavation'`. That closes the loop — every dig is a
  label, labels are the scarce asset, and the ranking should be evaluated against them
  quarterly. Active learning falls straight out: rank by *uncertainty × consequence*, not
  by severity alone, when choosing which segment to dig next.
- **Label corrections are revisions, not edits.** A dig can reclassify a defect. Because
  `truth_defect` is slowly-changing (`revision` / `valid_from` / `valid_to`) and every
  evaluation pins a `truth_as_of`, a correction cannot retroactively rewrite last
  quarter's reported metrics. A model whose historical track record silently changes has
  no track record.
- **Ranked output, not a verdict.** The deliverable to the integrity engineer is a ranked,
  budget-aware list with intervals and DQ flags. The model does not decide to dig; it
  decides what to look at first.
