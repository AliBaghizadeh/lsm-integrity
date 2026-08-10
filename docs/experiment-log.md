# Experiment log — what changed, what it moved

**Purpose.** An append-only, chronological record of every intervention and its measured
effect. One entry per change: *what was changed → what was measured → what moved.*

This exists because the project's other documents are **snapshots**, and snapshots lose
causality:

- `PLAN.md`'s status table shows where each stage stands *now*.
- `LSM_PROJECT.md`'s results section shows the *current* measured numbers.
- `serving/model_card.md` shows the *currently released* bundle.

All three overwrite superseded values in place. That is correct for their jobs, but it means
you cannot look at any of them and answer "which change caused severity coverage to move?"
This file answers that, and is **never rewritten** — entries are appended, and a number that
turns out to be superseded stays visible with its successor next to it.

> **Reading rule — the vector/scalar boundary.** Entries 1–4 were measured on the pre-Rig-v2
> **vector** rig (a single 3-axis magnetometer on a fixed 0.5 m grid). Entries 5+ are on the
> Rig-v2 **3-head scalar** rig (total-field `|B|` only, human walker, GPS dropout). These are
> different instruments measuring different physical quantities. **Do not read a delta across
> that boundary as an effect of anything except the rig change itself.** Within each side,
> deltas are meaningful.

---

## Metric trajectories at a glance

### Stage 3 — detection recall gap (IsolationForest − MAD) · gate: ≥ +0.15 at CI lower bound

| # | Date | Intervention | Defects | Recall gap | Verdict |
|---|---|---|---|---|---|
| 1 | 2026-07-30 | Initial corpus | 12 | −0.056 [−0.167, 0.056] | FAIL — CI far too wide to interpret |
| 2 | 2026-07-30 | `n_lines` 1→5 (more independent lines) | 60 | −0.028 [−0.083, 0.022] | FAIL — CI ~halved, effect still unresolved |
| 3 | 2026-07-30 | Stage 2.75 EDA fix: dropped 2 duplicate features, added `emphasis_repeats` | 60 | −0.006 [−0.067, 0.050] | FAIL — gap closed to ~0 |
| 4 | 2026-07-31 | Stage 6 scale rehearsal (vector rig) | 9,600 | **−0.029 [−0.033, −0.026]** | FAIL — CI now **excludes zero**: a real, small, negative effect |
| — | | *— rig change: vector → 3-head scalar —* | | | |
| 5 | 2026-08-06 | Rig-v2 rebuild | 60 | −0.006 [−0.061, 0.056] | FAIL |
| 6 | 2026-08-09 | Interference/defect physics refinement | 60 | 0.000 [−0.050, 0.061] | FAIL |
| 7 | 2026-08-10 | Scale rehearsal — block CV | 360 | −0.007 [−0.030, 0.017] | FAIL |
| 7b | 2026-08-10 | *same run* — whole-line holdout | 360 | −0.013 [−0.037, 0.012] | FAIL |
| 7c | 2026-08-10 | *same run* — temporal holdout | 360 | −0.014 [−0.047, 0.022] | FAIL |

**What this trajectory says:** nine measurements, four interventions, two rigs, three grouping
schemes — the gap never approaches +0.15, and the two *harder* splits push it slightly further
negative rather than toward the gate. This is not a tuning problem. Entry 4 remains the strongest
single result: at 9,600 defects the CI excluded zero, establishing IsolationForest is *genuinely
slightly worse*, not merely indistinguishable.

### Stage 4 — severity conformal coverage · gate: inside [0.87, 0.93]

| # | Date | Intervention | Matched indications | Coverage | vs baseline | Verdict |
|---|---|---|---|---|---|---|
| 1 | 2026-07-30 | Initial corpus | ~small | 0.727 | — | FAIL — undercovering |
| 2 | 2026-07-30 | `n_lines` 1→5 | ~137 | 0.905 → **0.920** [0.867, 0.967] | beats | **PASS** |
| 3 | 2026-07-31 | Scale rehearsal (vector) | large | **0.896** [0.891, 0.901] | beats | **PASS** |
| — | | *— rig change: vector → 3-head scalar —* | | | | |
| 4 | 2026-08-06 | Rig-v2 rebuild | 293 | 0.689 [0.420, 0.945] | beats (0.589) | FAIL |
| 5 | 2026-08-09 | Physics refinement | 555 | 0.618 [0.386, 0.834] | **under** (0.667) | FAIL — worse than predicting the mean |
| 6 | 2026-08-10 | Scale rehearsal under Rig-v2 — **block CV** | 3,741 | **0.872** [0.805, 0.928] | beats (0.838) | **PASS** |
| 6b | 2026-08-10 | *same run* — **whole-line holdout** | 2,938 | 0.845 [0.772, 0.911] | ~ties (0.840) | FAIL |
| 6c | 2026-08-10 | *same run* — **temporal holdout** | 1,192 | 0.860 [0.772, 0.947] | beats (0.825) | FAIL |

**What this trajectory says — the clearest causal finding in the project, with an important
qualifier.** Coverage recovered *twice* purely by adding calibration data (entry 1→2 on the vector
rig, entry 5→6 on the scalar rig), with no modelling change either time. Entry 5 looked alarming —
worse than a naive global-mean baseline — and it would have been easy to conclude the conformal
layer was broken under the scalar rig. It was **data-limited, not broken**.

**But the recovery is not robust to how honestly you split.** Stage 4 passes under block CV
(0.872), the grouping the gate is actually configured against — and fails under both harder tests
(0.845, 0.860), landing just below the band. So the defensible claim is *"more data moved severity
from badly broken to marginal,"* **not** *"more data fixed severity."* Quoting 6 without 6b/6c
would be the flattering read. This still closes the open question `LSM_PROJECT.md` and `PLAN.md`
both recorded as unanswered — the answer is "yes, substantially, but not all the way."

### Stage 5 — classification · gate: SCC recall ≥ 0.90 at CI lower bound

| # | Date | Intervention | SCC recall | Interference precision | Verdict |
|---|---|---|---|---|---|
| 1 | 2026-07-30 | Demo corpus (vector) | 0.125 [0.000, 0.264] | 0.909 | FAIL |
| 2 | 2026-07-31 | Scale rehearsal (vector) + dead-config fix | **0.636** [0.620, 0.652] | **0.999** | FAIL (but interference essentially solved) |
| — | | *— rig change: vector → 3-head scalar —* | | | |
| 3 | 2026-08-06 | Rig-v2 rebuild | 0.042 [0.000, 0.125] | 0.389 | FAIL |
| 4 | 2026-08-09 | Physics refinement (type-specific severity) | 0.023 [0.000, 0.068] | 0.333 [0.167, 0.500] | FAIL — but weld/dent/corrosion recall went ~0 → 0.258/0.144/0.171 |
| 5 | 2026-08-10 | Scale rehearsal — block CV | 0.129 [0.062, 0.207] | 0.271 [0.216, 0.342] · *baseline 0.281* | FAIL — **below** its own baseline |
| 5b | 2026-08-10 | *same run* — whole-line holdout | 0.100 [0.045, …] | 0.265 [0.214, 0.315] · *baseline 0.271* | FAIL — **below** baseline |
| 5c | 2026-08-10 | *same run* — temporal holdout | 0.000 | 0.298 [0.226, 0.363] · *baseline 0.288* | FAIL — ~ties baseline |

**What this trajectory says, and a correction it forces.** Entry 2's **0.999 interference
precision** was for a long time the project's strongest positive result — "the supervised
classifier has essentially solved defect-vs-buried-junk." Entries 5/5b/5c are the direct
scalar-rig test of whether that survives, and **it does not** — and the failure is worse than
"doesn't reach 0.999." Across all three splits the model sits at 0.265–0.298, and in two of the
three it is **at or below the trivial majority-class baseline**. On the scalar rig it is not
beating "guess the majority class."

Unlike severity, this is *not* data-limited: 6× the data and three grouping schemes move it
nowhere. Any claim that supervised classification solves interference discrimination is a
**vector-rig claim only** and must not be carried over to the scalar rig.

### Stage D — ablation ladder ("software or hardware?")

| # | Date | Intervention | Software total (arm 1→5) | Hardware arm 6 recall | Verdict |
|---|---|---|---|---|---|
| 1 | 2026-08-07 | Rig-v2 ablation ladder | +0.014 [−0.028, 0.056] | 0.597 [0.430, 0.764] | No software arm significant |
| 2 | 2026-08-09 | Physics refinement | **+0.056 [0.014, 0.111]** | 0.611 [0.444, 0.764] | Software total now **significant**, traced entirely to arm 1→2 (`g1`) |

**What this trajectory says:** the refinement flipped a headline. "No software step reaches
significance" was true of entry 1 and false after entry 2 — the first-difference `g1` step is a
real, free, significant gain. The *strategic* conclusion is unchanged: hardware's +0.403 point
gap is >7× the entire software gain.

### Supporting physics — POD vs defect-moment angle to `B̂0`

| # | Date | Intervention | Raw detection-rate corr. | Severity-normalised corr. |
|---|---|---|---|---|
| 1 | 2026-08-07 | Rig-v2 | +0.162 (p=0.078) | +0.183 (p=0.045) |
| 2 | 2026-08-09 | Physics refinement | **+0.240 (p=0.0083)** | +0.203 (p=0.0265) |

The scalar rig's central physics claim — a defect whose moment is near-perpendicular to the
ambient field is nearly invisible — strengthened under the refinement, most likely because
type-specific severity reshaped the severity confound rather than because the underlying
relationship changed.

### Background contrast

| # | Date | Configuration | Contrast | Note |
|---|---|---|---|---|
| 1 | 2026-07-29 | Vector rig, single head + detrend | **3.21×** | On `r_mag_nt`, a vector magnitude |
| 2 | 2026-07-29 | Vector rig, two-head vertical gradiometer | **1.45×** | Negative result: gradiometry made it *worse* |
| 3 | 2026-07-29 | Vector rig, real USGS storm / quiet backgrounds | 3.02× / ~3.0× | Detrend survived real geomagnetic data |
| 4 | 2026-07-30 | Vector rig, 60 defects packed on **one** line | ~2.7–3.0× | Why density is scaled by adding lines, never packing |
| — | | *— rig change —* | | |
| 5 | 2026-08-10 | Scalar rig, `\|r_mid_nt\|` | **1.62×** | **Not comparable to 1–4** — different physical quantity; first scalar-rig baseline |

---

## Chronological entries

### 2026-07-30 — `n_lines` 1 → 5
**Changed:** corpus scaled by adding independent lines (12 → 60 defects). A prior attempt packing
60 defects onto one line was rejected: measured degradation of background contrast 3.19× → ~2.7–3.0×,
because background rows start sitting inside a neighbour's 1/r³ tail.
**Moved:** detection gap −0.056 → −0.028 (CI ~halved). Severity coverage 0.727 → 0.920 (**gate
went from FAIL to PASS**).
**Impact:** established that adding *independent lines* is the safe way to buy statistical power,
and gave the first evidence severity was data-limited.

### 2026-07-30 — Stage 2.75 EDA-driven feature fix
**Changed:** dropped 2 exact-duplicate features (`feature_version` 1→2); added `emphasis_repeats`
so IsolationForest's uniform random split choice stops being dominated by the ~20-column amplitude
block over the 3 features that actually separate defect from interference.
**Moved:** detection gap −0.028 → −0.006. Interference-rejection mechanism reversed and became
significant (CI [0.007, 0.058], excluding zero).
**Impact:** the gate still failed, but "EDA-informed feature engineering measurably improved
interference rejection, confirmed by a significant CI" became a real reportable result.

### 2026-07-31 — Stage 6 scale rehearsal (vector rig, 9.6M rows, 9,600 defects)
**Changed:** 800× the defects of the original corpus.
**Moved:** detection gap −0.006 → **−0.029 [−0.033, −0.026]** — the CI excluded zero for the first
time. Severity 0.896 (PASS). SCC recall 0.125 → 0.636. Interference precision 0.909 → **0.999**.
**Also found three real bugs only visible at scale:** NaN `severity_smys` poisoning conformal
coverage (0.4% incidence → 0% coverage, a cliff not a gradient); `classify`'s
`n_estimators`/`num_leaves` were dead config; small-data hyperparameters (`num_leaves: 7`)
collapsed coverage at scale.
**Impact:** converted a fuzzy negative into a definite one, and demonstrated the point of scale
rehearsals — three production bugs surfaced that 60,000 rows never could.

### 2026-08-06 — Rig-v2: rebuild around the real instrument
**Changed:** a second-round developer interview revealed the real rig is a rod with **three scalar
total-field heads** (`|B|` only, never x/y/z), carried by a walking human with GPS dropout — not a
3-axis vector head on a fixed grid. Rebuilt generator/schema (`schema_version` 2→3), added a new
registration stage (GPS dead-reckoning + girth-weld-comb detection), rewrote features around
per-head first/second differences (`feature_version` 2→3).
**Moved (rig change — not comparable to prior entries):** detection gap −0.006. Severity 0.920 →
**0.689 (PASS → FAIL)**. SCC 0.125 → 0.042. Interference precision 0.909 → 0.389.
**Impact:** the project's largest single change. Established the scalar-rig baseline everything
after is measured against.

### 2026-08-07 — Stage D ablation ladder + POD-vs-angle
**Changed:** built a 6-arm ladder isolating each software improvement extractable from the real rod,
plus arm 6 asking what a hardware upgrade to vector output would buy.
**Measured:** software total +0.014 [−0.028, 0.056] (not significant); hardware arm 0.597 vs
software's best 0.208. POD-vs-angle Spearman +0.183 (p=0.045).
**Impact:** produced the project's most decision-relevant deliverable — a measured answer to
"improve the software or upgrade the hardware?"

### 2026-08-09 — Interference/defect physics refinement
**Changed:** two corrections from further developer-team feedback. (1) Interference amplitude was a
single fixed 50× moment scale for every source; real interference runs weaker and varies
source-to-source *independently of distance* → `interference_moment_scale_range: [30, 50]`, drawn
per source. (2) All 4 defect types shared one 20–80 severity range, carrying zero distinguishing
signal; team distrusts a *shape* distinction but expects *intensity* to differ by type →
`DEFECT_SEVERITY_RANGES` (dent 40–80, corrosion 20–70, SCC 15–50, weld 15–45).
**Moved:** detection gap −0.006 → 0.000. Severity 0.689 → **0.618** (now *under* its own baseline).
SCC 0.042 → 0.023, **but** weld/dent/corrosion recall went ~0 → 0.258/0.144/0.171. Ablation software
total +0.014 → **+0.056 [0.014, 0.111], now significant**. POD-vs-angle strengthened.
**Impact:** flipped a published headline ("no software arm reaches significance" became false), and
gave the classifier a real if indirect handle on defect type via severity-correlated features — with
no shape mechanism involved.

### 2026-08-10 — Scale rehearsal under Rig-v2 (18M rows, 360 defects)
**Changed:** re-derived the scale config for Rig-v2 (the old one was sized for the 0.5 m grid and
would have generated ~4.8×10⁸ rows — ~1.4 TB RAM extrapolated). Settled on 30 independent lines ×
2 km × 3 runs ≈ 18.0M rows, 360 defects, at unchanged per-km density.
**Fixed four staleness bugs in the rehearsal script first** — only one of which failed loudly; the
other three would have produced a wrong-but-plausible report (rows/sec wrong by ~50×; a contrast
ratio taken on a *signed* residual; and hardcoded narrative asserting *"Stage 4's gate now PASSES"*
regardless of measurement). See commit `48eb01b`.
**Moved:** detection gap 0.000 → −0.007 with CIs ~2.8× tighter (**4th consistent FAIL**). Severity
0.618 → **0.872 (FAIL → PASS)**. Interference precision 0.333 → **0.271**, CI now tight. SCC
0.023 → 0.129. First scalar-rig contrast figure: 1.62×.
**Impact — the key structural finding:** this run **separates data-limited from
information-limited failures.**
- **Severity was data-limited** — 6× the data fixed it, no modelling change.
- **Detection and interference discrimination are not** — 6× the data moved neither, and they fail
  under *both* unsupervised (IF ≈ MAD, four times) *and* supervised (precision 0.271, tight CI)
  approaches.

A failure that more data doesn't touch, under both paradigms, is the signature of **missing
information rather than a modelling deficiency** — independently corroborated by the POD-vs-angle
physics result and the ablation ladder's hardware finding. Three separate lines of evidence
converge on the same conclusion.

### 2026-08-10 (cont.) — the same run's two holdout CVs, and a correction they force

The block-CV numbers above are the *optimistic* grouping: folds are keyed on `(line, 100 m block)`
hash, so blocks 1 and 3 of a line can sit in training while block 2 is tested. The same run also
ran two strictly harder generalisation tests. **They change one of the conclusions above.**

| Metric | Block CV (optimistic) | Whole-line holdout | Temporal holdout (train runs 0–1, test run 2) |
|---|---|---|---|
| Detection gap (IF − MAD) | −0.007 [−0.030, 0.017] | −0.013 [−0.037, 0.012] | −0.014 [−0.047, 0.022] |
| **Severity coverage** (gate [0.87, 0.93]) | **0.872 — PASS** | **0.845 — FAIL** | **0.860 — FAIL** |
| Interference precision (model / baseline) | 0.271 / **0.281** | 0.265 / **0.271** | 0.298 / 0.288 |
| SCC recall (gate ≥0.90) | 0.129 | 0.100 | 0.000 |

**Correction to the block-CV entry above — severity's recovery is weaker than it first appeared.**
Stage 4 passes *as the gate is configured* (block CV, 0.872). But under both harder tests it lands
just **below** the band: 0.845 and 0.860. So the honest statement is not "more data fixed severity"
— it is **"more data moved severity from badly broken (0.618, under its own baseline) to marginal
(0.845–0.872, straddling the bottom of the band depending on how honestly you split)."** The
direction is real and large; the pass is not robust to the grouping choice. Reporting the block-CV
PASS on its own would have been the flattering read.

**Detection: confirmations 5, 6 and 7.** All three splits give a negative point estimate with a CI
straddling zero (−0.007 / −0.013 / −0.014). Seven measurements, two rigs, four interventions, three
grouping schemes — IsolationForest never approaches the +0.15 margin, and the two harder splits push
the point estimate slightly *further* negative, not toward the gate.

**Interference discrimination: the decisive negative.** Across all three splits the classifier sits
at 0.265–0.298 precision — and in two of the three it is **at or below the trivial majority-class
baseline** (0.271 vs 0.281 block; 0.265 vs 0.271 whole-line). It is not merely failing to reach the
vector rig's 0.999; on the scalar rig it is not beating "guess the majority class." Six independent
measurements (3 splits × model-vs-baseline) all agree.

**Net structural finding, now tested three ways:**

| Failure | More data? | Harder split? | Diagnosis |
|---|---|---|---|
| Severity calibration | **Helps a lot** (0.618 → 0.845–0.872) | Degrades it below the band | Data-limited, not yet robust |
| Detection (IF vs MAD) | No effect | No effect | Information-limited |
| Interference discrimination | No effect | No effect | Information-limited |

A failure that survives 6× more data, *both* unsupervised and supervised paradigms, and three
grouping schemes is the signature of **missing information, not modelling deficiency** — the same
conclusion the POD-vs-angle physics result and the ablation ladder's hardware finding reach
independently.

### 2026-08-10 — process finding: the scale-rehearsal path had silently rotted

Not a measurement, but it belongs in the causal record. Getting the run above to complete required
fixing **five** separate Rig-v2 staleness bugs (commits `48eb01b`, `d7c75d6`). Only two failed
loudly:

| # | Bug | How it failed |
|---|---|---|
| 1 | Scale config sized for the old 0.5 m grid | Would have generated ~4.8×10⁸ rows (~1.4 TB RAM) |
| 2 | `expected_rows` fed into `measure(n_rows=…)` | **Silent** — every rows/sec in the report wrong ~50× |
| 3 | `_background_contrast_check` read deleted `r_mag_nt` | Crash |
| 4 | Report hardcoded *"Stage 4's gate now PASSES"* | **Silent** — a false claim printed next to contradicting numbers |
| 5 | `scale_eval` never passed through `localisation_error_cm` | Crash — **after** ~30 min of finished computation |

**Root cause, common to all five:** Rig-v2 updated the primary path (`train.py`, `generate.py`,
`features.py`) but not the secondary one (`scale_eval.py`, `scripts/stage6_scale_rehearsal.py`).
That path has **no CI coverage by design** — it is explicitly excluded as a one-time manual run
(same convention as `check_observatory_background.py`). Nothing exercised it across a major
refactor, so it rotted silently for four days.

Bugs 2 and 4 are the instructive ones: neither would have failed. Both would have produced a
confident, plausible, wrong report — exactly the failure mode this project's whole gate-and-CI
discipline exists to prevent, in the one corner where that discipline was deliberately switched off.
The lesson is not "add these to CI" (an 80-minute job doesn't belong on every push) but **"code paths
excluded from CI need an explicit staleness check at every schema/feature-version bump."**
