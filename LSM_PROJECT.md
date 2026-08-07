# Above-Ground Magnetometry → Pipeline Integrity: an ML-Lifecycle Demonstrator

**One line:** A physicist's end-to-end ML pipeline that turns 3-head scalar (total-field) magnetometer + GPS survey data — walked by a human, not driven on rails — into a prioritised heat map of likely pipeline defects, built to demonstrate methodology and MLOps discipline, not physical fidelity.

## Project Context & Modeling Strategy
Real Large Stand-Off Magnetometry data is highly proprietary. To enable this research and demonstration, a physics-inspired synthetic generator was built — implementing a magnetic-dipole forward model, sensed by three total-field scalar heads on a vertical rod, over a realistic drifting geomagnetic background. **Rig-v2** (2026-08-06) rebuilt this generator around the real instrument, after a second-round interview with the LSM system's own developer (Richard Föcke) revealed it: a human walks the line at irregular speed and stand-off, GPS drops out in poor sky view, and each head reports only `|B|`, never x/y/z — replacing the original single 3-axis-vector-magnetometer-on-a-rail model. The project implements the full ML lifecycle on this dataset: raw data ingestion and validation, along-track **registration** (GPS dead-reckoning + girth-weld-comb detection — a new stage, since raw data no longer carries a usable chainage column), feature engineering (background detrending, per-head first/second differences, and stand-off inversion), uncertainty-calibrated modeling, anomaly tracking, and risk heat map serving.

## What it demonstrates (six scenarios, on one dataset)
1. **Anomaly detection** — find stress-concentration zones with mostly unsupervised methods after background removal.
2. **Severity regression** — map the magnetic signature to a `%SMYS`-style value *with calibrated uncertainty*.
3. **Defect classification** — SCC vs weld vs dent vs corrosion vs benign interference. **Demoted to a synthetic-only capability demonstration**, not a claim this generalises to real ROSEN data: the developer interview confirmed there is **no labelled defect-type data** on real surveys, so a supervised 5-class model cannot be trained on real ROSEN data at all. Kept because the calibration, per-class-recall discipline and SHAP physics-consistency gate around it are good MLOps worth demonstrating — not because the model itself is fit to deploy. It was already failing its own SCC-recall promotion gate before this rework (0.125, CI [0.000, 0.264]) — stated plainly because it costs nothing and buys credibility.
4. **Risk / prioritisation** — combine severity, class and (proxy) consequence into a ranked heat map.
5. **Growth forecasting** — three repeat surveys with growing defects → project remaining life.
6. **Dig-feedback active learning** — with no real defect-type labels, *which segment should we dig next, and what does that dig teach the model* is the actual path to ever training a supervised model on real ROSEN data. `src/lsm/dig_feedback.py` (`record_excavation`, `recompute_coverage_from_verifications`) already exists and is treated here as a headline stage, not a stretch-goal footnote — arguably the single most directly useful thing a data scientist could hand ROSEN, since it is the mechanism by which labels would ever start to exist.

## The data
`python -m lsm generate` (`src/lsm/generate.py`) writes one immutable Parquet file per survey. A 2 km line, **walked** (not driven on rails) by a human at irregular speed and stand-off, sampled at a fixed 120 Hz — irregular in distance, dense and monotonic in `sample_idx` — three repeat surveys per line, 12 growing defects plus 4 off-pipe interference sources (deliberate false-positive traps), plus a periodic girth-weld train (~every 12.2 m, a fact of how the pipe was built, not a rare event). Per-survey row count is therefore far denser than the old uniform-0.5 m grid — `config/base.yaml`'s own comment on `walk.sample_rate_hz` states ~50x — a direct consequence of the real rig's sample rate, not a demo convenience; exact post-Rig-v2 row counts are Stage D's to measure, not estimated here.

| column | meaning |
|---|---|
| `sample_idx` | dense, monotonic integer key — now a **time**-sample counter, not a distance-grid index |
| `t_s` | elapsed seconds since survey start — the walk is time-sampled, not distance-sampled |
| `chainage_true_m` | the generator's own exact along-track position — **truth tier**, may score registration, never a feature |
| `lat`, `lon` | GPS fixes, **nullable float64** — NaN during GPS dropout (poor sky view), not synthesised through |
| `b_lo_nt`, `b_mid_nt`, `b_hi_nt` | total-field **magnitude** `\|B\|` from each of 3 heads (−0.5 m / 0 / +0.5 m on the rod) — never x/y/z; large background + drift + defect dipoles + noise |
| `defect` | 1 within the label window of a true defect, else 0 |
| `defect_type` | scc / weld / dent / corrosion / none — see the classification demotion note above |
| `severity_smys` | proxy severity (~20–95), grows across surveys |
| `interference` | 1 for an off-pipe false-positive-trap source |
| `girth_weld` | 1 for the periodic joint train — a real, strong, non-cancelling source, but not damage |

`chainage_m` — distance along the pipe, used everywhere downstream — is **no longer a raw column**. Under an irregular human walk there is no physically-final chainage until Stage B's registration (`src/lsm/registration.py`) reconstructs it from GPS dead-reckoning plus girth-weld-comb detection; it is written to the feature layer, never to raw.

Signal-to-check: after the two-stage detrend, defect residuals are still visible against the background, but the exact contrast is being re-measured for the scalar rig (Stage D, in progress — see "Rig-v2 measured results" near the end of this document for the gate numbers measured so far, and note the 6-arm ablation ladder that would give a direct contrast comparison has not yet run). The pre-Rig-v2 vector-rig numbers (25.05 nT residual, 3.02–3.21× contrast) are a historical baseline, not current. Background removal is still the whole point of the project, and interference still makes naive thresholding fail.

**Optional realism:** swap the synthetic background for a real geomagnetic trace (BGS/NOAA observatory) so you can say the ambient field is real. Nice-to-have, not required.

## Repo structure
```
lsm-integrity-demo/
├── data/                    # generated CSVs (gitignored)
├── src/
│   ├── generate_lsm_data.py # the generator: 3-head scalar rig, human walker, GPS dropout, weld train (done)
│   ├── validate.py          # schema + range + GPS-dropout-aware + weld/saturation checks
│   ├── registration.py      # NEW: GPS dead-reckoning + girth-weld-comb detection -> registered chainage_m
│   ├── features.py          # background removal per head, first/second head-differences (g1/g2), stand-off inversion, window stats
│   ├── train.py             # anomaly + regression + classification, MLflow-tracked
│   └── forecast.py          # per-defect growth → remaining life
├── app/streamlit_app.py     # map + anomaly overlay + risk heat map
├── config.yaml              # one config for the whole run
├── mlruns/                  # MLflow tracking
├── requirements.txt
└── README.md                # this framing, condensed
```

## Lifecycle steps (where the effort goes)
**1 · Validation** — schema/dtypes, physical range checks on each axis, chainage/GPS continuity and monotonicity, per-axis noise sanity, and an interference flag. Log a short data-quality report. *This is what signals real experience — do it first and visibly.*

**2 · Features** — remove the background (two-stage robust-polynomial + rolling-median detrend, per head), compute the **first difference** across the 3 heads (`g1`, common-mode background rejection) and the **second difference** (`g2`, the third head's specific value: cancels a *linear* background gradient that `g1` cannot), invert the head-to-head amplitude ratio for a genuine per-row **measured stand-off** (`standoff_est_m`), sliding-window mean/std/peak, and normalise amplitude by that measured stand-off rather than an assumed constant depth. 45 columns total (`src/lsm/features.py::feature_columns()`). Keep a clear feature table.

**3 · Modelling** — split **grouped by chainage segment** so neighbours never leak across train/test.
- Unsupervised: robust-MAD threshold + IsolationForest on residual features (baseline).
- Supervised regression: gradient boosting with quantiles or NGBoost/conformal → severity + interval.
- Classification: gradient boosting or a 1-D CNN on the signal window; explicit `interference` class; report per-class recall (protect SCC).

**4 · MLOps** — MLflow for params/metrics/artefacts, one `config.yaml`, fixed seed, `make run` reproducibility, a couple of pytest checks on the validator and feature functions.

**5 · Serve** — Streamlit: plot the line, overlay detected anomalies on the GPS track, show a ranked risk heat map and a per-defect growth curve. This is the money shot in a portfolio.

## Suggested priority (if time is short)
Do the thin end-to-end slice first: **validate → detrend + gradient features → IsolationForest → Streamlit map with anomalies**. That alone tells the whole story. Then add the regression + uncertainty, then classification, then growth forecasting as time allows. A tracked, reproducible, deployed thin slice beats a half-finished deep model.

## Tech stack
Python · numpy/pandas · scikit-learn · LightGBM · MLflow · Streamlit · pytest.

## Key Engineering & Physics Insights
- **Preprocessing Focus:** The majority of LSM data analysis complexity lies in data validation and ambient background removal, rather than standard model optimization.
- **Sensor Configuration & Gradiometry (Rig-v2, corrected after a developer interview):** the real rig carries **three scalar total-field magnetometers** (middle + two, 50 cm apart) — each head outputs only `|B|`, never x/y/z, so what is actually measured is the *total-field anomaly*, `|B0+dB| − |B0| ≈ dB·B̂0`: only the projection of a defect's field onto the ambient field direction is visible. Three heads give both a **first difference** (`g1`, common-mode background rejection — the old two-head story) and a **second difference** (`g2 = b_hi + b_lo − 2·b_mid`, which additionally cancels a *linear* background gradient — the specific extra value the third head buys over a two-head gradiometer). The head-to-head amplitude ratio also inverts via 1/r³ for a genuine per-row measured stand-off. `|B|` is rotation-invariant, so rod sway/tilt moves head positions but never corrupts a reading by itself — almost certainly why the real instrument reports scalar in the first place. Whether/how much this configuration outperforms the pre-Rig-v2 vector-head model (which measured 2.3× vs 4.5× contrast for a two-head gradiometer vs single-head detrending) is exactly what Stage D's 6-arm ablation ladder is re-measuring; that older comparison is a historical baseline now, not the current default model's result — see "Rig-v2 measured results" below.
- **Interference Separation:** Incorporating synthetic external magnetic interference sources is necessary to evaluate the model's ability to distinguish true localized stress defects from benign external objects.
- **Uncertainty Quantification:** Point predictions are insufficient for pipeline safety assessments. Predictive outputs must include calibrated uncertainty intervals to assess structural excavation risk.
- **Temporal Analysis:** Fusing repeat surveys allows modeling of defect growth over time, enabling remaining life projections.

## Stretch (only if you have spare time)
Real observatory background; a tiny ILI-vs-LSM "data fusion" mock; conformal prediction for the regression.

(The dig-feedback active-learning loop — "which segment should we dig next, and what does that dig teach the model" — used to be listed here. It has been promoted to a headline stage, scenario 6 above: with no labelled real defect-type data, it is the actual path to ever training a supervised model on real ROSEN data, not a nice-to-have.)

## Insights & Additions from Literature (NeurIPS 2020)
Based on *Mitra et al.*'s workshop paper on ML-based LSM anomaly detection (real experimental
data, two physical defects on a lab rig) — the closest peer to this project in spirit, and
independently re-read in full (not just summarized) to check this comparison honestly rather
than assume it. They are ahead of us on breadth of technique (a working multi-output 1D-CNN,
FastDTW alignment, RAPIDS cuML GPU acceleration); we are ahead of them on statistical rigor
(bootstrap CIs on every headline metric vs. their bare accuracy/MCC with no interval at all,
and a documented, *re-verified* negative-result investigation vs. their "found a leakage bug,
fixed it, moved on"). Below is what we considered from their approach and the actual reasoning
for building vs. deferring each one — not a to-do list, since three of the five were
deliberately not built, for reasons specific to this project, not lack of time.

### Project vs. research paper: what we considered, and why we did or didn't build it

| Category | Their approach | Our approach | Built? Why / why not |
| :--- | :--- | :--- | :--- |
| **Data alignment** | FastDTW (time-series alignment) | Detrending + along-track gradients; no alignment step | **Not built, deliberately.** Our synthetic generator keeps every defect's chainage position exactly fixed across a line's repeat runs — there is no phase/chainage misalignment in this data for DTW to correct. Real LSM data (odometer slip, GPS drift) would need it; building DTW here would be solving a problem the demo doesn't have. |
| **Unsupervised baseline** | k-NN (`pyod`) — their best-performing unsupervised model | MAD threshold + IsolationForest | **Partially open.** A k-NN baseline alongside MAD/IsolationForest is a real, cheap candidate addition, not yet done — unlike the other four rows, this one has no principled reason to skip, just hasn't been prioritized yet. |
| **Supervised modelling** | Unified multi-output 1D-CNN (mask + depth + volume, Keras) | Separate LightGBM models per stage (anomaly, severity) | **Not built, deliberately.** LightGBM was chosen because Stage 6's scale target favours a fast, interpretable tabular model with native SHAP support, at a data volume (tens of thousands of rows) where a CNN's main advantage — learning spatial features from raw windows — is less clear-cut than on their scale. A 1D-CNN remains an explicit "stretch" comparison, not a gap papered over. |
| **Hardware / scaling** | RAPIDS cuML for GPU-accelerated SVC (their SVC scales as *O(N³)*) | CPU-bound LightGBM/scikit-learn (`n_jobs=16`) | **Not applicable, not just "not built."** We never use an SVC anywhere in this pipeline, so their specific *O(N³)* problem doesn't exist here. LightGBM/IsolationForest are CPU-native and don't benefit meaningfully from GPU at this row count; GPU only becomes relevant if a future 1D-CNN stage is actually started (and then needs a Blackwell/CUDA 12.8+-matched PyTorch build first — the installed one today is CPU-only). |
| **Leakage-safe splitting** | Hold out entire contiguous defect regions with a buffer | `GroupKFold` by whole `line_id` (a defect's entire physical line goes to one fold) | **Already handled, arguably more conservatively.** Grouping by whole line is a strictly larger exclusion zone than a buffer around each defect — a real, CI-enforced test (`tests/test_leakage.py`) already pins that no group crosses folds. This is a case where our approach and theirs converge on the same guarantee via a different mechanism, not a gap. |
| **Data augmentation** | Magnetostriction-informed physical augmentation | Full-control synthetic forward generator (no augmentation step) | **Not applicable.** Augmentation exists to stretch scarce *real* data further; since we already control ground truth completely and can generate more scenarios directly, "augmenting" a synthetic dataset via magnetostriction modelling would add complexity without adding real signal. This idea earns its place once real proprietary data is in play, not before.

---

## Microstructural & LSM Profiles of Pipeline Defects
As a materials scientist, you can map the microstructural changes to their magnetic footprint via the **Villari effect** (magnetostriction). This connection mirrors the legacy of **Italian metallurgical craftsmanship**—from the Renaissance armorers of Brescia and Milan who mastered grain boundary refinement and heat treatments to control residual stress, to modern Italian pipe manufacturing giants like *TenarisDalmine* who refine microstructures to optimize mechanical and magnetic properties in seamless pipeline steels.

### 1. Stress Corrosion Cracking (SCC)
*   **Microstructure (SEM/TEM):** 
    *   *SEM:* Reveals highly branched crack colonies. These can be **intergranular** (following grain boundaries) or **transgranular** (cleaving through grain bodies).
    *   *TEM:* Shows dislocation pile-ups at boundaries and localized electrochemical dissolution zones at the crack tip.
*   **LSM Signature:** Creates a very sharp, localized anomaly. The high stress concentration at crack tips acts as a localized barrier to magnetic flux lines, forcing them out of the pipe wall (high-frequency dipole).

### 2. Welds (Girth and Seam)
*   **Microstructure (SEM/TEM):**
    *   *SEM:* Displays microstructural gradients across the Heat-Affected Zone (HAZ). Transitions from coarse-grained bainite/martensite near the fusion line to fine acicular ferrite (which increases toughness, similar to the differential tempering used by historic Italian sword-makers to blend a hard edge with a flexible core).
    *   *TEM:* Reveals variations in precipitate distribution (carbides/nitrides) and micro-segregation.
*   **LSM Signature:** Presents as a wide, stable anomaly. The transition in permeability between parent metal, HAZ, and weld metal creates a permanent, predictable magnetic signature.

### 3. Dents (Mechanical Damage)
*   **Microstructure (SEM/TEM):**
    *   *SEM:* Shows severe plastic deformation; grains are heavily flattened, containing slip bands and twins.
    *   *TEM:* Characterized by extremely high density of entangled dislocation networks and sub-grain boundaries due to cold working.
*   **LSM Signature:** Produces a high-amplitude magnetic dipole. The physical deflection changes the sensor standoff, while the residual stress warps local magnetic permeability.

### 4. Corrosion (Metal Loss)
*   **Microstructure (SEM/TEM):**
    *   *SEM:* Characterized by pitting cavities, scalloped surfaces, and oxide scale layers (magnetite/hematite).
    *   *TEM:* Focuses on oxide-metal interfaces and lattice mismatches.
*   **LSM Signature:** Broad, lower-amplitude anomaly. Reducing wall thickness restricts the flux path, causing magnetic flux lines to leak outside the pipe.

### 5. Benign Interference (Metallic debris / parallel utilities)
*   **Microstructure (SEM/TEM):** Unrelated to the pipeline's microstructure.
*   **LSM Signature:** Lacks the sharp stress-related dipole characteristics of localized defects, presenting as long-wavelength drifts or isolated external dipoles.

---

## Comprehensive Literature Review & Project Alignment

An overview of the technical papers and reference documents found in the folder, read in full
(not just skimmed for keywords) so the comparisons below are checked against what each
document actually contains, not assumed from its title. Two corrections from that re-read,
stated plainly rather than quietly fixed: item 2's paper contains no dataset, experiment, or
results table anywhere — treat it as a proposal, not an empirical benchmark. Item 5 below is
**my own interview-prep synthesis** (built from the flyer and general domain knowledge for the
2nd-round interview with Richard Föcke), not a document ROSEN published — the earlier title
"ROSEN Reference Standards & Best Practices" implied an external authority this doesn't have,
and has been corrected.

### 1. npj Materials Degradation Review (Olawole et al., 2026)
*   **Title:** *Advanced sensor systems and machine learning for pipeline integrity management: a review of corrosion monitoring and prediction strategies*
*   **Focus:** A comprehensive evaluation of In-Line Inspection (ILI) techniques (MFL, UT, EMAT) vs. long-range continuous monitoring (GWUT, AET, DFOS) and the machine learning architectures used to interpret their signals. **This is a review, not primary research** — it cites other groups' numbers (e.g. a cited GAN's NCC 0.82) rather than running its own experiments, so it's a source of vocabulary and of what the field considers emerging (multimodal fusion, PINNs, digital twins), not a rigor benchmark this project should be measured against directly.
*   **Key Findings:**
    *   **The "ML Hook":** Sensors create either *volume bottlenecks* (e.g., DFOS producing 30 million points/sec, needing compression) or *complexity bottlenecks* (e.g., MFL and GWUT, where signals are indirect, non-linear, and obscured by operational noise).
    *   **Multimodal Data Fusion:** Combining MFL (magnetic) and UT (acoustic) compensates for individual sensor blind spots (e.g., MFL is blind to mid-wall laminations, while UT fails in gas pockets). Fusing them increases Probability of Identification (POI).
    *   **Advanced ML Paradigms:**
        *   *GAN Translation:* Using Generative Adversarial Networks (e.g., Res-Pix2Pix) to translate raw 2D MFL maps into pseudo-UT thickness profiles to ease fusion.
        *   *FEA-ANN Prognosis:* Training an Artificial Neural Network on finite element simulations of parameterized defects (varying length, width, and depth) to predict pipeline failure pressure instantly (relative error 0%–7%), avoiding slow real-time FEA.
        *   *Trust & PINNs:* Emphasizes the need for Explainable AI (SHAP/LIME) and Physics-Informed Neural Networks (PINNs) that constrain the loss function using physical laws (Maxwell's equations and Ramberg-Osgood stress-strain relations).
*   **Project alignment, honestly stated:** none of GAN fusion, FEA-ANN, or PINNs are built here — this project is single-modality (magnetometry only), and multimodal fusion isn't reachable without a second sensor's real data. What genuinely carries over is the framing: this project's synthetic-forward-model-plus-feature-engineering approach is a "sim-to-real" story in the same spirit this review describes as emerging practice, not evidence we've implemented what the review surveys.

### 2. Subsea Pipeline Integrity Model (Wegner et al., 2021)
*   **Title:** *A Machine Learning-Enhanced Model for Predicting Pipeline Integrity in Offshore Oil and Gas Fields*
*   **Focus:** Proposes fusing heterogeneous operational registries to predict structural degradation in subsea flowlines. **Read in full and worth being direct about: this paper contains no dataset, no experiment, and no results table or figure with a number in it anywhere — the "Modeling" and "Explainability" content is prose describing what a model would do, not results from one that was built.** It reads as a proposal/literature synthesis, not a completed empirical study.
*   **What it describes (proposed, not demonstrated):**
    *   **Data Fusion:** Integrating tabular data from ROV visual inspection reports, Cathodic Protection (CP) electrical surveys, and maintenance registries.
    *   **Modeling:** Supervised learning (SVMs, Random Forests) to forecast armor loss, coating damage, and corrosion fatigue.
    *   **Explainability:** SHAP value analysis for risk-driver interpretation.
*   **Project alignment, corrected:** this project has real bootstrap CIs, a real leakage test, and real (including negative) measured results — rigor this paper doesn't demonstrate having, whatever techniques it names. SHAP/feature-importance analysis is now built (Stage 5, 2026-07-30): LightGBM's native `pred_contrib=True` TreeSHAP values, not the external `shap` package (its `numba` dependency doesn't support the installed numpy), feeding a physics-consistency gate proven with an engineered-leak integration test, not just an absent-name check.

### 3. Pipeline Defect Detection using SVM (Isa, Rajkumar, & Woo, 2007)
*   **Title:** *Pipeline Defect Detection Using Support Vector Machines*
*   **Focus:** Continuous monitoring of pipe wall thinning using guided ultrasonic wave propagation on a lab-scale rig, one defect.
*   **Key Findings:**
    *   **Signal Processing:** Discrete Wavelet Transform (DWT) (Haar vs. Daubechies DB2) to compress raw 1D acoustic signals and filter high-frequency noise.
    *   **Classification:** DWT coefficients fed to an SVM (LIBSVM); RBF kernel performed best, **89.65%** classification accuracy, Haar wavelet, frame window 25. No cross-validation, no confidence interval, no baseline comparison, and a single lab defect — a reasonable 2007-vintage proof of concept, not a rigor bar.
*   **Project alignment, corrected:** this project does **not** use an SVM anywhere — Stage 5 is built as LightGBM multiclass, not RBF-SVC. A previous version of this section claimed this paper "supports the choice of RBF SVC in our classification benchmarks," which was wrong on two counts: no SVC exists in this project, and this 18-year-old single-defect result isn't grounds to choose one over LightGBM if it did. What legitimately carries over is the general idea that window-based features work for localized classification — the classifier choice doesn't.

### 4. ROSEN LSM Technical Flyer (Product Specifications)
*   **Focus:** Commercial capabilities of ROSEN's above-ground Large Stand-Off Magnetometry (LSM) system for unpiggable pipelines.
*   **Key Specifications:**
    *   *Pipeline Diameter:* 152–1820 mm (6"–72").
    *   *Optimal Standoff:* Up to 12 times the pipe diameter (e.g., ~3 m or 9.8 ft for a 10" pipeline).
    *   *Accuracy:* Lateral accuracy within 100 mm (0.33 ft); mapping accuracy ±5% of actual position.
    *   *Performance:* Probability of Detection (POD) >80% at a 95% confidence level, **explicitly stated "in the absence of magnetic interference."**
    *   *Output:* Identifies Stress Concentration Zones (SCZs) and reports stress magnitude in MPa or %SMYS.
*   **Project alignment, precisely stated (not simplified):** these specs define the physical bounds the synthetic generator was built to respect (standoff distances, lateral resolution, noise regime). On the *measured* comparison, two numbers now exist and must not be conflated: the **pre-Rig-v2** model measured recall-at-dig-budget at ~64% (both MAD and IsolationForest, at Stage 6 scale — see `docs/interview-reference.md` §10), while the **Rig-v2** scalar rig, re-measured by Stage D at demo scale (60 defects), measures **recall@budget 0.144 (MAD) / 0.139 (IsolationForest)** — see `LSM_PROJECT.md`'s own "Rig-v2 measured results" section above. Both sit below the flyer's 80% figure — the comparison is **not apples-to-apples** either way, because this project's evaluation corpus *always* includes interference by design (it's the deliberate false-positive trap the whole project is built to stress-test), while the flyer's 80% figure explicitly excludes interference. The Rig-v2 number is also not apples-to-apples against the pre-Rig-v2 64%: less information per scalar sample, a harder walked/GPS-dropout acquisition, and — not yet ruled out — Stage D has not run the 800×-scale rehearsal that would show whether 0.144 is itself depressed by small-sample variance the way early pre-Rig-v2 measurements were. The honest statement is "both numbers are under a harder condition their own spec doesn't cover, and the two project numbers are not yet directly comparable to each other," not a direct read of either against the flyer's 80%.

### 5. My own interview-prep synthesis (not a ROSEN document)
Built for the 2nd-round interview (5 Aug 2026, Richard Föcke, Head of NDT Apps & Products) from
the flyer plus general domain reasoning — notes to prepare with, not something ROSEN
published. Its own 7-stage framing of the pipeline (**1** ingest & integrate → **2** register/align
→ **3** signal processing & features → **4** label from digs → **5** model → **6** calibrate +
uncertainty → **7** serve heat map) maps closely onto this project's actual stage structure. One
gap this synthesis originally named honestly (below) has since been **resolved** by the same
developer interview that motivated the Rig-v2 rework — worth stating plainly, since it's a real,
good update, not something to downplay:
*   **Stage 2, "register/align" — RESOLVED, not a gap anymore.** This was true when written: the
    old synthetic generator kept every defect's chainage position exactly fixed across a line's
    repeat runs, so there was no misalignment for a registration step to fix. Rig-v2 (2026-08-06)
    changed the underlying physics: the generator now models a human walker at irregular speed
    with GPS that drops out, so raw data no longer carries a usable chainage column at all (see
    "Project Context" above). That forced exactly the registration stage this synthesis flagged
    as missing: `src/lsm/registration.py` (Stage B) reconstructs along-track position via GPS
    dead-reckoning through dropout, then locks it to the recovered girth-weld lattice (welds
    ~12 m apart, detected by periodicity rather than amplitude template, since each joint's extra
    steel has arbitrary orientation) — the same registration concept this synthesis's own
    7-stage framing named as stage 2. The gap closed because the physics forced the issue, not
    because a feature was retrofitted for its own sake.
*   **Its "recall/POD, PR-AUC, sizing error, calibrated uncertainty, cost-sensitive evaluation"
    metrics guidance is genuinely what this project already does** (Stage 3's recall-at-budget +
    false-dig-rate + PR-AUC-as-diagnostic, Stage 4's split-conformal calibration, both reported
    with bootstrap CIs, never bare accuracy).
*   **Forecasting (its stage-6-adjacent mention of corrosion growth/remaining life)** maps to
    this project's **Stage 8, a CLI stub, not built** — and forecasting is separately named as
    required experience in the actual job posting, which makes this the more load-bearing gap
    of the two, not just a nice-to-have.

---

## Rig-v2 measured results (Stage D, in progress)

Rig-v2 (2026-08-06) rebuilt the generator, registration and feature layers around the real
instrument (see "Project Context" above). Stage D re-ran the production training pipeline
against the rebuilt scalar-rig corpus and re-measured every existing gate. The numbers below
are real CLI output, captured directly — not estimates, not projected from the pre-Rig-v2
model.

**Do not read the historical numbers in `docs/interview-reference.md` (25.05 nT residual,
3.02–3.21× contrast, the pre-Rig-v2 IsolationForest/MAD recall table, etc.) as current.** They
were measured on the pre-Rig-v2 vector-head model and are kept there, explicitly labelled, as a
historical/comparison baseline — specifically, the value Stage D's ablation arm 6 ("what would
a hardware upgrade to full vector output buy") compares against.

### Gate results (measured, Stage D re-run against the Rig-v2 scalar corpus)

| Stage | Model | Baseline | Result | Gate | Verdict |
|---|---|---|---|---|---|
| 3 · Detect | IsolationForest | MAD | recall gap **−0.006** [−0.061, 0.056] | ≥0.15 at CI lower bound | **FAIL** |
| 4 · Severity | LightGBM CQR | global-mean | coverage **0.689** [0.420, 0.945] | in [0.87, 0.93] | **FAIL** |
| 5 · Classify | LightGBM multiclass | majority-class | SCC recall **0.042** [0.000, 0.125] | ≥0.90 at CI lower bound | **FAIL** |
| 8 · Growth | pooled log-linear | no-growth | log-growth-rate **0.1398** (n=14 defects) | beats no-growth baseline | **PASS** |

Raw output:

```
=== Stage 3: MAD baseline vs IsolationForest (grouped CV, out-of-fold) ===
MAD baseline:            recall@budget 0.144 [0.067,0.233]  false-dig 0.827 [0.773,0.874] (interference 0.067 [0.027,0.113])  localisation 6.359m [4.358,8.228]  PR-AUC 0.043
IsolationForest:          recall@budget 0.139 [0.067,0.222]  false-dig 0.833 [0.767,0.893] (interference 0.053 [0.020,0.080])  localisation 5.644m [4.013,7.484]  PR-AUC 0.037
Recall gap (IF - MAD): -0.006 [-0.061, 0.056] -- Stage 3 gate (>=0.15 at CI lower bound): DOES NOT PASS
(60 physical defects, 15 surveys evaluated)

=== Stage 4: severity -- LightGBM CQR vs global-mean baseline ===
Global-mean:  coverage@90% 0.589 [0.313,0.865]  MAE 29.087 [19.006,38.649]
LightGBM CQR: coverage@90% 0.689 [0.420,0.945]  MAE 28.952 [20.548,38.243]
Stage 4 gate (coverage in [0.87,0.93]): DOES NOT PASS
(n=293 matched, severity-labelled indications)

=== Stage 5: classification -- LightGBM multiclass vs majority-class baseline ===
Majority baseline: interference recall 1.000, everything else 0.000
LightGBM: SCC recall 0.042 [0.000,0.125]
Stage 5 gate: DOES NOT PASS
(n=376 matched, classifiable indications)

=== Stage 8 (forecast): growth ===
population log-growth-rate: 0.1398 (n=14 defects) -- matches ln(1.15)=0.1398 exactly
gate (beats no-growth baseline): PASSED
```

### What these numbers say, honestly

- **Stage 3 (detection) is numerically almost unchanged.** The IsolationForest-vs-MAD recall
  gap is **−0.006 [−0.061, 0.056]** under Rig-v2 — the same value, to three decimal places, as
  the pre-Rig-v2 measurement at the same 60-defect scale (also −0.006, see
  `docs/interview-reference.md` §10 / §11). Stated plainly, as "unchanged," not as a causal
  claim: the CI still straddles zero at this sample size, so this could be a small-sample
  coincidence, or it could reflect something about MAD-vs-IsolationForest on this feature set
  that doesn't actually depend on scalar vs. vector sensing. Telling those apart would need the
  same kind of 800×-scale rehearsal that resolved the equivalent pre-Rig-v2 question (Stage 6:
  −0.006 at 60 defects → −0.029 [−0.033, −0.026] at 9,600), which Stage D has not yet run.
  Either way the gate verdict is unchanged: **FAIL**.
- **Stage 4 (severity conformal coverage) genuinely regressed**, and this is reported as a real
  finding, not softened: **0.920 [0.867, 0.967] PASS** pre-Rig-v2, at a comparable ~137-matched-
  indication scale (`docs/interview-reference.md` §10/§14), versus **0.689 [0.420, 0.945] FAIL**
  under Rig-v2 at n=293. A harder, GPS-dropout, irregular-stand-off scalar rig is producing
  noisier severity calibration at this sample size. Whether more calibration data closes this
  the way it closed the equivalent pre-Rig-v2 gap (0.727 → 0.920 → 0.896 as the corpus scaled
  up) is an open question Stage D has not yet answered — flagged here rather than assumed.
- **Stage 5 (SCC classification)** was already failing its gate pre-Rig-v2 (0.125 [0.000,
  0.264], demo scale) and is still failing, slightly worse, under Rig-v2 (0.042 [0.000, 0.125]).
  Consistent with the classification-demotion decision above: neither rig's synthetic
  defect-type assignment carries a real physical signal for a classifier to separate on, so an
  unchanged failing verdict is the expected result, not a new problem introduced by Rig-v2.
- **Stage 8 (growth) still passes**, recovering `ln(1.15) = 0.1398` almost exactly — this is an
  estimator-correctness check on a deterministic growth law, not a claim about real defect
  physics, and it is unaffected by which rig generated the underlying residuals. An unchanged
  pass here is expected, not newsworthy.

### Ablation ladder — measured

The 6-arm ablation ladder (middle head only → +first difference `g1` → +second difference `g2`
→ +stand-off inversion → +weld-comb registration → full vector output as arm 6, the "what would
a hardware upgrade buy" comparison) has now run end to end (`scripts/ablation_ladder.py`), on a
real, non-toy corpus: 2 lines at the production per-line density (2000 m, 12 defects, 4
interference, 3 runs — a deliberate scale-down from the full 5-line default purely for runtime,
documented in the script's own module docstring; not yet re-run at 5 lines).

| Arm | recall @ dig budget | false-dig rate | localisation error (cm) |
|---|---|---|---|
| 1 · mid-head only, GPS chainage | 0.194 [0.069, 0.333] | 0.767 [0.717, 0.833] | 617 [379, 859] |
| 2 · + first difference `g1` | 0.208 [0.083, 0.375] | 0.750 [0.717, 0.783] | 425 [244, 632] |
| 3 · + second difference `g2` | 0.194 [0.069, 0.361] | 0.767 [0.733, 0.800] | 306 [158, 490] |
| 4 · + stand-off inversion/normalisation | 0.222 [0.083, 0.389] | 0.733 [0.700, 0.767] | 439 [235, 670] |
| 5 · + weld-comb registration | 0.208 [0.069, 0.361] | 0.750 [0.717, 0.783] | 536 [321, 745] |
| 6 · full 3-axis vector output (hardware) | **0.597 [0.430, 0.764]** | **0.272 [0.233, 0.311]** | **48 [40, 58]** |

Arm-to-arm recall deltas (paired bootstrap over the SAME defect set for arms 1-5; arm 5→6 is
**not** a paired comparison — `rig: vector` is a structurally different corpus/generator, so only
CI overlap is meaningful there, not a paired delta):

```
1->2                         0.014 [-0.042, 0.083]
2->3                         -0.014 [-0.056, 0.028]
3->4                         0.028 [-0.028, 0.097]
4->5                         -0.014 [-0.042, 0.000]
1->5 (software total)        0.014 [-0.028, 0.056]
5->6 (hardware headline)     software 0.208 [0.069, 0.361] vs hardware 0.597 [0.430, 0.764]
                              -- independent CIs, point gap +0.389
```

**The honest, unflattering answer: none of arms 2-5 move recall by a statistically distinguishable
amount over arm 1's floor.** Every arm-to-arm software delta's CI straddles zero, including the
cumulative "1→5" software total (+0.014 [-0.028, 0.056]). This is the plan's own explicitly
anticipated possible outcome ("If they do not [close most of the gap], that is a finding worth
delivering too — do not tune it toward the convenient answer") — and it is what was measured, not
what would make the best story. Arm 6 (full vector output, i.e. a genuine hardware upgrade) is
where the real movement is: recall roughly triples (point estimate 0.597 vs 0.208), false-dig rate
drops by nearly 3x (0.272 vs 0.750), and localisation error drops by roughly 10x (48 cm vs 536 cm)
— consistent with the underlying physics (a vector head recovers the anomaly's DIRECTION, which a
scalar total-field magnitude structurally cannot, per generate.py's own module docstring). Two
honest caveats on that comparison: (1) it is not a paired delta (independent corpora, so read it
as "the two CIs barely overlap on recall and don't overlap at all on false-dig/localisation," not
as a formal hypothesis test), and (2) `rig: vector`'s reference arm has none of the scalar rig's
walk/GPS-dropout/sensor-calibration-noise physics, so part of its localisation advantage in
particular reflects a simpler, more idealised world, not purely "vector vs scalar sensing" in
isolation — reported plainly rather than adjusted for, since disentangling the two would need a
walked, GPS-dropout **vector**-rig generator that does not currently exist.

**What arms 2-5 individually show, read across the point estimates (none reach significance, but
the direction is at least worth stating and not over-claiming beyond):** `g1`/`g2` and stand-off
normalisation each nudge the point estimate up or sideways by ~0.01-0.03 recall, and weld-comb
registration (arm 5) does NOT improve recall over arm 4 at all (point estimate goes slightly
*down*, -0.014) — registration's real, measured value is in localisation (see below), not
detection recall, which matches its own design intent (Stage B answers "where," not "was there a
defect there at all").

### Localisation, in the unit that actually matters: centimetres

`localisation_errors_cm` (new in `evaluate.py`/`train.py`) reports the SAME matched-indication
distance the metre-unit number already used, just in the unit the ~1 cm dig-marking requirement
is actually stated in. On the full production corpus (5 lines, the real gate re-run, not the
ablation ladder's 2-line scale-down):

```
MAD baseline:      localisation error (cm)   635.9 [435.8, 822.8]
IsolationForest:    localisation error (cm)   564.4 [401.3, 748.4]
```

**Not close to 1 cm.** Three separate, honestly-reported measurements chain together to explain
why:
- **Naive GPS-only dead reckoning** (no registration at all, `chainage_provisional_m` vs
  `chainage_true_m`, measured directly against every production raw survey): per-survey median
  error **~11.3 m** [8.7, 14.1].
- **Registered chainage** (`registration.register_survey`, the real weld-comb-locked output):
  per-survey median error **~1.40 m** [1.08, 1.69] — a genuine ~8x improvement over naive GPS,
  consistent with (and slightly better than) `tests/test_registration.py`'s own demonstrated
  2.7x-12.7x range on a shorter 500 m test survey (0.57-2.03 m median across seeds). Registration
  is real and it works; it just does not reach cm-level at this survey length/weld density.
- **Indication-level localisation error** (564-636 cm) is **larger than the pure registration
  residual** (~140 cm) — meaning the dominant remaining error is downstream of registration:
  which row a detector's peak lands on, and `MATCH_TOLERANCE_M`'s generous 15 m credit radius
  (revisited and kept, see `train.py`'s own comment on `MATCH_TOLERANCE_M`: the champion bundle's
  actual matched-defect distances top out at 12.65 m with nothing observed between there and the
  15 m cutoff, so the tolerance is not currently the thing inflating this number). Registration
  buys real, measured accuracy; it is not yet the bottleneck standing between this corpus and a
  cm-level dig mark — detection/clustering precision is.

### POD vs defect-moment angle to B_hat0

`scripts/pod_by_angle.py` measures the module docstring's central claim directly (a defect whose
magnetic moment is near-perpendicular to `B_hat0 = unit(background_nT)` is nearly invisible to a
scalar rig) rather than leaving it as prose, on 120 physical defects (10 lines, 3 runs each,
config/base.yaml density):

```
Spearman correlation(angle_cos, detected_fraction)   = +0.162 (p=0.078)
Spearman correlation(angle_cos, mean_peak_z)         = +0.163 (p=0.075)
Spearman correlation(angle_cos, peak_z_per_severity) = +0.183 (p=0.045)  <- severity-normalised

Detection rate by angle bin:            near-perpendicular 0.250 [0.13,0.38] | mid 0.217 [0.12,0.33] | near-parallel 0.392 [0.26,0.53]
Severity-normalised peak SNR by bin:    near-perpendicular 0.078 [0.06,0.11] | mid 0.069 [0.05,0.09] | near-parallel 0.092 [0.07,0.11]
```

A real, positive, population-level relationship exists between `|cos(angle to B_hat0)|` and
detectability — reaching conventional significance (p=0.045) once severity (a 4x confound, 20-80
range) is normalised out — and the near-parallel bin is clearly the most detectable of the three.
It is **not** a clean monotonic relationship (the "mid" bin is slightly, not significantly, lower
than "near-perpendicular" in both views) — expected, not a bug: unlike
`tests/test_generate.py`'s own null-anomaly unit test (which places the observation point exactly
along the moment's own axis to get an EXACT null), a real along-track survey pass sweeps the
sensor-to-source direction through many angles as the walker goes by, diluting the idealised
single-geometry relationship into a real-but-noisy population trend, not a deterministic one. The
physics claim is confirmed at population scale, honestly reported with its actual noise, not
overstated.

---

## Open questions for ROSEN (Rig-v2 assumptions)

The developer interview (5 Aug 2026, Richard Föcke) supplied the real rig description, but a
handful of the numbers given admit more than one honest reading, and Rig-v2 had to pick one to
build against without being able to ask ROSEN directly. Written up here with the reasoning
already used, not as vague hedges:

- **Is the ~100 µT sensor figure the ADC's full-scale range, or a noise/resolution figure?**
  The code assumes **full-scale range** (`SensorConfig.full_scale_ut`, `src/lsm/config.py`) and
  models 24-bit ADC quantization over it — Earth's field is ~50 µT, so ±100 µT is a standard
  fluxgate range, and this is almost certainly what was meant. If it were actually a noise floor
  instead, the instrument would be unusable for 25 nT anomalies — implausible, but worth
  confirming rather than assuming.
- **Is the ~1 cm figure the dig-marking accuracy target, or the along-track sample spacing?**
  Both are modelled (`registration.py` exists to hit the first; 120 Hz at ~1.2 m/s gives the
  second incidentally), but only the accuracy-target reading drove the registration design
  (Stage B, weld-comb detection locking GPS dead-reckoning to a centimetre-level lattice).
- **Is the rod vertical, or across-track (horizontal)?** Assumed **vertical**
  (`ArrayConfig.orientation`, the code's Stage-A "Decisions taken"). Vertical gives stand-off
  inversion and a genuine curvature term (`g2`); horizontal would trade that for lateral-offset
  estimation instead. The code deliberately raises `NotImplementedError` rather than silently
  guessing if `orientation != "vertical"` is ever set — a deliberate choice to surface this as an
  open question rather than paper over it.
- **Are the three heads factory-matched or field-calibrated?** This is the single largest lever
  on gradiometric performance: `SensorConfig.gain_sigma` models a 0.2% per-head gain mismatch,
  which leaves ~100 nT of uncancelled common-mode signal against a ~25 nT anomaly — verified
  directly in
  `tests/test_generate.py::test_gain_mismatch_leaves_the_predicted_common_mode_residual`.
- **Is walk speed logged via an odometer/IMU, or is dead reckoning
  (`src/lsm/registration.py`) truly GPS-only?** An odometer would make Stage B's registration
  substantially more accurate than pure GPS dead-reckoning through dropout.
