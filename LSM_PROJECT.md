# Above-Ground Magnetometry → Pipeline Integrity: an ML-Lifecycle Demonstrator

**One line:** A physicist's end-to-end ML pipeline that turns 3-axis magnetometer + GPS data into a prioritised heat map of likely pipeline defects — built to demonstrate methodology and MLOps discipline, not physical fidelity.

## Project Context & Modeling Strategy
Real Large Stand-Off Magnetometry data is highly proprietary. To enable this research and demonstration, a physics-inspired synthetic generator was built—implementing a magnetic-dipole forward model over a realistic drifting geomagnetic background. The project implements the full ML lifecycle on this dataset: raw data ingestion and validation, feature engineering (background detrending and gradients), uncertainty-calibrated modeling, anomaly tracking, and risk heat map serving.

## What it demonstrates (the five scenarios, on one dataset)
1. **Anomaly detection** — find stress-concentration zones with mostly unsupervised methods after background removal.
2. **Severity regression** — map the magnetic signature to a `%SMYS`-style value *with calibrated uncertainty*.
3. **Defect classification** — SCC vs weld vs dent vs corrosion vs benign interference.
4. **Risk / prioritisation** — combine severity, class and (proxy) consequence into a ranked heat map.
5. **Growth forecasting** — three repeat surveys with growing defects → project remaining life.

## The data
`generate_lsm_data.py` writes `lsm_synthetic.csv` in seconds. A 2 km line sampled every 0.5 m, three surveys, 12 growing defects plus 4 off-pipe interference sources (deliberate false-positive traps).

| column | meaning |
|---|---|
| `run_id` | survey index 0–2 (defects grow ~15% per survey) |
| `chainage_m` | distance along the pipe |
| `lat`, `lon` | GPS along a fixed bearing near Stans |
| `Bx_nT`, `By_nT`, `Bz_nT` | 3-axis field: large background + drift + defect dipoles + noise |
| `defect` | 1 within ±2 m of a true defect, else 0 |
| `defect_type` | scc / weld / dent / corrosion / none |
| `severity_smys` | proxy severity (~20–95), grows across surveys |

Signal-to-check: after a cubic detrend, defect residuals are ~25 nT against ~4 nT elsewhere — learnable, but only *after* you remove the background. That background-removal step is the whole point, and interference makes naive thresholding fail.

**Optional realism:** swap the synthetic background for a real geomagnetic trace (BGS/NOAA observatory) so you can say the ambient field is real. Nice-to-have, not required.

## Repo structure
```
lsm-integrity-demo/
├── data/                    # generated CSVs (gitignored)
├── src/
│   ├── generate_lsm_data.py # the generator (done)
│   ├── validate.py          # schema + range + GPS-continuity checks
│   ├── features.py          # background removal, gradients, window stats
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

**2 · Features** — remove the background (polynomial/robust detrend or high-pass), compute 3-sensor **gradients** (common-mode rejection sharpens local anomalies), field magnitude/orientation, sliding-window mean/std/peak, and normalise by depth/stand-off. Keep a clear feature table.

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
- **Sensor Configuration & Gradiometry:** A single 3-axis magnetometer allows calculation of the along-track spatial derivative ($dB/ds$). Adding a second sensor head (0.5 m vertical baseline) introduces true vertical gradiometry. In testing, this setup provided ~10 dB common-mode rejection. However, the vertical gradiometer showed a worse detection contrast (2.3× vs 4.5× for single-head detrending) because the difference operation accumulates sensor noise ($\sigma\sqrt{2}$) after detrending. The gradiometer is primarily beneficial when the ambient background is highly non-uniform along-track, and because it requires no fitting window, avoiding edge-row loss.
- **Interference Separation:** Incorporating synthetic external magnetic interference sources is necessary to evaluate the model's ability to distinguish true localized stress defects from benign external objects.
- **Uncertainty Quantification:** Point predictions are insufficient for pipeline safety assessments. Predictive outputs must include calibrated uncertainty intervals to assess structural excavation risk.
- **Temporal Analysis:** Fusing repeat surveys allows modeling of defect growth over time, enabling remaining life projections.

## Stretch (only if you have spare time)
Real observatory background; a tiny ILI-vs-LSM "data fusion" mock; conformal prediction for the regression; a simple active-learning demo ("which segment should we dig next?").

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
*   **Project alignment, corrected:** this project has real bootstrap CIs, a real leakage test, and real (including negative) measured results — rigor this paper doesn't demonstrate having, whatever techniques it names. The one real gap it points at honestly: SHAP/feature-importance analysis is **Stage 5 in this project's own plan, not yet built** — a previous version of this section claimed it was already incorporated, which wasn't true.

### 3. Pipeline Defect Detection using SVM (Isa, Rajkumar, & Woo, 2007)
*   **Title:** *Pipeline Defect Detection Using Support Vector Machines*
*   **Focus:** Continuous monitoring of pipe wall thinning using guided ultrasonic wave propagation on a lab-scale rig, one defect.
*   **Key Findings:**
    *   **Signal Processing:** Discrete Wavelet Transform (DWT) (Haar vs. Daubechies DB2) to compress raw 1D acoustic signals and filter high-frequency noise.
    *   **Classification:** DWT coefficients fed to an SVM (LIBSVM); RBF kernel performed best, **89.65%** classification accuracy, Haar wavelet, frame window 25. No cross-validation, no confidence interval, no baseline comparison, and a single lab defect — a reasonable 2007-vintage proof of concept, not a rigor bar.
*   **Project alignment, corrected:** this project does **not** use an SVM anywhere — Stage 5's plan is LightGBM multiclass, not RBF-SVC. A previous version of this section claimed this paper "supports the choice of RBF SVC in our classification benchmarks," which was wrong on two counts: no SVC exists in this project, and this 18-year-old single-defect result isn't grounds to choose one over LightGBM if it did. What legitimately carries over is the general idea that window-based features work for localized classification — the classifier choice doesn't.

### 4. ROSEN LSM Technical Flyer (Product Specifications)
*   **Focus:** Commercial capabilities of ROSEN's above-ground Large Stand-Off Magnetometry (LSM) system for unpiggable pipelines.
*   **Key Specifications:**
    *   *Pipeline Diameter:* 152–1820 mm (6"–72").
    *   *Optimal Standoff:* Up to 12 times the pipe diameter (e.g., ~3 m or 9.8 ft for a 10" pipeline).
    *   *Accuracy:* Lateral accuracy within 100 mm (0.33 ft); mapping accuracy ±5% of actual position.
    *   *Performance:* Probability of Detection (POD) >80% at a 95% confidence level, **explicitly stated "in the absence of magnetic interference."**
    *   *Output:* Identifies Stress Concentration Zones (SCZs) and reports stress magnitude in MPa or %SMYS.
*   **Project alignment, precisely stated (not simplified):** these specs define the physical bounds the synthetic generator was built to respect (standoff distances, lateral resolution, noise regime). On the *measured* comparison: this project's real recall-at-dig-budget is ~64% (both the MAD baseline and IsolationForest), which sits below the flyer's 80% figure — but the comparison is **not apples-to-apples**, because this project's evaluation corpus *always* includes interference by design (it's the deliberate false-positive trap the whole project is built to stress-test), while the flyer's 80% figure explicitly excludes interference. The honest statement is "64% under a harder condition their own spec doesn't cover," not a direct miss against their number.

### 5. My own interview-prep synthesis (not a ROSEN document)
Built for the 2nd-round interview (5 Aug 2026, Richard Föcke, Head of NDT Apps & Products) from
the flyer plus general domain reasoning — notes to prepare with, not something ROSEN
published. Its own 7-stage framing of the pipeline (**1** ingest & integrate → **2** register/align
→ **3** signal processing & features → **4** label from digs → **5** model → **6** calibrate +
uncertainty → **7** serve heat map) maps closely onto this project's actual stage structure, with
one honest, deliberate gap worth naming proactively rather than hoping it doesn't come up:
*   **Stage 2, "register/align," has no counterpart here.** This project's synthetic generator
    keeps every defect's chainage position exactly fixed across a line's repeat runs, so there is
    no misalignment for a registration step to fix. Real LSM data (odometer slip, GPS drift
    between runs) would need one — this demonstrator doesn't simulate that failure mode, so it
    doesn't build the fix either.
*   **Its "recall/POD, PR-AUC, sizing error, calibrated uncertainty, cost-sensitive evaluation"
    metrics guidance is genuinely what this project already does** (Stage 3's recall-at-budget +
    false-dig-rate + PR-AUC-as-diagnostic, Stage 4's split-conformal calibration, both reported
    with bootstrap CIs, never bare accuracy).
*   **Forecasting (its stage-6-adjacent mention of corrosion growth/remaining life)** maps to
    this project's **Stage 8, a CLI stub, not built** — and forecasting is separately named as
    required experience in the actual job posting, which makes this the more load-bearing gap
    of the two, not just a nice-to-have.
