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
Based on *Mitra et al.*'s paper on ML-based LSM anomaly detection, here is a detailed comparison and actionable additions to our pipeline.

### Project vs. Research Paper Comparison

| Category | Our Implementation | Research Paper Proposal | Actionable Addition |
| :--- | :--- | :--- | :--- |
| **Data Alignment** | Detrending and along-track gradients. | **FastDTW** (linear time-series alignment). | FastDTW to align multi-run surveys (Runs 0-2). |
| **Unsupervised Model** | Isolation Forest + robust-MAD threshold. | **k-Nearest Neighbors (k-NN)** (`pyod`). | Incorporate k-NN from `pyod` as baseline. |
| **Supervised Modeling** | Separate regression and classification. | **Unified Multi-Output 1D-CNN** in Keras. | Multi-output Conv1D for mask + depth + volume. |
| **Hardware / Scaling** | CPU-bound training. | **RAPIDS AI cuML** for GPU-accelerated SVC. | Leverage cuML to solve $O(N^3)$ SVC complexity. |
| **Data Leakage Split** | Grouped split by chainage segment. | Hold out **entire contiguous defect regions**. | Define 3-foot buffer bounds around defects. |
| **Data Augmentation** | Synthetic dipole forward generator. | **Magneto-restriction principles** for physics. | Augment training via stress-induced permeability. |

### Key Additions Detail
1. **Dynamic Time Warping (FastDTW) for Multi-Survey Alignment:** Physical offsets across different inspections cause phase shifts. FastDTW aligns different surveys prior to feature extraction in [features.py](file:///c:/Ali/kaggle/Rosen/project/src/features.py).
2. **k-NN as Primary Unsupervised Baseline:** In the paper, k-NN was the only `pyod` model that effectively isolated anomalies with minimal noise. We will incorporate `pyod.models.knn.KNN` in [train.py](file:///c:/Ali/kaggle/Rosen/project/src/train.py).
3. **Multi-Output 1D-CNN:** The authors built a 1D-CNN with multi-level spatial concatenations to predict classification mask and regression depth/volume simultaneously.
4. **RAPIDS cuML Acceleration:** RBF-kernel SVC scales poorly ($O(N^3)$). Using cuML on GPU achieves a 200x speedup, making scale-up to millions of field points computationally feasible.

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

Below is an overview of the technical papers and reference documents found in the folder, detailing how they align with or expand upon the methodologies implemented in this project.

### 1. npj Materials Degradation Review (Olawole et al., 2026)
*   **Title:** *Advanced sensor systems and machine learning for pipeline integrity management: a review of corrosion monitoring and prediction strategies*
*   **Focus:** A comprehensive evaluation of In-Line Inspection (ILI) techniques (MFL, UT, EMAT) vs. long-range continuous monitoring (GWUT, AET, DFOS) and the machine learning architectures used to interpret their signals.
*   **Key Findings:**
    *   **The "ML Hook":** Sensors create either *volume bottlenecks* (e.g., DFOS producing 30 million points/sec, needing compression) or *complexity bottlenecks* (e.g., MFL and GWUT, where signals are indirect, non-linear, and obscured by operational noise).
    *   **Multimodal Data Fusion:** Combining MFL (magnetic) and UT (acoustic) compensates for individual sensor blind spots (e.g., MFL is blind to mid-wall laminations, while UT fails in gas pockets). Fusing them increases Probability of Identification (POI).
    *   **Advanced ML Paradigms:**
        *   *GAN Translation:* Using Generative Adversarial Networks (e.g., Res-Pix2Pix) to translate raw 2D MFL maps into pseudo-UT thickness profiles to ease fusion.
        *   *FEA-ANN Prognosis:* Training an Artificial Neural Network on finite element simulations of parameterized defects (varying length, width, and depth) to predict pipeline failure pressure instantly (relative error 0%–7%), avoiding slow real-time FEA.
        *   *Trust & PINNs:* Emphasizes the need for Explainable AI (SHAP/LIME) and Physics-Informed Neural Networks (PINNs) that constrain the loss function using physical laws (Maxwell's equations and Ramberg-Osgood stress-strain relations).
*   **Project Alignment:** Our project aligns directly with the "Sim-to-Real" transfer learning workflow highlighted here. We generate synthetic anomalies using a physical forward model, implement feature engineering, and design robust grouped-splitting schemes to handle the complexity bottleneck before modeling.

### 2. Subsea Pipeline Integrity Model (Wegner et al., 2021)
*   **Title:** *A Machine Learning-Enhanced Model for Predicting Pipeline Integrity in Offshore Oil and Gas Fields*
*   **Focus:** Fuses heterogeneous operational registries to predict structural degradation in subsea flowlines.
*   **Key Findings:**
    *   **Data Fusion:** Integrates tabular data from Remotely Operated Vehicle (ROV) visual inspection reports, Cathodic Protection (CP) electrical surveys, and maintenance registries.
    *   **Modeling:** Applies supervised learning (Support Vector Machines and Random Forests) to forecast armor loss, coating damage, and corrosion fatigue.
    *   **Explainability:** Evaluates model risk features using SHAP value analysis to help pipeline engineers understand key risk drivers.
*   **Project Alignment:** Validates our use of auxiliary metadata (like GIS, operational runs, and pipe attributes) and highlights the importance of using feature importance and SHAP analysis for model trust, which we incorporate in Stage 6 of our [PLAN.md](file:///c:/Ali/kaggle/Rosen/project/PLAN.md).

### 3. Pipeline Defect Detection using SVM (Isa, Rajkumar, & Woo, 2007)
*   **Title:** *Pipeline Defect Detection Using Support Vector Machines*
*   **Focus:** Continuous monitoring of pipe wall thinning using guided ultrasonic wave propagation on a lab-scale rig.
*   **Key Findings:**
    *   **Signal Processing:** Applies **Discrete Wavelet Transform (DWT)** (comparing Haar and Daubechies DB2 wavelets) to compress raw 1D acoustic signals and filter out high-frequency environmental noise.
    *   **Classification:** Feeds the DWT coefficients into an SVM (LIBSVM). Comparing polynomial, RBF, and sigmoid kernels, the **RBF kernel** performed best, achieving **89.65% classification accuracy** with a Haar wavelet and a frame window size of 25.
*   **Project Alignment:** Confirms the utility of window-based features and RBF-kernel SVMs for localized classification, supporting the choice of RBF SVC in our classification benchmarks.

### 4. ROSEN LSM Technical Flyer (Product Specifications)
*   **Focus:** Commercial capabilities of ROSEN's above-ground Large Stand-Off Magnetometry (LSM) system for unpiggable pipelines.
*   **Key Specifications:**
    *   *Pipeline Diameter:* 152–1820 mm (6"–72").
    *   *Optimal Standoff:* Up to 12 times the pipe diameter (e.g., ~3 m or 9.8 ft for a 10" pipeline).
    *   *Accuracy:* Lateral accuracy within 100 mm (0.33 ft); mapping accuracy ±5% of actual position.
    *   *Performance:* Probability of Detection (POD) >80% at a 95% confidence level (in the absence of magnetic interference).
    *   *Output:* Identifies Stress Concentration Zones (SCZs) and reports stress magnitude in MPa or %SMYS.
*   **Project Alignment:** These operational specifications define the physical bounds of our synthetic generator (e.g., simulating sensor standoff distances, lateral resolutions, and noise regimes) and establish our target metrics (beating POD >80% at 95% confidence).

### 5. ROSEN Reference Standards & Best Practices
*   **Methodology Guidelines:**
    *   **Decision-Driven Pipelines:** All model predictions must integrate with pipeline integrity decision logic (e.g., categorizing anomalies to guide excavation vs. continuous monitoring plans).
    *   **Priority on Ingestion & Alignment:** Sensor data integration, spatial alignment (FastDTW), and data quality validation form the high-value core of the NDT processing chain.
    *   **Evaluation Metrics:** Standard accuracy is rejected due to extreme class imbalance. Instead, the model is evaluated on Recall (Probability of Detection), PR-AUC, and sizing error (MAE). A high penalty is placed on False Negatives (unregistered defects) relative to False Positives.
    *   **Calibration & Reliability:** Estimations are reported as calibrated probabilities or physical uncertainty intervals to ensure high engineering reliability.
*   **Project Alignment:** These structural guidelines dictate our focus on data validation and quarantine (Stage 1), FastDTW sequence alignment (Stage 2), cost-sensitive evaluations (Stage 3), and calibrated uncertainty outputs (Stage 6).
