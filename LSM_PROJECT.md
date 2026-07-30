# Above-Ground Magnetometry → Pipeline Integrity: an ML-Lifecycle Demonstrator

**One line:** A physicist's end-to-end ML pipeline that turns 3-axis magnetometer + GPS data into a prioritised heat map of likely pipeline defects — built to demonstrate methodology and MLOps discipline, not physical fidelity.

## The honest framing (use this verbatim if asked)
> "Real Large Stand-Off Magnetometry data is proprietary, so I built a physics-inspired synthetic generator — a magnetic-dipole forward model over a realistic drifting geomagnetic background — and then ran the *full* ML lifecycle on it exactly as I would on their data: validation, feature engineering, modelling with uncertainty, tracking, and a served heat map. The point isn't the data; it's showing how I'd approach the problem as someone learning the technique."

This maps directly to what ROSEN said in the first interview: the team is all physicists, the technique is rare, and they expect the person to learn it. The project *is* that story.

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
Python · numpy/pandas · scikit-learn + XGBoost/LightGBM · MLflow · Streamlit · pytest. (All already on your CV — lean into that.)

## Talking points for the interview
- "Most of the work was validation and background removal, not the model — same as it would be on real LSM."
- ~~"Three sensors let me use gradiometry to reject the common-mode background."~~ **Wrong, and corrected in Stage 2:** three *axes* are not three *sensors*. With one magnetometer there is no gradiometry, only the along-track derivative dB/ds. Stage 2 added a second sensor head at a 0.5 m vertical baseline, which makes the claim literally true — and then measured it: ~10 dB common-mode rejection, but *worse* detection contrast than detrending (2.3× vs 4.5×), because differencing adds √2 sensor noise after the detrend has already removed the background. The gradiometer earns its place where the background is *not* smooth along-track, and because it needs no fitting window and therefore has no edge rows.
- "I built in interference sources on purpose, because distinguishing a real defect from benign magnetic interference is the hard part."
- "Uncertainty is a first-class output — an inspection engineer needs to know how much to trust a number before authorising a dig."
- "Repeat surveys give growth, and growth plus a limit state gives remaining life — that's the proactive-maintenance loop."

## Stretch (only if you have spare time)
Real observatory background; a tiny ILI-vs-LSM "data fusion" mock; conformal prediction for the regression; a simple active-learning demo ("which segment should we dig next?").
