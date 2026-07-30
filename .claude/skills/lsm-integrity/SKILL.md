---
name: lsm-integrity
description: Workflow, invariants and conventions for the LSM pipeline-integrity demonstrator (magnetometry + GPS → defect indications → risk heat map). Use for ANY work in this repo: generating or ingesting survey data, data validation, background removal / feature engineering, training anomaly / severity / classification / growth models, MLflow tracking, conformal uncertainty, SQLite + S3 storage, batch inference, the Streamlit app, GitHub Actions CI/CD, or preparing ROSEN interview talking points. Triggers: LSM, magnetometry, magnetometer, nT, chainage, stand-off, pipeline defect, SMYS, indication, dig budget, heat map, ROSEN.
---

# LSM Integrity Demonstrator

A physics-inspired synthetic magnetometry dataset put through the **full production ML
lifecycle**. The deliverable is the *methodology*, not the physics. Every choice must be
defensible to a room of physicists who know the technique better than we do.

## Prime directive

> Be honest about what is synthetic, precise about what is physics, and rigorous about
> what is engineering. Never let a synthetic-data convenience masquerade as a real result.

If a step cannot be justified on real proprietary LSM data, it does not belong here.

## Non-negotiable invariants

Break any of these and the demo stops being credible.

1. **Never split rows randomly.** Neighbouring 0.5 m samples are near-duplicates and the
   same defect appears in every survey. Split by `line_id` (preferred) or by 100 m
   chainage block, using `GroupKFold`, with the group key **`(line_id, block)` — never
   `block` alone**, so overlapping surveys of the same stretch cannot cross folds. A test
   that asserts no group appears in both folds lives in `tests/test_leakage.py` and must
   stay green.
2. **Two-stage architecture.** Row-level scoring → cluster peaks into **indications** →
   per-indication severity, class and risk. Never regress `severity_smys` row-wise: it is
   0 off-defect and constant inside the ±2 m label box, so a row-level regressor learns
   "predict 0" and reports a flattering MAE.
3. **Split transforms by statefulness.** Per-survey transforms (detrend, high-pass,
   gradient) are *fit at inference on the incoming survey* — that is correct, they are
   background removal. Cross-survey statistics (feature scalers, calibration maps,
   conformal quantiles, decision thresholds) are *fit on train only* and shipped inside
   the model bundle. `features.py` must make this distinction explicit in code, not in a
   comment. See `references/architecture.md`.
4. **Metrics that an inspection engineer can act on.** PR-AUC, not ROC-AUC (positives are
   ~2.7%). Headline number is **recall at a fixed dig budget** (top-k indications per km)
   plus **false-dig rate** and **localisation error in metres**. Accuracy is banned.
5. **Every prediction ships with its uncertainty and its data-quality flag.** An indication
   with no interval or an unresolved DQ warning is not a prediction, it is a rumour.
6. **One `config.yaml`, one seed, one entry point.** `python -m lsm <step>`. Any run must
   be reproducible from `git_sha` + `config_sha256` + `data_sha256`, all three logged to
   MLflow and to the `model_run` table.
7. **Interference is the adversary, not noise.** Off-centreline sources are the designed
   false-positive trap. Report interference precision separately, always. A model that
   beats the MAD baseline only by flagging interference has failed.
8. **Never key or join on a float.** The physical key is the integer `sample_idx`;
   `chainage_m = sample_idx * step_m` is derived and is never a key, join column or
   grouping boundary. `lat`/`lon` are float64 — mandatory, since float32 quantises GPS to
   ~0.42 m at this latitude. Field values are float32 in storage and **float64 in
   arithmetic**. Full contract in `references/data-contract.md`.
9. **Three versions, all explicit:** `schema_version`, `feature_version`, `model_version`.
   `feature_version` is the one that bites — edit `features.py` and every stored feature is
   silently stale, which is training/serving skew no *data* check can see. Bundles pin it
   and refuse to load on a mismatch.
10. **Report intervals, not point estimates.** Every headline metric carries a bootstrap CI
    resampled over groups. With ~12 defects, a bare "recall = 0.83" is sampling noise, and
    a gate comparing two point estimates promotes models on luck.
11. **`pipeline_version` is the deployable unit, not `model_version`.** An indication comes
    from up to four models plus a feature version; they are promoted, served, referenced and
    rolled back together. Every analytical query filters on `pipeline_version` and
    `is_shadow` — the table holds output from several releases at once.
12. **Point-in-time correctness.** Every feature for a survey must be computable from data
    available at that survey's `surveyed_at`. No cross-survey normalisation, no attribute
    borrowed from a later run. Spatial grouping does not catch temporal leakage.
13. **Config is layered, and only the code layer is hashed.** `config/base.yaml` (hashed) /
    `config/{dev,prod}.yaml` / runtime params. A flat config makes `config_sha256` change
    with the environment, which silently voids reproducibility.
14. **The serving layer is the batch scoring job.** The Streamlit app is a *consumer* of the
    `indication` table, never "the server". Do not conflate them in code or in conversation.
15. **MLflow 3 removed model stages.** Use `@champion` / `@challenger` **aliases**.
    `transition_model_version_stage` is a dead API.

## Physics facts to state correctly

- The scanner carries **one 3-axis magnetometer**. Three axes are not three sensors.
  Without a second spatially separated sensor there is **no gradiometry** — only the
  **along-track spatial derivative** dB/ds. Say "along-track gradient" unless the
  two-sensor generator upgrade (Stage 2) is in place, then say "vertical gradiometer".
- Dipole field falls off as **1/r³**. Stand-off normalisation therefore scales residual
  amplitude by `depth³`. Off-pipe interference is farther away *and lateral*, so it is
  **weaker and broader** — width/decay-shape features separate it from on-pipe defects far
  better than amplitude alone. That is the core physical insight of the whole project.
- Background is ~19000/1000/45000 nT with drift; defect residuals are ~25 nT against a
  ~4 nT floor. The signal is ~0.05% of the raw field. **Background removal is the project.**

## Pipeline stages and their commands

The CLI is the *implementation*; the **Dagster asset graph is the interface**, partitioned
by `survey_id`, with backfill over the archive as a first-class partition backfill. Retries:
3× on IO, **zero on DQ failure** — a bad survey is not a flake.

```
raw_survey (S3 sensor) → ingested_survey → dq_report → survey_features(fv=n)
                                              ↓              ↓
                                        quarantine/    indications → gis_export
                                                                  → drift_report
```

```
python -m lsm generate    # synthetic surveys -> data/raw/*.parquet (+ S3 mirror)
python -m lsm ingest      # raw -> SQLite (survey, reading, truth_*) with content hashing
python -m lsm validate    # DQ gates -> dq_report table + MLflow artifact; exits non-zero on FAIL
python -m lsm features    # background removal + window/shape features -> feature store
python -m lsm train       # MLflow-tracked; emits a versioned model bundle
python -m lsm predict     # batch inference on a survey -> indication table + GeoJSON
python -m lsm forecast    # per-defect growth -> remaining life
python -m lsm monitor     # drift: PSI/KS vs bundle reference, indications/km, regime shift
python -m lsm serve       # Streamlit map + risk heat map (a CONSUMER of `indication`)
```

`validate` runs **before** `features` and again inside `predict`. Same validator, same
code path, both times. That symmetry is the point.

## Where things live

| Concern | Location |
|---|---|
| System-level design | `docs/production-architecture.md` — environments, serving contract, delayed labels, SLOs, runbook, and what is built vs designed |
| Orchestration | Dagster assets in `src/lsm/dagster_defs.py`, partitioned by `survey_id` |
| The data contract | `src/lsm/schemas.py`, enforced at **every** boundary |
| Bulk signal + labels | SQLite `data/lsm.db` (schema in `references/architecture.md`) |
| Raw immutable surveys | `data/raw/*.parquet`, mirrored to `s3://$LSM_BUCKET/raw/` |
| Rejected surveys | `data/quarantine/` + DQ report — never deleted, never silently skipped |
| Experiment tracking | MLflow, local `mlruns/` in dev, server + S3 artifacts in "prod" |
| Model bundles | MLflow artifact + `s3://$LSM_BUCKET/models/<version>/bundle.joblib` |
| Config | `config/base.yaml` (hashed) + `config/{dev,prod}.yaml` (not hashed) |
| Releases | `pipeline_release` table + MLflow `@champion` / `@challenger` aliases |
| CI | `.github/workflows/{ci,train,deploy}.yml` |

## Reference material

- `docs/production-architecture.md` — **the system view**: environments and config layering,
  orchestration and backfill, the batch serving contract, release/rollback, the
  delayed-label operating model, monitoring and runbook, SLOs/retention/cost, and an
  explicit built-vs-designed split. Read this before any question that starts "in
  production, how would you…".
- `references/data-contract.md` — **read before touching any schema, dtype or key**: keys,
  precision, units, nulls and edge policy, enums, the two hashes and deduplication,
  schema/feature versioning, Parquet layout, data classification.
- `references/architecture.md` — repo layout, `config.yaml` contract, SQLite DDL, model
  bundle contents, MLflow conventions, S3 layout, serving artifact and app, scale path
  beyond SQLite.
- `references/validation-and-trust.md` — the five validation layers, concrete DQ checks
  with thresholds, promotion gates, drift monitoring, the dig-feedback loop.
- `references/interview-drills.md` — the hard questions ROSEN can ask, mapped to the stage
  that answers them, with the honest one-paragraph answer.

## Working conventions

- Training runs in the `ml_gpu` conda env: **Python 3.12.13, dagster 1.13.15, mlflow 3.13.0,
  lightgbm 4.6.0**. The local venv is Python 3.13 — that mismatch is a real skew risk, so
  pin one Python version across training, local and the app runtime, and make
  `bundle.load()` **hard-fail** on a LightGBM/numpy/sklearn version mismatch rather than
  just recording versions.
- No `make` on this machine — the Typer CLI is the entry point; `Makefile` targets are thin
  wrappers only.
- **LightGBM throughout** — industry default, materially faster on the Stage-6 row counts,
  native SHAP contributions. Set `deterministic=True`, `force_row_wise=True`, fixed
  `num_threads`, or the CI determinism test flakes.
- Serving is **CPU-only and takes milliseconds**; the app runs on Hugging Face Spaces
  (Streamlit SDK) and the iPad is a thin client. Anything trained on GPU must be saved
  CPU-loadable (`map_location="cpu"` or ONNX).
- Hardware: Ryzen 9 7950X (16C/32T), 128 GB RAM, RTX 5080 16 GB. Tabular models are
  **CPU-bound and fine** — use `n_jobs=16`. The *base* interpreter's torch is CPU-only; do
  not write GPU code paths unless the 1-D CNN stage is explicitly started, and then note
  Blackwell (sm_120) needs a CUDA 12.8+ wheel.
- Style: match the existing `generate_lsm_data.py` — module docstring explaining the
  physics, short functions, comments that state *why*, no defensive boilerplate.
- When adding a model, add its baseline first. A model with no baseline is not a result.
- **Structured JSON logs** correlated by `survey_id` / `run_id` / `pipeline_version`. An
  alert with no destination is not monitoring; a log with no correlation key is not
  debuggable.
