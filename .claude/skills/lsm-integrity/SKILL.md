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
   with no interval or an unresolved DQ warning is not a prediction, it is a rumour. The
   flag's own reported scope has to be honest too: a validation check that reports
   "n_affected = len(df)" when only one row actually failed (a placeholder from early
   scaffolding, not a deliberate choice) just relocates the rumour into the DQ report
   itself — extract the underlying validator's structured failure info (which rows, which
   check) rather than defaulting to "the whole thing," especially before surfacing it to a
   user.
6. **One `config.yaml`, one seed, one entry point.** `python -m lsm <step>`. Any run must
   be reproducible from `git_sha` + `config_sha256` + `data_sha256`, all three logged to
   MLflow and to the `model_run` table.
7. **Interference is the adversary, not noise.** Off-centreline sources are the designed
   false-positive trap. Report interference precision separately, always. A model that
   beats the MAD baseline only by flagging interference has failed.
8. **Never key or join on a float.** The physical key is the integer `sample_idx`, still
   dense and monotonic under Rig-v2 — but it is now a **time-sample counter**
   (`walk.sample_rate_hz`), not a distance-grid index. `chainage_m = sample_idx * step_m`
   is **no longer true**: the rig is a rod carried by a human walker at irregular speed,
   so along-track spacing is irregular by construction. Raw carries `chainage_true_m`
   (truth tier — the generator's own exact position, may be used to score registration,
   never as a feature) and, only as an interim convenience column, a naive
   constant-speed `chainage_provisional_m` — deliberately **not** named `chainage_m`, so
   nothing downstream mistakes it for a physically-final chainage. The real, registered
   `chainage_m` is a Stage B (`registration.py`) OUTPUT, written to the feature layer,
   built from GPS (which drops out) dead-reckoned and then locked to the recovered
   girth-weld lattice — see `references/data-contract.md`. `lat`/`lon` are float64 and
   **nullable**: GPS drops out in poor sky view and the gap is written as NaN, not
   synthesised. Field values are float32 in storage and **float64 in arithmetic**.
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

- The rig (Rig-v2, confirmed by the instrument's own developer) is a rod carrying
  **three magnetometers, 50 cm apart** (middle + two) — real gradiometry, including a
  genuine **second difference**, not just the along-track derivative. But each head
  reports only its **total-field magnitude `|B|`**, one scalar, never x/y/z — there is no
  vector output anywhere in the scalar rig. Never say "one 3-axis magnetometer" or
  "along-track gradient only"; that described the pre-Rig-v2 model.
- **Scalar output is total-field-anomaly physics, not a format detail.** With the anomaly
  (~25 nT) tiny against the ambient field (~48,800 nT), `|B0+dB| - |B0| ~= dB . B_hat0` to
  <0.01 nT — you only ever see the anomaly's **projection onto the ambient field
  direction**. Consequences to state, not hand-wave: (1) a defect whose moment is
  near-perpendicular to `B_hat0` is nearly invisible — detection probability genuinely
  varies with defect orientation; (2) `|B|` is **rotation-invariant**, so rod sway/tilt
  moves head positions but cannot by itself corrupt a reading — almost certainly why the
  instrument reports scalar in the first place; (3) anomaly *shape* depends on ambient
  inclination and walk bearing, so shape features are bearing-dependent.
- **Three heads give a second difference, not just a first.** With `b_lo, b_mid, b_hi` at
  −0.5/0/+0.5 m along the mast: the first difference `(b_hi − b_lo)` cancels the
  common-mode background (the old Stage 2 story); the second difference
  `(b_hi + b_lo − 2*b_mid)` additionally cancels any **linear** background gradient, which
  the first difference does not — the specific extra value the third head buys. The
  head-to-head amplitude ratio also inverts for source distance via 1/r³, giving a
  *measured* stand-off instead of an assumed constant one.
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
python -m lsm features    # Stage B registration (chainage from GPS dead-reckoning + weld-comb
                           # lock) THEN background removal + window/shape features -> feature store
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
| Along-track registration | `src/lsm/registration.py` (Stage B) — GPS dead-reckoning through dropout + girth-weld-comb detection -> registered `chainage_m` + `dist_to_weld_m`; runs inside the `survey_features` asset, not as its own Dagster asset (see `pipeline.py::run_feature_pipeline`) |
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

## Building a Streamlit app: hard-won conventions, not LSM-specific

Learned building Stage 4.5's demo app, but every point below is a Streamlit/data-app
convention, not a physics or pipeline one — apply it on any project with a Streamlit
front end.

- **Streamlit does NOT add the executed script's own directory to `sys.path`**, unlike
  plain `python script.py`. If the app imports sibling modules (a shared plotting helper, a
  data-access module), add `sys.path.insert(0, str(Path(__file__).resolve().parent))`
  explicitly, or the import only fails the first time the app is actually *run* — not when
  it's written, not when its functions are unit-tested.
- **The whole script reruns top-to-bottom on every widget interaction — including every
  `st.tabs` body, even the ones not currently visible.** Anything expensive or side-
  effecting called at module level or inside a tab will silently re-run on a click that has
  nothing to do with it (a slider in one tab retriggering a live pipeline run meant for a
  different control entirely). Cache pure data loads with `st.cache_data`; cache anything
  with side effects (a live scoring run, a DB write) in `st.session_state`, keyed by the
  actual input that should invalidate it (e.g. a scenario id) — not "has this ever run this
  session." Verify the fix, don't just assume it: assert the cached object is the *same
  object* (not just equal) before and after an unrelated widget change.
- **`st.cache_resource` is a process-wide singleton shared across every concurrent session
  hitting the app — never put a writable or session-scoped resource in it.** It's correct
  for a read-only, shareable connection; wrong the moment anything writes through it (one
  user's action would then affect every other session). Use `st.session_state` for anything
  written to or scoped per browser session.
- **A session-scoped SQLite connection needs `check_same_thread=False`, and the reason is
  not what it looks like.** It is not a real concurrent-access hazard (`st.session_state`
  already ensures only one session owns it) — it's that Streamlit can dispatch one
  session's own *consecutive reruns* onto different worker threads from its internal thread
  pool, which trips sqlite3's same-thread guard even though only one rerun ever executes at
  a time. A false-positive guard, not a real one — but it still raises unless disabled.
- **Any charting library with a "big data" safety cap (Altair/Vega-Lite's default 5000-row
  limit is the common one) fails SILENTLY on real but modest data, not loudly.** Melting a
  few thousand rows into several series (e.g. 3 axes of a sensor reading) multiplies the row
  count past the cap fast, and the failure mode is an empty or partially-rendered chart, not
  a visible error — reproduce the exact chart-building call directly (`chart.to_dict()`) when
  a chart looks wrong before assuming the data or the logic is broken. If the true row count
  is bounded and modest (thousands, not millions), it's legitimate to disable the cap
  rather than pre-aggregate; say explicitly in code why the scale is bounded, so a future
  genuinely-huge dataset doesn't inherit the same "just disable the check" fix.
- **A segmented/toggle-style single-select widget (`st.segmented_control` and equivalents)
  can return `None` when clicked on its own already-selected option** (default "toggle off"
  behaviour) — code that assumes the return value always matches one of the offered options
  (`next(x for x in options if x == value)`) will crash on exactly that click. Prevent it at
  the widget (a `required=True`-style flag, if the widget offers one) AND defend in the
  code that consumes the value (`next((...), fallback)`), since a widget's deselection
  behaviour is framework/version detail, not something to rely on staying fixed.
- **Headless verification means `streamlit.testing.v1.AppTest`, not a hand-written mirror
  script and not just unit-testing the functions the app calls.** `AppTest` runs the actual
  script through Streamlit's real `ScriptRunner` and catches exactly the class of bug pure
  unit tests structurally cannot see, because the bug is in how the *framework* runs the
  script, not in the business logic (both gotchas above were found this way, before a human
  ever opened a browser). Any Streamlit app in a project like this should have an
  `AppTest`-based test file, in addition to (not instead of) unit tests for its non-UI
  helper functions.
- **Demo-visualization UX, learned from a real user's first reaction, not guessed:**
  - Don't collapse a multi-channel signal into one aggregate-magnitude line when the point
    of the chart is "can you see structure here" — plot the components separately.
  - A literal log-scale axis doesn't work on signed data centred away from zero (a raw
    sensor field, say). If a log view is requested to "see" something small against a huge
    background, show the *deviation from a robust baseline* (e.g. `|x - median(x)|`) on the
    log axis instead — that's the transform that actually reveals structure.
  - Overlay ground-truth/reference markers so a "there's nothing visible here" claim is
    something the viewer can verify themselves, not just something the app asserts in a
    caption.
  - Caption *why* a view matters, not just what it shows, and define domain jargon in an
    intro line (what is a "survey", a "scenario", etc.) — skipping this reads as an
    unfinished/"naked" app even when the underlying logic is completely correct.
