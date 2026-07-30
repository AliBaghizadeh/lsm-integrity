# Implementation plan — LSM integrity demonstrator

**Goal.** Rehearse bringing magnetometry + GPS survey data into production, end to end, on
the infrastructure I actually use (MLflow, SQLite, S3, GitHub Actions), so that ROSEN's
questions about validation, trust and scale have worked answers behind them.

**Ordering principle.** Ship a *tracked, tested, reproducible thin slice* before deepening
any model. Stages 0–4.5 are the demonstrator. Stages 5–8 are the depth that turns it from a
portfolio piece into an argument.

Conventions, invariants and the DDL live in `.claude/skills/lsm-integrity/`. Read
`SKILL.md` before starting any stage, and `references/data-contract.md` before touching any
schema, dtype or key. The system-level design — environments, serving contract, delayed
labels, SLOs, runbook — is `docs/production-architecture.md`, and it states explicitly what
is **built** here versus **designed** here.

**Settled stack:** LightGBM for all models; **Dagster** for orchestration; MLflow 3 for
tracking and the registry (**aliases, not the removed stages**); SQLite for metadata +
Parquet for bulk signal; S3 for artifacts; GitHub Actions for CI/CD; Streamlit on Hugging
Face Spaces for the demo. Training env: `ml_gpu` (Python 3.12.13, dagster 1.13.15,
mlflow 3.13.0, lightgbm 4.6.0).

**Design centre:** the primary workload is **backfill over an existing archive**, not
steady-state scoring. Anywhere the plan looks over-engineered for 12,000 rows, that is why.

**Effort.** The stage estimates below are focused-work days and are optimistic by roughly
2–2.5× once tests and fixtures are counted. Budget ~20 days for the whole plan, not 11.
Use the cut lines.

---

## Two corrections baked into this plan

**1. There is no gradiometry in the current data.** The brief says "compute 3-sensor
gradients — common-mode rejection". The generator has *one* 3-axis magnetometer, so what is
available today is the **along-track derivative** dB/ds. Stage 2 adds a second sensor head
at a vertical baseline, which makes the common-mode-rejection claim literally true and gives
a measurable answer to "how much did the gradiometer buy you?". Until then, the correct
phrase is "along-track gradient".

**2. Row-level severity regression is a trap.** `severity_smys` is 0 outside the ±2 m label
box and constant inside it. A row-level regressor learns "predict 0" and reports a
flattering MAE. Stage 3 therefore uses a **two-stage architecture**: row-level scoring →
peak clustering into *indications* → per-indication severity, class and risk. This is also
how real ILI/LSM analysis works, so the fix and the realism are the same decision.

A third, smaller issue: 12 defects on one 2 km line is too few for a stable supervised
estimate. Stage 6 fixes it by generating many lines; until then, every supervised number
carries a wide confidence interval and should be reported as provisional — literally, as a
bootstrap CI, not as a caveat in prose.

**Two data-contract decisions are structural and belong in Stage 0/1, not bolted on later:**

- **The physical key is the integer `sample_idx`, not the float `chainage_m`.** `0.5 * 3`
  is not `1.5` in binary, so a float join key silently drops rows between tables computed
  in different code paths. `chainage_m` is derived.
- **Precision is chosen from the physics.** `lat`/`lon` must be float64 — float32 ULP at
  47° N is ≈ 0.42 m, comparable to the 0.5 m sample spacing, so float32 GPS would quantise
  the survey. Field values are float32 in storage (ULP ≈ 0.004 nT at 45 000 nT, below sensor
  resolution) but **float64 in arithmetic**, because detrend fits and window sums accumulate
  error.

**And three system-level decisions that are also structural:**

- **`pipeline_version` is the deployable unit, not `model_version`.** An indication is
  produced by up to four models plus a feature version; one model column cannot express its
  lineage, and the four must be promoted and rolled back together.
- **Config is layered** — code config (hashed) / environment config (dev, prod) / runtime
  params. One flat `config.yaml` means `config_sha256` changes when you change environment,
  which silently voids the reproducibility guarantee.
- **Point-in-time correctness.** Every feature for a survey must be computable from data
  available at that survey's `surveyed_at`. Spatial grouping does not catch temporal
  leakage, and Stage 8 leaks without this rule.

---

## Stage 0 — Skeleton and reproducibility  *(~half a day)*

Make the repo runnable by someone else, on one command, before there is anything to run.

- `uv init`, `pyproject.toml`, console script `lsm` → Typer CLI with all nine subcommands
  stubbed and `--help` accurate.
- **Layered config**: `config/base.yaml` (code config — the only layer hashed into
  `config_sha256`), `config/{dev,prod}.yaml` (bucket, DB URI, MLflow URI, concurrency),
  runtime params on the CLI. `config.py` merges them and hashes only the code layer.
- **Repo hygiene from day one**, because retrofitting it is miserable: committed `uv.lock`,
  `pre-commit` (ruff + ruff-format + mypy), `pip-audit` in CI. Pin **one** Python version
  across the training env, the local venv and the app runtime — they are 3.12 / 3.13 / TBD
  today, which is three dependency graphs for one bundle.
- **`src/lsm/schemas.py` — the data contract, declared once** (pandera/pyarrow): columns,
  dtypes, units, closed enums, null policy, `schema_version`. Imported by `validate.py`,
  `features.py` and `predict.py`. A boundary that does not validate is a boundary where
  skew appears.
- **`src/lsm/hashing.py`** — `file_sha256` (raw bytes, provenance) vs canonical
  `content_sha256` (normalised content, identity/dedup), plus the Merkle `data_sha256` over
  a training corpus. BLAKE3/xxh3 for the bulk pass.
- `src/lsm/db.py`: SQLite DDL from `references/architecture.md`, WAL mode, foreign keys on.
- LightGBM determinism defaults wired in from the start (`deterministic=True`,
  `force_row_wise=True`, fixed `num_threads`) — retrofitting this after the determinism
  test starts flaking wastes an afternoon.
- `.gitignore` (`data/`, `mlruns/`, `.venv/`), `README.md` with the honest framing verbatim.
- `git init` — the repo is not currently under version control, and `git_sha` provenance
  depends on it.

**Gate:** `lsm --help` lists every step; `pytest` runs (zero tests is fine); fresh clone +
`uv sync` works; the same config + seed produces byte-identical model output twice; the
same code config hashes identically under `dev` and `prod`.

---

## Stage 1 — Ingest and validation  *(~1 day)* — do this first and make it visible

The stage that signals real experience. It comes before any modelling.

- Move `generate_lsm_data.py` → `src/lsm/generate.py`; write **Parquet per survey** (zstd,
  sorted by `sample_idx`) into `data/raw/line_id=…/run_id=…/`, with both hashes recorded.
- `lsm ingest`: raw → SQLite `survey` / `reading` / `truth_defect` / `truth_observation`.
  **Idempotent on `content_sha256`**: same hash is a no-op, a changed hash for an existing
  `(line_id, run_id)` is an error, not an overwrite. Raw data is immutable. Ground truth is
  **slowly-changing** — corrections insert a `revision`, never update in place, so a label
  fix cannot retroactively rewrite last quarter's metrics.
- `lsm validate`: the fourteen checks in `references/validation-and-trust.md` — schema
  (against `schemas.py`), range, saturation, `sample_idx` monotonicity/gaps/duplicates,
  **exact-duplicate `content_sha256`**, **partial survey overlap** (interval intersection
  plus residual cross-correlation — neither hash catches this, and it is a leakage risk,
  not just untidiness), GPS jump, GPS-vs-chainage consistency, noise floor, background
  regime, interference density, coverage.
- **Quarantine, not crash.** A hard failure writes the survey to `data/quarantine/` with its
  DQ report and marks `status='quarantined'`. `validate` exits non-zero for CI, but the
  production path raises a ticket and continues to the next survey. A pipeline that halts
  the nightly run over one stuck channel gets switched off by its operators.
- **Edge/NaN policy** decided here, not later: rolling windows produce NaN in the first and
  last *w*/2 metres; keep the rows, mark `dq_flag='edge'`, exclude from metrics, score with
  a widened interval. NaN in a *raw* field stays a hard fail.
- `tests/test_validate.py`: for each check, one clean survey and one deliberately corrupted
  survey. Plus `tests/golden/` — a frozen survey with expected outputs, so later refactors
  are provably behaviour-preserving. This is the test suite that gets read in an interview.

**Gate:** a corrupted survey (dropped rows, GPS teleport, stuck channel, re-exported
duplicate, overlapping re-run) is caught with the right check name and the right status,
and lands in quarantine. Clean data passes with zero `fail`.

**Answers:** "how do you know the data is fit to use?", "what happens when it isn't?",
"how do you handle duplicates and re-runs?", "what breaks in production?"

---

## Stage 1.5 — Orchestration and backfill  *(~1 day)* — the biggest single gap to close

Eight CLI subcommands run in order by a human is a *script*, not a pipeline. The primary
workload here is backfill over an archive, and backfill is exactly what needs a task graph.

- **Dagster asset graph** over `ingested_survey → dq_report → survey_features → indications
  → gis_export`, **partitioned by `survey_id`** (dynamic partitions). The CLI steps stay as
  the implementation; the assets are the interface.
- `feature_version` maps onto asset **code versions**, so a bump re-materialises exactly the
  affected partitions. That correspondence is why Dagster over Airflow here — say so.
- **Retries: 3× exponential backoff on IO/transport, zero on DQ failure.** A bad survey is
  not a flake, and retrying it three times just delays the ticket.
- **One real backfill**, run and timed: a failure at survey N costs one partition, not the
  run. Demonstrate resumability by killing it halfway and restarting.
- **Concurrency, stated honestly**: feature computation fans out across 16 cores; SQLite WAL
  is single-writer, so registry writes serialise through one Dagster resource. Do not write
  a parallel ingest against SQLite and expect it to work.
- **Structured JSON logging** correlated by `survey_id` / `run_id` / `pipeline_version`,
  plus per-asset duration, rows/sec, failure count and quarantine rate. Cheapest
  observability win available and it makes the DAG debuggable.

**Gate:** a 120-partition backfill runs, is killed mid-flight, and resumes without
reprocessing completed partitions or duplicating rows. One deliberately corrupt survey
quarantines itself while the other 119 succeed.

**Answers:** "how would you process our archive?", "what happens when survey 4,000 of 6,000
fails at 3am?"

---

## Stage 2 — Features, and the gradiometer upgrade  *(~1 day)*

- Extend `generate.py` with `gradiometer.enabled` — a second sensor head at
  `baseline_m` above the first, filling `bx2/by2/bz2`. Same dipole model, different
  observation height.
- `features.py` with the **stateless / fitted** split enforced in the type signatures, not
  in comments (see `references/architecture.md`): per-survey detrend, residual magnitude and
  orientation, along-track dr/ds and d²r/ds², vertical gradient when available,
  sliding-window stats over {2, 5, 10, 25} m, peak-shape features (FWHM, asymmetry, decay
  exponent) and stand-off normalisation by depth³.
- Feature store → `data/features/fv=<feature_version>/`, cache-keyed on
  `content_sha256 + feature_version`. **`feature_version` is introduced here and is the
  version people forget:** edit `features.py` and every stored feature is silently stale —
  training/serving skew that no data check can see, because the data is fine. Bundles pin
  it and refuse to load on a mismatch.
- **Point-in-time correctness, enforced here.** Every feature for a survey must be
  computable from data available at that survey's `surveyed_at` — no normalisation across
  surveys, no defect location borrowed from a later run, no population statistic fitted on
  the full set. A grouped *spatial* split does not catch temporal leakage, and Stage 8 leaks
  without this. Enforce with an as-of join and a test that fails if a feature function is
  handed a frame containing a later `surveyed_at`.
- `tests/test_features.py`: detrend on a synthetic pure-drift trace returns ~zero residual;
  a known dipole at known depth produces the expected peak width; the fitted transforms
  refuse to be re-fit at inference. Add **property-based tests** (Hypothesis): detrending a
  pure polynomial of degree ≤ 3 must return ~zero residual for *any* coefficients — this
  finds the edge cases hand-written fixtures miss.
- One figure: raw field, detrended residual, along-track gradient, vertical gradient, with
  true defect and interference positions marked. **Quantify the rejection the second sensor
  buys** — that number is the payoff of the whole stage.

**Gate:** defect residual ≈ 25 nT vs ≈ 4 nT background, reproduced from the data, not
quoted from the brief.

**Answers:** "how do you separate the signal from a 45 000 nT background?", the gradiometry
question, and "which features actually carry the defect information?".

---

## Stage 2.5 — Data fidelity  *(all three items done 2026-07-29)*

Added 2026-07-29 after Stage 2 measured the generator and found three ways it is kinder
than reality. Full measurements and the exact commands to reproduce them are in
`.claude/skills/lsm-integrity/` context, `scripts/plot_features.py LINE000_R0 --no-plot` and
`scripts/check_observatory_background.py`. **All three items landed 2026-07-29** (66/66
tests).

**1. Give the second sensor head its own background — done.** `generate.py` now adds a
`main_field_gradient_nT_per_m` term (0.02, ~constant) plus a spatially-varying
`geology_gradient_scale_nT_per_m` term (10.0, its own independent drift+wave-shaped draw)
scaled by `gradiometer.baseline_m`, so B2 is no longer built from the exact same arrays as B.
Re-measured: head-difference std is now 6.98–8.37 nT (was 6.91–7.02 = σ√2 exactly), i.e.
~2.4 nT of real background gradient on top of doubled sensor noise. Detection contrast
(detrended single head vs vertical gradiometer) is now 3.21× vs 1.45× — still worse for the
gradiometer here, same qualitative conclusion as before, but no longer an artifact of a
bit-identical background. `scripts/plot_features.py`'s printed caveat was rewritten to
report the real vs sensor-noise split instead of claiming perfect-by-construction
cancellation.

**2. Derive the label extent from the physics instead of a constant — done.** Half-width is
now `label_window_scale * r_eff`, where `r_eff = sqrt(depth_m^2 + y_off_m^2)` is the actual
dipole source-to-sensor distance — on-pipe defects (`y_off_m=0`) get `r_eff=depth_m`,
interference (farther + lateral) automatically gets a wider box. `label_window_scale`
calibrated by sweeping against the real generated data (not by construction) until
unlabelled clearly-anomalous background rows (`|r-median| > 4·MAD`, the same robust
threshold `features.py`'s own peak detector uses) dropped to the 0–2 per survey attributable
to pure sensor noise (confirmed: 12–99 m from the nearest labelled feature), landing at
`label_window_scale = 2.0`. A tighter scale that exactly matched measured FWHM was tried
first and made things *worse* (dozens of unlabelled rows per survey) because FWHM is a
half-max width and the 1/r³ tail stays clearly anomalous well past it — matching FWHM is the
wrong physics to calibrate against for this gate. Side effect caught during calibration: at
`scale=2.0` some defect/interference windows now overlap, so labelling was changed from
independent per-feature masks to nearest-feature assignment (each row goes to whichever
feature centre it is closest to) to avoid double-labelling a row as both. Re-measured Stage 2
gate: defect residual is now 25.05 nT / 0.051% of raw field — matches the project's own
stated physics claim ("~25 nT against a ~4 nT floor") almost exactly, better than the old
constant's diluted number.

**3. Promote the real observatory background from stretch to a stage — done 2026-07-29.**
The generator's synthetic background (drift + one sinusoid) let a degree-5 polynomial reach
the 5.0 nT sensor floor exactly. `generate.py` can now load a real geomagnetic observatory
trace instead: `data/reference/*.csv`, fetched live from the USGS Geomagnetism Program's
public API (`https://geomag.usgs.gov/ws/data/`, station BOU/Boulder CO, 1 Hz, public domain).
Two traces are checked in — a geomagnetically **quiet** day (2024-03-15) and the **G5
"Mother's Day" storm** (2024-05-10, one of the strongest storms in two decades). New
`data.observatory_background` config block (`enabled`, `csv_path`, `survey_speed_m_per_s`);
`_load_observatory_background()` resamples the trace's native 1 Hz cadence onto chainage
assuming a constant survey speed (default 1 m/s) — a stated modelling simplification, since
what the gate needs is the trace's genuine broadband shape, not a real speed match. **Off by
default** — see the gate result below for why.

**Result, measured by `scripts/check_observatory_background.py` (three full generate→features
runs, same code path the pipeline uses):**
- **Raw claim confirmed:** a bare degree-5 polynomial fit to the storm trace leaves 6.9–11.3 nT
  residual (per axis) — clearly above the 5.0 nT noise floor, unlike the synthetic sinusoid's
  5.03 nT. `tests/test_features.py::test_observatory_background_is_not_polynomial_like_the_synthetic_one`
  pins this.
- **The actual pipeline detrend is NOT broken by it.** `features.detrend_axis` is two-stage —
  robust polynomial *then* a 40 m rolling-median high-pass, explicitly to remove "whatever the
  polynomial cannot fit" (see its docstring, written before this test existed). That high-pass
  is non-parametric, so it doesn't care whether the residual structure is a sinusoid or a real
  storm: background residual after the full pipeline is 7.81 nT (synthetic) vs 8.09 nT (storm)
  vs 7.83 nT (quiet) — barely different — and the Stage 2 detectability gate holds at 3.02–3.21×
  contrast in all three cases (gate is >3.0×). Pinned by
  `test_stage2_gate_holds_with_a_real_observatory_background`.
- **Why it's off by default anyway:** the *point* of item 3 was to stress-test the detrend
  design, not to quietly swap the operating background — and the honest headline is a good
  one: the two-stage design was already robust to a real G5 storm, which is stronger evidence
  for the design than only ever having tested it against a sinusoid it was tuned next to.

**Gate — met.** Items 1–2: the gradiometer comparison is re-run and re-stated on a background
with a real vertical gradient; unlabelled clearly-anomalous rows are down to a handful of
confirmed noise outliers, not a window-sizing artifact. Item 3: a bare cubic/degree-5
polynomial no longer reaches the noise floor on a real observatory trace, and the residual is
reported — and, going further than the gate asked, the actual pipeline detrend is confirmed to
still hold up against it.

**Answers:** "how realistic is your synthetic data, and where does it flatter you?" — which
is the question this whole section exists to have a worked answer for.

---

## Stage 3 — Thin end-to-end slice: detection → indications → map  *(built 2026-07-29)*

The smallest thing that tells the whole story.

**Code built by Opus 5; the real training run is deliberately left for Ali to execute by
hand** (his call — see the run guide below). Verified here only via `tiny_cfg` (200 m, 1
line, 1 run) smoke tests, which confirm the whole `generate → ingest → features → train →
predict` path wires together and writes what it claims to — **not** a statistically
meaningful gate verdict. 109/109 tests pass; ruff clean.

- `src/lsm/models/anomaly.py`: `MADBaseline` (robust z-score on `r_mag_nt`) and
  `IsolationForestAnomalyModel` (sklearn, over the full residual+shape feature set, NaN
  shape features filled 0.0). Both scored so **higher = more anomalous**; IsolationForest
  uses `decision_function` (not `score_samples`) so threshold=0 is already
  contamination-calibrated, and `calibrated_threshold()` gives MAD the same nominal
  flagged-fraction on its own train fold — a fair comparison, not two differently-tuned
  sensitivities.
- `src/lsm/truth.py`: a truth-defect/interference **registry derived from the raw label
  columns** (`defect`, `defect_type`, `interference`), not from the SQLite `truth_defect`/
  `truth_observation` tables — those still aren't populated by `ingest` (a known gap, see
  memory). Contiguous same-label chainage runs in one reference run of a line become one
  physical source with a stable `source_id`; position is fixed across a line's 3 runs, so
  any one run's labels describe the whole line.
- `src/lsm/indications.py`: `cluster_indications()` (contiguous supra-threshold rows -> one
  peak + extent, `dq_flag` carried through from the per-row feature flag, deterministic
  `indication_id` so re-running `predict` is a no-op), `select_dig_budget()` (top-k by
  score — the budget is a serving-time ranking choice, so the DB table holds every
  indication above threshold, not just the dug slice), `write_indications()`.
- `src/lsm/evaluate.py`: `assign_group`/`assign_fold` — **sha256, not blake3** (this
  project already decided "stdlib sha256 throughout, no blake3/xxhash needed" in
  `hashing.py`; same property either way). Recall is bootstrapped over the **~12 physical
  defects** (not the ~36 defect×run observations — the 3 runs re-observe the same defects);
  false-dig-rate and interference-dig-fraction over the **3 runs**; localisation error over
  the pooled matched pairs. `paired_bootstrap_ci()` resamples both models' arrays with the
  *same* indices, which is what the gate actually needs (two separately-bootstrapped CIs
  can both be wide and overlapping even when the paired difference is consistently
  positive). `tests/test_leakage.py` passes (4 tests): no `(line_id, block)` group crosses
  folds, and the same block across a line's 3 runs always lands in the same fold.
- `src/lsm/train.py`: runs the grouped CV for both models, logs to MLflow (experiment from
  config, all three hashes, the DQ report as a JSON artifact, an overlaid PR curve, metrics
  with CIs), refits both on the **full** corpus (CV models exist only for an honest
  out-of-fold evaluation), saves an **ad hoc joblib artifact** to `data/models/<version>/
  bundle.joblib` — deliberately **not** the full `bundle.py` contract (pinned
  `defect_type` category order, calibration map, conformal quantiles, hard
  library-version-mismatch check): that is explicitly Stage 4 scope, not duplicated here.
  Writes `model_run` rows for both models and one `pipeline_release` (alias=`challenger`,
  `anomaly_version` = the IsolationForest run) referencing the isolation-forest artifact,
  then persists that model's indications for every survey in the corpus. A real bug was
  caught and fixed here: the first version persisted indications from the CV **out-of-fold**
  scores while `predict.py` reloads and re-scores with the **full-corpus-fit** model — two
  different scoring passes that can legitimately disagree. Fixed to score with the shipped
  model both times, honouring architecture.md's stated bundle invariant ("`predict()` on the
  training set... reproduces the in-training predictions bit-for-bit").
- `src/lsm/predict.py`: loads the latest `pipeline_release` (there is no `@champion` yet in
  this demo, only ever `@challenger` — promotion is explicitly Stage 7 CI scope, so "most
  recent by `released_at`" is the honest stand-in), reloads its bundle, scores one already-
  featurised survey, writes indications. `indications.geojson` export is Stage 4 scope.
- `app/streamlit_app.py`: a genuinely thin app (not the polished Stage 4.5 demo app) — GPS
  track (pydeck `PathLayer`), true defects and interference shown separately (orange/grey
  `ScatterplotLayer`s), detected indications overlaid (blue), ranked table below the map.
  Verified: the server starts and serves cleanly (health check OK) and the data/pydeck logic
  was exercised directly against the real (untrained) project DB without error; the
  "indications present" branch is exercised by the `test_predict.py` suite, not by a live
  browser session (no browser automation available in this environment) — Ali should give
  it a look once he's trained a real model.
- **A real, unrelated environment bug found and fixed along the way**: mlflow 3.13's
  documented dev backend (`file:./mlruns`) is now in **maintenance mode** and raises on
  init — confirmed empirically, not assumed from docs. `config/dev.yaml` and
  `references/architecture.md` were updated to `sqlite:///mlruns.db`, the minimal supported
  local backend; artifacts still land on local disk under `./mlruns/`.

**Gate:** IsolationForest beats the MAD baseline by ≥0.15 recall @ budget, **compared on
intervals rather than point estimates**, *and* the gap is attributable to interference
rejection, shown explicitly. If it is not, say so — a negative result reported honestly is
stronger than a tuned one. **Not yet evaluated for real — run `lsm train` on the full
2000 m / 3-run dataset to get the actual verdict** (`train.py` prints it directly: recall
gap with CI, gate pass/fail, and whether the gap is attributable to interference rejection).

**Answers:** "why not just threshold?", "what does the output look like to an engineer?"

---

## Stage 4 — Severity with calibrated uncertainty  *(built 2026-07-30)*

**Code built by Opus 5; the real training run is Ali's, same division of labour as Stage
3.** Verified via a deliberately larger-than-`tiny_cfg` fixture (500 m, 3 runs, 10 defects
— `tiny_cfg` alone never produces enough matched indications to exercise this path at all)
so the actual fit/predict/bundle-round-trip code runs for real, not just its "too little
data, skip" branch. 134/134 tests pass; ruff clean.

- Per-**indication** severity regression: LightGBM quantile (0.05 / 0.5 / 0.95) wrapped in
  **split conformal** (CQR) for a distribution-free 90% interval. Baseline: global mean
  severity. Feature vector = the indication's peak row's own residual/shape features (least
  noisy estimate) + `extent_m` (`indications.attach_indication_features` — SKILL invariant
  #2: severity is never regressed row-wise).
- **Training/eval universe is the anomaly detector's own out-of-fold indications** (all of
  them, not just the dig-budget-selected slice — severity applies to whatever got flagged,
  budget is a serving-time ranking choice), matched to the truth registry, kept only where
  `matched_kind == 'defect'`. This makes severity's honesty conditional on Stage 3's
  detector actually flagging real defects — worth restating if Stage 3's gate result changes.
- Grouped CV reuses Stage 3's exact fold assignment; each fold's train/calibration split is
  **by physical defect**, never by row, so conformal's finite-sample guarantee isn't violated
  by the same defect straddling both halves. With ~12 defects this is a genuinely tiny split
  (documented in code, not hidden) — `min_child_samples=3`, `n_estimators=50` are a stated
  small-sample accommodation, not a production default; Stage 6's scale rehearsal would need
  to revisit them.
- Coverage, MAE, mean interval width — all bootstrapped over **physical defects**, the same
  unit as Stage 3's recall (`evaluate.per_group_severity_metrics`); MAE by severity decile
  reported separately (a model accurate only on benign defects is useless).
- `bundle.py` — the real versioned artifact contract, replacing Stage 3's ad hoc joblib dict
  (used for anomaly bundles too now): ordered feature list, pinned `defect_type` category
  order (prepared for Stage 5, harmless before it), `feature_version`, `schema_version`, all
  three hashes, `truth_as_of`, library versions, training-set feature summary (the Stage 8
  drift reference). **`load_bundle()` hard-fails** on a `feature_version`/`schema_version`
  mismatch or a numpy/sklearn/lightgbm version mismatch — confirmed by
  `tests/test_bundle_roundtrip.py`, including a test that tampers a stored library version
  to prove it's enforced, not just recorded.
- `pipeline_release` now carries **both** `anomaly_version` and `severity_version` in one row
  (`classify_version`/`growth_version` stay NULL until Stage 5/8).
- Auto-generated **model card** (`model_card.py`) per pipeline release: intended use,
  training corpus, Stage 3 + Stage 4 metrics with intervals, known limitations, explicit
  synthetic-data statement.
- `lsm predict`: now loads both released bundles, fills `sev_pred`/`sev_lo`/`sev_hi`/
  `interval_nominal` on every indication (NULL if no severity model exists yet — not an
  error), and writes `indications.geojson` next to the DB rows.

**A real bug caught while sizing the test fixture, not by inspection**: `defect_hit_rates`'s
multi-line bug (see Stage 3 entry) had a severity-side twin risk that turned out NOT to bite
here, but forced writing `_severity_sized_cfg` in the first place — `tiny_cfg` silently
produces zero severity training data and the code path that matters was never exercised
until a bigger fixture forced it to run. Worth remembering for any future "add a model"
work: a smoke test that never triggers the interesting branch is not a passing test, it's an
untested one that happens not to fail.

**An unrelated environment issue found while running the bigger fixture**: `matplotlib`
picked an interactive Tk backend and crashed with no display available. Training scripts
never need a GUI backend; `train.py` now calls `matplotlib.use("Agg")` before importing
`pyplot`.

**A real correctness bug caught from Ali's actual first run, not from my own tests: split
conformal used a plain `np.quantile(scores, 1 - alpha)` for the calibration margin instead
of the finite-sample-corrected level (Romano et al. 2019: `ceil((n+1)(1-alpha))/n`).** The
naive quantile systematically undershoots the required margin once the calibration set is
small — exactly this project's regime (a handful of rows per fold, ~12 defects total). First
real run measured 72.7% empirical coverage against a 90% nominal target; after the fix, both
`SeverityModel` and `GlobalMeanSeverityBaseline` use a shared `conformal_margin()` helper
with the correction, pinned by 4 new unit tests (including one proving the naive quantile
would give a materially different, wrong answer on a 9-point calibration set). **Ali needs
to re-run `lsm train` to get corrected numbers** — the coverage/MAE/width values from the
first run are understated relative to what the (now-fixed) calibration actually produces.

**Gate:** coverage ∈ [0.87, 0.93] on the grouped holdout. Bundle round-trip exact; a
`feature_version` mismatch raises. First real run (pre-fix): LightGBM CQR beat the
global-mean baseline on every metric (MAE 8.91 vs 13.87, interval width 32.7 vs 35.8,
coverage 0.727 vs 0.682) but did not hit the coverage gate — expected to improve after the
conformal fix above; re-run for the real number. As with Stage 3, treat the severity numbers
as a methodology check first given Stage 3's own gate did not pass.

**Answers:** "how much should I trust this number before I authorise a dig?"

---

## Stage 4.5 — The demo app *(a consumer, not the serving layer)*  *(~1 day)*

**The serving layer is the batch scoring job** — the `indications` asset writing to the
`indication` table and `indications.geojson`, with a versioned output schema, a consumer
contract test in CI, idempotency on `(survey_id, pipeline_version)`, and a < 24 h freshness
SLO. That already exists from Stage 1.5. The Streamlit app **consumes** that output. Say it
that way in the interview — calling a Streamlit app "the serving layer" is the kind of thing
an MLOps interviewer probes.

Inference only, CPU only, and deliberately hard to break in a room full of people.

- Extract a **serving artifact** the app can read without importing training code:
  `bundle.joblib` (pulled from S3 by pinned `pipeline_version`, baked-in fallback copy),
  3–4 pre-baked demo surveys including one deliberately corrupted, precomputed indications,
  and the model card.
- `APP_MODE=demo` serves precomputed results (zero compute, cannot fail); `APP_MODE=live`
  runs the real `validate → features → score` path (~1 s). Same code, so the impressive
  path and the safe path are the same path.
- Deploy to **Hugging Face Spaces, Streamlit SDK** (~16 GB RAM removes OOM risk and lets
  the Stage-6 dataset be shown; the SDK avoids Docker's slow cold start), pushed by the
  GitHub Action **after** promotion gates — not auto-deployed from a branch, which would
  undermine the gated-release story.
- iPad is a thin client: pre-baked scenario **buttons** not file upload, `segmented_control`
  not sliders, `layout="wide"` with nothing essential behind the sidebar. Never push on
  demo day; wake the Space 5 minutes ahead. Fallback ladder: cloud URL → laptop
  `streamlit run` → recorded GIF.
- Footer strip showing `pipeline_version` / `feature_version` / `config_sha` / `git_sha`, so
  provenance is visible throughout.

**Build the app around four beats:** (1) raw 45 000 nT signal — the defect is invisible;
(2) detrend + gradient — defects appear, and so do four interference sources;
(3) ranked indications on the map with intervals and a dig-budget control;
(4) **load the corrupted survey and watch the app refuse to score it**, naming the failed
check. Beat 4 is the one nobody else demos, and it is aimed straight at "validation and
trust".

**Gate:** the whole demo runs end to end on an iPad over cellular, with the laptop closed.

---

## Stage 5 — Classification and risk ranking  *(~1 day)*

- LightGBM multiclass over indications: scc / weld / dent / corrosion / **interference** as
  an explicit class. Isotonic calibration, reliability diagram, Brier score.
- Report **per-class recall with SCC protected (≥0.90)** and interference precision.
- `risk_score` = calibrated P(defect) × predicted severity × proxy consequence, with the
  interval carried through so risk is a range. Ranked heat map in the app.
- Global + per-indication SHAP, and the **physics consistency check**: the model should key
  on residual amplitude, gradient and peak width. If it keys on absolute chainage, that is a
  shortcut, and finding it is the point.

**Gate:** SCC recall ≥ 0.90; no leaky feature in the top-10 SHAP list (denylist test).

**Answers:** "how do you avoid false digs?", "can you explain a prediction?"

---

## Stage 6 — Scale rehearsal  *(~1 day)* — the answer to "millions of rows"

The single most valuable stage for their stated situation, and the one most people skip.

- `data.n_lines: 40` → ~10⁷ rows. Chunked generation and ingest; measure wall-clock and
  peak RSS at each step on the 16-core / 128 GB box.
- Switch the bulk read path to Parquet + DuckDB, keeping SQLite for the small high-value
  tables (survey registry, truth, DQ, indications, model runs). Same CLI, same code.
- This is where the float32 feature decision pays: ~30 features × 10⁷ rows is 1.2 GB in
  float32 versus 2.4 GB in float64. Verify the arithmetic still runs in float64.
- Also where the **small-files problem** shows up: 120 tiny per-survey Parquet files is
  fine, 10⁵ is not. Note the production answer (periodic compaction to line-level files)
  and whether you hit it here.
- Re-run Stages 3–5 with `GroupKFold` on **whole held-out lines** — the real generalisation
  test — and add the **temporal holdout** (train surveys 0–1, test survey 2).
- Write down where SQLite stopped being the right tool, with numbers.

**Gate:** full pipeline runs on ~10⁷ rows within memory budget; a documented
rows/sec and peak-RSS table for each step.

**Answers:** "does this scale to our archive?", "how do you generalise to a new pipeline?"

---

## Stage 7 — CI/CD and the release gate  *(~1 day)*

- `ci.yml`: ruff + mypy + pip-audit + pytest + a tiny-config smoke run on every push. Add an
  **integration test against containerised MinIO** — `moto` mocks S3's API, not its
  behaviour, and the difference is where multipart uploads and consistency bugs live.
- `train.yml`: dispatch/tag → full train, MLflow logging, evaluate the six promotion gates
  in `references/validation-and-trust.md`, cut a `pipeline_release`, set the **`@challenger`
  alias**, push the bundle to S3, post the metrics table to the run summary.
- **MLflow 3 removed model stages** — promotion moves the **`@champion` alias**, which makes
  it an atomic pointer swap rather than a state mutation. Anything calling
  `transition_model_version_stage` is on a dead API.
- **Shadow scoring.** `@challenger` scores every survey alongside `@champion`, written with
  `is_shadow=1`. Because ground truth is months away, the near-term comparison is a
  **disagreement report**: change in indications/km, rank churn at the dig budget, and
  whether disagreements concentrate in DQ-warned surveys. This is the only pre-label
  validation available in this domain, and it works from day one.
- `deploy.yml`: on promotion, push the serving artifact to the HF Space pinned to a
  `pipeline_version`. Rollback = repoint `@champion`; bundles are immutable.
- **Rollback drill in CI.** Untested rollback is not rollback. The usual failure is that the
  *old* bundle no longer loads against current code — which our own `feature_version` guard
  would cause. Policy: the last **N = 3** feature versions stay loadable, and CI proves it by
  actually rolling back and scoring.
- AWS via **GitHub OIDC**, no long-lived keys; HF via a repo secret. `moto` mocks S3 in
  tests so the repo runs with no cloud account (`storage.enabled: false` locally). No
  credentials of any kind in the public Space — the bundle is fetched via a read-only
  policy or a presigned URL.

- **Final-test-set discipline.** The promotion gate compares candidates against the
  incumbent on a holdout; do that fifty times and you have selected on holdout noise — the
  multiple-comparisons problem applied to the *release process*. Keep a final test set
  touched rarely, count its uses in `model_run`, and rotate as new lines arrive.

**Gate:** a deliberately degraded model (metric interval below the incumbent, or coverage
out of band) is **blocked by CI**; and a rollback to the previous `pipeline_version` scores
a survey successfully. Demonstrating the block and the rollback is the deliverable, not the
pass.

**Answers:** "how does a model get to production, what stops a bad one, and how do you
undo it?"

---

## Stage 8 — Growth, remaining life, and monitoring  *(~1 day)*

- Match indications across surveys by chainage (this is a small alignment problem in
  miniature — record the match residual).
- **Partially pooled** growth rate per defect, shrunk toward the population rate, because
  three points per defect cannot support twelve independent fits. Project to a limit state
  → remaining life, with the severity interval propagated so the output is a range.
- Baseline: "no growth". State plainly that with n=3 the method is right and the numbers are
  provisional.
- Drift monitoring: PSI/KS vs the bundle's stored reference, indications-per-km, background
  regime shift; a `lsm monitor` command and a panel in the app.
- **Vintage backtesting.** Every evaluation pins `truth_as_of` and reconstructs what was
  known on that date, then scores forward. `truth_as_of` exists in the schema from Stage 1;
  this is the stage that finally uses it. Reported metrics are always as-of a stated date.
- The **dig-feedback loop**: a verification writes `source='excavation'` truth rows and
  triggers re-measurement of empirical coverage. Track **median days from indication to
  verification** — it bounds how fast the system can learn anything, and it is the number
  to ask ROSEN for.
- All growth queries filter on `pipeline_version` — the `indication` table holds output from
  several releases, and comparing surveys scored by different models is a correctness bug,
  not untidiness.

**Gate:** beats the "no growth" baseline on held-out survey 2; drift monitor fires on a
survey with a deliberately shifted background.

**Answers:** "how do you support proactive maintenance?", "how do you know when the model
has gone stale?"

---

## Stretch, in value order

1. **Real geomagnetic background** from a BGS/NOAA observatory trace — lets you say the
   ambient field is real, and makes diurnal-variation removal a real problem.
2. **Chainage alignment / data fusion mock** — two sources with different chainage datums,
   aligned by cross-correlating girth welds, with the residual alignment error propagated
   into localisation uncertainty. This is the one closest to their actual pain.
3. **Active learning** — rank the next dig by *uncertainty × consequence*, and show the
   censoring bias that comes from only ever digging where you predicted.
4. **1-D CNN** on the residual window as a challenger to the boosted model — only once
   there are enough labelled indications for it to be a fair fight.

---

## Suggested cut lines

Real elapsed time, not the optimistic per-stage estimates:

- **~1 week:** Stages 0, 1, 1.5, 3. A validated, orchestrated, backfillable thin slice with
  a map. The orchestration is what makes it a *system* — do not drop 1.5 to save a day.
- **~2 weeks:** add 2, 4, 4.5. Gradiometry, calibrated uncertainty, and an app you can hand
  over on an iPad — which beats a better model they never see.
- **~3 weeks:** add 7 and 6. Release gates, shadow scoring, rollback drill, scale rehearsal.
- **~4 weeks:** add 5 and 8. Then the trust, scale and proactive-maintenance stories are all
  complete.

Do not start Stage 5 before Stage 1's tests are green. The validation layer is the part of
this project that is hardest to fake and most likely to be probed.

## What is deliberately not built

Named in `docs/production-architecture.md` §10 with a decision behind each, rather than left
as a gap: a staging environment with its own infrastructure, blue/green deployment,
Airflow/K8s at full scale, multi-year vintage backtests, a real GIS consumer contract, and
paging/on-call. A demonstrator that tries to build all of it ships nothing — but every one
of them should have an answer ready.
