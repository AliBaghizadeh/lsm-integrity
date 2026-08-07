# Architecture reference

## Repo layout

```
lsm-integrity-demo/
├── config.yaml                  # single source of truth, hashed into every run
├── pyproject.toml               # uv-managed; console script `lsm`
├── Makefile                     # thin wrappers over the CLI (optional on Windows)
├── data/                        # gitignored
│   ├── raw/                     # immutable parquet, one file per survey
│   ├── quarantine/              # surveys that failed a hard DQ gate, + their dq.json
│   ├── lsm.db                   # SQLite: surveys, readings, truth, DQ, indications
│   └── features/fv=<n>/         # parquet feature store, keyed by feature_version
├── tests/golden/                # frozen survey + expected outputs (refactor regression)
├── mlruns.db, mlruns/           # MLflow local backend (dev): sqlite tracking store + local artifacts
├── src/lsm/
│   ├── __main__.py              # Typer CLI: generate|ingest|validate|features|train|predict|forecast|serve
│   ├── config.py                # pydantic model of config.yaml + sha256
│   ├── schemas.py               # THE data contract: pandera/pyarrow schemas, enums, dtypes
│   ├── generate.py              # synthetic survey generator, Rig-v2: 3-head scalar rig,
│   │                             # human walker, GPS dropout, weld train (rig: vector
│   │                             # preserves the pre-Rig-v2 model as an ablation reference arm)
│   ├── db.py                    # SQLite connection, DDL, migrations, upserts
│   ├── hashing.py               # file_sha256 vs canonical content_sha256; Merkle data_sha256
│   ├── storage.py               # S3 put/get + local mirror; quarantine path
│   ├── validate.py              # DQ checks -> DQReport; hard-fail vs warn gates
│   ├── registration.py          # Stage B: GPS dead-reckoning through dropout + girth-weld-
│   │                             # comb detection -> registered chainage_m + dist_to_weld_m;
│   │                             # runs inside the features step (pipeline.py::run_feature_pipeline)
│   ├── features.py              # StatelessTransform vs FittedTransform (see below)
│   ├── indications.py           # peak clustering: row scores -> indication objects
│   ├── models/
│   │   ├── anomaly.py           # MAD baseline + IsolationForest
│   │   ├── severity.py          # LightGBM quantile + split conformal
│   │   ├── classify.py          # LightGBM multiclass + isotonic calibration
│   │   └── growth.py            # partially-pooled growth rate -> remaining life
│   ├── bundle.py                # save/load the versioned model bundle
│   ├── evaluate.py              # grouped CV, dig-budget metrics, coverage, drift refs
│   └── predict.py               # batch inference: validate -> features -> score -> DB
├── app/streamlit_app.py
├── tests/
│   ├── test_validate.py
│   ├── test_features.py
│   ├── test_leakage.py          # asserts no group crosses train/test
│   ├── test_determinism.py      # same seed + config -> identical metrics
│   └── test_bundle_roundtrip.py # train -> save -> load -> identical predictions
└── .github/workflows/{ci,train,deploy}.yml
```

## `config.yaml` contract

```yaml
seed: 42

data:
  n_lines: 1              # 1 for the showcase; 5+ for the CI-narrowing rehearsal
  length_m: 2000.0
  step_m: 0.5              # nominal/mean spacing metadata only under rig: scalar --
                            # actual row spacing is irregular by construction (see walk below)
  depth_m: 1.5              # DEFECT/WELD BURIAL DEPTH (dipole source depth, r_eff =
                             # hypot(depth_m, y_off_m)) -- NOT the sensor stand-off under
                             # rig: scalar, which is separately modelled and varies
                             # (walk.standoff_m below). The two happen to share a value in
                             # this config; they are not the same field. (rig: vector, the
                             # legacy path, DOES use depth_m directly as a fixed stand-off.)
  background_nT: [19000.0, 1000.0, 45000.0]
  noise_nT: 5.0
  n_defects: 12           # per line
  n_interference: 4
  interference_moment_scale: 50.0  # interference is a LARGE steel object; at 1.0 the
                                   # 1/r^3 falloff from 3-8 m lateral puts every source
                                   # below the noise floor and the trap traps nothing
  label_window_scale: 2.0  # half-width = scale * r_eff (r_eff = sqrt(depth_m^2 + y_off_m^2))
  n_runs: 3
  growth: 1.15
  rig: scalar               # Rig-v2 default: 3-head scalar total-field rig, human walker,
                             # GPS dropout. "vector" preserves the pre-Rig-v2 single-3-axis-
                             # head-on-a-rail model byte-for-byte, kept solely as the
                             # reference arm of the Stage D hardware-upgrade ablation -- not
                             # a second maintained product, and it does not produce
                             # RawReadingSchema-conformant output.
  stress_polarity: random   # a defect's total-field sign is whatever its isotropic moment
                             # orientation gives (stress can enhance OR degrade the field --
                             # honest default). "positive" reproduces the old, now-known-
                             # false always-enhancing assumption; only affects rig: scalar.
  array:                    # the physical rod: middle + two heads, 50 cm apart, vertical mast
    n_heads: 3               # fixed physical fact under rig: scalar (b_lo/b_mid/b_hi are
                              # hardcoded schema columns to match) -- not a free dial
    spacing_m: 0.5
    orientation: vertical     # "horizontal" raises NotImplementedError rather than
                               # silently running vertical-mast physics under a different
                               # label -- an open question for ROSEN, not guessed
    main_field_gradient_nT_per_m: 0.02   # without these, every head would see an identical
    geology_gradient_scale_nT_per_m: 10.0 # background and gradient rejection would be
                                           # perfect by construction, which is not real
  walk:                     # a human walker, not a cart on rails -- speed/stand-off/lateral
                             # position all wander as OU (mean-reverting) processes
    speed_m_per_s: 1.2
    sample_rate_hz: 120.0    # -> ~1 cm mean spacing at nominal speed; a deliberate
                              # consequence of the real rig, ~50x the old uniform 0.5 m
                              # grid's row count, not a demo convenience
    standoff_m: 1.5           # nominal rod height above ground/pipe -- THE actual sensor
                               # stand-off under rig: scalar (see depth_m note above)
    standoff_sigma_m: 0.20
    # ... speed/standoff/lateral/tilt each carry their own OU sigma + correlation
    # length (see WalkConfig) -- omitted here for brevity
  gps:                      # Markov good/bad lock state, not "always on"
    dropout_rate: 0.05        # steady-state FRACTION of survey time spent unlocked
    mean_gap_s: 15.0           # mean dwell time of one dropout (~18 m at nominal speed)
    sigma_m: 1.5                # horizontal noise while locked
  weld:                     # a periodic joint train, NOT one of the rare point-defect types
    pitch_m: 12.2              # girth welds every ~12 m -- a fact of how the pipe was
                                # built, not a rare event (pre-Rig-v2 wrongly drew `weld`
                                # ~12 TIMES TOTAL on a 2 km line via rng.choice)
    pitch_jitter_m: 0.15
    moment_scale_range: [5.0, 20.0]  # a joint's extra steel is 5-20x a defect's moment
  sensor:                   # per-head imperfections a real fluxgate head has
    gain_sigma: 0.002          # 0.2% per-head gain MISMATCH -- leaves ~100 nT of
                                # uncancelled common-mode against a ~25 nT anomaly; the
                                # single largest lever on gradiometric performance
    offset_nt: 2.0
    adc_bits: 24                # 24-bit over +/-100 uT -> LSB ~0.012 nT
    full_scale_ut: 100.0
    noise_nt: 5.0
  geo: {lat0: 46.958, lon0: 8.365, bearing_deg: 35.0}

validate:
  gates:                  # see validation-and-trust.md for the full list + definitions
    schema: fail
    range: fail
    saturation: fail
    sample_idx_monotonic: fail   # renamed from chainage_monotonic -- chainage_m is no
                                   # longer a raw column, so monotonicity is checked on
                                   # the physical key (sample_idx) instead
    sample_idx_gap: warn          # renamed from chainage_gap for the same reason
    gps_jump: warn
    gps_chainage_consistency: warn  # new: cross-checks GPS against chainage_provisional_m
    noise_floor: warn
  max_gap_m: 2.0
  max_gps_jump_m: 5.0
  field_range_nT: [20000.0, 80000.0]   # MAGNITUDE-only range: |B| is never negative, so
                                        # the old signed bound ([-80000, 80000]) was a
                                        # check.range gate that could never fail

features:
  version: 3             # feature_version: bump on ANY behaviour change in features.py
  detrend: {method: robust_poly, degree: 3, window_m: 40.0}   # poly, then rolling-median high-pass
  windows_m: [2.0, 5.0, 10.0, 25.0]
  peak: {prominence_mad: 4.0, flank_fit_span: [0.5, 3.0], assign_radius_fwhm: 2.0}
  edge_policy: flag      # flag|drop -- see data-contract.md §5

model:
  split: {by: line_id, n_folds: 5, fallback_block_m: 100.0}   # group key is (line_id, block)
  bootstrap: {n_resamples: 1000, level: 0.95}                 # CIs on every headline metric
  lightgbm: {deterministic: true, force_row_wise: true, num_threads: 16}
  anomaly:  {contamination: 0.03, n_estimators: 300}
  severity: {quantiles: [0.05, 0.5, 0.95], conformal_alpha: 0.10}
  classify: {calibration: isotonic, class_weight: balanced}
  dig_budget_per_km: 5

mlflow:
  tracking_uri: sqlite:///mlruns.db   # mlflow 3.13's file:./mlruns backend is in
                                       # maintenance mode and refuses to init
  experiment: lsm-integrity

storage:
  sqlite_path: data/lsm.db
  s3: {bucket: ${LSM_BUCKET}, prefix: lsm-demo, enabled: false}
```

`config.py` loads this into a pydantic model and exposes `config_sha256` over the
canonicalised YAML. Every MLflow run logs the file *and* the hash.

## SQLite schema

Dtypes, keys, units, nulls and hashing rules are specified in `data-contract.md` — read it
before touching this DDL. Two rules it imposes here: **the physical key is the integer
`sample_idx`, never the float `chainage_m`**, and there are **two hashes** with different
jobs (`file_sha256` for provenance, canonical `content_sha256` for identity/dedup).

```sql
CREATE TABLE survey (
  survey_id        TEXT PRIMARY KEY,     -- LINE003_R2
  line_id          TEXT NOT NULL,
  run_id           INTEGER NOT NULL,
  surveyed_at      TEXT NOT NULL,        -- ISO-8601 UTC, always
  step_m           REAL NOT NULL,
  n_samples        INTEGER NOT NULL,
  chainage_start_m REAL NOT NULL,        -- interval, for overlap detection
  chainage_end_m   REAL NOT NULL,
  standoff_m       REAL NOT NULL,
  schema_version   INTEGER NOT NULL,
  source_uri       TEXT NOT NULL,        -- s3:// or file:// of the raw parquet
  file_sha256      TEXT NOT NULL,        -- raw bytes: provenance / immutability
  content_sha256   TEXT NOT NULL,        -- canonicalised content: identity / dedup
  status           TEXT NOT NULL,        -- accepted|quarantined
  ingested_at      TEXT NOT NULL,
  UNIQUE (line_id, run_id),
  UNIQUE (content_sha256)                -- exact duplicates cannot be ingested twice
);

-- Rig-v2 (schema_version 3): bx/by/bz(+bx2/by2/bz2) are gone -- the real rig has
-- three total-field HEADS (b_lo/b_mid/b_hi), never a vector reading. lat/lon
-- are nullable now: GPS dropout is a real, expected acquisition state
-- (GpsConfig), not corrupt data. t_s, girth_weld and chainage_true_m are new.
-- chainage_true_m is TRUTH TIER, stored directly on `reading` the same way
-- severity_smys already is (nullable there, NOT NULL here since the generator
-- always knows it for every row). There is deliberately NO registered-chainage
-- column here: chainage_m is a FEATURE-layer output of Stage B registration
-- (registration.py — GPS dead-reckoning through dropout, locked to the
-- recovered girth-weld lattice), never written to raw. See data-contract.md
-- §1 for why `chainage_m = sample_idx * step_m` stopped being true.
CREATE TABLE reading (
  survey_id  TEXT    NOT NULL REFERENCES survey(survey_id),
  sample_idx INTEGER NOT NULL,           -- exact key; a dense TIME-sample counter
                                          -- under Rig-v2 (walk.sample_rate_hz), not
                                          -- a distance-grid index
  t_s REAL NOT NULL,                     -- elapsed seconds since survey start --
                                          -- the walk is time-sampled, not distance-sampled
  lat REAL, lon REAL,                    -- nullable: GPS drops out in poor sky view
                                          -- (GpsConfig.dropout_rate); the gap is NaN,
                                          -- not synthesised through
  b_lo_nt REAL NOT NULL, b_mid_nt REAL NOT NULL, b_hi_nt REAL NOT NULL,  -- total-field
                                          -- |B| per head, never x/y/z
  girth_weld INTEGER NOT NULL,           -- periodic joint train ground truth, tracked
                                          -- separately from `defect` -- a real, strong,
                                          -- non-cancelling source, but not damage
  chainage_true_m REAL NOT NULL,         -- TRUTH TIER: the generator's own exact
                                          -- along-track position; may score
                                          -- registration, never a feature
  PRIMARY KEY (survey_id, sample_idx)
) WITHOUT ROWID;

-- Ground truth lives apart from the signal: in production it arrives months later,
-- from ILI or an excavation report, and it is the scarce asset. It is also SLOWLY
-- CHANGING -- a dig can reclassify a defect -- so corrections insert a revision
-- rather than updating in place, and evaluations pin an as_of timestamp.
-- Stable identity is its own table. Observations attach to the DEFECT, not to a
-- revision of its metadata -- so the FK needs a single-column unique parent.
CREATE TABLE defect (
  defect_id TEXT PRIMARY KEY,
  line_id   TEXT NOT NULL
);

CREATE TABLE truth_defect (               -- slowly-changing attributes of a defect
  defect_id   TEXT NOT NULL REFERENCES defect(defect_id),
  revision    INTEGER NOT NULL,
  chainage_m  REAL NOT NULL,
  defect_type TEXT NOT NULL,             -- closed enum, see data-contract.md §4
  source      TEXT NOT NULL,             -- synthetic_truth|ili|excavation
  verified_at TEXT,
  valid_from  TEXT NOT NULL,
  valid_to    TEXT,                      -- NULL = current revision
  PRIMARY KEY (defect_id, revision)
);

CREATE TABLE truth_observation (          -- severity per (defect, survey) -> growth
  defect_id     TEXT NOT NULL REFERENCES defect(defect_id),
  survey_id     TEXT NOT NULL REFERENCES survey(survey_id),
  severity_smys REAL NOT NULL,
  PRIMARY KEY (defect_id, survey_id)
);

CREATE TABLE dq_report (
  dq_id      INTEGER PRIMARY KEY AUTOINCREMENT,
  survey_id  TEXT NOT NULL, checked_at TEXT NOT NULL,
  check_name TEXT NOT NULL,
  status     TEXT NOT NULL,               -- pass|warn|fail
  n_affected INTEGER NOT NULL,
  detail_json TEXT
);

CREATE TABLE model_run (
  model_version  TEXT PRIMARY KEY,        -- e.g. sev-2026.07.28-a1b2c3d
  task           TEXT NOT NULL,           -- anomaly|severity|classify|growth
  mlflow_run_id  TEXT NOT NULL,
  git_sha        TEXT NOT NULL,
  config_sha256  TEXT NOT NULL,
  data_sha256    TEXT NOT NULL,           -- Merkle hash over member content_sha256 list
  feature_version INTEGER NOT NULL,       -- pinned; bundle refuses a mismatch
  truth_as_of    TEXT NOT NULL,           -- truth revisions are slowly-changing
  trained_at     TEXT NOT NULL,
  metrics_json   TEXT NOT NULL,           -- includes bootstrap CIs, not point estimates
  artifact_uri   TEXT NOT NULL,
  final_test_uses INTEGER NOT NULL DEFAULT 0  -- holdout-reuse counter; see §Evaluation
);
-- No `stage` column: MLflow 3 removed model stages. Serving state lives on the
-- `pipeline_release.alias` pointer (@champion / @challenger), not on a model row.

-- The deployable, rollbackable unit. An indication is produced by up to four models
-- plus a feature version; one model_version column cannot express that lineage.
CREATE TABLE pipeline_release (
  pipeline_version  TEXT PRIMARY KEY,     -- 2026.07.29-a1b2c3d
  anomaly_version   TEXT REFERENCES model_run(model_version),
  severity_version  TEXT REFERENCES model_run(model_version),
  classify_version  TEXT REFERENCES model_run(model_version),
  growth_version    TEXT REFERENCES model_run(model_version),
  feature_version   INTEGER NOT NULL,
  schema_version    INTEGER NOT NULL,
  container_digest  TEXT NOT NULL,        -- pins the runtime, not just the code
  released_at       TEXT NOT NULL,
  alias             TEXT                  -- champion|challenger|NULL (MLflow 3: no stages)
);

CREATE TABLE indication (                 -- the model's unit of output
  indication_id    TEXT PRIMARY KEY,
  survey_id        TEXT NOT NULL REFERENCES survey(survey_id),
  pipeline_version TEXT NOT NULL REFERENCES pipeline_release(pipeline_version),
  is_shadow        INTEGER NOT NULL DEFAULT 0,   -- challenger output, not served
  chainage_peak_m  REAL NOT NULL,
  chainage_start_m REAL NOT NULL,
  chainage_end_m   REAL NOT NULL,
  lat REAL, lon REAL,
  anomaly_score    REAL NOT NULL,
  p_defect_cal     REAL NOT NULL,         -- calibrated probability
  pred_type        TEXT, pred_type_conf REAL,
  sev_pred REAL, sev_lo REAL, sev_hi REAL, interval_nominal REAL,
  risk_score       REAL,
  dq_flag          TEXT NOT NULL,         -- clean|warn|suspect
  created_at       TEXT NOT NULL
);

CREATE INDEX ix_indication_survey ON indication(survey_id, pipeline_version, risk_score DESC);
CREATE INDEX ix_dq_survey ON dq_report(survey_id, status);
```

**Every analytical query filters on `pipeline_version` and `is_shadow`.** Historical
surveys are not automatically re-scored when a release is promoted, so the table holds
output from several releases at once; comparing across surveys without that filter compares
different models. Growth forecasting is the case where this bites hardest.

**Concurrency.** SQLite WAL allows exactly **one writer at a time**. Feature computation and
scoring fan out across cores; **registry writes are serialised through a single Dagster
resource**. Do not write a parallel ingest against SQLite and expect it to work — above
~10⁷ rows or any genuine multi-writer ingest, the registry moves to Postgres and the bulk
path is already Parquet.

Conventions: SQLite in **WAL** mode, `PRAGMA foreign_keys=ON`, ingest is **idempotent** —
re-ingesting the same `content_sha256` is a no-op, a *different* hash for the same
`(line_id, run_id)` is an error, not an overwrite. Raw data is immutable. A survey that
fails a hard gate is written to `quarantine/` with its DQ report and recorded with
`status='quarantined'` — rejection is normal operations, not a crash.

### Scale path (the answer to "we have millions of rows")

SQLite is correct here and comfortably handles 10⁷ rows read-mostly with the composite
primary key above. It stops being the right tool at concurrent multi-writer ingest. The
honest production answer, in order: **Parquet on S3 partitioned by `line_id`/`run_id`,
queried with DuckDB or Athena, with SQLite/Postgres retained only for the small
high-value tables** (survey registry, truth, DQ, indications, model runs). Signal data is
append-only columnar; metadata is relational. The Stage 6 scale rehearsal proves this by
generating ~40 lines (≈10⁷ rows) and running the same CLI over the Parquet path.

## Feature layer: stateless vs fitted

`features.py` exposes two clearly separated kinds of transform, and the bundle only
carries the second kind.

**Per-survey / stateless** — legitimately recomputed at inference on the incoming survey,
because they *are* background removal. Rig-v2 (`feature_version` 3) replaced every
vector-axis column (`rx/ry/rz`, `r_mag_nt`, inclination/declination, `gx/gy/gz`) with
their scalar-rig equivalents below — there is no vector output under `rig: scalar`, so
there is nothing left to take an orientation of:
- robust polynomial detrend (degree 3) or Savitzky–Golay high-pass, **independently per
  head** → three SIGNED residuals `r_lo_nt`, `r_mid_nt`, `r_hi_nt` (a total-field anomaly
  can enhance or degrade the ambient field, so unlike the old `r_mag_nt` there is no
  `Check.ge(0)`)
- along-track derivative `dr_ds_nt_per_m` and `d²r_ds²_nt_per_m2` (central differences of
  `r_mid_nt`, which takes over `r_mag_nt`'s old role as the primary signal)
- first difference `g1_nt_per_m` (common-mode background rejection) and second difference
  `g2_nt_per_m2 = b_hi + b_lo − 2·b_mid` (additionally cancels a *linear* background
  gradient) — **always computed, never conditional.** Every `rig: scalar` survey has
  exactly 3 heads (`ArrayConfig.n_heads` is a documented physical fact, not a free dial),
  so the old "gradiometer enabled" branch is gone — there is nothing left to be
  conditional on.
- `standoff_est_m`: a genuine **per-row measured stand-off**, inverted from the
  head-to-head amplitude ratio via 1/r³ — replacing the assumed-constant `depth_m` the
  pre-Rig-v2 model normalised by
- sliding-window stats over `windows_m` (on `r_mid_nt`): mean, std, max, peak-to-peak,
  kurtosis, zero-crossing rate, energy
- peak-shape features: FWHM, asymmetry, fitted decay exponent — **the interference
  discriminators**, since off-pipe sources are broader and shallower-decaying
- stand-off normalisation: `r_mag_norm_nt_m3` / `peak_prominence_norm_nt_m3`, amplitude
  multiplied by `standoff_est_m**3` — real information only since `feature_version` 3;
  under `feature_version` 1 this used one global `depth_m` and was an exact duplicate of
  the un-normalised column, so it was dropped at `feature_version` 2 and reinstated once
  `standoff_est_m` made it genuine per-row information again
- registration/DQ companions, new in Rig-v2, from Stage B (`registration.py`) rather than
  from the field readings themselves: `dist_to_weld_m` and `gps_locked`

**Fitted on train only** — serialised into the bundle, never recomputed at inference:
- feature scaler / quantile transformer
- IsolationForest and LightGBM models
- isotonic calibration map for `p_defect`
- split-conformal residual quantiles for the severity interval
- the operating threshold chosen from the training-fold PR curve at the target dig budget

45 columns total (`features.py::feature_columns()` derives the exact ordered list from
config — a bundle pins it and refuses to load on a mismatch). A unit test asserts that
`predict()` on the training set through the bundle reproduces the in-training predictions
bit-for-bit.

## Model bundle

One `bundle.joblib` per model version, containing: fitted transforms, model object,
feature name list **in order**, the pinned `defect_type` category order, calibration map,
conformal quantiles, threshold, `feature_version`, `schema_version`, `config_sha256`,
`git_sha`, `data_sha256`, `truth_as_of`, library versions, and the training-set feature
summary used later as the **drift reference**. Loading a bundle raises immediately on a
feature-list or `feature_version` mismatch — no silent column reordering, no silently stale
features.

**LightGBM determinism.** Set `deterministic=True`, `force_row_wise=True` and a fixed
`num_threads`. LightGBM is not bit-reproducible across thread counts by default, and
without these the determinism test in CI flakes intermittently — worse than not having it.

## MLflow conventions

Installed in `ml_gpu`: **MLflow 3.13**, LightGBM 4.6, Dagster 1.13, Python 3.12.13.

- One experiment per task: `lsm-anomaly`, `lsm-severity`, `lsm-classify`, `lsm-growth`.
- Always logged: the **code-config layer only** (see §Environments), `config_sha256`,
  `data_sha256`, `git_sha`, seed, row and defect counts, fold assignment, DQ report,
  PR/reliability/coverage plots, SHAP summary, the bundle itself.
- **Aliases, not stages.** MLflow 3 *removed* model stages; use `@champion` (served) and
  `@challenger` (shadow). Promotion and rollback are then atomic pointer moves with no state
  mutation. Anything still writing `transition_model_version_stage` is on a dead API.
- Promotion is done by CI on the gates in `validation-and-trust.md`, never by hand.
- Dev backend `sqlite:///mlruns.db` (confirmed empirically: mlflow 3.13's older
  `file:./mlruns` filesystem backend is in maintenance mode and raises on init).
  The stated production form is an MLflow server with a
  Postgres backend store and S3 artifact store — say this when asked, it is a one-line
  config change here.

## Environments and config layering

Three layers with different lifecycles; conflating them voids the reproducibility claim.
**Only the code layer feeds `config_sha256`** — otherwise the same code and data hash
differently in prod than in dev.

| Layer | File | Contents |
|---|---|---|
| Code config | `config/base.yaml` | detrend, windows, quantiles, gate thresholds — **hashed** |
| Environment | `config/{dev,staging,prod}.yaml` | bucket, DB URI, MLflow URI, concurrency |
| Runtime | CLI / Dagster partition | which survey, which `pipeline_version` |

**Runtime parity is enforced, not recorded.** Training runs in `ml_gpu` (Python 3.12.13),
the local venv is 3.13, and the app runtime is a third graph. `bundle.load()` **hard-fails**
on a LightGBM / numpy / sklearn version mismatch rather than merely storing the versions —
a silent minor-version difference changes predictions with no error.

## Orchestration (Dagster)

The eight CLI steps are the *implementation*; the **asset graph** is the interface.

```
raw_survey (S3 sensor) → ingested_survey → dq_report → survey_features(fv=n)
                                              ↓              ↓
                                        quarantine/    indications → gis_export
                                                                  → drift_report
```

- Partitioned by `survey_id` (dynamic partitions). **Backfill over the archive is a
  partition backfill**, not a bespoke script — this is why Dagster over Airflow here, since
  `feature_version` maps directly onto asset code versions.
- Retries: 3× exponential backoff on IO/transport; **zero on DQ failure** — a bad survey is
  not a flake.
- One poison survey quarantines itself; the run continues.
- Registry writes serialised (see Concurrency above); feature computation fans out.

## S3 layout

```
s3://$LSM_BUCKET/lsm-demo/
  raw/line_id=LINE003/run_id=2/survey.parquet     # immutable, versioning enabled
  quarantine/<survey_id>/{survey.parquet,dq.json} # failed a hard gate; not deleted
  features/fv=3/line_id=.../run_id=.../features.parquet
  models/<model_version>/bundle.joblib
  models/<model_version>/{metrics.json,model_card.md}
  reports/<survey_id>/dq.json
  reports/<survey_id>/indications.geojson         # what the GIS team consumes
```

`lat`/`lon` are **restricted columns** (see `data-contract.md` §10): excluded from MLflow
artifacts and from any drift-monitoring export, even though this project's are synthetic.

Local-first: `storage.enabled: false` keeps everything on disk so the repo runs with no
cloud account. Tests mock S3 with `moto`; CI authenticates to AWS with **GitHub OIDC**, not
long-lived access keys.

## Serving layer vs the demo app

**The serving layer is the batch scoring job** — the `indications` asset writing to the
`indication` table and `indications.geojson`. Its contract: versioned output schema with a
consumer contract test in CI, idempotent on `(survey_id, pipeline_version)`, indications
available < 24 h after upload. The Streamlit app is a **consumer** of that output, not the
server. Do not conflate them; full contract in `docs/production-architecture.md` §4.

The app **never imports the training code**. It reads a self-contained serving artifact:

```
serving/
  bundle.joblib            # pulled from S3 by pinned model_version, baked-in copy as fallback
  demo_surveys/*.parquet   # 3-4 pre-baked scenarios, including one deliberately corrupted
  precomputed/*.parquet    # indications for each demo survey
  model_card.md
```

`APP_MODE=demo` serves precomputed results (zero compute, cannot fail);
`APP_MODE=live` runs the real `validate → features → score` path on a selected survey
(~1 s). Same code both ways; ship `demo` as the default with a "run it live" toggle, so the
impressive path and the safe path are the same path.

**Deployment:** Hugging Face Spaces, **Streamlit SDK not Docker** (~16 GB RAM removes any
OOM risk and lets the Stage-6 scale dataset be demoed; the Streamlit SDK avoids Docker's
slow cold start). Pushed by the GitHub Action *after* promotion gates pass — not
auto-deployed from a branch, which would undermine the gated-release story. Inference is
CPU-only and takes milliseconds; nothing here needs a GPU. Anything trained on GPU must be
saved CPU-loadable (`map_location="cpu"` or ONNX).

**iPad constraints** (the device is a thin client — Safari only):
- Pre-baked scenario **buttons**, not `st.file_uploader` — iPadOS file picking is painful.
- `st.segmented_control` / `st.pills`, not sliders — touch targets. Dig budget as
  "3 / 5 / 10 per km" buttons.
- `layout="wide"`, sidebar collapsed by default; nothing essential behind the sidebar.
- Operational rule: **never push on demo day** (a push triggers a rebuild); deploy the day
  before and wake the Space 5 minutes ahead.
- Fallback ladder decided in advance: cloud URL → `streamlit run` on the laptop →
  recorded GIF.

## CI/CD

- `ci.yml` (every push/PR): ruff + pytest + `generate → ingest → validate → features →
  train → predict` on a tiny config. Fails on any DQ `fail` gate, leakage test failure, or
  golden-dataset mismatch.
- `train.yml` (dispatch / tag): full training, MLflow logging, gate evaluation, cut a
  `pipeline_release`, set the **`@challenger` alias**, push bundle to S3. Posts the metrics
  table to the run summary.
- `deploy.yml` (on promotion): move the **`@champion` alias**, push the serving artifact
  pinned to a `pipeline_version`. Rollback = repoint `@champion` at the previous release;
  bundles are immutable so a rollback target always exists. **CI drills the rollback** —
  the last N = 3 feature versions must stay loadable, otherwise the `feature_version` guard
  blocks the very path it was meant to protect.
