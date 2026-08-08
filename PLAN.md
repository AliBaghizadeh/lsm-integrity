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
Face Spaces for the demo. Training env: nominally `ml_gpu` (Python 3.12.13, dagster 1.13.15,
mlflow 3.13.0, lightgbm 4.6.0) — **but `ml_gpu` has a broken numpy install on this machine
(confirmed, unrelated to project code); use `C:\Users\aliba\.conda\envs\llm_gpu2\python.exe`
instead for everything, see `SKILL.md`'s "Working conventions" for the full note.**

**Design centre:** the primary workload is **backfill over an existing archive**, not
steady-state scoring. Anywhere the plan looks over-engineered for 12,000 rows, that is why.

**Effort.** The stage estimates below are focused-work days and are optimistic by roughly
2–2.5× once tests and fixtures are counted. Budget ~20 days for the whole plan, not 11.
Use the cut lines.

---

## Current status (as of 2026-08-08)

| Stage | Status |
|---|---|
| 0 — Skeleton and reproducibility | Done |
| 1 — Ingest and validation | Done |
| 1.5 — Orchestration and backfill | Done |
| 2 — Features, gradiometer | Done |
| 2.5 — Data fidelity | Done, all 3 items (2026-07-29) |
| 2.75 — EDA on the feature store | Done, findings acted on and re-run for real (2026-07-30) |
| 3 — Detection (MAD vs IsolationForest) | Done. Pre-Rig-v2: recall gap closed to -0.006 after the EDA fix, confirmed at 800× scale (-0.029 [-0.033,-0.026], still a real small negative effect). **Re-measured under Rig-v2 (2026-08-06): -0.006 [-0.061, 0.056], numerically unchanged — gate still does not pass the 0.15 margin.** |
| 4 — Severity (LightGBM CQR + conformal) | Done pre-Rig-v2 (gate PASSED, coverage 0.920). **Re-measured under Rig-v2: coverage 0.689 — gate now FAILS, a real reported regression on the harder scalar-rig acquisition, not smoothed over.** |
| 4.5 — Demo app | Built, headlessly verified, and iterated through several real-use feedback rounds (last: 2026-08-07/08 — Rig-v2 row-count/GPS-dropout fixes, see below). HF Spaces deploy still deferred, see next-steps. |
| 5 — Classification and risk ranking | Done (2026-07-30). Demoted to a synthetic-only capability demo after the ROSEN developer interview confirmed no real labelled defect-type data exists — kept for the MLOps discipline around it (calibration, SHAP physics-consistency gate), not as a real-world claim. Gate did not pass pre-Rig-v2 (SCC recall 0.125) or under Rig-v2 (0.042) — expected, not new. |
| 6 — Scale rehearsal | Done (2026-07-31), pre-Rig-v2, ~9.6M rows. **Rig-v2 has not yet re-run this at scale** — see next-steps. |
| 7 — CI/CD and release gate | `ci.yml` + `train.yml` built and green/red as designed; `deploy.yml`/S3/rollback still deferred pending a cloud-account decision (unchanged since 2026-07-30). |
| 8 — Growth, remaining life, monitoring | Done (2026-07-31). Gate passes pre- and post-Rig-v2 (recovers `ln(1.15)=0.1398` either way) — an estimator-correctness check, not affected by which rig produced the residuals. |
| **Rig-v2** — rebuild around the real ROSEN instrument | **Done (2026-08-06/07).** A 2nd-round developer interview revealed the real rig: 3 scalar total-field heads (never x/y/z), human walker, GPS dropout — not the single 3-axis vector head this project originally assumed. Rebuilt the generator/schema (Stage A), added a new registration stage (Stage B, `src/lsm/registration.py`), rewrote features for per-head g1/g2 differences (Stage C), re-measured every gate honestly and built a 6-arm ablation ladder answering "software or hardware?" (Stage D — **software: +0.014 [-0.028,0.056], indistinguishable from zero; hardware: recall roughly triples**), and updated every doc (Stage E). Full detail: `LSM_PROJECT.md`'s "Rig-v2 measured results" section. |
| **Post-Rig-v2 app fixes** | Done (2026-08-07/08). Rig-v2 changed a raw survey from ~4,000 to ~200,000 rows and introduced real GPS dropout — the app wasn't updated for either, causing two real bugs: the map rendered broken/random lines (NaN GPS coordinates fed straight to pydeck) and the app was very slow (full-resolution signal charts melted into Vega-Lite on every rerun). Both fixed; `docs/app-walkthrough.md` and the app's own "How it works" tab updated to include the new Register stage and current numbers. |

Stages 0–2 have no individually-recorded completion date: this repo's first commit
(`d34084e`, 2026-07-30) already bundled Stages 0–4 together as an initial import, so no
finer-grained per-stage date exists in git history for the earliest stages. Everything from
Stage 2.5 onward has a real recorded date, either from this table's own prior entries or
from `git log`.

311/311 tests pass, ruff + mypy clean. GitHub (`AliBaghizadeh/lsm-integrity`, private) is a
**curated public snapshot**, not a mirror of this local repo's full history — internal
planning/interview docs (`PLAN.md`, `LSM_PROJECT.md`, `docs/`, `.claude/`, `reports/`) are
deliberately excluded from every push, same convention as the very first push. See this
project's Claude Code memory for the exact exclude list and the rebuild procedure; local
`master` does not track `origin/main` on purpose, so a plain `git push` can't leak the
private history.

**What's actually left open, roughly in order of what unblocks the most:**

1. **A cloud-account decision** — still the single blocker on the rest of Stage 7 (S3 +
   GitHub OIDC, designed in `docs/production-architecture.md` but not built,
   `storage.s3.enabled: false` throughout) and on Stage 4.5's Hugging Face Spaces deploy.
   Unchanged since 2026-07-30: decide whether this demonstrator stays local-only (defensible
   on its own terms, has a fully working local demo) or is worth standing up real
   infrastructure for.
2. **Rig-v2's own explicitly-flagged not-yet-run items** (all stated honestly in
   `LSM_PROJECT.md`, not hidden):
   - The 800×-scale rehearsal that resolved the pre-Rig-v2 Stage 3 "is this gap real"
     question has not been re-run under Rig-v2 — the current -0.006 [-0.061, 0.056] result
     is demo-scale only, so whether it's a stable small effect or sample-size noise is
     still open.
   - The 6-arm ablation ladder ran at a 2-line scale-down (`ABLATION_N_LINES`) for runtime,
     not the full 5-line default — easy to re-run larger.
   - Whether more calibration data closes Stage 4's Rig-v2 severity-coverage regression
     (0.920→0.689) the way it closed the equivalent pre-Rig-v2 gap is unanswered.
**Historical negative result that motivated Stage 3's design (kept for context):** three
`lsm train` runs pre-Rig-v2 told one consistent story before the current -0.006 result —
12-defect corpus recall gap -0.056 CI [-0.167, 0.056] (too wide to trust), 5-line/~60-defect
corpus -0.028 CI [-0.083, 0.022] (about half the width, still real), then after Stage 2.75's
EDA-driven fix (dropped 2 duplicate features, gave IsolationForest an `emphasis_repeats`
knob) -0.006 CI [-0.067, 0.050] with the interference-rejection mechanism reversing and
becoming significant (CI [0.007, 0.058], excludes zero) — the gate still didn't pass, but
"EDA-informed feature engineering measurably improved interference rejection, confirmed by a
significant CI, even without clearing an arbitrary recall-margin bar" was a strong result in
its own right, and remains the mechanism story behind the now-Rig-v2-re-measured number
above.

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

## Stage 0 — Skeleton and reproducibility  *(built, bundled into the initial commit)*

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

## Stage 1 — Ingest and validation  *(built, bundled into the initial commit)* — do this first and make it visible

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

## Stage 1.5 — Orchestration and backfill  *(built, bundled into the initial commit)* — the biggest single gap to close

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

## Stage 2 — Features, and the gradiometer upgrade  *(built, bundled into the initial commit; the two-head gradiometer story here is superseded by Rig-v2's three-head g1/g2 differences — see `LSM_PROJECT.md`'s "Rig-v2 measured results")*

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

**4. Circumferential (clock) position is not modelled at all — found 2026-08-04, not yet
addressed.** `_build_features()` (`generate.py:132-144`) sets `y_off_m = 0.0` for *every*
defect — always placed directly on the sensor's own path. Only *interference* sources get a
randomised lateral offset (`y_off_m = rng.uniform(3, 8) * rng.choice([-1, 1])`, line 155), and
that represents an object sitting beside the pipe (fence post, buried scrap), not a defect at a
different position on the pipe wall itself. So the corpus never varies "is this defect at 3
o'clock or 9 o'clock" for real defects — zero examples of that dimension exist to learn from.
This is **not** caused by merging `rx/ry/rz` into `r_mag` for peak detection (Section 6) —
keeping the three axes fully separate through detection wouldn't fix it, because the generator
never produced that variation in the first place. `r_incl_deg`/`r_decl_deg` (dipole
orientation) are kept as separate features and are informative about defect *type/shape*, but
they are not a substitute for source *position*, and the corpus gives no lateral-offset target
to regress or classify against for defects. Recovering true circumferential position in a real
tool needs either multiple sensor heads/arrays at distinct clock positions on the crawler, or
genuine dipole-source inversion (fit the full 3-axis field to solve for 3D source location) —
this project does neither; it's row classification / anomaly detection against a corpus with
zero circumferential variation, not source localisation. State this explicitly if asked "can
this tool tell you which side of the pipe the defect is on" — the honest answer is no, and not
as a detection-stage modelling shortcut: the generator never taught the corpus that dimension
exists.

---

## Stage 2.75 — EDA on the feature store  *(built 2026-07-30)*

Not a pipeline stage — a one-off, reproducible analysis (`scripts/eda_features.py`, no
`feature_version`, no CLI subcommand) run before trusting the 48-feature set Stage 3 models
on. Prompted directly by "do we really have EDA, and is it industry-acceptable?" — the
answer before this was no: two ad hoc dev plotting scripts and 14 DQ gates that check known
invariants, nothing that explores the feature space itself. This is that missing piece:
class balance, per-feature distributions, correlation/redundancy, and per-feature
separability, over the real 5-line/15-survey corpus (58,800 clean rows).

**Class balance:** defect 3.67%, interference 4.81%, background 91.52%.

**Missingness is not uniform across classes, and it's informative, not a bug:** the five
shape features (`fwhm_m`, `peak_asymmetry`, `decay_exponent`, `peak_prominence_nt`,
`peak_distance_m`) are NaN on 91.3% of background rows (no nearby peak to describe), 17–17.7%
of interference rows, and only 0.5% of defect rows. `anomaly.py` fills these with 0.0 before
scoring — which means "shape feature present vs zero-filled" is itself a real, physically
meaningful signal (a peak exists nearby or it doesn't), not leakage, but worth stating
explicitly since a model doesn't know the difference between "genuinely zero" and "filled".

**Redundancy: confirmed and worse than assumed.** 35 feature pairs (of 1,128 possible) had
`|r| ≥ 0.9`. Two pairs were **exact duplicates under the current config** (`r = 1.000`):
`r_mag_nt` / `r_mag_norm_nt_m3`, and `peak_prominence_nt` / `peak_prominence_norm_nt_m3` —
because standoff-normalisation divides by `depth_m³`, and `depth_m` is a single global
config value, not a per-row measurement, so it's a constant scalar multiply, not new
information. **Acted on, not just noted:** `feature_columns()` no longer includes them, and
`compute_survey_features()` no longer computes them at all (`feature_version` 1 → 2,
regenerated the real feature store — 46 features now, not 48; redundant pairs ≥0.9 dropped
35 → 27). Reintroduce them, with another version bump, once a future stage makes stand-off a
genuine per-row measurement. The rest of the redundant block is the window-statistic family
(`w2m`/`w5m`/`w10m`/`w25m` × `mean`/`std`/`max`/`ptp`/`energy`) — expected, since they're all
computed from the same underlying residual at overlapping window lengths, but the
correlation heatmap (`docs/img/eda_feature_correlation.png`) makes the size of that block
visible for the first time: roughly 20–24 of the 48 features move together as one cluster.

**Separability (per-feature PR-AUC, diagnostic only — no CV, no model, just "does this one
feature carry signal") is the finding that matters most for Stage 3's open question.** The
big correlated amplitude block (`w5m_mean_nt` 0.968, `w5m_energy_nt2` 0.959, `w5m_std_nt`
0.930, …) separates defect from background almost perfectly on its own — but those same
features only separate **interference** from background at 0.72–0.80, because interference
genuinely produces amplitude too (by design — it's the false-positive trap). Defect vs
background is not the hard problem; **defect vs interference is**, and the amplitude block
is mediocre at it (most under 0.55). The features that actually separate defect from
interference are a small, different set: `w25m_kurt` (**0.985**), `w25m_zcr` (0.786),
`w10m_kurt` (0.680), `g_mag_nt_per_m` (0.638) — wide-window shape descriptors, exactly the
"width/decay-shape separates interference far better than amplitude alone" physical claim
this project has stated since Stage 0, now measured rather than asserted
(`docs/img/eda_bottom_separating_features.png` shows `w25m_zcr`'s clean three-way spread).

**This upgrades a hypothesis in `interview-drills.md` from a guess to an evidence-backed
mechanism.** That doc previously speculated IsolationForest's loss to MAD "might" be because
48 unweighted, near-duplicate features dilute the forest. EDA now shows concretely *which*
features are redundant (the ~20-feature amplitude block, plus two literal duplicates) and
*which* 1–3 features actually carry the defect-vs-interference signal (`w25m_kurt` above
all). An isolation forest splitting roughly uniformly across 48 dimensions, ~20 of which are
mutually correlated and redundant, has much more opportunity to isolate on the dominant
amplitude cluster than on the one or two shape features that would correctly reject
interference — consistent with Stage 3's own finding that IsolationForest's recall loss
shows up specifically at the tight dig-budget cutoff via extra interference let through.

**Implemented 2026-07-30, same day.** Two changes, both config-reversible, neither touching
what MAD or the severity model see:
1. The two exact duplicates dropped from the model entirely (`feature_version` 1 → 2, above).
2. `IsolationForestAnomalyModel` gained `emphasize_features` / `emphasis_repeats`
   (`models/anomaly.py`): sklearn's IsolationForest picks a feature *uniformly at random* at
   every split, so repeating `w25m_kurt`/`w25m_zcr`/`w10m_kurt` in the fitted/scored matrix
   (`config/base.yaml`: `emphasis_repeats: 5`) raises their selection probability without
   changing what they mean or touching the bundle-pinned `feature_cols` contract. Set
   `emphasis_repeats: 1` (or `emphasize_features: []`) to fall back to the original,
   unweighted behaviour — this is a knob, not a rewrite. New unit tests pin the matrix-
   widening behaviour, the no-op-at-repeats=1 case, and a loud `ValueError` on a config typo
   naming a column outside `feature_cols`.

**Ali re-ran `lsm train` for real the same day. It worked, and worked convincingly:**

| Metric | Before this fix | After this fix |
|---|---|---|
| Recall gap (IF − MAD) | −0.028, CI [−0.083, 0.022] | **−0.006, CI [−0.067, 0.050]** |
| False-dig rate, MAD vs IF | 0.227 vs 0.233 | 0.227 vs 0.216 |
| Interference-dig-fraction, MAD vs IF | 0.227 vs 0.233 (not attributable) | 0.227 vs 0.196 — **CI [0.007, 0.058], does not cross zero** |
| IsolationForest PR-AUC | 0.431 | 0.432 |

Recall is now statistically indistinguishable between the two models (CI straddles zero,
nearly symmetric) — the gap closed from −0.056 (original 12-defect run) to −0.028 (5-line
run) to −0.006 (this run), each step attributable to something specific and understood, not
noise. More importantly, the mechanism reversed and became **statistically significant**: MAD
now wastes significantly more of the dig budget on interference than IsolationForest does
(`train.py`'s own attribution check confirms it, CI excludes zero) — the *opposite* of the
uncredited pattern before the fix. **The Stage 3 gate still does not pass** (needs a ≥0.15
recall-gap margin at the CI lower bound; the honest recall result is parity, not an
IsolationForest win), but this is a materially better, better-understood, evidence-backed
result than "doesn't beat baseline, don't know why" — and it is a genuinely reportable
finding: feature engineering informed by real EDA measurably improved interference rejection,
even though it wasn't enough to clear an arbitrary recall-margin bar.

**Stage 4, same re-run:** coverage 0.920 (was 0.905) — still comfortably inside [0.87, 0.93],
gate still PASSES. MAE 7.46 vs the global-mean baseline's 15.09.

**Outputs:** `scripts/eda_features.py` (reproducible), `docs/img/eda_feature_correlation.png`,
`docs/img/eda_top_separating_features.png`, `docs/img/eda_bottom_separating_features.png`,
`docs/img/eda_feature_stats.csv`, `docs/img/eda_separability.csv`.

**Answers:** "do you actually do EDA, or just validation gates and modelling?" — and gives
Stage 3's open mechanism question an evidenced answer instead of a shrug, then a real,
measured improvement once acted on.

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
stronger than a tuned one. **Run for real three times**, and the gate never passes, but the
story sharpens each time: 1-line/12-defect corpus (recall gap -0.056, CI [-0.167, 0.056]);
5-line/~60-defect corpus after the `generate.py` spacing fix (recall gap -0.028, CI
[-0.083, 0.022] — confirms a real, small effect, not sampling noise); same 5-line corpus
after Stage 2.75's EDA-driven fix (dropped 2 exact-duplicate features, added an
`emphasis_repeats` knob so the 3 features that actually separate defect from interference
aren't drowned out by ~20 redundant amplitude features) — recall gap **-0.006, CI [-0.067,
0.050]**, and the interference-attribution check flips to **significant** (CI [0.007, 0.058],
excludes zero): MAD now wastes significantly more dig budget on interference than
IsolationForest does. Recall parity, not an IsolationForest win, so the ≥0.15 gate still
fails — but this is a materially improved, mechanism-understood result, not an unexplained
miss. See the "Current status" section at the top of this file and Stage 2.75 for the full
writeup.

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
`feature_version` mismatch raises. **Run for real three times.** Pre-fix (naive quantile,
1-line corpus): coverage 0.727 — the calibration bug above. Post-fix, same 12-defect
corpus: coverage still short of the gate but the CI contained the target band, an
inconclusive small-sample result rather than a broken model. Post-fix, 5-line/~60-defect
corpus (2026-07-30, same run as Stage 3's second evaluation): LightGBM CQR beats the
global-mean baseline on every metric (MAE 7.20 vs 14.95, interval width 45.4 vs 60.2,
coverage 0.905 vs 0.912) and **coverage 0.905 lands inside [0.87, 0.93] — gate PASSES.**
The same extra data that sharpened Stage 3's CI gave conformal calibration enough held-out
points to hit its target band. Its honesty is still conditional on Stage 3's detector,
which does not beat baseline — see Stage 3's gate result above.

**Answers:** "how much should I trust this number before I authorise a dig?"

---

## Stage 4.5 — The demo app *(a consumer, not the serving layer)*  *(built 2026-07-30, iterated through multiple feedback rounds through 2026-08-08 — see "Current status" above)*

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

**Built 2026-07-30 (local scope): the serving artifact, both APP_MODE paths, and all four
beats are real and headlessly verified.** HF Spaces deploy is deliberately NOT built yet —
it needs a cloud-account decision (item 6 above) and Stage 7's `deploy.yml`, neither of
which exist. Run it: `streamlit run app/demo_app.py` or `lsm serve`.

- `scripts/bake_demo_assets.py` builds a committed `serving/` directory (~6 MB) matching
  `architecture.md`'s spec exactly: 3 clean scenarios reused from the real, already-trained
  5-line corpus (different `line_id`s each, so `check_survey_overlap`/`check_background_
  regime` never produce order-dependent surprises across scenario picks in one session) —
  genuine model output, not fabricated — plus one freshly-corrupted scenario (`LINEDEMO_R0`:
  one real raw survey's bytes, relabelled, then `bx_nt` pushed out of range at row 3, the
  exact recipe `tests/test_validate.py` already proves trips `check_range`). The bake script
  runs the corrupted scenario through the real validator against a throwaway db and
  **asserts it actually fails** — if a future generator/validator change ever stops tripping
  the gate, baking fails loudly instead of silently shipping a "corrupted" scenario that
  isn't. Model bundles are copied verbatim from `data/models/`, not re-derived, so
  `load_bundle`'s hard version-mismatch check still applies exactly as it would in
  production.
- `src/lsm/pipeline.py` gained `run_full_pipeline()`: register → validate → features →
  score in one call, degrading to `(report, None)` (no exception) on a hard DQ fail — the
  one new "real" function this stage needed, reused by nothing demo-specific.
- `app/demo_lib.py` is the Streamlit-free seam: DEMO mode reads baked Parquet/JSON directly
  (zero SQLite, zero compute, cannot fail — literally, not just in spirit). LIVE mode builds
  one real, session-private SQLite db (a real tempfile, deliberately not the literal string
  `":memory:"` through `db.connect()`'s `Path(...)` wrapping, which is ambiguous on
  Windows), seeded from the manifest's own real `pipeline_release`/`model_run` rows so
  `predict_survey`'s FK reads resolve, then drives `run_full_pipeline` for real.
- `app/demo_app.py`: `st.segmented_control` for mode and scenario (never file upload), four
  beats as tabs, dig-budget control as a pure client-side top-k slice, footer provenance
  strip from the manifest. `app/map_utils.py` was extracted from Stage 3's `streamlit_app.py`
  so the two apps' pydeck rendering can never quietly diverge.
- **Two real bugs, both caught by headless testing before they'd have surfaced as "it's
  broken" during an actual demo:**
  1. Streamlit does **not** add the executed script's own directory to `sys.path` the way
     plain `python script.py` does — `import demo_lib` raised `ModuleNotFoundError` the
     first time the app was actually run (via `streamlit.testing.v1.AppTest`, not by eye).
     Fixed by adding the same explicit `sys.path.insert` both apps already do for `src`.
  2. The live-mode SQLite connection needed `check_same_thread=False` — **not** because of
     `st.cache_resource` (correctly avoided here; that cache is a process-wide singleton,
     wrong for a per-session writable connection) but because Streamlit can dispatch one
     session's *own* consecutive reruns onto different worker threads from its thread pool,
     which trips sqlite3's same-thread guard even though only one rerun ever executes at a
     time. Caught by scripting exactly that sequence — switch to live mode, run one
     scenario, then rerun on a *different* scenario in the same session — via `AppTest`.
- **Verification, precisely scoped:** `tests/test_pipeline_full.py` (3 tests),
  `tests/test_demo_lib.py` (10 tests, exercises every `demo_lib` function against the real
  committed `serving/` dir, including a rebuild-drift check: bundles must load against
  whatever `feature_version`/`schema_version` `config/base.yaml` *currently* declares, not
  just what they were baked against), and `tests/test_demo_app.py` (5 tests, via
  `streamlit.testing.v1.AppTest` — a real script execution through Streamlit's own
  ScriptRunner, not a mirror script: demo mode boots clean, the corrupted scenario shows the
  refusal, live mode survives the exact multi-rerun/thread-hop sequence that broke before
  the fix). 170/170 tests total, ruff + mypy clean. **What's still NOT machine-verified**:
  visual layout, the pydeck map's actual rendering, and the "~1 s" live-mode feel — `AppTest`
  proves the script runs without exception and produces the right widgets/errors, not that
  it looks right. That needs Ali to actually look at a browser.

**Answers:** "do you have something I can actually click through, not just metrics?" — and
demonstrates the habit of catching integration bugs with a headless test harness before
they'd show up mid-demo, not after.

**First real-use feedback from Ali, acted on same day.** After actually running it: Beat 1
showed only one line (`|B|` magnitude — threw away axis information), no way to confirm the
buried anomaly was genuinely there rather than just asserted, the app never explained what a
"survey" even is, and it felt slow. Fixed:
- `app/chart_utils.py` (new, Streamlit-free like `map_utils.py`/`demo_lib.py`) — Beat 1 now
  plots bx/by/bz as three separate lines, with a log-scale "|B| deviation from its own
  median" toggle (the literal answer to "I need log scale to see anomalies" — raw bx/by/bz
  are signed and can't be log-scaled directly, but the absolute deviation can, and that's
  the view where the anomaly stops being invisible). Both Beat 1 and Beat 2's charts now
  overlay dashed vertical markers at the true defect/interference chainages (contiguous-run
  midpoints, not one mark per row), so it's visually obvious whether the raw trace shows
  anything there.
- The slowness had a real, findable cause, not a vague "Streamlit is slow": the whole script
  reruns on **every** widget interaction, and the app was reloading raw/features/indications
  from disk AND, in live mode, re-running the **entire** validate→features→score pipeline on
  every rerun — including totally unrelated ones, like moving the Beat 3 dig-budget control.
  Fixed with `st.cache_data` on the demo-mode loaders and a `survey_id`-keyed cache in
  `st.session_state` for live-mode results, so the pipeline now runs once per scenario per
  session, not once per click. Pinned by a new test that asserts the cached result object is
  the *same object* (not just equal) before and after an unrelated widget change — proof of
  no recompute, not just proof of no crash.
- Added an intro paragraph explaining what a "survey" and a "scenario" are, and per-beat
  subheaders/captions explaining *why* each view matters, not just what it shows.
- `tests/test_chart_utils.py` (5 tests) + a new `test_demo_app.py` caching-regression test.
  176/176 tests total, ruff + mypy clean.

---

## Stage 5 — Classification and risk ranking  *(built 2026-07-30; demoted to a synthetic-only capability demo post-Rig-v2, see "Current status" above)*

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

**Built 2026-07-30 (local scope): the full classify pipeline is real and headlessly
verified — model, calibration, metrics, `risk_score`, model card, and the physics-consistency
gate.** Real corpus numbers (SCC recall at the CI lower bound, the real Brier score/
reliability diagram, how often the degenerate-calibration fallback fires) still need Ali's
`lsm train` run — same division of labour as every prior stage.

- The codebase had already anticipated this stage extensively before it was built:
  `config/base.yaml`'s bare `model.classify` block, `pipeline_release.classify_version`,
  `indication.pred_type`/`pred_type_conf`/`risk_score` columns, and `bundle.py`'s
  `defect_type_categories` stamp were all pre-built placeholders. `truth.build_truth_
  registry` already carried `defect_type` per physical source (including the literal
  `"interference"`), so Stage 5 needed one new join, not new label-derivation logic.
- `src/lsm/models/classify.py` (new): `ClassifyModel` (LightGBM multiclass, dynamic
  `num_class` per fold — hardcoding it to the full class count breaks on any fold with fewer
  distinct labels), calibrated via `sklearn.frozen.FrozenEstimator` wrapping the fitted model
  (`CalibratedClassifierCV(cv="prefit")` is **removed** in the installed sklearn 1.9.0) fit
  against a class-stratified-then-defect-grouped calibration split. Degenerate folds (a
  class missing from calib entirely, or only one class in train at all — both real,
  reproducible sklearn failure modes at this project's small per-class counts, confirmed
  empirically not assumed) fall back to uncalibrated softmax / a constant prediction, logged
  loudly rather than silently swallowed. `MajorityClassBaseline` is the required baseline.
- **SHAP via LightGBM's native `pred_contrib=True`, not the external `shap` package.** `shap`
  is installed but `import shap` raises `ImportError: Numba needs NumPy 2.4 or less` — numba
  0.66.0 (latest available) doesn't support the installed numpy 2.5.1, and this isn't
  fixable by upgrading. LightGBM's own `pred_contrib=True` gives genuine TreeSHAP
  contributions with no new dependency (verified: an engineered informative feature scores
  ~3.3 mean|contrib| vs ~0.02 for noise). The physics-consistency test is an **engineered-
  leak integration test** (not just an absent-name check, which would pass trivially since
  `chainage_m` is already structurally excluded from `feature_columns()`): a chainage-derived
  column is deliberately made informative by construction, a real model fit on it, and the
  denylist check must catch it — proving the gate has teeth.
- `risk_score = calibrated P(defect) × predicted severity × consequence proxy`. The
  consequence proxy (`config/base.yaml: model.classify.consequence_proxy`) is a stated
  engineering-judgment ranking, not derived from real data — SCC 1.0 (crack-like, sudden
  failure), corrosion 0.6, dent 0.5, weld 0.4 (best-controlled), interference 0.0 (never a
  pipe defect, so it can never dominate a dig ranking). Called out as a judgment call in both
  the config comment and the model card, not hidden as a bare constant. `p_defect_cal` (an
  explicitly-flagged, previously-uncalibrated percentile-rank stand-in) is now a real
  `IsotonicRegression` fit on the full out-of-fold matched-indication population (matched-
  defect, matched-interference, and unmatched false alarms alike).
- `train.py`'s matching step was refactored into a shared `_match_all_indications()` (the
  clustering half, `_cluster_all_indications()`, was already factored out) so severity and
  classify both build off one pass — verified zero regression in the existing severity path
  by re-running its pre-existing test unmodified before adding anything new.
- **Verification:** `tests/test_models_classify.py` (10), Stage 5 additions to
  `tests/test_evaluate.py` (~14), `tests/test_indications.py` (4), `tests/test_predict.py`
  (2), `tests/test_bundle_roundtrip.py` (1), `tests/test_model_card.py` (2), plus the
  extended `test_train.py` end-to-end smoke test and the two engineered-leak SHAP tests.
  212/212 tests total, ruff + mypy clean.

---

## Stage 6 — Scale rehearsal  *(built 2026-07-31, pre-Rig-v2 vector rig; not yet re-run under Rig-v2, see "Current status" above)* — the answer to "millions of rows"

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

**Built and run for real 2026-07-31.** Full detail, numbers, and the report itself:
`docs/stage6-scale-rehearsal.md` (generated by `scripts/stage6_scale_rehearsal.py`, run once
against a new `config/scale/` — never touches the demo `data/`). Highlights:

- **A real arithmetic gap in this plan's own numbers**: "`n_lines: 40` → ~10⁷ rows" doesn't
  close at the demo's `length_m=2000` (only ~480,000 rows). Resolved: `length_m=40000` (a
  realistic full-segment survey length) + `n_lines=40` = 9.6M rows, density-preserving
  (`n_defects=240`, `n_interference=80`, same 6/km, 2/km ratio as the demo).
- **Gate result: full pipeline ran on 9.6M rows, ~34 min end to end** (generate 505s, ingest
  34s, features 1s cache-permitting, train's block CV 765s, whole-line+temporal 1017s), peak
  RSS 23.7–28.4 GB (IsolationForest's 300-tree fit on 9.4M×46 features is the dominant cost,
  well within the 128 GB budget). Full rows/sec + peak-RSS table in the report.
- **Two real bugs found and fixed, both invisible at demo scale:**
  1. `generate.py` initialises `severity_smys` to NaN off-defect (not 0, contradicting this
     project's own documented convention) — a detector's peak occasionally lands just
     outside a defect's label window while still within the looser dig-matching tolerance,
     so a NaN true-severity value reaches the matched frame (0.4% of matched defects at this
     scale, 0 at demo scale). One NaN in a fold's conformal calibration set silently NaN'd
     that WHOLE fold's margin (`np.quantile` propagates NaN), collapsing pooled Stage 4
     coverage to exactly 0%. Fixed by dropping NaN-truth rows upstream
     (`_build_severity_training_frame`) plus a loud belt-and-braces guard in
     `SeverityModel.fit` itself. Confirmed: Stage 4's gate now **passes** at scale (coverage
     0.896 block-CV, 0.902 whole-line-holdout).
  2. `config/base.yaml`'s `model.classify.n_estimators`/`num_leaves` had been **silently dead
     since Stage 5** — never threaded from config into `ClassifyModel`, coincidentally
     matching its hardcoded fallback (50/7), so editing them did nothing. Found while
     investigating why scale's larger capacity values weren't changing classify's behaviour.
     Fixed; SCC recall improved 0.50→0.64 (block-CV) once actually wired.
     `SeverityModel`'s own `n_estimators`/`num_leaves` were similarly hardcoded (not even
     config-driven at all) — now read from `model.severity`, demo values unchanged.
- **Background-contrast gate did NOT pass at 40 km** (2.70x vs the 3.0x Stage 2 gate).
  Diagnosed, not just reported: background residual is essentially unchanged (~7.7 nT vs the
  demo's ~7.7–8.1 nT — detrending is not degrading at this length), but the defect-residual
  median is lower than the demo's ~25 nT while the max is 139 nT — a strongly right-skewed
  distribution, most likely meaning the demo's own 12-defect median was itself an optimistic
  small-sample read, not that this stage broke anything. Reported honestly, not adjusted
  away; a larger single-survey sample for the Stage 2 check itself is future work.
- **A real DQ finding**: 2 of 120 surveys (1.7%) were quarantined by `check_survey_overlap`
  on a correlation just above the fixed 0.9 threshold — an order-statistic effect (the check
  takes the MAX correlation across an ever-growing number of same-line prior runs) compounded
  by a 20x longer line giving two runs' shared deterministic structure far more samples to
  correlate on. The DQ layer did exactly its job (quarantine, not crash); revisiting the
  threshold for long-line, many-run deployments is future work.
- **Whole-line-holdout CV** and **temporal holdout** (train runs 0–1, test run 2) both built
  as new, explicitly opt-in paths (`src/lsm/scale_eval.py`) reusing the existing Stage 3–5
  evaluation body via one small, behaviour-preserving `train.py` refactor
  (`_run_grouped_cv`/`_evaluate_corpus`) — the default 5-fold block-CV path is byte-for-byte
  unchanged, verified by re-running the existing test suite unmodified. Temporal holdout
  surfaced its own honest negative result: severity coverage drops to ~0.71 when tested on a
  genuinely later run (run 2's grown severity values extend past what runs 0–1's calibration
  ever saw) — a real, meaningful conformal-coverage-under-shift finding, not a bug.
- **DuckDB comparison, reported honestly**: at 9.4M rows, the existing pandas-concat corpus
  read (2.7s) was actually *faster* than the new DuckDB glob read (8.3s) — stated as
  measured, not assumed to favour DuckDB by default. The small-files problem (PLAN.md's own
  "120 files is fine" example) was not hit at this scale, by design — noted, not fabricated.
- Ingest's rewritten bulk-insert path (vectorized NaN→None + dtype-cast, replacing a
  per-row `itertuples()`+`float()` loop) was a real, modest win (1.19x at 9.6M rows) — SQLite
  itself, not the row conversion, is the majority of the remaining cost at this row count,
  matching `architecture.md`'s own claim that SQLite is fine up to 10⁷ rows.
- 231/231 tests, ruff + mypy clean.

---

## Stage 7 — CI/CD and the release gate  *(partially built 2026-07-30; scoped down against real infrastructure gaps)*

**What's actually built, and works:** `.github/workflows/ci.yml` (ruff + mypy + pip-audit +
the full 146-test suite + a CLI smoke run of `generate → ingest → features`, on every
push/PR) and `.github/workflows/train.yml` (manual dispatch or a `train-*` tag: the full
`generate → ingest → features → train`, evaluating the real Stage 3/4 gates). Both were
verified by running their exact commands locally before being trusted, not just written and
assumed correct — this caught two real problems worth remembering:
- `mypy src/lsm` without `--ignore-missing-imports` fails immediately (24 errors) on missing
  stubs for pandas/pyarrow/sklearn/etc — I'd only ever tested it locally *with* that flag and
  had written the workflow *without* it. Fixed by adding `ignore_missing_imports = true` under
  `[tool.mypy]` in `pyproject.toml`, so local and CI behaviour can't drift apart on a
  forgotten flag.
- `ruff check .` over the **whole repo** (not the scoped `ruff check <file>` calls used
  throughout earlier stages) found 5 real, pre-existing dead-code issues (unused imports in
  four files, one unused local variable) that had simply never been swept. Fixed. mypy also
  caught a real (if minor) type issue: `mlflow.log_dict()` was passed a bare list where its
  stub declares `dict[str, Any]` — fixed by wrapping both call sites in a named key, which
  also makes the logged JSON artifact more self-describing.
- Also surfaced: `matplotlib` and `pydeck` were imported directly by production code
  (`train.py`, `app/streamlit_app.py`) but never declared as dependencies — relying on
  transitive installation that a long-lived shared conda env happens to provide but a fresh
  CI environment isn't guaranteed to. Added both to `pyproject.toml`.

**`ci.yml` and `train.yml` are deliberately separate workflows, not one.** `ci.yml`'s badge
tracks code correctness (lint, types, the test suite) and should stay reliably green.
`train.yml` runs the real, *statistical* promotion gate — and since Stage 3's real gate
currently does **not** pass (IsolationForest doesn't beat MAD on the real corpus), running
`train.yml` for real produces a legitimately **red** job. That is not a broken workflow —
it is the literal Stage 7 deliverable stated below, actually demonstrated: a candidate that
doesn't clear the bar is blocked, visibly, by CI. Conflating the two workflows would make
the code-quality badge red for reasons that have nothing to do with a given commit's code.

**Deliberately NOT built — needs infrastructure this project doesn't have, not a code gap:**
- `deploy.yml`, HF Spaces push, `@champion` promotion — Stage 4.5's demo app doesn't exist
  yet, so there is nothing to deploy.
- S3 push, GitHub OIDC, the MinIO integration test — no cloud account exists
  (`storage.s3.enabled: false` throughout); building the OIDC wiring and a MinIO container
  job against infrastructure nobody will ever point at a real bucket would be theatre.
- The **rollback drill** — needs at least two `feature_version` bumps to have something to
  roll back *between*; there is only `feature_version=1` today.
- **Shadow scoring** — needs an existing `@champion` release to shadow against; nothing has
  ever been promoted past `@challenger` (which itself requires Stage 3/4 to actually pass).
- Final-test-set-reuse counting (`model_run.final_test_uses`) — the column exists in the
  schema but nothing increments it yet; meaningful once there's a real promotion history to
  protect from multiple-comparisons selection.

None of this is hidden — it's stated in `train.yml`'s own header comment, not just here.

**Gate:** a deliberately degraded model is blocked by CI — **demonstrated for real**, not
hypothetically, since the actual current candidate is exactly such a case. The rollback-drill
half of the original gate is not yet meaningful (see above) and is deferred, honestly, rather
than faked with a single feature version.

**Answers:** "how does a model get to production, what stops a bad one, and how do you
undo it?" — with the honest addendum that this repo currently demonstrates the "stops a bad
one" half for real, and the "undo it" half once Stage 6 gives it something to roll back
between.

---

## Stage 8 — Growth, remaining life, and monitoring  *(built 2026-07-31)*

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

**Built and run for real 2026-07-31 (backend; app panel deferred).** `lsm forecast` and
`lsm monitor` are real commands (plus a new `lsm verify`, not in this stage's original named
list, needed to close the dig-feedback loop). 268/268 tests, ruff+mypy clean.

- **Cross-survey defect identity was already solved, for free**: `truth.build_truth_registry`
  builds one per-line registry (position fixed across a line's runs), and `train._match_all_
  indications` already matches every run against it, so a defect's `matched_source_id` is
  already stable across runs. `match_dug_indications` already returns `distance_m` (the match
  residual this stage asks to record). No new spatial-alignment logic was needed.
- **Growth-rate estimation is closed-form empirical-Bayes shrinkage** (per-defect OLS
  log-linear slope, inverse-variance-weighted pooling toward a population rate) — no PyMC/
  Stan, matching this project's practice throughout (bootstrap CIs, split conformal,
  isotonic calibration). On the real corpus's `1.15` growth law, the fitted population rate
  recovered **0.1398**, matching `ln(1.15)` to 4 decimal places, confirmed via the actual CLI,
  not just synthetic unit tests.
- **The gate passed on the first real run**: model MAE well below the no-growth baseline's,
  CI lower bound of the gap positive by a wide margin (a genuinely easy statistical target,
  since the growth law is deterministic-plus-noise, unlike Stage 3's real detection problem).
- **`truth_as_of` finally gets exercised end-to-end** for the first time anywhere in this
  project — a `--as-of` flag on `forecast` threads into `features.load_feature_corpus`'s
  already-existing (but, until now, always-"now") point-in-time filter.
- **`monitor` correctly flagged real drift** on a later run of the same line, on exactly the
  features tied to defect shape/amplitude (`fwhm_m`, `peak_asymmetry`, `decay_exponent`,
  `peak_prominence_nt`) — because severity genuinely grows across runs by construction, so a
  later run's feature distribution legitimately differs from the pooled training reference.
  The monitor doing its job, not a false alarm.
- **No schema change for the dig-feedback loop.** `indication.created_at` and
  `truth_defect.verified_at` already existed; "median days from indication to verification"
  is computed by a chainage-proximity join between them at query time, not a new table —
  deliberately avoiding a `schema_version` bump, which would have hard-invalidated every
  already-trained bundle (`bundle.load_bundle`'s own version check) for a metric that didn't
  need one.
- **App panel deliberately NOT built this pass.** `app/demo_lib.py` is single-survey-scoped
  throughout; a growth/monitoring panel needs a genuinely new per-line, multi-run loading
  layer plus new baked `serving/` artifacts — a distinct piece of work from the backend, not
  a thin UI wrapper around it. Same honest-scoping treatment as Stage 7's infra gaps.

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
