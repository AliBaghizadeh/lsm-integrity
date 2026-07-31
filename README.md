# LSM Pipeline Integrity — an ML Lifecycle Demonstrator

A physicist's end-to-end ML pipeline that turns 3-axis magnetometer + GPS survey data into
a prioritised, risk-ranked dig list — built to demonstrate **methodology and MLOps
discipline**, not physical fidelity. Real Large Stand-Off Magnetometry (LSM) data is
proprietary, so a physics-inspired synthetic generator (a magnetic-dipole forward model over
a drifting geomagnetic background) stands in for it. Every modelling and engineering
decision here is meant to be defensible on real data, and every result reported — including
the ones that don't clear their own gate — is real, not cherry-picked.

## What's implemented

| Stage | What it does | Status |
|---|---|---|
| 0 — Skeleton & reproducibility | Config layering, hashing (`git_sha`/`config_sha256`/`data_sha256`), structured logs | Done |
| 1 — Ingest & validation | 14 DQ gates, quarantine (not crash), content-hash dedup | Done |
| 1.5 — Orchestration | Dagster asset graph, partitioned by `survey_id` | Done |
| 2 — Features & gradiometer | Background removal, windowed/shape features, second sensor head | Done |
| 2.5 — Data fidelity | Real geomagnetic observatory background, label-window physics | Done |
| 2.75 — EDA | Feature correlation/redundancy/separability analysis on the real feature store | Done |
| 3 — Detection | MAD baseline vs IsolationForest, grouped CV, bootstrap CIs | Run for real — gate doesn't pass (see below), reason understood |
| 4 — Severity | LightGBM quantile regression + split conformal prediction | Run for real — **gate passes** |
| 4.5 — Demo app | Streamlit app, `demo`/`live` modes, headlessly tested | Done |
| 5 — Classification & risk ranking | LightGBM multiclass + isotonic calibration, physics-consistency SHAP gate | Run for real — recall gate doesn't pass (see below), physics-consistency gate **passes** |
| 6 — Scale rehearsal | Full pipeline at ~9.6M rows, whole-line + temporal holdout | Run for real — see gate results below |
| 7 — CI/CD | GitHub Actions (`ci.yml`, `train.yml`); cloud deploy deferred | Partial — `ci.yml`/`train.yml` built and green; `deploy.yml`/S3 deferred (no cloud account) |
| 8 — Growth & monitoring | Partially-pooled growth rate, drift monitor, dig-feedback loop | Run for real (backend) — **gate passes**; no app panel yet |

## The honest results, not the flattering ones

- **Detection (Stage 3):** IsolationForest does **not** clear its stated bar (≥0.15 recall
  improvement over a simple MAD-threshold baseline) on the real generated corpus — recall
  gap **-0.006**, 95% CI **[-0.067, 0.050]**, statistically indistinguishable from the
  baseline. That's reported as the actual result, not tuned away. What real EDA and three
  successive training runs *did* establish: the negative result is real and small, not a
  sampling-noise artifact (confirmed by re-running at increasing scale and watching the
  estimate sharpen, not just shrink), and the mechanism is understood — IsolationForest
  genuinely wastes significantly less of the dig budget on interference than MAD does (a
  bootstrap CI that excludes zero), even though that wasn't enough to clear the stated
  recall margin.
- **Severity (Stage 4):** clears its gate — 90%-nominal coverage lands at 0.920, inside the
  [0.87, 0.93] target band, MAE roughly half the global-mean baseline's.
- **Classification (Stage 5):** the SCC-recall gate does **not** pass (0.125 against a ≥0.90
  target) — traced to a real, stated limitation in the synthetic generator: defect subtype
  (SCC/weld/dent/corrosion) is assigned uniformly at random and carries no distinguishing
  physical signal today, so a classifier cannot learn it. **Interference vs. defect**, which
  *is* physically real in the generator, separates well (0.778 recall, 0.909 precision). The
  physics-consistency gate (no absolute-position feature in the top-10 SHAP contributors)
  **passes**.
- **Scale (Stage 6):** the full pipeline ran on ~9.6M rows in ~34 minutes end to end, within
  a 128 GB memory budget; two real bugs (a NaN-propagation bug that silently zeroed a whole
  fold's conformal coverage, and a dead config knob) were found and fixed only because
  testing moved past demo scale.
- **Growth (Stage 8):** clears its gate — the fitted population growth rate recovered the
  corpus's real 1.15 growth law to 4 decimal places, and beats a "no growth" baseline well
  outside its confidence interval.

## Quickstart

```bash
conda create -n ml_gpu python=3.12
conda activate ml_gpu
pip install -e ".[dev]"

python -m lsm generate    # synthetic surveys -> data/raw/*.parquet
python -m lsm ingest      # raw -> SQLite, with content hashing + DQ quarantine
python -m lsm features    # background removal + feature engineering
python -m lsm train       # trains + evaluates the real Stage 3/4/5 gates (see above)
python -m lsm forecast    # Stage 8: growth rate -> remaining life, vs a no-growth baseline
python -m lsm predict <survey_id>   # batch inference on one survey
python -m lsm serve       # launch the demo app (or: streamlit run app/demo_app.py)
```

`lsm serve` opens a Streamlit app with two modes: `demo` (precomputed results from the
committed `serving/` artifact, zero compute, cannot fail) and `live` (re-runs the real
`validate → features → score` pipeline on the spot, ~1s) — same code either way. After
retraining, `python scripts/bake_demo_assets.py` refreshes `serving/` with the new models.

## Testing

```bash
pytest              # 276 tests
ruff check .
mypy src
```

Dependency versions that `bundle.py` hard-checks at load time (`numpy`, `scikit-learn`,
`lightgbm`) are pinned in `pyproject.toml` to match what the committed `serving/` bundles
were actually trained with — a fresh install that drifted from those versions would either
fail the bundle's own version guard or silently score differently with no error.

`tests/` covers the pipeline stages, the data contract, and the demo app itself — the
Streamlit app is tested headlessly via `streamlit.testing.v1.AppTest`, which runs the
actual script through Streamlit's real runtime rather than a hand-written approximation of
it.

## Repo layout

```
src/lsm/          the pipeline: generate, ingest, validate, features, train, predict, growth, monitor
app/               the Streamlit demo app + its Streamlit-free helper modules
config/            base.yaml (hashed) + dev.yaml/prod.yaml (environment, not hashed)
serving/           committed demo artifact: model bundles + baked scenarios + manifest
scripts/           dev tools: EDA, plotting, baking the demo artifact
tests/             276 tests
```

## License

MIT — see [`LICENSE`](LICENSE).
