# LSM Pipeline Integrity — an ML Lifecycle Demonstrator

A physicist's end-to-end ML pipeline that turns 3-axis magnetometer + GPS survey data into
a prioritised heat map of likely pipeline defects — built to demonstrate **methodology and
MLOps discipline**, not physical fidelity. Real Large Stand-Off Magnetometry (LSM) data is
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
| 3 — Detection | MAD baseline vs IsolationForest, grouped CV, bootstrap CIs | Built, run for real — gate doesn't pass (see below), reason understood |
| 4 — Severity | LightGBM quantile regression + split conformal prediction | Built, run for real — **gate passes** |
| 4.5 — Demo app | Streamlit app, `demo`/`live` modes, headlessly tested | Built |
| 5 / 6 / 8 — Classification, scale rehearsal, growth/monitoring | — | Not started |
| 7 — CI/CD | GitHub Actions (`ci.yml`, `train.yml`); cloud deploy deferred | Partial |

See [`PLAN.md`](PLAN.md) for the full stage-by-stage writeup, including exact numbers, the
reasoning behind every design decision, and what's still open.

## The honest result, not the flattering one

IsolationForest does **not** clear its stated bar (≥0.15 recall improvement over a simple
MAD-threshold baseline) on the real generated corpus. That's reported as the actual result,
not tuned away. What EDA and three successive real training runs *did* establish:

- The negative result is real and small (recall gap **-0.006**, 95% CI **[-0.067, 0.050]**
  on the final run) — not a sampling-noise artifact from too few defects, confirmed by
  re-running at increasing scale and watching the estimate sharpen, not just shrink.
- The mechanism is understood, not hand-waved: a real EDA pass found ~20 of the model's 46
  features are a mutually-redundant amplitude block that separates defects from background
  almost perfectly but is mediocre at defect-vs-**interference** — the actual hard problem,
  since interference is a deliberate false-positive trap. Three other features do that job.
  Re-weighting toward them measurably improved interference rejection (a statistically
  significant effect, confirmed by a bootstrap CI that excludes zero) even though it wasn't
  enough to clear the stated recall margin.
- The severity model (LightGBM quantile regression + split conformal), evaluated on the same
  corpus, **does** clear its gate: 90%-nominal coverage lands at 0.92, inside the [0.87, 0.93]
  target band.

## Quickstart

```bash
conda create -n ml_gpu python=3.12
conda activate ml_gpu
pip install -e ".[dev]"

python -m lsm generate    # synthetic surveys -> data/raw/*.parquet
python -m lsm ingest      # raw -> SQLite, with content hashing + DQ quarantine
python -m lsm features    # background removal + feature engineering
python -m lsm train       # trains + evaluates the real Stage 3/4 gates (see above)
python -m lsm predict <survey_id>   # batch inference on one survey
python -m lsm serve       # launch the Stage 4.5 demo app (or: streamlit run app/demo_app.py)
```

`lsm serve` opens a Streamlit app with two modes: `demo` (precomputed results from the
committed `serving/` artifact, zero compute, cannot fail) and `live` (re-runs the real
`validate → features → score` pipeline on the spot, ~1s) — same code either way.

## Testing

```bash
pytest              # 181 tests
ruff check .
mypy src
```

`tests/` covers the pipeline stages, the data contract, and the demo app itself — the
Streamlit app is tested headlessly via `streamlit.testing.v1.AppTest`, which runs the
actual script through Streamlit's real runtime rather than a hand-written approximation of
it.

## Repo layout

```
src/lsm/          the pipeline: generate, ingest, validate, features, train, predict
app/               the Streamlit demo app + its Streamlit-free helper modules
config/            base.yaml (hashed) + dev.yaml/prod.yaml (environment, not hashed)
serving/           committed demo artifact: model bundles + baked scenarios + manifest
scripts/           dev tools: EDA, plotting, baking the demo artifact
tests/             181 tests
docs/               system-level architecture notes
.claude/skills/    the working invariants and conventions this project was built against
PLAN.md            the full stage-by-stage history, numbers, and reasoning
```

## Documentation

- [`PLAN.md`](PLAN.md) — the real, incremental history: every stage's deliverables,
  acceptance gate, and the actual measured result, including negative ones.
- [`LSM_PROJECT.md`](LSM_PROJECT.md) — project context, physics background, and literature
  notes.
- [`docs/production-architecture.md`](docs/production-architecture.md) — the system-level
  view: environments, orchestration, serving contract, monitoring, what's built vs designed.
- [`.claude/skills/lsm-integrity/`](.claude/skills/lsm-integrity/) — the non-negotiable
  invariants and engineering conventions this project holds itself to.

## License

MIT — see [`LICENSE`](LICENSE).
