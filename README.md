# LSM Pipeline Integrity — an ML Lifecycle Demonstrator

[![CI](https://github.com/AliBaghizadeh/lsm-integrity/actions/workflows/ci.yml/badge.svg)](https://github.com/AliBaghizadeh/lsm-integrity/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A physicist's end-to-end ML pipeline that turns walked, 3-head scalar magnetometer + GPS
survey data into a prioritised, risk-ranked dig list — ingest → validation → along-track
registration → feature engineering → detection → severity/classification → risk ranking →
growth & remaining-life, orchestrated with Dagster, tracked with MLflow, served through a
Streamlit app, gated by CI. It exists to demonstrate **MLOps and ML-pipeline-engineering
discipline**, not physical fidelity: real Large Stand-Off Magnetometry (LSM) data is
proprietary, so every dataset, model, and reported number here runs on a physics-inspired
**synthetic** generator instead — including results that don't clear their own gate, since
nothing in this project is cherry-picked.

**Rig-v2 (2026-08-06):** the generator, registration, and feature layers were rebuilt around
a more realistic instrument model — a rod carrying three **scalar** total-field heads (never
x/y/z), carried by a human walker with GPS dropout, not the single 3-axis vector head on a
fixed grid this project originally assumed. Every existing gate was re-measured honestly
against the rebuilt corpus, and a new 6-arm **ablation ladder**
(`scripts/ablation_ladder.py`) answers a real instrumentation question — "improve the
software, or upgrade the hardware?" The honest answer: one software step (a first-difference
across heads) gives a real, statistically significant recall gain, but a genuine hardware
upgrade to full vector output still buys roughly seven times more than the entire software
gain combined.

**The clean-room experiment (2026-08-10):** the detection model's failure had been established as
*information*-limited rather than data-limited — it survives 6× more data, three grouping schemes,
and both the unsupervised and supervised paradigms. `scripts/cleanroom_experiment.py` asks which
information is missing, by generating a counterfactual corpus: an isolated test spool with defects
and nothing else — no external interference, no girth welds, chainage from a tape measure. There,
the same detector beats its baseline by **+0.199 recall [0.139, 0.255]**, the first time in this
project it beats that baseline at all, and localises **10.8× better** (812 cm → 75 cm). So the
confounders, not the scalar rig's sensing physics, are what break detection. The second half of the
experiment is the one that matters for deciding anything, though: a model *trained* on that clean
corpus and evaluated on realistic data is **worse** than one trained on realistic data, and cannot
recognise interference at all, having never seen it. Controlled acquisition is a diagnostic
instrument here, not a training corpus — a distinction the flattering half of the experiment would
have hidden.

![Project components: pipeline stages, infrastructure, and consumers](img/project-components.png)
*Pipeline stages, infrastructure, and consumers.*

## Contents

- [Repo layout](#repo-layout)
- [Designed for production, built on synthetic data](#designed-for-production-built-on-synthetic-data)
- [Quickstart](#quickstart)
- [Demo vs. live mode](#demo-vs-live-mode)
- [Testing & CI](#testing--ci)
- [Development](#development)
- [Model card](#model-card)
- [Data & security](#data--security)
- [License](#license)

## Repo layout

```
.
├── src/lsm/                 the pipeline
│   ├── generate.py          synthetic survey generator -- Rig-v2: 3-head scalar rig, walker
│   │                         physics, GPS dropout (real background trace option preserved)
│   ├── ingest.py            raw -> SQLite, content-hashed, idempotent
│   ├── validate.py          14 data-quality checks, quarantine on hard failure
│   ├── registration.py      Rig-v2: GPS dead-reckoning + girth-weld-comb detection -> chainage_m
│   │                         (raw no longer carries a usable chainage column)
│   ├── features.py          detrend, per-head first/second difference (g1/g2), stand-off
│   │                         inversion, window + peak-shape features
│   ├── indications.py       row scores -> clustered indications, severity/classify attachment
│   ├── models/               anomaly.py (MAD/IsolationForest), severity.py (LightGBM CQR),
│   │                         classify.py (LightGBM multiclass)
│   ├── train.py              grouped CV, gate evaluation, model persistence
│   ├── predict.py            batch inference on one survey -> indications + GeoJSON
│   ├── growth.py             partially-pooled growth rate -> remaining life
│   ├── monitor.py            drift monitoring (PSI/KS vs. training reference)
│   ├── dagster_defs.py       the asset graph
│   └── bundle.py, db.py, config.py, schemas.py, hashing.py   the contracts everything else relies on
├── app/                     the Streamlit demo app
│   ├── demo_app.py           entry point -- landing screen, workflow diagram, model performance,
│   │                         signal views, ranked indications, corrupted-survey refusal
│   ├── demo_lib.py            Streamlit-free seam: demo/live mode logic, testable without a browser
│   └── chart_utils.py, map_utils.py   Altair/pydeck chart builders
├── config/                  base.yaml (hashed into config_sha256) + dev.yaml/prod.yaml (not hashed)
├── serving/                 baked demo artifact: model bundles + 4 demo scenarios + manifest
├── scripts/                 EDA, plotting, the scale rehearsal, the Stage D ablation ladder,
│                            the clean-room counterfactual experiment, baking the demo artifact
├── tests/                   312 tests
└── .github/workflows/       ci.yml (lint+type+test, every push) / train.yml (the real promotion gate)
```

## Designed for production, built on synthetic data

Every engineering decision here is made the way it would need to be made against real data,
even though the data itself is synthetic:

- **A declared data contract, not a loose CSV.** Schema, dtypes, units, and null policy
  declared once (`schemas.py`) and validated at every boundary.
- **Idempotent, content-hashed ingest.** The same bytes twice is a no-op; a changed hash under
  an existing key is a hard error, never a silent overwrite.
- **14 real data-quality checks**, quarantine — not crash — on failure. A pipeline that halts
  a nightly run over one bad sensor gets switched off by its own operators.
- **Everything that can silently go stale is versioned**: `feature_version`, `schema_version`,
  and `pipeline_version` — the actual deployable unit (up to four models plus a feature
  version, promoted and rolled back together, never one model in isolation).
- **Hard version guards, not soft warnings.** Loading a model bundle trained against a
  different `numpy`/`scikit-learn`/`lightgbm` fails loudly at load time — a silent
  minor-version difference is exactly the failure mode that doesn't show up until predictions
  are already wrong.
- **Grouped, leakage-aware evaluation.** Splits by line/block, never randomly; point-in-time
  correctness enforced by an as-of join, not just a spatial split.
- **Real orchestration, not a notebook.** A Dagster asset graph partitioned by survey, with a
  resumable backfill demonstrated by killing a run mid-flight and confirming no reprocessing.
- **A real CI/CD promotion gate.** `train.yml` runs the actual statistical gate and blocks a
  candidate that doesn't clear it — demonstrated for real, since the current candidate
  genuinely doesn't clear it, not hypothetically.

What's explicitly **not** built, and why: cloud artifact storage (S3) and a real deploy
pipeline are designed but not wired up, because no cloud account exists for this
demonstrator — standing up real infrastructure nobody will ever point at a real bucket would
be theatre, not rigor. The gap between designed and built is stated wherever it matters, not
glossed over.

## Quickstart

### Minimal — just see the demo

The committed `serving/` artifact ships pre-baked model outputs, so the demo needs no data
generation, training, or GPU — just an install:

```bash
pip install -e .
python -m lsm serve   # opens the Streamlit app; pick "demo" mode in the UI
```

![Ranked indications by risk score, colored by predicted defect type](img/risk-score-analysis.png)
*What you'll see: the demo app's ranked-indications view, dig-budget-limited, sorted by
`risk_score` (calibrated P(defect) × severity × a stated consequence proxy).*

### Full dev setup

```bash
conda create -n ml_gpu python=3.12
conda activate ml_gpu
pip install -e ".[dev]"
```

(the env name is historical — nothing in this repo requires a GPU; LightGBM, IsolationForest,
and scikit-learn all run on CPU)

...or without conda:

```bash
python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

Then run the pipeline end to end:

```bash
python -m lsm generate    # synthetic surveys -> data/raw/*.parquet
python -m lsm ingest      # raw -> SQLite, with content hashing + DQ quarantine
python -m lsm features    # background removal + feature engineering
python -m lsm train       # trains + evaluates the real detection/severity/classification gates
python -m lsm forecast    # growth rate -> remaining life, vs. a no-growth baseline
python -m lsm predict <survey_id>   # batch inference on one survey
python -m lsm serve       # launch the demo app (or: streamlit run app/demo_app.py)
```

`generate`, `ingest`, and `features` run in well under a minute on the demo corpus (a handful
of synthetic surveys). `train` is the heavy step — it fits and cross-validates multiple models
(anomaly, severity, classification) — budget the most time for it; exact wall-clock depends on
your machine.

## Demo vs. live mode

`lsm serve` opens a Streamlit app with two modes, same code path either way:

- **`demo`** — precomputed results from the committed `serving/` artifact. Zero compute,
  cannot fail — the right first click.
- **`live`** — re-runs the real `validate → features → score` pipeline on the spot, roughly a
  second per scenario.

After retraining, `python scripts/bake_demo_assets.py` refreshes `serving/` with the new
models.

**Why the `Train` GitHub Action can be red on purpose:** `train.yml` runs the actual Stage 3/4
statistical promotion gate, and its exit code *is* the gate — it fails if the candidate
detection model doesn't beat the MAD baseline on the synthetic corpus. A red `Train` run
doesn't mean broken code; it means "the current candidate hasn't cleared the bar yet," and
demonstrating that block is itself part of the point — see [`serving/model_card.md`](serving/model_card.md)
for the actual gate numbers behind it. The `CI` badge at the top of this README (lint,
type-check, tests) is the one that reflects code health; `train.yml` reflects model quality
and is expected to fail until a candidate clears the gate.

## Testing & CI

```bash
pytest              # unit, leakage, determinism, bundle round-trip, golden regression
ruff check .
mypy src
python -m pip_audit  # dependency vulnerability scan — informational in CI, not blocking
```

Dependency versions that `bundle.py` hard-checks at load time (`numpy`, `scikit-learn`,
`lightgbm`) are pinned in `pyproject.toml` to match what the committed `serving/` bundles were
actually trained with — a fresh install that drifted from those versions would either fail the
bundle's own version guard or silently score differently with no error.

The Streamlit app is tested headlessly via `streamlit.testing.v1.AppTest`, which runs the
actual script through Streamlit's real runtime rather than a hand-written approximation of it.

## Development

Each pipeline stage is idempotent and safe to run in isolation, so you don't need to rerun the
whole pipeline to iterate on one part of it:

```bash
python -m lsm generate
python -m lsm ingest
python -m lsm features
python -m lsm predict <survey_id>
```

Run the Streamlit app headlessly (no browser needed), the same way the test suite does:

```python
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("app/demo_app.py").run()
```

Where to extend things:

- New data-quality checks → `src/lsm/validate.py`
- New features → `src/lsm/features.py`
- New or alternate models → `src/lsm/models/` (`anomaly.py`, `severity.py`, `classify.py`),
  wired into `train.py`

## Model card

[`serving/model_card.md`](serving/model_card.md) is the real, generated model card for the
bundles committed to `serving/`: gate results (pass/fail) for detection, severity,
classification, and growth, with bootstrap confidence intervals, provenance hashes, and
stated scope limits — not a hand-written summary.

## Data & security

Real LSM survey data is proprietary and was never used anywhere in this project. Every
dataset, model, and reported number here comes from a physics-inspired synthetic generator (a
magnetic-dipole forward model, sensed by a walked 3-head scalar rig over a drifting geomagnetic
background, with an optional real USGS observatory trace used only for the background
component). No proprietary or customer data is present in this repository.

## License

MIT — see [`LICENSE`](LICENSE).
