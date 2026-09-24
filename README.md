# LSM Pipeline Integrity - ML Lifecycle Demonstrator

[![CI](https://github.com/AliBaghizadeh/lsm-integrity/actions/workflows/ci.yml/badge.svg)](https://github.com/AliBaghizadeh/lsm-integrity/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

An end-to-end machine-learning pipeline for walked, three-head scalar magnetometer surveys. It
turns synthetic survey data into validated, registered, risk-ranked pipeline indications and a
dig-budget-aware output. The project demonstrates reproducible ML and data-engineering patterns:

`generate -> ingest -> validate -> register -> features -> detect -> classify -> rank -> monitor`

All survey data, labels, models, and reported metrics are synthetic. The repository is a
technical demonstrator, not a claim of field performance.

![Pipeline components and consumers](img/project-components.png)
*Pipeline stages, infrastructure, and consumers.*

## Current state

- A committed `serving/` artifact lets the Streamlit demo run without data generation or model
  training.
- The pipeline includes data contracts, content-hashed ingest, quarantine-oriented validation,
  along-track registration, feature engineering, grouped evaluation, model bundles, monitoring,
  and a Dagster asset graph.
- The current synthetic benchmark deliberately reports failed detection, severity, and
  classification promotion gates where the evidence does not meet the configured thresholds.
  The growth comparison passes on its synthetic benchmark.
- Detailed metrics, confidence intervals, provenance hashes, and scope limits are in the
  generated [model card](serving/model_card.md).
- Local storage and GitHub Actions are wired up. Cloud object storage and cloud deployment are
  not included.

## Repository layout

```text
src/lsm/                 pipeline stages and contracts
app/                     Streamlit demo and chart/map helpers
config/                  base, development, production, and scale settings
serving/                 committed demo bundles, scenarios, and model card
data/reference/          small public reference background traces
scripts/                 generation, evaluation, plotting, and asset tools
tests/                   unit, integration, leakage, and regression tests
img/                     README and demo figures
.github/workflows/       lint, type-check, test, and training workflows
```

## Quickstart

Requires Python 3.12 or newer.

### Run the demo

```bash
pip install -e .
python -m lsm serve
```

Open the Streamlit app and select `demo` mode. The committed serving artifact supplies the
precomputed results, so this path does not require training or a GPU.

![Risk-ranked indications](img/risk-score-analysis.png)
*Example risk-ranked output from the demo artifact.*

### Run the pipeline locally

For the development dependencies:

```bash
pip install -e ".[dev]"
```

Then run the stages as needed:

```bash
python -m lsm generate
python -m lsm ingest
python -m lsm features
python -m lsm train
python -m lsm forecast
python -m lsm predict <survey_id>
python -m lsm serve
```

Generated data and reports are written to ignored local directories. After retraining, refresh the
committed demo artifact with:

```bash
python scripts/bake_demo_assets.py
```

## What is implemented

- **Data contract and validation:** schemas, units, null policy, range checks, duplicate checks,
  sensor checks, and quarantine for hard failures.
- **Reproducible ingest:** content hashes make repeated ingestion idempotent and prevent silent
  replacement of changed inputs.
- **Registration:** GPS dead reckoning and weld-comb alignment produce a common along-track
  coordinate.
- **Features:** background removal, per-head first/second differences, stand-off inversion,
  window statistics, and peak-shape features.
- **Models:** MAD and IsolationForest anomaly detection, LightGBM severity estimation with
  conformal intervals, calibrated multiclass classification, and partially pooled growth.
- **Evaluation:** grouped cross-validation, leakage checks, bootstrap intervals, promotion gates,
  and a baseline comparison for each modeled task.
- **Serving:** versioned bundles, batch prediction, GeoJSON indications, drift monitoring, and a
  Streamlit demo that consumes served output.

## Demo versus live mode

The app supports two modes through the same consumer-facing interface:

- `demo` reads the committed `serving/` artifact and is the fastest way to explore the project.
- `live` validates and scores a selected survey with the current local pipeline.

The demo also includes a deliberately corrupted survey so the validation and refusal path can be
seen without supplying external data.

## Testing and CI

```bash
pytest
ruff check .
mypy src
python -m pip_audit
```

The Streamlit app is exercised headlessly with `streamlit.testing.v1.AppTest`. The CI workflow
runs the code-quality and test checks; the training workflow evaluates the configured model gates.

## Data and scope

The repository contains synthetic surveys, small public reference background traces, generated
demo artifacts, and code. It does not require or include proprietary survey data, customer data,
personal documents, or field inspection outcomes. Consequence weights used in `risk_score` are
explicit engineering placeholders for the synthetic demo and must be replaced before any real
operational use.

The committed metrics should be read with their confidence intervals and limitations. They are
useful for demonstrating pipeline behavior and honest evaluation, not for estimating performance
on an unseen operating environment.

## License

MIT - see [LICENSE](LICENSE).
