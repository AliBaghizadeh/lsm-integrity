# Production architecture — LSM integrity pipeline

How this would run at ROSEN scale: thousands of surveys, years of archive, ground truth
that arrives months late. The demonstrator builds a subset of this (§10); the rest is
designed, costed and written down rather than hand-waved.

**Design centre:** the primary workload is **backfill over an existing archive**, not
steady-state scoring. Everything below follows from that.

---

## 1. System at a glance

```
S3 raw/ ──sensor──▶ ┌─────────── Dagster asset graph, partitioned by survey_id ───────────┐
                    │ ingested_survey → dq_report → survey_features → indications        │
                    │                      │              (fv=n)         │               │
                    │                      └── quarantine/               ├─▶ gis_export  │
                    │                                                    └─▶ drift_report│
                    └────────────────────────────────────────────────────────────────────┘
                                    │                          │
                          SQLite/Postgres registry      MLflow (models, metrics)
                                    │
                    ┌───────────────┴───────────────┐
              GIS / integrity system          Streamlit app (a CONSUMER, not the server)
```

Dagster's software-defined assets map 1:1 onto this: each box is an asset, `survey_id` is a
dynamic partition, `feature_version` is the asset's code version, and **backfill is a
first-class partition backfill** rather than a bespoke script. That fit is why Dagster over
Airflow here.

## 2. Environments and config layering

Three environments, three separate buckets / databases / MLflow experiments. Three config
layers with different lifecycles — conflating them breaks reproducibility:

| Layer | Contents | Lifecycle | Hashed into the run? |
|---|---|---|---|
| **Code config** `config/base.yaml` | detrend degree, windows, quantiles, gate thresholds | git commit | **yes** — this is `config_sha256` |
| **Environment config** `config/{dev,staging,prod}.yaml` | bucket, DB URI, MLflow URI, concurrency | per environment | no |
| **Runtime params** | which partition, which `pipeline_version` | per invocation | logged, not hashed |

Only the code layer feeds `config_sha256`. Otherwise the same code and data produce a
different reproducibility hash in prod than in dev, which silently voids the guarantee.

**Runtime parity is enforced, not recorded.** Training (`ml_gpu`, Python 3.12.13), the local
venv (3.13) and the app runtime are three dependency graphs today. Production pins one
Python version and one lockfile, training runs in a container referenced **by digest** in
the bundle, and `bundle.load()` **hard-fails** on a LightGBM/numpy/sklearn version mismatch
rather than merely recording it. A silent minor-version difference changes predictions with
no error — the worst class of failure.

## 3. Orchestration, triggering and backfill

| Concern | Production |
|---|---|
| Trigger | S3 sensor on `raw/` for new surveys; nightly schedule as a safety net |
| Unit of work | one survey = one partition; idempotent and independently retryable |
| Retries | 3× exponential backoff on transport/IO; **zero** on DQ failure — a bad survey is not a flake |
| Backfill | Dagster partition backfill with bounded concurrency; resumable — a failure at survey 4,000 of 6,000 costs one partition, not the run |
| Concurrency | feature computation fans out across cores; **registry writes are serialised** (SQLite is single-writer; Postgres above ~10⁷ rows or any multi-writer ingest) |
| Isolation | one poison survey quarantines itself and the run continues |

## 4. Serving contract — batch, and distinct from the demo app

The serving layer is the **batch scoring job**. The Streamlit app is a *consumer* of its
output. Conflating the two is a category error.

| | |
|---|---|
| Trigger | asset materialisation on new/updated `survey_features` |
| Output | `indication` rows + `reports/<survey_id>/indications.geojson` (versioned schema) |
| Consumer | GIS / integrity management system |
| Contract | output schema is versioned and covered by a **consumer contract test** in CI; a breaking change requires a new schema version, not an edit |
| Idempotency | re-scoring the same `(survey_id, pipeline_version)` replaces its rows exactly |
| Latency SLO | indications available < 24 h after survey upload; < 30 s compute per 20 km survey |

## 5. Release and rollback

The deployable unit is a **`pipeline_version` release manifest**, not an individual model —
an indication is produced by up to four models plus a feature version, and all of them must
be pinned together for its lineage to be answerable.

```yaml
pipeline_version: 2026.07.29-a1b2c3d
models: {anomaly: anom-…, severity: sev-…, classify: cls-…, growth: grw-…}
feature_version: 3
schema_version: 2
container_digest: sha256:…
```

- **MLflow 3 removed model stages.** Promotion sets the **`@champion` alias**;
  `@challenger` marks the shadow candidate. Aliases make promotion and rollback atomic
  pointer moves with no state mutation.
- **Rollback = repoint `@champion` to the previous `pipeline_version`.** Bundles are
  immutable, so a rollback target always exists.
- **Rollback is drilled in CI**, because the usual way it fails is that the old bundle no
  longer loads against current code — which our own `feature_version` guard would cause.
  Policy: the last **N = 3** feature versions stay loadable, and CI proves it.
- **Re-scoring policy on promotion:** historical surveys are *not* automatically re-scored;
  every analytical query filters on `pipeline_version`. Re-scoring the archive is a
  deliberate, scheduled backfill. Without this rule, growth forecasting silently compares
  outputs from different models.

## 6. Delayed labels — the operating model

Ground truth arrives months late and is **censored by our own past ranking**: we only dig
where we predicted. This is the defining constraint of the domain and it drives three
mechanisms.

1. **Vintage backtesting.** Every evaluation pins `truth_as_of` and reconstructs what was
   known on that date, then scores forward. Reported metrics are always as-of a stated date.
2. **Shadow scoring.** Champion and challenger both score every live survey. Since outcome
   comparison must wait months, the near-term signal is **disagreement analysis**: change in
   indication count per km, rank churn at the dig budget, and whether disagreements
   concentrate in DQ-warned surveys. This is the only pre-label validation available, and it
   works from day one.
3. **Feedback loop and exploration.** Each excavation writes a `truth_defect` revision.
   Because labels are censored, a small fraction of the dig budget should be allocated by
   *uncertainty × consequence* rather than by rank — otherwise the model can only ever
   confirm itself. Tracked metric: **median days from indication to verification**, which
   bounds how fast anything can be learned.

## 7. Monitoring, alerting, runbook

**Model observability:** feature PSI/KS vs the bundle's training reference (>0.2 warn,
>0.3 block), indications-per-km vs training rate, background regime shift per line,
measured conformal coverage as verifications arrive.

**System observability:** structured JSON logs correlated by `survey_id` / `run_id` /
`pipeline_version`; per-asset duration, rows/sec, failure count, quarantine rate. Alerts
route to the integrity-analytics on-call channel — an alarm with no destination is not
monitoring.

| Symptom | First check | Action |
|---|---|---|
| Survey quarantined | `dq_report` check name | Analyst review; re-acquire if sensor fault |
| Quarantine rate > 5% | Is it one line or one scanner? | Suspect acquisition, not data |
| PSI > 0.3 | Background regime vs that line's history | Block scoring; usually a new/recalibrated scanner |
| Bundle load fails | `feature_version` / library digest | Blocked deploy — rebuild or roll back |
| Coverage below nominal | Recent verification residuals | **Refit conformal quantiles, do not retrain** |

## 8. Evaluation discipline

- Grouped splits by `(line_id, block)`; spatial **and** temporal holdouts.
- **Point-in-time correctness:** every feature for a survey must be computable from data
  available at that survey's `surveyed_at`. Enforced by an as-of join and a test — spatial
  grouping does not catch temporal leakage.
- **Final test set is touched rarely and its use is counted.** Repeatedly promoting against
  a fixed holdout selects on holdout noise; the release process itself is subject to
  multiple comparisons. Rotate as new lines arrive.
- All headline metrics reported as bootstrap CIs over groups; gates compare intervals.

## 9. SLOs, retention, cost

| SLO | Target |
|---|---|
| Survey ingest + validate | < 2 min |
| Indications available | < 24 h after upload |
| Backfill throughput | ≥ 500 surveys/hour |
| Quarantine rate | < 5% (alert above) |
| Scheduled GIS export delivered | 99% of runs |

**Retention:** raw surveys indefinitely (immutable — they *are* the asset; Glacier IR after
12 months); features 90 days then recompute on demand (they're derivable); quarantine 12
months; indications indefinitely (audit trail); MLflow artifacts — champion versions
forever, others pruned at 12 months.

**Cost:** ~500 surveys/year at 20 km and 0.5 m spacing is ~2×10⁷ rows/year, well under
1 GB/year compressed. **Storage is not the cost.** The cost is compute during archive
backfill and analyst time reviewing indications — which is exactly why *false-dig rate*,
not accuracy, is the metric the system optimises.

## 10. Built here vs designed here

| Built in the demonstrator | Designed and documented only |
|---|---|
| Dagster asset graph, retries, one real backfill | Airflow/K8s or AWS Batch at full scale |
| Config layering, env separation (dev/prod) | Staging environment with its own infrastructure |
| `pipeline_version` manifest, MLflow aliases, rollback drill | Blue/green infrastructure |
| Shadow scoring + disagreement report | Multi-year vintage backtest |
| Point-in-time test, bundle version assertion | Full consumer contract with a real GIS system |
| Structured logging, drift report | Paging, on-call rotation, dashboards |

The distinction is deliberate: a demonstrator that tries to build all of this ships nothing.
What matters is that every box on the right has a named decision behind it, not a gap.
