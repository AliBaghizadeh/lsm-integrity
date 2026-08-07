# Data contract

The schema, dtypes, keys, units and versioning rules that every stage must honour. This is
declared **once** and enforced at every boundary — ingest, feature output, model input,
prediction output — not re-checked ad hoc in each module.

Implementation: one Pandera (or pyarrow) schema per table in `src/lsm/schemas.py`,
imported by `validate.py`, `features.py` and `predict.py`. A boundary that does not
validate is a boundary where skew will appear.

---

## 1. Keys — never join on a float

`0.5 * 3 != 1.5` in binary. A float join key produces silent partial misses between the
reading table and any feature table computed in a different code path.

**The physical key is `(survey_id, sample_idx)` with `sample_idx` an `int32`.** Under Rig-v2
(`schema_version` 3) `sample_idx` is a dense, monotonic **time**-sample counter (the walk is
sampled at a fixed rate, `walk.sample_rate_hz`, not on a distance grid) rather than a
distance-grid index — but it stays the sole physical key regardless; nothing about joins or
`content_sha256` changes.

`chainage_m` is **no longer a raw column at all**, and no longer a simple derivation. Under the
pre-Rig-v2 model it was `chainage_m = sample_idx * step_m`; under Rig-v2 a human walks the line
at irregular speed, so there is no fixed step to multiply by. Instead `chainage_m` is an
**output of Stage B registration** (`src/lsm/registration.py`: GPS dead-reckoning through
dropout, then locked onto the recovered girth-weld lattice), written only to the **feature
layer**, computed once per survey by `pipeline.run_feature_pipeline`. Raw carries
`chainage_true_m` instead — a **truth-tier** column (the generator's own exact position, same
tier as `defect`/`interference`) that may be used to *score* registration but must never become
a feature (enforced by `features.py`'s `TRUTH_DENYLIST` / `_assert_no_truth_leakage`) — and,
only as an interim convenience, a naive constant-speed `chainage_provisional_m`, deliberately
*not* named `chainage_m` so nothing downstream mistakes it for the physically-final value.
Whichever column is in play, the rule is unchanged: never a key, a join column, or a grouping
boundary.

Consequences to keep consistent:
- Chainage blocks for `GroupKFold` are computed from `sample_idx`, integer division.
- Group key is **`(line_id, block)`** — never `block` alone. Two overlapping surveys of the
  same stretch must land in the same fold, or the overlap is leakage.
- Defect matching across surveys uses a tolerance on `chainage_m`, and that tolerance is a
  config value, not a magic number.

## 2. Dtypes and precision

Chosen from the physics, not from habit. **Updated for Rig-v2 (`schema_version` 3)** — verified
against `src/lsm/schemas.py`'s current `RawReadingSchema`/`FeatureFrameSchema`, not carried
forward from the pre-Rig-v2 contract.

| Column | Parquet dtype | Reasoning |
|---|---|---|
| `survey_id`, `line_id` | `string` (dictionary-encoded) | low cardinality |
| `run_id` | `int16` | |
| `sample_idx` | `int32` | exact key; now a dense **time**-sample counter under Rig-v2, not a distance-grid index (see §1) |
| `t_s` | `float64` | elapsed seconds since survey start — the walk is time-sampled, not distance-sampled, so this (not chainage) is the raw table's regular axis |
| `chainage_true_m` | `float64`, **not nullable** | truth-tier only (§1) — the generator's own exact position; never a raw contract column before Rig-v2, since there was previously no distinction between "true" and "derived" chainage |
| `chainage_m` | `float64`, **feature-layer only** — no longer a raw column | a Stage B registration output (§1); never a key |
| `lat`, `lon` | **`float64`, nullable** (was mandatory pre-Rig-v2) | float32 ULP at 47° N ≈ 3.8×10⁻⁶ ° ≈ **0.42 m**, still too coarse even before considering nulls. Nullable because GPS genuinely drops out in poor sky view (`GpsConfig.dropout_rate`) and the gap is written as NaN, not synthesised through — a survey with mandatory lat/lon could not represent the instrument's actual failure mode. |
| `b_lo_nt`, `b_mid_nt`, `b_hi_nt` | `float32` storage | total-field **magnitude** per head (never negative — see §4's range note), replacing the removed `bx_nt`/`by_nt`/`bz_nt`/`bx2_nt`/`by2_nt`/`bz2_nt` vector-axis columns. ULP at ~45 000 nT ≈ 0.004 nT, finer than any magnetometer's resolution. |
| `girth_weld` | `int8` (0/1) | periodic joint train ground truth, tracked separately from `defect` — a weld is a real, strong, non-cancelling source, but not damage |
| residuals, window features | `float32` | halves Stage-6 memory |
| `defect` | `int8` | |
| `defect_type` | `dictionary<string>`, **pinned category order** | see §4 |
| `severity_smys` | `float32` | |
| `*_at` timestamps | `timestamp[us, tz=UTC]` | §3 |

**Store float32, compute float64.** Applies unchanged to the new `b_lo/mid/hi_nt` raw columns
and to `r_lo_nt`/`r_mid_nt`/`r_hi_nt` in the feature layer. Storage precision is not arithmetic precision:
detrend least-squares fits, cumulative window sums and gradient differences accumulate
error. Cast to float64 inside the transform, cast back on write. State this explicitly —
it is the kind of detail a physicist will check.

> **SQLite note:** SQLite `REAL` is always 8-byte float64 — there is no float32 storage
> class. The float32 decision applies to the Parquet layer, which is where the row counts
> live. Do not claim memory savings from the SQLite side.

## 3. Units, naming and time

- Every physical column carries its unit as a suffix: `_m`, `_nt`, `_deg`, `_s`. No
  exceptions, no bare `chainage`. Unit conversion happens once, at ingest.
- All names `lower_snake_case`. Sensor heads lowercase (`b_lo_nt`, `b_mid_nt`, `b_hi_nt` —
  Rig-v2's 3 total-field heads, replacing the pre-Rig-v2 vector axes `bx_nt`/`by_nt`/`bz_nt`),
  matching the DB.
- **All timestamps UTC, ISO-8601, `_at` suffix.** Naive local times are a data-quality
  failure, not a formatting preference — surveys are compared across years and a DST
  boundary silently reorders them.
- `severity_smys` is a percentage of SMYS in [0, 100]; it is a *proxy* in this project and
  the column comment must say so.

## 4. Categoricals and class order

`defect_type ∈ {scc, weld, dent, corrosion, interference, none}` is a **closed enum
declared in `schemas.py`**, and the ordered category list is serialised into the model
bundle. Never let pandas infer category order from whatever happened to be in the training
frame — an alphabetical reshuffle silently permutes the classifier's output columns, and
the failure looks like a mediocre model rather than a bug.

An unseen category at inference is a hard `fail`, not a silent `other`.

## 5. Null semantics — explicit, per column

Never impute silently. LightGBM handles NaN natively; that is a reason to *pass* NaN
through, not a reason to stop thinking about what it means.

| Situation | Meaning | Policy |
|---|---|---|
| NaN in a raw head reading (`b_lo_nt`/`b_mid_nt`/`b_hi_nt`) | sensor dropout | **hard fail** — quarantine the survey |
| NaN in `lat`/`lon` | **GPS dropout (poor sky view), Rig-v2** — expected, not a sensor fault | **not** a hard fail — a real, modelled acquisition state (`GpsConfig.dropout_rate`); Stage B registration reconstructs `chainage_m` through the gap by dead reckoning, and `gps_locked=0` on the affected feature rows carries the flag forward |
| NaN in a window feature at survey ends | rolling window truncation | expected — see below |
| NaN in `severity_smys` | not a defect | distinct from `0.0`; use NaN, not 0 |

Pre-Rig-v2 this table also carried a `bx2_nt NULL` row ("no second sensor head fitted") for the
old optional gradiometer. There is no equivalent under Rig-v2: `rig: scalar` always has exactly
3 heads (`ArrayConfig.n_heads` is a documented physical fact there, not a free dial), so
`b_lo_nt`/`b_mid_nt`/`b_hi_nt` are never independently optional the way the old second head was.

**Edge policy.** Rolling windows produce NaN in the first and last *w*/2 metres. Dropping
those rows silently discards the pipe ends; padding fabricates data. The rule: keep the
rows, mark `dq_flag='edge'`, **exclude them from metric computation**, and still score
them but with a widened interval. Report how much length is edge-affected — for a 25 m
window on a 2 km line it is 1.25%, and on real short segments it can be much worse.

## 6. Hashing, identity and deduplication

Two hashes with different jobs. Conflating them is the usual mistake.

| Hash | Over | Job |
|---|---|---|
| `file_sha256` | the raw file bytes | provenance and immutability — "this exact artifact" |
| `content_sha256` | **canonicalised** content | semantic identity — "this data, however exported" |

Canonicalisation for `content_sha256`: select the contract columns in declared order, sort
by `sample_idx`, cast to contract dtypes, round floats to a declared precision, hash the
Arrow buffers. A re-export with different float formatting or column order then produces
the **same** content hash. Use BLAKE3 or xxh3-128 for the bulk pass — this is integrity and
dedup, not an adversarial setting; keep sha256 for the small artifacts people cite.

**Ingest is idempotent on `content_sha256`:** same hash → no-op. A *different*
`content_sha256` for an existing `(line_id, run_id)` is an **error**, never an overwrite.
Raw data is immutable; a genuine correction gets a new `run_id` or a superseding revision.

**Duplicate detection runs at three levels:**
1. *Exact* — `content_sha256` collision across surveys.
2. *Row-level* — duplicate `sample_idx` within a survey. Enforced by the primary key, and
   surfaced as a named DQ check so the failure message is useful.
3. *Partial overlap* — the hard one. Two surveys of the same `line_id` whose
   `[chainage_start_m, chainage_end_m]` intervals intersect: a re-run over a bad section, or
   one survey split across files. Neither hash catches it. Check interval intersection,
   then cross-correlate the detrended residual in the overlap; high correlation means the
   same physical acquisition. **This is a leakage risk, not just untidiness** — hence the
   `(line_id, block)` group key in §1.

**Label-level deduplication** is entity resolution, not hashing: the same defect reported by
both ILI and an excavation must be matched on `(line_id, chainage_m ± tolerance)`, not on
exact equality.

**Hashes also buy three things beyond dedup:**
- **Stable fold assignment** — `blake3(line_id) % n_folds` instead of positional
  assignment, so adding lines to the archive does not reshuffle existing fold membership
  and invalidate historical comparisons.
- **Feature-store cache key** — `content_sha256 + feature_version + relevant config hash`;
  skip recompute on a hit. This is what makes Stage 6 re-runs tolerable.
- **Training-set identity** — `data_sha256` is a Merkle-style hash over the sorted member
  `content_sha256` list, so one value pins the exact training corpus.

## 7. Versioning — three independent versions

| Version | Changes when | Enforced by |
|---|---|---|
| `schema_version` | a table's columns or dtypes change | stored on `survey`; migration script required |
| `feature_version` | `features.py` changes behaviour | in the feature-store path **and** in the bundle; a bundle refuses to load against a mismatch |
| `model_version` | a model is trained | `model_run` table + MLflow registry |

`feature_version` is the one people forget, and it is the nastiest: edit `features.py` and
every stored feature is silently stale, producing training/serving skew that no data check
can see because the *data* is fine. Bumping it invalidates the feature cache and requires a
documented backfill decision — recompute, or pin old models to old features. Both are
acceptable; leaving it implicit is not.

## 8. Ground truth is slowly-changing

An excavation can reclassify or re-measure a defect. `truth_defect` therefore carries
`revision`, `valid_from`, `valid_to` and `source`, and corrections **insert** a new
revision rather than updating in place. Without this, a label correction silently rewrites
history and last quarter's reported metrics change retroactively — which is exactly the
kind of thing that destroys trust in a model's track record. Evaluations pin an
`as_of` timestamp.

## 9. Parquet physical layout

- One file per survey under `raw/line_id=…/run_id=…/`, **zstd** compression.
- Rows sorted by `sample_idx`; write min/max statistics so chainage-range predicates push
  down.
- Dictionary-encode `survey_id`, `line_id`, `defect_type`.
- Row-group size targeted at ~64–128 MB *at scale*; at demo scale a survey is a few hundred
  KB and a single row group. Note the small-files problem honestly: 120 tiny files is fine,
  10⁵ is not, and the production answer is periodic compaction into line-level files.

## 10. Data classification

Synthetic here, so nothing is sensitive. State the real-world position anyway, because it
signals having thought about deployment at a customer rather than on a laptop:

> Real pipeline route coordinates are **critical-infrastructure data**. In most
> jurisdictions and most customer contracts they do not leave the customer's environment,
> which pushes you toward training in the customer's VPC or on-prem, shipping model
> artifacts rather than data, and keeping GPS out of any telemetry or experiment-tracking
> payload. `lat`/`lon` are therefore treated as restricted columns: excluded from MLflow
> artifacts, and excluded from any drift-monitoring export.
