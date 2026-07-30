# Interview drills

Questions ROSEN can reasonably ask, the stage of the project that answers each, and the
honest answer. Rehearse the *reasoning*, not the wording. Where the true answer is "I don't
know yet", say so and follow with how you would find out — that is the answer a team of
physicists respects.

## The framing question — "you have no real LSM data, so what is this?"

> Real LSM data is proprietary, so I built a physics-inspired forward model — magnetic
> dipoles at burial depth under a drifting geomagnetic background — and then ran the full
> lifecycle on it exactly as I would on yours: validation first, background removal,
> features, grouped splits, calibrated uncertainty, tracking, gated release, a served heat
> map. The data is a stand-in. The methodology, and the places where I chose to be
> paranoid, are the real content. I also built in off-pipe interference on purpose,
> because separating a defect from benign magnetic clutter is the hard part, not fitting
> the model.

## Physics and signal

**"What actually generates the signal?"** Stress magnetisation — mechanical stress changes
local permeability and remanent magnetisation, so a stress-concentration zone perturbs the
ambient field. I model it as a point dipole, which is the leading multipole term and right
in the far field at ~1.5 m stand-off. It is a crude model for an extended flaw and I would
expect the real signature to be closer to a line of dipoles or a dipole sheet.

**"Three axes — so you can do gradiometry?"** No, and this is worth being precise about.
Three *axes* is one vector magnetometer at one point. Gradiometry needs two spatially
separated heads. With a single moving sensor you get the **along-track derivative** dB/ds,
which does suppress the slowly varying background but is not common-mode rejection. Stage 2
of my plan adds a second head at a 0.5 m vertical baseline so I can compute a true vertical
gradient and quantify how much rejection it buys.

**"How do you remove the background?"** The defect residual is ~25 nT on a ~45 000 nT
field — 0.05%. Robust polynomial detrend per axis, degree 3 over the survey, or a
Savitzky–Golay high-pass with a ~40 m window; the window has to be long compared with a
defect footprint (a few metres at 1.5 m stand-off) and short compared with geomagnetic
drift. Choosing that scale separation *is* the engineering. In production I would prefer a
real observatory trace for the diurnal variation over fitting it away.

**"How do you tell a defect from a fence post?"** Amplitude alone will not do it — an
off-pipe source can be strong. The discriminators are geometric: dipole fields fall as
1/r³, so an off-centreline source is both weaker *and* broader, with a slower decay in the
along-track profile. So I use peak width (FWHM), asymmetry and a fitted decay exponent, not
just peak height. And I keep `interference` as an explicit class with its own reported
precision, rather than letting it hide inside "negative".

## Data and scale

**"We have millions of rows from many years — does this scale?"** The demo uses SQLite,
which is the right tool at 10⁷ rows read-mostly with a composite primary key on
`(survey_id, chainage_m)`. It stops being right at concurrent multi-writer ingest. The
production shape is columnar: Parquet on S3 partitioned by line and run, queried with
DuckDB or Athena, with a relational store kept only for the small high-value tables —
survey registry, ground truth, DQ reports, indications, model versions. Signal is
append-only and immutable; metadata is relational and mutable. I built a scale rehearsal
into the plan (~40 lines, ~10⁷ rows) specifically so I am not guessing about that.

**"Multiple data sources have to be integrated."** The join key is **chainage plus a
line identifier**, and the hard part is that every source has its own chainage datum —
ILI odometer, GPS-derived, as-built drawings — so they disagree by metres. I would treat
alignment as a first-class step with its own validation: cross-correlate a common feature
(girth welds are the natural fiducial), estimate a per-segment offset and stretch, and
record the residual alignment error as a data-quality metric that propagates into
localisation uncertainty. Fusing before you have quantified the misalignment is how you
manufacture false correlations.

**"How do you handle duplicates? You'll get the same survey twice."** Two hashes with
different jobs. A hash of the file bytes proves provenance and immutability, but it gives
false negatives for deduplication — re-export the same survey with different float
formatting and you get a new hash for identical data. So there is also a canonical content
hash over normalised content: contract columns in declared order, sorted by sample index,
cast to contract dtypes and rounded to a declared precision. Ingest is idempotent on that,
and a *different* content hash for an existing (line, run) is an error rather than an
overwrite — raw data is immutable. The case neither hash catches is **partial overlap**:
two surveys of the same line that intersect, from a re-run over a bad section or a survey
split across files. That needs an interval check plus signal cross-correlation in the
overlap — and it matters beyond tidiness, because an undetected overlap landing in two
different folds is leakage. It's why my group key is (line, block) and never block alone.

**"What's in your data contract?"** Dtypes chosen from the physics, not habit. Latitude and
longitude must be float64 — float32 ULP at 47° N is about 0.42 m, comparable to my 0.5 m
sample spacing, so float32 GPS would quantise the survey. Field values can be float32,
since ULP at 45 000 nT is about 0.004 nT, below any magnetometer's resolution — but the
arithmetic runs in float64, because detrend least-squares fits and cumulative window sums
accumulate error that storage precision says nothing about. And the physical key is an
integer sample index, never the float chainage: `0.5 * 3` is not `1.5` in binary, and a
float join key silently drops rows between tables built in different code paths.

**"Ground truth?"** Scarce, delayed and biased — you only dig where you predicted, so your
labels are censored by your own past ranking. That means naive precision on dug sites
overstates performance, and it argues for a small randomised or uncertainty-driven
exploration budget in the dig programme. That is the active-learning stretch goal, and it
is a genuine business argument, not a research nicety.

## Modelling

**"Why not just threshold?"** I do — MAD threshold is the baseline, and it must be beaten
by a stated margin. It fails specifically because of interference, which is exactly why
interference is in the generator.

**"Why two stages?"** Because the unit of decision is an **indication**, not a 0.5 m
sample. Row-level scoring, then peak clustering into indications, then per-indication
severity and class. It also removes an evaluation trap: `severity_smys` is zero off-defect
and constant inside the label window, so a row-level regressor scores beautifully by
predicting zero everywhere.

**"How do you split?"** By line, or by 100 m chainage block, with `GroupKFold` — never
randomly. Adjacent 0.5 m samples are near-duplicates and the same defect recurs in every
survey, so a random split leaks in two directions at once and produces a number I would not
be able to defend. There is a CI test that fails if any group crosses folds.

**"Which metric?"** Not accuracy, not ROC-AUC — positives are under 3% of rows. PR-AUC as a
diagnostic, and as the headline **recall at a fixed dig budget**: if you fund five digs per
kilometre, what fraction of real defects do you find, and how many empty holes do you dig?
That maps directly onto what an integrity engineer is optimising.

**"Uncertainty?"** LightGBM quantile regression for the shape, wrapped in split conformal
for a distribution-free 90% interval with a finite-sample coverage guarantee — under
exchangeability. A new line or a new scanner breaks exchangeability, which is why coverage
is monitored in production and re-fit on new calibration data rather than assumed.

**"Deep learning?"** A 1-D CNN over the residual window is a reasonable next step and I have
it planned, but gradient boosting on well-designed physics features is the right first
model at this data scale, it trains in seconds on 16 cores, and it is far easier to
interrogate with SHAP. I would move to a CNN when I have enough labelled indications that
learned filters beat hand-designed shape features — and I would keep the boosted model as
the challenger baseline, not delete it.

**"Growth and remaining life?"** Three surveys with ~15% growth. Fitting twelve independent
growth curves off three points would be over-fitting; I use partial pooling so each defect's
rate is shrunk toward the population rate, then project to a limit state for remaining
life, and I propagate the severity interval through so the remaining-life output is a range,
not a date. With n=3 the honest statement is that the *method* is right and the *numbers*
are not yet trustworthy.

## MLOps

**"How is a run reproducible?"** `git_sha` + `config_sha256` + `data_sha256`, all three
logged to MLflow and written into the `model_run` table, one `config.yaml`, one seed, and a
determinism test in CI that re-runs training and compares metrics. One practical catch:
LightGBM is not bit-reproducible across thread counts by default, so `deterministic=True`,
`force_row_wise=True` and a fixed `num_threads` are set from day one — otherwise that test
flakes intermittently, which is worse than not having it.

**"What happens when a survey fails validation?"** It goes to a quarantine bucket with its
DQ report attached and raises a ticket; it is never deleted and never silently skipped. In
CI the same failure exits non-zero, but in production the run continues to the next survey.
A pipeline that halts the whole nightly job because one survey had a stuck channel gets
switched off by its operators within a month, and then you have no validation at all.

**"You changed the feature code — what happens to everything you already computed?"** This
is the version people forget, and it is the nastiest, because it is training/serving skew
that no *data* check can detect — the data is fine, the code moved. So `feature_version` is
explicit: it is in the feature-store path, it is in the model bundle, and a bundle refuses
to load against a mismatch. Bumping it forces a documented decision — backfill, or pin old
models to old features. Both are fine; leaving it implicit is not.

**"How confident are you in that metric?"** Not very, and I report that rather than hiding
it. Every headline number carries a bootstrap confidence interval resampled over *groups* —
lines and defects — not rows. With twelve defects, "recall 0.83" is a point estimate on a
sample of twelve, and the interval is wide. The promotion gate compares intervals, not
point estimates, precisely so it does not promote a model on sampling variation.

**"Training vs inference — what stops them diverging?"** They share the code path. The same
validator runs at ingest and inside `predict`. Features are split into per-survey
transforms, which are legitimately re-fit at inference because they *are* background
removal, and fitted transforms — scalers, calibration, conformal quantiles, threshold —
which live inside the versioned bundle and are never recomputed. The bundle stores its
feature list in order and refuses to load against a mismatched feature set.

**"How does a model reach production?"** CI gates: DQ pass, leakage and determinism tests
green, primary metric beats the incumbent on the same holdout *compared on intervals*,
conformal coverage inside its band, no leaky feature in the top-10 SHAP list, bundle
round-trip exact. Passing cuts a `pipeline_release` and sets the `@challenger` alias
automatically; it then runs in shadow, and moving `@champion` is a human decision with a
recorded reason. Deployment pins the `pipeline_version`; rollback repoints the alias, and
bundles are immutable so a rollback target always exists. One caveat I'd raise unprompted:
comparing every candidate against one fixed holdout eventually selects on holdout noise, so
the final test set is touched rarely and its uses are counted.

**"How would you process our archive?"** That is the workload the whole design is centred
on — backfill, not steady-state scoring. It's a Dagster asset graph partitioned by survey,
so processing the archive is a partition backfill rather than a bespoke script, and a
failure at survey 4,000 of 6,000 costs one partition rather than the run. Retries are 3× on
IO and **zero on a data-quality failure**, because a bad survey isn't a flake — retrying it
three times just delays the ticket. I chose Dagster over Airflow specifically because its
asset code-version model maps onto my `feature_version`, so bumping the feature code
re-materialises exactly the affected partitions.

**"Where does a prediction actually get served?"** The serving layer is the batch scoring
job that writes the indication table and the GeoJSON the GIS system consumes — versioned
output schema, a consumer contract test in CI, idempotent on (survey, pipeline version),
with a 24-hour freshness target. The Streamlit app is a *consumer* of that table, not the
server. I'd be careful about that distinction because calling a dashboard "the serving
layer" hides the fact that nobody has defined the actual contract.

**"How do you validate a new model when the ground truth is six months away?"** You can't,
on outcomes — so the only pre-label evidence is **shadow scoring**. Champion and challenger
both score every live survey, the challenger's output is written flagged as shadow, and I
analyse *disagreement*: change in indications per kilometre, rank churn at the dig budget,
and whether the disagreements cluster in surveys that already had data-quality warnings. It
gives you signal on day one instead of month six. Then when labels do arrive, I evaluate by
vintage — pin a `truth_as_of`, reconstruct what was known on that date, score forward — so
a label correction can't retroactively improve a historical result.

**"What's your deployable unit?"** Not a model — a `pipeline_version` manifest pinning four
model versions, the feature version, the schema version and the container digest. An
indication is produced by all of them, so its lineage is only answerable if they're pinned
together, and they have to be promoted and rolled back together. Rollback is repointing the
`@champion` alias, and I drill it in CI, because the usual way rollback fails is that the
old bundle no longer loads against current code — which my own feature-version guard would
cause. So the policy is that the last three feature versions stay loadable and CI proves it.

**"What breaks in production and how do you know?"** Drift, in three flavours: feature
drift (PSI vs the bundle's training reference), prediction drift (indications per km), and
background regime shift (median field and drift slope vs that line's history). Plus
measured conformal coverage as verification results arrive. The failure I would most
expect is a new scanner unit or recalibration shifting the background — a data problem that
looks exactly like a model problem if you are not monitoring the inputs.

## Questions to ask them

- How is chainage reconciled between LSM, ILI and as-built records today, and what is the
  residual alignment error?
- What fraction of LSM indications get excavated, and is any of the dig budget randomised?
- Is stand-off distance measured per survey or assumed from as-built depth of cover?
- Do you have paired LSM and ILI runs on the same line — that is the supervised dataset.
- What is the current false-dig rate, and what would a useful improvement look like in
  euros?
- Where does the model output actually land — a report, a GIS layer, an integrity
  management system?
