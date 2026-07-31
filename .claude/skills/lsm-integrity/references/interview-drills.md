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

**"Why not just threshold?"** I do — MAD threshold is the baseline, and IsolationForest has
to beat it by a stated margin, on the confidence interval, not the point estimate. When I
actually ran it: **it didn't**, across three real runs, though the story gets better each
time. First on a 1-line/12-defect corpus: recall gap -0.056, CI [-0.167, 0.056] — straddling
zero, too little data to tell if the effect was real. Rather than accept that as the final
word, I scaled the corpus to 5 independent lines (~60 defects — more lines, not more defects
crammed onto one line, which I tried first and it degraded a different gate by contaminating
background rows with neighbouring sources' field tails) specifically to narrow that CI.
Re-ran: gap -0.028, CI [-0.083, 0.022] — about half the point estimate, about half the width,
still just barely straddling zero, a **more precise measurement of a small real effect**, not
"still inconclusive." I could have stopped there and called it a legitimate negative result.
Instead I ran a real EDA (below) that found a specific, fixable mechanism, fixed it, and
re-ran a third time: gap closed to **-0.006, CI [-0.067, 0.050]** — recall is now
statistically indistinguishable between the two models, and the interference-rejection
mechanism the EDA predicted turned out to be real and significant (see below). I could have
quietly re-tuned IsolationForest until it passed the ≥0.15 margin; I reported the honest
result — parity, not a win — instead, because that's what the gate is *for*.

I went further and ran a real EDA over the 48-feature set (Stage 2.75) to stop guessing at
that mechanism. It found the model is fit on ~20 mutually-correlated amplitude features
(the window-statistic family, plus two features that are *exact* duplicates under the
current config — normalising by `depth_m` is a constant scalar multiply when `depth_m` isn't
per-row) that separate defect from background almost perfectly (PR-AUC up to 0.97) but are
mediocre at defect-vs-interference specifically (mostly under 0.55), because interference
genuinely produces amplitude too by design. The features that actually separate defect from
interference are a different, smaller set — wide-window shape descriptors like `w25m_kurt`
(PR-AUC 0.985). An isolation forest splitting roughly uniformly across 48 dimensions, a
third of which are redundant amplitude features, has more chances to isolate on that
dominant cluster than on the one or two shape features that would correctly reject
interference. That's a specific, evidenced mechanism, not a hedge — so I acted on it: dropped
the two duplicate features entirely (a real `feature_version` bump, not just excluding them
from the model), and gave IsolationForest an `emphasis_repeats` knob that repeats
`w25m_kurt`/`w25m_zcr`/`w10m_kurt` in its fitted matrix to raise their selection odds. Re-ran:
recall gap closed to **-0.006, CI [-0.067, 0.050]** (was -0.028) — recall is now
statistically indistinguishable between the two models — and the interference-attribution
check flipped from "doesn't cleanly explain it" to **significant** (CI [0.007, 0.058],
excludes zero): MAD now wastes significantly more of the dig budget on interference than
IsolationForest does. The ≥0.15 recall-margin gate still doesn't pass — the honest result is
parity, not an IsolationForest win — but "I found a specific mechanism via EDA, fixed it, and
measurably improved the thing the fix targeted, confirmed by a CI that excludes zero" is a
complete, evidence-driven story, end to end, not a shrug.

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
is monitored in production and re-fit on new calibration data rather than assumed. First
real run (12-defect corpus) measured 0.727 against the 0.87-0.93 target band — below it,
though the CI was [0.485, 0.939], containing the target, so at n=11 matched defects the
honest statement was "can't confirm calibration is good, can't rule it out either," not
"calibration is broken." I'd also found a real bug in my own conformal code on that first
run: a plain (1-alpha) quantile for the calibration margin instead of the finite-sample-
corrected level the theory (Romano et al. 2019) specifies, which systematically undercovers
on small calibration sets. Fixed it, verified with unit tests that hand-check the correction
against a naive quantile. On the later 5-line/~60-defect corpus (n=140 matched severity
indications, same scale-up that sharpened the Stage 3 measurement above), coverage came in
at **0.905 — inside the target band, gate passes**, with MAE roughly halved versus the
global-mean baseline (7.2 vs 15.0 nT). Same conformal code both times; more calibration
points is what actually closed the gap, which is the expected story for a distribution-free
method under a genuine small-sample regime, not a coincidence. Still conditional on Stage
3's detector, which doesn't beat baseline — a passing severity gate validates the CQR
methodology, not the whole pipeline's choice of anomaly model.

**"How do you decide defect type — SCC vs weld vs dent vs corrosion?"** Multiclass LightGBM
(isotonic-calibrated P(defect), degenerate-fold fallbacks for the small per-class counts, a
physics-consistency SHAP check that denylists absolute chainage) — that machinery is real and
built. But the honest number, from my real run: recall is scc 0.125, weld 0.111, dent 0.646,
corrosion 0.383, versus **interference at 0.778 recall / 0.909 precision**. I didn't just
report that gap, I checked why. In `generate.py`, each synthetic defect's `type` is assigned
`rng.choice(["scc","weld","dent","corrosion"])` at generation time, and grepping the file
confirms that value is read in exactly one place afterward — writing the ground-truth label
column. It never reaches the dipole model: amplitude, orientation and decay all come from
`severity`/`orientation`, drawn independently of `type`. So in this synthetic dataset the four
defect subtypes are **physically indistinguishable in the sensor data by construction** —
there is no signal in the features for a classifier to find, and scc/weld sitting at or below
the 25% random-guess floor for a 4-way choice is the model doing exactly what the data
supports, not underperforming. Interference is a genuinely different physical source (bigger
magnetic moment, broader/off-pipe geometry — see the interference section above), which is
why the model separates it well and can't separate the other four. I'd rather say that
plainly than let a strong interference number imply the subtype numbers are equally real.
Fixing it for real means extending the generator so each defect type carries a distinguishing
physical signature (the decay/shape differences a metallurgist would actually expect between
SCC, a weld anomaly, a dent, and corrosion) — a stated, scoped data gap, not a modelling
failure, and the same "add its baseline first, report the honest number" discipline as
everything else in this project.

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

## Process, honesty, and what was new to you

Questions in this section aren't about the domain — they're about how you actually worked,
and in a lot of interviews they matter more than the domain answers because they're the ones
that transfer to a role that isn't this exact project. Answer with your own true specifics,
not a generic "I learned a lot about MLOps" — name the actual thing.

**"Tell me about something in this project you hadn't used before."** Be concrete: which
technique, when you ran into needing it, what you did to get up to speed (read the original
paper, worked a toy example by hand, implemented it and tested it against a naive baseline
before trusting it), and what surprised you once you had it running. Naming a specific gap
and how you closed it is a stronger answer than implying you walked in knowing everything —
this project genuinely has two candidates for this: IsolationForest, and split conformal
prediction (CQR). If either was new to you, say so plainly and say what studying it involved.

**"Walk me through something that didn't work."** Two real ones, fully reportable without
hedging:
1. IsolationForest vs the MAD baseline (Stage 3) — never beat it, across three real runs.
   First (12 defects): recall gap -0.056, CI [-0.167, 0.056]. Rather than leave it there, I
   scaled the corpus 5x (more independent lines, not more defects packed onto one line —
   that move alone surfaced a real data-generation bug I fixed first) specifically to
   sharpen that CI, and re-ran: gap -0.028, CI [-0.083, 0.022] — a tighter measurement
   confirming a real, small effect, not "still not enough data." Then I ran a real EDA over
   the feature set, found the model was fit on ~20 redundant amplitude features that were
   drowning out the 3 that actually separate defect from interference, fixed it (dropped 2
   exact-duplicate features, added a knob to up-weight the other 3), and re-ran a third
   time: gap closed to -0.006, CI [-0.067, 0.050] — recall is now statistically
   indistinguishable between the models, and the interference-rejection mechanism the EDA
   predicted turned out real and significant. Still doesn't clear the gate's ≥0.15 margin —
   parity isn't a win — but reported as the actual result at every step, not tuned until it
   passed.
2. Split-conformal undercoverage (Stage 4) — traced to a real bug in my own code (a naive
   quantile where the theory calls for a finite-sample-corrected one), fixed it, and the
   gate *still* didn't clear on the original 12-defect data (0.727 vs a [0.87,0.93] target,
   CI containing the target band). Left it there rather than manufacture a happy ending —
   until the same 5x corpus scale-up used for Stage 3 gave conformal enough calibration
   points to actually pass (0.905) on a later run. Worth being clear in an interview that
   the fix was necessary but not sufficient by itself; scale is what closed the gap.

**"How do you know your negative results aren't just bugs?"** Because each one got checked
before being accepted, not just shrugged at: the conformal undercoverage was traced to a
specific formula error and confirmed by hand-computing the correction on a small example and
by inspecting per-defect coverage row by row, not by assumption. The IsolationForest result
was checked by re-running at 5x the scale specifically to see if the effect held up or was
noise — it held up, tighter and smaller — and then by running a real EDA to find and fix a
specific, named mechanism (redundant amplitude features drowning out the ones that actually
separate defect from interference), which produced a predicted, measurable, significant
improvement on re-run rather than a shrug. A suspicious number deserves inspection; a
genuinely small effect, once you've ruled out a bug, ruled out "just not enough data," and
acted on the mechanism you found, is a real finding, not an excuse.

**"What would you do next with more time or data?"** Stage 6 (the full scale rehearsal —
~40 lines, ~10⁷ rows) is the designed answer, and it's not purely hypothetical anymore: a
first, smaller step (1→5 lines, ~60 defects) already narrowed both Stage 3's and Stage 4's
CIs meaningfully and flipped Stage 4's gate from fail to pass, which is direct evidence that
sample size — not the underlying models — was the binding constraint on 12 defects. Stage 6
is the same lever, taken further.

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
