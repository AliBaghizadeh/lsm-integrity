# App walkthrough — what to say, tab by tab

One page you can read once and then explain the demo app yourself, without opening any
code. Run it with `streamlit run app/demo_app.py` (or `lsm serve`) from the `llm_gpu2` conda
env (see the environment note in `.claude/skills/lsm-integrity/SKILL.md` — `ml_gpu` has a
broken numpy install on this machine; use `llm_gpu2` for everything).

**Rig-v2 (2026-08-06):** this walkthrough describes the app as rebuilt around the real
3-head scalar rig (walker, GPS dropout, `b_lo/mid/hi_nt` heads — never `bx/by/bz`). If you
see `bx/by/bz` or a two-head "gradiometer" mentioned anywhere else, that's the pre-Rig-v2
model — historical, not what this app runs today.

## What this app is, in one breath

A Streamlit app that walks through an LSM (magnetometer) pipeline-integrity pipeline end to
end — raw sensor signal in, a risk-ranked dig list out — using **real trained models on
real (synthetic) surveys**, not mocked output. It is a *consumer* of the pipeline's real
batch-scoring output, not the serving layer itself (the serving layer is `lsm train` /
`lsm predict`, which write to a database and a GeoJSON export the app then reads).

Two modes, same underlying code path:
- **demo** — reads precomputed results straight off disk (`serving/`). Zero compute, cannot
  fail. This is what you run in front of people.
- **live** — re-runs the actual `validate → register → features → score` pipeline on the
  spot (~1s), against a real session-private database. Proves it isn't smoke and mirrors.
  Toggle to this once, live, as the "watch it actually work" moment.

## Landing page

Title + tagline (now stating the real ~48,800 nT scalar-rig signal, not the old ~45,000 nT
vector-head number) + the current `img/project-components.png` diagram (all pipeline
stages, infra, and consumers, click-through source in `docs/project-components.html`),
gated behind a "Launch the demo" button so nobody lands mid-tab on a shared screen. No
ROSEN-branded photo here anymore (dropped deliberately, see `267b3e8` in git log) — a
centred text hero plus the components image.

## The controls above the tabs

- **Mode**: demo / live, as above.
- **Scenario**: 3 **clean** surveys (passed all validation checks — this is about data
  quality, *not* whether the model did well) + 1 **corrupted** survey (one sensor reading
  deliberately pushed out of physical range, to demo tab 6). All three clean scenarios are
  real surveys the trained model genuinely scored, not fabricated for the demo.
- A **survey** = one walked pass of the magnetometer rod along a stretch of buried pipe.
  The same physical line gets surveyed repeatedly over time (like a baseline inspection
  plus follow-ups), which is how growth/change gets tracked.

## Tab 1 — How it works

A workflow diagram of the **seven** real pipeline stages (Ingest → Validate → Register →
Features → Detect → Severity+Classify → Risk rank), each one labeled with the actual Python
module behind it — so if someone asks "where does X happen," you can point at the box and
name the file. **Register is new in Rig-v2**: raw data no longer carries a usable position
column (irregular walk speed + GPS dropout), so along-track chainage is reconstructed from
GPS dead-reckoning plus girth-weld-comb detection, not read off a column.

Includes an expander listing all **14 real data-quality checks** the validator runs
(schema, per-head range, saturation, GPS jump, survey-overlap cross-correlation, etc.).

**Say:** "Every scenario in this app runs this exact same seven-stage path — this tab is the
map of it."

## Tab 2 — Model performance

**This is the tab that answers "where's the baseline, where's IsolationForest, where's the
boosting model."** It renders the real, auto-generated model card (`serving/model_card.md`)
for whatever pipeline is currently loaded — the same numbers `lsm train` prints to the
console, not a re-derived summary. Four real comparisons, each with a bootstrap confidence
interval, not a point estimate — **current Rig-v2 numbers, scalar rig, demo scale (60
defects)**:

| Stage | Model | vs. baseline | Real result |
|---|---|---|---|
| 3 — Detection | IsolationForest | MAD (median-absolute-deviation) threshold | Recall gap **-0.006** [-0.061, 0.056] — statistically indistinguishable, numerically unchanged from the pre-Rig-v2 result. Gate (≥0.15 gap) **did not pass**. |
| 4 — Severity | LightGBM quantile regression + split conformal | Global-mean baseline | Coverage **0.689** [0.420, 0.945] vs target band [0.87, 0.93] — **gate did not pass**. This is a real regression from the pre-Rig-v2 result (0.920, PASS) — a harder, GPS-dropout, magnitude-only acquisition is producing noisier calibration at this sample size. |
| 5 — Classification | LightGBM multiclass + isotonic calibration | Majority-class baseline | SCC recall **0.042** [0.000, 0.125] — **gate did not pass**. Demoted to a synthetic-only capability demo — the interview confirmed no real labelled defect-type data exists at ROSEN, so this isn't a claim the model generalises. |
| 8 — Growth | Partially-pooled log-linear | "No growth" baseline | **Gate passed** — recovers `ln(1.15) = 0.1398` almost exactly. Unaffected by Rig-v2 (estimator-correctness check on a deterministic law, not real defect physics). |

**Say:** "We don't just report the fancy model's number — every model here is compared
against a stated, honest baseline, and when the fancy model *doesn't* clear the bar, that's
reported as-is, not hidden. Rig-v2 made that harder, not easier — severity coverage
genuinely regressed when we rebuilt around the real instrument, and that's on the model card
too, not smoothed over."

**If someone asks "so should ROSEN build new hardware or hire more data scientists"** —
that's the actual finding from the Stage D ablation ladder (not shown in this tab; see
`LSM_PROJECT.md`'s "Rig-v2 measured results" / `reports/ablation-ladder.md`): none of five
software-only improvements move detection recall by a statistically distinguishable amount
over the bare middle head, while a genuine hardware upgrade to full vector output roughly
triples recall, cuts false-digs ~3×, and cuts localisation error ~10×. Reported plainly —
it leans toward hardware for this specific gap.

**Known caveat, worth knowing before someone asks:** the four defect subtypes (SCC, weld,
dent, corrosion) are assigned **at random** in the synthetic data generator and carry no
distinguishing physical signal — so the classifier can't tell them apart. If asked, say this
plainly rather than over-claiming subtype accuracy — full detail in
`.claude/skills/lsm-integrity/references/interview-drills.md` under "How do you decide
defect type."

## Tab 3 — Raw signal

The raw **~48,800 nT** field, `b_lo/mid/hi` — the rod's three scalar total-field heads, not
vector axes — as three separate lines; the defect is genuinely invisible here. A log-scale
"deviation from median" toggle is the first place it becomes visible. Dashed markers show
true defect/interference locations for reference.

**Say:** "This is the point of the whole pipeline — you cannot see the defect in the raw
signal. Everything downstream exists to make it visible. And notice these are three scalar
magnitude readings, not x/y/z — the real instrument only ever reports `|B|`."

## Tab 4 — Detrend + gradient

Same signal after background removal (two-stage detrend: robust polynomial + rolling-median
high-pass), plus the first-difference gradient across the three heads (`g1_nt_per_m`). Now
both true defects *and* true interference sources show up — separating the two is the
model's actual job, not detecting a bump exists.

## Tab 5 — Ranked indications

The output an inspection engineer actually reads: a map, a severity/type scatter (with 90%
confidence intervals), a table, and a ranked bar chart — same dig-budget-limited set of
indications shown four ways. Ranked by `risk_score` (calibrated P(defect) × severity ×
consequence proxy) once a classify model is available, falling back to `anomaly_score`
otherwise. The dig-budget control (3/5/10 per km) is a pure client-side re-slice, nothing
recomputes.

The map itself is real GPS track, thinned for display and filtered to rows where the walker
actually had GPS lock (Rig-v2 has real dropout — unlocked rows are dropped before drawing,
not fed to the map as-is, which used to render as broken lines through the survey area).

**Say:** "This is the answer to 'what do I dig first' — not a probability nobody can act on,
a ranked list with a stated budget."

## Tab 6 — Corrupted survey

Loads the deliberately-broken scenario and shows the validator's actual refusal: which of
the 14 checks failed, on which rows, with what value. This is the tab that proves the system
knows the difference between "the model is uncertain" and "the data is bad" — most demos
never show this half.

**Say:** "Watch it refuse to score bad data instead of silently producing a confident, wrong
answer."

## Tab 7 — Risk heatmap

`(line, 100 m block)` risk matrix across every survey the currently released pipeline has
scored — not just the one scenario selected above. Lets you point at "where along the whole
corpus is risk concentrated," not just one survey's own indications.

## Footer

`pipeline_version` / `feature_version` / `schema_version` / `config_sha256` / `git_sha` —
full provenance of exactly what produced whatever's on screen. Point at it if anyone asks
"how do you know what model made this prediction."

## If you only have 90 seconds

1. Launch → tab 3 (defect is invisible) → tab 4 (now it isn't) → tab 5 (ranked list + map).
2. Flip Mode to **live** on one clean scenario — proves it's not canned.
3. Tab 2 for one sentence: "every model here beat, tied, or honestly lost to a stated
   baseline — we don't hide the losses, and Rig-v2's severity regression is right there too."
4. Tab 6, corrupted scenario: "and it knows when to refuse."
