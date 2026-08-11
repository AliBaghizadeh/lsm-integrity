# My Model Failed Its Own Test. That Was the Point.

*What I learned building a machine learning pipeline for buried pipeline inspection.*

---

I am a physicist. I understand magnetism well. What I did not know was how to turn that knowledge into software that other people could rely on.

So I built a full machine learning pipeline to find out. It takes magnetometer readings from above a buried pipeline and produces a ranked list of places to dig.

Two things to say before I start.

The data is synthetic. Real inspection data of this type is commercial and I do not have any, so I wrote a physics-based generator instead. Every number below is measured on that generated data.

And the main model does not work. It fails its own quality check, the check runs in CI, and the job shows red. I now think this is the most useful part of the project.

---

## The problem

Buried pipelines corrode and crack. The normal way to inspect them is to send a robot inside. But many pipelines cannot take one, because of tight bends or missing entry points.

One alternative is to walk a magnetometer along the ground above the pipe. Steel under stress changes the magnetic field around it slightly. If you can measure that change, you can guess where the damage is, without digging.

The difficulty is the size of the signal.

The Earth's magnetic field where I placed my survey is about 45,000 nT. The signal from a defect, after I remove the background, is about 25 nT. That is around 0.05 percent of the raw measurement.

So the real work is not detection. It is background removal. Everything else in the pipeline follows from that.

There is a second problem. A magnetometer cannot tell useful steel from useless steel. A buried fence post, a piece of scrap metal, or an old well casing all produce a signal that looks like a defect. If your detector only looks at signal strength, every piece of buried junk becomes a recommendation to dig. Each dig costs tens of thousands of euros.

So the task is not "find strong signals". It is "find strong signals that are actually defects".

---

## Making the data

I could not get real data, so I wrote a forward model.

A small damaged region in a steel pipe behaves, from a distance, roughly like a magnetic dipole. The field of a dipole is standard physics and falls off as 1/r³. I place defects along a line, give each one a strength and a random orientation, and compute what a sensor walking above them would measure.

I should be honest about the limits. The scaling constant in my code is not the real physical constant. I chose it so the numbers land in a realistic range. I do not model the pipe body, or induced versus permanent magnetisation. The generator gives data with the right *structure*, not data a physics reviewer would accept. That was a deliberate trade: I wanted ground truth I know exactly, so I can check whether the pipeline recovers it.

The corpus is five separate 2 km lines, measured every 0.5 m, surveyed three times each. That is 15 surveys and 60,000 rows. Each line has 12 defects and 4 pieces of interference. Defects grow 15 percent between surveys, so I can also test growth prediction.

### The interference is the interesting part

I wanted the false alarms to be genuinely hard, not decoration.

Interference sources sit 3 to 8 m to the side of the line, not under it. Being further away, their signal is much weaker, because of the 1/r³ falloff. If I left it there, they would sit below the noise and never fool anything.

So I made them stronger, by a factor of about 50. That number is not arbitrary: (5.5/1.5)³ is roughly 50, which is exactly the penalty for sitting 5.5 m away instead of 1.5 m below.

The result is interference that arrives at the sensor with the *same strength* as a defect, but a different *shape*, because it comes from a different geometry.

I measured this. Defects under the pipe produce a peak about 2.2 m wide. Interference beside the pipe produces a peak about 6.2 m wide. Same height, different width.

This is the core idea of the project: **shape separates them, strength does not.** A threshold on signal strength cannot work here, even in principle.

Later analysis confirmed it in numbers. Features based on signal strength separate defects from empty ground almost perfectly. For separating defects from interference, the same features are close to useless. The features that work are shape features.

---

## Twice my simulator fooled me

This is the part I would most want to tell someone starting out.

**First mistake: my background was a polynomial, so my polynomial method looked excellent.**

I built the background drift as a slow ramp plus one slow wave. I then removed it by fitting a polynomial. Of course it worked. The leftover noise came down to 5.03 nT, and my sensor noise was set to 5.0 nT. It matched exactly.

My method was not good. I was fitting a polynomial to a polynomial.

To check properly, I needed a real background. The USGS publishes magnetic observatory recordings for free. I took two days from Boulder, Colorado: one quiet day, and 10 May 2024, the largest magnetic storm in twenty years. My generator can use these real recordings instead of my invented background.

The test was useful. A simple polynomial fit leaves 6.9 to 11.3 nT of error on the storm data, well above the noise floor. It does not fit real magnetic behaviour the way it fit mine.

But my actual two-step method (a robust polynomial, then a 40 m rolling median filter) held up: about 7.8 nT on my synthetic background, 8.1 nT on the storm, 7.8 nT on the quiet day. Detection quality stayed roughly the same in all three cases.

So the pipeline survived contact with real physics. But I only know that because I went looking for a way to prove myself wrong.

**Second mistake: my two-sensor setup could not fail, because I built it that way.**

A standard trick is to use two sensors, one above the other, and subtract. The background is nearly the same at both, so it cancels, while the nearby defect signal does not.

In my first version, I built the second sensor's reading from the same background arrays as the first. So the background cancelled perfectly. Not because the method is good, but because I had made the two backgrounds identical. The measured difference was pure sensor noise.

Fixing it meant modelling what really differs between two sensors half a metre apart: the Earth's field gradient, and local geology.

Then came the honest result. **The two-sensor version is worse.** Detection contrast is about 3.2 times for a single sensor with background removal, and about 1.45 times for the two-sensor difference. Subtracting cancels the background but adds the noise of both sensors together. After background removal has already done most of the work, you pay the noise cost for a benefit you already had.

That is a negative result about a technique the field treats as standard. It is in the repository, written down as such.

That was true of the instrument I thought I was building for. Later, I found out that instrument does not exist. More on that below.

---

## Checking the data before using it

In the lab, my data checking was "look at the plot, and if it looks strange, investigate". That does not work for a pipeline that runs at night on data nobody looks at.

So every survey passes 14 automatic checks at loading time. Seven can stop a survey. Seven only warn. They cover things like schema, physical range, a stuck sensor repeating the same value, GPS jumps, and noise level.

Two decisions here took me time to appreciate.

**Do not crash. Set aside.** When a survey fails a hard check, the pipeline does not stop. It records all 14 results, marks that survey as quarantined, copies it with a report into a separate folder, and continues with the other surveys.

The reason is practical. A pipeline that stops a whole nightly run because of one bad sensor gets switched off by the people who operate it. Refusing bad data is correct behaviour, and correct behaviour should not look like a crash.

**Raw data is never overwritten.** Every survey is identified by a hash of its contents. If I load the same file twice, nothing happens. If I load a *different* file under the same name, it stops with an error saying raw data is immutable and it refuses to overwrite.

That error has saved me several times. Data quietly changing under a name I already trained on is a problem you do not notice until your results move and you cannot explain why.

---

## Testing honestly

This is where I changed how I think.

**I had far fewer independent samples than rows.** My three surveys measure the *same* 12 defects. That is 12 independent objects, not 36 measurements. Every confidence interval in the project resamples whole defects, not rows. If I had resampled rows, I would have produced very narrow and completely false error bars.

**I split the data by location, not randomly.** Neighbouring measurements are almost identical, so a random split puts nearly the same data in training and testing, and the score looks great for no reason.

**I refused to use one popular metric.** Only about 3 percent of rows contain a defect. With such unbalanced data, ROC-AUC looks flattering and says little. I use it nowhere.

Then I set quality gates the model must pass before it can be released. The important detail is that almost every gate is measured on the *lower end of the confidence interval*, not the best estimate. A model that wins on average, but not with confidence, has not shown anything.

---

## The model failed, and then I checked whether the failure was real

The result on my first corpus:

```
Recall difference (new model − simple baseline): −0.006  [−0.067, +0.050]
Gate: DID NOT PASS
```

The new model was slightly worse than a simple threshold. But look at the interval. It goes from −0.067 to +0.050. It contains zero. With only 60 defects, I could not tell "slightly worse" from "no idea".

A weak negative result is not a result. So the question became: is this real, or is my dataset too small to see it?

I first tried the obvious fix, putting more defects on one line. That made things worse: packing them closer means the background around each defect is polluted by its neighbours. More data, worse data.

The correct fix was more separate lines. I scaled up to 40 lines of 40 km each, three surveys per line. That is **9.6 million rows**, with the same defect density as before.

```
Recall difference (new model − simple baseline): −0.029  [−0.033, −0.026]
Gate: DOES NOT PASS
```

Now the interval is narrow and does not contain zero. With 800 times more defects, the answer is clear. The complex model is genuinely, repeatedly, slightly *worse* than a simple robust threshold on this data.

The numbers show exactly where it loses.

A real inspection team cannot dig everywhere. They have a fixed budget, so I allow the model 5 holes per kilometre and judge it only on those. Both models get the same number of holes, which makes them directly comparable: whatever they suggest beyond the budget costs nothing and is never checked.

| Percent of all holes | Simple threshold | Complex model |
|---|---|---|
| Found no defect | 23.3 | 26.6 |
| — landed on buried junk | 23.3 | 23.0 |
| — landed on empty ground | 0.0 | 3.6 |
| Average position error (metres) | 0.28 | 0.42 |

The two middle rows add up to the first one.

The simple threshold does make mistakes, but every one of them lands on buried junk. That is the honest kind of mistake, and it is exactly the trap I built into the data.

The complex model digs 3.6 percent of its holes in empty ground, where there is nothing at all, and it places every hole about 50 percent less accurately. It does reject junk slightly better, 23.0 against 23.3, but that gain is ten times smaller than what it loses elsewhere.

There is a deeper reason, and it is my mistake in how I framed the problem. I built the interference to have the *same strength* as a defect. So interference is exactly as unusual as a defect. But an anomaly detector looks for things that are unusual. Being unusual is the property the two things **share**. To separate them you need to know that a narrow peak means damage and a wide peak means junk, and that knowledge can only come from labelled examples. I asked an unlabelled method to answer a question that needs labels.

I know the information is there, because my classifier, which does see labels, identifies buried junk with 99.9 percent precision using the same features.

The training command exits with an error code when this gate fails, so the CI job is red. It is supposed to be. A separate CI job runs the code checks and 312 tests, and that one is green. I keep them apart so that a real model failure never looks like broken code, and broken code never hides behind a model excuse.

Nothing has ever been promoted to production status. Every release is still marked "challenger".

### Running at scale also found three real bugs

**A 0.4 percent data problem caused a 100 percent failure.** In 36 of 18,917 cases, a detection landed just outside a defect's labelled zone, so its true severity was missing. One missing value passed into the uncertainty calibration, and because of how the maths propagates, it destroyed the calibration for that entire group. A tiny data issue became a total failure, not a small one.

**Some settings did nothing.** Two model settings in my configuration file were never actually passed to the model. Changing them had no effect at all, silently, because they happened to match the built-in defaults. I only found this when larger values changed nothing. Fixing it improved crack detection from 0.50 to 0.64.

**Settings that work on small data can break on large data.** I had reduced the model size so it would work with only 60 defects. Applied unchanged to 9,600 defects, that same limit destroyed the uncertainty calibration completely. Small-data workarounds are not production settings.

### What did work

Severity prediction with uncertainty works well. I ask for intervals that contain the true value 90 percent of the time, and measured 92 percent on the small corpus and 89.6 percent on the large one. That is the result I trust most.

Interference rejection works very well. The classifier identifies buried junk with 99.9 percent precision. That was the hard problem I set out to solve. *(I take this back later — see "Then I ran it big." The number is real, but it belongs to the instrument I assumed existed, not the one that does.)*

Crack detection does not reach my target. I required 90 percent, and measured 64 percent. Better than the baseline's zero, but not good enough to release.

Growth prediction passes its gate, but I do not think it means much, and I want to say so. My generator grows defects by exactly 15 percent every time, with no randomness. So a correct estimator recovers that number almost exactly, and mine does. That tells me my arithmetic is right. It does not tell me the method works on real defect growth, which is messy and irregular.

---

## The instrument I built this around wasn't the real one

That two-head story above was true for as long as I believed it. Then I had a second-round interview with the person who actually built the real instrument, and it stopped being true.

I had built the whole project around one mental picture: a 3-axis vector magnetometer riding a cart on a fixed 0.5 m grid, with an optional second head half a metre up for gradiometry. The real rig is a different machine. It is a rod carrying three sensors — a middle head and two more, 50 cm apart — carried by a person walking at whatever speed a person walks, with GPS that drops out wherever sky view is poor. Each of those three heads reports only one number: the size of the field, |B|. Never x, y, or z.

That single fact changes the physics of the whole problem.

The ambient field in the rebuilt generator is about 48,800 nT. The defect signal is still about 25 nT. A sensor that measures only the *size* of the field, not its direction, does not see the defect's field. What it sees is:

|B0 + dB| − |B0| ≈ dB · B̂0

the projection of the defect's field onto the direction of the ambient field, and nothing more. A defect whose stress-induced moment happens to point close to perpendicular to the ambient field is close to invisible, however strong it is. A vector sensor would have caught it. A scalar one, structurally, cannot.

That is also, I now think, why the real instrument reports scalar in the first place. |B| does not care which way the rod is pointing. A rod swaying in someone's hand corrupts a vector reading. It does not corrupt a total-field reading. Scalar is the sensible choice for a walked survey, and I had modelled the wrong instrument.

Three heads, not two, buy something specific. Two heads give a first difference, cancelling the common background — the story I told above. Three heads also give a second difference, `g2 = b_hi + b_lo − 2·b_mid`, which cancels a *linear* background gradient the first difference cannot touch. And because the walk is irregular, GPS alone is no longer enough to say where you are along the pipe. A new stage now reconstructs position from GPS dead-reckoning plus the pipe's own girth welds, about 12 m apart, found in the signal itself.

I re-ran the existing gates on the rebuilt corpus before asking anything new. Detection barely moved: the recall gap came back at −0.006, the same figure to three decimal places as the original small-scale measurement. Severity got worse: the uncertainty intervals that used to cover the true value about 92 percent of the time now cover it about 69 percent of the time, on this harder, noisier rig. I am stating that as a real regression, not smoothing over it.

Then the real question: given the real rod, how much can software claw back? I ran an ablation ladder — six versions of the pipeline, adding one more thing at a time, from a bare middle head with GPS position, up through the first difference, the second difference, a stand-off correction, and full weld-comb registration. A sixth arm asks something different: what if, instead of more software, the hardware had a fourth axis — full vector output?

| Arm | recall @ dig budget | false-dig rate | localisation error (cm) |
|---|---|---|---|
| 1. mid-head only, GPS chainage | 0.194 [0.069, 0.333] | 0.767 [0.717, 0.833] | 617 [379, 859] |
| 2. + first difference | 0.208 [0.083, 0.375] | 0.750 [0.717, 0.783] | 425 [244, 632] |
| 3. + second difference | 0.194 [0.069, 0.361] | 0.767 [0.733, 0.800] | 306 [158, 490] |
| 4. + stand-off correction | 0.222 [0.083, 0.389] | 0.733 [0.700, 0.767] | 439 [235, 670] |
| 5. + weld-comb registration | 0.208 [0.069, 0.361] | 0.750 [0.717, 0.783] | 536 [321, 745] |
| 6. full vector output (hardware) | 0.597 [0.430, 0.764] | 0.272 [0.233, 0.311] | 48 [40, 58] |

Read the intervals, not just the point estimates. Every step from arm 1 to arm 5 has a confidence interval on its change that straddles zero. Added together, the whole software ladder moves recall by +0.014 [−0.028, 0.056] — not distinguishable from doing nothing. Arm 6 is a different story: recall roughly triples, false digs drop by nearly three times, and localisation error drops by roughly ten times.

I want to be exact about what that last comparison is and isn't. Arm 6 runs on a simpler generator, one that does not model the walker's gait, the GPS dropout, or the sensor noise the real scalar rig has to live with. Part of its advantage is a simpler world, not only vector-versus-scalar sensing. It is not a paired comparison, so the honest statement is that the confidence intervals barely overlap on recall and do not overlap at all on the other two numbers — not a hypothesis test with a clean p-value.

Even with that caveat, the reading is uncomfortable in the same way the earlier result was. After doing everything I could think of, algorithmically, with the real three-head instrument, the honest answer is that you cannot fully compensate for scalar sensing with better software. A genuine hardware upgrade, one more axis, buys far more than any feature I engineered on top of the existing rod. A separate check, correlating detection against the angle between a defect's moment and the ambient field direction over 120 physical defects, found a weak but real relationship pointing the same way (Spearman correlation +0.183, p = 0.045, after controlling for severity) — noisy, not a clean textbook curve, but consistent with the physics.

That is arguably more useful to tell an instrumentation company than "my model works." It tells them where the next dollar should go.

## More feedback came in, and one number flipped

I want to show you what the signal actually looks like, broken into its physical pieces, before I get to what changed.

![From raw field to isolated defect signal](img/signal_decomposition.png)
*Same survey, five views: the raw field (background hides everything), each source's own contribution isolated against background, the residual after two-stage detrending, the same residual zoomed in so bump *shape* is visible, and finally the residual with weld and interference windows masked out — just the defect signal left.*

![Each physical source, isolated](img/signal_components.png)
*The same four sources, each on its own scale this time, over the full 2 km line. Girth welds are the loudest thing in the data by a wide margin — up to 330 nT, and there are 163 of them. Interference is rarer (4 sources) but still reaches over 150 nT. Defects are the quietest — a peak of about 11 nT — which is the whole reason this project exists: the thing you actually care about is the smallest signal in the room.*

After I had the numbers above, the team gave me two more pieces of real feedback. First: interference is not one fixed strength. In the real data, some external objects sit close to a defect's own amplitude and some are noticeably weaker — the variation doesn't track distance, it's just genuine spread in what junk happens to be buried nearby. My generator had interference at a single fixed multiplier. Second: I had all four defect types — cracking, welds, dents, corrosion — drawing their severity from the same range, which the team was right to distrust as a shape claim, but they did expect a real difference in typical *intensity* between a sharp mechanical dent and a diffuse crack.

Both are one-line changes to the generator: interference strength now varies randomly instead of sitting at one fixed value, and each defect type draws its severity from its own range instead of sharing one.

I regenerated the corpus, retrained everything, and re-ran the ablation ladder. Here is where I have to correct something I said above, not quietly leave it next to the new numbers.

| Arm | recall @ dig budget | false-dig rate | localisation error (cm) |
|---|---|---|---|
| 1. mid-head only, GPS chainage | 0.153 [0.056, 0.278] | 0.817 [0.767, 0.867] | 854 [617, 1072] |
| 2. + first difference | 0.208 [0.083, 0.347] | 0.750 [0.683, 0.817] | 951 [773, 1124] |
| 3. + second difference | 0.208 [0.097, 0.347] | 0.750 [0.650, 0.800] | 913 [720, 1112] |
| 4. + stand-off correction | 0.222 [0.097, 0.361] | 0.733 [0.667, 0.783] | 887 [719, 1050] |
| 5. + weld-comb registration | 0.208 [0.097, 0.347] | 0.750 [0.683, 0.800] | 829 [659, 989] |
| 6. full vector output (hardware) | 0.611 [0.444, 0.764] | 0.211 [0.128, 0.300] | 47 [39, 56] |

The step from arm 1 to arm 2 — adding the first difference across heads — now shows a real, statistically significant jump: +0.056 [0.014, 0.111], an interval that does not touch zero. I said above that no software step reached significance. That was true of the first measurement. It is not true anymore, and I would rather correct it in public than leave the old sentence standing.

What has not changed: the first difference explains the *entire* software gain. Arms 3 through 5 still move the point estimate by amounts their own intervals can't distinguish from zero. And hardware still dwarfs everything software did — arm 6 reaches 0.611 recall against software's best of 0.208, a gap more than seven times the size of the one real software win. The instrument-company answer from before still holds. It just has one more honest footnote now: keep the first-difference calculation, it's free and it works, but don't expect it or anything downstream of it to close the gap hardware would.

Severity got worse again, too. The uncertainty intervals that covered the true value 92 percent of the time before Rig-v2, then 69 percent under it, now cover it 62 percent of the time — and for the first time, worse than just predicting the population mean for every defect (67 percent). I don't have a confirmed reason yet. My best guess is that type-dependent severity makes the population more varied, and I don't have enough calibration examples at this scale to learn the split. I'm stating that as a guess, not a finding.

---

## Then I ran it big, and something I had been proud of stopped being true

Everything above was measured on a small corpus. I finally ran the whole thing at 18 million rows and 360 defects, and I checked it three different ways instead of one — the easy split, a split that holds out whole pipelines, and a split that trains on the first two visits and tests on the third.

Two things came out of it.

The first is good news. Severity had been getting steadily worse, and I had guessed it was starved of calibration examples rather than broken. That guess was right: with more data the intervals went from covering the true value 62 percent of the time to 87 percent, with no change to the model at all. But it only reaches 87 percent under the easiest of the three splits. Under the two harder ones it lands at 85 and 86 percent, just under the bar. So the honest sentence is "more data moved severity from badly broken to marginal," not "more data fixed severity." I wrote the first version of that sentence as "fixed," looked at the other two splits, and rewrote it.

The second is the part I have to correct. Twice above I told you that identifying buried junk was the hard problem I set out to solve, and that my classifier had essentially solved it at 99.9 percent precision. That number was real. It was also measured on the *old* instrument — the one with a full three-axis sensor, before I learned the real device only reports a single magnitude per head. On the real instrument, that same classifier scores between 26 and 30 percent, depending on the split. In two of the three splits it does not beat "always guess the most common answer."

So the sentence "the classifier has solved interference discrimination" is a claim about a machine that does not exist. I am leaving the original sentences above where they are, because that is genuinely what I believed at that point in the story, and correcting them here rather than editing them out.

And unlike severity, more data does nothing for it. Six times the data, three ways of splitting, both a model that uses labels and one that does not — nothing moves. When a failure ignores all of that, it usually means the information you need is not in the measurement.

## So I asked what the missing information actually was

I had been arguing with the team for a while that the real gap is ground truth: nobody knows exactly what each kind of defect looks like in this data, because nobody has ever measured a defect in isolation. My suggestion was to build test pipes in-house. Clean pipe, known defects, nothing else buried nearby, no welds, position measured with a tape instead of a satellite.

I can simulate exactly that, so before asking anyone to spend money on it, I tested whether it would work.

The first half was a straight vindication. On clean pipe, the detection model that had never once beaten a simple threshold beat it by 0.199 recall, with an interval that stays clear of zero. It also located defects about eleven times more accurately — 75 centimetres instead of 812. Same sensor, same physics, same algorithm. The only difference was that I removed the buried junk and the pipe joints.

That is the answer to "what information is missing." It was never that the defect signal is too weak for this sensor. It is that the interference is designed to be exactly as strange-looking as a defect, and a detector whose whole job is to find strange things cannot separate two things whose strangeness is identical. Take the junk away and it works immediately.

Here is where I had to be careful with myself, because I had proposed this idea and I wanted it to be right.

Making the clean-pipe problem easier proves nothing. Of course it is easier. The question that decides whether to build the test facility is different: if I *train* a model on beautiful clean data, does it work on the messy real pipe I actually have to deploy on?

No. It gets worse. Severity error went from 15.8 to 19.4. The confidence scores got worse too. And there is a failure that is not a matter of statistics at all: a model trained on clean pipe has never seen a buried fence post in its life, so it never predicts one. The one discrimination the real-data model still does above chance, the clean-data model cannot do at all.

One number in that comparison did favour the clean-pipe model, and it is a good example of how a metric can flatter you. Its uncertainty intervals contained the true answer 88 percent of the time on real data, against 60 percent for the model trained on real data. That looks like better calibration until you notice the intervals are more than twice as wide. It was not more accurate. It was vaguer, and being vague is an easy way to be right. I had run the first version of this experiment without printing the interval width at all, which meant I could not tell those two explanations apart. I added the number and ran it again.

The direction held at both corpus sizes I tried. And it hurts in both directions — a model trained on messy data is badly miscalibrated on clean data too, which is what a genuine mismatch between two worlds looks like, as opposed to one of them simply being harder.

Meanwhile, the flattering version of this experiment looks wonderful. Train on clean pipe, test on clean pipe, and every number improves: better severity than the baseline, non-zero recall on all four defect types, well-calibrated intervals. If I had run only that comparison — and it is the obvious one to run — I would have walked into a meeting with a chart showing my idea works.

So the proposal survives, with its purpose changed. Building test pipes is worth it as a *measuring instrument*: it tells you how much of your failure is caused by clutter rather than by physics, and that is a number nobody currently has. It is not worth it as a source of training data. The models that go in the field have to be trained on the field.

I would rather have found that out in a simulation than in a budget.

---

## What I learned

The modelling was the smallest part of the work. Feature engineering and model choice were maybe 15 percent. The rest was data contracts, validation, provenance, and the methods for deciding whether a result is real.

Most of that is not new to a physicist. It is laboratory discipline in a different form.

Recording the exact contents of every input file is a lab notebook you cannot backdate. Knowing that 12 defects measured three times is 12 samples and not 36 is just knowing your independent samples. Refusing to release a model that fails its gate is refusing to publish a result that is not significant.

What I had to learn is that in software these habits must be written as code. Gates, checks, and automatic tests. In the lab, I am the check. In a pipeline that runs at night on data nobody reads, the check must be code, or it does not exist.

The most useful thing I built is a job that turns red when the model is not good enough. Not because red is good, but because a system that can only say yes tells you nothing when it says yes.

My detection model does not beat a simple threshold. I know this with a confidence interval of −0.033 to −0.026, measured over 9,600 defects, on the original vector-head version of the instrument. I know why. And it is written down in the model card, the README, and a red CI badge, instead of in a drawer.

Months later, after the real instrument turned out to be a different, harder machine, the same question came back and got a related but not identical answer: one software step (a first-difference calculation across the three heads) gave a real, statistically significant improvement — small, and it's the only one of five that did — while a genuine hardware upgrade still bought roughly seven times more than that entire software gain. I said in an earlier draft of this piece that no software step reached significance at all. Further feedback on the data changed that, and I corrected it above rather than quietly editing it out. Two different projects, the same discipline, and this time the uncomfortable answer needed revising once, in public, when the numbers actually moved.

Then it needed revising twice. The result I was proudest of — 99.9 percent precision at telling a real defect from buried junk — turned out to belong to an instrument that doesn't exist, and it does not survive on the real one. I found that by running the thing at scale instead of taking the small-corpus number at face value.

The last lesson is the one I did not expect, and it is about my own idea rather than my model. I proposed building test pipes, I believed in it, and I got to test it before anyone spent money. The obvious experiment — is clean data easier to work with — would have told me I was right. The experiment that mattered — does a model trained on clean data work in the field — told me I was half wrong, and told me which half. Designing an experiment that can embarrass you is worth more than running one that can only agree with you, and it is much harder to make yourself do when the idea is yours.

---

*The full project is on GitHub at [AliBaghizadeh/lsm-integrity](https://github.com/AliBaghizadeh/lsm-integrity). The data is synthetic. The results are real, including the ones that did not pass.*
