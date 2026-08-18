# DA-ZVAD — Project Handbook

*Everything you need to know about this project, in one place.
Last updated 19 August 2026, after five GPU runs across two benchmarks.*

> **This document replaces** the separate notes on ML basics, foundations,
> experiments, results and viva questions. Those are archived in
> `docs/_archive/`. The one companion still worth keeping open is
> `03_domain_adaptation_deep_dive.md` — the detailed study of the six survey
> papers your guide assigned.

---

# How to use this

| If you have… | Read |
|---|---|
| 10 minutes | Part 0 |
| 1 hour | Parts 0, 1, 4, 5 |
| An afternoon before the presentation | All of it, then the self-quiz at the end |
| A specific hard question | Part 7 |

---

# Part 0 — The whole project in one page

**What we built.** A system that watches CCTV video and flags unusual events,
which you adapt to a new location by *writing a sentence in English* instead of
retraining it. Nothing in it is trained. Ever.

**Why that's interesting.** Anomaly detectors are normally tied to the place
they learned. Moving one means collecting new footage and retraining. We remove
that cost.

**Why it's research and not just engineering.** The domain-adaptation literature
almost entirely addresses "the same things look different" (covariate shift).
Anomaly detection's actual problem is "the same thing means something different"
(concept shift) — a bicycle is fine on a road and anomalous on a footpath. The
surveys say explicitly that they set concept shift aside. Nobody has attacked it
directly in video anomaly detection.

**What we found.**

1. **0.734 AUROC** on ShanghaiTech with zero training data — matching the
   benchmark's own 2018 trained baseline (0.728), which used its training set.
2. Giving the system a **wrong** scene description costs **0.105 AUROC**. Since
   every model is frozen, nothing else could have caused that. The text is doing
   real work.
3. **Where** you inject the description matters more than what it says — putting
   it in the wrong place *reverses* the effect. Not previously reported.
4. It **doesn't replicate** on a single-scene benchmark, and we measured why: the
   sentence mainly tells the system *which* place it is looking at.
5. **Six** attempted improvements all failed. The simplest configuration wins.

**The five sentences to memorise**

1. Anomaly detectors are tied to where they learned; we adapt by writing a
   sentence.
2. The DA literature targets covariate shift; anomaly detection is concept shift.
3. 0.734 with no training, and a wrong sentence costs 10 points.
4. Where the sentence is injected matters more than what it says.
5. It works where there are several scenes to tell apart, and we proved that.

---

# Part 1 — The background you actually need

Skip this if you're comfortable with embeddings, softmax and AUROC.

## 1.1 Training vs inference

**Training** is when a model adjusts its internal numbers (parameters) to fit
data. **Inference** is using a trained model without changing it.

Our entire thesis sits on this distinction: **we only ever do inference.** No
parameter in our system is ever updated. This is not laziness — see §4.2.

## 1.2 Embeddings

A neural network converts an image into a list of numbers — say 768 of them.
That list is an **embedding**. Similar images get similar lists.

The useful trick: CLIP was trained so that **an image of a dog** and **the words
"a photo of a dog"** produce *nearby* lists. Pictures and sentences live in the
same space and can be compared directly.

**Cosine similarity** measures how close two lists are, from −1 to +1. That
single number is the foundation of everything we do.

## 1.3 How CLIP got its knowledge

CLIP was trained on ~400 million image–caption pairs scraped from the web. It was
never told about anomalies, campuses or bicycles specifically. It learned a
general association between pictures and language.

That breadth is why it works on our task without training — and why it's robust
to lighting and camera changes (see covariate shift, §2.2).

## 1.4 Softmax

Given two raw scores, softmax converts them into probabilities that sum to 1.
We give it two numbers — how well the frame matches "normal" text and how well
it matches "abnormal" text — and it returns *the probability the frame is
abnormal*. That's our anomaly score.

## 1.5 AUROC — the only metric that matters here

> Pick one anomalous frame and one normal frame at random. **AUROC is how often
> the system gives the anomalous one a higher score.**

- 0.50 = coin flip
- 0.734 = right about 73 times in 100
- 1.00 = perfect

**Why not accuracy?** Because anomalies are rare. A detector that says "normal"
to everything gets 95% accuracy and detects nothing. AUROC can't be fooled that
way because it measures *ranking*, not a threshold.

**Micro vs macro.** Micro pools every frame from every clip into one ranking.
Macro scores each clip separately and averages. When they disagree, something is
wrong with how clips compare to each other — which is exactly what happened to
us (§5.3).

---

# Part 2 — The research problem

## 2.1 The practical problem

A mall installs an anomaly detector. It learns over weeks that slow walking is
normal, shutters close at 9pm, nobody's around at 3am.

Sell the same system to a factory and it fails. A forklift in the aisles is
routine there and alarming in a mall. **Same footage, opposite answer.**

So: new customer, new footage, new training run. That's the cost.

## 2.2 The four kinds of "difference between places"

This vocabulary is from the DA literature and you must own it. A domain is a
joint distribution `P(x, y)` over inputs `x` and labels `y`, which factors as
`P(y|x) · P(x)`.

| Type | What changes | Example |
|---|---|---|
| **Covariate shift** | `p(x)` — the inputs | A stop sign in fog vs sunshine |
| Conditional shift | `p(x\|y)` | A class looks different across domains |
| Label shift | `p(y)` | Class proportions differ |
| **Concept shift** | `p(y\|x)` — **the rule** | A bicycle on a road vs a footpath |

## 2.3 The gap — say this precisely

**Liu et al. (2022)** state their scope outright:

> *"[Concept shift] is, however, usually not a common problem in popular object
> classification or semantic segmentation tasks. As such, this review mainly
> focuses on the covariate shift alignment in UDA."*

**Singhal et al. (2023)** list stable `p(y|x)` as the **first condition** under
which domain adaptation is theoretically justified.

So classical DA theory **assumes concept shift away.**

**But anomaly detection is defined by it.** "Normal" is a property of the
deployment context, not of the object. That's the definition of the task.

**Consequence:** methods that align `p(x)` — the dominant family in the whole
literature — structurally cannot help, because `p(x)` can be *identical* across
two domains while only the labelling function differs.

## 2.4 One honest caveat

A paper published April 2026 (**Wilkinghoff et al.**) argues the same premise
independently: normality is context-dependent and should be judged
conditionally. We cite it.

**This is good for you.** It means the problem is real and recognised, not
invented to have something to solve. What they leave open is whether supplied
context actually *does* anything — which is what we measure. Their paper is a
position piece with no experiments.

---

# Part 3 — Where we sit in the literature

## 3.1 The neighbours

| Work | What it does | What it leaves open |
|---|---|---|
| **LAVAD** (CVPR'24) | Training-free VAD, ~0.85 | Three large models per frame; no channel for an operator to state site normality; adaptation never measured |
| **VERA** (CVPR'25) | Frozen VLM + verbal guiding questions | Questions are **learned offline on source data** — not test-time |
| **AnyAnomaly** (2025) | User names the anomaly in text at inference | Demonstrates text works; never tests whether text is the **cause** |
| **OVVAD** (CVPR'24) | Frozen CLIP, open-vocabulary | **Trains** a temporal adapter |
| **Ada-VAD, zxVAD** | Cross-domain VAD | Adaptation via pixels and parameters — never language |

## 3.2 What is genuinely ours

Be precise here. Overclaiming is how you lose a viva.

**Not novel:** supplying text at inference time to steer a frozen detector.
AnyAnomaly does that. **Name it yourself before they do.**

**Novel:**

1. **The characterisation** — framing VAD as concept-shift-dominated in the DA
   literature's own vocabulary, and showing the surveys scope it out.
2. **The protocol** — a four-condition sweep including a *deliberately wrong*
   description as a falsifying control. Others supply text and report it works;
   nobody supplies wrong text to check the system is listening.
3. **The finding** — injection point dominates wording, to the point of
   reversing the effect.
4. **The boundary** — measured, not asserted (§5.7).

## 3.3 The argument that ties it together

> "You cannot run our central experiment on any of those systems. VERA's text is
> optimised on source data. LAVAD's notion of anomaly is entangled in a language
> model's prior. OVVAD has a trained adapter. **Ours is the only pipeline where
> text is the sole free variable — which is the precondition for measuring
> whether text does anything.** The architecture exists to make the experiment
> possible."

---

# Part 4 — The method

## 4.1 The four modules

**M1 — Visual scoring (frozen CLIP ViT-L/14)**
Each frame is embedded. Two sets of sentences — "normal" and "abnormal" — are
also embedded and each set averaged into a single vector (a *prototype*). The
frame's score is the softmax probability that it matches the abnormal prototype
rather than the normal one.

*In plain terms:* which of two descriptions does this picture look more like?

**M2 — Temporal smoothing**
Average each frame's score with its neighbours over a window of `w` frames. Real
events last seconds; noise doesn't. No parameters. Our best `w` is **31**.

**M3 — Verbalised domain context** ← *the research*
Your sentence about the deployment site is injected into the prompts, so
"normal" means normal *here*.

**M4 — Explanation (frozen LLaVA, 4-bit)**
For each flagged event, generates a sentence explaining what looks wrong.
**Never run on video. This is Phase 3 work.**

## 4.2 Why freezing everything is the point

If the system learned *anything* from the new site and performance improved, you
could not say what caused it — the sentence, or the learning. They're tangled.

Freezing every parameter means **exactly one thing can vary: the text.** So any
measured difference is caused by the text. There is no other candidate.

This is called **identifiability**. It's the strongest methodological property
of your design, and it's why the architecture is shaped the way it is.

## 4.3 The context sweep — the central experiment

Run the identical pipeline four times, changing only the sentence:

| Condition | Sentence | Purpose |
|---|---|---|
| `none` | M3 off | Lower reference |
| `generic` | "a generic scene" | Controls for merely *having* context |
| `matched` | Correct description | The proposed operating condition |
| `mismatched` | **Wrong domain's** description | **Falsifying control** |

**Predicted signature, fixed before measurement:**
matched ≥ generic ≥ none, with mismatched measurably **worse**.

If all four coincide, the method ignores its context and the claim is refuted.
This is a test that could have failed — and partly did.

---

# Part 5 — What we did and what we found

## 5.1 The setup

| | |
|---|---|
| Benchmarks | ShanghaiTech (13 camera views), CUHK Avenue (1 view) |
| Scale | 128 test clips, 28,118 frames scored, frame-level labels |
| Hardware | NVIDIA A40, college GPU server |
| Backbone | CLIP ViT-L/14 (LAION-2B), frozen |
| Runs | 5, each with a manifest recording code commit, GPU, config, counts |

## 5.2 Day one: it failed completely

First full run on ShanghaiTech returned **0.49** — chance. And the context sweep
came out **backwards**: the correct description was the *worst* condition.

## 5.3 Problem 1 — we were measuring it wrong

ShanghaiTech has 13 cameras. CLIP sits at a different baseline score under each
(different lighting, angle, crowd density).

We were pooling every frame from all 13 into one ranking. **That's like ranking
students from different schools by raw marks when the schools grade
differently** — the comparison destroys the ordering that exists within each
school.

The benchmark's own published protocol normalises each clip to a common range
first. We weren't doing it.

> **Applying it took 0.49 → 0.71 with no change to the system.**

**Be ready to defend this.** It uses **no labels**. It's the protocol from the
paper that created the benchmark. And we report **both** figures in the paper so
anyone can check.

**The diagnostic that proved it:** the macro figure (immune to cross-clip
offsets by construction) sat at 0.67 while raw micro sat at 0.52. That gap
identifies the cause. Only 31 of 105 scorable clips were below chance
individually — the signal was there, pooling was hiding it.

## 5.4 Problem 2 — the description cancelled itself out

We were appending the scene sentence to **both** prompt sets:

- normal: *"a campus walkway with pedestrians, everything is normal"*
- abnormal: *"a campus walkway with pedestrians, but something is wrong"*

Both start with the same words. Each set is averaged into one prototype, so
shared text becomes a **common component of both** — pulling them toward each
other and shrinking the margin the decision depends on.

**Which explains the inversion.** A description that accurately describes the
imagery aligns strongly with *every* frame, so it absorbs the most contrast — the
more accurate, the more damage. A factory description aligns with nothing, so it
leaves the original prompts intact.

**The fix:** attach the description to the **normal ensemble only**. This also
fits the idea: the scene tells you what normal looks like here; an anomaly is a
departure from it.

## 5.5 Day two: a second dataset, and a bug of our own

We ran **CUHK Avenue** to check the finding holds elsewhere. It didn't (§5.7).

While investigating, we found **an error in our own analysis code**. A script we
had been exploring with scored the *cosine margin* where the real pipeline
scores its *softmax probability*.

Those rank frames identically — we checked, and the rank correlation was exactly
**1.00**. But the benchmark's metric normalises each clip to [0,1] first, and
that step is **affine**. An affine map applied after a nonlinear monotone one is
not the same operation. So the two produced different pooled figures despite
identical rankings.

**About four hours of conclusions from that script were void.** We corrected
every affected number.

> **The lesson, now a project rule:** any new analysis tool must reproduce a
> known result from the existing pipeline before its output is trusted. A rank
> correlation of 1.0 does *not* prove two scorings are equivalent under a metric
> that normalises per clip.

**This is worth presenting.** It produced a genuine methodological finding — the
metric this field uses is sensitive to the *scale* of your score, not just its
ordering — which is now a limitation in your paper and applies to everyone using
the protocol.

**What the correction changed:**

| | before | after |
|---|---|---|
| Headline | 0.706 | **0.734** |
| Best window | 15 | **31** |
| Best components | language **+ motion** | **language alone** |

**A finding we retracted:** we previously believed a motion signal combined
usefully with language. Measured correctly, motion adds nothing.

## 5.6 The results

### Result A — the detector works, and language alone is best

| Signal | Held-out | Full test set |
|---|---|---|
| **Language only** | **0.718 ± 0.036** | **0.707** |
| Language + motion | 0.711 ± 0.034 | 0.706 |
| Motion only (no language) | 0.685 ± 0.015 | 0.686 |
| Language + clip-average | 0.645 ± 0.034 | 0.640 |
| Clip-average only | 0.585 ± 0.025 | 0.585 |

Adding motion changes nothing. That surprised us — ShanghaiTech's anomalies look
kinematic (a bicycle at cycling speed), so the two *should* combine. They don't,
which suggests the frozen encoder already registers enough of the motion.

With the correct description and `w=31`, the full system reaches **0.734**.

### Result B — the temporal window, and the pooling protocol

| Pooling convention | w=1 | w=5 | w=15 | **w=31** |
|---|---|---|---|---|
| Micro, raw scores | 0.493 | 0.502 | 0.513 | 0.519 |
| Macro (per-clip mean) | 0.614 | 0.628 | 0.648 | 0.670 |
| **Micro, per-clip normalised** | 0.667 | 0.685 | 0.702 | **0.707** |

`w=31` is a genuine optimum — performance *falls* to 0.683 at `w=61`, so the
metric isn't simply rewarding heavier smoothing.

### Result C — the central experiment

All 107 clips, `w=31`, per-clip normalised. Every model frozen; only the sentence
and its injection point vary.

| Injection point | none | generic | matched | mismatched | **gap** |
|---|---|---|---|---|---|
| **Both** prompt sets | 0.707 | 0.670 | 0.666 | 0.695 | **−0.029** |
| **Normal set only** | 0.707 | 0.691 | **0.734** | 0.628 | **+0.105** |

The `none` column is **identical** in both rows — that's the internal control,
confirming nothing but the injection point changed.

> **State the claim precisely.** The correct description beats none by **+0.027**
> — positive at every window and growing with it, so the direction is real, but
> inside the ±0.036 split-to-split spread, so we report the direction and not the
> size. **The claim rests on the wrong description costing 0.105**, which is
> unambiguous.

### Result D — it did not replicate on Avenue

| Condition | ShanghaiTech | Avenue |
|---|---|---|
| No sentence | 0.707 | 0.706 |
| Correct sentence | **0.734** | 0.677 |
| Wrong sentence | 0.628 | 0.657 |
| **Gap** | **+0.105** | **+0.020** |

**Detection transfers perfectly** — 0.706 vs 0.707. The detector works equally
well on both benchmarks with no retuning.

**The adaptation does not.** The gap is five times smaller, and on Avenue the
*correct* description scores **below** no description at all. A vague placeholder
("a generic scene") is the best condition there.

**Concede this openly:** since the placeholder carries no information about the
environment, whatever it contributes on Avenue **cannot be domain adaptation** —
most likely it just improves the prompt ensemble as an ensemble.

## 5.7 Result E — and then we worked out why

ShanghaiTech has **13 camera views**. Avenue has **one**.

Hypothesis: a scene description only has work to do when there are several
places to tell apart. On Avenue every clip shows the same view, so the base
prompts already cover the only environment present.

**This is testable without a third dataset**, because ShanghaiTech is really 13
single-view datasets stacked together (clip `01_0014` is view 01). So we ran the
same sweep **inside each view separately**:

| Evaluation | Gap |
|---|---|
| Pooled across 13 views | **+0.105** |
| Within a single view (mean of 9) | **+0.034** |
| Avenue (single view) | +0.020 |

**The prediction held.** A within-view gap near +0.105 would have refuted it.

> **What the sentence mainly does is tell the system *which* place it is looking
> at — not what counts as normal within that place.**

**Caveats to state:** the per-view estimates rest on 5–34 clips each, three of
nine are negative, and the standard deviation (0.058) exceeds the mean (0.034).
One view (12) scores 0.415 — below chance — under every condition, and we have no
explanation for that.

## 5.8 Result F — six things that did not work

| Modification | Outcome |
|---|---|
| Quadrant scoring, to catch small objects | No improvement in any configuration |
| Prompts naming bicycles and vehicles | Much worse alone — 0.486 |
| Clip's own average as normality reference | 0.585, and degrades the language signal |
| Adding a motion signal | Costs 0.001 — not complementary |
| Local temporal deviation before projection | Better scorer, but *narrows* the context gap |
| Per-prompt max pooling | 0.678 vs 0.707 |

> **This is a strength, not an admission.** Your claim is that the *simplest*
> configuration is right. That's only believable next to the alternatives you
> tried. And a clean table of successes is the artefact that's easy to fabricate;
> six diagnosed failures are not.

---

# Part 6 — Honest assessment

## 6.1 As a detector: modest

LAVAD reaches ~0.85. You reach 0.734. That gap is real and you should not
minimise it.

**But the fair comparison is different.** Liu et al. (2018) — the paper that
*created* ShanghaiTech — reach ~0.728 by training on ShanghaiTech's own training
split.

> **You reach 0.734 having never seen a frame of it.**

And the deployment framing matters: at a new site, a trained model's accuracy
isn't 0.728 — it's *undefined*, until someone spends weeks producing it.

**Don't overstate the practical case.** 0.734 is not a deployable autonomous
detector. The honest positioning is **cold start**: coverage from day one while
data is collected for a trained system, or triage that reduces hours of footage
to minutes of review.

## 6.2 As research: genuinely sound

- A claim that could have been proven wrong
- A test designed to prove it wrong — which partly did
- An explanation for the failure, and a control that confirmed it
- Six failures reported alongside successes
- Two of your own errors caught and corrected
- Every number traceable to code and hardware

## 6.3 Will it get much better?

Probably not dramatically, and you should understand why.

Your decision rule is **one vector compared against two sentences**. That's a
linear discriminant — it cannot *reason*. LAVAD captions each frame and has a
language model think about the caption, which can hold "bicycle" and "footpath"
as separate facts and combine them. No amount of tuning a two-prototype
comparison recovers compositional reasoning.

Evidence that you're near this architecture's ceiling:
- Prompt content barely matters (generic ≈ specific)
- A signal with zero semantics (motion) gets within 0.02
- Six modifications all absorbed

**Patch-level scoring** is the best remaining idea — might reach 0.78–0.82.
Reaching 0.85 would mean adding a captioner and an LLM, **which would destroy
the identifiability that makes your claim measurable.**

> Say it as a trade-off, not an apology: *"The architecture is capped by the same
> property that makes its central claim testable."*

## 6.4 Is it publishable?

As it stands, not for a strong venue. You have a characterisation, a protocol,
two findings and a measured boundary — but no new architecture, and the central
effect holds on one of two video benchmarks.

For the **thesis**, it's comfortably sufficient. For a paper, Phase 3 would need
to add the MVTec sweep and ideally patch-level scoring.

---

# Part 7 — Defending it

Each entry: what to say first, the fuller version, and the phrasing that loses
the room.

## A. Novelty

**"Isn't this already done? Zero-shot, domain adaptation and explanation all exist."**

> Each ingredient exists; the combination doesn't. But the combination isn't the
> argument — what it *enables* is. You cannot run our central experiment on any
> competing system, because in each of them something other than the text is free
> to move. The architecture exists to make the experiment possible.

⚠️ Don't claim "text at inference time" as novel. AnyAnomaly does that.

**"What exactly is the research gap?"**

> Two halves. The DA literature explicitly excludes the shift type anomaly
> detection is built on. And the VAD papers that use language never test whether
> the language is causally responsible — they supply text, report it works, and
> stop.

**"Isn't 'nobody ran a control' trivial?"**

> A control is trivial when it confirms. Ours overturned the result. The
> contribution isn't the ablation — it's the failure mode it exposed: the obvious
> implementation actively harms, and the harm scales with how *accurate* your
> description is. That's a property of a whole class of methods, not our code.

⚠️ Never say "we ran an ablation nobody ran." That phrasing *is* trivial.

## B. The accuracy

**"Existing models do better. Why do this?"**

> They do better *when they have training data from the deployment site*. On a
> benchmark everyone has it; at a new site nobody does. Also, accuracy was never
> the question — we asked whether language alone can carry adaptation and whether
> that can be measured.

⚠️ Never say "we're not competing on accuracy" and stop. Always pair it with what
you *are* competing on.

**"Your MVTec result was 88.5%. Why is video only 0.734?"**

> Different tasks — but the MVTec breakdown predicted it. Performance there fell
> as the defect occupied fewer pixels: textures 99.5%, objects 83%, transistor
> 72.7%. ShanghaiTech's anomaly is 1–2% of the frame. One trend, two benchmarks.

**Follow-up — "then why didn't quadrant scoring fix it?"**
> It should have and didn't. Mean aggregation beat max, which is the reverse of
> what recovering a small object looks like. Resolution is our leading
> explanation, not a proven one. Patch-level scoring is the proper test and we
> haven't run it.

## C. Rigour

**"Did you tune on the test set?"**

> Partly, and we control for it. No validation split is defined for these
> benchmarks, so we split clips in half, chose settings on one half, and report
> the half never used — averaged over five splits with the spread reported. Where
> configurations sit inside that spread we report them as tied.

**"Isn't the normalisation flattering the number?"**

> It uses no labels — it puts 13 camera views on a common scale before comparing.
> It's the benchmark's own protocol, and we report the un-normalised figure in
> the same table.

**"The correct description barely beats no description."**

> +0.027, positive at every window and growing with it. But it's inside the
> ±0.036 split spread, so I report the direction, not the size. The claim rests
> on the *wrong* description costing 0.105.

**"Motion alone gets 0.686 without language. What's language for?"**

> Language reaches 0.707 and adding motion costs 0.001 — they're not
> complementary. That surprised us; we expected them to combine. It suggests the
> frozen encoder already registers enough of the motion.

**"Why did your numbers change between drafts?"**

> An analysis script scored the cosine margin where the pipeline scores its
> softmax. They rank identically, but the per-clip normalisation in the metric is
> affine and doesn't commute with a nonlinear monotone map. We caught it by
> requiring the analysis tool to reproduce a pipeline result, corrected every
> affected number, and the metric's scale-sensitivity is now a stated limitation.

## D. The hardest one

**"Your central finding doesn't replicate on Avenue. Doesn't that refute it?"**

> It bounds it rather than refuting it. Detection transfers almost exactly —
> 0.706 against 0.707. What doesn't transfer is the context effect: the gap falls
> from +0.105 to +0.020 and the correct description scores below none.
>
> We tested why. ShanghaiTech has thirteen camera views; Avenue has one.
> Confining ShanghaiTech to a single view reproduces Avenue's flat result
> (+0.034). So the descriptor mainly identifies *which* environment is in view.
>
> And I can't explain away that a placeholder beats an accurate description on
> Avenue. But the mechanism is now measured, not guessed.

⚠️ Don't say "it works on ShanghaiTech and needs more investigation on Avenue."
That's evasive. State the near-vanishing effect and the placeholder result
*first*, then the mechanism.

## E. Credibility

**"How do I know you ran this?"**

> Every run writes a manifest — code commit, working-tree state, host, GPU,
> driver and library versions, full configuration, frame and label counts. Five
> of them, committed with the results.
>
> And the internal control is checkable: two runs differ only in where the
> descriptor is injected, and their "no context" columns agree to four decimals.

*(Have `results/runs/2026-08-14_162056_surv_normal/MANIFEST.txt` open.)*

**"Walk me through what went wrong and how you found it."** ← *your best question*

Take your time. The first run was chance and the sweep came out backwards. Three
causes: pooling across incomparable camera baselines; the descriptor entering
both prompt sets and collapsing the margin; and a hypothesis of ours about
campus-specific prompts that made things worse. Then day two: Avenue didn't
replicate, and an error in our own analysis code invalidated four hours of work.

## F. What you cannot answer yet — know these

1. **Why context helps on one benchmark and not the other** — the scene-diversity
   account is supported but the per-view estimates are noisy.
2. **The dilution mechanism is inferred, not directly measured.** The prototype
   distance under each condition has never been computed. Takes minutes.
3. **The industrial↔surveillance contrast is argued, not measured.** The sweep
   has never been run on MVTec, and both video benchmarks are outdoor pedestrian
   surveillance.
4. **M4 has produced nothing.**
5. **Does AnyAnomaly already run a wrong-text control?** Needs checking. Read the
   paper (arXiv 2503.04504) before the viva.
6. **The temporal window was selected on the test set** — no validation split
   exists for these benchmarks.

## The three sentences to fall back on

> The domain-adaptation literature excludes concept shift by explicit choice, and
> anomaly detection is built on concept shift.
>
> The papers that use language demonstrate it works but never test whether it is
> causally responsible.
>
> We characterise the first and supply a protocol for the second — including a
> falsifying control that our own framework initially failed.

---

# Part 8 — What's left

## Phase 3 (this semester)

| Priority | Work | Cost | Why |
|---|---|---|---|
| 1 | Per-scene descriptors | ~1 hr | 13 views currently share one sentence. Follows directly from Result E |
| 2 | Context sweep on MVTec | ~1 hr | Your concept-shift argument needs the industrial contrast measured |
| 3 | Patch-level scoring | 1–2 sessions | Best shot at a materially higher number |
| 4 | Run M4 | ~1 session | A quarter of the framework; strong demo material |
| 5 | Cross-domain descriptor swap | ~1 hr | ShanghaiTech's sentence on Avenue and back — the fair transfer test |
| 6 | Direct dilution measurement | minutes | Turns the mechanism from inferred to measured |

## Phase 4

Paper submission. Possibly a reproduced baseline if your guide wants one — if so,
request Llama-2 access early, since approval is the long pole.

---

# Part 9 — Where everything lives

| Need | Location |
|---|---|
| This handbook | `docs/08_understanding/00_HANDBOOK.md` |
| The six DA surveys, in depth | `docs/08_understanding/03_domain_adaptation_deep_dive.md` |
| The paper | `docs/09_paper/main.tex` |
| The deck (21 slides + notes) | `docs/06_presentations/DA-ZVAD_Phase2_Review.pptx` |
| **Proof you ran it** | `results/runs/*/MANIFEST.txt` |
| Raw result tables | `results/runs/*/tables/*.csv` |
| Post-hoc analysis | `results/runs/analysis/` |

⚠️ **Read `results/runs/README.md` before showing anything from `analysis/`** —
about half those CSVs come from the buggy analysis path and are marked
superseded.

**On the server:** `~/dazvad/` holds the datasets, the 300 MB embedding cache and
cached scores. Reconnect with `ssh m251250cs@192.168.41.119`, then
`tmux new -s dazvad` and `source ~/dazvad/venv/bin/activate`.

---

# Self-quiz

Answer out loud, without looking. If you can't, reread the section named.

1. What is concept shift, and why does anomaly detection have it? *(§2.2–2.3)*
2. Why does freezing every model make the claim measurable? *(§4.2)*
3. What is the mismatched condition for, and what would refute you? *(§4.3)*
4. Why did pooling raw scores across 13 cameras give 0.52? *(§5.3)*
5. Why did an **accurate** description do the most damage? *(§5.4)*
6. What does +0.105 mean, and why is +0.027 stated differently? *(§5.6 C)*
7. What happened on Avenue, and what did the within-view control show? *(§5.6 D, §5.7)*
8. Name three of the six failed modifications. *(§5.8)*
9. Why won't this architecture reach 0.85? *(§6.3)*
10. What can you *not* yet answer? *(§7 F)*
