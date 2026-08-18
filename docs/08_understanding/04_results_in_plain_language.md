# What We Did, In Plain Language

*Written 14 August 2026. **Substantially revised 19 August 2026** after a second
round of experiments corrected several results — including one finding that
turned out to be wrong. Read this before your Phase 2 presentation. No prior
technical reading needed.*

---

## 1. The whole thing in one paragraph

We are building a system that watches CCTV video and flags unusual events. The
normal way to build one is to show it thousands of hours of ordinary footage
from *that specific camera* until it learns what "ordinary" looks like there.
Our system skips that entirely. Instead, you **write a sentence in English**
describing the place — "a university campus walkway with pedestrians" — and the
system uses that sentence to decide what counts as unusual. Nothing is trained.
We tested it on two standard benchmarks, found that it works on one and not the
other, and then worked out why.

---

## 2. The problem we are attacking

### An anomaly detector is tied to the place it learned

Suppose a company sells an anomaly-detection system to a shopping mall. Over
weeks of footage it learns that people walking slowly is normal, that the
shutters close at 9pm, that nobody is around at 3am.

Now sell the same system to a factory. It fails completely. Everything is
different — the lighting, the layout, and above all the *meaning* of what it
sees. A forklift moving through the aisles is perfectly normal in a factory and
alarming in a mall.

So the company must collect new footage and retrain. For every new customer.
That is the cost we want to remove.

### The academic name for this problem

Making a model work in a new place is called **domain adaptation**. It has a
large research literature — our guide gave you six survey papers about it.

Here is the important thing, and it is the foundation of your entire thesis:

> **Almost all of that literature solves a different problem from ours.**

Researchers distinguish between kinds of "difference between places":

**Covariate shift** — *the same things look different.*
A stop sign in bright sun versus a stop sign in fog. It is still a stop sign;
the answer is unchanged. Only the appearance moved.

**Concept shift** — *the same thing gets a different answer.*
A bicycle on a road is fine. The identical bicycle on a pedestrian footpath is
an anomaly. Nothing about the picture changed. The *rule* changed.

The surveys say plainly that they focus on covariate shift, and that concept
shift is rare enough to set aside. That is a fair judgement for their
problems — in object recognition, a cat is a cat everywhere.

**But anomaly detection is built on concept shift.** "Normal" is defined by
where you are. That is the definition of the task. So the field's main toolkit
is aimed somewhere other than our actual problem.

> **One honest note.** A paper published this April (Wilkinghoff et al.) argues
> the same premise independently — that normality is context-dependent and
> should be judged conditionally. We cite it. That is *good* for you: it means
> the problem is real and recognised, not invented to be solved. What they leave
> open is whether supplied context actually does anything, which is what we
> measure.

---

## 3. Our idea

If the difference between two places is a difference in *rules*, and rules are
the kind of thing you can state in words — then let the operator state them.

So: **freeze everything, and adapt by changing a sentence.**

### Why "freeze everything" is a clever move, not a lazy one

This is the strongest thing in your whole design and a panel member may well
probe it.

If we let the system learn even a little from the new place, and performance
improves, we cannot tell *why*. Did the sentence help? Or did the learning help?
The two are tangled.

Because we freeze every single parameter, only one thing in the entire system
can vary: **the sentence**. So if performance changes, the sentence caused it.

**And this is not available to the competing systems.** You could not run our
central experiment on them — VERA optimises its text on source data, LAVAD's
notion of anomaly is buried in a language model's prior, OVVAD trains an
adapter. In each, something else is free to move.

---

## 4. What the system actually does

Four parts.

**M1 — The looker (CLIP)**
A pre-trained model that compares a picture to a sentence and says how well they
match. We hand it two sets of sentences — describing normal, and describing
abnormal — and for each frame it reports which side matches better. **No
training happens.** The language itself does the classifying.

**M2 — The smoother**
Single frames are noisy. We average each frame's score with its neighbours. Real
events last several seconds; noise does not.

**M3 — The scene description** ← *this is the research*
Where your sentence about the deployment site gets mixed into M1's sentences.

**M4 — The explainer (LLaVA)**
For each flagged event, a second model writes a sentence saying what is wrong.
**Still not run.**

---

## 5. What happened, as a story

This matters more than the numbers. It is what you should actually present.

### Day one: it failed completely

We ran the full experiment on **ShanghaiTech** (campus CCTV, 107 clips, 40,791
frames, every frame human-labelled). Fifty minutes.

**Result: 0.49.** Explained below, but for now: **0.50 means random guessing.**

Worse, the central experiment came out *backwards*. We predicted that the
**correct** description of the scene would help and a **wrong** one would hurt.
We got the opposite.

### Problem 1 — we were measuring it wrong

ShanghaiTech has 13 different cameras. CLIP gives slightly different baseline
scores under each — different lighting, different angle.

We were dumping every frame from all 13 cameras into one list and ranking them
together. That is like ranking students from different schools by raw marks when
the schools grade differently.

The standard fix, used by every published paper on this benchmark, is to put
each clip's scores on a common scale before combining. **That alone took the
score from 0.49 to 0.71**, with no change to the system.

> Not cheating: it uses **no labels**, it is the benchmark's own published
> protocol, and we report **both** numbers in the paper.

### Problem 2 — the description was cancelling itself out

We were adding your scene sentence to *both* the normal and the abnormal
sentences:

- normal: *"a campus walkway with pedestrians, everything is normal"*
- abnormal: *"a campus walkway with pedestrians, but something is wrong"*

Both now start with the same words. The system averages each group into a single
summary. When both summaries share a phrase, they become **more similar to each
other** — and the method depends on them being *different*.

And now the backwards result makes sense: **the more accurate your description,
the worse the damage.** An accurate description matches every frame strongly, so
it dominates both sides. A description of a factory matches nothing, so it does
no harm.

The fix: put the description **only on the normal side**. After that, the
experiment behaved exactly as predicted.

### Day two: a second dataset, and a bug of our own

We then ran **CUHK Avenue**, a second surveillance benchmark, to check the
finding holds elsewhere. It did not — more on that below.

While investigating, we found **an error in our own analysis code**. A script
we had been using to explore ideas was scoring the numbers slightly differently
from the real system. The two ranked frames identically — we checked, and the
correlation was exactly 1.0 — but the benchmark's scoring recipe involves
rescaling each clip, and that step is affected by the difference.

Roughly four hours of conclusions drawn from that script were void. We corrected
every affected number.

> **This is worth presenting.** It produced a genuine methodological finding:
> the metric this whole field uses is sensitive to the *scale* of your score,
> not just its ordering. That is now a limitation in your paper, and it applies
> to everyone using the protocol, not just us.

### What the correction changed

| | before | after |
|---|---|---|
| Headline | 0.706 | **0.734** |
| Best temporal window | 15 | **31** |
| Best components | language **+ motion** | **language alone** |

**A finding we had to retract:** we previously believed a motion signal combined
usefully with the language signal. Measured correctly, motion adds nothing. If
you saw an earlier version saying otherwise, this is the correction.

---

## 6. The results, and what the numbers mean

### First: what does 0.734 mean?

**AUROC.** The only definition you need:

> Pick one anomalous frame and one normal frame at random. AUROC is **how often
> the system gives the anomalous one a higher suspicion score.**

- **0.50** = coin flip
- **0.734** = correct about 73 times out of 100
- **1.00** = perfect

### Result A — the detector works

| what is measured | full test set |
|---|---|
| **Language only** | **0.707** |
| Motion only (no language) | 0.686 |
| Language + motion | 0.706 |
| Clip's own average as reference | 0.585 |

**Language alone is best.** Adding motion changes nothing. That surprised us —
ShanghaiTech's anomalies look kinematic, so the two *should* combine. They
don't, which suggests the frozen encoder already registers enough of the motion.

With the correct scene description and the best temporal window, the full system
reaches **0.734**.

### Result B — the central experiment

Four runs, changing **only the sentence**, every model frozen:

| sentence given | score |
|---|---|
| No sentence | 0.707 |
| Vague ("a generic scene") | 0.691 |
| **The correct one** | **0.734** |
| **A deliberately wrong one** | **0.628** |

**A wrong description costs 10 points.** Nothing else could have caused it —
same model, same weights, same video.

> **Be precise.** The correct description beats none by +0.027. That is positive
> at every temporal window and grows with the window, so the direction is real —
> but it sits inside the ±0.036 variation between data splits, so we report the
> direction and don't claim the size. **The claim rests on the wrong description
> failing, not on the right one succeeding.**

### Result C — it did not replicate on the second benchmark

| | ShanghaiTech | Avenue |
|---|---|---|
| No sentence | 0.707 | 0.706 |
| Correct sentence | **0.734** | 0.677 |
| Wrong sentence | 0.628 | 0.657 |
| **Gap** | **+0.105** | **+0.020** |

**Detection transfers perfectly** — 0.706 versus 0.707. The detector works
equally well on both.

**The adaptation does not.** The gap is five times smaller, and on Avenue the
*correct* description scores below no description at all. A vague placeholder
beats an accurate one.

That was the falsifying control doing exactly what it was built to do.

### Result D — and then we worked out why

ShanghaiTech has **13 camera views**. Avenue has **one**.

Our idea: a scene description only has work to do when there are several places
to tell apart. On Avenue every clip shows the same view, so there is nothing to
resolve.

That is testable *without a third dataset*, because ShanghaiTech is really 13
single-view datasets stacked together. So we ran the same experiment **inside
each camera view separately**:

| evaluation | gap |
|---|---|
| Pooled across 13 views | **+0.105** |
| Within a single view (average) | **+0.034** |
| Avenue (single view) | +0.020 |

**The prediction held.** Confine it to one scene and the effect collapses to
near Avenue's level.

> **So what the sentence mainly does is tell the system *which* place it is
> looking at — not what counts as normal within that place.**

That is narrower than we originally claimed, and it is *measured* rather than
argued. It also predicts where the method should be useful: installations
covering several environments, not a single fixed camera.

### Result E — six things that did not work

Quadrant scoring, motion, the clip's own average, local temporal deviation,
per-prompt max pooling, and prompts naming bicycles and vehicles. All neutral or
harmful.

> **This is a strength.** Your claim is that the *simplest* configuration is the
> right one. That is only believable next to the alternatives you tried.

---

## 7. Are these results promising?

### As a detector: modest, with one comparison worth making

LAVAD (CVPR 2024) reaches ~0.85 — but runs three large models on every frame.
You run one frozen model and a sentence, in about 7 GB.

**The fair comparison is this one:**

> Liu et al. (2018), the paper that created ShanghaiTech, reach ~0.728 by
> training on ShanghaiTech's own training data. **You reach 0.734 having never
> seen a frame of it.**

### As research: yes, genuinely

- A claim that could have been proven wrong
- A test designed to prove it wrong — which it partly did
- An explanation for the failure, then a control that confirmed the explanation
- Six failures reported alongside the successes
- Two of your own errors caught and corrected
- Every number traceable to the code and machine that produced it

### Will it get much better?

Probably not dramatically, and it is worth knowing why. Your decision rule is
one vector compared against two sentences — it cannot *reason*. LAVAD captions
each frame and has a language model think about the caption, which can hold
"bicycle" and "footpath" as separate facts and combine them.

Patch-level scoring (looking at parts of the frame instead of the whole) is the
best remaining idea and might reach 0.78–0.82. Getting to 0.85 would mean adding
a captioner and a language model — **which would destroy the thing that makes
your claim measurable.**

That is a real trade-off, and it belongs in your answer rather than in your
apology:

> "The architecture is capped by the same property that makes its central claim
> testable."

---

## 8. Your Phase 2 presentation

| Phase | Needs | Status |
|---|---|---|
| **Phase 2** (now) | Show the summer's work | **Comfortably covered** |
| Phase 3 | Full implementation | Six items queued |
| Phase 4 | Publication | Needs Phase 3 |

Your deck is **21 slides** with speaker notes:
`docs/06_presentations/DA-ZVAD_Phase2_Review.pptx`

**The two that matter most:**
- **Slide 11** — the dilution explanation. Put the two prompts on screen and let
  people see the shared words.
- **Slide 15** — the within-view control. This shows a complete cycle: odd
  result → explanation → test that could have killed it → it held.

Rehearse both **out loud**. They land when spoken and die when read.

---

## 9. Questions they will ask

**"0.734 isn't very good. LAVAD gets 0.85."**
> Correct. LAVAD runs a captioner, a language model and a refiner on every
> frame. We run one frozen encoder and a sentence, in about 7 GB. The fair
> comparison is the benchmark's own trained baseline at 0.728 — we match it
> using none of its training data.

**"Did you tune on the test set?"**
> Partly, and we control for it. No validation split is defined for these
> benchmarks, so we split the clips in half, chose settings on one half, and
> report the half we never looked at — averaged over five different splits, with
> the spread reported.

**"Isn't the normalisation just flattering the number?"**
> It uses no labels — it puts 13 camera views on a common scale before
> comparing. It is the benchmark's own protocol, and we report the
> un-normalised figure in the same table.

**"The correct description barely beats no description."**
> +0.027, positive at every window and growing with it — so the direction is
> consistent. But it is inside the ±0.036 split spread, so I report the
> direction, not the size. The claim rests on the *wrong* description costing
> 0.105.

**"Your finding doesn't replicate on Avenue."** ← *the hard one*
> It bounds it rather than refuting it. Detection transfers exactly — 0.706
> against 0.707. What doesn't transfer is the context effect. And we tested why:
> confining ShanghaiTech to a single camera view reproduces Avenue's flat
> result. The sentence mainly tells the system which environment it is in, and
> Avenue has only one. **I can't explain away that a placeholder beats an
> accurate description there** — but the mechanism is now measured, not guessed.

**"Why did your numbers change from an earlier version?"**
> An analysis script scored slightly differently from the real pipeline. The two
> ranked frames identically, but the benchmark's per-clip rescaling is affected
> by the difference. We caught it by requiring the analysis tool to reproduce a
> pipeline result, corrected every affected figure, and the metric's sensitivity
> is now a stated limitation.

**"How do I know you ran this?"**
> Every run writes a manifest — code commit, GPU, library versions, config,
> frame counts. Five of them, committed with the results.
> *(Have one open on screen.)*

---

## 10. Glossary

**AUROC** — Show one anomalous and one normal frame; how often is the anomalous
one ranked higher? 0.5 = guessing, 1.0 = perfect.

**Zero-shot / training-free** — Never shown examples from this task or place.

**Frozen** — No part of the model changes. No training, no updates.

**CLIP** — A pre-trained model scoring how well a picture matches a sentence.

**Prompt ensemble** — Several phrasings of the same idea, used together.

**Covariate shift** — Same things, different appearance (fog versus sun).

**Concept shift** — Same appearance, different correct answer. **Our problem.**

**Ablation** — Turning parts off one at a time to see what each contributes.

**Held-out split** — Keeping data hidden while choosing settings, then reporting
on the hidden part.

**Manifest** — The record each run writes about itself.

---

## 11. If you remember only five things

1. **The problem.** Anomaly detectors are tied to where they learned. We adapt
   them by writing a sentence instead of retraining.

2. **The gap.** The domain-adaptation literature explicitly focuses on "things
   look different". Anomaly detection's real problem is "the same thing means
   something different".

3. **The result.** 0.734 on ShanghaiTech with no training — matching the
   benchmark's own trained baseline. A *wrong* sentence costs 10 points, which
   proves the text is doing real work.

4. **The surprise.** *Where* you insert the description matters more than what
   it says. Inserting it wrongly reversed the entire result.

5. **The boundary.** It doesn't work on a single-scene benchmark, and we
   measured why: the sentence mainly tells the system *which* scene it is
   looking at. The claim is scoped, and the scoping is evidence, not excuse.
