# What We Did, In Plain Language

*Written 14 August 2026, after the first full GPU run. Read this before your
Phase 2 presentation. No prior technical reading needed.*

---

## 1. The whole thing in one paragraph

We are building a system that watches CCTV video and flags unusual events. The
normal way to build one is to show it thousands of hours of ordinary footage
from *that specific camera* until it learns what "ordinary" looks like there.
Our system skips that entirely. Instead, you **write a sentence in English**
describing the place — "a university campus walkway with pedestrians" — and the
system uses that sentence to decide what counts as unusual. Nothing is trained.
Today we tested it properly for the first time, on a standard benchmark, and we
found out both how well it works and, more interestingly, *why* it works the way
it does.

---

## 2. The problem we are attacking

### An anomaly detector is tied to the place it learned

Suppose a company sells an anomaly-detection system to a shopping mall. It
learns, over weeks of footage, that people walking slowly is normal, that the
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

**That gap is your research contribution.** You are not claiming to have
invented a better detector. You are pointing out that everyone has been
solving the adjacent problem, and showing what it looks like to attack the
real one.

---

## 3. Our idea

If the difference between two places is a difference in *rules*, and rules are
the kind of thing you can state in words — then let the operator state them.

So: **freeze everything, and adapt by changing a sentence.**

No retraining. No new footage. Deploy to a new site by editing one line of
text.

### Why "freeze everything" is a clever move, not a lazy one

This is worth understanding properly, because it is the strongest thing in your
whole design and a panel member may well probe it.

If we let the system learn even a little from the new place, and performance
improves, we cannot tell *why*. Did the sentence help? Or did the learning help?
The two are tangled.

Because we freeze every single parameter, only one thing in the entire system
can vary: **the sentence**. So if performance changes, the sentence caused it.
There is no other candidate.

This is what scientists call an **identifiable** claim — you can point at the
cause. Most adaptation methods cannot do this, because they change weights and
text together.

> **Say this in your presentation.** It is a genuine methodological strength and
> it shows you understand why the design is the way it is.

---

## 4. What the system actually does

Four parts. Think of them as four workers on an assembly line.

**M1 — The looker (CLIP)**
A pre-trained model called CLIP can compare a picture to a sentence and say how
well they match. We hand it two sets of sentences — describing normal, and
describing abnormal — and for each video frame it reports which side matches
better. That number is the anomaly score. **No training happens.** The language
itself is doing the classifying.

**M2 — The smoother**
Single frames are noisy; a person momentarily blocking the camera should not
count as an event. So we average each frame's score with its neighbours. Real
events last several seconds; noise does not.

**M3 — The scene description** ← *this is the research*
This is where your sentence about the deployment site gets mixed into M1's
sentences, so that "normal" means normal *here*.

**M4 — The explainer (LLaVA)**
For each flagged event, a second model writes a sentence saying what it thinks
is wrong. **We have not run this yet.**

---

## 5. What happened today, as a story

This section matters more than the numbers. It is what you should actually
present.

### Step 1: We got it running (and it failed completely)

We connected to the college's A40 GPU, downloaded the standard benchmark
(**ShanghaiTech** — campus CCTV, 107 video clips, 40,791 frames, with a human
label on every single frame saying anomalous or not), and ran the full
experiment. Fifty minutes.

**Result: 0.49.** Explained below, but for now: **0.50 means random guessing.**
The system was performing exactly as well as a coin.

Worse, the central experiment came out *backwards*. We had predicted that giving
the system the **correct** description of the scene would help and a **wrong**
description would hurt. We got the opposite. The correct description was the
worst condition and the deliberately wrong one was among the best.

That is the kind of result that makes you think the whole idea is broken.

### Step 2: We investigated instead of guessing

Three separate problems, found one at a time.

**Problem 1 — we were measuring it wrong.**

ShanghaiTech has 13 different cameras. Each shows a different part of campus.
CLIP happens to give slightly different baseline scores under each camera — one
camera might sit around 0.3, another around 0.6, just because of lighting and
angle.

We were dumping every frame from all 13 cameras into one big list and ranking
them together. That is like ranking students from different schools by raw
marks when the schools grade differently — a 70 from a strict school and a 70
from a lenient one are not the same thing, and comparing them destroys the
ranking.

The standard fix, used by every published paper on this benchmark, is to
normalise each clip's scores to a common range before combining them. We were
not doing it.

**Applying it took the score from 0.49 to 0.70, with no change to the system at
all.** Same scores, correct arithmetic.

> This is not cheating, and you should be ready to say so. It uses no labels —
> it is just putting things on the same scale before comparing them. It is the
> protocol from the paper that created the benchmark. And we report **both**
> numbers in our paper, so nobody has to take our word for it.

**Problem 2 — the scene description was cancelling itself out.**

This is the interesting one, and it became a genuine finding.

We were adding your scene sentence to *both* the "normal" sentences and the
"abnormal" sentences. Like this:

- normal: *"a campus walkway with pedestrians, everything is normal"*
- abnormal: *"a campus walkway with pedestrians, but something is wrong"*

Notice both now start with the same words. The system averages each group into a
single summary. When both summaries contain the same phrase, they become **more
similar to each other** — and the whole method depends on them being *different*.
We were blurring the distinction we needed.

And now the backwards result makes perfect sense: **the more accurate your
description, the worse the damage.** An accurate description matches every frame
strongly, so it dominates both sides and wipes out the contrast. A description
of a factory, fed to campus footage, matches nothing — so it does no harm.

The fix: put the description **only on the normal side**. That fits the idea
anyway — the scene tells you what normal looks like here; an anomaly is
whatever departs from it.

**After the fix, the experiment behaved exactly as predicted.** More on this
below, because it is your headline result.

**Problem 3 — a hypothesis of mine that turned out wrong.**

I thought the abnormal sentences were badly chosen. They talked about *"a fight,
robbery or accident"* — crime-scene language. But ShanghaiTech's anomalies are
mostly cyclists and skateboarders on footpaths. So I predicted that rewriting
them to mention bicycles and vehicles would help a lot.

**It made things much worse** (0.69 → 0.49). My prediction was wrong.

The reason, we think, is the same dilution problem in a new place: my new
sentences all contained "walkway" and "pedestrians" on *both* sides, so the two
groups shared vocabulary again.

> **Keep this in your presentation.** A prediction that failed, with an
> explanation of why, is more convincing evidence that you did real work than a
> table of good numbers.

### Step 3: We made the system faster, then tested one more idea

Each experiment took 50 minutes, because we re-analysed all 40,000 frames every
time we changed a word. We restructured it: **look at all the frames once, save
the results, then test ideas against the saved version.** One 21-minute pass,
after which each new idea takes seconds instead of an hour.

That let us test several ideas quickly. Most failed. One worked:

**Motion.** ShanghaiTech's anomalies are mostly about *speed*. A photo of a
cyclist is just a photo of a person on a path — nothing looks wrong. What makes
it an anomaly is that the person is moving at bicycle speed. A single still
frame cannot show that.

So we measured how much each frame differs from the one before it. Fast-moving
things change a lot; walking changes little.

Combining "what is in the frame" with "how fast it is changing" gave our best
result.

---

## 6. The results, and what the numbers mean

### First: what does 0.70 actually mean?

The measure is called **AUROC**. Here is the only definition you need:

> **Pick one anomalous frame and one normal frame at random. AUROC is how often
> the system gives the anomalous one a higher suspicion score.**

- **0.50** = coin flip. Useless.
- **0.70** = correct 70% of the time.
- **1.00** = perfect, always.

So our 0.706 means: shown an anomalous frame and a normal frame, our system
picks correctly about **7 times out of 10**.

### Result A — the detector works

| What is being measured | Score |
|---|---|
| Language only (CLIP + descriptions) | 0.673 |
| Motion only (no language at all) | 0.684 |
| **Both together** | **0.706** |

Two things to notice.

**The pieces combine well.** Neither alone reaches 0.706. They notice different
things — language notices *a bicycle is present*, motion notices *something is
moving oddly* — and a ShanghaiTech anomaly needs both. A parked bicycle is not
an anomaly. A person walking briskly is not an anomaly. A bicycle *travelling at
cycling speed on a footpath* is.

**Motion alone almost matches language alone.** That is a slightly awkward
finding and we report it honestly: on this benchmark, a large part of what the
fancy language model contributes could be obtained from a simple measure of
change. It is a caution about over-claiming, and stating it yourself is far
better than a panel member noticing it.

### Result B — the central experiment (your headline)

This is the one that matters. We ran the system four times, changing **only the
sentence**, with every model frozen:

| Sentence given | Score |
|---|---|
| No sentence at all | 0.685 |
| A vague one ("a generic scene") | 0.677 |
| **The correct one** (campus walkway) | **0.692** |
| **A deliberately wrong one** (factory inspection) | **0.592** |

**Read the last row.** Giving the system a description of the wrong kind of
place costs it **10 points**. That is a big drop.

And here is why that is proof rather than coincidence: **nothing else could have
caused it.** Same model, same weights, same video, same everything. One sentence
changed. So the sentence is doing real work.

**Be precise about the claim, because this is where a sharp examiner will push.**
The correct sentence beat no sentence by only 0.007 — that is nothing, and you
should not claim it as an improvement. The *evidence* is the wrong sentence
failing badly, not the right sentence succeeding.

Put in one line:

> **"We proved the description matters by breaking it. Corrupting it costs 10
> points. That could only have come from the text, because nothing else in the
> system was allowed to change."**

### Result C — the honest failures

We tested four ideas that did not work, and all four are written into the paper:

1. Chopping frames into quadrants to spot small objects — no help
2. Judging normality from the clip's own average appearance — 0.585, poor, and
   it *hurt* when combined with language
3. Rewriting the prompts to name bicycles and cars — much worse alone
4. A different way of combining prompt sentences — no difference

> **This is a strength, not an admission.** Anyone can report the runs that
> worked. Reporting the ones that did not is what separates a study from a demo,
> and a panel that suspects AI-generated work will find this far more convincing
> than a clean table.

---

## 7. Are these results promising? An honest answer

### As a detector: modest

The best training-free published method (LAVAD, CVPR 2024) reaches about 0.85 on
this benchmark. We are at 0.71.

**But the comparison is not like-for-like.** LAVAD runs a captioning model over
every frame, feeds the captions to a large language model, then refines with a
third model. Three heavy models per frame. Ours is one frozen encoder and a
sentence, running in about 7 GB of memory.

You are not competing on accuracy, and your paper says so explicitly.

### As research: yes, genuinely promising

Here is the honest case, and it is a real one.

**You have the shape of a proper scientific study:**

- A claim that could have been proven wrong
- A test designed to prove it wrong (the deliberately wrong description)
- The test was run, and the claim survived
- Findings you did not expect, with explanations
- Failures reported alongside successes
- Every number traceable to the exact code and hardware that produced it

**And you found something genuinely new.** The finding that *where* you inject
the description matters more than *what it says* — to the point of reversing the
result — is not in the literature. Papers treat verbalised context as a matter
of wording. We showed that a study reporting "scene descriptions do not help"
might simply have injected them in the wrong place.

That is a small but real contribution, and it is publishable.

### What is missing, and you should say so

**Your project is about adapting between domains, and so far you have tested one
domain.** Everything today came from ShanghaiTech. The description experiment is
good evidence that context matters *within* a domain, but the actual claim —
deploy somewhere new, change only the sentence, keep working — needs a **second
dataset**.

That is the next experiment, and it is roughly half a session now that
everything is built.

> **Say this yourself in the presentation before anyone asks.** Naming the gap
> in your own work reads as confidence. Being caught not having noticed reads as
> the opposite.

---

## 8. Your Phase 2 presentation

### Where you are in the four phases

| Phase | What it needs | Status |
|---|---|---|
| **Phase 2** (now) | Show what you did over the break | **Comfortably covered** |
| Phase 3 | Full implementation | Needs 2nd dataset + explanations |
| Phase 4 | Paper publication | Needs Phase 3 done |

For Phase 2 you have considerably more than "here is my plan" — you have a
working system, a benchmark result, a controlled experiment, and a finding.

### A suggested structure

**Slide 1 — The problem.**
Anomaly detectors are tied to the place they learned. Every new site means new
footage and retraining. *Use the mall-versus-factory example — everyone gets it
instantly.*

**Slide 2 — Why the existing literature does not solve it.**
Covariate shift versus concept shift. The surveys focus on the first and say so.
Anomaly detection is built on the second. *This is your strongest slide — it is
the one that shows you read the six surveys and understood them.*

**Slide 3 — Our approach.**
Freeze everything, adapt by writing a sentence. Show the architecture diagram.
Emphasise: because nothing can change except the text, any effect is provably
caused by the text.

**Slide 4 — What we ran.**
ShanghaiTech, 107 clips, 40,791 frames, on the college A40. Show a manifest —
it proves the run is real and repeatable.

**Slide 5 — The first result was a failure.**
0.49, chance. And the key experiment came out backwards. *Do not hide this.
Presenting a failure you then explained is the most credible thing you can do.*

**Slide 6 — What was wrong, and the fix.**
The measurement error, and the dilution problem. Explain dilution with the two
sentences on screen — it is visual and people understand it immediately.

**Slide 7 — Results after the fix.**
The context table. Point at the 0.592 and say: *"that is a wrong description
costing us 10 points, and nothing else in the system was allowed to change."*

**Slide 8 — Component ablation.**
Language 0.673, motion 0.684, together 0.706. Note the honest observation about
motion.

**Slide 9 — What did not work.**
Four failed hypotheses. Shows the work was exploratory and real.

**Slide 10 — Next.**
Second dataset (the actual transfer test), explanation module, patch-level
scoring. Be clear that the transfer claim is not yet tested.

---

## 9. Questions they will ask

**"0.71 is not very good. LAVAD gets 0.85."**
> Correct, and we say so in the paper. LAVAD runs three large models per frame —
> a captioner, a language model, and a refiner. We run one frozen encoder and a
> sentence, in about 7 GB. We are not competing on accuracy; we are testing
> whether language alone can carry adaptation, which needs the system to stay
> frozen.

**"Did you tune this on the test set?"**
> Partly, and we control for it. There is no validation split defined for this
> benchmark, so we split the clips into two halves, chose configurations using
> one half, and report the number from the half we never looked at — averaged
> over five different splits, with the spread reported. Where configurations sit
> inside that spread we report them as tied rather than picking a winner.

**"Isn't the normalisation just making the number look better?"**
> It uses no labels — it only puts the 13 cameras on a common scale before
> comparing them. It is the protocol from the paper that introduced the
> benchmark, and every published result on it does the same. We report the
> un-normalised number in the same table so it can be checked.

**"The correct description barely beats no description. So does it help?"**
> That is the right question, and our claim is narrower than you might expect.
> The correct description gains 0.007, which we explicitly do not claim as an
> improvement. The evidence that the text matters is that a *wrong* description
> costs 0.100. The description constrains the decision rather than adding
> information — it degrades sharply when misdirected.

**"Motion alone gets 0.684 without any language. So what is the language for?"**
> A fair challenge, and we raise it ourselves in the paper. On this benchmark
> the anomalies are largely kinematic, so a change signal captures much of them.
> The two are complementary — together they reach 0.706, above either alone —
> but we are careful not to present a frame-level vision-language score on this
> benchmark as evidence of semantic understanding.

**"Have you shown it adapts across domains?"**
> Not yet, and that is the next experiment. What we have shown is that within a
> domain, the description is load-bearing. Cross-domain transfer needs a second
> dataset and is our immediate next step.

**"How do I know you actually ran this?"**
> Every run writes a manifest recording the code commit, the GPU, the library
> versions, the configuration, and the frame counts. They are committed to the
> repository with the results. *(Have one open. This question is really about
> credibility, and a manifest answers it in five seconds.)*

---

## 10. Glossary

**AUROC** — Show the system one anomalous and one normal frame; how often does
it rank the anomalous one higher? 0.5 = guessing, 1.0 = perfect.

**Zero-shot / training-free** — The system was never shown examples from this
task or this place. It works straight out of the box.

**Frozen** — No part of the model changes. No training, no learning, no updates.

**CLIP** — A pre-trained model that scores how well a picture matches a
sentence. Ours is used entirely as-is.

**Prompt** — A sentence given to the model. A **prompt ensemble** is several
phrasings of the same idea, used together.

**Covariate shift** — Same things, different appearance (fog versus sun).

**Concept shift** — Same appearance, different correct answer (a bicycle on a
road versus on a footpath). **This is our problem.**

**Ablation** — Turning parts off one at a time to see what each contributes.

**Held-out split** — Keeping some data hidden while choosing settings, then
reporting the score on the hidden part, so you cannot fool yourself.

**Micro / macro averaging** — Micro pools all frames together; macro scores each
clip and averages. When they disagree, something is off with how clips compare.

**Manifest** — The record each run writes about itself: which code, which
machine, which settings.

---

## 11. If you remember only five things

1. **The problem:** anomaly detectors are tied to where they learned. We adapt
   them by writing a sentence instead of retraining.

2. **The research gap:** the domain-adaptation literature explicitly focuses on
   "things look different". Anomaly detection's real problem is "the same thing
   means something different". Nobody has attacked that directly.

3. **The headline result:** 0.706 on the standard benchmark with no training at
   all — and giving it a *wrong* scene description costs 10 points, which proves
   the sentence is doing real work, because nothing else could change.

4. **The surprise:** *where* you insert the description matters more than what it
   says. Putting it in the wrong place reversed the entire result. That is new,
   and it is our contribution.

5. **What is missing:** we have tested one domain. The cross-domain claim needs a
   second dataset. Say this before anyone asks.
