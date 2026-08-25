# DA-ZVAD — Project Handbook

*A document to learn from. Written to be read straight through.
Last updated 19 August 2026, after five GPU runs across two benchmarks.*

---

## How to read this

This is written to **teach**, not to look things up in. Each idea is built before
it's used, so reading in order costs less effort than jumping around.

| Part | What it gives you |
|---|---|
| 1 | The idea, in ordinary words |
| 2 | The machinery — how a computer compares a picture to a sentence |
| 3 | The research question, and why it's a real gap |
| 4 | What we actually built |
| 5 | What happened when we ran it — the story, with the reasoning shown |
| 6 | Every result, explained |
| 7 | Is it any good? An honest answer |
| 8 | Answering questions |
| 9 | What's left, and where everything lives |
| 10 | Glossary and a self-quiz |

**If you only have twenty minutes:** Part 1, then §6.9, then the self-quiz.

The one companion document worth keeping is
`03_domain_adaptation_deep_dive.md` — the detailed study of the six survey papers
your supervisor assigned. Everything else has been folded in here.

---

---

# Part 1 — The idea

## 1.1 What the system does

It watches CCTV video and flags moments that look unusual — someone cycling
through a pedestrian area, a car where cars shouldn't be, people running.

That's not new. What's new is **how you move it to a new location.**

## 1.2 Why moving it is the hard part

Almost every anomaly detector works like this: show it many hours of *ordinary*
footage from one camera. It gradually builds a picture of what ordinary looks
like there. Anything that doesn't fit gets flagged.

This works, and it has a problem.

**The system's idea of "normal" belongs to that one camera.** Move it to a
different building, a different angle, a different kind of place, and it breaks.
Not slightly — completely.

Take a concrete case. A detector installed in a shopping mall learns that people
drift slowly, that shutters come down at 9pm, that the floor is empty at 3am.
Now install the same system in a warehouse. A forklift moving down an aisle is
completely routine there, and would have been an emergency in the mall.

**The footage is the same. The correct answer is the opposite.**

So the vendor collects fresh footage from the new site, waits, retrains, and
redeploys. For every customer. That is the cost we set out to remove.

## 1.3 Our idea, in one sentence

> **Don't retrain the system. Tell it, in English, where it is.**

You write one sentence — *"a university campus walkway with pedestrians"* — and
the system uses that sentence to work out what counts as normal there.

Nothing inside the system ever changes. No training, at any point, ever.

## 1.4 Why that could possibly work

This only makes sense because of a specific development of the last few years:
models that understand pictures *and* words together.

The one we use is called **CLIP**. It was shown roughly 400 million pictures from
the internet, each with its caption. It was never taught about anomalies,
campuses or bicycles specifically. It learned a general association between what
things look like and what people call them.

Because of that, you can hand CLIP a photograph and a sentence, and it will tell
you how well they match. That single ability is what the whole project is built
on. Part 2 explains how.

## 1.5 Why this counts as research, not just engineering

There's a whole research field about moving models between places. It's called
**domain adaptation**, and your supervisor gave you six survey papers about it.

Here's what makes this a thesis rather than a project:

> **That field almost entirely solves a different problem from ours.**

It handles cases where the same things *look* different — a stop sign in fog
versus sunshine. It largely sets aside cases where the same thing *means*
something different — a bicycle on a road versus on a footpath.

But that second case is the whole of anomaly detection. Part 3 develops this
properly, and it is the foundation of your argument.

---

---

# Part 2 — The machinery

*How the system works inside. If you're comfortable with embeddings, CLIP and
AUROC you can skim — but read §2.6, because every number in Part 6 is expressed
in it.*

## 2.1 What it means for a model to "know" something

A neural network is, in the end, a very large pile of numbers — millions of them.
Those numbers are called **parameters** or **weights**.

**Training** is adjusting those numbers so the model does something useful. It's
slow and needs a lot of data.

**Inference** is using the model afterwards, without changing anything.

> **Our system only ever does inference.** Not one parameter is ever adjusted.
> This is the most important fact about the design, and §4.4 explains why it
> isn't laziness.

## 2.2 Embeddings — turning things into lists of numbers

Feed a picture into a model and it produces a list of numbers. In our case, 768
of them. That list is called an **embedding**.

The list isn't arbitrary. It's arranged so that **similar things produce similar
lists**. Two photos of dogs give lists close together. A photo of a dog and a
photo of a bridge give lists far apart.

Picture each list as an arrow pointing somewhere. Similar things point in similar
directions.

### Measuring "similar direction"

To ask how closely two arrows point the same way, we use **cosine similarity**.
One number:

- **+1** — pointing exactly the same way
- **0** — at right angles, unrelated
- **−1** — pointing exactly opposite

That's the only geometry you need. Everything else follows from it.

## 2.3 CLIP — pictures and sentences in the same space

Here's the part that makes the project possible.

CLIP has two halves. One turns **pictures** into lists of 768 numbers. The other
turns **sentences** into lists of 768 numbers. They were trained together, on
those 400 million image–caption pairs, so that **a picture and its caption end up
pointing in similar directions.**

A photo of a dog, and the words *"a photograph of a dog"*, land close together.

So you can compare a picture to a sentence directly: take the picture's arrow,
take the sentence's arrow, measure the angle between them. **That's the whole
trick.**

## 2.4 Turning that into an anomaly score

**Step 1 — write two sets of sentences.**

*Normal:*
- "a normal day with people behaving ordinarily"
- "a calm and safe public area"
- "nothing dangerous or abnormal is happening"

*Abnormal:*
- "a dangerous or violent event"
- "a fight, robbery or accident in progress"
- "an abnormal and unsafe situation"

**Step 2 — turn each set into a single arrow.** Encode every sentence, average
them, then adjust the result back to unit length. That gives one arrow for
"normal" and one for "abnormal".

These averaged arrows are called **prototypes**. Remember the word — the
project's central finding is about them.

**Step 3 — score each frame.** Encode it, then measure how closely it points to
each prototype. Two numbers.

**Step 4 — convert to a probability** using **softmax**, a standard function that
turns scores into probabilities adding to 1. What comes out is *the probability
this frame is abnormal* — a number from 0 to 1. That's our anomaly score.

Notice what didn't happen: **nothing was trained.** The sentences did the
classifying.

## 2.5 Smoothing over time

Single frames are noisy. Someone walks in front of the camera, a light flickers,
and one frame scores oddly for no meaningful reason.

So we average each frame's score with its neighbours. With a window of 31, each
frame's final score is the average of it and the 15 either side.

Real events last a second or more and survive the averaging. One-frame noise
doesn't. This costs nothing — there's no parameter to learn, only a choice of
window size. Ours is 31, and §6.2 shows why.

## 2.6 AUROC — how we measure success

**Read this properly.** Every number in Part 6 is an AUROC.

### Why not just "accuracy"?

Anomalies are rare — maybe 5% of frames. A system that says "normal" to
everything is 95% accurate and completely useless. Accuracy can't tell the
difference.

### The measure we use

> Pick one anomalous frame and one normal frame at random. **AUROC is how often
> the system gives the anomalous one a higher suspicion score.**

That's it.

- **0.50** — right half the time. A coin flip. Useless.
- **0.734** — right about 73 times in 100. Ours.
- **1.00** — always right.

It's the standard measure because it doesn't depend on where you set your alarm
threshold. It measures whether the system **ranks** things correctly, which is
the underlying ability.

### Two ways of averaging, and why it matters

We have 107 separate clips. Two ways to get one number:

**Micro** — pour every frame from every clip into one pile and rank them together.

**Macro** — score each clip on its own, then average the 107 scores.

They usually agree. **When they disagree badly, something is wrong with how clips
compare to each other** — and in §5.3 you'll see that happen, and what it
revealed.

---

---

# Part 3 — The research question

## 3.1 Domain adaptation

A **domain** is a setting a model works in — a particular camera, building, or
kind of scene. **Domain adaptation** is the study of moving a model between them.

It's a mature field. Your supervisor gave you six surveys of it, which is a
signal: he wants you to know it properly, and a panel may test the vocabulary.

## 3.2 The four kinds of difference — the key idea

When two places differ, researchers distinguish *what* differs. There are four
possibilities, and getting them straight is essential.

| Name | What changes | Everyday example |
|---|---|---|
| **Covariate shift** | The inputs look different | Same street, foggy instead of sunny |
| Conditional shift | A category looks different | "Cars" look different in Japan and India |
| Label shift | The proportions change | A quiet site has fewer anomalies |
| **Concept shift** | **The rule itself changes** | **A bicycle: fine on a road, anomalous on a footpath** |

The first and last matter here.

**Covariate shift** — the picture changed, the answer didn't. A stop sign in fog
is still a stop sign.

**Concept shift** — the picture didn't change, the answer did. Same bicycle, same
appearance, opposite label, because the *place* is different.

## 3.3 The gap — and how to state it precisely

This is the argument your whole thesis rests on. Learn it in this order.

**Step one: the field says which one it handles.**

Liu et al. (2022), one of your six surveys, states it outright:

> *"[Concept shift] is, however, usually not a common problem in popular object
> classification or semantic segmentation tasks. As such, this review mainly
> focuses on the covariate shift alignment."*

Singhal et al. (2023) go further: they list a *stable* rule — the answer not
changing between places — as the **first condition** under which domain
adaptation is theoretically justified.

So the field's theory **assumes concept shift doesn't happen.**

**Step two: for their problems, that's reasonable.** In object recognition a cat
is a cat everywhere. The rule genuinely is stable. They aren't being careless.

**Step three: but anomaly detection is built on the thing they excluded.**
"Normal" is a property of *where you are*, not of the object. That isn't an edge
case — it's the definition of the task.

**Step four: so the field's main tool doesn't apply.** The dominant family of
methods works by making two places *look* more alike. But in our case the two
places can look **identical** and still disagree about the answer. Making them
look more alike achieves nothing at all.

> **In one sentence:** *The domain-adaptation literature excludes the kind of
> difference that anomaly detection is made of.*

## 3.4 An honest complication — and why it helps you

In April 2026 a paper appeared (Wilkinghoff et al.) making the same core
observation independently: that what counts as normal depends on context, and
that judging it against one fixed standard is a mistake.

**Your first instinct may be that this weakens your contribution. It doesn't.**

Their paper is a *position paper*. It argues the problem exists and lists open
challenges. It runs no experiments.

So:
- The problem is **independently recognised** — you didn't invent it to have
  something to solve. That's a real objection you can now pre-empt.
- What they leave open is whether supplied context actually *does* anything.
  **That's exactly what we measure.**

Cite it and say so. Being caught not knowing about a paper on your own premise
would be far worse than citing it confidently.

---

---

# Part 4 — What we built

## 4.1 The system, in four parts

**M1 — the looker.** Frozen CLIP. Scores each frame against the two prototypes,
as in §2.4.

**M2 — the smoother.** Averages scores over time, as in §2.5.

**M3 — the scene description.** *This is the research.* Your sentence about the
site gets mixed into the prompts, so "normal" means normal *here*.

**M4 — the explainer.** A second frozen model (LLaVA) writes a sentence saying
what looks wrong in a flagged frame. **We have never run this on video.** It's
Phase 3 work.

## 4.2 How M3 actually works

This matters, because the project's most interesting finding is about it.

Take the normal sentences from §2.4. Now add two more, built from your
description:

- *"a university campus walkway with pedestrians, everything is normal"*
- *"a university campus walkway with pedestrians, a usual and safe moment"*

Those go into the normal set. Then the set is averaged into a prototype as
before.

So the scene description changes **where the normal prototype points**. That's
the entire adaptation mechanism. One sentence, one arrow moved.

## 4.3 The obvious version, which turned out to be wrong

The natural thing — and what we did first — was to add the description to
**both** sets:

- normal: *"a campus walkway with pedestrians, everything is normal"*
- abnormal: *"a campus walkway with pedestrians, but something is wrong"*

That looks symmetric and sensible. §5.4 explains why it fails badly, and it's the
finding you'll be asked about most.

## 4.4 Why everything is frozen — the identifiability argument

**Take time with this. It's the strongest thing in your design.**

Suppose we let the system learn a little from each new place. We change the
sentence, we let it learn, performance improves.

**Now: what caused the improvement?**

You can't say. The sentence changed *and* the model changed. The two effects are
tangled and no amount of analysis separates them.

Now suppose nothing in the model can change. Ever. Then when you run the system
twice, changing only the sentence, and the results differ —

> **the sentence caused it. There is no other candidate.**

That property is called **identifiability**: you can point at the cause.

**And here's what makes it a contribution rather than a constraint:** none of the
competing systems could run our experiment.

- **VERA** learns its text from source data — text and training are tangled by design
- **LAVAD** buries its notion of anomaly inside a language model's prior
- **OVVAD** trains an adapter alongside

In every one, something other than the text is free to move. **Ours is the only
design where text is the sole variable — which is the precondition for measuring
whether text does anything at all.**

> The architecture exists to make the experiment possible.

## 4.5 The experiment that could prove us wrong

Most papers show their method working. We designed a test that could show ours
*failing*.

Run the identical system four times. Change **only the sentence**.

| Condition | Sentence given | What it's for |
|---|---|---|
| **none** | M3 switched off | The baseline |
| **generic** | "a generic scene" | Controls for merely *having* a sentence |
| **matched** | The correct description | The intended way to use it |
| **mismatched** | A description of an **industrial factory**, fed to campus footage | **The falsifying control** |

The last one is the important one. If describing the wrong place costs nothing,
the system is ignoring its sentence and our claim is dead.

**What we predicted, before running anything:**
matched ≥ generic ≥ none, with mismatched clearly **worse**.

Writing the prediction down first matters. It means the experiment could
genuinely fail — and the first time, it did.

---

---

# Part 5 — What happened

*Told in order, with the reasoning shown. This is the part to present.*

## 5.1 What we ran it on

| | |
|---|---|
| **ShanghaiTech** | Campus CCTV, **12 camera views** in the test split, 107 clips, 40,791 frames |
| **CUHK Avenue** | Outdoor walkway, **1 camera**, 21 clips |
| Labels | A human marked every frame normal or anomalous |
| Hardware | The college's NVIDIA A40 |

That "12 cameras versus 1" difference turns out to matter enormously (§6.7).

## 5.2 The first run failed completely

Fifty minutes of GPU time. Result: **0.49**.

From §2.6, 0.50 is a coin flip. We had built something no better than guessing.

And the central experiment came out **backwards**: the *correct* description was
the worst condition, and the deliberately wrong one among the best.

That's the point where the idea looks broken. Three separate things were wrong,
and finding them is most of the work.

## 5.3 Problem one — we were measuring it wrong

### What went wrong

The ShanghaiTech test split spans 12 cameras, each looking at a different part of campus with
different lighting and angles. CLIP produces slightly different baseline scores
under each — not because anything is anomalous, simply because the scenes differ.

We were taking every frame from all 12 cameras and ranking them in one big list.

### Why that breaks things — a worked example

Suppose camera A's scores all sit between 0.30 and 0.40, and its anomalies score
0.38. Camera B's scores sit between 0.60 and 0.70, and its anomalies score 0.68.

**Within each camera the system is working perfectly.** In camera A, 0.38 is near
the top. In camera B, 0.68 is near the top.

Now pool them and rank. Every one of camera B's **normal** frames (0.60–0.67)
ranks *above* every one of camera A's **anomalies** (0.38).

The ranking is destroyed — not because the detector failed, but because we
compared numbers that were never on the same scale.

> **The analogy:** ranking students from different schools by raw marks when the
> schools grade differently. A 70 from a strict school and a 70 from a lenient
> one are not the same thing.

### The fix

Before combining, rescale each clip's scores to run from 0 to 1. Camera A's 0.38
and camera B's 0.68 both become roughly 0.8 — now comparable.

This is the standard published protocol for this benchmark. We simply weren't
doing it.

> **Applying it took the score from 0.49 to 0.71 — with no change to the system.**

### How we knew this was the problem

Remember micro and macro from §2.6. **Macro** scores each clip separately, so
it's immune to the scale problem by construction.

Macro said **0.67**. Micro on raw scores said **0.52**.

That gap is the fingerprint. The signal existed *inside* each clip; pooling was
destroying it. Only 31 of 105 clips were below chance individually.

### Be ready to defend this

Someone will ask whether the rescaling is just flattering the number. Three
parts to the answer:

1. **It uses no labels.** It only puts cameras on a common scale.
2. **It's the benchmark's own published protocol**, from the paper that created it.
3. **We report both numbers** in the paper, so anyone can check.

## 5.4 Problem two — the description was cancelling itself out

**This is the project's most interesting finding. Take your time.**

### What we were doing

Adding the scene description to *both* prompt sets:

- normal: *"a campus walkway with pedestrians, everything is normal"*
- abnormal: *"a campus walkway with pedestrians, but something is wrong"*

### Why it fails

Look at what those two sentences have in common. **Most of their words.**

Now recall §2.4: each set gets averaged into one prototype arrow. If both sets
contain the same long phrase, that phrase pulls **both arrows in the same
direction**.

The two prototypes drift toward each other.

And the system works by asking *which of these two arrows does the frame point
more towards?* If the arrows point almost the same way, **that question has no
useful answer.** Every frame scores about the same either way.

> **The analogy:** two reviewers who both begin every review with the same three
> paragraphs. Their reviews now look nearly identical whatever they're reviewing.
> The thing that distinguished them has been drowned out.

### Why an accurate description does the most damage

This is the counter-intuitive part, and it's what a panel will ask about.

An **accurate** description of the scene matches *every single frame* strongly —
that's what makes it accurate. So it contributes a large shared component to both
prototypes, and drags them together hard.

A description of a **factory**, fed to campus footage, matches nothing on screen.
It contributes little, so the original prompts survive underneath.

> **So the better your description, the worse your result.** That's why the
> experiment came out backwards.

### The fix

Add the description to the **normal set only**.

This fits the idea better anyway. The scene tells you what *normal* looks like
here; an anomaly is whatever departs from it. There was never a good reason to
describe the scene on the abnormal side.

After this fix, the experiment behaved exactly as predicted.

### And we measured it, rather than just arguing it

The prototypes are just averaged sentence embeddings, so the angle between them
can be computed directly — no video needed, seconds to run.

| Where the description goes | none | generic | matched | mismatched |
|---|---|---|---|---|
| **Both sets** | 35.7° | 26.3° | 26.3° | 25.0° |
| **Normal set only** | 35.7° | 34.5° | **41.9°** | 37.6° |

**First row:** adding *any* description to both sets squeezes the angle from
about 36° to 25°. The gap the whole decision depends on **nearly halves**. That's
the collapse, measured.

**Second row:** with the fix, no collapse. And the correct description actually
pushes the arrows *further apart* than having no description at all — 41.9°, up
from 35.7°. That's better than we expected.

**But one prediction of ours failed.** We expected the *accurate* description to
squeeze the arrows most. It doesn't — the mismatched one squeezes very slightly
more. And all three descriptions sit within about 1° of each other despite
producing noticeably different results.

> **So the explanation is right about the big effect and incomplete about the
> detail.** The angle explains why grounding *both* sets is harmful. It does
> *not* explain why an accurate description is the worst of the three. What's
> missing is the *direction* the arrows move — toward where the video actually
> sits, or away from it — and we haven't measured that.

Say precisely that if asked. A mechanism that explains most of an effect and is
honest about the rest is far stronger than one claimed to explain everything.

## 5.5 Problem three — a hypothesis of ours that was simply wrong

We thought the abnormal sentences were badly chosen. They mention *"a fight,
robbery or accident"* — crime-scene language. But ShanghaiTech's anomalies are
mostly cyclists and skateboarders on footpaths.

So we rewrote them to name bicycles, vehicles and running.

**It made things much worse** — 0.685 down to 0.486.

Why? The same dilution problem in a new place. The new sentences all contained
"walkway" and "pedestrians" on *both* sides, so the two sets shared vocabulary
again.

> **Keep this in your presentation.** A prediction that failed, with an
> explanation for the failure, is more convincing evidence of real work than any
> table of good numbers.

## 5.6 Day two — a second dataset, and a mistake of our own

We ran CUHK Avenue to check the finding held elsewhere. It didn't — §6.6.

While investigating, we found **an error in our own analysis code.**

### What the error was

We had two pieces of software: the real pipeline, and a faster script for trying
ideas out. The fast script scored things *slightly* differently — it used the raw
difference between the two similarities, where the real pipeline converted that
difference into a probability first (the softmax step from §2.4).

Mathematically, that ordering is identical. We checked: the two agreed on the
ranking of every single frame, correlation exactly **1.00**.

**But the benchmark's recipe rescales each clip to 0–1 first** (§5.3), and that
rescaling behaves differently depending on the shape of the numbers going in. So
the two produced different final scores despite ranking every frame identically.

About four hours of conclusions from that script were void. We corrected every
affected number.

### Why this is worth presenting

It produced a genuine methodological finding: **the metric this whole field uses
depends on the scale of your scores, not just their ordering.** That's now a
limitation in your paper, and it applies to everyone using the protocol.

And it produced a rule we now follow: *any new analysis tool must reproduce a
known result from the real pipeline before its output is trusted.* A perfect rank
correlation is not enough to prove two methods equivalent.

### What changed

| | before | after |
|---|---|---|
| Headline | 0.706 | **0.734** |
| Best smoothing window | 15 | **31** |
| Best combination | language **+ motion** | **language alone** |

**One finding we retracted.** We had believed a motion signal combined usefully
with the language signal. Measured correctly, motion adds nothing. If you
remember reading otherwise, this is the correction.

---

---

# Part 6 — The results, explained

## 6.1 Which parts of the system earn their place

We tested each component separately.

| Signal | Held-out | Full test set |
|---|---|---|
| **Language only** | **0.718 ± 0.036** | **0.707** |
| Language + motion | 0.711 ± 0.034 | 0.706 |
| Motion only, no language at all | 0.685 ± 0.015 | 0.686 |
| Language + clip's own average | 0.645 ± 0.034 | 0.640 |
| Clip's own average only | 0.585 ± 0.025 | 0.585 |

**Language alone is the best configuration.** Nothing added to it helps.

### What "motion only" means, and why it's uncomfortable

We measured how much each frame's embedding differs from the one before it. Fast
movement changes it a lot; walking barely at all. **No language involved.**

That alone reaches 0.686 — close to the full language pathway. It's a caution
against over-claiming, and you should volunteer it rather than wait to be asked.

### What "held-out" means and why it's there

Once we could test ideas quickly, a danger appeared: try enough things and one
looks good by luck.

So we split the 107 clips into two halves **before measuring anything**. We chose
settings using one half and report the number from the half we never looked at —
averaged over five different random splits. The **±** is how much the number
moves depending on which clips land in which half.

That ± matters. When two configurations differ by less than it, they're tied, and
we say so.

## 6.2 How much smoothing helps

| Pooling method | w=1 | w=5 | w=15 | **w=31** | w=61 |
|---|---|---|---|---|---|
| Micro, raw scores | 0.493 | 0.502 | 0.513 | 0.519 | — |
| Macro (per clip) | 0.614 | 0.628 | 0.648 | 0.670 | — |
| **Micro, rescaled per clip** | 0.667 | 0.685 | 0.702 | **0.707** | 0.683 |

Two things to read here.

**Smoothing helps, up to a point.** From none to a 31-frame window gains four
points. At 61 it gets *worse* — so 31 is a genuine best value, not an artefact of
"more smoothing is always better."

**The top row never rises above chance.** That's §5.3's problem in one line: with
raw pooling the same scores never beat 0.52; with correct pooling they reach
0.707.

## 6.3 The central experiment

All 107 clips, 31-frame window, correct pooling. Every model frozen; only the
sentence and where it goes are changed.

| Where the description goes | none | generic | matched | mismatched | **gap** |
|---|---|---|---|---|---|
| **Both sets** | 0.707 | 0.670 | 0.666 | 0.695 | **−0.029** |
| **Normal set only** | 0.707 | 0.691 | **0.734** | 0.628 | **+0.105** |

### How to read this table

**The "gap" is matched minus mismatched** — the difference between telling the
system the truth and telling it about a factory. That's the number the whole
experiment exists to produce.

**Top row:** with the description in both sets, the gap is *negative*. The truth
does worse than the lie. That's the broken version from §5.4.

**Bottom row:** with the fix, the gap is +0.105. Ten points.

**Now look at the "none" column: identical in both rows, 0.707.** It has to be —
if there's no description, it can't matter where you'd have put it. That's a
built-in check that nothing else changed between the two experiments. It's the
kind of detail that makes an experiment trustworthy, and worth pointing at.

### How to state the claim — precisely

Two halves, carrying very different weight.

**The weaker half.** The correct description beats no description by **+0.027**.
It's positive at every smoothing window and grows with the window, so the
*direction* is consistent. But 0.027 is smaller than the ±0.036 spread between
data splits. **So report the direction and don't claim the size.**

**The stronger half.** The wrong description costs **0.105**. Far outside any
noise — and since every model is frozen, nothing else could have caused it.

> **Say this:** *"We proved the description matters by breaking it. Getting it
> wrong costs ten points, and nothing else in the system was allowed to change."*

Understating protects you. Claim the +0.027 as a solid improvement and someone
who checks the spread will take the room off you.

## 6.4 The finding that isn't in the literature

Everything above adds up to something nobody has reported:

> **Where you put the description matters more than what it says.**

Same sentence, same models, same video. Move it from one prompt set to two, and
the effect *reverses* — from +0.105 to −0.029.

That has a practical consequence for the field. A researcher who tried scene
descriptions the obvious way, saw them fail, and published "scene descriptions
don't help" would have published something false. They'd have been measuring an
implementation detail.

## 6.5 Six things that did not work

| What we tried | What happened |
|---|---|
| Cutting frames into quarters to catch small objects | No improvement anywhere |
| Prompts naming bicycles and vehicles | Much worse — 0.486 |
| Using each clip's own average as the "normal" reference | 0.585, and it hurt the language signal |
| Adding a motion signal | Costs 0.001 — not complementary |
| Subtracting a local time-average before scoring | Better scorer, but *shrinks* the context gap |
| A different way of combining the prompts | 0.678 against 0.707 |

> **This is a strength, not an admission.** Your claim is that the *simplest*
> configuration is right. That's only believable next to the alternatives you
> tried. And a clean table of successes is exactly the artefact that's easy to
> fabricate — six diagnosed failures are not.

## 6.6 The second dataset — where it stopped working

We ran the identical system on CUHK Avenue. Nothing retuned. Only the sentence
changed, which is what the method claims is sufficient.

| Condition | ShanghaiTech | Avenue |
|---|---|---|
| No sentence | 0.707 | 0.706 |
| Correct sentence | **0.734** | 0.677 |
| Wrong sentence | 0.628 | 0.657 |
| **Gap** | **+0.105** | **+0.020** |

**The detector transfers perfectly.** 0.706 against 0.707 with no description —
as close as this measurement can resolve. The system works just as well on a
benchmark it was never adjusted for.

**The adaptation does not.** The gap collapses from ten points to two. And worse:
on Avenue the *correct* description scores **below** having no description at
all. The best condition there is the vague placeholder, "a generic scene".

### Concede this openly

A placeholder contains no information about the environment. So whatever benefit
it gives on Avenue **cannot be domain adaptation** — most likely it just makes
the prompt set slightly better as a set.

Don't try to explain that away. Say it, then give the mechanism below.

## 6.7 …and then we worked out why

ShanghaiTech's test split has **12 camera views**. Avenue has **one**.

**The idea:** a scene description only has a job to do when there are several
places to tell apart. On Avenue every clip shows the same view. There's nothing
for the sentence to disambiguate, and the basic prompts already cover the only
scene there is.

### The clever part — testing it without a third dataset

ShanghaiTech is really **12 single-camera datasets stacked together**. The clip
names even say which camera each came from (`01_0014` is camera 01).

So we ran the same experiment **inside each camera separately**. If the idea is
right, the effect should collapse — because now there's only one scene, just like
Avenue.

| How it's evaluated | Gap |
|---|---|
| All 12 cameras pooled | **+0.105** |
| Inside a single camera (average of 9) | **+0.033** |
| Avenue, which has one camera | +0.020 |

**The prediction held.** Confine it to one scene and the effect falls to roughly
Avenue's level.

**And note it could have failed.** A within-camera gap near +0.105 would have
destroyed the explanation. We ran a test that could have refuted us, and it
didn't.

> **What this means:** the sentence mainly tells the system *which* place it's
> looking at — not what counts as normal within that place.

That's narrower than we originally claimed. It's also **measured** rather than
argued, and it predicts where the method should be useful: sites with several
cameras covering different environments, not a single fixed installation.

**Caveats to state:** each camera's estimate rests on only 5–34 clips, three of
the nine are negative, and the variation between cameras (0.058) is larger than
the average effect (0.033). One camera scores 0.415 — below chance — under every
condition, and we have no explanation for that.

## 6.8 Everything in one place

| | |
|---|---|
| **Headline** | **0.734** AUROC on ShanghaiTech, no training data |
| Comparison | Liu et al. (2018) reach 0.728 **by training on it** |
| Context effect | Wrong description costs **0.105** |
| Mechanism | Prototype angle collapses 35.7° → 25° when misapplied |
| Boundary | Effect nearly vanishes on a single-camera benchmark |
| Why | The sentence identifies *which* scene — measured |
| Failed attempts | Six, all reported |

## 6.9 The five sentences to memorise

1. Anomaly detectors are stuck in the place they learned; we move them by
   writing a sentence.
2. The domain-adaptation field handles "things look different"; anomaly
   detection is "the same thing means something different".
3. 0.734 with no training at all — matching the benchmark's own trained
   baseline — and a *wrong* sentence costs ten points.
4. Where you put the sentence matters more than what it says; putting it wrongly
   reverses the result.
5. It works where there are several scenes to tell apart, and we measured that
   rather than assuming it.

---

---

# Part 7 — Is it any good?

## 7.1 As a detector: modest, and one comparison saves it

The best training-free method published (LAVAD, 2024) reaches about 0.85. You
reach 0.734. That's a real gap and you shouldn't minimise it.

**But that isn't the comparison that matters.**

> Liu et al. (2018) — the researchers who *created* ShanghaiTech — reach about
> **0.728** by training a model on ShanghaiTech's own training footage.
>
> **You reach 0.734 having never seen a frame of it.**

Matching the benchmark's original trained baseline with zero training data is a
genuine result.

There's also a deployment argument. At a brand-new site, the trained model's
accuracy isn't 0.728 — it's *undefined*, because it doesn't exist yet. Someone
has to spend weeks producing it. Yours runs on day one.

**Don't overstate that, though.** 0.734 is not good enough to run a building
unsupervised. The honest framing is **cold start**: something from day one while
footage is collected for a trained system, or narrowing hours of video down to
minutes worth reviewing.

## 7.2 As research: genuinely sound

- A claim that could have been proven wrong
- A test designed to prove it wrong — which partly did
- An explanation for that failure, and a control that confirmed the explanation
- Six failures reported next to the successes
- Two errors of our own, found and corrected
- Every number traceable to the exact code and machine that produced it

That's the shape of real experimental work.

## 7.3 Will the number get much better?

Probably not dramatically, and it's worth understanding why rather than hoping.

**Your decision rule is very simple.** One arrow compared against two other
arrows, producing one number. It cannot *reason*.

LAVAD writes a caption for each frame — *"a man riding a bicycle on a
sidewalk"* — and hands it to a language model, which can hold "bicycle" and
"sidewalk" as separate facts and judge the combination. No amount of tuning a
two-arrow comparison recovers that.

**Evidence you're near this design's ceiling:**
- The wording of the prompts barely matters
- A signal with no language at all gets within 0.02
- All six attempted improvements were absorbed

**The one idea left** is scoring parts of the frame rather than the whole thing —
the anomaly is often 1–2% of the picture. That might reach 0.78–0.82.

Getting to 0.85 would mean adding a captioner and a language model — **which
would destroy the thing that makes your claim measurable** (§4.4). You'd gain ten
points and lose the contribution.

> **Say it as a trade-off, not an apology:** *"The architecture is limited by the
> same property that makes its central claim testable."*

## 7.4 Is it publishable?

Honestly: not yet for a strong venue. You have a characterisation, a protocol,
two findings and a measured boundary — but no new architecture, and the central
effect holds on one of the two video benchmarks.

**For the thesis, it's comfortably enough.** For a paper, Phase 3 would want the
industrial sweep and ideally the region-level scoring.

---

---

# Part 8 — Answering questions

*For each: what to say, and the phrasing that loses the room.*

## 8.1 On novelty

**"Isn't this already done?"**

> Each ingredient exists somewhere. What doesn't exist is a system where the text
> is the *only* thing that can vary — and without that you can't test whether the
> text does anything. The architecture exists to make the experiment possible.

⚠️ Don't claim "text at inference time" as new. A 2025 paper (AnyAnomaly) does
that. **Name it yourself before they do.**

**"What exactly is your research gap?"**

> Two halves. The domain-adaptation literature explicitly excludes the kind of
> difference anomaly detection is made of. And the papers that do use language
> supply it, report that it works, and never test whether it's causally
> responsible.

**"Isn't 'nobody ran a control' a trivial contribution?"**

> A control is trivial when it confirms. Ours overturned the result. The
> contribution isn't the ablation — it's the failure mode it exposed: the obvious
> implementation actively harms, and the harm grows with how *accurate* your
> description is. That's a property of a class of methods, not of our code.

⚠️ Never phrase it as "we ran an ablation nobody ran." That phrasing *is* trivial.

## 8.2 On the accuracy

**"Existing methods do better. Why do this?"**

> They do better *when they have training footage from the site*. On a benchmark
> everyone has it; at a new site nobody does. And accuracy was never the
> question — we asked whether language alone can carry the adaptation, and
> whether that can be measured.

⚠️ Never say "we're not competing on accuracy" and stop. On its own it reads as an
excuse. Always pair it with what you *are* competing on.

**"Your industrial result was 88.5%. Why is video only 0.734?"**

> Different tasks — but the industrial breakdown predicted it. Performance there
> fell as the defect got smaller: textures 99.5%, whole objects 83%, small
> localised defects 72.7%. In surveillance the anomaly is 1–2% of the frame. One
> trend across two benchmarks.

**Follow-up: "then why didn't quartering the frame fix it?"**
> It should have and it didn't. Averaging over quarters beat taking the maximum,
> which is the opposite of what finding a small object looks like. Resolution is
> our leading explanation, not a proven one. Proper region-level scoring is the
> real test and we haven't run it.

## 8.3 On method

**"Did you tune on the test set?"**

> Partly, and we control for it. These benchmarks define no validation split, so
> we split the clips in half, chose settings on one half, and report the half we
> never looked at — averaged over five splits, with the spread reported. Where
> configurations sit inside that spread we call them tied.

**"Isn't the rescaling just flattering the number?"**

> It uses no labels — it only puts 12 cameras on a common scale before comparing
> them. It's the benchmark's own published protocol, and we report the
> un-rescaled figure in the same table.

**"The correct description barely beats no description."**

> +0.027, positive at every smoothing window and growing with it, so the
> direction is consistent. But it's inside the ±0.036 spread between splits, so I
> report the direction and not the size. The claim rests on the *wrong*
> description costing 0.105.

**"Motion alone gets 0.686 without any language. So what's the language for?"**

> Language reaches 0.707, and adding motion to it costs 0.001 — they're not
> complementary. That surprised us; we expected them to combine. It suggests the
> frozen encoder already picks up enough of the motion on its own.

**"Why did your numbers change between drafts?"**

> An analysis script scored slightly differently from the real pipeline. The two
> rank every frame identically, but the benchmark's per-clip rescaling is
> affected by the difference. We caught it by requiring the analysis tool to
> reproduce a pipeline result, corrected every affected number, and the metric's
> sensitivity is now a stated limitation.

## 8.4 The hardest question

**"Your central finding doesn't replicate on the second dataset. Doesn't that refute it?"**

> It bounds it rather than refuting it. The detector transfers almost exactly —
> 0.706 against 0.707. What doesn't transfer is the context effect: the gap falls
> from ten points to two, and the correct description scores below none.
>
> We tested why. ShanghaiTech's test split has twelve camera views; Avenue has one.
> Confining ShanghaiTech to a single view reproduces Avenue's flat result. So the
> sentence mainly identifies *which* environment is in view — and Avenue has only
> one.
>
> And I can't explain away that a placeholder beats an accurate description
> there. But the mechanism is now measured, not guessed.

⚠️ Don't say "it works on ShanghaiTech and needs more investigation on Avenue."
That's evasive and will be heard as evasive. State the near-vanishing effect and
the placeholder result **first**, then give the mechanism.

## 8.5 On credibility

**"How do I know you actually ran this?"**

> Every run writes a manifest — the exact code version, the machine, the GPU, the
> library versions, the full configuration, and the frame counts. Five of them,
> committed alongside the results.
>
> And the internal check is verifiable: two of our runs differ only in where the
> description is injected, and their "no description" columns agree to four
> decimal places.

*Have `results/runs/2026-08-14_162056_surv_normal/MANIFEST.txt` open on screen.*

**"Walk me through what went wrong and how you found it."** ← *your best question*

Take your time. The first run was chance and the key experiment came out
backwards. Three causes: pooling across cameras with incomparable scales; the
description entering both prompt sets and collapsing them together; and a
hypothesis of ours about better prompts that made things worse. Then day two:
Avenue didn't replicate, and an error in our own analysis code invalidated four
hours of work.

## 8.6 What you cannot answer yet — know these

1. **Why context helps on one benchmark and not the other** — the scene-count
   explanation is supported, but the per-camera estimates are noisy.
2. **Which direction the prototypes collapse.** We measured *how much* (§5.4),
   not whether they move toward or away from where the video sits — which is what
   would explain why an accurate description is the worst of the three.
3. **The industrial-versus-surveillance contrast is argued, not measured.** The
   sweep has never been run on the industrial benchmark, and both video datasets
   are outdoor pedestrian scenes.
4. **M4 has produced nothing.** The explanation module has never run on video.
5. **Does AnyAnomaly already run a wrong-text control?** Needs checking. Read it
   (arXiv 2503.04504) before the viva.
6. **The smoothing window was chosen on the test set** — no validation split
   exists for these benchmarks.

## 8.7 Three sentences to fall back on

If a question goes somewhere you didn't prepare, return to these.

> The domain-adaptation literature excludes concept shift by explicit choice, and
> anomaly detection is built on concept shift.
>
> The papers that use language show it works but never test whether it is
> causally responsible.
>
> We characterise the first and supply a protocol for the second — including a
> control that our own framework initially failed.

---

---

# Part 9 — What's left, and where things are

## 9.1 Phase 3

| Priority | Work | Cost | Why |
|---|---|---|---|
| 1 | A separate description per camera | ~1 hr | All 12 views currently share one sentence — and §6.7 says identifying the view *is* the mechanism |
| 2 | The context sweep on the industrial benchmark | ~1 hr | Your concept-shift argument needs that contrast measured, not just argued |
| 3 | Region-level scoring | 1–2 sessions | The best remaining shot at a higher number |
| 4 | Run M4 | ~1 session | A quarter of the framework; makes a strong demo |
| 5 | Swap descriptions between the two video datasets | ~1 hr | A fairer transfer test than the factory description |
| 6 | Which direction the prototypes move | minutes | Completes the mechanism from §5.4 |

## 9.2 Phase 4

Paper submission. Possibly reproducing a competing method if your guide wants
one — if so, request Llama-2 access early, since the approval wait is the long
pole.

## 9.3 Where everything lives

| Need | Location |
|---|---|
| This handbook | `docs/08_understanding/` (`.md` and `.docx`) |
| The six surveys, in depth | `docs/08_understanding/03_domain_adaptation_deep_dive.md` |
| The paper | `docs/09_paper/main.tex` |
| The slides (21, with speaker notes) | `docs/06_presentations/DA-ZVAD_Phase2_Review.pptx` |
| **Proof you ran it** | `results/runs/*/MANIFEST.txt` |
| Raw result tables | `results/runs/*/tables/*.csv` |

⚠️ **Read `results/runs/README.md` before showing anything from `analysis/`** —
about half those files come from the buggy analysis path and are marked
superseded.

**Reconnecting to the server:**
```
ssh m251250cs@192.168.41.119
tmux new -s dazvad
source ~/dazvad/venv/bin/activate
```

---

---

# Part 10 — Glossary and self-quiz

## 10.1 Glossary

**Anomaly detection** — finding rare, unexpected events.

**Parameters / weights** — the numbers inside a model that training adjusts.
Ours never change.

**Training / inference** — learning from data, versus using what's already
learned. We only do the second.

**Embedding** — the list of numbers a model produces for an input. Similar
inputs give similar lists.

**Cosine similarity** — how closely two such lists point the same way. +1
identical, 0 unrelated.

**CLIP** — the pre-trained model that puts pictures and sentences in the same
space so they can be compared.

**Prompt** — a sentence given to the model. A **prompt ensemble** is several
phrasings used together.

**Prototype** — the single averaged arrow representing a whole prompt ensemble.
The dilution finding is about these.

**Zero-shot / training-free** — the system has never seen an example from this
task or place.

**Frozen** — nothing inside the model changes, ever.

**Covariate shift** — the same things look different (fog versus sun).

**Concept shift** — the same appearance gets a different answer. **Our problem.**

**AUROC** — pick one anomalous and one normal frame; how often does the system
rank the anomalous one higher? 0.5 = guessing, 1.0 = perfect.

**Micro / macro** — pooling all frames together, versus scoring each clip and
averaging.

**Ablation** — turning parts off one at a time to see what each contributes.

**Held-out split** — hiding some data while choosing settings, then reporting on
the hidden part, so you can't fool yourself.

**Manifest** — the record each run writes about the conditions it ran under.

## 10.2 Self-quiz

Answer each **out loud**, without looking. If you can't, reread the section.

1. Why does an anomaly detector break when you move it to a new building? *(§1.2)*
2. How can a computer compare a photograph to a sentence? *(§2.3)*
3. What does an AUROC of 0.734 actually mean? *(§2.6)*
4. What's the difference between covariate shift and concept shift — with an
   example of each? *(§3.2)*
5. Why does freezing everything make the claim measurable? *(§4.4)*
6. What is the mismatched condition for, and what result would have refuted you?
   *(§4.5)*
7. Why did pooling raw scores across 12 cameras give 0.52? Use the two-camera
   example. *(§5.3)*
8. Why does an **accurate** description do the most damage when added to both
   prompt sets? *(§5.4)*
9. What does the +0.105 gap mean, and why do you describe +0.027 differently?
   *(§6.3)*
10. What happened on Avenue, and how did you test *why*? *(§6.6–6.7)*
11. Name three of the six things that didn't work. *(§6.5)*
12. Why won't this design reach 0.85? *(§7.3)*
13. What can you **not** yet answer? *(§8.6)*

If you can do all thirteen aloud, you know this project.
