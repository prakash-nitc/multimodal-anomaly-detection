# DA-ZVAD Handbook, November 2026 edition

*Plain language, built to be learned, remembered, taught and defended.
Written 6 November 2026, after the improvement round and the freeze. The older,
longer handbook (`00_HANDBOOK.md`) stays as the detailed reference for Phases 1–3.*

---

## How to use this

| If you have | Read |
|---|---|
| 5 minutes | Part 0 (one page) and the memory card at the end of it |
| 30 minutes | Parts 0, 1, 4 and 6 |
| 2 hours | Everything, then the self-quiz in Part 9 |
| The night before a viva or interview | Part 0, Part 6, and say Part 7 out loud |

Each part builds on the one before. Numbers marked *illustrative* are made up to
show how something works; every other number is a real, measured result.

---

# Part 0 — The whole project on one page

**What it does.** Watches CCTV video and gives every frame an anomaly score: how
likely it is that something unusual is happening, like a fight, a car on a footpath,
or someone collapsing.

**What makes it different.** It is **training-free**. Nothing is trained, ever. It uses
a frozen, ready-made model that understands pictures and words, and compares each
frame with two lists of sentences: one describing normal scenes and one describing
anomalies.

**The journey in four lines.**
1. We first gave the detector one sentence about each camera. It helped on campus video (0.751).
2. On XD-Violence it failed badly (0.624 → 0.417), because the sentence was written from
   frames that already showed the violence.
3. So we stopped relying on that sentence and changed *how the detector describes normal and abnormal*.
4. The final model: **PE-Core + calm phrases + a frozen phrase bank**, with no sentence for
   each video. XD went from 41.7 to 72.2.

**Final scores** (AUROC; AP for XD):

| ShanghaiTech | UCF-Crime | XD-Violence | NWPU Campus |
|---|---|---|---|
| 78.3 | 85.6 | 72.2 | 63.9 |

**Status.** These are exploratory, because the model was chosen on these four datasets.
It is now frozen and will be tested once on a dataset it has never seen, UBnormal.

### The memory card

> **Three ingredients: P-C-B.** **P**E-Core (the encoder), **C**alm normal phrases, **B**ank of concrete actions.
> Think of a **PCB**, a circuit board: three parts, no moving parts, nothing trained.
>
> **Three results:** XD 41.7 → 72.2; no worse than the old model anywhere; no captioning model.
>
> **Three limits:** chosen on these datasets (sealed test to come); an LLM wrote the bank knowing
> the datasets; one frame at a time, so motion and small objects are hard.
>
> **One sentence:** *"A frozen model compares each frame with calm descriptions of normal scenes and
> concrete descriptions of anomalies — no training, the same setup for every camera."*

---

# Part 1 — The problem, in everyday words

## 1.1 What is a video anomaly?

Anything unusual for that place: a fight, a theft, a fire, a car accident, a bicycle
racing through pedestrians, someone falling down. Anomalies are **rare**. A camera can
run for days with nothing happening, which is exactly why nobody can watch it all.

## 1.2 Why it is hard

**Normal depends on the place.**
- A car on a road is normal; a car on a campus footpath is an anomaly.
- Running on a sports track is normal; running through a train station may not be.

The picture can look almost the same while the right answer flips. This has a name
in the research your guides gave you: **concept shift**. The meaning changes, not the
appearance. Most domain-adaptation methods deal with the opposite case, where
appearance changes but meaning stays the same (a stop sign in fog or sunshine).

## 1.3 Why "training-free" matters

The usual approach is to collect hours of normal video from one camera and train a
model on it. Move the camera, and you start again: collect, label, retrain, for every
site. A training-free detector skips all of that. You plug it in, and it works from
day one. The price is accuracy: it will rarely beat a model trained on that exact camera.

---

# Part 2 — The tools, explained simply

## 2.1 Embeddings: turning things into lists of numbers

A computer can't compare a picture with a sentence directly. So both are turned into
a long list of numbers (about a thousand), called an **embedding** or **vector**. Things
that mean similar things get similar lists.

**Analogy:** every picture and every sentence gets a pin on a huge map. A photo of a
fight and the sentence "two people punching each other" land close together. The
sentence "an empty street" lands far away.

## 2.2 The encoder: the machine that makes the lists

The **encoder** is the model that turns pictures and sentences into vectors. It was trained
once, by its makers, on hundreds of millions of image-caption pairs. We never train it
further, which is why it is called **frozen**.

| | Encoder | Notes |
|---|---|---|
| Earlier phases | CLIP ViT-L | The classic picture-and-words model |
| Final model | **PE-Core** (Meta, 2025) | Newer, used by NOVA and Flashback too |

## 2.3 Similarity: how close two pins are

**Cosine similarity** measures how closely two vectors point in the same direction. It is high when a
frame and a sentence match, and low when they don't.

## 2.4 The anomaly score: one subtraction

For each frame:

```
score = (how close it is to the ABNORMAL sentences) − (how close it is to the NORMAL sentences)
```

- Each list of sentences is averaged into one vector first, called a **prototype**.
- The difference is squashed into the range 0 to 1 (a sigmoid, after multiplying by 100).
- Scores are smoothed over a few neighbouring frames, so a single odd frame doesn't cause an alarm.

*Illustrative example:* frame-to-normal similarity 0.21, frame-to-abnormal similarity 0.24.
Difference +0.03, times 100 = 3, sigmoid(3) ≈ 0.95, so the frame looks anomalous.

## 2.5 How we measure success

**AUROC** (used for ShanghaiTech, UCF-Crime and NWPU Campus):
> Pick one anomalous frame and one normal frame at random. AUROC is the chance that the
> anomalous one got the higher score.
- 0.5 = a coin flip. 1.0 = perfect. We write it as 0–100 too (85.6 means 0.856).
- Scores **can't be compared across datasets**. 64 on NWPU and 86 on UCF don't mean the model is
  "worse" on NWPU. Each dataset has its own difficulty.

**AP, average precision** (used for XD-Violence, because its official benchmark uses it):
> Of the frames the detector flags, how many are truly anomalous, averaged over every
> possible alarm threshold.

## 2.6 Confidence intervals: is a difference real or luck?

A score from 290 videos could be a bit different with another 290. To check, we
**resample**: we redo the measurement 1,000 times on random re-draws of the video list.
This is called a **bootstrap**. The middle 95% of those results is the **95% confidence interval (CI)**.

- If the CI for "new minus old" is **entirely above zero**, the improvement is **significant**, very
  unlikely to be luck.
- If it **includes zero**, we say **"within noise"**.

## 2.7 Calibration vs re-ranking

Two ways a change can raise a score:
- **Calibration:** it moves whole videos up or down, so different cameras end up on the same scale.
  *"Scale, not sort."*
- **Re-ranking:** it puts the frames *inside* a video in a better order, so the anomaly stands out from
  its own surroundings.

Both count. Re-ranking is the stronger kind of evidence, because it shows the detector is
seeing the event itself. An audit we ran found most of the old sentence's gains were
calibration. The calm phrases' gains on ShanghaiTech and NWPU were mostly re-ranking.

---

# Part 3 — The journey, as a story

## 3.1 The first idea: tell it where it is (Phases 1–2)

*"Don't retrain the detector. Tell it, in English, where it is."* A sentence such as
"a university campus walkway with pedestrians" was added to the normal side.
- ShanghaiTech: 0.734 with one shared sentence, and **0.751** with a sentence for each camera written by LLaVA, a captioning model.
- A *wrong* sentence cost about 10 points. So the sentence really carried information about the scene.

## 3.2 Crime video (Phase 3)

- UCF-Crime: **0.824** with a LLaVA sentence for each video.
- **The audit:** about 82–87% of the gain was calibration. The sentence mainly put videos on one scale.

## 3.3 The failure: XD-Violence

- No sentence: **0.624** AP. With the LLaVA sentence: **0.417**.
- **Why:** LLaVA described the first 3 frames of each video. In **127 of the 500 violent videos**, the violence
  had already begun in those frames. So the sentence called the violence normal, and the detector ignored it.
- **The lesson:** a fix must not depend on what the first frames happen to show.

## 3.4 Reading the papers (November)

We read 15 recent papers. Three lessons:
1. **NOVA:** the normal side must *never* contain look-alike actions. Our earlier "near-miss" list did
   the opposite, adding "hugging" as normal next to fighting. **That was our mistake, and we disclosed it.**
2. **ASK-Hint:** concrete actions ("punching") work better than abstract labels ("fighting").
3. The top methods use a newer encoder (PE-Core), rich phrase lists, or heavy models.

## 3.5 The improvement round (5–6 November)

We tested one idea at a time, always with rules written first (Part 4.3). What we kept:
- **PE-Core** as the encoder, which alone wasn't enough;
- **C, the calm phrases**, which passed, and passed again with a second wording;
- **B, the phrase bank**, which passed.

What we dropped is listed in Part 5.4. Then we **froze** the model for a single test on unseen data.

---

# Part 4 — The final model, step by step

## 4.1 The picture

![The simple version: three inputs, one frozen encoder, four numbered steps. Learn to redraw this one.](../09_paper/dazvad_architecture_v3_simple.png)

The detailed version, with notation, for the thesis and slides:
[`docs/09_paper/dazvad_architecture_v3.png`](../09_paper/dazvad_architecture_v3.png).
The earlier design (with the sentence and LLaVA) stays at `docs/09_paper/dazvad_architecture_detailed.png`
for reference.

In plain text:

```
 Video frame (every 16th) ──► PE-Core image encoder ──► frame vector ─┐
                                                                      ├─► score = sim(abnormal) − sim(normal)
 Normal side:   3 general + 8 calm phrases ──► PE-Core text ─► avg ───┤        ──► sigmoid(100 ×)
 Abnormal side: 48 concrete actions        ──► PE-Core text ─► avg ───┘        ──► smooth over 5 frames
                                                                                ──► anomaly score (0 to 1)
```

**Memorise it as four verbs: Encode → Compare → Subtract → Smooth.**

*One exception to know:* ShanghaiTech follows its own benchmark protocol, using every 2nd frame,
smoothing over 31 frames, and rescaling scores within each clip. The other datasets, and the sealed test,
use every 16th frame, smoothing over 5, and raw scores.

## 4.2 The three ingredients (P-C-B)

### P — PE-Core, the encoder
- Replaces CLIP ViT-L and does the same job.
- **Alone it is not the fix.** With no other change it was better only on UCF-Crime (84.7 vs 75.6), and much
  worse on XD (49.4 vs 62.4). The gains came from the text.

### C — Calm normal phrases
Eight fixed phrases added to the normal side:

| Quiet places | Calm people |
|---|---|
| an empty walkway | a person standing still |
| an empty street | a person sitting quietly |
| an empty room | two people talking calmly |
| a quiet open area with nobody around | a person walking slowly |

- **Why it works:** normal should be described as still and calm, with nothing that looks like an anomaly.
  Then the normal and abnormal sides stay clearly apart.
- **Effect on PE-Core:** ShanghaiTech +9.1, XD +14.1, NWPU +5.1 (all significant); UCF −3.0 (within noise).
- **Was it luck?** We wrote a second, completely different set of eight calm phrases. It passed too.

### B — A bank of 48 concrete actions
- **12 types × 4 phrases:** fighting, weapon, theft, vandalism, road accident, fire or explosion, riot or
  arrest, abuse, falling, vehicle among pedestrians, sudden running, trespass.
- **Examples:** "two people punching each other", "a cyclist riding through a crowd of pedestrians", "a car on fire".
- It replaces the old three vague abnormal phrases, like "a dangerous or violent event".
- **Written once by an LLM, before any score existed, then frozen.** The same list is used for every dataset.
- **Effect on top of C:** XD +8.8 and UCF +3.9 (significant); ShanghaiTech +2.6 and NWPU −3.0 (within noise).

## 4.3 How we decided: the rule

Every idea had its rules **written and committed to Git before** any score existed. An idea was kept only if,
across the four datasets, it was:
1. **higher on more than half** of them,
2. **significantly better on at least two,**
3. **significantly worse on none.**

Every run also had to reproduce the old scores exactly before anything new was reported.
That catches bugs.

> **Why this matters in a viva:** it shows we didn't keep trying things until the numbers looked good.
> The rule decided, and the date in Git proves the rules came first.

---

# Part 5 — The results and how to read them

## 5.1 Old model to final model

| | ShanghaiTech | UCF-Crime | XD-Violence (AP) | NWPU Campus |
|---|---|---|---|---|
| Old: ViT-L + LLaVA sentence | 75.1 | 82.4 | **41.7** | 66.7 |
| Plain PE-Core | 66.6 | 84.7 | 49.4 | 61.8 |
| PE-Core + C | 75.7 | 81.7 | 63.5 | 66.9 |
| **Final: PE-Core + C + B** | **78.3** | **85.6** | **72.2** | **63.9** |

- Against the old model, on the same videos: **no worse on any dataset, significantly better on XD.**
- NWPU is 3 points lower than with C alone, within noise. Say this before anyone asks.

## 5.2 Against published methods

Every figure below was checked in its paper's PDF (`docs/07_reference_papers/PROVENANCE.md`).

| Dataset | Ours | Below us | Above us |
|---|---|---|---|
| UCF-Crime | 85.6 | Frozen VLM 85.3, PANDA 84.9, VADTree 84.7, AnyAnomaly 80.7, LAVAD 80.3 | Flashback-PE 87.3, ASK-Hint 89.8, NOVA 89.9 |
| XD-Violence (AP) | 72.2 | VERA 70.5, PANDA 70.2, Frozen VLM 69.0, VADTree 68.9, LAVAD 62.0 | Flashback-PE 75.1, NOVA 84.8 |
| ShanghaiTech | 78.3 | MPN 73.8, Liu et al. 72.8 (trained) | FPDM 78.6, AnyAnomaly 79.7, MULDE 81.3, AnomalyRuler 85.2 |
| NWPU Campus | 63.9 | HF2-VAD 63.7, MNAD 62.5, MemAE 61.9 (trained) | MPN 64.4, AMMC-Net 64.5, Cao et al. 65.8 and 68.2 (trained) |

**The rule for talking about this table:** gaps of 1–2 points may be noise. Say "above" or "below",
never "beats" or "state of the art".

## 5.3 Why these numbers are good

1. **No training at all.** No labels, no fine-tuning, no captioning model at run time.
2. **One setup for crime and campus.** NOVA, Flashback and ASK-Hint report only on crime datasets.
3. **Campus is hard for everyone.** NWPU's own authors reach 68.2 *with* training. Trained one-class methods
   score about 62–66. We reach 63.9 with none. On ShanghaiTech we're level with trained FPDM (78.6).
4. **The numbers can be trusted:** rules first, confidence intervals, failures reported, a sealed test still to come.

## 5.4 What we tried and dropped (one line each)

| Idea | Why it was dropped |
|---|---|
| "Normal scene / Anomalous scene" labels | Gained on three datasets, but significantly worse on NWPU |
| Both encoders together | Only XD improved, at double the cost |
| Scoring by anomaly type (NOVA) | XD rose to 76.9, but ShanghaiTech fell 8 points |
| Different templates per side (Flashback) | Significantly worse on NWPU |
| Phrases chosen by scene | No gain, though it recognised scenes well (11 of 12 ShanghaiTech cameras picked "campus walkway") |
| Camera memory | Significantly worse on ShanghaiTech and XD |

## 5.5 Why some methods score higher, and why we don't copy them

| Method | What gives it the edge | Why we don't copy it |
|---|---|---|
| NOVA | An LLM writes 20 normal + 20 abnormal phrases per type, with guidance for each dataset | Tailored to crime datasets; its scoring cost us 8 points on campus |
| Flashback | A memory of 1 million generated captions (GPT-4o, 76 hours, $181), and a video encoder | Expensive to build; needs a different encoder |
| ASK-Hint | A large video-language model | Heavy; it chose its prompt count from its own results table |
| Frozen VLM | A large model asked about every frame | Heavy at run time |

**What limits our model** (memorise these three):
1. **One frame at a time:** running looks like walking in a single frame.
2. **The whole frame at once:** a small bicycle in a crowd barely changes the frame's vector.
3. **Tailoring has a price:** what helps crime video can hurt campus video.

## 5.6 Honest limits

- **Exploratory until the sealed test:** the model was chosen on these four datasets.
- **The bank isn't independent:** the LLM that wrote it knew these datasets' anomaly types.
- **NWPU dropped a little** with the bank (66.9 → 63.9, within noise).
- **XD is still behind the leaders** (72.2 against Flashback-PE's 75.1 and NOVA's 84.8).
- **UBnormal is computer-generated,** so its result won't prove performance on real cameras.

## 5.7 The sealed test, and why it matters

On 6 November 2026 the model was **frozen**: every setting was fixed and committed (Git commit `3792b86`)
before any UBnormal data was downloaded. UBnormal has 29 scenes the model has never seen. It will be
scored **once**, and the result reported whatever it shows. The wording for each possible result was
fixed in advance (`docs/10_phase3/PREREG_sealed_ubnormal.md`).

> *"We chose the model on four datasets, then froze it and tested it once on a fifth it had never
> seen. So the final number shows how it does on new data, not how well we tuned it."*

---

# Part 6 — Defending it: 20 likely questions

**1. What does your system do, in one line?**
It gives every CCTV frame an anomaly score by comparing it with calm descriptions of normal scenes and
concrete descriptions of anomalies, using a frozen model and no training.

**2. Why training-free? Isn't training always better?**
Training usually scores higher on the camera it was trained for, but needs labelled video from every new
site. Training-free works from day one anywhere. I measure how close it gets, and on campus data it's
inside the range of trained methods.

**3. What is PE-Core?**
Meta's 2025 Perception Encoder. It turns images and text into comparable vectors, like CLIP but newer.
On its own it wasn't the improvement. The text changes were.

**4. Why do calm phrases help?**
Without them, the normal side is vague, and some normal phrases sit close to anomalies. Calm, still
scenes keep the two sides far apart, so a frame showing action moves clearly towards the abnormal side.
It's NOVA's rule: never put look-alike actions on the normal side.

**5. How do you know the calm phrases weren't just lucky wording?**
I wrote a second, completely different set of eight calm phrases and tested it under the same rule. It
passed too.

**6. Who wrote the phrase bank? Did an AI write it?**
Yes, an LLM wrote it once, before it was scored, and then it was frozen. NOVA and Flashback also use
LLM-written text. I state openly that the LLM knew these datasets' anomaly types. That's why the sealed
test on an unseen dataset matters.

**7. Didn't you tune on the test data?**
The four datasets shaped my choices, so I call those results exploratory. To deal with that, I froze the
model and will test it once on UBnormal, which played no part in any decision.

**8. Why is NWPU only 63.9?**
NWPU is hard for everyone. Its own authors reach 68.2 with training, and trained one-class methods score
62–66. Its anomalies are often small or depend on the scene, which a whole-frame model struggles with.

**9. NOVA gets 89.9 on UCF. Why not just use NOVA?**
NOVA writes 20 + 20 phrases for each anomaly type with guidance for each dataset, and only tests on crime
datasets. When I tried its scoring, XD rose but ShanghaiTech fell 8 points. I want one setup that works
on any camera.

**10. What is AUROC?**
The chance that a random anomalous frame scores higher than a random normal frame. 0.5 is a coin flip,
1.0 is perfect.

**11. Why AP for XD-Violence?**
Because XD's official benchmark uses it, so my numbers are comparable with published ones.

**12. What does "significant" mean here?**
I redo the measurement on 1,000 random re-draws of the videos. If the middle 95% of "new minus old" is
above zero, the gain is very unlikely to be luck.

**13. What failed?**
Six ideas: keyword labels, two encoders, scoring by type, different templates, scene-chosen phrases, and
camera memory. All are reported with their reasons.

**14. Why did the scene idea fail if scenes were recognised correctly?**
Recognition worked: 11 of 12 ShanghaiTech cameras picked "campus walkway". But the scene's shorter phrase
list scored the same as the whole bank, so the extra step added nothing to the score.

**15. What happened to your original sentence idea?**
It helped on campus video, but it broke XD-Violence because the sentence was written from frames already
showing violence. Better phrases for normal and abnormal turned out to matter more than a sentence for
each video.

**16. What is concept shift?**
When the same picture means something different in a different place, like a car on a road and a car on a
footpath. Anomaly detection is full of it, while most domain adaptation deals with appearance changes instead.

**17. Is this novel?**
The individual parts come from earlier work. What's new is testing one training-free setup on crime *and*
campus benchmarks, with rules written in advance and confidence intervals, and showing where it works
and where it doesn't.

**18. What would you do with more time?**
Score parts of the frame, for small objects. Try a video encoder, for motion. Test on more unseen datasets.

**19. Would it work in real life?**
It needs no setup data and runs one frame through one encoder, so it's light. Its weak spots are small
objects and motion, and the sealed test will show how it handles new scenes.

**20. What is the one result you are proudest of?**
XD-Violence going from 41.7 to 72.2 without a captioning model, decided by a rule I wrote before seeing
any number.

---

# Part 7 — Teaching it to anyone

## 7.1 The 30-second version (anyone)

*"CCTV cameras record far more than anyone can watch. My system scores each moment for how unusual it
looks. It uses a ready-made AI model that understands pictures and words, and compares each frame with
calm descriptions of normal scenes and concrete descriptions of trouble, like a fight or a fire. Nothing
is trained, so it can be used on a new camera straight away."*

## 7.2 The 2-minute version (a technical interviewer)

1. **Problem:** video anomaly detection without training, where normal depends on the place (concept shift).
2. **Method:** a frozen PE-Core encoder. Score = similarity to an abnormal prototype minus similarity to a
   normal prototype, then sigmoid and smoothing.
3. **Key idea:** the text. The normal side is calm, with no look-alike actions, and the abnormal side is 48
   concrete actions written once and frozen.
4. **Story:** a per-camera sentence broke XD because it was written from frames already showing
   violence. The new text fixed it: 41.7 → 72.2.
5. **Rigour:** one idea at a time, rules committed first, confidence intervals, six failures reported.
6. **Next:** the model is frozen, with one sealed test on unseen UBnormal.

## 7.3 The whiteboard version

### Drawing the architecture in 60 seconds

Draw from left to right, and say each line as you draw it:

1. **Three boxes, stacked on the left:** FRAME, NORMAL, ABNORMAL. *"Three inputs: the video frame, calm
   descriptions of normal, and 48 concrete descriptions of anomalies."*
2. **One tall box in the middle, labelled PE-Core with a snowflake.** Draw three arrows into it.
   *"Step 1, encode: one frozen model turns all three into vectors. Nothing is trained."*
3. **Three boxes on its right:** frame vector, normal average, abnormal average. *"Each list of phrases is
   averaged into one vector."*
4. **A COMPARE box** with all three arrows going in. *"Step 2, compare: how close is the frame to each side?"*
5. **SUBTRACT, then SMOOTH**, stacked on the far right. *"Step 3: abnormal closeness minus normal
   closeness. Step 4: smooth over a few frames, and that's the anomaly score."*
6. **Write P-C-B underneath.** *"PE-Core, Calm phrases, Bank of actions."*

**If asked for detail**, add λ = 100 and the sigmoid next to SUBTRACT, "5 frames" next to SMOOTH, and
"1024 numbers" next to the frame vector. The detailed figure has everything else.

### The quick sketch

If you only have a corner of the board, draw this, top to bottom, saying the four verbs aloud:

```
   FRAME ──encode──► vector
                        │  compare with
         ┌──────────────┴──────────────┐
   NORMAL (calm)                  ABNORMAL (48 actions)
         └──────────────┬──────────────┘
                     subtract
                        │
                      smooth
                        │
                  ANOMALY SCORE
```

## 7.4 An analogy for non-technical people

*"Imagine a guard with two photo albums: one of calm, ordinary scenes and one of trouble, like fights,
fires and accidents. For every moment of video, the guard asks which album it looks more like. My system
does that with an AI model instead of a guard, and nobody has to train it first."*

---

# Part 8 — Glossary in plain words

| Term | Plain meaning |
|---|---|
| Anomaly | Something unusual for that place |
| Training-free / zero-shot | Used as is, with nothing learned from the target data |
| Frozen | The model's internals are never changed |
| Encoder | The model that turns pictures and text into vectors |
| Embedding / vector | A list of numbers that stands for a picture or sentence |
| Cosine similarity | How closely two vectors point the same way |
| Prototype | The average of a list of phrase vectors |
| Margin | Abnormal similarity minus normal similarity |
| Sigmoid | Squashes any number into the range 0–1 |
| AUROC | Chance an anomalous frame outscores a normal one |
| AP | How precise the alarms are, averaged over thresholds |
| Bootstrap | Redoing a measurement on many random re-draws |
| 95% CI | The range the true difference very likely lies in |
| Significant | CI entirely above (or below) zero |
| Calibration | Moving whole videos onto one scale |
| Re-ranking | Ordering frames better within a video |
| Concept shift | The same picture means something different elsewhere |
| Pre-registration | Writing the rules and predictions before seeing results |
| Sealed test | One test on data never used for any decision |
| PE-Core | Meta's 2025 encoder used in the final model |
| C | The eight calm normal phrases |
| B | The 48-phrase bank of concrete actions |

---

# Part 9 — Self-quiz

Cover the answers and say each one out loud.

| # | Question | Answer |
|---|---|---|
| 1 | What are the three ingredients? | PE-Core, calm phrases, phrase bank (P-C-B) |
| 2 | The four verbs of the pipeline? | Encode, compare, subtract, smooth |
| 3 | XD before and after? | 41.7 → 72.2 |
| 4 | Final scores on the four datasets? | 78.3, 85.6, 72.2, 63.9 |
| 5 | Why did the LLaVA sentence break XD? | It was written from first frames that already showed violence (127 of 500 videos) |
| 6 | NOVA's rule for the normal side? | Never include look-alike actions |
| 7 | How was luck in the calm phrases ruled out? | A second, different wording also passed |
| 8 | The three parts of the keep rule? | Better on more than half, significant on at least 2, significantly worse on none |
| 9 | Name three dropped ideas | Any three: labels, two encoders, scoring by type, templates, scene choice, camera memory |
| 10 | Why is NWPU hard? | Small and scene-dependent anomalies; trained methods only reach 62–68 |
| 11 | Why not copy NOVA? | Tailored to crime datasets; its scoring cost 8 points on ShanghaiTech |
| 12 | What is the sealed test? | One frozen run on unseen UBnormal, reported whatever it shows |
| 13 | AUROC in one sentence? | Chance a random anomalous frame outscores a random normal one |
| 14 | Calibration or re-ranking: which is stronger evidence? | Re-ranking; it shows the event itself is seen |
| 15 | Three limits? | Chosen on these datasets; the bank isn't independent; one frame at a time |

---

## Where things live

| What | Where |
|---|---|
| Final model's rules and freeze | `docs/10_phase3/PREREG_sealed_ubnormal.md` |
| The improvement round, step by step | `docs/10_phase3/PREREG_ideas.md` |
| The fixes run that found C | `docs/10_phase3/PREREG_fixes.md` |
| Phrases (C, the second wording, B) | `da_zvad/prompts/fixes.py`, `da_zvad/prompts/ideas.py` |
| Experiment code | `notebooks/ideas_sweep.py`, `notebooks/fixes_sweep.py` |
| Results | `results/runs/analysis/ideas_step*`, `fixes_*` |
| Published numbers, checked | `docs/07_reference_papers/PROVENANCE.md` |
| Presentation | `docs/06_presentations/DA-ZVAD_Phase4_Review.pptx` |
| Detailed earlier handbook | `docs/08_understanding/00_HANDBOOK.md` |
