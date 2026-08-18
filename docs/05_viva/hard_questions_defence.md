# Hard Questions — Defence Notes

*Adversarial questions, with prepared answers. Compiled 14 August 2026 after the
first full GPU run. Companion to `viva_questions_final.md`, which covers
slide-wise explanatory questions — this file covers the ones designed to test
whether the work holds up.*

**How to use this.** Each entry has a short answer to say first, a fuller
version if pressed, and a trap to avoid. Learn the short answers. The fuller
ones are for when someone follows up. The traps are the phrasings that lose the
room.

Most of these questions came from the student, not from a supervisor. That is
worth noting: they are the questions a careful person asks about their own work.

---

# A. Novelty and contribution

These are the hardest. Answer them precisely or not at all.

---

### A1. Is this not already done? Zero-shot, domain adaptation and explanation all exist.

**Short answer**
> "Each ingredient exists. The combination doesn't — no published system is
> fully frozen, adapts by text alone, evaluates that as domain adaptation, and
> explains. But the combination isn't the argument. What the combination
> *enables* is."

**If pressed**
> "You cannot run our central experiment on any of those systems. LAVAD's notion
> of anomaly is entangled in an LLM's prior, so changing the text tells you
> nothing about cause. VERA optimises its text on source data, so text and
> training are confounded by construction. OVVAD trains a temporal adapter. Ours
> is the only pipeline where text is the *only* free variable — which is the
> precondition for measuring whether text does anything. The architecture exists
> to make the experiment possible."

**Trap**
Do not claim "text at inference time" as novel. AnyAnomaly (2025) does that.
Name it yourself before they do.

---

### A2. What exactly is the research gap you are filling?

**Short answer**
> "Two halves. The domain-adaptation literature explicitly excludes the shift
> type anomaly detection is built on. And the VAD papers that use language never
> test whether the language is causally responsible."

**If pressed — the theory half**
> "Liu et al. state they focus on covariate shift because concept shift 'is
> usually not a common problem'. Singhal et al. list stable p(y|x) as the *first*
> condition under which domain adaptation is justified. So classical DA theory
> assumes away exactly what defines our task — normality is a property of
> deployment context, not of the object."

**If pressed — the empirical half**
> "Papers supply text, report that it works, and stop. None supplies
> deliberately *wrong* text to check whether the system is listening."

**The 'so what' follow-up**
> "Because the assumption turns out to be fragile. We ran the control and in our
> first configuration the text was actively hurting — an accurate scene
> description was the worst condition we tested. Had we done what the literature
> does, we would have published a wrong conclusion. The gap isn't academic."

---

### A3. Isn't "nobody ran a control" a trivial contribution?

**This is a fair challenge. Concede the framing, then correct it.**

**Short answer**
> "A control is trivial when it confirms. Ours overturned the result. The
> contribution isn't that we ran an ablation — it's the failure mode it exposed."

**If pressed**
> "The obvious implementation of verbalised adaptation actively harms
> performance, and the harm scales with how *accurate* the description is. That
> is a structural property of any method that mean-pools prompt ensembles and
> injects shared context into both sides — it pulls the two prototypes together
> and erodes the margin the decision rests on. It's a claim about a class of
> methods, not about our code."

**Trap**
Do not say "we ran an ablation nobody ran." That is the phrasing that makes it
sound trivial, because it is.

---

### A4. Is the contribution strong enough to publish?

**Answer honestly. This is not a question to bluff.**

> "As it stands, not for a strong venue. We have a characterisation, a protocol,
> two findings and a boundary condition — but no new architecture, and the
> central effect holds on one of the two video benchmarks we tried. What would
> strengthen it is the within-view control that would turn our explanation for
> that boundary into a result, and a direct measurement of the dilution
> mechanism. For the thesis, the work is a characterisation and a measurement
> method that already caught two real errors in its own execution."

---

### A5. Your central finding does not replicate on the second dataset. Doesn't that refute it?

**The hardest question you will get. Do not soften it.**

**Short answer**
> "It bounds it rather than refuting it. Detection transfers almost exactly —
> 0.706 on Avenue against 0.707 on ShanghaiTech with no descriptor. What does not
> transfer is the context effect: the gap falls from +0.105 to +0.020, and the
> correct description scores below no description at all."

**The explanation, offered as a conjecture**
> "ShanghaiTech has thirteen camera views; Avenue has one. A scene description
> has a discrimination to perform only when there are several environments to
> tell apart. On Avenue every clip shows the same view, so the base prompts
> already cover the only environment present and there is nothing for the
> descriptor to resolve."

**The part that must be conceded**
> "The best condition on Avenue is the *generic* placeholder, which carries no
> information about the environment. So whatever it contributes there cannot be
> domain adaptation — most likely it just improves the prompt ensemble as an
> ensemble. I can't explain that away."

**Why this is still worth having**
> "A residual effect survives: mismatched remains the worst condition on both
> benchmarks. And the conjecture is testable — partition ShanghaiTech by camera
> and run the sweep inside a single view. If the effect disappears there, scene
> diversity is established as the operative variable. That experiment is named in
> the paper and is our first Phase 3 task."

**Trap**
Do not claim the method "works on ShanghaiTech and needs more investigation on
Avenue." That is evasive. Say the effect nearly vanishes, say the placeholder
wins, then give the conjecture and the test.

**Why answer this way**
Overselling invites exactly the follow-up you cannot answer. Naming the shortfall
yourself and stating the remedy reads as command of the work.

---

# B. The modest accuracy

---

### B1. Existing models perform better. Why do zero-shot domain adaptation at all?

**The strongest answer you have. Learn it properly.**

**Short answer**
> "Existing models do better *when they have training data from the deployment
> site*. On a benchmark everyone has it. At a new site nobody does — the trained
> model's accuracy there isn't 0.728, it's undefined until someone spends weeks
> producing it."

**If pressed**
> "Liu et al. reach roughly 72.8% by training on ShanghaiTech's own training
> split. We reach 0.734 using none of it — so we match the benchmark's original
> trained baseline with zero training data. Deploy to a different campus and
> their model does not yet exist. LAVAD reaches 0.85 but runs three large models
> per frame and offers no channel for an operator to state what normal means at
> their site. We run on day one, on one frozen encoder, adaptable by editing a
> sentence."

**The research half**
> "But accuracy was never the question. We asked whether language alone can carry
> adaptation and whether that can be measured. Accuracy is the instrument, not
> the objective."

**The honest concession — volunteer it**
> "0.734 is not a deployable autonomous detector, and I won't claim it is. The
> positioning is cold start: coverage from day one while data is collected for a
> trained system, or triage that reduces hours of footage to minutes of review."

**Trap**
Never say "we're not competing on accuracy" and stop. Alone it reads as an
excuse. Always pair it with what you *are* competing on.

---

### B2. Your MVTec result was 88.5%. Why is video only 0.734?

**Short answer**
> "Different tasks and different metrics — but more importantly, the MVTec
> breakdown predicted this. Performance there fell as the defect occupied a
> smaller fraction of the pixels, and ShanghaiTech is the extreme end of that
> same axis."

**The evidence**

| MVTec category type | AUROC | Anomaly occupies |
|---|---|---|
| Textures (carpet, grid, leather, tile, wood) | ~99.5% | most of the image |
| Objects (bottle, cable, capsule) | ~83.1% | a region |
| Transistor — small localised defect | 72.7% | a few percent |
| ShanghaiTech — cyclist in a wide frame | 73.4% | ~1–2% |

> "One monotone trend across two benchmarks. And the MVTec prompt ablation said
> the same thing independently — generic 89.1%, category-specific 88.6%, ensemble
> 88.5%. Prompt engineering bought nothing, so the binding constraint is spatial
> resolution, not linguistic specification. ShanghaiTech confirmed it: writing
> campus-specific prompts naming bicycles made things *worse*."

**The follow-up you must be ready for — "then why didn't quadrant scoring fix it?"**
> "It should have, and it didn't. Mean aggregation over quadrants beat max, which
> is the opposite of what recovering a small object looks like. Either quadrants
> are far too coarse, or resolution isn't the whole story. Patch-level scoring
> against the encoder's 256 spatial tokens is the proper test and we haven't run
> it. Resolution is our leading explanation, not a proven one."

**Why concede that**
The quadrant result genuinely sits in tension with the resolution story. Asserting
the story as settled invites the person who noticed to dismantle it.

---

### B3. Why not just ensemble several models to raise the number?

> "It would help, probably one to three points, and it's cheap now that
> embeddings are cached. But it doesn't answer any question — 'we used three
> models instead of one' is an engineering choice, not a finding. The same effort
> spent on patch-level scoring attacks the limitation we actually diagnosed and
> produces a result either way. And the untested cross-domain claim matters more
> than either."

---

# C. Methodology and rigour

---

### C1. Did you tune on the test set?

> "Partly, and we control for it. There is no validation split defined for this
> benchmark, so we split the clips into halves, selected configurations on one
> half, and report the number from the half never used for selection — averaged
> over five different partitions, with the spread reported. Where configurations
> sit inside that spread we report them as tied rather than picking a winner."

**If pressed on prompt design**
> "The campus prompt vocabulary came from the dataset's published anomaly
> taxonomy in Liu et al., not from inspecting test labels. And we report both
> prompt sets we tried, including the one that performed worse."

---

### C2. Isn't per-clip normalisation just making the number look better?

> "It uses no labels — it puts thirteen camera views on a common scale before
> comparing them. It is the protocol from the paper that introduced the
> benchmark, and published results on it do the same. We report the
> un-normalised figure in the same table so it can be checked."

**If pressed on why it matters so much**
> "The macro figure, which is immune to cross-clip offsets by construction, sits
> at 0.648 while raw micro sits at 0.513. That gap identifies the cause — the
> signal exists within clips and pooling across incomparable scales destroys it."

---

### C3. Does a correct description actually help, or only a wrong one hurt?

**Answer precisely. Understating protects you.**

> "Both, but with very different confidence. A correct description beats none by
> +0.027, and that margin is positive at every temporal window we measured and
> grows with the window — so the direction is consistent, not a coincidence of
> one setting. But it sits inside the ±0.036 spread across data splits, so I
> report the direction and decline to claim the magnitude.
>
> The claim rests on the other half: a *wrong* description costs 0.105, which is
> unambiguous. The descriptor constrains a decision boundary rather than adding
> information — it degrades sharply when misdirected."

---

### C4. Motion alone reaches 0.686 without any language. What is the language for?

**The answer here changed when a measurement error was corrected. Know the current one.**

> "Motion reaches 0.686 alone, and the language pathway reaches 0.707. Adding
> motion to language costs 0.001 — they are not complementary. That surprised us:
> ShanghaiTech's anomalies look kinematic, a bicycle at cycling speed, so we
> expected the two to combine. Measured properly they do not, which suggests the
> frozen encoder's pooled embedding already registers enough of the motion that
> an explicit term adds nothing."

**If asked why this differs from an earlier draft**
> "An analysis script scored the cosine margin where the pipeline scores its
> softmax. Those rank frames identically, but the per-clip normalisation in the
> metric is affine, so applying it after a nonlinear monotone map is not the same
> operation — the pooled figures differ by up to 0.05. We caught it by requiring
> the analysis tool to reproduce a pipeline result, corrected every affected
> number, and the metric's scale-sensitivity is now recorded as a limitation."

---

### C5. How do you know the dilution mechanism is real? Did you measure it?

**Currently the weakest link. Do not bluff.**

> "It is inferred, not measured. The account predicts the ordering we observed —
> matched worst, mismatched nearest to none — and fixing the injection point
> inverted the sign of the effect as predicted. But we have not directly measured
> the distance between the two prompt prototypes under each condition. That
> measurement takes minutes and is the next thing on the list."

---

# D. Comparison with prior work

---

### D1. How does your model compare to recent methods?

**Build the comparison on multiple axes, not one number.**

| Method | Training | Target data | Models at inference | Adaptation channel | ShanghaiTech AUC |
|---|---|---|---|---|---|
| Liu et al. 2018 | Full, on target | Required | 1 (trained) | none — retrain | ~0.728 |
| Ada-VAD 2024 | Source + target | Required | trained | pixels / parameters | — |
| zxVAD 2023 | Source training | None | trained | none | — |
| VERA 2025 | Text learned on source | Source needed | 1 VLM | text, learned offline | — |
| LAVAD 2024 | None | None | **3 large models** | none | ~0.85 |
| **DA-ZVAD** | **None** | **None** | **1 frozen encoder** | **text, at test time** | **0.734** |

*Verify every figure from the source papers before quoting.*

> "We lose one column and win four. We occupy a different point on the
> cost/capability curve."

**Three traps**
- Do not compare across protocols without saying so in the caption
- Do not omit LAVAD because you lose to it — a table of only methods you beat is
  the clearest signal of cherry-picking
- Do not add a column purely because you win it

---

### D2. Did you reproduce any baseline yourself?

> "Not for video. LAVAD requires BLIP-2, Llama-2 and ImageBind, and Llama-2 is
> access-gated — realistically several days with a real chance of failure. What
> we did do on MVTec is the controlled version: a one-class SVM on *identical*
> CLIP features and the identical test split, so the only difference is whether
> training happened. That gave 92.4% trained against 88.5% zero-shot. The same
> comparison on ShanghaiTech is a planned experiment."

---

# E. Credibility and provenance

---

### E1. How do I know you actually ran these experiments?

**Have a manifest open in another window.**

> "Every run writes a manifest recording the exact code commit and whether the
> working tree was clean, the host, GPU, driver and library versions, the full
> configuration, and per-dataset sequence, frame and positive-label counts. They
> are committed to the repository alongside the results."

**The strongest supporting detail**
> "The internal control is checkable rather than asserted. Two of our runs differ
> only in where the descriptor is injected, and their 'no context' columns agree
> at 0.685 to three decimals — which they must, if nothing else changed."

---

### E2. Walk me through what went wrong and how you found it.

**This is your best question. Take your time on it.**

> "The first full run returned 0.49 — chance — and the context sweep came out
> backwards, with the correct description worst. Three separate causes.
>
> First, we pooled raw scores across thirteen camera views that sit at different
> similarity baselines, which destroys the within-clip ordering. Normalising per
> clip took the same scores from 0.49 to 0.70.
>
> Second, we were appending the scene description to both prompt ensembles. Each
> is mean-pooled into one vector, so shared text enters both and pulls them
> together — which is why an *accurate* description did the most damage.
> Injecting into the normal ensemble only produced the predicted signature.
>
> Third, my hypothesis that campus-specific anomaly prompts would help was
> wrong — it made things substantially worse, for the same dilution reason."

---

# F. Questions I cannot answer well yet

**Know these. Being caught without an answer is worse than volunteering one.**

**F1. Why does context help on one benchmark and not the other?**
The scene-diversity account is a conjecture from two datasets. The within-view
control that would settle it is not yet run.

**F2. Have you measured the dilution mechanism directly?**
No — inferred from the performance ordering and from the fusion experiment
inverting the effect. The direct measurement, comparing the distance between the
two prompt prototypes under each condition, takes minutes and is not done.

**F3. Does the industrial/surveillance contrast hold for adaptation?**
Unknown. MVTec is evaluated for detection only; the context sweep has never been
run on it. Both video benchmarks are outdoor pedestrian surveillance, so the
concept-shift contrast that motivates the whole argument is currently argued
rather than measured.

**F4. Has M4 produced anything?**
No. The explanation module has not been run on video.

**F5. Does AnyAnomaly already do the falsifying control?**
Needs checking. Read the paper before the viva — this is the one that could
weaken the protocol claim, and you want to find out from your own reading rather
than from a panel member.

**F6. How much does the headline depend on the temporal window?**
More than is comfortable. The optimum is w=31 and performance falls at w=61;
the driver's original sweep stopped at 15 and understated the result by 0.03.
The window was selected on the same test set, since these benchmarks define no
validation split.

---

## The three sentences to fall back on

If a question goes somewhere you did not prepare, return to these.

> The domain-adaptation literature excludes concept shift by explicit choice, and
> anomaly detection is built on concept shift.
>
> The papers that use language demonstrate it works but never test whether it is
> causally responsible.
>
> We characterise the first and supply a protocol for the second — including a
> falsifying control that our own framework initially failed.
