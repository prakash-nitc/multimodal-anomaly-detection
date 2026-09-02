# Phase 2 Review — Speaking Script

**Deck:** `DA-ZVAD_Phase2_Review.pptx` (29 slides) · **Runs ~24 min** + questions.
Cut list at the end gets you to ~18 min if the slot is shorter.

> **Do not memorise this word for word.** Learn the *beat* of each slide — the one thing it
> exists to say. A recited talk sounds recited, and this panel is listening for whether you
> own the work. Where a sentence is worth saying exactly, it is marked **[SAY]**.
> Everything else, put in your own words.

---

## Slide 1 — Title · 20s

Don't read the slide. Just:

**[SAY]** "This is Phase 2 of my thesis work — domain-adaptive zero-shot video anomaly
detection. The one-line version: I adapt a detector to a new environment by writing a
sentence, with no training and every model frozen. I'll show you what worked, and also the
part that didn't."

That last clause buys you enormous goodwill. Say it.

---

## Slide 2 — The problem · 60s

Lead with the picture, not the definition.

- A detector learns "normal" from weeks of footage at one camera. Move it, it fails.
- Mall vs factory: **the same forklift**. Alarm in one, routine in the other.
- **[SAY]** "The footage didn't change. The rule changed."
- Cost today: every new site means new footage and a retraining cycle. That's the cost I'm removing.

---

## Slide 3 — Research gap 1 of 2 · 75s

This is the slide that proves you read the six surveys. Slow down.

- Domain adaptation splits differences between places into kinds.
- **Covariate shift** — appearance moves. Stop sign in fog vs sunshine. Still a stop sign.
- **Concept shift** — the rule moves. Bicycle on a road vs on a footpath. Identical image, opposite label.
- Read the Liu quote **aloud** off the slide.
- **[SAY]** "The field scopes concept shift out by explicit decision — and for object
  recognition that's the right call. A cat is a cat everywhere. But anomaly detection is
  built on concept shift."

You are not criticising them. Say that.

---

## Slide 4 — Research gap 2 of 2 · 60s

- Walk one row of the table. A person running: normal in a park, anomalous in a bank vault.
- **[SAY]** "In anomaly detection, normal is defined by the deployment context, not by the
  object. That is the definition of the task."
- So methods that align appearance can't help — the appearance is already identical. Two
  domains can share p(x) exactly while the labelling function differs.
- **Volunteer this:** a 2026 position paper reached the same premise independently. Saying
  it yourself answers "did you invent this problem?" before it's asked. What's still
  missing is a test of whether supplied context does any work.

---

## Slide 5 — The framework · 60s

Four modules, left to right, about thirty seconds. Don't linger.

- M1 frozen CLIP turns each frame into 768 numbers of unit length.
- Frame scoring: a softmax over how well that vector matches the "normal" prototype
  versus the "abnormal" one. **[SAY]** "$s_t$ is just how far the frame leans abnormal, on a 0–1 scale."
- M2 centred moving average, window 31 frames. **[SAY]** "About a second either side, so a
  single odd frame can't raise an alarm." No parameters.
- M3 the scene description. **Stop here.**
- M4 frozen LLaVA explains a detected event.

**[SAY]** "M3 is the only place the domain enters the system. Moving to a new site means
editing one sentence — no target data, no gradients."

> The figure now carries the actual operation for each module (the softmax, the window, the
> prototype construction). You do not have to read those out — but if a panelist asks
> "what exactly does M2 do", point at the formula and the small before/after plot beside it.

---

## Slide 6 — Why freezing everything is the point · 60s

The strongest methodological point in the deck. Say it slowly.

- **The risk:** if the system learned anything from the new site and got better, we
  couldn't say what caused it — the sentence, or the learning.
- **The design:** no parameter anywhere changes. Exactly one thing varies: the text.
- **[SAY]** "So any measured difference is attributable to the sentence. There is no other candidate."
- Then the point that lands: **you cannot run this experiment on the competing systems.**
  Their text is learned on source data, or entangled in an LLM prior, or paired with a
  trained adapter. Freezing is what makes the claim testable instead of asserted.

---

## Slide 7 — Evaluation design · 60s

- Same pipeline four times. Only the sentence changes: none / generic / matched / mismatched.
- Generic controls for merely *having* context. Mismatched is the falsifying control.
- **[SAY]** "We fixed the interpretation before we measured anything: matched at least
  generic at least none, with mismatched measurably worse. If a deliberately wrong
  description costs nothing, the method ignores its context and the claim is refuted."

A panel respects a test you could have failed. Add: "and later in this deck, one we partly did."

---

## Slide 8 — Setup · 30s

Brief. This is the credibility slide, so let the content carry it.

- Two benchmarks, 128 test clips, 28,118 frames, frame-level ground truth. A40. CLIP
  ViT-L/14, frozen. Five full runs.
- **[SAY]** "Every run writes a manifest — the code commit, whether the tree was clean,
  host, GPU, driver, library versions, full config, frame and label counts. Committed with
  the results. Every figure in this deck traces back to the state that produced it."
- **If anyone doubts the work is yours, offer to open a manifest.** Out loud.

---

## Slide 9 — What the data looks like · 30s

Thirty seconds, no more. Point down one column.

**[SAY]** "Each column is one fixed camera — scene, angle and lighting constant. Only the
event changes. An empty walkway and the same walkway with a cyclist look alike and are
labelled opposite."

---

## Slide 10 — The first result was a failure · 45s

**Do not rush this and do not apologise for it.**

- 0.49 AUROC. Fifty minutes of GPU time to match a coin flip.
- And the central experiment came out **backwards**: matched was *worst* at 0.666,
  mismatched among the *best* at 0.695.
- **[SAY]** "This is the point where the idea looks broken."

Then **pause.** Let it sit. The next three slides are why this talk is worth hearing.

---

## Slide 11 — Diagnosis 1: we were measuring it wrong · 75s

- 12 camera views. CLIP sits at a different baseline score under each — different lighting,
  different angle.
- Our error: we pooled every frame from all 12 cameras into one ranking.
- **Use the analogy, it works on everyone.** **[SAY]** "It's like ranking students from
  different schools by raw marks when the schools grade differently. The comparison
  destroys the ordering."
- 0.49 → 0.71 on identical scores.
- **Stress twice:** the fix **uses no labels**, and it is the benchmark's **own published
  protocol**. We weren't following it.
- **[SAY]** "We report both figures in the paper."

---

## Slide 12 — The evidence for that diagnosis · 45s

- Left: every frame in two dimensions, coloured by camera. The views sit in **separate regions**.
- Right: similarity to the average frame differs by more than 0.13 between views.
- **[SAY]** "Neither panel uses labels — so this isn't hindsight. The cameras occupy
  separate regions at different similarity levels, which is exactly why pooling raw scores
  destroyed the ordering."

---

## Slide 13 — Diagnosis 2: the description cancelled itself out · 90s

The most interesting slide in the deck. Give it the time.

- We were appending the scene sentence to **both** prompt sets. Point at the two prompts
  and let them see the shared words.
- Each prompt set is averaged into one summary vector. Shared text enters both, so the two
  summaries move **toward each other** — and the method depends on them being different.
- **The punchline. [SAY]** "An accurate description matches every frame strongly, so it
  absorbs the most contrast. A wrong one matches nothing, so it does no damage. That is why
  the result inverted."
- **The fix:** attach the description to the **normal** prompts only. The scene defines what
  normal looks like here; an anomaly is a departure from it.

---

## Slide 14 — After the fix: the predicted signature · 75s

The headline slide.

- ShanghaiTech, all 107 clips, per-clip normalised. Every model frozen.
- **Point at the `none` column: 0.707 in both rows.** **[SAY]** "That's the control — it
  confirms nothing but the injection point changed."
- Both prompt sets: gap −0.029. Normal set only: **+0.105**.
- Point at 0.628. **[SAY]** "A wrong sentence costs ten points. Nothing else in the system
  was permitted to change. The text caused it."

---

## Slide 15 — The same experiment, drawn · 30s

- Trace the two arrows with a finger. Grey = sentence in both sets, distance negative.
  Orange = normal set only, +0.105.
- **[SAY]** "The only thing separating them is where the sentence was attached."

---

## Slide 16 — One clip, two sentences · 40s

- Same frozen models, same frames, same smoothing. Only difference: campus walkway vs
  industrial site.
- **[SAY]** "The curves coincide outside the event — that's on purpose, that's the control —
  and separate inside it."
- Worth naming: **there is no aggregation here to argue with.** Two runs over identical
  frames with identical frozen weights.

---

## Slide 17 — Stating the claim precisely · 60s

Be scrupulous here. Understating protects you.

- **The weaker half:** a correct description beats none by +0.027. Positive at every window,
  but inside the ±0.036 split-to-split spread. **[SAY]** "So we report direction, not magnitude."
- **The claim we make:** a wrong description costs −0.105. Unambiguous.
- **[SAY]** "The description constrains a decision boundary rather than adding information.
  It doesn't reliably lift performance when correct; it degrades it sharply when misdirected."
- Secondary finding, not in the literature: **where** you inject dominates **what** it says.

---

## Slide 18 — Which components earn their place · 60s

- Language alone is best: 0.718 held-out. Adding motion costs 0.001. Adding scene-centre
  normality costs 0.067.
- **Volunteer the surprise. [SAY]** "We expected motion to help. It doesn't. ShanghaiTech's
  anomalies look kinematic — a bicycle at cycling speed — so appearance and motion should be
  complementary. Measured here, they aren't. The pooled embedding already registers enough of it."
- Reporting the prediction that failed is stronger than reporting only the ones that held.

---

## Slide 19 — Window length, and what each component adds · 35s

- Left: smoothing helps up to w=31, then hurts — a real optimum, not the metric rewarding
  blur. The grey curve is the wrong pooling; it never leaves chance.
- Right: **volunteer the weakness before anyone asks.** **[SAY]** "The error bars overlap, so
  the top three are statistically tied. The honest claim is that nothing we added beat plain
  language — that's a negative result about our own elaborations, not a win over the alternatives."

---

## Slide 20 — A second domain · 75s

- Identical frozen configuration on CUHK Avenue. Nothing retuned. Only the sentence changed.
- **Detection transfers:** 0.706 vs 0.707 with no descriptor.
- **Adaptation does not:** gap 5× smaller, and matched scores *below* none.
- **Give the honest reading. [SAY]** "On Avenue a placeholder beats an accurate description.
  Whatever that benefit is, it cannot be domain adaptation."
- Then the conjecture **and** the test: ShanghaiTech has 12 views, Avenue has one. A scene
  description has work to do only when there are several environments to tell apart.
  Testable — run the sweep inside a single ShanghaiTech view.

---

## Slide 21 — We tested that explanation, and it held · 75s

This slide shows a full research cycle. Name it as one.

- ShanghaiTech is twelve single-view datasets stacked together. If the descriptor works by
  saying *which* scene you're in, confining the sweep to one view should reproduce Avenue's
  flat result.
- Pooled +0.105 → within-view +0.033 → Avenue +0.020.
- **[SAY]** "A within-view gap near +0.105 would have refuted this outright. It came back at
  a third of that, next to Avenue's figure."
- **Give the caveat yourself:** three of nine views are negative. Inside one scene the effect
  is not reliable.
- **[SAY]** "What the sentence mainly supplies is which scene you're in — not what counts as
  normal within it."

---

## Slide 22 — Every camera view, one at a time · 35s

- **Show the spread yourself.** Orange line is the pooled result; each bar is one view alone.
- Three are negative; views 03 and 07 nearly reach the pooled figure. Bars rest on 5–34 clips each.
- **[SAY]** "So this is a shift in the average, not a clean collapse."

Saying this before the panel spots it is the difference between a caveat and a hole.

---

## Slide 23 — Six things that did not work · 45s

- Don't read all six. Pick two — quadrant scoring, and prompts naming bicycles (0.486, much worse).
- **[SAY]** "The claim is that the minimal configuration is the right one. That's only
  credible next to the alternatives that were tried. A clean table of successes is the
  artefact that's easy to fabricate. Six diagnosed failures are not."

**If the panel suspects generated work, this slide is your best answer.** Deliver it looking at them.

---

## Slide 24 — Where this sits against the literature · 60s

Confident, not apologetic.

- **Anchor on the trained baseline, not LAVAD.** **[SAY]** "We match the benchmark's own 2018
  baseline — 0.734 against 0.728 — using none of its training data."
- **Then concede LAVAD openly:** about 0.85, but it runs a captioner, an LLM and a refiner per
  frame. We run one frozen encoder and a sentence, in about 7 GB.
- **[SAY]** "We don't expect to exceed trained state-of-the-art on absolute AUROC, and the
  paper makes no such claim."

---

## Slide 25 — Limitations we are stating ourselves · 50s

Deliver these as findings, not confessions.

- The mechanism is bounded — works on ShanghaiTech, nearly vanishes on Avenue; the
  scene-diversity explanation rests on two datasets.
- Resolution ceiling at 224×224; quadrant scoring didn't close it.
- **Dwell on the metric one.** Per-clip normalisation is affine, so a monotone rescaling
  changes the pooled figure even though the ranking is identical. **[SAY]** "That's a real
  methodological point about a protocol the whole field uses — and we found it by chasing our
  own bug."
- No validation split exists, so we split clips and report the half never used to select.
- Both video domains are outdoor pedestrian surveillance. M4 has produced no video results yet.

---

## Slide 26 — Phase 3 plan · 40s

- Priority 1: the within-view sweep — **[SAY]** "it turns our explanation from a conjecture
  into a result."
- Then the MVTec context sweep, patch-level scoring, and running M4.
- **[SAY]** "The infrastructure cost is now paid. Encoding a benchmark takes 21 minutes, after
  which a new scoring hypothesis is evaluated in seconds instead of fifty."

---

## Slide 27 — Summary · 45s

Five beats, one line each. **Close on the boundary, not the number.**

1. The field targets covariate shift by explicit scoping; anomaly detection is dominated by concept shift.
2. Every model frozen — which is what makes the claim identifiable.
3. 0.734 with no training, and a wrong description costs 0.105.
4. Where the description is injected dominates what it says. Not in the literature.
5. It doesn't replicate on a single-scene benchmark, and the within-view control shows why.

**[SAY]** "The claim is scoped rather than abandoned, and we named the experiment that settles
it. All results reproducible from committed code and run manifests."

Then hand over. Don't trail off.

---

## Slides 28–29 — References

**Do not read these out.** Advance past them; they are there as an evidence base.

The numbering is the paper's own, so `[7]` on a slide is `[7]` in the report — if a panelist
asks where something is cited, the two agree. Be ready for one question: *which survey has
the quote?* — **Liu et al., 2022, reference [14].**

---

# If you need ~18 minutes

Cut in this order. Advance through the cut slides without stopping — don't delete them.

1. Slide 15 (drawn version of 14) — 14 already made the point. **−30s**
2. Slide 19 (window chart) — fold "error bars overlap" into slide 18. **−35s**
3. Slide 22 (per-view bars) — fold "three of nine negative" into slide 21. **−35s**
4. Slide 12 (t-SNE evidence) — keep only if slide 11 gets challenged. **−45s**
5. Slide 9 (data figure). **−30s**
6. Trim slide 23 to one example instead of two. **−20s**

**Never cut:** 6, 7, 10, 11, 13, 14, 20, 21. Those eight are the talk.

---

# The three sentences to fall back on

If a question goes somewhere you didn't prepare:

> The domain-adaptation literature excludes concept shift by explicit choice, and anomaly
> detection is built on concept shift.
>
> The papers that use language show it works but never test whether it is causally responsible.
>
> We characterise the first and supply a protocol for the second — including a control that
> our own framework initially failed.

And if you genuinely don't know: **"I haven't tested that yet — here's the experiment that
would answer it."** That answer is always available to you, and it never sounds weak.
