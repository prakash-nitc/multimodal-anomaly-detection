# 02 — The experiments: what we measure, and why the design is airtight

> Read after [01_foundations.md](01_foundations.md). This note covers the
> domain-adaptability landscape (what the guide asked about), the three
> experiments our framework now implements, and the logic that makes them
> defensible. Concepts first, as always.

---

## 1. Concepts first

### The four "lanes" of domain adaptability (the map in one minute)

When a detection model moves to a new environment ("domain"), the literature
differs on what that move should cost:

- **Lane A — learned adaptation:** collect target data, run a training loop
  (adversarial alignment). Works, but every deployment is a project.
- **Lane B — few-shot/meta-learning:** meta-train so a few target frames +
  a few gradient steps adapt the model. Cheaper, still gradient updates.
- **Lane C — zero-shot vision-only (zxVAD):** no target data at all, but the
  source model is still trained, and there is no way to *tell* it what
  normal means in the new scene.
- **Lane D — training-free VLM/LLM (us, LAVAD, VERA...):** frozen pre-trained
  models; adaptation must come from somewhere other than weights. Our claim:
  it comes from **text**.

### The controlled-experiment logic (why "frozen" is a superpower)

In any experiment you want ONE variable to change. Because *nothing* in our
pipeline trains, when we change only the scene description and the AUROC
moves, the movement is attributable to the text — there is no other moving
part it could hide in. A trained method can never make this claim cleanly:
any effect might live in the updated weights. This is the single most
important sentence in our methodology.

### Micro vs. macro AUROC (you'll see both in every table)

- **Micro:** concatenate every frame from every clip, compute one AUROC.
  Weights long clips more; the field standard for ShanghaiTech.
- **Macro:** compute AUROC per clip, average. Every clip counts equally;
  exposes methods that only work on a few easy clips.
Reporting both is cheap honesty: if they diverge, that divergence is itself
a finding.

---

## 2. The three experiments (now fully implemented)

### E1 — The grid (`da_zvad/grid.py`, from before)
Dataset × context on/off × smoothing window → the breadth table. Answers:
how much does temporal smoothing buy, per dataset?

### E2 — The context sweep (`da_zvad/context_sweep.py`) — the core experiment
Everything frozen; only the scene description changes across four variants:

| Variant | Text used |
|---|---|
| none | no scene description (prompt ensembles only) |
| generic | "a generic scene" |
| matched | the correct description of the scene |
| **mismatched** | **a wrong-domain description** (factory text on campus video) |

The predicted signature if our thesis claim is true: `matched ≥ generic ≥
none`, and `mismatched` measurably *hurts*. The mismatched cell is the
clever part — it's the negative control. Anyone can show "adding context
helps a bit"; showing that *wrong* context **damages** performance proves the
model actually *reads* the context rather than ignoring it.

### E3 — Explanations under shift (`da_zvad/explain_shift.py`)
For each detected event, the MLLM explains the same frame twice: once with
the matched scene description, once with the mismatched one. Output is a
side-by-side gallery. Per our literature review, no published work has
examined explanation behavior under domain shift — this is a small first.

### How they run
One Kaggle session (`notebooks/full_experiments_kaggle.py`): installs deps,
clones the framework, auto-finds datasets, runs E1+E2 (+E3 optionally),
caches everything (a dead session resumes, not restarts), zips the tables.

---

## 3. What to say if asked

**"How do you know the text is doing anything?"**
> "The mismatched-context cell. Same frozen pipeline, same frames — only the
> description is wrong. If AUROC drops there and rises with the matched
> description, the text is causally involved; nothing else changed."

**"Why is your comparison with VERA fair — they also verbalize?"**
> "VERA's questions are *learned offline on source data* — a training step,
> just in words. Ours is a plain description supplied at test time with zero
> optimization. Different assumption, and ours is the one that matches
> deploy-to-a-new-camera-today."

**"What if the context sweep shows no effect?"**
> "Then that's a publishable negative result: it would say CLIP's prompt
> ensembles already saturate the language channel and scene descriptions add
> nothing — which sharpens exactly where adaptation must come from instead.
> The experiment is informative in every outcome; that's why it's designed
> this way."

**"Why both micro and macro AUROC?"**
> "Micro is the comparable field standard; macro catches a method that only
> works on a few long clips. Divergence between them is itself diagnostic."

---

## 4. Self-quiz (answer before peeking)

1. Name the four lanes of domain adaptability and their target-domain cost.
2. Why can a fully-frozen pipeline attribute effects to text more cleanly
   than a trained one?
3. What is the mismatched-context variant *for*?
4. What does it mean if matched ≈ none in the context sweep?
5. Why does E3 explain the *same frame twice*?
6. Micro vs macro AUROC — what does each weight, and why report both?

<details><summary>Answers</summary>

1. A: target data + training run · B: few frames + gradient steps ·
   C: nothing at target time but a trained source model (zxVAD) ·
   D: nothing anywhere — frozen models (ours).
2. No parameters update, so a change in the only varied input (the text) is
   the only possible cause of a change in output — a controlled experiment.
3. It's the negative control: wrong-domain text should *hurt* if the model
   truly reads the context; it separates "context is used" from "context is
   ignored".
4. The scene description adds nothing beyond the prompt ensembles — an
   honest negative result that redirects where adaptation must come from.
5. To isolate the context's effect on the *explanation* while holding the
   visual evidence fixed — same logic as the sweep, applied to G4.
6. Micro weights long clips (field standard); macro weights clips equally
   (robustness check). Divergence is diagnostic.
</details>

## 5. What's next

One GPU session runs everything. The tables that come back become the
September review's results section, and the sweep's four columns become the
central figure of the thesis.
