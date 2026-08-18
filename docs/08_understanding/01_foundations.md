# 00 — Foundations: the idea, CLIP, and the DA-ZVAD framework

> Read this once, slowly. Everything else in the project builds on these ideas.

---

## 1. What we built

A modular Python framework (`da_zvad/`) that takes video frames and outputs, for each
frame, **an anomaly score plus (eventually) a sentence explaining why** — with **no
training**. Before that, we validated the core idea on still images: a CLIP-based detector
that reaches **88.5% AUROC on the MVTec industrial defect benchmark with zero training
data**.

---

## 2. Why — the one idea underneath everything

Classical anomaly detection learns "normal" from hundreds of normal examples, then flags
deviations. That means: collect data, train, and **retrain every time the environment
changes**. Our approach replaces the learned model of "normal" with **language**.

CLIP (OpenAI, 2021) was trained on ~400 million image–caption pairs to place images and
text in the **same embedding space**: an image of a cracked bottle lands *near* the sentence
"a damaged bottle" and *far* from "a flawless bottle". So we can classify without training:

> Score = how much more the image resembles the sentence *"something abnormal"* than the
> sentence *"a normal scene"*.

**The language prior is the classifier.** No training data, no training loop. To move the
detector from a factory to a subway station, you change the *sentences*, not the model.
That single property is the thesis: **training-free, domain-adaptive, explainable**.

---

## 3. How it works — the four modules

Think of the pipeline as four stations a frame passes through:

**M1 — CLIP encoder** (`da_zvad/encoders/`). The frame is embedded and compared with two
*prompt ensembles* — a set of "normal" sentences and a set of "abnormal" sentences (sets,
because averaging several phrasings is more stable than one sentence). A softmax over the
two similarities gives P(abnormal) ∈ [0,1]. That's the raw per-frame anomaly score.

**M2 — Temporal aggregation** (`da_zvad/temporal/`). Raw per-frame scores are noisy — one
frame can randomly spike. A **moving average over a window of frames** smooths the sequence:
real events (many consecutive high frames) survive; isolated spikes get damped. This is how
"video" enters the picture without any training. Our demo figure shows exactly this: a false
spike at one frame is suppressed while the true event stays above threshold.

**M3 — Verbalized context** (`da_zvad/context/`). "A person running" is normal in a park,
alarming in a bank. M3 injects a plain-text scene description ("a quiet campus walkway")
into both prompt ensembles, so "normal" is defined *for that place*. This is the
domain-adaptation mechanism — and it costs zero parameter updates.

**M4 — LLM reasoning** (`da_zvad/reasoning/`). For frames the detector flags, a multimodal
LLM (e.g. LLaVA) is asked to *explain* the anomaly ("a person is fighting near the
entrance"). Currently an interface with a clearly-marked stub; the real model is the next
implementation step.

**The design trick worth remembering:** every module is an on/off switch in one config
object (`da_zvad/config.py`). Running the pipeline with different switch combinations *is*
the ablation study — the experiment that measures what each module contributes — with no
code changes.

---

## 4. Key terms (own these)

| Term | Plain meaning |
|---|---|
| **Zero-shot** | The model performs a task it was never explicitly trained for — here, spotting anomalies it has never seen, using language descriptions. |
| **Training-free** | No parameter is updated anywhere in our pipeline. Frozen models + text only. |
| **VLM / CLIP** | A model embedding images and text in one space, so image–text similarity is meaningful. |
| **Prompt ensemble** | Several phrasings of the same concept, averaged, for a stabler text embedding. |
| **AUROC** | Area under the ROC curve. Probability that a random anomalous sample scores higher than a random normal one. 0.5 = coin flip, 1.0 = perfect. It's threshold-independent — it measures *ranking* quality. |
| **Frame-level AUROC** | AUROC computed over every video frame's score vs. its ground-truth label — the standard metric on ShanghaiTech / Avenue. |
| **Ablation study** | Remove/disable one component at a time and re-measure, to prove each part earns its place. |
| **Frozen encoder** | The pretrained model's weights are never touched — preserves its general knowledge and avoids overfitting to one domain. |

---

## 5. What to say if asked

**"Why does CLIP work for defects it never saw?"**
> "CLIP learned a general association between images and language from 400M pairs. 'Damaged',
> 'cracked', 'flawless' are concepts it already grounds visually — we're reusing that prior,
> not teaching it anything new."

**"What did the 88.5% on MVTec tell you?"**
> "Two things. Zero-shot CLIP is within ~4 points of a *trained* one-class SVM — and the gap
> is structured: CLIP wins on texture categories (~99%) where defects change global
> appearance, and loses on small localized defects (~83%) that a whole-image embedding can't
> resolve. That split is what motivates the video and explanation work."

**"Each frame is scored independently — isn't that wrong for video?"**
> "Yes, and that's deliberate as a baseline. Temporal smoothing (M2) is the training-free
> correction; a learned temporal adapter is the planned extension. We quantify exactly how
> much temporal context adds — that's one axis of the ablation."

**"Where does the domain adaptation actually happen?"**
> "In text. M3 rewrites what 'normal' means using a scene description. No adversarial
> alignment, no fine-tuning — it's test-time, source-free adaptation through language, and
> the cross-domain experiments measure how far that gets us."

**"What's implemented versus planned?"**
> "M1, M2, M3, the dataset adapters, and the evaluation harness are implemented and tested.
> M4 is an interface with a stub — the pipeline runs end-to-end today, and the real MLLM
> explanation model is the next step."

---

## 6. Self-quiz (answer before peeking)

1. In one sentence: how can CLIP classify a defect it was never trained on?
2. What exactly does an AUROC of 0.885 mean? Why is it threshold-independent?
3. Why do we use *ensembles* of prompts instead of one sentence?
4. Why did CLIP score ~99% on textures but ~83% on objects in MVTec?
5. What problem does temporal smoothing (M2) solve, and why does it need no training?
6. Where—precisely—does domain adaptation happen in this pipeline?
7. Which modules are implemented and which is a stub, right now?
8. What is the difference between "zero-shot" and "training-free" as we use them?

<details><summary>Answers</summary>

1. CLIP embeds images and text in one space, so similarity to the sentence "a damaged X"
   vs. "a flawless X" is itself a classifier — the language prior replaces training.
2. Pick one random anomalous and one random normal sample: 88.5% chance the anomalous one
   gets the higher score. It evaluates the *ranking* of scores, so no threshold is involved.
3. One phrasing is noisy; averaging several phrasings of the same concept gives a stabler
   text embedding (less sensitivity to word choice).
4. Texture defects change the image's *global* appearance, which a whole-image embedding
   captures; object defects are small and local, below the resolution of one global vector.
5. Raw per-frame scores fluctuate; isolated spikes cause false alarms. A moving average
   keeps multi-frame events and damps single-frame noise — it's just arithmetic on scores,
   no parameters to learn.
6. In the text (M3): the scene description is injected into both prompt ensembles, so
   "normal" is re-defined for that environment — zero parameter updates.
7. M1 (CLIP encoder), M2 (temporal), M3 (context), datasets, and evaluation are implemented;
   M4 (LLM reasoning) is an interface with a clearly-marked stub.
8. Zero-shot = handles categories/anomalies never seen in training. Training-free = we never
   update any weights anywhere. Ours is both; a fine-tuned open-vocabulary model would be
   zero-shot but not training-free.
</details>

## 7. What's next

1. **Phase-0 probe:** measure the zero-shot detector on real ShanghaiTech frames
   (frame-level AUROC) — this number decides how we weight detection vs. explanation in the
   story.
2. **Real M4:** replace the stub with an actual MLLM and produce the first explanation
   gallery.
3. **First ablation grid** on real data → the core of the September review package.
