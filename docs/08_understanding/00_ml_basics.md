# 00 — ML basics: the minimum you need, from zero

> Read this **before** [01_foundations.md](01_foundations.md). Every concept here appears
> somewhere in our project — each section ends with **"Where it appears for us."**
> Nothing here is filler; if it's on this page, a panelist could ask it.

---

## 1. What machine learning actually is

Classical programming: a human writes the rules (`if temperature > 40: alert`). Machine
learning: we don't know the rules, so we show a model many **examples** and it finds the
rules itself. A "model" is just a mathematical function with millions of adjustable numbers
(**parameters / weights**). "Learning" means automatically adjusting those numbers until the
function's outputs match the examples.

**Where it appears for us:** CLIP is such a function — 400M image–text examples shaped its
weights. We never adjust them; we only *use* the function. That's why our method is
"training-free."

## 2. Training vs. inference — the distinction our whole thesis sits on

- **Training:** show examples → measure how wrong the model is (the **loss**) → nudge every
  weight slightly to be less wrong → repeat millions of times. Expensive, slow, needs data.
- **Inference:** freeze the weights, feed an input, read the output. Cheap, fast, needs
  nothing but the input.

**Where it appears for us:** our entire pipeline is inference-only. When we say "frozen CLIP",
frozen = weights never change. Every competitor method that trains needs labelled data from
the target environment; we don't. That is the thesis in one sentence.

## 3. Supervised, unsupervised, and where anomaly detection sits

- **Supervised:** every example has a label ("this image = cat"). Learns input→label.
- **Unsupervised:** no labels; the model finds structure on its own (e.g. clustering).
- **Anomaly detection** is usually framed as **one-class learning**: you only have *normal*
  examples (defect-free products, ordinary CCTV footage), the model learns what normal looks
  like, and anything far from it is flagged. Getting *anomalous* examples is hard — defects
  are rare and unpredictable — which is why the field avoids supervised learning.

**Where it appears for us:** we skip even the one-class training. Our "model of normal" is a
*sentence* ("a flawless bottle"), not a trained model. That's the leap the examiners should
understand from your presentation.

## 4. Embeddings — the single most important concept in this project

A neural network can convert any input (image, sentence) into a list of numbers — a
**vector**, e.g. 768 numbers. This vector is called an **embedding**. The magic property:
the network is trained so that *similar things get nearby vectors*. "King" lands near
"queen"; a photo of a cat lands near a drawing of a cat.

Think of it as a map: every image and sentence gets GPS coordinates in a 768-dimensional
space, and distance on the map = difference in meaning.

**How do you measure "nearby"? Cosine similarity:** the angle between two vectors. Pointing
the same direction → similarity ≈ 1; unrelated → ≈ 0. Before comparing, vectors are
**normalized** (scaled to length 1) so only direction matters, not magnitude — that's the
`F.normalize` you see in our code.

**Where it appears for us:** CLIP gives *images and text coordinates on the same map*. Our
anomaly score is literally: "is this frame's point closer to the 'normal' sentences or the
'abnormal' sentences?"

## 5. Softmax — turning scores into probabilities

We end up with two similarity numbers: sim(image, normal-text) and sim(image, abnormal-text).
Raw similarities are hard to interpret. **Softmax** converts a list of numbers into
probabilities (all between 0 and 1, summing to 1), exaggerating whichever is larger.

Example: similarities (0.24, 0.31) → softmax → (0.30, 0.70) → "70% abnormal."

**Where it appears for us:** the last line of our scoring function. Our per-frame anomaly
score *is* the softmax probability of the abnormal side. (The `logit_scale` you see in code
is a temperature CLIP learned during its training — it sharpens the softmax; we just reuse it.)

## 6. Neural networks and Transformers — as much as you need

A neural network = layers of simple number-mixing operations stacked deep; each layer's
output feeds the next. A **Transformer** is a network architecture whose key trick is
**attention**: every part of the input can "look at" every other part and decide what's
relevant. It powers both modern language models and modern vision models.

A **Vision Transformer (ViT)** applies this to images: chop the image into small patches
(like words in a sentence), let patches attend to each other, output one embedding for the
whole image. "ViT-L/14" = a Large ViT using 14×14-pixel patches.

**Where it appears for us:** CLIP's image encoder is a ViT-L/14. You don't need its internals;
you need: *patches in → attention layers → one 768-number embedding out.*

## 7. How CLIP was trained (so you can answer "where does its knowledge come from?")

**Contrastive learning:** take 400M (image, caption) pairs from the web. Push each image's
embedding *toward* its own caption's embedding and *away* from every other caption in the
batch. After enough of this, the image map and the text map become the *same* map. That's
the entire trick — and it's why an unseen sentence like "a scratched metal nut" still lands
in a meaningful place.

**Where it appears for us:** this is the answer to "why does zero-shot work at all?" — we're
harvesting the alignment CLIP already learned, not teaching anything new.

## 8. Generalization and overfitting — why we keep CLIP frozen

**Overfitting:** a model memorizes its training examples instead of learning the general
rule — perfect on training data, poor on new data. It's why models are always evaluated on a
held-out **test set** they never saw. **Generalization** is the opposite: performing well on
new data.

**Where it appears for us:** fine-tuning CLIP on one dataset would specialize (and possibly
overfit) it to that domain, destroying exactly the broad knowledge we rely on for
cross-domain transfer. Keeping it frozen preserves generalization — say this when asked
"why didn't you fine-tune?"

## 9. Evaluation: why not just "accuracy"?

If 95 frames in 100 are normal, a model that says "everything is normal" scores 95%
accuracy while catching zero anomalies. Anomaly detection is **imbalanced**, so accuracy is
misleading. Instead:

- **Precision** — of the frames I flagged, how many were truly anomalous? (few false alarms)
- **Recall** — of the truly anomalous frames, how many did I flag? (few misses)
- **F1** — the balance of the two.
- **AUROC** — the headline metric; threshold-free ranking quality. Explained properly in
  [01_foundations.md](01_foundations.md) §4 — internalize that one especially.

**Where it appears for us:** `da_zvad/evaluation/metrics.py` computes exactly these, and
every results table you'll present is built from them.

## 10. Quick glossary

| Term | One line |
|---|---|
| Parameters / weights | The adjustable numbers inside a model; training = adjusting them |
| Frozen | Weights locked; the model only does inference |
| Embedding | A vector of numbers representing an input's *meaning* |
| Cosine similarity | Angle-based closeness of two embeddings (1 = same direction) |
| Normalize | Scale a vector to length 1 so only direction matters |
| Softmax | Converts raw scores into probabilities summing to 1 |
| ViT | Transformer over image patches → one image embedding |
| Contrastive learning | Train by pulling matched pairs together, pushing others apart |
| Zero-shot | Doing a task with no task-specific training examples |
| One-class learning | Learning "normal" only, flagging deviations |
| Overfitting | Memorizing training data; failing on new data |
| Inference | Running a frozen model on new input |

## 11. Self-quiz (answer before peeking)

1. What is the difference between training and inference, and which one does our entire
   pipeline use?
2. Why is anomaly detection usually *not* done with supervised learning?
3. What is an embedding, and what makes CLIP's embeddings special?
4. Why do we normalize vectors before comparing them?
5. What does softmax do to our two similarity values?
6. In one sentence, how was CLIP trained?
7. Why don't we fine-tune CLIP on our datasets?
8. Why is plain accuracy a bad metric for anomaly detection, and what do we use instead?

<details><summary>Answers</summary>

1. Training adjusts weights from examples; inference runs a frozen model. Ours is 100%
   inference — that's what "training-free" means.
2. Anomalies are rare and unpredictable, so you can't collect a labelled set of all possible
   anomalies; the field uses one-class (normal-only) formulations instead.
3. A vector encoding the input's meaning, where similar things are nearby. CLIP's are
   special because images and text share the *same* space, so image–text similarity is
   meaningful.
4. Normalizing removes magnitude so cosine similarity measures direction (meaning) only.
5. Turns (sim-to-normal, sim-to-abnormal) into two probabilities summing to 1; our score is
   the abnormal-side probability.
6. Contrastively, on ~400M web image–caption pairs: pull each image toward its own caption,
   push away from others, until images and text share one embedding space.
7. Fine-tuning would specialize it to one domain and erode the broad knowledge that makes
   cross-domain, training-free transfer possible (and it would end the "training-free" claim).
8. With ~95% normal frames, "always say normal" gets 95% accuracy while catching nothing.
   We use precision/recall/F1 and, primarily, AUROC (threshold-free ranking quality).
</details>
