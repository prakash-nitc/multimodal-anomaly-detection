# 03 — Domain Adaptation: a deep dive from the six survey papers

> **This note is for studying, not for submitting.** Your guide's concern is that the
> understanding isn't there. The fix is that you can explain this material in your own
> words, unprompted. Read this alongside the PDFs in `SurveyPapersStudy/` — this note
> tells you *what to look for* in each and how the six fit together, but the sentences
> you say in the next meeting have to be yours.
>
> Time to work through properly: **6–8 hours**. Suggested split: Part 1 (2h, the
> foundation — do not skip), Parts 2–4 (3h, skim each PDF alongside), Parts 5–6 (2h,
> the part that is genuinely ours).

---

## The six papers, and why he gave you *these* six

They are not six versions of the same thing. Laid out by date, they trace the entire
history of the field, and one is deliberately from another discipline:

| # | Paper | Year | Era it represents |
|---|---|---|---|
| 1 | Patel, Gopalan, Li, Chellappa — *Visual Domain Adaptation: A Survey of Recent Advances* (IEEE Signal Processing Magazine) | 2015 | **Shallow era** — before deep learning dominated |
| 2 | Wang & Deng — *Deep Visual Domain Adaptation: A Survey* (Neurocomputing) | 2018 | **Deep era** — the canonical taxonomy |
| 3 | Wilson & Cook — *A Survey of Unsupervised Deep Domain Adaptation* (ACM CSUR) | 2020 | **Unsupervised deep DA**, method components + theory |
| 4 | Liu, Yoo, Xing, Oh, El Fakhri, Kang, Woo — *Deep Unsupervised Domain Adaptation: Recent Advances and Perspectives* (APSIPA) | 2022 | **Modern UDA** + the future-directions list |
| 5 | Singhal, Walambe, Ramanna, Kotecha — *Domain Adaptation: Challenges, Methods, Datasets, Applications* (IEEE Access) | 2023 | **Broad consolidation**, shallow → deep, datasets |
| 6 | Fan, Liu, Chen — *Domain Adaptation of LLMs for Geotechnical Applications* | 2025 | **LLM era, different field entirely** |

The sixth is the tell. Geotechnical engineering has nothing to do with your project —
he included it so you'd see that **"domain adaptation" means something structurally
different in the LLM world than in the vision world.** That contrast is the intellectual
core of what he's asking you to understand, and it turns out to be where our
contribution lives.

---

# PART 1 — What domain adaptation actually is

*(If you learn only one part of this note, learn this one. Every question he asks will
bottom out here.)*

## 1.1 The setup

You train a model on a **source domain** and deploy it on a **target domain**. A
"domain" is a distribution over inputs and labels — a source distribution $P_S(x, y)$
and a target distribution $P_T(x, y)$. Standard machine learning assumes these are the
same distribution (train and test drawn i.i.d. from one pot). **Domain adaptation is
the field that exists because they usually aren't.**

Patel et al. (2015) open exactly there: *any distributional change occurring after
learning a classifier can degrade its performance at test time.* Their example is face
recognition — a model trained on frontal studio photographs degrades on surveillance
footage. Same task, same label set, different distribution.

DA is a **subfield of transfer learning**. Transfer learning is the umbrella (transfer
anything — knowledge, features, parameters — between tasks or domains); DA is the
specific case where the **task stays the same and the domain changes**.

## 1.2 The four types of shift — the most important idea in this note

Liu et al. (2022) §2, citing Kouw (2018), decompose $P(x,y) = P(y|x)P(x)$ and note that
a "domain shift" can be a shift in any factor. Four types:

| Shift type | What changes | Concrete example |
|---|---|---|
| **Covariate shift** | $p(x)$ differs; $p(y|x)$ stays the same | Same objects, different camera/lighting/weather |
| **Conditional shift** | $p(x|y)$ differs | "Street lamps: some glitter, others are dim at night" (Liu's own example) — the same class *looks* different per domain |
| **Label / target shift** | $p(y)$ differs — class proportions change | Source has 50% cars; target has 5% cars |
| **Concept shift** | $p(y|x)$ differs — **the same input carries a different label** | Liu's example: *tomato classified as a vegetable in one country, a fruit in another* |

**Now read this sentence from Liu et al. (2022) §2 carefully — it is the single most
useful sentence in all six PDFs for our project:**

> "Furthermore, the concept shift (Kouw, 2018) can arise, when classifying, for example,
> tomato as a vegetable or fruit in different countries; **it is, however, usually not a
> common problem in popular object classification or semantic segmentation tasks. As
> such, this review mainly focuses on the covariate shift alignment in UDA, as is most
> commonly studied.** The challenges of aligning the other shifts and their combinations
> are also discussed as directions for future research."

Unpack what that admits:

1. The DA field's dominant effort — the adversarial aligners, the MMD losses, the
   feature-alignment machinery in all five vision papers — is aimed at **covariate
   shift**.
2. Concept shift is **explicitly scoped out**, on the grounds that in object
   classification a cat is a cat everywhere, so $p(y|x)$ rarely changes.
3. The survey itself flags the other shifts as **open future work**.

Hold onto this. Part 5 shows why that "rare" assumption is *false* for our task — and
that is our gap, stated for us by the literature rather than invented by us.

## 1.3 The standard problem settings (vocabulary he may test)

By **label availability in the target**:
- **Supervised DA** — some labelled target data
- **Semi-supervised DA** — a few labelled + many unlabelled target samples
- **Unsupervised DA (UDA)** — unlabelled target data only ← *by far the most studied; papers 3 and 4 are entirely about this*

By **label-set relationship** (Singhal et al. 2023, Table 4):
- **Closed-set DA** — source and target share the same classes
- **Partial DA** — target classes are a subset of source classes (risk: *negative transfer*)
- **Open-set DA** — target contains classes unseen in the source
- **Universal DA** — no prior knowledge of the label-set relationship at all

By **feature space**: **homogeneous** (same feature space) vs **heterogeneous** (different
feature spaces, e.g. text ↔ image — Patel 2015 handles this by projecting both into a
common latent subspace).

By **number of steps** (Wang & Deng 2018): **one-step DA** (source and target are close
enough to bridge directly) vs **multi-step / transitive DA** (too far apart; you build
*intermediate* domains to hop through).

**Domain generalization (DG)** is the neighbouring problem, and the distinction matters:
DA gets *access to target data* (unlabelled at least); DG has **no target data at all**
and must generalize blindly. Our setting is unusual and worth naming precisely — see 5.3.

---

# PART 2 — What each paper actually contributes

Read these summaries, then skim each PDF's tables and figures. If you can say a sentence
about each paper's *distinct* contribution, you've done what he asked.

## 2.1 Patel et al. (2015) — the shallow era

The pre-deep-learning picture. Methods here manipulate features and subspaces rather
than training networks:
- **Subspace alignment**: represent each domain by a linear subspace and align them.
- **Geodesic-flow methods**: treat the source and target subspaces as points on a
  Grassmann manifold and derive *intermediate subspaces* along the geodesic path between
  them — literally interpolating a sequence of domains between source and target.
- **Dictionary / sparse-coding methods**: learn a dictionary that represents both domains.
- **Heterogeneous DA**: project source ($N$-dim) and target ($M$-dim) features into a
  shared $l$-dim latent domain via two projection matrices, so incomparable features
  become comparable.

**Why it's in your reading list:** it shows the *problem* predates deep learning. The
geodesic idea also anticipates "gradual/continuous adaptation," which reappears in 2022
as an open problem.

## 2.2 Wang & Deng (2018) — the canonical deep taxonomy

Memorize this taxonomy; it is the most-cited organizing scheme in the field. Deep
**one-step DA** splits into three families (their Table 1):

| Family | Mechanism | Sub-criteria |
|---|---|---|
| **Discrepancy-based** | Fine-tune the network to *reduce a measured domain gap* | class, statistic (e.g. MMD), architecture (e.g. BN stats), geometric criteria |
| **Adversarial-based** | A **domain discriminator** tries to tell source from target; the feature extractor learns to fool it → domain-confused features | generative / non-generative |
| **Reconstruction-based** | Use *data reconstruction* as an auxiliary task to force features to stay domain-invariant | encoder–decoder, adversarial reconstruction |

**Multi-step DA** (their Table 2): hand-crafted (a human picks the intermediate domains),
instance-based (select auxiliary data to bridge), representation-based.

## 2.3 Wilson & Cook (2020) — UDA methods and their parts

A different cut of the same space, organized by *what the method does*: domain-invariant
feature learning, domain mapping (image-to-image translation from source to target
style), normalization statistics, ensemble methods, target-discriminative methods, and
combinations.

Its distinctive value is §4 **Components** — it decomposes methods into interchangeable
parts (losses, weight sharing, training stages, multi-level alignment) — and §6
**Theory**, which covers the generalization bound: target error is bounded by source
error + a divergence term between domains + an irreducible term. That bound is *why*
everyone minimizes a domain divergence.

Its §8 research directions worth knowing: hyperparameter tuning without target labels
(you have no validation labels in UDA — a real and under-discussed problem), class
imbalance, dataset limitations, and better experimental comparability.

## 2.4 Liu et al. (2022) — modern UDA and the future list

Methods (§3): statistic divergence alignment, adversarial learning, normalization
statistics, generative domain mapping, **self-training** (pseudo-labels on the target),
**self-supervision**, low-density target boundary (decision boundaries should fall in
low-density regions), and combinations.

Applications (§4): image analysis, **medical imaging**, **video analysis**, NLP, time series.

**§5 is the part to study hardest — it is a list of open problems, i.e. a list of
candidate gaps:**
- **§5.1 Realistic shift assumptions** — the four shift types above; the admission quoted in 1.2
- **§5.2 Partial / open-set DA**
- **§5.3 Source-free DA** — driven by *data privacy*: you cannot always share the source data (medical/cross-institution). Adapt using only the pre-trained model — "white-box" (weights available) or even "black-box" (weights unavailable). Note the striking detail: *deep-inversion techniques can recover original training data from a shared model*, which is why black-box settings matter.
- **§5.4 Continuous & test-time adaptation** — real distributions drift *smoothly*, not in discrete jumps ("driving from Seattle to Boston" crossing mountains, desert, plains with no boundaries). Needs lifelong adaptation, low-cost updates, no catastrophic forgetting.
- **§5.5 Adaptation in the Foundation Model Era** — read this one twice; see 5.2 below.
- §5.6 Semi-supervised DA · §5.7 Domain generalization · §5.8 Out-of-distribution detection

## 2.5 Singhal et al. (2023) — consolidation, conditions, datasets

Broadest of the six. Two things to take from it:

**The three conditions under which DA is theoretically justified** (their §II):
1. **Covariate shift** — conditional label distribution is preserved: $P(Y_s|X_s) = Q(Y_t|X_t)$
2. **Somewhat similar distributions** — source and target must not be arbitrarily far apart (measured by $\mathcal{H}$-divergence)
3. **Joint error minimization** — DA minimizes joint source+target error

Note condition 1 **assumes concept shift away by definition**. That is the field's
foundational assumption, and it is worth saying aloud in a meeting: *classical DA theory
presumes $p(y|x)$ is stable.* The paper also notes later theoretical work (Zhao et al.)
questioning whether these conditions are sufficient in practice.

**The label-set taxonomy** (closed / partial / open-set / universal DA) from 1.3, plus
coverage of federated DA and source-free settings, and a catalogue of standard datasets
(Office-31, Office-Home, VisDA, digits, etc.).

## 2.6 Fan et al. (2025) — DA in a completely different field, and a different paradigm

Geotechnical engineering: soil, rock, foundations, site investigation. The problem: a
general LLM doesn't know specialist terminology or domain logic. Adapting it is domain
adaptation — but **nothing here resembles adversarial feature alignment.** Their four
strategies, described as a **continuum balancing cost, complexity, and performance**:

| Strategy | What it does | Cost | Weakness (their words) |
|---|---|---|---|
| **Prompt engineering** | Encode domain knowledge in the *input text* | Lowest — immediate, flexible | "often lacks depth and robustness" |
| **RAG** (retrieval-augmented generation) | Retrieve domain documents at inference and inject them | Low–medium | Depends entirely on retrieval corpus quality |
| **DAPT** (domain-adaptive pretraining) | Continue pretraining on domain corpora | High compute | Needs large-scale resources |
| **Fine-tuning** | Update weights on labelled domain tasks (LoRA/PEFT reduce this) | Highest — compute + labels | Needs substantial labelled data |

Their conclusion: **hybrid strategies are emerging as most effective** (e.g. DAPT + RAG;
fine-tuned adapters + optimized prompting). Reported limitations across the field: data
scarcity, validation difficulty, and explainability.

**Take this away:** in the foundation-model era, *writing text is a recognized domain
adaptation method* — the cheapest point on an accepted continuum. That is not a hack
someone invented to avoid training; it is a documented strategy with known trade-offs.

---

# PART 3 — The method families, consolidated

Everything in the five vision papers reduces to a handful of mechanisms:

1. **Align the statistics.** Minimize a divergence (MMD, CORAL, $\mathcal{H}$-divergence)
   between source and target feature distributions.
2. **Fool a discriminator.** Adversarial DA: features good enough that a domain
   classifier can't tell which domain they came from.
3. **Translate the data.** Domain mapping / image-to-image translation: make source
   images *look* like target images (or vice versa), then train normally.
4. **Fix the normalization.** Recompute batch-norm statistics on the target — remarkably
   cheap and often surprisingly effective.
5. **Pseudo-label and self-train.** Predict on the target, keep confident predictions,
   retrain on them.
6. **Reconstruct.** Force features to be able to rebuild inputs from both domains.
7. **Push boundaries into low-density regions.** Decision surfaces shouldn't cut through
   dense target clusters.

**Every one of these needs (a) target-domain data and (b) a training run.** That is the
common structural cost of the classical toolkit — and the axis along which the LLM-era
strategies differ.

---

# PART 4 — How DA is used in different fields

Exactly what he asked you to survey.

| Field | Typical domain shift | How DA is done there |
|---|---|---|
| **Computer vision** (all 5 vision papers) | Camera, lighting, pose, synthetic→real | Adversarial alignment, domain mapping, subspace methods |
| **Medical imaging** (Liu §4.2, §5.3) | Different scanners, hospitals, protocols | Same toolkit — **but privacy forbids sharing source data**, which is what created source-free DA |
| **Video analysis** (Liu §4.3) | Scene, viewpoint, activity distribution | Vision toolkit + temporal modelling |
| **NLP** (Wilson §7.2, Liu §4.4) | Domain vocabulary/genre shift | Feature alignment historically; now prompting, RAG, DAPT, fine-tuning |
| **Time series** (Wilson §7.3, Liu §4.5) | Sensor, subject, session | Alignment adapted for temporal structure |
| **Geotechnical / LLM** (Fan 2025) | General text → specialist domain knowledge | **Prompt engineering, RAG, DAPT, fine-tuning** |

**The single most important observation across this table** — and the thing to say in
the meeting: *the mechanism of domain adaptation changed fundamentally when foundation
models arrived.* In the vision literature, adaptation means **changing the model** to
match the target distribution. In the LLM literature, adaptation increasingly means
**changing the model's input context** while the model stays frozen. The vision surveys
(2015–2023) largely predate that shift, so their taxonomies contain no category for
"adapt by describing the domain in language."

---

# PART 5 — Where our project sits (the genuinely defensible position)

This section is ours. It is built only from claims the six papers actually make.

## 5.1 The observation that makes our project a research contribution

Recall two facts established above:

- **Fact A** (Liu 2022 §2; Singhal 2023 §II): classical DA overwhelmingly targets
  **covariate shift**, and explicitly treats **concept shift** — $p(y|x)$ changing
  between domains — as rare enough to set aside, because in object classification a
  tomato is a tomato and a cat is a cat.
- **Fact B** (about our task, not from the surveys): **in anomaly detection, "normal" is
  defined by the deployment context, not by the object.** Consider:
  - a person *running* — normal in a park, anomalous in a bank vault
  - a person *lying on the ground* — normal on a beach, anomalous in a factory aisle
  - a *vehicle* — normal on a road, anomalous on a pedestrian walkway (this is literally
    a ShanghaiTech anomaly class)

  In each case the *image content is identical* and the *label flips with the domain*.
  By the Kouw taxonomy that is **concept shift**, exactly.

**Therefore: the shift type that the domain-adaptation literature deliberately set aside
as uncommon is the shift type that dominates video anomaly detection.** The field's
scoping assumption is reasonable for classification and false for our task.

That is a genuine gap. We did not invent it; we located a stated assumption and showed
the task where it fails. That is precisely what a literature review is supposed to produce.

## 5.2 The foundation-model angle (Liu 2022 §5.5, near-verbatim)

The survey asks: if a foundation model has trained on enormous, diverse data, does it
just generalize everywhere? Its answer:

> foundation models "are robust to the covariate shift in many cases" ... but "it is
> challenging to alleviate the label shift, without access to target domain data. In
> addition, the **concept shift can also cause a problem**, even though there are
> sufficient training data."

Read that against our setting. CLIP *is* a foundation model. It has seen every lighting
condition, camera type and viewpoint on the internet — so **covariate shift, the thing
the whole classical toolkit was built to fix, is largely already handled for us.** What
remains unhandled is exactly what the survey names: concept shift. And the survey lists
this as an open direction for the foundation-model era.

So our project is not "CLIP applied to anomaly detection." It is: *given that foundation
models neutralize covariate shift, what remains is concept shift — and here is a
mechanism for it, plus a way to measure whether that mechanism works.*

## 5.3 Naming our setting precisely

Using the vocabulary from Part 1, our setting is:

- **Source-free** (Liu §5.3) — we never touch source-domain training data
- **Test-time** (Liu §5.4) — adaptation happens at inference, per deployment
- **Training-free / zero-shot** — no gradient updates anywhere, unlike every classical method
- **Concept-shift-oriented** (Liu §5.1, §5.5) — the shift we target is $p(y|x)$, not $p(x)$
- **Language-mediated** — adaptation is performed by *text*, which the LLM-era DA
  literature (Fan 2025) recognizes as the lowest-cost point on the adaptation continuum

Our M3 module — the verbalized scene description — is, stated in the field's own terms,
**prompt-engineering-based domain adaptation transplanted from the LLM literature into a
vision task, aimed at the shift type classical visual DA excluded.**

Say it that way and it stops sounding like a shortcut. It is a defensible methodological
position with citations behind every clause.

## 5.4 The honest counter-arguments (be ready for these)

He may push back. Prepare these:

**"Prompt engineering isn't real domain adaptation."**
> Fan et al. (2025) catalogue it as one of four recognized adaptation strategies, with a
> stated cost/benefit profile — lowest cost, "often lacks depth and robustness." We
> position ourselves at that known-weak end deliberately, and our experiment measures
> exactly how much it buys.

**"You're avoiding the hard part — there's no learning."**
> Correct, and that's the design. Because nothing trains, when only the text changes and
> performance moves, the effect is attributable to the text alone. A trained method
> cannot make that claim cleanly — the effect could hide in the weights. Being
> training-free is what makes the measurement valid.

**"Concept shift is a stretch — isn't this just prompting?"**
> The definition is precise: same $x$, different $p(y|x)$. A vehicle on a walkway is
> anomalous; on a road it is not. Identical pixels, flipped label, purely from domain
> context. That is concept shift by Kouw's definition, and no classical DA method
> addresses it because they align $p(x)$.

---

# PART 6 — The gaps we can genuinely fill

He asked for *a few genuine* gaps, not a list of everything. Four candidates, honestly ranked.

### ✅ Gap 1 (primary) — Concept shift is the dominant shift in VAD, and it is unstudied
**Evidence:** Liu 2022 §2 scopes concept shift out as uncommon; §5.1/§5.5 list it as open.
**Our claim:** in VAD it is not uncommon, it is the defining shift.
**What we deliver:** the argument + a mechanism (verbalized context) + a measurement.
**Realistic?** Yes — the framing costs nothing and it reframes work already built.

### ✅ Gap 2 (primary) — No protocol exists for *measuring* language-mediated adaptation
**Evidence:** the classical evaluation protocol is "source-trained accuracy vs
target-adapted accuracy," which presumes a training step. There is no standard way to
test whether a *description* adapted anything.
**Our claim:** the **matched / mismatched / generic / none** context sweep is such a
protocol — and the mismatched arm is the critical control, because it can *falsify* the
claim (wrong-domain text should actively hurt if the model is genuinely reading context).
**What we deliver:** the protocol + results across industrial and surveillance domains.
**Realistic?** Yes — already implemented in `da_zvad/context_sweep.py`.

### 🔶 Gap 3 (secondary) — Foundation-model-era DA is empirically under-characterized
**Evidence:** Liu 2022 §5.5 states it as an open direction and poses it as a question.
**Our claim:** we contribute a data point — which shifts a frozen VLM absorbs by itself,
and which it does not.
**Realistic?** Yes as a supporting contribution; too broad to be the headline.

### 🔶 Gap 4 (secondary) — Explanation behaviour under domain shift is unexamined
**Evidence:** none of the six surveys discusses explanation quality under shift; the VAD
explainability work (VERA) evaluates in-domain only.
**What we deliver:** the matched-vs-mismatched explanation gallery.
**Realistic?** Yes, but qualitative at our scale — a supporting section, not the claim.

### ❌ Explicitly *not* claiming
- Beating state-of-the-art AUROC (we won't, and shouldn't pretend to)
- A new adaptation *algorithm* (we contribute framing, mechanism, and measurement)
- Solving concept shift (we characterize and partially address it)

**Recommended headline for the paper:**
> *Concept Shift, Not Covariate Shift: Characterizing Domain Adaptation for Training-Free
> Video Anomaly Detection*

That title states a thesis, is falsifiable, and is defensible from these six papers alone.

---

# PART 7 — What you must be able to say without notes

Your guide's concern is understanding, so the deliverable is your ability to talk. Target
these six, aloud, in your own phrasing:

1. **Define domain adaptation and its relationship to transfer learning** (30 seconds).
2. **Name the four shift types with an example each** — especially concept shift.
3. **Name the three classical deep DA families** (discrepancy / adversarial /
   reconstruction) and what each does mechanically.
4. **Explain how LLM-era DA differs** — the four strategies, the cost continuum, and the
   key structural difference: change the model vs. change the input context.
5. **State our gap in three sentences** — the field targets covariate shift and sets
   concept shift aside as rare; in anomaly detection normality is context-defined so
   concept shift dominates; foundation models already absorb covariate shift, leaving
   concept shift as the live problem.
6. **Explain why the mismatched-context experiment matters** — it can prove us wrong,
   which is what makes the claim scientific rather than rhetorical.

## Self-quiz (answer aloud before checking)

1. What distinguishes domain adaptation from domain generalization?
2. Write out $P(x,y)$'s decomposition and say which factor each of the four shifts moves.
3. In adversarial DA, what are the two competing objectives?
4. Why did source-free DA emerge, and from which application field?
5. What are the three theoretical conditions Singhal et al. give for DA to work, and
   which one silently assumes concept shift away?
6. Rank the four LLM adaptation strategies by cost, and give each one's main weakness.
7. Give a VAD example of concept shift, and explain why feature alignment cannot fix it.
8. Why does a fully frozen pipeline let you attribute an effect to the text?

<details><summary>Answers</summary>

1. DA has access to target-domain data (usually unlabelled); DG has *no* target data and
   must generalize blindly.
2. $P(x,y) = P(y|x)P(x)$. Covariate shift moves $p(x)$; conditional shift moves $p(x|y)$;
   label/target shift moves $p(y)$; concept shift moves $p(y|x)$.
3. A domain discriminator tries to classify features as source vs target; the feature
   extractor tries to produce features that fool it — yielding domain-invariant features.
4. Data-privacy constraints on sharing source data across institutions — medical imaging
   (Liu §5.3). Deep-inversion attacks recovering training data from shared weights pushed
   it further toward black-box settings.
5. Covariate shift, somewhat-similar distributions, joint error minimization. The first —
   it states $P(Y_s|X_s) = Q(Y_t|X_t)$, i.e. no concept shift, by assumption.
6. Prompt engineering (lowest; lacks depth/robustness) → RAG (retrieval-corpus dependent)
   → DAPT (large compute) → fine-tuning (compute + labelled data).
7. A vehicle on a pedestrian walkway is anomalous; the same vehicle on a road is normal.
   Feature alignment matches $p(x)$ across domains, but here $p(x)$ can be identical —
   what differs is the label assigned to it, so aligning features cannot help.
8. No parameters change, so the text is the only varied input; any change in output is
   attributable to it. A controlled experiment.
</details>

---

## What to do next (in order)

1. **Read this note fully** (2h), then **skim each PDF** for the tables named above (3h).
2. **Answer the self-quiz aloud.** Where you stumble, reread that section — that's the
   real signal, not how long you spent.
3. **Ask me to run a mock session** ("grill me on DA") — I'll play your guide, ask
   follow-ups, and tell you honestly where you're thin.
4. Only then present the position in Part 5. Present it *verbally*, from understanding.
   If he sees you reason through concept shift unaided, the credibility problem
   dissolves — and it dissolves permanently, because the understanding will be real.
