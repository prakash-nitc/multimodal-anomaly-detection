# Domain Adaptability in Video Anomaly Detection — Literature Map, Models, and Gaps

> Prepared following the guide's directive (July 2026): survey the domain-adaptation
> literature for VAD, identify which models each line of work uses, locate the gaps,
> and position our model choice. Sources: papers up to March 2026 (arXiv IDs given).

---

## 1. The problem, precisely

A VAD model learns "normal" from a source environment. Deployed in a new scene —
different camera, lighting, activity patterns, or even a different *domain* (industrial
inspection vs. street surveillance) — its notion of normal breaks. **Domain adaptability**
asks: what does it cost to move the model? The literature answers in four distinct ways,
with decreasing target-domain cost:

| Lane | Target-domain cost | Representative papers |
|---|---|---|
| A. Learned / adversarial DA | target data + training run | Ada-VAD, DA-VAD, Industrial-AdaVAD |
| B. Meta-learning / few-shot scene adaptation | few target frames + gradient steps | Lu et al. (ECCV'20), Anomaly Crossing, adversarial-diffusion FS |
| C. Zero-shot cross-domain (vision-only) | none (but source training needed) | zxVAD (WACV'23) |
| D. Training-free VLM/LLM-based | none — frozen pre-trained models | LAVAD, VERA, AnyAnomaly, MM-VAD (2026) |

---

## 2. Lane A — Learned / adversarial domain adaptation

**Ada-VAD** (SDM 2024, [doi:10.1137/1.9781611978032.73]) — few-shot cross-domain VAD.
Pretrains a **domain-invariant 3D-CNN/autoencoder-style predictor** with synthesized
abnormal samples and self-supervised prediction; adapts to a target domain with a few
target frames + adversarial alignment. *Model: 3D conv predictive network (no VLM).*

**DA-VAD** (2025) — weakly-supervised; treats video-level and clip-level features as
source/target and inserts **domain aligner** modules to reduce scene overfitting.
*Model: I3D-style clip features + alignment heads.*

**Industrial-AdaVAD** (Mathematics, 2025) — edge-deployed industrial VAD with a
**multilayer adversarial DA** mechanism for cross-scene transfer. *Model: lightweight
CNN backbone + adversarial discriminators.*

**Shared limitation:** every deployment needs target data and a training loop; the
adapted model is scene-specific; none produces explanations.

## 3. Lane B — Meta-learning / few-shot scene adaptation

**Lu et al., "Few-shot Scene-adaptive Anomaly Detection"** (ECCV 2020) — the classic:
MAML-style meta-learning over many scenes so a **future-frame-prediction r-GAN** adapts
to a new scene with a handful of frames.

**Anomaly Crossing** (arXiv 2112.06320) — reframes cross-domain VAD as **cross-domain
few-shot learning**; representation fusion between source and target.

**Adversarial diffusion for few-shot scene-adaptive VAD** (Neurocomputing 2024) —
diffusion-based generator adapted per scene with few samples.

**Shared limitation:** still gradient updates per scene; backbone models are pre-VLM
(GANs/AEs/3D-CNNs) with no language interface and no explanations.

## 4. Lane C — Zero-shot cross-domain, vision-only

**zxVAD** (WACV 2023, arXiv 2212.07010) — the closest *non-VLM* relative of our goal:
cross-domain VAD **without any target adaptation**. A **Normalcy Classifier** contrasts
normal features against pseudo-anomalies synthesized by pasting foreign objects via an
**untrained CNN**; future-frame prediction backbone. Transfers UCF/ShanghaiTech-style
domains zero-shot.

**Limitations:** the source model is still *trained*; "normal" is defined only by source
visual statistics — there is no mechanism to *tell* the system what normal means in the
target scene (no language channel); no explanations; evaluated within surveillance
domains only.

## 5. Lane D — Training-free VLM/LLM methods (2024 → 2026, the active frontier)

**LAVAD** (CVPR 2024, arXiv 2404.01014) — fully training-free: **BLIP-2** captions
frames → **Llama-family LLM** scores temporal windows of captions → **ImageBind**-style
cross-modal similarity cleans noisy captions. Strong on UCF-Crime/XD-Violence.
*Cost: three large frozen models chained; no explicit domain-context mechanism.*

**VERA** (CVPR 2025, arXiv 2412.01095) — keeps the VLM **frozen (InternVL2)** and makes
only the *verbal* part learnable: it optimizes a set of **guiding questions** via
learner/optimizer VLM interactions on a source dataset. Explainable by construction.
*Key nuance for us: the verbalized questions are **learned offline on source data** —
adaptation is not test-time; cross-domain transfer of the learned questions is not the
studied problem.*

**AnyAnomaly** (arXiv 2503.04504) — **customizable VAD**: the user's *text description
of what counts as abnormal* drives a zero-shot LVLM (segment-level VQA + context-aware
scoring). No training. SOTA-competitive on UBnormal/UCF-Crime. *The "describe it in
text" philosophy, applied to the anomaly side.*

**MM-VAD** (arXiv 2603.13374, 2026) — training-free VAD as **adaptive test-time
inference**: caption-derived scene representations in hyperbolic space + adaptive QA
over a frozen LLM. Confirms the field is moving from fixed pipelines to test-time
reasoning.

**Also in this wave:** MoniTor (online training-free VAD, 2025); "No Need for Real
Anomaly" (MLLM zero-shot VAD, arXiv 2602.19248); "Sparse Reasoning is Enough"
(arXiv 2511.17094); latent-manifold steering in frozen MLLMs (arXiv 2602.24021);
language-guided open-world VAD under weak supervision (arXiv 2503.13160). Weakly
supervised CLIP-adapters (VadCLIP, HeadCLIP with learnable prompt tokens) sit between
lanes A and D — CLIP-based but trained.

**Shared blind spot of Lane D:** every one of these is developed and evaluated
**within surveillance video only**, and none isolates *domain adaptation* as the
measured variable: there is no study that (i) moves a single training-free pipeline
across **industrial ↔ surveillance** domains, (ii) injects the target domain as a
**test-time text description** (rather than learned prompts/questions), and
(iii) ablates **which component carries the transfer** (visual prior vs. prompt wording
vs. scene context vs. temporal aggregation).

---

## 6. Gap analysis → our positioning

| Gap | Who comes closest | What's missing |
|---|---|---|
| G1. Cross-domain evaluation industrial ↔ surveillance | zxVAD (surveillance-only) | no VLM method evaluated across *both* domain families with one frozen pipeline |
| G2. Test-time, source-free domain injection via language | VERA (learned questions), AnyAnomaly (user text for *anomalies*) | describing the *scene/normality* in text at test time, zero optimization, and measuring its effect under domain shift |
| G3. Component-wise transfer ablation | — | nobody quantifies which module (prompts / context / temporal / backbone) carries or breaks cross-domain performance |
| G4. DA + explanations together | VERA (explainable, in-domain) | explanation quality under domain shift is unstudied |
| G5. Compute floor | LAVAD (3 large models) | a minimal single-GPU training-free stack for the same task |

**DA-ZVAD targets G1–G5 jointly:** one frozen pipeline (no training anywhere), domain
adaptation as a **test-time verbalized scene description**, evaluated on MVTec
(industrial) + ShanghaiTech/Avenue (surveillance), with a full module-toggle ablation
and an explanation stage — on a single consumer GPU.

## 7. Models: what they use vs. what we use

| Work | Backbone(s) | Trained? | Language channel | Explains? |
|---|---|---|---|---|
| Ada-VAD / DA-VAD / Industrial-AdaVAD | 3D-CNN / I3D / AE + adversarial heads | yes (source + target) | none | no |
| Lu et al. ECCV'20 | r-GAN (frame prediction) + MAML | yes (meta + per-scene) | none | no |
| zxVAD | frame-prediction CNN + untrained-CNN synthesis | yes (source only) | none | no |
| LAVAD | BLIP-2 + Llama LLM + ImageBind | no | captions (implicit) | partial |
| VERA | InternVL2 (frozen) | verbal params learned | learned guiding questions | yes |
| AnyAnomaly | open LVLM (VQA) | no | user-defined anomaly text | partial |
| MM-VAD (2026) | captioner + frozen LLM (hyperbolic reps) | no | adaptive QA | partial |
| **DA-ZVAD (ours)** | **frozen OpenCLIP ViT-L/14 + LLaVA-1.5-7B (4-bit)** | **no — nothing, anywhere** | **test-time verbalized scene context (M3)** | **yes (M4)** |

**Why this model choice:** CLIP ViT-L/14 is the common denominator of the zero-shot AD
literature (WinCLIP, AnomalyCLIP lineage) → our cross-domain numbers stay comparable;
LLaVA-1.5-7B in 4-bit is the lightest capable open MLLM (~5.5 GB) → the whole stack
fits one T4/RTX-3060-class GPU, in deliberate contrast to LAVAD's multi-model chain;
and keeping *everything* frozen makes the domain-adaptation claim clean: any adaptation
effect is attributable to the text, not to hidden parameter updates.

## 8. References (key)

- Aich et al., *Cross-Domain VAD without Target Domain Adaptation*, WACV 2023. arXiv:2212.07010
- Ada-VAD, SDM 2024. doi:10.1137/1.9781611978032.73
- Lu et al., *Few-shot Scene-adaptive Anomaly Detection*, ECCV 2020
- Zanella et al., *LAVAD*, CVPR 2024. arXiv:2404.01014
- Ye et al., *VERA*, CVPR 2025. arXiv:2412.01095
- *AnyAnomaly*, 2025. arXiv:2503.04504
- *MM-VAD: Geometry-Aware Semantic Reasoning for Training-Free VAD*, 2026. arXiv:2603.13374
- *Anomaly Crossing*, arXiv:2112.06320 · *Industrial-AdaVAD*, Mathematics 2025
- Related 2025–26 wave: arXiv:2602.19248, 2511.17094, 2602.24021, 2503.13160
