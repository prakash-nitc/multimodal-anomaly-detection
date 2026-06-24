# Viva Question Bank — Final Presentation

---

# PART A: SLIDE-WISE QUESTIONS

---

## Slide 1 — Title

**Q1: What does "zero-shot" mean in your context?**
> "It means detecting anomalies without any training data from the target domain. The model uses only pre-trained knowledge from CLIP and natural language prompts — no fine-tuning, no labelled examples."

**Q2: What is "verbalized prompting"?**
> "Instead of retraining the model for a new environment, we describe the environment in natural language — like 'a dimly lit warehouse with conveyor belts'. The model adapts its understanding of normal based on this text description alone."

**Q3: Why NIT Calicut? What resources do you have?**
> "We have access to GPU clusters and can also use Kaggle T4 GPUs. Our pipeline needs only ~7.1 GB VRAM, so even a single consumer GPU is sufficient."

---

## Slide 2 — Problem & Challenge

**Q4: Why can't traditional methods like PatchCore scale?**
> "PatchCore needs to build a memory bank of normal features for each product category. If you have 50 products, you need 50 separate training runs. And if you move to a new factory, you start from scratch."

**Q5: Give a real-world example where explainability matters.**
> "In pharmaceutical manufacturing, if a pill is flagged as defective, the operator needs to know — is it a crack, a discoloration, or a contamination? Just a score of 0.92 isn't actionable. Regulatory compliance requires documented reasoning."

**Q6: What do you mean by temporal blindness?**
> "Image-based methods treat each video frame independently. If someone walks normally in frame 1, pulls out a weapon in frame 5, and attacks in frame 10 — each frame alone might look normal. Only the temporal sequence reveals the anomaly."

**Q7: Is zero-shot detection really practical? Won't it always be less accurate?**
> "Yes, there's a trade-off. PatchCore achieves 99.1% with training vs our 88.5% zero-shot. But the advantage is deployment speed — you can deploy to a new environment in minutes instead of weeks. For many real-world scenarios, 88-90% accuracy with instant deployment is more valuable than 99% after months of data collection."

---

## Slide 3 — PatchCore & WinCLIP

**Q8: Explain PatchCore in simple terms.**
> "PatchCore extracts patch-level features from normal images using a pretrained CNN and stores them in a memory bank. At test time, it compares each patch of the test image against this bank. If a patch is far from all stored normal patches — it's anomalous. Think of it as 'remembering what normal looks like' and flagging anything different."

**Q9: What is a memory bank in PatchCore? How big is it?**
> "It's a collection of representative feature vectors from normal training images. PatchCore uses coreset subsampling to keep it compact — typically a few thousand vectors per category, stored as a matrix in GPU memory."

**Q10: How does WinCLIP achieve zero-shot detection?**
> "It uses CLIP's ability to match images with text. It creates text prompts like 'a photo of a damaged bottle' and 'a photo of a good bottle', encodes them, then compares the test image embedding against both. If the image is closer to 'damaged', it's flagged as anomalous."

**Q11: What are Compositional Prompt Ensembles in WinCLIP?**
> "Instead of using one prompt, WinCLIP combines multiple state words (good, perfect, flawless, damaged, broken) with multiple templates ('a photo of a...', 'a close-up of a...'). This creates many prompts whose embeddings are averaged — reducing sensitivity to any single prompt's wording."

**Q12: Why is your 88.5% lower than WinCLIP's 91.8%?**
> "WinCLIP uses a multi-scale sliding window approach — it extracts patches at different scales and scores each locally. We deliberately skipped this to keep a simpler architecture that extends cleanly to video frames. The 3.3% gap is the cost of that design choice, which we consider worthwhile."

**Q13: What is AUROC? Why use it instead of accuracy?**
> "AUROC is the Area Under the Receiver Operating Characteristic curve. It measures performance across all possible thresholds. We use it instead of accuracy because anomaly detection datasets are highly imbalanced — 90% normal, 10% anomalous. A model that always says 'normal' gets 90% accuracy but 50% AUROC."

---

## Slide 4 — AnomalyCLIP, AnomalyGPT & MMAD

**Q14: What does "object-agnostic" mean in AnomalyCLIP?**
> "It means the learned prompts work across all product categories — bottles, screws, leather — without category-specific training. The prompts capture the general concept of 'normal vs anomalous' rather than 'what a normal bottle looks like'."

**Q15: How is AnomalyCLIP different from WinCLIP?**
> "WinCLIP uses hand-crafted text prompts that someone manually writes. AnomalyCLIP replaces these with learnable token embeddings that are optimized through training. The learned prompts generalize better to unseen products."

**Q16: How does AnomalyGPT generate explanations?**
> "It fine-tunes a multimodal LLM (based on LLaMA/Vicuna) on synthetic anomaly data. The model receives an image and can answer questions like 'Is this defective?' or 'Where is the defect?' in natural language. It uses a prompt learner to bridge visual anomaly features to the LLM."

**Q17: Why not just use GPT-4o for anomaly detection?**
> "The MMAD benchmark tested exactly this. GPT-4o achieves only moderate accuracy on industrial AD tasks. It struggles with fine-grained defects because it wasn't trained specifically for this domain. Plus, it's proprietary and expensive — we target open-source solutions."

**Q18: What are the 7 subtasks in MMAD?**
> "Anomaly Classification, Anomaly Localization, Anomaly Type Classification, Severity Assessment, Defect Description, Root Cause Analysis, and Repair Suggestion. Our work primarily targets classification, description, and localization."

---

## Slide 5 — Pivot to Video

**Q19: Why did you pivot from image to video AD?**
> "Two reasons. First, our professor's guidance to explore domain adaptability beyond static images. Second, real-world deployment uses cameras that capture video, not photos. Video has temporal context — things happening over time — that static methods completely miss."

**Q20: What is the difference between image AD and video AD?**
> "Image AD detects defects in a single static image — like a scratch on a bottle. Video AD detects anomalous events in a stream — like a person shoplifting or a machine malfunctioning over time. Video adds temporal dynamics, motion cues, and scene evolution."

---

## Slide 6 — OVVAD & Survey

**Q21: Why does OVVAD freeze CLIP? Why not fine-tune it?**
> "CLIP was trained on 2 billion image-text pairs. Fine-tuning it on a small video dataset would overfit and destroy its zero-shot generalization. By freezing CLIP and adding a tiny adapter on top, we preserve its broad knowledge while learning domain-specific temporal patterns."

**Q22: What is the temporal adapter in OVVAD?**
> "It's a lightweight module with less than 1 million parameters placed after CLIP's frozen encoder. It takes embeddings from consecutive frames and learns to capture inter-frame dynamics — like 'what changed between frame t and t+1'."

**Q23: What is Semantic Knowledge Injection in OVVAD?**
> "It uses LLM-generated text descriptions of anomaly categories and injects them into the visual pipeline. This helps the model understand what different anomaly types look like semantically, even if it hasn't seen them visually."

**Q24: What did the Gao et al. survey conclude?**
> "Three key findings: (1) Open-vocabulary VAD using VLMs is the most promising emerging direction. (2) Domain adaptation in VAD is severely underexplored. (3) Training-free methods are catching up to supervised ones. All three validate our research direction."

**Q25: Why include a survey paper in your literature review?**
> "ACM Computing Surveys has an impact factor of ~16 — it's a top-tier journal. The survey covers 400+ methods and provides a systematic taxonomy. It validates that our identified gaps — open-vocabulary, domain-adaptive, explainable VAD — are recognized by the broader community."

---

## Slide 7 — LAVAD & VERA

**Q26: How does LAVAD work without any training?**
> "Three steps: (1) BLIP-2 generates a text caption for each frame. (2) Captions from adjacent frames are grouped into snippets and sent to an LLM. (3) The LLM reasons about whether the described events are anomalous and assigns a score. No model is trained or fine-tuned."

**Q27: What is cross-modal refinement in LAVAD?**
> "BLIP-2 sometimes hallucinates — it might describe a normal scene as 'a person fighting'. Cross-modal refinement uses CLIP to check: does the original video actually match this caption? If CLIP similarity is low, the caption was likely wrong, and we downweight that score."

**Q28: How does VERA adapt to new domains?**
> "You describe the deployment environment in text — 'a dimly lit warehouse corridor with conveyor belts'. The VLM then scores anomalies relative to this context. A running person gets a low score when context says 'gym' but a high score when context says 'hospital'. No parameters change."

**Q29: Who writes the verbalized description in VERA?**
> "In deployment, the site engineer writes a 2-3 sentence description when installing the system. It's a one-time effort per site. Alternatively, a VLM could auto-generate the description from sample frames."

**Q30: What are the limitations of LAVAD?**
> "Two main ones: (1) Caption quality — BLIP-2 captions can be noisy or hallucinated. (2) Inference cost — running an LLM per snippet is computationally expensive for real-time use."

---

## Slide 8 — Comparative Analysis

**Q31: Why can't you just combine existing methods?**
> "Each method is designed as a standalone system with different architectures, inputs, and training paradigms. OVVAD needs training, LAVAD doesn't. VERA works on single frames, OVVAD needs sequences. Our contribution is designing a unified architecture that integrates the best ideas from each."

**Q32: What are your five desiderata?**
> "Zero-shot — no training data. Training-free — no parameter updates. Explainable — natural language output. Temporal — video-level reasoning. Domain-adaptive — works across environments. No existing method satisfies all five."

---

## Slide 9 — Challenges

**Q33: What is closed-vocabulary detection and why is it a problem?**
> "It means the system can only detect anomaly types it was trained on. If trained on 'fighting' and 'robbery', it can't detect 'arson' at test time. In real-world surveillance, new anomaly types appear that were never in the training set."

**Q34: Give an example of domain rigidity.**
> "A forklift moving is normal in a warehouse but anomalous on a sidewalk. A person running is normal in a sports arena but anomalous in a hospital corridor. Current methods have a fixed definition of 'normal' that can't adapt to these context changes."

---

# PART B: PAPER-SPECIFIC DEEP QUESTIONS

---

## PatchCore (CVPR 2022)

**Q35: What backbone does PatchCore use?**
> "A pretrained ImageNet CNN — typically a Wide ResNet-50. It extracts features from intermediate layers, not the final classification layer, to get richer patch-level representations."

**Q36: What is coreset subsampling?**
> "The full memory bank would be too large. Coreset subsampling selects a representative subset of patches that maximally covers the feature space. This keeps the memory bank small while maintaining coverage of all normal patterns."

**Q37: How does PatchCore detect anomalies at test time?**
> "For each patch in the test image, it computes the nearest-neighbor distance to the memory bank. The maximum distance across all patches becomes the image-level anomaly score. High distance = the patch looks different from anything in the normal memory bank."

---

## WinCLIP (CVPR 2023)

**Q38: What is CLIP's contrastive learning objective?**
> "CLIP is trained on 400M image-text pairs. For each pair, it learns to maximize cosine similarity between matching image-text embeddings and minimize it for non-matching pairs. This creates a shared embedding space where images and text can be directly compared."

**Q39: Why does WinCLIP use a sliding window?**
> "CLIP is designed for whole-image understanding. Industrial defects are often localized — a tiny scratch on a large product. The sliding window extracts overlapping patches at multiple scales, letting CLIP focus on local regions where defects might exist."

**Q40: Could you improve WinCLIP's prompts?**
> "Yes — that's part of our contribution. WinCLIP uses manually designed prompts. Our systematic prompt engineering study explores which combinations of state words, templates, and ensemble sizes work best, potentially closing the performance gap without adding complexity."

---

## AnomalyCLIP (ICLR 2024)

**Q41: What is prompt learning?**
> "Instead of writing text prompts manually, you initialize learnable token embeddings and optimize them through backpropagation. The tokens aren't real words — they're continuous vectors in CLIP's embedding space that the model learns to interpret as 'normal' or 'anomalous'."

**Q42: Why is AnomalyCLIP not truly zero-shot?**
> "It requires a training phase to learn the prompt embeddings. Even though the prompts generalize to unseen categories, you still need a dataset to train them initially. Our approach uses hand-crafted prompt ensembles — truly zero training."

---

## AnomalyGPT (AAAI 2024)

**Q43: What is NSA (Novelty Synthesis for Anomaly)?**
> "It's a technique to generate synthetic anomaly images by cutting patches from one image and pasting them onto another. This creates training data for AnomalyGPT since real anomaly data is scarce."

**Q44: What does "threshold-free" mean in AnomalyGPT?**
> "Traditional methods output a score and you need to manually set a threshold — above 0.5 is anomalous. AnomalyGPT's LLM directly answers 'yes' or 'no' in natural language, eliminating the need for threshold tuning."

---

## OVVAD (CVPR 2024)

**Q45: What does "open-vocabulary" mean in OVVAD?**
> "It can detect anomaly types never seen during training. If trained on 'fighting' and 'robbery', it can still detect 'explosion' at test time by leveraging CLIP's language understanding to generalize beyond the training vocabulary."

**Q46: What is anomaly synthesis in OVVAD?**
> "During training, OVVAD creates pseudo-novel anomaly videos by mixing visual features from known categories. This exposes the model to unseen patterns, improving its ability to generalize to truly novel anomaly types at test time."

---

## LAVAD (CVPR 2024)

**Q47: Why use an LLM instead of a learned classifier?**
> "A learned classifier outputs a number with no reasoning. An LLM can reason temporally: 'Frame 1 shows a person walking, frame 5 shows them running, frame 10 shows an attack — this is a developing assault.' The LLM understands narrative sequences from its pretraining."

**Q48: What captioning model does LAVAD use?**
> "BLIP-2 — a pretrained vision-language model that generates text descriptions of images. It's used off-the-shelf, completely frozen, no fine-tuning."

---

## VERA (CVPR 2025)

**Q49: How is VERA's prompting different from regular CLIP prompts?**
> "Regular prompts describe what to look for — 'a damaged bottle'. VERA prompts describe where you're looking — 'a dimly lit warehouse'. This shifts the entire baseline of what's normal. It's about context-dependent normality, not object-specific detection."

**Q50: Can VERA handle day vs night changes?**
> "Yes. You can have multiple verbalized contexts and switch between them — 'Daytime: busy parking lot' vs 'Nighttime: empty lot, no pedestrians expected'. The system applies the right context at the right time."

---

# PART C: KAGGLE PROJECT QUESTIONS (THEY WILL GRILL YOU HERE)

---

## Project 1: CLIP Baseline on MVTec AD

**Q51: Why did you choose ViT-L/14 over ViT-B/16?**
> "ViT-L/14 is larger (307M vs 86M parameters) and produces richer 768-dimensional embeddings. Since we're not fine-tuning, we want the strongest possible frozen encoder. The extra compute is minimal since it's inference only."

**Q52: Why LAION-2B weights instead of OpenAI's original?**
> "LAION-2B was trained on 2 billion image-text pairs — 5× more than OpenAI's 400M. The open-source checkpoint from OpenCLIP achieves better zero-shot performance. It's also fully open-source, aligning with our reproducibility goals."

**Q53: Walk me through how a single image is scored.**
> "Step 1: Preprocess the image to 224×224 and normalize. Step 2: Pass through CLIP's visual encoder to get a 768-d embedding. Step 3: Compute cosine similarity against the pre-encoded normal and abnormal text embeddings. Step 4: Multiply by CLIP's learned temperature (logit_scale). Step 5: Apply softmax. Step 6: P(abnormal) is the anomaly score."

**Q54: What is logit_scale in CLIP?**
> "It's a learned temperature parameter that sharpens the softmax distribution. Raw cosine similarities are between -1 and 1, which gives weak softmax outputs. The logit_scale (typically ~4.6) amplifies the difference, making the model more confident in its predictions."

**Q55: Why 24 prompts? Why not 5 or 100?**
> "24 is the cross-product of 6 state words × 4 templates. Too few prompts make the representation sensitive to specific wording. Too many add redundancy without improving coverage. 24 provides good diversity while keeping encoding fast. WinCLIP uses a similar ensemble size."

**Q56: Why do textures score 99.5% but objects only 83.1%?**
> "Texture defects — scratches on leather, holes in carpet — change the global visual appearance. CLIP's single image embedding captures this easily. Object defects — a bent transistor pin, a tiny crack on a pill — are localized to maybe 2% of pixels. The global embedding averages them away."

**Q57: Transistor is your worst at 72.7%. Why?**
> "Transistor defects are extremely subtle — bent leads, misplaced components — that even humans need magnification to spot. CLIP was trained on internet images, not micro-electronics. The visual difference between a normal and defective transistor is minimal at 224×224 resolution."

**Q58: How would you improve the object category scores?**
> "Three approaches: (1) Add WinCLIP's sliding window for spatial localization. (2) Use higher resolution inputs — maybe 336×336 or 448×448. (3) Use patch-level features from intermediate CLIP layers instead of the final CLS token."

**Q59: 135 seconds for 1,725 images — is that fast enough?**
> "That's about 78ms per image or ~13 FPS. For batch quality inspection, this is very practical. For real-time video at 30 FPS, we'd need optimization — batching, TensorRT compilation, or using a lighter CLIP variant."

**Q60: What if you used different prompts? Would results change?**
> "Yes — prompt design is critical. That's why systematic prompt engineering is one of our contributions. We tested 6 state words and 4 templates. Different combinations shift results by 2-5%. The ensemble approach makes it more robust than any single prompt."

**Q61: Why F1-score is 90.7% but AUROC is 88.5%? Shouldn't they be similar?**
> "AUROC measures performance across all thresholds. F1 is computed at the optimal threshold. A model can have high F1 at one specific threshold even if its overall ranking (AUROC) is lower. The F1 is high because at the optimal threshold, both precision and recall are balanced."

---

## Project 2: Video Extension (Pseudo-Video)

**Q62: Why pseudo-video instead of real UCF-Crime videos?**
> "UCF-Crime is 128 GB — impractical to download in a Kaggle session. More importantly, using MVTec images gives us a controlled experiment where we know the exact ground truth and can compare directly with Project 1's results."

**Q63: How did you construct the pseudo-video?**
> "I took real MVTec bottle test images: 10 normal (good) bottles as frames 0-9, 10 anomalous (broken/contaminated) bottles as frames 10-19, and 10 normal bottles again as frames 20-29. This simulates a conveyor belt where a batch of defective products passes through."

**Q64: The score gap is 0.587. Is that good enough?**
> "Very good. Normal frames average 0.307, anomalous frames average 0.894. A simple threshold at 0.5 would perfectly separate them. The gap is wide enough that even with noise, temporal smoothing, or domain shift, the signal should remain detectable."

**Q65: Why do scores within the anomalous segment vary (0.60 to 1.0)?**
> "Different defect types produce different scores. A 'broken_large' defect (obvious crack) scores ~0.98 while a 'contamination' (subtle surface stain) scores ~0.60. The variation reflects defect severity — not model noise."

**Q66: You score each frame independently. What's wrong with that?**
> "Three problems: (1) No temporal context — frame 15 doesn't know frames 10-14 were also anomalous. (2) Score fluctuations — a contamination frame might dip below threshold and get missed. (3) No explanations — we get a number but not 'why'. These are exactly the limitations DA-ZVAD's temporal adapter and LLM reasoning will address."

**Q67: What if the pseudo-video had gradual transitions instead of sharp ones?**
> "That's a great point and a limitation of this experiment. In real surveillance, anomalies develop gradually. Per-frame independent scoring would still catch the peak anomaly but might miss the build-up. This is exactly why temporal reasoning — looking at score trends over time — is essential."

**Q68: How does this experiment connect to Project 1?**
> "Project 1 validates CLIP as a strong zero-shot anomaly detector on images (88.5% AUROC). Project 2 shows the same model's discriminative ability transfers to per-frame video scoring (0.587 gap). Together, they prove CLIP is a viable foundation for video AD — but needs temporal modeling on top."

**Q69: What would change with real surveillance video?**
> "Three things: (1) Image quality — surveillance cameras have lower resolution, motion blur, and compression artifacts. (2) Prompt design — industrial prompts won't work for surveillance; you need domain-specific prompts. (3) Temporal dynamics — real anomalies develop over seconds, not sharp frame-to-frame transitions."

**Q70: ~5ms per frame — could this work in real-time?**
> "At 5ms per frame, that's 200 FPS — well above real-time requirements. But this is just the CLIP encoding step. The full DA-ZVAD pipeline with LLM reasoning will be slower. We'd need to optimize — perhaps scoring every 5th frame, or batching LLM calls."

---

# PART D: RESULTS & ARCHITECTURE QUESTIONS (Slides 10-15)

---

**Q71: Explain your problem statement in one sentence.**
> "We're building a video anomaly detection system that needs no training data, adapts to new environments through text descriptions, and explains every detection in natural language."

**Q72: Why five objectives? Aren't you being too ambitious?**
> "The five objectives are interconnected — they all emerge from one pipeline. The frozen VLM gives zero-shot + open-vocabulary. Verbalized context gives domain adaptation. The LLM head gives explanations. Temporal adapter gives video-level reasoning. It's one architecture, not five separate projects."

**Q73: Total VRAM is 7.1 GB. Break it down.**
> "CLIP ViT-L/14 frozen: ~1.2 GB. LLaVA-7B quantized to 4-bit: ~5.5 GB. Temporal adapter: ~0.1 GB. Frame buffer (16 frames): ~0.3 GB. Total: ~7.1 GB. Fits a single 8 GB GPU."

**Q74: Why LLaVA and not GPT-4o?**
> "Three reasons: (1) Open-source and reproducible. (2) Can run locally without API costs. (3) 4-bit quantization fits in our GPU budget. GPT-4o is proprietary, expensive per API call, and the MMAD benchmark showed it's only moderately accurate anyway."

**Q75: Your roadmap says 15 papers. Earlier it said 9. Which is it?**
> "We deeply studied 9 papers with full architectural analysis. The remaining 6 were surveyed at a higher level — reading abstracts, intros, and results. Total exposure is 15 papers across 5 categories."

**Q76: What are your evaluation metrics?**
> "Primary: frame-level AUROC. Secondary: F1-score, Average Precision (AP). For explainability, we'll use qualitative evaluation of generated explanations — correctness, specificity, and relevance."

**Q77: What if DA-ZVAD doesn't beat existing methods?**
> "The primary contribution isn't necessarily beating AUROC numbers — it's achieving competitive performance while satisfying all five desiderata simultaneously. Even if AUROC is slightly lower than OVVAD, providing explanations and domain adaptation at zero training cost is a significant practical advantage."

**Q78: What datasets will you use for cross-domain evaluation?**
> "UCF-Crime (1,900 videos, surveillance), XD-Violence (4,754 videos, multi-domain), and MVTec AD (5,354 images, industrial). Training on none, evaluating on all — that's the zero-shot cross-domain test."

---

# 🧠 LAST-MINUTE CHEAT SHEET

| Topic | Number to Remember |
|---|---|
| PatchCore AUROC | 99.1% (needs training) |
| WinCLIP AUROC | 91.8% (zero-shot) |
| Our CLIP baseline | 88.5% (zero-shot, no sliding window) |
| Texture categories | 99.5% mean |
| Object categories | 83.1% mean |
| Worst category | Transistor 72.7% |
| Video normal score | 0.307 |
| Video anomaly score | 0.894 |
| Score gap | 0.587 |
| Inference speed | ~5ms/frame on T4 |
| Total VRAM needed | ~7.1 GB |
| Papers studied | 9 deep + 6 surveyed = 15 total |
| CLIP model | ViT-L/14, LAION-2B, 768-d embeddings |
| Prompts per category | 24 normal + 24 abnormal |
| Total MVTec test images | 1,725 |
| MVTec categories | 15 (5 texture + 10 object) |
| OVVAD AUC | 86.40% on UCF-Crime |
