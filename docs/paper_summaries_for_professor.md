# Research Papers Studied — Summary for Professor Meeting
**Topic:** Multimodal Anomaly Detection in Industrial Quality Inspection using Vision-Language Models  
**Date:** March 2, 2026

---

## Paper 1: WinCLIP — Zero-/Few-Shot Anomaly Classification and Segmentation
**Authors:** Jeong et al. | **Venue:** CVPR 2023  
**Code:** [github.com/caoyunkang/WinClip](https://github.com/caoyunkang/WinClip)

### Summary
WinCLIP is the first major work applying OpenAI's CLIP vision-language model to industrial anomaly detection in a **zero-shot** setting (no training on product images). It introduces two key techniques: **(1) Compositional Prompt Ensembles (CPE)** — combining multiple text templates (e.g., "a photo of a [state] [object]") with state words (flawless, damaged, broken, etc.) to create robust text representations for normal and anomalous classes; and **(2) a multi-scale sliding window approach** — since CLIP is designed for whole-image understanding, WinCLIP extracts local patch-level features using overlapping windows at multiple scales, enabling pixel-level anomaly localization.

### Key Results
| Metric | MVTec AD |
|--------|----------|
| Image-level AUROC (zero-shot) | **91.8%** |
| Pixel-level AUROC (zero-shot) | **85.1%** |

### Limitations & Relevance to My Work
- Relies on **hand-crafted prompts** — no systematic study of which prompts work best
- Provides anomaly **scores/heatmaps** but cannot **explain** what the defect is or why it's anomalous
- These two limitations directly motivate my research: systematic prompt engineering + adding natural language explanations via MLLMs

---

## Paper 2: AnomalyGPT — Detecting Industrial Anomalies Using Large Vision-Language Models
**Authors:** Gu et al. | **Venue:** AAAI 2024  
**Code:** [github.com/CASIA-IVA-Lab/AnomalyGPT](https://github.com/CASIA-IVA-Lab/AnomalyGPT)

### Summary
AnomalyGPT is the first paper to use a **Large Vision-Language Model (LVLM)** for industrial anomaly detection. It fine-tunes a multimodal LLM (based on Vicuna/LLaMA) on synthetically generated anomaly data using Novelty Synthesis for Anomaly (NSA). The architecture consists of: **(1) a frozen image encoder** that extracts multi-layer visual features, **(2) an image decoder** that compares patch features against "normal"/"abnormal" text prompts for localization, and **(3) a prompt learner** that bridges visual anomaly information to the LLM. The model is **threshold-free** — instead of requiring manual anomaly score thresholds, the LLM directly answers "Is this defective?" in natural language. It also supports **multi-turn dialogue**, allowing follow-up questions about defect type, location, and severity.

### Key Results
- **94.1%** image-level AUROC on MVTec AD (with 1-shot reference)
- Provides **natural language explanations** of detected defects (e.g., "There is a crack on the upper-left surface of the pill")
- Pixel-level localization via anomaly heatmaps

### Limitations & Relevance to My Work
- **Not truly zero-shot** — requires fine-tuning on simulated anomaly data, limiting scalability
- Dependent on quality of synthetic anomalies for training
- This motivates combining CLIP's zero-shot scoring strength with MLLM reasoning, avoiding the need for any fine-tuning

---

## Paper 3: AnomalyCLIP — Object-Agnostic Prompt Learning for Zero-Shot Anomaly Detection
**Authors:** Zhou et al. | **Venue:** ICLR 2024  
**Code:** [github.com/zqhang/AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP)

### Summary
AnomalyCLIP addresses a key weakness of WinCLIP: its reliance on hand-crafted, object-specific text prompts. Instead of manually writing prompts for each product category, AnomalyCLIP introduces **learnable object-agnostic text prompts** — the model learns universal prompt embeddings that capture the concept of "normal" vs "anomalous" **across all object categories**. This is achieved through prompt learning (optimizing continuous token embeddings while keeping CLIP frozen). The learned prompts generalize to unseen product types without retraining, making it truly object-agnostic.

### Key Results
- **Significant improvement over WinCLIP** on cross-domain evaluation (unseen product categories)
- Approximately **~94% image-level AUROC** on MVTec AD
- Better generalization to new, unseen domains compared to hand-crafted prompt methods

### Relevance to My Work
- Directly informs my **prompt engineering study** — comparing hand-crafted prompts (WinCLIP) vs learned prompts (AnomalyCLIP) vs my systematic prompt strategy
- Shows that prompt design is a critical factor in CLIP-based anomaly detection performance
- AnomalyCLIP requires a training phase for prompt learning; I want to explore if structured manual prompts can achieve competitive results without any learning

---

## Paper 4: MMAD — Multi-Modal Anomaly Detection Benchmark
**Authors:** Jiang et al. | **Venue:** ICLR 2025  
**Code:** [github.com/jam-cc/MMAD](https://github.com/jam-cc/MMAD)

### Summary
MMAD is a comprehensive benchmark designed to evaluate how well current Multimodal LLMs perform on industrial anomaly detection. It defines **7 subtasks**: (1) Anomaly Classification, (2) Anomaly Localization, (3) Anomaly Type Classification, (4) Severity Assessment, (5) Defect Description, (6) Root Cause Analysis, and (7) Repair Suggestion. The benchmark evaluates both proprietary models (GPT-4o, Claude) and open-source models (LLaVA, etc.) on these tasks.

### Key Findings
| Model Category | Performance |
|----------------|-------------|
| GPT-4o (best proprietary) | Best overall, but **still not sufficient** for industrial deployment |
| Claude | Moderate performance |
| Open-source MLLMs (LLaVA etc.) | **Poor** — struggle with fine-grained defect detection |

### Relevance to My Work
- **Validates the research gap** — even state-of-the-art MLLMs cannot reliably perform industrial anomaly detection out-of-the-box
- The 7-subtask framework provides an **evaluation methodology** I can adopt or extend
- Published at **ICLR 2025** — confirms this is an active, high-impact research area
- Motivates my hybrid approach: rather than relying on MLLM alone, combine CLIP's strong visual scoring with MLLM's language reasoning

---

## Paper 5: PatchCore — Towards Total Recall in Industrial Anomaly Detection
**Authors:** Roth et al. | **Venue:** CVPR 2022  
**Code:** [github.com/amazon-science/patchcore-inspection](https://github.com/amazon-science/patchcore-inspection)

### Summary
PatchCore is the **strongest traditional (non-VLM) baseline** for industrial anomaly detection. It extracts patch-level features from a pre-trained CNN (ImageNet-trained), stores representative normal features in a **memory bank** using coreset subsampling, and detects anomalies by computing the **nearest-neighbor distance** between test patches and the memory bank. If a test patch is far from all stored normal patches, it is flagged as anomalous. The approach is simple, elegant, and highly effective.

### Key Results
| Metric | MVTec AD |
|--------|----------|
| Image-level AUROC | **99.1%** |
| Near-perfect detection for most categories | ✅ |

### Limitations & Relevance to My Work
- **Requires training data** — needs normal samples for each product category; cannot generalize to unseen products
- **No explanations** — only outputs an anomaly score and heatmap, no reasoning about *what* the defect is
- Serves as the **upper-bound baseline** for comparison — my zero-shot approach targets the gap between WinCLIP's 91.8% and PatchCore's 99.1%
- If my zero-shot method can approach ~92–95% AUROC while providing natural language explanations and requiring no training data, it would represent a significant practical advantage

---

## Summary Table

| # | Paper | Venue | Method | Zero-Shot? | MVTec AUROC | Key Contribution |
|---|-------|-------|--------|------------|-------------|------------------|
| 1 | WinCLIP | CVPR 2023 | CLIP + prompt ensemble + sliding window | ✅ Yes | 91.8% | First zero-shot CLIP-based AD |
| 2 | AnomalyGPT | AAAI 2024 | Fine-tuned LVLM + image decoder | ❌ No | 94.1% (1-shot) | First MLLM for AD with explanations |
| 3 | AnomalyCLIP | ICLR 2024 | CLIP + learnable prompts | ✅ Yes | ~94% | Object-agnostic prompt learning |
| 4 | MMAD | ICLR 2025 | Benchmark / evaluation | — | — | Proves MLLMs still struggle with AD |
| 5 | PatchCore | CVPR 2022 | Memory bank + KNN | ❌ No | 99.1% | Strongest traditional baseline |

---

## How These Papers Inform My Research Direction

```
PatchCore (99.1%) ← Upper-bound baseline (needs training)
       ↑
   THE GAP I target (92–95% zero-shot)
       ↑
WinCLIP (91.8%) ← Zero-shot scoring, no explanations
       +
AnomalyCLIP ← Better prompt strategies
       +
AnomalyGPT ← Natural language reasoning (but needs fine-tuning)
       +
MMAD Benchmark ← Proves the gap exists
       ↓
MY PROPOSED WORK:
  Stage 1: CLIP + systematic prompt engineering → anomaly scoring
  Stage 2: Open-source MLLM (LLaVA) → defect explanation
  = Zero-shot + Explainable + Open-source
```
