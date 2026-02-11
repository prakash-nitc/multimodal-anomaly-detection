# 📚 Curated Paper Reading List — Multimodal Anomaly Detection

> **Instructions:** You don't need to read every paper fully. Read the **abstract + intro + results** of all papers (~15 min each). Deep-read only the ⭐ starred papers (method section + experiments).

---

## Category A: Foundational / Dataset Papers

### A1. ⭐ MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection
- **Authors:** Bergmann et al.
- **Venue:** CVPR 2019
- **Summary:** Introduces the MVTec Anomaly Detection dataset — the standard benchmark with 15 industrial product categories (bottle, cable, capsule, carpet, grid, hazelnut, leather, metal nut, pill, screw, tile, toothbrush, transistor, wood, zipper). Contains 5,354 high-res images with pixel-level anomaly annotations. **This is the dataset you'll use.**
- **Why read:** You MUST understand this dataset to do any experiments or answer questions about it.
- **Key result:** Establishes baseline AUROC scores for traditional methods.

### A2. CLIP: Learning Transferable Visual Models From Natural Language Supervision
- **Authors:** Radford et al. (OpenAI)
- **Venue:** ICML 2021
- **Summary:** Introduces Contrastive Language-Image Pre-training (CLIP) — trains on 400M image-text pairs to learn joint vision-language representations. Can classify images via text prompts without any training (zero-shot). **This is the core model your research builds on.**
- **Why read:** High-level understanding of how CLIP works (contrastive learning, image-text similarity).
- **Key result:** Zero-shot ImageNet accuracy competitive with supervised ResNet-50.

### A3. LLaVA: Visual Instruction Tuning
- **Authors:** Liu et al.
- **Venue:** NeurIPS 2023
- **Summary:** Introduces Large Language and Vision Assistant — connects a vision encoder (CLIP ViT) with a language model (Vicuna/LLaMA) via a projection layer. Can see images and have conversations about them. **This is the MLLM you'll use.**
- **Why read:** Understand multimodal LLM architecture at a high level.
- **Key result:** Strong visual reasoning and conversation abilities.

---

## Category B: Traditional Anomaly Detection Methods (Baselines)

### B1. ⭐ Towards Total Recall in Industrial Anomaly Detection (PatchCore)
- **Authors:** Roth et al.
- **Venue:** CVPR 2022
- **Summary:** Extracts patch-level features from a pre-trained CNN, stores them in a memory bank, and detects anomalies by finding the nearest-neighbor distance. Simple yet highly effective. **Best performing traditional baseline.**
- **Why read:** This is the strongest baseline you'll compare against.
- **Key result:** 99.1% image-level AUROC on MVTec AD.

### B2. PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection and Localization
- **Authors:** Defard et al.
- **Venue:** ICPR 2020
- **Summary:** Models the distribution of patch-level features from normal images using multivariate Gaussian. Detects anomalies as out-of-distribution patches. Uses Mahalanobis distance for scoring.
- **Why read:** Key comparison baseline, uses statistical approach.
- **Key result:** 95.3% image-level AUROC on MVTec AD.

### B3. Student-Teacher Feature Pyramid Matching (STFPM)
- **Authors:** Wang et al.
- **Venue:** 2021
- **Summary:** Uses a teacher-student network architecture. Teacher (pre-trained) and student (trained on normal data) produce feature pyramids. Anomalies cause discrepancies between teacher and student features.
- **Why read:** Represents knowledge-distillation approach to AD.
- **Key result:** Competitive results on MVTec AD with efficient inference.

---

## Category C: CLIP-Based Anomaly Detection (Most Relevant)

### C1. ⭐ WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation
- **Authors:** Jeong et al.
- **Venue:** CVPR 2023
- **Summary:** First major work applying CLIP to industrial anomaly detection. Uses compositional prompt ensembles ("a photo of a damaged [object]" etc.) and a sliding window approach for pixel-level anomaly detection. Achieves strong zero-shot results without any training.
- **Why read:** THE most important prior work for your research. Read carefully.
- **Key result:** 91.8% image-AUROC, 85.1% pixel-AUROC on MVTec AD (zero-shot).

### C2. CLIP-AD: A Language-Guided Staged Dual-Path Model for Zero-Shot Anomaly Detection
- **Authors:** Chen et al.
- **Venue:** 2023
- **Summary:** Reinterprets text prompt design for CLIP-based AD. Introduces staged dual-path architecture with separate global and local feature paths. Improves segmentation over WinCLIP.
- **Why read:** Shows how prompt design matters for anomaly detection.
- **Key result:** Improved pixel-level AUROC over WinCLIP on MVTec AD.

### C3. ⭐ AnomalyCLIP: Object-Agnostic Prompt Learning for Zero-Shot Anomaly Detection
- **Authors:** Zhou et al.
- **Venue:** ICLR 2024
- **Summary:** Learns object-agnostic text prompts that generalize across different product categories. Uses prompt learning (learnable text tokens) instead of hand-crafted prompts. Improves generalization to unseen domains.
- **Why read:** Shows learnable prompts vs hand-crafted prompts — directly relevant to your work.
- **Key result:** Significant improvement over WinCLIP on cross-domain evaluation.

### C4. APRIL-GAN: A Zero-/Few-Shot Anomaly Classification and Segmentation Method for CVPR 2023 VAND Challenge
- **Authors:** Chen et al.
- **Venue:** CVPR 2023 Workshop
- **Summary:** Uses CLIP features with additional linear layers for anomaly scoring. Won the VAND (Visual Anomaly and Novelty Detection) challenge at CVPR 2023.
- **Why read:** Simple effective approach, won a competition.
- **Key result:** Competition-winning results on MVTec AD.

### C5. VCP-CLIP: Visual Context Prompting for Zero-Shot Anomaly Segmentation
- **Authors:** Qu et al.
- **Venue:** ECCV 2024
- **Summary:** Introduces visual context prompting to activate CLIP's perception of anomalous patterns. Uses visual context from reference images to guide the text prompts.
- **Why read:** Recent ECCV paper, shows visual prompting direction.
- **Key result:** State-of-the-art pixel-level anomaly segmentation.

### C6. CLIPFUSION: Combining CLIP with Diffusion for Zero-Shot Anomaly Detection
- **Authors:** 2024
- **Summary:** Combines CLIP's semantic understanding with diffusion model's generative capabilities for anomaly detection. Uses diffusion reconstruction error + CLIP similarity fusion.
- **Why read:** Shows multimodal fusion idea (two different model types).
- **Key result:** Outperforms WinCLIP in both classification and segmentation.

---

## Category D: MLLM-Based Anomaly Detection (Your Research Direction)

### D1. ⭐ AnomalyGPT: Detecting Industrial Anomalies Using Large Vision-Language Models
- **Authors:** Gu et al.
- **Venue:** AAAI 2024
- **Summary:** First LVLM-based industrial anomaly detection method. Fine-tunes a vision-language model on simulated anomaly data. Can detect and localize anomalies through natural language dialogue without manual thresholds. Supports few-shot in-context learning.
- **Why read:** THE most important MLLM-based AD paper. Read carefully.
- **Key result:** State-of-the-art on MVTec AD with natural language explanations of defects.

### D2. ⭐ Exploring Grounding Potential of VLMs for Generic Anomaly Detection (GPT-4V study)
- **Authors:** Cao et al.
- **Venue:** 2023
- **Summary:** Systematically evaluates GPT-4V's ability to detect anomalies across industrial, medical, and video domains. Tests zero-shot and one-shot prompting strategies. Shows GPT-4V can reason about anomalies and provide explanations.
- **Why read:** Shows what commercial MLLMs can do — strong motivation for your work.
- **Key result:** GPT-4V shows promising but imperfect anomaly detection, gap exists for improvement.

### D3. MMAD: Multi-Modal Anomaly Detection Benchmark
- **Authors:** Jiang et al.
- **Venue:** ICLR 2025
- **Summary:** Introduces a comprehensive benchmark for evaluating MLLMs in industrial anomaly detection. Defines 7 key subtasks. Evaluates GPT-4o, Claude, Gemini, and open-source models. Shows current MLLMs still have significant room for improvement.
- **Why read:** Directly relevant — provides evaluation framework you can use/extend.
- **Key result:** Even GPT-4o achieves only moderate accuracy, highlighting research opportunities.

### D4. VELM: Vision Expert-Enhanced LLM for Anomaly Detection
- **Authors:** 2024
- **Summary:** Combines unsupervised anomaly detection methods (as vision experts) with LLMs for anomaly classification. The vision expert detects anomalies, the LLM interprets and classifies them using text descriptions.
- **Why read:** Shows hybrid approach (traditional AD + LLM) — potential direction for your work.
- **Key result:** Better anomaly classification through vision expert + LLM reasoning.

### D5. Echo: A Multi-Expert Framework for MLLM-Based Industrial Anomaly Detection
- **Authors:** 2025
- **Summary:** Proposes a multi-expert framework that integrates reference extraction, knowledge guidance, reasoning, and decision-making modules to enhance MLLM performance for industrial AD.
- **Why read:** Latest work, shows how to structure an MLLM-based AD system.
- **Key result:** Improved performance over baseline MLLMs.

---

## Category E: Survey / Review Papers

### E1. ⭐ A Survey on Visual Anomaly Detection: Challenge and Approach
- **Authors:** Recent survey (2024)
- **Summary:** Comprehensive survey covering the full landscape of visual anomaly detection — from traditional methods to deep learning to foundation models. Provides taxonomy and comparison tables.
- **Why read:** Great source for your literature survey. Borrow the taxonomy structure.
- **Key result:** Provides organized overview of 100+ AD methods.

### E2. Anomaly Detection in Industrial Quality Inspection: A Review
- **Authors:** 2024
- **Summary:** Focuses specifically on industrial quality inspection use cases. Covers datasets, methods, and practical deployment considerations.
- **Why read:** Good for your "Background" section — connects AD to real-world applications.
- **Key result:** Identifies key challenges in industrial AD deployment.

---

## 📊 Quick Reference Table

| # | Paper | Year | Category | Method Type | Zero-Shot? | Dataset | Read Priority |
|---|---|---|---|---|---|---|---|
| A1 | MVTec AD | 2019 | Dataset | - | - | MVTec AD | ⭐ Must read |
| A2 | CLIP | 2021 | Foundation | Contrastive Learning | ✅ | ImageNet | Skim |
| A3 | LLaVA | 2023 | Foundation | MLLM | ✅ | Various | Skim |
| B1 | PatchCore | 2022 | Traditional AD | Memory Bank + KNN | ❌ | MVTec AD | ⭐ Must read |
| B2 | PaDiM | 2020 | Traditional AD | Gaussian Modeling | ❌ | MVTec AD | Skim |
| B3 | STFPM | 2021 | Traditional AD | Teacher-Student | ❌ | MVTec AD | Skim |
| C1 | WinCLIP | 2023 | CLIP-based AD | Prompt Ensemble + Window | ✅ | MVTec AD | ⭐ Must read |
| C2 | CLIP-AD | 2023 | CLIP-based AD | Dual-Path | ✅ | MVTec AD | Skim |
| C3 | AnomalyCLIP | 2024 | CLIP-based AD | Prompt Learning | ✅ | MVTec AD | ⭐ Must read |
| C4 | APRIL-GAN | 2023 | CLIP-based AD | Linear Probing | ✅ | MVTec AD | Skim |
| C5 | VCP-CLIP | 2024 | CLIP-based AD | Visual Context Prompt | ✅ | MVTec AD | Skim |
| C6 | CLIPFUSION | 2024 | CLIP-based AD | CLIP + Diffusion | ✅ | MVTec AD | Skim |
| D1 | AnomalyGPT | 2024 | MLLM-based AD | Fine-tuned LVLM | ❌ | MVTec AD | ⭐ Must read |
| D2 | GPT-4V study | 2023 | MLLM-based AD | Zero-shot GPT-4V | ✅ | Multi-domain | ⭐ Must read |
| D3 | MMAD | 2025 | MLLM-based AD | Benchmark | - | Custom | Skim |
| D4 | VELM | 2024 | MLLM-based AD | Vision Expert + LLM | ❌ | MVTec AD | Skim |
| D5 | Echo | 2025 | MLLM-based AD | Multi-Expert MLLM | ❌ | MVTec AD | Skim |
| E1 | AD Survey | 2024 | Survey | - | - | - | ⭐ Must read |
| E2 | Industrial AD Review | 2024 | Survey | - | - | - | Skim |

---

## 📖 Reading Strategy (15 min per paper for "Skim", 45 min for "Must read")

1. **Day 1–2:** Read all ⭐ papers (A1, B1, C1, C3, D1, D2, E1) = ~7 papers × 45 min = ~5 hrs
2. **Day 3–4:** Skim remaining papers = ~12 papers × 15 min = ~3 hrs
3. **Total reading time: ~8 hours spread over 4 days**
