# 🚀 Mentor Update: Domain Adaptability & Video Anomaly Detection (VAD)

Based on the professor's suggestion to incorporate **domain adaptability** and **Video Anomaly Detection (VAD)** alongside your current multimodal approach, here is a structured update you can present tomorrow. 

This directly builds upon your existing proposal (zero-shot VLM/MLLM approach) but extends it from static industrial images to dynamic video domains.

---

## 1. Why Domain Adaptation is Critical for Video AD
In industrial image anomaly detection (like MVTec AD), the background is usually static and controlled. In Video Anomaly Detection (e.g., surveillance, complex manufacturing lines), the environment, camera angles, and lighting constantly change. 
**Domain Adaptation (DA)** is required to transfer the model's understanding of "normal" from a source domain (where we have labels/clean data) to a target domain (a new video scene) without requiring full retraining.

## 2. Recent Papers to Mention (2024–2025)
Show your mentor that you have looked into the latest, state-of-the-art literature combining your core tools (CLIP/MLLMs) with Video AD and Domain Adaptation.

### Top Papers on Domain Adaptation in VAD
1. **DA-VAD: Domain Adaptation Video Anomaly Detection (2025)**
   - *Concept:* A weakly-supervised method that treats video-level and clip-level features as distinct source and target domains. Uses domain aligners to prevent overfitting to specific scenes.
   - *Why it matters:* Shows how to adapt features dynamically without needing full supervision in the target domain.
2. **Ada-VAD: Adaptive Video Anomaly Detection (2024)**
   - *Concept:* Tackles the few-shot cross-domain VAD problem. Pretrains with synthesized abnormal samples and uses adversarial training to mitigate distribution shifts when introduced to a new video domain.
   - *Why it matters:* Perfectly aligns with your goal of minimizing training data for new domains.

### Top Papers on MLLMs/CLIP in VAD
3. **MoniTor (2025)**
   - *Concept:* An online, **training-free** VAD framework utilizing LLMs with instructional guidance. 
   - *Why it matters:* Validates your proposal's "zero-shot/training-free" direction but applied to video.
4. **VadCLIP (2024)** & **HeadCLIP (2025)**
   - *Concept:* VadCLIP adapts CLIP for weakly supervised VAD without requiring fine-tuning of the base model. HeadCLIP (2025) adapts attention heads of models like CLIP to generalize concepts of normality across domains via learnable prompts.
   - *Why it matters:* Directly relates to your methodology of using CLIP for anomaly scoring and prompt engineering.

---

## 3. How to Update Your Proposed Methodology
When speaking to your mentor, suggest extending your **3-Component Framework** (from your `research_proposal.tex`) as follows to handle video:

*   **Component 1 (VLM Scoring): From Spatial to Temporal Windows**
    Instead of passing static sliding windows through CLIP (like WinCLIP), use temporal windows (e.g., 16-frame clips). You can use models like **Video-LLaMA** or add a temporal projection layer to CLIP to capture motion anomalies, not just appearance.
*   **Component 2 (Prompt Engineering): Domain-Adaptive Prompts**
    Use **Learnable Prompts** (inspired by AnomalyCLIP and HeadCLIP). Instead of hardcoding "a photo of a defective toy", train domain-agnostic text tokens that adapt to the target video's specific environment automatically.
*   **Component 3 (MLLM Reasoning): Event-Level Explanations**
    Instead of describing a single defect, the MLLM (like LLaVA-Video or GPT-4o) takes sequence frames and reasons about *events* over time (e.g., "The robotic arm moved too fast and dropped the component").

---

## 4. Suggested Script for Tomorrow's Meeting

> *"Professor, I've looked into your suggestion regarding Domain Adaptability and Video Anomaly Detection. It's a very active area right now, especially merging it with our Vision-Language Model approach. 
> 
> I found very recent works from 2024 and 2025, such as **DA-VAD** and **HeadCLIP**, which show that we can use CLIP and MLLMs for video by treating different video scenes as different target domains. 
> 
> To integrate this into our project, I propose we extend our zero-shot VLM pipeline to handle temporal frame sequences. For domain adaptation, we can use learnable prompt tokens (like in HeadCLIP) so that our text queries automatically adapt to the specific background or lighting of the new video stream, without needing to fine-tune the entire model. The MLLM will then provide descriptions of anomalous temporal events rather than just static defects."*
