# Presentation Script — 17 Slides
### Easy to remember, conversational, and confident

> **Tip:** Don't memorize word-for-word. Memorize the **bold key phrases** and the flow will come naturally.

---

## Slide 1 — Title Slide
**⏱ ~30 seconds**

*"Good morning sir. My name is Prakash Kumar Sarangi, M.Tech student, Department of CSE, NIT Calicut, under the guidance of Prof. Dr. Pranesh Das.*

*My research topic is — **Zero-Shot Video Anomaly Detection using Vision-Language Models with Domain-Adaptive Verbalized Prompting**.*

*In simple terms — I'm building a system that can **detect anomalies in video without any training data**, and can **adapt to new environments just by describing them in text**."*

---

## Slide 2 — The Problem & The Challenge
**⏱ ~1 minute**

*"So why is this important? Current anomaly detection systems have **three critical problems**:*

*First — **Data Dependency**. Traditional methods like PatchCore need training data for every new product or every new camera location. If you move a camera from a warehouse to a parking lot — you need to collect data and retrain. This doesn't scale.*

*Second — **No Explainability**. These systems output a score like 0.87 — but never tell you **what** is wrong. For an industrial operator, that's not useful. They need to know — is it a crack? A contamination? Where exactly?*

*Third — **Temporal Blindness**. Image-based methods look at each frame alone. They can't understand that a person **walking, then running, then attacking** is a developing anomaly. Each frame looks fine individually.*

*The critical gap is — **no existing method solves all three simultaneously**. That's what our work targets."*

---

## Slide 3 — Literature: PatchCore & WinCLIP
**⏱ ~1.5 minutes**

*"Let me walk you through the key papers we studied. Starting with our **two primary baselines**:*

*First — **PatchCore**, CVPR 2022. This is the **strongest traditional method**. It stores patch-level features from normal images in a memory bank, then flags anything that looks different. It achieves **99.1% AUROC** on MVTec AD — near perfect. But it **needs training data** for every category and gives **no explanations**.*

*Second — **WinCLIP**, CVPR 2023. This was the **first paper to use CLIP for zero-shot anomaly detection**. It uses text prompts like 'a damaged bottle' vs 'a good bottle' and compares them against the image. Achieves **91.8% AUROC with zero training**. But it relies on hand-crafted prompts and still gives no explanations.*

*Our own CLIP baseline achieves **88.5% AUROC** — the 3.3% gap from WinCLIP is because we intentionally **skipped the sliding window** to keep the architecture simple for video extension."*

---

## Slide 4 — AnomalyCLIP, AnomalyGPT & MMAD
**⏱ ~1.5 minutes**

*"Next, three more advanced papers:*

*First — **AnomalyCLIP**, ICLR 2024. Instead of hand-crafting prompts, it **learns** object-agnostic prompts automatically. Gets ~94% AUROC. But it still needs a training phase for prompt learning, and has no video capability.*

*Second — **AnomalyGPT**, AAAI 2024. This is the **first paper to use a multimodal LLM** for anomaly detection. It can actually **explain defects in natural language** — like 'there is a crack on the upper-left surface'. Gets 94.1% AUROC. But it requires fine-tuning on synthetic anomaly data — so it's **not truly zero-shot**.*

*Third — **MMAD Benchmark**, ICLR 2025. This tested GPT-4o, Claude, and open-source models on 7 anomaly detection tasks. The key finding? **Even GPT-4o performs only moderately**. This confirms that MLLMs alone are not enough — you need a structured pipeline.*

*Bottom line — **no existing method is simultaneously zero-shot, explainable, and temporal**. That gap is validated."*

---

## Slide 5 — Pivot to Video AD
**⏱ ~45 seconds**

*"Up to this point, all the methods I discussed work on **static images**. But our professor suggested — look into **domain adaptability beyond static images**.*

*That's when we pivoted to **Video Anomaly Detection**. The reasoning is simple — in real-world deployment, cameras capture **video, not photos**. Video has **temporal context** — things happening over time — motion cues, developing events.*

*The flow goes from static image analysis → sequential frames → temporal context → motion cues → full video anomaly detection. Each step adds more understanding."*

---

## Slide 6 — OVVAD & VAD Survey
**⏱ ~1.5 minutes**

*"For the video extension, we studied four key papers. First two on this slide:*

*First — **OVVAD**, CVPR 2024. This paper showed you can keep CLIP **completely frozen** and add a tiny **temporal adapter** on top — less than 1 million parameters. It also injects text descriptions of anomaly categories and synthesizes fake anomaly videos for training. It achieves **86.4% AUC on UCF-Crime**. The key insight for us — **temporal modeling on top of a frozen VLM works**.*

*Second — the **Gao et al. Survey**, ACM Computing Surveys 2025. This covers **400+ VAD methods** and explicitly identifies the gaps — generalization across domains, temporal modeling, and label-heavy training. It validates that **open-vocabulary, zero-shot VAD** is the future direction. This is exactly where our work sits."*

---

## Slide 7 — LAVAD & VERA
**⏱ ~1.5 minutes**

*"The next two papers complete our architectural puzzle:*

*First — **LAVAD**, CVPR 2024. This is a **fully training-free** approach. It converts video frames to text captions using BLIP-2, then asks an LLM to reason — 'is this sequence of events anomalous?' It **beats all unsupervised methods on UCF-Crime without any training**. The limitation is — captions can be noisy or hallucinated, and inference cost is high.*

*Second — **VERA**, CVPR 2025. This is the most elegant solution for **domain adaptation**. Instead of retraining, you simply describe the environment in text — 'a dimly lit warehouse with conveyor belts'. The model adapts its definition of normal based on this text. A running person is normal in a gym but anomalous in a hospital. VERA handles this through **text alone — no parameter changes**.*

*LAVAD gives us the **reasoning engine**, VERA gives us the **domain adaptation**."*

---

## Slide 8 — Comparative Analysis Table
**⏱ ~1 minute**

*"This table summarizes all the methods across five key criteria — zero-shot, training-free, explainable, temporal, and domain-adaptive.*

*As you can see — PatchCore fails on all five. WinCLIP is zero-shot but lacks everything else. AnomalyCLIP and AnomalyGPT each solve one or two problems. OVVAD adds temporal but needs training. LAVAD is training-free with temporal reasoning but no domain adaptation. VERA adds domain adaptation and explainability.*

***No single existing method has all five checkmarks.***

*Our proposed DA-ZVAD — shown in the last row — is designed to be the **first method satisfying all five criteria simultaneously**. That's our contribution."*

---

## Slide 9 — Challenges in Existing Study
**⏱ ~1 minute**

*"To summarize the six key challenges we identified from the literature:*

*One — **Training Dependency**. PatchCore, AnomalyGPT, OVVAD all need costly domain-specific data.*

*Two — **Closed Vocabulary**. Traditional methods fail on anomaly types they haven't seen before.*

*Three — **No Explainability**. Scores without reasoning are useless for real-world deployment.*

*Four — **Domain Rigidity**. What's normal in one place is anomalous in another — current methods can't adapt.*

*Five — **No Temporal Reasoning**. Frame-by-frame scoring misses developing events.*

*Six — **Caption Hallucination**. Methods like LAVAD that rely on VLM captions suffer from inaccurate descriptions.*

*These six gaps directly motivated our problem statement."*

---

## Slide 10 — Problem Statement & Objectives
**⏱ ~1.5 minutes**

*"Our problem statement is:*

***'Design and Development of a Zero-Shot Video Anomaly Detection System using Vision-Language Models with Domain-Adaptive Verbalized Prompting for Cross-Domain Surveillance and Industrial Monitoring.'***

*We have five specific objectives:*

*First — develop a **training-free** pipeline using pre-trained VLMs.*

*Second — enable **domain adaptation** through natural language descriptions — just describe the environment in text.*

*Third — provide **natural language explanations** for every detection — not just a score.*

*Fourth — achieve **open-vocabulary** detection — detect anomaly types never seen before.*

*Fifth — evaluate across **cross-domain benchmarks** — UCF-Crime, XD-Violence, and MVTec AD.*

*The key idea is — **the anomaly score depends on where the camera is deployed**. Same event, different context, different score."*

---

## Slide 11 — DA-ZVAD Architecture
**⏱ ~1.5 minutes**

*"This is our proposed architecture — DA-ZVAD.*

*The pipeline is simple — video frames go through a **frozen CLIP ViT-L/14 encoder** to get visual embeddings. Then an optional **lightweight temporal adapter** captures inter-frame dynamics. The domain context is injected as **verbalized text** — describing the deployment environment. Finally, an **LLM reasoning head** takes everything and outputs both an anomaly score and a natural language explanation.*

*The beauty is — to deploy to a new environment, you just **change the text description**. No retraining, no new data, no parameter changes.*

*And it's computationally feasible — total VRAM is about **7.1 GB**. CLIP takes 1.2 GB, LLaVA quantized takes 5.5 GB, the rest is minimal. It runs on a **single RTX 3060 or a Kaggle T4 GPU**. No multi-GPU setup needed."*

---

## Slide 12 — CLIP Baseline Results
**⏱ ~1.5 minutes**

*"Now let me show our actual experimental results. We ran CLIP zero-shot on all **15 MVTec AD categories** — 1,725 test images — with zero training data.*

*We achieved **88.5% mean AUROC** and **90.7% F1 score** in just **135 seconds on a T4 GPU**.*

*The results split into two clear clusters:*

***Textures** scored **99.5% mean** — leather, wood, grid, carpet, tile. Why? Because texture defects like scratches or holes change the **global appearance** of the image, which CLIP captures easily.*

***Objects** scored **83.1% mean** — bottles, cables, transistors. Why lower? Because object defects are **localized** — a tiny scratch on a metal nut occupies maybe 2% of pixels. CLIP's single image embedding doesn't have the spatial resolution for that.*

*The 3.3% gap from WinCLIP's 91.8% is because we **deliberately skipped the sliding window** — we wanted a simpler architecture that extends cleanly to video frames."*

---

## Slide 13 — AUROC by Category
**⏱ ~45 seconds**

*"This bar chart shows the performance breakdown across all 15 categories.*

*You can clearly see — the **top five are all textures**: leather and wood at 100%, grid at 99.7%, carpet at 99.2%, tile at 98.7%.*

*The **bottom five are all objects**: transistor at 72.7% is the hardest — transistor defects are extremely subtle, even humans need magnification. Cable and capsule are also challenging due to small, localized defects.*

*This pattern tells us — for the next phase, we need to improve **spatial reasoning** for fine-grained defect localization."*

---

## Slide 14 — Video Extension Results
**⏱ ~1.5 minutes**

*"Next, we tested — **can CLIP score video frames effectively?***

*We created a controlled experiment — took real MVTec bottle images and arranged them as a 30-frame pseudo-video: 10 normal frames, then 10 anomalous frames, then 10 normal again. Like a conveyor belt where a batch of defective products passes through.*

*The results were clear:*

*Normal frames scored **0.307** on average. Anomalous frames scored **0.894**. That's a **0.587 score gap** — very strong separation.*

*The score jumps sharply at frame 10 when anomalies appear, and drops back at frame 20 when they stop. A simple threshold at 0.5 would correctly segment this entire video.*

*This proves that **CLIP's discriminative ability transfers to video frames**. The signal is strong. But — each frame is scored independently. There's no temporal context, no smoothing, no explanations. That's exactly what DA-ZVAD's temporal adapter and LLM reasoning will add."*

---

## Slide 15 — Roadmap & Contributions
**⏱ ~1 minute**

*"Here's our roadmap:*

*Semester 2 — which is **done** — we completed the literature survey of 15 papers, built the CLIP baseline achieving 88.5% AUROC, ran the video proof-of-concept, and designed the DA-ZVAD architecture.*

*Semester 3 — July to November — we'll set up UCF-Crime dataset, implement OVVAD's temporal adapter, LAVAD's caption-then-reason pipeline, and VERA's verbalized prompting.*

*Semester 4 — January to May — full DA-ZVAD integration, cross-domain evaluation, ablation studies, and thesis writing.*

*Our expected contributions are:*
1. *A systematic prompt engineering study*
2. *A hybrid VLM + LLM video AD pipeline*
3. *Cross-domain benchmark evaluation*
4. *Domain-adaptive verbalized prompting"*

---

## Slide 16 — References
**⏱ ~15 seconds**

*"These are the key references cited throughout our presentation — spanning CVPR, ICLR, AAAI, NeurIPS, and ACM Computing Surveys from 2019 to 2025."*

*(Don't read them out — just acknowledge and move on.)*

---

## Slide 17 — Thank You
**⏱ ~15 seconds**

*"Thank you for your time. I'm happy to take any questions."*

*(Smile. Make eye contact. Wait confidently.)*

---

# ⏱ TOTAL ESTIMATED TIME: ~16–18 minutes

---

# 🧠 QUICK MEMORY TRICKS

| Slide | Remember This One Thing |
|---|---|
| 2 | **Three problems**: data, explain, temporal |
| 3 | **PatchCore = 99.1% but needs training, WinCLIP = 91.8% zero-shot** |
| 4 | **AnomalyCLIP = learned prompts, AnomalyGPT = first MLLM, MMAD = benchmark** |
| 6 | **OVVAD = frozen CLIP + temporal adapter, Survey = 400+ methods** |
| 7 | **LAVAD = caption→reason, VERA = domain via text** |
| 8 | **Table punchline: DA-ZVAD is the only all-green row** |
| 9 | **Six challenges: train, vocab, explain, domain, temporal, hallucination** |
| 10 | **Problem statement = Design & Dev of zero-shot VAD using VLMs** |
| 12 | **88.5% AUROC, textures 99.5%, objects 83.1%** |
| 14 | **Normal 0.307, Anomalous 0.894, Gap 0.587** |
