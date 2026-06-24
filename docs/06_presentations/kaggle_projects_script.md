# Kaggle Projects — Explanation Script
## For Professor Walkthrough / Viva

---

# PROJECT 1: CLIP Zero-Shot Anomaly Detection on MVTec AD
**File:** `clip_baseline_kaggle.py` (298 lines, single self-contained script)

---

## 1. WHY I DID THIS

*"Sir, the first question in our research was: can a vision-language model like CLIP detect anomalies without any training data? The literature says yes — WinCLIP showed 91.8% AUROC — but we needed our own baseline to:*

1. *Validate that the approach actually works in practice*
2. *Understand where it succeeds and where it fails*
3. *Establish a foundation we can extend to video in the next step*

*So I implemented a complete zero-shot anomaly detection pipeline from scratch on Kaggle using the MVTec AD benchmark — the standard dataset in this field with 15 product categories."*

---

## 2. WHAT THE CODE DOES — Section by Section

### Section 1: Dataset Loader (Lines 40-84)

*"I wrote a custom PyTorch Dataset class called `MVTecDataset`. It scans the MVTec AD folder structure where each category has a `test/` directory containing subfolders — `good/` for normal images and named defect folders like `broken_large/`, `scratch/` for anomalous images.*

*The loader automatically assigns label 0 to 'good' images and label 1 to everything else. For each image, it returns: the preprocessed image tensor, the binary label, the category name, and the file path for debugging."*

### Section 2: Prompt Engineering (Lines 88-113)

*"This is the core of zero-shot detection. Instead of learning from data, we tell CLIP what to look for using natural language.*

*I defined 6 normal states — 'good', 'perfect', 'flawless', 'pristine', 'normal', 'unblemished' — and 6 abnormal states — 'damaged', 'defective', 'broken', 'flawed', 'abnormal', 'imperfect'.*

*These are crossed with 4 sentence templates:*
- *'a photo of a {state} {object}'*
- *'a {state} {object}'*  
- *'a photo of a {state} {object} for quality inspection'*
- *'a close-up photo of a {state} {object}'*

*This gives us 24 normal prompts and 24 abnormal prompts per category. For example, for 'bottle':*
- *'a photo of a good bottle'*
- *'a close-up photo of a flawless bottle'*
- *'a photo of a damaged bottle for quality inspection'*

*Why 24 prompts? Prompt ensembling. A single prompt like 'a damaged bottle' might not capture all ways CLIP understands damage. By averaging across many prompts, we get a more robust representation. This is the same idea as WinCLIP's Compositional Prompt Ensembles."*

### Section 3: CLIP Detector Class (Lines 118-154)

*"The `CLIPZeroShotDetector` class does three things:*

**Model loading:** *We use OpenCLIP's ViT-L/14 model pretrained on LAION-2B — that's 2 billion image-text pairs, much larger than OpenAI's original 400M. ViT-L/14 means a Vision Transformer with 'Large' size and 14×14 patch size. This specific checkpoint is the best open-source CLIP model available.*

**Text encoding:** *`encode_text_prompts()` takes our 24 prompts, encodes each through CLIP's text encoder, L2-normalizes them, then mean-pools into a single embedding vector. So 24 prompts become 1 representative vector for 'normal' and 1 for 'abnormal'.*

**Scoring:** *`compute_anomaly_scores()` is where the detection happens:*
1. *Encode the image through CLIP's visual encoder → get image embedding*
2. *Compute cosine similarity against both text embeddings*
3. *Multiply by CLIP's learned temperature parameter (logit_scale)*
4. *Apply softmax to get probabilities*
5. *P(abnormal) = the anomaly score*

*If the image looks more like 'damaged bottle' than 'good bottle', the anomaly score will be high.*

*The key line is:*
```python
sim = img_feats @ text_embeds.T * self.model.logit_scale.exp()
probs = sim.softmax(dim=-1)
```
*This is a standard CLIP zero-shot classification — we're just classifying into 'normal' vs 'abnormal' instead of '1000 ImageNet classes'."*

### Section 4: Evaluation Loop (Lines 185-253)

*"I iterate over all 15 MVTec AD categories. For each:*
1. *Load the test set using our custom Dataset*
2. *Generate category-specific prompts ('bottle' → 'a damaged bottle')*
3. *Score all test images in batches of 32*
4. *Compute AUROC and optimal F1*

*The total runtime is 135.8 seconds on a Kaggle T4 GPU for all 1,725 test images — about 9 seconds per category."*

### Section 5: Visualization (Lines 256-297)

*"The script outputs a CSV file with all results, a sorted horizontal bar chart color-coded by performance (green >85%, yellow 70-85%, red <70%), and prints a formatted results table."*

---

## 3. RESULTS & INTERPRETATION

### Overall Numbers

| Metric | Value |
|---|---|
| Mean AUROC | **88.5%** |
| Mean F1 | **90.7%** |
| Total time | 135.8 seconds |
| Training needed | **None** |

### Category-Level Analysis

*"The results split into two clear clusters:*

**Textures (99.5% mean):**
| Category | AUROC |
|---|---|
| Leather | 100.0% |
| Wood | 100.0% |
| Grid | 99.7% |
| Carpet | 99.2% |
| Tile | 98.7% |

*Why so high? Texture defects — like a scratch on leather or a hole in carpet — change the global visual appearance of the entire image. CLIP processes the whole image as one embedding, so it catches these global changes easily.*

**Objects (83.1% mean):**
| Category | AUROC |
|---|---|
| Toothbrush | 90.8% |
| Hazelnut | 86.8% |
| Bottle | 85.6% |
| Pill | 85.6% |
| Metal Nut | 84.3% |
| Screw | 84.3% |
| Capsule | 80.9% |
| Zipper | 80.3% |
| Cable | 79.5% |
| Transistor | 72.7% |

*Why lower? Object defects are localised — a tiny scratch on a metal nut, one bent pin on a transistor. CLIP's single image embedding doesn't have the spatial resolution to catch defects that occupy maybe 2% of the image pixels. This is exactly what WinCLIP's sliding window solves, but we chose not to use it for video extensibility.*

*Transistor at 72.7% is the worst case — transistor defects are extremely subtle electronic component issues that even humans need magnification to spot. CLIP was trained on internet images, not micro-electronics."*

---

## 4. DESIGN DECISIONS & WHY

| Decision | Why |
|---|---|
| ViT-L/14 (not ViT-B/16) | Larger model = better embeddings, worth the extra compute |
| LAION-2B (not OpenAI weights) | Open-source, trained on 5x more data, better performance |
| No sliding window | Keep architecture simple for video extension |
| 24 prompts per category | Prompt ensembling reduces sensitivity to wording |
| Batch size 32 | Fits T4 GPU memory comfortably |

---
---

# PROJECT 2: Video AD Preliminary Study
**File:** `video_ad_preliminary_kaggle.py` (411 lines)

---

## 1. WHY I DID THIS

*"After the image baseline, the natural question was: can the same CLIP approach work on video? The professor had suggested exploring domain adaptability beyond static images. Before building the full DA-ZVAD pipeline, I needed to answer a simpler question first:*

**Can CLIP distinguish normal vs anomalous video frames when scored independently?**

*If the answer is no — if per-frame CLIP scoring produces random noise — then there's no point building temporal adapters on top. If yes, then the temporal adapter only needs to smooth and contextualize already-discriminative scores."*

---

## 2. WHAT THE CODE DOES — Section by Section

### Section 1: Model Setup (Lines 36-47)

*"Same CLIP ViT-L/14 model as Notebook 1. Identical encoding pipeline. This is intentional — we want to compare apples to apples and see if the same model works across modalities."*

### Section 2: Video Scoring Functions (Lines 50-116)

*"Three key functions:*

**`encode_text_ensemble()`** — Same as Notebook 1, mean-pools multiple prompts into one embedding.

**`score_frame()`** — Takes a single PIL image (one video frame), encodes it, computes similarity against normal/abnormal text embeddings, returns P(abnormal). This is the atomic operation — everything else builds on this.

**`extract_frames()`** — Reads a video file using OpenCV, samples frames at a specified FPS (default 2 FPS), converts BGR to RGB, returns a list of PIL images. This function isn't used in the final experiment but demonstrates the full video pipeline.

**`score_video()`** — Orchestrates the full pipeline: extract frames → encode text → score each frame → return timestamps + scores. Again, included to show the complete video pipeline even though we use pseudo-video."*

### Section 3: Surveillance Prompts (Lines 119-140)

*"For video, I designed domain-specific prompts:*

**Normal:** *'a normal scene', 'people walking normally', 'a calm and safe environment'*

**Abnormal:** *'a dangerous scene', 'a violent or criminal activity', 'an accident or emergency'*

*These are more general than the industrial prompts in Notebook 1 because surveillance videos cover diverse scenarios. This is a design choice — and exactly where VERA's verbalized prompting would help, by adapting these prompts to specific deployment environments."*

### Section 4: Pseudo-Video Construction (Lines 186-248)

*"Since UCF-Crime is 128 GB and impractical to download in a Kaggle session, I created a controlled experiment:*

*I take real MVTec AD bottle test images and arrange them as a 30-frame pseudo-video:*

| Frames | Content | Label |
|---|---|---|
| 0-9 | 10 normal (good) bottles | 0 |
| 10-19 | 10 anomalous (broken/contaminated) bottles | 1 |
| 20-29 | 10 normal bottles again | 0 |

*This simulates a conveyor belt inspection scenario where a batch of defective products passes through. Using real MVTec images instead of synthetic images means the difficulty level is identical to Notebook 1."*

### Section 5: Scoring (Lines 252-284)

*"I use the same CLIP scoring approach but with inspection-specific prompts:*
- *'a photo of a good bottle', 'a perfect bottle without any defects'*
- *'a photo of a damaged bottle', 'a defective bottle with cracks'*

*Each of the 30 frames is scored independently. No temporal information is used — frame 15 doesn't know what happened in frame 14."*

### Section 6: Temporal Plot (Lines 288-339)

*"The main output is a two-panel temporal plot:*

*Top panel: Anomaly score vs frame index, with the line plot showing score trajectory. Background shading marks the ground truth segments (green for normal, red for anomalous). A horizontal dashed line at 0.5 shows the threshold.*

*Bottom panel: Ground truth as a color bar — green blocks for normal frames, red for anomalous."*

### Section 7: Analysis (Lines 342-371)

*"Computes summary statistics and prints key observations with identified limitations."*

---

## 3. RESULTS & INTERPRETATION

### Quantitative Results

| Metric | Value |
|---|---|
| Mean score (normal frames) | **0.307** |
| Mean score (anomalous frames) | **0.894** |
| Score gap | **0.587** |

### What the Temporal Plot Shows

*"Looking at the temporal plot:*

*Frames 0-9 (normal): Scores stay low, fluctuating between 0.15 and 0.35. All well below the 0.5 threshold.*

*Frame 10 (transition): Score jumps sharply from ~0.35 to ~0.60 — the system immediately detects something changed.*

*Frames 10-19 (anomalous): Scores are consistently high, mostly between 0.86 and 1.0. The different defect types (broken_large, broken_small, contamination) produce slightly different scores, which is why there's variation.*

*Frame 20 (recovery): Score drops back to ~0.35 — the system correctly identifies return to normal.*

*Frames 20-29 (normal again): Scores return to the low baseline around 0.15-0.44."*

### What This Proves

*"Three things:*

1. **CLIP's discriminative ability transfers to frame-level video scoring** — the 0.587 score gap means there's a very clear signal to work with.

2. **Transition detection works** — the score change at frames 10 and 20 is sharp, not gradual. A simple threshold at 0.5 would correctly segment this pseudo-video.

3. **The approach is fast** — each frame takes ~5ms on a T4, making real-time processing feasible."*

---

## 4. LIMITATIONS FOUND → WHY DA-ZVAD IS NEEDED

*"This experiment was designed not just to show success, but to systematically identify what's missing. I found three limitations:*

### Limitation 1: No Temporal Context
*"Each frame is scored alone. Frame 15 showing an anomaly doesn't consider that frames 10-14 were also anomalous. In real surveillance video, this matters — is someone running briefly (normal) or running for 30 seconds (suspicious)?*

**Solution → OVVAD's Temporal Adapter**: Processes sequences of frame embeddings together, learning patterns like 'anomaly scores have been high for N consecutive frames'."*

### Limitation 2: Score Fluctuations
*"Within the anomalous segment, scores vary from 0.60 to 1.0. A 'broken_large' defect scores 0.98 but a 'contamination' scores 0.60. Without smoothing, a simple threshold might miss the contamination frames.*

**Solution → LAVAD's LLM Aggregation**: Instead of thresholding raw scores, an LLM reasons over a window of scores/captions and produces a more stable anomaly judgment."*

### Limitation 3: No Explanations
*"The system says 'anomaly score 0.92' but never says 'broken glass detected on the left side of the bottle'. For industrial deployment, operators need to know what's wrong, not just that something is wrong.*

**Solution → VERA's Verbalized Prompting + LAVAD's LLM**: The LLM generates natural language descriptions like 'contamination detected on bottle surface, likely production residue'."*

---

## 5. HOW THE TWO PROJECTS CONNECT

*"The two notebooks form a logical progression:*

```
Notebook 1 (Image)          Notebook 2 (Video)
     |                            |
     v                            v
  88.5% AUROC on            Can CLIP score
  15 categories              video frames?
     |                            |
     v                            v
  CLIP works for             YES — 0.587 gap
  zero-shot AD               but 3 limitations
     |                            |
     +----------+  +  +-----------+
                |     |
                v     v
        DA-ZVAD Architecture
   (fixes all 3 limitations)
```

*Notebook 1 validates that CLIP is a strong zero-shot anomaly detector. Notebook 2 shows it extends to video but needs temporal modelling. Together, they justify every component of our proposed DA-ZVAD architecture."*

---

## 6. QUICK REFERENCE — Numbers to Remember

| Fact | Value |
|---|---|
| CLIP model | ViT-L/14, LAION-2B |
| Image AUROC | 88.5% (mean), 99.5% (textures), 83.1% (objects) |
| Worst category | Transistor (72.7%) |
| Best category | Leather/Wood (100%) |
| Video normal score | 0.307 |
| Video anomaly score | 0.894 |
| Score gap | 0.587 |
| Inference speed | ~5ms per frame on T4 |
| Total prompts per category | 24 normal + 24 abnormal |
| WinCLIP comparison | 88.5% vs 91.8% (no sliding window) |
| GPU memory | ~1.2 GB for ViT-L/14 |
