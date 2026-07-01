# 🎥 Future Scope: Video Anomaly Detection via Frame-Level VLM Analysis

> **Context:** As discussed with the professor (March 2, 2026) — extending the multimodal anomaly detection framework from static industrial images to video anomaly detection using a frame-level approach.

---

## 1. Core Idea

Apply the **same CLIP + MLLM hybrid pipeline** to video by treating it as a sequence of images:

```
Video Stream → Frame Extraction (1-5 FPS) → Per-Frame CLIP Scoring → Flag Anomalous Frames → MLLM Explanation
```

**Key advantage:** No new models needed — the pipeline is inherently domain-agnostic. A frame from a surveillance camera is just an image.

---

## 2. Proposed Frame-Level Pipeline

```
INPUT: Surveillance / traffic / campus video
         │
         ▼
  ┌─────────────────────────┐
  │  Frame Extraction        │
  │  • 1 FPS (for long video)│
  │  • 5 FPS (for fast events│
  │    like accidents)       │
  └──────────┬──────────────┘
             │
             ▼ Batch of frames
  ┌─────────────────────────┐
  │  Stage 1: CLIP Scoring   │
  │  Prompts:                │
  │  • "a normal scene"      │
  │  • "an abnormal event"   │
  │  • "a violent act"       │
  │  • "a traffic accident"  │
  │  → Anomaly score/frame   │
  └──────────┬──────────────┘
             │
        Score > threshold?
        ┌────┴────┐
        No        Yes
        │         │
     discard      ▼
              ┌─────────────────────────┐
              │  Stage 2: MLLM Reasoning │
              │  "Describe the anomaly   │
              │   in this frame"         │
              │                          │
              │  Output: "A person is    │
              │  breaking into a car in  │
              │  the parking lot"        │
              └──────────┬──────────────┘
                         │
                         ▼
              ┌─────────────────────────┐
              │  Temporal Aggregation    │
              │  (optional enhancement) │
              │  • Group consecutive     │
              │    anomalous frames      │
              │  • Report anomaly with   │
              │    start/end timestamps  │
              └─────────────────────────┘
```

---

## 3. Video Anomaly Datasets

| Dataset | Domain | Videos | Anomaly Types | Evaluation Metric |
|---------|--------|--------|---------------|-------------------|
| **UCF-Crime** | Surveillance | 1,900 videos | 13 types (robbery, assault, explosion, shoplifting, etc.) | Frame-level AUC |
| **ShanghaiTech** | Campus CCTV | 437 videos | Cycling, vehicles in pedestrian zones, jumping | Frame-level AUC |
| **CUHK Avenue** | Subway entrance | 37 videos | Running, throwing objects, loitering | Frame-level AUC |
| **XD-Violence** | Mixed sources | 4,754 videos | 6 types (fighting, shooting, riot, abuse, etc.) | AP (Average Precision) |

**Recommended starting point:** UCF-Crime — most widely used, diverse anomaly types.

---

## 4. Prompt Design for Video Anomaly Domains

### Surveillance / Security
```
Normal:  "a normal day in a parking lot", "people walking normally", "an empty corridor"
Anomaly: "a robbery in progress", "a person fighting", "someone breaking into a vehicle"
```

### Traffic
```
Normal:  "normal traffic flow", "cars driving on the road", "a clear intersection"
Anomaly: "a car accident", "a wrong-way driver", "a pedestrian jaywalking dangerously"
```

### Campus / Public Space
```
Normal:  "students walking on campus", "a quiet library", "normal classroom activity"
Anomaly: "a person running suspiciously", "an unattended bag", "unusual crowd gathering"
```

---

## 5. Challenges Specific to Video

| Challenge | Impact | Mitigation |
|-----------|--------|------------|
| **Temporal context** | A person running is normal in a park, anomalous in a bank | Use scene-specific prompts; future work: temporal modeling |
| **Frame rate selection** | Too low → miss fast events; too high → slow processing | Adaptive: 1 FPS default, 5 FPS for high-risk areas |
| **Redundant frames** | Most frames in surveillance are boring/normal | CLIP scoring is fast (~50ms/frame on GPU); only send anomalous frames to MLLM |
| **Lighting / camera quality** | Night vision, low-res CCTV, occlusion | CLIP is relatively robust; may need domain-specific prompts |
| **Scale of data** | Hours of video = thousands of frames | Batch processing; CLIP can handle ~20 FPS on a V100 |

---

## 6. What Changes vs Current Industrial Pipeline

| Component | Industrial (current) | Video Extension |
|-----------|---------------------|-----------------|
| Input | Single product image | Video → extracted frames |
| CLIP model | Same ✅ | Same ✅ |
| MLLM model | Same ✅ | Same ✅ |
| Prompts | Product-specific ("damaged screw") | Scene-specific ("robbery in progress") |
| Evaluation metric | Image-level AUROC | **Frame-level AUC** |
| Localization | Pixel-level heatmap | Frame-level timestamp |
| Extra code needed | — | `cv2.VideoCapture()` + frame batching (~50 lines) |

**Bottom line:** ~90% of the code is reused. Only prompt templates and data loading change.

---

## 7. Implementation Timeline (Estimated)

| Task | Effort | When |
|------|--------|------|
| Download video datasets (UCF-Crime) | 1 day | After core pipeline is built |
| Write frame extraction code | Half day | Trivial with OpenCV |
| Design video-domain prompts | 1-2 days | Iterative prompt tuning |
| Run CLIP scoring on video frames | 1 day | Same inference code |
| Run MLLM on flagged frames | 1 day | Same inference code |
| Evaluate & compare with baselines | 2-3 days | Compute frame-level AUC |
| **Total additional effort** | **~1 week** | Near the end of project |

---

## 8. How This Strengthens the Thesis

1. **Demonstrates domain-agnosticism** — same zero-shot pipeline works on factory products AND surveillance video
2. **Broader impact** — video anomaly detection has applications in security, traffic, healthcare
3. **Demonstrates engineering versatility** — one framework serves both images and video, showing practical AI engineering skill
4. **Stronger publication potential** — "cross-domain zero-shot anomaly detection" is a more compelling story
5. **Professor's explicit suggestion** — aligns with advisor expectations

---

## 9. Future Extensions Beyond Frame-Level

*(For thesis future work section or PhD continuation — NOT for current scope)*

- **Temporal modeling:** Use Video-LLaVA or VideoChat to understand actions across frames
- **Real-time streaming:** Optimize pipeline for live CCTV feed processing
- **Multi-camera fusion:** Correlate anomalies across multiple camera views
- **Audio-visual anomaly:** Add audio (screams, gunshots) as another modality
