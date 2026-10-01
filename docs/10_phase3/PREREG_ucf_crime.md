# Pre-registration — context sweep on UCF-Crime

**Written and committed before any UCF-Crime score was computed.** The commit
timestamp of this file is the evidence. Nothing below may be edited after the
results exist; corrections go in a dated addendum at the bottom.

## Why this benchmark

Result 5 and Result 5b conclude that the scene descriptor works mainly by telling
the model **which scene is in view**. The gap it produces was +0.105 across
ShanghaiTech's 12 camera views, +0.033 within one view, +0.020 on single-view
Avenue. UCF-Crime's 290 test videos come from ~290 different cameras and
settings — streets, shops, homes, highways. It is the strongest available test
of that account on data it was not developed on.

## Data and protocol (fixed)

| | |
|---|---|
| Split | official test split, 290 videos (140 anomalous, 150 normal) |
| Labels | official temporal annotation, intervals 1-based inclusive |
| Sampling | every 16th frame (69,634 frames) |
| Encoder | CLIP ViT-L/14 LAION-2B, whole frame, frozen |
| Prompt ensemble | `surveillance` (the crime-oriented ensemble), unchanged |
| Fusion | descriptor appended to the **normal** ensemble only |
| Window | w = 5 at step 16 (≈2.7 s, the time-equivalent of w = 31 at step 2 on ShanghaiTech); w = 1 also reported |
| **Primary metric** | frame-level AUROC over **raw pooled** scores — the UCF-Crime convention |
| Secondary | the same after per-video min–max normalisation (the ShanghaiTech protocol) |

**Why raw is primary here, when it was secondary on ShanghaiTech:** 150 of the
290 videos contain no anomaly at all. Per-video min–max forces every normal
video to span [0, 1], so each one acquires frames scored as maximally anomalous.
That is harmless when nearly every clip contains an event (ShanghaiTech) and
distorting when half contain none. Both are reported.

## Conditions

| Condition | Descriptor |
|---|---|
| none | M3 off |
| generic | "a generic scene" |
| shared matched | "real-world CCTV surveillance footage of streets, shops and buildings" |
| **per-video (generated)** | LLaVA-1.5-7B caption of the video's **first 3 sampled frames** (frames 0, 16, 32), medoid of the three; same instruction as ShanghaiTech Result 5b |
| per-video shuffled | the per-video captions assigned to other videos, 20 random derangements (seed 20261001) |
| mismatched | the industrial description used throughout |

**Label-free descriptor generation.** Frames to caption are chosen by position,
never by label. In 3 of the 140 anomalous videos the anomaly has already begun
within those frames; those captions are kept, not filtered, because filtering
would require the labels.

## Predictions

| # | Prediction | Refuted if |
|---|---|---|
| P1 | per-video − mismatched gap is **larger than on ShanghaiTech** (> +0.105) | gap ≤ +0.105 |
| P2 | per-video beats the shuffled mean | per-video ≤ shuffled mean |
| P3 | per-video beats the shared matched descriptor | per-video ≤ shared |
| P4 | matched (shared) > mismatched | matched ≤ mismatched |

P1 is the scene-diversity account's quantitative claim and the one that can
fail most informatively. P2 is the mechanism test. P3 is weakest — on
ShanghaiTech the per-view gain over shared was inside the noise — and is
reported as a direction if it holds.

**No prediction is made about absolute AUROC against other methods.** The
comparison with LAVAD (UCF-Crime AUC 80.28, verified in the provenance ledger)
will be reported whatever it shows.

## Known risks, stated in advance

- UCF-Crime videos are low-resolution (320×240) and the anomalies often small;
  the resolution limitation found on MVTec and Avenue applies.
- Several anomaly classes (shoplifting, stealing, abuse) are subtle and
  unlikely to be visible to whole-frame semantics.
- Captioning from the first second of a video is a harder setting than
  captioning a camera from install-time footage.

## Addenda

*(dated, below this line, after results)*

### Addendum 1 — 30 Sep 2026, results (nothing above was edited)

Primary metric raw pooled AUROC, w = 5:

| Condition | raw w5 | raw w1 | norm w5 | norm w1 |
|---|---|---|---|---|
| none | 0.7561 | 0.7577 | 0.7589 | 0.7607 |
| generic | 0.7293 | 0.7308 | 0.7400 | 0.7414 |
| shared matched | 0.8129 | 0.8033 | 0.7666 | 0.7553 |
| **per-video (generated)** | **0.8235** | 0.8150 | 0.7892 | 0.7850 |
| shuffled, mean of 20 | 0.7133 | 0.7173 | 0.7410 | 0.7383 |
| shuffled, best of 20 | 0.7787 | 0.7804 | 0.7841 | 0.7872 |
| mismatched | 0.7271 | 0.7307 | 0.7457 | 0.7431 |

| # | Outcome |
|---|---|
| P1 | **Refuted.** per-video − mismatched = +0.096, not > +0.105. The gap is comparable to ShanghaiTech's, not larger. |
| P2 | Holds. per-video beats the shuffled mean by +0.110 and the best of 20 by +0.045. |
| P3 | Holds as a direction, +0.011. |
| P4 | Holds, +0.086. |

Notes. Shuffled captions (0.713) score below no descriptor (0.756): another
video's description actively misleads. Per-video 0.824 is above LAVAD's reported
80.28, but LAVAD scores every frame and this protocol every 16th; the comparison
is close, not identical. As anticipated, per-video normalisation lowers every
figure on a benchmark where half the videos contain no anomaly.

### Addendum 2 — 30 Sep 2026, ShanghaiTech per-view rerun without labels

The published Result 5b captions were generated from frames chosen using ground
truth labels. Rerun with frames chosen by position only (`--label-free`):
per-view 0.7507 (was 0.7550), shuffled mean 0.7074 / best 0.7224, beats all 11
rotations. The conclusion is unchanged; 0.751 replaces 0.755 in the paper.
