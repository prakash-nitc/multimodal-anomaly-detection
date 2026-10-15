# Pre-registration — context sweep on XD-Violence

**Written and committed before any XD-Violence video was scored.** The commit
timestamp of this file is the evidence. What was looked at beforehand: the
paper, the annotation file and the archive's file list (to build the loader and
the download script), and at most two of the smallest videos, downloaded only
to test the download script — not decoded, captioned or scored. Nothing below may be edited
after results exist; corrections go in a dated addendum at the bottom.

## Why this benchmark

XD-Violence is LAVAD's second benchmark, so it completes the comparison begun on
UCF-Crime. It is also the first benchmark here that is **not fixed-camera
surveillance**: by their file names, the 800 test videos are 377 clips from 95
films and 423 YouTube clips from 352 sources — sports, games, news, cartoons,
hand-held and car cameras, and some CCTV — and a single test video often cuts
between shots. Everything found so far was found on fixed
cameras. This tests two things on data the method was not developed on:

1. whether a per-video sentence still helps when a video is not one fixed scene;
2. the reading from Phase 3 that the sentence works mainly by **calibrating**
   videos against each other rather than by re-ranking frames inside a video
   (report, "What the descriptor changes"). That reading is registered here as
   prediction P4, using the decision rule fixed in
   `notebooks/measurement_audit.py` — before that audit has been run on any
   dataset.

## Data and protocol (fixed)

| | |
|---|---|
| Split | official test split, 800 videos (500 violent, 300 normal) — Wu et al. 2020, §3.2 |
| Source | the authors' `videos.zip` (10,804,733,011 bytes) and `annotations.txt` (sha256 `27b583ba…e04c2f`), fetched by `notebooks/fetch_xd_violence_test.py`; member list in `data_manifests/xd_violence_test_members.txt` |
| Labels | the authors' own convention from `list/make_gt.py`: 0-based frames, `gt[start:end] = 1` (end exclusive), each video cut to frame count − 1 and then to a multiple of 16. Our implementation matches a literal port of `make_gt.py` on all 800 videos (checked 16 Oct 2026) |
| Sampling | one frame per 16-frame snippet — the snippet's first frame (decoded frames 0, 16, 32, …), the same step as UCF-Crime |
| Encoder | CLIP ViT-L/14 LAION-2B, whole frame, frozen; visual only (XD-Violence's audio is not used) |
| Prompt ensemble | `surveillance` (the violence/crime-oriented ensemble), unchanged |
| Fusion | descriptor appended to the **normal** ensemble only |
| Window | w = 5 snippets (80 frames, 3.3 s at 24 fps) — the same 5 snippets as UCF-Crime; w = 1 also reported |
| **Primary metric** | frame-level **AP**, computed exactly as the authors' `test.py`: each snippet's score repeated 16 times, all test frames pooled, raw scores, `sklearn.metrics.auc(recall, precision)` |
| Secondary | AUROC on the same frames; AP after per-video min–max normalisation; AUROC over the sampled frames only (comparable with the UCF-Crime numbers and the audit) |

**Why AP is primary:** it is the benchmark's own metric. The dataset authors
chose it over AUC "since AUC usually shows an optimistic result when dealing
with class-imbalanced data" (§5.1). AUROC is reported alongside because LAVAD
and VERA report it.

**Why raw scores are primary:** as on UCF-Crime, 300 of the 800 videos contain
no violence, and per-video min–max would force each of them to span [0, 1].

## Conditions

| Condition | Descriptor |
|---|---|
| none | M3 off |
| generic | "a generic scene" |
| shared matched | "scenes from films, television, news, sports and online videos" |
| **per-video (generated)** | LLaVA-1.5-7B caption of the video's **first 3 sampled frames** (decoded frames 0, 16, 32), medoid of the three |
| per-video shuffled | the per-video captions assigned to other videos, 20 random derangements (seed 20261016) |
| mismatched | the industrial description used throughout |

**Caption instruction.** Identical to ShanghaiTech and UCF-Crime except one
phrase: "this fixed surveillance camera frame" becomes "this video frame",
because these videos are not fixed-camera footage. Frames are chosen by
position, never by label. Videos whose violence has already begun within the
three captioned frames are counted and reported, and their captions kept —
removing them would require the labels.

## Predictions

Primary metric (AP, raw, w = 5) unless stated.

| # | Prediction | Refuted if |
|---|---|---|
| P1 | per-video beats the shuffled mean | per-video ≤ shuffled mean |
| P2 | shared matched beats mismatched | shared ≤ mismatched |
| P3 | per-video beats no descriptor | per-video ≤ none |
| P4 | the per-video − none gap is **CALIBRATION-dominated** under `measurement_audit.py`'s rule (level share ≥ 2/3 and \|within-video gap\| ≤ 1/3 of the gap), on sampled-frame raw AUROC at w = 5 | any other verdict, including "no effect" |
| P5 | per-video − shuffled mean is **smaller than on UCF-Crime** (+0.110, sampled-frame raw AUROC, w = 5) | gap ≥ +0.110 |

P1 is the mechanism test, as on UCF-Crime. P2 and P3 test the direction of the
effect on a new kind of data. P4 is the calibration reading's first test on a
benchmark it was not derived from; it can fail. P5 follows from the shot cuts:
a caption of the opening second describes less of an XD-Violence video than of
a fixed-camera video, and clips of the same film share scenery, so swapping
captions should cost less than on UCF-Crime.

**No prediction is made about absolute AP against other methods.** For
reference, verified in `docs/07_reference_papers/PROVENANCE.md`: LAVAD AP 62.01
/ AUC 85.36; LLaVA-1.5 50.26 / 79.62; zero-shot CLIP 17.83 / 38.21 (all LAVAD
Table 2); VERA AUC 88.26 (VERA Table 3; VERA learns its questions on training
data). These are reported whatever the result. LAVAD does not state its AP
formula, so the comparison is reported as comparable or not, never as a margin.

## Known risks, stated in advance

- **Shot cuts.** The descriptor comes from the first second; later shots may be
  different places. This is expected to weaken the per-video condition (P5).
- **Repeated sources.** Up to 10 test clips share one film (and up to 6 one
  YouTube source), so a random derangement can give a video the caption of
  another clip from the same source — a weaker control than on UCF-Crime.
  Reported as is.
- **Non-photographic content.** Cartoons and games are out of CLIP's usual
  photographic domain.
- **AP is threshold-free but prevalence-sensitive.** AP values are not
  comparable to AUROC values and are not compared with them.
- The official AP formula (trapezoidal `auc(recall, precision)`) differs slightly
  from `average_precision_score`; the official one is used.

## Run order (server)

```bash
pip install -r requirements-server.txt
python notebooks/fetch_xd_violence_test.py                    # ~10.8 GB
python notebooks/cache_embeddings.py --dataset xd_violence \
       --root ~/dazvad/data/xd_violence --frame-step 16 --crops 1
python notebooks/xd_violence_sweep.py caption                 # LLaVA, label-free
python notebooks/xd_violence_sweep.py run                     # prints P1–P5
```

## Addenda

*(dated, below this line, after results)*
