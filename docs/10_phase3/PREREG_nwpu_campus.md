# Pre-registration — context sweep on NWPU Campus

**Written and committed before any NWPU Campus video was scored.** The commit
timestamp of this file is the evidence. Nothing below may be edited after
results exist; corrections go in a dated addendum at the bottom.

What was looked at beforehand, to build the loader and the download script: the
paper and its supplementary material; the ground-truth file's names and totals
(242 videos, 384,059 frames, 65,266 anomalous, 124 anomalous videos — the same
totals as the paper's Table 2); the archive's file index; and two test videos,
D036_01 and D043_04, downloaded and decoded only to check that frame *i* of a
video is label *i* (both contain no anomaly). Nothing was captioned or scored.

## Why this benchmark

It is the only benchmark whose authors designate **scene-dependent anomalies** —
"an event is normal in one scene but abnormal in another" (Cao et al. 2023, §1)
— which is the concept shift this thesis is about. It also has 43 fixed-camera
scenes with 1 to 17 test videos each, so effects can be measured inside a scene
as well as across scenes, and a campus setting close to ShanghaiTech, where the
per-camera result (0.751) was found. Every scene has at least one test video
with an anomaly; in the four scene-dependent scenes below, 2–3 of their 5–8
test videos do.

## What "scene-dependent" means here — read before the predictions

The supplementary material defines it by the scene's **normal training
footage**, not by its appearance: photographing is anomalous in scenes D036 and
D038 "because the training videos do not contain photographing", and normal in
D055, whose training videos include it (Supplementary §1). Four classes are
scene-dependent (Supp. Table 2), and they occur in four scenes (Supp. Table 1):

| Scene | Scene-dependent class | Other anomaly classes in the scene |
|---|---|---|
| D003 | Trucks | Dogs |
| D036 | Photographing in restricted area | Climbing tree |
| D038 | Photographing in restricted area | Jaywalking |
| D129 | Cycling on footpath | Playing with water |

(The fourth class, "Wrong turn", is listed in Supp. Table 2 but in no scene of
Supp. Table 1.) The ground truth marks frames as anomalous without saying which
class, so scene-dependent frames cannot be separated from the others.

The method's scene sentence describes **the place**. A rule such as "nobody
photographs here" is a fact about behaviour, usually not visible in the place.
So this benchmark can test where the method's boundary lies — prediction P5.

## Data and protocol (fixed)

| | |
|---|---|
| Split | official test split, 242 videos in 43 scenes (124 contain anomalies), 25 fps |
| Source | the authors' Google Drive release; only the 242 test videos are read out of the 106-volume archive by `notebooks/fetch_nwpu_campus_test.py`, each checked against the CRC-32 in the archive's own header (`data_manifests/nwpu_campus_test_members.json`) |
| Labels | the authors' `NWPU_Campus_gt.npz` (sha256 pinned in the manifest), one label per decoded frame |
| Sampling | every 16th frame (0.64 s), as on UCF-Crime and XD-Violence |
| Encoder | CLIP ViT-L/14 LAION-2B, whole frame, frozen |
| Prompt ensemble | `surveillance`, unchanged |
| Fusion | descriptor appended to the **normal** ensemble only |
| Window | w = 5 (3.2 s); w = 1 also reported |
| **Primary metric** | frame-level AUROC over all sampled frames pooled, **raw** scores — the paper's metric ("we concatenate all the frames in a dataset and then compute the overall frame-level AUC", §5.1) |
| Secondary | the same after per-video min–max normalisation |

**Why raw is primary:** 118 of the 242 test videos contain no anomaly, and the
paper describes plain concatenation; per-video min–max would force each normal
video to span [0, 1] (the same reasoning as on UCF-Crime).

## Conditions

| Condition | Descriptor |
|---|---|
| none | M3 off |
| generic | "a generic scene" |
| shared matched | "a university campus with roads, footpaths, squares and buildings" |
| **per-scene (generated)** | one sentence per scene: LLaVA-1.5-7B captions the first 3 sampled frames (decoded frames 0, 16, 32) of the scene's first test video by name, medoid kept; the caption instruction is ShanghaiTech's, unchanged (these are fixed cameras) |
| per-scene shuffled | every rotation of the 43 scene captions (42 rotations), each scene given another scene's sentence |
| mismatched | the industrial description used throughout |

Frames are chosen by position, never by label. Scenes whose captioned frames
already show an anomaly are counted and reported, and their captions kept.

## Predictions

Primary metric (raw pooled AUROC, w = 5) unless stated.

| # | Prediction | Refuted if |
|---|---|---|
| P1 | per-scene beats the shuffled mean | per-scene ≤ shuffled mean |
| P2 | per-scene beats no descriptor | per-scene ≤ none |
| P3 | shared matched beats mismatched | shared ≤ mismatched |
| P4 | the per-scene − none gap is **CALIBRATION-dominated** under `notebooks/measurement_audit.py`'s rule (level share ≥ 2/3 and \|within-video gap\| ≤ 1/3 of the gap) | any other verdict, including "no effect" |
| P5 | the per-scene gain over shuffled — raw AUROC within each scene, w = 5 — averaged over the four scene-dependent scenes, is **no larger** than the median gain over the other scenes | the four-scene mean exceeds the median |

P1–P3 repeat the ShanghaiTech and UCF-Crime tests on a new benchmark. P4 is a
second out-of-sample test of the calibration reading (the first is XD-Violence
P4). P5 is the boundary test: if a sentence about the place carried the
scene's *rules*, it would help most where the rules make the anomalies; the
calibration reading says it carries *which scene*, and should help there no
more than elsewhere. P5 has little power — four scenes, each with other anomaly
classes mixed in — and is reported as a direction whichever way it falls.

**No prediction is made about absolute AUROC against other methods.** For
reference, the paper's semi-supervised methods, trained on the training split
and evaluated on all frames (Table 4, verified in PROVENANCE.md): MemAE 61.9,
MNAD 62.5, OG-Net 62.5, HF²-VAD 63.7, MPN 64.4, AMMC-Net 64.5; the authors'
model 65.8 without and 68.2 with its scene-conditioning module.

## Known risks, stated in advance

- Full-HD frames are shrunk to 224×224; small or distant events (littering,
  jaywalking, forgetting a backpack) may be invisible to whole-frame semantics.
- The per-scene sentence comes from the first second of a test video; in some
  scenes an anomaly may already be under way (counted, kept).
- Faces are blurred in the release.
- Google Drive may refuse downloads for a while ("quota exceeded"); the fetch
  script waits and is resumable.

## Planned extension — not registered here

A descriptor of the scene's **normal activity**, written by LLaVA from the
scene's normal-only training footage, would put behavioural rules into the
sentence and is the natural follow-up if P5 holds. It will get its own dated
addendum, written before any training frame is captioned.

## Run order (server)

```bash
python notebooks/fetch_nwpu_campus_test.py                     # 21.1 GB packed
python notebooks/cache_embeddings.py --dataset nwpu_campus \
       --root ~/dazvad/data/nwpu_campus --frame-step 16 --crops 1
python notebooks/nwpu_campus_sweep.py caption                  # LLaVA, 43 scenes
python notebooks/nwpu_campus_sweep.py run                      # prints P1–P5
```

## Addenda

*(dated, below this line, after results)*
