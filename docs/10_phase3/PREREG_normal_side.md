# Pre-registration — normal-side prompt banks and retrieved normality

**Approved and committed before any score under these conditions existed.** The
commit timestamp of this file is the evidence. The run records the SHA-256 of this file, of
`da_zvad/prompts/banks.py`, of `da_zvad/context/retrieval.py` and of
`notebooks/normal_side_sweep.py`, so any later edit is visible. Nothing below may
be edited after results exist; corrections go in a dated addendum at the bottom.

## Why

Three findings from Phase 3 motivate this run.

1. **The generated sentence can be wrong in the worst way.** On XD-Violence, a
   per-video LLaVA sentence lowered AP from 0.624 (no sentence) to 0.417. In 127
   of the 500 violent videos, the violence had already begun in the three frames
   LLaVA captioned, so the "description of normal" may describe the anomaly.
2. **The normal side is where training-free methods now gain most.** NOVA
   (arXiv 2609.06360, Sep 2026) reports its largest ablation gain from richer
   normal-side prompts ("confusing" look-alike actions, calm scenes). Our
   ensembles have three generic phrases per side.
3. **Our gains are mostly calibration** (measurement audit, 23 Oct 2026). The
   same audit is applied to every new component here.

The question is practical: can a camera be described safely and cheaply
enough to use on day one at a real site? Retrieval answers "describe" by
*choosing* from vetted phrases instead of *writing*, so it cannot name an
anomaly as normal.

## What was known before writing this

- Every legacy number on all five datasets, including the XD-Violence and NWPU
  Campus sweeps of 1 Nov 2026.
- **August 2026, ShanghaiTech:** the dataset's own anomaly vocabulary ("a person
  riding a bicycle on a pedestrian walkway", …) did worse than the crime-oriented
  `surveillance` ensemble (0.486 alone; below it when mixed in; `scoring_lab.csv`).
  This informs prediction P5a and the rule that worry phrases do not name the place.
- **Smoke check on 3 Nov 2026** (laptop, `try_retrieval_on_photo.py`) on ten
  images in `figure_assets/`: two web photos, three Avenue frames and five
  ShanghaiTech frames from two views. The eight dataset frames are test frames,
  labelled normal or anomalous in their file names.
  - Adding the five retrieved phrases *raw* to the normal side pushed every
    score towards zero, and the anomalous Avenue frame lost its top rank.
  - Passing the top phrase through the two M3 templates kept the ranking.
  - The design was changed to the templated top phrase **for parity with the
    validated LLaVA injection**. The raw-five design is kept as a registered
    secondary condition (`FULL_R5raw`), so the effect of this choice is reported
    either way.
  - No other setting was looked at on any image.

## Data and protocol (fixed)

| Dataset | Cache (CLIP ViT-L/14 LAION-2B, whole frame) | Unit for retrieval | Window | Primary metric |
|---|---|---|---|---|
| ShanghaiTech (107 clips) | step 2 | camera view (12) | 31 | pooled AUROC, per-clip min–max |
| CUHK Avenue (21 clips), *if its cache is built* | step 2 | the one camera | 31 | pooled AUROC, per-clip min–max |
| UCF-Crime (290 videos) | step 16 | video | 5 | pooled AUROC, raw |
| XD-Violence (800 videos) | step 16 | video | 5 | AP as the authors' `test.py` |
| NWPU Campus (242 videos) | step 16 | scene (43) | 5 | pooled AUROC, raw |

Everything else is unchanged from the earlier pre-registrations:
- λ = 100;
- the `surveillance` base ensembles;
- M3 normal-side fusion;
- the same captions for the LLaVA conditions.

XD-Violence AP is computed from per-snippet counts of positive and negative
frames. This is mathematically identical to repeating each score 16 times, and
the self-test checks it to 1e-12. Its AUROC and step-wise AP
(`average_precision_score`) are reported as secondary metrics.

## Components (all training-free, all on cached embeddings)

| | Component | Definition |
|---|---|---|
| **W** | Worries | The benchmark's published anomaly classes as event phrases, added to the abnormal side (`banks.WORRIES`): UCF-Crime 13, XD-Violence 6, ShanghaiTech 6, Avenue 3, NWPU 28. Only the **dataset-level** taxonomy is used, never NWPU's per-scene list, because that would tell the detector what each camera's test videos contain. Phrases name the event, not the place. |
| **N** | Near-miss + calm | Look-alike normal actions written from the class names only (`banks.NEAR_MISS`: 13 / 6 / 6 / 4 / 19), plus four calm anchors shared by all datasets, added to the normal side. |
| **P** | Per-phrase scoring | Margin = log-mean-exp over abnormal phrases − log-mean-exp over normal phrases (λ = 100, each side normalised by its size), instead of one mean prototype per side. With one phrase per side it equals the legacy rule (self-test). |
| **E** | Embedding window | Centred moving average of 5 sampled frames' unit embeddings (edge padding), renormalised, before scoring. Score smoothing (window above) is unchanged. |
| **R** | Retrieval | Per unit, the warm-up video is the unit's first test video by name (for UCF and XD, the video itself), as LLaVA used. The steps: (1) of its first `n_warm` sampled frames (32 at step 16; 256 at step 2, about 512 decoded frames either way), take the 5 with the lowest legacy no-sentence score; (2) average and renormalise their embeddings; (3) rank the 207-phrase library (`banks.library()`, "&lt;place&gt; with &lt;activity&gt;", no banned word) by cosine similarity; (4) **self-check:** if the best phrase is no more similar than the best of three neutral phrases ("a photo of a place", …), use no sentence; (5) otherwise the best phrase enters the normal side through the two M3 templates, exactly as LLaVA's sentence did. |

**FULL** = W + N + P + E + R.

## Conditions

| Name | What |
|---|---|
| legacy_none, legacy_shared, legacy_best | Today's system through the old code path: no sentence, the shared sentence, the LLaVA sentence (per view, video or scene; Avenue has none). **Gates.** |
| legacy_R | Today's system with LLaVA's sentence replaced by the retrieved one. The like-for-like test of "describe". |
| FULL; FULL−W, −N, −P, −E, −R | All components; each left out in turn |
| ladder_W, ladder_WN, ladder_WNP | Components added one at a time (for presentation) |
| FULL_llava, FULL_shared | FULL with R replaced by LLaVA's sentence or by the shared sentence, same templates |
| FULL_Rperm, legacy_Rperm | Control: each unit gets another unit's retrieved sentence. 20 derangements (seed 20261104) for per-video datasets; every rotation for views and scenes. The bootstrap uses the first 5. |
| FULL_R5raw | Secondary: the five raw retrieved phrases instead of the templated top one |
| FULL_nocampus, legacy_R_nocampus | Campus datasets only: every campus-tagged place removed from the library (207 → 170 phrases). Tests what happens when the library does not know the scene. |

## Validation gates (nothing new is reported for a dataset that fails)

1. The new scoring code, with all components off, reproduces the legacy margins
   (max |difference| ≤ 1e-5).
2. legacy_none, legacy_shared and legacy_best reproduce the saved sweep JSON
   values (tolerance 6e-4) where they exist, otherwise the published values
   (tolerance 1.5e-3). These are:
   - ShanghaiTech: 0.7067 / 0.7336 / 0.751;
   - Avenue: 0.706 / 0.677;
   - UCF-Crime: 0.756 / 0.813 / 0.824;
   - XD-Violence and NWPU Campus: their 1 Nov sweep JSONs.
3. No retrieved phrase contains a banned word (asserted).

## Statistics

Paired video-level bootstrap: 1,000 resamples, seed 20261104, with 95%
percentile intervals for every comparison below. The resampled unit is the video
(clip), as in the measurement audit. The audit's level/shape split, with its
fixed decision rule, is applied to FULL vs legacy_none, FULL vs FULL−R and
legacy_R vs legacy_none.

Comparisons:
- FULL − legacy_none and FULL − legacy_best;
- legacy_R − legacy_none and legacy_R − legacy_best;
- FULL − FULL_llava and FULL − FULL_shared;
- FULL − FULL−c for each component c;
- FULL − FULL_Rperm and legacy_R − legacy_Rperm;
- FULL − FULL_R5raw;
- FULL_nocampus − FULL and legacy_R_nocampus − legacy_R.

## Keep or drop (decides the final system)

A component c is **kept** only if all three conditions hold for FULL − FULL−c
across the datasets that pass their gates:
- the point estimate is above 0 on more than half of them;
- the 95% CI lies above 0 on at least 2;
- the 95% CI lies below 0 on none.

Otherwise it is dropped. The **final system** is FULL minus the dropped
components. Its numbers are reported as *selected by this rule on the same
data*, never as an independent test. If every component is dropped, the final
system is today's system with no sentence, and that is the result.

## Leave one dataset out

For each dataset d, the rule above is applied to the other datasets only, and
the components it keeps are scored on d against legacy_none and legacy_best,
with the same bootstrap. This checks the *decision rule* on data it did not see.
It does **not** make the phrases independent of d, because they were written
knowing all five.

## Stronger frozen encoder

Run after the ViT-L/14 sweep, because ViT-L's keep/drop table fixes the recipe;
a new encoder does not choose again.

- **Encoder:** `PE-Core-L-14-336` (open_clip, weights `meta`), the backbone NOVA
  uses. `ViT-H-14` (`laion2b_s32b_b79k`) is the fallback if PE-Core's weights
  cannot be fetched on the server.
- **Fixed:** λ = 100 for comparability (the learned scale is logged), whole frame,
  all conditions and windows as above, the same LLaVA sentences.
- **Validation:** no published numbers exist for this encoder, so only the
  code-path gates apply. Results are compared with ViT-L by a paired video-level
  bootstrap on identical videos.
- **Audit:** the level/shape split is repeated on the new encoder, to test whether
  the calibration finding depends on the backbone.

## Not in this run

**Multi-crop.** On ShanghaiTech, with the same scoring rule (`margin_pooled`), the
same prompts and w = 31, the whole frame scores 0.707 (`sweep_whole2.csv`), while
aggregating over the whole frame and four quadrants scores 0.558 (mean) or 0.567
(max) (`scoring_lab2.csv`). Three full-height square crops keep the frame's
scale, so they are a different test, and are left for later.

## Sealed test (later, on data nobody has touched)

None of the five datasets can show that this generalises. Once this run is
finished and the recipe is frozen, it runs **once** on a benchmark not used in
any design decision.

- **Candidates:** UBnormal (public; 29 virtual scenes, so synthetic) and MSAD (14
  real scenarios; its videos need a request form).
- **Two versions, both reported:** with the benchmark's published classes as
  worries and near-misses, and with none, because a real site may have no list.
- **Protocol:** loader, metric and predictions go in a dated addendum committed
  **before** any of its videos is scored. No test video or label from it may be
  viewed before then, except to check the loader's file structure.

## Predictions

| | Prediction | Why |
|---|---|---|
| P1 | FULL beats legacy_none on all datasets but at most one | The richer normal side was NOVA's largest gain |
| P2 | On XD-Violence, legacy_R beats legacy_best (AP) | The library cannot describe violence as normal |
| P3 | Own retrieved sentence beats another unit's (FULL − FULL_Rperm > 0) on at least 3 of the multi-unit datasets | Retrieval carries scene information, as LLaVA's sentence did (+0.110 on UCF-Crime) |
| P4 | The retrieval gain (FULL vs FULL−R) is never RE-RANKING-dominated | Every sentence effect measured so far was calibration or mixed |
| P5a | ShanghaiTech: FULL − FULL−W ≤ 0 | The August campus-vocabulary result |
| P5b | UCF-Crime and XD-Violence: FULL − FULL−W > 0 | Their classes are the crimes and violence the base ensemble already targets |
| P6 | Removing campus phrases lowers FULL on ShanghaiTech and NWPU | Retrieval there relies on the campus places |
| P7 | With the same recipe, the stronger encoder beats ViT-L on at least 3 datasets | NOVA reports high numbers on PE-Core, though its setup differs in other ways too |

Honest expectation:
- small gains on most datasets from the text changes;
- most of the XD-Violence loss recovered;
- any large jump coming from the encoder, not the text.

## Exploratory, not tests

- `xd_contamination` asks whether the XD-Violence loss comes from the 127 videos
  whose captioned frames already show violence, or from captions containing a
  banned word. It does this by removing the sentence from only those videos. It
  uses labels and is reported as a diagnosis.
- Fallback rates, the retrieved sentence per unit, and contact sheets of calm
  frames beside their phrases (`review`).

## Known limitations

- ShanghaiTech's worry list comes from the repo's August `campus` ensemble, with
  place words removed. It has not been checked against the dataset's own paper
  (Luo et al. 2017), which is not in `docs/07_reference_papers/`.
- Avenue's classes come from Liu et al. 2018's one-line description, not
  Avenue's own paper.
- The warm-up uses each test video's opening frames, which is transductive, as
  LLaVA's captioning was. In deployment these would be setup footage.
- The phrase library and near-miss lists were written for this study. Other
  wordings would give other numbers. They are fixed here so that no wording is
  tuned on results.
- **This run does not test generalisation to new data.** All five datasets
  influenced the design: retrieval was motivated by the XD-Violence failure, and
  the library was written knowing these datasets (campus places, and film and
  television places for XD-Violence). Worries and near-misses come from each
  benchmark's published classes, which a real site would not have.

## Run

```bash
cd ~/dazvad/repo && git pull && source ~/dazvad/venv/bin/activate
python notebooks/normal_side_sweep.py selftest
# optional, enables Avenue (~5 min):
python notebooks/cache_embeddings.py --dataset avenue --root ~/dazvad/data/avenue --frame-step 2 --crops 1
python notebooks/normal_side_sweep.py run 2>&1 | tee ~/dazvad/logs/normal_side.log
python notebooks/normal_side_sweep.py xd_contamination
python notebooks/normal_side_sweep.py review
```

Then the stronger encoder (encode once per dataset, then score):

```bash
for spec in "shanghaitech 2" "avenue 2" "ucf_crime 16" "xd_violence 16" "nwpu_campus 16"; do
  set -- $spec
  python notebooks/cache_embeddings.py --dataset $1 --root ~/dazvad/data/$1 --frame-step $2 \
      --crops 1 --clip-model PE-Core-L-14-336 --clip-pretrained meta
done
python notebooks/normal_side_sweep.py run --backbone pe-l 2>&1 | tee ~/dazvad/logs/normal_side_pe.log
python notebooks/normal_side_sweep.py backbones --backbone pe-l
```

Outputs, under `~/dazvad/work/`:
- `tables/normal_side_<dataset>[_pe-l].json`
- `tables/normal_side_summary[_pe-l].json`
- `tables/normal_side_scores_<dataset>_<backbone>.npz`
- `tables/normal_side_backbones_pe-l.json`
- `tables/normal_side_xd_contamination.json`
- `review/review_<dataset>.jpg`

## Addenda

*(none yet)*
