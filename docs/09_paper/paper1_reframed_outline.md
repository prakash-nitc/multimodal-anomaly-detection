# Paper 1, reframed — outline

*Draft, 7 November 2026. A new outline: `paper1_concept_shift.tex` is unchanged
and stays as the full record. Every number below is already measured and appears
in that draft or in a committed result file; none is new. Items marked
**[optional]** need work we have not done yet.*

---

## 1. The decision in one paragraph

The current Paper 1 reads as a **method paper**: it proposes a scene description as
the way to adapt a frozen detector. Its own results then show the method mostly
calibrates, does nothing on one-camera data and NWPU Campus, breaks XD-Violence, and
is replaced in Paper 2. Method-focused reviewers will reject that. The reframed paper
is an **analysis paper**: it asks *what scene text actually does* in a training-free
detector, answers with controls designed to fail, and turns the answers into **rules
for anyone who builds or evaluates such detectors**. Nothing has to be re-run. What
changes is the question, the order, and about half the length.

| | Current draft | Reframed |
|---|---|---|
| Question | "Can a scene description adapt a frozen detector?" | "What does scene text do in a training-free detector, and how should its effect be measured?" |
| Main claim | The description adapts the detector | Scene text is load-bearing, but it mostly **calibrates cameras**, its effect depends on **where it enters**, and it **fails** when written from abnormal footage |
| Failures (Avenue, NWPU, XD) | Limitations at the end | **Results**, each one a finding with a cause |
| Contribution type | A method | Findings + a measurement audit others can reuse |
| Comparison with other methods | A large SOTA table and long discussion | One small table "for scale"; no ranking claims |
| Length | ~2,600 lines of LaTeX | Aim for roughly half; check the target journal's limit |

---

## 2. Title options

1. **What Scene Text Does in Training-Free Video Anomaly Detection: Calibration, Injection, and Failure** *(recommended: says the three findings)*
2. Is the Scene Description Doing the Work? A Falsification-First Audit of Language-Conditioned Video Anomaly Detection
3. Scene Text Calibrates, It Does Not Detect: Measuring Language Conditioning in Frozen Video Anomaly Detectors

---

## 3. The five findings (the spine of the paper)

Each finding is one results subsection. The order matters: each one raises the
question the next answers.

| # | Finding | Key evidence (already measured) |
|---|---|---|
| F1 | **Scene text is load-bearing**, and a deliberately wrong text proves it | ShanghaiTech: wrong description −0.105 (0.734 vs 0.628). UCF-Crime: own description beats other videos' by +0.110 [+0.069, +0.157]; others' fall *below* no text (0.713 vs 0.756) |
| F2 | **Where the text enters matters more than what it says** | Added to both prompt sets, the gap is −0.029; normal set only, +0.105. Mechanism: prototype angle 35.7° → ~26° (both) vs 41.9° (normal only); accurate text also drags both prototypes onto the images (+0.067 vs +0.014 generic, −0.017 wrong) |
| F3 | **It mostly calibrates cameras rather than re-ranking frames** | Pre-registered audit: 82%, 83%, 87% of the headline gaps are per-video level shifts; within-video change ≤ +0.024. Per-camera text is the exception (34%, mixed). A label-free per-video centring does *not* reproduce it (0.657 ShT, 0.702 UCF) |
| F4 | **It needs several scenes to help** | One camera (Avenue): gap +0.020, correct text below none (0.677 vs 0.706). Inside one ShanghaiTech view: +0.033 [−0.005, +0.070], excluding the pooled +0.105. But ~290 scenes (UCF) give +0.096, no more than 12: the pre-registered "grows with scenes" prediction was **refuted** |
| F5 | **It fails when written from abnormal footage** | XD-Violence: 0.624 → 0.417 AP, −0.207 [−0.261, −0.155]. In 127 of 500 violent videos the first three frames already show violence; removing the text there alone recovers 66% (0.417 → 0.554), removing it where captions contain violent words recovers 7%. NWPU Campus: no effect (+0.008 [−0.025, +0.043]) although own beats other scenes' (+0.066) |

**Plus one measurement finding that applies to everyone (M1):** pooled AUROC from
CLIP softmax scores can show large differences that ranking inside a video does
not. The same ShanghaiTech gap is +0.017, +0.022, +0.047, +0.105, +0.133 at
λ = 1, 10, 30, 100, 300; and raw pooling gives 0.519 where per-clip normalisation gives
0.707 for identical scores. This is the paper's most reusable result.

---

## 4. Contributions (as they would be written)

1. **A falsification-first protocol** for language-conditioned, training-free
   detection: a frozen pipeline in which the text is the only free quantity, and
   controls (generic, mismatched, shuffled) that can refute an adaptation claim.
2. **Three findings about what scene text does** (F2–F4): injection point dominates
   wording, gains are mostly calibration, and the benefit needs several scenes
   (with a pre-registered prediction that failed).
3. **A diagnosed failure** (F5): text written from a camera's opening footage
   inherits that footage, and on XD-Violence describes the anomaly as normal.
4. **A measurement audit** others can reuse (M1): an exact split of a pooled gain
   into a per-video level part and a remainder, with a decision rule fixed in
   advance, and evidence that softmax-scale pooling inflates text effects.

*Dropped as a contribution:* "the characterisation of concept shift". It stays as
motivation (one paragraph and one table), because a reviewer will read it as framing,
not as a result.

---

## 5. Draft abstract (≈190 words)

> Training-free video anomaly detectors score frames against text with a frozen
> vision-language model, and several derive that text, or a normal reference, from
> the target video itself.
> We ask what such scene text actually does. We build a frozen detector in which a
> one-sentence scene description is the only free quantity, and test it with
> controls designed to fail: generic, mismatched and shuffled descriptions. On five
> benchmarks, four findings emerge. The text is load-bearing: a wrong description
> costs 0.105 AUROC on ShanghaiTech, and on UCF-Crime a video's own description beats
> other videos' by 0.110 [0.069, 0.157]. Where it enters matters more than what it
> says: added to both prompt sets, it reverses its own effect. It mostly calibrates:
> a pre-registered audit attributes 82–87% of its gains to per-video score shifts,
> not to better ranking inside a video, and the measured effect grows sixfold with
> the softmax temperature. And it fails when written from abnormal footage: on
> XD-Violence, descriptions captioned from opening frames that already show violence
> lower average precision from 0.624 to 0.417. We conclude with rules for building and
> evaluating language-conditioned detectors.

*(+0.017 → +0.105 is "sixfold"; check the wording against the final numbers.)*

---

## 6. Section-by-section plan

Word budgets assume a journal article of roughly 7,000–8,000 words. Scale to the
chosen journal's limit.

### 1. Introduction (~700 words)
- Training-free VAD conditions on text; several recent methods derive the text or a
  normal reference from the target video itself (LAVAD's frame captions, NOVA's visual
  anchor), while others fix it in advance (AnyAnomaly, ASK-Hint, Flashback).
- The open question: *does the scene text do what it is assumed to do?* Nobody tests it
  with a control that could fail.
- Concept shift in one paragraph, as motivation: normality depends on the place, so
  telling the model the place is the natural idea.
- Preview F1–F5 and M1 in one sentence each; contributions list (Section 4 above).
- **New closing sentence:** the findings point to what a better text should be, which
  later work can test. *(No details of Paper 2 here; one forward pointer only, so the
  papers do not overlap.)*

### 2. Related work (~700 words, one table)
- Training-free VAD and where its text comes from (short; Paper 2 has the long version).
- Domain adaptation and concept shift: two paragraphs, condensed from the current
  sections 2.2–2.4 and 3.
- Evaluation practice in VAD: pooled vs per-clip AUROC, micro vs macro. *(Currently
  scattered; bring together, because M1 builds on it.)*
- **One** compact table replacing the three literature tables: method, what text it
  uses, whether the text depends on the target video, what was tested.

### 3. Setting and protocol (~900 words)
- 3.1 The frozen detector, one equation for the score and one for smoothing
  (edge-padded, as implemented). CLIP ViT-L/14, three normal and three abnormal
  phrases, λ = 100.
- 3.2 Where the text enters: the two fusion rules (both sets / normal set only).
- 3.3 The controls: none, generic, matched, mismatched, shuffled (Table: one row each
  with its purpose). Which outcome would refute the claim.
- 3.4 Where descriptions come from: by hand, and by LLaVA from the first three frames,
  label-free.
- 3.5 Benchmarks: ShanghaiTech (12 views), CUHK Avenue (1), UCF-Crime (~290 scenes),
  XD-Violence, NWPU Campus (43 scenes, scene-dependent anomalies). One table.
- 3.6 Metrics and the audit: AUROC (micro, macro), AP, video-level bootstrap, and the
  **level/remainder decomposition** written out as an equation with its decision rule.
  *(Currently buried in Results; promote it to Method because it is a contribution.)*
- 3.7 Pre-registration: which predictions were committed before which runs (UCF-Crime
  P1–P4; the audit's rule).

### 4. Findings (~3,000 words; the bulk)
- **4.1 F1: the text is load-bearing.** ShanghaiTech sweep (normal-only row) and
  UCF-Crime own vs others'. Figure: the one-clip curves (`fig_context_effect`).
- **4.2 F2: injection point dominates wording.** Fusion table with angles; direction of
  collapse. Figure: `fig_chart_sweep`.
- **4.3 F3: mostly calibration.** Audit table; centring control; the per-camera
  exception. **[new figure, no new run]** a bar chart of level part vs remainder for
  each comparison, drawn from `measurement_audit.json`.
- **4.4 F4: several scenes needed, but not "more is better".** Avenue, within-view,
  per-camera vs shuffled, UCF-Crime P1 refuted. Figures: `fig_chart_within_view`,
  `fig_chart_per_camera`.
- **4.5 F5: failure from abnormal footage.** XD-Violence diagnosis (127/500; 66% vs
  7%); NWPU Campus as "scene identity without a gain". Figure: ROC/PR curves
  (`fig_roc_curves`).
- **4.6 M1: what the metric does to text effects.** Raw vs per-clip pooling
  (0.519 vs 0.707); the λ sweep. Figure: `fig_camera_baselines`, plus **[new figure,
  no new run]** gap vs λ, pooled vs within-video, from `measurement_audit.json`.

### 5. Rules for language-conditioned detectors (~600 words) — *new section*
Turn the findings into advice. This is what makes an analysis paper useful.
1. Put scene text on the **normal side only**; text shared by both sides cancels.
2. **Always include a mismatched or shuffled control**; without it, an effect cannot
   be attributed to the text.
3. **Report macro AUROC and a level/remainder split** alongside pooled AUROC; pooled
   softmax AUROC can inflate text effects.
4. **Never write scene text from footage that may contain the event**; opening
   frames are not guaranteed normal.
5. **Expect little benefit on single-scene deployments**; the text mainly tells
   scenes apart.

### 6. Limitations (~400 words)
One encoder for the descriptor experiments; whole-frame scoring and the centre crop;
selection on test splits (development partitions on ShanghaiTech); XD diagnosis is
exploratory (uses labels); caption quality (LLaVA ignored the instruction in 6 of 12
first-run captions); the explanation stage is not evaluated.

### 7. Conclusion (~250 words)
Scene text is load-bearing but fragile; it calibrates more than it detects; and it
must not depend on what the camera happens to show first.

---

## 7. What moves out (to the thesis or nowhere)

| Current content | Where it goes | Why |
|---|---|---|
| MVTec AD baseline | Thesis | Image anomaly detection; distracts from the video question |
| M4 explanation module, algorithm box | Thesis (one sentence remains) | Not evaluated; invites "why is it here?" |
| Three literature tables | One compact table | Length; Paper 2 carries a full literature table |
| Long comparison with AnyAnomaly, LAVAD and the SOTA table | One small "for scale" table, no ranking | An analysis paper should not compete on scores |
| "Six things that did not work" | One table in the setting section | Supports "the minimal configuration", not a finding |
| Design-rationale and implementation prose | Half a paragraph | Length |
| Concept-shift observations 1–3 and gaps G1–G4 | One paragraph + one table in the introduction | Framing, not a result |

---

## 8. Figures and tables

| Item | Status |
|---|---|
| Pipeline diagram | **New, simpler** version without the explanation branch (redraw from `make_architecture_paper1.py`) |
| `fig_context_effect`, `fig_chart_sweep`, `fig_chart_within_view`, `fig_chart_per_camera`, `fig_roc_curves`, `fig_camera_baselines`, `fig_dataset_samples` | Exist; reuse |
| Audit decomposition (level vs remainder per comparison) | **New figure** from `measurement_audit.json`; no new run |
| Gap vs λ, pooled vs within-video | **New figure** from `measurement_audit.json`; no new run |
| `fig_comparison`, `fig_chart_ablation`, `fig_detection_example`, `fig_chart_ucf_crime` | Drop or move to the thesis |
| Tables: controls, benchmarks, fusion + angles, audit, within-view/scenes, UCF-Crime bootstrap, XD/NWPU, compact related work, small "for scale" | ~9 tables, down from 23 |

---

## 9. Keeping Paper 1 and Paper 2 apart

| Paper 1 (reframed) contains | Paper 2 contains |
|---|---|
| CLIP ViT-L/14 with scene descriptions only | PE-Core, calm phrases, the bank |
| The audit as a method, applied to descriptions | The audit applied to calm phrases (cites Paper 1) |
| The XD failure and its diagnosis | The XD failure in two sentences, citing Paper 1 |
| Rules for language-conditioned detectors | The fixed text prior and the sealed test |
| No sealed test, no PE-Core numbers | All PE-Core and UBnormal numbers |

The only shared numbers are the CLIP baselines (0.707, 0.756, 0.624, 0.659), which
Paper 2 needs as its starting point and cites.

---

## 10. Likely reviewer objections, and the answer each section gives

| Objection | Answer in the paper |
|---|---|
| "The method doesn't work, so why publish?" | It is not a method paper: the contribution is what scene text does and how to measure it (Sections 1, 5) |
| "Only one encoder" | Stated as a limitation; **[optional]** add the descriptor results on PE-Core from the normal-side run, if they hold up when re-checked |
| "Calibration is useful too" | Agreed, and said: one alarm threshold across many cameras needs it (Section 4.3); the point is that it is not what the method claims |
| "The XD analysis uses labels" | Labelled exploratory; the main XD result (0.624 → 0.417) uses none |
| "Scores are below state of the art" | No ranking claim; one table for scale only |
| "Pre-registration of a few predictions is not full pre-registration" | Say exactly which predictions were committed when, and that one failed |

---

## 11. Optional additions that would strengthen it

- **[optional] Second encoder.** The November normal-side run scored description
  conditions on PE-Core too (`results/runs/analysis/normal_side_*`); our notes say the
  XD rescue by retrieval and the word flip did *not* replicate there. If re-checked,
  this becomes an honest "the findings are encoder-dependent" paragraph. No GPU needed
  if the stored results suffice.
- **[optional] Audit on XD-Violence and NWPU Campus in the same table.** The XD loss is
  already measured as 69% level; NWPU would complete the table.

---

## 12. Venue and next steps

- **Venues to discuss with the guides:** an analysis paper fits journals that publish
  empirical studies and evaluation methodology (for example Pattern Recognition
  Letters, Image and Vision Computing, Neurocomputing), or a workshop on video
  understanding or evaluation. Check each one's word limit and AI policy first.
- **Order:** Paper 2 first or together; Paper 2 cites this one.
- **Steps once the guides agree:**
  1. Write the new LaTeX file from this outline (`paper1_analysis.tex`), reusing
     existing text where it fits.
  2. Draw the two new figures and the simpler pipeline diagram.
  3. Cut to the word limit.
  4. You rewrite it in your own words, section by section.
