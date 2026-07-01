# Research Plan — Semester 3 & 4

> **DA-ZVAD: Domain-Adaptive Zero-shot Video Anomaly Detection.**
> Canonical project plan. Last updated: 2026-07-02.

---

## 1. Thesis statement

> A **training-free** vision-language + LLM pipeline for video anomaly detection, with
> **verbalized domain context** and **natural-language explanations**, evaluated
> **across industrial and surveillance domains**.

The framework operates without task-specific training: a frozen CLIP encoder scores frames
against natural-language prompt ensembles; training-free temporal aggregation smooths the
score sequence; a verbalized scene description adapts the notion of "normal" to a new
environment with zero parameter updates; and an MLLM produces a natural-language explanation
for each flagged event.

---

## 2. Key milestones

| When | Milestone |
|---|---|
| Mid-July 2026 | Plan discussion with the research guide |
| **September 2026** | **Panel review — presentation (slides) + report (IEEE format) + demonstration** |
| Nov/Dec 2026 | End-semester review — complete results |
| Jan–Feb 2027 | Paper draft (target: workshop / mid-tier journal) |
| Mar–May 2027 | Thesis writing, submission, defense |

---

## 3. Architecture and implementation status

| Module | Role | Status |
|---|---|---|
| M1 — Frozen CLIP encoder (ViT-L/14) | frame → anomaly score via language prompts | ✅ implemented |
| M2 — Temporal aggregation | training-free smoothing of score sequences | ✅ implemented |
| M3 — Verbalized domain context | scene description grounds "what is normal" | ✅ implemented |
| M4 — LLM reasoning | natural-language explanation per flagged frame | 🔶 interface defined; implementation in progress |
| Datasets | MVTec AD, ShanghaiTech, CUHK Avenue adapters | ✅ / in progress |
| Evaluation | frame-level AUROC / AP / F1 + ablation runner | ✅ implemented |

Code: `da_zvad/` package (modular, configuration-driven — every module is a config toggle,
so the ablation study is produced by configuration alone). End-to-end smoke test:
`python -m da_zvad.demo`.

Foundation result (Semester 2): CLIP zero-shot baseline at **88.5% image-level AUROC on
MVTec AD** with zero training data.

---

## 4. Semester 3 work plan (working back from the September review)

| Phase | Period | Activities → Deliverable |
|---|---|---|
| A — Baseline measurement | early July | Zero-shot detector measured on ShanghaiTech (frame-level AUROC) → baseline number + prompt/smoothing comparison |
| B — Core results | mid-July–Aug | Detection results on two surveillance benchmarks + MVTec; first ablation table; first explanation examples (M4) |
| C — Review package | late Aug | IEEE-format report, slides, live demonstration script |
| D — Panel review | September | Presentation + demo |
| E — Full evaluation | Oct–Nov | Complete ablation grid (multi-seed), failure-case analysis, cross-domain consolidation → draft results chapter |

---

## 5. Datasets and evaluation

- **MVTec AD** — industrial images (15 categories); cross-domain reference point.
- **ShanghaiTech Campus** — surveillance video, frame-level ground truth; primary video benchmark.
- **CUHK Avenue** (or UCSD Ped2 as alternative) — second video benchmark.
- Metrics: image/frame-level **AUROC** (standard for these benchmarks), AP, best-F1;
  qualitative assessment of generated explanations.

---

## 6. Semester 4 work plan

1. **Paper** (Jan–Feb): written from Semester-3 results. Candidate venues: computer-vision
   workshop (CVPR/WACV), ICIP, or journals with rolling submission (Pattern Recognition
   Letters, Neurocomputing).
2. **Optional extension:** a lightweight learned temporal adapter (few-shot) behind the
   existing M2 interface — treated as an enhancement, not a dependency; the training-free
   results stand alone.
3. **Thesis** (Mar–May): Semester-3/4 results become the core chapters; the existing
   literature survey and proposal provide the remainder.

---

## 7. Risk management

- No training loops on the critical path — removes the largest schedule risk.
- The contribution is framed as a training-free framework **plus an empirical cross-domain
  study** (where zero-shot VLM detection works and where it fails) — informative regardless
  of absolute benchmark numbers.
- Dataset contingency: UCSD Ped2 substitutes for CUHK Avenue if sourcing is a problem;
  UCF-Crime, if requested, is handled as a subset evaluation.

---

## 8. Documentation convention

Each completed phase is accompanied by a plain-language technical note in
`docs/08_understanding/` (what was built, why, how it works, key terms). These serve as
the project's running technical documentation and as preparation material for reviews.

---

## 9. Related repository

A standalone demonstration of the underlying zero-shot technique (industrial defect
detection on MVTec AD with an interactive UI) is maintained separately at
`github.com/prakash-nitc/clip-anomaly-detection`.
