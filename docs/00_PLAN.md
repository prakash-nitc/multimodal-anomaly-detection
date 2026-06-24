# Research Plan — Semester 3 & 4 (DA-ZVAD)

> **Canonical planning document.** Supersedes the older `2nd_year_plan.md`.
> Last updated: 2026-06-25. Share this with the guide; revise after his sign-off.

---

## 0. One-line thesis

> *A **training-free** vision-language + LLM pipeline for video anomaly detection,
> with **verbalized domain context** and **natural-language explanations**, evaluated
> **across industrial and surveillance domains**.*

This is the de-risked **DA-ZVAD** spine: the fully training-free configuration is the
thesis; the trained temporal adapter is an optional Sem-4 upgrade, never a dependency.
Breadth (and the "this is a lot of work" impression) comes from **sweeps and ablations**,
not from re-implementing three papers' training loops.

---

## 1. Working context & constraints

- **Priority order:** placements first, research second — but Sem-3/Sem-4 grades depend
  on execution, and a paper is expected in the final semester.
- **Division of labour:** the assistant carries all code, experiments, debugging,
  analysis, plots, and writing. The user runs **batched, paste-ready GPU commands** on the
  college machine and returns the output, learns a small high-yield defense kit, and owns
  the guide/professor interface.
- **Non-negotiables:** results are **real** (the user actually runs them); every artifact
  ships with a plain-language "how to explain this" companion for the viva.
- **Compute:** college GPU available — compute is not the bottleneck; *debugging time* is.
  This is why the critical path avoids training loops.

**Effort key (used below):** 🟦 assistant produces · 🟩 user's bounded GPU step ·
🎓 guide checkpoint · ⭐ artifact that reads as "serious work" to the panel.

---

## 2. Semester 3 (Jul–Dec 2026) — Build the core result

**Goal:** end the semester with a complete, reproducible **results chapter** — multi-dataset
AUC + a large ablation grid + an explanation gallery. That chapter *is* the end-sem review
and doubles as the draft paper's results section.

### Phase 0 — Foundation + de-risk (Jul, wk 1–2)
- 🟦 Reproducibility scaffold: configs, fixed seeds, logging, a standard **frame-level AUC**
  evaluation harness (do once so Sem 4 is "write," not "rerun").
- 🟦 Dataset prep scripts for **ShanghaiTech** + **CUHK Avenue** (MVTec already in hand).
- 🟦 **De-risk probe:** naive per-frame CLIP scoring on ShanghaiTech.
- 🟩 Download datasets + run the probe → paste the number.
- **🚦 Decision gate:** probe ≥ ~75% → detection can be a co-headline; ~60s (likely) →
  explanations become the headline and detection is "supporting." Plan continues either way;
  only emphasis shifts. 🎓 Share the number with the guide.

### Phase 1 — Detector + temporal + context (Aug, wk 3–6)
- 🟦 Frame-level CLIP scoring on **3 datasets** (ShanghaiTech, Avenue, MVTec).
- 🟦 **Training-free temporal aggregation** (score smoothing) + window-size sweep.
- 🟦 **Verbalized context** (Module 3) — per-domain prompts + with/without ablation.
- 🟩 One batched run → CSVs.
- **Deliverable ⭐:** AUC tables on 3 datasets + temporal-window sweep plot + context ablation.

### 🎓 Mid-sem review (~Sep)
- 🟦 Slides + script built from the Phase-1 tables. Already reads as substantial
  (3 datasets × multiple configurations).

### Phase 2 — LLM reasoning + the big grid (Sep–Oct, wk 7–12)
- 🟦 MLLM (LLaVA / caption-then-reason) on flagged frames → **natural-language explanations**.
- 🟦 **Full ablation matrix:** {M1, M1+M2, M1+M3, M1+M2+M3, +M4} × 3 datasets × 3 seeds.
- 🟦 **Sweeps:** sampling FPS, CLIP backbone (ViT-B/16 vs ViT-L/14), threshold.
- 🟦 **Failure-case analysis** — curated examples of where it breaks and why.
- 🟩 One overnight batch run → CSVs.
- **Deliverable ⭐:** ~30–50-row ablation matrix · sweep plots · explanation gallery · failure cases.

### Phase 3 — Consolidate + end-sem (Nov–Dec, wk 13–16)
- 🟦 **Results chapter** draft (tables + plots + cross-domain analysis).
- 🟦 Updated viva defense kit + end-sem deck.
- **🎓 End-sem review:** defend the complete results story.

**Sem-3 deliverables:** reproducible code · AUC on 3 datasets · ~40-row ablation grid ·
3–4 sweeps with plots · multi-seed error bars · explanation gallery · failure analysis ·
results-chapter draft · viva kit.

---

## 3. Semester 4 (Jan–May 2027) — Paper + thesis

**Goal:** convert Sem-3 results into a submitted paper and a finished thesis. The trained
adapter is an optional upgrade gated on placement status — never a dependency.

### Phase 4 — Write the paper (Jan–Feb)
- 🟦 Full paper draft from Sem-3 results.
- **Target venue:** workshop / mid-tier — WACV/BMVC/ICIP workshop, or
  *Pattern Recognition Letters* / *Neurocomputing*.
- 🎓 Guide reviews + submits.

### Phase 5 — ⚠️ Stretch: trained temporal adapter (Feb–Mar, gated)
- **Only if placements wrapped:** lightweight OVVAD-style adapter trained on ShanghaiTech
  weak labels → one extra "learned DA" results row + ablation.
- **If not:** stays as thesis future-work (fully defensible). No dependency.

### Phase 6 — Thesis + defense (Mar–May)
- 🟦 Assemble thesis (paper → 2–3 chapters + extended analysis + existing survey).
- 🟦 Final deck + full viva kit + dry-run Q&A.
- **🎓 Thesis submission + defense.**

---

## 4. Decision points for the guide

1. **Datasets:** ShanghaiTech + Avenue primary, MVTec for cross-domain, UCF-Crime optional.
   (UCF-Crime is 100+ GB and harder — recommend against it as primary.)
2. **Training-free spine** as the thesis, trained adapter as a Sem-4 stretch — agree, or does
   he want the learned adapter as a core contribution? (Raises effort + a zero-shot-vs-trained
   tension that must then be defended carefully.)
3. **Framing / venue:** explanation-led empirical cross-domain study; workshop/mid-tier target.
4. **Confirm the real department review dates** so milestones pin to them.

---

## 5. Novelty framing (say it precisely)

Lead claim is a **training-free, verbalized-context, explainable VAD pipeline + an empirical
study of where zero-shot VLM AD works and fails across domains** — a *study*, not a SOTA bet,
so it cannot be proven wrong in a viva. The "first method satisfying all five desiderata
(zero-shot, training-free, explainable, temporal, domain-adaptive)" table is a *summary*, not
the primary claim — a reviewer who knows VERA (CVPR 2025) will otherwise read it as
"VERA + a temporal adapter."

Domain adaptation here is **training-free / source-free / test-time** (adapt "what is normal"
via a text description, zero parameter updates) — *not* learned/adversarial DA. State it that
way to pre-empt the "where is your domain aligner?" question.

---

## 6. Risk posture

The trained adapter is a bonus, never a dependency. A bad placement season cannot sink the
thesis or paper, because the training-free cross-domain result stands on its own.
**Publication calibration:** "reputable" = workshop / mid-tier conference / journal (a normal
target for a strong MTech thesis), not CVPR/ICCV main track.

---

## 7. Parallel track — resume ML project (placement double-dip)

A productized slice of the research, **not** a separate project. Lives in its own repo at
`p:\Research\clip-anomaly-detection\` (`app.py`, `benchmark.py`, `README.md`). Already has a
real headline: **88.5% image-level AUROC on MVTec AD, zero training data** (within ~4 pts of a
trained OC-SVM, beats it on all texture categories). Remaining steps are the user's: create a
Hugging Face account, deploy the Gradio app to Spaces → public link for the resume.
(An older duplicate exists at `clip-zero-shot-anomaly/` — `clip-anomaly-detection` is canonical.)
