# Guide Meeting Brief — Project Plan & Framework

> For the meeting on **2026-06-26**. Goal: align on the Sem-3/4 plan and show the framework
> design is ready to build. Full plan: [`00_PLAN.md`](00_PLAN.md). Read this once before you go in.

---

## 1. 60-second status (lead with this)

- **Literature survey done** — 15 papers; the gap is clear: existing VAD is training-heavy,
  closed-vocabulary, and gives scores without explanations.
- **CLIP baseline built & measured** — **88.5% image AUROC on MVTec AD, zero training data**
  (texture categories ~99%, objects ~83%). Real numbers, reproducible.
- **Video proof-of-concept done** — per-frame CLIP scoring separates normal vs anomalous
  frames (score gap 0.59), but exposes the gap: no temporal context, no explanations.
- **Architecture designed** — DA-ZVAD, a 4-module training-free pipeline (below).
- **This meeting:** lock the plan + start building the framework he asked for.

---

## 2. The plan in one breath

> A **training-free** vision-language + LLM pipeline for video anomaly detection, with
> **verbalized domain context** and **natural-language explanations**, evaluated **across
> industrial and surveillance domains**.

- **Sem 3 (Jul–Dec):** build the framework → multi-dataset results (MVTec + ShanghaiTech +
  Avenue), a large ablation grid, and an explanation gallery = a complete results chapter.
- **Sem 4 (Jan–May):** paper (workshop / PRL / Neurocomputing) + thesis. Trained temporal
  adapter is an *optional* upgrade, never a dependency.

---

## 3. The framework 

**DA-ZVAD — 4 modules, fully training-free in the default mode:**

| Module | Role | Source idea |
|---|---|---|
| **M1 — Frozen CLIP encoder** (ViT-L/14) | frame → visual embedding | OVVAD |
| **M2 — Temporal aggregation** | smooth per-frame scores over time (no training) | OVVAD (training-free variant) |
| **M3 — Verbalized context** | a text description of the scene grounds "what is normal" | VERA |
| **M4 — LLM reasoning** | flagged frames → anomaly score + natural-language explanation | LAVAD |

**Output:** per-frame anomaly score **+** a sentence explaining *why*.
**Key property:** to deploy to a new environment you change the **text description**, not the
weights — zero retraining. Runs on a single GPU (~7 GB).

**Planned code framework** (fresh `da_zvad/` package — built after your sign-off):

```
da_zvad/
├── config.py              # one config object drives every experiment
├── encoders/clip_encoder.py     # M1
├── temporal/aggregation.py      # M2  (training-free)
├── context/verbalized.py        # M3
├── reasoning/llm_reasoner.py    # M4  (optional)
├── prompts/templates.py         # per-domain prompt ensembles
├── datasets/  {mvtec, shanghaitech, avenue}.py
├── pipeline.py            # orchestrates M1 → M2 → M3 → M4
└── evaluation/  metrics.py (frame-level AUROC) + runner.py (ablation grid)
scripts/run_experiment.py
```

**Why this structure (say this if asked):** each module is swappable and can be turned on/off,
so the *same* framework produces the full ablation study by config alone — that is how the
results chapter gets its breadth without re-writing code.

---

## 4. Decisions to get his input on (your agenda)

1. **Datasets** — ShanghaiTech + CUHK Avenue as primary (small, standard frame-level AUC),
   MVTec for the cross-domain angle. UCF-Crime is 100+ GB and harder — keep optional? *(Get his view.)*
2. **Training-free spine** as the thesis, trained adapter as an optional Sem-4 stretch —
   agree, or does he want the learned adapter as a core contribution?
3. **Framing / venue** — explanation-led, empirical cross-domain study; workshop / mid-tier
   journal target. Aligned with his expectations?
4. **Framework scope** — confirm the `da_zvad/` package above is the "framework" he means
   (code pipeline), vs. a written design document.
5. **Review dates** — confirm the actual department mid-sem / end-sem review dates so the
   plan's milestones pin to them.

---

## 5. Likely questions he'll ask — and your answers

**"What's novel here?"**
> "A training-free pipeline that adds *explanations* and *domain adaptation via text* to
> zero-shot VAD, plus an empirical study of where it works and fails across industrial and
> surveillance domains. It's a study, not just a SOTA-chasing model."

**"Why training-free? Why not train a model?"**
> "Training needs domain-specific labelled data and a debugging-heavy training loop. Training-
> free lets the same framework move across domains by changing a text prompt — which is exactly
> the domain-adaptation claim. The trained temporal adapter is on the roadmap as an optional upgrade."

**"This is just LAVAD / VERA combined."**
> "Those are the building blocks — LAVAD for reasoning, VERA for verbalized context. The
> contribution is the unified training-free framework *and* the cross-domain empirical
> characterization; no single existing method gives training-free + explainable + domain-
> adaptive + a cross-domain study together."

**"Will it beat the state of the art on AUROC?"**
> "Not necessarily, and we don't claim that. The value is competitive detection *plus*
> explanations and domain adaptation at zero training cost. The first Sem-3 step is a quick
> probe to measure exactly where the zero-shot detector stands."

**"What have you actually built?"**
> "The CLIP baseline (88.5% on MVTec), the video proof-of-concept, the full architecture, and
> the framework design ready to implement. Next is scaffolding the `da_zvad/` package."

---

## 6. Questions to ask him

- Does he prefer ShanghaiTech/Avenue, or does he specifically want UCF-Crime / XD-Violence?
- Is the explanation-led, cross-domain framing the right thesis story, or does he want a
  stronger emphasis on detection accuracy?
- Any conference/journal he already has in mind for the final-semester paper?
- Confirm the review/milestone dates.

---

## 7. One honest note for yourself

If he pushes for the *trained* adapter as a core (not optional) contribution, that raises the
effort and debugging load significantly — flag your bandwidth honestly and propose it as a
Sem-4 item gated on progress. The training-free result stands on its own either way, so the
thesis is safe regardless of how this lands.
