# Provenance ledger — every external number in the paper

**The rule: no figure from another paper enters `main.tex` until it is listed
here with the source PDF in this folder and a location inside that PDF.**

This exists because two wrong numbers reached the draft in September 2026, both
from the same cause — a figure taken from a web search summary rather than read
out of the paper it belongs to:

- **LAVAD was compared against on ShanghaiTech.** LAVAD reports on UCF-Crime and
  XD-Violence and on neither benchmark used here. The comparison was invalid and
  the number (`≈0.85`) matched no figure LAVAD reports.
- **Four one-class rows** (MemAE, MNAD, HF²-VAD, RTFM) came from a search
  summary of some third paper's table. They were removed rather than trusted.

Both would have been caught by opening one PDF.

`scripts/check_provenance.py` cross-checks the comparison table in `main.tex`
against this file and fails on any number that is missing, mismatched, or marked
unverified. Run it before every upload to Overleaf.

---

## Source PDFs required in this folder

| Key | File | Supplies |
|---|---|---|
| `anyanomaly` | `AnyAnomaly_WACV2026.pdf` | every comparison row in Table `tbl:sota` |
| `shanghaitech` | `Liu2018_FutureFramePrediction_CVPR.pdf` | the benchmark's own baseline |
| `lavad` | `LAVAD_CVPR2024.pdf` | the UCF-Crime / XD-Violence figures in Limitations |

---

## Ledger

`status` is `verified` only when someone has opened the named PDF at the named
location and seen the figure. `unverified` means the number is in the draft on
weaker evidence and must be either confirmed or removed.

| Value | Metric | Method | Source | Location in source | Status |
|---|---|---|---|---|---|
| 85.1 | Avenue AUROC | Liu et al. 2018 | `shanghaitech` | Table 1, p.6, row "Our proposed method", CUHK Avenue column | verified |
| 72.8 | ShT AUROC | Liu et al. 2018 | `shanghaitech` | Table 1, p.6, row "Our proposed method", ShanghaiTech column | verified |
| 89.5 | Avenue AUROC | MPN | `anyanomaly` | Table 5, row MPN[21] | verified |
| 73.8 | ShT AUROC | MPN | `anyanomaly` | Table 5, row MPN[21] | verified |
| 90.1 | Avenue AUROC | FPDM | `anyanomaly` | Table 5, row FPDM[34] | verified |
| 78.6 | ShT AUROC | FPDM | `anyanomaly` | Table 5, row FPDM[34] | verified |
| 91.3 | Avenue AUROC | MA-PDM | `anyanomaly` | Table 5, row MA-PDM[40] | verified |
| 79.2 | ShT AUROC | MA-PDM | `anyanomaly` | Table 5, row MA-PDM[40] | verified |
| 81.3 | ShT AUROC | MULDE | `anyanomaly` | Table 5, row MULDE[24]; Avenue not reported | verified |
| 62.3 | Avenue AUROC | Zero-shot CLIP | `anyanomaly` | Table 6, row ZS CLIP[25] | verified |
| 60.9 | ShT AUROC | Zero-shot CLIP | `anyanomaly` | Table 6, row ZS CLIP[25] | verified |
| 64.5 | Avenue AUROC | Zero-shot ImageBind | `anyanomaly` | Table 6, row ZS ImageBind[9] | verified |
| 61.3 | ShT AUROC | Zero-shot ImageBind | `anyanomaly` | Table 6, row ZS ImageBind[9] | verified |
| 67.4 | Avenue AUROC | LLaVA-1.5 | `anyanomaly` | Table 6, row LLaVA-1.5[16] | verified |
| 59.6 | ShT AUROC | LLaVA-1.5 | `anyanomaly` | Table 6, row LLaVA-1.5[16] | verified |
| 76.9 | Avenue AUROC | Video-ChatGPT | `anyanomaly` | Table 6, row Video-ChatGPT[22] | verified |
| 69.1 | ShT AUROC | Video-ChatGPT | `anyanomaly` | Table 6, row Video-ChatGPT[22] | verified |
| 87.3 | Avenue AUROC | AnyAnomaly | `anyanomaly` | Tables 5 and 6, both give 87.3 | verified |
| 79.7 | ShT AUROC | AnyAnomaly | `anyanomaly` | Tables 5 and 6, both give 79.7 | verified |
| 80.28 | UCF-Crime AUC | LAVAD | `lavad` | Table 1, p.6, row LAVAD | verified |
| 80.3 | UCF AUROC | LAVAD | `lavad` | Table 1, p.6, row LAVAD (80.28, shown to one decimal in `tbl:sota`) | verified |
| 53.2 | UCF AUROC | Zero-shot CLIP | `lavad` | Table 1, p.6, row ZS CLIP[22] (53.16) | verified |
| 53.7 | UCF AUROC | Zero-shot ImageBind | `lavad` | Table 1, p.6, row ZS IMAGEBIND (IMAGE)[6] (53.65); the video variant is 55.78 | verified |
| 72.8 | UCF AUROC | LLaVA-1.5 | `lavad` | Table 1, p.6, row LLAVA-1.5[17] (72.84) | verified |
| 74.7 | UCF AUROC | FPDM | `anyanomaly` | Table 5, p.8, row FPDM[34], UCF column | verified |
| 78.5 | UCF AUROC | MULDE | `anyanomaly` | Table 5, p.8, row MULDE[24], UCF column | verified |
| 80.7 | UCF AUROC | AnyAnomaly | `anyanomaly` | Table 5, p.8, row AnyAnomaly, UCF column (77.8 without context) | verified |
| 62.01 | XD-Violence AP | LAVAD | `lavad` | Table 2, p.6, row LAVAD (AUC there is 85.36) | verified |

**Our own rows** — Avenue 67.7 and ShanghaiTech 73.4 — are not in this ledger.
They are measurements, and their provenance is the run manifests under
`results/runs/`, which record the code commit, hardware, library versions and
frame counts that produced them.

---

## Quoted claims about other papers

Numbers are not the only thing that can be wrong. These are direct quotations or
close paraphrases the argument leans on, and they carry the same rule.

| Claim | Source | Location | Status |
|---|---|---|---|
| Concept shift "is, however, usually not a common problem…" | Liu et al. 2022 survey | quoted verbatim in §3.1 | verified |
| AnyAnomaly is given the benchmark's anomaly classes: "each anomaly class in the dataset was treated as X" | `anyanomaly` | §4.6, Comparison with SOTA | verified |
| AnyAnomaly never supplies a deliberately wrong text | `anyanomaly` | absence across Tables 3–6 and the supplementary prompt study | verified |
| Abnormal events are "rare and diverse, making it difficult to construct large-scale datasets" | `anyanomaly` | §1, opening paragraph | verified |
| Avenue's anomaly classes include "too close" | `anyanomaly` | Table 2, C-Ave appearance classes | verified |
| Avenue anomalies are "throwing objects, loitering and running", and "the size of people may change because of the camera position and angle" | `shanghaitech` | §4.1, dataset description | verified |
| Both papers report frame-level AUC, so the rows are commensurable | `shanghaitech` | §4.2, Evaluation Metric | verified |
| LAVAD evaluates only on UCF-Crime and XD-Violence, so it is not comparable on our benchmarks | `lavad` | §4, Datasets | verified |
| A zero-shot CLIP VAD baseline is built from two single prompts scored by softmax over cosine similarity, on ViT-B/32 | `lavad` | §4.1, description of the ZS CLIP baseline | verified |
| LAVAD samples each video every 16 frames — the same as our UCF-Crime protocol | `lavad` | §4, Implementation Details: "We sample each video every 16 frames for computational efficiency" | verified |
| ~~LAVAD scores every frame~~ — **this was wrong** (stated in the paper, deck, script and handbook 30 Sep – 2 Oct 2026, corrected 2 Oct). Whether LAVAD computes AUROC over sampled frames or propagated scores is not stated | `lavad` | §4 | corrected |

---

## Adding a number

1. Put the PDF in this folder and add it to the source table above.
2. Open it. Find the figure. Note the table or section.
3. Add a ledger row with `status: verified`.
4. Only then put it in `main.tex`.
5. Run `python scripts/check_provenance.py`.
