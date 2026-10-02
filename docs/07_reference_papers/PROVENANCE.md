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

**The PDFs are kept on the local machine only and are not committed** — most are publisher copies licensed to the downloader, and the repository is public. `.gitignore` excludes them; `scripts/check_provenance.py` still checks they are present locally.

| Key | File | Supplies |
|---|---|---|
| `anyanomaly` | `AnyAnomaly_WACV2026.pdf` | every comparison row in Table `tbl:sota` |
| `shanghaitech` | `Liu2018_FutureFramePrediction_CVPR.pdf` | the benchmark's own baseline |
| `lavad` | `LAVAD_CVPR2024.pdf` | the UCF-Crime / XD-Violence figures in Limitations |
| `wilkinghoff` | `Wilkinghoff et al. 2026.pdf` | two quotations; Observation 2 and C1 |
| `vera` | `VERA_…Vision-Language_Models.pdf` | literature review; gap G4 |
| `ucfcrime` | `Sultani et al. 2018, UCF-Crime (CVPR 2018).pdf` | dataset description; Result 7 |
| `ovvad` | `Wu et al. 2024, OVVAD (CVPR)-….pdf` | literature review |
| `zxvad` | `Aich et al. 2023, zxVAD (WACV)-….pdf` | literature review; also the secondary source for Lu et al. 2020 |
| — | (none — paywalled) | Ada-VAD verified from the publisher's abstract page. The file once saved under that name was a different paper (Zhu et al., TKDD 2024) and was removed |

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
| Wilkinghoff: "a specific observation may be normal under one operating condition, yet anomalous under another" | `wilkinghoff` | p.1, Abstract — verbatim | verified |
| Wilkinghoff: "structural ambiguity" | `wilkinghoff` | p.1 Abstract and p.4 §3 | verified |
| Wilkinghoff is a position paper; its context comes from other modalities (sensors) | `wilkinghoff` | p.4 §3 "Position Statement"; p.1 "modalities play asymmetric roles, separating context from observation" | verified |
| Liu 2022: "it is challenging to alleviate the label shift … even though there are sufficient training data" | Liu et al. 2022 survey | verbatim; fails the automatic check only because the PDF prints "sufficient" with an ffi ligature | verified |
| Singhal: covariate shift, P(Ys\|Xs)=Q(Yt\|Xt), is the first condition for DA | Singhal et al. 2023 survey | "three primary conditions … 1) Covariate Shift" | verified |
| Per-clip min–max normalisation is the benchmark's own protocol | `shanghaitech` | §3.4: "we normalize PSNR of all frames in each testing video to the range [0, 1]" | verified |
| VERA keeps the VLM frozen and optimises guiding questions on labelled training data | `vera` | p.1 Abstract and Fig. 1 | verified |
| ~~VERA is evaluated in-domain only~~ — **wrong**: VERA Table 9 transfers questions between UCF-Crime and XD-Violence (detection AUC). Gap G4 reworded: explanation quality under shift is not measured | `vera` | p.7, Table 9 | corrected |
| OVVAD: frozen CLIP encoders, a "nearly weight-free temporal adapter", trained detection/classification heads (earlier text called the adapter "trained" — corrected) | `ovvad` | p.2 contributions; p.3 §3.2; p.6 implementation | corrected |
| zxVAD: no target-domain adaptation; an untrained CNN synthesises pseudo-abnormal frames; future-frame prediction | `zxvad` | p.1 Abstract; p.2 contributions | verified |
| Lu et al. 2020: meta-learned scene adaptation from few frames, future-frame prediction | `zxvad` (secondary) | zxVAD p.3 and ref. [1]: "use meta-learning approaches and adapt to the target domain with few scenes" | verified (secondary source) |
| UCF-Crime: 1,900 videos, 13 anomaly classes; test split 150 normal + 140 anomalous | `ucfcrime` | p.1 Abstract; p.5 "Training and testing sets" | verified |
| Page ranges: UCF-Crime 6479–6488, VERA 8679–8688, OVVAD 18297–18307, zxVAD 2578–2590, Lu 2020 125–141 | PDFs | running page numbers; Lu 2020 via zxVAD ref. [1] | verified |
| Ada-VAD: synthesised abnormal samples, then adversarial adaptation to a few target frames | publisher abstract | SIAM SDM 2024 abstract page (doi:10.1137/1.9781611978032.73): "we synthesize abnormal samples … pretrain a domain invariant model … adapt the pre-trained model to target domain with few-shot samples … with an adversarial training approach"; authors Guo, Fu, Li; pp. 634–642 | verified (abstract; full text paywalled) |
| LAVAD samples each video every 16 frames — the same as our UCF-Crime protocol | `lavad` | §4, Implementation Details: "We sample each video every 16 frames for computational efficiency" | verified |
| ~~LAVAD scores every frame~~ — **this was wrong** (stated in the paper, deck, script and handbook 30 Sep – 2 Oct 2026, corrected 2 Oct). Whether LAVAD computes AUROC over sampled frames or propagated scores is not stated | `lavad` | §4 | corrected |

---

## Adding a number

1. Put the PDF in this folder and add it to the source table above.
2. Open it. Find the figure. Note the table or section.
3. Add a ledger row with `status: verified`.
4. Only then put it in `main.tex`.
5. Run `python scripts/check_provenance.py`.
