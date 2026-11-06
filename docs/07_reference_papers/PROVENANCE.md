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
| `xdviolence` | `Wu et al., ECCV 2020.pdf` | XD-Violence dataset description and metric; PREREG_xd_violence.md |
| `nwpu` | `Cao et al. 2023, NWPU Campus (CVPR).pdf` | NWPU Campus description, metric, comparison figures; PREREG_nwpu_campus.md |
| `nwpusupp` | `Cao et al. 2023, NWPU Campus (CVPR) supplementary.pdf` | anomaly classes per scene; definition of scene-dependent anomalies |
| `nova` | `NOVA_ Normal-Side Modeling….pdf` | closest related work; Tables 1-3 |
| `flashback` | `Flashback-Memory-Driven….pdf` | related work; encoder and prompt ablation |
| `askhint` | `Unlocking_Vision-Language_Models….pdf` | related work; prompt count, transfer |
| `vlmanswer` | `A VLM Answer Is Not an Anomaly Score….pdf` | AP vs trapezoidal PR-AUC; frozen VLM baseline |
| `csivad` | `Context-structured Video Anomaly Detection….pdf` | video-level results only |
| `anomalyruler` | `Follow the Rules….pdf` | few-normal-shot results |
| `ubnormal` | `UBnormal_New_Benchmark….pdf` | sealed-test candidate description |
| `msad` | `NeurIPS-2024-advancing-video-anomaly-detection….pdf` | sealed-test candidate description |
| `luo2017` | `A_Revisit_of_Sparse_Coding….pdf` | ShanghaiTech description |
| `lu2013` | `Lu_Abnormal_Event_Detection_2013_ICCV_paper.pdf` | Avenue description |
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
| 85.36 | XD-Violence AUC | LAVAD | `lavad` | Table 2, p.6, row LAVAD | verified |
| 50.26 | XD-Violence AP | LLaVA-1.5 | `lavad` | Table 2, p.6, row LLAVA-1.5[17] (AUC 79.62) | verified |
| 17.83 | XD-Violence AP | Zero-shot CLIP | `lavad` | Table 2, p.6, row ZS CLIP[22] (AUC 38.21) | verified |
| 27.25 | XD-Violence AP | Zero-shot ImageBind | `lavad` | Table 2, p.6, row ZS IMAGEBIND (IMAGE)[6] (AUC 58.81); the video variant is 25.36 / 55.06 | verified |
| 88.26 | XD-Violence AUC | VERA | `vera` | Table 3, p.8684, row VERA; VERA learns its questions on training data and reports AUC only | verified |
| 89.86 | UCF AUROC | NOVA | `nova` | Table 1, p.7; mean of five prompt banks | verified |
| 84.82 | XD-Violence AP | NOVA | `nova` | Table 1, p.7 | verified |
| 95.07 | XD-Violence AUC | NOVA | `nova` | Table 1, p.7 | verified |
| 66.85 | UCF AUROC | NOVA template, classes only | `nova` | Table 3, p.7 | verified |
| 77.94 | XD-Violence AP | NOVA template, classes only | `nova` | Table 3, p.7 (AUC 90.99) | verified |
| 84.74 | UCF AUROC | VADTree, as quoted by NOVA | `nova` | Table 1, p.7 (XD AP 68.85, AUC 90.55); original PDF not read | verified |
| 84.89 | UCF AUROC | PANDA, as quoted by NOVA | `nova` | Table 1, p.7 (XD AP 70.16); original PDF not read | verified |
| 82.03 | UCF AUROC | EventVAD, as quoted by NOVA | `nova` | Table 1, p.7 (XD AP 64.04, AUC 87.51); original PDF not read | verified |
| 87.29 | UCF AUROC | Flashback-PE | `flashback` | Table 1, p.6 | verified |
| 75.13 | XD-Violence AP | Flashback-PE | `flashback` | Table 1, p.6 (AUC 90.54) | verified |
| 81.65 | UCF AUROC | Flashback-ImageBind | `flashback` | Table 1, p.6 | verified |
| 60.13 | XD-Violence AP | Flashback-ImageBind | `flashback` | Table 1, p.6 (AUC 83.52) | verified |
| 70.54 | XD-Violence AP | VERA, as quoted by Flashback | `flashback` | Table 1, p.6; VERA's own paper reports AUC only | verified |
| 89.83 | UCF AUROC | ASK-Hint | `askhint` | Table 1, PDF p.5 | verified |
| 90.31 | XD-Violence AUC | ASK-Hint | `askhint` | Table 2, PDF p.6; no AP reported | verified |
| 85.32 | UCF AUROC | Frozen VLM, yes/no probability | `vlmanswer` | Table 2, p.4; mean over four VLMs | verified |
| 68.98 | XD-Violence AP | Frozen VLM, yes/no probability | `vlmanswer` | Table 2, p.4; mean over four VLMs | verified |
| 92.90 | UCF video-level AUROC | CSI-VAD | `csivad` | Table I, p.4; VIDEO level, not comparable with frame level | verified |
| 85.2 | ShanghaiTech AUROC | AnomalyRuler | `anomalyruler` | Table 3, p.12; few-normal-shot, uses GPT-4 | verified |
| 71.9 | UBnormal AUROC | AnomalyRuler | `anomalyruler` | Table 3, p.12; few-normal-shot, uses GPT-4. Micro AUROC: the same column gives BAF 59.3, the UBnormal paper's micro test figure for that method (Table 2, p.7) | verified |
| 62.94 | UBnormal AUROC | OVVAD | `ovvad` | Table 3, p.6; trained. Micro AUROC: its Georgescu rows (59.30, 61.30) equal the UBnormal paper's micro test figures | verified |
| 59.3 | UBnormal micro AUROC, test | Georgescu et al. (BAF) | `ubnormal` | Table 2, p.7, Test / AUC Micro; trained on UBnormal normal training data | verified |
| 84.9 | UBnormal macro AUROC, test | Georgescu et al. (BAF) | `ubnormal` | Table 2, p.7, Test / AUC Macro | verified |
| 61.3 | UBnormal micro AUROC, test | Georgescu et al. + UBnormal anomalies | `ubnormal` | Table 2, p.7, Test / AUC Micro; trained with UBnormal's training anomalies | verified |
| 85.6 | UBnormal macro AUROC, test | Georgescu et al. + UBnormal anomalies | `ubnormal` | Table 2, p.7, Test / AUC Macro | verified |
| 68.5 | UBnormal micro AUROC, test | Bertasius et al. (1/32 sample rate, fine-tuned) | `ubnormal` | Table 2, p.7, Test / AUC Micro; best micro test figure in the table; supervised, fine-tuned on UBnormal | verified |
| 80.3 | UBnormal macro AUROC, test | Bertasius et al. (1/32 sample rate, fine-tuned) | `ubnormal` | Table 2, p.7, Test / AUC Macro | verified |
| 71.8 | UBnormal AUROC | STG-NF, as tabulated by AnomalyRuler | `anomalyruler` | Table 3, p.12, UB column (micro, as above); trained; original PDF not read | verified |
| 62.7 | UBnormal AUROC | FPDM, as tabulated by AnomalyRuler | `anomalyruler` | Table 3, p.12, UB column (micro, as above); trained; original PDF not read | verified |
| 61.9 | NWPU Campus AUC | MemAE | `nwpu` | Table 4, p.8 (also Table 3, p.7, column Cam) | verified |
| 62.5 | NWPU Campus AUC | MNAD | `nwpu` | Table 4, p.8 | verified |
| 62.5 | NWPU Campus AUC | OG-Net | `nwpu` | Table 4, p.8 | verified |
| 63.7 | NWPU Campus AUC | HF2-VAD | `nwpu` | Table 4, p.8 | verified |
| 64.4 | NWPU Campus AUC | MPN | `nwpu` | Table 4, p.8 | verified |
| 64.5 | NWPU Campus AUC | AMMC-Net | `nwpu` | Table 4, p.8 | verified |
| 65.8 | NWPU Campus AUC | Cao et al. (γ=0, no scene conditioning) | `nwpu` | Table 4, p.8 | verified |
| 68.2 | NWPU Campus AUC | Cao et al. (γ=1, scene-conditioned) | `nwpu` | Table 4, p.8 | verified |
| 17.8 | XD-Violence AP | Zero-shot CLIP | `lavad` | Table 2, p.6, row ZS CLIP[22] (17.83, shown to one decimal in Paper 1 `tbl:sota`) | verified |
| 27.3 | XD-Violence AP | Zero-shot ImageBind | `lavad` | Table 2, p.6, row ZS IMAGEBIND (IMAGE)[6] (27.25, shown to one decimal) | verified |
| 50.3 | XD-Violence AP | LLaVA-1.5 | `lavad` | Table 2, p.6, row LLAVA-1.5[17] (50.26, shown to one decimal) | verified |
| 62.0 | XD-Violence AP | LAVAD | `lavad` | Table 2, p.6, row LAVAD (62.01, shown to one decimal) | verified |
| 85.3 | UCF AUROC | Frozen VLM, yes/no probability | `vlmanswer` | Table 2, p.4 (85.32, shown to one decimal); mean over four VLMs | verified |
| 69.0 | XD-Violence AP | Frozen VLM, yes/no probability | `vlmanswer` | Table 2, p.4 (68.98, shown to one decimal); mean over four VLMs | verified |
| 87.3 | UCF AUROC | Flashback-PE | `flashback` | Table 1, p.6 (87.29, shown to one decimal) | verified |
| 75.1 | XD-Violence AP | Flashback-PE | `flashback` | Table 1, p.6 (75.13, shown to one decimal) | verified |
| 89.8 | UCF AUROC | ASK-Hint | `askhint` | Table 1, PDF p.5 (89.83, shown to one decimal) | verified |
| 89.9 | UCF AUROC | NOVA | `nova` | Table 1, p.7 (89.86, shown to one decimal); mean of five prompt banks | verified |
| 84.8 | XD-Violence AP | NOVA | `nova` | Table 1, p.7 (84.82, shown to one decimal) | verified |

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
| XD-Violence: 4,754 untrimmed videos, 217 h; test set 800 = 500 violent + 300 non-violent; six classes (Abuse, Car Accident, Explosion, Fighting, Riot, Shooting); sources include movies, cartoons, sports, games, news, CCTV, hand-held and car cameras | `xdviolence` | p.1 Abstract; §3.1–3.3 | verified |
| XD-Violence is evaluated by frame-level AP (precision–recall), not AUC, "since AUC usually shows an optimistic result when dealing with class-imbalanced data"; videos fixed at 24 fps, 16-frame snippets | `xdviolence` | §5.1–5.2 | verified |
| Official AP = sklearn `auc(recall, precision)` over snippet scores repeated ×16; ground truth `gt[start:end] = 1` (0-based, end exclusive), each video cut to frame-count − 1 and then to a multiple of 16 | authors' code, github.com/Roc-Ng/XDVioDet: `test.py`, `list/make_gt.py` | read 16 Oct 2026; our `snippet_gt` matches a literal port of `make_gt.py` on all 800 videos × 3 random lengths | verified |
| XD-Violence label codes: B1 Fighting, B2 Shooting, B4 Riot, B5 Abuse, B6 Car accident, G Explosion; A = normal | dataset ReadMe, roc-ng.github.io/XD-Violence/images/ReadMe.md | read 16 Oct 2026 | verified |
| NWPU Campus: 43 scenes, 28 anomaly classes, 25 fps; 305 training (normal only) + 242 test videos; test 384,059 frames, 65,266 anomalous, 124 videos with anomalies; frame labels carry no class | `nwpu` | Abstract; §3.1; Table 2; also matches the released ground-truth file | verified |
| NWPU Campus metric: "we concatenate all the frames in a dataset and then compute the overall frame-level AUC" | `nwpu` | §5.1, Evaluation Metric | verified |
| Scene dependency: "an event is normal in one scene but abnormal in another" | `nwpu` | §1 | verified |
| Scene-dependent classes: cycling on footpath, wrong turn, trucks, photographing in restricted area; defined by the scene's training footage (photographing absent from D036/D038 training videos, present in D055) | `nwpusupp` | §1; Table 2 | verified |
| Scenes with a scene-dependent class: D003 (Trucks), D036 and D038 (Photographing), D129 (Cycling on footpath); "Wrong turn" appears in no scene of Table 1 | `nwpusupp` | Table 1 | verified |
| Page ranges: UCF-Crime 6479–6488, VERA 8679–8688, OVVAD 18297–18307, zxVAD 2578–2590, Lu 2020 125–141 | PDFs | running page numbers; Lu 2020 via zxVAD ref. [1] | verified |
| Ada-VAD: synthesised abnormal samples, then adversarial adaptation to a few target frames | publisher abstract | SIAM SDM 2024 abstract page (doi:10.1137/1.9781611978032.73): "we synthesize abnormal samples … pretrain a domain invariant model … adapt the pre-trained model to target domain with few-shot samples … with an adversarial training approach"; authors Guo, Fu, Li; pp. 634–642 | verified (abstract; full text paywalled) |
| LAVAD samples each video every 16 frames — the same as our UCF-Crime protocol | `lavad` | §4, Implementation Details: "We sample each video every 16 frames for computational efficiency" | verified |
| ~~LAVAD scores every frame~~ — **this was wrong** (stated in the paper, deck, script and handbook 30 Sep – 2 Oct 2026, corrected 2 Oct). Whether LAVAD computes AUROC over sampled frames or propagated scores is not stated | `lavad` | §4 | corrected |

---
| NOVA forbids look-alike actions from normal prompts ("must NEVER appear in a normal description") | `nova` | §3.2, pp.3-4; supp F.1, F.3, pp.13-14 | verified |
| UBnormal: 29 virtual scenes, 236,902 frames | `ubnormal` | p.2, contributions | verified |
| MSAD protocol i tests on 120 normal + 240 abnormal videos; protocol ii on 120 + 120 | `msad` | p.7 | verified |
| ShanghaiTech adds sudden-motion anomalies "such as chasing and brawling"; the paper gives no full class list | `luo2017` | p.5 | verified |
| Avenue: 14 unusual events "including running, throwing objects, and loitering" | `lu2013` | p.5 | verified |
| Trapezoidal PR-AUC can exceed non-interpolated AP on tied scores (74.06 vs 50.66 decoded; 70.63 vs 70.25 probability) | `vlmanswer` | Fig. 5, pp.8-9 | verified |

## Adding a number

1. Put the PDF in this folder and add it to the source table above.
2. Open it. Find the figure. Note the table or section.
3. Add a ledger row with `status: verified`.
4. Only then put it in `main.tex`.
5. Run `python scripts/check_provenance.py`.
