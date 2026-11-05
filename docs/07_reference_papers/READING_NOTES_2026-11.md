# Reading notes, November 2026

Page numbers are PDF pages.

## NOVA (arXiv 2609.06360v1, preprint, no venue)
- Numbers, Table 1 p.7: UCF AUC 89.86; XD AP 84.82, AUC 95.07. Mean over five prompt banks; no video-level CIs.
- Components: NA (prompt construction), VNA (visual anchor), MAS (window + motion gate). Backbone PE-Core-L14-336 (p.6).
- **NA forbids look-alike actions from the normal side** (p.3-4, supp F.1/F.3 p.13-14: "must NEVER appear in a normal
  description"). OUR N DOES THE OPPOSITE: `banks.NEAR_MISS` ADDS look-alikes to the normal side.
- NA also: static/low-motion normals, 4 calm anchors, geometry-gated rewrite (cos to anomaly centroid > 0.70, max 3 rounds).
- VNA: first 32 sampled frames, softmax weights on preliminary scores (gamma 0.3), weighted mean IMAGE embedding added to the
  normal pool (p.5). Uses the image vector directly, not text retrieved from it.
- Scoring: per class c, top-50% mean of 20 positive and 20 negative prompts, alpha=0.9, tau=0.03; score = max over classes (p.5).
- Needs: predefined anomaly vocabulary + one footage descriptor per dataset + hand-written role guidance/examples per dataset
  (p.3-4, supp p.13-14). Uses GPT-5.4-mini for banks. Only UCF and XD tested: both have crime/violence classes.
- Ablation (Table 2-3, p.7):
  template, classes only: UCF 66.85, XD AUC 90.99, XD AP 77.94
  LLM, anomaly only: 72.50 / 91.81 / 76.44
  LLM normal+anomaly, no NA: 85.62 / 93.51 / 80.36
  NA: 88.79 / 94.29 / 84.33; VNA alone adds +1.37 / +0.57 / +0.11; MAS alone XD AP -7.48.
- So: VNA small; XD AP is already 77.94 with PE-Core and class names, before any NOVA module.
- Limits they admit (p.8): fixed anchor from early prefix; not for long-term evolving normality.
- Their template baseline drops UCF to 66.85: normal-side text matters a lot THERE because there is otherwise no normal reference.

## A VLM Answer Is Not an Anomaly Score (arXiv 2608.21244v3, preprint)
- Frozen VLMs, yes/no probability ("PROB") with a generic prompt, no sentence, no training (Table 2 p.4):
  UCF AUROC 85.32, XD AP 68.98. Per model Table 5-6 p.15-16. Decoded answer (DEC) 72.20 / 49.44.
- Metric warning (p.8-9): trapezoidal PR-AUC vs non-interpolated AP. DEC example XD: 74.06 vs 50.66; PROB 70.63 vs 70.25.
  Inflation is for tied scores. Our xd_ap is trapezoidal (official XD code). Our secondary `ap_step` matches to 4 digits
  (legacy_none .6243/.6243, legacy_best .4170/.4171, FULL .7507/.7508), so OUR numbers are not inflated. LAVAD's and other
  decoded-score figures may be: that paper does not test LAVAD directly, so say "may".
- Paired video-level bootstrap with 10,000 replicates, same design as ours; we use 1000.
- Relevance to our audit: different mechanism. Theirs = ties from decoding. Ours = level/shape. Do not conflate.

## Flashback (arXiv 2505.15205v2, preprint)
- Table 1 p.6: Flashback-PE UCF 87.29, XD AP 75.13, XD AUC 90.54. Flashback-IB (ImageBind, same method) 81.65 / 60.13 / 83.52.
  So encoder swap alone: +5.64 UCF, +15.00 XD AP. Useful prior for P7, though it is a video encoder setting.
- Repulsive prompting p.5, Table 3b p.8: keyword "Normal"/"Anomalous" + different templates per side:
  none 74.98 / 71.01 / 87.08; keyword only 81.24 / 72.20 / 88.42; template only 82.12 / 72.21 / 88.82; both 87.29 / 75.13 / 90.54.
- Scaled anomaly penalization p.5: alpha<1 on anomaly embeddings; encoder biased toward anomalous captions.
- Memory: gpt-4o, 1M pairs, $181.43, 76 h (p.6). 10k pairs already 82.61 / 72.17 / 88.63 (Table 3d p.8).
- Retrieves top-10 captions per SEGMENT, score = soft label average. Different from our per-camera scene retrieval.
- Real-time 42 fps. Failure shown: benchmark disagrees with its labels (pickpocketing) p.7.

## ASK-Hint (WACV 2026)
- UCF 89.83 (Table 1, p.5), XD AUC 90.31 (Table 2, p.6); no AP. Heavy frozen VLM (Qwen2.5-VL).
- More prompts hurt: 42 prompts 67.17, 3: 78.71, 6: 89.83, 9: 87.67, 12: 83.36 (Tables 3, 5, pp.6-7). 6 picked looking at this table.
- Cross-dataset prompt transfer loses AUC: UCF 89.83 to 81.86; XD 90.31 to 87.11 (Table 6, p.7). Transfer is hard for them too.

## Others (method and results)
- MaskCLIP: frozen CLIP, remove query/key, use value embedding + last linear at each location; no training for the base version (p.6-7).
  Segmentation work, not video anomaly.
- AnomalyCLIP: learns prompts on auxiliary data, so it trains; DPAM (V-V attention) is the reusable local-feature idea.
- PE: best features for spatial tasks sit at intermediate layers (p.2-3); PE-Core tuned on synthetic video captions. Not tested here.
- AnomalyRuler: 10 normal frames (10 batches x 1) from training set, GPT-4V + GPT-4 (p.10). Tests Ped2/Avenue/ShT/UB.
- SlowFastVAD: RAG knowledge base from few normal samples + trained AED-MAE fast detector. Not training-free. Tests Ped2/Avenue/ShT/UB.
- LaGoVAD: trained on 35,279 videos of its own dataset; user defines anomaly in language at inference. Not training-free.
- Pattern: none of NOVA, Flashback, ASK-Hint tests a campus dataset. Ruler and SlowFast test campus but not UCF/XD.
  Across the papers read, no training-free system reports crime AND campus benchmarks with video-level CIs.

## Dataset facts
- ShT (Luo 2017, p.5): names only chasing and brawling; no full class list. Our worry list cannot be verified from this paper.
- Avenue (Lu 2013, p.5): running, throwing objects, loitering: matches ours. Paper says 15 sequences/14 events; later papers use 16 train/21 test.
- UBnormal: 29 scenes, 236,902 frames (p.2); open-set; micro and macro frame AUC plus RBDC/TBDC.
- MSAD: protocol i test 120 normal + 240 abnormal; protocol ii 120 + 120 (p.7).

## CSI-VAD (arXiv 2607.19077v1)
- VIDEO-level evaluation only (p.3): whole-video score, not frame. Not comparable with any frame-level figure of ours.
- UCF video AUC 92.90 (Table I p.4); UBnormal video AUC about 72 (Table III p.5). Heavy: Qwen2.5-VL, detector, tracker, several LLM calls.
- Env branch: scene prior from 3 keyframes (first/middle/LAST) so uses the end of the video.
