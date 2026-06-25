# DA-ZVAD — framework skeleton

**Domain-Adaptive Zero-shot Video Anomaly Detection.** A modular, training-free
pipeline for anomaly detection with vision-language and large language models.

```
frames → [M1 CLIP] → [M2 temporal] → [M3 context] → [M4 LLM reasoning]
       → per-frame anomaly score + natural-language explanation
```

Every module is a toggle in the config, so the **same code path produces the full
ablation study** by configuration alone.

## Module status

| Module | File | Status |
|---|---|---|
| M1 — Frozen CLIP encoder | `encoders/clip_encoder.py` | ✅ implemented (OpenCLIP, lazy-loaded) |
| M2 — Temporal aggregation | `temporal/aggregation.py` | ✅ implemented (training-free smoothing) |
| M3 — Verbalized context | `context/verbalized.py` | ✅ implemented |
| M4 — LLM reasoning | `reasoning/llm_reasoner.py` | 🔶 interface + stub (real MLLM = next step) |
| Datasets | `datasets/` | ✅ MVTec · 🔶 video (cv2 frame extraction ready; GT-label parsing stubbed) |
| Evaluation | `evaluation/metrics.py` | ✅ frame-level AUROC / AP / F1 |
| Runner | `runner.py` | ✅ config-driven; `run_grid` = the ablation engine |

## Quick start

```bash
# end-to-end smoke test — no GPU, no data, no model download:
python -m da_zvad.demo

# real run on the GPU machine:
python scripts/run_da_zvad.py --dataset mvtec --data-root <path> --category bottle --domain industrial
```

The smoke test exercises the whole pipeline on synthetic scores and saves a score
timeline to `results/figures/da_zvad_demo.png`. It is a **plumbing test, not a
research result** — real numbers come from the GPU runs above.

## What's next
1. Implement the real M4 reasoner (LLaVA / caption-then-reason) behind the existing interface.
2. Wire up frame-level ground-truth label parsing for ShanghaiTech / CUHK Avenue.
3. Run the ablation grid (module on/off × dataset × seed) via `runner.run_grid`.
