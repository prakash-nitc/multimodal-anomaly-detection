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
| M4 — LLM reasoning | `reasoning/llava_reasoner.py` | ✅ implemented (LLaVA-1.5, 4-bit, needs GPU; CPU runs use the stub). One explanation per flagged *event* (peak frame), budget-capped |
| Datasets | `datasets/` | ✅ MVTec · ✅ ShanghaiTech (frame folders + .npy GT) · ✅ Avenue (.avi + .mat GT) |
| Evaluation | `evaluation/metrics.py` | ✅ frame-level AUROC / AP / F1 |
| Grid engine | `grid.py` + `scripts/run_grid.py` | ✅ dataset × context × window grid; raw-score caching = resumable + one M1 pass serves all windows |

## Quick start

```bash
# end-to-end smoke test — no GPU, no data, no model download:
python -m da_zvad.demo

# grid smoke test (also CPU-only):
python scripts/run_grid.py --synthetic

# single real run on the GPU machine:
python scripts/run_da_zvad.py --dataset mvtec --data-root <path> --category bottle --domain industrial

# the batched results job on the GPU machine (resumable):
python scripts/run_grid.py --shanghaitech <root> --frame-step 2 --mvtec <root> --categories bottle,screw
```

The smoke test exercises the whole pipeline on synthetic scores and saves a score
timeline to `results/figures/da_zvad_demo.png`. It is a **plumbing test, not a
research result** — real numbers come from the GPU runs above.

## What's next
1. Run the ShanghaiTech probe / grid on a GPU (Kaggle or college) → first real frame-level AUROC.
2. Generate the first explanation gallery with the LLaVA reasoner on flagged events.
3. Extend the grid outputs into the September review tables and figures.
