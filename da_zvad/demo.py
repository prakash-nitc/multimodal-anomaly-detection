"""End-to-end smoke test of the DA-ZVAD pipeline on SYNTHETIC data.

Runs with no GPU, no model download and no dataset -- it validates the framework
wiring (temporal aggregation -> detection -> metrics -> reasoning interface) and
saves an anomaly-score timeline figure.

This is a plumbing test, NOT a research result. Real numbers come from running the
same pipeline on MVTec / ShanghaiTech / Avenue on the GPU.

Run:  python -m da_zvad.demo
"""
from __future__ import annotations

import os
import numpy as np

from .config import DAZVADConfig
from .pipeline import DAZVADPipeline
from .datasets import get_dataset


def main(out_dir: str = "results"):
    print("=" * 64)
    print("DA-ZVAD pipeline smoke test (synthetic data, no GPU)")
    print("=" * 64)

    cfg = DAZVADConfig(
        dataset="synthetic",
        domain="surveillance",
        domain_description="a quiet campus walkway",
        use_temporal=True, use_context=True, use_reasoning=True,
        temporal_window=5, threshold=0.5, seed=0,
    )
    pipe = DAZVADPipeline(cfg)
    print("Pipeline:", pipe)
    n, a = pipe.build_prompts()
    print(f"M3 grounded prompts: {len(n)} normal / {len(a)} abnormal "
          f"(context = '{cfg.domain_description}')")

    seq = get_dataset(cfg).sequences()[0]
    out = pipe.run_sequence(seq)

    m = out["metrics"]
    print("\n-- metrics (aggregated scores vs. ground truth) --")
    for k, v in m.items():
        print(f"   {k:10s}: {v:.3f}")
    n_flagged = int(out["predictions"].sum())
    print(f"   flagged frames: {n_flagged} / {len(seq)}")
    if out["explanations"]:
        i = sorted(out["explanations"])[0]
        print(f"   M4 example: frame {i} -> {out['explanations'][i]}")

    _save_figure(out, cfg, out_dir)
    print("\nDone. (Synthetic plumbing test -- not a research result.)")
    return out


def _save_figure(out, cfg, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("(matplotlib not available -- skipping figure)")
        return
    raw, agg, labels = out["raw_scores"], out["scores"], out["labels"]
    t = np.arange(len(agg))
    fig, ax = plt.subplots(figsize=(11, 4))
    # shade ground-truth anomalous span
    in_anom = labels == 1
    if in_anom.any():
        start = int(np.argmax(in_anom)); end = len(labels) - int(np.argmax(in_anom[::-1]))
        ax.axvspan(start - 0.5, end - 0.5, color="red", alpha=0.12, label="ground-truth anomaly")
    ax.plot(t, raw, color="0.7", lw=1, label="raw per-frame score (M1)")
    ax.plot(t, agg, color="C0", lw=2, marker="o", ms=3, label=f"temporal (M2, w={cfg.temporal_window})")
    ax.axhline(cfg.threshold, ls="--", color="orange", lw=1.2, label=f"threshold {cfg.threshold}")
    ax.set_title(f"DA-ZVAD pipeline smoke test  |  AUROC={out['metrics']['auroc']:.3f}  "
                 f"(synthetic)", fontweight="bold")
    ax.set_xlabel("frame"); ax.set_ylabel("anomaly score"); ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.3)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    path = os.path.join(out_dir, "figures", "da_zvad_demo.png")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"   figure saved: {path}")


if __name__ == "__main__":
    main()
