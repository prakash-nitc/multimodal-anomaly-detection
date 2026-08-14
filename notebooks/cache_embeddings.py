# -*- coding: utf-8 -*-
"""Encode a dataset's frames once and cache the CLIP embeddings.

Everything expensive about DA-ZVAD is the image encoder, and it is entirely
independent of the prompts, the scoring rule and the temporal window. Coupling
them -- as ``score_frames`` does -- means every new idea costs a full re-encode:
50 minutes on an A40 to answer one question about prompt wording.

Caching embeddings breaks that coupling. One pass over the dataset produces a
file that ``scoring_lab.py`` can evaluate any number of hypotheses against in
seconds. The embeddings depend only on (dataset, CLIP model, frame_step, crops),
so the cache stays valid across every prompt and scoring change.

Storage is trivial: 20k frames x 768 dims x float32 is ~63 MB whole-frame, or
~315 MB with quadrant crops.

Usage:
    python notebooks/cache_embeddings.py \
        --shanghaitech ~/dazvad/data/shanghaitech --crops 5
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--shanghaitech", required=True)
    p.add_argument("--out", default=os.path.expanduser("~/dazvad/work/embeddings"))
    p.add_argument("--frame-step", type=int, default=2)
    p.add_argument("--crops", type=int, default=5, choices=[1, 5],
                   help="1 = whole frame only; 5 = whole frame + four quadrants")
    p.add_argument("--clip-model", default="ViT-L-14")
    p.add_argument("--clip-pretrained", default="laion2b_s32b_b82k")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--gpu-frac", type=float, default=0.40)
    args = p.parse_args()

    import numpy as np
    import torch
    from da_zvad.config import DAZVADConfig
    from da_zvad.datasets import get_dataset
    from da_zvad.encoders import CLIPEncoder

    if not torch.cuda.is_available():
        print("CUDA unavailable -- refusing to encode on CPU.", file=sys.stderr)
        return 2
    torch.cuda.set_per_process_memory_fraction(args.gpu_frac, 0)
    print(f"[gpu] {torch.cuda.get_device_properties(0).name} "
          f"capped at {args.gpu_frac:.0%}")

    cfg = DAZVADConfig(dataset="shanghaitech",
                       data_root=os.path.expanduser(args.shanghaitech),
                       frame_step=args.frame_step)
    seqs = get_dataset(cfg).sequences()
    total = sum(len(s.frames) for s in seqs)
    print(f"[data] {len(seqs)} clips / {total} frames (step={args.frame_step})")

    enc = CLIPEncoder(args.clip_model, args.clip_pretrained, "cuda")

    feats, labels, clip_ids, names = [], [], [], []
    t0 = time.time()
    for i, seq in enumerate(seqs):
        f = enc.encode_images(seq.frames, batch_size=args.batch_size,
                              num_workers=args.workers, crops=args.crops)
        feats.append(f)
        labels.append(np.asarray(seq.labels, dtype=np.int8))
        clip_ids.append(np.full(len(seq.frames), i, dtype=np.int32))
        names.append(seq.name)
        done = sum(len(x) for x in labels)
        el = time.time() - t0
        print(f"[enc] {i + 1:>3}/{len(seqs)}  {seq.name:<28} "
              f"{done:>6}/{total} frames  {el / 60:.1f} min  "
              f"eta {(el / max(done, 1) * (total - done)) / 60:.1f} min", flush=True)

    os.makedirs(os.path.expanduser(args.out), exist_ok=True)
    path = os.path.join(os.path.expanduser(args.out),
                        f"shanghaitech_{args.clip_model}_step{args.frame_step}"
                        f"_crops{args.crops}.npz")
    np.savez(path,
             feats=np.concatenate(feats, axis=0),
             labels=np.concatenate(labels),
             clip_ids=np.concatenate(clip_ids),
             names=np.array(names, dtype=object),
             frame_step=args.frame_step, crops=args.crops,
             clip_model=args.clip_model, clip_pretrained=args.clip_pretrained)
    size_mb = os.path.getsize(path) / 1024 ** 2
    print(f"\n[done] {(time.time() - t0) / 60:.1f} min")
    print(f"[done] {path}  ({size_mb:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
