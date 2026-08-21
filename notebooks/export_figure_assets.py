# -*- coding: utf-8 -*-
"""Collect everything needed to draw the paper's figures, in one pass.

The figures the supervisor asked for -- an architecture diagram carrying real
frames, dataset samples, a worked detection example, and a picture of why
per-camera pooling failed -- all need material that lives on the GPU server:
video frames, cached scores, and image embeddings. Pulling them piecemeal means
repeated round trips.

This writes a single small bundle instead. Frames are exported as JPEG, scores
and labels for one worked example as arrays, and a subsample of embeddings with
their camera identity so the thirteen views can be projected to two dimensions
and shown as thirteen separate clusters -- which is the pooling problem made
visible rather than described.

Everything is chosen deterministically so the figures can be regenerated.

Usage:
    python notebooks/export_figure_assets.py \
        --shanghaitech ~/dazvad/data/shanghaitech \
        --avenue ~/dazvad/data/avenue \
        --raw ~/dazvad/work/raw \
        --embeddings ~/dazvad/work/embeddings/shanghaitech_ViT-L-14_step2_crops5.npz \
        --out ~/dazvad/figure_assets
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def pick_frames(seq, n_normal=2, n_anom=2):
    """Indices of clearly-normal and clearly-anomalous frames in one sequence.

    Anomalous frames are taken from the middle of the longest contiguous
    anomalous run, where the event is unambiguously under way rather than
    beginning or ending. Normal frames are taken far from any anomalous run so a
    reader comparing them is not looking at the moments just before an event.
    """
    lab = np.asarray(seq.labels, dtype=int)
    if lab.sum() == 0:
        return [], []

    runs, start = [], None
    for i, v in enumerate(lab):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i)); start = None
    if start is not None:
        runs.append((start, len(lab)))
    if not runs:
        return [], []

    lo, hi = max(runs, key=lambda r: r[1] - r[0])
    mid = (lo + hi) // 2
    anom = [mid + k for k in range(-(n_anom // 2), n_anom - n_anom // 2)
            if 0 <= mid + k < len(lab) and lab[mid + k]]

    far = [i for i in range(len(lab))
           if not lab[i] and all(abs(i - m) > 30 for m in anom)]
    normal = [far[len(far) // 4], far[3 * len(far) // 4]][:n_normal] if far else []
    return normal, anom


def export_dataset(name, cfg_kwargs, out_dir, max_clips=3):
    from da_zvad.config import DAZVADConfig
    from da_zvad.datasets import get_dataset
    from PIL import Image

    cfg = DAZVADConfig(**cfg_kwargs)
    seqs = get_dataset(cfg).sequences()
    seqs = [s for s in seqs if np.asarray(s.labels).sum() > 0]
    if not seqs:
        print(f"[{name}] no labelled anomalies found")
        return

    # spread the chosen clips across the dataset rather than taking the first few
    picks = [seqs[i] for i in np.linspace(0, len(seqs) - 1, max_clips, dtype=int)]
    d = os.path.join(out_dir, "frames", name)
    os.makedirs(d, exist_ok=True)

    for seq in picks:
        clip = os.path.basename(str(seq.name))
        normal, anom = pick_frames(seq)
        for tag, idxs in (("normal", normal), ("anomaly", anom)):
            for i in idxs:
                src = seq.frames[i]
                dst = os.path.join(d, f"{clip}_{tag}_{i:05d}.jpg")
                if isinstance(src, str):
                    im = Image.open(src).convert("RGB")
                else:
                    im = src.convert("RGB")
                im.thumbnail((640, 640))
                im.save(dst, quality=88)
        print(f"[{name}] {clip}: {len(normal)} normal, {len(anom)} anomalous")


def export_worked_example(raw_dir, out_dir):
    """One clip's scores under all four sweep conditions, for a score-vs-time plot."""
    pat = os.path.join(raw_dir, "shanghaitech_surveillance_{}_normal_*.npz")
    conds = ("none", "generic", "matched", "mismatched")
    data = {}
    for c in conds:
        hits = sorted(glob.glob(pat.format(c)))
        if not hits:
            print(f"[example] missing cache for {c}")
            return
        data[c] = np.load(hits[0], allow_pickle=True)

    z = data["none"]
    n = int(z["n"])
    names = [str(x) for x in z["names"]]
    # the clip with the most anomalous frames makes the clearest illustration
    counts = [int(np.asarray(z[f"l{i}"]).sum()) for i in range(n)]
    best = int(np.argmax(counts))

    out = {"clip": names[best], "labels": z[f"l{best}"]}
    for c in conds:
        out[f"scores_{c}"] = data[c][f"s{best}"]
    np.savez(os.path.join(out_dir, "worked_example.npz"), **out)
    print(f"[example] {names[best]}  {len(out['labels'])} frames, "
          f"{counts[best]} anomalous")


def export_embedding_sample(emb_path, out_dir, per_clip=25, seed=0):
    """A subsample of frame embeddings tagged with camera view.

    Enough to project to two dimensions and show that the thirteen views form
    thirteen separate clusters -- the reason raw pooling across them destroyed
    the ranking.
    """
    z = np.load(os.path.expanduser(emb_path), allow_pickle=True)
    feats, clip_ids = z["feats"][:, 0, :], z["clip_ids"]
    names = [str(x) for x in z["names"]]
    labels = z["labels"].astype(int)

    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(clip_ids):
        idx = np.where(clip_ids == c)[0]
        keep.append(rng.choice(idx, size=min(per_clip, len(idx)), replace=False))
    keep = np.concatenate(keep)

    view = np.array([os.path.basename(names[c]).split("_")[0]
                     for c in clip_ids[keep]])
    np.savez_compressed(os.path.join(out_dir, "embedding_sample.npz"),
                        feats=feats[keep].astype(np.float32),
                        view=view, labels=labels[keep])
    print(f"[embeddings] {len(keep)} frames from {len(set(view))} views")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--shanghaitech", default=os.path.expanduser("~/dazvad/data/shanghaitech"))
    p.add_argument("--avenue", default=os.path.expanduser("~/dazvad/data/avenue"))
    p.add_argument("--raw", default=os.path.expanduser("~/dazvad/work/raw"))
    p.add_argument("--embeddings", default=os.path.expanduser(
        "~/dazvad/work/embeddings/shanghaitech_ViT-L-14_step2_crops5.npz"))
    p.add_argument("--out", default=os.path.expanduser("~/dazvad/figure_assets"))
    p.add_argument("--frame-step", type=int, default=2)
    args = p.parse_args()

    out = os.path.expanduser(args.out)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out, exist_ok=True)

    if os.path.isdir(os.path.expanduser(args.shanghaitech)):
        export_dataset("shanghaitech",
                       dict(dataset="shanghaitech",
                            data_root=os.path.expanduser(args.shanghaitech),
                            frame_step=args.frame_step), out)
    if os.path.isdir(os.path.expanduser(args.avenue)):
        export_dataset("avenue",
                       dict(dataset="avenue",
                            data_root=os.path.expanduser(args.avenue),
                            frame_step=args.frame_step), out, max_clips=2)

    export_worked_example(os.path.expanduser(args.raw), out)
    if os.path.isfile(os.path.expanduser(args.embeddings)):
        export_embedding_sample(args.embeddings, out)

    total = sum(os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(out) for f in fs)
    print(f"\n[done] {out}  ({total / 1024 ** 2:.1f} MB)")
    print("Bundle it with:  tar czf ~/figure_assets.tgz -C ~/dazvad figure_assets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
