# -*- coding: utf-8 -*-
"""Context sweep on UCF-Crime, exactly as pre-registered in
docs/10_phase3/PREREG_ucf_crime.md. Do not change the protocol here without a
dated addendum there.

    python notebooks/ucf_crime_sweep.py caption   # LLaVA, per video, label-free (~15 min)
    python notebooks/ucf_crime_sweep.py run       # scores every condition from the cache

Captioning uses the first three SAMPLED frames of each video (frames 0, 16, 32),
chosen by position and never by label.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from da_zvad import evaluation                                   # noqa: E402
from da_zvad.temporal import moving_average                      # noqa: E402
from per_view_descriptors import (CAPTION_PROMPT, clean, encoder,  # noqa: E402
                                  prototypes)

ROOT = "~/dazvad/data/ucf_crime"
CACHE = "~/dazvad/work/embeddings/ucf_crime_ViT-L-14_step16_crops1.npz"
CAPS = os.path.expanduser("~/dazvad/work/tables/ucf_crime_captions.json")
OUT = os.path.expanduser("~/dazvad/work/tables/ucf_crime_sweep.json")
SHARED = "real-world CCTV surveillance footage of streets, shops and buildings"
GENERIC = "a generic scene"
MISMATCHED = ("an industrial quality-inspection image of a manufactured product "
              "on a factory line")
SCALE, WINDOWS, SEED, N_SHUFFLE = 100.0, (5, 1), 20261001, 20


def sequences():
    from da_zvad.config import DAZVADConfig
    from da_zvad.datasets import get_dataset
    return get_dataset(DAZVADConfig(dataset="ucf_crime",
                                    data_root=os.path.expanduser(ROOT),
                                    frame_step=16)).sequences()


def caption(args) -> int:
    from PIL import Image
    from da_zvad.reasoning import LlavaReasoner

    llava = LlavaReasoner(max_new_tokens=40)
    pool = encoder()
    done = json.load(open(CAPS)) if os.path.exists(CAPS) else {}
    seqs = sequences()
    for k, s in enumerate(seqs, 1):
        if s.name in done:
            continue
        caps = [clean(llava._generate(Image.open(f).convert("RGB"), CAPTION_PROMPT))
                for f in s.frames[:3]]                     # by position, never by label
        emb = np.stack([pool([c]) for c in caps])
        done[s.name] = caps[int(np.argmax((emb @ emb.T).sum(1)))]
        print("[%3d/%d] %-34s %s" % (k, len(seqs), s.name, done[s.name]), flush=True)
        if k % 10 == 0:
            json.dump(done, open(CAPS, "w"), indent=1)     # resumable
    json.dump(done, open(CAPS, "w"), indent=1)
    print("\nwrote %s (%d captions)" % (CAPS, len(done)))
    return 0


def run(args) -> int:
    z = np.load(os.path.expanduser(CACHE), allow_pickle=True)
    feats = z["feats"]
    feats = feats[:, 0, :] if feats.ndim == 3 else feats
    feats = feats.astype(np.float64)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    labels, clip_ids, names = z["labels"].astype(int), z["clip_ids"], list(z["names"])
    print("cache: %d frames, %d videos, %.1f%% anomalous frames"
          % (len(feats), len(names), 100 * labels.mean()))
    if not os.path.exists(CAPS):
        print("run the caption step first")
        return 2
    caps = json.load(open(CAPS))
    missing = [n for n in names if n not in caps]
    if missing:
        print("%d videos lack a caption (e.g. %s) -- rerun caption" % (len(missing), missing[:2]))
        return 2

    pool = encoder()
    clips = np.unique(clip_ids)

    def evaluate(desc_for_clip):
        s = np.empty(len(feats))
        for c in clips:
            m = clip_ids == c
            pn, pa = prototypes(pool, desc_for_clip(int(c)))
            s[m] = 1.0 / (1.0 + np.exp(-SCALE * (feats[m] @ pa - feats[m] @ pn)))
        out = {}
        for w in WINDOWS:
            seqs = [moving_average(s[clip_ids == c], w) for c in clips]
            labs = [labels[clip_ids == c] for c in clips]
            out["raw_w%d" % w] = evaluation.pooled_auroc(seqs, labs, normalize=False)
            out["norm_w%d" % w] = evaluation.pooled_auroc(seqs, labs, normalize=True)
        return out

    per_clip = [caps[names[c]] for c in range(len(names))]
    res = {
        "none": evaluate(lambda c: None),
        "generic": evaluate(lambda c: GENERIC),
        "shared": evaluate(lambda c: SHARED),
        "mismatched": evaluate(lambda c: MISMATCHED),
        "per_video": evaluate(lambda c: per_clip[c]),
    }
    rng = np.random.default_rng(SEED)
    shuf = []
    n = len(names)
    for _ in range(N_SHUFFLE):
        while True:                                   # derangement: nobody keeps their own
            perm = rng.permutation(n)
            if not np.any(perm == np.arange(n)):
                break
        shuf.append(evaluate(lambda c, p=perm: per_clip[p[c]]))
    keys = shuf[0].keys()
    res["shuffled_mean"] = {k: float(np.mean([r[k] for r in shuf])) for k in keys}
    res["shuffled_max"] = {k: float(np.max([r[k] for r in shuf])) for k in keys}

    order = ("none", "generic", "shared", "per_video", "shuffled_mean", "shuffled_max", "mismatched")
    cols = ("raw_w5", "raw_w1", "norm_w5", "norm_w1")
    print("\n%-14s" % "" + "".join("%10s" % c for c in cols) + "    (raw_w5 is PRIMARY)")
    for k in order:
        print("%-14s" % k + "".join("%10.4f" % res[k][c] for c in cols))

    p = "raw_w5"
    gap = res["per_video"][p] - res["mismatched"][p]
    print("\npre-registered predictions (primary metric %s)" % p)
    print("  P1 per_video - mismatched = %+.4f  > +0.105 ?  %s"
          % (gap, "HOLDS" if gap > 0.105 else "REFUTED"))
    print("  P2 per_video vs shuffled mean  %+.4f         %s"
          % (res["per_video"][p] - res["shuffled_mean"][p],
             "HOLDS" if res["per_video"][p] > res["shuffled_mean"][p] else "REFUTED"))
    print("  P3 per_video vs shared         %+.4f         %s"
          % (res["per_video"][p] - res["shared"][p],
             "HOLDS" if res["per_video"][p] > res["shared"][p] else "REFUTED"))
    print("  P4 shared vs mismatched        %+.4f         %s"
          % (res["shared"][p] - res["mismatched"][p],
             "HOLDS" if res["shared"][p] > res["mismatched"][p] else "REFUTED"))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"results": res, "shuffled_all": shuf}, open(OUT, "w"), indent=1)
    print("\nsaved %s" % OUT)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["caption", "run"])
    a = ap.parse_args()
    sys.exit({"caption": caption, "run": run}[a.step](a))
