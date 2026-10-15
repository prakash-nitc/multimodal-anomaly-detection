# -*- coding: utf-8 -*-
"""Context sweep on XD-Violence, exactly as pre-registered in
docs/10_phase3/PREREG_xd_violence.md. Do not change the protocol here without a
dated addendum there.

    python notebooks/xd_violence_sweep.py caption   # LLaVA, per video, label-free (~40 min)
    python notebooks/xd_violence_sweep.py run       # scores every condition from the cache

Captioning uses the first three SAMPLED frames of each video (decoded frames 0,
16 and 32), chosen by position and never by label.

Primary metric (the benchmark's own): frame-level AP computed as the authors'
test.py does -- sklearn ``auc(recall, precision)`` over every official frame,
with each snippet's score repeated 16 times against make_gt.py's ground truth
(see da_zvad/datasets/xd_violence.py).
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

from da_zvad import evaluation                                    # noqa: E402
from da_zvad.datasets.xd_violence import (SNIPPET, parse_annotation,  # noqa: E402
                                          snippet_gt)
from da_zvad.temporal import moving_average                       # noqa: E402
from per_view_descriptors import clean, encoder, prototypes      # noqa: E402

ROOT = "~/dazvad/data/xd_violence"
CACHE = "~/dazvad/work/embeddings/xd_violence_ViT-L-14_step16_crops1.npz"
CAPS = os.path.expanduser("~/dazvad/work/tables/xd_violence_captions.json")
OUT = os.path.expanduser("~/dazvad/work/tables/xd_violence_sweep.json")
SHARED = "scenes from films, television, news, sports and online videos"
GENERIC = "a generic scene"
MISMATCHED = ("an industrial quality-inspection image of a manufactured product "
              "on a factory line")
SCALE, WINDOWS, SEED, N_SHUFFLE = 100.0, (5, 1), 20261016, 20
UCF_OWN_VS_SHUFFLED = 0.110        # UCF-Crime per-video minus shuffled mean, AUROC raw w=5

# Identical to the ShanghaiTech / UCF-Crime instruction except that XD-Violence
# is not fixed-camera surveillance: "fixed surveillance camera frame" becomes
# "video frame". Registered in the pre-registration.
CAPTION_PROMPT = (
    "USER: <image>\nDescribe the place shown in this video frame in one short "
    "sentence. Describe only the location and its layout (paths, roads, "
    "buildings, trees, water). Do not mention people or what anyone is doing. "
    "ASSISTANT:")


def sequences():
    from da_zvad.config import DAZVADConfig
    from da_zvad.datasets import get_dataset
    return get_dataset(DAZVADConfig(dataset="xd_violence",
                                    data_root=os.path.expanduser(ROOT),
                                    frame_step=SNIPPET)).sequences()


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
        print("[%3d/%d] %-50s %s" % (k, len(seqs), s.name[:50], done[s.name]), flush=True)
        if k % 10 == 0:
            json.dump(done, open(CAPS, "w"), indent=1)     # resumable
    json.dump(done, open(CAPS, "w"), indent=1)
    print("\nwrote %s (%d captions)" % (CAPS, len(done)))
    return 0


def official_ap_auc(seqs, gts):
    """AP and AUROC exactly as test.py: snippet scores repeated 16x, all frames pooled."""
    from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
    s = np.concatenate([np.repeat(x, SNIPPET) for x in seqs])
    y = np.concatenate(gts)
    p, r, _ = precision_recall_curve(y, s)
    return float(auc(r, p)), float(roc_auc_score(y, s))


def run(args) -> int:
    from measurement_audit import split, verdict

    z = np.load(os.path.expanduser(CACHE), allow_pickle=True)
    feats = z["feats"]
    feats = (feats[:, 0, :] if feats.ndim == 3 else feats).astype(np.float64)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    labels_all, clip_ids, names = z["labels"].astype(int), z["clip_ids"], list(z["names"])
    clips = np.unique(clip_ids)
    F = [feats[clip_ids == c] for c in clips]
    L = [labels_all[clip_ids == c] for c in clips]
    ann = parse_annotation(os.path.join(os.path.expanduser(ROOT), "annotations.txt"))
    G = [snippet_gt(len(l), ann.get(names[c].split("/", 1)[1], [])) for c, l in zip(clips, L)]
    for g, l in zip(G, L):                          # sampled labels = official labels at 16k
        assert np.array_equal(g[::SNIPPET], l)
    n_viol = sum(int(g.max()) for g in G)
    print("cache: %d snippets, %d videos (%d violent), %d official frames, %.1f%% violent"
          % (len(feats), len(names), n_viol, sum(len(g) for g in G),
             100 * np.concatenate(G).mean()))
    early = sum(int(l[:3].max()) for l in L)
    print("videos whose violence has begun within the 3 captioned frames: %d of %d" % (early, n_viol))
    if not os.path.exists(CAPS):
        print("run the caption step first")
        return 2
    caps = json.load(open(CAPS))
    missing = [n for n in names if n not in caps]
    if missing:
        print("%d videos lack a caption (e.g. %s) -- rerun caption" % (len(missing), missing[:2]))
        return 2

    pool = encoder()
    per = [caps[names[c]] for c in clips]

    def margins(desc_for_video):
        out = []
        for i, f in enumerate(F):
            pn, pa = prototypes(pool, desc_for_video(i))
            out.append(f @ pa - f @ pn)
        return out

    def evaluate(m):
        s = [1.0 / (1.0 + np.exp(-SCALE * x)) for x in m]
        out = {}
        for w in WINDOWS:
            sm = [moving_average(x, w) for x in s]
            out["ap_w%d" % w], out["auc_w%d" % w] = official_ap_auc(sm, G)
            if w == 5:
                nm = [evaluation.normalize_per_sequence(x) for x in sm]
                out["ap_norm_w5"], out["auc_norm_w5"] = official_ap_auc(nm, G)
                out["auc_sampled_w5"] = evaluation.pooled_auroc(sm, L, normalize=False)
        return out

    M = {"none": margins(lambda i: None),
         "generic": margins(lambda i: GENERIC),
         "shared": margins(lambda i: SHARED),
         "mismatched": margins(lambda i: MISMATCHED),
         "per_video": margins(lambda i: per[i])}
    res = {k: evaluate(m) for k, m in M.items()}
    rng = np.random.default_rng(SEED)
    shuf = []
    n = len(F)
    for _ in range(N_SHUFFLE):
        while True:                                   # derangement: nobody keeps their own
            perm = rng.permutation(n)
            if not np.any(perm == np.arange(n)):
                break
        shuf.append(evaluate(margins(lambda i, p=perm: per[p[i]])))
    keys = shuf[0].keys()
    res["shuffled_mean"] = {k: float(np.mean([r[k] for r in shuf])) for k in keys}
    res["shuffled_max"] = {k: float(np.max([r[k] for r in shuf])) for k in keys}

    order = ("none", "generic", "shared", "per_video", "shuffled_mean", "shuffled_max", "mismatched")
    cols = ("ap_w5", "ap_w1", "auc_w5", "auc_w1", "ap_norm_w5", "auc_sampled_w5")
    print("\n%-14s" % "" + "".join("%15s" % c for c in cols) + "    (ap_w5 is PRIMARY)")
    for k in order:
        print("%-14s" % k + "".join("%15.4f" % res[k][c] for c in cols))

    p = "ap_w5"
    a = lambda k: res[k][p]                              # noqa: E731
    cal = split(M["per_video"], M["none"], L, 5, False)
    # like for like with UCF-Crime's +0.110: pooled AUROC over the sampled frames
    own_vs_shuf = res["per_video"]["auc_sampled_w5"] - res["shuffled_mean"]["auc_sampled_w5"]
    print("\npre-registered predictions (primary metric %s unless stated)" % p)
    print("  P1 per_video vs shuffled mean   %+.4f   %s"
          % (a("per_video") - a("shuffled_mean"),
             "HOLDS" if a("per_video") > a("shuffled_mean") else "REFUTED"))
    print("  P2 shared vs mismatched         %+.4f   %s"
          % (a("shared") - a("mismatched"), "HOLDS" if a("shared") > a("mismatched") else "REFUTED"))
    print("  P3 per_video vs none            %+.4f   %s"
          % (a("per_video") - a("none"), "HOLDS" if a("per_video") > a("none") else "REFUTED"))
    print("  P4 per_video vs none is calibration-dominated (sampled AUROC, raw, w=5):")
    print("     gap %+.4f, level share %.2f, macro_m gap %+.4f  ->  %s   %s"
          % (cal["gap"], cal["level_share"], cal["macro_m_gap"], verdict(cal),
             "HOLDS" if verdict(cal) == "CALIBRATION-dominated" else "REFUTED"))
    print("  P5 per_video - shuffled (sampled AUROC w=5) %+.4f < UCF-Crime's %+.3f ?   %s"
          % (own_vs_shuf, UCF_OWN_VS_SHUFFLED,
             "HOLDS" if own_vs_shuf < UCF_OWN_VS_SHUFFLED else "REFUTED"))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"results": res, "shuffled_all": shuf, "p4_split": cal,
               "early_violent_captions": early}, open(OUT, "w"), indent=1, default=float)
    print("\nsaved %s" % OUT)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["caption", "run"])
    a = ap.parse_args()
    sys.exit({"caption": caption, "run": run}[a.step](a))
