# -*- coding: utf-8 -*-
"""Context sweep on NWPU Campus, exactly as pre-registered in
docs/10_phase3/PREREG_nwpu_campus.md. Do not change the protocol here without a
dated addendum there.

    python notebooks/nwpu_campus_sweep.py caption   # LLaVA, one sentence per scene (~5 min)
    python notebooks/nwpu_campus_sweep.py run       # scores every condition from the cache

One sentence per SCENE (43), as for ShanghaiTech's camera views: LLaVA captions
the first three sampled frames (decoded frames 0, 16, 32) of the scene's first
test video, by position and never by label, and the medoid caption is kept.
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
from da_zvad.datasets.nwpu_campus import SCENE_DEPENDENT_SCENES, scene_of  # noqa: E402
from da_zvad.temporal import moving_average                       # noqa: E402
from per_view_descriptors import (CAPTION_PROMPT, clean, encoder,  # noqa: E402
                                  prototypes)

ROOT = "~/dazvad/data/nwpu_campus"
CACHE = "~/dazvad/work/embeddings/nwpu_campus_ViT-L-14_step16_crops1.npz"
CAPS = os.path.expanduser("~/dazvad/work/tables/nwpu_campus_captions.json")
OUT = os.path.expanduser("~/dazvad/work/tables/nwpu_campus_sweep.json")
SHARED = "a university campus with roads, footpaths, squares and buildings"
GENERIC = "a generic scene"
MISMATCHED = ("an industrial quality-inspection image of a manufactured product "
              "on a factory line")
SCALE, WINDOWS = 100.0, (5, 1)


def sequences():
    from da_zvad.config import DAZVADConfig
    from da_zvad.datasets import get_dataset
    return get_dataset(DAZVADConfig(dataset="nwpu_campus",
                                    data_root=os.path.expanduser(ROOT),
                                    frame_step=16)).sequences()


def caption(args) -> int:
    from PIL import Image
    from da_zvad.reasoning import LlavaReasoner

    llava = LlavaReasoner(max_new_tokens=40)
    pool = encoder()
    first = {}
    for s in sequences():                          # sorted by name: D001_01 before D001_02
        first.setdefault(scene_of(s.name), s)
    out, raw = {}, {}
    for k, (scene, s) in enumerate(sorted(first.items()), 1):
        caps = [clean(llava._generate(Image.open(f).convert("RGB"), CAPTION_PROMPT))
                for f in s.frames[:3]]             # by position, never by label
        emb = np.stack([pool([c]) for c in caps])
        out[scene], raw[scene] = caps[int(np.argmax((emb @ emb.T).sum(1)))], caps
        print("[%2d/%d] %s  (from %s)  %s" % (k, len(first), scene, s.name, out[scene]), flush=True)
    os.makedirs(os.path.dirname(CAPS), exist_ok=True)
    json.dump({"captions": out, "all": raw,
               "source_video": {sc: s.name for sc, s in first.items()}},
              open(CAPS, "w"), indent=1)
    print("\nwrote %s (%d scenes)" % (CAPS, len(out)))
    return 0


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
    V = [names[c] for c in clips]
    S = [scene_of(n) for n in V]
    scenes = sorted(set(S))
    print("cache: %d frames, %d videos, %d scenes, %.1f%% anomalous frames, %d anomalous videos"
          % (len(feats), len(V), len(scenes), 100 * labels_all.mean(), sum(int(l.max()) for l in L)))
    if not os.path.exists(CAPS):
        print("run the caption step first")
        return 2
    capfile = json.load(open(CAPS))
    caps, src = capfile["captions"], capfile["source_video"]
    if sorted(caps) != scenes:
        print("captions do not cover the cached scenes -- rerun caption")
        return 2
    early = [sc for sc, v in src.items() if int(L[V.index(v)][:3].max())]
    print("scenes whose captioned frames already show an anomaly: %d %s" % (len(early), early))

    pool = encoder()

    def margins(desc_for_scene):
        out = []
        for f, sc in zip(F, S):
            pn, pa = prototypes(pool, desc_for_scene(sc))
            out.append(f @ pa - f @ pn)
        return out

    def scores(m, w):
        return [moving_average(1.0 / (1.0 + np.exp(-SCALE * x)), w) for x in m]

    def evaluate(m):
        out = {}
        for w in WINDOWS:
            sm = scores(m, w)
            out["raw_w%d" % w] = evaluation.pooled_auroc(sm, L, normalize=False)
            if w == 5:
                out["norm_w5"] = evaluation.pooled_auroc(sm, L, normalize=True)
        return out

    def per_scene_auroc(m):
        """Pooled raw AUROC inside each scene that has both classes (w=5)."""
        sm = scores(m, 5)
        res = {}
        for sc in scenes:
            idx = [i for i, s in enumerate(S) if s == sc]
            lab = np.concatenate([L[i] for i in idx])
            if lab.min() != lab.max():
                res[sc] = evaluation.pooled_auroc([sm[i] for i in idx], [L[i] for i in idx],
                                                  normalize=False)
        return res

    M = {"none": margins(lambda sc: None),
         "generic": margins(lambda sc: GENERIC),
         "shared": margins(lambda sc: SHARED),
         "mismatched": margins(lambda sc: MISMATCHED),
         "per_scene": margins(lambda sc: caps[sc])}
    res = {k: evaluate(m) for k, m in M.items()}
    # Shuffled control: every rotation of the scene list, as for ShanghaiTech's views.
    rot_m = [margins(lambda sc, k=k: caps[scenes[(scenes.index(sc) + k) % len(scenes)]])
             for k in range(1, len(scenes))]
    shuf = [evaluate(m) for m in rot_m]
    keys = shuf[0].keys()
    res["shuffled_mean"] = {k: float(np.mean([r[k] for r in shuf])) for k in keys}
    res["shuffled_max"] = {k: float(np.max([r[k] for r in shuf])) for k in keys}

    order = ("none", "generic", "shared", "per_scene", "shuffled_mean", "shuffled_max", "mismatched")
    cols = ("raw_w5", "raw_w1", "norm_w5")
    print("\n%-14s" % "" + "".join("%10s" % c for c in cols) + "    (raw_w5 is PRIMARY)")
    for k in order:
        print("%-14s" % k + "".join("%10.4f" % res[k][c] for c in cols))

    # P5: is the per-scene gain over shuffled larger where anomalies are scene-dependent?
    own = per_scene_auroc(M["per_scene"])
    rot = [per_scene_auroc(m) for m in rot_m]
    gain = {sc: own[sc] - float(np.mean([r[sc] for r in rot])) for sc in own}
    sd = [sc for sc in SCENE_DEPENDENT_SCENES if sc in gain]
    other = [sc for sc in gain if sc not in SCENE_DEPENDENT_SCENES]
    sd_mean = float(np.mean([gain[sc] for sc in sd])) if sd else float("nan")
    other_median = float(np.median([gain[sc] for sc in other]))
    print("\nper-scene gain over shuffled (within-scene raw AUROC, w=5)")
    for sc in sd:
        print("  %s  %+.4f   (%s)" % (sc, gain[sc], SCENE_DEPENDENT_SCENES[sc]))
    print("  scene-dependent scenes, mean  %+.4f   (n=%d)" % (sd_mean, len(sd)))
    print("  other scenes, median          %+.4f   (n=%d)" % (other_median, len(other)))

    p = "raw_w5"
    a = lambda k: res[k][p]                              # noqa: E731
    cal = split(M["per_scene"], M["none"], L, 5, False)
    print("\npre-registered predictions (primary metric %s unless stated)" % p)
    print("  P1 per_scene vs shuffled mean  %+.4f   %s"
          % (a("per_scene") - a("shuffled_mean"),
             "HOLDS" if a("per_scene") > a("shuffled_mean") else "REFUTED"))
    print("  P2 per_scene vs none           %+.4f   %s"
          % (a("per_scene") - a("none"), "HOLDS" if a("per_scene") > a("none") else "REFUTED"))
    print("  P3 shared vs mismatched        %+.4f   %s"
          % (a("shared") - a("mismatched"), "HOLDS" if a("shared") > a("mismatched") else "REFUTED"))
    print("  P4 per_scene vs none is calibration-dominated:")
    print("     gap %+.4f, level share %.2f, macro_m gap %+.4f  ->  %s   %s"
          % (cal["gap"], cal["level_share"], cal["macro_m_gap"], verdict(cal),
             "HOLDS" if verdict(cal) == "CALIBRATION-dominated" else "REFUTED"))
    print("  P5 scene-dependent mean gain %+.4f <= other-scene median %+.4f ?   %s"
          % (sd_mean, other_median, "HOLDS" if sd_mean <= other_median else "REFUTED"))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"results": res, "shuffled_all": shuf, "p4_split": cal, "scene_gain": gain,
               "early_anomalous_caption_scenes": early}, open(OUT, "w"), indent=1, default=float)
    print("\nsaved %s" % OUT)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["caption", "run"])
    a = ap.parse_args()
    sys.exit({"caption": caption, "run": run}[a.step](a))
