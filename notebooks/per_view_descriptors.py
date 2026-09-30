# -*- coding: utf-8 -*-
"""Per-camera scene descriptors on ShanghaiTech (Phase 3, priority 1).

Every published ShanghaiTech figure gives all twelve camera views the SAME
sentence ("a university campus walkway with pedestrians"). Result 5 concludes
that what the descriptor mainly supplies is WHICH scene is in view. If so, a
sentence written for each camera should beat one shared sentence -- and that is
the direct test of the paper's own mechanism.

Conditions, all on the cached embeddings, normal-ensemble fusion, w=31:
    none           no descriptor                         (published 0.707)
    shared         one sentence for all views            (published 0.734)
    per_view       each view gets its own sentence       <- the test
    shuffled       each view gets ANOTHER view's sentence <- the control
The shuffled control matters: every sentence in it is a plausible campus scene,
so a per_view win over shuffled cannot come from recognising an implausible
description, only from the sentence matching the camera.

Validation gate: `none` and `shared` must reproduce the published 0.707 / 0.734
before anything else is printed.

Step 1 (on the server):  python notebooks/per_view_descriptors.py sheet
    -> ~/dazvad/work/view_sheet.jpg, one normal frame per view; copy to laptop.
Step 2 (after descriptors are written into notebooks/view_descriptors.json):
                          python notebooks/per_view_descriptors.py run
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from da_zvad import evaluation                                   # noqa: E402
from da_zvad.context.verbalized import VerbalizedContext         # noqa: E402
from da_zvad.prompts.templates import get_prompts                # noqa: E402
from da_zvad.temporal import moving_average                      # noqa: E402

CACHE = "~/dazvad/work/embeddings/shanghaitech_ViT-L-14_step2_crops5.npz"
DATA = "~/dazvad/data/shanghaitech"
DESC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "view_descriptors.json")
SHARED = "a university campus walkway with pedestrians"
DOMAIN, WINDOW, SCALE = "surveillance", 31, 100.0
PUBLISHED = {"none": 0.707, "shared": 0.734}


def view_of(name: str) -> str:
    return os.path.basename(str(name))[:2]


# ---------------------------------------------------------------- step 1
def sheet(args) -> int:
    from PIL import Image, ImageDraw
    from da_zvad.config import DAZVADConfig
    from da_zvad.datasets import get_dataset

    seqs = get_dataset(DAZVADConfig(dataset="shanghaitech",
                                    data_root=os.path.expanduser(DATA),
                                    frame_step=2)).sequences()
    picked = {}
    for s in seqs:
        v = view_of(s.name)
        if v in picked:
            continue
        normal = np.where(np.asarray(s.labels) == 0)[0]
        if len(normal):
            picked[v] = s.frames[int(normal[len(normal) // 2])]
    tw, th = 320, 180
    views = sorted(picked)
    cols = 4
    rows = (len(views) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * tw, rows * th), "white")
    for i, v in enumerate(views):
        im = picked[v]
        im = (Image.open(im) if isinstance(im, str) else im).convert("RGB").resize((tw, th))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 58, 26], fill="black")
        d.text((6, 6), "view " + v, fill="white")
        canvas.paste(im, ((i % cols) * tw, (i // cols) * th))
    out = os.path.expanduser("~/dazvad/work/view_sheet.jpg")
    canvas.save(out, quality=85)
    print("wrote %s  (%d views: %s)" % (out, len(views), " ".join(views)))
    return 0


# ---------------------------------------------------------------- step 2
def encoder():
    import torch
    import torch.nn.functional as F
    import open_clip

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="laion2b_s32b_b82k")
    m = m.to(dev).eval()
    tok = open_clip.get_tokenizer("ViT-L-14")
    cache = {}

    def pool(prompts):
        key = tuple(prompts)
        if key not in cache:
            with torch.no_grad():
                f = F.normalize(m.encode_text(tok(list(prompts)).to(dev)), dim=-1)
                cache[key] = F.normalize(f.mean(0), dim=-1).cpu().numpy().astype(np.float64)
        return cache[key]
    return pool


def prototypes(pool, desc):
    n, a = get_prompts(DOMAIN)
    if desc is not None:
        n, a = VerbalizedContext(description=desc, mode="normal").ground(n, a)
    return pool(n), pool(a)


def score(feats, clip_ids, labels, views, pool, desc_for_view):
    """desc_for_view: view -> sentence or None. Returns published-protocol AUROC."""
    s = np.empty(len(feats))
    for v in np.unique(views):
        m = views == v
        pn, pa = prototypes(pool, desc_for_view(v))
        s[m] = 1.0 / (1.0 + np.exp(-SCALE * (feats[m] @ pa - feats[m] @ pn)))
    seqs, labs = [], []
    for c in np.unique(clip_ids):
        m = clip_ids == c
        seqs.append(moving_average(s[m], WINDOW))
        labs.append(labels[m])
    return evaluation.pooled_auroc(seqs, labs, normalize=True), seqs, labs


def run(args) -> int:
    z = np.load(os.path.expanduser(CACHE), allow_pickle=True)
    feats = z["feats"][:, 0, :].astype(np.float64)          # crop 0 = whole frame
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    labels, clip_ids = z["labels"].astype(int), z["clip_ids"]
    names = z["names"]
    views = np.array([view_of(names[c]) for c in clip_ids])
    print("cache: %d frames, %d clips, %d views" % (len(feats), len(names), len(set(views))))

    pool = encoder()
    res = {}
    res["none"], _, _ = score(feats, clip_ids, labels, views, pool, lambda v: None)
    res["shared"], _, _ = score(feats, clip_ids, labels, views, pool, lambda v: SHARED)

    print("\nvalidation")
    ok = True
    for k, pub in PUBLISHED.items():
        good = abs(res[k] - pub) < 0.002
        ok &= good
        print("  %-7s %.4f   published %.3f   %s" % (k, res[k], pub, "ok" if good else "MISMATCH"))
    if not ok:
        print("VALIDATION FAILED -- nothing further reported.")
        return 1

    if not os.path.exists(DESC):
        print("\nno %s yet -- write the per-view sentences, then rerun." % DESC)
        return 2
    desc = json.load(open(DESC, encoding="utf-8"))
    vs = sorted(set(views))
    missing = [v for v in vs if v not in desc]
    if missing:
        print("descriptor missing for views %s" % missing)
        return 2

    res["per_view"], _, _ = score(feats, clip_ids, labels, views, pool, lambda v: desc[v])

    # Shuffled control: every derangement-by-rotation of the view list, averaged,
    # so the result does not hinge on one lucky or unlucky pairing.
    shuf = []
    for k in range(1, len(vs)):
        rot = {v: desc[vs[(i + k) % len(vs)]] for i, v in enumerate(vs)}
        shuf.append(score(feats, clip_ids, labels, views, pool, lambda v, r=rot: r[v])[0])
    res["shuffled_mean"], res["shuffled_max"] = float(np.mean(shuf)), float(np.max(shuf))

    print("\nresults (w=%d, per-clip normalised micro AUROC, all %d clips)" % (WINDOW, len(names)))
    for k in ("none", "shared", "per_view", "shuffled_mean", "shuffled_max"):
        print("  %-14s %.4f" % (k, res[k]))
    print("\n  per_view - shared      %+.4f" % (res["per_view"] - res["shared"]))
    print("  per_view - shuffled    %+.4f   (beats all %d rotations: %s)"
          % (res["per_view"] - res["shuffled_mean"], len(shuf),
             "YES" if res["per_view"] > res["shuffled_max"] else "no"))

    out = os.path.expanduser("~/dazvad/work/tables/per_view_descriptors.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"results": res, "shuffled_all": shuf, "descriptors": desc}, open(out, "w"), indent=1)
    print("\nsaved %s" % out)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["sheet", "run"])
    a = p.parse_args()
    sys.exit(sheet(a) if a.step == "sheet" else run(a))
