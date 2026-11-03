# -*- coding: utf-8 -*-
"""Try retrieved normality on your own photos (laptop, CPU, no dataset needed).

Give it one or more photos of the SAME place, as if they were a new camera's
setup footage. It prints what the system would decide "normal" means there:

    python notebooks/try_retrieval_on_photo.py gate1.jpg gate2.jpg gate3.jpg
    python notebooks/try_retrieval_on_photo.py canteen.jpg --no-campus

Steps, exactly as the experiment (da_zvad/context/retrieval.py):
  1. CLIP encodes each photo (the same frozen ViT-L/14 as the detector);
  2. the photos' embeddings are averaged into one "camera vector" (with real
     footage, only the calmest frames are used; with your own photos, all are);
  3. that vector is compared with every phrase in the library
     (da_zvad/prompts/banks.py) and the 5 closest are kept;
  4. self-check: if the best phrase matches no better than "a photo of a place",
     the library does not know this scene and the detector would fall back.

Finally each photo is scored by today's detector without and with the retrieved
phrases, so you can see what they change. First run downloads nothing if the
CLIP weights are already cached (they are on this laptop); otherwise ~1.7 GB.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from da_zvad.context import retrieval as rt                 # noqa: E402
from da_zvad.prompts import banks                           # noqa: E402
from da_zvad.prompts.templates import get_prompts           # noqa: E402

LAM = 100.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("photos", nargs="+")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--no-campus", action="store_true",
                    help="remove every campus phrase from the library first")
    a = ap.parse_args()

    import open_clip
    import torch
    import torch.nn.functional as F
    from PIL import Image

    print("loading CLIP ViT-L/14 (CPU, ~30 s) ...")
    model, _, prep = open_clip.create_model_and_transforms("ViT-L-14", pretrained="laion2b_s32b_b82k")
    model.eval()
    tok = open_clip.get_tokenizer("ViT-L-14")

    def text(phrases):
        with torch.no_grad():
            return F.normalize(model.encode_text(tok(phrases)), dim=-1).numpy().astype(np.float64)

    with torch.no_grad():
        imgs = torch.stack([prep(Image.open(p).convert("RGB")) for p in a.photos])
        feats = F.normalize(model.encode_image(imgs), dim=-1).numpy().astype(np.float64)

    lib = banks.library(exclude_tags=("campus",) if a.no_campus else ())
    texts = [e["text"] for e in lib]
    lib_emb, neutral = text(texts), text(banks.NEUTRAL)

    v = rt.camera_vector(feats, range(len(feats)))
    idx, sims = rt.retrieve(v, lib_emb, a.k)
    ok = rt.passes_self_check(v, float(sims[0]), neutral)
    print("\nlibrary: %d phrases%s" % (len(texts), " (campus removed)" if a.no_campus else ""))
    print("camera vector from %d photo(s)\n" % len(feats))
    print("  top %d normal descriptions for this place:" % a.k)
    for j, s in zip(idx, sims):
        print("    %.3f  %s" % (s, texts[j]))
    print("  best neutral phrase:  %.3f  (%s)"
          % (float(np.max(neutral @ v)), banks.NEUTRAL[int(np.argmax(neutral @ v))]))
    print("  self-check: %s" % ("PASS -- these phrases join the normal prompts" if ok else
                                "FAIL -- library does not know this place; base prompts only"))

    n0, a0 = get_prompts("surveillance")
    nb, ab = text(list(n0)), text(list(a0))
    # As in the experiment: the best phrase becomes the scene sentence and enters
    # through the same two templates LLaVA's sentence used.
    from da_zvad.context.verbalized import VerbalizedContext
    sentence = texts[idx[0]] if ok else None
    extra = (text(VerbalizedContext(description=sentence).ground([], [])[0]) if ok
             else np.zeros((0, nb.shape[1])))
    print("\n  scene sentence used: %s" % (sentence or "(none -- fallback)"))

    def p_abnormal(N, A):
        pn = N.mean(0) / np.linalg.norm(N.mean(0))
        pa = A.mean(0) / np.linalg.norm(A.mean(0))
        m = feats @ pa - feats @ pn
        return 1.0 / (1.0 + np.exp(-LAM * m))

    before, after = p_abnormal(nb, ab), p_abnormal(np.vstack([nb, extra]), ab)
    print("\n  anomaly score per photo (today's detector):")
    print("    %-40s %10s %10s" % ("photo", "without", "with"))
    for p, b, c in zip(a.photos, before, after):
        print("    %-40s %10.2e %10.2e" % (os.path.basename(p)[:40], b, c))
    print("\n  Lower 'with' = the retrieved phrases make this view look more normal.")
    print("  Try a photo with something unusual in it: its score should stay higher.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
