# -*- coding: utf-8 -*-
"""Measure prototype dilution directly, instead of inferring it from AUROC.

The framework's central mechanism claim is that injecting the scene descriptor
into both prompt ensembles pulls the normal and abnormal prototypes toward each
other, contracting the margin the decision rests on -- and that this is why an
accurate descriptor did the most damage, since accurate text aligns with every
frame and so contributes the largest common component.

Every piece of evidence for that account so far is indirect: the ordering of the
sweep inverted, and correcting the fusion rule inverted it back. The quantity the
account is actually about -- the angle between the two prototypes -- has never
been computed.

It is cheap to compute. The prototypes are mean-pooled text embeddings, so this
needs the text encoder and nothing else: no images, no GPU time worth counting.

What the account predicts, and what would refute it:

    Under ``both`` fusion, prototype similarity should be HIGHEST for the matched
    descriptor and lowest for mismatched, and should track the measured AUROC
    inversely -- more similar prototypes, worse detection.

    Under ``normal`` fusion the abnormal prototype never moves, so the collapse
    mechanism cannot operate and the similarities should vary far less.

Similarities that are flat across conditions, or that order the wrong way, would
mean the dilution account is wrong however well it fits the AUROC numbers.

Usage:
    python notebooks/prototype_distance.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from da_zvad.prompts import get_prompts              # noqa: E402
from da_zvad.context import VerbalizedContext        # noqa: E402
from da_zvad.context_sweep import MISMATCHED         # noqa: E402

# Measured AUROC at w=31, all 107 ShanghaiTech clips, per-clip normalised,
# read from the driver's cached scores. Listed here so the relationship between
# prototype geometry and detection can be seen in one table.
AUROC = {
    "both":   {"none": 0.7067, "generic": 0.6704, "matched": 0.6662, "mismatched": 0.6950},
    "normal": {"none": 0.7067, "generic": 0.6907, "matched": 0.7336, "mismatched": 0.6283},
}


def prototype(enc, prompts):
    """Mean-pool an ensemble into one unit vector -- what the pipeline scores against."""
    f = enc.encode_texts(list(prompts))
    v = f.mean(0)
    return v / np.linalg.norm(v)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="surveillance")
    p.add_argument("--matched",
                   default="a university campus walkway with pedestrians")
    p.add_argument("--clip-model", default="ViT-L-14")
    p.add_argument("--clip-pretrained", default="laion2b_s32b_b82k")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    from da_zvad.encoders import CLIPEncoder
    enc = CLIPEncoder(args.clip_model, args.clip_pretrained, args.device)

    base_n, base_a = get_prompts(args.domain)
    mismatched = MISMATCHED.get(args.domain, MISMATCHED["generic"])
    descriptions = {
        "generic": "a generic scene",
        "matched": args.matched,
        "mismatched": mismatched,
    }

    print(f"domain={args.domain}  model={args.clip_model}")
    print(f"matched     : \"{args.matched}\"")
    print(f"mismatched  : \"{mismatched}\"\n")

    rows = []
    for mode in ("both", "normal"):
        print(f"=== fusion = {mode} " + "=" * 46)
        print(f"{'condition':<14}{'cos(P+,P-)':>12}{'angle':>9}"
              f"{'margin':>9}{'AUROC':>9}")
        print("-" * 53)
        for cond in ("none", "generic", "matched", "mismatched"):
            if cond == "none":
                pn, pa = base_n, base_a
            else:
                pn, pa = VerbalizedContext(descriptions[cond],
                                           mode=mode).ground(base_n, base_a)
            vn, va = prototype(enc, pn), prototype(enc, pa)
            cos = float(vn @ va)
            angle = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
            auroc = AUROC[mode][cond]
            print(f"{cond:<14}{cos:>12.4f}{angle:>8.1f}°"
                  f"{1 - cos:>9.4f}{auroc:>9.4f}")
            rows.append({"mode": mode, "condition": cond, "cos": cos,
                         "angle": angle, "auroc": auroc})
        print()

    # Does prototype separation predict detection, within each fusion mode?
    print("=" * 60)
    for mode in ("both", "normal"):
        sub = [r for r in rows if r["mode"] == mode]
        c = np.array([r["cos"] for r in sub])
        a = np.array([r["auroc"] for r in sub])
        if c.std() < 1e-9:
            print(f"{mode:<8} prototypes identical across conditions")
            continue
        r = float(np.corrcoef(c, a)[0, 1])
        spread = c.max() - c.min()
        print(f"{mode:<8} similarity spread {spread:.4f}   "
              f"corr(similarity, AUROC) = {r:+.3f}")

    print()
    print("The dilution account predicts, under 'both': matched most similar,")
    print("mismatched least, and a NEGATIVE correlation -- prototypes closer")
    print("together means worse detection. Under 'normal' the abnormal")
    print("prototype never moves, so the spread should be much smaller.")
    print()
    print("Flat similarities, or the wrong ordering, would refute the account")
    print("regardless of how well it fits the AUROC numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
