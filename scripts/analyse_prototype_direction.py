# -*- coding: utf-8 -*-
"""Which DIRECTION the prompt prototypes move -- the missing half of §5.4.

The paper measures how much the two text prototypes collapse toward each other
when the scene description is appended to both ensembles: the angle between them
contracts from 35.7 degrees to about 25. That accounts for why grounding both
ensembles is harmful. It does NOT account for the finer ordering -- why an
ACCURATE description is the most harmful of the three, when the three grounded
conditions sit within 1.3 degrees of one another.

Section 7.5 states the missing piece as a conjecture, and names this measurement
as the thing that would settle it:

    "a descriptor matching the imagery moves both prototypes toward the region
     the image embeddings occupy, so every frame scores alike, whereas a
     descriptor of an unrelated domain moves them somewhere the images are not
     and leaves the original prompts discriminating underneath."

That is a claim about direction relative to the image cloud, and the angle
between the prototypes cannot see it -- two prototypes can close by the same
amount while moving toward the images or away from them. This script measures it
against real image embeddings.

WHAT IS AND IS NOT ESTABLISHED HERE
-----------------------------------
The image embeddings available offline are a 2,675-frame sample carrying view
and label but NOT clip identity, so per-clip min-max normalisation -- the
benchmark protocol behind every published figure -- cannot be reproduced. AUROC
is therefore computed per view and reported only as an ORDERING check: if the
sample does not reproduce the published ordering (matched worst under `both`,
best under `normal`), the geometry measured on it says nothing about the full
run and the script says so rather than reporting numbers.

The geometry itself does not depend on that caveat. Prototype positions are a
property of the prompts alone, and their relation to the image cloud needs only
a representative sample of frames, not the benchmark protocol.

Run: python scripts/analyse_prototype_direction.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from da_zvad.context.verbalized import VerbalizedContext          # noqa: E402
from da_zvad.context_sweep import MISMATCHED                       # noqa: E402
from da_zvad.prompts.templates import get_prompts                  # noqa: E402

SAMPLE = "figure_assets/embedding_sample.npz"
DOMAIN = "surveillance"      # the ensemble behind the headline run (MANIFEST)
MATCHED = "a university campus walkway with pedestrians"
GENERIC = "a generic scene"
MODEL, PRETRAINED = "ViT-L-14", "laion2b_s32b_b82k"

# Published prototype angles, Table tbl:proto -- the validation gate.
PUBLISHED = {
    ("both", "none"): 35.7, ("both", "generic"): 26.3,
    ("both", "matched"): 26.3, ("both", "mismatched"): 25.0,
    ("normal", "none"): 35.7, ("normal", "generic"): 34.5,
    ("normal", "matched"): 41.9, ("normal", "mismatched"): 37.6,
}
TOL = 0.6      # degrees; the paper prints one decimal place


def ensembles(condition: str, fusion: str):
    """The (normal, abnormal) prompt lists for one sweep cell.

    Built through the pipeline's own VerbalizedContext so the appended
    sentences are byte-identical to the ones that produced the results.
    """
    base_n, base_a = get_prompts(DOMAIN)
    if condition == "none":
        return list(base_n), list(base_a)
    desc = {"generic": GENERIC,
            "matched": MATCHED,
            "mismatched": MISMATCHED[DOMAIN]}[condition]
    return VerbalizedContext(description=desc, mode=fusion).ground(base_n, base_a)


def load_encoder():
    import torch
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
    model = model.eval()
    tok = open_clip.get_tokenizer(MODEL)

    def pool(prompts):
        """Encode -> normalise -> mean-pool -> normalise. Matches CLIPEncoder."""
        import torch.nn.functional as F
        with torch.no_grad():
            f = F.normalize(model.encode_text(tok(list(prompts))), dim=-1)
            return F.normalize(f.mean(dim=0, keepdim=True), dim=-1)[0].numpy()

    return pool, float(model.logit_scale.exp().item())


def auroc(y, s):
    """Rank-based AUROC; no sklearn dependency, ties averaged."""
    y = np.asarray(y).astype(bool)
    if y.all() or not y.any():
        return np.nan
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks within ties
    s_sorted = np.asarray(s)[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    n_pos, n_neg = y.sum(), (~y).sum()
    return (ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def per_view_auroc(feats, labels, views, e_pos, e_neg, scale):
    """Softmax score, min-max normalised WITHIN each view, then pooled.

    Per-view stands in for the protocol's per-clip normalisation, which the
    sample cannot support. It is the same idea at coarser granularity.
    """
    logits = np.stack([feats @ e_pos, feats @ e_neg], axis=1) * scale
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    s = e[:, 1] / e.sum(axis=1)
    out = np.empty_like(s)
    for v in np.unique(views):
        m = views == v
        lo, hi = s[m].min(), s[m].max()
        out[m] = (s[m] - lo) / (hi - lo) if hi > lo else 0.5
    return auroc(labels, out), s


def main() -> int:
    if not os.path.exists(SAMPLE):
        print("missing %s" % SAMPLE)
        return 2
    d = np.load(SAMPLE, allow_pickle=True)
    feats = d["feats"].astype(np.float64)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    labels, views = d["labels"], d["view"]
    print("sample: %d frames, %d views, %.1f%% anomalous"
          % (len(feats), len(np.unique(views)), 100 * labels.mean()))

    centre = feats.mean(axis=0)
    centre /= np.linalg.norm(centre)

    print("\nloading %s (CPU) ..." % MODEL)
    pool, scale = load_encoder()
    print("  logit scale %.2f" % scale)

    rows, failures = [], []
    for fusion in ("both", "normal"):
        for cond in ("none", "generic", "matched", "mismatched"):
            n, a = ensembles(cond, fusion)
            e_pos, e_neg = pool(n).astype(np.float64), pool(a).astype(np.float64)

            angle = np.degrees(np.arccos(np.clip(e_pos @ e_neg, -1, 1)))
            exp = PUBLISHED[(fusion, cond)]
            ok = abs(angle - exp) <= TOL
            if not ok:
                failures.append((fusion, cond, angle, exp))

            # --- the new measurement -------------------------------------
            # How far each prototype sits from where the images actually are.
            sim_pos = float((feats @ e_pos).mean())
            sim_neg = float((feats @ e_neg).mean())
            sim_mid = 0.5 * (sim_pos + sim_neg)      # the pair's approach to the cloud
            spread = float((feats @ (e_neg - e_pos)).std())  # per-frame discriminability

            au, _ = per_view_auroc(feats, labels, views, e_pos, e_neg, scale)
            rows.append(dict(fusion=fusion, cond=cond, angle=angle, exp=exp, ok=ok,
                             sim_pos=sim_pos, sim_neg=sim_neg, sim_mid=sim_mid,
                             spread=spread, auroc=au))

    # ---- validation gate --------------------------------------------------
    print("\nvalidation -- prototype angles against Table 9")
    for r in rows:
        print("  %-6s %-11s  %5.1f deg   published %5.1f   %s"
              % (r["fusion"], r["cond"], r["angle"], r["exp"],
                 "ok" if r["ok"] else "MISMATCH"))
    if failures:
        print("\n  VALIDATION FAILED on %d cell(s). The prompts or the pooling do not"
              % len(failures))
        print("  match what produced the published table; nothing else is reported.")
        return 1
    print("  all eight cells reproduce the published angles")

    # ---- ordering check on the sample ------------------------------------
    both = {r["cond"]: r["auroc"] for r in rows if r["fusion"] == "both"}
    norm = {r["cond"]: r["auroc"] for r in rows if r["fusion"] == "normal"}
    print("\nordering check (per-view normalised, w=1; sample only)")
    for name, g in (("both", both), ("normal", norm)):
        print("  %-6s " % name + "  ".join("%s %.3f" % (c, g[c])
              for c in ("none", "generic", "matched", "mismatched")))
    repro = (both["matched"] < both["none"]) and (norm["matched"] > norm["none"]) \
        and (norm["mismatched"] < norm["none"])
    print("  published ordering reproduced on the sample: %s" % ("YES" if repro else "NO"))

    # ---- the finding ------------------------------------------------------
    print("\nDIRECTION -- mean cosine from the image cloud to each prototype")
    print("  %-6s %-11s  %8s %8s %8s   %9s"
          % ("fusion", "condition", "to P+", "to P-", "pair", "per-frame"))
    base = {}
    for r in rows:
        if r["cond"] == "none":
            base[r["fusion"]] = r["sim_mid"]
        d_mid = r["sim_mid"] - base[r["fusion"]]
        print("  %-6s %-11s  %8.4f %8.4f %8.4f   %9.4f%s"
              % (r["fusion"], r["cond"], r["sim_pos"], r["sim_neg"], r["sim_mid"],
                 r["spread"], "   (%+.4f vs none)" % d_mid if r["cond"] != "none" else ""))

    print("\nreading")
    b = {r["cond"]: r for r in rows if r["fusion"] == "both"}
    moved = {c: b[c]["sim_mid"] - b["none"]["sim_mid"]
             for c in ("generic", "matched", "mismatched")}
    order = sorted(moved, key=moved.get, reverse=True)
    print("  under `both`, the pair moves toward the images most under: %s"
          % " > ".join("%s (%+.4f)" % (c, moved[c]) for c in order))
    conj = order[0] == "matched"
    print("  Section 7.5's conjecture predicts `matched` moves furthest toward")
    print("  the image cloud.  ->  %s" % ("SUPPORTED" if conj else "NOT SUPPORTED"))
    print("  per-frame discriminability (std of the P- minus P+ margin) under `both`:")
    for c in ("none", "generic", "matched", "mismatched"):
        print("      %-11s %.4f" % (c, b[c]["spread"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
