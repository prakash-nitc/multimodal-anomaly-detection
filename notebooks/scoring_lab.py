# -*- coding: utf-8 -*-
"""Evaluate many scoring hypotheses against cached embeddings, honestly.

Once ``cache_embeddings.py`` has run, every question that does not change the
image encoder -- prompt wording, pooling rule, crop aggregation, temporal
window, how normality is defined -- can be answered in seconds instead of the
50 minutes a re-encode costs.

That speed creates its own hazard. Evaluate enough variants against a test set
and one of them scores well by chance; report only that one and the number is
fiction. So the clips are split into a DEVELOPMENT half and a HELD-OUT half
before anything is measured. All exploration and all ranking happen on dev. The
held-out column exists to be read once, at the end, for the configuration dev
selected -- it is the number that goes in the paper.

The split is by CLIP, not by frame: frames within a clip are highly correlated,
so a frame-level split would leak almost everything across the boundary.

Scoring strategies
------------------
margin_pooled
    Mean-pool each prompt ensemble into one prototype, score by the difference
    of cosine similarities. This is the original DA-ZVAD rule (its softmax is a
    monotone function of this margin, so AUROC is identical). Mean-pooling is
    also what produced the prototype-dilution failure.

margin_maxmax
    Score against each prompt individually: max similarity over the abnormal
    ensemble minus max over the normal ensemble. Never averages prompts, so a
    specific anomaly phrase that fires on a few frames is not washed out by the
    other phrases in its ensemble.

center
    Ignore the abnormal text entirely. Estimate what normal looks like IN THIS
    CLIP as the mean embedding of its own frames, and score by distance from
    it. Uses no labels and no training. Anomalies are the minority and are
    mutually dissimilar, so the mean is dominated by normal content.

    This is transductive -- it reads the unlabelled test clip before scoring --
    which must be stated plainly in the paper. It is standard in the VAD
    literature, and it is the most direct expression of the concept-shift
    claim: the decision boundary is re-estimated per scene, with no gradients.

*_plus_center
    Per-clip z-score of the text margin plus the centre distance. Combines a
    semantic prior with a scene-specific empirical one.

Usage:
    python notebooks/scoring_lab.py --cache ~/dazvad/work/embeddings/<file>.npz
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from da_zvad import evaluation                    # noqa: E402
from da_zvad.prompts import get_prompts           # noqa: E402
from da_zvad.temporal import moving_average       # noqa: E402


# --- extra prompt ensembles to compare against the packaged ones ------------
EXTRA_PROMPTS = {
    # Union of the crime vocabulary and the campus vocabulary: tests whether
    # the two are complementary or whether one simply drowns the other.
    "union": (
        list(get_prompts("surveillance")[0]) + list(get_prompts("campus")[0]),
        list(get_prompts("surveillance")[1]) + list(get_prompts("campus")[1]),
    ),
    # Deliberately minimal and scene-free: the two ensembles share no
    # vocabulary at all, which is the condition prototype dilution violates.
    "disjoint": (
        ["a photo of an ordinary calm moment"],
        ["a photo of an unusual and alarming moment"],
    ),
}


def prompt_sets():
    out = {name: get_prompts(name) for name in ("generic", "surveillance", "campus")}
    out.update(EXTRA_PROMPTS)
    return out



# --- the four context-sweep conditions, as prompt sets ----------------------
# The sweep varies the scene DESCRIPTION injected into the prompts, not the
# prompt vocabulary. Reconstructing it here lets any scoring strategy be
# evaluated under all four conditions against cached embeddings, which is how
# we test whether a change to the scoring rule alters the size of the
# matched-versus-mismatched gap rather than only the headline number.
SWEEP_DEFAULTS = {
    "matched": "a university campus walkway with pedestrians",
    "mismatched": ("an industrial quality-inspection image of a manufactured "
                   "product on a factory line"),
    "generic": "a generic scene",
}


def sweep_prompt_sets(domain: str, matched: str, mismatched: str):
    from da_zvad.context import VerbalizedContext
    n, a = get_prompts(domain)
    out = {"none": (list(n), list(a))}
    for name, desc in (("generic", SWEEP_DEFAULTS["generic"]),
                       ("matched", matched),
                       ("mismatched", mismatched)):
        out[name] = VerbalizedContext(desc, mode="normal").ground(n, a)
    return out


def print_sweep_gaps(rows, order=("none", "generic", "matched", "mismatched")):
    """Per strategy: the four conditions, and the gap the thesis predicts.

    The gap -- matched minus mismatched -- is the quantity the falsifying
    control measures. A scoring change that raises every condition equally has
    improved the detector without saying anything about whether the descriptor
    is load-bearing; one that widens the gap has.
    """
    by = {}
    for r in rows:
        by.setdefault((r["strategy"], r["window"]), {})[r["prompts"]] = r
    best = {}
    for (strat, win), conds in by.items():
        if not all(c in conds for c in order):
            continue
        score = sum(float(conds[c]["dev_mean"]) for c in order) / len(order)
        if strat not in best or score > best[strat][0]:
            best[strat] = (score, win, conds)

    print()
    print("=" * 78)
    print("CONTEXT SWEEP BY SCORING STRATEGY  (held-out AUROC)")
    print("=" * 78)
    print(f"{'strategy':<28}{'win':>4}" + "".join(f"{c:>11}" for c in order)
          + f"{'gap':>9}")
    print("-" * 78)
    ranked = sorted(best.items(),
                    key=lambda kv: -(float(kv[1][2]["matched"]["heldout_mean"])
                                     - float(kv[1][2]["mismatched"]["heldout_mean"])))
    for strat, (_, win, conds) in ranked:
        vals = [float(conds[c]["heldout_mean"]) for c in order]
        gap = vals[order.index("matched")] - vals[order.index("mismatched")]
        print(f"{strat:<28}{win:>4}" + "".join(f"{v:>11.4f}" for v in vals)
              + f"{gap:>+9.4f}")
    print()
    print("gap = matched - mismatched. Positive and large is the signature the")
    print("falsifying control looks for; near zero means the descriptor is inert")
    print("under that scoring rule, whatever the headline number says.")



# CLIP's learned inverse temperature. The driver scores the softmax probability
# of the abnormal prototype, which is sigmoid(LOGIT_SCALE * margin); the margin
# itself is a monotone transform of that, so within-clip AUROC is identical.
#
# Pooled AUROC is not. Per-clip min-max normalisation is affine, and an affine
# map applied after a nonlinear monotone one is not the same as the affine map
# alone -- so the two scorings normalise each clip onto [0, 1] differently and
# rank differently once clips are pooled. Reproducing the driver therefore
# requires reproducing its transform, not merely its ordering.
#
# This is a property of the protocol rather than of either implementation: a
# metric defined through per-clip normalisation is sensitive to the score's
# scale, not only to its ranking.
LOGIT_SCALE = 100.0


def as_probability(margin):
    """Margin -> softmax probability, matching the driver's frame score."""
    return 1.0 / (1.0 + np.exp(-LOGIT_SCALE * np.asarray(margin, dtype=np.float64)))


def zscore_per_clip(x: np.ndarray, clip_ids: np.ndarray) -> np.ndarray:
    z = np.empty_like(x, dtype=float)
    for c in np.unique(clip_ids):
        m = clip_ids == c
        v = x[m]
        sd = v.std()
        z[m] = (v - v.mean()) / sd if sd > 1e-8 else 0.0
    return z



def local_reference(feats: np.ndarray, clip_ids: np.ndarray,
                    window: int, guard: int) -> np.ndarray:
    """Mean embedding of each frame's temporal neighbourhood, per clip.

    ``guard`` excludes a band around the frame itself so that an anomaly does
    not contaminate the reference it is measured against. Anomalous events in
    these benchmarks last a second or two, so a guard wider than the event and
    a window several times wider leaves the reference dominated by normal
    content without needing labels to say which frames those are.

    Computed by cumulative sums, so cost is linear in frame count.
    """
    out = np.empty_like(feats, dtype=np.float32)
    for c in np.unique(clip_ids):
        idx = np.where(clip_ids == c)[0]
        f = feats[idx]
        n = len(f)
        cs = np.concatenate([np.zeros((1, f.shape[1]), dtype=np.float64),
                             np.cumsum(f.astype(np.float64), axis=0)])
        t = np.arange(n)
        lo, hi = np.maximum(t - window, 0), np.minimum(t + window + 1, n)
        glo, ghi = np.maximum(t - guard, 0), np.minimum(t + guard + 1, n)
        tot = cs[hi] - cs[lo]
        inner = cs[ghi] - cs[glo]
        cnt = (hi - lo) - (ghi - glo)
        # clips shorter than the guard leave nothing outside it; fall back to
        # the whole-clip mean rather than dividing by zero
        bad = cnt <= 0
        ref = np.where(bad[:, None], (cs[n] - cs[0]) / max(n, 1),
                       (tot - inner) / np.maximum(cnt, 1)[:, None])
        out[idx] = ref.astype(np.float32)
    return out

def build_scores(feats, clip_ids, texts, crop_agg: str):
    """feats (N, C, D); texts dict name -> (normal (P,D), abnormal (Q,D))."""
    # "whole" uses crop 0 only -- the full frame, which is what the experiment
    # driver scores. Keeping it available lets a lab result be compared against
    # a driver run without the crop aggregation confounding the comparison.
    if crop_agg == "whole":
        agg = lambda a: a[:, 0]
    elif crop_agg == "max":
        agg = lambda a: a.max(axis=1)
    else:
        agg = lambda a: a.mean(axis=1)
    out = {}

    for pname, (tn, ta) in texts.items():
        pn = tn.mean(0); pn /= np.linalg.norm(pn)
        pa = ta.mean(0); pa /= np.linalg.norm(pa)
        out[f"margin_pooled|{pname}"] = as_probability(
            agg(feats @ pa) - agg(feats @ pn))
        out[f"margin_maxmax|{pname}"] = as_probability(
            agg((feats @ ta.T).max(axis=-1)) - agg((feats @ tn.T).max(axis=-1)))

    # Scene-conditional normality, from the clip's own (unlabelled) frames.
    whole = feats[:, 0, :]
    center = np.empty(len(whole), dtype=float)
    for c in np.unique(clip_ids):
        m = clip_ids == c
        mu = whole[m].mean(0)
        n = np.linalg.norm(mu)
        center[m] = 1.0 - whole[m] @ (mu / n if n > 1e-8 else mu)
    out["center|-"] = center

    # Motion. ShanghaiTech's anomaly classes are overwhelmingly kinematic --
    # cyclists, runners, vehicles, chasing. A single frame of a cyclist is an
    # unremarkable photo of a person on a path, so a frame-level scorer is
    # being asked to judge appearance for what is really an event in time.
    # Embedding drift between neighbouring frames measures that directly, and
    # needs no extra model: fast-moving content changes the frame embedding
    # quickly, ordinary walking does not.
    for lag in (1, 5):
        d = np.zeros(len(whole), dtype=float)
        for c in np.unique(clip_ids):
            idx = np.where(clip_ids == c)[0]
            f = whole[idx]
            if len(f) <= lag:
                continue
            sim = np.einsum("ij,ij->i", f[lag:], f[:-lag])
            d[idx[lag:]] = 1.0 - sim
            d[idx[:lag]] = d[idx[lag]]          # pad the head with the first value
        out[f"motion{lag}|-"] = d

    # Text-directed local deviation.
    #
    # The margin strategies compare each frame independently against two fixed
    # text prototypes. CLIP similarity on a surveillance frame is dominated by
    # scene appearance -- lighting, crowd density, viewpoint -- which varies
    # continuously within a clip and swamps the component that carries "a
    # bicycle is present". Per-clip normalisation removes that baseline
    # globally, which is why it moved pooled AUROC from 0.51 to 0.70, but a
    # clip is a minute of video and the baseline drifts inside it.
    #
    # Subtracting a local temporal reference first, then projecting onto the
    # text direction, asks a different question: how far does this frame depart
    # from its recent context, in the direction the text calls anomalous? The
    # direction still comes entirely from the prompts, so nothing is learned and
    # the identifiability argument is unaffected.
    #
    # This also predicts the observed results. Motion works because it measures
    # departure from recent context but carries no direction; the text margin
    # works weakly because it carries direction but has no reference.
    for pname, (tn, ta) in texts.items():
        pn = tn.mean(0); pn /= np.linalg.norm(pn)
        pa = ta.mean(0); pa /= np.linalg.norm(pa)
        d = pa - pn
        nd = np.linalg.norm(d)
        if nd < 1e-8:
            continue
        d = d / nd
        for w, g in ((50, 15), (150, 30)):
            ref = local_reference(whole, clip_ids, w, g)
            out[f"textdev_w{w}|{pname}"] = as_probability((whole - ref) @ d)
        out[f"textdev_clip|{pname}"] = as_probability((whole - np.stack(
            [whole[clip_ids == c].mean(0) for c in clip_ids])) @ d)

    base = [k for k in list(out) if k.startswith("margin_")]
    base = base + [k for k in list(out) if k.startswith("textdev_")]
    for key in base:
        strat, pname = key.split("|")
        zk = zscore_per_clip(out[key], clip_ids)
        out[f"{strat}_plus_center|{pname}"] = zk + zscore_per_clip(center, clip_ids)
        for lag in (1, 5):
            out[f"{strat}_plus_motion{lag}|{pname}"] = (
                zk + zscore_per_clip(out[f"motion{lag}|-"], clip_ids))
    return out


def per_clip_smoothed(scores, labels, clip_ids, window: int):
    """Smooth once per clip; every split then reuses the result."""
    return {int(c): (moving_average(scores[clip_ids == c], window),
                     labels[clip_ids == c])
            for c in np.unique(clip_ids)}


def auroc_subset(pc, keep) -> float:
    return evaluation.pooled_auroc([pc[c][0] for c in keep],
                                   [pc[c][1] for c in keep], normalize=True)


def make_splits(clips, dev_frac: float, seeds):
    """Several independent clip-level dev/held-out partitions.

    One partition of 107 clips is noisy: differences of 0.002 between the top
    configurations are indistinguishable from which clips happened to land
    where. Averaging over partitions, and reporting the spread, says whether a
    ranking is real or an artefact of one draw.
    """
    out = []
    for s in seeds:
        perm = np.random.default_rng(s).permutation(clips)
        n = int(round(dev_frac * len(clips)))
        out.append((set(perm[:n].tolist()), set(perm[n:].tolist())))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--windows", type=int, nargs="+", default=[1, 5, 15, 31])
    p.add_argument("--crop-agg", nargs="+", default=["max", "mean"],
                   choices=["max", "mean", "whole"],
                   help="'whole' scores the full frame only, matching the "
                        "experiment driver")
    p.add_argument("--dev-frac", type=float, default=0.5)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--sweep", action="store_true",
                   help="evaluate the four context-sweep conditions instead of "
                        "the packaged prompt vocabularies, and report the "
                        "matched-minus-mismatched gap per scoring strategy")
    p.add_argument("--sweep-domain", default="surveillance",
                   choices=["surveillance", "campus", "generic"])
    p.add_argument("--matched-desc", default=SWEEP_DEFAULTS["matched"])
    p.add_argument("--mismatched-desc", default=SWEEP_DEFAULTS["mismatched"])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    z = np.load(os.path.expanduser(args.cache), allow_pickle=True)
    feats, labels, clip_ids = z["feats"], z["labels"].astype(int), z["clip_ids"]
    n_crops = feats.shape[1]
    print(f"cache: {feats.shape[0]} frames, {n_crops} crop(s), dim {feats.shape[2]}, "
          f"{len(np.unique(clip_ids))} clips, {labels.sum()} anomalous "
          f"({100 * labels.mean():.1f}%)")

    clips = np.unique(clip_ids)
    splits = make_splits(clips, args.dev_frac, args.seeds)
    print(f"splits: {len(splits)} x ({len(splits[0][0])} dev / "
          f"{len(splits[0][1])} held-out) clips, seeds {args.seeds}\n")

    from da_zvad.encoders import CLIPEncoder
    enc = CLIPEncoder(str(z["clip_model"]), str(z["clip_pretrained"]), "cpu")
    sets = (sweep_prompt_sets(args.sweep_domain, args.matched_desc,
                              args.mismatched_desc)
            if args.sweep else prompt_sets())
    texts = {name: (enc.encode_texts(n), enc.encode_texts(a))
             for name, (n, a) in sets.items()}

    rows = []
    for crop_agg in (args.crop_agg if n_crops > 1 else ["whole"]):
        variants = build_scores(feats, clip_ids, texts, crop_agg)
        for key, s in variants.items():
            strat, pname = key.split("|")
            for w in args.windows:
                pc = per_clip_smoothed(s, labels, clip_ids, w)
                dev_a = [auroc_subset(pc, d) for d, _ in splits]
                held_a = [auroc_subset(pc, h) for _, h in splits]
                full = auroc_subset(pc, [int(c) for c in clips])
                rows.append({
                    "strategy": strat, "prompts": pname,
                    "crops": crop_agg if n_crops > 1 else "n/a", "window": w,
                    "dev_mean": round(float(np.mean(dev_a)), 4),
                    "heldout_mean": round(float(np.mean(held_a)), 4),
                    "heldout_std": round(float(np.std(held_a)), 4),
                    "all_clips": round(float(full), 4),
                })

    rows.sort(key=lambda r: -r["dev_mean"])
    print(f"{'strategy':<30}{'prompts':<12}{'crops':<7}{'win':>4}"
          f"{'DEV':>9}{'held-out':>11}{'all':>9}")
    print("-" * 82)
    for r in rows[:args.top]:
        print(f"{r['strategy']:<30}{r['prompts']:<12}{r['crops']:<7}{r['window']:>4}"
              f"{r['dev_mean']:>9.4f}"
              f"{r['heldout_mean']:>8.4f}+-{r['heldout_std']:<4.3f}"
              f"{r['all_clips']:>9.4f}")

    best = rows[0]
    spread = max(r["dev_mean"] for r in rows[:5]) - min(r["dev_mean"] for r in rows[:5])
    print("\n" + "=" * 82)
    print("SELECTED ON DEV (mean over splits) -- report the held-out number:")
    print(f"  {best['strategy']} / prompts={best['prompts']} / "
          f"crops={best['crops']} / window={best['window']}")
    print(f"  dev {best['dev_mean']:.4f}   HELD-OUT {best['heldout_mean']:.4f} "
          f"+- {best['heldout_std']:.4f}   all-clips {best['all_clips']:.4f}")
    print("=" * 82)
    if spread < 2 * max(r["heldout_std"] for r in rows[:5]):
        print(f"\nCAUTION: the top 5 configurations span only {spread:.4f} on dev,")
        print("which is inside the split-to-split noise. Treat them as tied --")
        print("the winner is not meaningfully better than its neighbours.")
    print("\nheld-out +- is the standard deviation across splits, i.e. how much")
    print("the number moves purely from which clips landed in which half.")
    print("'all' scores every clip, for comparison with published numbers that")
    print("use the full test set.")

    if args.sweep:
        print_sweep_gaps(rows)

    if args.out:
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader(); wr.writerows(rows)
        print(f"\n-> {args.out}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
