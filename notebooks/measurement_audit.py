# -*- coding: utf-8 -*-
"""Measurement audit: what does the scene sentence actually change?

Plan step 1 after Phase 3 (report_ieee.tex, "What the descriptor changes").

THE QUESTION
------------
Every context gap in the paper is a difference in POOLED AUROC, which ranks the
frames of all videos against each other. A sentence can raise pooled AUROC in
two different ways:

  re-ranking   inside a video, anomalous frames move above normal ones;
  calibration  each video's scores move up or down as a block, so frames from
               different cameras become comparable -- the order of frames
               inside any one video does not change.

Two figures from earlier runs point to calibration: on ShanghaiTech at w=5 the
matched-mismatched gap is +0.100 pooled but +0.006 macro, and scoring the raw
cosine margin instead of the softmax cuts the pooled gap at w=31 to +0.017.
Both come from runs made for other purposes. This script measures the question
directly, on every benchmark and every descriptor condition.

WHAT IS MEASURED
----------------
Per condition (columns of the printed tables):
  pooled   the published number: softmax(lambda=100) -> moving average ->
           [per-video min-max] -> one AUROC over all frames. Min-max for
           ShanghaiTech and Avenue, raw for UCF-Crime, exactly as published.
  other    the same with the other normalisation choice.
  macro    mean of per-video AUROCs on the published scores: within-video
           ranking only. Videos with one class are skipped (on UCF-Crime that
           leaves the 140 anomalous videos). NOT fully blind to a per-video
           shift: smoothing after the softmax lets a shift change saturation
           and so reorder frames slightly (exactly blind only at w=1).
  macro_m  the same on the smoothed raw margin. Smoothing is linear, so a
           per-video shift cannot change it at all: this is the pure
           within-video ranking measure, and the one the decision rule uses.
  margin   pooled AUROC of the raw cosine margin <v,e-> - <v,e+>, no softmax,
           same smoothing and normalisation.
  centred  pooled AUROC after subtracting each video's own mean margin before
           the softmax: a label-free, sentence-free per-video calibration. If
           this alone matches the sentence, the sentence is acting as a
           calibrator that a one-line rule could replace.

Per pair of conditions A vs B (rows of the pair tables):
  gap      pooled(A) - pooled(B), the published gap.
  level    the part of the gap reproduced by shifting each video of B by its
           mean margin change d_c, keeping B's within-video pattern; averaged
           with the reverse (A shifted back by d_c), so the split does not
           depend on which condition is taken first.
  shape    the rest (gap - level): what needs a change in within-video pattern.
  share    level / gap.
  macro    macro(A) - macro(B).
  macro_m  macro_m(A) - macro_m(B): the within-video gap a shift cannot fake.
  lambda   the gap re-computed with the softmax temperature at 1, 10, 30, 100
           (CLIP's own) and 300. A gap that grows with lambda is a level shift
           being amplified by the logit scale.

DECISION RULE -- fixed before any audit number exists
-----------------------------------------------------
For each headline pair, on its published protocol and window:
  CALIBRATION-dominated   level share >= 2/3  and  |macro_m gap| <= 1/3 |gap|
  RE-RANKING-dominated    level share <= 1/3
  MIXED                   otherwise
  (|gap| < 0.01: no effect to decompose.)
Headline pairs: ShanghaiTech matched vs mismatched (w=31); ShanghaiTech
per-camera LLaVA label-free vs its shuffled control (w=31); UCF-Crime per-video
vs none and per-video vs shuffled (raw, w=5).

VALIDATION GATES -- nothing past a failed gate is trusted
---------------------------------------------------------
  G1  driver scores (ShanghaiTech, Avenue): margins recovered from the saved
      softmax scores reproduce the driver's own context_sweep.csv (w=1, 5;
      raw, normalised and macro AUROC) to its 4 decimals.
  G2  the same files at w=31 reproduce the published sweep: ShanghaiTech
      0.707 / 0.691 / 0.734 / 0.628, Avenue 0.706 / 0.729 / 0.677 / 0.657.
  G3  ShanghaiTech embedding cache reproduces the four sweep conditions and
      per_view_descriptors.py's per-camera results (0.749, 0.751, shuffled).
  G4  UCF-Crime embedding cache reproduces ucf_crime_sweep.py's results.
  G5  where both exist, driver margins and cache margins for the same frames
      agree (r >= 0.99): the logit recovery and lambda = 100 are right.
Saved result files on the server are the reference when present; otherwise the
published 3-decimal values are used, with a tolerance stated per gate.

Runs on CPU (only CLIP's text encoder is loaded, for the cache conditions);
about 10 minutes. No GPU, no labels beyond the benchmarks' own.

    python notebooks/measurement_audit.py --selftest   # synthetic checks, no data
    python notebooks/measurement_audit.py              # everything under ~/dazvad/work
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import numpy as np                                                # noqa: E402
from scipy.special import expit                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from da_zvad import evaluation                                    # noqa: E402
from da_zvad.temporal import moving_average                       # noqa: E402

LAMBDA = 100.0                         # CLIP ViT-L/14 (LAION-2B) logit scale
LAMBDAS = (1.0, 10.0, 30.0, 100.0, 300.0)
BOOT_SEED = 20261015
WORK = os.path.expanduser("~/dazvad/work")
OUT = os.path.join(WORK, "tables", "measurement_audit.json")

# The driver runs whose tables G1 reproduces, and the descriptors they used.
DRIVER = {
    "shanghaitech": {
        "table": "results/runs/2026-08-14_162056_surv_normal/tables/context_sweep.csv",
        "spec_desc": "a university campus walkway with pedestrians",
        "published_w31": {"none": 0.707, "generic": 0.691,
                          "matched": 0.734, "mismatched": 0.628},
    },
    "avenue": {
        "table": "results/runs/2026-08-18_140941_avenue_fixed/tables/context_sweep.csv",
        "spec_desc": "an outdoor campus walkway in front of a building entrance",
        "published_w31": {"none": 0.706, "generic": 0.729,
                          "matched": 0.677, "mismatched": 0.657},
    },
}
SWEEP = ("none", "generic", "matched", "mismatched")

# ShanghaiTech cache path: sweep_whole2.csv (lab, whole frame, w=31, all clips)
# and the published per-camera table.
SHT_CACHE_REF = {"none": 0.7067, "generic": 0.6907, "matched": 0.7336,
                 "mismatched": 0.6283}
SHT_PERCAM_PUB = {"per_view_hand": 0.749, "per_view_lf": 0.751,
                  "shuffled_hand_mean": 0.715, "shuffled_hand_max": 0.737,
                  "shuffled_lf_mean": 0.707, "shuffled_lf_max": 0.722}
UCF_PUB = {   # raw_w5 / norm_w5, main.tex tbl:ucf
    "none": (0.756, 0.759), "generic": (0.729, 0.740), "shared": (0.813, 0.767),
    "per_video": (0.824, 0.789), "shuffled_mean": (0.713, 0.741),
    "shuffled_max": (0.779, 0.784), "mismatched": (0.727, 0.746),
}

TOL_TABLE = 6e-5        # a 4-decimal table, plus float noise
TOL_SAVED = 6e-4        # a saved JSON from a GPU run, re-scored on CPU text features
TOL_PUB = 1.5e-3        # a published 3-decimal value reached by a second code path


# ======================================================================
# core: scores, metrics, the level/shape split
# ======================================================================
def to_margin(p, lam=LAMBDA):
    """Driver score P(abnormal) = sigmoid(lam * margin)  ->  margin."""
    p = np.asarray(p, dtype=np.float64)
    if np.any((p <= 0.0) | (p >= 1.0)):
        raise ValueError("saturated score: the margin is not recoverable")
    return (np.log(p) - np.log1p(-p)) / lam


def margin_error_bound(p, lam=LAMBDA):
    """Worst-case error of to_margin() when p was stored as float32.

    The driver's softmax runs in float32, whose spacing near p = 1 is ~6e-8, so
    1 - p -- and with it the recovered margin -- loses precision as scores
    saturate. Bound per frame: ulp(p) / (p (1 - p)) / lambda.
    """
    p = np.asarray(p, dtype=np.float64)
    ulp = np.spacing(p.astype(np.float32)).astype(np.float64)
    return ulp / (p * (1.0 - p)) / lam


def processed(margins, w, normalize, lam=LAMBDA, centre=False):
    """Per-video score arrays as the metric sees them.

    lam=None scores the margin itself. centre=True subtracts each video's mean
    margin first. Normalisation is applied per video, exactly as
    evaluation.pooled_auroc does, so concatenating these and taking one AUROC
    is the pooled AUROC.
    """
    out = []
    for m in margins:
        x = m - m.mean() if centre else m
        x = expit(lam * x) if lam else x
        x = moving_average(x, w)
        out.append(evaluation.normalize_per_sequence(x) if normalize else x)
    return out


def auc(seqs, labels):
    return evaluation.frame_auroc(np.concatenate(seqs), np.concatenate(labels))


def per_video_auc(seqs, labels):
    return np.array([evaluation.frame_auroc(s, l) if len(np.unique(l)) == 2 else np.nan
                     for s, l in zip(seqs, labels)])


def condition_metrics(margins, labels, w, normalize):
    pub = processed(margins, w, normalize)
    pv = per_video_auc(pub, labels)
    return {
        "pooled": auc(pub, labels),
        "other": auc(processed(margins, w, not normalize), labels),
        "macro": float(np.nanmean(pv)),
        "macro_m": float(np.nanmean(per_video_auc(processed(margins, w, normalize, lam=None),
                                                  labels))),
        "n_macro": int(np.sum(~np.isnan(pv))),
        "margin": auc(processed(margins, w, normalize, lam=None), labels),
        "centred": auc(processed(margins, w, normalize, centre=True), labels),
    }


def split(mA, mB, labels, w, normalize, boot=0):
    """Level/shape decomposition of pooled(A) - pooled(B). See module docstring."""
    d = [a.mean() - b.mean() for a, b in zip(mA, mB)]
    P = {"A": processed(mA, w, normalize),
         "B": processed(mB, w, normalize),
         "cfab": processed([b + di for b, di in zip(mB, d)], w, normalize),
         "cfba": processed([a - di for a, di in zip(mA, d)], w, normalize)}
    a = {k: auc(v, labels) for k, v in P.items()}
    gap = a["A"] - a["B"]
    level = 0.5 * ((a["cfab"] - a["B"]) + (a["A"] - a["cfba"]))
    pa, pb = per_video_auc(P["A"], labels), per_video_auc(P["B"], labels)
    pma = per_video_auc(processed(mA, w, normalize, lam=None), labels)
    pmb = per_video_auc(processed(mB, w, normalize, lam=None), labels)
    res = {
        "gap": gap, "level": level, "shape": gap - level,
        "level_share": level / gap if abs(gap) > 1e-9 else float("nan"),
        "macro_gap": float(np.nanmean(pa) - np.nanmean(pb)),
        "macro_m_gap": float(np.nanmean(pma) - np.nanmean(pmb)),
        "margin_gap": (auc(processed(mA, w, normalize, lam=None), labels)
                       - auc(processed(mB, w, normalize, lam=None), labels)),
        "lambda_gap": {str(int(l)): auc(processed(mA, w, normalize, lam=l), labels)
                       - auc(processed(mB, w, normalize, lam=l), labels)
                       for l in LAMBDAS},
        "offset_mean": float(np.mean(d)), "offset_sd": float(np.std(d)),
    }
    if boot:
        res["ci95"] = bootstrap(P, labels, pma - pmb, boot)
    return res


def split_avg(mA, mBs, labels, w, normalize):
    """A against each of several B (shuffled controls), averaged."""
    rs = [split(mA, mB, labels, w, normalize) for mB in mBs]
    out = {k: float(np.mean([r[k] for r in rs]))
           for k in ("gap", "level", "shape", "macro_gap", "macro_m_gap", "margin_gap",
                     "offset_mean", "offset_sd")}
    out["level_share"] = out["level"] / out["gap"] if abs(out["gap"]) > 1e-9 else float("nan")
    out["lambda_gap"] = {k: float(np.mean([r["lambda_gap"][k] for r in rs]))
                         for k in rs[0]["lambda_gap"]}
    out["n_controls"] = len(rs)
    return out


def bootstrap(P, labels, dmacro, B):
    """Video-level bootstrap: 95% intervals for gap, level and macro_m gap."""
    rng = np.random.default_rng(BOOT_SEED)
    n = len(labels)
    acc = {"gap": [], "level": [], "macro_m_gap": []}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        lab = np.concatenate([labels[i] for i in idx])
        if lab.min() == lab.max():
            continue
        a = {k: evaluation.frame_auroc(np.concatenate([P[k][i] for i in idx]), lab)
             for k in P}
        acc["gap"].append(a["A"] - a["B"])
        acc["level"].append(0.5 * ((a["cfab"] - a["B"]) + (a["A"] - a["cfba"])))
        acc["macro_m_gap"].append(np.nanmean(dmacro[idx]))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in acc.items()}


def verdict(r):
    if abs(r["gap"]) < 0.01:
        return "NO EFFECT TO DECOMPOSE"
    ratio = abs(r["macro_m_gap"]) / abs(r["gap"])
    if r["level_share"] >= 2 / 3 and ratio <= 1 / 3:
        return "CALIBRATION-dominated"
    if r["level_share"] <= 1 / 3:
        return "RE-RANKING-dominated"
    return "MIXED"


# ======================================================================
# reporting
# ======================================================================
def print_conditions(title, rows, primary):
    print("\n" + title)
    print("  %-20s%9s%9s%9s%9s%9s%9s"
          % ("condition", primary, "other", "macro", "macro_m", "margin", "centred"))
    for name, m in rows.items():
        print("  %-20s%9.4f%9.4f%9.4f%9.4f%9.4f%9.4f"
              % (name, m["pooled"], m["other"], m["macro"], m["macro_m"], m["margin"],
                 m["centred"]))


def print_pairs(title, pairs):
    print("\n" + title)
    print("  %-30s%8s%8s%8s%7s%8s%8s%8s   lambda 1 / 10 / 30 / 100 / 300"
          % ("A vs B", "gap", "level", "shape", "share", "macro", "macro_m", "margin"))
    for name, r in pairs.items():
        lg = " / ".join("%+.3f" % r["lambda_gap"][k] for k in ("1", "10", "30", "100", "300"))
        print("  %-30s%+8.3f%+8.3f%+8.3f%7.2f%+8.3f%+8.3f%+8.3f   %s"
              % (name, r["gap"], r["level"], r["shape"], r["level_share"],
                 r["macro_gap"], r["macro_m_gap"], r["margin_gap"], lg))
        if "ci95" in r:
            c = r["ci95"]
            print("  %-30s  95%% CI  gap [%+.3f, %+.3f]  level [%+.3f, %+.3f]  macro_m [%+.3f, %+.3f]"
                  % ("", *c["gap"], *c["level"], *c["macro_m_gap"]))


class Gates:
    def __init__(self):
        self.rows, self.failed = [], False

    def check(self, gate, what, got, ref, tol, source):
        ok = abs(got - ref) <= tol
        self.failed |= not ok
        self.rows.append({"gate": gate, "what": what, "got": got, "ref": ref,
                          "tol": tol, "source": source, "ok": ok})
        print("  %-3s %-34s %.4f  ref %.4f  (%s, tol %.4f)  %s"
              % (gate, what, got, ref, source, tol, "ok" if ok else "MISMATCH"))
        return ok


# ======================================================================
# sources
# ======================================================================
def driver_path(raw_dir, dataset, cond, step=2, domain="surveillance", mode="normal"):
    """The exact cache file the context sweep wrote (same key function)."""
    from da_zvad.config import DAZVADConfig
    from da_zvad.context_sweep import variants_for
    from da_zvad.grid import DatasetSpec, _cache_key
    spec = DatasetSpec(dataset, domain=domain, description=DRIVER[dataset]["spec_desc"])
    var = {v["name"]: v for v in variants_for(spec)}[cond]
    key = _cache_key(spec, cond, DAZVADConfig(frame_step=step, context_mode=mode),
                     var["description"])
    return os.path.join(raw_dir, key + ".npz")


def load_driver(raw_dir, dataset):
    """-> names, {cond: margins}, labels  (or None if the files are gone)."""
    out, names, labels = {}, None, None
    for c in SWEEP:
        path = driver_path(raw_dir, dataset, c)
        if not os.path.exists(path):
            print("  missing %s" % path)
            return None
        z = np.load(path, allow_pickle=True)
        n = int(z["n"])
        nm = [str(x) for x in z["names"]]
        if names is None:
            names, labels = nm, [np.asarray(z["l%d" % i], dtype=int) for i in range(n)]
        elif nm != names:
            raise RuntimeError("video order differs between conditions in %s" % dataset)
        out[c] = [to_margin(z["s%d" % i]) for i in range(n)]
        bound = np.concatenate([margin_error_bound(z["s%d" % i]) for i in range(n)])
        print("  %-11s margin precision: median %.1e, max %.1e (margins are ~1e-2)"
              % (c, float(np.median(bound)), float(np.max(bound))))
        back = max(float(np.max(np.abs(expit(LAMBDA * m) - z["s%d" % i])))
                   for i, m in enumerate(out[c]))
        if back > 1e-9:
            raise RuntimeError("margin round trip error %.2e in %s/%s" % (back, dataset, c))
    return names, out, labels


def read_table(path):
    with open(os.path.join(REPO, path), newline="") as f:
        return {(r["context_variant"], int(r["window"])): r for r in csv.DictReader(f)}


def load_cache(path):
    z = np.load(os.path.expanduser(path), allow_pickle=True)
    f = z["feats"]
    f = (f[:, 0, :] if f.ndim == 3 else f).astype(np.float64)
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    clip_ids = z["clip_ids"]
    clips = np.unique(clip_ids)
    feats = [f[clip_ids == c] for c in clips]
    labels = [z["labels"][clip_ids == c].astype(int) for c in clips]
    names = [str(z["names"][int(c)]) for c in clips]
    return feats, labels, names


def margins_for(feats, pool, desc_for_video):
    """One margin array per video; desc_for_video(i) -> sentence or None."""
    import per_view_descriptors as pv
    out = []
    for i, f in enumerate(feats):
        pn, pa = pv.prototypes(pool, desc_for_video(i))
        out.append(f @ pa - f @ pn)
    return out


def derangements(n, seed, k):
    """Exactly ucf_crime_sweep.py's shuffled control."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(k):
        while True:
            perm = rng.permutation(n)
            if not np.any(perm == np.arange(n)):
                break
        out.append(perm)
    return out


# ======================================================================
# sections
# ======================================================================
def audit_driver(dataset, raw_dir, gates, boot, report):
    print("\n" + "=" * 78 + "\n%s -- the driver's own scores\n" % dataset.upper() + "=" * 78)
    got = load_driver(raw_dir, dataset)
    if got is None:
        print("  driver files not found -- section skipped")
        return None
    names, M, labels = got
    print("  %d videos, %d frames" % (len(names), sum(len(l) for l in labels)))

    tab = read_table(DRIVER[dataset]["table"])
    for c in SWEEP:
        for w in (1, 5):
            r = tab[(c, w)]
            gates.check("G1", "%s %s w=%d raw" % (dataset[:3], c, w),
                        auc(processed(M[c], w, False), labels), float(r["auroc_micro"]),
                        TOL_TABLE, "driver table")
            gates.check("G1", "%s %s w=%d normalised" % (dataset[:3], c, w),
                        auc(processed(M[c], w, True), labels), float(r["auroc_micro_norm"]),
                        TOL_TABLE, "driver table")
            gates.check("G1", "%s %s w=%d macro" % (dataset[:3], c, w),
                        float(np.nanmean(per_video_auc(processed(M[c], w, True), labels))),
                        float(r["auroc_macro"]), TOL_TABLE, "driver table")
        gates.check("G2", "%s %s w=31" % (dataset[:3], c),
                    auc(processed(M[c], 31, True), labels),
                    DRIVER[dataset]["published_w31"][c], TOL_PUB, "published")
    if gates.failed:
        return None

    sec = {"conditions": {}, "pairs": {}}
    for w in (31, 5, 1):
        rows = {c: condition_metrics(M[c], labels, w, True) for c in SWEEP}
        print_conditions("conditions, w=%d (pooled = per-video normalised, as published)" % w,
                         rows, "pooled")
        pairs = {"matched vs mismatched": split(M["matched"], M["mismatched"], labels, w, True,
                                                boot if w == 31 else 0),
                 "matched vs none": split(M["matched"], M["none"], labels, w, True),
                 "mismatched vs none": split(M["mismatched"], M["none"], labels, w, True)}
        print_pairs("pairs, w=%d" % w, pairs)
        sec["conditions"]["w%d" % w], sec["pairs"]["w%d" % w] = rows, pairs
    if dataset == "shanghaitech":
        report["headline"]["ShanghaiTech matched vs mismatched (w=31)"] = \
            sec["pairs"]["w31"]["matched vs mismatched"]
    report["_driver_margins"][dataset] = (names, M["none"])
    return sec


def audit_sht_cache(cache, desc_files, gates, boot, report, pool):
    import per_view_descriptors as pv
    from da_zvad.context_sweep import MISMATCHED
    print("\n" + "=" * 78 + "\nSHANGHAITECH -- embedding cache (sweep + per-camera)\n" + "=" * 78)
    if not os.path.exists(os.path.expanduser(cache)):
        print("  no cache at %s -- section skipped" % cache)
        return None
    feats, labels, names = load_cache(cache)
    views = [pv.view_of(n) for n in names]
    vs = sorted(set(views))
    print("  %d videos, %d views" % (len(names), len(vs)))

    sentence = {"none": None, "generic": "a generic scene", "matched": pv.SHARED,
                "mismatched": MISMATCHED["surveillance"]}
    M = {c: margins_for(feats, pool, lambda i, s=s: s) for c, s in sentence.items()}
    for c in SWEEP:
        gates.check("G3", "ShT cache %s w=31" % c, auc(processed(M[c], 31, True), labels),
                    SHT_CACHE_REF[c], TOL_SAVED, "sweep_whole2.csv")

    shuffled = {}
    for tag, path in desc_files.items():
        if not os.path.exists(path):
            print("  no descriptor file %s -- '%s' skipped" % (path, tag))
            continue
        desc = json.load(open(path, encoding="utf-8"))
        M["per_view_" + tag] = margins_for(feats, pool, lambda i, d=desc: d[views[i]])
        shuffled[tag] = []
        for k in range(1, len(vs)):                      # per_view_descriptors.py's rotations
            rot = {v: desc[vs[(i + k) % len(vs)]] for i, v in enumerate(vs)}
            shuffled[tag].append(margins_for(feats, pool, lambda i, r=rot: r[views[i]]))
        sh = [auc(processed(m, 31, True), labels) for m in shuffled[tag]]
        saved = os.path.join(WORK, "tables", "per_view_%s.json"
                             % os.path.splitext(os.path.basename(path))[0])
        if os.path.exists(saved):
            ref = json.load(open(saved))["results"]
            refs = {"per_view": ref["per_view"], "shuffled_mean": ref["shuffled_mean"],
                    "shuffled_max": ref["shuffled_max"]}
            src, tol = "saved run", TOL_SAVED
        else:
            refs = {"per_view": SHT_PERCAM_PUB["per_view_" + tag],
                    "shuffled_mean": SHT_PERCAM_PUB["shuffled_%s_mean" % tag],
                    "shuffled_max": SHT_PERCAM_PUB["shuffled_%s_max" % tag]}
            src, tol = "published", TOL_PUB
        gates.check("G3", "ShT per-camera %s" % tag,
                    auc(processed(M["per_view_" + tag], 31, True), labels), refs["per_view"], tol, src)
        gates.check("G3", "ShT shuffled %s mean" % tag, float(np.mean(sh)), refs["shuffled_mean"], tol, src)
        gates.check("G3", "ShT shuffled %s max" % tag, float(np.max(sh)), refs["shuffled_max"], tol, src)

    drv = report["_driver_margins"].get("shanghaitech")
    if drv is not None:
        dn, dm = drv
        if dn == names:
            a, b = np.concatenate(dm), np.concatenate(M["none"])
            r = float(np.corrcoef(a, b)[0, 1])
            print("  G5  driver vs cache margins (none): r = %.5f, median |diff| = %.2e"
                  % (r, float(np.median(np.abs(a - b)))))
            # The exact checks are G1-G4; this one only has to show that the two
            # files describe the same frames scored the same way (lambda = 100).
            gates.check("G5", "driver~cache margin correlation", r, 1.0, 1e-2, "identity")
        else:
            print("  G5  skipped: driver and cache list the videos differently")
    if gates.failed:
        return None

    sec = {"conditions": {}, "pairs": {}}
    for w in (31, 5):
        rows = {c: condition_metrics(m, labels, w, True) for c, m in M.items()}
        for tag, sh in shuffled.items():
            ms = [condition_metrics(m, labels, w, True) for m in sh]
            rows["shuffled_" + tag] = {k: float(np.mean([x[k] for x in ms])) for k in ms[0]}
        print_conditions("conditions, w=%d" % w, rows, "pooled")
        pairs = {"matched vs mismatched": split(M["matched"], M["mismatched"], labels, w, True),
                 "matched (shared) vs none": split(M["matched"], M["none"], labels, w, True)}
        for tag, sh in shuffled.items():
            A = M["per_view_" + tag]
            pairs["per-camera %s vs shared" % tag] = split(A, M["matched"], labels, w, True,
                                                           boot if w == 31 else 0)
            pairs["per-camera %s vs shuffled" % tag] = split_avg(A, sh, labels, w, True)
            pairs["per-camera %s vs none" % tag] = split(A, M["none"], labels, w, True)
        print_pairs("pairs, w=%d (shuffled rows average the %d rotations)" % (w, len(vs) - 1), pairs)
        sec["conditions"]["w%d" % w], sec["pairs"]["w%d" % w] = rows, pairs
    if "per-camera lf vs shuffled" in sec["pairs"]["w31"]:
        report["headline"]["ShanghaiTech per-camera (LLaVA, label-free) vs shuffled (w=31)"] = \
            sec["pairs"]["w31"]["per-camera lf vs shuffled"]
    if "ShanghaiTech matched vs mismatched (w=31)" not in report["headline"]:
        report["headline"]["ShanghaiTech matched vs mismatched (w=31)"] = \
            sec["pairs"]["w31"]["matched vs mismatched"]
    return sec


def audit_ucf(cache, caps_path, gates, boot, report, pool):
    import ucf_crime_sweep as u
    print("\n" + "=" * 78 + "\nUCF-CRIME -- embedding cache (pre-registered conditions)\n" + "=" * 78)
    if not os.path.exists(os.path.expanduser(cache)) or not os.path.exists(caps_path):
        print("  cache or captions missing -- section skipped")
        return None
    feats, labels, names = load_cache(cache)
    caps = json.load(open(caps_path))
    per = [caps[n] for n in names]
    print("  %d videos (%d anomalous), %d frames"
          % (len(names), sum(int(l.max()) for l in labels), sum(len(l) for l in labels)))

    M = {"none": margins_for(feats, pool, lambda i: None),
         "generic": margins_for(feats, pool, lambda i: u.GENERIC),
         "shared": margins_for(feats, pool, lambda i: u.SHARED),
         "mismatched": margins_for(feats, pool, lambda i: u.MISMATCHED),
         "per_video": margins_for(feats, pool, lambda i: per[i])}
    shuffled = [margins_for(feats, pool, lambda i, p=p: per[p[i]])
                for p in derangements(len(names), u.SEED, u.N_SHUFFLE)]

    saved = json.load(open(u.OUT))["results"] if os.path.exists(u.OUT) else None
    sh_raw = [auc(processed(m, 5, False), labels) for m in shuffled]
    sh_norm = [auc(processed(m, 5, True), labels) for m in shuffled]
    got = {c: (auc(processed(M[c], 5, False), labels), auc(processed(M[c], 5, True), labels))
           for c in M}
    got["shuffled_mean"] = (float(np.mean(sh_raw)), float(np.mean(sh_norm)))
    got["shuffled_max"] = (float(np.max(sh_raw)), float(np.max(sh_norm)))
    for c, (gr, gn) in got.items():
        if saved:
            ref, src, tol = (saved[c]["raw_w5"], saved[c]["norm_w5"]), "saved run", TOL_SAVED
        else:
            ref, src, tol = UCF_PUB[c], "published", TOL_PUB
        gates.check("G4", "UCF %s raw w=5" % c, gr, ref[0], tol, src)
        gates.check("G4", "UCF %s norm w=5" % c, gn, ref[1], tol, src)
    if gates.failed:
        return None

    sec = {"conditions": {}, "pairs": {}}
    for w in (5, 1):
        rows = {c: condition_metrics(m, labels, w, False) for c, m in M.items()}
        ms = [condition_metrics(m, labels, w, False) for m in shuffled]
        rows["shuffled"] = {k: float(np.mean([x[k] for x in ms])) for k in ms[0]}
        print_conditions("conditions, w=%d (pooled = raw, the pre-registered primary; "
                         "macro over anomalous videos)" % w, rows, "raw")
        b = boot if w == 5 else 0
        pairs = {"per-video vs none": split(M["per_video"], M["none"], labels, w, False, b),
                 "per-video vs shuffled": split_avg(M["per_video"], shuffled, labels, w, False),
                 "per-video vs shared": split(M["per_video"], M["shared"], labels, w, False),
                 "shared vs mismatched": split(M["shared"], M["mismatched"], labels, w, False),
                 "per-video vs mismatched": split(M["per_video"], M["mismatched"], labels, w, False)}
        print_pairs("pairs, w=%d (raw pooled)" % w, pairs)
        sec["conditions"]["w%d" % w], sec["pairs"]["w%d" % w] = rows, pairs
    report["headline"]["UCF-Crime per-video vs none (raw, w=5)"] = sec["pairs"]["w5"]["per-video vs none"]
    report["headline"]["UCF-Crime per-video vs shuffled (raw, w=5)"] = sec["pairs"]["w5"]["per-video vs shuffled"]
    if boot:
        sec["paired_bootstrap"] = ucf_paired_bootstrap(M, shuffled, labels, boot)
    return sec


UCF_BOOT_SEED = 20261023
PUBLISHED_UCF = {"LAVAD": 0.803, "AnyAnomaly": 0.807}     # fixed published figures


def paired_bootstrap(P, labels, B, seed, thresholds=None):
    """Video-level PAIRED bootstrap of pooled AUROC.

    P maps a configuration to its per-video score arrays (already smoothed, as
    the metric sees them); a value may be a LIST of such configurations, whose
    AUROCs are averaged (the shuffled control). Each iteration resamples VIDEOS
    with replacement, pools their frames and scores every configuration on the
    same resample, so gaps are measured on identical data.
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    draws = {k: [] for k in P}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        lab = np.concatenate([labels[i] for i in idx])
        if lab.min() == lab.max():
            continue
        for k, v in P.items():
            group = v if isinstance(v[0], list) else [v]
            draws[k].append(float(np.mean([
                evaluation.frame_auroc(np.concatenate([g[i] for i in idx]), lab) for g in group])))
    return {k: np.asarray(d) for k, d in draws.items()}


def ucf_paired_bootstrap(M, shuffled, labels, B):
    """95% percentile CIs for UCF-Crime's 0.824 and its pre-registered gaps."""
    P = {"per_video": processed(M["per_video"], 5, False),
         "none": processed(M["none"], 5, False),
         "shared": processed(M["shared"], 5, False),
         "shuffled_mean20": [processed(m, 5, False) for m in shuffled]}
    point = {k: (float(np.mean([auc(g, labels) for g in v])) if isinstance(v[0], list)
                 else auc(v, labels)) for k, v in P.items()}
    print("\nUCF-Crime paired video-level bootstrap: %d resamples of %d videos, seed %d"
          % (B, len(labels), UCF_BOOT_SEED))
    d = paired_bootstrap(P, labels, B, UCF_BOOT_SEED)
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]  # noqa: E731
    out = {"n_resamples": int(len(d["per_video"])), "n_videos": len(labels),
           "seed": UCF_BOOT_SEED, "metric": "raw pooled frame AUROC, w=5, step 16",
           "per_video": {"point": point["per_video"], "ci95": ci(d["per_video"])}}
    for ref in ("none", "shared", "shuffled_mean20"):
        g = d["per_video"] - d[ref]
        out["per_video_minus_" + ref] = {"point": point["per_video"] - point[ref],
                                         "ci95": ci(g), "frac_positive": float(np.mean(g > 0))}
    out["frac_above_published"] = {name: float(np.mean(d["per_video"] > v))
                                   for name, v in PUBLISHED_UCF.items()}
    out["note"] = ("Published figures are fixed points; their own uncertainty is not "
                   "available, so only our side is resampled.")
    print("  per-video AUROC            %.4f   95%% CI [%.4f, %.4f]"
          % (out["per_video"]["point"], *out["per_video"]["ci95"]))
    for ref in ("none", "shared", "shuffled_mean20"):
        r = out["per_video_minus_" + ref]
        print("  per-video - %-15s %+.4f   95%% CI [%+.4f, %+.4f]   >0 in %.1f%%"
              % (ref, r["point"], *r["ci95"], 100 * r["frac_positive"]))
    for name, v in PUBLISHED_UCF.items():
        print("  P(AUROC > %.3f, %s)       %.1f%% of resamples"
              % (v, name, 100 * out["frac_above_published"][name]))
    print("  (published figures are fixed; their uncertainty cannot be resampled)")
    return out


# ======================================================================
# self-test: synthetic data with known answers
# ======================================================================
def selftest() -> int:
    rng = np.random.default_rng(0)
    fails = []

    def expect(cond, msg):
        print("  %-62s %s" % (msg, "ok" if cond else "FAIL"))
        if not cond:
            fails.append(msg)

    labels, B = [], []
    for i in range(14):
        n = int(rng.integers(120, 320))
        lab = np.zeros(n, dtype=int)
        if i % 3:
            s = int(rng.integers(10, n - 40))
            lab[s:s + int(rng.integers(15, 35))] = 1
        level = rng.normal(0, 0.03)                  # each camera its own baseline
        m = level + 0.01 * rng.standard_normal(n) + 0.004 * lab   # mid-range AUROC
        labels.append(lab)
        B.append(m)
    shift = rng.normal(0, 0.03, len(B))
    A_level = [b + s for b, s in zip(B, shift)]
    A_shape = []
    for b, lab in zip(B, labels):
        e = 0.015 * lab - 0.005 * rng.standard_normal(len(b))
        A_shape.append(b + e - e.mean())

    print("\nlevel-only change (A = B + per-video constant)")
    r = split(A_level, B, labels, 9, True)
    expect(abs(r["shape"]) < 1e-12, "shape part is exactly zero")
    expect(abs(r["level"] - r["gap"]) < 1e-12, "level part equals the whole gap")
    mA = np.nanmean(per_video_auc(processed(A_level, 9, True, lam=None), labels))
    mB = np.nanmean(per_video_auc(processed(B, 9, True, lam=None), labels))
    expect(abs(mA - mB) < 1e-12, "macro AUROC of the margin is unchanged")
    r = split(A_level, B, labels, 9, False)
    expect(abs(r["shape"]) < 1e-12, "raw pooling: shape part is exactly zero too")
    expect(abs(r["gap"]) > 0.005,
           "yet raw pooled AUROC moves (%+.3f): the trap" % r["gap"])

    big = [b + 3 * s for b, s in zip(B, shift)]
    r = split(big, B, labels, 9, False)
    expect(verdict(r) == "CALIBRATION-dominated",
           "pure shift, smoothed (w=9) -> %s" % verdict(r))
    r = split(big, B, labels, 31, True)
    expect(abs(r["macro_m_gap"]) < 1e-12, "pure shift leaves macro_m exactly unchanged at w=31")

    print("\nshape-only change (zero-mean per video)")
    r = split(A_shape, B, labels, 9, True)
    expect(abs(r["level"]) < 1e-12, "level part is exactly zero")
    expect(r["macro_m_gap"] > 0.01, "macro_m AUROC rises")
    r = split(A_shape, B, labels, 9, False)
    expect(verdict(r) == "RE-RANKING-dominated", "pure re-ranking -> %s" % verdict(r))

    print("\nidentities")
    r = split(A_shape, A_level, labels, 5, False, boot=50)
    expect(abs(r["level"] + r["shape"] - r["gap"]) < 1e-12, "level + shape = gap")
    expect(r["ci95"]["gap"][0] <= r["gap"] <= r["ci95"]["gap"][1], "bootstrap interval brackets the gap")
    seqs = [moving_average(expit(LAMBDA * m), 5) for m in B]
    expect(abs(auc(processed(B, 5, True), labels)
               - evaluation.pooled_auroc(seqs, labels, normalize=True)) < 1e-12,
           "pooled AUROC equals evaluation.pooled_auroc (normalised)")
    expect(abs(auc(processed(B, 5, False), labels)
               - evaluation.pooled_auroc(seqs, labels, normalize=False)) < 1e-12,
           "pooled AUROC equals evaluation.pooled_auroc (raw)")
    driver_macro = float(np.mean([evaluation.frame_auroc(s, l) for s, l in zip(seqs, labels)
                                  if len(np.unique(l)) == 2]))
    expect(abs(np.nanmean(per_video_auc(processed(B, 5, True), labels)) - driver_macro) < 1e-12,
           "macro AUROC equals the driver's definition")

    print("\nmargin recovery from a float32 softmax, as the driver computes it")
    cn = rng.uniform(0.15, 0.30, 5000).astype(np.float32)
    ca = (cn + rng.normal(0, 0.04, 5000)).astype(np.float32)
    lg = np.stack([cn, ca], 1) * np.float32(LAMBDA)
    e = np.exp(lg - lg.max(1, keepdims=True))
    p = (e[:, 1] / e.sum(1)).astype(np.float32)
    rec = to_margin(p.astype(np.float64))
    err = np.abs(rec - (ca.astype(np.float64) - cn.astype(np.float64)))
    bound = margin_error_bound(p.astype(np.float64))
    expect(bool(np.all(err <= 2 * bound + 1e-7)),
           "error stays within the stated bound (max err %.1e)" % float(err.max()))
    real = np.abs(100 * (ca - cn)) < 6          # |lambda * margin| < 6, i.e. 0.0025 < p < 0.9975
    expect(float(err[real].max()) < 1e-5,
           "below 1e-5 for unsaturated scores (max %.1e)" % float(err[real].max()))

    print("\ndriver files: exact names, loading, G1-style check")
    import tempfile
    tmp = tempfile.mkdtemp()
    for c in SWEEP:
        payload = {"n": len(B), "names": np.array(["shanghaitech/%02d_%04d" % (i % 3 + 1, i)
                                                   for i in range(len(B))], dtype=object)}
        for i, (m, l) in enumerate(zip(B, labels)):
            payload["s%d" % i] = expit(LAMBDA * (m + 0.001 * SWEEP.index(c)))
            payload["l%d" % i] = l
        np.savez(driver_path(tmp, "shanghaitech", c), **payload)
    got = load_driver(tmp, "shanghaitech")
    expect(got is not None and len(got[1]) == 4, "all four condition files found by key")
    direct = evaluation.pooled_auroc(
        [moving_average(np.load(driver_path(tmp, "shanghaitech", "matched"),
                                allow_pickle=True)["s%d" % i], 5) for i in range(len(B))],
        labels, normalize=True)
    expect(abs(auc(processed(got[1]["matched"], 5, True), got[2]) - direct) < 1e-9,
           "recovered margins reproduce the driver's normalised AUROC")
    expect(driver_path(tmp, "avenue", "none") != driver_path(tmp, "avenue", "generic"),
           "different descriptions give different files")

    print("\ncache sections end to end (fake text encoder, gates not applied)")
    import per_view_descriptors as pv
    dim = 32

    def fake_pool(prompts):
        h = abs(hash(tuple(prompts))) % (2 ** 32)
        v = np.random.default_rng(h).standard_normal(dim)
        return v / np.linalg.norm(v)

    feats = [rng.standard_normal((len(l), dim)) for l in labels]
    feats = [f / np.linalg.norm(f, axis=1, keepdims=True) for f in feats]
    names = ["%02d_%04d" % (i % 4 + 1, i) for i in range(len(labels))]
    desc = {pv.view_of(n): "scene %s" % pv.view_of(n) for n in names}
    m1 = margins_for(feats, fake_pool, lambda i: desc[pv.view_of(names[i])])
    m0 = margins_for(feats, fake_pool, lambda i: None)
    r = split(m1, m0, labels, 31, True, boot=20)
    expect(np.isfinite(r["gap"]) and np.isfinite(r["level_share"]), "per-camera split runs")
    r = split_avg(m1, [m0, m1], labels, 5, False)
    expect(np.isfinite(r["gap"]), "averaged split runs")
    perms = derangements(10, 1, 3)
    expect(all(not np.any(p == np.arange(10)) for p in perms), "derangements fix no video")
    print("\npaired video-level bootstrap")
    Pa, Pb = processed(B, 5, False), processed(A_shape, 5, False)
    d = paired_bootstrap({"a": Pa, "a2": [p.copy() for p in Pa], "b": Pb,
                          "ab": [Pa, Pb]}, labels, 200, 7)
    expect(np.all(d["a"] == d["a2"]), "identical configurations differ by exactly 0 in every draw")
    expect(np.allclose(d["ab"], (d["a"] + d["b"]) / 2), "a list of configurations is averaged")
    lo, hi = np.percentile(d["b"] - d["a"], [2.5, 97.5])
    full = auc(Pb, labels) - auc(Pa, labels)
    expect(lo <= full <= hi, "the full-data gap lies inside its 95%% interval (%+.3f)" % full)
    d2 = paired_bootstrap({"a": Pa}, labels, 50, 7)
    expect(np.array_equal(d2["a"], d["a"][:50]), "a fixed seed gives the same resamples")

    print("\n%s" % ("SELF-TEST PASSED" if not fails else "SELF-TEST FAILED: %s" % fails))
    return 1 if fails else 0


# ======================================================================
def run_manifest(args):
    """What produced this file: code commit, machine, library versions, settings."""
    import datetime
    import platform
    import subprocess
    import sklearn

    def git(*a):
        try:
            return subprocess.run(["git", "-C", REPO, *a], capture_output=True,
                                  text=True, timeout=30).stdout.strip()
        except Exception:                        # noqa: BLE001
            return "unknown"
    return {"started_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
            "git_commit": git("rev-parse", "HEAD"),
            "working_tree": "clean" if not git("status", "--porcelain") else "DIRTY",
            "host": platform.node(), "python": platform.python_version(),
            "numpy": np.__version__, "sklearn": sklearn.__version__,
            "args": vars(args), "boot_seed": BOOT_SEED, "ucf_boot_seed": UCF_BOOT_SEED}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--raw", default=os.path.join(WORK, "raw"))
    ap.add_argument("--sht-cache", default="~/dazvad/work/embeddings/shanghaitech_ViT-L-14_step2_crops5.npz")
    ap.add_argument("--ucf-cache", default="~/dazvad/work/embeddings/ucf_crime_ViT-L-14_step16_crops1.npz")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    import per_view_descriptors as pv
    import ucf_crime_sweep as u
    gates = Gates()
    report = {"headline": {}, "_driver_margins": {}}
    sections = {}
    print("validation gates are printed as each section loads; a MISMATCH stops the audit")
    for ds in ("shanghaitech", "avenue"):
        sections["driver_" + ds] = audit_driver(ds, os.path.expanduser(args.raw), gates,
                                                args.boot, report)
        if gates.failed:
            break
    have_cache = any(os.path.exists(os.path.expanduser(c)) for c in (args.sht_cache, args.ucf_cache))
    if not gates.failed and not have_cache:
        print("\nno embedding caches found -- cache sections skipped")
    elif not gates.failed:
        print("\nloading CLIP's text encoder (CPU is fine) ...")
        pool = pv.encoder()
        sections["sht_cache"] = audit_sht_cache(
            args.sht_cache,
            {"hand": pv.DESC, "lf": os.path.join(HERE, "view_descriptors_llava_labelfree.json")},
            gates, args.boot, report, pool)
        if not gates.failed:
            sections["ucf"] = audit_ucf(args.ucf_cache, u.CAPS, gates, args.boot, report, pool)

    if gates.failed:
        print("\nVALIDATION FAILED -- the audit does not reproduce the published numbers,")
        print("so none of its decompositions are reported. Fix the mismatch first.")
    else:
        print("\n" + "=" * 78 + "\nHEADLINE VERDICTS (decision rule fixed in the docstring)\n" + "=" * 78)
        for name, r in report["headline"].items():
            ratio = abs(r["macro_m_gap"]) / abs(r["gap"]) if abs(r["gap"]) > 1e-9 else float("nan")
            print("  %-58s gap %+.3f  level share %.2f  |macro_m|/|gap| %.2f  ->  %s"
                  % (name, r["gap"], r["level_share"], ratio, verdict(r)))
            r["verdict"] = verdict(r)

    report.pop("_driver_margins")
    out = {"manifest": run_manifest(args), "gates": gates.rows,
           "gates_passed": not gates.failed,
           "headline": report["headline"], "sections": sections}
    os.makedirs(os.path.dirname(os.path.expanduser(args.out)), exist_ok=True)
    with open(os.path.expanduser(args.out), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("\nsaved %s  -- copy it to results/runs/analysis/ on the laptop" % args.out)
    return 0 if not gates.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
