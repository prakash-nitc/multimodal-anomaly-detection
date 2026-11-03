# -*- coding: utf-8 -*-
"""Normal-side prompt banks and retrieved normality on five benchmarks, exactly
as pre-registered in docs/10_phase3/PREREG_normal_side.md. Do not change the
protocol here without a dated addendum there.

    python notebooks/normal_side_sweep.py selftest           # laptop, no data, ~1 min
    python notebooks/normal_side_sweep.py run                # server, all cached datasets
    python notebooks/normal_side_sweep.py run --datasets ucf_crime xd_violence
    python notebooks/normal_side_sweep.py review             # contact sheets, after run
    python notebooks/normal_side_sweep.py xd_contamination   # exploratory, after run
    python notebooks/normal_side_sweep.py run --backbone pe-l  # same recipe, new encoder
    python notebooks/normal_side_sweep.py backbones --backbone pe-l  # paired vs ViT-L

Five components, each one switch, all training-free and all on the cached CLIP
embeddings (no image is re-encoded):

    W  worries     the benchmark's published anomaly classes on the abnormal side
    N  near-miss   look-alike normal actions + four calm anchors on the normal side
    P  per-phrase  score against every phrase (log-mean-exp) instead of averaging
                   each side into one prototype
    E  emb. window average 5 neighbouring frame embeddings before scoring
    R  retrieval   each camera's scene sentence CHOSEN from a filtered library by
                   its calm frames (da_zvad/context/retrieval.py) instead of
                   written by LLaVA; the top phrase enters through the same two
                   M3 templates as LLaVA's sentence (FULL_R5raw, the five raw
                   phrases, is kept as a secondary condition)

FULL switches all five on. Each component is judged by FULL minus FULL-without-
it, under the keep/drop rule fixed in the pre-registration. Legacy conditions
(today's system) go through the old code path and must reproduce the published
numbers before anything new is printed.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import itertools
import json
import os
import platform
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from da_zvad.context import retrieval as rt                       # noqa: E402
from da_zvad.prompts import banks                                 # noqa: E402
from da_zvad.prompts.templates import get_prompts                 # noqa: E402

LAM = 100.0                    # CLIP logit scale, as every earlier sweep
BOOT_SEED = 20261104
CTRL_SEED = 20261104
N_CTRL, BOOT_CTRL = 20, 5      # shuffled controls: point estimate / bootstrap
K_CALM, K_RET, W_EMB = 5, 5, 5
TOL_SAVED, TOL_PUB, TOL_PATH = 6e-4, 1.5e-3, 1e-5
COMPONENTS = ("W", "N", "P", "E", "R")
WORK = os.path.expanduser("~/dazvad/work")
TABLES = os.path.join(WORK, "tables")
EMB = "~/dazvad/work/embeddings/%s_ViT-L-14_step%d_crops%d.npz"
# Frozen encoders. vitl is the one every published number uses and the only one
# with validation gates; lambda stays 100 for all, so the audit stays comparable.
BACKBONES = {"vitl": ("ViT-L-14", "laion2b_s32b_b82k"),
             "pe-l": ("PE-Core-L-14-336", "meta"),
             "vith": ("ViT-H-14", "laion2b_s32b_b79k")}

SPECS = {
    "shanghaitech": dict(
        cache=EMB % ("shanghaitech", 2, 5), root="~/dazvad/data/shanghaitech", step=2,
        w=31, normalize=True, unit="view", n_warm=256, metric="auroc", campus=True,
        shared="a university campus walkway with pedestrians",
        best=("views", os.path.join(HERE, "view_descriptors_llava_labelfree.json")),
        published={"none": (0.7067, TOL_SAVED), "shared": (0.7336, TOL_SAVED),
                   "best": (0.751, TOL_PUB)}),
    "avenue": dict(
        cache=EMB % ("avenue", 2, 1), root="~/dazvad/data/avenue", step=2,
        w=31, normalize=True, unit="camera", n_warm=256, metric="auroc", campus=True,
        shared="an outdoor campus walkway in front of a building entrance", best=None,
        published={"none": (0.706, TOL_PUB), "shared": (0.677, TOL_PUB)}),
    "ucf_crime": dict(
        cache=EMB % ("ucf_crime", 16, 1), root="~/dazvad/data/ucf_crime", step=16,
        w=5, normalize=False, unit="video", n_warm=32, metric="auroc", campus=False,
        shared="real-world CCTV surveillance footage of streets, shops and buildings",
        best=("videos", [os.path.join(TABLES, "ucf_crime_captions.json"),
                         os.path.join(REPO, "results/runs/analysis/ucf_crime_captions.json")]),
        saved=(os.path.join(TABLES, "ucf_crime_sweep.json"),
               {"none": ("none", "raw_w5"), "shared": ("shared", "raw_w5"),
                "best": ("per_video", "raw_w5")}),
        published={"none": (0.756, TOL_PUB), "shared": (0.813, TOL_PUB),
                   "best": (0.824, TOL_PUB)}),
    "xd_violence": dict(
        cache=EMB % ("xd_violence", 16, 1), root="~/dazvad/data/xd_violence", step=16,
        w=5, normalize=False, unit="video", n_warm=32, metric="xd_ap", campus=False,
        shared="scenes from films, television, news, sports and online videos",
        best=("videos", [os.path.join(TABLES, "xd_violence_captions.json")]),
        saved=(os.path.join(TABLES, "xd_violence_sweep.json"),
               {"none": ("none", "ap_w5"), "shared": ("shared", "ap_w5"),
                "best": ("per_video", "ap_w5")}),
        published={"none": (0.624, TOL_PUB), "best": (0.417, TOL_PUB)}),
    "nwpu_campus": dict(
        cache=EMB % ("nwpu_campus", 16, 1), root="~/dazvad/data/nwpu_campus", step=16,
        w=5, normalize=False, unit="scene", n_warm=32, metric="auroc", campus=True,
        shared="a university campus with roads, footpaths, squares and buildings",
        best=("scenes", [os.path.join(TABLES, "nwpu_campus_captions.json")]),
        saved=(os.path.join(TABLES, "nwpu_campus_sweep.json"),
               {"none": ("none", "raw_w5"), "shared": ("shared", "raw_w5"),
                "best": ("per_scene", "raw_w5")}),
        published={}),
}
ORDER = ("shanghaitech", "avenue", "ucf_crime", "xd_violence", "nwpu_campus")


def spec_for(ds, backbone="vitl"):
    """A dataset's spec with its cache path for one backbone (whole frame only)."""
    spec = dict(SPECS[ds])
    if backbone != "vitl":
        spec["cache"] = "~/dazvad/work/embeddings/%s_%s_step%d_crops1.npz" % (
            ds, BACKBONES[backbone][0], spec["step"])
    return spec


def subset_key(on):
    """A set of components as a stable name: {'R', 'W'} -> 'WR'; empty -> 'none'."""
    return "".join(c for c in COMPONENTS if c in on) or "none"


# ======================================================================
# scoring
# ======================================================================
def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def margins_proto(F, N, A):
    """Legacy rule: each side mean-pooled into one prototype."""
    return F @ unit(A.mean(0)) - F @ unit(N.mean(0))


def _lme(S, lam):
    """log-mean-exp over phrases, in cosine units: (logsumexp(lam S) - log n) / lam."""
    z = lam * S
    m = z.max(axis=1, keepdims=True)
    return (m[:, 0] + np.log(np.exp(z - m).mean(axis=1))) / lam


def margins_lme(F, N, A, lam=LAM):
    """Per-phrase rule: sigmoid(lam * margin) is the softmax mass on the abnormal
    phrases when both sides are reweighted to equal size (the log n terms), so
    adding phrases to one side does not shift its score by count alone."""
    return _lme(F @ A.T, lam) - _lme(F @ N.T, lam)


def window_feats(F, w):
    """Centred moving average of unit embeddings (edge padding), renormalised."""
    if w <= 1:
        return F
    if w % 2 == 0:
        raise ValueError("embedding window must be odd")
    pad = w // 2
    P = np.pad(F, ((pad, pad), (0, 0)), mode="edge")
    c = np.cumsum(np.vstack([np.zeros((1, F.shape[1])), P]), axis=0)
    return unit((c[w:] - c[:-w]) / w)


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def processed(margins, w, normalize):
    from da_zvad import evaluation
    from da_zvad.temporal import moving_average
    out = []
    for m in margins:
        x = moving_average(sigmoid(LAM * m), w)
        out.append(evaluation.normalize_per_sequence(x) if normalize else x)
    return out


# ======================================================================
# metrics
# ======================================================================
class Metric:
    """Primary metric of one dataset, on any subset (with repeats) of its videos."""

    def __init__(self, kind, labels, wpos=None, wneg=None):
        self.kind, self.labels, self.wpos, self.wneg = kind, labels, wpos, wneg

    def valid(self, idx):
        if self.kind == "xd_ap":
            return (sum(self.wpos[i].sum() for i in idx) > 0
                    and sum(self.wneg[i].sum() for i in idx) > 0)
        lab = np.concatenate([self.labels[i] for i in idx])
        return lab.min() != lab.max()

    def __call__(self, seqs, idx=None):
        idx = range(len(seqs)) if idx is None else idx
        s = np.concatenate([seqs[i] for i in idx])
        if self.kind == "xd_ap":
            return weighted_ap(s, np.concatenate([self.wpos[i] for i in idx]),
                               np.concatenate([self.wneg[i] for i in idx]))
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(np.concatenate([self.labels[i] for i in idx]), s))


def _weighted(scores, wpos, wneg):
    y = np.r_[np.ones(len(scores)), np.zeros(len(scores))]
    s = np.r_[scores, scores]
    w = np.r_[wpos, wneg].astype(float)
    k = w > 0
    return y[k], s[k], w[k]


def weighted_ap(scores, wpos, wneg):
    """XD-Violence AP exactly as the authors' test.py (auc(recall, precision) over
    every frame, each snippet score repeated 16x), computed on per-snippet
    positive/negative frame counts -- identical result, 16x fewer points."""
    from sklearn.metrics import auc, precision_recall_curve
    y, s, w = _weighted(scores, wpos, wneg)
    p, r, _ = precision_recall_curve(y, s, sample_weight=w)
    return float(auc(r, p))


def weighted_secondary(scores, wpos, wneg):
    """Frame AUROC and step-wise AP (sklearn average_precision_score)."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    y, s, w = _weighted(scores, wpos, wneg)
    return {"auc": float(roc_auc_score(y, s, sample_weight=w)),
            "ap_step": float(average_precision_score(y, s, sample_weight=w))}


def paired_bootstrap(P, metric, n, B, seed):
    """Video-level paired bootstrap. P: name -> per-video score list, or a tuple
    of such lists (controls; their metrics are averaged per resample)."""
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in P}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if not metric.valid(idx):
            continue
        for k, v in P.items():
            group = v if isinstance(v, tuple) else (v,)
            draws[k].append(float(np.mean([metric(g, idx) for g in group])))
    return {k: np.asarray(v) for k, v in draws.items()}


def ci95(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def derangements(n, seed, k):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < k:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            out.append(perm)
    return out


# ======================================================================
# text
# ======================================================================
def text_encoder(backbone="vitl"):
    """phrases -> unit CLIP text embeddings, one row per phrase (cached)."""
    import open_clip
    import torch
    import torch.nn.functional as tF

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    name, pretrained = BACKBONES[backbone]
    m, _, _ = open_clip.create_model_and_transforms(name, pretrained=pretrained)
    m = m.to(dev).eval()
    tok = open_clip.get_tokenizer(name)
    print("  %s / %s: learned logit scale %.3f (lambda stays %.0f)"
          % (name, pretrained, float(m.logit_scale.exp()), LAM))
    cache = {}

    def enc(phrases):
        need = [p for p in dict.fromkeys(phrases) if p not in cache]
        for i in range(0, len(need), 128):
            chunk = need[i:i + 128]
            with torch.no_grad():
                f = tF.normalize(m.encode_text(tok(chunk).to(dev)), dim=-1)
            for p, row in zip(chunk, f.cpu().numpy().astype(np.float64)):
                cache[p] = row
        return np.stack([cache[p] for p in phrases])
    return enc


def fake_encoder(dim=64):
    """Deterministic stand-in for the self-test: a phrase's vector is seeded by
    its hash, plus a shared component for phrases that share words."""
    vocab = {}

    def word(wd):
        if wd not in vocab:
            seed = int(hashlib.sha1(wd.encode()).hexdigest()[:8], 16)
            vocab[wd] = np.random.default_rng(seed).standard_normal(dim)
        return vocab[wd]

    def enc(phrases):
        return unit(np.stack([np.mean([word(w) for w in p.lower().split()], axis=0)
                              for p in phrases]))
    return enc


def legacy_prompts(desc):
    """Exactly per_view_descriptors.prototypes' prompt lists (surveillance, M3 normal)."""
    from da_zvad.context.verbalized import VerbalizedContext
    n, a = get_prompts("surveillance")
    if desc is not None:
        n, a = VerbalizedContext(description=desc, mode="normal").ground(n, a)
    return list(n), list(a)


def templated(desc):
    """A scene sentence as M3 injects it: the two normal-side templates
    ("<desc>, everything is normal", "<desc>, a usual and safe moment").
    Retrieved, LLaVA and shared sentences all enter the bank this way, so the
    three differ only in who wrote the sentence."""
    from da_zvad.context.verbalized import VerbalizedContext
    if desc is None:
        return []
    return VerbalizedContext(description=desc, mode="normal").ground([], [])[0]


# ======================================================================
# data
# ======================================================================
def unit_of(spec, name):
    if spec["unit"] == "view":
        return os.path.basename(str(name))[:2]
    if spec["unit"] == "scene":
        from da_zvad.datasets.nwpu_campus import scene_of
        return scene_of(name)
    if spec["unit"] == "camera":
        return "camera"
    return str(name)


def load(ds, spec):
    from measurement_audit import load_cache
    F, L, names = load_cache(spec["cache"])
    d = {"F": F, "L": L, "names": names, "units": [unit_of(spec, n) for n in names]}
    if spec["metric"] == "xd_ap":
        from da_zvad.datasets.xd_violence import SNIPPET, parse_annotation, snippet_gt
        ann = parse_annotation(os.path.join(os.path.expanduser(spec["root"]), "annotations.txt"))
        d["wpos"], d["wneg"] = [], []
        for n, l in zip(names, L):
            g = snippet_gt(len(l), ann.get(n.split("/", 1)[1], []))
            assert len(g) == SNIPPET * len(l) and np.array_equal(g[::SNIPPET], l)
            pos = g.reshape(len(l), SNIPPET).sum(1).astype(float)
            d["wpos"].append(pos)
            d["wneg"].append(SNIPPET - pos)
    return d


def load_best(spec, units):
    """unit -> LLaVA sentence of the published best condition (or None)."""
    if not spec.get("best"):
        return None
    kind, paths = spec["best"]
    paths = [paths] if isinstance(paths, str) else paths
    path = next((p for p in paths if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError("no caption file among %s" % paths)
    obj = json.load(open(path, encoding="utf-8"))
    if kind == "scenes":
        obj = obj["captions"]
    missing = sorted(set(units) - set(obj))
    if missing:
        raise KeyError("captions missing for %d units, e.g. %s" % (len(missing), missing[:2]))
    return {u: obj[u] for u in set(units)}


def fake_dataset(ds, spec, rng):
    """Synthetic stand-in with the real structure (units, labels, XD weights)."""
    dim = 64
    n_units = {"camera": 1, "view": 6, "scene": 8}.get(spec["unit"], 30)
    n_vid = max(n_units, 30) if spec["unit"] != "video" else 30
    proto = {u: unit(rng.standard_normal(dim)) for u in range(n_units)}
    abn = unit(rng.standard_normal(dim))
    F, L, names = [], [], []
    for v in range(n_vid):
        u = v % n_units
        T = int(rng.integers(40, 90))
        lab = np.zeros(T, dtype=int)
        if v % 3 == 0:
            a = int(rng.integers(10, T - 15))
            lab[a:a + 12] = 1
        x = proto[u] + 0.6 * rng.standard_normal((T, dim)) + 0.8 * lab[:, None] * abn
        F.append(unit(x))
        L.append(lab)
        uname = {"view": "%02d" % u, "scene": "D%03d" % u}.get(spec["unit"], "")
        names.append({"view": "%s_%04d" % (uname, v), "scene": "%s_%02d" % (uname, v),
                      "camera": "%02d" % v}.get(spec["unit"], "%s/v%03d" % (ds, v)))
    d = {"F": F, "L": L, "names": names, "units": [unit_of(spec, n) for n in names]}
    if spec["metric"] == "xd_ap":
        d["wpos"] = [16.0 * l for l in L]
        d["wneg"] = [16.0 - 16.0 * l for l in L]
    return d


# ======================================================================
# one dataset
# ======================================================================
class Run:
    def __init__(self, ds, spec, data, enc, pool_legacy, best, B, fake=False):
        self.ds, self.spec, self.d, self.enc, self.B, self.fake = ds, spec, data, enc, B, fake
        self.pool = pool_legacy
        self.best = best
        F = data["F"]
        self.n = len(F)
        self.Fw = [window_feats(f, W_EMB) for f in F]
        labels = data["L"]
        self.metric = Metric(spec["metric"], labels, data.get("wpos"), data.get("wneg"))
        self.units = data["units"]
        self.unit_list = sorted(set(self.units))
        self.base_n, self.base_a = get_prompts("surveillance")

    # -- margins ----------------------------------------------------------
    def legacy(self, desc_for_unit):
        """Old code path (per_view_descriptors.prototypes) for gates and legacy rows."""
        out = []
        for f, u in zip(self.d["F"], self.units):
            desc = desc_for_unit(u)
            if self.pool is not None:
                import per_view_descriptors as pv
                pn, pa = pv.prototypes(self.pool, desc)
            else:
                n, a = legacy_prompts(desc)
                pn, pa = unit(self.enc(n).mean(0)), unit(self.enc(a).mean(0))
            out.append(f @ pa - f @ pn)
        return out

    def margins(self, on, extra_for_unit=None):
        """on: set of components. extra_for_unit(u) -> extra normal phrases."""
        out = []
        cache = {}
        for f, fw, u in zip(self.d["F"], self.Fw, self.units):
            extra = tuple(extra_for_unit(u)) if extra_for_unit else ()
            if extra not in cache:
                n, a = banks.bank(self.ds, self.base_n, self.base_a,
                                  near_miss="N" in on, worries="W" in on)
                cache[extra] = (self.enc(list(n) + list(extra)), self.enc(a))
            N, A = cache[extra]
            x = fw if "E" in on else f
            out.append(margins_lme(x, N, A) if "P" in on else margins_proto(x, N, A))
        return out

    def score(self, m):
        return processed(m, self.spec["w"], self.spec["normalize"])

    # -- retrieval ----------------------------------------------------------
    def describe_units(self, lib, base_margins):
        texts = [e["text"] for e in lib]
        lib_emb = self.enc(texts)
        neu = self.enc(banks.NEUTRAL)
        first = {}
        for i, (name, u) in enumerate(zip(self.d["names"], self.units)):
            if u not in first or name < self.d["names"][first[u]]:
                first[u] = i
        out = {}
        for u in self.unit_list:
            i = first[u]
            r = rt.describe(self.d["F"][i], base_margins[i], lib_emb, neu,
                            self.spec["n_warm"], K_CALM, K_RET)
            r.update(video=self.d["names"][i], texts=[texts[j] for j in r["phrases"]],
                     top_texts=[texts[j] for j in r["top"]])
            out[u] = r
        return out


def best_sentences(spec, units, fake, notes):
    """unit -> LLaVA sentence, or None when this dataset has none available."""
    if fake:
        return ({u: "a place number %s with people" % u for u in set(units)}
                if spec.get("best") else None)
    try:
        return load_best(spec, units)
    except (FileNotFoundError, KeyError) as e:
        notes.append("legacy best condition unavailable: %s" % e)
        return None


def evaluate_dataset(ds, spec, data, enc, pool_legacy, B, fake=False, log=print, validate=None):
    """validate=False skips the published-number gates (a new backbone has none)."""
    validate = (not fake) if validate is None else validate
    rng_ctrl = np.random.default_rng(CTRL_SEED)
    report = {"dataset": ds, "n_videos": len(data["F"]), "gates": [], "notes": []}
    units = data["units"]
    best = best_sentences(spec, units, fake, report["notes"])
    R = Run(ds, spec, data, enc, pool_legacy, best, B, fake)
    met = R.metric
    log("\n" + "=" * 78 + "\n%s  (%d videos, %d units, primary %s)\n" % (
        ds.upper(), R.n, len(R.unit_list), spec["metric"]) + "=" * 78)

    # ---------------------------------------------------------------- gates
    M = {"legacy_none": R.legacy(lambda u: None),
         "legacy_shared": R.legacy(lambda u: spec["shared"])}
    if best is not None:
        M["legacy_best"] = R.legacy(lambda u: best[u])
    S = {k: R.score(m) for k, m in M.items()}
    point = {k: met(s) for k, s in S.items()}

    gates_ok = True
    path_err = float(max(np.max(np.abs(a - b)) for a, b in zip(
        M["legacy_none"], R.margins(set(), None))))
    ok = path_err <= TOL_PATH
    gates_ok &= ok
    report["gates"].append({"what": "new code path == legacy (none)", "max_abs": path_err,
                            "tol": TOL_PATH, "ok": ok})
    log("  gate  new scoring path reproduces the legacy margins   max |diff| %.1e  %s"
        % (path_err, "ok" if ok else "MISMATCH"))
    refs = {}
    if validate:
        saved = spec.get("saved")
        if saved and os.path.exists(saved[0]):
            sj = json.load(open(saved[0]))["results"]
            refs = {k: (float(sj[c][col]), TOL_SAVED, os.path.basename(saved[0]))
                    for k, (c, col) in saved[1].items() if c in sj}
        for k, (v, tol) in spec.get("published", {}).items():
            refs.setdefault(k, (v, tol, "published"))
        for k in ("none", "shared", "best"):
            key = "legacy_" + k
            if key not in point:
                continue
            if k not in refs:
                report["notes"].append("no reference value for %s" % key)
                log("  gate  %-14s %.4f   no reference value -- not checked" % (key, point[key]))
                continue
            v, tol, src = refs[k]
            ok = abs(point[key] - v) <= tol
            gates_ok &= ok
            report["gates"].append({"what": key, "got": point[key], "ref": v, "tol": tol,
                                    "source": src, "ok": ok})
            log("  gate  %-14s %.4f   ref %.4f (%s, tol %.4f)  %s"
                % (key, point[key], v, src, tol, "ok" if ok else "MISMATCH"))
        if "none" not in refs:
            gates_ok = False
            log("  gate  no reference for legacy_none -- cannot validate this dataset")
    report["gates_passed"] = bool(gates_ok)
    if not gates_ok:
        log("  VALIDATION FAILED -- %s is not reported." % ds)
        return report

    # ---------------------------------------------------------------- retrieval
    lib = banks.library()
    desc = R.describe_units(lib, M["legacy_none"])
    dirty = [t for r in desc.values() for t in r["texts"] if not banks.is_clean(t)]
    assert not dirty, "banned word in a retrieved phrase: %s" % dirty[:3]
    fb = [u for u, r in desc.items() if r["fallback"]]
    report["retrieval"] = {"library_size": len(lib), "fallback_units": len(fb),
                           "units": len(desc), "contaminated_phrases": 0, "per_unit": desc}
    log("  retrieval: %d-phrase library, %d of %d units fell back to base prompts"
        % (len(lib), len(fb), len(desc)))
    for u in R.unit_list[:3]:
        log("    %-24s %s" % (str(u)[:24], desc[u]["texts"][:2] or "(fallback)"))
    top1 = lambda u: desc[u]["texts"][0] if desc[u]["texts"] else None   # noqa: E731
    own = lambda u: templated(top1(u))                                   # noqa: E731
    raw5 = lambda u: desc[u]["texts"]                                    # noqa: E731

    multi = len(R.unit_list) > 1
    if multi:
        if spec["unit"] == "video":
            perms = derangements(len(R.unit_list), int(rng_ctrl.integers(1 << 31)), N_CTRL)
        else:
            k = len(R.unit_list)
            perms = [np.roll(np.arange(k), -s) for s in range(1, k)]
        idx_of = {u: i for i, u in enumerate(R.unit_list)}
        moved = [{u: R.unit_list[p[idx_of[u]]] for u in R.unit_list} for p in perms]

    full = set(COMPONENTS)
    on_wo = lambda c: full - {c}                                          # noqa: E731
    M["legacy_R"] = R.legacy(top1)
    M["FULL"] = R.margins(full, own)
    for c in COMPONENTS:
        M["FULL-" + c] = R.margins(on_wo(c), own if c != "R" else None)
    for name, on in (("ladder_W", {"W"}), ("ladder_WN", {"W", "N"}),
                     ("ladder_WNP", {"W", "N", "P"})):
        M[name] = R.margins(on, None)
    M["FULL_R5raw"] = R.margins(full, raw5)          # secondary: the first design
    M["FULL_shared"] = R.margins(on_wo("R"), lambda u: templated(spec["shared"]))
    if best is not None:
        M["FULL_llava"] = R.margins(on_wo("R"), lambda u: templated(best[u]))
    if spec["campus"]:
        lib_nc = banks.library(exclude_tags=("campus",))
        desc_nc = R.describe_units(lib_nc, M["legacy_none"])
        report["retrieval_nocampus"] = {
            "library_size": len(lib_nc),
            "fallback_units": sum(r["fallback"] for r in desc_nc.values()),
            "per_unit": {u: {"texts": r["texts"], "fallback": r["fallback"]}
                         for u, r in desc_nc.items()}}
        M["FULL_nocampus"] = R.margins(full, lambda u: templated(
            desc_nc[u]["texts"][0] if desc_nc[u]["texts"] else None))
        M["legacy_R_nocampus"] = R.legacy(
            lambda u: desc_nc[u]["texts"][0] if desc_nc[u]["texts"] else None)
    S.update({k: R.score(m) for k, m in M.items() if k not in S})
    point.update({k: met(S[k]) for k in M if k not in point})

    # Every combination of components: the ablation table.
    report["subsets"] = {}
    for k in range(len(COMPONENTS) + 1):
        for combo in itertools.combinations(COMPONENTS, k):
            on = set(combo)
            report["subsets"][subset_key(on)] = met(R.score(R.margins(on, own if "R" in on else None)))
    for key, ref in (("none", "legacy_none"), ("R", "legacy_R")):
        err = abs(report["subsets"][key] - point[ref])
        ok = err <= TOL_SAVED
        report["gates"].append({"what": "subset %s == %s" % (key, ref), "max_abs": err,
                                "tol": TOL_SAVED, "ok": ok})
        log("  gate  subset %-5s reproduces %-12s |diff| %.1e  %s"
            % (key, ref, err, "ok" if ok else "MISMATCH"))
        if not ok:
            report["gates_passed"] = False
            log("  VALIDATION FAILED -- %s is not reported." % ds)
            return report

    ctrl = {}
    if multi:
        ctrl["FULL_Rperm"] = [R.score(R.margins(full, lambda u, mv=mv: templated(top1(mv[u]))))
                              for mv in moved]
        ctrl["legacy_Rperm"] = [R.score(R.legacy(lambda u, mv=mv: top1(mv[u])))
                                for mv in moved]
        for k, v in ctrl.items():
            point[k] = float(np.mean([met(s) for s in v]))

    log("\n  %-22s %8s" % ("condition", spec["metric"]))
    for k in point:
        log("  %-22s %8.4f" % (k, point[k]))
    if spec["metric"] == "xd_ap":
        report["secondary"] = {k: weighted_secondary(np.concatenate(S[k]),
                                                     np.concatenate(data["wpos"]),
                                                     np.concatenate(data["wneg"]))
                               for k in ("legacy_none", "legacy_best", "legacy_R", "FULL")
                               if k in S}

    # ---------------------------------------------------------------- comparisons
    comps = [("FULL", "legacy_none"), ("legacy_R", "legacy_none")]
    if best is not None:
        comps += [("FULL", "legacy_best"), ("legacy_R", "legacy_best"), ("FULL", "FULL_llava")]
    comps += [("FULL", "FULL-" + c) for c in COMPONENTS]
    comps += [("FULL", "FULL_shared"), ("FULL", "FULL_R5raw")]
    if multi:
        comps += [("FULL", "FULL_Rperm"), ("legacy_R", "legacy_Rperm")]
    if spec["campus"]:
        comps += [("FULL_nocampus", "FULL"), ("legacy_R_nocampus", "legacy_R")]
    need = {k for pair in comps for k in pair}
    P = {k: (tuple(ctrl[k][:BOOT_CTRL]) if k in ctrl else S[k]) for k in need}
    log("\n  paired bootstrap: %d resamples of %d videos, seed %d" % (B, R.n, BOOT_SEED))
    draws = paired_bootstrap(P, met, R.n, B, BOOT_SEED)
    report["comparisons"] = {}
    for a, b in comps:
        g = draws[a] - draws[b]
        r = {"point": point[a] - point[b], "ci95": ci95(g), "frac_positive": float(np.mean(g > 0))}
        report["comparisons"]["%s - %s" % (a, b)] = r
        log("  %-36s %+.4f   95%% CI [%+.4f, %+.4f]"
            % ("%s - %s" % (a, b), r["point"], *r["ci95"]))
    report["n_resamples"] = int(len(next(iter(draws.values()))))

    # ---------------------------------------------------------------- audit
    from measurement_audit import split, verdict
    report["audit"] = {}
    for a, b in (("FULL", "legacy_none"), ("FULL", "FULL-R"), ("legacy_R", "legacy_none")):
        r = split(M[a], M[b], data["L"], spec["w"], spec["normalize"])
        r["verdict"] = verdict(r)
        report["audit"]["%s vs %s" % (a, b)] = r
        log("  audit %-26s gap %+.4f  level share %5.2f  macro_m gap %+.4f  ->  %s"
            % ("%s vs %s" % (a, b), r["gap"], r["level_share"], r["macro_m_gap"], r["verdict"]))
    report["point"] = point
    return report


# ======================================================================
# across datasets
# ======================================================================
def keep_or_drop(reports):
    """The pre-registered rule (PREREG_normal_side.md, section 'Keep or drop')."""
    out = {}
    done = [r for r in reports if r.get("gates_passed") and "comparisons" in r]
    for c in COMPONENTS:
        rows = [r["comparisons"]["FULL - FULL-" + c] for r in done]
        pos = sum(x["point"] > 0 for x in rows)
        sig = sum(x["ci95"][0] > 0 for x in rows)
        harm = sum(x["ci95"][1] < 0 for x in rows)
        keep = len(rows) > 0 and pos > len(rows) / 2 and sig >= 2 and harm == 0
        out[c] = {"datasets": len(rows), "positive": pos, "ci_above_zero": sig,
                  "ci_below_zero": harm, "decision": "KEEP" if keep else "DROP"}
    return out


def kept(kd):
    return {c for c, r in kd.items() if r["decision"] == "KEEP"}


def leave_one_dataset_out(reports):
    """For each dataset, the components the rule keeps when that dataset is hidden
    from it. Those are then scored on the hidden dataset (evaluate_final), so the
    decision and the test use different data."""
    done = [r for r in reports if r.get("gates_passed") and "comparisons" in r]
    return {r["dataset"]: kept(keep_or_drop([x for x in done if x is not r])) for r in done}


def evaluate_final(ds, spec, data, enc, pool_legacy, B, systems, per_unit, fake=False, log=print):
    """Score named component sets against today's system, with paired CIs.

    systems: name -> set of components; per_unit: the retrieval result per unit.
    Returns the report and the smoothed per-video scores.
    """
    notes = []
    best = best_sentences(spec, data["units"], fake, notes)
    R = Run(ds, spec, data, enc, pool_legacy, best, B, fake)
    met = R.metric

    def sentence(u):
        return per_unit[u]["texts"][0] if per_unit[u]["texts"] else None

    M = {"legacy_none": R.legacy(lambda u: None)}
    if best is not None:
        M["legacy_best"] = R.legacy(lambda u: best[u])
    for name, on in systems.items():
        M[name] = R.margins(on, (lambda u: templated(sentence(u))) if "R" in on else None)
    S = {k: R.score(m) for k, m in M.items()}
    point = {k: met(s) for k, s in S.items()}
    draws = paired_bootstrap(S, met, R.n, B, BOOT_SEED)
    out = {"systems": {k: subset_key(v) for k, v in systems.items()}, "point": point,
           "comparisons": {}, "notes": notes}
    log("\n  %s: %s" % (ds, ", ".join("%s = %s" % (k, subset_key(v)) for k, v in systems.items())))
    for name in systems:
        for ref in ("legacy_none", "legacy_best"):
            if ref not in S:
                continue
            g = draws[name] - draws[ref]
            r = {"point": point[name] - point[ref], "ci95": ci95(g),
                 "frac_positive": float(np.mean(g > 0))}
            out["comparisons"]["%s - %s" % (name, ref)] = r
            log("    %-28s %.4f vs %.4f  %+.4f  95%% CI [%+.4f, %+.4f]"
                % ("%s - %s" % (name, ref), point[name], point[ref], r["point"], *r["ci95"]))
    from measurement_audit import split, verdict
    r = split(M["FINAL"], M["legacy_none"], data["L"], spec["w"], spec["normalize"])
    r["verdict"] = verdict(r)
    out["audit"] = r
    log("    audit FINAL vs legacy_none  gap %+.4f  level share %5.2f  ->  %s"
        % (r["gap"], r["level_share"], r["verdict"]))
    return out, S


def predictions(reports):
    """The pre-registered predictions, evaluated mechanically."""
    by = {r["dataset"]: r for r in reports if r.get("gates_passed") and "comparisons" in r}
    gap = lambda ds, k: by[ds]["comparisons"][k]["point"]                # noqa: E731
    res = {}
    n = len(by)
    wins = [ds for ds in by if gap(ds, "FULL - legacy_none") > 0]
    res["P1 FULL beats legacy_none on all but at most one dataset"] = {
        "value": "%d of %d" % (len(wins), n), "holds": n > 0 and len(wins) >= n - 1}
    if "xd_violence" in by and "legacy_R - legacy_best" in by["xd_violence"]["comparisons"]:
        v = gap("xd_violence", "legacy_R - legacy_best")
        res["P2 XD-Violence: retrieval beats LLaVA's sentence in today's system (AP)"] = {
            "value": v, "holds": v > 0}
    multi = [ds for ds in by if "FULL - FULL_Rperm" in by[ds]["comparisons"]]
    own = [ds for ds in multi if gap(ds, "FULL - FULL_Rperm") > 0]
    res["P3 own retrieved phrases beat another camera's on >= 3 of the multi-camera datasets"] = {
        "value": "%d of %d" % (len(own), len(multi)), "holds": len(own) >= min(3, len(multi))}
    vs = {ds: by[ds]["audit"]["FULL vs FULL-R"]["verdict"] for ds in by}
    res["P4 the retrieval gain is never RE-RANKING-dominated"] = {
        "value": vs, "holds": all(v != "RE-RANKING-dominated" for v in vs.values())}
    if "shanghaitech" in by:
        v = gap("shanghaitech", "FULL - FULL-W")
        res["P5a ShanghaiTech: worries do not help (FULL - FULL-W <= 0)"] = {
            "value": v, "holds": v <= 0}
    crime = [ds for ds in ("ucf_crime", "xd_violence") if ds in by]
    if crime:
        vals = {ds: gap(ds, "FULL - FULL-W") for ds in crime}
        res["P5b UCF-Crime and XD-Violence: worries help (FULL - FULL-W > 0)"] = {
            "value": vals, "holds": all(v > 0 for v in vals.values())}
    camp = [ds for ds in ("shanghaitech", "nwpu_campus") if ds in by]
    if camp:
        vals = {ds: gap(ds, "FULL_nocampus - FULL") for ds in camp}
        res["P6 removing campus phrases lowers FULL on ShanghaiTech and NWPU"] = {
            "value": vals, "holds": all(v < 0 for v in vals.values())}
    return res


def run_manifest(args):
    def git(*a):
        try:
            return subprocess.run(["git", "-C", REPO, *a], capture_output=True,
                                  text=True, timeout=30).stdout.strip()
        except Exception:                        # noqa: BLE001
            return "unknown"
    import sklearn
    def sha(*parts):
        p = os.path.join(REPO, *parts)
        return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else "missing"
    files = {"prereg": sha("docs", "10_phase3", "PREREG_normal_side.md"),
             "banks.py": sha("da_zvad", "prompts", "banks.py"),
             "retrieval.py": sha("da_zvad", "context", "retrieval.py"),
             "normal_side_sweep.py": sha("notebooks", "normal_side_sweep.py")}
    return {"started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git("rev-parse", "HEAD"),
            "working_tree": "clean" if not git("status", "--porcelain") else "DIRTY",
            "sha256": files, "host": platform.node(),
            "python": platform.python_version(), "numpy": np.__version__,
            "sklearn": sklearn.__version__, "args": vars(args),
            "constants": {"LAM": LAM, "BOOT_SEED": BOOT_SEED, "CTRL_SEED": CTRL_SEED,
                          "N_CTRL": N_CTRL, "BOOT_CTRL": BOOT_CTRL, "K_CALM": K_CALM,
                          "K_RET": K_RET, "W_EMB": W_EMB}}


def run(args) -> int:
    fake = getattr(args, "fake", False)
    tag = getattr(args, "backbone", "vitl")
    default = tag == "vitl"
    suffix = "" if default else "_" + tag
    specs = {d: spec_for(d, tag) for d in ORDER}
    names = args.datasets or [d for d in ORDER if fake or
                              os.path.exists(os.path.expanduser(specs[d]["cache"]))]
    if not fake:
        for d in ORDER:
            if d not in names:
                print("skipping %s: no cache at %s" % (d, specs[d]["cache"]))
    if fake:
        enc, pool = fake_encoder(), None
    else:
        print("loading %s's text encoder ..." % BACKBONES[tag][0])
        enc = text_encoder(tag)
        pool = None
        if default:
            import per_view_descriptors as pv
            pool = pv.encoder()
    reports, datas = [], {}
    rng = np.random.default_rng(0)

    def save(rep):
        if fake:
            return
        os.makedirs(TABLES, exist_ok=True)
        out = os.path.join(TABLES, "normal_side_%s%s.json" % (rep["dataset"], suffix))
        json.dump({"manifest": run_manifest(args), **rep}, open(out, "w"), indent=1, default=float)
        print("  saved %s" % out)

    def data_for(ds):
        if fake:
            return datas.setdefault(ds, fake_dataset(ds, specs[ds], rng))
        return load(ds, specs[ds])

    for ds in names:
        rep = evaluate_dataset(ds, specs[ds], data_for(ds), enc, pool, args.boot, fake=fake,
                               validate=(not fake) and default)
        reports.append(rep)
        save(rep)

    summary = {"manifest": run_manifest(args), "datasets": names, "backbone": tag,
               "gates": {r["dataset"]: r.get("gates_passed") for r in reports}}
    if default:
        kd = keep_or_drop(reports)
        pr = predictions(reports)
        lodo = leave_one_dataset_out(reports)
        print("\n" + "=" * 78 + "\nKEEP OR DROP (rule fixed in the pre-registration)\n" + "=" * 78)
        for c, r in kd.items():
            print("  %s  positive on %d/%d, CI above 0 on %d, CI below 0 on %d  ->  %s"
                  % (c, r["positive"], r["datasets"], r["ci_above_zero"], r["ci_below_zero"],
                     r["decision"]))
        print("\nPRE-REGISTERED PREDICTIONS")
        for k, r in pr.items():
            print("  %-80s %s   (%s)" % (k, "HOLDS" if r["holds"] else "REFUTED", r["value"]))
        final_on = kept(kd)
        summary.update(keep_or_drop=kd, predictions=pr, final_system=subset_key(final_on),
                       leave_one_dataset_out={d: subset_key(v) for d, v in lodo.items()})
    else:
        # A new backbone gets the recipe ViT-L's run chose; it does not choose again.
        lodo = {}
        final_on = set(COMPONENTS)
        if not fake:
            ref = os.path.join(TABLES, "normal_side_summary.json")
            if not os.path.exists(ref):
                print("run the ViT-L sweep first: its keep/drop table fixes the recipe")
                return 2
            final_on = kept(json.load(open(ref))["keep_or_drop"])
        summary["final_system"] = subset_key(final_on)

    print("\n" + "=" * 78 + "\nFINAL SYSTEM = %s  (selected by the rule on these same data)"
          % subset_key(final_on))
    if default:
        print("LODO = chosen with that dataset hidden from the rule, then scored on it")
    print("=" * 78)
    for rep in reports:
        ds = rep["dataset"]
        if not (rep.get("gates_passed") and "comparisons" in rep):
            continue
        systems = {"FINAL": final_on}
        if ds in lodo:
            systems["LODO"] = lodo[ds]
        data = data_for(ds)
        rep["final"], S = evaluate_final(ds, specs[ds], data, enc, pool, args.boot, systems,
                                         rep["retrieval"]["per_unit"], fake=fake)
        save(rep)
        if not fake:
            out = os.path.join(TABLES, "normal_side_scores_%s_%s.npz" % (ds, tag))
            np.savez(out, names=np.array(data["names"], dtype=object),
                     lengths=np.array([len(x) for x in S["FINAL"]]),
                     FINAL=np.concatenate(S["FINAL"]), none=np.concatenate(S["legacy_none"]))
    summary["headline"] = {r["dataset"]: r["final"]["comparisons"] for r in reports if "final" in r}
    if not fake:
        out = os.path.join(TABLES, "normal_side_summary%s.json" % suffix)
        json.dump(summary, open(out, "w"), indent=1, default=float)
        print("\nsaved %s -- copy normal_side_*.json to results/runs/analysis/ on the laptop" % out)
    return 0 if all(r.get("gates_passed") for r in reports) else 1


def split_by_lengths(x, lengths):
    return np.split(np.asarray(x), np.cumsum(lengths)[:-1])


def backbones(args) -> int:
    """Paired video-level comparison of a new backbone with ViT-L, same recipe."""
    tag = args.backbone
    if tag == "vitl":
        print("give --backbone pe-l or vith")
        return 2
    res = {}
    for ds in ORDER:
        pa = os.path.join(TABLES, "normal_side_scores_%s_vitl.npz" % ds)
        pb = os.path.join(TABLES, "normal_side_scores_%s_%s.npz" % (ds, tag))
        if not (os.path.exists(pa) and os.path.exists(pb)):
            continue
        za, zb = np.load(pa, allow_pickle=True), np.load(pb, allow_pickle=True)
        if list(za["names"]) != list(zb["names"]) or not np.array_equal(za["lengths"], zb["lengths"]):
            print("%s: videos differ between backbones -- skipped" % ds)
            continue
        data = load(ds, spec_for(ds, "vitl"))
        met = Metric(SPECS[ds]["metric"], data["L"], data.get("wpos"), data.get("wneg"))
        P = {"vitl_none": split_by_lengths(za["none"], za["lengths"]),
             "vitl_final": split_by_lengths(za["FINAL"], za["lengths"]),
             "new_none": split_by_lengths(zb["none"], zb["lengths"]),
             "new_final": split_by_lengths(zb["FINAL"], zb["lengths"])}
        draws = paired_bootstrap(P, met, len(data["L"]), args.boot, BOOT_SEED)
        point = {k: met(v) for k, v in P.items()}
        res[ds] = {"point": point}
        for a, b in (("new_none", "vitl_none"), ("new_final", "vitl_final")):
            g = draws[a] - draws[b]
            res[ds][a + " - " + b] = {"point": point[a] - point[b], "ci95": ci95(g)}
            print("  %-14s %-24s %.4f vs %.4f  %+.4f  95%% CI [%+.4f, %+.4f]"
                  % (ds, a + " - " + b, point[a], point[b], point[a] - point[b], *ci95(g)))
    wins = sum(r["new_final - vitl_final"]["point"] > 0 for r in res.values())
    p7 = {"value": "%d of %d" % (wins, len(res)), "holds": len(res) > 0 and wins >= min(3, len(res))}
    print("\n  P7 the new backbone beats ViT-L on >= 3 datasets, same recipe: %s (%s)"
          % ("HOLDS" if p7["holds"] else "REFUTED", p7["value"]))
    out = os.path.join(TABLES, "normal_side_backbones_%s.json" % tag)
    json.dump({"manifest": run_manifest(args), "P7": p7, "datasets": res}, open(out, "w"),
              indent=1, default=float)
    print("saved %s" % out)
    return 0


# ======================================================================
# review sheet and exploratory analysis (after run)
# ======================================================================
def review(args) -> int:
    """One contact sheet per dataset: each unit's calm frames beside the phrases
    retrieval chose for it (and LLaVA's sentence, for comparison)."""
    from PIL import Image, ImageDraw
    from da_zvad.config import DAZVADConfig
    from da_zvad.datasets import get_dataset

    out_dir = os.path.join(WORK, "review")
    os.makedirs(out_dir, exist_ok=True)
    for ds in args.datasets or ORDER:
        path = os.path.join(TABLES, "normal_side_%s.json" % ds)
        if not os.path.exists(path):
            continue
        rep = json.load(open(path))
        if "retrieval" not in rep:
            continue
        spec = SPECS[ds]
        seqs = {s.name: s for s in get_dataset(DAZVADConfig(
            dataset=ds, data_root=os.path.expanduser(spec["root"]),
            frame_step=spec["step"])).sequences()}
        try:
            best = load_best(spec, list(rep["retrieval"]["per_unit"]))
        except Exception:                      # noqa: BLE001
            best = None
        units = sorted(rep["retrieval"]["per_unit"])
        units = [units[i] for i in np.random.default_rng(BOOT_SEED).permutation(len(units))[:20]]
        tw, th, rowh = 160, 90, 200
        sheet = Image.new("RGB", (5 * tw + 620, rowh * len(units)), "white")
        dr = ImageDraw.Draw(sheet)
        for r, u in enumerate(sorted(units)):
            info = rep["retrieval"]["per_unit"][u]
            seq = seqs.get(info["video"])
            for j, fi in enumerate(info["calm"][:5]):
                if seq is None:
                    break
                im = seq.frames[fi]
                im = (Image.open(im) if isinstance(im, str) else im).convert("RGB")
                sheet.paste(im.resize((tw, th)), (j * tw, r * rowh))
            x, y = 5 * tw + 10, r * rowh + 4
            dr.text((x, y), "%s  (%s)%s" % (str(u)[:40], info["video"][-30:],
                                             "   FALLBACK" if info["fallback"] else ""),
                    fill="black")
            for k, (t, s) in enumerate(zip(info["top_texts"], info["sims"])):
                dr.text((x, y + 18 + 16 * k), "%.3f  %s" % (s, t[:80]), fill="black")
            dr.text((x, y + 18 + 16 * 5), "neutral %.3f" % info["neutral_max"], fill="gray")
            if best:
                dr.text((x, y + 18 + 16 * 6), "LLaVA: %s" % best.get(u, "")[:80], fill="blue")
        out = os.path.join(out_dir, "review_%s.jpg" % ds)
        sheet.save(out, quality=85)
        print("wrote %s (%d units)" % (out, len(units)))
    return 0


def xd_contamination(args) -> int:
    """EXPLORATORY (uses labels; not a pre-registered test). Does the XD-Violence
    loss come from the videos whose captioned frames, or captions, show violence?
    Removes the sentence from only those videos and re-scores."""
    import per_view_descriptors as pv
    spec = SPECS["xd_violence"]
    data = load("xd_violence", spec)
    best = load_best(spec, data["units"])
    pool = pv.encoder()
    met = Metric("xd_ap", data["L"], data["wpos"], data["wneg"])
    frames_hit = {u for u, l in zip(data["units"], data["L"]) if l[:3].max() > 0}
    text_hit = {u for u in set(data["units"]) if not banks.is_clean(best[u])}

    def ap(desc_for_unit):
        m = []
        for f, u in zip(data["F"], data["units"]):
            pn, pa = pv.prototypes(pool, desc_for_unit(u))
            m.append(f @ pa - f @ pn)
        return met(processed(m, spec["w"], False))

    res = {"n_videos": len(data["F"]), "frames_show_violence": len(frames_hit),
           "caption_names_violence": len(text_hit),
           "caption_examples": sorted(best[u] for u in text_hit)[:15],
           "ap_none": ap(lambda u: None), "ap_per_video": ap(lambda u: best[u]),
           "ap_drop_sentence_where_frames_violent":
               ap(lambda u: None if u in frames_hit else best[u]),
           "ap_drop_sentence_where_caption_violent":
               ap(lambda u: None if u in text_hit else best[u]),
           "ap_sentence_only_where_frames_violent":
               ap(lambda u: best[u] if u in frames_hit else None)}
    for k, v in res.items():
        print("  %-42s %s" % (k, v if not isinstance(v, float) else "%.4f" % v))
    out = os.path.join(TABLES, "normal_side_xd_contamination.json")
    json.dump({"note": "EXPLORATORY: label-using diagnosis, not a pre-registered test", **res},
              open(out, "w"), indent=1)
    print("saved %s" % out)
    return 0


# ======================================================================
# self-test (no data, no GPU)
# ======================================================================
def selftest(args) -> int:
    fails = []

    def expect(cond, what):
        print("  %-66s %s" % (what, "ok" if cond else "FAIL"))
        if not cond:
            fails.append(what)

    rng = np.random.default_rng(1)
    F = unit(rng.standard_normal((50, 32)))
    N, A = unit(rng.standard_normal((1, 32))), unit(rng.standard_normal((1, 32)))
    expect(np.allclose(margins_lme(F, N, A), margins_proto(F, N, A)),
           "per-phrase rule == prototype rule with one phrase per side")
    N3, A3 = unit(rng.standard_normal((3, 32))), unit(rng.standard_normal((4, 32)))
    expect(np.allclose(margins_lme(F, np.vstack([N3, N3]), A3), margins_lme(F, N3, A3)),
           "per-phrase rule ignores duplicated phrases (count-normalised)")
    expect(np.allclose(margins_proto(F, N3, A3),
                       F @ unit(A3.mean(0)) - F @ unit(N3.mean(0))),
           "prototype rule = abnormal minus normal prototype similarity")
    expect(templated(None) == [] and templated("a quiet lane") == legacy_prompts("a quiet lane")[0][3:]
           and len(templated("a quiet lane")) == 2,
           "templated() = exactly the two sentences M3 adds to the normal side")
    expect(np.array_equal(window_feats(F, 1), F), "embedding window 1 is the identity")
    Fw = window_feats(F, 5)
    expect(Fw.shape == F.shape and np.allclose(np.linalg.norm(Fw, axis=1), 1),
           "embedding window keeps length and unit norm")
    expect(np.allclose(Fw[10], unit(F[8:13].mean(0))), "embedding window is centred")

    s = np.array([5, 1, 4, 0.5, 3, 9, 0.1, 2], dtype=float)
    expect(list(rt.calm_indices(s, 6, 2)) == [1, 3], "calm frames: lowest scores inside warm-up only")
    expect(list(rt.calm_indices(s[:2], 32, 5)) == [0, 1], "calm frames: short video uses all frames")
    lib = unit(rng.standard_normal((40, 32)))
    v = unit(lib[7] + 0.05 * rng.standard_normal(32))
    idx, sims = rt.retrieve(v, lib, 5)
    expect(idx[0] == 7 and np.all(np.diff(sims) <= 0), "retrieval returns the nearest phrase first")
    neu = unit(np.vstack([v + 0.01 * rng.standard_normal(32)]))
    expect(not rt.passes_self_check(v, float(sims[0]), neu),
           "self-check fails when a neutral phrase matches better")
    expect(rt.passes_self_check(v, float(sims[0]), unit(-v[None])),
           "self-check passes when the library matches better")
    d = rt.describe(np.vstack([lib[7]] * 10), np.arange(10.0), lib, unit(-lib[7][None]), 8, 3, 5)
    expect(d["calm"] == [0, 1, 2] and d["phrases"][0] == 7 and not d["fallback"],
           "describe(): calm frames, retrieval and self-check together")

    expect(banks.check_all() == [], "library and near-miss phrases contain no banned word")
    expect(not banks.is_clean("two men fighting in a street"), "filter catches 'fighting'")
    expect(not banks.is_clean("A building on FIRE"), "filter is case-insensitive")
    expect(banks.is_clean("fireworks over a river") and banks.is_clean("a runway"),
           "filter matches whole words only")
    lib_all, lib_nc = banks.library(), banks.library(exclude_tags=("campus",))
    expect(len(lib_all) >= 150 and len(lib_nc) < len(lib_all), "library size; campus removal shrinks it")
    expect(not any(w in e["text"] for e in lib_nc for w in ("campus", "university", "school",
                                                            "college", "classroom")),
           "remove-a-domain leaves no campus phrase")
    for ds in ORDER:
        n, a = banks.bank(ds, ["n0"], ["a0"])
        expect(n[0] == "n0" and a[0] == "a0" and len(a) > 1 and len(n) > 1,
               "bank(%s) keeps the base prompts and adds to both sides" % ds)
    worries_scene = [p for ps in banks.WORRIES.values() for p in ps
                     if any(w in p for w in ("walkway", "campus", "path "))]
    expect(not worries_scene, "worry phrases do not name the place (dilution rule)")

    from sklearn.metrics import auc, precision_recall_curve
    seqs, gts, wp, wn = [], [], [], []
    for _ in range(25):
        n = int(rng.integers(5, 30))
        sc = np.round(rng.random(n), 2)                 # rounding creates ties
        g = (rng.random(16 * n) < (0.3 if rng.random() < 0.4 else 0.0)).astype(int)
        seqs.append(sc)
        gts.append(g)
        wp.append(g.reshape(n, 16).sum(1).astype(float))
        wn.append(16 - wp[-1])
    p, r, _ = precision_recall_curve(np.concatenate(gts), np.concatenate([np.repeat(x, 16) for x in seqs]))
    expanded = auc(r, p)
    weighted = weighted_ap(np.concatenate(seqs), np.concatenate(wp), np.concatenate(wn))
    expect(abs(expanded - weighted) < 1e-12, "weighted XD AP == authors' expanded AP (%.6f)" % expanded)

    expect(all(not np.any(pp == np.arange(9)) for pp in derangements(9, 3, 10)),
           "derangements leave nobody in place")
    labs = [np.r_[np.zeros(10), np.ones(5)].astype(int) for _ in range(12)]
    m = Metric("auroc", labs)
    good = [np.r_[np.zeros(10), np.ones(5)] + 0.01 * rng.random(15) for _ in labs]
    bad = [rng.random(15) for _ in labs]
    dr = paired_bootstrap({"g": good, "b": bad}, m, 12, 200, 7)
    dr2 = paired_bootstrap({"g": good, "b": bad}, m, 12, 200, 7)
    expect(np.array_equal(dr["g"], dr2["g"]) and np.all(dr["g"] - dr["b"] > 0),
           "paired bootstrap is deterministic and paired")

    fake_reports = [{"dataset": d, "gates_passed": True, "comparisons": {
        "FULL - FULL-" + c: {"point": pt, "ci95": ci} for c, (pt, ci) in zip(
            COMPONENTS, [(0.02, [0.01, 0.03]), (-0.01, [-0.02, -0.001]), (0.01, [-0.01, 0.02]),
                         (0.02, [0.005, 0.03]), (0.0, [-0.01, 0.01])])}} for d in ORDER[:4]]
    kd = keep_or_drop(fake_reports)
    expect(kd["W"]["decision"] == "KEEP" and kd["N"]["decision"] == "DROP"
           and kd["P"]["decision"] == "DROP" and kd["E"]["decision"] == "KEEP"
           and kd["R"]["decision"] == "DROP", "keep/drop rule on a known table")

    expect(subset_key(set()) == "none" and subset_key({"R", "W"}) == "WR",
           "subset names are stable and independent of set order")
    lodo = leave_one_dataset_out([dict(r) for r in fake_reports])
    expect(all(v == {"W", "E"} for v in lodo.values()) and len(lodo) == 4,
           "leave-one-dataset-out applies the rule to the other datasets")
    hide = dict(fake_reports[0], comparisons={"FULL - FULL-" + c: {"point": 0.05, "ci95": [0.01, 0.09]}
                                              for c in COMPONENTS})
    rest = [dict(r, comparisons={"FULL - FULL-" + c: {"point": -0.01, "ci95": [-0.02, 0.01]}
                                 for c in COMPONENTS}) for r in fake_reports[1:]]
    expect(leave_one_dataset_out([hide] + rest)[hide["dataset"]] == set(),
           "leave-one-dataset-out cannot see the hidden dataset's own gain")
    parts = split_by_lengths(np.arange(6.0), [1, 2, 3])
    expect([len(x) for x in parts] == [1, 2, 3] and parts[2][0] == 3.0, "score arrays split back by video")
    expect(spec_for("ucf_crime", "pe-l")["cache"].endswith("ucf_crime_PE-Core-L-14-336_step16_crops1.npz")
           and spec_for("shanghaitech")["cache"] == SPECS["shanghaitech"]["cache"],
           "backbone cache names match cache_embeddings.py")

    print("\n  end-to-end run on synthetic data (fake encoder, 200 resamples) ...")
    ns = argparse.Namespace(datasets=None, boot=200, fake=True, backbone="vitl")
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = run(ns)
    text = buf.getvalue()
    expect(code == 0, "synthetic run completes with every gate passing")
    expect("KEEP OR DROP" in text and "PRE-REGISTERED PREDICTIONS" in text,
           "synthetic run reaches the keep/drop table and the predictions")
    expect("FINAL SYSTEM" in text and "LODO - legacy_none" in text and "audit FINAL" in text,
           "synthetic run scores the final system and the left-out datasets")
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        code2 = run(argparse.Namespace(datasets=["ucf_crime"], boot=50, fake=True, backbone="pe-l"))
    expect(code2 == 0 and "FINAL - legacy_none" in buf2.getvalue() and "KEEP OR DROP" not in buf2.getvalue(),
           "a new backbone reuses the recipe and does not choose again")
    if code != 0 or getattr(args, "verbose", False):
        print(text[-4000:])
    print("\n%s (%d failures)" % ("SELF-TEST PASSED" if not fails else "SELF-TEST FAILED", len(fails)))
    return 0 if not fails else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["selftest", "run", "review", "xd_contamination", "backbones"])
    ap.add_argument("--backbone", default="vitl", choices=list(BACKBONES))
    ap.add_argument("--datasets", nargs="*", choices=list(SPECS))
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--fake", action="store_true", help="synthetic data (testing only)")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    sys.exit({"selftest": selftest, "run": run, "review": review,
              "xd_contamination": xd_contamination, "backbones": backbones}[a.step](a))
