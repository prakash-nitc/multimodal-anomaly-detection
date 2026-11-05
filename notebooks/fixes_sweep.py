# -*- coding: utf-8 -*-
"""Four training-free fixes, each added to the plain detector, on two encoders.
Pre-registered in docs/10_phase3/PREREG_fixes.md; do not change the protocol here
without a dated addendum there.

    python notebooks/fixes_sweep.py selftest                # laptop, no data, ~1 min
    python notebooks/fixes_sweep.py run --backbone vitl     # server
    python notebooks/fixes_sweep.py run --backbone pe-l     # server
    python notebooks/fixes_sweep.py decide                  # server, after both runs

    C  calm normal side   eight calm, low-motion phrases on the normal side, the same
                          for every dataset; no look-alike actions
    A  visual anchor      the camera's calm frames' mean image vector on the normal side
    K  side keywords      "Normal scene: " / "Anomalous scene: " before every phrase
    S  anomaly weight     the abnormal similarity multiplied by 0.95

A fix is judged by adding it to the plain detector, not by removing it from the
full stack: the earlier rule kept a component that hurt when it stood alone.
Each encoder also gets every combination (16) with each sentence condition (none,
shared, LLaVA, retrieved), as point estimates; only the named conditions are
bootstrapped.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import normal_side_sweep as ns                                    # noqa: E402
from da_zvad.context import retrieval as rt                       # noqa: E402
from da_zvad.prompts import banks, fixes                          # noqa: E402

FIXES = ("C", "A", "K", "S")
SENTENCES = ("none", "shared", "llava", "R")
N_SHUFFLE = 5                  # the same shuffles give the point estimate and its CI
ORDER = ("shanghaitech", "ucf_crime", "xd_violence", "nwpu_campus")
ENCODERS = ("vitl", "pe-l")
SEED = ns.BOOT_SEED


def key(on):
    return "".join(c for c in FIXES if c in on) or "base"


def subsets():
    for k in range(len(FIXES) + 1):
        for combo in itertools.combinations(FIXES, k):
            yield set(combo)


# ======================================================================
# scoring
# ======================================================================
def phrases(on, normal_base, abnormal_base, extra=()):
    """-> (normal phrases, abnormal phrases) with fixes C and K applied."""
    n = list(normal_base) + (fixes.CALM_NORMAL if "C" in on else []) + list(extra)
    a = list(abnormal_base)
    if "K" in on:
        n = [fixes.KEY_NORMAL + p for p in n]
        a = [fixes.KEY_ANOMALOUS + p for p in a]
    return n, a


def margin(f, pn, pa, alpha=1.0, anchor=None, weight=0.0):
    """alpha * <f, pa> - [(1 - weight) * <f, pn> + weight * <f, anchor>]."""
    normal = f @ pn
    if weight:
        normal = (1.0 - weight) * normal + weight * (f @ anchor)
    return alpha * (f @ pa) - normal


class Bench:
    """One dataset on one encoder, wrapping the earlier run's data and retrieval."""

    def __init__(self, ds, spec, data, enc, B, fake=False):
        self.ds, self.spec, self.data, self.enc = ds, spec, data, enc
        notes = []
        self.best = ns.best_sentences(spec, data["units"], fake, notes)
        self.notes = notes
        self.R = ns.Run(ds, spec, data, enc, None, self.best, B, fake)
        self.anchors = {}
        self._proto = {}

    def describe(self, base_margins):
        desc = self.R.describe_units(banks.library(), base_margins)
        where = {n: i for i, n in enumerate(self.data["names"])}
        for u, r in desc.items():
            self.anchors[u] = rt.camera_vector(self.data["F"][where[r["video"]]], r["calm"])
        return desc

    def margins(self, on, extra_for_unit=None, alpha=None, weight=None):
        alpha = alpha if alpha is not None else (fixes.ALPHA_FLASHBACK if "S" in on else 1.0)
        weight = weight if weight is not None else (fixes.ANCHOR_WEIGHT if "A" in on else 0.0)
        tag = key(on & {"C", "K"})
        out = []
        for f, u in zip(self.data["F"], self.data["units"]):
            extra = tuple(extra_for_unit(u)) if extra_for_unit else ()
            pk = (tag, extra)
            if pk not in self._proto:
                n, a = phrases(on, self.R.base_n, self.R.base_a, extra)
                self._proto[pk] = (ns.unit(self.enc(n).mean(0)), ns.unit(self.enc(a).mean(0)))
            pn, pa = self._proto[pk]
            out.append(margin(f, pn, pa, alpha, self.anchors.get(u), weight))
        return out

    def score(self, m):
        return self.R.score(m)


def shuffles(unit_list, per_video):
    """N_SHUFFLE reassignments of one unit's sentence to another unit."""
    k = len(unit_list)
    if per_video:
        perms = ns.derangements(k, SEED, N_SHUFFLE)
    else:
        step = max(1, (k - 1) // N_SHUFFLE)
        perms = [np.roll(np.arange(k), -(1 + i * step)) for i in range(min(N_SHUFFLE, k - 1))]
    idx = {u: i for i, u in enumerate(unit_list)}
    return [{u: unit_list[p[idx[u]]] for u in unit_list} for p in perms]


# ======================================================================
# one dataset
# ======================================================================
def comparisons(has_best):
    c = [("C", "base"), ("A", "base"), ("K", "base"), ("S", "base"), ("S90", "base"),
         ("CAKS", "base"), ("shared", "base"), ("R", "base"), ("R", "shared"),
         ("CAKS_shared", "shared"), ("CAKS_R", "R"), ("R", "R_perm")]
    if has_best:
        c += [("llava", "base"), ("llava", "shared"), ("R", "llava"),
              ("CAKS_llava", "llava"), ("llava", "llava_perm")]
    return c


AUDITED = ("C", "A", "K", "S", "CAKS", "shared", "llava", "R")


def previous_points(ds, tag, fake):
    if fake:
        return None
    suffix = "" if tag == "vitl" else "_" + tag
    path = os.path.join(ns.TABLES, "normal_side_%s%s.json" % (ds, suffix))
    return json.load(open(path))["point"] if os.path.exists(path) else {}


def evaluate(ds, spec, data, enc, B, tag, fake=False, log=print):
    b = Bench(ds, spec, data, enc, B, fake)
    met = b.R.metric
    report = {"dataset": ds, "backbone": tag, "n_videos": len(data["F"]), "gates": [],
              "notes": b.notes}
    log("\n" + "=" * 78 + "\n%s on %s  (%d videos, %d units, primary %s)\n" % (
        ds.upper(), tag, b.R.n, len(b.R.unit_list), spec["metric"]) + "=" * 78)

    base_m = b.margins(set())
    desc = b.describe(base_m)
    top1 = lambda u: desc[u]["texts"][0] if desc[u]["texts"] else None   # noqa: E731
    sentence = {"none": None,
                "shared": lambda u: ns.templated(spec["shared"]),
                "R": lambda u: ns.templated(top1(u))}
    if b.best is not None:
        sentence["llava"] = lambda u: ns.templated(b.best[u])
    report["fallback_units"] = sum(r["fallback"] for r in desc.values())
    report["units"] = len(desc)

    # ---------------------------------------------------------------- gates
    ok_all = True
    legacy = {"legacy_none": lambda u: None, "legacy_shared": lambda u: spec["shared"],
              "legacy_R": top1}
    if b.best is not None:
        legacy["legacy_best"] = lambda u: b.best[u]
    legacy_m = {k: b.R.legacy(v) for k, v in legacy.items()}
    err = float(max(np.max(np.abs(x - y)) for x, y in zip(legacy_m["legacy_none"], base_m)))
    ok = err <= ns.TOL_PATH
    ok_all &= ok
    report["gates"].append({"what": "base == legacy path", "max_abs": err, "ok": ok})
    log("  gate  base margins == legacy path                 max |diff| %.1e  %s"
        % (err, "ok" if ok else "MISMATCH"))
    prev = previous_points(ds, tag, fake)
    for k, m in legacy_m.items():
        got = met(b.score(m))
        if prev is None:
            continue
        if k not in prev:
            ok_all = False
            log("  gate  %-14s %.4f   no earlier value -- cannot validate" % (k, got))
            continue
        ok = abs(got - prev[k]) <= ns.TOL_PATH
        ok_all &= ok
        report["gates"].append({"what": k, "got": got, "ref": prev[k], "ok": ok})
        log("  gate  %-14s %.6f  earlier run %.6f  %s" % (k, got, prev[k], "ok" if ok else "MISMATCH"))
    report["gates_passed"] = bool(ok_all)
    if not ok_all:
        log("  VALIDATION FAILED -- %s on %s is not reported." % (ds, tag))
        return report, {}

    # ---------------------------------------------------------------- table
    table = {}
    for s, fn in sentence.items():
        for on in subsets():
            table["%s|%s" % (s, key(on))] = met(b.score(b.margins(on, fn)))
    report["table"] = table

    # ---------------------------------------------------------------- named conditions
    full = set(FIXES)
    M = {"base": base_m,
         "C": b.margins({"C"}), "A": b.margins({"A"}), "K": b.margins({"K"}), "S": b.margins({"S"}),
         "S90": b.margins(set(), alpha=fixes.ALPHA_NOVA),
         "A_heavy": b.margins(set(), weight=fixes.ANCHOR_WEIGHT_HEAVY),
         "CAKS": b.margins(full)}
    for s, fn in sentence.items():
        if fn is not None:
            M[s] = b.margins(set(), fn)
            M["CAKS_" + s] = b.margins(full, fn)
    S = {k: b.score(m) for k, m in M.items()}
    S_subset = {key(on): b.score(b.margins(on)) for on in subsets()}
    point = {k: met(v) for k, v in S.items()}

    # shuffled controls: the same shuffles for the point estimate and the bootstrap
    ctrl = {}
    moved = shuffles(b.R.unit_list, spec["unit"] == "video")
    if len(b.R.unit_list) > 1:
        ctrl["R_perm"] = tuple(b.score(b.margins(set(), lambda u, mv=mv: ns.templated(top1(mv[u]))))
                               for mv in moved)
        if b.best is not None:
            ctrl["llava_perm"] = tuple(
                b.score(b.margins(set(), lambda u, mv=mv: ns.templated(b.best[mv[u]])))
                for mv in moved)
        for k, v in ctrl.items():
            point[k] = float(np.mean([met(x) for x in v]))

    log("\n  %-16s %8s" % ("condition", spec["metric"]))
    for k, v in point.items():
        log("  %-16s %8.4f" % (k, v))
    if spec["metric"] == "xd_ap":
        report["secondary"] = {
            k: ns.weighted_secondary(np.concatenate(S[k]), np.concatenate(data["wpos"]),
                                     np.concatenate(data["wneg"]))
            for k in ("base", "shared", "llava", "R", "C", "A", "K", "S", "CAKS") if k in S}
    tie = {k: float(1.0 - len(np.unique(np.concatenate(v))) / sum(len(x) for x in v))
           for k, v in S.items()}
    report["tied_score_fraction"] = tie
    log("  tied score fraction, worst: %s %.4f" % max(tie.items(), key=lambda kv: kv[1]))

    # ---------------------------------------------------------------- bootstrap
    pairs = [(x, y) for x, y in comparisons(b.best is not None) if x in point and y in point]
    need = {k for p in pairs for k in p}
    P = {k: ctrl[k] if k in ctrl else S[k] for k in need}
    log("\n  paired bootstrap: %d resamples of %d videos, seed %d" % (B, b.R.n, SEED))
    draws = ns.paired_bootstrap(P, met, b.R.n, B, SEED)
    report["comparisons"] = {}
    for x, y in pairs:
        g = draws[x] - draws[y]
        r = {"point": point[x] - point[y], "ci95": ns.ci95(g), "frac_positive": float(np.mean(g > 0))}
        report["comparisons"]["%s - %s" % (x, y)] = r
        log("  %-26s %+.4f   95%% CI [%+.4f, %+.4f]" % ("%s - %s" % (x, y), r["point"], *r["ci95"]))

    # ---------------------------------------------------------------- audit
    from measurement_audit import split, verdict
    report["audit"] = {}
    for k in AUDITED:
        if k in M:
            r = split(M[k], M["base"], data["L"], spec["w"], spec["normalize"])
            r["verdict"] = verdict(r)
            report["audit"][k] = r
            log("  audit %-8s vs base  gap %+.4f  level share %5.2f  macro_m gap %+.4f  ->  %s"
                % (k, r["gap"], r["level_share"], r["macro_m_gap"], r["verdict"]))
    report["point"] = point
    keep = {k: S[k] for k in S}
    keep.update({"subset_" + k: v for k, v in S_subset.items()})
    return report, keep


def manifest(args):
    m = ns.run_manifest(args)

    def sha(*parts):
        p = os.path.join(REPO, *parts)
        return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else "missing"
    m["sha256"].update(prereg_fixes=sha("docs", "10_phase3", "PREREG_fixes.md"),
                       fixes_py=sha("da_zvad", "prompts", "fixes.py"),
                       fixes_sweep=sha("notebooks", "fixes_sweep.py"))
    m["constants"].update(N_SHUFFLE=N_SHUFFLE, FIXES=FIXES)
    return m


def run(args):
    tag = args.backbone
    specs = {d: ns.spec_for(d, tag) for d in ORDER}
    names = args.datasets or list(ORDER)
    print("loading %s's text encoder ..." % ns.BACKBONES[tag][0])
    enc = ns.text_encoder(tag)
    rc = 0
    for ds in names:
        if not os.path.exists(os.path.expanduser(specs[ds]["cache"])):
            print("MISSING cache for %s: %s" % (ds, specs[ds]["cache"]))
            rc = 1
            continue
        data = ns.load(ds, specs[ds])
        rep, keep = evaluate(ds, specs[ds], data, enc, args.boot, tag)
        save(args, ds, tag, rep, keep, data)
        if not rep["gates_passed"]:
            rc = 1
    return rc


def save(args, ds, tag, rep, keep, data):
    os.makedirs(ns.TABLES, exist_ok=True)
    out = os.path.join(ns.TABLES, "fixes_%s_%s.json" % (ds, tag))
    json.dump({"manifest": manifest(args), **rep}, open(out, "w"), indent=1, default=float)
    print("  saved %s" % out)
    if rep["gates_passed"]:
        np.savez(os.path.join(ns.TABLES, "fixes_scores_%s_%s.npz" % (ds, tag)),
                 names=np.array(data["names"], dtype=object),
                 lengths=np.array([len(x) for x in data["F"]]),
                 **{k: np.concatenate(v).astype(np.float64) for k, v in keep.items()})


# ======================================================================
# across datasets and encoders
# ======================================================================
def passes(rows):
    """The rule: a positive point on more than half, CI above 0 on >= 2, below 0 on none."""
    pos = sum(r["point"] > 0 for r in rows)
    sig = sum(r["ci95"][0] > 0 for r in rows)
    harm = sum(r["ci95"][1] < 0 for r in rows)
    return {"datasets": len(rows), "positive": pos, "ci_above_zero": sig, "ci_below_zero": harm,
            "passes": len(rows) > 0 and pos > len(rows) / 2 and sig >= 2 and harm == 0}


def adopted(reports, names=None):
    use = [r for r in reports if names is None or r["dataset"] in names]
    return {f: passes([r["comparisons"]["%s - base" % f] for r in use]) for f in FIXES}


def combination_check(chosen, gain, compare):
    """Adopted fixes must also work together: while the combination has a CI below 0
    on any dataset, or is not positive on more than half, drop the adopted fix with
    the smallest mean gain. compare(set) -> list of {point, ci95}, one per dataset."""
    chosen, trail = set(chosen), []
    while chosen:
        rows = compare(chosen)
        r = passes(rows)
        trail.append({"set": key(chosen), **r})
        if r["ci_below_zero"] == 0 and r["positive"] > r["datasets"] / 2:
            break
        chosen.discard(min(chosen, key=lambda f: gain[f]))
    return chosen, trail


def scores_loader(tag):
    cache = {}

    def get(ds):
        if ds not in cache:
            cache[ds] = np.load(os.path.join(ns.TABLES, "fixes_scores_%s_%s.npz" % (ds, tag)),
                                allow_pickle=True)
        return cache[ds]
    return get


def predictions(rep, kept_on):
    """The pre-registered predictions, evaluated mechanically."""
    def gap(tag, ds, name):
        return rep[tag][ds]["comparisons"][name]["point"]
    res = {}
    earlier = {t: {d: json.load(open(os.path.join(ns.TABLES, "normal_side_%s%s.json" % (
        d, "" if t == "vitl" else "_" + t))))["subsets"]["N"] for d in ORDER} for t in ENCODERS}
    vals = {t: {d: rep[t][d]["point"]["C"] - earlier[t][d] for d in ("ucf_crime", "xd_violence")}
            for t in ENCODERS}
    res["F1 calm normal side beats the backwards N on UCF on both encoders, and on XD on ViT-L"] = {
        "value": vals, "holds": vals["vitl"]["ucf_crime"] > 0 and vals["pe-l"]["ucf_crime"] > 0
        and vals["vitl"]["xd_violence"] > 0}
    vals = {t: {d: gap(t, d, "A - base") for d in ("ucf_crime", "xd_violence")} for t in ENCODERS}
    res["F2 visual anchor changes UCF and XD by under 0.03 on both encoders"] = {
        "value": vals, "holds": all(abs(x) < 0.03 for v in vals.values() for x in v.values())}
    vals = {t: gap(t, "ucf_crime", "K - base") for t in ENCODERS}
    res["F3 side keywords gain under +0.05 on UCF on both encoders"] = {
        "value": vals, "holds": all(v < 0.05 for v in vals.values())}
    vals = {t: {d: rep[t][d]["audit"]["S"]["verdict"] for d in ORDER} for t in ENCODERS}
    res["F4 anomaly weight is calibration or no effect on >= 3 of 4 datasets on both encoders"] = {
        "value": vals, "holds": all(sum(x in ("CALIBRATION-dominated", "NO EFFECT TO DECOMPOSE")
                                        for x in v.values()) >= 3 for v in vals.values())}
    both = sorted(set(kept_on["vitl"]) & set(kept_on["pe-l"]))
    res["F5 at most one fix passes the rule on both encoders"] = {"value": both, "holds": len(both) <= 1}
    vals = {t: {"llava": rep[t]["xd_violence"]["point"]["llava"], "base": rep[t]["xd_violence"]["point"]["base"],
                "CAKS_llava": rep[t]["xd_violence"]["point"]["CAKS_llava"],
                "CAKS": rep[t]["xd_violence"]["point"]["CAKS"]} for t in ENCODERS}
    res["F6 LLaVA stays below no sentence on XD on both encoders, with and without all fixes"] = {
        "value": vals, "holds": all(v["llava"] < v["base"] and v["CAKS_llava"] < v["CAKS"]
                                    for v in vals.values())}
    flips = []
    for d in ORDER:
        for f in FIXES:
            a = rep["vitl"][d]["comparisons"]["%s - base" % f]["ci95"]
            b = rep["pe-l"][d]["comparisons"]["%s - base" % f]["ci95"]
            if (a[0] > 0 and b[1] < 0) or (a[1] < 0 and b[0] > 0):
                flips.append("%s on %s" % (f, d))
    res["F7 some fix has CIs on opposite sides of 0 on the two encoders, same dataset"] = {
        "value": flips, "holds": len(flips) > 0}
    vals = {t: sum(gap(t, d, "llava - llava_perm") > 0 for d in ORDER) for t in ENCODERS}
    res["F8 each unit's own LLaVA sentence beats another's on >= 3 of 4 on both encoders"] = {
        "value": vals, "holds": all(v >= 3 for v in vals.values())}
    return res


def decide(args):
    rep = {t: {d: json.load(open(os.path.join(ns.TABLES, "fixes_%s_%s.json" % (d, t)))) for d in ORDER}
           for t in ENCODERS}
    if not all(r["gates_passed"] for v in rep.values() for r in v.values()):
        print("a dataset failed its gates -- no decision")
        return 1
    out, kept_on = {"manifest": manifest(args)}, {}
    datas = {d: ns.load(d, ns.spec_for(d, "vitl")) for d in ORDER}
    mets = {d: ns.Metric(ns.SPECS[d]["metric"], datas[d]["L"], datas[d].get("wpos"), datas[d].get("wneg"))
            for d in ORDER}

    def vs_base(tag, d, name, get):
        z = get(d)
        P = {"x": ns.split_by_lengths(z[name], z["lengths"]),
             "base": ns.split_by_lengths(z["base"], z["lengths"])}
        draws = ns.paired_bootstrap(P, mets[d], len(z["lengths"]), args.boot, SEED)
        g = draws["x"] - draws["base"]
        return {"point": mets[d](P["x"]) - mets[d](P["base"]), "ci95": ns.ci95(g)}

    for t in ENCODERS:
        get = scores_loader(t)
        reports = [rep[t][d] for d in ORDER]
        rule = adopted(reports)
        print("\n" + "=" * 78 + "\nRULE on %s (each fix added to the plain detector)\n" % t + "=" * 78)
        for f, r in rule.items():
            print("  %s  positive on %d/%d, CI above 0 on %d, CI below 0 on %d  ->  %s"
                  % (f, r["positive"], r["datasets"], r["ci_above_zero"], r["ci_below_zero"],
                     "PASS" if r["passes"] else "fail"))
        chosen = {f for f, r in rule.items() if r["passes"]}
        kept_on[t] = sorted(chosen)
        gain = {f: float(np.mean([rep[t][d]["comparisons"]["%s - base" % f]["point"] for d in ORDER]))
                for f in FIXES}
        final, trail = combination_check(
            chosen, gain, lambda s: [vs_base(t, d, "subset_" + key(s), get) for d in ORDER])
        print("  adopted: %s; after the combination check: %s" % (key(chosen), key(final)))
        lodo = {}
        for d in ORDER:
            others = [x for x in ORDER if x != d]
            on = {f for f, r in adopted(reports, others).items() if r["passes"]}
            r = vs_base(t, d, "subset_" + key(on), get)
            lodo[d] = {"set": key(on), **r}
            print("  LODO %-14s rule on the other three picks %-5s  on hidden: %+.4f  95%% CI [%+.4f, %+.4f]"
                  % (d, key(on), r["point"], *r["ci95"]))
        out[t] = {"rule": rule, "adopted": key(chosen), "after_check": key(final), "check": trail,
                  "lodo": lodo, "gain": gain}

    print("\n" + "=" * 78 + "\nENCODERS, paired on the same videos: PE-Core minus ViT-L\n" + "=" * 78)
    out["encoders"] = {}
    get_a, get_b = scores_loader("vitl"), scores_loader("pe-l")
    for d in ORDER:
        za, zb = get_a(d), get_b(d)
        if list(za["names"]) != list(zb["names"]):
            print("  %s: videos differ between encoders -- skipped" % d)
            continue
        names = [k for k in ("base", "shared", "llava", "R", "CAKS") if k in za.files and k in zb.files]
        P = {}
        for k in names:
            P["a_" + k] = ns.split_by_lengths(za[k], za["lengths"])
            P["b_" + k] = ns.split_by_lengths(zb[k], zb["lengths"])
        draws = ns.paired_bootstrap(P, mets[d], len(za["lengths"]), args.boot, SEED)
        out["encoders"][d] = {}
        for k in names:
            g = draws["b_" + k] - draws["a_" + k]
            pa, pb = mets[d](P["a_" + k]), mets[d](P["b_" + k])
            out["encoders"][d][k] = {"vitl": pa, "pe-l": pb, "point": pb - pa, "ci95": ns.ci95(g)}
            print("  %-14s %-8s %.4f -> %.4f  %+.4f  95%% CI [%+.4f, %+.4f]"
                  % (d, k, pa, pb, pb - pa, *ns.ci95(g)))

    pr = predictions(rep, kept_on)
    print("\nPRE-REGISTERED PREDICTIONS")
    for k, r in pr.items():
        print("  %-88s %s   (%s)" % (k, "HOLDS" if r["holds"] else "REFUTED", r["value"]))
    out["predictions"] = pr
    path = os.path.join(ns.TABLES, "fixes_summary.json")
    json.dump(out, open(path, "w"), indent=1, default=float)
    print("\nsaved %s -- copy fixes_* to results/runs/analysis/ on the laptop" % path)
    return 0


# ======================================================================
# self-test (no data, no GPU)
# ======================================================================
def selftest(args):
    fails = []

    def expect(cond, what):
        print("  %-68s %s" % (what, "ok" if cond else "FAIL"))
        if not cond:
            fails.append(what)

    rng = np.random.default_rng(3)
    F = ns.unit(rng.standard_normal((40, 32)))
    pn, pa, anc = (ns.unit(rng.standard_normal(32)) for _ in range(3))
    expect(np.allclose(margin(F, pn, pa), F @ pa - F @ pn), "all fixes off == the legacy margin")
    expect(np.allclose(margin(F, pn, pa, alpha=0.9), 0.9 * (F @ pa) - F @ pn), "S multiplies only the abnormal side")
    w = fixes.ANCHOR_WEIGHT
    expect(np.allclose(margin(F, pn, pa, anchor=anc, weight=w), F @ pa - ((1 - w) * (F @ pn) + w * (F @ anc))),
           "A mixes the anchor into the normal side only")
    expect(abs(w - 1 / 11) < 1e-12, "anchor weight is NOVA's one part in eleven")

    n, a = phrases({"C"}, ["x"], ["y"], ["z"])
    expect(n == ["x"] + fixes.CALM_NORMAL + ["z"] and a == ["y"], "C adds the calm phrases to the normal side only")
    n, a = phrases({"K"}, ["x"], ["y"], ["z"])
    expect(n == ["Normal scene: x", "Normal scene: z"] and a == ["Anomalous scene: y"],
           "K labels each side, sentences included")
    expect(all(banks.is_clean(p) for p in fixes.CALM_NORMAL), "no calm phrase contains a banned word")
    nm = {p for v in banks.NEAR_MISS.values() for p in v}
    expect(not nm & set(fixes.CALM_NORMAL), "no calm phrase comes from the near-miss lists")
    expect(len(list(subsets())) == 16 and key(set()) == "base" and key(set(FIXES)) == "CAKS",
           "16 subsets, stable names")

    ok = {"point": 0.02, "ci95": [0.01, 0.03]}
    flat = {"point": 0.01, "ci95": [-0.01, 0.03]}
    bad = {"point": -0.02, "ci95": [-0.03, -0.01]}
    expect(passes([ok, ok, flat, flat])["passes"], "rule: 4/4 positive, 2 significant, no harm -> pass")
    expect(not passes([ok, flat, flat, bad])["passes"], "rule: harm on one dataset -> fail")
    expect(not passes([ok, flat, bad, {"point": -0.01, "ci95": [-0.02, 0.01]}])["passes"], "rule: 2/4 positive -> fail")
    expect(not passes([ok, flat, flat, flat])["passes"], "rule: one significant -> fail")

    calls = []

    def fake_compare(s):
        calls.append(key(s))
        return [ok, ok, ok, ok] if s == {"A"} else [ok, ok, ok, bad]
    final, trail = combination_check({"A", "K"}, {"A": 0.03, "K": 0.01}, fake_compare)
    expect(final == {"A"} and calls == ["AK", "A"], "combination check drops the weakest fix when the pair harms")
    final, _ = combination_check(set(), {}, fake_compare)
    expect(final == set(), "no adopted fix means the plain detector")

    u = ["a", "b", "c", "d", "e", "f", "g", "h"]
    for per_video in (True, False):
        s = shuffles(u, per_video)
        expect(len(s) == N_SHUFFLE and all(m[x] != x for m in s for x in u),
               "shuffles: %d, nobody keeps their own (%s)" % (N_SHUFFLE, "video" if per_video else "camera"))
    expect(len(shuffles(u[:3], False)) == 2, "shuffles: a two-rotation unit list stays valid")

    print("\n  full pipeline on synthetic data (fake encoder, no gates) ...")
    rng = np.random.default_rng(0)
    enc = ns.fake_encoder()
    reports = []
    logged = []
    for ds in ORDER:
        spec = ns.spec_for(ds, "vitl")
        data = ns.fake_dataset(ds, spec, rng)
        rep, keep = evaluate(ds, spec, data, enc, 30, "vitl", fake=True, log=logged.append)
        reports.append(rep)
        b = rep["comparisons"]
        expect(rep["gates_passed"], "%s: margin path gate passes" % ds)
        expect(len(rep["table"]) == 64, "%s: 16 subsets x 4 sentences" % ds)
        expect(abs(rep["table"]["none|base"] - rep["point"]["base"]) < 1e-12, "%s: table agrees with named base" % ds)
        expect(abs(rep["table"]["none|C"] - rep["point"]["C"]) < 1e-12, "%s: table agrees with named C" % ds)
        expect(abs(rep["table"]["llava|base"] - rep["point"]["llava"]) < 1e-12, "%s: table agrees with named LLaVA" % ds)
        expect("R - R_perm" in b and "llava - llava_perm" in b, "%s: shuffled controls compared" % ds)
        expect("subset_CAKS" in keep and "subset_base" in keep, "%s: subset scores kept for the decide step" % ds)
        expect(all(len(x) == len(y) for x, y in zip(keep["base"], data["F"])), "%s: one score per frame" % ds)
    kept = adopted(reports)
    expect(set(kept) == set(FIXES) and all("passes" in r for r in kept.values()), "rule runs across datasets")
    expect(any("paired bootstrap" in str(x) for x in logged), "bootstrap ran")

    print("\n  the decide step, end to end, on synthetic results in a temporary folder ...")
    tables, real_load = ns.TABLES, ns.load
    fake_data = {}
    with tempfile.TemporaryDirectory() as tmp:
        ns.TABLES = tmp
        ns.load = lambda ds, spec: fake_data[ds]
        try:
            rng = np.random.default_rng(1)
            for ds in ORDER:
                spec = ns.spec_for(ds, "vitl")
                fake_data[ds] = ns.fake_dataset(ds, spec, rng)
                for t in ENCODERS:
                    rep, keep = evaluate(ds, spec, fake_data[ds], enc, 20, t, fake=True, log=logged.append)
                    save(args, ds, t, rep, keep, fake_data[ds])
                    earlier = "normal_side_%s%s.json" % (ds, "" if t == "vitl" else "_" + t)
                    json.dump({"subsets": {"N": 0.5}}, open(os.path.join(tmp, earlier), "w"))
            args.boot = 20
            rc = decide(args)
            summary = json.load(open(os.path.join(tmp, "fixes_summary.json")))
            expect(rc == 0, "decide finishes")
            expect(all(t in summary and "lodo" in summary[t] for t in ENCODERS), "rule and LODO on both encoders")
            expect(len(summary["predictions"]) == 8, "all eight predictions evaluated")
            expect(set(summary["encoders"]) == set(ORDER), "encoders compared on every dataset")
        finally:
            ns.TABLES, ns.load = tables, real_load
    print("\n%s" % ("SELF-TEST FAILED: %s" % fails if fails else "self-test passed"))
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["selftest", "run", "decide"])
    ap.add_argument("--backbone", default="vitl", choices=ENCODERS)
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    return {"selftest": selftest, "run": run, "decide": decide}[args.step](args)


if __name__ == "__main__":
    sys.exit(main())
