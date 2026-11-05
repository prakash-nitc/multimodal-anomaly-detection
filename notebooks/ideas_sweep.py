# -*- coding: utf-8 -*-
"""The improvement round, one idea at a time, on PE-Core with cached embeddings.
Pre-registered in docs/10_phase3/PREREG_ideas.md; do not change a step's protocol
here without a dated addendum there.

    python notebooks/ideas_sweep.py selftest     # laptop, no data, ~1 min
    python notebooks/ideas_sweep.py step1        # server, after fixes_sweep.py
    python notebooks/ideas_sweep.py step2        # server
    python notebooks/ideas_sweep.py step3        # server
    python notebooks/ideas_sweep.py step4        # server, after step3
    python notebooks/ideas_sweep.py step5        # server, after step3

Step 1 checks the current best before anything is built on it:

    C2    a second wording of the calm normal side
    K     whether the side keywords earn their place on top of C
    old   PE-Core with and without the fixes against the ViT-L models, same videos

Step 2 averages the two encoders' margins:

    E     PE-Core + C with plain ViT-L
    EC    PE-Core + C with ViT-L + C

Step 3 takes three ideas from the papers, each on top of C:

    B     a frozen bank of concrete anomalous actions as the abnormal side
    T     the same bank, scored by anomaly type (NOVA)
    D     different templates on each side (Flashback)

Step 4 chooses part of the bank from what the camera shows:

    S     each unit's calm frames pick a scene, and that scene's types form the abnormal side

Step 5 adds what the camera has seen before, with no text:

    M     B plus how unlike the unit's earlier frames each frame is
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import fixes_sweep as fs                                          # noqa: E402
import normal_side_sweep as ns                                    # noqa: E402
from da_zvad.prompts import banks, fixes, ideas                   # noqa: E402

ORDER = fs.ORDER
TAG = "pe-l"
SEED = fs.SEED

STEP1_PAIRS = (("C2", "base"), ("C", "base"), ("C2", "C"),
               ("CK", "base"), ("CK", "C"), ("C2K", "base"), ("C2K", "C2"), ("C2K", "CK"),
               ("base", "old_llava"), ("C", "old_llava"), ("CK", "old_llava"),
               ("base", "old_plain"), ("C", "old_plain"), ("CK", "old_plain"))
NEW = ("base", "C", "CK")
OLD = ("old_llava", "old_plain")


def compare_with_old(rows):
    """What may be said of a new model against an old one, from one row per dataset."""
    r = fs.passes(rows)
    if r["passes"]:
        return "better"
    return "no worse" if r["ci_below_zero"] == 0 else "worse on %d" % r["ci_below_zero"]


def choose(rows):
    """The current best after step 1, from each dataset's comparisons."""
    get = lambda name: [r[name] for r in rows]   # noqa: E731
    wording = fs.passes(get("C2 - base"))
    keywords = fs.passes(get("CK - C"))
    second = sum(x["point"] > 0 for x in get("C2K - C2"))
    k_ok = keywords["passes"] and second > len(rows) / 2
    best = "base" if not wording["passes"] else ("CK" if k_ok else "C")
    return {"wording": wording, "keywords": keywords, "keywords_second_wording_positive": second,
            "keywords_adopted": bool(k_ok), "best": best}


def check_one(ds, spec, data, enc, B, fake=False, log=print):
    b = fs.Bench(ds, spec, data, enc, B, fake)
    met = b.R.metric
    report = {"dataset": ds, "backbone": TAG, "n_videos": len(data["F"]), "gates": [], "notes": b.notes}
    log("\n" + "=" * 78 + "\n%s on %s  (%d videos, primary %s)\n" % (
        ds.upper(), TAG, b.R.n, spec["metric"]) + "=" * 78)

    second = lambda u: ideas.CALM_SECOND                                      # noqa: E731
    M = {"base": b.margins(set()), "C": b.margins({"C"}), "CK": b.margins({"C", "K"}),
         "C2": b.margins(set(), second), "C2K": b.margins({"K"}, second),
         "C_places": b.margins(set(), lambda u: banks.CALM_ANCHORS),
         "C_people": b.margins(set(), lambda u: fixes.CALM_PEOPLE),
         "C2_places": b.margins(set(), lambda u: ideas.CALM_SECOND_PLACES),
         "C2_people": b.margins(set(), lambda u: ideas.CALM_SECOND_PEOPLE)}
    S = {k: b.score(m) for k, m in M.items()}

    # ---------------------------------------------------------------- gates
    ok_all = True

    def gate(what, got, ref, tol=ns.TOL_PATH):
        nonlocal ok_all
        ok = ref is not None and abs(got - ref) <= tol
        ok_all &= ok
        report["gates"].append({"what": what, "got": got, "ref": ref, "ok": ok})
        log("  gate  %-34s %.6f  reference %s  %s" % (
            what, got, "none" if ref is None else "%.6f" % ref, "ok" if ok else "MISMATCH"))

    same = b.margins(set(), lambda u: fixes.CALM_NORMAL)
    gate("calm phrases via the second path", float(max(np.max(np.abs(x - y)) for x, y in zip(same, M["C"]))), 0.0)
    pe = json.load(open(os.path.join(ns.TABLES, "fixes_%s_%s.json" % (ds, TAG))))["table"]
    for k, cell in (("base", "none|base"), ("C", "none|C"), ("CK", "none|CK")):
        gate("PE-Core %s == fixes run" % k, met(S[k]), pe.get(cell))
    z = fs.scores_loader("vitl")(ds)
    same_videos = list(z["names"]) == list(data["names"])
    report["same_videos"] = same_videos
    gate("same videos as the ViT-L run", float(same_videos), 1.0)
    vit = json.load(open(os.path.join(ns.TABLES, "fixes_%s_vitl.json" % ds)))["point"]
    for k, name in (("old_llava", "llava"), ("old_plain", "base")):
        S[k] = ns.split_by_lengths(z[name], z["lengths"])
        if same_videos:
            gate("ViT-L %s == fixes run" % name, met(S[k]), vit.get(name))
    report["gates_passed"] = bool(ok_all)
    if not ok_all:
        log("  VALIDATION FAILED -- %s is not reported." % ds)
        return report, {}

    # ---------------------------------------------------------------- points
    point = {k: met(v) for k, v in S.items()}
    log("\n  %-16s %8s" % ("condition", spec["metric"]))
    for k, v in point.items():
        log("  %-16s %8.4f" % (k, v))
    report["point"] = point
    if spec["metric"] == "xd_ap":
        report["secondary"] = {
            k: ns.weighted_secondary(np.concatenate(S[k]), np.concatenate(data["wpos"]),
                                     np.concatenate(data["wneg"])) for k in ("C", "CK", "C2", "C2K")}
    tie = {k: float(1.0 - len(np.unique(np.concatenate(v))) / sum(len(x) for x in v)) for k, v in S.items()}
    report["tied_score_fraction"] = tie

    # ---------------------------------------------------------------- bootstrap
    log("\n  paired bootstrap: %d resamples of %d videos, seed %d" % (B, b.R.n, SEED))
    need = {k for p in STEP1_PAIRS for k in p}
    draws = ns.paired_bootstrap({k: S[k] for k in need}, met, b.R.n, B, SEED)
    report["comparisons"] = {}
    for x, y in STEP1_PAIRS:
        g = draws[x] - draws[y]
        r = {"point": point[x] - point[y], "ci95": ns.ci95(g), "frac_positive": float(np.mean(g > 0))}
        report["comparisons"]["%s - %s" % (x, y)] = r
        log("  %-22s %+.4f   95%% CI [%+.4f, %+.4f]" % ("%s - %s" % (x, y), r["point"], *r["ci95"]))

    # ---------------------------------------------------------------- audit
    from measurement_audit import split, verdict
    report["audit"] = {}
    for k in ("C2", "C2K", "CK"):
        r = split(M[k], M["base"], data["L"], spec["w"], spec["normalize"])
        r["verdict"] = verdict(r)
        report["audit"][k] = r
        log("  audit %-4s vs base  gap %+.4f  level share %5.2f  macro_m gap %+.4f  ->  %s"
            % (k, r["gap"], r["level_share"], r["macro_m_gap"], r["verdict"]))
    return report, {k: S[k] for k in ("C", "CK", "C2", "C2K")}


def predictions(reports, decision):
    cmp = lambda d, name: reports[d]["comparisons"][name]             # noqa: E731
    res = {}
    res["G1 the second wording passes the rule against plain PE-Core"] = {
        "value": decision["wording"], "holds": decision["wording"]["passes"]}
    spans = [d for d in ORDER if cmp(d, "C2 - C")["ci95"][0] <= 0 <= cmp(d, "C2 - C")["ci95"][1]]
    res["G2 the two wordings differ within noise on at least 3 of 4 datasets"] = {
        "value": spans, "holds": len(spans) >= 3}
    res["G3 side keywords pass the rule on top of C"] = {
        "value": decision["keywords"], "holds": decision["keywords"]["passes"]}
    v = cmp("ucf_crime", "C2 - base")["point"]
    res["G4 the second wording also lowers UCF-Crime"] = {"value": v, "holds": v < 0}
    rows = {d: cmp(d, "C - old_llava")["ci95"] for d in ORDER}
    res["G5 C beats the old LLaVA model on XD with a CI above 0, and no CI is below 0"] = {
        "value": rows, "holds": rows["xd_violence"][0] > 0 and all(x[1] >= 0 for x in rows.values())}
    return res


def summarise(args, reports):
    rows = [reports[d]["comparisons"] for d in ORDER]
    decision = choose(rows)
    print("\n" + "=" * 78 + "\nSTEP 1 DECISION\n" + "=" * 78)
    for name, r in (("second wording vs plain", decision["wording"]), ("keywords on top of C", decision["keywords"])):
        print("  %-26s positive on %d/%d, CI above 0 on %d, CI below 0 on %d  ->  %s"
              % (name, r["positive"], r["datasets"], r["ci_above_zero"], r["ci_below_zero"],
                 "PASS" if r["passes"] else "fail"))
    print("  keywords on the second wording: positive on %d/%d" % (
        decision["keywords_second_wording_positive"], len(rows)))
    print("  CURRENT BEST: PE-Core %s" % decision["best"])
    verdicts = {}
    print("\n  against the old models, same videos:")
    for x in NEW:
        for y in OLD:
            v = compare_with_old([r["%s - %s" % (x, y)] for r in rows])
            verdicts["%s vs %s" % (x, y)] = v
            print("    %-5s vs %-10s %s" % (x, y, v))
    pr = predictions(reports, decision)
    print("\nPRE-REGISTERED PREDICTIONS")
    for k, r in pr.items():
        print("  %-82s %s" % (k, "HOLDS" if r["holds"] else "REFUTED"))
    out = {"manifest": manifest(args), "decision": decision, "old_models": verdicts, "predictions": pr}
    path = os.path.join(ns.TABLES, "ideas_step1_summary.json")
    json.dump(out, open(path, "w"), indent=1, default=float)
    print("\nsaved %s" % path)
    return out


def manifest(args):
    m = fs.manifest(args)

    def sha(*parts):
        p = os.path.join(REPO, *parts)
        return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else "missing"
    m["sha256"].update(prereg_ideas=sha("docs", "10_phase3", "PREREG_ideas.md"),
                       ideas_py=sha("da_zvad", "prompts", "ideas.py"),
                       ideas_sweep=sha("notebooks", "ideas_sweep.py"))
    return m


def save(args, ds, rep, keep, data):
    json.dump({"manifest": manifest(args), **rep},
              open(os.path.join(ns.TABLES, "ideas_step1_%s.json" % ds), "w"), indent=1, default=float)
    if rep["gates_passed"]:
        np.savez(os.path.join(ns.TABLES, "ideas_step1_scores_%s.npz" % ds),
                 names=np.array(data["names"], dtype=object),
                 lengths=np.array([len(x) for x in data["F"]]),
                 **{k: np.concatenate(v).astype(np.float64) for k, v in keep.items()})


def step1(args):
    specs = {d: ns.spec_for(d, TAG) for d in ORDER}
    print("loading %s's text encoder ..." % ns.BACKBONES[TAG][0])
    enc = ns.text_encoder(TAG)
    reports = {}
    for ds in ORDER:
        data = ns.load(ds, specs[ds])
        rep, keep = check_one(ds, specs[ds], data, enc, args.boot)
        save(args, ds, rep, keep, data)
        reports[ds] = rep
    if not all(r["gates_passed"] for r in reports.values()):
        print("a dataset failed its gates -- no decision")
        return 1
    summarise(args, reports)
    return 0


# ======================================================================
# step 2: both encoders together
# ======================================================================
STEP2_PAIRS = (("E", "C"), ("EC", "C"), ("EC", "E"),
               ("E", "old_llava"), ("EC", "old_llava"), ("E", "old_plain"), ("EC", "old_plain"))
COMBINED = ("E", "EC")


def fuse(a, b):
    """Mean of two encoders' margins, frame by frame, before the sigmoid and smoothing."""
    return [(x + y) / 2.0 for x, y in zip(a, b)]


def choose_combined(rows):
    """Which combination replaces PE-Core + C, if any, from each dataset's comparisons."""
    r = {k: fs.passes([x["%s - C" % k] for x in rows]) for k in COMBINED}
    gain = {k: float(np.mean([x["%s - C" % k]["point"] for x in rows])) for k in COMBINED}
    passed = [k for k in COMBINED if r[k]["passes"]]
    return {"rule": r, "gain": gain, "best": max(passed, key=lambda k: gain[k]) if passed else "C"}


def combine_one(ds, spec, data, enc, spec_v, data_v, enc_v, B, fake=False, log=print):
    pe = fs.Bench(ds, spec, data, enc, B, fake)
    vit = fs.Bench(ds, spec_v, data_v, enc_v, B, fake)
    met = pe.R.metric
    report = {"dataset": ds, "n_videos": len(data["F"]), "gates": [], "notes": pe.notes + vit.notes}
    log("\n" + "=" * 78 + "\n%s, both encoders  (%d videos, primary %s)\n" % (
        ds.upper(), pe.R.n, spec["metric"]) + "=" * 78)

    # ---------------------------------------------------------------- gates
    ok_all = True

    def gate(what, got, ref):
        nonlocal ok_all
        ok = ref is not None and abs(got - ref) <= ns.TOL_PATH
        ok_all &= ok
        report["gates"].append({"what": what, "got": got, "ref": ref, "ok": ok})
        log("  gate  %-34s %.6f  reference %s  %s" % (
            what, got, "none" if ref is None else "%.6f" % ref, "ok" if ok else "MISMATCH"))

    same = list(data["names"]) == list(data_v["names"])
    same = same and all(len(x) == len(y) and np.array_equal(x, y) for x, y in zip(data["L"], data_v["L"]))
    same = same and all(len(x) == len(y) for x, y in zip(data["F"], data_v["F"]))
    gate("same videos, frames and labels", float(same), 1.0)
    if not same:
        report["gates_passed"] = False
        log("  VALIDATION FAILED -- %s is not reported." % ds)
        return report, {}
    M = {"C": pe.margins({"C"}), "vit": vit.margins(set()), "vit_C": vit.margins({"C"})}
    S = {k: pe.score(m) for k, m in M.items()}
    table = {t: json.load(open(os.path.join(ns.TABLES, "fixes_%s_%s.json" % (ds, t))))["table"] for t in fs.ENCODERS}
    gate("PE-Core C == fixes run", met(S["C"]), table[TAG].get("none|C"))
    gate("ViT-L plain == fixes run", met(S["vit"]), table["vitl"].get("none|base"))
    gate("ViT-L C == fixes run", met(S["vit_C"]), table["vitl"].get("none|C"))
    z = fs.scores_loader("vitl")(ds)
    for k, name in (("old_llava", "llava"), ("old_plain", "base")):
        S[k] = ns.split_by_lengths(z[name], z["lengths"])
    gate("ViT-L LLaVA == fixes run", met(S["old_llava"]), table["vitl"].get("llava|base"))
    report["gates_passed"] = bool(ok_all)
    if not ok_all:
        log("  VALIDATION FAILED -- %s is not reported." % ds)
        return report, {}

    # ---------------------------------------------------------------- points
    M["E"], M["EC"] = fuse(M["C"], M["vit"]), fuse(M["C"], M["vit_C"])
    S["E"], S["EC"] = pe.score(M["E"]), pe.score(M["EC"])
    point = {k: met(v) for k, v in S.items()}
    log("\n  %-16s %8s" % ("condition", spec["metric"]))
    for k, v in point.items():
        log("  %-16s %8.4f" % (k, v))
    report["point"] = point
    if spec["metric"] == "xd_ap":
        report["secondary"] = {
            k: ns.weighted_secondary(np.concatenate(S[k]), np.concatenate(data["wpos"]),
                                     np.concatenate(data["wneg"])) for k in ("C", "E", "EC")}
    report["tied_score_fraction"] = {
        k: float(1.0 - len(np.unique(np.concatenate(v))) / sum(len(x) for x in v)) for k, v in S.items()}

    # ---------------------------------------------------------------- bootstrap
    log("\n  paired bootstrap: %d resamples of %d videos, seed %d" % (B, pe.R.n, SEED))
    need = {k for p in STEP2_PAIRS for k in p}
    draws = ns.paired_bootstrap({k: S[k] for k in need}, met, pe.R.n, B, SEED)
    report["comparisons"] = {}
    for x, y in STEP2_PAIRS:
        g = draws[x] - draws[y]
        r = {"point": point[x] - point[y], "ci95": ns.ci95(g), "frac_positive": float(np.mean(g > 0))}
        report["comparisons"]["%s - %s" % (x, y)] = r
        log("  %-22s %+.4f   95%% CI [%+.4f, %+.4f]" % ("%s - %s" % (x, y), r["point"], *r["ci95"]))

    # ---------------------------------------------------------------- audit
    from measurement_audit import split, verdict
    report["audit"] = {}
    for k in COMBINED:
        r = split(M[k], M["C"], data["L"], spec["w"], spec["normalize"])
        r["verdict"] = verdict(r)
        report["audit"][k] = r
        log("  audit %-3s vs C  gap %+.4f  level share %5.2f  macro_m gap %+.4f  ->  %s"
            % (k, r["gap"], r["level_share"], r["macro_m_gap"], r["verdict"]))
    return report, {k: S[k] for k in COMBINED}


def predictions_step2(reports, decision):
    cmp = lambda d, name: reports[d]["comparisons"][name]             # noqa: E731
    res = {}
    passed = [k for k in COMBINED if decision["rule"][k]["passes"]]
    res["H1 at least one combination passes the rule against PE-Core + C"] = {"value": passed, "holds": len(passed) > 0}
    v = cmp("xd_violence", "EC - C")["point"]
    res["H2 EC is above PE-Core + C on XD"] = {"value": v, "holds": v > 0}
    v = {k: cmp("ucf_crime", "%s - C" % k)["ci95"] for k in COMBINED}
    res["H3 neither combination has a CI above 0 on UCF"] = {"value": v, "holds": all(x[0] <= 0 for x in v.values())}
    v = {k: cmp("nwpu_campus", "%s - C" % k)["point"] for k in COMBINED}
    res["H4 both combinations are above PE-Core + C on NWPU"] = {"value": v, "holds": all(x > 0 for x in v.values())}
    return res


def summarise_step2(args, reports):
    rows = [reports[d]["comparisons"] for d in ORDER]
    decision = choose_combined(rows)
    print("\n" + "=" * 78 + "\nSTEP 2 DECISION\n" + "=" * 78)
    for k in COMBINED:
        r = decision["rule"][k]
        print("  %-3s vs C  mean gain %+.4f, positive on %d/%d, CI above 0 on %d, CI below 0 on %d  ->  %s"
              % (k, decision["gain"][k], r["positive"], r["datasets"], r["ci_above_zero"], r["ci_below_zero"],
                 "PASS" if r["passes"] else "fail"))
    print("  CURRENT BEST: %s" % {"C": "PE-Core + C", "E": "PE-Core + C with plain ViT-L",
                                  "EC": "PE-Core + C with ViT-L + C"}[decision["best"]])
    verdicts = {}
    print("\n  against the old models, same videos:")
    for x in COMBINED:
        for y in OLD:
            v = compare_with_old([r["%s - %s" % (x, y)] for r in rows])
            verdicts["%s vs %s" % (x, y)] = v
            print("    %-3s vs %-10s %s" % (x, y, v))
    pr = predictions_step2(reports, decision)
    print("\nPRE-REGISTERED PREDICTIONS")
    for k, r in pr.items():
        print("  %-70s %s" % (k, "HOLDS" if r["holds"] else "REFUTED"))
    out = {"manifest": manifest(args), "decision": decision, "old_models": verdicts, "predictions": pr}
    path = os.path.join(ns.TABLES, "ideas_step2_summary.json")
    json.dump(out, open(path, "w"), indent=1, default=float)
    print("\nsaved %s" % path)
    return out


def save_step2(args, ds, rep, keep, data):
    json.dump({"manifest": manifest(args), **rep},
              open(os.path.join(ns.TABLES, "ideas_step2_%s.json" % ds), "w"), indent=1, default=float)
    if rep["gates_passed"]:
        np.savez(os.path.join(ns.TABLES, "ideas_step2_scores_%s.npz" % ds),
                 names=np.array(data["names"], dtype=object),
                 lengths=np.array([len(x) for x in data["F"]]),
                 **{k: np.concatenate(v).astype(np.float64) for k, v in keep.items()})


def step2(args):
    print("loading both text encoders ...")
    enc, enc_v = ns.text_encoder(TAG), ns.text_encoder("vitl")
    reports = {}
    for ds in ORDER:
        spec, spec_v = ns.spec_for(ds, TAG), ns.spec_for(ds, "vitl")
        data, data_v = ns.load(ds, spec), ns.load(ds, spec_v)
        rep, keep = combine_one(ds, spec, data, enc, spec_v, data_v, enc_v, args.boot)
        save_step2(args, ds, rep, keep, data)
        reports[ds] = rep
    if not all(r["gates_passed"] for r in reports.values()):
        print("a dataset failed its gates -- no decision")
        return 1
    summarise_step2(args, reports)
    return 0


def selftest_step2(expect):
    ok = {"point": 0.02, "ci95": [0.01, 0.03]}
    big = {"point": 0.04, "ci95": [0.01, 0.07]}
    flat = {"point": 0.01, "ci95": [-0.01, 0.03]}
    bad = {"point": -0.02, "ci95": [-0.03, -0.01]}

    def rows(e, ec):
        return [{"E - C": a, "EC - C": b} for a, b in zip(e, ec)]
    fail = [ok, ok, flat, bad]
    expect(choose_combined(rows(fail, fail))["best"] == "C", "both combinations fail -> PE-Core + C stays")
    expect(choose_combined(rows([ok, ok, flat, flat], fail))["best"] == "E", "only E passes -> E")
    expect(choose_combined(rows([ok, ok, flat, flat], [big, big, flat, flat]))["best"] == "EC",
           "both pass -> the larger mean gain")
    a, b = [np.array([0.2, 0.4])], [np.array([0.4, 0.0])]
    expect(np.allclose(fuse(a, b)[0], [0.3, 0.2]), "fusion is the mean of the margins")
    expect(np.allclose(fuse(a, a)[0], a[0]), "fusing a score with itself changes nothing")

    print("\n  step 2, end to end, on synthetic results in a temporary folder ...")
    tables, logged = ns.TABLES, []
    enc = ns.fake_encoder()
    rng = np.random.default_rng(2)
    with tempfile.TemporaryDirectory() as tmp:
        ns.TABLES = tmp
        try:
            reports = {}
            fargs = SimpleNamespace(step="selftest", boot=20)
            for ds in ORDER:
                spec = ns.spec_for(ds, "vitl")
                data = ns.fake_dataset(ds, spec, rng)
                other = dict(data, F=[ns.unit(f + 0.3 * rng.standard_normal(f.shape)) for f in data["F"]])
                fs.save(fargs, ds, TAG, *_fixes_fixture(ds, spec, data, enc, logged), data)
                fs.save(fargs, ds, "vitl", *_fixes_fixture(ds, spec, other, enc, logged), other)
                rep, keep = combine_one(ds, spec, data, enc, spec, other, enc, 20, fake=True, log=logged.append)
                expect(rep["gates_passed"], "%s: every gate passes" % ds)
                expect(set(rep["comparisons"]) == {"%s - %s" % p for p in STEP2_PAIRS}, "%s: all comparisons made" % ds)
                expect(rep["point"]["E"] != rep["point"]["C"], "%s: fusion changes the score" % ds)
                expect(all(len(x) == len(y) for x, y in zip(keep["E"], data["F"])), "%s: one score per frame" % ds)
                save_step2(fargs, ds, rep, keep, data)
                reports[ds] = rep

            ds = ORDER[0]
            spec = ns.spec_for(ds, "vitl")
            data = ns.fake_dataset(ds, spec, np.random.default_rng(5))
            short = dict(data, F=[data["F"][0][:-1]] + data["F"][1:])
            rep, keep = combine_one(ds, spec, data, enc, spec, short, enc, 5, fake=True, log=logged.append)
            expect(not rep["gates_passed"] and keep == {}, "a different frame count stops the dataset")

            out = summarise_step2(fargs, reports)
            expect(out["decision"]["best"] in ("C", "E", "EC"), "the decision names a current best")
            expect(len(out["predictions"]) == 4 and len(out["old_models"]) == 4,
                   "all predictions and old-model verdicts made")
            expect(os.path.exists(os.path.join(tmp, "ideas_step2_summary.json")), "summary saved")
        finally:
            ns.TABLES = tables


def _fixes_fixture(ds, spec, data, enc, logged):
    """A stand-in for one encoder's fixes run, for the self-test's gates to read."""
    return fs.evaluate(ds, spec, data, enc, 5, TAG, fake=True, log=logged.append)


# ======================================================================
# step 3: the papers' ideas
# ======================================================================
STEP3 = ("B", "T", "D", "BD", "TD")
STEP3_PAIRS = tuple((k, "C") for k in STEP3) + tuple((k, "old_llava") for k in STEP3)
COMBO = {"B": "BD", "T": "TD"}


def top_half_mean(S):
    """Each row's mean over its top 50% of columns (NOVA eq. 4)."""
    k = int(np.ceil(S.shape[1] / 2.0))
    return np.partition(S, S.shape[1] - k, axis=1)[:, -k:].mean(axis=1)


def margin_by_type(F, N, groups):
    """Top-half mean over the normal phrases, against the best-matching anomaly type."""
    abnormal = np.max([top_half_mean(F @ g.T) for g in groups], axis=0)
    return abnormal - top_half_mean(F @ N.T)


def choose_papers(rows):
    """What replaces PE-Core + C, if anything, from each dataset's comparisons. A pair
    is only considered when its two parts pass alone."""
    r = {k: fs.passes([x["%s - C" % k] for x in rows]) for k in STEP3}
    gain = {k: float(np.mean([x["%s - C" % k]["point"] for x in rows])) for k in STEP3}
    bank = [k for k in ("B", "T") if r[k]["passes"]]
    candidates = bank + (["D"] if r["D"]["passes"] else [])
    if bank and r["D"]["passes"]:
        pair = COMBO[max(bank, key=lambda k: gain[k])]
        if r[pair]["passes"]:
            candidates.append(pair)
    return {"rule": r, "gain": gain, "bank_useful": bool(bank),
            "best": max(candidates, key=lambda k: gain[k]) if candidates else "C"}


def papers_one(ds, spec, data, enc, B, fake=False, log=print):
    b = fs.Bench(ds, spec, data, enc, B, fake)
    met = b.R.metric
    report = {"dataset": ds, "backbone": TAG, "n_videos": len(data["F"]), "gates": [], "notes": b.notes}
    log("\n" + "=" * 78 + "\n%s on %s  (%d videos, primary %s)\n" % (
        ds.upper(), TAG, b.R.n, spec["metric"]) + "=" * 78)
    normal, base_a = fs.phrases({"C"}, b.R.base_n, b.R.base_a)
    N, T_N = enc(normal), enc([ideas.template_normal(p) for p in normal])
    A = enc(ideas.BANK_PHRASES)
    groups = [enc(v) for v in ideas.BANK.values()]
    T_groups = [enc([ideas.template_anomalous(p, c) for p in v]) for c, v in ideas.BANK.items()]
    T_A = np.concatenate(T_groups)
    T_base = enc([ideas.template_anomalous(p) for p in base_a])

    M = {"C": b.margins({"C"}),
         "B": [ns.margins_proto(f, N, A) for f in data["F"]],
         "T": [margin_by_type(f, N, groups) for f in data["F"]],
         "D": [ns.margins_proto(f, T_N, T_base) for f in data["F"]],
         "BD": [ns.margins_proto(f, T_N, T_A) for f in data["F"]],
         "TD": [margin_by_type(f, T_N, T_groups) for f in data["F"]]}
    S = {k: b.score(m) for k, m in M.items()}

    # ---------------------------------------------------------------- gates
    ok_all = True

    def gate(what, got, ref):
        nonlocal ok_all
        ok = ref is not None and abs(got - ref) <= ns.TOL_PATH
        ok_all &= ok
        report["gates"].append({"what": what, "got": got, "ref": ref, "ok": ok})
        log("  gate  %-34s %.6f  reference %s  %s" % (
            what, got, "none" if ref is None else "%.6f" % ref, "ok" if ok else "MISMATCH"))

    same = [ns.margins_proto(f, N, enc(base_a)) for f in data["F"]]
    gate("C through the new scoring path", float(max(np.max(np.abs(x - y)) for x, y in zip(same, M["C"]))), 0.0)
    table = json.load(open(os.path.join(ns.TABLES, "fixes_%s_%s.json" % (ds, TAG))))["table"]
    gate("PE-Core C == fixes run", met(S["C"]), table.get("none|C"))
    z = fs.scores_loader("vitl")(ds)
    same_videos = list(z["names"]) == list(data["names"])
    gate("same videos as the ViT-L run", float(same_videos), 1.0)
    S["old_llava"] = ns.split_by_lengths(z["llava"], z["lengths"])
    vit = json.load(open(os.path.join(ns.TABLES, "fixes_%s_vitl.json" % ds)))["table"]
    if same_videos:
        gate("ViT-L LLaVA == fixes run", met(S["old_llava"]), vit.get("llava|base"))
    report["gates_passed"] = bool(ok_all)
    if not ok_all:
        log("  VALIDATION FAILED -- %s is not reported." % ds)
        return report, {}

    # ---------------------------------------------------------------- points
    point = {k: met(v) for k, v in S.items()}
    log("\n  %-16s %8s" % ("condition", spec["metric"]))
    for k, v in point.items():
        log("  %-16s %8.4f" % (k, v))
    report["point"] = point
    if spec["metric"] == "xd_ap":
        report["secondary"] = {
            k: ns.weighted_secondary(np.concatenate(S[k]), np.concatenate(data["wpos"]),
                                     np.concatenate(data["wneg"])) for k in ("C",) + STEP3}
    tie = {k: float(1.0 - len(np.unique(np.concatenate(v))) / sum(len(x) for x in v)) for k, v in S.items()}
    report["tied_score_fraction"] = tie
    log("  tied score fraction, worst: %s %.4f" % max(tie.items(), key=lambda kv: kv[1]))

    # ---------------------------------------------------------------- bootstrap
    log("\n  paired bootstrap: %d resamples of %d videos, seed %d" % (B, b.R.n, SEED))
    need = {k for p in STEP3_PAIRS for k in p}
    draws = ns.paired_bootstrap({k: S[k] for k in need}, met, b.R.n, B, SEED)
    report["comparisons"] = {}
    for x, y in STEP3_PAIRS:
        g = draws[x] - draws[y]
        r = {"point": point[x] - point[y], "ci95": ns.ci95(g), "frac_positive": float(np.mean(g > 0))}
        report["comparisons"]["%s - %s" % (x, y)] = r
        log("  %-22s %+.4f   95%% CI [%+.4f, %+.4f]" % ("%s - %s" % (x, y), r["point"], *r["ci95"]))

    # ---------------------------------------------------------------- audit
    from measurement_audit import split, verdict
    report["audit"] = {}
    for k in STEP3:
        r = split(M[k], M["C"], data["L"], spec["w"], spec["normalize"])
        r["verdict"] = verdict(r)
        report["audit"][k] = r
        log("  audit %-3s vs C  gap %+.4f  level share %5.2f  macro_m gap %+.4f  ->  %s"
            % (k, r["gap"], r["level_share"], r["macro_m_gap"], r["verdict"]))
    return report, {k: S[k] for k in STEP3}


def predictions_step3(reports, decision):
    cmp = lambda d, name: reports[d]["comparisons"][name]             # noqa: E731
    res = {}
    passed = [k for k in ("B", "T", "D") if decision["rule"][k]["passes"]]
    res["J1 at least one of B, T and D passes the rule against PE-Core + C"] = {
        "value": passed, "holds": len(passed) > 0}
    v = reports["xd_violence"]["point"]["T"]
    res["J2 scoring by type reaches 0.70 on XD"] = {"value": v, "holds": v >= 0.70}
    v = cmp("shanghaitech", "T - C")["point"]
    res["J3 scoring by type lowers ShanghaiTech"] = {"value": v, "holds": v < 0}
    v = cmp("ucf_crime", "D - C")["point"]
    res["J4 different templates raise UCF"] = {"value": v, "holds": v > 0}
    v = {"ucf_crime": max(reports["ucf_crime"]["point"][k] for k in STEP3),
         "xd_violence": max(reports["xd_violence"]["point"][k] for k in STEP3)}
    res["J5 nothing reaches NOVA's UCF 89.86 or XD 84.82"] = {
        "value": v, "holds": v["ucf_crime"] < 0.8986 and v["xd_violence"] < 0.8482}
    return res


def summarise_step3(args, reports):
    rows = [reports[d]["comparisons"] for d in ORDER]
    decision = choose_papers(rows)
    print("\n" + "=" * 78 + "\nSTEP 3 DECISION\n" + "=" * 78)
    for k in STEP3:
        r = decision["rule"][k]
        print("  %-3s vs C  mean gain %+.4f, positive on %d/%d, CI above 0 on %d, CI below 0 on %d  ->  %s"
              % (k, decision["gain"][k], r["positive"], r["datasets"], r["ci_above_zero"], r["ci_below_zero"],
                 "PASS" if r["passes"] else "fail"))
    print("  bank useful, so step 4 runs: %s" % ("yes" if decision["bank_useful"] else "no"))
    print("  CURRENT BEST: %s" % ("PE-Core + C" if decision["best"] == "C" else "PE-Core + C, " + decision["best"]))
    verdicts = {}
    print("\n  against the old LLaVA model, same videos:")
    for x in STEP3:
        v = compare_with_old([r["%s - old_llava" % x] for r in rows])
        verdicts["%s vs old_llava" % x] = v
        print("    %-3s %s" % (x, v))
    pr = predictions_step3(reports, decision)
    print("\nPRE-REGISTERED PREDICTIONS")
    for k, r in pr.items():
        print("  %-70s %s" % (k, "HOLDS" if r["holds"] else "REFUTED"))
    out = {"manifest": manifest(args), "decision": decision, "old_models": verdicts, "predictions": pr}
    path = os.path.join(ns.TABLES, "ideas_step3_summary.json")
    json.dump(out, open(path, "w"), indent=1, default=float)
    print("\nsaved %s" % path)
    return out


def save_step3(args, ds, rep, keep, data):
    json.dump({"manifest": manifest(args), **rep},
              open(os.path.join(ns.TABLES, "ideas_step3_%s.json" % ds), "w"), indent=1, default=float)
    if rep["gates_passed"]:
        np.savez(os.path.join(ns.TABLES, "ideas_step3_scores_%s.npz" % ds),
                 names=np.array(data["names"], dtype=object),
                 lengths=np.array([len(x) for x in data["F"]]),
                 **{k: np.concatenate(v).astype(np.float64) for k, v in keep.items()})


def step3(args):
    specs = {d: ns.spec_for(d, TAG) for d in ORDER}
    print("loading %s's text encoder ..." % ns.BACKBONES[TAG][0])
    enc = ns.text_encoder(TAG)
    reports = {}
    for ds in ORDER:
        data = ns.load(ds, specs[ds])
        rep, keep = papers_one(ds, specs[ds], data, enc, args.boot)
        save_step3(args, ds, rep, keep, data)
        reports[ds] = rep
    if not all(r["gates_passed"] for r in reports.values()):
        print("a dataset failed its gates -- no decision")
        return 1
    summarise_step3(args, reports)
    return 0


def selftest_step3(expect):
    bank = ideas.BANK_PHRASES
    expect(len(ideas.BANK) == 12 and all(len(v) == 4 for v in ideas.BANK.values()), "twelve types of four phrases")
    expect(len(set(bank)) == 48, "no phrase repeats in the bank")
    expect(not set(bank) & set(fixes.CALM_NORMAL + ideas.CALM_SECOND), "no bank phrase is on the normal side")
    n, a = ideas.template_normal("x"), ideas.template_anomalous("x", "y")
    expect(n != a and "x" in n and "x" in a and "y" in a, "the two sides get different wrappers")
    expect(n == "Normal Scene:\n- x\n", "the normal wrapper matches Flashback's yaml")
    expect(a == "Anomalous Scene:\n    Action Category: y\n    Scene Description: x\n",
           "the anomalous wrapper matches Flashback's yaml")

    S = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 0.0, 2.0, 0.0]])
    expect(np.allclose(top_half_mean(S), [3.5, 3.0]), "top-half mean, even count")
    expect(np.allclose(top_half_mean(S[:, :3]), [2.5, 3.0]), "top-half mean rounds up, odd count")
    F = np.eye(3)
    N = np.array([[0.0, 0.0, 1.0]])
    g = [np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]])]
    expect(np.allclose(margin_by_type(F, N, g), [1.0, 1.0, -1.0]), "scoring by type takes the best-matching type")

    ok = {"point": 0.02, "ci95": [0.01, 0.03]}
    big = {"point": 0.04, "ci95": [0.01, 0.07]}
    flat = {"point": 0.01, "ci95": [-0.01, 0.03]}
    bad = {"point": -0.02, "ci95": [-0.03, -0.01]}
    win, lose = [ok, ok, flat, flat], [ok, ok, flat, bad]

    def rows(**kw):
        return [{"%s - C" % k: kw.get(k, lose)[i] for k in STEP3} for i in range(4)]
    expect(choose_papers(rows())["best"] == "C", "nothing passes -> PE-Core + C stays")
    expect(not choose_papers(rows(D=win))["bank_useful"], "only templates pass -> the bank is not useful")
    expect(choose_papers(rows(B=win))["best"] == "B" and choose_papers(rows(B=win))["bank_useful"], "only B passes -> B")
    strong = [big, big, flat, flat]
    expect(choose_papers(rows(B=win, T=strong))["best"] == "T", "B and T both pass -> the larger gain")
    expect(choose_papers(rows(T=win, D=win, TD=strong))["best"] == "TD", "parts and pair pass, pair larger -> pair")
    expect(choose_papers(rows(T=strong, D=win))["best"] == "T", "the pair fails -> the best single")
    expect(choose_papers(rows(TD=strong))["best"] == "C", "a pair is ignored unless its parts pass")

    print("\n  step 3, end to end, on synthetic results in a temporary folder ...")
    tables, logged = ns.TABLES, []
    enc = ns.fake_encoder()
    rng = np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as tmp:
        ns.TABLES = tmp
        try:
            reports = {}
            fargs = SimpleNamespace(step="selftest", boot=20)
            first = {}
            for ds in ORDER:
                spec = ns.spec_for(ds, "vitl")
                data = ns.fake_dataset(ds, spec, rng)
                first[ds] = data
                for t in fs.ENCODERS:
                    fs.save(fargs, ds, t, *_fixes_fixture(ds, spec, data, enc, logged), data)
                rep, keep = papers_one(ds, spec, data, enc, 20, fake=True, log=logged.append)
                expect(rep["gates_passed"], "%s: every gate passes" % ds)
                expect(set(rep["comparisons"]) == {"%s - %s" % p for p in STEP3_PAIRS}, "%s: all comparisons made" % ds)
                expect(len({round(rep["point"][k], 9) for k in STEP3}) == 5, "%s: the five conditions differ" % ds)
                expect(all(len(x) == len(y) for x, y in zip(keep["TD"], data["F"])), "%s: one score per frame" % ds)
                save_step3(fargs, ds, rep, keep, data)
                reports[ds] = rep

            ds = ORDER[0]
            path = os.path.join(tmp, "fixes_%s_%s.json" % (ds, TAG))
            saved = json.load(open(path))
            saved["table"]["none|C"] += 0.01
            json.dump(saved, open(path, "w"))
            rep, keep = papers_one(ds, ns.spec_for(ds, "vitl"), first[ds], enc, 5, fake=True, log=logged.append)
            expect(not rep["gates_passed"] and keep == {}, "a mismatched reference stops the dataset")

            out = summarise_step3(fargs, reports)
            expect(out["decision"]["best"] in ("C",) + STEP3, "the decision names a current best")
            expect(len(out["predictions"]) == 5 and len(out["old_models"]) == 5,
                   "all predictions and old-model verdicts made")
            expect(os.path.exists(os.path.join(tmp, "ideas_step3_summary.json")), "summary saved")
        finally:
            ns.TABLES = tables


# ======================================================================
# step 4: phrases chosen by scene
# ======================================================================
STEP4_PAIRS = (("S", "B"), ("S", "S_wrong"), ("S_wrong", "B"), ("S", "C"), ("S", "old_llava"))
SCENE_NAMES = tuple(ideas.SCENES)


def pick_scenes(v, scene_emb, fallback):
    """-> (best scene, least-matching scene), or (None, None) when the self-check failed."""
    if fallback:
        return None, None
    sims = scene_emb @ v
    return SCENE_NAMES[int(np.argmax(sims))], SCENE_NAMES[int(np.argmin(sims))]


def bank_for(scene):
    """A chosen scene's part of the bank; the whole bank when no scene was chosen."""
    return ideas.scene_phrases(scene) if scene else ideas.BANK_PHRASES


def choose_scene_rule(rows):
    """Scene choice replaces B only if it passes the rule and also beats the wrong scene."""
    r = fs.passes([x["S - B"] for x in rows])
    ctrl = sum(x["S - S_wrong"]["ci95"][0] > 0 for x in rows)
    ok = r["passes"] and ctrl >= 2
    return {"rule": r, "above_wrong_scene": ctrl, "best": "S" if ok else "B"}


def scene_one(ds, spec, data, enc, B, fake=False, log=print):
    b = fs.Bench(ds, spec, data, enc, B, fake)
    met = b.R.metric
    report = {"dataset": ds, "backbone": TAG, "n_videos": len(data["F"]), "gates": [], "notes": b.notes}
    log("\n" + "=" * 78 + "\n%s on %s  (%d videos, %d units, primary %s)\n" % (
        ds.upper(), TAG, b.R.n, len(b.R.unit_list), spec["metric"]) + "=" * 78)
    normal, _ = fs.phrases({"C"}, b.R.base_n, b.R.base_a)
    N = enc(normal)

    # the scene of each unit, from its calm frames
    scene_emb = enc(list(SCENE_NAMES))
    desc = b.R.describe_units([{"text": t} for t in SCENE_NAMES], b.margins(set()))
    where = {n: i for i, n in enumerate(data["names"])}
    best, worst = {}, {}
    for u, r in desc.items():
        v = fs.rt.camera_vector(data["F"][where[r["video"]]], r["calm"])
        best[u], worst[u] = pick_scenes(v, scene_emb, r["fallback"])
    counts = {s: sum(x == s for x in best.values()) for s in SCENE_NAMES}
    report["scene_counts"] = counts
    report["fallback_units"] = sum(x is None for x in best.values())
    report["units"] = len(best)
    log("  units using the whole bank: %d of %d" % (report["fallback_units"], len(best)))
    for s, c in counts.items():
        log("    %3d  %s" % (c, s))

    def by_unit(chosen):
        return [ns.margins_proto(f, N, enc(bank_for(chosen[u]))) for f, u in zip(data["F"], data["units"])]

    M = {"C": b.margins({"C"}),
         "B": [ns.margins_proto(f, N, enc(ideas.BANK_PHRASES)) for f in data["F"]],
         "S": by_unit(best), "S_wrong": by_unit(worst)}
    S = {k: b.score(m) for k, m in M.items()}

    # ---------------------------------------------------------------- gates
    ok_all = True

    def gate(what, got, ref):
        nonlocal ok_all
        ok = ref is not None and abs(got - ref) <= ns.TOL_PATH
        ok_all &= ok
        report["gates"].append({"what": what, "got": got, "ref": ref, "ok": ok})
        log("  gate  %-34s %.6f  reference %s  %s" % (
            what, got, "none" if ref is None else "%.6f" % ref, "ok" if ok else "MISMATCH"))

    prev = json.load(open(os.path.join(ns.TABLES, "ideas_step3_%s.json" % ds)))["point"]
    gate("B == step 3", met(S["B"]), prev.get("B"))
    gate("C == step 3", met(S["C"]), prev.get("C"))
    z = fs.scores_loader("vitl")(ds)
    same_videos = list(z["names"]) == list(data["names"])
    gate("same videos as the ViT-L run", float(same_videos), 1.0)
    S["old_llava"] = ns.split_by_lengths(z["llava"], z["lengths"])
    if same_videos:
        gate("ViT-L LLaVA == step 3", met(S["old_llava"]), prev.get("old_llava"))
    report["gates_passed"] = bool(ok_all)
    if not ok_all:
        log("  VALIDATION FAILED -- %s is not reported." % ds)
        return report, {}

    # ---------------------------------------------------------------- points
    point = {k: met(v) for k, v in S.items()}
    log("\n  %-16s %8s" % ("condition", spec["metric"]))
    for k, v in point.items():
        log("  %-16s %8.4f" % (k, v))
    report["point"] = point
    if spec["metric"] == "xd_ap":
        report["secondary"] = {
            k: ns.weighted_secondary(np.concatenate(S[k]), np.concatenate(data["wpos"]),
                                     np.concatenate(data["wneg"])) for k in ("B", "S", "S_wrong")}

    # ---------------------------------------------------------------- bootstrap
    log("\n  paired bootstrap: %d resamples of %d videos, seed %d" % (B, b.R.n, SEED))
    need = {k for p in STEP4_PAIRS for k in p}
    draws = ns.paired_bootstrap({k: S[k] for k in need}, met, b.R.n, B, SEED)
    report["comparisons"] = {}
    for x, y in STEP4_PAIRS:
        g = draws[x] - draws[y]
        r = {"point": point[x] - point[y], "ci95": ns.ci95(g), "frac_positive": float(np.mean(g > 0))}
        report["comparisons"]["%s - %s" % (x, y)] = r
        log("  %-22s %+.4f   95%% CI [%+.4f, %+.4f]" % ("%s - %s" % (x, y), r["point"], *r["ci95"]))

    # ---------------------------------------------------------------- audit
    from measurement_audit import split, verdict
    r = split(M["S"], M["B"], data["L"], spec["w"], spec["normalize"])
    r["verdict"] = verdict(r)
    report["audit"] = {"S": r}
    log("  audit S vs B  gap %+.4f  level share %5.2f  macro_m gap %+.4f  ->  %s"
        % (r["gap"], r["level_share"], r["macro_m_gap"], r["verdict"]))
    return report, {k: S[k] for k in ("S", "S_wrong")}


def predictions_step4(reports, decision):
    cmp = lambda d, name: reports[d]["comparisons"][name]             # noqa: E731
    res = {}
    res["K1 scene choice fails the rule against B"] = {"value": decision["rule"], "holds": not decision["rule"]["passes"]}
    v = cmp("nwpu_campus", "S - B")["point"]
    res["K2 scene choice is above B on NWPU"] = {"value": v, "holds": v > 0}
    v = decision["above_wrong_scene"]
    res["K3 the right scene beats the wrong scene, CI above 0, on at least 2"] = {"value": v, "holds": v >= 2}
    v = {d: reports[d]["fallback_units"] / reports[d]["units"] for d in ORDER}
    res["K4 XD falls back to the whole bank more often than any other dataset"] = {
        "value": v, "holds": all(v["xd_violence"] > x for d, x in v.items() if d != "xd_violence")}
    v = cmp("xd_violence", "S - B")["point"]
    res["K5 scene choice lowers XD"] = {"value": v, "holds": v < 0}
    return res


def summarise_step4(args, reports):
    rows = [reports[d]["comparisons"] for d in ORDER]
    decision = choose_scene_rule(rows)
    r = decision["rule"]
    print("\n" + "=" * 78 + "\nSTEP 4 DECISION\n" + "=" * 78)
    print("  S vs B  positive on %d/%d, CI above 0 on %d, CI below 0 on %d  ->  %s"
          % (r["positive"], r["datasets"], r["ci_above_zero"], r["ci_below_zero"], "PASS" if r["passes"] else "fail"))
    print("  S beats the wrong scene, CI above 0, on %d datasets" % decision["above_wrong_scene"])
    print("  CURRENT BEST: PE-Core + C, %s" % decision["best"])
    verdict = compare_with_old([x["S - old_llava"] for x in rows])
    print("\n  S against the old LLaVA model, same videos: %s" % verdict)
    pr = predictions_step4(reports, decision)
    print("\nPRE-REGISTERED PREDICTIONS")
    for k, x in pr.items():
        print("  %-72s %s" % (k, "HOLDS" if x["holds"] else "REFUTED"))
    out = {"manifest": manifest(args), "decision": decision, "old_models": {"S vs old_llava": verdict}, "predictions": pr}
    path = os.path.join(ns.TABLES, "ideas_step4_summary.json")
    json.dump(out, open(path, "w"), indent=1, default=float)
    print("\nsaved %s" % path)
    return out


def save_step4(args, ds, rep, keep, data):
    json.dump({"manifest": manifest(args), **rep},
              open(os.path.join(ns.TABLES, "ideas_step4_%s.json" % ds), "w"), indent=1, default=float)
    if rep["gates_passed"]:
        np.savez(os.path.join(ns.TABLES, "ideas_step4_scores_%s.npz" % ds),
                 names=np.array(data["names"], dtype=object),
                 lengths=np.array([len(x) for x in data["F"]]),
                 **{k: np.concatenate(v).astype(np.float64) for k, v in keep.items()})


def step4(args):
    specs = {d: ns.spec_for(d, TAG) for d in ORDER}
    print("loading %s's text encoder ..." % ns.BACKBONES[TAG][0])
    enc = ns.text_encoder(TAG)
    reports = {}
    for ds in ORDER:
        data = ns.load(ds, specs[ds])
        rep, keep = scene_one(ds, specs[ds], data, enc, args.boot)
        save_step4(args, ds, rep, keep, data)
        reports[ds] = rep
    if not all(r["gates_passed"] for r in reports.values()):
        print("a dataset failed its gates -- no decision")
        return 1
    summarise_step4(args, reports)
    return 0


def selftest_step4(expect):
    expect(all(set(v) <= set(ideas.BANK) and len(v) >= 2 for v in ideas.SCENES.values()),
           "every scene uses at least two real bank types")
    expect(set(ideas.BANK) == {t for v in ideas.SCENES.values() for t in v}, "every bank type belongs to some scene")
    expect(all(set(ideas.scene_phrases(s)) < set(ideas.BANK_PHRASES) for s in ideas.SCENES),
           "each scene uses part of the bank and no new phrase")
    expect(ideas.scene_phrases("a highway with moving vehicles")[:1] == ["two cars crashing into each other"],
           "a scene's phrases come from its own types")
    expect(bank_for(None) == ideas.BANK_PHRASES, "no scene chosen -> the whole bank")
    E = np.eye(len(SCENE_NAMES))
    v = E[2] * 0.9 + E[5] * 0.1 - E[1] * 0.4
    expect(pick_scenes(v, E, False) == (SCENE_NAMES[2], SCENE_NAMES[1]), "picks the closest and the least-matching scene")
    expect(pick_scenes(v, E, True) == (None, None), "a failed self-check picks no scene")

    ok = {"point": 0.02, "ci95": [0.01, 0.03]}
    flat = {"point": 0.01, "ci95": [-0.01, 0.03]}
    bad = {"point": -0.02, "ci95": [-0.03, -0.01]}

    def rows(s, w):
        return [{"S - B": a, "S - S_wrong": b} for a, b in zip(s, w)]
    win = [ok, ok, flat, flat]
    expect(choose_scene_rule(rows(win, win))["best"] == "S", "passes and beats the wrong scene -> S")
    expect(choose_scene_rule(rows(win, [ok, flat, flat, flat]))["best"] == "B", "does not beat the wrong scene -> B stays")
    expect(choose_scene_rule(rows([ok, ok, ok, bad], win))["best"] == "B", "fails the rule -> B stays")

    print("\n  step 4, end to end, on synthetic results in a temporary folder ...")
    tables, logged = ns.TABLES, []
    enc = ns.fake_encoder()
    rng = np.random.default_rng(4)
    with tempfile.TemporaryDirectory() as tmp:
        ns.TABLES = tmp
        try:
            reports = {}
            fargs = SimpleNamespace(step="selftest", boot=20)
            first = {}
            for ds in ORDER:
                spec = ns.spec_for(ds, "vitl")
                data = ns.fake_dataset(ds, spec, rng)
                first[ds] = data
                for t in fs.ENCODERS:
                    fs.save(fargs, ds, t, *_fixes_fixture(ds, spec, data, enc, logged), data)
                rep, keep = papers_one(ds, spec, data, enc, 5, fake=True, log=logged.append)
                save_step3(fargs, ds, rep, keep, data)
                rep, keep = scene_one(ds, spec, data, enc, 20, fake=True, log=logged.append)
                expect(rep["gates_passed"], "%s: every gate passes" % ds)
                expect(set(rep["comparisons"]) == {"%s - %s" % p for p in STEP4_PAIRS}, "%s: all comparisons made" % ds)
                expect(sum(rep["scene_counts"].values()) + rep["fallback_units"] == rep["units"],
                       "%s: every unit has a scene or the whole bank" % ds)
                expect(all(len(x) == len(y) for x, y in zip(keep["S"], data["F"])), "%s: one score per frame" % ds)
                save_step4(fargs, ds, rep, keep, data)
                reports[ds] = rep

            ds = ORDER[0]
            path = os.path.join(tmp, "ideas_step3_%s.json" % ds)
            saved = json.load(open(path))
            saved["point"]["B"] += 0.01
            json.dump(saved, open(path, "w"))
            rep, keep = scene_one(ds, ns.spec_for(ds, "vitl"), first[ds], enc, 5, fake=True, log=logged.append)
            expect(not rep["gates_passed"] and keep == {}, "a mismatched reference stops the dataset")

            out = summarise_step4(fargs, reports)
            expect(out["decision"]["best"] in ("B", "S"), "the decision names a current best")
            expect(len(out["predictions"]) == 5, "all predictions made")
            expect(os.path.exists(os.path.join(tmp, "ideas_step4_summary.json")), "summary saved")
        finally:
            ns.TABLES = tables


# ======================================================================
# step 5: camera memory
# ======================================================================
STEP5_PAIRS = (("M", "B"), ("M", "M_wrong"), ("M", "C"), ("M", "old_llava"))
KNN = 5


def novelty(X, gap, k=KNN):
    """1 - mean similarity to the k nearest earlier frames, leaving out the last gap
    frames. NaN until k earlier frames exist."""
    X = np.asarray(X, dtype=np.float32)
    S = X @ X.T
    out = np.full(len(X), np.nan)
    for t in range(len(X)):
        end = t - gap
        if end >= k:
            out[t] = 1.0 - np.partition(S[t, :end], end - k)[end - k:].mean()
    return out


def novelty_against(X, Y, k=KNN):
    """The same measure against another unit's frames, all of them."""
    S = np.asarray(X, dtype=np.float32) @ np.asarray(Y, dtype=np.float32).T
    k = min(k, S.shape[1])
    return 1.0 - np.partition(S, S.shape[1] - k, axis=1)[:, -k:].mean(axis=1)


def add_memory(m, n):
    """The margin plus the novelty's deviation from its mean, scaled to the margin's
    spread, so each counts equally and the margin's level across units is kept."""
    ok = np.isfinite(n)
    if ok.sum() < 2 or np.std(n[ok]) == 0:
        return np.asarray(m, dtype=float).copy()
    d = np.zeros_like(n, dtype=float)
    d[ok] = (n[ok] - n[ok].mean()) / np.std(n[ok])
    return m + np.std(m) * d


def unit_order(data):
    """unit -> its video indices in name order."""
    out = {}
    for i, u in enumerate(data["units"]):
        out.setdefault(u, []).append(i)
    return {u: sorted(v, key=lambda i: data["names"][i]) for u, v in sorted(out.items())}


def with_memory(data, base, gap, wrong=False):
    """-> (combined margins, novelty alone), one array per video."""
    units = unit_order(data)
    names = list(units)
    M, N = [None] * len(base), [None] * len(base)
    for j, u in enumerate(names):
        idx = units[u]
        X = np.concatenate([data["F"][i] for i in idx])
        if wrong:
            other = units[names[(j + 1) % len(names)]]
            n = novelty_against(X, np.concatenate([data["F"][i] for i in other]))
        else:
            n = novelty(X, gap)
        c = add_memory(np.concatenate([base[i] for i in idx]), n)
        ends = np.cumsum([len(data["F"][i]) for i in idx])[:-1]
        for i, part, q in zip(idx, np.split(c, ends), np.split(n, ends)):
            M[i], N[i] = part, q
    return M, N


def choose_memory(rows):
    """Memory replaces B only if it passes the rule and also beats another unit's memory."""
    r = fs.passes([x["M - B"] for x in rows])
    ctrl = sum(x["M - M_wrong"]["ci95"][0] > 0 for x in rows)
    ok = r["passes"] and ctrl >= 2
    return {"rule": r, "above_wrong_memory": ctrl, "best": "M" if ok else "B"}


def memory_one(ds, spec, data, enc, B, fake=False, log=print):
    b = fs.Bench(ds, spec, data, enc, B, fake)
    met = b.R.metric
    report = {"dataset": ds, "backbone": TAG, "n_videos": len(data["F"]), "gates": [], "notes": b.notes}
    log("\n" + "=" * 78 + "\n%s on %s  (%d videos, %d units, primary %s)\n" % (
        ds.upper(), TAG, b.R.n, len(b.R.unit_list), spec["metric"]) + "=" * 78)
    normal, _ = fs.phrases({"C"}, b.R.base_n, b.R.base_a)
    N, A = enc(normal), enc(ideas.BANK_PHRASES)
    base = [ns.margins_proto(f, N, A) for f in data["F"]]
    own, alone = with_memory(data, base, spec["w"])
    wrong, _ = with_memory(data, base, spec["w"], wrong=True)
    known = np.concatenate(alone)
    report["frames_without_memory"] = float(np.mean(~np.isfinite(known)))
    centred = [np.nan_to_num(x - np.nanmean(known)) for x in alone]
    M = {"C": b.margins({"C"}), "B": base, "M": own, "M_wrong": wrong, "N_only": centred}
    S = {k: b.score(m) for k, m in M.items()}
    log("  frames with no memory yet: %.1f%%" % (100.0 * report["frames_without_memory"]))

    # ---------------------------------------------------------------- gates
    ok_all = True

    def gate(what, got, ref):
        nonlocal ok_all
        ok = ref is not None and abs(got - ref) <= ns.TOL_PATH
        ok_all &= ok
        report["gates"].append({"what": what, "got": got, "ref": ref, "ok": ok})
        log("  gate  %-34s %.6f  reference %s  %s" % (
            what, got, "none" if ref is None else "%.6f" % ref, "ok" if ok else "MISMATCH"))

    prev = json.load(open(os.path.join(ns.TABLES, "ideas_step3_%s.json" % ds)))["point"]
    gate("B == step 3", met(S["B"]), prev.get("B"))
    gate("C == step 3", met(S["C"]), prev.get("C"))
    z = fs.scores_loader("vitl")(ds)
    same_videos = list(z["names"]) == list(data["names"])
    gate("same videos as the ViT-L run", float(same_videos), 1.0)
    S["old_llava"] = ns.split_by_lengths(z["llava"], z["lengths"])
    if same_videos:
        gate("ViT-L LLaVA == step 3", met(S["old_llava"]), prev.get("old_llava"))
    report["gates_passed"] = bool(ok_all)
    if not ok_all:
        log("  VALIDATION FAILED -- %s is not reported." % ds)
        return report, {}

    # ---------------------------------------------------------------- points
    point = {k: met(v) for k, v in S.items()}
    log("\n  %-16s %8s" % ("condition", spec["metric"]))
    for k, v in point.items():
        log("  %-16s %8.4f" % (k, v))
    report["point"] = point
    if spec["metric"] == "xd_ap":
        report["secondary"] = {
            k: ns.weighted_secondary(np.concatenate(S[k]), np.concatenate(data["wpos"]),
                                     np.concatenate(data["wneg"])) for k in ("B", "M", "M_wrong", "N_only")}

    # ---------------------------------------------------------------- bootstrap
    log("\n  paired bootstrap: %d resamples of %d videos, seed %d" % (B, b.R.n, SEED))
    need = {k for p in STEP5_PAIRS for k in p}
    draws = ns.paired_bootstrap({k: S[k] for k in need}, met, b.R.n, B, SEED)
    report["comparisons"] = {}
    for x, y in STEP5_PAIRS:
        g = draws[x] - draws[y]
        r = {"point": point[x] - point[y], "ci95": ns.ci95(g), "frac_positive": float(np.mean(g > 0))}
        report["comparisons"]["%s - %s" % (x, y)] = r
        log("  %-22s %+.4f   95%% CI [%+.4f, %+.4f]" % ("%s - %s" % (x, y), r["point"], *r["ci95"]))

    # ---------------------------------------------------------------- audit
    from measurement_audit import split, verdict
    r = split(M["M"], M["B"], data["L"], spec["w"], spec["normalize"])
    r["verdict"] = verdict(r)
    report["audit"] = {"M": r}
    log("  audit M vs B  gap %+.4f  level share %5.2f  macro_m gap %+.4f  ->  %s"
        % (r["gap"], r["level_share"], r["macro_m_gap"], r["verdict"]))
    return report, {k: S[k] for k in ("M", "M_wrong", "N_only")}


def predictions_step5(reports, decision):
    cmp = lambda d, name: reports[d]["comparisons"][name]             # noqa: E731
    res = {}
    res["L1 camera memory fails the rule against B"] = {"value": decision["rule"], "holds": not decision["rule"]["passes"]}
    v = cmp("nwpu_campus", "M - B")["point"]
    res["L2 camera memory raises NWPU"] = {"value": v, "holds": v > 0}
    v = cmp("xd_violence", "M - B")["point"]
    res["L3 camera memory lowers XD"] = {"value": v, "holds": v < 0}
    v = decision["above_wrong_memory"]
    res["L4 own memory beats another unit's, CI above 0, on at least 2"] = {"value": v, "holds": v >= 2}
    v = {d: reports[d]["point"]["N_only"] - reports[d]["point"]["B"] for d in ORDER}
    res["L5 memory alone is below B on every dataset"] = {"value": v, "holds": all(x < 0 for x in v.values())}
    return res


def summarise_step5(args, reports):
    rows = [reports[d]["comparisons"] for d in ORDER]
    decision = choose_memory(rows)
    r = decision["rule"]
    print("\n" + "=" * 78 + "\nSTEP 5 DECISION\n" + "=" * 78)
    print("  M vs B  positive on %d/%d, CI above 0 on %d, CI below 0 on %d  ->  %s"
          % (r["positive"], r["datasets"], r["ci_above_zero"], r["ci_below_zero"], "PASS" if r["passes"] else "fail"))
    print("  M beats another unit's memory, CI above 0, on %d datasets" % decision["above_wrong_memory"])
    print("  CURRENT BEST: PE-Core + C, %s" % ("B with camera memory" if decision["best"] == "M" else "B"))
    verdict = compare_with_old([x["M - old_llava"] for x in rows])
    print("\n  M against the old LLaVA model, same videos: %s" % verdict)
    pr = predictions_step5(reports, decision)
    print("\nPRE-REGISTERED PREDICTIONS")
    for k, x in pr.items():
        print("  %-68s %s" % (k, "HOLDS" if x["holds"] else "REFUTED"))
    out = {"manifest": manifest(args), "decision": decision, "old_models": {"M vs old_llava": verdict}, "predictions": pr}
    path = os.path.join(ns.TABLES, "ideas_step5_summary.json")
    json.dump(out, open(path, "w"), indent=1, default=float)
    print("\nsaved %s" % path)
    return out


def save_step5(args, ds, rep, keep, data):
    json.dump({"manifest": manifest(args), **rep},
              open(os.path.join(ns.TABLES, "ideas_step5_%s.json" % ds), "w"), indent=1, default=float)
    if rep["gates_passed"]:
        np.savez(os.path.join(ns.TABLES, "ideas_step5_scores_%s.npz" % ds),
                 names=np.array(data["names"], dtype=object),
                 lengths=np.array([len(x) for x in data["F"]]),
                 **{k: np.concatenate(v).astype(np.float64) for k, v in keep.items()})


def step5(args):
    specs = {d: ns.spec_for(d, TAG) for d in ORDER}
    print("loading %s's text encoder ..." % ns.BACKBONES[TAG][0])
    enc = ns.text_encoder(TAG)
    reports = {}
    for ds in ORDER:
        data = ns.load(ds, specs[ds])
        rep, keep = memory_one(ds, specs[ds], data, enc, args.boot)
        save_step5(args, ds, rep, keep, data)
        reports[ds] = rep
    if not all(r["gates_passed"] for r in reports.values()):
        print("a dataset failed its gates -- no decision")
        return 1
    summarise_step5(args, reports)
    return 0


def selftest_step5(expect):
    X = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    n = novelty(X, gap=0, k=2)
    expect(np.isnan(n[:2]).all(), "no score until k earlier frames exist")
    expect(np.allclose(n[2:], [0.0, 1.0, 0.0]), "a frame unlike the past is novel, a familiar one is not")
    n = novelty(X, gap=1, k=2)
    expect(np.isnan(n[:3]).all() and np.allclose(n[3:], [1.0, 0.0]), "the gap leaves out the latest frames")
    Y = np.random.default_rng(0).standard_normal((30, 8))
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    a, c = novelty(Y, 2)[:20], novelty(np.concatenate([Y[:20], Y[:10] * -1.0]), 2)[:20]
    expect(np.allclose(a, c, equal_nan=True), "the memory uses no future frame")
    expect(np.allclose(novelty_against(X[:2], X[3:4]), [1.0, 1.0]), "memory from another unit works")
    expect(novelty_against(X[:2], X[3:], k=9).shape == (2,), "another unit shorter than k is fine")

    m = np.array([0.1, 0.2, 0.3, 0.4])
    expect(np.allclose(add_memory(m, np.full(4, np.nan)), m), "no memory at all -> the margin is unchanged")
    expect(np.allclose(add_memory(m, np.full(4, 0.5)), m), "flat novelty -> the margin is unchanged")
    q = add_memory(m, np.array([np.nan, 0.0, 1.0, 0.5]))
    expect(np.isclose(q[0], m[0]), "a frame with no memory keeps its margin")
    expect(np.isclose(add_memory(m + 5.0, np.array([0.0, 1.0, 2.0, 3.0])).mean(), m.mean() + 5.0),
           "the margin's level across units is kept")
    d = add_memory(m, np.array([1.0, 2.0, 3.0, 4.0])) - m
    expect(np.isclose(np.std(d), np.std(m)), "novelty counts as much as the margin")

    ok = {"point": 0.02, "ci95": [0.01, 0.03]}
    flat = {"point": 0.01, "ci95": [-0.01, 0.03]}
    bad = {"point": -0.02, "ci95": [-0.03, -0.01]}

    def rows(s, w):
        return [{"M - B": a, "M - M_wrong": b} for a, b in zip(s, w)]
    win = [ok, ok, flat, flat]
    expect(choose_memory(rows(win, win))["best"] == "M", "passes and beats another unit's memory -> M")
    expect(choose_memory(rows(win, [ok, flat, flat, flat]))["best"] == "B", "does not beat it -> B stays")
    expect(choose_memory(rows([ok, ok, ok, bad], win))["best"] == "B", "fails the rule -> B stays")

    print("\n  step 5, end to end, on synthetic results in a temporary folder ...")
    tables, logged = ns.TABLES, []
    enc = ns.fake_encoder()
    rng = np.random.default_rng(5)
    with tempfile.TemporaryDirectory() as tmp:
        ns.TABLES = tmp
        try:
            reports = {}
            fargs = SimpleNamespace(step="selftest", boot=20)
            first = {}
            for ds in ORDER:
                spec = ns.spec_for(ds, "vitl")
                data = ns.fake_dataset(ds, spec, rng)
                first[ds] = data
                for t in fs.ENCODERS:
                    fs.save(fargs, ds, t, *_fixes_fixture(ds, spec, data, enc, logged), data)
                rep, keep = papers_one(ds, spec, data, enc, 5, fake=True, log=logged.append)
                save_step3(fargs, ds, rep, keep, data)
                rep, keep = memory_one(ds, spec, data, enc, 20, fake=True, log=logged.append)
                expect(rep["gates_passed"], "%s: every gate passes" % ds)
                expect(set(rep["comparisons"]) == {"%s - %s" % p for p in STEP5_PAIRS}, "%s: all comparisons made" % ds)
                expect(rep["point"]["M"] != rep["point"]["B"], "%s: memory changes the score" % ds)
                expect(all(len(x) == len(y) for x, y in zip(keep["M"], data["F"])), "%s: one score per frame" % ds)
                save_step5(fargs, ds, rep, keep, data)
                reports[ds] = rep

            ds = ORDER[0]
            path = os.path.join(tmp, "ideas_step3_%s.json" % ds)
            saved = json.load(open(path))
            saved["point"]["B"] += 0.01
            json.dump(saved, open(path, "w"))
            rep, keep = memory_one(ds, ns.spec_for(ds, "vitl"), first[ds], enc, 5, fake=True, log=logged.append)
            expect(not rep["gates_passed"] and keep == {}, "a mismatched reference stops the dataset")

            out = summarise_step5(fargs, reports)
            expect(out["decision"]["best"] in ("B", "M"), "the decision names a current best")
            expect(len(out["predictions"]) == 5, "all predictions made")
            expect(os.path.exists(os.path.join(tmp, "ideas_step5_summary.json")), "summary saved")
        finally:
            ns.TABLES = tables


# ======================================================================
# self-test (no data, no GPU)
# ======================================================================
def selftest(args):
    fails = []

    def expect(cond, what):
        print("  %-68s %s" % (what, "ok" if cond else "FAIL"))
        if not cond:
            fails.append(what)

    second = ideas.CALM_SECOND
    expect(len(second) == len(fixes.CALM_NORMAL) == 8, "both wordings have eight phrases")
    expect(not set(second) & set(fixes.CALM_NORMAL), "no phrase in common with the first wording")
    expect(all(banks.is_clean(p) for p in second), "no second-wording phrase contains a banned word")
    nm = {p for v in banks.NEAR_MISS.values() for p in v}
    expect(not nm & set(second), "no second-wording phrase comes from the near-miss lists")

    ok = {"point": 0.02, "ci95": [0.01, 0.03]}
    flat = {"point": 0.01, "ci95": [-0.01, 0.03]}
    low = {"point": -0.01, "ci95": [-0.03, 0.01]}
    bad = {"point": -0.02, "ci95": [-0.03, -0.01]}
    expect(compare_with_old([ok, ok, flat, flat]) == "better", "old models: the rule passes -> better")
    expect(compare_with_old([ok, low, low, flat]) == "no worse", "old models: no CI below 0 -> no worse")
    expect(compare_with_old([ok, ok, ok, bad]) == "worse on 1", "old models: a CI below 0 -> worse")

    def rows(c2, ck, c2k):
        return [{"C2 - base": a, "CK - C": b, "C2K - C2": c} for a, b, c in zip(c2, ck, c2k)]
    good = [ok, ok, flat, flat]
    expect(choose(rows(good, good, good))["best"] == "CK", "wording and keywords pass -> CK")
    expect(choose(rows(good, [ok, flat, flat, flat], good))["best"] == "C", "keywords fail -> C")
    expect(choose(rows(good, good, [low, low, low, flat]))["best"] == "C", "keywords fail on the second wording -> C")
    expect(choose(rows([ok, flat, flat, flat], good, good))["best"] == "base", "wording fails -> plain PE-Core")

    print("\n  step 1, end to end, on synthetic results in a temporary folder ...")
    tables, logged = ns.TABLES, []
    enc = ns.fake_encoder()
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as tmp:
        ns.TABLES = tmp
        try:
            reports = {}
            fargs = SimpleNamespace(step="selftest", boot=20)
            first = {}
            for ds in ORDER:
                spec = ns.spec_for(ds, "vitl")
                data = ns.fake_dataset(ds, spec, rng)
                first[ds] = data
                for t in fs.ENCODERS:
                    rep, keep = fs.evaluate(ds, spec, data, enc, 5, t, fake=True, log=logged.append)
                    fs.save(fargs, ds, t, rep, keep, data)
                rep, keep = check_one(ds, spec, data, enc, 20, fake=True, log=logged.append)
                expect(rep["gates_passed"], "%s: every gate passes" % ds)
                expect(rep["point"]["old_llava"] == json.load(open(os.path.join(
                    tmp, "fixes_%s_vitl.json" % ds)))["point"]["llava"], "%s: the old model is read back exactly" % ds)
                expect(set(rep["comparisons"]) == {"%s - %s" % p for p in STEP1_PAIRS}, "%s: all comparisons made" % ds)
                expect(all(len(x) == len(y) for x, y in zip(keep["C2"], data["F"])), "%s: one score per frame" % ds)
                save(fargs, ds, rep, keep, data)
                reports[ds] = rep

            # a wrong reference must stop the dataset
            ds = ORDER[0]
            path = os.path.join(tmp, "fixes_%s_%s.json" % (ds, TAG))
            saved = json.load(open(path))
            saved["table"]["none|C"] += 0.01
            json.dump(saved, open(path, "w"))
            rep, keep = check_one(ds, ns.spec_for(ds, "vitl"), first[ds], enc, 5, fake=True, log=logged.append)
            expect(not rep["gates_passed"] and keep == {}, "a mismatched reference stops the dataset")

            out = summarise(fargs, reports)
            expect(out["decision"]["best"] in ("base", "C", "CK"), "the decision names a current best")
            expect(len(out["predictions"]) == 5 and len(out["old_models"]) == 6,
                   "all predictions and old-model verdicts made")
            expect(os.path.exists(os.path.join(tmp, "ideas_step1_summary.json")), "summary saved")
        finally:
            ns.TABLES = tables
    selftest_step2(expect)
    selftest_step3(expect)
    selftest_step4(expect)
    selftest_step5(expect)
    print("\n%s" % ("SELF-TEST FAILED: %s" % fails if fails else "self-test passed"))
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["selftest", "step1", "step2", "step3", "step4", "step5"])
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    return {"selftest": selftest, "step1": step1, "step2": step2, "step3": step3, "step4": step4,
            "step5": step5}[args.step](args)


if __name__ == "__main__":
    sys.exit(main())
