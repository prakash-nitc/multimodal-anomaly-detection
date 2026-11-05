# -*- coding: utf-8 -*-
"""The improvement round, one idea at a time, on PE-Core with cached embeddings.
Pre-registered in docs/10_phase3/PREREG_ideas.md; do not change a step's protocol
here without a dated addendum there.

    python notebooks/ideas_sweep.py selftest     # laptop, no data, ~1 min
    python notebooks/ideas_sweep.py step1        # server, after fixes_sweep.py

Step 1 checks the current best before anything is built on it:

    C2    a second wording of the calm normal side
    K     whether the side keywords earn their place on top of C
    old   PE-Core with and without the fixes against the ViT-L models, same videos
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
    print("\n%s" % ("SELF-TEST FAILED: %s" % fails if fails else "self-test passed"))
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["selftest", "step1"])
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    return {"selftest": selftest, "step1": step1}[args.step](args)


if __name__ == "__main__":
    sys.exit(main())
