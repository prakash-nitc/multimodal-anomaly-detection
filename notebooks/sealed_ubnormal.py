# -*- coding: utf-8 -*-
"""The single sealed test of the frozen configuration on UBnormal.
Pre-registered in docs/10_phase3/PREREG_sealed_ubnormal.md (frozen at commit 3792b86).

    python notebooks/sealed_ubnormal.py selftest                 # laptop, no data
    python notebooks/sealed_ubnormal.py gates                    # server, before scoring
    python notebooks/sealed_ubnormal.py score --root ~/dazvad/data/ubnormal   # server, ONCE

``gates`` checks that the frozen files are unchanged and that the frozen configuration
reproduces its step-3 scores on the four development benchmarks. ``score`` refuses to run
if those checks fail, and refuses to run a second time unless given ``--technical-rerun``
with a reason, which is written into the output for the dated addendum. A rerun is for a
technical failure only, never because of the score.

Conditions, all on the same test videos and with no sentence:
    B          the frozen configuration: PE-Core, base + calm normal phrases, the 48-action bank
    C          PE-Core, base + calm normal phrases, the base abnormal phrases
    pe_plain   PE-Core, base phrases only
    vit_plain  CLIP ViT-L/14, base phrases only
"""
from __future__ import annotations

import argparse
import hashlib
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
from da_zvad.datasets import ubnormal as ub                       # noqa: E402
from da_zvad.prompts import fixes, ideas                          # noqa: E402
from da_zvad.prompts.templates import get_prompts                 # noqa: E402

SEED, BOOT, W, LAM = 20261104, 1000, 5, ns.LAM
EMB = "~/dazvad/work/embeddings/ubnormal_%s_step16_crops1.npz"
OUT = os.path.join(ns.TABLES, "sealed_ubnormal.json")

# sha256 (first 16 hex digits) of each frozen file with line endings normalised to LF, as
# committed. The freeze note's table was computed on a Windows working copy; there
# templates.py has CRLF endings (5a6456c8bf24bca5), and its committed content hashes as below.
FROZEN = {
    "da_zvad/prompts/ideas.py": "d889dbc6a76a4410",
    "da_zvad/prompts/fixes.py": "32ab541b436c77f5",
    "da_zvad/prompts/banks.py": "0ab005d55f05a5a8",
    "da_zvad/prompts/templates.py": "4de350b2382843e4",
    "notebooks/ideas_sweep.py": "d0ee71aca2ad9c5b",
    "notebooks/fixes_sweep.py": "5efe7532daafff5c",
    "notebooks/normal_side_sweep.py": "d41be068c7dd1b4b",
}
DEV = ("shanghaitech", "ucf_crime", "xd_violence", "nwpu_campus")


# ======================================================================
# the frozen configuration
# ======================================================================
def phrase_sets():
    base_n, base_a = get_prompts("surveillance")
    calm_n = list(base_n) + list(fixes.CALM_NORMAL)
    return {"B": (calm_n, list(ideas.BANK_PHRASES)), "C": (calm_n, list(base_a)),
            "plain": (list(base_n), list(base_a))}


def margins(feats, enc, which):
    n, a = phrase_sets()[which]
    N, A = enc(n), enc(a)
    return [ns.margins_proto(f, N, A) for f in feats]


def scores(feats, enc, which, w=W, normalize=False):
    return ns.processed(margins(feats, enc, which), w, normalize)


def norm_sha(path):
    b = open(path, "rb").read().replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()[:16]


def check_hashes(log=print):
    ok = True
    for rel, want in FROZEN.items():
        got = norm_sha(os.path.join(REPO, rel))
        good = got == want
        ok &= good
        log("  hash  %-34s %s  %s" % (rel, got, "ok" if good else "CHANGED (frozen %s)" % want))
    return ok


# ======================================================================
# metrics, as the authors' compute_auc_score.py
# ======================================================================
def micro_auc(scores_full, gts):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.concatenate(gts), np.concatenate(scores_full)))


def macro_auc(scores_full, gts):
    """Mean per-video AUROC, each video padded with one 0 and one 1 as the authors do."""
    from sklearn.metrics import roc_auc_score
    vals = [roc_auc_score(np.r_[0, g, 1], np.r_[0.0, s, 1.0]) for s, g in zip(scores_full, gts)]
    return float(np.mean(vals))


def bootstrap(full, gts, names, B=BOOT, seed=SEED):
    """Paired video-level resamples of the primary metric for every named condition."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    n = len(gts)
    draws = {k: [] for k in names}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        y = np.concatenate([gts[i] for i in idx])
        if y.min() == y.max():
            continue
        for k in names:
            draws[k].append(roc_auc_score(y, np.concatenate([full[k][i] for i in idx])))
    return {k: np.asarray(v) for k, v in draws.items()}


def ci95(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


# ======================================================================
# the wording fixed in the freeze note
# ======================================================================
def detection_words(ci):
    if ci[0] > 0.5:
        return "Detects anomalies on an unseen dataset, without training"
    return "Did not detect anomalies reliably on UBnormal"


def transfer_words(ci):
    if ci[0] > 0:
        return "The calm phrases and bank carried over to unseen data"
    if ci[1] < 0:
        return "The calm phrases and bank hurt on unseen data"
    return "Unseen data did not show a gain from the calm phrases and bank"


def old_model_words(ci, point):
    """The step-1 rule applied literally to one benchmark: positive on more than half, CI
    above 0 on at least two, below 0 on none. With one benchmark "better" cannot be reached."""
    if ci[1] < 0:
        return "worse"
    better = point > 0 and ci[0] > 0 and False      # needs a second benchmark
    return "better" if better else "no worse"


# ======================================================================
# gates
# ======================================================================
def dev_gates(enc, log=print):
    ok = True
    for ds in DEV:
        spec = ns.spec_for(ds, "pe-l")
        data = ns.load(ds, spec)
        met = ns.Metric(spec["metric"], data["L"], data.get("wpos"), data.get("wneg"))
        got = met(scores(data["F"], enc, "B", spec["w"], spec["normalize"]))
        ref = json.load(open(os.path.join(ns.TABLES, "ideas_step3_%s.json" % ds)))["point"]["B"]
        good = abs(got - ref) <= ns.TOL_PATH
        ok &= good
        log("  gate  B on %-14s %.6f  step 3 %.6f  %s" % (ds, got, ref, "ok" if good else "MISMATCH"))
    return ok


def gates(args):
    print("frozen files")
    ok = check_hashes()
    print("frozen configuration on the development benchmarks")
    ok &= dev_gates(ns.text_encoder("pe-l"))
    print("\nGATES %s" % ("PASSED" if ok else "FAILED -- the sealed test must not run"))
    return 0 if ok else 1


# ======================================================================
# the sealed run
# ======================================================================
def evaluate(F_pe, F_vit, names, labels_sampled, data, enc_pe, enc_vit, B=BOOT, log=print):
    """data: object with frame_count(video) and full_gt(video). Returns the report."""
    rep = {"n_videos": len(names), "gates": []}
    videos = [n.split("/", 1)[1] for n in names]
    gts = [data.full_gt(v) for v in videos]
    # the sampled labels in the cache must be the full labels at frames 0, 16, 32, ...
    agree = all(np.array_equal(g[ub.STEP * np.arange(len(l))], l) for g, l in zip(gts, labels_sampled))
    rep["gates"].append({"what": "cached sampled labels == full labels at sampled frames", "ok": bool(agree)})
    log("  gate  sampled labels agree with the frame-level labels  %s" % ("ok" if agree else "MISMATCH"))
    if not agree:
        rep["gates_passed"] = False
        return rep
    rep["gates_passed"] = True
    rep["frames"] = int(sum(len(g) for g in gts))
    rep["anomalous_frames"] = int(sum(int(g.sum()) for g in gts))

    sampled = {"B": scores(F_pe, enc_pe, "B"), "C": scores(F_pe, enc_pe, "C"),
               "pe_plain": scores(F_pe, enc_pe, "plain"), "vit_plain": scores(F_vit, enc_vit, "plain")}
    full = {k: [ub.hold_scores(s, len(g)) for s, g in zip(v, gts)] for k, v in sampled.items()}

    rep["primary_micro_auroc"] = {k: micro_auc(full[k], gts) for k in full}
    rep["macro_auroc"] = {k: macro_auc(full[k], gts) for k in full}
    met = ns.Metric("auroc", labels_sampled)
    rep["sampled_micro_auroc"] = {k: met(sampled[k]) for k in sampled}
    log("\n  %-10s %10s %10s %10s" % ("condition", "micro", "macro", "sampled"))
    for k in full:
        log("  %-10s %10.4f %10.4f %10.4f" % (k, rep["primary_micro_auroc"][k], rep["macro_auroc"][k],
                                              rep["sampled_micro_auroc"][k]))

    draws = bootstrap(full, gts, list(full), B=B)
    P = rep["primary_micro_auroc"]
    rep["ci95"] = {k: ci95(v) for k, v in draws.items()}
    rep["comparisons"] = {}
    for x, y in (("B", "pe_plain"), ("B", "vit_plain"), ("B", "C"), ("C", "pe_plain")):
        g = draws[x] - draws[y]
        rep["comparisons"]["%s - %s" % (x, y)] = {"point": P[x] - P[y], "ci95": ci95(g)}
    log("\n  paired bootstrap: %d resamples of %d videos, seed %d" % (B, len(names), SEED))
    log("  B micro AUROC %.4f  95%% CI [%.4f, %.4f]" % (P["B"], *rep["ci95"]["B"]))
    for k, r in rep["comparisons"].items():
        log("  %-16s %+.4f  95%% CI [%+.4f, %+.4f]" % (k, r["point"], *r["ci95"]))

    c = rep["comparisons"]
    rep["wording"] = {
        "detection": detection_words(rep["ci95"]["B"]),
        "transfer": transfer_words(c["B - pe_plain"]["ci95"]),
        "against plain ViT-L": old_model_words(c["B - vit_plain"]["ci95"], c["B - vit_plain"]["point"]),
    }
    rep["predictions"] = {
        "U1 B's 95% CI lies above 0.5": rep["ci95"]["B"][0] > 0.5,
        "U2 B - plain PE-Core is above 0": c["B - pe_plain"]["point"] > 0,
        "U3 B's micro AUROC is between 0.60 and 0.75": 0.60 <= P["B"] <= 0.75,
    }
    log("\nWHAT THE RESULT ALLOWS US TO SAY (fixed in the freeze note)")
    for k, v in rep["wording"].items():
        log("  %-20s %s" % (k, v))
    log("  (with one benchmark, the step-1 rule cannot return \"better\"; it is applied literally)")
    log("\nPRE-REGISTERED PREDICTIONS")
    for k, v in rep["predictions"].items():
        log("  %-50s %s" % (k, "HOLDS" if v else "REFUTED"))
    return rep


def score(args):
    if os.path.exists(OUT) and not args.technical_rerun:
        print("%s already exists. The sealed test runs once; a rerun needs --technical-rerun "
              "\"<reason>\" and is for a technical failure only." % OUT)
        return 1
    print("frozen files")
    if not check_hashes():
        print("a frozen file changed -- the sealed test must not run")
        return 1
    enc_pe, enc_vit = ns.text_encoder("pe-l"), ns.text_encoder("vitl")
    print("frozen configuration on the development benchmarks")
    if not dev_gates(enc_pe):
        print("development gates failed -- the sealed test must not run")
        return 1
    from measurement_audit import load_cache
    data = ub.UBnormalDataset(args.root)
    counts = data.check_files()
    print("\nUBnormal test split: %d abnormal + %d normal videos" % (counts["abnormal"], counts["normal"]))
    F_pe, L_pe, names = load_cache(EMB % "PE-Core-L-14-336")
    F_vit, _L, names_v = load_cache(EMB % "ViT-L-14")
    if names != names_v or len(names) != counts["abnormal"] + counts["normal"]:
        print("the two caches do not hold the same %d test videos" % (counts["abnormal"] + counts["normal"]))
        return 1
    rep = evaluate(F_pe, F_vit, names, L_pe, data, enc_pe, enc_vit)
    rep["manifest"] = ns.run_manifest(args)
    rep["manifest"]["frozen_sha256_lf"] = {k: norm_sha(os.path.join(REPO, k)) for k in FROZEN}
    rep["technical_rerun"] = args.technical_rerun
    os.makedirs(ns.TABLES, exist_ok=True)
    json.dump(rep, open(OUT, "w"), indent=1, default=float)
    print("\nsaved %s" % OUT)
    return 0 if rep["gates_passed"] else 1


# ======================================================================
# self-test (no data, no GPU)
# ======================================================================
def selftest(args):
    fails = []

    def expect(cond, what):
        print("  %-70s %s" % (what, "ok" if cond else "FAIL"))
        if not cond:
            fails.append(what)

    expect(ub.scene_of("abnormal_scene_7_scenario_2_fog") == "7", "scene folder from the name, as the authors do")
    t = ub.read_tracks("3,10,14\n")
    expect(t.shape == (1, 3), "a tracks file with one row")
    t2 = ub.read_tracks("3,10,14\n5,20,21\n")
    gt = ub.frame_level_gt(30, t2)
    expect(gt[10] == 1 and gt[14] == 1 and gt[15] == 0 and gt[9] == 0, "a track's end frame is included")
    expect(int(gt.sum()) == 7, "two tracks mark 5 + 2 frames")
    expect(ub.frame_level_gt(12, t2).sum() == 2, "tracks past the frame count are cut off")
    h = ub.hold_scores(np.array([0.1, 0.9]), 40)
    expect(len(h) == 40 and h[15] == 0.1 and h[16] == 0.9 and h[39] == 0.9, "scores held until the next sample")
    expect(len(ub.hold_scores(np.array([0.1, 0.9, 0.3]), 20)) == 20, "held scores are cut to the frame count")

    from sklearn.metrics import roc_auc_score
    s = [np.array([0.1, 0.8, 0.2]), np.array([0.3, 0.3])]
    g = [np.array([0, 1, 0]), np.array([0, 0])]
    want = np.mean([roc_auc_score([0, 0, 1, 0, 1], [0, .1, .8, .2, 1]), roc_auc_score([0, 0, 0, 1], [0, .3, .3, 1])])
    expect(abs(macro_auc(s, g) - want) < 1e-12, "macro AUROC pads each video with a 0 and a 1, as the authors do")
    expect(abs(micro_auc(s, g) - roc_auc_score([0, 1, 0, 0, 0], [.1, .8, .2, .3, .3])) < 1e-12, "micro AUROC pools every frame")

    expect(detection_words([0.55, 0.7]).startswith("Detects"), "CI above 0.5 -> detects")
    expect(detection_words([0.45, 0.7]).startswith("Did not"), "CI including 0.5 -> did not detect reliably")
    expect(transfer_words([0.01, 0.05]).endswith("carried over to unseen data"), "gain CI above 0 -> carried over")
    expect(transfer_words([-0.05, -0.01]).endswith("hurt on unseen data"), "gain CI below 0 -> hurt")
    expect(transfer_words([-0.01, 0.05]).startswith("Unseen data did not show"), "gain CI spanning 0 -> no gain shown")
    expect(old_model_words([0.01, 0.05], 0.03) == "no worse", "one benchmark cannot reach 'better' under the frozen rule")
    expect(old_model_words([-0.05, -0.01], -0.03) == "worse", "a CI below 0 -> worse")

    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "x.py")
        open(p, "wb").write(b"a = 1\r\nb = 2\r\n")
        q = os.path.join(tmp, "y.py")
        open(q, "wb").write(b"a = 1\nb = 2\n")
        expect(norm_sha(p) == norm_sha(q), "file hashes ignore Windows line endings")
    expect(set(FROZEN) == {"da_zvad/prompts/ideas.py", "da_zvad/prompts/fixes.py", "da_zvad/prompts/banks.py",
                           "da_zvad/prompts/templates.py", "notebooks/ideas_sweep.py", "notebooks/fixes_sweep.py",
                           "notebooks/normal_side_sweep.py"}, "the seven files of the freeze note are checked")
    expect(check_hashes(log=lambda *_: None), "the frozen files in this checkout are unchanged")
    sets = phrase_sets()
    expect(len(sets["B"][0]) == 11 and len(sets["B"][1]) == 48, "B is 11 normal and 48 abnormal phrases")
    expect(sets["C"][0] == sets["B"][0] and len(sets["C"][1]) == 3, "C shares B's normal side and keeps 3 abnormal phrases")

    print("\n  the sealed evaluation, end to end, on synthetic videos ...")
    rng = np.random.default_rng(0)
    enc = ns.fake_encoder()
    names, F, Ls, fake = [], [], [], {}
    bank = ns.unit(enc(ideas.BANK_PHRASES).mean(0))

    class Fake:
        def frame_count(self, v):
            return fake[v][0]

        def full_gt(self, v):
            return fake[v][1]
    for k in range(30):
        n_frames = int(rng.integers(200, 400))
        gt = np.zeros(n_frames, dtype=int)
        if k % 2 == 0:
            a = int(rng.integers(40, n_frames - 80))
            gt[a: a + 60] = 1
        n_s = len(range(0, n_frames, ub.STEP))
        lab = gt[ub.STEP * np.arange(n_s)]
        x = ns.unit(rng.standard_normal((n_s, 64)) + 8.0 * lab[:, None] * bank)
        v = "abnormal_scene_%d_scenario_%d" % (k % 5 + 1, k)
        names.append("ubnormal/" + v); F.append(x); Ls.append(lab); fake[v] = (n_frames, gt)
    logged = []
    rep = evaluate(F, F, names, Ls, Fake(), enc, enc, B=50, log=logged.append)
    expect(rep["gates_passed"], "the label agreement gate passes")
    expect(set(rep["primary_micro_auroc"]) == {"B", "C", "pe_plain", "vit_plain"}, "all four conditions scored")
    expect(rep["primary_micro_auroc"]["B"] > 0.7, "a planted anomaly along the bank direction is detected")
    expect(len(rep["predictions"]) == 3 and len(rep["wording"]) == 3, "three predictions and three wording lines")
    bad = [l.copy() for l in Ls]
    bad[0][0] = 1 - bad[0][0]
    rep2 = evaluate(F, F, names, bad, Fake(), enc, enc, B=10, log=logged.append)
    expect(not rep2["gates_passed"], "cached labels that disagree with the frame labels stop the run")

    print("\n%s" % ("SELF-TEST FAILED: %s" % fails if fails else "self-test passed"))
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["selftest", "gates", "score"])
    ap.add_argument("--root", default="~/dazvad/data/ubnormal")
    ap.add_argument("--technical-rerun", default=None, help="reason for a rerun after a technical failure")
    args = ap.parse_args()
    return {"selftest": selftest, "gates": gates, "score": score}[args.step](args)


if __name__ == "__main__":
    sys.exit(main())
