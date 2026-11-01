# -*- coding: utf-8 -*-
"""Two follow-up checks on the UCF-Crime result (dated addendum material).

These are NOT part of the UCF-Crime pre-registration; they are later, separately
reported checks. Neither changes a reported number.

CHECK 1 -- Is the per-video sentence acting as a scene-type prior?
------------------------------------------------------------------
Raw pooled AUROC rewards anything that lowers whole normal videos relative to
anomalous ones. A caption describes a setting, and settings may correlate with
which videos contain a crime. If so, the gain would come from separating normal
videos from anomalous videos, not from finding the anomalous moment. Measured,
for per-video sentences (A) against no sentence (B):

  shift AUROC     does the video's mean margin change d predict whether the video
                  is anomalous? 0.5 = no relation. Repeated with shuffled captions.
  gain vs videos  AUROC gain: anomalous frames vs frames of NORMAL videos.
  gain within     AUROC gain: anomalous frames vs normal frames of the ANOMALOUS
                  videos, pooled across those videos. Correction (1 Nov 2026, after
                  the first run): this removes normal videos, but it still compares
                  frames across different anomalous videos with different captions,
                  so cross-video calibration can still move it. It is not a purely
                  within-video measure; that is macro_m in measurement_audit.py. The
                  decision rule is left as committed. Its "NOT NEEDED" outcome is
                  weaker than first described; it did not fire on the real data.

DECISION RULE -- fixed before any of these numbers was computed:
  SCENE-PRIOR SUPPORTED    shift AUROC 95% CI above 0.5, gain vs videos CI above
                           0, and gain within CI not above 0
  SCENE-PRIOR NOT NEEDED   gain within CI above 0
  INCONCLUSIVE             otherwise
Intervals: paired video-level bootstrap, 1,000 resamples, seed 20261101.

CHECK 2 -- Does 0.824 depend on scoring only every 16th frame?
--------------------------------------------------------------
Each sampled frame's smoothed score is held for its 16-frame snippet and AUROC is
computed over every decoded frame, against labels on every frame. No prediction is
registered; the result is reported whatever it shows.

Gate: the cache must first reproduce the published sampled-frame figures.
CPU only, about 10 minutes.

    python notebooks/ucf_followups.py --selftest
    python notebooks/ucf_followups.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from da_zvad import evaluation                                    # noqa: E402
from measurement_audit import (TOL_PUB, TOL_SAVED, UCF_PUB, auc,  # noqa: E402
                               derangements, load_cache, margins_for, processed)

W, STEP, SEED, B = 5, 16, 20261101, 1000


# ---------------------------------------------------------------- check 1
def shifts(mA, mB):
    return np.array([a.mean() - b.mean() for a, b in zip(mA, mB)])


def frame_sets(seqs, labels, idx):
    """-> (anomalous frames, normal frames of anomalous videos, frames of normal videos)."""
    pos, nin, nvid = [], [], []
    for i in idx:
        s, y = seqs[i], labels[i]
        if y.max():
            pos.append(s[y == 1])
            nin.append(s[y == 0])
        else:
            nvid.append(s)
    return tuple(np.concatenate(x) if x else np.empty(0) for x in (pos, nin, nvid))


def two_set_auc(pos, neg):
    if not len(pos) or not len(neg):
        return float("nan")
    y = np.r_[np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)]
    return evaluation.frame_auroc(np.r_[pos, neg], y)


def prior_stats(A, B_, labels, d, idx):
    """A, B_ = per-video processed scores; d = per-video mean margin change."""
    pa, ia, na = frame_sets(A, labels, idx)
    pb, ib, nb = frame_sets(B_, labels, idx)
    vy = np.array([int(labels[i].max()) for i in idx])
    return {"shift_auc": evaluation.frame_auroc(d[idx], vy),
            "gain_vs_videos": two_set_auc(pa, na) - two_set_auc(pb, nb),
            "gain_within": two_set_auc(pa, ia) - two_set_auc(pb, ib)}


def prior_bootstrap(A, B_, labels, d, n, seed):
    rng = np.random.default_rng(seed)
    N = len(labels)
    acc = {"shift_auc": [], "gain_vs_videos": [], "gain_within": []}
    for _ in range(n):
        idx = rng.integers(0, N, N)
        kinds = {int(labels[i].max()) for i in idx}
        if len(kinds) < 2:
            continue
        st = prior_stats(A, B_, labels, d, idx)
        for k, v in st.items():
            if np.isfinite(v):
                acc[k].append(v)
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in acc.items()}


def prior_verdict(ci):
    if ci["gain_within"][0] > 0:
        return "SCENE-PRIOR NOT NEEDED"
    if ci["shift_auc"][0] > 0.5 and ci["gain_vs_videos"][0] > 0:
        return "SCENE-PRIOR SUPPORTED"
    return "INCONCLUSIVE"


# ---------------------------------------------------------------- check 2
def hold_to_all_frames(snippet_scores, total, step=STEP):
    """Hold each sampled frame's score for its snippet, out to the decoded frame count."""
    s = np.repeat(np.asarray(snippet_scores, dtype=float), step)[:total]
    return np.r_[s, np.full(total - len(s), s[-1])] if len(s) < total else s


def every_frame_auc(seqs, full_labels):
    return auc([hold_to_all_frames(s, len(y)) for s, y in zip(seqs, full_labels)], full_labels)


def full_frame_labels(names, root):
    """Labels on every decoded frame; the frame count comes from each video's cache sentinel."""
    from da_zvad.datasets.ucf_crime import interval_labels, parse_annotation
    ann = parse_annotation(os.path.join(root, "Temporal_Anomaly_Annotation.txt"))
    out = []
    for n in names:
        stem = n.split("/", 1)[1]
        sentinel = os.path.join(root, f"_frames_step{STEP}", stem, ".complete")
        total = int(open(sentinel).read().strip())
        out.append(interval_labels(total, ann[stem + ".mp4"][1]))
    return out


# ---------------------------------------------------------------- self-test
def selftest() -> int:
    rng = np.random.default_rng(0)
    labels, base = [], []
    for i in range(60):
        n = int(rng.integers(50, 120))
        y = np.zeros(n, dtype=int)
        if i % 2:
            s = int(rng.integers(5, n - 20))
            y[s:s + 12] = 1
        labels.append(y)
        base.append(rng.normal(0, .01) + .01 * rng.standard_normal(n) + .004 * y)
    fails = []

    def expect(ok, msg):
        print("  %-66s %s" % (msg, "ok" if ok else "FAIL"))
        if not ok:
            fails.append(msg)

    def run(mA):
        A, B_ = processed(mA, W, False), processed(base, W, False)
        d = shifts(mA, base)
        return prior_stats(A, B_, labels, d, np.arange(60)), prior_bootstrap(A, B_, labels, d, 200, 1)

    print("a pure scene-type prior: normal videos lowered as blocks")
    prior = [b - (0 if y.max() else .03) for b, y in zip(base, labels)]
    st, ci = run(prior)
    expect(abs(st["gain_within"]) < 1e-12, "gain within anomalous videos is exactly zero")
    expect(st["gain_vs_videos"] > .05, "gain against normal videos is large (%+.3f)" % st["gain_vs_videos"])
    expect(st["shift_auc"] == 1.0, "the shift perfectly predicts the video label")
    expect(prior_verdict(ci) == "SCENE-PRIOR SUPPORTED", "verdict: %s" % prior_verdict(ci))

    print("real within-video improvement: no relation between shift and label")
    better = []
    for b, y in zip(base, labels):
        e = .02 * y
        better.append(b + e - e.mean())
    st, ci = run(better)
    expect(st["gain_within"] > .05, "gain within anomalous videos is large (%+.3f)" % st["gain_within"])
    expect(prior_verdict(ci) == "SCENE-PRIOR NOT NEEDED", "verdict: %s" % prior_verdict(ci))

    print("holding snippet scores on every frame")
    expect(np.array_equal(hold_to_all_frames([1, 2], 32), np.repeat([1., 2.], 16)), "exact multiple of 16")
    expect(np.array_equal(hold_to_all_frames([1, 2], 20), np.r_[np.full(16, 1.), np.full(4, 2.)]), "video ends mid-snippet")
    expect(np.array_equal(hold_to_all_frames([1, 2], 35), np.r_[np.full(16, 1.), np.full(19, 2.)]), "video runs past the last snippet")
    print("\n%s" % ("SELF-TEST PASSED" if not fails else "SELF-TEST FAILED: %s" % fails))
    return 1 if fails else 0


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--boot", type=int, default=B)
    ap.add_argument("--out", default="~/dazvad/work/tables/ucf_followups.json")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    import per_view_descriptors as pv
    import ucf_crime_sweep as u

    feats, labels, names = load_cache(u.CACHE)
    caps = json.load(open(u.CAPS))
    per = [caps[n] for n in names]
    pool = pv.encoder()
    M = {"none": margins_for(feats, pool, lambda i: None),
         "shared": margins_for(feats, pool, lambda i: u.SHARED),
         "mismatched": margins_for(feats, pool, lambda i: u.MISMATCHED),
         "per_video": margins_for(feats, pool, lambda i: per[i])}
    shuffled = [margins_for(feats, pool, lambda i, p=p: per[p[i]])
                for p in derangements(len(names), u.SEED, u.N_SHUFFLE)]
    P = {k: processed(m, W, False) for k, m in M.items()}

    # gate: the published sampled-frame numbers, or stop
    saved = json.load(open(u.OUT))["results"] if os.path.exists(u.OUT) else None
    ok = True
    print("validation (sampled frames, raw, w=5)")
    for c in M:
        got = auc(P[c], labels)
        ref, tol = (saved[c]["raw_w5"], TOL_SAVED) if saved else (UCF_PUB[c][0], TOL_PUB)
        good = abs(got - ref) <= tol
        ok &= good
        print("  %-11s %.4f  ref %.4f  %s" % (c, got, ref, "ok" if good else "MISMATCH"))
    if not ok:
        print("VALIDATION FAILED -- nothing further reported.")
        return 1

    # check 1
    idx = np.arange(len(labels))
    d = shifts(M["per_video"], M["none"])
    vy = np.array([int(y.max()) for y in labels])
    st = prior_stats(P["per_video"], P["none"], labels, d, idx)
    ci = prior_bootstrap(P["per_video"], P["none"], labels, d, args.boot, SEED)
    shuf_auc = [evaluation.frame_auroc(shifts(m, M["none"]), vy) for m in shuffled]
    verdict = prior_verdict(ci)
    print("\nCHECK 1 -- per-video sentence vs none")
    print("  mean shift: anomalous videos %+.4f, normal videos %+.4f"
          % (d[vy == 1].mean(), d[vy == 0].mean()))
    print("  shift AUROC (predicts video label)  %.3f  95%% CI [%.3f, %.3f]" % (st["shift_auc"], *ci["shift_auc"]))
    print("    same with shuffled captions: mean %.3f, range %.3f to %.3f"
          % (np.mean(shuf_auc), np.min(shuf_auc), np.max(shuf_auc)))
    print("  gain vs normal videos   %+.4f  95%% CI [%+.4f, %+.4f]" % (st["gain_vs_videos"], *ci["gain_vs_videos"]))
    print("  gain within anomalous   %+.4f  95%% CI [%+.4f, %+.4f]" % (st["gain_within"], *ci["gain_within"]))
    print("  VERDICT: %s" % verdict)

    # check 2
    print("\nCHECK 2 -- every decoded frame, scores held per snippet")
    full = full_frame_labels(names, os.path.expanduser(u.ROOT))
    print("  %d frames, %.1f%% anomalous" % (sum(len(y) for y in full), 100 * np.concatenate(full).mean()))
    every = {c: every_frame_auc(P[c], full) for c in M}
    every["shuffled_mean"] = float(np.mean([every_frame_auc(processed(m, W, False), full) for m in shuffled]))
    sampled = {c: auc(P[c], labels) for c in M}
    for c, v in every.items():
        print("  %-14s every frame %.4f   sampled %s" % (c, v, "%.4f" % sampled[c] if c in sampled else "  --  "))

    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"check1": {"point": st, "ci95": ci, "verdict": verdict,
                          "mean_shift_anomalous": float(d[vy == 1].mean()),
                          "mean_shift_normal": float(d[vy == 0].mean()),
                          "shuffled_shift_auc": shuf_auc,
                          "per_video_shift": dict(zip(names, map(float, d)))},
               "check2": {"every_frame": every, "sampled": sampled},
               "seed": SEED, "n_resamples": args.boot}, open(out, "w"), indent=1, default=float)
    print("\nsaved %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
