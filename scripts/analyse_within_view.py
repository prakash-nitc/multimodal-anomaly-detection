# -*- coding: utf-8 -*-
"""Uncertainty on the within-view context gap (Result 5's control).

The paper reports the within-view gap as a mean over nine camera views with the
standard deviation beside it, and qualifies it three ways: the per-view
estimates rest on 5-34 clips each, three of nine are negative, and the spread
exceeds the mean. Those qualifications are stated but never quantified, so the
claim they attach to -- that confining the sweep to one view SHRINKS the gap --
has no interval on it.

This puts one there. Two questions are separable and the paper currently blurs
them:

  (a) Is the within-view gap greater than zero?      -- the weak claim
  (b) Is it smaller than the pooled +0.105?          -- the claim actually made

(b) is what Result 5 needs; (a) is a stronger statement the data may not carry.

Resampling is over VIEWS, not clips: the nine gaps are the observations, and a
view is the unit that would differ if the benchmark had sampled other cameras.
Resampling clips inside a view would answer a different and easier question.

Validation gate: the script recomputes the two figures already published
(mean +0.033, sd 0.058) and refuses to report anything if they do not match, so
its output cannot silently disagree with the paper it is annotating.
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

CSV = "results/runs/analysis/within_view.csv"
POOLED = 0.105          # Table tbl:sweep, normal-ensemble row
AVENUE = 0.020          # Table tbl:avenue, single-view benchmark
PUBLISHED_MEAN, PUBLISHED_SD = 0.033, 0.058   # sd as printed: numpy default, ddof=0
DRAWS = 50000
SEED = 20260911


def load():
    with open(CSV, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    view = np.array([r["view"] for r in rows])
    n = np.array([int(r["n_clips"]) for r in rows])
    gap = np.array([float(r["gap"]) for r in rows])
    none = np.array([float(r["none"]) for r in rows])
    # The CSV carries the gap; recompute it from its parts so a stale or
    # hand-edited column cannot pass unnoticed.
    recomputed = np.array([float(r["matched"]) - float(r["mismatched"]) for r in rows])
    assert np.allclose(gap, recomputed, atol=5e-4), "gap column disagrees with matched-mismatched"
    return view, n, gap, none


def boot(x, w, rng, draws=DRAWS):
    """Percentile bootstrap of a (optionally weighted) mean over views."""
    idx = rng.integers(0, len(x), size=(draws, len(x)))
    if w is None:
        return x[idx].mean(axis=1)
    xs, ws = x[idx], w[idx]
    return (xs * ws).sum(axis=1) / ws.sum(axis=1)


def pct(a, lo=2.5, hi=97.5):
    return np.percentile(a, lo), np.percentile(a, hi)


def main() -> int:
    if not os.path.exists(CSV):
        print("missing %s" % CSV)
        return 2
    view, n, gap, none = load()
    rng = np.random.default_rng(SEED)

    mean = gap.mean()
    sd_pop, sd_sample = gap.std(), gap.std(ddof=1)
    print("nine views, gaps from %.4f to %.4f" % (gap.min(), gap.max()))
    print("  mean %+.4f   negative in %d of %d" % (mean, (gap < 0).sum(), len(gap)))
    print("  sd   %.4f (ddof=0, as published)   %.4f (ddof=1, sample)"
          % (sd_pop, sd_sample))

    # --- validation gate ---------------------------------------------------
    # The paper prints numpy's default sd, so that is what the gate checks. The
    # sample sd is the better estimator for nine views drawn from the cameras
    # that might have been installed, and it is the one reported below; the two
    # differ by 0.003 and both exceed the mean, so nothing in the argument turns
    # on the choice. Noted here rather than silently switched.
    if not (abs(mean - PUBLISHED_MEAN) < 5e-4 and abs(sd_pop - PUBLISHED_SD) < 1e-3):
        print("  VALIDATION FAILED: does not reproduce the published %+.3f (sd %.3f)"
              % (PUBLISHED_MEAN, PUBLISHED_SD))
        return 1
    print("  validation: reproduces the published mean and sd")

    # --- (a) is it above zero? --------------------------------------------
    bs = boot(gap, None, rng)
    lo, hi = pct(bs)
    print("\n(a) within-view gap vs zero")
    print("      mean %+.4f   95%% CI [%+.4f, %+.4f]" % (mean, lo, hi))
    print("      P(mean gap <= 0) = %.3f" % (bs <= 0).mean())
    print("      -> %s" % ("above zero" if lo > 0 else
                           "NOT separable from zero at 95%"))

    # --- (b) is it below the pooled figure? -------------------------------
    print("\n(b) within-view gap vs the pooled +%.3f  <- the claim Result 5 makes" % POOLED)
    print("      P(mean gap >= pooled) = %.4f" % (bs >= POOLED).mean())
    print("      -> %s" % ("SHRINKS, and the CI excludes the pooled value"
                           if hi < POOLED else "not separable from pooled"))

    # --- clip-weighted, since views carry 5 to 34 clips --------------------
    wmean = float((gap * n).sum() / n.sum())
    bw = boot(gap, n.astype(float), rng)
    wlo, whi = pct(bw)
    print("\nclip-weighted (views carry %d-%d clips)" % (n.min(), n.max()))
    print("      mean %+.4f   95%% CI [%+.4f, %+.4f]" % (wmean, wlo, whi))

    # --- is the effect an artefact of small views? ------------------------
    r_n = float(np.corrcoef(n, gap)[0, 1])
    r_d = float(np.corrcoef(none, gap)[0, 1])
    print("\ndiagnostics")
    print("      corr(gap, n_clips)       %+.3f  %s"
          % (r_n, "(small views are not driving it)" if abs(r_n) < 0.5
             else "(WARNING: gap tracks view size)"))
    print("      corr(gap, view accuracy) %+.3f" % r_d)
    print("      Avenue, single view:     %+.4f  %s"
          % (AVENUE, "inside the CI" if lo <= AVENUE <= hi else "outside the CI"))

    print("\nwhat this licenses")
    print("      Result 5's claim is (b), and (b) holds: %s."
          % ("the gap is measurably smaller within a view" if hi < POOLED
             else "NOT established"))
    print("      (a) is %s, so do not claim the within-view effect is"
          % ("supported" if lo > 0 else "not supported"))
    print("      reliably positive -- the paper already declines to, and this is why.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
