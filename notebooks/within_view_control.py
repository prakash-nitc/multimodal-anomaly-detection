# -*- coding: utf-8 -*-
"""Does verbalized context help only where there are several scenes to tell apart?

The context sweep produces a large matched-minus-mismatched gap on ShanghaiTech
(+0.105) and almost none on CUHK Avenue (+0.020). The proposed explanation is
scene diversity: ShanghaiTech spans thirteen camera views, so a scene descriptor
has a discrimination to perform, whereas Avenue is a single fixed view where the
base prompts already cover the only environment present.

That explanation is a conjecture drawn from two datasets. This script tests it
without needing a third, by exploiting the fact that ShanghaiTech is itself
thirteen single-view datasets stacked together. Clip names carry the view in
their prefix (``01_0014`` is view 01), so the same sweep can be evaluated inside
each view separately.

The prediction is sharp. If the effect comes from the descriptor resolving which
environment the model is in, the gap should collapse when only one environment
is present, approaching Avenue's figure. If instead the gap survives within
single views, the scene-diversity account is wrong and the difference between
the two benchmarks needs another explanation.

Scores are read from the experiment driver's own cache rather than recomputed
from embeddings. The driver is the artefact whose numbers the paper reports, and
re-deriving them through a second code path is how an earlier analysis error was
introduced.

Usage:
    python notebooks/within_view_control.py --raw ~/dazvad/work/raw --window 31
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from da_zvad import evaluation                    # noqa: E402
from da_zvad.temporal import moving_average       # noqa: E402

CONDITIONS = ("none", "generic", "matched", "mismatched")


def load_condition(raw_dir: str, dataset: str, domain: str, mode: str,
                   cond: str):
    """-> (names, scores, labels) from the driver's cached npz for one condition."""
    pat = os.path.join(raw_dir, f"{dataset}_{domain}_{cond}_{mode}_*.npz")
    hits = sorted(glob.glob(pat))
    if not hits:
        raise FileNotFoundError(f"no cache matching {pat}")
    z = np.load(hits[-1], allow_pickle=True)
    n = int(z["n"])
    return (list(z["names"]),
            [z[f"s{i}"] for i in range(n)],
            [z[f"l{i}"] for i in range(n)])


def view_of(name: str) -> str:
    """'shanghaitech/01_0014' -> '01'. The prefix is the camera view."""
    return os.path.basename(str(name)).split("_")[0]


def auroc(scores, labels, window: int) -> float:
    return evaluation.pooled_auroc([moving_average(s, window) for s in scores],
                                   labels, normalize=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default=os.path.expanduser("~/dazvad/work/raw"))
    p.add_argument("--dataset", default="shanghaitech")
    p.add_argument("--domain", default="surveillance")
    p.add_argument("--mode", default="normal")
    p.add_argument("--window", type=int, default=31)
    p.add_argument("--min-clips", type=int, default=4,
                   help="skip views with fewer scorable clips than this")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    data = {}
    for c in CONDITIONS:
        data[c] = load_condition(os.path.expanduser(args.raw), args.dataset,
                                 args.domain, args.mode, c)
    names = data["none"][0]
    for c in CONDITIONS:
        if data[c][0] != names:
            print(f"clip order differs between 'none' and {c!r}", file=sys.stderr)
            return 2

    views = {}
    for i, nm in enumerate(names):
        views.setdefault(view_of(nm), []).append(i)

    print(f"{args.dataset}: {len(names)} clips across {len(views)} views, "
          f"window {args.window}\n")

    rows = []
    hdr = f"{'view':<8}{'clips':>6}" + "".join(f"{c:>12}" for c in CONDITIONS) + f"{'gap':>10}"
    print(hdr)
    print("-" * len(hdr))

    for v in sorted(views):
        idx = views[v]
        # a view can only be scored if its pooled labels contain both classes
        lab = np.concatenate([data["none"][2][i] for i in idx])
        if len(idx) < args.min_clips or len(np.unique(lab)) < 2:
            print(f"{v:<8}{len(idx):>6}   skipped (too few clips or one class)")
            continue
        vals = {}
        for c in CONDITIONS:
            _, sc, lb = data[c]
            vals[c] = auroc([sc[i] for i in idx], [lb[i] for i in idx], args.window)
        gap = vals["matched"] - vals["mismatched"]
        rows.append({"view": v, "n_clips": len(idx),
                     **{c: round(vals[c], 4) for c in CONDITIONS},
                     "gap": round(gap, 4)})
        print(f"{v:<8}{len(idx):>6}" + "".join(f"{vals[c]:>12.4f}" for c in CONDITIONS)
              + f"{gap:>+10.4f}")

    # pooled across every view, which is what the paper reports
    allidx = list(range(len(names)))
    pooled = {c: auroc(data[c][1], data[c][2], args.window) for c in CONDITIONS}
    pgap = pooled["matched"] - pooled["mismatched"]
    print("-" * len(hdr))
    print(f"{'POOLED':<8}{len(allidx):>6}"
          + "".join(f"{pooled[c]:>12.4f}" for c in CONDITIONS) + f"{pgap:>+10.4f}")

    if rows:
        gaps = np.array([r["gap"] for r in rows])
        print()
        print(f"mean within-view gap : {gaps.mean():+.4f}  "
              f"(sd {gaps.std():.4f}, n={len(gaps)} views)")
        print(f"pooled gap           : {pgap:+.4f}")
        print()
        print("The scene-diversity account predicts the within-view mean to be")
        print("substantially the smaller of the two, approaching the +0.020")
        print("measured on single-view CUHK Avenue. A within-view gap comparable")
        print("to the pooled one refutes it, and the difference between the two")
        print("benchmarks then requires another explanation.")

    if args.out and rows:
        import csv
        with open(os.path.expanduser(args.out), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
