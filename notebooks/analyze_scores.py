# -*- coding: utf-8 -*-
"""Re-analyse cached CLIP scores without re-running the GPU.

``run_grid`` caches raw per-frame scores to ``<work>/raw/*.npz``, so any change
that lives *after* scoring -- a different temporal window, a different pooling
protocol -- can be evaluated in seconds instead of an hour.

What this script answers: how much of DA-ZVAD's near-chance pooled AUROC is a
detection failure, and how much is a *measurement* artefact?

The ShanghaiTech test split spans 12 camera views. A frozen scorer sits at a different
similarity baseline under each, so concatenating raw scores from all clips into
one ranking compares frames across incomparable scales. The standard protocol
(Liu et al., 2018) min-maxes each clip first. That uses no labels and so leaks
nothing -- it just stops the metric from destroying signal the scorer did find.

The macro column is the diagnostic: macro is the mean of per-clip AUROCs, and it
is unaffected by cross-clip offsets. A large macro/micro-raw gap that closes
under normalisation means the scorer was working better than the raw pooled
number suggested.

Usage:
    python notebooks/analyze_scores.py --work ~/dazvad/work
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from da_zvad import evaluation                       # noqa: E402
from da_zvad.temporal import moving_average          # noqa: E402


def load(path: str):
    z = np.load(path, allow_pickle=True)
    n = int(z["n"])
    return ([z[f"s{i}"] for i in range(n)],
            [z[f"l{i}"] for i in range(n)],
            list(z["names"]))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--work", default=os.path.expanduser("~/dazvad/work"))
    p.add_argument("--windows", type=int, nargs="+", default=[1, 5, 9, 15])
    p.add_argument("--out", default=None, help="optional CSV path")
    args = p.parse_args()

    raw_dir = os.path.join(os.path.expanduser(args.work), "raw")
    files = sorted(glob.glob(os.path.join(raw_dir, "*.npz")))
    if not files:
        print(f"no cached scores under {raw_dir}", file=sys.stderr)
        return 1

    rows = []
    print(f"{'variant':<38}{'win':>4}{'micro_raw':>11}{'micro_norm':>12}"
          f"{'macro':>9}{'clips<0.5':>11}")
    print("-" * 85)

    for f in files:
        scores, labels, names = load(f)
        variant = os.path.basename(f)[:-4]
        for w in args.windows:
            sm = [moving_average(s, w) for s in scores]
            micro_raw = evaluation.pooled_auroc(sm, labels, normalize=False)
            micro_norm = evaluation.pooled_auroc(sm, labels, normalize=True)
            per_clip = [evaluation.frame_auroc(s, l) for s, l in zip(sm, labels)
                        if len(np.unique(l)) == 2]
            macro = float(np.mean(per_clip)) if per_clip else float("nan")
            below = int(sum(1 for a in per_clip if a < 0.5))
            print(f"{variant:<38}{w:>4}{micro_raw:>11.4f}{micro_norm:>12.4f}"
                  f"{macro:>9.4f}{below:>7}/{len(per_clip):<4}")
            rows.append({"variant": variant, "window": w,
                         "auroc_micro_raw": round(micro_raw, 4),
                         "auroc_micro_norm": round(micro_norm, 4),
                         "auroc_macro": round(macro, 4),
                         "clips_below_chance": below,
                         "n_clips_scored": len(per_clip)})

    if args.out:
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        print(f"\n-> {args.out}")

    print("\nmicro_raw  : all clips pooled, no normalisation (what we reported)")
    print("micro_norm : per-clip min-max then pooled -- the published protocol")
    print("macro      : mean of per-clip AUROCs (immune to cross-clip offsets)")
    print("clips<0.5  : clips scored worse than chance -- if this is near half,")
    print("             the scorer has no per-clip signal and normalisation")
    print("             will not rescue it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
