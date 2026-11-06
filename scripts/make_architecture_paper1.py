# -*- coding: utf-8 -*-
"""The CLIP-era architecture figure (make_architecture.py), sized for Paper 1.

make_architecture.py draws on an 8.4 x 4.95 in canvas. In the cas-sc paper the
text block is about 6.9 in wide, so the figure was scaled to ~0.82 and its body
text printed at ~5.7 pt -- too small to read. This wrapper draws the identical
diagram on a smaller canvas (same data coordinates, so the layout is unchanged)
while every font keeps its size in points. Printed at text width, labels then
appear at their intended 7-9.6 pt.

Nothing in make_architecture.py changes, so main.tex and the decks keep their
figure; only docs/09_paper/figures/fig_architecture_clip.{pdf,png} is written.

Usage:  python scripts/make_architecture_paper1.py [--canvas-width 7.0]
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import arch_style  # noqa: E402
import make_architecture as ma  # noqa: E402


def main() -> int:
    root = os.path.dirname(HERE)
    p = argparse.ArgumentParser()
    p.add_argument("--canvas-width", type=float, default=8.4,
                   help="drawn width in inches; the figure sits on a landscape page")
    p.add_argument("--assets", default=os.path.join(root, "figure_assets"))
    p.add_argument("--out", default=os.path.join(root, "docs", "09_paper", "figures",
                                                 "fig_architecture_clip.png"))
    a = p.parse_args()

    k = a.canvas_width / ma.W
    original = plt.subplots

    def smaller(*args, **kw):
        if kw.get("figsize") == (ma.W, ma.H):
            kw["figsize"] = (ma.W * k, ma.H * k)
        return original(*args, **kw)

    plt.subplots = smaller
    try:
        with arch_style.bold_print_style():
            print("saved:", ma.build(a.assets, a.out))
    finally:
        plt.subplots = original
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
