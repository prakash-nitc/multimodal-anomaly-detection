# -*- coding: utf-8 -*-
"""Architecture figure for Paper 2 (the frozen final model), in its own visual style.

Paper 1 shows the CLIP + scene-sentence design with dashed panels in teal and rust
(make_architecture.py). Paper 2 shows a different model and should not look like a
copy, so this script draws the verified layout of make_architecture_paper_v3.py with:

  * a different palette  -- indigo encoder, plum text side, teal calm phrases,
                             amber smoothing, deep-red detection;
  * a different panel style -- solid, lightly tinted rounded panels with a filled
                             title tab, instead of dashed outlines on white;
  * no outer gold frame, and Segoe UI instead of DejaVu Sans;
  * the print-size canvas (see make_architecture_paper1.py), so labels print at
    their intended size in the 6.9 in text block.

Neither existing figure script changes. Output: docs/09_paper/figures/fig_architecture_final.{pdf,png}

Usage:  python scripts/make_architecture_paper2.py [--canvas-width 7.8]
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
from matplotlib.patches import FancyBboxPatch  # noqa: E402

import make_architecture as base  # noqa: E402
import make_architecture_paper_v3 as v3  # noqa: E402

# Text colours hold >= ~7:1 on white; *_BG are fills only.
PALETTE = {
    "VIS": "#3A3685", "VIS_BG": "#E9E8F6",      # encoder: indigo
    "CTX": "#7D2E61", "CTX_BG": "#F6E6F0",      # text side: plum
    "CALM": "#0E6660", "CALM_BG": "#E1F2F0",    # calm normal phrases: teal
    "TMP": "#8A5300", "TMP_BG": "#FBEFD9",      # smoothing: amber
    "OUT_BAD": "#8E1B2C", "OUT_OK": "#2F6B2A",  # detection
    "GOLD": "#6B4E00",
}
PANEL_FILL_ALPHA = 0.055


def tinted_container(ax, x, y, w, h, ec, label):
    """Solid rounded panel with a pale fill and a filled title tab (Paper 2 style)."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.06",
                                facecolor=ec, alpha=PANEL_FILL_ALPHA, edgecolor="none", zorder=0.5))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.06",
                                facecolor="none", edgecolor=ec, linewidth=1.15, zorder=1))
    ax.text(x + 0.09, y + h - 0.05, " %s " % label, fontsize=base.FS_PANEL - 0.6, color="#FFFFFF",
            fontweight="bold", va="center", zorder=6,
            bbox=dict(facecolor=ec, edgecolor="none", boxstyle="round,pad=0.25,rounding_size=0.12"))
    base.CONTAINERS.append((x, y, w, h, label))


def main() -> int:
    root = os.path.dirname(HERE)
    p = argparse.ArgumentParser()
    p.add_argument("--canvas-width", type=float, default=7.8)
    p.add_argument("--assets", default=os.path.join(root, "figure_assets"))
    p.add_argument("--out", default=os.path.join(root, "docs", "09_paper", "figures",
                                                 "fig_architecture_final.png"))
    a = p.parse_args()

    for mod in (base, v3):
        for k, v in PALETTE.items():
            if hasattr(mod, k):
                setattr(mod, k, v)
        mod.container = tinted_container
    # Segoe UI has no snowflake glyph; matplotlib falls back along this list per glyph
    # (fallback works across font.family entries, not across font.sans-serif ones)
    plt.rcParams["font.family"] = ["Segoe UI", "Segoe UI Symbol", "DejaVu Sans"]

    # drop the outer dashed gold frame that marks Paper 1's figures
    real_rect = v3.Rectangle

    def rect(*args, **kw):
        if str(kw.get("edgecolor", "")).upper() == "#B39A3C":
            kw["edgecolor"] = "none"
        return real_rect(*args, **kw)

    v3.Rectangle = rect

    k = a.canvas_width / base.W
    original = plt.subplots

    def smaller(*args, **kw):
        if kw.get("figsize") == (base.W, base.H):
            kw["figsize"] = (base.W * k, base.H * k)
        return original(*args, **kw)

    plt.subplots = smaller
    try:
        print("saved:", v3.build(a.assets, a.out))
    finally:
        plt.subplots = original
        v3.Rectangle = real_rect
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
