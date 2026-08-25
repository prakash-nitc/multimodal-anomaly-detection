# -*- coding: utf-8 -*-
"""DA-ZVAD architecture diagram, drawn to a supervisor-supplied reference style.

The previous diagram was five labelled boxes in a row. The reference carries
considerably more: real video frames at input and output, dashed colour-coded
containers grouping each branch, a snowflake on every frozen module, the actual
text the system is given rather than a description of it, mathematical notation
for every intermediate quantity, and different shapes for different kinds of
object.

Drawn at single-column width. The earlier version was authored wide and then
scaled into a 6.3in column, which shrank every label past legibility; laying the
flow out vertically instead keeps the type at its designed size in print.

Layout is banded, with every element placed in the axes' data coordinates so
positions can be reasoned about directly in inches. The embedded score plot uses
``ax.inset_axes(..., transform=ax.transData)`` for the same reason: an earlier
version placed it in figure fractions while everything around it was in data
coordinates, and it landed across three other bands.

Usage:  python scripts/make_architecture_v2.py [--assets DIR] [--out PATH]
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# ---------------------------------------------------------------- palette
INK = "#141C1E"
MUTED = "#5F6E70"
VIS, VIS_BG = "#0D6B67", "#E4F2F0"          # M1 visual
CTX, CTX_BG = "#B4571A", "#FBEDE1"          # M3 verbalised context
TMP, TMP_BG = "#3B5BA5", "#E6EBF7"          # M2 temporal
RSN, RSN_BG = "#6B4C9A", "#EFEAF7"          # M4 reasoning
OUT_OK, OUT_BAD = "#1E8449", "#B03A2E"
PAPER = "#FFFFFF"

W, H = 6.9, 9.4
FROZEN = "❄"

# band boundaries, in inches
A_Y, A_H = 8.05, 1.18          # input
B_Y, B_H = 5.70, 2.15          # M1 / M3
C_Y, C_H = 4.10, 1.42          # scoring
D_Y, D_H = 2.42, 1.50          # M2
E_Y, E_H = 0.42, 1.82          # detection / M4


def box(ax, x, y, w, h, fc, ec, lw=1.1, r=0.05, z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0.012,rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))


def container(ax, x, y, w, h, ec, label):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=ec,
                           linewidth=1.0, linestyle=(0, (4, 2.5)), zorder=1))
    ax.text(x + 0.10, y + h - 0.055, label, fontsize=6.6, color=ec,
            fontweight="bold", va="center", zorder=6,
            bbox=dict(facecolor=PAPER, edgecolor="none", pad=1.6))


def arrow(ax, p, q, color=INK, lw=1.1, rad=0.0, z=4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8,
                                 linewidth=lw, color=color, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1, shrinkB=1))


def module(ax, x, y, w, h, title, sub, color, bg, frozen=True, ts=7.2):
    box(ax, x, y, w, h, bg, color, lw=1.3)
    if frozen:
        ax.text(x + 0.085, y + h - 0.09, FROZEN, fontsize=7.5, color=color,
                va="center", ha="center", zorder=5)
    ax.text(x + w / 2, y + h * (0.60 if sub else 0.5), title, fontsize=ts,
            fontweight="bold", color=INK, ha="center", va="center", zorder=5)
    if sub:
        ax.text(x + w / 2, y + h * 0.24, sub, fontsize=6.0, color=MUTED,
                ha="center", va="center", zorder=5, style="italic")


def scroll(ax, x, y, w, h, lines, color, bg, tag):
    box(ax, x, y, w, h, bg, color, lw=1.0, r=0.03)
    ax.text(x + 0.07, y + h - 0.10, tag, fontsize=5.9, color=color,
            fontweight="bold", va="center", zorder=5)
    for i, ln in enumerate(lines):
        ax.text(x + 0.07, y + h - 0.245 - i * 0.115, ln, fontsize=5.3,
                color=INK, va="center", zorder=5, family="monospace")


def cylinder(ax, cx, cy, w, h, label, color, bg, fs=7.4):
    box(ax, cx - w / 2, cy - h / 2, w, h, bg, color, lw=1.2, r=0.05, z=4)
    ax.text(cx, cy, label, fontsize=fs, fontweight="bold", color=INK,
            ha="center", va="center", zorder=5)


def frame_img(ax, path, cx, cy, zoom, border=None, lw=1.6):
    im = OffsetImage(plt.imread(path), zoom=zoom)
    ax.add_artist(AnnotationBbox(
        im, (cx, cy), frameon=border is not None, pad=0.0, zorder=4,
        bboxprops=dict(edgecolor=border, linewidth=lw) if border else None))


def build(assets: str, out: str) -> str:
    fa = os.path.join(assets, "frames", "shanghaitech")
    normal = os.path.join(fa, "01_0014_normal_00017.jpg")
    anom = os.path.join(fa, "01_0014_anomaly_00095.jpg")
    have = os.path.isfile(normal) and os.path.isfile(anom)

    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    fig.patch.set_facecolor(PAPER)

    ax.add_patch(Rectangle((0.08, 0.08), W - 0.16, H - 0.16, facecolor="none",
                           edgecolor="#C9B458", linewidth=1.0,
                           linestyle=(0, (5, 3)), zorder=0))

    # ======================================================= A. input
    container(ax, 0.18, A_Y, W - 0.36, A_H, MUTED, "INPUT VIDEO")
    if have:
        for dx in (0.0, 0.07, 0.14):
            frame_img(ax, normal, 0.68 + dx, 8.66 + dx * 0.5, 0.108,
                      border="#9AA6A7", lw=0.7)
        frame_img(ax, anom, 2.18, 8.70, 0.126, border="#D4A017", lw=1.9)
    ax.text(0.82, 8.24, "frames  $f_1 \\ldots f_T$", fontsize=6.5, color=MUTED,
            ha="center")
    ax.text(2.18, 8.24, "current frame  $f_t$", fontsize=6.5, color="#9A7500",
            ha="center", fontweight="bold")

    box(ax, 3.32, 8.20, 3.24, 0.86, "#FBFBF9", MUTED, lw=0.9)
    ax.text(3.44, 8.96, f"{FROZEN}  EVERY MODEL FROZEN", fontsize=6.4, color=VIS,
            fontweight="bold", va="center", zorder=5)
    ax.text(4.94, 8.53,
            "No parameter is updated at any stage.\n"
            "Only the sentence $c$ can vary, so any change\n"
            "in output is attributable to the text alone.",
            fontsize=6.1, color=INK, ha="center", va="center", zorder=5,
            linespacing=1.6)

    arrow(ax, (2.18, 8.42), (2.18, 7.90), color=MUTED)

    # ======================================================= B. branches
    container(ax, 0.18, B_Y, 3.02, B_H, VIS, "M1  ·  VISUAL SCORING")
    module(ax, 0.42, 6.98, 2.54, 0.56, "CLIP image encoder",
           "$E_I$  ·  ViT-L/14, LAION-2B", VIS, VIS_BG)
    arrow(ax, (1.69, 6.96), (1.69, 6.68), color=VIS)
    cylinder(ax, 1.69, 6.44, 1.24, 0.42, "$v_t \\in \\mathbb{R}^{768}$", VIS, "#FFFFFF")
    ax.text(1.69, 6.04, "unit-norm frame embedding", fontsize=6.0, color=MUTED,
            ha="center", style="italic")

    container(ax, 3.38, B_Y, 3.26, B_H, CTX, "M3  ·  VERBALISED CONTEXT")
    box(ax, 3.56, 7.14, 2.90, 0.50, CTX_BG, CTX, lw=1.4)
    ax.text(3.66, 7.56, "OPERATOR WRITES ONE SENTENCE  $c$", fontsize=5.9,
            color=CTX, fontweight="bold", va="center", zorder=5)
    ax.text(5.01, 7.29, '"a university campus walkway with pedestrians"',
            fontsize=5.9, color=INK, ha="center", va="center", zorder=5,
            style="italic")

    scroll(ax, 3.56, 6.22, 1.38, 0.76,
           ['"a calm and safe', ' public area"', '+ c, "everything', '  is normal"'],
           CTX, "#FFFFFF", "$P^{+}$  NORMAL")
    scroll(ax, 5.08, 6.22, 1.38, 0.76,
           ['"a dangerous or', ' violent event"', '"a fight, robbery', ' or accident"'],
           CTX, "#F5F4F1", "$P^{-}$  ABNORMAL")

    # c reaches only the normal ensemble -- shown by there being one arrow
    arrow(ax, (4.25, 7.12), (4.25, 7.02), color=CTX, lw=1.4)
    ax.text(4.44, 7.07, "$c$ enters the normal ensemble only", fontsize=5.6,
            color=CTX, ha="left", va="center", zorder=6)

    # scroll bottoms sit at 6.22; the encoder's top edge at 6.10 -- the arrows
    # occupy the gap between them rather than being drawn over the module
    arrow(ax, (4.25, 6.20), (4.25, 6.12), color=CTX)
    arrow(ax, (5.77, 6.20), (5.77, 6.12), color=CTX)
    module(ax, 3.56, 5.82, 2.90, 0.28, "CLIP text encoder   $E_T$", "",
           CTX, CTX_BG, ts=6.8)

    arrow(ax, (1.69, 5.86), (1.69, 5.56), color=VIS)
    arrow(ax, (5.01, 5.80), (5.01, 5.56), color=CTX)

    # ======================================================= C. scoring
    container(ax, 0.18, C_Y, W - 0.36, C_H, INK, "FRAME SCORING")
    cylinder(ax, 4.28, 5.16, 0.78, 0.32, "$e^{+}$", CTX, "#FFFFFF", fs=7.8)
    cylinder(ax, 5.74, 5.16, 0.78, 0.32, "$e^{-}$", CTX, "#EFECE7", fs=7.8)
    ax.text(5.01, 4.90, "mean-pooled prototypes", fontsize=5.9, color=MUTED,
            ha="center", style="italic")

    box(ax, 0.42, 4.28, 3.32, 0.80, "#FFFFFF", INK, lw=1.2)
    ax.text(2.08, 4.92, "softmax over two cosine similarities",
            fontsize=6.2, color=MUTED, ha="center", zorder=5)
    ax.text(2.08, 4.60,
            r"$s_t=\dfrac{\exp(\lambda\langle v_t,e^{-}\rangle)}"
            r"{\exp(\lambda\langle v_t,e^{+}\rangle)+\exp(\lambda\langle v_t,e^{-}\rangle)}$",
            fontsize=7.4, color=INK, ha="center", va="center", zorder=5)
    arrow(ax, (3.92, 5.06), (3.80, 4.82), color=CTX, rad=-0.18)
    arrow(ax, (5.40, 5.04), (3.86, 4.70), color=CTX, rad=-0.10)
    arrow(ax, (2.08, 4.26), (2.08, 3.90), color=INK)

    # ======================================================= D. temporal
    container(ax, 0.18, D_Y, W - 0.36, D_H, TMP, "M2  ·  TEMPORAL AGGREGATION")
    box(ax, 0.42, 2.66, 2.18, 0.86, TMP_BG, TMP, lw=1.3)
    ax.text(1.51, 3.34, "centred moving average", fontsize=6.6,
            fontweight="bold", color=INK, ha="center", zorder=5)
    ax.text(1.51, 3.00, r"$\tilde{s}_t=\frac{1}{|\mathcal{W}_t|}"
                        r"\sum_{k\in\mathcal{W}_t}s_k$", fontsize=7.4,
            color=INK, ha="center", va="center", zorder=5)
    ax.text(1.51, 2.76, "$w = 31$ frames", fontsize=6.2, color=MUTED,
            ha="center", zorder=5)

    # data coordinates, so this sits inside its band rather than across three
    axi = ax.inset_axes([2.82, 2.70, 3.62, 0.78], transform=ax.transData)
    wex = os.path.join(assets, "worked_example.npz")
    if os.path.isfile(wex):
        z = np.load(wex, allow_pickle=True)
        raw, lab = z["scores_matched"], z["labels"].astype(int)
        # Scale by the SMOOTHED signal's range. Min-maxing the raw score lets a
        # single outlier spike compress the smoothed curve onto the floor, which
        # hides the very thing this panel exists to show.
        sm_raw = np.convolve(raw, np.ones(31) / 31, mode="same")
        lo, span = sm_raw.min(), sm_raw.max() - sm_raw.min() + 1e-12
        sm = (sm_raw - lo) / span
        s = np.clip((raw - lo) / span, -0.08, 1.12)
        axi.plot(s, color="#C2CECD", lw=0.55)
        axi.plot(sm, color=TMP, lw=1.5)
        idx = np.where(lab == 1)[0]
        if len(idx):
            axi.axvspan(idx[0], idx[-1], color=OUT_BAD, alpha=0.13, lw=0)
            axi.text((idx[0] + idx[-1]) / 2, 1.16, "true event", fontsize=5.4,
                     color=OUT_BAD, ha="center", va="top")
        tau = 0.45
        axi.axhline(tau, color=OUT_BAD, lw=0.8, ls=(0, (3, 2)))
        axi.text(len(s) * 0.995, tau + 0.03, r"$\tau$", fontsize=6.6,
                 color=OUT_BAD, ha="right", va="bottom")
        axi.set_xlim(0, len(s)); axi.set_ylim(-0.12, 1.24)
    axi.set_xticks([]); axi.set_yticks([])
    for sp in axi.spines.values():
        sp.set_color("#C9D2D1"); sp.set_linewidth(0.6)
    axi.set_title("raw $s_t$ (grey)  vs  smoothed $\\tilde{s}_t$ (blue)",
                  fontsize=5.8, color=MUTED, pad=2.5)

    arrow(ax, (3.45, 2.40), (3.45, 2.28), color=TMP)

    # ======================================================= E. detect + explain
    container(ax, 0.18, E_Y, W - 0.36, E_H, RSN, "DETECTION  ·  M4 REASONING")
    box(ax, 0.42, 1.62, 1.96, 0.34, "#FFFFFF", INK, lw=1.1)
    ax.text(1.40, 1.79, r"flag if  $\tilde{s}_t \geq \tau$", fontsize=6.8,
            color=INK, ha="center", va="center", zorder=5)

    if have:
        frame_img(ax, normal, 0.72, 1.16, 0.066, border=OUT_OK, lw=1.5)
        frame_img(ax, anom, 1.40, 1.16, 0.066, border=OUT_BAD, lw=2.1)
        frame_img(ax, normal, 2.08, 1.16, 0.066, border=OUT_OK, lw=1.5)
    ax.text(1.40, 0.76, "per-frame decision", fontsize=5.9, color=MUTED,
            ha="center")
    ax.text(1.40, 0.60, "peak frame of each event is explained", fontsize=5.9,
            color=MUTED, ha="center")

    module(ax, 2.66, 1.58, 1.44, 0.42, "LLaVA-1.5", "4-bit, frozen", RSN, RSN_BG,
           ts=6.9)
    arrow(ax, (2.42, 1.79), (2.64, 1.79), color=RSN)
    arrow(ax, (4.12, 1.79), (4.34, 1.79), color=RSN)

    box(ax, 4.36, 0.56, 2.14, 1.44, "#FFFFFF", RSN, lw=1.0, r=0.03)
    ax.text(4.45, 1.88, "GENERATED EXPLANATION", fontsize=5.6, color=RSN,
            fontweight="bold", va="center", zorder=5)
    ax.text(5.43, 1.66, "grounded by the same sentence $c$", fontsize=5.4,
            color=CTX, ha="center", va="center", zorder=5, style="italic")
    ax.text(5.43, 1.10, '"A cyclist is riding through\na pedestrian walkway,\n'
                        'which is unusual for this\nscene."',
            fontsize=6.1, color=INK, ha="center", va="center", zorder=5,
            style="italic", linespacing=1.5)

    # ======================================================= footer
    ax.text(W / 2, 0.24, "Deploying to a new site means editing sentence $c$   ·   "
                         "no target data   ·   no gradients",
            fontsize=6.6, color=CTX, ha="center", fontweight="bold")

    fig.savefig(out, dpi=400, facecolor=PAPER)
    plt.close(fig)
    return out


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser()
    p.add_argument("--assets", default=os.path.join(root, "figure_assets"))
    p.add_argument("--out", default=os.path.join(root, "docs", "09_paper",
                                                 "dazvad_architecture.png"))
    a = p.parse_args()
    print("saved:", build(a.assets, a.out))
