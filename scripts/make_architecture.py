# -*- coding: utf-8 -*-
"""DA-ZVAD architecture figure -- compact landscape version.

Matches the layout of the reference the supervisor supplied: input at the left,
two parallel branches through the middle, fusion and output at the right, with
dashed colour-coded containers, a snowflake on every frozen module, real video
frames, and mathematical notation for the intermediate quantities.

Deliberately less detailed than the tall variant in make_architecture_v2.py.
Every module is named and every quantity labelled, but the prose, the full
scoring expression and the score trace are dropped -- the figure's job here is to
show the shape of the system at a glance, and the paper's text carries the rest.

Sized so that it prints at close to 1:1 in a single-column layout rather than
being scaled down until the labels stop being readable.

Usage:  python scripts/make_architecture.py [--assets DIR] [--out PATH]
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

INK, MUTED = "#141C1E", "#5F6E70"
VIS, VIS_BG = "#0D6B67", "#E4F2F0"
CTX, CTX_BG = "#B4571A", "#FBEDE1"
TMP, TMP_BG = "#3B5BA5", "#E6EBF7"
RSN, RSN_BG = "#6B4C9A", "#EFEAF7"
OUT_OK, OUT_BAD = "#1E8449", "#B03A2E"
PAPER = "#FFFFFF"

W, H = 7.15, 3.95
FROZEN = "❄"


def box(ax, x, y, w, h, fc, ec, lw=1.0, r=0.045, z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0.010,rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))


def container(ax, x, y, w, h, ec, label):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=ec,
                           linewidth=0.9, linestyle=(0, (3.5, 2.2)), zorder=1))
    ax.text(x + 0.08, y + h - 0.045, label, fontsize=5.9, color=ec,
            fontweight="bold", va="center", zorder=6,
            bbox=dict(facecolor=PAPER, edgecolor="none", pad=1.3))


def arrow(ax, p, q, color=INK, lw=1.0, rad=0.0, z=5):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7,
                                 linewidth=lw, color=color, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1, shrinkB=1))


def module(ax, x, y, w, h, title, sub, color, bg, frozen=True, ts=6.5):
    box(ax, x, y, w, h, bg, color, lw=1.15)
    if frozen:
        ax.text(x + 0.075, y + h - 0.075, FROZEN, fontsize=6.4, color=color,
                va="center", ha="center", zorder=5)
    ax.text(x + w / 2, y + h * (0.62 if sub else 0.5), title, fontsize=ts,
            fontweight="bold", color=INK, ha="center", va="center", zorder=5)
    if sub:
        ax.text(x + w / 2, y + h * 0.25, sub, fontsize=5.4, color=MUTED,
                ha="center", va="center", zorder=5, style="italic")


def pill(ax, cx, cy, w, h, label, color, bg, fs=6.6):
    box(ax, cx - w / 2, cy - h / 2, w, h, bg, color, lw=1.1, r=0.05, z=4)
    ax.text(cx, cy, label, fontsize=fs, fontweight="bold", color=INK,
            ha="center", va="center", zorder=5)


def frame_img(ax, path, cx, cy, zoom, border=None, lw=1.4):
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

    ax.add_patch(Rectangle((0.06, 0.06), W - 0.12, H - 0.12, facecolor="none",
                           edgecolor="#C9B458", linewidth=0.9,
                           linestyle=(0, (4.5, 2.8)), zorder=0))

    # ------------------------------------------------ inputs (left)
    # Two distinct inputs, kept separate because they are: the camera supplies
    # frames to M1, the operator supplies one sentence to M3. Drawing a single
    # "input" feeding both would misdescribe where the adaptation comes from.
    container(ax, 0.14, 1.86, 1.06, 1.80, MUTED, "VIDEO")
    if have:
        for dx in (0.0, 0.05, 0.10):
            frame_img(ax, normal, 0.56 + dx, 3.16 + dx * 0.55, 0.062,
                      border="#9AA6A7", lw=0.6)
        frame_img(ax, anom, 0.67, 2.36, 0.072, border="#D4A017", lw=1.6)
    ax.text(0.67, 2.84, "$f_1 \\ldots f_T$", fontsize=6.2, color=MUTED, ha="center")
    ax.text(0.67, 2.02, "current frame $f_t$", fontsize=5.8, color="#9A7500",
            ha="center", fontweight="bold")

    container(ax, 0.14, 0.30, 1.06, 1.42, MUTED, "OPERATOR")
    box(ax, 0.26, 0.74, 0.82, 0.62, "#FBFBF9", MUTED, lw=0.9)
    ax.text(0.67, 1.20, "writes one", fontsize=5.6, color=INK, ha="center",
            va="center", zorder=5)
    ax.text(0.67, 1.04, "sentence", fontsize=5.6, color=INK, ha="center",
            va="center", zorder=5)
    ax.text(0.67, 0.87, "$c$", fontsize=7.6, color=CTX, ha="center",
            va="center", fontweight="bold", zorder=5)
    ax.text(0.67, 0.46, f"{FROZEN} = frozen", fontsize=5.5, color=VIS,
            ha="center", fontweight="bold")

    # ------------------------------------------------ M1 (top branch)
    container(ax, 1.32, 2.16, 2.52, 1.50, VIS, "M1 · VISUAL SCORING")
    module(ax, 1.48, 2.86, 2.20, 0.50, "CLIP image encoder",
           "$E_I$ · ViT-L/14", VIS, VIS_BG)
    arrow(ax, (2.58, 2.84), (2.58, 2.66), color=VIS)
    pill(ax, 2.58, 2.46, 1.16, 0.32, "$v_t \\in \\mathbb{R}^{768}$", VIS, "#FFFFFF", fs=6.4)

    # ------------------------------------------------ M3 (bottom branch)
    container(ax, 1.32, 0.30, 2.52, 1.76, CTX, "M3 · VERBALISED CONTEXT")
    box(ax, 1.48, 1.52, 2.20, 0.40, CTX_BG, CTX, lw=1.15)
    ax.text(2.58, 1.80, "scene sentence  $c$", fontsize=5.8, color=CTX,
            ha="center", va="center", fontweight="bold", zorder=5)
    ax.text(2.58, 1.63, '"a campus walkway with pedestrians"', fontsize=5.5,
            color=INK, ha="center", va="center", style="italic", zorder=5)

    box(ax, 1.48, 0.94, 1.04, 0.44, "#FFFFFF", CTX, lw=1.0, r=0.03)
    ax.text(2.00, 1.26, "$P^{+}$ normal", fontsize=5.7, color=CTX, ha="center",
            va="center", fontweight="bold", zorder=5)
    ax.text(2.00, 1.07, "prompts $+\\, c$", fontsize=5.5, color=INK,
            ha="center", va="center", zorder=5)

    box(ax, 2.64, 0.94, 1.04, 0.44, "#F5F4F1", CTX, lw=1.0, r=0.03)
    ax.text(3.16, 1.26, "$P^{-}$ abnormal", fontsize=5.7, color=CTX, ha="center",
            va="center", fontweight="bold", zorder=5)
    ax.text(3.16, 1.07, "prompts only", fontsize=5.5, color=INK,
            ha="center", va="center", zorder=5)

    arrow(ax, (2.00, 1.50), (2.00, 1.40), color=CTX, lw=1.2)
    ax.text(2.30, 1.45, "$c$ enters $P^{+}$ only", fontsize=5.2, color=CTX,
            ha="left", va="center", zorder=6)

    arrow(ax, (2.00, 0.92), (2.00, 0.82), color=CTX)
    arrow(ax, (3.16, 0.92), (3.16, 0.82), color=CTX)
    module(ax, 1.48, 0.44, 2.20, 0.36, "CLIP text encoder  $E_T$", "",
           CTX, CTX_BG, ts=6.2)

    # ------------------------------------------------ fusion (right-middle)
    container(ax, 3.96, 1.44, 1.60, 2.22, INK, "FRAME SCORING")
    pill(ax, 4.44, 3.18, 0.60, 0.30, "$e^{+}$", CTX, "#FFFFFF")
    pill(ax, 5.14, 3.18, 0.60, 0.30, "$e^{-}$", CTX, "#EFECE7")
    ax.text(4.79, 2.93, "prototypes", fontsize=5.4, color=MUTED, ha="center",
            style="italic")

    box(ax, 4.10, 2.16, 1.32, 0.60, "#FFFFFF", INK, lw=1.1)
    ax.text(4.76, 2.60, "softmax over", fontsize=5.6, color=MUTED, ha="center",
            zorder=5)
    ax.text(4.76, 2.44, r"$\langle v_t,e^{+}\rangle$ , $\langle v_t,e^{-}\rangle$",
            fontsize=6.2, color=INK, ha="center", va="center", zorder=5)
    ax.text(4.76, 2.26, "$\\rightarrow\\; s_t$", fontsize=6.6, color=INK,
            ha="center", va="center", fontweight="bold", zorder=5)

    module(ax, 4.10, 1.58, 1.32, 0.44, "M2 · moving average",
           "$\\tilde{s}_t$ ,  $w=31$", TMP, TMP_BG, frozen=False, ts=6.0)

    arrow(ax, (3.70, 2.46), (4.06, 2.46), color=VIS)          # v_t -> scoring
    arrow(ax, (3.70, 0.62), (3.86, 0.62), color=CTX)
    arrow(ax, (3.86, 0.62), (3.86, 3.18), color=CTX, lw=0.9)  # up the side
    arrow(ax, (3.86, 3.18), (4.12, 3.18), color=CTX)
    arrow(ax, (4.44, 3.02), (4.60, 2.78), color=CTX, rad=-0.15)
    arrow(ax, (5.14, 3.02), (4.94, 2.78), color=CTX, rad=0.15)
    arrow(ax, (4.76, 2.14), (4.76, 2.04), color=INK)

    # ------------------------------------------------ output (far right)
    container(ax, 5.68, 0.30, 1.34, 3.36, RSN, "DETECTION · M4")
    ax.text(6.35, 3.40, r"flag if $\tilde{s}_t \geq \tau$", fontsize=6.0,
            color=INK, ha="center", va="center", zorder=5)
    if have:
        # zoom 0.034 on a 640px frame is ~0.30in wide; 0.37in spacing keeps a gap
        frame_img(ax, normal, 5.98, 3.02, 0.034, border=OUT_OK, lw=1.2)
        frame_img(ax, anom, 6.35, 3.02, 0.034, border=OUT_BAD, lw=1.7)
        frame_img(ax, normal, 6.72, 3.02, 0.034, border=OUT_OK, lw=1.2)
    ax.text(6.35, 2.74, "per-frame decision", fontsize=5.3, color=MUTED,
            ha="center")

    module(ax, 5.84, 2.16, 1.02, 0.40, "LLaVA-1.5", "4-bit", RSN, RSN_BG, ts=6.2)
    arrow(ax, (6.35, 2.70), (6.35, 2.60), color=RSN)
    arrow(ax, (6.35, 2.12), (6.35, 1.98), color=RSN)

    box(ax, 5.82, 0.86, 1.06, 1.10, "#FFFFFF", RSN, lw=1.0, r=0.03)
    ax.text(6.35, 1.84, "explanation", fontsize=5.5, color=RSN, ha="center",
            va="center", fontweight="bold", zorder=5)
    ax.text(6.35, 1.36, '"A cyclist is\nriding through\na pedestrian\nwalkway."',
            fontsize=5.4, color=INK, ha="center", va="center", style="italic",
            linespacing=1.45, zorder=5)
    ax.text(6.35, 0.62, "grounded by $c$", fontsize=5.2, color=CTX,
            ha="center", style="italic")

    # video -> M1, operator -> M3: two separate paths, never crossing
    arrow(ax, (1.22, 2.90), (1.44, 3.10), color=MUTED)
    arrow(ax, (1.22, 1.05), (1.44, 1.60), color=CTX, lw=1.2)
    arrow(ax, (5.50, 1.80), (5.74, 1.80), color=TMP)

    ax.text(W / 2, 0.155, "Deploying to a new site = editing $c$   ·   "
                          "no target data   ·   no gradients",
            fontsize=5.9, color=CTX, ha="center", fontweight="bold")

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
