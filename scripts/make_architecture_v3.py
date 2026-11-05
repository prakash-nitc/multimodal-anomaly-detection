# -*- coding: utf-8 -*-
"""DA-ZVAD architecture, November 2026: the frozen final model.

PE-Core + calm normal phrases (C) + a frozen bank of concrete actions (B), as frozen in
docs/10_phase3/PREREG_sealed_ubnormal.md. Drawn in the style of make_architecture_v2.py
(banded, dashed colour-coded containers, a snowflake on every frozen module, the actual
text the system is given), which stays as the reference for the Phase 1-3 design.

Two figures:

  detailed   the full pipeline with notation, for the paper, thesis and slides;
  simple     five boxes and four verbs, drawn to be redrawn on a whiteboard from memory.

The score curve in the detailed figure is an illustration and is labelled as one; every
other element is the model itself.

Usage:  python scripts/make_architecture_v3.py [--assets DIR] [--outdir DIR]
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
VIS, VIS_BG = "#0D6B67", "#E4F2F0"          # image branch (P)
CALM, CALM_BG = "#2E7D4F", "#E6F3EA"        # normal side (C)
BANK, BANK_BG = "#B4571A", "#FBEDE1"        # abnormal side (B)
TXT, TXT_BG = "#5B4A8B", "#EEEAF6"          # text encoder
TMP, TMP_BG = "#3B5BA5", "#E6EBF7"          # smoothing
OUT_OK, OUT_BAD = "#1E8449", "#B03A2E"
PAPER = "#FFFFFF"
FROZEN = "❄"


def box(ax, x, y, w, h, fc, ec, lw=1.1, r=0.05, z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.012,rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))


def container(ax, x, y, w, h, ec, label):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=ec,
                           linewidth=1.0, linestyle=(0, (4, 2.5)), zorder=1))
    ax.text(x + 0.10, y + h - 0.055, label, fontsize=6.6, color=ec, fontweight="bold",
            va="center", zorder=6, bbox=dict(facecolor=PAPER, edgecolor="none", pad=1.6))


def arrow(ax, p, q, color=INK, lw=1.1, rad=0.0, z=4, ms=8):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, linewidth=lw,
                                 color=color, zorder=z, connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1, shrinkB=1))


def module(ax, x, y, w, h, title, sub, color, bg, frozen=True, ts=7.2):
    box(ax, x, y, w, h, bg, color, lw=1.3)
    if frozen:
        ax.text(x + 0.085, y + h - 0.09, FROZEN, fontsize=7.5, color=color,
                va="center", ha="center", zorder=5)
    ax.text(x + w / 2, y + h * (0.60 if sub else 0.5), title, fontsize=ts,
            fontweight="bold", color=INK, ha="center", va="center", zorder=5)
    if sub:
        ax.text(x + w / 2, y + h * 0.25, sub, fontsize=6.0, color=MUTED,
                ha="center", va="center", zorder=5, style="italic")


def scroll(ax, x, y, w, h, tag, lines, color, bg, fs=5.3, step=0.112):
    box(ax, x, y, w, h, bg, color, lw=1.1, r=0.03)
    ax.text(x + 0.07, y + h - 0.10, tag, fontsize=5.9, color=color, fontweight="bold",
            va="center", zorder=5)
    for i, ln in enumerate(lines):
        ax.text(x + 0.07, y + h - 0.25 - i * step, ln, fontsize=fs, color=INK,
                va="center", zorder=5, family="monospace")


def pill(ax, cx, cy, w, h, label, color, bg, fs=7.6):
    box(ax, cx - w / 2, cy - h / 2, w, h, bg, color, lw=1.2, r=0.05, z=4)
    ax.text(cx, cy, label, fontsize=fs, fontweight="bold", color=INK,
            ha="center", va="center", zorder=5)


def frame_img(ax, path, cx, cy, zoom, border=None, lw=1.6):
    im = OffsetImage(plt.imread(path), zoom=zoom)
    ax.add_artist(AnnotationBbox(im, (cx, cy), frameon=border is not None, pad=0.0, zorder=4,
                                 bboxprops=dict(edgecolor=border, linewidth=lw) if border else None))


# ====================================================================== detailed
def build_detailed(assets: str, out: str) -> str:
    W, H = 6.9, 9.6
    fa = os.path.join(assets, "frames", "shanghaitech")
    normal = os.path.join(fa, "01_0014_normal_00017.jpg")
    anom = os.path.join(fa, "01_0014_anomaly_00095.jpg")
    have = os.path.isfile(normal) and os.path.isfile(anom)

    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    fig.patch.set_facecolor(PAPER)
    ax.add_patch(Rectangle((0.08, 0.08), W - 0.16, H - 0.16, facecolor="none",
                           edgecolor="#C9B458", linewidth=1.0, linestyle=(0, (5, 3)), zorder=0))

    # ---------------------------------------------------------- A. input
    container(ax, 0.18, 8.25, W - 0.36, 1.18, MUTED, "INPUT VIDEO")
    if have:
        for dx in (0.0, 0.07, 0.14):
            frame_img(ax, normal, 0.68 + dx, 8.86 + dx * 0.5, 0.108, border="#9AA6A7", lw=0.7)
        frame_img(ax, anom, 2.18, 8.90, 0.126, border="#D4A017", lw=1.9)
    ax.text(0.82, 8.44, "every 16th frame*", fontsize=6.5, color=MUTED, ha="center")
    ax.text(2.18, 8.44, "current frame  $f_t$", fontsize=6.5, color="#9A7500",
            ha="center", fontweight="bold")
    box(ax, 3.32, 8.40, 3.24, 0.86, "#FBFBF9", MUTED, lw=0.9)
    ax.text(3.44, 9.16, f"{FROZEN}  EVERY MODEL FROZEN  ·  NOTHING TRAINED", fontsize=6.3, color=VIS,
            fontweight="bold", va="center", zorder=5)
    ax.text(4.94, 8.73,
            "No parameter is updated at any stage.\n"
            "The 59 phrases were fixed before scoring, and\n"
            "the same lists are used for every camera.",
            fontsize=6.1, color=INK, ha="center", va="center", zorder=5, linespacing=1.6)
    arrow(ax, (2.18, 8.62), (2.18, 8.08), color=MUTED)

    # ---------------------------------------------------------- B. two branches
    container(ax, 0.18, 5.62, 2.62, 2.43, VIS, "P  ·  IMAGE BRANCH")
    module(ax, 0.36, 7.02, 2.26, 0.62, "PE-Core image encoder",
           "PE-Core-L-14-336  ·  336 × 336 px", VIS, VIS_BG)
    arrow(ax, (1.49, 7.00), (1.49, 6.70), color=VIS)
    pill(ax, 1.49, 6.45, 1.30, 0.42, "$v_t \\in \\mathbb{R}^{1024}$", VIS, "#FFFFFF")
    ax.text(1.49, 6.07, "unit-norm frame vector", fontsize=6.0, color=MUTED,
            ha="center", style="italic")

    container(ax, 2.98, 5.62, 3.74, 2.43, TXT, "TEXT BRANCH  ·  59 FIXED PHRASES")
    scroll(ax, 3.10, 6.36, 1.74, 1.42, "NORMAL SIDE  (11)", [
        '"a calm and safe',
        ' public area"   + 2 more',
        "C: 8 calm phrases",
        '"an empty walkway"',
        '"a person standing still"',
        '"two people talking calmly"',
        "  ... no look-alike actions",
    ], CALM, CALM_BG, fs=5.0, step=0.150)
    scroll(ax, 4.94, 6.36, 1.66, 1.42, "ABNORMAL SIDE  (48)", [
        "B: 12 types × 4 actions",
        '"two people punching',
        '  each other"',
        '"a cyclist riding through',
        '  a crowd of pedestrians"',
        '"a car on fire"',
        "  ... written once, frozen",
    ], BANK, BANK_BG, fs=5.0, step=0.150)
    arrow(ax, (3.97, 6.34), (3.97, 6.24), color=CALM)
    arrow(ax, (5.77, 6.34), (5.77, 6.24), color=BANK)
    module(ax, 3.10, 5.84, 3.50, 0.38, "PE-Core text encoder,  averaged per side", "",
           TXT, TXT_BG, ts=6.8)

    arrow(ax, (1.49, 5.88), (1.49, 5.40), color=VIS)
    arrow(ax, (4.85, 5.82), (4.85, 5.40), color=TXT)

    # ---------------------------------------------------------- C. scoring
    container(ax, 0.18, 3.90, W - 0.36, 1.42, INK, "FRAME SCORING")
    pill(ax, 4.10, 4.90, 0.86, 0.34, "$e^{+}$  normal", CALM, CALM_BG, fs=6.6)
    pill(ax, 5.62, 4.90, 0.96, 0.34, "$e^{-}$  abnormal", BANK, BANK_BG, fs=6.6)
    ax.text(4.86, 5.15, "prototypes: each side's unit vectors averaged, then renormalised",
            fontsize=5.6, color=MUTED, ha="center", style="italic")
    box(ax, 0.36, 4.06, 3.10, 0.88, "#FFFFFF", INK, lw=1.2)
    ax.text(1.91, 4.79, "abnormal similarity minus normal similarity", fontsize=6.0,
            color=MUTED, ha="center", zorder=5)
    ax.text(1.91, 4.43, r"$s_t=\sigma\!\left(\lambda\left(\langle v_t,e^{-}\rangle-\langle v_t,e^{+}\rangle\right)\right)$",
            fontsize=7.6, color=INK, ha="center", va="center", zorder=5)
    ax.text(1.91, 4.16, r"$\lambda = 100$, fixed  ·  $\sigma$ = sigmoid", fontsize=5.8,
            color=MUTED, ha="center", zorder=5)
    arrow(ax, (3.66, 4.86), (3.50, 4.64), color=CALM, rad=-0.15)
    arrow(ax, (5.20, 4.74), (3.52, 4.46), color=BANK, rad=-0.12)
    arrow(ax, (1.91, 4.04), (1.91, 3.70), color=INK)

    # ---------------------------------------------------------- D. smoothing
    container(ax, 0.18, 2.22, W - 0.36, 1.48, TMP, "TEMPORAL SMOOTHING")
    box(ax, 0.36, 2.44, 2.20, 0.90, TMP_BG, TMP, lw=1.3)
    ax.text(1.46, 3.16, "centred moving average", fontsize=6.6, fontweight="bold",
            color=INK, ha="center", zorder=5)
    ax.text(1.46, 2.84, r"$\tilde{s}_t=\frac{1}{|\mathcal{W}_t|}\sum_{k\in\mathcal{W}_t}s_k$",
            fontsize=7.4, color=INK, ha="center", va="center", zorder=5)
    ax.text(1.46, 2.56, "5 sampled frames*  ·  raw scores", fontsize=5.8,
            color=MUTED, ha="center", zorder=5)
    axi = ax.inset_axes([2.80, 2.46, 3.70, 0.84], transform=ax.transData)
    rng = np.random.default_rng(7)
    t = np.arange(160)
    lab = (t >= 92) & (t < 122)
    raw = 0.18 + 0.05 * rng.standard_normal(len(t)) + 0.55 * lab
    raw[30] += 0.45                                       # one stray frame
    sm = np.convolve(raw, np.ones(5) / 5, mode="same")
    axi.plot(raw, color="#C2CECD", lw=0.6)
    axi.plot(sm, color=TMP, lw=1.5)
    axi.axvspan(92, 121, color=OUT_BAD, alpha=0.13, lw=0)
    axi.text(106, 1.02, "event", fontsize=5.4, color=OUT_BAD, ha="center", va="top")
    axi.axhline(0.5, color=OUT_BAD, lw=0.8, ls=(0, (3, 2)))
    axi.text(158, 0.53, r"$\tau$", fontsize=6.4, color=OUT_BAD, ha="right", va="bottom")
    axi.set_xlim(0, 159); axi.set_ylim(-0.05, 1.08)
    axi.set_xticks([]); axi.set_yticks([])
    for sp in axi.spines.values():
        sp.set_color("#C9D2D1"); sp.set_linewidth(0.6)
    axi.set_title("illustration: raw $s_t$ (grey) vs smoothed $\\tilde{s}_t$ (blue)",
                  fontsize=5.6, color=MUTED, pad=2.5)
    arrow(ax, (3.45, 2.20), (3.45, 2.06), color=TMP)

    # ---------------------------------------------------------- E. output
    container(ax, 0.18, 0.52, W - 0.36, 1.50, OUT_BAD, "OUTPUT")
    box(ax, 0.36, 1.48, 1.98, 0.34, "#FFFFFF", INK, lw=1.1)
    ax.text(1.35, 1.65, r"flag if  $\tilde{s}_t \geq \tau$", fontsize=6.8, color=INK,
            ha="center", va="center", zorder=5)
    if have:
        frame_img(ax, normal, 0.70, 1.02, 0.060, border=OUT_OK, lw=1.5)
        frame_img(ax, anom, 1.35, 1.02, 0.060, border=OUT_BAD, lw=2.1)
        frame_img(ax, normal, 2.00, 1.02, 0.060, border=OUT_OK, lw=1.5)
    ax.text(1.35, 0.66, "anomaly score for every frame", fontsize=5.9, color=MUTED, ha="center")
    box(ax, 2.62, 0.78, 3.92, 1.04, "#FBFBF9", MUTED, lw=0.9, r=0.03)
    ax.text(2.74, 1.68, "REMOVED FROM THE EARLIER DESIGN", fontsize=5.9, color=MUTED,
            fontweight="bold", va="center", zorder=5)
    ax.text(2.74, 1.22,
            "· no scene sentence written for each camera\n"
            "· no captioning model (LLaVA) at any stage\n"
            "· nothing that depends on the first frames",
            fontsize=6.0, color=INK, va="center", zorder=5, linespacing=1.55)

    ax.text(W / 2, 0.34, "A new camera needs no new data:  the same 59 phrases  ·  no labels  ·  no gradients",
            fontsize=6.6, color=VIS, ha="center", fontweight="bold")
    ax.text(W / 2, 0.17, "* ShanghaiTech follows its benchmark protocol: every 2nd frame, a window of 31, "
                         "and scores rescaled within each clip", fontsize=5.6, color=MUTED, ha="center")
    fig.savefig(out, dpi=400, facecolor=PAPER)
    fig.savefig(os.path.splitext(out)[0] + ".pdf", facecolor=PAPER)
    plt.close(fig)
    return out


# ====================================================================== simple
def build_simple(out: str) -> str:
    """Three inputs, one encoder, four numbered steps: the version to redraw on a whiteboard."""
    W, H = 10.4, 4.9
    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    fig.patch.set_facecolor(PAPER)

    def b(x, y, w, h, title, sub, ec, fc, step=None, word=None):
        box(ax, x, y, w, h, fc, ec, lw=1.6, r=0.06)
        ax.text(x + w / 2, y + h * (0.62 if sub else 0.5), title, fontsize=10.5, fontweight="bold",
                color=INK, ha="center", va="center", zorder=5)
        if sub:
            ax.text(x + w / 2, y + h * 0.30, sub, fontsize=7.8, color=MUTED, ha="center",
                    va="center", zorder=5)
        if step:
            ax.add_patch(plt.Circle((x + 0.02, y + h - 0.02), 0.19, color=INK, zorder=6))
            ax.text(x + 0.02, y + h - 0.02, str(step), fontsize=9, color=PAPER, fontweight="bold",
                    ha="center", va="center", zorder=7)
            ax.text(x + 0.27, y + h + 0.17, word, fontsize=9, color=INK, fontweight="bold",
                    va="center", zorder=7)

    rows = (3.45, 2.05, 0.65)                          # bottoms of the three input rows
    b(0.30, rows[0], 2.00, 0.85, "FRAME", "sampled from video", MUTED, "#F4F6F6")
    b(0.30, rows[1], 2.00, 0.85, "NORMAL", "3 general + 8 calm", CALM, CALM_BG)
    b(0.30, rows[2], 2.00, 0.85, "ABNORMAL", "48 concrete actions", BANK, BANK_BG)
    b(2.85, 0.65, 1.50, 3.65, "PE-Core", "frozen", VIS, VIS_BG, 1, "Encode")
    b(4.90, rows[0], 1.80, 0.85, "frame vector", "1024 numbers", VIS, "#FFFFFF")
    b(4.90, rows[1], 1.80, 0.85, "normal average", "one vector", CALM, "#FFFFFF")
    b(4.90, rows[2], 1.80, 0.85, "abnormal average", "one vector", BANK, "#FFFFFF")
    b(7.25, 1.75, 1.20, 1.45, "COMPARE", "closeness\nto each", INK, "#FFFFFF", 2, "Compare")
    b(8.85, 2.70, 1.35, 1.00, "SUBTRACT", "abn. − normal", INK, "#FFFFFF", 3, "Subtract")
    b(8.85, 0.95, 1.35, 1.00, "SMOOTH", "→ score 0–1", TMP, TMP_BG, 4, "Smooth")

    for y in rows:
        arrow(ax, (2.32, y + 0.42), (2.83, y + 0.42), lw=1.5, ms=11)
        arrow(ax, (4.37, y + 0.42), (4.88, y + 0.42), lw=1.5, ms=11)
    arrow(ax, (6.72, rows[0] + 0.42), (7.23, 3.00), lw=1.4, ms=10)
    arrow(ax, (6.72, rows[1] + 0.42), (7.23, 2.47), lw=1.4, ms=10)
    arrow(ax, (6.72, rows[2] + 0.42), (7.23, 1.95), lw=1.4, ms=10)
    arrow(ax, (8.47, 2.75), (8.83, 3.05), lw=1.5, ms=11)
    arrow(ax, (10.0, 2.68), (10.0, 1.97), lw=1.5, ms=11)
    ax.text(0.30, 0.25, "P-C-B:  PE-Core · Calm phrases · Bank of actions   —   nothing trained, "
            "the same 59 phrases for every camera", fontsize=8.6, color=VIS, fontweight="bold", va="center")
    fig.savefig(out, dpi=300, facecolor=PAPER)
    plt.close(fig)
    return out


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser()
    p.add_argument("--assets", default=os.path.join(root, "figure_assets"))
    p.add_argument("--outdir", default=os.path.join(root, "docs", "09_paper"))
    a = p.parse_args()
    print("saved:", build_detailed(a.assets, os.path.join(a.outdir, "dazvad_architecture_v3.png")))
    print("saved:", build_simple(os.path.join(a.outdir, "dazvad_architecture_v3_simple.png")))
