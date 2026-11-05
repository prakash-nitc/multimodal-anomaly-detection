# -*- coding: utf-8 -*-
"""DA-ZVAD architecture figure for the paper -- the frozen final model, November 2026.

The same compact landscape layout, palette, type scale and overflow check as
make_architecture.py, which draws the earlier CLIP + scene-sentence design and stays
as the reference for it. Panel by panel:

    VIDEO               unchanged
    TEXT · FIXED ONCE   replaces SETUP · ONCE: no caption is written for each camera;
                        the 59 phrases were fixed before testing
    M1                  PE-Core replaces CLIP ViT-L/14 (1024-d, 336 px)
    M3                  calm normal phrases (C) and the 48-phrase bank (B) replace the
                        scene sentence c_k
    FRAME SCORING       unchanged rule, written as a sigmoid of the margin
    M2                  w = 5 sampled frames (31 on ShanghaiTech, its benchmark protocol)
    OUTPUT              the score and the decision; LLaVA is no longer used

The smoothing inset is a schematic, as in the earlier figure.

Usage:  python scripts/make_architecture_paper_v3.py [--assets DIR] [--out PATH]
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")

import figsave
import make_architecture as base
import matplotlib.pyplot as plt
import numpy as np
from make_architecture import (CTX, CTX_BG, FROZEN, FS_BODY, FS_BOX, FS_FORM, FS_GLOSS,
                               FS_TINY, FS_TITLE, GOLD, H, INK, MUTED, OUT_BAD, OUT_OK, PAPER,
                               RSN, TMP, TMP_BG, VIS, VIS_BG, W, arrow, box, check_overflow,
                               container, frame_img, gloss, module, pill)
from matplotlib.patches import Rectangle

# The normal side gets its own green so C reads apart from the bank (B) at a glance.
CALM, CALM_BG = "#1F6B3E", "#E5F2E9"


def smoothing_inset(ax, x, y, w, h):
    """Schematic of M2 at w = 5: a jagged score and its centred moving average."""
    ins = ax.inset_axes([x, y, w, h], transform=ax.transData, zorder=6)
    rng = np.random.default_rng(7)
    n = 64
    t = np.arange(n)
    base_curve = 0.28 + 0.46 * ((t >= 36) & (t < 48))
    raw = np.clip(base_curve + rng.normal(0, 0.07, n), 0, 1)
    raw[14] = 0.80                                          # one stray frame
    k = 5
    sm = np.convolve(np.pad(raw, k // 2, mode="edge"), np.ones(k) / k, "same")[k // 2: k // 2 + n]
    ins.axvspan(36, 47.5, color="#F2DFC4", zorder=0)
    ins.plot(t, raw, lw=0.6, color="#9AA3A4", zorder=2)
    ins.plot(t, sm, lw=1.5, color=TMP, zorder=3)
    ins.set_xticks([]); ins.set_yticks([])
    ins.set_ylim(0, 1.32)                                  # headroom so the labels clear the curve
    for s in ins.spines.values():
        s.set_edgecolor(MUTED); s.set_linewidth(0.7)
    ins.set_facecolor("#FFFFFF")
    ins.text(0.03, 0.93, "$s_t$ raw", transform=ins.transAxes, fontsize=7.0,
             color="#4A5456", va="top", fontweight="bold")
    ins.text(0.97, 0.93, "$\\tilde{s}_t$ smoothed", transform=ins.transAxes,
             fontsize=7.0, color=TMP, va="top", ha="right", fontweight="bold")


def build(assets: str, out: str) -> str:
    base.BOXES.clear(); base.CONTAINERS.clear()
    fa = os.path.join(assets, "frames", "shanghaitech")
    normal = os.path.join(fa, "01_0014_normal_00017.jpg")
    anom = os.path.join(fa, "01_0014_anomaly_00095.jpg")
    have = os.path.isfile(normal) and os.path.isfile(anom)

    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    fig.patch.set_facecolor(PAPER)
    ax.add_patch(Rectangle((0.05, 0.05), W - 0.10, H - 0.10, facecolor="none",
                           edgecolor="#B39A3C", linewidth=1.0, linestyle=(0, (4.5, 2.8)), zorder=0))

    # ------------------------------------------------ inputs (left)
    container(ax, 0.12, 2.72, 1.34, 1.86, MUTED, "VIDEO")
    if have:
        for dx in (0.0, 0.055, 0.11):
            frame_img(ax, normal, 0.62 + dx, 3.95 + dx * 0.5, 0.070, border="#7E8A8B", lw=0.7)
        frame_img(ax, anom, 0.79, 3.16, 0.081, border="#C08A00", lw=1.8)
    ax.text(0.79, 3.56, "every 16th frame", fontsize=FS_TINY, color=MUTED, ha="center")
    ax.text(0.79, 2.86, "current frame $f_t$", fontsize=FS_GLOSS, color=GOLD,
            ha="center", fontweight="bold")

    # The text is not produced per camera any more: it was fixed once, before any test.
    container(ax, 0.12, 0.72, 1.34, 1.72, CALM, "FIXED TEXT")
    module(ax, 0.24, 1.66, 1.10, 0.46, "59 phrases", "11 + 48",
           CALM, CALM_BG, frozen=False, ts=8.4)
    gloss(ax, 0.79, 1.46, "fixed before scoring;", fs=FS_TINY)
    gloss(ax, 0.79, 1.33, "same for every camera", fs=FS_TINY)
    gloss(ax, 0.79, 1.13, "no caption, no sentence", fs=FS_TINY)
    gloss(ax, 0.79, 1.00, "per camera", fs=FS_TINY)
    ax.text(0.79, 0.81, f"{FROZEN} = frozen", fontsize=FS_TINY, color=VIS, ha="center",
            fontweight="bold")

    # ------------------------------------------------ M1 (top branch)
    container(ax, 1.60, 2.72, 2.92, 1.86, VIS, "M1 · VISUAL SCORING")
    module(ax, 1.78, 3.78, 2.56, 0.58, "PE-Core image encoder",
           "$E_I$ · PE-Core-L/14 · 336 px", VIS, VIS_BG)
    arrow(ax, (3.06, 3.76), (3.06, 3.62), color=VIS)
    pill(ax, 3.06, 3.44, 1.80, 0.34, "$v_t = E_I(f_t) \\in \\mathbb{R}^{1024}$",
         VIS, "#FFFFFF", fs=8.0)
    gloss(ax, 3.06, 3.14, "each frame becomes 1024 numbers of unit length;")
    gloss(ax, 3.06, 2.98, "the direction encodes what it contains")

    # ------------------------------------------------ M3 (bottom branch)
    container(ax, 1.60, 0.44, 2.92, 2.10, CTX, "M3 · NORMAL AND ABNORMAL TEXT")
    box(ax, 1.78, 1.78, 1.22, 0.52, CALM_BG, CALM, lw=1.3, r=0.03)
    ax.text(2.39, 2.17, "C · calm normal", fontsize=FS_BOX - 0.6, color=CALM, ha="center",
            va="center", fontweight="bold", zorder=5)
    ax.text(2.39, 1.98, '"a person', fontsize=FS_TINY, color=INK, ha="center", va="center",
            style="italic", zorder=5)
    ax.text(2.39, 1.86, 'standing still"', fontsize=FS_TINY, color=INK, ha="center",
            va="center", style="italic", zorder=5)
    box(ax, 3.12, 1.78, 1.22, 0.52, CTX_BG, CTX, lw=1.3, r=0.03)
    ax.text(3.73, 2.17, "B · action bank", fontsize=FS_BOX - 0.6, color=CTX, ha="center",
            va="center", fontweight="bold", zorder=5)
    ax.text(3.73, 1.98, '"two people punching', fontsize=FS_TINY, color=INK, ha="center",
            va="center", style="italic", zorder=5)
    ax.text(3.73, 1.86, 'each other"', fontsize=FS_TINY, color=INK, ha="center",
            va="center", style="italic", zorder=5)

    box(ax, 1.78, 1.08, 1.22, 0.50, "#FFFFFF", CALM, lw=1.15, r=0.03)
    ax.text(2.39, 1.43, "$P^{+}$ normal", fontsize=FS_BOX, color=CALM, ha="center",
            va="center", fontweight="bold", zorder=5)
    ax.text(2.39, 1.23, "3 general + 8 calm", fontsize=FS_TINY, color=INK, ha="center",
            va="center", zorder=5)
    box(ax, 3.12, 1.08, 1.22, 0.50, "#F4F2EE", CTX, lw=1.15, r=0.03)
    ax.text(3.73, 1.43, "$P^{-}$ abnormal", fontsize=FS_BOX, color=CTX, ha="center",
            va="center", fontweight="bold", zorder=5)
    ax.text(3.73, 1.23, "12 types × 4 actions", fontsize=FS_TINY, color=INK, ha="center",
            va="center", zorder=5)

    arrow(ax, (2.39, 1.76), (2.39, 1.61), color=CALM, lw=1.3)
    arrow(ax, (3.73, 1.76), (3.73, 1.61), color=CTX, lw=1.3)
    arrow(ax, (2.39, 1.08), (2.39, 1.02), color=CALM)
    arrow(ax, (3.73, 1.08), (3.73, 1.02), color=CTX)
    module(ax, 1.78, 0.58, 2.56, 0.42, "PE-Core text encoder  $E_T$", "", CTX, CTX_BG, ts=FS_TITLE)

    # ------------------------------------------------ fusion (right-middle)
    container(ax, 4.66, 2.34, 1.86, 2.24, INK, "FRAME SCORING")
    pill(ax, 5.14, 4.20, 0.66, 0.32, "$e^{+}$", CALM, CALM_BG)
    pill(ax, 6.04, 4.20, 0.66, 0.32, "$e^{-}$", CTX, "#EDE9E3")
    box(ax, 4.82, 2.98, 1.54, 0.80, "#FFFFFF", INK, lw=1.25)
    ax.text(5.59, 3.63, "abnormal minus normal", fontsize=FS_BODY - 0.4, color=MUTED,
            ha="center", zorder=5)
    ax.text(5.59, 3.40, r"$m_t=\langle v_t,e^{-}\rangle-\langle v_t,e^{+}\rangle$",
            fontsize=FS_FORM, color=INK, ha="center", va="center", zorder=5)
    ax.text(5.59, 3.15, r"$\rightarrow\; s_t=\sigma(\lambda\, m_t)$",
            fontsize=FS_FORM, color=INK, ha="center", va="center", fontweight="bold", zorder=5)
    gloss(ax, 5.59, 2.74, "$e^{\\pm}$: each side's mean, renormalised", fs=FS_TINY)
    gloss(ax, 5.59, 2.58, "$\\lambda$ = 100, fixed", fs=FS_TINY)

    # ------------------------------------------------ M2 (its own module)
    container(ax, 4.66, 0.44, 1.86, 1.74, TMP, "M2 · SMOOTHING")
    module(ax, 4.82, 1.58, 1.54, 0.44, "centred moving average",
           "window $w = 5$ frames", TMP, TMP_BG, frozen=False, ts=7.6)
    ax.text(5.59, 1.36, r"$\tilde{s}_t=\frac{1}{w}\sum_{i=t-2}^{\,t+2} s_i$",
            fontsize=8.4, color=INK, ha="center", va="center", zorder=6)
    smoothing_inset(ax, 4.86, 0.72, 1.46, 0.40)
    gloss(ax, 5.59, 0.64, "about 3 s of video:", fs=6.4)
    gloss(ax, 5.59, 0.52, "one spike cannot raise an alarm", fs=6.4)

    # ------------------------------------------------ wiring
    arrow(ax, (4.38, 3.44), (4.78, 3.36), color=VIS)              # v_t -> scoring
    arrow(ax, (4.38, 0.79), (4.59, 0.79), color=CTX)              # E_T out
    arrow(ax, (4.59, 0.79), (4.59, 4.20), color=CTX, lw=1.0)      # up the side
    arrow(ax, (4.59, 4.20), (4.78, 4.20), color=CTX)
    arrow(ax, (5.14, 4.04), (5.34, 3.84), color=CALM, rad=-0.15)
    arrow(ax, (6.04, 4.04), (5.84, 3.84), color=CTX, rad=0.15)
    arrow(ax, (5.59, 2.42), (5.59, 2.16), color=TMP, lw=1.35)     # s_t -> M2
    ax.text(5.68, 2.27, "$s_t$", fontsize=FS_TINY, color=TMP, ha="left", va="center",
            zorder=6, fontweight="bold")

    # ------------------------------------------------ output (far right)
    container(ax, 6.72, 0.44, 1.56, 4.14, OUT_BAD, "DETECTION")
    ax.text(7.47, 4.32, r"flag if $\tilde{s}_t \geq \tau$", fontsize=FS_FORM, color=INK,
            ha="center", va="center", fontweight="bold", zorder=5)
    gloss(ax, 7.47, 4.10, "$\\tau$: the alarm threshold", fs=FS_TINY)
    if have:
        frame_img(ax, normal, 7.05, 3.74, 0.038, border=OUT_OK, lw=1.3)
        frame_img(ax, anom, 7.47, 3.74, 0.038, border=OUT_BAD, lw=1.9)
        frame_img(ax, normal, 7.89, 3.74, 0.038, border=OUT_OK, lw=1.3)
    gloss(ax, 7.47, 3.44, "a score for every frame", fs=FS_TINY)

    box(ax, 6.84, 1.86, 1.26, 1.34, "#FFFFFF", MUTED, lw=1.1, r=0.03)
    ax.text(7.47, 3.04, "not used", fontsize=FS_BOX, color=MUTED, ha="center", va="center",
            fontweight="bold", zorder=5)
    ax.text(7.47, 2.52, "scene sentence\ncaptioning model\ntarget-site data\ngradient updates",
            fontsize=FS_TINY, color=INK, ha="center", va="center", linespacing=1.55, zorder=5)

    box(ax, 6.84, 0.62, 1.26, 1.04, VIS_BG, VIS, lw=1.1, r=0.03)
    ax.text(7.47, 1.46, "run-time cost", fontsize=FS_TINY, color=VIS, ha="center", va="center",
            fontweight="bold", zorder=5)
    ax.text(7.47, 1.04, "one image pass per\nsampled frame;\ntext encoded once", fontsize=FS_TINY,
            color=INK, ha="center", va="center", linespacing=1.45, zorder=5)

    # video -> M1, fixed text -> M3
    arrow(ax, (1.46, 3.60), (1.74, 3.90), color=MUTED)
    arrow(ax, (1.46, 1.89), (1.74, 2.02), color=CALM, lw=1.35)
    # M2 -> detection, routed round the bottom so it enters at the decision rule
    arrow(ax, (6.52, 0.68), (6.61, 0.68), color=TMP)
    arrow(ax, (6.61, 0.68), (6.61, 4.32), color=TMP, lw=1.0)
    arrow(ax, (6.61, 4.32), (6.84, 4.32), color=TMP)

    ax.text(W / 2, 0.28, "New camera = no new data   ·   the same 59 phrases   ·   "
                         "no labels   ·   no gradients   ·   every model frozen",
            fontsize=FS_BODY, color=VIS, ha="center", fontweight="bold")
    ax.text(W / 2, 0.13, "Evaluation: frame-level AUROC (AP on XD-Violence); ShanghaiTech uses every 2nd "
                         "frame, w = 31 and per-clip normalisation, its benchmark protocol",
            fontsize=FS_TINY, color=INK, ha="center", fontweight="medium")

    check_overflow(fig, ax)
    figsave.save(fig, out, facecolor=PAPER)
    plt.close(fig)
    return out


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser()
    p.add_argument("--assets", default=os.path.join(root, "figure_assets"))
    p.add_argument("--out", default=os.path.join(root, "docs", "09_paper", "dazvad_architecture_v3_paper.png"))
    a = p.parse_args()
    print("saved:", build(a.assets, a.out))
