# -*- coding: utf-8 -*-
"""DA-ZVAD architecture figure -- compact landscape version.

Matches the layout of the reference the supervisor supplied: input at the left,
two parallel branches through the middle, fusion and output at the right, with
dashed colour-coded containers, a snowflake on every frozen module, real video
frames, and mathematical notation for the intermediate quantities.

Revision (supervisor review, Phase 2):

  * Contrast. Every label was drawn in a mid-grey (#5F6E70) or a mid-orange at
    5.2-6.5pt, which survives a paper column but washes out under a projector.
    All text colours are darkened to at least ~7:1 on white and the type scale
    is raised ~40%.

  * M2 is now its own labelled container. It was previously a single box inside
    the FRAME SCORING group, so it did not read as a module at all next to the
    dashed panels around M1, M3 and M4 -- the supervisor's report was that M2
    was missing from the figure, which is a fair reading of what was drawn.

  * Every module carries its actual operation (the softmax with CLIP's logit
    scale, the centred moving average with its window) plus one plain-language
    line saying what that operation does, so the figure can be read by someone
    who does not already know the method.

Sized so that it prints at close to 1:1 in a single-column layout rather than
being scaled down until the labels stop being readable. On a slide, place it at
~11.5in wide: the type is designed to survive that enlargement.

Usage:  python scripts/make_architecture.py [--assets DIR] [--out PATH]
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")

import figsave
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# Palette. Every colour used for TEXT is held at >= ~7:1 contrast on white;
# the pale *_BG values are fills only and never carry type.
INK, MUTED = "#101718", "#394547"
VIS, VIS_BG = "#0A5B58", "#E2F1EF"
CTX, CTX_BG = "#8F4413", "#FAEBDE"
TMP, TMP_BG = "#2B4680", "#E4EAF6"
RSN, RSN_BG = "#553A7D", "#EDE8F6"
OUT_OK, OUT_BAD = "#186B3A", "#96261C"
GOLD = "#7A5C00"
PAPER = "#FFFFFF"

W, H = 8.40, 4.95
FROZEN = "❄"

# Type scale -- one place, so the whole figure can be re-tuned at once.
FS_PANEL = 9.2      # dashed-container labels
FS_TITLE = 9.6      # module titles
FS_SUB = 7.8        # module subtitles
FS_BOX = 8.5        # titles inside plain boxes
FS_BODY = 8.0       # ordinary body text
FS_MATH = 9.4       # pills carrying a symbol
FS_FORM = 8.3       # formulae
FS_GLOSS = 7.4      # the plain-language lines
FS_TINY = 7.0


# Rounded boxes are recorded alongside the dashed panels; a subtitle running
# past its own box is the overflow that is easiest to miss and was the one that
# actually shipped.
BOXES = []


def box(ax, x, y, w, h, fc, ec, lw=1.0, r=0.045, z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0.010,rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))
    BOXES.append((x, y, w, h))


# Every dashed panel records its bounds so check_overflow can test the labels
# that sit inside it. Eyeballing a thumbnail does not reliably catch a line that
# runs a few points past a panel edge, and at this type scale that is exactly
# the failure mode.
CONTAINERS = []


def container(ax, x, y, w, h, ec, label):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=ec,
                           linewidth=1.1, linestyle=(0, (3.5, 2.2)), zorder=1))
    ax.text(x + 0.09, y + h - 0.05, label, fontsize=FS_PANEL, color=ec,
            fontweight="bold", va="center", zorder=6,
            bbox=dict(facecolor=PAPER, edgecolor="none", pad=1.6))
    CONTAINERS.append((x, y, w, h, label))


def check_overflow(fig, ax, pad=0.015):
    """Warn about any text whose rendered box runs outside its dashed panel.

    Each text is assigned to the panel containing its centre, then its full
    extent is compared against that panel. The panel's own title is skipped: it
    is drawn deliberately straddling the top edge.
    """
    fig.canvas.draw()
    inv = ax.transData.inverted()
    panel_labels = {lab for *_r, lab in CONTAINERS}
    problems = []
    for t in ax.texts:
        s = t.get_text().strip()
        if not s:
            continue
        bb = t.get_window_extent(fig.canvas.get_renderer())
        (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

        # A panel's own title straddles the top edge by design, so only its
        # width is meaningful -- but that width is exactly what overflowed.
        if s in panel_labels:
            for px, py, pw, ph, lab in CONTAINERS:
                if lab == s and x1 > px + pw - pad:
                    problems.append((lab, "panel title too wide", s[:46]))
            continue

        # Smallest enclosing region wins: a rounded box before the dashed panel
        # it sits in, since the box is the tighter constraint.
        region = None
        for bx, by, bw, bh in BOXES:
            if bx <= cx <= bx + bw and by <= cy <= by + bh:
                region = (bx, by, bw, bh, "box")
                break
        if region is None:
            for px, py, pw, ph, lab in CONTAINERS:
                if px <= cx <= px + pw and py <= cy <= py + ph:
                    region = (px, py, pw, ph, lab)
                    break
        if region is None:
            continue
        rx, ry, rw, rh, lab = region
        over = []
        if x0 < rx - pad:        over.append("left")
        if x1 > rx + rw + pad:   over.append("right")
        if y0 < ry - pad:        over.append("bottom")
        if y1 > ry + rh + pad:   over.append("top")
        if over:
            problems.append((lab, "/".join(over), s[:46]))
    for lab, side, s in problems:
        print("  OVERFLOW: %-24s %-12s %r" % (lab, side, s))
    if not problems:
        print("  panels: no text overflows its container")
    return problems


def arrow(ax, p, q, color=INK, lw=1.15, rad=0.0, z=5):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8.5,
                                 linewidth=lw, color=color, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1, shrinkB=1))


def module(ax, x, y, w, h, title, sub, color, bg, frozen=True, ts=FS_TITLE):
    box(ax, x, y, w, h, bg, color, lw=1.3)
    if frozen:
        ax.text(x + 0.085, y + h - 0.085, FROZEN, fontsize=7.4, color=color,
                va="center", ha="center", zorder=5)
    ax.text(x + w / 2, y + h * (0.63 if sub else 0.5), title, fontsize=ts,
            fontweight="bold", color=INK, ha="center", va="center", zorder=5)
    if sub:
        # Upright and in the body ink, not italic grey. Figure labels are read
        # at a glance rather than as prose, so weight helps here in a way it
        # would not in a paragraph of running text.
        ax.text(x + w / 2, y + h * 0.26, sub, fontsize=FS_SUB, color=INK,
                ha="center", va="center", zorder=5, fontweight="medium")


def pill(ax, cx, cy, w, h, label, color, bg, fs=FS_MATH):
    box(ax, cx - w / 2, cy - h / 2, w, h, bg, color, lw=1.25, r=0.05, z=4)
    ax.text(cx, cy, label, fontsize=fs, fontweight="bold", color=INK,
            ha="center", va="center", zorder=5)


def gloss(ax, cx, y, text, color=INK, fs=FS_GLOSS):
    """One plain-language line.

    These carry the explanation a non-specialist reads, so they were the worst
    thing on the figure to have set in italic at the smallest size in a
    secondary grey. Now upright, body ink, medium weight -- matching how
    published framework figures label their parts.
    """
    ax.text(cx, y, text, fontsize=fs, color=color, ha="center", va="center",
            fontweight="medium", zorder=6)


def frame_img(ax, path, cx, cy, zoom, border=None, lw=1.5):
    im = OffsetImage(plt.imread(path), zoom=zoom)
    ax.add_artist(AnnotationBbox(
        im, (cx, cy), frameon=border is not None, pad=0.0, zorder=4,
        bboxprops=dict(edgecolor=border, linewidth=lw) if border else None))


def smoothing_inset(ax, x, y, w, h):
    """A schematic of what M2 does: a jagged per-frame score and its centred
    moving average, with the event shaded. Illustrative, not measured data --
    its job is to show that smoothing turns spikes into one sustained bump."""
    ins = ax.inset_axes([x, y, w, h], transform=ax.transData, zorder=6)
    rng = np.random.default_rng(7)
    n = 220
    t = np.arange(n)
    base = 0.30 + 0.42 * np.exp(-0.5 * ((t - 128) / 21.0) ** 2)
    raw = np.clip(base + rng.normal(0, 0.085, n), 0, 1)
    k = 31
    sm = np.convolve(np.pad(raw, k // 2, mode="edge"), np.ones(k) / k, "same")[k // 2: k // 2 + n]
    ins.axvspan(100, 156, color="#F2DFC4", zorder=0)
    ins.plot(t, raw, lw=0.6, color="#9AA3A4", zorder=2)
    ins.plot(t, sm, lw=1.5, color=TMP, zorder=3)
    ins.set_xticks([]); ins.set_yticks([])
    ins.set_ylim(0, 1.02)
    for s in ins.spines.values():
        s.set_edgecolor(MUTED); s.set_linewidth(0.7)
    ins.set_facecolor("#FFFFFF")
    ins.text(0.03, 0.93, "$s_t$ raw", transform=ins.transAxes, fontsize=7.0,
             color="#4A5456", va="top", fontweight="bold")
    ins.text(0.97, 0.93, "$\\tilde{s}_t$ smoothed", transform=ins.transAxes,
             fontsize=7.0, color=TMP, va="top", ha="right", fontweight="bold")


def build(assets: str, out: str) -> str:
    fa = os.path.join(assets, "frames", "shanghaitech")
    normal = os.path.join(fa, "01_0014_normal_00017.jpg")
    anom = os.path.join(fa, "01_0014_anomaly_00095.jpg")
    have = os.path.isfile(normal) and os.path.isfile(anom)

    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    fig.patch.set_facecolor(PAPER)

    ax.add_patch(Rectangle((0.05, 0.05), W - 0.10, H - 0.10, facecolor="none",
                           edgecolor="#B39A3C", linewidth=1.0,
                           linestyle=(0, (4.5, 2.8)), zorder=0))

    # ------------------------------------------------ inputs (left)
    # Two distinct inputs, kept separate because they are: the camera supplies
    # frames to M1, the operator supplies one sentence to M3. Drawing a single
    # "input" feeding both would misdescribe where the adaptation comes from.
    container(ax, 0.12, 2.72, 1.34, 1.86, MUTED, "VIDEO")
    if have:
        for dx in (0.0, 0.055, 0.11):
            frame_img(ax, normal, 0.62 + dx, 3.95 + dx * 0.5, 0.070,
                      border="#7E8A8B", lw=0.7)
        frame_img(ax, anom, 0.79, 3.16, 0.081, border="#C08A00", lw=1.8)
    ax.text(0.79, 3.56, "$f_1 \\ldots f_T$", fontsize=FS_BODY, color=MUTED,
            ha="center")
    ax.text(0.79, 2.86, "current frame $f_t$", fontsize=FS_GLOSS, color=GOLD,
            ha="center", fontweight="bold")

    container(ax, 0.12, 0.72, 1.34, 1.72, MUTED, "OPERATOR")
    box(ax, 0.28, 1.34, 1.02, 0.70, "#FBFBF9", MUTED, lw=1.0)
    ax.text(0.79, 1.86, "writes one", fontsize=FS_BODY, color=INK, ha="center",
            va="center", zorder=5)
    ax.text(0.79, 1.68, "sentence", fontsize=FS_BODY, color=INK, ha="center",
            va="center", zorder=5)
    ax.text(0.79, 1.48, "$c$", fontsize=9.6, color=CTX, ha="center",
            va="center", fontweight="bold", zorder=5)
    gloss(ax, 0.79, 1.12, "the only input that", fs=FS_TINY)
    gloss(ax, 0.79, 1.00, "changes per site", fs=FS_TINY)
    ax.text(0.79, 0.84, f"{FROZEN} = frozen", fontsize=FS_TINY,
            color=VIS, ha="center", fontweight="bold")

    # ------------------------------------------------ M1 (top branch)
    container(ax, 1.60, 2.72, 2.92, 1.86, VIS, "M1 · VISUAL SCORING")
    module(ax, 1.78, 3.78, 2.56, 0.58, "CLIP image encoder",
           "$E_I$ · ViT-L/14 · LAION-2B", VIS, VIS_BG)
    arrow(ax, (3.06, 3.76), (3.06, 3.62), color=VIS)
    pill(ax, 3.06, 3.44, 1.72, 0.34, "$v_t = E_I(f_t) \\in \\mathbb{R}^{768}$",
         VIS, "#FFFFFF", fs=8.0)
    gloss(ax, 3.06, 3.14, "each frame becomes 768 numbers of unit length;")
    gloss(ax, 3.06, 2.98, "the direction encodes what it contains")

    # ------------------------------------------------ M3 (bottom branch)
    container(ax, 1.60, 0.44, 2.92, 2.10, CTX, "M3 · VERBALISED CONTEXT")
    box(ax, 1.78, 1.92, 2.56, 0.48, CTX_BG, CTX, lw=1.3)
    ax.text(3.06, 2.26, "scene sentence  $c$", fontsize=FS_BOX, color=CTX,
            ha="center", va="center", fontweight="bold", zorder=5)
    ax.text(3.06, 2.06, '"a campus walkway with pedestrians"', fontsize=FS_BODY,
            color=INK, ha="center", va="center", style="italic", zorder=5)

    box(ax, 1.78, 1.16, 1.22, 0.52, "#FFFFFF", CTX, lw=1.15, r=0.03)
    ax.text(2.39, 1.53, "$P^{+}$ normal", fontsize=FS_BOX, color=CTX,
            ha="center", va="center", fontweight="bold", zorder=5)
    ax.text(2.39, 1.32, "prompts $+\\, c$", fontsize=FS_BODY, color=INK,
            ha="center", va="center", zorder=5)

    box(ax, 3.12, 1.16, 1.22, 0.52, "#F4F2EE", CTX, lw=1.15, r=0.03)
    ax.text(3.73, 1.53, "$P^{-}$ abnormal", fontsize=FS_BOX, color=CTX,
            ha="center", va="center", fontweight="bold", zorder=5)
    ax.text(3.73, 1.32, "prompts only", fontsize=FS_BODY, color=INK,
            ha="center", va="center", zorder=5)

    arrow(ax, (2.39, 1.90), (2.39, 1.72), color=CTX, lw=1.35)
    ax.text(2.56, 1.82, "$c$ enters $P^{+}$ only", fontsize=FS_TINY, color=CTX,
            ha="left", va="center", zorder=6, fontweight="bold")

    arrow(ax, (2.39, 1.14), (2.39, 1.02), color=CTX)
    arrow(ax, (3.73, 1.14), (3.73, 1.02), color=CTX)
    module(ax, 1.78, 0.58, 2.56, 0.42, "CLIP text encoder  $E_T$", "",
           CTX, CTX_BG, ts=FS_TITLE)

    # ------------------------------------------------ fusion (right-middle)
    container(ax, 4.66, 2.34, 1.86, 2.24, INK, "FRAME SCORING")
    pill(ax, 5.14, 4.20, 0.66, 0.32, "$e^{+}$", CTX, "#FFFFFF")
    pill(ax, 6.04, 4.20, 0.66, 0.32, "$e^{-}$", CTX, "#EDE9E3")
    box(ax, 4.82, 2.98, 1.54, 0.80, "#FFFFFF", INK, lw=1.25)
    ax.text(5.59, 3.63, "softmax over", fontsize=FS_BODY, color=MUTED,
            ha="center", zorder=5)
    ax.text(5.59, 3.43,
            r"$\lambda\langle v_t,e^{+}\rangle$ ,  $\lambda\langle v_t,e^{-}\rangle$",
            fontsize=FS_FORM, color=INK, ha="center", va="center", zorder=5)
    ax.text(5.59, 3.18, "$\\rightarrow\\; s_t = P(\\mathrm{abnormal})$",
            fontsize=FS_FORM, color=INK, ha="center", va="center",
            fontweight="bold", zorder=5)
    gloss(ax, 5.59, 2.66, "$\\lambda$: CLIP logit scale $\\approx$ 100", fs=FS_TINY)

    # ------------------------------------------------ M2 (its own module)
    container(ax, 4.66, 0.44, 1.86, 1.74, TMP, "M2 · SMOOTHING")
    module(ax, 4.82, 1.58, 1.54, 0.44, "centred moving average",
           "window $w = 31$ frames", TMP, TMP_BG, frozen=False, ts=7.6)
    ax.text(5.59, 1.36,
            r"$\tilde{s}_t=\frac{1}{w}\sum_{i=t-15}^{\,t+15} s_i$",
            fontsize=8.4, color=INK, ha="center", va="center", zorder=6)
    smoothing_inset(ax, 4.86, 0.72, 1.46, 0.40)
    gloss(ax, 5.59, 0.64, "averages over $\\pm$15 frames ($\\approx$1 s):", fs=6.4)
    gloss(ax, 5.59, 0.52, "one spike cannot raise an alarm", fs=6.4)

    # ------------------------------------------------ wiring
    arrow(ax, (4.38, 3.44), (4.78, 3.36), color=VIS)              # v_t -> scoring
    arrow(ax, (4.38, 0.79), (4.59, 0.79), color=CTX)              # E_T out
    arrow(ax, (4.59, 0.79), (4.59, 4.20), color=CTX, lw=1.0)      # up the side
    arrow(ax, (4.59, 4.20), (4.78, 4.20), color=CTX)
    arrow(ax, (5.14, 4.04), (5.34, 3.84), color=CTX, rad=-0.15)
    arrow(ax, (6.04, 4.04), (5.84, 3.84), color=CTX, rad=0.15)
    arrow(ax, (5.59, 2.32), (5.59, 2.16), color=TMP, lw=1.35)     # s_t -> M2
    ax.text(5.68, 2.25, "$s_t$", fontsize=FS_TINY, color=TMP, ha="left",
            va="center", zorder=6, fontweight="bold")

    # ------------------------------------------------ output (far right)
    container(ax, 6.72, 0.44, 1.56, 4.14, RSN, "DETECTION · M4")
    ax.text(7.47, 4.32, r"flag if $\tilde{s}_t \geq \tau$", fontsize=FS_FORM,
            color=INK, ha="center", va="center", fontweight="bold", zorder=5)
    gloss(ax, 7.47, 4.10, "$\\tau$ chosen on held-out clips", fs=FS_TINY)
    if have:
        frame_img(ax, normal, 7.05, 3.74, 0.038, border=OUT_OK, lw=1.3)
        frame_img(ax, anom, 7.47, 3.74, 0.038, border=OUT_BAD, lw=1.9)
        frame_img(ax, normal, 7.89, 3.74, 0.038, border=OUT_OK, lw=1.3)

    module(ax, 6.86, 2.84, 1.22, 0.44, "LLaVA-1.5", "4-bit · frozen", RSN, RSN_BG,
           ts=FS_TITLE)
    arrow(ax, (7.47, 3.30), (7.47, 3.20), color=RSN)
    arrow(ax, (7.47, 2.80), (7.47, 2.68), color=RSN)

    box(ax, 6.82, 1.34, 1.30, 1.30, "#FFFFFF", RSN, lw=1.15, r=0.03)
    ax.text(7.47, 2.50, "explanation", fontsize=FS_BOX, color=RSN, ha="center",
            va="center", fontweight="bold", zorder=5)
    ax.text(7.47, 1.94, '"A cyclist is\nriding through\na pedestrian\nwalkway."',
            fontsize=FS_BODY, color=INK, ha="center", va="center", style="italic",
            linespacing=1.5, zorder=5)
    gloss(ax, 7.47, 1.14, "grounded by $c$ — the same", CTX, fs=FS_TINY)
    gloss(ax, 7.47, 1.02, "sentence steers the words", CTX, fs=FS_TINY)

    # video -> M1, operator -> M3: two separate paths, never crossing
    arrow(ax, (1.46, 3.60), (1.74, 3.90), color=MUTED)
    arrow(ax, (1.46, 1.70), (1.74, 2.10), color=CTX, lw=1.35)
    # M2 -> M4, routed round the bottom so it enters at the decision rule
    arrow(ax, (6.52, 0.68), (6.61, 0.68), color=TMP)
    arrow(ax, (6.61, 0.68), (6.61, 4.32), color=TMP, lw=1.0)
    arrow(ax, (6.61, 4.32), (6.84, 4.32), color=TMP)

    ax.text(W / 2, 0.28, "Deploying to a new site = editing $c$   ·   "
                         "no target data   ·   no gradients   ·   every model frozen",
            fontsize=FS_BODY, color=CTX, ha="center", fontweight="bold")
    ax.text(W / 2, 0.13, "Evaluation: scores min–max normalised per clip before "
                         "pooling, following the benchmark protocol",
            fontsize=FS_TINY, color=INK, ha="center", fontweight="medium")

    check_overflow(fig, ax)
    figsave.save(fig, out, facecolor=PAPER)
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
