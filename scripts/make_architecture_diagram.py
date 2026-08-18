# -*- coding: utf-8 -*-
"""DA-ZVAD architecture diagram, sized for a single-column paper.

Designed at ~6.8in wide so that \\includegraphics[width=\\linewidth] in a
single-column A4 layout applies almost no downscaling -- the previous 14.5in
version was shrunk ~2.3x and became unreadable in print.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BLUE, AQUA, ORANGE = "#2a78d6", "#1baf7a", "#d1541f"
INK, MUTED = "#1a1a19", "#5f5e58"
NAVY = "#14405c"

FIG_W, FIG_H = 6.8, 4.15
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100 * FIG_H / FIG_W)      # keeps geometry square
ax.set_aspect("equal")
ax.axis("off")

def box(x, y, w, h, title, sub, edge, face="white", lw=1.6,
        tsize=8.6, ssize=7.0, tcol=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.5,rounding_size=1.3",
                                linewidth=lw, edgecolor=edge, facecolor=face,
                                zorder=3))
    ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center",
            fontsize=tsize, fontweight="bold", color=tcol or INK, zorder=4)
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
            fontsize=ssize, color=MUTED, zorder=4, linespacing=1.45)

def arrow(x1, y1, x2, y2, color=INK, lw=1.5, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>", mutation_scale=11,
                                 linewidth=lw, color=color, zorder=2,
                                 connectionstyle=f"arc3,rad={rad}"))

# ---------------- headings ----------------
ax.text(50, 54.0, "DA-ZVAD", ha="center", fontsize=12,
        fontweight="bold", color=NAVY)
ax.text(50, 49.3,
        "every model frozen  ·  domain adaptation carried entirely by text",
        ha="center", fontsize=8.2, color=MUTED, style="italic")

# ---------------- pipeline row ----------------
# gaps: 5, 5, 13, 5 -- the wide one leaves room for the threshold label
Y, H, W = 29, 13, 14
box(1,  Y, W, H, "Input",        "video frames\nor images",    MUTED, "#f6f5f1", lw=1.3)
box(20, Y, W, H, "M1 Encoder",   "frozen CLIP\nViT-L/14",      BLUE)
box(39, Y, W, H, "M2 Temporal",  "moving average\nwindow $w$", BLUE)
box(66, Y, W, H, "M4 Reasoning", "frozen LLaVA\n4-bit",        BLUE)
box(85, Y, W, H, "Output",       "score +\nexplanation",       AQUA, "#f1faf6", lw=1.4)

mid = Y + H / 2
for x1, x2 in [(15, 20), (34, 39), (53, 66), (80, 85)]:
    arrow(x1, mid, x2, mid)

# threshold label sits in the wide gap, clear of both boxes
ax.text(59.5, mid + 2.4, r"score $\geq \tau$", ha="center", va="bottom",
        fontsize=7.0, color=INK)
ax.text(59.5, mid - 2.6, "peak frame", ha="center", va="top",
        fontsize=6.4, color=MUTED)

# ---------------- M3: the adaptation mechanism ----------------
MY, MH = 6, 12
ax.add_patch(FancyBboxPatch((20, MY), 60, MH,
                            boxstyle="round,pad=0.5,rounding_size=1.3",
                            linewidth=2.0, edgecolor=ORANGE,
                            facecolor="#fdf3ed", zorder=3))
ax.text(50, MY + MH * 0.66, "M3  ·  Verbalised domain context",
        ha="center", va="center", fontsize=9.0, fontweight="bold",
        color=ORANGE, zorder=4)
ax.text(50, MY + MH * 0.26,
        '"a university campus walkway with pedestrians"',
        ha="center", va="center", fontsize=7.4, color=MUTED,
        style="italic", zorder=4)

# arrows from M3 up into M1 and M4 (aligned with their box centres)
arrow(27, MY + MH, 27, Y, color=ORANGE, lw=1.7)
arrow(73, MY + MH, 73, Y, color=ORANGE, lw=1.7)
ax.text(28.5, (MY + MH + Y) / 2, "grounds the\nprompt ensembles",
        fontsize=6.6, color=ORANGE, va="center", ha="left", linespacing=1.4)
ax.text(74.5, (MY + MH + Y) / 2, "grounds the\nexplanation",
        fontsize=6.6, color=ORANGE, va="center", ha="left", linespacing=1.4)

# ---------------- footer ----------------
ax.text(50, MY - 2.4,
        "deploying to a new scene = editing this sentence"
        "  ·  no target data, no gradients",
        ha="center", va="top", fontsize=7.0, color=ORANGE, fontweight="bold")

fig.tight_layout(pad=0.25)
out = r"p:\Research\multimodal-anomaly-detection\docs\09_paper\dazvad_architecture.png"
fig.savefig(out, dpi=400, bbox_inches="tight", facecolor="white")
print("saved:", out, f"({FIG_W}x{FIG_H} in @ 400 dpi)")
