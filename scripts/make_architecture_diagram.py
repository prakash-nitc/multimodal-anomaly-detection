# -*- coding: utf-8 -*-
"""DA-ZVAD architecture diagram."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BLUE, AQUA, ORANGE = "#2a78d6", "#1baf7a", "#eb6834"
INK, MUTED, LINE = "#1a1a19", "#6b6a63", "#c9c8c0"
NAVY = "#1A5276"

fig, ax = plt.subplots(figsize=(14.5, 8.2))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

def box(x, y, w, h, title, sub=None, color=BLUE, fill="white", lw=2.0,
        tsize=11, ssize=8.6, tcolor=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.6",
                                linewidth=lw, edgecolor=color, facecolor=fill, zorder=3))
    ax.text(x + w/2, y + h*(0.62 if sub else 0.5), title, ha="center", va="center",
            fontsize=tsize, fontweight="bold", color=tcolor or INK, zorder=4)
    if sub:
        ax.text(x + w/2, y + h*0.27, sub, ha="center", va="center",
                fontsize=ssize, color=MUTED, zorder=4, linespacing=1.5)

def arrow(x1, y1, x2, y2, color=INK, style="-|>", lw=1.8, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=15, linewidth=lw, color=color,
                                 linestyle=ls, zorder=2,
                                 connectionstyle=f"arc3,rad={rad}"))

# ---------------- title ----------------
ax.text(50, 89, "DA-ZVAD — Domain-Adaptive Zero-shot Video Anomaly Detection",
        ha="center", fontsize=16, fontweight="bold", color=NAVY)
ax.text(50, 84.4, "Every model is frozen. Domain adaptation is performed by text, not by weight updates.",
        ha="center", fontsize=10.5, color=MUTED, style="italic")

# ---------------- main pipeline row ----------------
Y, H = 60, 15
box(3, Y, 13, H, "Input", "video frames\nor images", color=MUTED, fill="#f7f6f2", lw=1.6)
box(21, Y, 17, H, "M1 · Visual encoder", "frozen CLIP ViT-L/14\nframe → anomaly score", color=BLUE)
box(43, Y, 17, H, "M2 · Temporal", "training-free\nscore smoothing", color=BLUE)
box(65, Y, 13, H, "Detection", "score ≥ τ\n→ flag event", color=BLUE)
box(82, Y, 15, H, "M4 · Reasoning", "LLaVA-1.5-7B (4-bit)\nwhy is it anomalous?", color=BLUE)

for x1, x2 in [(16, 21), (38, 43), (60, 65), (78, 82)]:
    arrow(x1, Y + H/2, x2, Y + H/2)

# frozen tags
for x, w in [(21, 17), (43, 17), (82, 15)]:
    ax.text(x + w/2, Y + H + 1.6, "frozen · no training", ha="center",
            fontsize=7.6, color=BLUE, fontweight="bold")

# ---------------- M3: the adaptation mechanism ----------------
MY, MH = 31, 15
box(21, MY, 56, MH, "M3 · Verbalized scene context   —   the domain-adaptation mechanism",
    '"a university campus walkway with pedestrians"',
    color=ORANGE, fill="#fdf4ef", lw=2.6, tsize=12, ssize=11)
ax.text(49, MY - 3.2,
        "Deploying to a new scene = editing this sentence.  No target data, no gradients, no retraining.",
        ha="center", fontsize=9.6, color=ORANGE, fontweight="bold")

# arrows from M3 up into M1 and M4
arrow(29.5, MY + MH, 29.5, Y, color=ORANGE, lw=2.2)
arrow(70, MY + MH, 86, Y, color=ORANGE, lw=2.2, rad=-0.16)
ax.text(31.3, (MY + MH + Y)/2, "grounds the\nprompt ensembles", fontsize=8, color=ORANGE,
        va="center", ha="left", linespacing=1.4)
ax.text(76.5, MY + MH - 0.5, "grounds the\nexplanation", fontsize=8, color=ORANGE,
        va="top", ha="left", linespacing=1.4)

# ---------------- outputs ----------------
box(86, 8.5, 14, 12, "Output", "frame-level score\n+ explanation", color=AQUA,
    fill="#f0faf6", lw=1.8)
arrow(93, Y, 93, 20.5, color=AQUA, lw=1.8)

# ---------------- experiment inset ----------------
ax.add_patch(FancyBboxPatch((3, 8.5), 34, 14, boxstyle="round,pad=0.6,rounding_size=1.6",
                            linewidth=1.4, edgecolor=LINE, facecolor="#fafaf7", zorder=3))
ax.text(20, 19.6, "The experiment: vary ONLY the M3 text", ha="center",
        fontsize=9.8, fontweight="bold", color=INK, zorder=4)
variants = [("none", "no context"), ("generic", '"a generic scene"'),
            ("matched", "correct scene"), ("mismatched", "WRONG domain ← falsifies")]
for i, (nm, desc) in enumerate(variants):
    yy = 16.4 - i*2.5
    ax.text(6, yy, f"• {nm}", fontsize=8.4, color=ORANGE if nm == "mismatched" else INK,
            fontweight="bold" if nm == "mismatched" else "normal", zorder=4)
    ax.text(17, yy, desc, fontsize=8.2,
            color=ORANGE if nm == "mismatched" else MUTED, zorder=4)

# ---------------- framing note ----------------
ax.text(50, 2.6,
        "Targets concept shift  p(y|x)  — the same input carries a different label across domains "
        "(a vehicle: normal on a road, anomalous on a walkway),\nthe shift type classical domain adaptation "
        "explicitly sets aside while aligning  p(x).",
        ha="center", fontsize=9, color=MUTED, linespacing=1.6)

fig.tight_layout()
out = r"p:\Research\multimodal-anomaly-detection\docs\06_presentations\dazvad_architecture.png"
fig.savefig(out, dpi=190, bbox_inches="tight", facecolor="white")
print("saved:", out)
