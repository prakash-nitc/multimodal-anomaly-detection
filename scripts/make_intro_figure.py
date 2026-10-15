# -*- coding: utf-8 -*-
"""Introductory figure for the deck: what video anomaly detection is.

Three panels:
  (a) NORMAL  -- a busy pedestrian crossing seen from above, people crossing as
      usual ("Scramble from above, SHIBUYA SKY", Sei F, CC BY-SA 2.0),
  (b) ANOMALY -- a collision on a crossing, also from above, like a CCTV view
      ("Japanese car accident", Shuets Udono, CC BY-SA 2.0),
  (c) an ILLUSTRATION of a detector's output: a score per frame that rises at
      the unusual moment. Schematic, not measured -- the panel says so.

The two photographs come from Wikimedia Commons and are kept out of git in
figure_assets/web/ (CREDITS.json records title, author, licence and source).
The figure is used only in the presentation, not in the paper, and the slide
carries the attribution both licences require.

Usage:  python scripts/make_intro_figure.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")

import figsave  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
from PIL import Image  # noqa: E402

INK, MUTED = "#101718", "#394547"
OK, BAD, TMP = "#1E8449", "#B03A2E", "#3B5BA5"
GRID = "#D8E0DF"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "axes.edgecolor": "#8E9E9D", "axes.linewidth": 0.9,
    "xtick.color": INK, "ytick.color": INK,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.labelsize": 9.5, "axes.labelweight": "bold",
    "font.weight": "medium",
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "figure_assets", "web")


def crop_16x9(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    th = int(round(w * 9 / 16))
    if th <= h:                                   # too tall: keep the middle band
        top = (h - th) // 2
        im = im.crop((0, top, w, top + th))
    else:                                         # too wide: keep the middle
        tw = int(round(h * 16 / 9))
        left = (w - tw) // 2
        im = im.crop((left, 0, left + tw, h))
    return np.asarray(im)


def frame_panel(ax, img, colour, head, sub):
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(colour); sp.set_linewidth(4)
    ax.set_title(head, fontsize=12.5, fontweight="bold", color=colour, pad=6)
    ax.text(0.5, -0.06, sub, transform=ax.transAxes, ha="center", va="top",
            fontsize=10, color=INK)


def build(out_dir):
    fig = plt.figure(figsize=(13.2, 3.95))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.24,
                          left=0.012, right=0.985, top=0.87, bottom=0.2)

    a = fig.add_subplot(gs[0])
    frame_panel(a, crop_16x9(os.path.join(WEB, "normal_shibuya.jpg")), OK,
                "NORMAL", "People crossing a busy junction, as usual")

    b = fig.add_subplot(gs[1])
    img = crop_16x9(os.path.join(WEB, "anomaly_collision.jpg"))
    frame_panel(b, img, BAD, "ANOMALY", "A collision on the crossing")
    h, w = img.shape[:2]
    # the two cars meet near (0.53 w, 0.28 h) in this photograph
    b.add_patch(Ellipse((0.53 * w, 0.30 * h), 0.24 * w, 0.42 * h, fill=False,
                        edgecolor=BAD, lw=2.8))

    c = fig.add_subplot(gs[2])
    t = np.arange(200)
    rng = np.random.default_rng(7)
    score = 0.10 + 0.03 * rng.standard_normal(len(t))
    event = (t >= 120) & (t <= 150)
    score[event] += 0.75 * np.sin(np.linspace(0, np.pi, event.sum())) ** 0.5
    score = np.convolve(score, np.ones(7) / 7, mode="same").clip(0, 1)
    c.axvspan(120, 150, color=BAD, alpha=0.13, lw=0)
    c.text(135, 1.08, "the unusual moment", ha="center", fontsize=9.5,
           color=BAD, fontweight="bold")
    c.plot(t, score, color=TMP, lw=2.2)
    c.axhline(0.5, color=MUTED, lw=1.1, ls=(0, (4, 3)))
    c.text(4, 0.52, "alarm threshold", ha="left", va="bottom", fontsize=9, color=MUTED)
    c.set_xlim(0, len(t) - 1); c.set_ylim(-0.04, 1.2)
    c.set_yticks([0, 0.5, 1.0]); c.set_xticks([])
    c.set_xlabel("time  →"); c.set_ylabel("anomaly score")
    c.grid(axis="y", color=GRID, lw=0.5); c.set_axisbelow(True)
    c.set_title("WHAT A DETECTOR OUTPUTS", fontsize=12.5, fontweight="bold",
                color=TMP, pad=6)
    c.text(0.5, -0.17, "Illustration: one score per frame; high = unusual",
           transform=c.transAxes, ha="center", va="top", fontsize=10, color=INK)

    p = os.path.join(out_dir, "fig_intro_anomaly_detection.png")
    figsave.save(fig, p)
    plt.close(fig)
    return p


if __name__ == "__main__":
    out = os.path.join(ROOT, "docs", "06_presentations")
    print("saved:", build(out))
