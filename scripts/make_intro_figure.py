# -*- coding: utf-8 -*-
"""Introductory figure: what video anomaly detection is, in one picture.

Three panels, all real data:
  (a) an ordinary moment on ShanghaiTech camera 01 (clip 01_0014, frame 17),
  (b) the same camera, same view, a cyclist on the pedestrian walkway
      (clip 01_0014, frame 95) -- ShanghaiTech's textbook anomaly,
  (c) a detector's score over one clip (04_0004, the worked example used in the
      paper), with the true anomaly shaded.

Panels (a) and (b) are the same camera so that the only thing that differs is
the event. Panel (c) is a different clip because it is the one whose scores are
saved locally; its title says so.

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
FR = os.path.join(ROOT, "figure_assets", "frames", "shanghaitech")


def frame_panel(ax, path, colour, head, sub):
    im = np.asarray(Image.open(path).convert("RGB"))
    ax.imshow(im)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(colour); sp.set_linewidth(4)
    ax.set_title(head, fontsize=12.5, fontweight="bold", color=colour, pad=6)
    ax.text(0.5, -0.06, sub, transform=ax.transAxes, ha="center", va="top",
            fontsize=10, color=INK)
    return im.shape


def build(out_dir):
    fig = plt.figure(figsize=(13.2, 4.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.24,
                          left=0.012, right=0.985, top=0.87, bottom=0.27)

    a = fig.add_subplot(gs[0])
    frame_panel(a, os.path.join(FR, "01_0014_normal_00017.jpg"), OK,
                "NORMAL", "An ordinary moment on a campus walkway")

    b = fig.add_subplot(gs[1])
    frame_panel(b, os.path.join(FR, "01_0014_anomaly_00095.jpg"), BAD,
                "ANOMALY", "Same camera: a cyclist on a pedestrian-only walkway")
    # the cyclist sits at roughly (357, 135) in this 640x359 frame
    b.add_patch(Ellipse((357, 135), 120, 110, fill=False, edgecolor=BAD, lw=2.6))

    c = fig.add_subplot(gs[2])
    z = np.load(os.path.join(ROOT, "figure_assets", "worked_example.npz"),
                allow_pickle=True)
    raw, lab = z["scores_matched"], z["labels"].astype(int)
    sm = np.convolve(raw, np.ones(31) / 31, mode="same")
    sm = (sm - sm.min()) / (sm.max() - sm.min() + 1e-12)
    idx = np.where(lab == 1)[0]
    if len(idx):
        c.axvspan(idx[0], idx[-1], color=BAD, alpha=0.13, lw=0)
        c.text((idx[0] + idx[-1]) / 2, 1.08, "true anomaly", ha="center",
               fontsize=9.5, color=BAD, fontweight="bold")
    c.plot(sm, color=TMP, lw=2.2)
    c.axhline(0.45, color=MUTED, lw=1.1, ls=(0, (4, 3)))
    c.text(len(sm) - 4, 0.47, "alarm threshold", ha="right", va="bottom",
           fontsize=9, color=MUTED)
    c.set_xlim(0, len(sm)); c.set_ylim(-0.04, 1.2)
    c.set_yticks([0, 0.5, 1.0])
    c.set_xlabel("time (frames)"); c.set_ylabel("anomaly score")
    c.grid(axis="y", color=GRID, lw=0.5); c.set_axisbelow(True)
    c.set_title("THE DETECTOR'S OUTPUT", fontsize=12.5, fontweight="bold",
                color=TMP, pad=6)
    c.text(0.5, -0.24, "A score for every frame; high = unusual  (ShanghaiTech clip 04_0004)",
           transform=c.transAxes, ha="center", va="top", fontsize=10, color=INK)

    p = os.path.join(out_dir, "fig_intro_anomaly_detection.png")
    figsave.save(fig, p)
    plt.close(fig)
    return p


if __name__ == "__main__":
    out = os.path.join(ROOT, "docs", "09_paper", "figures")
    print("saved:", build(out))
