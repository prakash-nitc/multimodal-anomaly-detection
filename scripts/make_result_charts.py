# -*- coding: utf-8 -*-
"""Result charts for the report.

Three charts, each showing a result the tables already state. They exist because
a comparison is easier to see than to read off a row, not to add new claims.

  context_sweep     the four sweep conditions under both fusion rules. The
                    quantity of interest is the distance between matched and
                    mismatched, so it is drawn rather than left to be computed
                    by eye.

  window_ablation   AUROC against temporal window under all three pooling
                    conventions, plus the component ablation. Shows the optimum
                    at w=31 and the fall beyond it, and that the raw-pooled
                    curve never leaves chance.

  within_view       the per-camera gaps against the pooled figure. The pooled
                    bar stands well clear of almost every individual camera,
                    which is the scene-diversity finding in one picture.

Bar charts throughout. Pie charts are poor for comparing magnitudes and every
comparison here is of magnitudes.

Provenance: the within-view chart reads results/runs/analysis/within_view.csv.
The sweep and window figures are computed at w=31, which post-dates the driver's
own window sweep, so they are carried here as constants read off the driver's
cached scores on 19 Aug 2026 (run 2026-08-14_162056_surv_normal). Phase 3 should
write those into a CSV and read them the same way.

Usage:  python scripts/make_result_charts.py [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")

import figsave
import matplotlib.pyplot as plt
import numpy as np

# MUTED was #5F6E70, ~4.4:1 on white -- the same washed-out grey the
# architecture figure carried, and the reason secondary labels read as
# faded in print. Darkened to ~9:1, matching make_architecture.py so every
# figure in the report now uses one ink for secondary text.
INK, MUTED = "#101718", "#394547"
VIS, CTX, TMP = "#0D6B67", "#B4571A", "#3B5BA5"
OK, BAD = "#1E8449", "#B03A2E"
GRID = "#D8E0DF"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "axes.edgecolor": "#8E9E9D", "axes.linewidth": 0.9,
    "xtick.color": INK, "ytick.color": INK,
    "xtick.labelsize": 8.0, "ytick.labelsize": 8.0,
    "axes.labelsize": 8.7, "axes.titlesize": 9.2,
    "axes.labelweight": "bold", "axes.titleweight": "bold",
    "font.weight": "medium",
})

CONDS = ["none", "generic", "matched", "mismatched"]

# ShanghaiTech, all 107 clips, w=31, per-clip normalised micro AUROC.
SWEEP = {
    "both":   [0.7067, 0.6704, 0.6662, 0.6950],
    "normal": [0.7067, 0.6907, 0.7336, 0.6283],
}
WINDOWS = [1, 5, 9, 15, 31, 61]
CURVES = {
    "micro, raw":              [0.4926, 0.5022, 0.5075, 0.5134, 0.5185, None],
    "macro (per clip)":        [0.6143, 0.6276, 0.6365, 0.6482, 0.6699, None],
    "micro, per-clip normed":  [0.6666, 0.6852, 0.6914, 0.7017, 0.7067, 0.6827],
    "matched descriptor":      [0.6744, 0.6924, 0.7014, 0.7132, 0.7336, 0.7210],
}
# Held-out means over five clip-level partitions, with the standard deviation
# across those partitions. Deliberately NOT the full-set column of
# tbl:components: the spread is a property of the held-out estimate, so pairing
# a full-set mean with a held-out error bar would draw an interval around a
# number the interval does not describe.
ABLATION = [
    ("scene-centre only",       0.585, 0.025),
    ("semantic + scene-centre", 0.645, 0.034),
    ("kinematic only",          0.685, 0.015),
    ("semantic + kinematic",    0.711, 0.034),
    ("semantic only",           0.718, 0.036),
]


def context_sweep(out):
    fig, ax = plt.subplots(figsize=(6.9, 2.7))
    x = np.arange(len(CONDS))
    w = 0.36
    b1 = ax.bar(x - w / 2, SWEEP["both"], w, color="#B7C2C1",
                edgecolor="#8895A0", lw=0.7, label="descriptor in BOTH ensembles")
    b2 = ax.bar(x + w / 2, SWEEP["normal"], w, color=CTX, alpha=0.88,
                edgecolor="#8A430F", lw=0.7, label="descriptor in NORMAL only")

    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.004,
                    f"{r.get_height():.3f}", ha="center", va="bottom",
                    fontsize=8.6, color=INK, fontweight="bold")

    ax.axhline(0.5, color=BAD, lw=0.8, ls=(0, (4, 3)))
    ax.text(len(CONDS) - 0.45, 0.507, "chance", fontsize=8.2, color=BAD,
            ha="right", va="bottom", fontweight="bold")

    # The gap is the experiment's output, so draw it. Both annotations now sit
    # ABOVE every bar. The previous layout put the BOTH-ensembles arrow at
    # y=0.556 to keep it clear of the orange bars, but the two series are
    # interleaved on the x axis, so an arrow spanning matched->mismatched at any
    # height below 0.734 crosses the orange matched bar -- an arrow measuring
    # one series was drawn over the other's data. Each label also names its
    # series, so the reader does not have to match arrow colour to the legend.
    # The colour #8895A0 that matched the bar edge sat at ~3:1 on white; the
    # bars stay pale (deliberate -- this is the rule we reject) but the label
    # reads at ~7:1.
    for series, off, col, y, name in (
            ("both",   -w / 2, "#4A5560", 0.790, "BOTH ensembles"),
            ("normal",  w / 2, CTX,       0.850, "NORMAL only")):
        m, mm = SWEEP[series][2], SWEEP[series][3]
        xm, xmm = 2 + off, 3 + off
        ax.annotate("", xy=(xmm, y), xytext=(xm, y),
                    arrowprops=dict(arrowstyle="<->", color=col, lw=1.1))
        ax.text((xm + xmm) / 2, y + 0.006,
                f"{name}:  gap {m - mm:+.3f}", ha="center", va="bottom",
                fontsize=8.8, color=col, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(CONDS)
    ax.set_ylim(0.48, 0.90); ax.set_ylabel("frame-level AUROC")
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.legend(fontsize=8.6, loc="upper left", frameon=True, edgecolor=GRID,
              framealpha=0.95)
    ax.set_title("Context sweep, ShanghaiTech, $w{=}31$ — everything frozen; "
                 "only the sentence and where it goes change", pad=6)
    fig.tight_layout()
    p = os.path.join(out, "fig_chart_sweep.png")
    figsave.save(fig, p); plt.close(fig)
    return p


def window_and_ablation(out):
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.75),
                             gridspec_kw={"width_ratios": [1.0, 1.05]})

    ax = axes[0]
    styles = {
        "micro, raw":             dict(color="#9AA6A7", ls=(0, (3, 2)), marker="s"),
        "macro (per clip)":       dict(color=VIS, ls="-.", marker="^"),
        "micro, per-clip normed": dict(color=TMP, ls="-", marker="o"),
        "matched descriptor":     dict(color=CTX, ls="-", marker="D"),
    }
    for name, ys in CURVES.items():
        xs = [w for w, y in zip(WINDOWS, ys) if y is not None]
        vals = [y for y in ys if y is not None]
        ax.plot(xs, vals, lw=1.4, ms=3.4, label=name, **styles[name])
    ax.axhline(0.5, color=BAD, lw=0.8, ls=(0, (4, 3)))
    ax.axvline(31, color=MUTED, lw=0.7, ls=(0, (2, 2)))
    ax.text(31, 0.767, "optimum", fontsize=8.4, color=INK, fontweight="bold",
            ha="center",
            va="top", bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
    ax.set_xscale("log"); ax.set_xticks(WINDOWS)
    ax.set_xticklabels([str(w) for w in WINDOWS])
    ax.set_xlabel("temporal window $w$ (frames)")
    ax.set_ylabel("frame-level AUROC")
    ax.set_ylim(0.44, 0.78)
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.legend(fontsize=8.0, loc="lower left", frameon=True, edgecolor=GRID,
              framealpha=0.95)
    ax.set_title("Temporal window and pooling convention", pad=5)

    ax = axes[1]
    names = [n for n, _, _ in ABLATION]
    vals = [v for _, v, _ in ABLATION]
    errs = [e for _, _, e in ABLATION]
    y = np.arange(len(names))
    cols = ["#B7C2C1"] * len(names)
    cols[-1] = VIS
    ax.barh(y, vals, xerr=errs, color=cols, edgecolor="#8895A0", lw=0.7,
            error_kw=dict(ecolor=MUTED, lw=0.8, capsize=2.2))
    for i, v in enumerate(vals):
        ax.text(v + errs[i] + 0.006, i, f"{v:.3f}", va="center", fontsize=7.7,
                color=INK)
    ax.axvline(0.5, color=BAD, lw=0.8, ls=(0, (4, 3)))
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8.8)
    ax.set_xlim(0.48, 0.82)
    # Raw string: "\p" is not an escape sequence and Python 3.12+ warns on it.
    # Shortened too -- at the larger bold type the full wording ran off the
    # right edge of the canvas, which figsave's clip check caught.
    ax.set_xlabel(r"held-out AUROC (mean $\pm$ sd, 5 splits)")
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.set_title("Component ablation — nothing added\nto the semantic pathway "
                 "improves on it", pad=5, fontsize=9.1)

    fig.tight_layout(w_pad=1.6)
    p = os.path.join(out, "fig_chart_ablation.png")
    figsave.save(fig, p); plt.close(fig)
    return p


def within_view(csv_path, out):
    rows = list(csv.DictReader(open(csv_path)))
    views = [r["view"] for r in rows]
    gaps = np.array([float(r["gap"]) for r in rows])
    n = np.array([int(r["n_clips"]) for r in rows])
    pooled, mean_within = 0.1053, float(gaps.mean())

    fig, ax = plt.subplots(figsize=(6.9, 2.6))
    x = np.arange(len(views))
    cols = [OK if g > 0 else BAD for g in gaps]
    ax.bar(x, gaps, 0.62, color=cols, alpha=0.70, edgecolor="#8895A0", lw=0.7)
    for i, (g, c) in enumerate(zip(gaps, n)):
        ax.text(i, g + (0.004 if g >= 0 else -0.004), f"{g:+.3f}", ha="center",
                va="bottom" if g >= 0 else "top", fontsize=7.2, color=INK)
        ax.text(i, -0.088, f"n={c}", ha="center", fontsize=6.7, color=MUTED)

    ax.axhline(0, color="#8895A0", lw=0.8)
    ax.axhline(pooled, color=CTX, lw=1.5)
    ax.text(len(views) - 0.4, pooled + 0.004, f"pooled across views  {pooled:+.3f}",
            ha="right", va="bottom", fontsize=8.2, color=CTX, fontweight="bold")
    ax.axhline(mean_within, color=TMP, lw=1.2, ls=(0, (4, 2.5)))
    ax.text(-0.4, mean_within + 0.004, f"mean within a view  {mean_within:+.3f}",
            ha="left", va="bottom", fontsize=7.9, color=TMP, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(views)
    ax.set_xlabel("camera view"); ax.set_ylabel("matched $-$ mismatched")
    ax.set_ylim(-0.10, 0.14)
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    # "collapses" overstates it: two views nearly reach the pooled figure. The
    # claim the data supports is about the mean and the spread, so say that.
    # Wrapped explicitly. tight_layout pads the axes but never shrinks or wraps
    # a title, so at the larger type this ran off the right edge of the canvas.
    ax.set_title("Per-camera gaps against the pooled figure — the mean within\n"
                 "a view is a third of pooled, and three views are negative",
                 pad=5, fontsize=9.1)
    fig.tight_layout()
    p = os.path.join(out, "fig_chart_within_view.png")
    figsave.save(fig, p); plt.close(fig)
    return p


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--within-view-csv", default=os.path.join(
        root, "results", "runs", "analysis", "within_view.csv"))
    ap.add_argument("--out", default=os.path.join(root, "docs", "09_paper", "figures"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print("  ", context_sweep(a.out))
    print("  ", window_and_ablation(a.out))
    print("  ", within_view(a.within_view_csv, a.out))
