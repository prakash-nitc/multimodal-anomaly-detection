# -*- coding: utf-8 -*-
"""Charts added for Paper 1 (the CLIP + scene-description paper).

  comparison   published methods against ours on five benchmarks, one panel per
               benchmark. Every external figure is a row of the provenance ledger
               (docs/07_reference_papers/PROVENANCE.md) and of tbl:sota; nothing
               here is new. Ours: no description, and the configuration the paper
               proposes (shared sentence on the campus benchmarks, generated
               description per video / per scene elsewhere).

  roc          ROC curves on ShanghaiTech, UCF-Crime and NWPU Campus and the
               precision-recall curve on XD-Violence, from the ViT-L scores in
               results/runs/analysis/fixes_scores_<ds>_vitl.npz and the frame labels
               exported from the server (--labels). The script first reproduces the
               published figures from those scores and refuses to draw if any
               differs by more than 1e-3.

Usage:  python scripts/make_paper1_charts.py comparison
        python scripts/make_paper1_charts.py roc --labels PATH/paper_labels.npz
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figsave  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "09_paper", "figures")

INK, MUTED, GRID = "#101718", "#394547", "#D8E0DF"
ONECLASS, OURS, OTHER = "#3B5BA5", "#B4571A", "#7D8A89"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "axes.edgecolor": "#8E9E9D", "axes.linewidth": 0.9,
    "xtick.color": INK, "ytick.color": INK, "xtick.labelsize": 7.6,
    "ytick.labelsize": 7.6, "axes.labelsize": 8.2, "axes.titlesize": 9.0,
    "axes.labelweight": "bold", "axes.titleweight": "bold",
})

# kind: "one" one-class training on target normal video; "few" a few normal
# target frames + an LLM; "tf" training-free, published; "ours0" ours without a
# description; "ours" ours with the proposed description.
BENCH = {
    "ShanghaiTech (AUROC)": [
        ("Liu et al.", 72.8, "one"), ("MPN", 73.8, "one"), ("FPDM", 78.6, "one"),
        ("MA-PDM", 79.2, "one"), ("MULDE", 81.3, "one"), ("AnomalyRuler", 85.2, "few"),
        ("Zero-shot CLIP", 60.9, "tf"), ("Zero-shot ImageBind", 61.3, "tf"),
        ("LLaVA-1.5", 59.6, "tf"), ("Video-ChatGPT", 69.1, "tf"),
        ("AnyAnomaly", 79.7, "tf"),
        ("Ours, no description", 70.7, "ours0"), ("Ours, shared description", 73.4, "ours")],
    "CUHK Avenue (AUROC)": [
        ("Liu et al.", 85.1, "one"), ("MPN", 89.5, "one"), ("FPDM", 90.1, "one"),
        ("MA-PDM", 91.3, "one"),
        ("Zero-shot CLIP", 62.3, "tf"), ("Zero-shot ImageBind", 64.5, "tf"),
        ("LLaVA-1.5", 67.4, "tf"), ("Video-ChatGPT", 76.9, "tf"),
        ("AnyAnomaly", 87.3, "tf"),
        ("Ours, no description", 70.6, "ours0"), ("Ours, shared description", 67.7, "ours")],
    "UCF-Crime (AUROC)": [
        ("FPDM", 74.7, "one"), ("MULDE", 78.5, "one"),
        ("Zero-shot CLIP", 53.2, "tf"), ("Zero-shot ImageBind", 53.7, "tf"),
        ("LLaVA-1.5", 72.8, "tf"), ("LAVAD", 80.3, "tf"), ("AnyAnomaly", 80.7, "tf"),
        ("Frozen VLM, yes/no", 85.3, "tf"), ("Flashback-PE", 87.3, "tf"),
        ("ASK-Hint", 89.8, "tf"), ("NOVA", 89.9, "tf"),
        ("Ours, no description", 75.6, "ours0"), ("Ours, per-video description", 82.4, "ours")],
    "XD-Violence (AP)": [
        ("Zero-shot CLIP", 17.8, "tf"), ("Zero-shot ImageBind", 27.3, "tf"),
        ("LLaVA-1.5", 50.3, "tf"), ("LAVAD", 62.0, "tf"),
        ("Frozen VLM, yes/no", 69.0, "tf"), ("Flashback-PE", 75.1, "tf"),
        ("NOVA", 84.8, "tf"),
        ("Ours, no description", 62.4, "ours0"), ("Ours, per-video description", 41.7, "ours")],
    "NWPU Campus (AUROC)": [
        ("MPN", 64.4, "one"), ("Cao et al., scene-conditioned", 68.2, "one"),
        ("Ours, no description", 65.9, "ours0"), ("Ours, per-scene description", 66.7, "ours")],
}
STYLE = {"one": dict(color=ONECLASS), "few": dict(color=ONECLASS, hatch="////"),
         "tf": dict(color=OTHER), "ours0": dict(color="#FFFFFF", hatch="////"),
         "ours": dict(color=OURS)}


def comparison(out):
    # 7.3 in tall so that, with its caption, the figure fits on one page of the
    # cas-sc text block; at 9.4 in it could not float and was pushed to the end.
    fig, axes = plt.subplots(3, 2, figsize=(6.9, 7.3),
                             gridspec_kw=dict(height_ratios=[13, 13, 5]))
    axes = axes.ravel()
    for ax, (title, rows) in zip(axes, BENCH.items()):
        rows = sorted(rows, key=lambda r: r[1])
        lo = 0 if "AP" in title else 50
        y = np.arange(len(rows))
        for yi, (lab, v, kind) in zip(y, rows):
            st = STYLE[kind]
            edge = OURS if kind.startswith("ours") else "#FFFFFF"
            ax.barh(yi, v - lo, left=lo, height=0.72, color=st["color"],
                    hatch=st.get("hatch"), edgecolor=edge, lw=1.0 if kind == "ours0" else 0)
            if "hatch" in st and kind == "few":
                ax.barh(yi, v - lo, left=lo, height=0.72, fill=False,
                        edgecolor="#FFFFFF", hatch="////", lw=0)
            ax.text(v + (1.0 if lo == 0 else 0.6), yi, "%.1f" % v, va="center",
                    fontsize=7.2, color=INK,
                    fontweight="bold" if kind.startswith("ours") else "normal")
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows])
        for tl, r in zip(ax.get_yticklabels(), rows):
            if r[2].startswith("ours"):
                tl.set_fontweight("bold")
        ax.set_xlim(lo, 95 if lo == 50 else 92)
        ax.set_title(title, loc="left", pad=4)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    legend_ax = axes[-1]
    legend_ax.axis("off")
    handles = [Patch(facecolor=ONECLASS, label="Trained on normal video\nof the target scene"),
               Patch(facecolor=ONECLASS, hatch="////", edgecolor="#FFFFFF",
                     label="A few normal target\nframes + GPT-4"),
               Patch(facecolor=OTHER, label="Training-free, published"),
               Patch(facecolor="#FFFFFF", hatch="////", edgecolor=OURS,
                     label="Ours, no description"),
               Patch(facecolor=OURS, label="Ours, with scene description")]
    legend_ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 1.05), frameon=False, fontsize=7.8,
                     handlelength=2.2, handleheight=1.2, labelspacing=0.9)
    fig.tight_layout(h_pad=1.4, w_pad=1.2)
    p = os.path.join(out, "fig_comparison.png")
    figsave.save(fig, p)
    plt.close(fig)
    return p


# ----------------------------------------------------------------------- ROC
PUBLISHED = {  # dataset -> condition -> published figure (paper tables)
    "shanghaitech": {"base": 0.707, "shared": 0.734, "llava": 0.751},
    "ucf_crime": {"base": 0.756, "shared": 0.813, "llava": 0.824},
    "nwpu_campus": {"base": 0.659, "shared": 0.648, "llava": 0.667},
    "xd_violence": {"base": 0.624, "shared": 0.582, "llava": 0.417},
}
NAMES = {"shanghaitech": "ShanghaiTech", "ucf_crime": "UCF-Crime",
         "nwpu_campus": "NWPU Campus", "xd_violence": "XD-Violence"}
CONDS = [("base", "No description", "#7D8A89", (0, (4, 2))),
         ("shared", "One shared description", ONECLASS, "-"),
         ("llava", "Generated by LLaVA, per camera / video / scene", OURS, "-")]


def roc(out, labels_path):
    from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve
    lab = np.load(labels_path, allow_pickle=True)
    tables = os.path.join(ROOT, "results", "runs", "analysis")
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 6.6))
    for ax, ds in zip(axes.ravel(), ["shanghaitech", "ucf_crime", "nwpu_campus", "xd_violence"]):
        sc = np.load(os.path.join(tables, "fixes_scores_%s_vitl.npz" % ds), allow_pickle=True)
        names = [str(n) for n in sc["names"]]
        if names != [str(n) for n in lab[ds + "_names"]]:
            raise SystemExit("video order differs between scores and labels for " + ds)
        if ds == "xd_violence":
            wpos, wneg = lab[ds + "_wpos"].astype(float), lab[ds + "_wneg"].astype(float)
            n = len(wpos)
            y = np.r_[np.ones(n), np.zeros(n)]
            w = np.r_[wpos, wneg]
            keep = w > 0
        else:
            y = lab[ds + "_L"].astype(int)
        for key, name, col, ls in CONDS:
            s = sc[key]
            if ds == "xd_violence":
                ss = np.r_[s, s]
                p, r, _ = precision_recall_curve(y[keep], ss[keep], sample_weight=w[keep])
                val = auc(r, p)
                ax.plot(r, p, color=col, ls=ls, lw=1.6, label="%s (AP %.3f)" % (name.split(",")[0], PUBLISHED[ds][key]))
            else:
                val = roc_auc_score(y, s)
                fpr, tpr, _ = roc_curve(y, s)
                ax.plot(fpr, tpr, color=col, ls=ls, lw=1.6,
                        label="%s (%.3f)" % (name.split(",")[0], PUBLISHED[ds][key]))
            if abs(val - PUBLISHED[ds][key]) > 1e-3:
                raise SystemExit("%s %s: %.4f does not reproduce the published %.3f"
                                 % (ds, key, val, PUBLISHED[ds][key]))
            print("  %-13s %-7s %.4f  (published %.3f)" % (ds, key, val, PUBLISHED[ds][key]))
        if ds == "xd_violence":
            base_rate = (w[keep] * y[keep]).sum() / w[keep].sum()
            ax.axhline(base_rate, color=MUTED, lw=0.8, ls=":")
            ax.text(0.98, base_rate + 0.015, "chance (positive rate %.2f)" % base_rate,
                    ha="right", fontsize=6.8, color=MUTED)
            ax.set_xlabel("recall")
            ax.set_ylabel("precision")
            ax.set_title("XD-Violence: precision-recall", loc="left", pad=4)
        else:
            ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=":")
            ax.set_xlabel("false-positive rate")
            ax.set_ylabel("true-positive rate")
            ax.set_title("%s: ROC" % NAMES[ds], loc="left", pad=4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.set_aspect("equal")
        ax.grid(color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.legend(loc="lower right" if ds != "xd_violence" else "upper right",
                  fontsize=6.6, frameon=True, framealpha=0.92, edgecolor=GRID)
    fig.tight_layout(h_pad=1.2, w_pad=1.0)
    p = os.path.join(out, "fig_roc_curves.png")
    figsave.save(fig, p)
    plt.close(fig)
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["comparison", "roc"])
    ap.add_argument("--labels")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.which == "comparison":
        print(comparison(a.out))
    else:
        print(roc(a.out, a.labels))
