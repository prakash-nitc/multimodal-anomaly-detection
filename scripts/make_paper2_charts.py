# -*- coding: utf-8 -*-
"""Charts for Paper 2 (the frozen final model and its sealed UBnormal test).

Paper 2 uses its own palette and chart forms, so that it does not read as a copy of
Paper 1 (which uses rust/teal horizontal bars). Every number is read from a committed
result file, never typed in, except the published figures of `comparison`, which are
rows of docs/07_reference_papers/PROVENANCE.md.

  progression  slope chart: the five configurations of the development round
  forest       paired effects with 95% intervals, one panel per benchmark,
               including the sealed UBnormal panel
  timeline     the evidence chain, from the first pre-registration to the sealed result
  dev_roc      ROC / PR of the final model against plain PE-Core and plain ViT-L on the
               four development benchmarks (needs --labels, the exported dev labels)
  comparison   dot plot of published methods and ours on five benchmarks
  ubnormal     sealed-test figures from the exported per-frame data (needs --export):
               ROC, per-scene AUROC, and per-anomaly-type AUROC of the bank (post hoc)

Usage:  python scripts/make_paper2_charts.py progression forest timeline comparison
        python scripts/make_paper2_charts.py dev_roc --labels PATH/paper_labels.npz
        python scripts/make_paper2_charts.py ubnormal --export PATH/ubnormal_paper2_export.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figsave  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "runs", "analysis")
OUT = os.path.join(ROOT, "docs", "09_paper", "figures")

INK, MUTED, GRID, FAINT = "#141A1F", "#3E4A52", "#E1E5EA", "#9AA5AE"
FINAL, CALM, PE, VIT = "#8E3A72", "#00897B", "#5550B0", "#C08A00"
OTHER = "#8C969E"
plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "axes.edgecolor": "#AAB3BA", "axes.linewidth": 0.8,
    "xtick.color": INK, "ytick.color": INK, "xtick.labelsize": 7.6, "ytick.labelsize": 7.6,
    "axes.labelsize": 8.2, "axes.titlesize": 8.8, "axes.titleweight": "bold",
    "axes.labelcolor": INK,
})
DEV = ["shanghaitech", "ucf_crime", "xd_violence", "nwpu_campus"]
NAME = {"shanghaitech": "ShanghaiTech", "ucf_crime": "UCF-Crime",
        "xd_violence": "XD-Violence (AP)", "nwpu_campus": "NWPU Campus", "ubnormal": "UBnormal"}


def j(name):
    return json.load(open(os.path.join(RES, name)))


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ progression
def progression(out):
    stages = ["ViT-L/14\nno text", "ViT-L/14\n+ LLaVA\ndescription", "PE-Core\nno text",
              "PE-Core\n+ calm (C)", "PE-Core\n+ C + bank (B)"]
    vals = {}
    for ds in DEV:
        fv, fp, s3 = j("fixes_%s_vitl.json" % ds), j("fixes_%s_pe-l.json" % ds), j("ideas_step3_%s.json" % ds)
        vals[ds] = [fv["point"]["base"], s3["point"]["old_llava"], fp["point"]["base"],
                    fp["point"]["C"], s3["point"]["B"]]
    colours = {"shanghaitech": PE, "ucf_crime": CALM, "xd_violence": FINAL, "nwpu_campus": VIT}
    marks = {"shanghaitech": "o", "ucf_crime": "s", "xd_violence": "D", "nwpu_campus": "^"}
    fig, ax = plt.subplots(figsize=(6.9, 3.3))
    x = np.arange(len(stages))
    ax.axvspan(1.5, 4.5, color="#F4EEF3", zorder=0)
    ax.text(3.0, 0.405, "frozen PE-Core encoder", ha="center", fontsize=7.4, color=MUTED, style="italic")
    for ds in DEV:
        v = vals[ds]
        ax.plot(x, v, color=colours[ds], lw=1.8, marker=marks[ds], ms=5.5, zorder=3,
                markeredgecolor="white", markeredgewidth=0.8)
        ax.text(x[-1] + 0.12, v[-1], "%s  %.3f" % (NAME[ds], v[-1]), va="center", fontsize=7.4,
                color=INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=7.4)
    ax.set_xlim(-0.3, 5.9)
    ax.set_ylim(0.39, 0.89)
    ax.set_ylabel("AUROC  (AP on XD-Violence)")
    clean(ax)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    p = os.path.join(out, "fig_p2_progression.png")
    figsave.save(fig, p)
    plt.close(fig)
    return p


# ------------------------------------------------------------------ forest
def forest(out):
    rows = ["Encoder: PE-Core − ViT-L/14", "Calm phrases on PE-Core (C − plain)",
            "Calm phrases on ViT-L/14", "Bank on top of C (B − C)",
            "Final − description-based system", "Final − plain ViT-L/14"]
    eff = {}
    bb = j("normal_side_backbones_pe-l.json")["datasets"]
    for ds in DEV:
        fp, fv, s3 = j("fixes_%s_pe-l.json" % ds), j("fixes_%s_vitl.json" % ds), j("ideas_step3_%s.json" % ds)
        e = {rows[0]: bb[ds]["new_none - vitl_none"], rows[1]: fp["comparisons"]["C - base"],
             rows[2]: fv["comparisons"]["C - base"], rows[3]: s3["comparisons"]["B - C"],
             rows[4]: s3["comparisons"]["B - old_llava"]}
        eff[ds] = {k: (v["point"], v["ci95"][0], v["ci95"][1]) for k, v in e.items()}
    s = j("sealed_ubnormal.json")["comparisons"]
    eff["ubnormal"] = {rows[1]: s["C - pe_plain"], rows[3]: s["B - C"], rows[5]: s["B - vit_plain"]}
    eff["ubnormal"] = {k: (v["point"], v["ci95"][0], v["ci95"][1]) for k, v in eff["ubnormal"].items()}
    cols = {rows[0]: PE, rows[1]: CALM, rows[2]: VIT, rows[3]: FINAL, rows[4]: INK, rows[5]: INK}

    order = DEV + ["ubnormal"]
    fig, axes = plt.subplots(1, 5, figsize=(6.9, 3.2), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, ds in zip(axes, order):
        ax.axvline(0, color=MUTED, lw=0.9)
        for yi, r in zip(y, rows):
            if r not in eff[ds]:
                ax.text(0, yi, "—", ha="center", va="center", fontsize=7, color=FAINT)
                continue
            pt, lo, hi = eff[ds][r]
            sig = lo > 0 or hi < 0
            ax.plot([lo, hi], [yi, yi], color=cols[r], lw=1.6, solid_capstyle="round")
            ax.plot([pt], [yi], marker="o", ms=5.2, color=cols[r] if sig else "white",
                    markeredgecolor=cols[r], markeredgewidth=1.4, zorder=3)
        ax.set_title(NAME[ds].replace(" (AP)", "\n(AP)") if ds != "ubnormal" else "UBnormal\n(sealed)",
                     fontsize=7.8, color=FINAL if ds == "ubnormal" else INK)
        ax.set_xlim(-0.25, 0.42)
        ax.set_xticks([-0.2, 0, 0.2, 0.4])
        ax.tick_params(axis="x", labelsize=6.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color=GRID, lw=0.6)
        if ds == "ubnormal":
            ax.set_facecolor("#FAF3F8")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(rows, fontsize=7.2)
    axes[0].set_ylim(-0.7, len(rows) - 0.3)
    fig.supxlabel("paired difference (95% interval); filled = interval excludes 0", fontsize=7.6, y=0.02)
    fig.tight_layout(w_pad=0.6)
    p = os.path.join(out, "fig_p2_forest.png")
    figsave.save(fig, p)
    plt.close(fig)
    return p


# ------------------------------------------------------------------ timeline
EVENTS = [  # (day offset in hours from 3 Nov 00:00, label, commit, kind)
    ("3 Nov 21:41", "Normal-side study\npre-registered", "a87ff87", "pre"),
    ("5 Nov 14:33", "Fixes study\npre-registered", "9ae7671", "pre"),
    ("5 Nov 20:04", "Development round:\nrules + step 1", "7168814", "pre"),
    ("6 Nov 01:01", "Steps 2-5 scored,\neach after its rules", "5a2dafb", "res"),
    ("6 Nov 01:11", "Model frozen;\nsealed test\npre-registered", "3792b86", "key"),
    ("6 Nov 13:21", "Sealed-test code\ncommitted", "5c52d41", "key"),
    ("6 Nov 19:31", "UBnormal\ndownloaded", "", "data"),
    ("6 Nov 22:12", "Scored once", "", "data"),
    ("6 Nov 22:18", "Result\ncommitted", "2764f93", "key"),
]


def _hours(stamp):
    d, hm = stamp.split(" Nov ")
    h, m = hm.split(":")
    return (int(d) - 3) * 24 + int(h) + int(m) / 60


def timeline(out):
    fig, ax = plt.subplots(figsize=(6.9, 2.35))
    hs = [_hours(e[0]) for e in EVENTS]
    # compress the long gaps so the dense last day stays legible
    pos = np.cumsum([0] + [min(1.0, 0.42 + (b - a) / 16) for a, b in zip(hs[:-1], hs[1:])])
    ax.plot([pos[0] - 0.3, pos[-1] + 0.3], [0, 0], color=FAINT, lw=2, zorder=1)
    ax.axvspan(pos[4] - 0.08, pos[-1] + 0.3, color="#F7EEF4", zorder=0)
    ax.text((pos[4] + pos[-1]) / 2, -0.95, "everything about the test fixed before any UBnormal data existed",
            ha="center", fontsize=7.2, color=FINAL, style="italic")
    col = {"pre": PE, "res": CALM, "key": FINAL, "data": VIT}
    for i, (e, x) in enumerate(zip(EVENTS, pos)):
        stamp, label, commit, kind = e
        up = i % 2 == 0
        ax.plot([x], [0], marker="o", ms=8 if kind == "key" else 6.5, color=col[kind],
                markeredgecolor="white", markeredgewidth=1.2, zorder=3)
        ax.plot([x, x], [0, 0.32 if up else -0.32], color=col[kind], lw=0.9, zorder=2)
        ty = 0.38 if up else -0.38
        ax.text(x, ty, label, ha="center", va="bottom" if up else "top", fontsize=6.9, color=INK,
                fontweight="bold" if kind == "key" else "normal", linespacing=1.05)
        sub = stamp + ("\n" + commit if commit else "")
        ax.text(x, -0.12 if up else 0.12, sub, ha="center", va="top" if up else "bottom",
                fontsize=6.0, color=MUTED, linespacing=1.0)
    ax.set_xlim(pos[0] - 0.45, pos[-1] + 0.45)
    ax.set_ylim(-1.05, 1.15)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    p = os.path.join(out, "fig_p2_timeline.png")
    figsave.save(fig, p)
    plt.close(fig)
    return p


# ------------------------------------------------------------------ dev ROC
def dev_roc(out, labels_path):
    from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve
    lab = np.load(labels_path, allow_pickle=True)
    pub = {ds: (j("ideas_step3_%s.json" % ds)["point"]["B"], j("fixes_%s_pe-l.json" % ds)["point"]["base"],
                j("fixes_%s_vitl.json" % ds)["point"]["base"]) for ds in DEV}
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 2.35))
    for ax, ds in zip(axes, DEV):
        sB = np.load(os.path.join(RES, "ideas_step3_scores_%s.npz" % ds), allow_pickle=True)
        sP = np.load(os.path.join(RES, "fixes_scores_%s_pe-l.npz" % ds), allow_pickle=True)
        sV = np.load(os.path.join(RES, "fixes_scores_%s_vitl.npz" % ds), allow_pickle=True)
        names = [str(n) for n in lab[ds + "_names"]]
        for z in (sB, sP, sV):
            if [str(n) for n in z["names"]] != names:
                raise SystemExit("video order differs for " + ds)
        curves = [("Final (B)", sB["B"], FINAL, pub[ds][0]), ("Plain PE-Core", sP["base"], PE, pub[ds][1]),
                  ("Plain ViT-L/14", sV["base"], VIT, pub[ds][2])]
        if ds == "xd_violence":
            wpos, wneg = lab[ds + "_wpos"].astype(float), lab[ds + "_wneg"].astype(float)
            yv = np.r_[np.ones(len(wpos)), np.zeros(len(wneg))]
            w = np.r_[wpos, wneg]
            k = w > 0
        else:
            yv = lab[ds + "_L"].astype(int)
        for name, s, c, want in curves:
            if ds == "xd_violence":
                ss = np.r_[s, s]
                pr, rc, _ = precision_recall_curve(yv[k], ss[k], sample_weight=w[k])
                got = auc(rc, pr)
                ax.plot(rc, pr, color=c, lw=1.5, label="%.3f" % want)
            else:
                got = roc_auc_score(yv, s)
                fpr, tpr, _ = roc_curve(yv, s)
                ax.plot(fpr, tpr, color=c, lw=1.5, label="%.3f" % want)
            if abs(got - want) > 1e-3:
                raise SystemExit("%s %s: %.4f does not reproduce %.4f" % (ds, name, got, want))
            print("  %-13s %-15s %.4f (result file %.4f)" % (ds, name, got, want))
        if ds != "xd_violence":
            ax.plot([0, 1], [0, 1], color=FAINT, lw=0.7, ls=":")
            ax.set_xlabel("false-positive rate", fontsize=7)
        else:
            ax.set_xlabel("recall", fontsize=7)
        ax.set_title(NAME[ds], fontsize=7.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6.4)
        clean(ax)
        ax.legend(loc="lower right" if ds != "xd_violence" else "upper right", fontsize=6.2,
                  frameon=True, framealpha=0.9, edgecolor=GRID, handlelength=1.2)
    axes[0].set_ylabel("true-positive rate", fontsize=7)
    axes[2].set_ylabel("precision", fontsize=7)
    fig.tight_layout(w_pad=0.5)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color=c, lw=2) for c in (FINAL, PE, VIT)],
               ["Final model (PE-Core + C + B)", "Plain PE-Core", "Plain ViT-L/14"],
               loc="lower center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.subplots_adjust(bottom=0.25)
    p = os.path.join(out, "fig_p2_dev_roc.png")
    figsave.save(fig, p)
    plt.close(fig)
    return p


# ------------------------------------------------------------------ comparison
PUBLISHED = {  # every value is a PROVENANCE.md row; kind: tf training-free, tgt uses target data
    "UCF-Crime": [("NOVA", 89.86, "tf"), ("ASK-Hint", 89.83, "tf"), ("Flashback-PE", 87.29, "tf"),
                  ("Frozen VLM", 85.32, "tf"), ("PANDA", 84.89, "tf"), ("VADTree", 84.74, "tf"),
                  ("EventVAD", 82.03, "tf"), ("AnyAnomaly", 80.7, "tf"), ("LAVAD", 80.28, "tf"),
                  ("MULDE", 78.5, "tgt"), ("FPDM", 74.7, "tgt")],
    "XD-Violence (AP)": [("NOVA", 84.82, "tf"), ("Flashback-PE", 75.13, "tf"), ("VERA", 70.54, "tgt"),
                         ("PANDA", 70.16, "tf"), ("Frozen VLM", 68.98, "tf"), ("VADTree", 68.85, "tf"),
                         ("EventVAD", 64.04, "tf"), ("LAVAD", 62.01, "tf")],
    "ShanghaiTech": [("AnomalyRuler", 85.2, "tgt"), ("MULDE", 81.3, "tgt"), ("AnyAnomaly", 79.7, "tf"),
                     ("MA-PDM", 79.2, "tgt"), ("FPDM", 78.6, "tgt"), ("MPN", 73.8, "tgt"),
                     ("Liu et al.", 72.8, "tgt")],
    "NWPU Campus": [("Cao et al. (scene)", 68.2, "tgt"), ("Cao et al.", 65.8, "tgt"), ("AMMC-Net", 64.5, "tgt"),
                    ("MPN", 64.4, "tgt"), ("HF2-VAD", 63.7, "tgt"), ("MNAD", 62.5, "tgt"), ("MemAE", 61.9, "tgt")],
    "UBnormal": [("AnomalyRuler", 71.9, "tgt"), ("STG-NF", 71.8, "tgt"), ("Bertasius et al.", 68.5, "tgt"),
                 ("OVVAD", 62.94, "tgt"), ("FPDM", 62.7, "tgt"), ("Georgescu et al.", 59.3, "tgt")],
}


def comparison(out):
    ours = {"UCF-Crime": j("ideas_step3_ucf_crime.json")["point"]["B"],
            "XD-Violence (AP)": j("ideas_step3_xd_violence.json")["point"]["B"],
            "ShanghaiTech": j("ideas_step3_shanghaitech.json")["point"]["B"],
            "NWPU Campus": j("ideas_step3_nwpu_campus.json")["point"]["B"],
            "UBnormal": j("sealed_ubnormal.json")["primary_micro_auroc"]["B"]}
    ub_ci = j("sealed_ubnormal.json")["ci95"]["B"]
    fig, axes = plt.subplots(1, 5, figsize=(6.9, 3.25))
    for ax, (bench, rows) in zip(axes, PUBLISHED.items()):
        rows = sorted(rows + [("Ours (final)", 100 * ours[bench], "ours")], key=lambda r: r[1])
        y = np.arange(len(rows))
        for yi, (name, v, kind) in zip(y, rows):
            c = FINAL if kind == "ours" else (OTHER if kind == "tf" else PE)
            ax.plot([ax.get_xlim()[0] if False else 0, v], [yi, yi], color=GRID, lw=0.8, zorder=1)
            ax.plot([v], [yi], marker="D" if kind == "ours" else "o", ms=6.5 if kind == "ours" else 5,
                    color=c, markeredgecolor="white", markeredgewidth=0.8, zorder=3)
            if kind == "ours" and bench == "UBnormal":
                ax.plot([100 * ub_ci[0], 100 * ub_ci[1]], [yi, yi], color=FINAL, lw=1.5, zorder=2)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows], fontsize=6.4)
        for tl, r in zip(ax.get_yticklabels(), rows):
            if r[2] == "ours":
                tl.set_color(FINAL); tl.set_fontweight("bold")
        lo = min(r[1] for r in rows)
        hi = max([r[1] for r in rows] + ([100 * ub_ci[1]] if bench == "UBnormal" else []))
        ax.set_xlim(lo - 4, hi + 3)
        ax.set_title(bench.replace(" (AP)", "\n(AP)") if "AP" in bench else bench + "\n",
                     fontsize=7.6, color=FINAL if bench == "UBnormal" else INK)
        ax.tick_params(axis="x", labelsize=6.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], marker="D", color=FINAL, ls="", ms=6),
                Line2D([], [], marker="o", color=OTHER, ls="", ms=5),
                Line2D([], [], marker="o", color=PE, ls="", ms=5)],
               ["Ours, final model (UBnormal: sealed, with 95% interval)", "Published, training-free",
                "Published, uses target-benchmark data"],
               loc="lower center", ncol=3, fontsize=6.8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(w_pad=0.4)
    fig.subplots_adjust(bottom=0.17)
    p = os.path.join(out, "fig_p2_comparison.png")
    figsave.save(fig, p)
    plt.close(fig)
    return p


# ------------------------------------------------------------------ UBnormal (sealed)
def ubnormal(out, export_path):
    from sklearn.metrics import roc_auc_score, roc_curve
    z = np.load(export_path, allow_pickle=True)
    sealed = j("sealed_ubnormal.json")
    gt = z["gt_full"].astype(int)
    nf = z["n_frames"]
    off = np.r_[0, np.cumsum(nf)]
    conds = [("B", "Final (B)", FINAL), ("C", "PE-Core + C", CALM), ("pe_plain", "Plain PE-Core", PE),
             ("vit_plain", "Plain ViT-L/14", VIT)]
    for k, _n, _c in conds:
        got = roc_auc_score(gt, z["held_" + k])
        want = sealed["primary_micro_auroc"][k]
        if abs(got - want) > 1e-9:
            raise SystemExit("export does not reproduce the sealed %s: %.10f vs %.10f" % (k, got, want))
    print("  export reproduces all four sealed micro AUROCs")
    paths = []

    # (a) ROC and (b) per-scene AUROC
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.9, 2.9), gridspec_kw=dict(width_ratios=[1, 1.55]))
    for k, n, c in conds:
        fpr, tpr, _ = roc_curve(gt, z["held_" + k])
        ax.plot(fpr, tpr, color=c, lw=1.6, label="%s  %.3f" % (n, sealed["primary_micro_auroc"][k]))
    ax.plot([0, 1], [0, 1], color=FAINT, lw=0.7, ls=":")
    ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate")
    ax.set_title("(a) ROC, all 92,640 test frames", fontsize=7.8, loc="left")
    ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.legend(loc="lower right", fontsize=6.2, frameon=True, framealpha=0.9, edgecolor=GRID)
    clean(ax)
    scenes = sorted(set(int(s) for s in z["scene"]))
    rowsB, rowsP = [], []
    for sc in scenes:
        idx = [i for i, s in enumerate(z["scene"]) if int(s) == sc]
        g = np.concatenate([gt[off[i]:off[i + 1]] for i in idx])
        if g.min() == g.max():
            continue
        b = np.concatenate([z["held_B"][off[i]:off[i + 1]] for i in idx])
        pp = np.concatenate([z["held_pe_plain"][off[i]:off[i + 1]] for i in idx])
        rowsB.append((sc, roc_auc_score(g, b), roc_auc_score(g, pp), len(idx)))
    rowsB.sort(key=lambda r: r[1])
    xs = np.arange(len(rowsB))
    for xi, (sc, b, pp, n) in zip(xs, rowsB):
        bx.plot([xi, xi], [pp, b], color=FINAL if b >= pp else VIT, lw=1.2, zorder=2)
        bx.plot([xi], [pp], marker="o", ms=3.8, color="white", markeredgecolor=PE, markeredgewidth=1.1, zorder=3)
        bx.plot([xi], [b], marker="o", ms=4.4, color=FINAL, markeredgecolor="white", markeredgewidth=0.6, zorder=4)
    bx.axhline(0.5, color=FAINT, lw=0.8, ls=":")
    bx.set_xticks(xs)
    bx.set_xticklabels([str(r[0]) for r in rowsB], fontsize=5.8)
    bx.set_xlabel("scene (sorted by the final model's AUROC)")
    bx.set_ylabel("AUROC within the scene")
    better = sum(1 for r in rowsB if r[1] > r[2])
    bx.set_title("(b) AUROC within each scene: final model (filled), plain PE-Core (open)",
                 fontsize=7.6, loc="left")
    clean(bx)
    fig.tight_layout(w_pad=1.2)
    p = os.path.join(out, "fig_p2_ubnormal.png")
    figsave.save(fig, p); plt.close(fig); paths.append(p)

    # (c) per-anomaly-type AUROC of the bank, post hoc
    Fp = z["F_pe"].astype(np.float64)
    Tn = z["T_pe_calm_normal"].astype(np.float64)
    Tb = z["T_pe_bank"].astype(np.float64)
    ty = z["bank_type_of_phrase"]
    types = list(z["bank_types"])
    unit = lambda v: v / np.linalg.norm(v)
    en = unit(Tn.mean(0))
    ys = z["labels_sampled"].astype(int)
    res = []
    for i, t in enumerate(types):
        ea = unit(Tb[ty == i].mean(0))
        res.append((t, roc_auc_score(ys, Fp @ ea - Fp @ en)))
    eall = unit(Tb.mean(0))
    allb = roc_auc_score(ys, Fp @ eall - Fp @ en)
    res.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(6.9, 2.6))
    yy = np.arange(len(res))
    for yi, (t, a) in zip(yy, res):
        ax.plot([0.5, a], [yi, yi], color=GRID, lw=1.0, zorder=1)
        ax.plot([a], [yi], marker="o", ms=5.5, color=FINAL if a >= allb else PE,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        ax.text(a + 0.004, yi, "%.3f" % a, va="center", fontsize=6.6, color=INK)
    ax.axvline(allb, color=FINAL, lw=1.0, ls="--")
    ax.text(allb - 0.003, -0.75, "whole bank %.3f" % allb, fontsize=6.8, color=FINAL, va="center", ha="right")
    ax.set_ylim(-1.15, len(res) - 0.4)
    ax.axvline(0.5, color=FAINT, lw=0.8, ls=":")
    ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in res], fontsize=7)
    ax.set_xlabel("AUROC on UBnormal's sampled frames, one anomaly type's four phrases against the normal side (unsmoothed)")
    clean(ax); ax.grid(axis="y", visible=False)
    fig.tight_layout()
    p = os.path.join(out, "fig_p2_ubnormal_types.png")
    figsave.save(fig, p); plt.close(fig); paths.append(p)
    json.dump({"per_scene": [dict(scene=r[0], auroc_B=r[1], auroc_pe_plain=r[2], videos=r[3]) for r in rowsB],
               "per_type": dict(res), "whole_bank_unsmoothed": allb},
              open(os.path.join(RES, "ubnormal_posthoc.json"), "w"), indent=1)
    print("  wrote results/runs/analysis/ubnormal_posthoc.json (post hoc, aggregate numbers only)")
    return paths


# ------------------------------------------------------------------ UBnormal ablations (post hoc)
def _ub_setup(export_path):
    """Per-video features and labels from the export, and a scorer that reproduces the sealed run."""
    sys.path.insert(0, ROOT)
    from da_zvad.temporal import moving_average
    from sklearn.metrics import roc_auc_score
    z = np.load(export_path, allow_pickle=True)
    step = int(z["step"])
    so = np.r_[0, np.cumsum(z["n_sampled"])]
    fo = np.r_[0, np.cumsum(z["n_frames"])]
    nv = len(z["names"])
    gts = [z["gt_full"][fo[i]:fo[i + 1]].astype(int) for i in range(nv)]
    feats = {"pe": z["F_pe"].astype(np.float64), "vit": z["F_vit"].astype(np.float64)}
    unit = lambda v: v / np.linalg.norm(v)

    def hold(s, n):
        full = np.repeat(s, step)[:n]
        return np.r_[full, np.full(n - len(full), s[-1])] if len(full) < n else full

    def margins(F, N, A, agg="mean", lam=100.0):
        if agg == "mean":
            return F @ unit(A.mean(0)) - F @ unit(N.mean(0))
        SA, SN = F @ A.T, F @ N.T
        if agg == "max":
            return SA.max(1) - SN.max(1)
        lme = lambda S: (np.log(np.exp(lam * (S - S.max(1, keepdims=True))).mean(1)) / lam + S.max(1))
        return lme(SA) - lme(SN)                                   # log-mean-exp over phrases

    def full_scores(N, A, enc="pe", w=5, lam=100.0, agg="mean"):
        m = margins(feats[enc], N, A, agg, lam)
        out = []
        for i in range(nv):
            s = 1.0 / (1.0 + np.exp(-lam * m[so[i]:so[i + 1]]))
            out.append(hold(moving_average(s, w), len(gts[i])))
        return out

    def micro(full):
        return roc_auc_score(np.concatenate(gts), np.concatenate(full))

    T = {"general": z["T_pe_base_normal"].astype(np.float64), "calm11": z["T_pe_calm_normal"].astype(np.float64),
         "base_a": z["T_pe_base_abnormal"].astype(np.float64), "bank": z["T_pe_bank"].astype(np.float64),
         "vit_n": z["T_vit_base_normal"].astype(np.float64), "vit_a": z["T_vit_base_abnormal"].astype(np.float64)}
    return z, gts, T, full_scores, micro


def ablation(out, export_path):
    sys.path.insert(0, os.path.join(ROOT, "notebooks"))
    import sealed_ubnormal as su
    z, gts, T, full_scores, micro = _ub_setup(export_path)
    sealed = j("sealed_ubnormal.json")["primary_micro_auroc"]
    calm = T["calm11"]
    # gate: the rebuilt scorer reproduces all four sealed figures exactly
    ref = {"B": full_scores(calm, T["bank"]), "C": full_scores(calm, T["base_a"]),
           "pe_plain": full_scores(T["general"], T["base_a"]),
           "vit_plain": full_scores(T["vit_n"], T["vit_a"], enc="vit")}
    for k, f in ref.items():
        if abs(micro(f) - sealed[k]) > 1e-9:
            raise SystemExit("ablation scorer does not reproduce sealed %s: %.10f vs %.10f" % (k, micro(f), sealed[k]))
    print("  rebuilt scorer reproduces all four sealed micro AUROCs exactly")
    phr = [str(p) for p in z["phrases_calm_normal"]]
    gen_idx, calm_idx = list(range(3)), list(range(3, 11))
    places, people = calm_idx[:4], calm_idx[4:]
    ty = z["bank_type_of_phrase"]
    types = [str(t) for t in z["bank_types"]]
    res = {"note": "post hoc, UBnormal test split, micro AUROC over all frames; not used to change the model"}

    # (1) components, with paired bootstrap against the final model
    comp = [("Final: general + calm (11) vs bank (48)", calm, T["bank"]),
            ("without the bank (vs 3 general abnormal phrases)", calm, T["base_a"]),
            ("without the calm phrases (3 general vs bank)", T["general"], T["bank"]),
            ("without both (plain PE-Core)", T["general"], T["base_a"]),
            ("calm phrases only, no general phrases (8 vs bank)", calm[calm_idx], T["bank"]),
            ("quiet places only (3 general + 4 vs bank)", calm[gen_idx + places], T["bank"]),
            ("calm people only (3 general + 4 vs bank)", calm[gen_idx + people], T["bank"]),
            ("bank plus the 3 general abnormal phrases", calm, np.r_[T["bank"], T["base_a"]])]
    fulls = {name: full_scores(N, A) for name, N, A in comp}
    draws = su.bootstrap(fulls, gts, list(fulls))
    base = comp[0][0]
    res["components"] = []
    for name, _N, _A in comp:
        p = micro(fulls[name])
        d = draws[name] - draws[base]
        res["components"].append(dict(name=name, auroc=p, diff=p - micro(fulls[base]),
                                      ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]))
    # (2) smoothing window, (3) temperature, (4) phrase aggregation
    res["window"] = {w: micro(full_scores(calm, T["bank"], w=w)) for w in (1, 3, 5, 9, 15, 31)}
    res["lambda"] = {l: micro(full_scores(calm, T["bank"], lam=l)) for l in (1, 10, 30, 100, 300)}
    res["aggregation"] = {a: micro(full_scores(calm, T["bank"], agg=a)) for a in ("mean", "lme", "max")}
    # (5) phrases per type (the first k of each type), (6) leave one type out
    res["per_type_k"] = {k: micro(full_scores(calm, T["bank"][np.isin(np.arange(48), [i for t in range(12)
                                                for i in np.where(ty == t)[0][:k]])])) for k in (1, 2, 3, 4)}
    res["leave_one_out"] = {types[t]: micro(full_scores(calm, T["bank"][ty != t])) for t in range(12)}
    json.dump(res, open(os.path.join(RES, "ubnormal_ablation.json"), "w"), indent=1)
    print("  wrote results/runs/analysis/ubnormal_ablation.json")
    for r in res["components"]:
        print("  %-55s %.4f  %+.4f [%+.4f, %+.4f]" % (r["name"], r["auroc"], r["diff"], *r["ci"]))
    for key in ("window", "lambda", "aggregation", "per_type_k"):
        print("  %-12s" % key, "  ".join("%s:%.4f" % kv for kv in res[key].items()))
    print("  leave-one-out", "  ".join("%s:%.4f" % kv for kv in res["leave_one_out"].items()))

    # figure: four panels
    fig, axs = plt.subplots(2, 2, figsize=(6.9, 4.3))
    axs = axs.flat
    fin = res["window"][5]
    for ax, key, xl, ticks in ((axs[0], "window", "smoothing window w (sampled frames)", None),
                               (axs[1], "per_type_k", "phrases per anomaly type", None)):
        xs = list(res[key].keys()); ys = list(res[key].values())
        ax.plot(range(len(xs)), ys, color=FINAL, lw=1.6, marker="o", ms=4)
        sel = xs.index(5 if key == "window" else 4)
        ax.plot([sel], [ys[sel]], marker="D", ms=6.5, color=FINAL, markeredgecolor="white")
        ax.set_xticks(range(len(xs))); ax.set_xticklabels([str(x) for x in xs], fontsize=6.6)
        ax.set_xlabel(xl, fontsize=6.8); clean(ax)
    for ax in (axs[0], axs[1]):
        ax.set_ylabel("micro AUROC (UBnormal)", fontsize=7)
    axs[0].set_title("(a) smoothing window (diamond: frozen choice)", fontsize=7.6, loc="left")
    axs[1].set_title("(b) bank size (diamond: frozen choice)", fontsize=7.6, loc="left")
    ag = res["aggregation"]; lab = {"mean": "mean of phrase vectors (ours)", "lme": "log-mean-exp of similarities",
                                    "max": "maximum similarity"}
    yy = np.arange(3)[::-1]
    axs[2].barh(yy, list(ag.values()), color=[FINAL, OTHER, OTHER], height=0.55)
    axs[2].set_yticks(yy); axs[2].set_yticklabels([lab[k] for k in ag], fontsize=6.6)
    axs[2].set_xlim(min(ag.values()) - 0.03, max(ag.values()) + 0.02)
    for y_, v in zip(yy, ag.values()):
        axs[2].text(v + 0.002, y_, "%.3f" % v, va="center", fontsize=6.4)
    axs[2].set_xlabel("micro AUROC (UBnormal)", fontsize=6.8)
    axs[2].set_title("(c) how phrases are pooled on each side", fontsize=7.6, loc="left"); clean(axs[2])
    lo_ = sorted(res["leave_one_out"].items(), key=lambda kv: kv[1])
    yy = np.arange(len(lo_))
    axs[3].barh(yy, [v - fin for _k, v in lo_], color=[PE if v < fin else CALM for _k, v in lo_], height=0.65)
    axs[3].axvline(0, color=INK, lw=0.7)
    axs[3].set_yticks(yy); axs[3].set_yticklabels([k for k, _v in lo_], fontsize=6.2)
    axs[3].set_xlabel("change in micro AUROC when one type is removed", fontsize=6.8)
    axs[3].set_title("(d) leave one anomaly type out", fontsize=7.6, loc="left"); clean(axs[3])
    for ax in axs:
        ax.tick_params(labelsize=6.3)
    fig.tight_layout(h_pad=1.2, w_pad=1.0)
    p = os.path.join(out, "fig_p2_ablation.png")
    figsave.save(fig, p); plt.close(fig)
    return p


# ------------------------------------------------------------------ qualitative score curves (dev)
def qualitative(out, labels_path):
    """Score curves for four test videos chosen by a stated rule (not by eye)."""
    from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
    lab = np.load(labels_path, allow_pickle=True)

    def load(ds):
        sB = np.load(os.path.join(RES, "ideas_step3_scores_%s.npz" % ds), allow_pickle=True)
        sV = np.load(os.path.join(RES, "fixes_scores_%s_vitl.npz" % ds), allow_pickle=True)
        sP = np.load(os.path.join(RES, "fixes_scores_%s_pe-l.npz" % ds), allow_pickle=True)
        names = [str(n) for n in lab[ds + "_names"]]
        for zz in (sB, sV, sP):
            if [str(n) for n in zz["names"]] != names:
                raise SystemExit("video order differs for " + ds)
        off = np.r_[0, np.cumsum(sB["lengths"])]
        return names, off, sB, sV, sP

    # gate: the stored scores reproduce the published figures (XD: Paper 1's 0.624 / 0.417)
    names, off, sB, sV, sP = load("xd_violence")
    wpos, wneg = lab["xd_violence_wpos"].astype(float), lab["xd_violence_wneg"].astype(float)
    yv = np.r_[np.ones(len(wpos)), np.zeros(len(wneg))]; w = np.r_[wpos, wneg]; k = w > 0
    ap = lambda s: auc(*precision_recall_curve(yv[k], np.r_[s, s][k], sample_weight=w[k])[1::-1])
    got = {"B": ap(sB["B"]), "vit_none": ap(sV["base"]), "vit_llava": ap(sV["llava"])}
    want = {"B": j("ideas_step3_xd_violence.json")["point"]["B"], "vit_none": 0.624, "vit_llava": 0.417}
    for kk in got:
        tol = 1e-3 if kk == "B" else 5e-4
        if abs(got[kk] - want[kk]) > tol:
            raise SystemExit("XD %s AP %.4f does not reproduce %.4f" % (kk, got[kk], want[kk]))
    print("  XD stored scores reproduce AP: final %.4f, ViT-L none %.4f, ViT-L + LLaVA %.4f" %
          (got["B"], got["vit_none"], got["vit_llava"]))

    def per_video(ds, a, b):
        names, off, sB, sV, sP = load(ds)
        L = lab[ds + "_L"].astype(int)
        rows = []
        for i, n in enumerate(names):
            y = L[off[i]:off[i + 1]]
            if y.min() == y.max():
                continue
            sa, sb = a(sB, sV, sP)[off[i]:off[i + 1]], b(sB, sV, sP)[off[i]:off[i + 1]]
            rows.append((roc_auc_score(y, sb) - roc_auc_score(y, sa), i, n))
        return rows, names, off, L

    picks = []
    # (a) XD: the largest gain over the LLaVA-sentence system among videos violent in their first 3 sampled frames
    rows, names, off, L = per_video("xd_violence", lambda B, V, P: V["llava"], lambda B, V, P: B["B"])
    rows = [r for r in rows if L[off[r[1]]:off[r[1]] + 3].all()]
    picks.append(("xd_violence", max(rows), "llava", "(a) XD-Violence: violence already in the first frames"))
    # (b), (c): the median gain over plain PE-Core
    for ds, title in (("shanghaitech", "(b) ShanghaiTech: median gain"), ("ucf_crime", "(c) UCF-Crime: median gain")):
        rows, *_ = per_video(ds, lambda B, V, P: P["base"], lambda B, V, P: B["B"])
        rows.sort()
        picks.append((ds, rows[len(rows) // 2], "base", title))
    # (d) NWPU: the largest loss against plain PE-Core (a failure case)
    rows, *_ = per_video("nwpu_campus", lambda B, V, P: P["base"], lambda B, V, P: B["B"])
    picks.append(("nwpu_campus", min(rows), "base", "(d) NWPU Campus: largest loss (failure case)"))

    fig, axs = plt.subplots(2, 2, figsize=(6.9, 4.1))
    info = []
    for ax, (ds, (gain, i, n), ref, title) in zip(axs.flat, picks):
        names, off, sB, sV, sP = load(ds)
        L = lab[ds + "_L"].astype(int)
        sl = slice(off[i], off[i + 1])
        y = L[sl]
        x = np.arange(len(y))
        for a0 in np.where(np.diff(np.r_[0, y, 0]) == 1)[0]:
            a1 = a0 + np.argmax(np.r_[y, 0][a0:] == 0)
            ax.axvspan(a0 - 0.5, a1 - 0.5, color="#F3DDE8", lw=0)
        other = sV["llava"][sl] if ref == "llava" else sP["base"][sl]
        ax.plot(x, other, color=VIT if ref == "llava" else PE, lw=1.1,
                label="ViT-L/14 + LLaVA description" if ref == "llava" else "plain PE-Core")
        ax.plot(x, sB["B"][sl], color=FINAL, lw=1.5, label="final model")
        ax.set_title(title, fontsize=7.4, loc="left")
        ax.set_xlim(0, len(y) - 1)
        ax.set_xlabel("sampled frame", fontsize=6.6); ax.set_ylabel("score", fontsize=6.6)
        ax.tick_params(labelsize=6.2); clean(ax)
        ax.legend(fontsize=5.9, loc="upper left", frameon=True, framealpha=0.9, edgecolor=GRID)
        info.append(dict(dataset=ds, video=n, within_video_auroc_gain=float(gain), reference=ref))
    fig.tight_layout(h_pad=0.9, w_pad=0.9)
    p = os.path.join(out, "fig_p2_qualitative.png")
    figsave.save(fig, p); plt.close(fig)
    json.dump(info, open(os.path.join(RES, "qualitative_picks.json"), "w"), indent=1)
    for d in info:
        print("  %-13s %-60s gain %+.3f" % (d["dataset"], d["video"][:60], d["within_video_auroc_gain"]))
    return p


# ------------------------------------------------------------------ the benchmarks at a glance
def datasets(out, labels_path, export_path):
    lab = np.load(labels_path, allow_pickle=True)
    z = np.load(export_path, allow_pickle=True)
    rows = []
    for ds in DEV:
        L = lab[ds + "_L"].astype(int)
        sB = np.load(os.path.join(RES, "ideas_step3_scores_%s.npz" % ds), allow_pickle=True)
        off = np.r_[0, np.cumsum(sB["lengths"])]
        per = [L[off[i]:off[i + 1]] for i in range(len(off) - 1)]
        rows.append((ds, len(per), L.mean(), np.mean([p.max() for p in per]),
                     [p.mean() for p in per if p.max() > 0]))
    fo = np.r_[0, np.cumsum(z["n_frames"])]
    per = [z["gt_full"][fo[i]:fo[i + 1]].astype(int) for i in range(len(fo) - 1)]
    rows.append(("ubnormal", len(per), z["gt_full"].mean(), np.mean([p.max() for p in per]),
                 [p.mean() for p in per if p.max() > 0]))
    fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.3), gridspec_kw=dict(width_ratios=[1, 1, 1.35]))
    nm = [NAME[r[0]].replace(" (AP)", "") for r in rows]
    cols = [OTHER] * 4 + [FINAL]
    yy = np.arange(len(rows))[::-1]
    axs[0].barh(yy, [100 * r[2] for r in rows], color=cols, height=0.6)
    axs[0].set_title("(a) anomalous frames (%)", fontsize=7.6, loc="left")
    axs[1].barh(yy, [100 * r[3] for r in rows], color=cols, height=0.6)
    axs[1].set_title("(b) test videos with an anomaly (%)", fontsize=7.6, loc="left")
    for ax, key in ((axs[0], 2), (axs[1], 3)):
        for y_, r in zip(yy, rows):
            ax.text(100 * r[key] + 1, y_, "%.0f" % (100 * r[key]), va="center", fontsize=6.2)
        ax.set_yticks(yy); ax.set_yticklabels(nm, fontsize=6.6); ax.set_xlim(0, 100); clean(ax)
    bp = axs[2].boxplot([[100 * v for v in r[4]] for r in rows][::-1], vert=False, widths=0.55,
                        patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1))
    for patch, c in zip(bp["boxes"], cols[::-1]):
        patch.set_facecolor(c); patch.set_alpha(0.55); patch.set_edgecolor(c)
    axs[2].set_yticks(range(1, len(rows) + 1)); axs[2].set_yticklabels(nm[::-1], fontsize=6.6)
    axs[2].set_title("(c) anomalous share of each\nanomalous video (%)", fontsize=7.4, loc="left")
    clean(axs[2])
    for ax in axs:
        ax.tick_params(labelsize=6.3)
    fig.tight_layout(w_pad=0.8)
    p = os.path.join(out, "fig_p2_datasets.png")
    figsave.save(fig, p); plt.close(fig)
    for r in rows:
        print("  %-13s videos %4d  anomalous frames %5.1f%%  videos with anomaly %5.1f%%  median share %5.1f%%" %
              (r[0], r[1], 100 * r[2], 100 * r[3], 100 * np.median(r[4])))
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="+", choices=["progression", "forest", "timeline", "dev_roc",
                                                 "comparison", "ubnormal", "ablation", "qualitative",
                                                 "datasets"])
    ap.add_argument("--labels")
    ap.add_argument("--export")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    for w in a.which:
        if w == "dev_roc":
            print(dev_roc(a.out, a.labels))
        elif w == "ubnormal":
            print(ubnormal(a.out, a.export))
        elif w == "ablation":
            print(ablation(a.out, a.export))
        elif w == "qualitative":
            print(qualitative(a.out, a.labels))
        elif w == "datasets":
            print(datasets(a.out, a.labels, a.export))
        else:
            print(globals()[w](a.out))
