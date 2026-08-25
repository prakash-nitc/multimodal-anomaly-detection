# -*- coding: utf-8 -*-
"""Figures for the report, beyond the architecture diagram.

The supervisor asked for images of the data, of a detection actually happening,
and of a component working -- on the grounds that a report of nothing but result
tables reads as though nobody ran anything.

Four figures, each carrying an argument the text also makes:

  dataset_samples     what "anomalous" means on these benchmarks. The label is
                      a property of the scene, not of the object: the same
                      walkway is normal empty and anomalous with a cyclist on
                      it, which is the paper's concept-shift claim in a picture.

  detection_example   one clip's score against time, raw and smoothed, with the
                      true event shaded. Shows both that the detector fires and
                      what M2's smoothing does to the noise around it.

  context_effect      the same clip scored with the correct descriptor and with
                      a description of an industrial site. Nothing else differs,
                      so the gap between the two curves is the contribution of
                      the text, visible per frame instead of as a summary
                      statistic.

  camera_baselines    why pooling raw scores across cameras failed. Frame
                      embeddings projected to two dimensions separate by camera,
                      and each camera's similarity to the global mean sits at a
                      different level -- so a score of 0.6 does not mean the same
                      thing under two different cameras.

Usage:  python scripts/make_report_figures.py [--assets DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

INK, MUTED = "#141C1E", "#5F6E70"
VIS, CTX, TMP = "#0D6B67", "#B4571A", "#3B5BA5"
OK, BAD = "#1E8449", "#B03A2E"
GRID = "#D8E0DF"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "axes.edgecolor": "#B9C6C5", "axes.linewidth": 0.7,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.labelsize": 7.2, "axes.titlesize": 7.6,
})


def _panel_label(ax, s):
    ax.text(-0.02, 1.06, s, transform=ax.transAxes, fontsize=8.4,
            fontweight="bold", color=INK, ha="right", va="bottom")


# ------------------------------------------------------------------ 1
def dataset_samples(assets, out):
    """Normal / anomalous pairs. Same camera in each pair, so only the event differs."""
    root = os.path.join(assets, "frames")
    pairs = []
    for ds, clip, label in (("shanghaitech", "01_0014", "ShanghaiTech · cam 01"),
                            ("shanghaitech", "04_0013", "ShanghaiTech · cam 04"),
                            ("shanghaitech", "12_0175", "ShanghaiTech · cam 12"),
                            ("avenue", "01", "CUHK Avenue · single view")):
        d = os.path.join(root, ds)
        n = sorted(glob.glob(os.path.join(d, f"{clip}_normal_*.jpg")))
        a = sorted(glob.glob(os.path.join(d, f"{clip}_anomaly_*.jpg")))
        if n and a:
            pairs.append((label, n[0], a[0]))
    if not pairs:
        return None

    fig, axes = plt.subplots(2, 4, figsize=(6.9, 2.95))
    for j, (label, npath, apath) in enumerate(pairs):
        for i, (path, tag, col) in enumerate(((npath, "normal", OK),
                                              (apath, "anomalous", BAD))):
            ax = axes[i][j]
            ax.imshow(plt.imread(path))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(col); sp.set_linewidth(1.8)
            if j == 0:
                ax.set_ylabel(tag, fontsize=7.4, color=col, fontweight="bold")
            if i == 0:
                ax.set_title(label, fontsize=6.8, color=INK, pad=3)
    fig.text(0.5, 0.015,
             "Each column is one camera. Only the event differs between the two "
             "rows — the scene, angle and lighting are identical.",
             ha="center", fontsize=6.4, color=MUTED)
    fig.tight_layout(rect=(0, 0.045, 1, 1), h_pad=0.6, w_pad=0.4)
    p = os.path.join(out, "fig_dataset_samples.png")
    fig.savefig(p, dpi=400, facecolor="white"); plt.close(fig)
    return p


# ------------------------------------------------------------------ 2
def detection_example(assets, out):
    z = np.load(os.path.join(assets, "worked_example.npz"), allow_pickle=True)
    raw, lab = z["scores_matched"], z["labels"].astype(int)
    clip = str(z["clip"])

    sm_raw = np.convolve(raw, np.ones(31) / 31, mode="same")
    lo, span = sm_raw.min(), sm_raw.max() - sm_raw.min() + 1e-12
    sm = (sm_raw - lo) / span
    rw = np.clip((raw - lo) / span, -0.15, 1.35)

    fig, ax = plt.subplots(figsize=(6.9, 2.35))
    idx = np.where(lab == 1)[0]
    if len(idx):
        ax.axvspan(idx[0], idx[-1], color=BAD, alpha=0.12, lw=0,
                   label="ground-truth event")
    ax.plot(rw, color="#BFCBCA", lw=0.7, label="raw score  $s_t$")
    ax.plot(sm, color=TMP, lw=1.9, label="smoothed  $\\tilde{s}_t$  ($w=31$)")
    ax.axhline(0.45, color=BAD, lw=1.0, ls=(0, (4, 3)), label=r"threshold  $\tau$")

    fired = np.where(sm >= 0.45)[0]
    if len(fired):
        ax.axvspan(fired[0], fired[-1], ymin=0.0, ymax=0.045, color=OK, lw=0)
        ax.text((fired[0] + fired[-1]) / 2, -0.10, "detected", fontsize=6.4,
                color=OK, ha="center", va="top", fontweight="bold")

    ax.set_xlim(0, len(sm)); ax.set_ylim(-0.18, 1.40)
    ax.set_xlabel("frame index"); ax.set_ylabel("anomaly score")
    ax.set_yticks([0, 0.5, 1.0])
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.2, loc="upper left", frameon=True, framealpha=0.95,
              edgecolor=GRID, ncol=2, handlelength=1.6)
    ax.set_title(f"Detection on {clip} — smoothing suppresses isolated spikes "
                 f"while the event survives", fontsize=7.2, color=INK, pad=5)
    fig.tight_layout()
    p = os.path.join(out, "fig_detection_example.png")
    fig.savefig(p, dpi=400, facecolor="white"); plt.close(fig)
    return p


# ------------------------------------------------------------------ 3
def context_effect(assets, out):
    z = np.load(os.path.join(assets, "worked_example.npz"), allow_pickle=True)
    lab = z["labels"].astype(int)
    clip = str(z["clip"])

    def smooth_norm(x):
        s = np.convolve(x, np.ones(31) / 31, mode="same")
        return (s - s.min()) / (s.max() - s.min() + 1e-12)

    m, mm = smooth_norm(z["scores_matched"]), smooth_norm(z["scores_mismatched"])

    fig, ax = plt.subplots(figsize=(6.9, 2.5))
    idx = np.where(lab == 1)[0]
    if len(idx):
        ax.axvspan(idx[0], idx[-1], color=BAD, alpha=0.12, lw=0,
                   label="ground-truth event")
    ax.plot(m, color=CTX, lw=1.9, label='matched:  "a campus walkway with pedestrians"')
    ax.plot(mm, color="#8895A0", lw=1.6, ls=(0, (5, 2.5)),
            label='mismatched:  "an industrial quality-inspection image…"')

    if len(idx):
        a, b = idx[0], idx[-1]
        ax.annotate("", xy=(b + 12, m[a:b].mean()), xytext=(b + 12, mm[a:b].mean()),
                    arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9))
        ax.text(b + 20, (m[a:b].mean() + mm[a:b].mean()) / 2,
                "separation during\nthe event", fontsize=6.2, color=INK,
                va="center", linespacing=1.4)

    ax.set_xlim(0, len(m)); ax.set_ylim(-0.08, 1.30)
    ax.set_xlabel("frame index"); ax.set_ylabel("smoothed score")
    ax.set_yticks([0, 0.5, 1.0])
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.legend(fontsize=6.2, loc="upper left", frameon=True, framealpha=0.95,
              edgecolor=GRID, handlelength=2.0)
    ax.set_title(f"Same clip ({clip}), same frozen models — only the sentence differs",
                 fontsize=7.2, color=INK, pad=5)
    fig.tight_layout()
    p = os.path.join(out, "fig_context_effect.png")
    fig.savefig(p, dpi=400, facecolor="white"); plt.close(fig)
    return p


# ------------------------------------------------------------------ 4
def camera_baselines(assets, out):
    z = np.load(os.path.join(assets, "embedding_sample.npz"), allow_pickle=True)
    feats, view = z["feats"], np.array([str(v) for v in z["view"]])
    views = sorted(set(view.tolist()))

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    p50 = PCA(n_components=min(50, feats.shape[1]), random_state=0).fit_transform(feats)
    xy = TSNE(n_components=2, init="pca", perplexity=30, random_state=0,
              max_iter=1000).fit_transform(p50)

    # each camera's similarity to the global mean: the baseline offset that
    # makes raw scores from different cameras incomparable
    mu = feats.mean(0); mu /= np.linalg.norm(mu)
    sim = feats @ mu

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.85),
                             gridspec_kw={"width_ratios": [1.05, 1.0]})
    cmap = plt.get_cmap("tab20")
    for i, v in enumerate(views):
        m = view == v
        axes[0].scatter(xy[m, 0], xy[m, 1], s=4.5, alpha=0.75,
                        color=cmap(i % 20), linewidths=0, label=v)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].set_title("Frame embeddings, coloured by camera", pad=4)
    axes[0].legend(fontsize=5.0, ncol=4, loc="lower center", frameon=True,
                   edgecolor=GRID, framealpha=0.95, handletextpad=0.2,
                   columnspacing=0.7, markerscale=1.6,
                   title="camera", title_fontsize=5.2)
    _panel_label(axes[0], "(a)")

    data = [sim[view == v] for v in views]
    bp = axes[1].boxplot(data, patch_artist=True, widths=0.62,
                         medianprops=dict(color=INK, lw=1.0),
                         flierprops=dict(marker=".", ms=1.6, mfc=MUTED,
                                         mec="none", alpha=0.5))
    for i, b in enumerate(bp["boxes"]):
        b.set(facecolor=cmap(i % 20), alpha=0.75, edgecolor="#8895A0", lw=0.6)
    for k in ("whiskers", "caps"):
        for a in bp[k]:
            a.set(color="#8895A0", lw=0.7)
    axes[1].set_xticklabels(views, fontsize=5.8)
    axes[1].set_xlabel("camera view"); axes[1].set_ylabel("similarity to global mean")
    axes[1].grid(axis="y", color=GRID, lw=0.5); axes[1].set_axisbelow(True)
    axes[1].set_title("Each camera sits at its own baseline", pad=4)
    _panel_label(axes[1], "(b)")

    fig.text(0.5, 0.012,
             "Twelve cameras occupy twelve regions of the embedding space (a) "
             "and score at different baselines (b),\nso the same numeric score "
             "means different things under different cameras.",
             ha="center", fontsize=6.3, color=MUTED, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.085, 1, 1), w_pad=1.4)
    p = os.path.join(out, "fig_camera_baselines.png")
    fig.savefig(p, dpi=400, facecolor="white"); plt.close(fig)
    return p


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=os.path.join(root, "figure_assets"))
    ap.add_argument("--out", default=os.path.join(root, "docs", "09_paper", "figures"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for fn in (dataset_samples, detection_example, context_effect, camera_baselines):
        try:
            p = fn(a.assets, a.out)
            print(f"  {fn.__name__:<20} -> {p}")
        except Exception as e:                       # keep going; report the gap
            print(f"  {fn.__name__:<20} FAILED: {type(e).__name__}: {e}")
