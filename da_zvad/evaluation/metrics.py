"""Evaluation metrics for (frame-level) anomaly detection.

Primary metric is AUROC -- the standard for MVTec AD (image-level) and for
ShanghaiTech / CUHK Avenue (frame-level). Uses scikit-learn when available with a
dependency-free NumPy fallback so the metric harness always runs.
"""
from __future__ import annotations

from typing import Dict
import numpy as np


def _auroc_numpy(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney U), no sklearn needed."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    avg_rank = {i: (cum[i] - (counts[i] - 1) / 2.0) for i in range(len(counts))}
    ranks = np.array([avg_rank[i] for i in inv])
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = ranks[pos].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def frame_auroc(scores, labels) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(labels)) < 2:
            return float("nan")
        return float(roc_auc_score(labels, scores))
    except Exception:
        return _auroc_numpy(scores, labels)


def average_precision(scores, labels) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    try:
        from sklearn.metrics import average_precision_score
        if len(np.unique(labels)) < 2:
            return float("nan")
        return float(average_precision_score(labels, scores))
    except Exception:
        return float("nan")


def best_f1(scores, labels) -> float:
    """Best F1 over all thresholds."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) < 2:
        return float("nan")
    thr = np.unique(scores)
    best = 0.0
    for t in thr:
        pred = (scores >= t).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        denom = 2 * tp + fp + fn
        if denom:
            best = max(best, 2 * tp / denom)
    return float(best)


def score_gap(scores, labels) -> float:
    """Mean(anomalous scores) - Mean(normal scores). Quick separability check."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if (labels == 1).sum() == 0 or (labels == 0).sum() == 0:
        return float("nan")
    return float(scores[labels == 1].mean() - scores[labels == 0].mean())


def summarize(scores, labels) -> Dict[str, float]:
    return {
        "auroc": frame_auroc(scores, labels),
        "ap": average_precision(scores, labels),
        "best_f1": best_f1(scores, labels),
        "score_gap": score_gap(scores, labels),
    }
