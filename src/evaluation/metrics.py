"""
Evaluation Metrics for Anomaly Detection
==========================================
Image-level AUROC, F1-Score, and per-category result formatting.
"""

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score


def compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute image-level Area Under the ROC Curve.

    Args:
        scores: Anomaly scores (higher = more anomalous).
        labels: Binary ground truth labels (0=normal, 1=anomalous).

    Returns:
        AUROC value in [0, 1].
    """
    if len(np.unique(labels)) < 2:
        # Only one class present — AUROC is undefined
        return float("nan")
    return roc_auc_score(labels, scores)


def compute_optimal_f1(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute F1-score at the optimal threshold.

    Searches over all unique score values to find the threshold
    that maximizes F1-score.

    Args:
        scores: Anomaly scores.
        labels: Binary ground truth labels.

    Returns:
        Dict with 'f1', 'precision', 'recall', 'threshold'.
    """
    thresholds = np.unique(scores)
    best_f1 = 0.0
    best_result = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "threshold": 0.0}

    for t in thresholds:
        preds = (scores >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_result = {
                "f1": f1,
                "precision": precision_score(labels, preds, zero_division=0),
                "recall": recall_score(labels, preds, zero_division=0),
                "threshold": float(t),
            }

    return best_result


def format_results_table(
    results: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None,
) -> str:
    """Format per-category results as a printable table.

    Args:
        results: Dict mapping category name -> dict of metric values.
        metrics: List of metric names to include. Defaults to ['auroc', 'f1'].

    Returns:
        Formatted table string.
    """
    if metrics is None:
        metrics = ["auroc", "f1"]

    # Header
    header = f"{'Category':<15}" + "".join(f"{m.upper():>10}" for m in metrics)
    separator = "-" * len(header)
    lines = [separator, header, separator]

    # Per-category rows
    for category, metric_vals in results.items():
        if category == "MEAN":
            continue
        row = f"{category:<15}"
        for m in metrics:
            val = metric_vals.get(m, float("nan"))
            if np.isnan(val):
                row += f"{'N/A':>10}"
            else:
                row += f"{val:>10.1%}"
        lines.append(row)

    # Mean row
    lines.append(separator)
    if "MEAN" in results:
        row = f"{'MEAN':<15}"
        for m in metrics:
            val = results["MEAN"].get(m, float("nan"))
            if np.isnan(val):
                row += f"{'N/A':>10}"
            else:
                row += f"{val:>10.1%}"
        lines.append(row)
    lines.append(separator)

    return "\n".join(lines)
