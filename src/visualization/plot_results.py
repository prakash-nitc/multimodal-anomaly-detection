"""
Result Visualization
====================
Generates bar charts and summary plots for anomaly detection results.
"""

from typing import Dict, Optional
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt


def plot_category_auroc(
    results: Dict[str, Dict[str, float]],
    save_path: str,
    title: str = "CLIP Zero-Shot Anomaly Detection — MVTec AD",
) -> str:
    """Generate a horizontal bar chart of per-category AUROC scores.

    Args:
        results: Dict mapping category name -> {'auroc': float, ...}.
                 Should include a 'MEAN' key for the average.
        save_path: Path to save the figure.
        title: Plot title.

    Returns:
        Path where the figure was saved.
    """
    # Separate categories from mean
    categories = [c for c in results if c != "MEAN"]
    aurocs = [results[c]["auroc"] for c in categories]

    # Sort by AUROC descending
    sorted_pairs = sorted(zip(categories, aurocs), key=lambda x: x[1], reverse=True)
    categories, aurocs = zip(*sorted_pairs)
    categories, aurocs = list(categories), list(aurocs)

    mean_auroc = results.get("MEAN", {}).get("auroc", np.mean(aurocs))

    # Color coding: green for good (>85%), yellow for moderate (70-85%), red for poor (<70%)
    colors = []
    for a in aurocs:
        if a >= 0.85:
            colors.append("#2ecc71")  # green
        elif a >= 0.70:
            colors.append("#f39c12")  # orange
        else:
            colors.append("#e74c3c")  # red

    fig, ax = plt.subplots(figsize=(10, 8))

    bars = ax.barh(range(len(categories)), aurocs, color=colors, edgecolor="white", height=0.7)

    # Add value labels on bars
    for bar, auroc in zip(bars, aurocs):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{auroc:.1%}",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    # Mean line
    ax.axvline(x=mean_auroc, color="#3498db", linestyle="--", linewidth=2, label=f"Mean: {mean_auroc:.1%}")

    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories, fontsize=11)
    ax.set_xlabel("Image-level AUROC", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.12)
    ax.legend(fontsize=11, loc="lower right")
    ax.invert_yaxis()

    # Style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure saved to: {save_path}")
    return str(save_path)
