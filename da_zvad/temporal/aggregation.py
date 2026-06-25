"""M2 - Training-free temporal aggregation.

Per-frame scores are noisy and lack temporal context. A centered moving average
smooths the score sequence so short anomalous events stand out and isolated
single-frame fluctuations are suppressed -- all without any learned parameters.

This is the training-free variant of the temporal module. A learned lightweight
adapter (few-shot) is a planned optional upgrade and would slot in behind the
same interface.
"""
from __future__ import annotations

import numpy as np


def moving_average(scores: np.ndarray, window: int = 5) -> np.ndarray:
    """Centered moving average with edge handling.

    Args:
        scores: 1-D array of per-frame anomaly scores.
        window: smoothing window in frames (>=1). window<=1 is a no-op.
    """
    scores = np.asarray(scores, dtype=float)
    if window <= 1 or scores.size == 0:
        return scores.copy()
    window = min(window, scores.size)
    kernel = np.ones(window) / window
    # 'same' keeps length; reflect padding avoids dark edges
    pad = window // 2
    padded = np.pad(scores, pad, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="same")[pad: pad + scores.size]
    return smoothed


class TemporalAggregator:
    """Wraps the temporal smoothing so it can be toggled/swapped in the pipeline."""

    def __init__(self, window: int = 5):
        self.window = window

    def apply(self, scores: np.ndarray) -> np.ndarray:
        return moving_average(scores, self.window)

    def __repr__(self) -> str:
        return f"TemporalAggregator(window={self.window})"
