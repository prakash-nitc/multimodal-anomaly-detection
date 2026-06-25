"""Dataset interface shared by every adapter.

A dataset yields a ``FrameSequence``: an ordered list of frames (PIL images or
paths) plus per-frame binary labels (0 normal, 1 anomalous). The pipeline never
needs to know which dataset it came from.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Any
import numpy as np


@dataclass
class FrameSequence:
    frames: List[Any]               # PIL images, file paths, or None (synthetic)
    labels: np.ndarray              # shape (T,), values in {0, 1}
    name: str = "sequence"
    raw_scores: Optional[np.ndarray] = None   # set for synthetic / precomputed runs

    def __len__(self) -> int:
        return len(self.labels)


class AnomalyDataset:
    """Base class. Subclasses implement ``sequences()``."""

    def sequences(self) -> List[FrameSequence]:
        raise NotImplementedError


class SyntheticDataset(AnomalyDataset):
    """A toy normal -> anomalous -> normal sequence with precomputed raw scores.

    Used to smoke-test the full pipeline end-to-end with no model or data: it
    exercises temporal aggregation, detection, metrics and the reasoning interface.
    """

    def __init__(self, length: int = 60, anomaly_span=(20, 35), seed: int = 0):
        self.length = length
        self.anomaly_span = anomaly_span
        self.seed = seed

    def sequences(self) -> List[FrameSequence]:
        rng = np.random.default_rng(self.seed)
        T = self.length
        labels = np.zeros(T, dtype=int)
        a, b = self.anomaly_span
        labels[a:b] = 1
        # raw per-frame "anomaly scores": low normally, high during the event,
        # plus noise and a couple of contamination spikes (so smoothing matters).
        base = np.where(labels == 1, 0.78, 0.25)
        noise = rng.normal(0, 0.08, T)
        scores = np.clip(base + noise, 0, 1)
        for idx in rng.choice(np.where(labels == 0)[0], size=2, replace=False):
            scores[idx] = 0.7   # isolated false spikes temporal smoothing should damp
        return [FrameSequence(frames=[None] * T, labels=labels,
                              name="synthetic", raw_scores=scores)]
