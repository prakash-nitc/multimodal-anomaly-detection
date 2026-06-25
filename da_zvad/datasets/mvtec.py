"""MVTec AD adapter (industrial images).

Treats one category's test split as a single 'sequence' of frames so the same
pipeline handles images and video uniformly. Labels: 0 = good, 1 = any defect.
"""
from __future__ import annotations

import os
from typing import List, Optional
import numpy as np

from .base import AnomalyDataset, FrameSequence


class MVTecDataset(AnomalyDataset):
    def __init__(self, root: Optional[str], category: Optional[str]):
        if not root or not category:
            raise ValueError("MVTecDataset requires data_root and category.")
        self.root = root
        self.category = category

    def sequences(self) -> List[FrameSequence]:
        from PIL import Image  # lazy

        test_dir = os.path.join(self.root, self.category, "test")
        if not os.path.isdir(test_dir):
            raise FileNotFoundError(f"MVTec test dir not found: {test_dir}")

        paths: List[str] = []
        labels: List[int] = []
        for defect in sorted(os.listdir(test_dir)):
            d = os.path.join(test_dir, defect)
            if not os.path.isdir(d):
                continue
            label = 0 if defect == "good" else 1
            for f in sorted(os.listdir(d)):
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    paths.append(os.path.join(d, f))
                    labels.append(label)

        # frames are loaded lazily by the pipeline via _load
        frames = [self._loader(p, Image) for p in paths]
        return [FrameSequence(frames=frames, labels=np.array(labels),
                              name=f"mvtec/{self.category}")]

    @staticmethod
    def _loader(path, Image):
        # return a thunk-free PIL image; categories are small enough to hold
        return Image.open(path).convert("RGB")
