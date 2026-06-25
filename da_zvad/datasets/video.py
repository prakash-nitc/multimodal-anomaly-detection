"""Video adapter for surveillance benchmarks (ShanghaiTech, CUHK Avenue).

Extracts frames at a target FPS via OpenCV. Frame-level ground-truth label
parsing is dataset-specific and is the piece to wire up during dataset setup;
``_load_labels`` is a clearly-marked stub for now.

NOTE: OpenCV (cv2) is imported lazily -- this module only needs it when actually
reading video on the GPU machine, not when importing the framework elsewhere.
"""
from __future__ import annotations

import os
from typing import List, Optional
import numpy as np

from .base import AnomalyDataset, FrameSequence


class VideoDataset(AnomalyDataset):
    def __init__(self, root: Optional[str], sample_fps: int = 2, name: str = "video"):
        if not root:
            raise ValueError("VideoDataset requires data_root.")
        self.root = root
        self.sample_fps = sample_fps
        self.name = name

    def _extract_frames(self, video_path: str):
        import cv2  # lazy
        from PIL import Image

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        step = max(1, int(round(fps / self.sample_fps)))
        frames, i = [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
            i += 1
        cap.release()
        return frames

    def _load_labels(self, video_path: str, n_frames: int) -> np.ndarray:
        # TODO: parse the dataset's frame-level ground truth (per-benchmark format).
        # Returning zeros lets the pipeline run; AUROC is reported once GT is wired.
        return np.zeros(n_frames, dtype=int)

    def sequences(self) -> List[FrameSequence]:
        vids = [os.path.join(self.root, f) for f in sorted(os.listdir(self.root))
                if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
        out = []
        for v in vids:
            frames = self._extract_frames(v)
            labels = self._load_labels(v, len(frames))
            out.append(FrameSequence(frames=frames, labels=labels,
                                     name=f"{self.name}/{os.path.basename(v)}"))
        return out
