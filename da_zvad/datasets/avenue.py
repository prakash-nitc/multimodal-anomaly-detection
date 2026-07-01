"""CUHK Avenue adapter (surveillance benchmark).

Test videos are .avi files; ground truth ships as per-video .mat files of
pixel masks (``volLabel``). A frame is anomalous iff its mask has any nonzero
pixel -- that reduction to frame-level labels happens here, once.

    <root>/testing_videos/01.avi ... 21.avi
    <root>/ground_truth_demo/testing_label_mask/1_label.mat ...

Requires cv2 (frame extraction) and scipy (.mat parsing); both are imported
lazily so the framework imports cleanly on machines without them.
"""
from __future__ import annotations

import os
import re
import warnings
from typing import List, Optional
import numpy as np

from .base import AnomalyDataset, FrameSequence
from .shanghaitech import _first_existing


class AvenueDataset(AnomalyDataset):
    def __init__(self, root: Optional[str], frame_step: int = 1):
        if not root:
            raise ValueError("AvenueDataset requires data_root.")
        self.root = root
        self.frame_step = max(1, frame_step)

        self.video_root = _first_existing(
            os.path.join(root, "testing_videos"),
            os.path.join(root, "testing", "videos"),
        )
        self.gt_root = _first_existing(
            os.path.join(root, "ground_truth_demo", "testing_label_mask"),
            os.path.join(root, "testing_label_mask"),
        )
        if self.video_root is None:
            raise FileNotFoundError(f"Avenue testing videos not found under {root!r}.")
        if self.gt_root is None:
            warnings.warn(f"Avenue GT masks not found under {root!r}; labels will be 0.")

    def _frame_labels(self, video_id: int, n_frames: int) -> np.ndarray:
        if self.gt_root is None:
            return np.zeros(n_frames, dtype=int)
        mat_path = os.path.join(self.gt_root, f"{video_id}_label.mat")
        if not os.path.isfile(mat_path):
            warnings.warn(f"No GT .mat for video {video_id}; labels set to 0.")
            return np.zeros(n_frames, dtype=int)

        from scipy.io import loadmat  # lazy
        vol = loadmat(mat_path)["volLabel"].ravel()
        labels = np.array([int(np.asarray(m).any()) for m in vol], dtype=int)
        if len(labels) != n_frames:
            warnings.warn(
                f"Avenue video {video_id}: {n_frames} frames vs {len(labels)} "
                "GT entries -- truncating to the shorter length."
            )
        return labels

    def sequences(self) -> List[FrameSequence]:
        import cv2  # lazy
        from PIL import Image

        out: List[FrameSequence] = []
        vids = sorted(
            f for f in os.listdir(self.video_root)
            if f.lower().endswith((".avi", ".mp4"))
        )
        for fname in vids:
            m = re.match(r"(\d+)", os.path.splitext(fname)[0])
            video_id = int(m.group(1)) if m else -1

            cap = cv2.VideoCapture(os.path.join(self.video_root, fname))
            frames = []
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if i % self.frame_step == 0:
                    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                i += 1
            cap.release()

            full_labels = self._frame_labels(video_id, i)
            n = min(i, len(full_labels))
            idx = np.arange(0, n, self.frame_step)
            k = min(len(frames), len(idx))
            out.append(FrameSequence(
                frames=frames[:k],
                labels=full_labels[idx][:k],
                name=f"avenue/{os.path.splitext(fname)[0]}",
            ))
        return out
