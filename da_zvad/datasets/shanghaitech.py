"""ShanghaiTech Campus adapter (surveillance benchmark).

The test split ships as per-clip frame folders plus per-clip frame-level
ground-truth arrays (.npy with one 0/1 entry per original frame):

    <root>/testing/frames/01_0014/*.jpg
    <root>/testing/test_frame_mask/01_0014.npy

Frames are kept as file paths (the encoder loads them lazily) so the ~40k-frame
test set never sits in RAM. ``frame_step`` subsamples frames and labels with the
SAME indices -- score/label misalignment is the classic silent bug in VAD
evaluation, so that alignment lives in exactly one place here.
"""
from __future__ import annotations

import os
import warnings
from typing import List, Optional
import numpy as np

from .base import AnomalyDataset, FrameSequence

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _first_existing(*candidates: str) -> Optional[str]:
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    return None


class ShanghaiTechDataset(AnomalyDataset):
    def __init__(self, root: Optional[str], frame_step: int = 1):
        if not root:
            raise ValueError("ShanghaiTechDataset requires data_root.")
        self.root = root
        self.frame_step = max(1, frame_step)

        self.frames_root = _first_existing(
            os.path.join(root, "testing", "frames"),
            os.path.join(root, "frames"),
            os.path.join(root, "testing_frames"),
        )
        self.mask_root = _first_existing(
            os.path.join(root, "testing", "test_frame_mask"),
            os.path.join(root, "test_frame_mask"),
            os.path.join(root, "testing", "frame_masks"),
            os.path.join(root, "frame_masks"),
        )
        if self.frames_root is None:
            raise FileNotFoundError(
                f"ShanghaiTech frames folder not found under {root!r} "
                "(looked for testing/frames, frames, testing_frames)."
            )
        if self.mask_root is None:
            warnings.warn(
                f"ShanghaiTech ground-truth masks not found under {root!r} -- "
                "sequences will have all-zero labels and AUROC will be NaN."
            )

    def sequences(self) -> List[FrameSequence]:
        out: List[FrameSequence] = []
        for clip in sorted(os.listdir(self.frames_root)):
            clip_dir = os.path.join(self.frames_root, clip)
            if not os.path.isdir(clip_dir):
                continue
            paths = sorted(
                os.path.join(clip_dir, f)
                for f in os.listdir(clip_dir)
                if f.lower().endswith(_IMG_EXTS)
            )
            if not paths:
                continue

            if self.mask_root is not None:
                mask_path = os.path.join(self.mask_root, clip + ".npy")
                if os.path.isfile(mask_path):
                    labels = np.load(mask_path).astype(int).ravel()
                else:
                    warnings.warn(f"No GT mask for clip {clip!r}; labels set to 0.")
                    labels = np.zeros(len(paths), dtype=int)
            else:
                labels = np.zeros(len(paths), dtype=int)

            if len(labels) != len(paths):
                warnings.warn(
                    f"{clip}: {len(paths)} frames vs {len(labels)} labels -- "
                    "truncating to the shorter length."
                )
                n = min(len(paths), len(labels))
                paths, labels = paths[:n], labels[:n]

            # one place where subsampling happens, identically for both
            idx = np.arange(0, len(paths), self.frame_step)
            out.append(FrameSequence(
                frames=[paths[i] for i in idx],
                labels=labels[idx],
                name=f"shanghaitech/{clip}",
            ))
        return out
