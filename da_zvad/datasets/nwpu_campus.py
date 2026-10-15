"""NWPU Campus adapter, test split only (Cao et al., CVPR 2023).

    <root>/videos/D001_01.avi ...       (242 test videos, 43 scenes, 25 fps)
    <root>/NWPU_Campus_gt.npz           (one 0/1 array per video, one entry per frame)

Fetched by ``notebooks/fetch_nwpu_campus_test.py``.

The scene is the name's prefix: ``D001_01`` is video 01 of scene D001. The
ground truth gives one label per decoded frame (0-based); on the videos checked
(16 Oct 2026) the decoded frame count, the container's frame count and the
ground-truth length agree exactly, and any disagreement is reported here.

Only every ``frame_step``-th frame is decoded to disk and scored, as on
UCF-Crime: the test split is 384,059 full-HD frames.

Requires cv2, imported lazily.
"""
from __future__ import annotations

import os
import warnings
from typing import List, Optional, Tuple

import numpy as np

from .base import AnomalyDataset, FrameSequence

_IMG_EXT = ".jpg"

# Supplementary Table 1/2 of the paper: the scenes whose anomalies include a
# class the authors mark scene-dependent ("s.d."). Each also has other classes.
SCENE_DEPENDENT_SCENES = {
    "D003": "Trucks",
    "D036": "Photographing in restricted area",
    "D038": "Photographing in restricted area",
    "D129": "Cycling on footpath",
}


def scene_of(name: str) -> str:
    """'nwpu_campus/D001_01' or 'D001_01' -> 'D001'."""
    return os.path.basename(str(name)).split("_")[0]


class NWPUCampusDataset(AnomalyDataset):
    def __init__(self, root: Optional[str], frame_step: int = 16,
                 cache_dir: Optional[str] = None, limit: Optional[int] = None):
        if not root:
            raise ValueError("NWPUCampusDataset requires data_root.")
        self.root = os.path.expanduser(root)
        self.frame_step = max(1, int(frame_step))
        self.limit = limit
        self.video_root = os.path.join(self.root, "videos")
        self.gt_path = os.path.join(self.root, "NWPU_Campus_gt.npz")
        for p in (self.video_root, self.gt_path):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"{p!r} missing -- run notebooks/fetch_nwpu_campus_test.py first.")
        self.cache_dir = cache_dir or os.path.join(self.root, f"_frames_step{self.frame_step}")
        z = np.load(self.gt_path)
        self.gt = {k: np.asarray(z[k]).astype(int) for k in z.files}

    def _extract(self, fname: str) -> Tuple[List[str], int]:
        """-> (paths of sampled frames, decoded frame count)."""
        import cv2  # lazy

        out_dir = os.path.join(self.cache_dir, os.path.splitext(fname)[0])
        sentinel = os.path.join(out_dir, ".complete")
        if os.path.isfile(sentinel):
            total = int(open(sentinel).read().strip() or 0)
            paths = sorted(os.path.join(out_dir, f)
                           for f in os.listdir(out_dir) if f.endswith(_IMG_EXT))
            return paths, total

        os.makedirs(out_dir, exist_ok=True)
        cap = cv2.VideoCapture(os.path.join(self.video_root, fname))
        paths, i = [], 0
        while True:
            if not cap.grab():
                break
            if i % self.frame_step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    p = os.path.join(out_dir, f"{i:07d}{_IMG_EXT}")
                    cv2.imwrite(p, frame)
                    paths.append(p)
            i += 1
        cap.release()
        if i == 0:
            raise RuntimeError(f"Decoded 0 frames from {fname!r} -- codec unavailable?")
        with open(sentinel, "w") as fh:
            fh.write(str(i))
        return paths, i

    def sequences(self) -> List[FrameSequence]:
        names = sorted(self.gt)
        on_disk = {os.path.splitext(f)[0] for f in os.listdir(self.video_root)
                   if f.endswith(".avi")}
        missing = [n for n in names if n not in on_disk]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} test videos not on disk (e.g. {missing[:3]}) "
                "-- rerun the fetch script.")
        if self.limit:
            names = names[:self.limit]

        out: List[FrameSequence] = []
        for name in names:
            paths, total = self._extract(name + ".avi")
            gt = self.gt[name]
            if total != len(gt):
                warnings.warn(f"nwpu_campus/{name}: decoded {total} frames, ground truth "
                              f"has {len(gt)}; frames beyond the shorter are dropped")
            idx = np.array([int(os.path.splitext(os.path.basename(p))[0]) for p in paths])
            keep = idx < len(gt)
            out.append(FrameSequence(
                frames=[p for p, k in zip(paths, keep) if k],
                labels=gt[idx[keep]],
                name=f"nwpu_campus/{name}",
            ))
        return out
