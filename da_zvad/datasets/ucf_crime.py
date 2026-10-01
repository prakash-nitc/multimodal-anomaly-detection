"""UCF-Crime adapter, test split only (Sultani et al., CVPR 2018).

    <root>/videos/Abuse028_x264.mp4 ...              (290 test videos)
    <root>/Temporal_Anomaly_Annotation.txt

Fetched by ``notebooks/fetch_ucf_crime_test.py``, which pulls only these files
out of the official archive.

Ground truth
------------
Each annotation line is

    Abuse028_x264.mp4  Abuse  165  240  -1  -1

-- video, class, then up to two anomalous intervals as start/end FRAME NUMBERS,
with -1 marking an absent interval. Normal videos carry -1 throughout. The
numbers are treated as 1-based and inclusive, following the MATLAB evaluation
released with the benchmark; frame n of the file is decoded index n-1 here. At
the subsampling steps used on this benchmark a one-frame convention difference
changes no sampled label in practice, but it is fixed in one place
(``interval_labels``) so that it is at least consistent.

Subsampling
-----------
UCF-Crime is ~30 fps and the test split alone is on the order of a million
frames, so only every ``frame_step``-th frame is decoded and scored, and labels
are taken at exactly those frames. Only the sampled frames are written to the
JPEG cache; decoding every frame to disk first, as the Avenue adapter does,
would cost tens of GB for no benefit. The cache is keyed by step, so changing
the step never silently reuses a cache built for another one.

Requires cv2, imported lazily.
"""
from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import AnomalyDataset, FrameSequence

_IMG_EXT = ".jpg"

Interval = Tuple[int, int]


def parse_annotation(path: str) -> Dict[str, Tuple[str, List[Interval]]]:
    """-> {video filename: (class, [(start, end), ...])}, intervals 1-based inclusive."""
    out: Dict[str, Tuple[str, List[Interval]]] = {}
    for line in open(path, encoding="utf-8"):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 6:
            raise ValueError(f"malformed annotation line: {line!r}")
        name, cls = parts[0], parts[1]
        nums = [int(x) for x in parts[2:6]]
        ivs = [(s, e) for s, e in (nums[0:2], nums[2:4]) if s >= 0 and e >= 0]
        for s, e in ivs:
            if e < s:
                raise ValueError(f"{name}: interval end {e} before start {s}")
        out[name] = (cls, ivs)
    return out


def interval_labels(n_frames: int, intervals: List[Interval]) -> np.ndarray:
    """Per-frame 0/1 labels for decoded indices 0..n-1 from 1-based inclusive intervals."""
    y = np.zeros(n_frames, dtype=int)
    for s, e in intervals:
        lo, hi = max(s - 1, 0), min(e - 1, n_frames - 1)
        if lo <= hi:
            y[lo:hi + 1] = 1
    return y


class UCFCrimeDataset(AnomalyDataset):
    def __init__(self, root: Optional[str], frame_step: int = 16,
                 cache_dir: Optional[str] = None, limit: Optional[int] = None):
        if not root:
            raise ValueError("UCFCrimeDataset requires data_root.")
        self.root = os.path.expanduser(root)
        self.frame_step = max(1, int(frame_step))
        self.limit = limit
        self.video_root = os.path.join(self.root, "videos")
        self.ann_path = os.path.join(self.root, "Temporal_Anomaly_Annotation.txt")
        for p in (self.video_root, self.ann_path):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"{p!r} missing -- run notebooks/fetch_ucf_crime_test.py first.")
        self.cache_dir = cache_dir or os.path.join(
            self.root, f"_frames_step{self.frame_step}")
        self.annotation = parse_annotation(self.ann_path)

    # ---- decode the sampled frames once, reuse thereafter -------------
    def _extract(self, fname: str) -> Tuple[List[str], int]:
        """-> (paths of sampled frames, total frame count of the video)."""
        import cv2  # lazy

        stem = os.path.splitext(fname)[0]
        out_dir = os.path.join(self.cache_dir, stem)
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
            # grab() advances without decoding pixels; retrieve() only on
            # sampled frames. Much faster than read() on every frame.
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
        names = sorted(self.annotation)
        on_disk = set(os.listdir(self.video_root))
        missing = [n for n in names if n not in on_disk]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} annotated test videos not on disk (e.g. {missing[:3]}) "
                "-- rerun the fetch script.")
        if self.limit:
            names = names[:self.limit]

        out: List[FrameSequence] = []
        for fname in names:
            cls, ivs = self.annotation[fname]
            paths, total = self._extract(fname)
            full = interval_labels(total, ivs)
            for s, e in ivs:
                if e > total:
                    warnings.warn(f"ucf_crime/{fname}: interval ends at {e} "
                                  f"but the video has {total} frames")
            # Sampled frame k sits at decoded index int(filename); take its label.
            idx = np.array([int(os.path.splitext(os.path.basename(p))[0]) for p in paths])
            out.append(FrameSequence(
                frames=paths,
                labels=full[idx],
                name=f"ucf_crime/{os.path.splitext(fname)[0]}",
            ))
        return out
