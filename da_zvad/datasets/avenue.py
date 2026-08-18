"""CUHK Avenue adapter (surveillance benchmark).

Test videos ship as ``.avi`` files with per-video ``.mat`` ground truth
(``volLabel``: one pixel mask per frame). A frame is anomalous iff its mask has
any nonzero pixel -- that reduction to frame-level labels happens here, once.

    <root>/testing_videos/01.avi ... 21.avi
    <root>/ground_truth_demo/testing_label_mask/1_label.mat ...

Frames are decoded to disk on first use and referenced by path thereafter, which
is the contract ShanghaiTech already uses. Decoding into memory instead would
hold every frame of all 21 videos as a decoded image simultaneously -- several
GB for a benchmark that is a few hundred MB on disk -- and would repeat the
decode on every run. The extracted cache carries a ``.complete`` sentinel per
video so an interrupted decode is redone rather than silently reused as a
partial sequence.

Requires cv2 (decoding) and scipy (.mat parsing); both are imported lazily so
the framework imports cleanly on machines without them.
"""
from __future__ import annotations

import os
import re
import warnings
from typing import List, Optional
import numpy as np

from .base import AnomalyDataset, FrameSequence
from .shanghaitech import _first_existing

_IMG_EXT = ".jpg"
_VIDEO_EXTS = (".avi", ".mp4")


class AvenueDataset(AnomalyDataset):
    def __init__(self, root: Optional[str], frame_step: int = 1,
                 cache_dir: Optional[str] = None):
        if not root:
            raise ValueError("AvenueDataset requires data_root.")
        self.root = root
        self.frame_step = max(1, frame_step)

        self.video_root = _first_existing(
            os.path.join(root, "testing_videos"),
            os.path.join(root, "testing", "videos"),
            os.path.join(root, "Avenue Dataset", "testing_videos"),
        )
        self.gt_root = _first_existing(
            os.path.join(root, "ground_truth_demo", "testing_label_mask"),
            os.path.join(root, "testing_label_mask"),
            os.path.join(root, "ground_truth", "testing_label_mask"),
            os.path.join(root, "Avenue Dataset", "ground_truth_demo",
                         "testing_label_mask"),
        )
        if self.video_root is None:
            raise FileNotFoundError(
                f"Avenue testing videos not found under {root!r} "
                "(looked for testing_videos, testing/videos)."
            )
        if self.gt_root is None:
            warnings.warn(
                f"Avenue ground truth not found under {root!r} -- sequences "
                "will have all-zero labels and AUROC will be NaN."
            )
        self.cache_dir = cache_dir or os.path.join(root, "_frames_cache")

    # ---- decode once, reuse thereafter --------------------------------
    def _extract(self, fname: str) -> List[str]:
        import cv2  # lazy

        stem = os.path.splitext(fname)[0]
        out_dir = os.path.join(self.cache_dir, stem)
        sentinel = os.path.join(out_dir, ".complete")

        def _listing() -> List[str]:
            return sorted(
                os.path.join(out_dir, f)
                for f in os.listdir(out_dir) if f.endswith(_IMG_EXT)
            )

        if os.path.isfile(sentinel):
            return _listing()

        os.makedirs(out_dir, exist_ok=True)
        cap = cv2.VideoCapture(os.path.join(self.video_root, fname))
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imwrite(os.path.join(out_dir, f"{n:06d}{_IMG_EXT}"), frame)
            n += 1
        cap.release()

        if n == 0:
            raise RuntimeError(
                f"Decoded 0 frames from {fname!r}. OpenCV could not read the "
                "video -- the codec is probably unavailable in this build of "
                "opencv-python-headless."
            )
        open(sentinel, "w").close()
        return _listing()

    # ---- pixel masks -> frame labels ----------------------------------
    def _frame_labels(self, video_id: int, n_frames: int) -> np.ndarray:
        if self.gt_root is None:
            return np.zeros(n_frames, dtype=int)
        mat_path = os.path.join(self.gt_root, f"{video_id}_label.mat")
        if not os.path.isfile(mat_path):
            warnings.warn(f"No GT .mat for video {video_id}; labels set to 0.")
            return np.zeros(n_frames, dtype=int)

        from scipy.io import loadmat  # lazy
        vol = loadmat(mat_path)["volLabel"].ravel()
        return np.array([int(np.asarray(m).any()) for m in vol], dtype=int)

    def sequences(self) -> List[FrameSequence]:
        vids = sorted(
            f for f in os.listdir(self.video_root)
            if f.lower().endswith(_VIDEO_EXTS)
        )
        if not vids:
            raise FileNotFoundError(
                f"No .avi/.mp4 files in {self.video_root!r}."
            )

        out: List[FrameSequence] = []
        for fname in vids:
            stem = os.path.splitext(fname)[0]
            m = re.match(r"(\d+)", stem)
            video_id = int(m.group(1)) if m else -1

            paths = self._extract(fname)
            labels = self._frame_labels(video_id, len(paths))

            if len(labels) != len(paths):
                warnings.warn(
                    f"avenue/{stem}: {len(paths)} frames vs {len(labels)} GT "
                    "entries -- truncating to the shorter length."
                )
            n = min(len(paths), len(labels))
            paths, labels = paths[:n], labels[:n]

            # one place where subsampling happens, identically for both
            idx = np.arange(0, n, self.frame_step)
            out.append(FrameSequence(
                frames=[paths[i] for i in idx],
                labels=labels[idx],
                name=f"avenue/{stem}",
            ))
        return out
