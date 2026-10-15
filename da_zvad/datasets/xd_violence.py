"""XD-Violence adapter, test split only (Wu et al., ECCV 2020).

    <root>/videos/<name>.mp4 ...        (800 test videos: 500 violent, 300 normal)
    <root>/annotations.txt              (official temporal annotation)

Fetched by ``notebooks/fetch_xd_violence_test.py``.

Ground truth -- the authors' own convention
-------------------------------------------
Each annotation line is a video name followed by start/end frame pairs:

    v=-fOWSLV6Esw__#1_label_B4-0-0 20 772
    Brick.Mansions.2014__#00-41-25_00-42-36_label_B1-0-0 523 610 900 1700

Only the 500 violent videos appear; normal videos (``_label_A``) are all-normal.
Fourteen lines carry a ``.mp4`` suffix on the name, which the authors' code never
noticed because it matches names by substring; it is stripped here.

The frame convention is taken from the authors' ``list/make_gt.py``
(github.com/Roc-Ng/XDVioDet), not guessed: frames are 0-based, an interval
marks ``gt[start:end] = 1`` (end EXCLUSIVE), the frame count is OpenCV's
``CAP_PROP_FRAME_COUNT``, and each video is cut to ``count - 1`` frames and then
down to a multiple of 16 -- the I3D snippet length. Their evaluation repeats
each snippet's score 16 times and computes AP over every remaining frame.

Sampling
--------
One frame per 16-frame snippet is scored: the snippet's first frame, i.e.
decoded indices 0, 16, 32, ... -- the same frames, and the same step, as on
UCF-Crime. Snippet k therefore covers official frames 16k .. 16k+15, and
repeating its score 16 times reproduces the official evaluation format exactly
(``snippet_gt`` gives the matching ground truth). Labels on the sampled frames
alone are also returned, for the per-frame metrics used elsewhere.

If a video decodes fewer frames than its metadata claims, the official length
is cut to the snippets that actually have a decoded first frame; the number of
videos affected is reported, never hidden.

Requires cv2, imported lazily.
"""
from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import AnomalyDataset, FrameSequence

_IMG_EXT = ".jpg"
SNIPPET = 16

Interval = Tuple[int, int]

CLASSES = {"A": "Normal", "B1": "Fighting", "B2": "Shooting", "B4": "Riot",
           "B5": "Abuse", "B6": "Car accident", "G": "Explosion"}


def video_classes(name: str) -> List[str]:
    """'..._label_G-B2-B6' -> ['Explosion', 'Shooting', 'Car accident']."""
    codes = name.rsplit("_label_", 1)[1].split("-")
    return [CLASSES[c] for c in codes if c != "0"]


def parse_annotation(path: str) -> Dict[str, List[Interval]]:
    """-> {video name without extension: [(start, end), ...]}, 0-based, end exclusive."""
    out: Dict[str, List[Interval]] = {}
    for line in open(path, encoding="utf-8"):
        parts = line.split()
        if not parts:
            continue
        name = parts[0][:-4] if parts[0].endswith(".mp4") else parts[0]
        nums = [int(x) for x in parts[1:]]
        if not nums or len(nums) % 2:
            raise ValueError(f"malformed annotation line: {line!r}")
        ivs = [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]
        for s, e in ivs:
            if e <= s or s < 0:
                raise ValueError(f"{name}: bad interval ({s}, {e})")
        if name in out:
            raise ValueError(f"{name}: annotated twice")
        out[name] = ivs
    return out


def official_length(frame_count: int, snippet: int = SNIPPET) -> int:
    """Frames the official evaluation keeps: count - 1, cut to a multiple of 16."""
    n = max(frame_count - 1, 0)
    return n - n % snippet


def official_gt(n_frames: int, intervals: List[Interval]) -> np.ndarray:
    """Per-frame labels exactly as make_gt.py builds them: gt[start:end] = 1."""
    y = np.zeros(n_frames, dtype=int)
    for s, e in intervals:
        y[s:min(e, n_frames)] = 1
    return y


def snippet_gt(n_snippets: int, intervals: List[Interval]) -> np.ndarray:
    """Ground truth for every official frame of a video scored as n snippets.

    Compare against ``np.repeat(snippet_scores, 16)``: this is the official
    evaluation format.
    """
    return official_gt(n_snippets * SNIPPET, intervals)


class XDViolenceDataset(AnomalyDataset):
    def __init__(self, root: Optional[str], frame_step: int = SNIPPET,
                 cache_dir: Optional[str] = None, limit: Optional[int] = None):
        if not root:
            raise ValueError("XDViolenceDataset requires data_root.")
        if int(frame_step) != SNIPPET:
            raise ValueError("XD-Violence is scored one frame per 16-frame snippet "
                             "(the official evaluation unit); frame_step must be 16.")
        self.root = os.path.expanduser(root)
        self.frame_step = SNIPPET
        self.limit = limit
        self.video_root = os.path.join(self.root, "videos")
        self.ann_path = os.path.join(self.root, "annotations.txt")
        for p in (self.video_root, self.ann_path):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"{p!r} missing -- run notebooks/fetch_xd_violence_test.py first.")
        self.cache_dir = cache_dir or os.path.join(self.root, f"_frames_step{SNIPPET}")
        self.annotation = parse_annotation(self.ann_path)
        self.truncated: List[str] = []

    # ---- decode the sampled frames once, reuse thereafter -------------
    def _extract(self, fname: str) -> Tuple[List[str], int, int, float]:
        """-> (sampled frame paths, decoded frame count, metadata frame count, fps)."""
        import cv2  # lazy

        out_dir = os.path.join(self.cache_dir, os.path.splitext(fname)[0])
        sentinel = os.path.join(out_dir, ".complete")
        if os.path.isfile(sentinel):
            decoded, meta, fps = open(sentinel).read().split()
            paths = sorted(os.path.join(out_dir, f)
                           for f in os.listdir(out_dir) if f.endswith(_IMG_EXT))
            return paths, int(decoded), int(meta), float(fps)

        os.makedirs(out_dir, exist_ok=True)
        cap = cv2.VideoCapture(os.path.join(self.video_root, fname))
        meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        paths, i = [], 0
        while True:
            if not cap.grab():               # advance without decoding pixels
                break
            if i % SNIPPET == 0:
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
            fh.write(f"{i} {meta} {fps}")
        return paths, i, meta, fps

    def sequences(self) -> List[FrameSequence]:
        names = sorted(os.path.splitext(f)[0] for f in os.listdir(self.video_root)
                       if f.endswith(".mp4"))
        missing = [n for n in self.annotation if n not in set(names)]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} annotated test videos not on disk (e.g. {missing[:3]}) "
                "-- rerun the fetch script.")
        unannotated = [n for n in names if "_label_A" not in n and n not in self.annotation]
        if unannotated:
            raise ValueError(f"{len(unannotated)} violent videos have no annotation "
                             f"(e.g. {unannotated[:3]})")
        if self.limit:
            names = names[:self.limit]

        out: List[FrameSequence] = []
        self.truncated = []
        for name in names:
            paths, decoded, meta, fps = self._extract(name + ".mp4")
            if fps and abs(fps - 24.0) > 0.5:
                warnings.warn(f"xd_violence/{name}: {fps:.2f} fps (dataset is 24 fps)")
            n_off = official_length(meta) // SNIPPET           # snippets the authors score
            idx = np.array([int(os.path.splitext(os.path.basename(p))[0]) for p in paths])
            n = min(n_off, int(np.sum(idx < n_off * SNIPPET)))
            if n < n_off:
                self.truncated.append(name)
            if not np.array_equal(idx[:n], SNIPPET * np.arange(n)):
                # a frame that failed to decode would shift every later snippet
                raise RuntimeError(f"xd_violence/{name}: sampled frames are not "
                                   f"0, 16, 32, ... -- delete its frame cache and rerun")
            ivs = self.annotation.get(name, [])
            gt = snippet_gt(n, ivs)
            out.append(FrameSequence(
                frames=paths[:n],
                labels=gt[idx[:n]],
                name=f"xd_violence/{name}",
            ))
        if self.truncated:
            warnings.warn(f"{len(self.truncated)} videos decode fewer frames than their "
                          f"metadata; their tail snippets are dropped (e.g. {self.truncated[:2]})")
        return out
