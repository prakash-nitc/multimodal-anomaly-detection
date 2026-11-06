"""UBnormal adapter, official test split only (Acsintoae et al., CVPR 2022).

Reserved for the single sealed test pre-registered in docs/10_phase3/PREREG_sealed_ubnormal.md.

    <root>/Scene<N>/<video>.mp4
    <root>/Scene<N>/<video>_annotations/<video>_tracks.txt     (abnormal videos only)
    <root>/abnormal_test_video_names.txt                       (official split lists,
    <root>/normal_test_video_names.txt                          from the authors' repo)

Ground truth -- the authors' own convention
-------------------------------------------
Ported from ``scripts/write_frame_level_ground_truth.py`` in github.com/lilygeorgescu/UBnormal:
the frame count is OpenCV's ``CAP_PROP_FRAME_COUNT``; each line of a tracks file is
``object_id,start_frame,end_frame``; every track marks ``gt[start:end + 1] = 1`` (end
INCLUSIVE); a normal video is all zeros; the scene folder is the third ``_`` field of the
video name. Nothing here reads a label except to build that array.

Sampling
--------
Decoded frames 0, 16, 32, ... are scored, as on UCF-Crime, XD-Violence and NWPU Campus.
Labels on those frames are returned for the sampled-frame metric; the sealed test also
evaluates every frame, holding each sampled score until the next one (``hold_scores``).

Requires cv2, imported lazily.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import AnomalyDataset, FrameSequence

STEP = 16
_IMG_EXT = ".jpg"
LISTS = ("abnormal_test_video_names.txt", "normal_test_video_names.txt")


def scene_of(video: str) -> str:
    """'abnormal_scene_7_scenario_2_fog' -> '7' (the authors' video_name.split('_')[2])."""
    return video.split("_")[2]


def read_tracks(text: str) -> np.ndarray:
    """Rows of (object_id, start, end) from a tracks file's text; one row or many."""
    rows = [r for r in (ln.strip() for ln in text.splitlines()) if r]
    arr = np.array([[float(x) for x in r.split(",")] for r in rows])
    return arr.reshape(-1, 3) if arr.size else np.zeros((0, 3))


def frame_level_gt(n_frames: int, tracks: np.ndarray) -> np.ndarray:
    """Per-frame labels exactly as the authors build them: gt[start:end + 1] = 1."""
    gt = np.zeros(n_frames, dtype=int)
    for _obj, s, e in tracks:
        gt[int(s): int(e + 1)] = 1
    return gt


def hold_scores(sampled: np.ndarray, n_frames: int, step: int = STEP) -> np.ndarray:
    """Every frame gets the score of the last sampled frame at or before it."""
    s = np.asarray(sampled, dtype=float)
    full = np.repeat(s, step)[:n_frames]
    if len(full) < n_frames:
        full = np.r_[full, np.full(n_frames - len(full), s[-1] if len(s) else 0.0)]
    return full


def read_lists(root: str) -> Tuple[List[str], List[str]]:
    """(abnormal test videos, normal test videos), each sorted as the authors do."""
    out = []
    for name in LISTS:
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path!r} missing -- fetch the official split lists first.")
        out.append(sorted(ln.strip() for ln in open(path, encoding="utf-8") if ln.strip()))
    return out[0], out[1]


class UBnormalDataset(AnomalyDataset):
    def __init__(self, root: Optional[str], frame_step: int = STEP, cache_dir: Optional[str] = None):
        if not root:
            raise ValueError("UBnormalDataset requires data_root.")
        if int(frame_step) != STEP:
            raise ValueError("The sealed test fixes sampling at every 16th frame.")
        self.root = os.path.expanduser(root)
        self.cache_dir = cache_dir or os.path.join(self.root, f"_frames_step{STEP}")
        self.abnormal, self.normal = read_lists(self.root)

    def video_path(self, video: str) -> str:
        return os.path.join(self.root, f"Scene{scene_of(video)}", video + ".mp4")

    def tracks_path(self, video: str) -> str:
        return os.path.join(self.root, f"Scene{scene_of(video)}", video + "_annotations",
                            video + "_tracks.txt")

    def check_files(self) -> Dict[str, int]:
        """Counts only: every test video, and every abnormal video's tracks file, must exist."""
        missing_v = [v for v in self.abnormal + self.normal if not os.path.isfile(self.video_path(v))]
        missing_t = [v for v in self.abnormal if not os.path.isfile(self.tracks_path(v))]
        if missing_v or missing_t:
            raise FileNotFoundError(f"{len(missing_v)} test videos and {len(missing_t)} tracks files "
                                    f"missing (e.g. {(missing_v + missing_t)[:3]})")
        return {"abnormal": len(self.abnormal), "normal": len(self.normal)}

    def _extract(self, video: str) -> Tuple[List[str], int, int, float]:
        """-> (sampled frame paths, decoded frame count, metadata frame count, fps)."""
        import cv2  # lazy

        out_dir = os.path.join(self.cache_dir, video)
        sentinel = os.path.join(out_dir, ".complete")
        if os.path.isfile(sentinel):
            decoded, meta, fps = open(sentinel).read().split()
            paths = sorted(os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(_IMG_EXT))
            return paths, int(decoded), int(meta), float(fps)
        os.makedirs(out_dir, exist_ok=True)
        cap = cv2.VideoCapture(self.video_path(video))
        meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        paths, i = [], 0
        while True:
            if not cap.grab():
                break
            if i % STEP == 0:
                ok, frame = cap.retrieve()
                if ok:
                    p = os.path.join(out_dir, f"{i:07d}{_IMG_EXT}")
                    cv2.imwrite(p, frame)
                    paths.append(p)
            i += 1
        cap.release()
        if i == 0:
            raise RuntimeError(f"Decoded 0 frames from {video!r} -- codec unavailable?")
        with open(sentinel, "w") as fh:
            fh.write(f"{i} {meta} {fps}")
        return paths, i, meta, fps

    def frame_count(self, video: str) -> int:
        """The authors' frame count (CAP_PROP_FRAME_COUNT), read from the decode record."""
        return self._extract(video)[2]

    def full_gt(self, video: str) -> np.ndarray:
        n = self.frame_count(video)
        if video in set(self.abnormal):
            return frame_level_gt(n, read_tracks(open(self.tracks_path(video)).read()))
        return np.zeros(n, dtype=int)

    def sequences(self) -> List[FrameSequence]:
        self.check_files()
        out: List[FrameSequence] = []
        for video in sorted(self.abnormal + self.normal):
            paths, _decoded, meta, _fps = self._extract(video)
            idx = np.array([int(os.path.splitext(os.path.basename(p))[0]) for p in paths])
            keep = idx < meta
            if not np.array_equal(idx[keep], STEP * np.arange(int(keep.sum()))):
                raise RuntimeError(f"ubnormal/{video}: sampled frames are not 0, 16, 32, ... -- "
                                   "delete its frame cache and rerun")
            gt = self.full_gt(video)
            out.append(FrameSequence(frames=[p for p, k in zip(paths, keep) if k],
                                     labels=gt[idx[keep]], name=f"ubnormal/{video}"))
        return out
