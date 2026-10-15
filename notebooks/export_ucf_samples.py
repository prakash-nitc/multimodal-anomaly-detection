# -*- coding: utf-8 -*-
"""Export sample UCF-Crime frames for figures and slides.

For each chosen class, takes the first test video of that class whose anomaly
starts late enough to leave ordinary footage before it, and copies two frames
from the step-16 frame cache the adapter already wrote:
  * a NORMAL frame, halfway between the start of the video and the event,
  * an ANOMALY frame, at the middle of the first labelled anomalous interval.
Same video, same camera -- only the event differs.

Graphic classes (Shooting, Abuse, Assault) are left out by default; the set is
meant for slides shown to a general audience.

    python notebooks/export_ucf_samples.py        # -> ~/dazvad/work/ucf_samples/
then, on the laptop:
    scp -r m251250cs@192.168.41.119:~/dazvad/work/ucf_samples .
"""
from __future__ import annotations

import csv
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from da_zvad.datasets.ucf_crime import parse_annotation  # noqa: E402

ROOT = os.path.expanduser("~/dazvad/data/ucf_crime")
CACHE = os.path.join(ROOT, "_frames_step16")
OUT = os.path.expanduser("~/dazvad/work/ucf_samples")
STEP = 16
CLASSES = ["RoadAccidents", "Robbery", "Burglary", "Arson",
           "Explosion", "Shoplifting", "Vandalism"]
MIN_ONSET = 10 * STEP          # at least ~5 s of ordinary footage before the event


def nearest_cached(stem: str, frame: int) -> str | None:
    """Path of the cached sampled frame at or just before decoded index `frame`."""
    idx = (frame // STEP) * STEP
    p = os.path.join(CACHE, stem, "%07d.jpg" % idx)
    return p if os.path.isfile(p) else None


def main() -> int:
    ann = parse_annotation(os.path.join(ROOT, "Temporal_Anomaly_Annotation.txt"))
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for cls in CLASSES:
        for name in sorted(ann):
            c, ivs = ann[name]
            if c != cls or not ivs or ivs[0][0] < MIN_ONSET:
                continue
            stem = os.path.splitext(name)[0]
            s, e = ivs[0]                              # 1-based inclusive
            normal = nearest_cached(stem, (s - 1) // 2)
            anomaly = nearest_cached(stem, (s - 1 + e - 1) // 2)
            if not (normal and anomaly):
                continue
            for kind, src in (("normal", normal), ("anomaly", anomaly)):
                dst = os.path.join(OUT, "%s_%s_%s" % (cls, kind, os.path.basename(src)))
                shutil.copyfile(src, dst)
            rows.append([cls, stem, s, e,
                         os.path.basename(normal), os.path.basename(anomaly)])
            print("  %-14s %-22s event frames %d-%d" % (cls, stem, s, e))
            break
        else:
            print("  %-14s no suitable video" % cls)

    with open(os.path.join(OUT, "samples.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "video", "event_start", "event_end",
                    "normal_frame", "anomaly_frame"])
        w.writerows(rows)
    print("\nwrote %d pairs to %s" % (len(rows), OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
