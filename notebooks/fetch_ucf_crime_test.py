# -*- coding: utf-8 -*-
"""Download ONLY the UCF-Crime test split from the official archive.

The official UCF_Crimes.zip is 103 GB, almost all of it the 1,610 training
videos. A training-free method needs none of those. The 290 test videos -- 140
anomalous, 150 normal -- total about 8 GB, and the archive server supports HTTP
range requests, so each member can be read straight out of the remote zip
without downloading the rest.

The member list (data_manifests/ucf_crime_test_members.txt) was built by
matching the official test annotation against the archive's index. Fifty normal
test videos appear twice in the archive (Testing_Normal_Videos_Anomaly and
z_Normal_Videos_event) with identical sizes; the Testing_Normal copy is used.

Resumable: a video already on disk at its full size is skipped, so an
interrupted run is simply rerun.

    pip install remotezip
    python notebooks/fetch_ucf_crime_test.py            # -> ~/dazvad/data/ucf_crime
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
import urllib.request
import zipfile
import ssl

ARCHIVE = "https://www.crcv.ucf.edu/data1/chenchen/UCF_Crimes.zip"
ANNOTATION = ("https://www.crcv.ucf.edu/projects/real-world/"
              "Temporal_Anomaly_Annotation_For_Testing_Videos.zip")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMBERS = os.path.join(HERE, "data_manifests", "ucf_crime_test_members.txt")

# The CRCV server's certificate chain is incomplete; Python cannot verify it.
# The files are public and are checked against the archive's own CRC-32 on
# extraction, which is what actually guards integrity here.
_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE


def fetch_annotation(out_dir: str) -> str:
    dst = os.path.join(out_dir, "Temporal_Anomaly_Annotation.txt")
    if os.path.exists(dst):
        return dst
    raw = urllib.request.urlopen(ANNOTATION, context=_CTX, timeout=120).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = next(n for n in z.namelist() if n.endswith("Temporal_Anomaly_Annotation.txt"))
        open(dst, "wb").write(z.read(name))
    return dst


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="~/dazvad/data/ucf_crime")
    a = p.parse_args()
    out = os.path.expanduser(a.out)
    vids = os.path.join(out, "videos")
    os.makedirs(vids, exist_ok=True)

    try:
        from remotezip import RemoteZip
    except ImportError:
        print("pip install remotezip  -- then rerun")
        return 2
    import urllib3
    urllib3.disable_warnings()

    ann = fetch_annotation(out)
    wanted = [l.strip() for l in open(MEMBERS, encoding="utf-8") if l.strip()]
    ann_names = {l.split()[0] for l in open(ann) if l.strip()}
    got = {os.path.basename(m) for m in wanted}
    if got != ann_names:
        print("member list does not match the annotation (%d vs %d) -- stopping"
              % (len(got), len(ann_names)))
        return 1
    print("annotation: %d test videos -> %s" % (len(ann_names), ann))

    t0, done_bytes = time.time(), 0
    with RemoteZip(ARCHIVE, verify=False) as z:
        index = {i.filename: i for i in z.infolist()}
        total = sum(index[m].file_size for m in wanted)
        print("to fetch: %.2f GB across %d videos\n" % (total / 1e9, len(wanted)))
        for k, m in enumerate(wanted, 1):
            info = index[m]
            dst = os.path.join(vids, os.path.basename(m))
            if os.path.exists(dst) and os.path.getsize(dst) == info.file_size:
                done_bytes += info.file_size
                continue
            for attempt in range(3):
                try:
                    tmp = dst + ".part"
                    with z.open(info) as src, open(tmp, "wb") as fh:
                        while True:
                            chunk = src.read(1 << 22)
                            if not chunk:
                                break
                            fh.write(chunk)
                    os.replace(tmp, dst)          # CRC checked by zipfile on read
                    break
                except Exception as e:           # noqa: BLE001 -- retry transient network errors
                    print("  retry %d on %s: %s" % (attempt + 1, os.path.basename(m), e))
                    time.sleep(5)
            else:
                print("FAILED %s -- rerun the script to resume" % m)
                return 1
            done_bytes += info.file_size
            el = time.time() - t0
            print("[%3d/%d] %-34s %6.2f/%.2f GB  %.1f min"
                  % (k, len(wanted), os.path.basename(m), done_bytes / 1e9, total / 1e9, el / 60),
                  flush=True)

    n = len([f for f in os.listdir(vids) if f.endswith(".mp4")])
    print("\ndone: %d videos in %s" % (n, vids))
    return 0 if n == len(wanted) else 1


if __name__ == "__main__":
    sys.exit(main())
