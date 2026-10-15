# -*- coding: utf-8 -*-
"""Download ONLY the NWPU Campus test split from the authors' Google Drive.

The authors ship all 547 videos (82 GB) as one 7z archive split into 106 volumes
of 700 MiB. The archive is not solid: every video is its own LZMA2 block. So,
as for UCF-Crime, the 242 test videos (21.1 GB packed, 22.4 GB on disk) can be
read straight out of the remote volumes with HTTP range requests, without
downloading the 305 training videos.

The byte offset, packed size, unpacked size and CRC-32 of every test video were
read from the archive's own 7z header on 16 Oct 2026 and are pinned in
data_manifests/nwpu_campus_test_members.json. This script needs no 7z tool:
Python's lzma decodes the raw LZMA2 stream and zlib checks the CRC. It stops if
the archive's size or header position has changed.

Resumable: a video already on disk at its full size is skipped.

Terms: the dataset is for academic research only and may not be redistributed
(campusvad.github.io). Nothing from it is committed to this repository.

    python notebooks/fetch_nwpu_campus_test.py            # -> ~/dazvad/data/nwpu_campus
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import struct
import sys
import time
import zlib

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(HERE, "data_manifests", "nwpu_campus_test_members.json")
URL = "https://drive.usercontent.google.com/download?id=%s&export=download&confirm=t"
NEXT_HEADER_OFFSET = 77484127207      # from the archive's signature header, 16 Oct 2026


def get(session, file_id, a, b, stream=False):
    """Range GET on one Drive file, waiting out rate limits and quota errors."""
    for attempt in range(8):
        r = session.get(URL % file_id, headers={"Range": "bytes=%d-%d" % (a, b)},
                        timeout=120, stream=stream)
        if r.status_code == 206:
            return r
        r.close()                                # never let a full-file reply stream in
        if r.status_code == 200:
            # Drive occasionally ignores the Range header (seen 16 Oct 2026); ask again
            time.sleep(5)
            continue
        if r.status_code in (403, 429, 503):
            # 403 here is Drive's "download quota exceeded" for a busy file
            wait = min(60 * 2 ** attempt, 1800)
            print("  Google Drive answered HTTP %d; waiting %d s" % (r.status_code, wait), flush=True)
            time.sleep(wait)
            continue
        r.raise_for_status()
        raise RuntimeError("expected a range response, got HTTP %d" % r.status_code)
    raise RuntimeError("Drive kept refusing -- rerun later; finished videos are kept")


def check_archive(session, m):
    ids = m["volume_ids"]
    sig = get(session, ids[0], 0, 31).content
    if sig[:6] != b"7z\xbc\xaf\x27\x1c" or struct.unpack("<Q", sig[12:20])[0] != NEXT_HEADER_OFFSET:
        raise SystemExit("the archive's 7z header moved -- the manifest no longer applies")
    last = get(session, ids[-1], 0, 0).headers["Content-Range"]
    total = m["volume_bytes"] * (m["n_volumes"] - 1) + int(last.split("/")[1])
    if total != m["archive_bytes"]:
        raise SystemExit("the archive's size changed -- the manifest no longer applies")


def fetch_member(session, m, row, dst):
    """Stream one LZMA2 block across volume boundaries, decode, and check its CRC."""
    vol = m["volume_bytes"]
    dec = lzma.LZMADecompressor(lzma.FORMAT_RAW, filters=[
        {"id": lzma.FILTER_LZMA2, "dict_size": m["lzma2_dict_size"]}])
    crc, size = 0, 0
    pos, end = row["offset"], row["offset"] + row["packed"]
    tmp = dst + ".part"
    with open(tmp, "wb") as out:
        while pos < end:
            v, o = divmod(pos, vol)
            n = min(end - pos, vol - o)
            r = get(session, m["volume_ids"][v], o, o + n - 1, stream=True)
            for chunk in r.iter_content(1 << 20):
                data = dec.decompress(chunk)
                crc = zlib.crc32(data, crc)
                size += len(data)
                out.write(data)
            pos += n
    if size != row["size"] or (crc & 0xFFFFFFFF) != row["crc32"]:
        os.remove(tmp)
        raise RuntimeError("%s: size %d / CRC %08x do not match the archive header"
                           % (row["name"], size, crc & 0xFFFFFFFF))
    os.replace(tmp, dst)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="~/dazvad/data/nwpu_campus")
    p.add_argument("--only", nargs="*", help="fetch just these videos (testing)")
    a = p.parse_args()
    import requests

    out = os.path.expanduser(a.out)
    vids = os.path.join(out, "videos")
    os.makedirs(vids, exist_ok=True)
    m = json.load(open(MANIFEST, encoding="utf-8"))
    s = requests.Session()
    check_archive(s, m)

    gt = os.path.join(out, "NWPU_Campus_gt.npz")
    if not os.path.exists(gt):
        r = s.get(URL % m["gt_file_id"], timeout=120)
        r.raise_for_status()
        open(gt, "wb").write(r.content)
    if hashlib.sha256(open(gt, "rb").read()).hexdigest() != m["gt_sha256"]:
        raise SystemExit("NWPU_Campus_gt.npz differs from the one the manifest was built "
                         "against -- stop and check")
    import numpy as np
    keys = set(np.load(gt).files)
    names = {r["name"][:-4] for r in m["members"]}
    if keys != names:
        raise SystemExit("ground truth lists %d videos, the manifest %d -- mismatch"
                         % (len(keys), len(names)))

    rows = [r for r in m["members"] if not a.only or r["name"] in a.only]
    total = sum(r["packed"] for r in rows)
    print("ground truth ok (%d test videos); to fetch: %.2f GB packed across %d videos\n"
          % (len(keys), total / 1e9, len(rows)))
    t0, done = time.time(), 0
    for k, row in enumerate(rows, 1):
        dst = os.path.join(vids, row["name"])
        if not (os.path.exists(dst) and os.path.getsize(dst) == row["size"]):
            for attempt in range(4):
                try:
                    fetch_member(s, m, row, dst)
                    break
                except Exception as e:           # noqa: BLE001 -- network errors
                    print("  retry %d on %s: %s" % (attempt + 1, row["name"], e), flush=True)
                    time.sleep(15)
                    s = requests.Session()
            else:
                print("FAILED %s -- rerun the script to resume" % row["name"])
                return 1
        done += row["packed"]
        print("[%3d/%d] %-14s %6.2f/%.2f GB  %.1f min"
              % (k, len(rows), row["name"], done / 1e9, total / 1e9, (time.time() - t0) / 60),
              flush=True)
    n = len([f for f in os.listdir(vids) if f.endswith(".avi")])
    print("\ndone: %d videos in %s" % (n, vids))
    return 0 if a.only or n == len(m["members"]) else 1


if __name__ == "__main__":
    sys.exit(main())
