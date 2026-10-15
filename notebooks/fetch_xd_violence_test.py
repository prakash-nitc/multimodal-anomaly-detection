# -*- coding: utf-8 -*-
"""Download the XD-Violence test split from the authors' official links.

Test videos: one 10.8 GB ``videos.zip`` (800 videos -- 500 violent, 300 normal)
on the authors' OneDrive, linked from https://roc-ng.github.io/XD-Violence/.
Annotations: ``annotations.txt`` on the same site.

OneDrive refuses a bare download request; it first wants the session cookie
that opening the share page sets. With that cookie the file supports HTTP range
requests, so -- as for UCF-Crime -- each video is read straight out of the
remote zip, CRC-checked, and written to disk on its own. An interrupted run is
simply rerun: a video already on disk at its full size is skipped, and an
expired session is renewed automatically.

The member list (data_manifests/xd_violence_test_members.txt) was read from the
remote zip's index on 16 Oct 2026; the script stops if the archive no longer
matches it, or if the annotation file's checksum has changed.

SharePoint rate-limits bursts of requests (HTTP 429; seen on 16 Oct 2026 after a
few dozen probes). The script waits these out, but if it keeps being throttled,
``--whole`` downloads the zip in a single resumable request instead and extracts
it locally -- about 11 GB of extra disk while it runs.

    pip install remotezip
    python notebooks/fetch_xd_violence_test.py            # -> ~/dazvad/data/xd_violence
    python notebooks/fetch_xd_violence_test.py --whole    # if HTTP 429 persists
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time

SHARE = ("https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/pengwu_stu_xidian_edu_cn/"
         "EdewYQSko-BHpew03RBfGLQBIAOUdF7jt1rC71rK5iBGag?e=v4Enay")
ANNOTATION = "https://roc-ng.github.io/XD-Violence/images/annotations.txt"
ANNOTATION_SHA256 = "27b583ba06fe2e096d7e3d281845365c3b24de7e30797ae5c1ec771667e04c2f"
ZIP_BYTES = 10804733011
N_ANNOTATED = 500                     # violent test videos; normal ones are not listed
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMBERS = os.path.join(HERE, "data_manifests", "xd_violence_test_members.txt")


def fetch_annotation(out_dir: str, session) -> str:
    dst = os.path.join(out_dir, "annotations.txt")
    if not os.path.exists(dst):
        r = polite_get(session, ANNOTATION)
        r.raise_for_status()
        open(dst, "wb").write(r.content)
    digest = hashlib.sha256(open(dst, "rb").read()).hexdigest()
    if digest != ANNOTATION_SHA256:
        raise SystemExit("annotations.txt has changed upstream (sha256 %s) -- the "
                         "pre-registered protocol names the old file; stop and check" % digest)
    return dst


def polite_get(s, url, **kw):
    """GET that waits out SharePoint's rate limit (HTTP 429/503) instead of failing."""
    for attempt in range(8):
        r = s.get(url, timeout=120, **kw)
        if r.status_code not in (429, 503):
            return r
        wait = int(r.headers.get("Retry-After") or 0) or min(30 * 2 ** attempt, 600)
        print("  SharePoint is rate-limiting (HTTP %d); waiting %d s" % (r.status_code, wait),
              flush=True)
        time.sleep(wait)
    r.raise_for_status()
    return r


def new_session():
    """A requests session holding the OneDrive cookie that opening the share page sets."""
    import requests
    s = requests.Session()
    polite_get(s, SHARE).raise_for_status()
    return s


def open_remote():
    """-> (session, RemoteZip) with a fresh OneDrive session cookie."""
    from remotezip import RemoteZip
    s = new_session()
    r = polite_get(s, SHARE + "&download=1", headers={"Range": "bytes=0-0"})
    if r.status_code != 206:
        raise RuntimeError("OneDrive did not grant a range request (HTTP %d)" % r.status_code)
    total = int(r.headers["Content-Range"].split("/")[1])
    if total != ZIP_BYTES:
        raise SystemExit("videos.zip is %d bytes, expected %d -- the archive changed"
                         % (total, ZIP_BYTES))
    return s, RemoteZip(r.url, session=s, support_suffix_range=False)


def download_whole(out: str, session) -> str:
    """videos.zip in one resumable stream: one long request instead of ~800 short ones.

    The fallback for when SharePoint keeps rate-limiting the per-video requests.
    An interrupted transfer resumes from the bytes already on disk.
    """
    path = os.path.join(out, "videos.zip")
    have = os.path.getsize(path) if os.path.exists(path) else 0
    t0, last, failures = time.time(), have, 0
    while have < ZIP_BYTES:
        try:
            r = polite_get(session, SHARE + "&download=1",
                           headers={"Range": "bytes=%d-" % have}, stream=True)
            if r.status_code != 206:
                raise RuntimeError("no range response (HTTP %d)" % r.status_code)
            if int(r.headers["Content-Range"].split("/")[1]) != ZIP_BYTES:
                raise SystemExit("videos.zip changed size upstream -- the archive changed")
            with open(path, "ab") as fh:
                for chunk in r.iter_content(1 << 22):
                    fh.write(chunk)
                    have += len(chunk)
                    if have - last >= 5e8 or have >= ZIP_BYTES:
                        print("  %.2f / %.2f GB  %.1f min"
                              % (have / 1e9, ZIP_BYTES / 1e9, (time.time() - t0) / 60), flush=True)
                        last = have
            failures = 0
        except SystemExit:
            raise
        except Exception as e:                   # noqa: BLE001 -- dropped connection, expired cookie
            failures += 1
            if failures > 10:
                raise SystemExit("download keeps failing (%s) -- rerun later to resume" % e)
            print("  interrupted at %.2f GB (%s); resuming" % (have / 1e9, e), flush=True)
            time.sleep(10)
            session = new_session()
            have = os.path.getsize(path)
    if have != ZIP_BYTES:
        raise SystemExit("videos.zip is %d bytes, expected %d -- delete it and rerun"
                         % (have, ZIP_BYTES))
    return path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="~/dazvad/data/xd_violence")
    p.add_argument("--whole", action="store_true",
                   help="download videos.zip in one resumable stream, then extract "
                        "locally. Use if SharePoint keeps answering 429. Needs ~11 GB of "
                        "extra disk while it runs; the zip is deleted afterwards.")
    p.add_argument("--keep-zip", action="store_true", help="with --whole: keep videos.zip")
    a = p.parse_args()
    out = os.path.expanduser(a.out)
    vids = os.path.join(out, "videos")
    os.makedirs(vids, exist_ok=True)

    if a.whole:
        import zipfile
        session = new_session()
        ann = fetch_annotation(out, session)
        zpath = download_whole(out, session)
        z = zipfile.ZipFile(zpath)
    else:
        try:
            import remotezip  # noqa: F401
        except ImportError:
            print("pip install remotezip  -- then rerun")
            return 2
        session, z = open_remote()
        ann = fetch_annotation(out, session)
    wanted = [l.strip() for l in open(MEMBERS, encoding="utf-8") if l.strip()]
    index = {i.filename: i for i in z.infolist()}
    if sorted(n for n in index if n.endswith(".mp4")) != sorted(wanted):
        print("the archive's video list no longer matches %s -- stopping" % MEMBERS)
        return 1
    names = {os.path.basename(m)[:-4] for m in wanted}
    annotated = {(l.split()[0][:-4] if l.split()[0].endswith(".mp4") else l.split()[0])
                 for l in open(ann, encoding="utf-8") if l.strip()}
    if not annotated <= names or len(annotated) != N_ANNOTATED:
        print("annotation does not match the video list -- stopping")
        return 1
    total = sum(index[m].file_size for m in wanted)
    print("annotation: %d violent videos annotated -> %s" % (len(annotated), ann))
    print("to fetch: %.2f GB across %d videos\n" % (total / 1e9, len(wanted)))

    t0, done_bytes = time.time(), 0
    for k, m in enumerate(wanted, 1):
        info = index[m]
        dst = os.path.join(vids, os.path.basename(m))
        if os.path.exists(dst) and os.path.getsize(dst) == info.file_size:
            done_bytes += info.file_size
            continue
        for attempt in range(4):
            try:
                tmp = dst + ".part"
                with z.open(info) as src, open(tmp, "wb") as fh:
                    while True:
                        chunk = src.read(1 << 22)
                        if not chunk:
                            break
                        fh.write(chunk)
                os.replace(tmp, dst)              # zipfile checks the CRC-32 on read
                break
            except Exception as e:               # noqa: BLE001 -- network errors, expired session
                print("  retry %d on %s: %s" % (attempt + 1, os.path.basename(m), e))
                if a.whole:                       # a local CRC error is not transient
                    print("  the downloaded videos.zip is damaged -- delete it and rerun")
                    return 1
                time.sleep(10)
                try:
                    session, z = open_remote()    # renew the cookie and the index
                    index = {i.filename: i for i in z.infolist()}
                    info = index[m]
                except Exception as e2:          # noqa: BLE001
                    print("  could not reopen the archive: %s" % e2)
        else:
            print("FAILED %s -- rerun the script to resume" % m)
            return 1
        done_bytes += info.file_size
        el = time.time() - t0
        print("[%3d/%d] %-60s %6.2f/%.2f GB  %.1f min"
              % (k, len(wanted), os.path.basename(m)[:60], done_bytes / 1e9, total / 1e9, el / 60),
              flush=True)

    n = len([f for f in os.listdir(vids) if f.endswith(".mp4")])
    print("\ndone: %d videos in %s" % (n, vids))
    if a.whole and n == len(wanted) and not a.keep_zip:
        z.close()
        os.remove(zpath)
        print("removed %s (every video extracted and CRC-checked)" % zpath)
    return 0 if n == len(wanted) else 1


if __name__ == "__main__":
    sys.exit(main())
