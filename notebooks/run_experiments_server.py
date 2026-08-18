# -*- coding: utf-8 -*-
"""DA-ZVAD experiment driver for a shared GPU server.

Replaces ``full_experiments_kaggle.py`` for runs on institutional hardware.
Three things differ from the Kaggle version, and each exists for a reason:

1. **Explicit dataset paths.** Kaggle mounts datasets at a predictable
   ``/kaggle/input/*`` so that script could guess. Here the caller says where
   the data is, and a wrong path fails immediately instead of silently
   producing an empty grid.

2. **Timestamped run folders with a manifest.** Every run writes its own
   ``MANIFEST.txt`` recording the git commit, GPU, library versions, config and
   dataset statistics. A results table whose provenance cannot be reconstructed
   is not evidence, and "which code produced this number?" is the first
   question anyone should ask of it.

3. **Courtesy limits.** This machine is shared with other students. Thread
   count and GPU memory are capped so a long run cannot starve someone else's
   job. The caps are far above what the pipeline actually needs, so they cost
   nothing.

The expensive CLIP scoring stays cached in a persistent work directory
(``--work``), NOT in the per-run folder -- so a second run reuses the cache and
finishes in seconds, while still getting its own manifest and tables.

Usage
-----
    python notebooks/run_experiments_server.py \
        --shanghaitech ~/dazvad/data/shanghaitech \
        --runs ~/dazvad/runs --work ~/dazvad/work
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

# Thread caps must be set BEFORE torch/cv2 import -- both read these at load
# time and will otherwise spawn one worker per core on a shared machine.
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", REPO, *args],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unavailable"


def build_manifest(args, specs, gpu_info: dict) -> str:
    """Everything needed to reproduce this run, in one plain-text block."""
    import torch

    dirty = _git("status", "--porcelain")
    lines = [
        "DA-ZVAD experiment run",
        "=" * 60,
        f"run id          : {args.run_id}",
        f"started (UTC)   : {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "-- provenance " + "-" * 46,
        f"git commit      : {_git('rev-parse', 'HEAD')}",
        f"git branch      : {_git('rev-parse', '--abbrev-ref', 'HEAD')}",
        f"working tree    : {'DIRTY (uncommitted changes!)' if dirty else 'clean'}",
        f"repo path       : {REPO}",
        "",
        "-- environment " + "-" * 46,
        f"host            : {socket.gethostname()}",
        f"user            : {os.environ.get('USER', 'unknown')}",
        f"os              : {platform.platform()}",
        f"python          : {platform.python_version()}",
        f"torch           : {torch.__version__}",
        f"cuda (torch)    : {torch.version.cuda}",
        f"gpu             : {gpu_info.get('name', 'CPU ONLY')}",
        f"gpu memory      : {gpu_info.get('total_gb', 'n/a')} GB",
        f"memory cap      : {int(args.gpu_frac * 100)}% of device",
        f"thread cap      : {os.environ['OMP_NUM_THREADS']}",
        "",
        "-- configuration " + "-" * 43,
        f"clip model      : {args.clip_model} / {args.clip_pretrained}",
        f"prompt domain   : {args.domain}",
        f"context mode    : {args.context_mode}",
        f"frame_step      : {args.frame_step}  (score every Nth frame)",
        f"grid windows    : {args.windows}",
        f"sweep windows   : {args.sweep_windows}",
        "",
        "-- datasets " + "-" * 48,
    ]
    for spec, stats in specs:
        lines += [
            f"{spec.label()}",
            f"    root        : {spec.data_root}",
            f"    domain      : {spec.domain}",
            f"    description : \"{spec.description}\"",
            f"    sequences   : {stats['n_seqs']}",
            f"    frames used : {stats['n_frames']}  (after frame_step)",
            f"    anomalous   : {stats['n_anom']} frames "
            f"({100.0 * stats['n_anom'] / max(stats['n_frames'], 1):.1f}%)",
        ]
    return "\n".join(lines) + "\n"


def dataset_stats(spec, frame_step: int) -> dict:
    """Count sequences/frames/positives up front.

    This is the cheap early warning for the failure that matters most: if
    ground-truth masks are missing the loader silently yields all-zero labels,
    every AUROC comes back NaN, and an hour of GPU time is wasted. Better to
    refuse to start.
    """
    import numpy as np
    from da_zvad.config import DAZVADConfig
    from da_zvad.datasets import get_dataset

    cfg = DAZVADConfig(dataset=spec.dataset, data_root=spec.data_root,
                       category=spec.category, frame_step=frame_step)
    seqs = get_dataset(cfg).sequences()
    n_frames = sum(len(s.frames) for s in seqs)
    n_anom = int(sum(int(np.asarray(s.labels).sum()) for s in seqs))
    return {"n_seqs": len(seqs), "n_frames": n_frames, "n_anom": n_anom}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shanghaitech", help="ShanghaiTech root (contains testing/frames)")
    p.add_argument("--avenue", help="CUHK Avenue root (contains testing_videos)")
    p.add_argument("--mvtec", help="MVTec AD root (contains bottle/, carpet/, ...)")
    p.add_argument("--mvtec-categories", nargs="+",
                   default=["bottle", "carpet", "screw", "transistor"])
    p.add_argument("--runs", default=os.path.expanduser("~/dazvad/runs"),
                   help="parent directory for timestamped run folders")
    p.add_argument("--work", default=os.path.expanduser("~/dazvad/work"),
                   help="persistent cache + tables (survives across runs)")
    p.add_argument("--frame-step", type=int, default=2)
    p.add_argument("--windows", type=int, nargs="+", default=[1, 5, 9, 15])
    p.add_argument("--sweep-windows", type=int, nargs="+", default=[1, 5])
    p.add_argument("--clip-model", default="ViT-L-14")
    p.add_argument("--clip-pretrained", default="laion2b_s32b_b82k")
    p.add_argument("--domain", default="campus",
                   choices=["campus", "surveillance", "generic"],
                   help="prompt ensemble for the video datasets. 'surveillance' "
                        "encodes a crime notion of anomaly (UCF-Crime style); "
                        "'campus' encodes ShanghaiTech's documented classes")
    p.add_argument("--context-mode", default="normal", choices=["normal", "both"],
                   help="'both' appends the description to both ensembles and "
                        "dilutes them; 'normal' grounds only the normal side")
    p.add_argument("--gpu-frac", type=float, default=0.40,
                   help="max fraction of GPU memory this process may hold")
    p.add_argument("--tag", default="grid", help="short label for the run folder")
    p.add_argument("--allow-unlabelled", action="store_true",
                   help="proceed even if a dataset has zero anomalous frames")
    args = p.parse_args()

    import numpy as np
    import torch
    from da_zvad.config import DAZVADConfig
    from da_zvad.grid import DatasetSpec, run_grid
    from da_zvad.context_sweep import run_context_sweep

    # ---- GPU -----------------------------------------------------------
    gpu_info = {}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_info = {"name": props.name,
                    "total_gb": round(props.total_memory / 1024 ** 3, 1)}
        # Hard ceiling: this process can never crowd out another student's job.
        torch.cuda.set_per_process_memory_fraction(args.gpu_frac, 0)
        print(f"[gpu] {props.name}  capped at {args.gpu_frac:.0%} of "
              f"{gpu_info['total_gb']} GB")
    else:
        print("[gpu] WARNING: CUDA unavailable -- this will run on CPU and take "
              "days, not minutes. Aborting.", file=sys.stderr)
        return 2

    # ---- datasets ------------------------------------------------------
    specs = []
    if args.shanghaitech:
        specs.append(DatasetSpec(
            "shanghaitech", os.path.expanduser(args.shanghaitech),
            domain=args.domain,
            description="a university campus walkway with pedestrians"))
    if args.avenue:
        specs.append(DatasetSpec(
            "avenue", os.path.expanduser(args.avenue), domain=args.domain,
            description="an outdoor campus walkway in front of a building entrance"))
    if args.mvtec:
        for cat in args.mvtec_categories:
            specs.append(DatasetSpec(
                "mvtec", os.path.expanduser(args.mvtec), category=cat,
                domain="industrial",
                description=f"an industrial quality-inspection image of a {cat}"))
    if not specs:
        p.error("no dataset given -- pass at least --shanghaitech")

    print("\n[data] inspecting datasets before committing GPU time ...")
    checked = []
    for spec in specs:
        stats = dataset_stats(spec, args.frame_step)
        flag = "" if stats["n_anom"] else "   <-- NO ANOMALOUS FRAMES"
        print(f"[data] {spec.label():<22} {stats['n_seqs']:>4} seqs  "
              f"{stats['n_frames']:>7} frames  {stats['n_anom']:>7} anomalous{flag}")
        if not stats["n_anom"] and not args.allow_unlabelled:
            print(f"\n[data] ABORT: {spec.label()} has no positive labels. Ground "
                  f"truth is missing or unreadable, so every AUROC would be NaN.\n"
                  f"       Check the mask folder, or pass --allow-unlabelled to "
                  f"override.", file=sys.stderr)
            return 3
        checked.append((spec, stats))

    # ---- run folder ----------------------------------------------------
    args.run_id = f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{args.tag}"
    run_dir = os.path.join(os.path.expanduser(args.runs), args.run_id)
    work = os.path.expanduser(args.work)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(work, exist_ok=True)

    manifest_path = os.path.join(run_dir, "MANIFEST.txt")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(build_manifest(args, checked, gpu_info))
    print(f"\n[run] {run_dir}")
    print(f"[run] cache/tables in {work} (shared across runs, resumable)")

    base = DAZVADConfig(clip_model=args.clip_model,
                        clip_pretrained=args.clip_pretrained,
                        frame_step=args.frame_step,
                        context_mode=args.context_mode)
    t0 = time.time()

    print("\n" + "=" * 70)
    print("GRID  (context on/off x temporal window)")
    print("=" * 70, flush=True)
    run_grid(specs, windows=args.windows, base=base, out_dir=work)

    print("\n" + "=" * 70)
    print("CONTEXT SWEEP  (none / generic / matched / mismatched)")
    print("=" * 70, flush=True)
    run_context_sweep(specs, windows=args.sweep_windows, base=base, out_dir=work)

    elapsed = time.time() - t0

    # ---- collect into the run folder -----------------------------------
    tables_src = os.path.join(work, "tables")
    if os.path.isdir(tables_src):
        shutil.copytree(tables_src, os.path.join(run_dir, "tables"),
                        dirs_exist_ok=True)
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write("\n-- completion " + "-" * 46 + "\n")
        f.write(f"finished (UTC)  : "
                f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}\n")
        f.write(f"wall clock      : {elapsed / 60:.1f} min\n")
        f.write(f"peak gpu memory : "
                f"{torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB\n")

    print(f"\n[done] {elapsed / 60:.1f} min")
    print(f"[done] tables   -> {os.path.join(run_dir, 'tables')}")
    print(f"[done] manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
