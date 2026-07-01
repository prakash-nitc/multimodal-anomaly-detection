"""One-command experiment grid — the batched (overnight) results job.

Examples
--------
# smoke test, no GPU/data needed:
python scripts/run_grid.py --synthetic

# the real Sem-3 grid on a GPU machine (any subset of datasets works):
python scripts/run_grid.py \
    --shanghaitech /path/to/shanghaitech --frame-step 2 \
    --avenue /path/to/avenue \
    --mvtec /path/to/mvtec --categories bottle,cable,screw

Raw CLIP scores are cached under results/raw/ -- rerunning the same command
resumes instead of rescoring (safe for Kaggle session limits).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from da_zvad.config import DAZVADConfig
from da_zvad.grid import DatasetSpec, run_grid


def main():
    p = argparse.ArgumentParser(description="Run the DA-ZVAD experiment grid.")
    p.add_argument("--synthetic", action="store_true", help="include the synthetic smoke dataset")
    p.add_argument("--shanghaitech", metavar="ROOT", default=None)
    p.add_argument("--avenue", metavar="ROOT", default=None)
    p.add_argument("--mvtec", metavar="ROOT", default=None)
    p.add_argument("--categories", default="bottle",
                   help="comma-separated MVTec categories (with --mvtec)")
    p.add_argument("--windows", default="1,5,9,15")
    p.add_argument("--frame-step", type=int, default=1)
    p.add_argument("--clip-model", default="ViT-L-14")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    specs = []
    if args.synthetic:
        specs.append(DatasetSpec("synthetic", domain="surveillance",
                                 description="a quiet campus walkway"))
    if args.shanghaitech:
        specs.append(DatasetSpec("shanghaitech", args.shanghaitech, domain="surveillance",
                                 description="a university campus walkway with pedestrians"))
    if args.avenue:
        specs.append(DatasetSpec("avenue", args.avenue, domain="surveillance",
                                 description="a subway station entrance with commuters"))
    if args.mvtec:
        for cat in args.categories.split(","):
            specs.append(DatasetSpec("mvtec", args.mvtec, category=cat.strip(),
                                     domain="industrial",
                                     description=f"an industrial quality-inspection image of a {cat.strip()}"))
    if not specs:
        p.error("no datasets selected — pass --synthetic and/or dataset roots")

    base = DAZVADConfig(clip_model=args.clip_model, frame_step=args.frame_step)
    windows = [int(w) for w in args.windows.split(",")]
    rows = run_grid(specs, windows=windows, base=base, out_dir=args.out_dir)

    print("\n=== GRID SUMMARY (auroc_micro) ===")
    print(f"{'dataset':<24}{'context':<9}" + "".join(f"w={w:<6}" for w in windows))
    keyed = {(r["dataset"], r["context"]): {} for r in rows}
    for r in rows:
        keyed[(r["dataset"], r["context"])][r["window"]] = r["auroc_micro"]
    for (ds, ctx), by_w in keyed.items():
        print(f"{ds:<24}{str(ctx):<9}" + "".join(f"{by_w[w]:<8.3f}" for w in windows))


if __name__ == "__main__":
    main()
