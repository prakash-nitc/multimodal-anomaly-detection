"""CLI entry point for DA-ZVAD experiments.

Examples
--------
# synthetic smoke test (no GPU/data):
python scripts/run_da_zvad.py --dataset synthetic

# MVTec category on the GPU:
python scripts/run_da_zvad.py --dataset mvtec --data-root /path/to/mvtec --category bottle \
       --domain industrial

# surveillance video:
python scripts/run_da_zvad.py --dataset shanghaitech --data-root /path/to/clips \
       --domain surveillance --domain-description "a campus walkway"
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from da_zvad.config import DAZVADConfig
from da_zvad.runner import run_experiment


def main():
    p = argparse.ArgumentParser(description="Run a DA-ZVAD experiment.")
    p.add_argument("--dataset", default="synthetic",
                   choices=["synthetic", "mvtec", "shanghaitech", "avenue"])
    p.add_argument("--data-root", default=None)
    p.add_argument("--category", default=None, help="MVTec category")
    p.add_argument("--domain", default="generic",
                   choices=["generic", "industrial", "surveillance"])
    p.add_argument("--domain-description", default="a generic scene")
    p.add_argument("--no-temporal", action="store_true")
    p.add_argument("--no-context", action="store_true")
    p.add_argument("--reasoning", action="store_true", help="enable M4 (stub for now)")
    p.add_argument("--temporal-window", type=int, default=5)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--sample-fps", type=int, default=2)
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    cfg = DAZVADConfig(
        dataset=args.dataset, data_root=args.data_root, category=args.category,
        domain=args.domain, domain_description=args.domain_description,
        use_temporal=not args.no_temporal, use_context=not args.no_context,
        use_reasoning=args.reasoning, temporal_window=args.temporal_window,
        threshold=args.threshold, sample_fps=args.sample_fps,
    )
    print(f"[da-zvad] config: {cfg.tag()}")
    row = run_experiment(cfg, out_dir=args.out_dir)
    print("[da-zvad] result:", json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
