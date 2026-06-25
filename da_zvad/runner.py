"""Configuration-driven experiment runner.

``run_experiment`` executes one config; ``run_grid`` sweeps many. Because each
module is a toggle in the config, ``run_grid`` over the toggle combinations *is*
the ablation study -- no code changes per experiment.
"""
from __future__ import annotations

from typing import List, Dict
import os
import json
import csv

from .config import DAZVADConfig
from .pipeline import DAZVADPipeline
from .datasets import get_dataset


def run_experiment(config: DAZVADConfig, out_dir: str = "results") -> Dict:
    """Run one config, persist per-config metrics, return the aggregate row."""
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    dataset = get_dataset(config)
    pipeline = DAZVADPipeline(config)
    results = pipeline.run(dataset)

    # mean metric across sequences (datasets with many clips)
    keys = ["auroc", "ap", "best_f1", "score_gap"]
    agg = {}
    for k in keys:
        vals = [r["metrics"].get(k) for r in results if r["metrics"].get(k) == r["metrics"].get(k)]
        agg[k] = sum(vals) / len(vals) if vals else float("nan")

    row = {"tag": config.tag(), **agg}
    with open(os.path.join(out_dir, "logs", f"{config.tag()}.json"), "w") as f:
        json.dump({"config": json.loads(config.to_json()), "metrics": agg}, f, indent=2)
    return row


def run_grid(configs: List[DAZVADConfig], out_dir: str = "results") -> List[Dict]:
    """Run many configs and write a single comparison table (the ablation grid)."""
    rows = [run_experiment(c, out_dir) for c in configs]
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    table_path = os.path.join(out_dir, "tables", "ablation_grid.csv")
    with open(table_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tag", "auroc", "ap", "best_f1", "score_gap"])
        w.writeheader()
        w.writerows(rows)
    return rows
