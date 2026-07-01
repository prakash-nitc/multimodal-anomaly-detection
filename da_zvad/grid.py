"""Experiment grid engine — the batched "one command fills the results chapter" job.

Key efficiency property: the expensive step (M1 CLIP scoring) depends only on
(dataset, context/prompt configuration), NOT on the temporal window. So raw
scores are computed once per (dataset, context) and cached to disk; every
temporal-window row is then derived from the cache in milliseconds.

The cache also makes runs RESUMABLE: if a Kaggle session dies mid-grid, rerun
the same command and finished datasets are loaded, not rescored.

Outputs:
    results/raw/<key>.npz            cached raw scores + labels per sequence
    results/tables/grid.csv          one row per (dataset, context, window)
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional, Dict
import csv
import os

import numpy as np

from .config import DAZVADConfig
from .pipeline import DAZVADPipeline
from .datasets import get_dataset
from . import evaluation


@dataclass
class DatasetSpec:
    """One dataset to include in the grid."""
    dataset: str                      # mvtec | shanghaitech | avenue | synthetic
    data_root: Optional[str] = None
    category: Optional[str] = None    # MVTec only
    domain: str = "generic"
    description: str = "a generic scene"

    def label(self) -> str:
        return f"{self.dataset}/{self.category}" if self.category else self.dataset


def _cache_key(spec: DatasetSpec, use_context: bool, cfg: DAZVADConfig) -> str:
    parts = [spec.dataset, spec.category or "", "ctx" if use_context else "noctx",
             cfg.clip_model.replace("/", "-"), f"step{cfg.frame_step}"]
    return "_".join(p for p in parts if p)


def _score_or_load(spec: DatasetSpec, use_context: bool, base: DAZVADConfig,
                   raw_dir: str) -> Dict:
    """Return {"scores": [np.ndarray...], "labels": [np.ndarray...], "names": [...]}."""
    key = _cache_key(spec, use_context, base)
    path = os.path.join(raw_dir, key + ".npz")
    if os.path.isfile(path):
        z = np.load(path, allow_pickle=True)
        n = int(z["n"])
        return {"scores": [z[f"s{i}"] for i in range(n)],
                "labels": [z[f"l{i}"] for i in range(n)],
                "names": list(z["names"]), "cached": True}

    cfg = replace(base, dataset=spec.dataset, data_root=spec.data_root,
                  category=spec.category, domain=spec.domain,
                  domain_description=spec.description,
                  use_context=use_context, use_temporal=False, use_reasoning=False)
    pipeline = DAZVADPipeline(cfg)
    seqs = get_dataset(cfg).sequences()

    scores, labels, names = [], [], []
    for seq in seqs:
        raw = seq.raw_scores if seq.raw_scores is not None else pipeline.score_frames(seq.frames)
        scores.append(np.asarray(raw, dtype=float))
        labels.append(np.asarray(seq.labels, dtype=int))
        names.append(seq.name)

    os.makedirs(raw_dir, exist_ok=True)
    payload = {"n": len(scores), "names": np.array(names, dtype=object)}
    for i, (s, l) in enumerate(zip(scores, labels)):
        payload[f"s{i}"], payload[f"l{i}"] = s, l
    np.savez(path, **payload)
    return {"scores": scores, "labels": labels, "names": names, "cached": False}


def _smooth(s: np.ndarray, w: int) -> np.ndarray:
    from .temporal import moving_average
    return moving_average(s, w)


def run_grid(specs: List[DatasetSpec],
             windows: List[int] = (1, 5, 9, 15),
             context_options: List[bool] = (False, True),
             base: Optional[DAZVADConfig] = None,
             out_dir: str = "results",
             verbose: bool = True) -> List[Dict]:
    """Run the full (dataset x context x window) grid. Returns the table rows."""
    base = base or DAZVADConfig()
    raw_dir = os.path.join(out_dir, "raw")
    rows: List[Dict] = []

    for spec in specs:
        for use_ctx in context_options:
            data = _score_or_load(spec, use_ctx, base, raw_dir)
            if verbose:
                src = "cache" if data.get("cached") else "scored"
                n_frames = sum(len(s) for s in data["scores"])
                print(f"[grid] {spec.label():<24} ctx={use_ctx!s:<5} "
                      f"{len(data['scores'])} seqs / {n_frames} frames ({src})")
            labels_cat = np.concatenate(data["labels"])
            for w in windows:
                smoothed = [_smooth(s, w) for s in data["scores"]]
                micro = evaluation.frame_auroc(np.concatenate(smoothed), labels_cat)
                per_seq = [evaluation.frame_auroc(s, l)
                           for s, l in zip(smoothed, data["labels"])
                           if len(np.unique(l)) == 2]
                macro = float(np.mean(per_seq)) if per_seq else float("nan")
                rows.append({
                    "dataset": spec.label(), "context": use_ctx, "window": w,
                    "auroc_micro": round(float(micro), 4),
                    "auroc_macro": round(macro, 4),
                    "n_seqs": len(data["scores"]),
                    "n_frames": int(labels_cat.size),
                })

    tables_dir = os.path.join(out_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    table_path = os.path.join(tables_dir, "grid.csv")
    with open(table_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    if verbose:
        print(f"[grid] {len(rows)} rows -> {table_path}")
    return rows
