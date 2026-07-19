"""The context sweep — the thesis's central experiment (gaps G2 + G3).

Question: does the verbalized scene description actually CARRY domain
adaptation, or is it decoration? Everything stays frozen; ONLY the text of
the scene description changes across four variants:

    none        M3 off — domain prompt ensembles alone
    generic     "a generic scene" (context present but uninformative)
    matched     the correct description of the target scene
    mismatched  a description of the WRONG domain family
                (industrial text on surveillance data, and vice versa)

Predicted signature if the DA-ZVAD claim holds: matched >= generic/none, and
mismatched measurably HURTS. Because no parameters update anywhere, any gap
between variants is attributable to the text alone — this is what makes the
"training-free domain adaptation" claim falsifiable rather than rhetorical.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional

import numpy as np

from .config import DAZVADConfig
from .grid import DatasetSpec, _score_or_load, _smooth
from . import evaluation

# Deliberately-wrong descriptions per domain family
MISMATCHED = {
    "surveillance": "an industrial quality-inspection image of a manufactured product on a factory line",
    "industrial": "a university campus walkway with pedestrians under CCTV surveillance",
    "generic": "a generic scene",
}


def variants_for(spec: DatasetSpec) -> List[Dict]:
    """The four context variants for one dataset spec."""
    return [
        {"name": "none",       "use_context": False, "description": None},
        {"name": "generic",    "use_context": True,  "description": "a generic scene"},
        {"name": "matched",    "use_context": True,  "description": spec.description},
        {"name": "mismatched", "use_context": True,
         "description": MISMATCHED.get(spec.domain, MISMATCHED["generic"])},
    ]


def run_context_sweep(specs: List[DatasetSpec],
                      windows: List[int] = (1, 5),
                      base: Optional[DAZVADConfig] = None,
                      out_dir: str = "results",
                      verbose: bool = True) -> List[Dict]:
    """dataset x context-variant x window -> AUROC rows + CSV + pivot print."""
    base = base or DAZVADConfig()
    raw_dir = os.path.join(out_dir, "raw")
    rows: List[Dict] = []

    for spec in specs:
        for var in variants_for(spec):
            data = _score_or_load(spec, var["name"], var["use_context"],
                                  base, raw_dir, description=var["description"])
            if verbose:
                src = "cache" if data.get("cached") else "scored"
                print(f"[ctx-sweep] {spec.label():<22} {var['name']:<11} ({src})", flush=True)
            labels_cat = np.concatenate(data["labels"])
            for w in windows:
                smoothed = [_smooth(s, w) for s in data["scores"]]
                micro = evaluation.frame_auroc(np.concatenate(smoothed), labels_cat)
                per_seq = [evaluation.frame_auroc(s, l)
                           for s, l in zip(smoothed, data["labels"])
                           if len(np.unique(l)) == 2]
                macro = float(np.mean(per_seq)) if per_seq else float("nan")
                rows.append({
                    "dataset": spec.label(), "context_variant": var["name"],
                    "description": var["description"] or "-", "window": w,
                    "auroc_micro": round(float(micro), 4),
                    "auroc_macro": round(macro, 4),
                    "n_frames": int(labels_cat.size),
                })

    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    path = os.path.join(out_dir, "tables", "context_sweep.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    if verbose:
        print(f"\n[ctx-sweep] {len(rows)} rows -> {path}")
        print("\n=== context sweep: auroc_micro (per window) ===")
        variant_order = ["none", "generic", "matched", "mismatched"]
        for w_ in sorted({r["window"] for r in rows}):
            print(f"-- window={w_} --")
            print(f"{'dataset':<22}" + "".join(f"{v:>12}" for v in variant_order))
            for ds in dict.fromkeys(r["dataset"] for r in rows):
                by_var = {r["context_variant"]: r["auroc_micro"] for r in rows
                          if r["dataset"] == ds and r["window"] == w_}
                print(f"{ds:<22}" + "".join(
                    f"{by_var.get(v, float('nan')):>12.3f}" for v in variant_order))
    return rows
