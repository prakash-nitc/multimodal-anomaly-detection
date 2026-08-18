"""G4 experiment: explanation quality under domain shift.

Per the DA findings report, no existing work studies how MLLM explanations
behave when the scene context is wrong. This harness makes that measurable:
for each flagged event (peak frame per contiguous detection), M4 generates
TWO explanations — one grounded in the matched scene description, one in a
deliberately mismatched (wrong-domain) description — and saves them
side-by-side as a gallery (markdown for reading, JSON for later scoring).

Runs with reasoner="stub" on CPU (plumbing test) or reasoner="llava" on a GPU
(the real study). Qualitative by design at this stage: the gallery is the
Sem-3 deliverable; quantitative explanation scoring is Sem-4 material.
"""
from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Dict, List, Optional

import numpy as np

from .config import DAZVADConfig
from .grid import DatasetSpec, _score_or_load
from .context_sweep import MISMATCHED
from .datasets import get_dataset
from .pipeline import build_reasoner, select_explanation_frames
from .temporal import moving_average


def run_explain_shift(spec: DatasetSpec,
                      base: Optional[DAZVADConfig] = None,
                      max_events: int = 6,
                      window: int = 5,
                      out_dir: str = "results",
                      verbose: bool = True) -> List[Dict]:
    """Generate matched-vs-mismatched explanations for top flagged events."""
    base = base or DAZVADConfig()
    raw_dir = os.path.join(out_dir, "raw")

    # 1) scores under the MATCHED context (cached if the sweep already ran)
    data = _score_or_load(spec, "matched", True, base, raw_dir,
                          description=spec.description)

    # 2) reload sequences once for frame access (scores cache holds no frames)
    cfg = replace(base, dataset=spec.dataset, data_root=spec.data_root,
                  category=spec.category, domain=spec.domain,
                  domain_description=spec.description)
    seqs = {s.name: s for s in get_dataset(cfg).sequences()}

    # 3) pick the top events across all sequences (peak score per event)
    candidates = []  # (peak_score, seq_name, frame_idx)
    for name, scores in zip(data["names"], data["scores"]):
        agg = moving_average(np.asarray(scores, dtype=float), window)
        preds = (agg >= base.threshold).astype(int)
        for idx in select_explanation_frames(preds, agg, budget=max_events):
            candidates.append((float(agg[idx]), name, int(idx)))
    candidates.sort(reverse=True)
    chosen = candidates[:max_events]
    if verbose:
        print(f"[explain-shift] {spec.label()}: {len(chosen)} events selected "
              f"(reasoner={base.reasoner})", flush=True)

    # 4) explain each event under matched vs mismatched context
    reasoner = build_reasoner(base)
    mismatched_desc = MISMATCHED.get(spec.domain, MISMATCHED["generic"])
    gallery = []
    for peak, name, idx in chosen:
        seq = seqs.get(name)
        frame = seq.frames[idx] if seq and idx < len(seq.frames) else None
        entry = {
            "dataset": spec.label(), "sequence": name, "frame_idx": idx,
            "score": round(peak, 3),
            "matched_context": spec.description,
            "mismatched_context": mismatched_desc,
            "explanation_matched": reasoner.explain(frame, spec.description, peak),
            "explanation_mismatched": reasoner.explain(frame, mismatched_desc, peak),
        }
        gallery.append(entry)
        if verbose:
            print(f"  {name} @f{idx} (score {peak:.2f})", flush=True)

    # 5) persist: JSON (for later scoring) + markdown gallery (for reading)
    gal_dir = os.path.join(out_dir, "explanations")
    os.makedirs(gal_dir, exist_ok=True)
    tag = spec.label().replace("/", "_")
    with open(os.path.join(gal_dir, f"{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(gallery, f, indent=2, ensure_ascii=False)
    md = [f"# Explanation gallery under domain shift — {spec.label()}",
          "", f"Matched context: *{spec.description}*",
          f"Mismatched context: *{mismatched_desc}*", ""]
    for e in gallery:
        md += [f"## {e['sequence']} · frame {e['frame_idx']} · score {e['score']}",
               f"**Matched:** {e['explanation_matched']}",
               "", f"**Mismatched:** {e['explanation_mismatched']}", "", "---", ""]
    with open(os.path.join(gal_dir, f"{tag}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    if verbose:
        print(f"[explain-shift] gallery -> {gal_dir}/{tag}.md (+.json)")
    return gallery
