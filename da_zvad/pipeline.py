"""DA-ZVAD pipeline -- orchestrates M1 -> M2 -> M3 -> M4.

Design note: scoring (M1, needs CLIP) is separated from post-processing (M2/M3
detection/metrics, model-free). This lets the framework be exercised end-to-end
with precomputed/synthetic scores -- which is exactly how the synthetic demo and
the unit-level checks run without a GPU.
"""
from __future__ import annotations

from typing import List, Optional, Dict, Any
import numpy as np

from .config import DAZVADConfig
from .prompts import get_prompts
from .context import VerbalizedContext
from .temporal import TemporalAggregator
from .datasets.base import FrameSequence
from . import evaluation


def select_explanation_frames(predictions: np.ndarray, scores: np.ndarray,
                              budget: int = 8) -> List[int]:
    """Pick the peak-score frame of each contiguous flagged event.

    One explanation per *event* (not per frame): cheaper, non-redundant, and the
    peak frame is where the anomaly is most visible. Events are ranked by peak
    score; at most ``budget`` are kept.
    """
    predictions = np.asarray(predictions, dtype=int)
    scores = np.asarray(scores, dtype=float)
    events = []  # (peak_score, peak_index)
    start = None
    for i, p in enumerate(predictions):
        if p == 1 and start is None:
            start = i
        elif p == 0 and start is not None:
            seg = slice(start, i)
            peak = start + int(np.argmax(scores[seg]))
            events.append((scores[peak], peak))
            start = None
    if start is not None:
        seg = slice(start, len(predictions))
        peak = start + int(np.argmax(scores[seg]))
        events.append((scores[peak], peak))
    events.sort(reverse=True)
    return sorted(idx for _, idx in events[:budget])


def build_reasoner(config: DAZVADConfig):
    if config.reasoner == "llava":
        from .reasoning import LlavaReasoner
        return LlavaReasoner(model_id=config.llava_model, device=config.device)
    from .reasoning import StubReasoner
    return StubReasoner()


class DAZVADPipeline:
    def __init__(self, config: DAZVADConfig):
        self.config = config
        # M3 verbalized context (optional)
        self.context = (VerbalizedContext(config.domain_description,
                                          mode=config.context_mode)
                        if config.use_context else None)
        # M2 temporal (optional)
        self.temporal = TemporalAggregator(config.temporal_window) if config.use_temporal else None
        # M4 reasoning (optional; stub or LLaVA per config)
        self.reasoner = build_reasoner(config) if config.use_reasoning else None
        self._encoder = None  # M1, lazy

    # ---- M3: build the (optionally context-grounded) prompt ensembles ----
    def build_prompts(self):
        normal, abnormal = get_prompts(self.config.domain)
        if self.context is not None:
            normal, abnormal = self.context.ground(normal, abnormal)
        return normal, abnormal

    # ---- M1: per-frame scoring (requires CLIP) ----
    def score_frames(self, frames: List[Any]) -> np.ndarray:
        if self._encoder is None:
            from .encoders import CLIPEncoder
            self._encoder = CLIPEncoder(
                self.config.clip_model, self.config.clip_pretrained, self.config.device
            )
        normal, abnormal = self.build_prompts()
        return self._encoder.score_frames(frames, normal, abnormal)

    # ---- M2 + detection + M4 + metrics on a score sequence ----
    def postprocess(self, raw_scores: np.ndarray, seq: FrameSequence) -> Dict:
        raw_scores = np.asarray(raw_scores, dtype=float)
        agg = self.temporal.apply(raw_scores) if self.temporal is not None else raw_scores
        preds = (agg >= self.config.threshold).astype(int)

        explanations = {}
        if self.reasoner is not None:
            ctx = self.config.domain_description
            chosen = select_explanation_frames(preds, agg, self.config.max_explanations)
            for i in chosen:
                frame = seq.frames[i] if i < len(seq.frames) else None
                explanations[int(i)] = self.reasoner.explain(frame, ctx, float(agg[i]))

        metrics = evaluation.summarize(agg, seq.labels) if seq.labels is not None else {}
        return {
            "name": seq.name,
            "raw_scores": raw_scores,
            "scores": agg,
            "predictions": preds,
            "labels": seq.labels,
            "explanations": explanations,
            "metrics": metrics,
        }

    # ---- full run on one sequence ----
    def run_sequence(self, seq: FrameSequence) -> Dict:
        if seq.raw_scores is not None:          # synthetic / precomputed
            raw = seq.raw_scores
        else:                                    # real frames -> M1
            raw = self.score_frames(seq.frames)
        return self.postprocess(raw, seq)

    def run(self, dataset) -> List[Dict]:
        return [self.run_sequence(s) for s in dataset.sequences()]

    def __repr__(self) -> str:
        mods = ["M1"]
        if self.temporal: mods.append("M2")
        if self.context: mods.append("M3")
        if self.reasoner: mods.append("M4")
        return f"DAZVADPipeline(modules={'+'.join(mods)}, domain={self.config.domain})"
