"""Retrieved normality: describe a camera by searching a phrase library.

Pre-registered in docs/10_phase3/PREREG_normal_side.md.

LLaVA writes one sentence from a camera's first frames, and when those frames
already show the anomaly the sentence describes it as normal (127 of 500
XD-Violence captions). Retrieval replaces writing with choosing:

1. calm frames  -- of the first ``n_warm`` sampled frames, the ``k_calm`` that
                   the plain detector finds least abnormal (no labels used);
2. camera vector -- their mean CLIP image embedding, renormalised;
3. search        -- cosine similarity against every library phrase; keep the
                   top ``k_ret``;
4. self-check    -- if the best phrase matches the camera no better than a
                   neutral phrase ("a photo of a place"), the library does not
                   know this scene: return nothing and let the detector run on
                   its base prompts.

The library is filtered so that no phrase names an anomaly; the worst a bad
match can do is describe the wrong ordinary scene.

Everything here is numpy on precomputed, unit-normalised embeddings.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def calm_indices(scores: Sequence[float], n_warm: int, k_calm: int) -> np.ndarray:
    """Indices of the ``k_calm`` lowest scores among the first ``n_warm``.

    Ties go to the earlier frame (stable sort), so the choice is deterministic.
    A video shorter than ``k_calm`` frames uses all of them.
    """
    s = np.asarray(scores, dtype=float)[:n_warm]
    return np.sort(np.argsort(s, kind="stable")[:min(k_calm, len(s))])


def camera_vector(feats: np.ndarray, idx: Sequence[int]) -> np.ndarray:
    """Unit mean of the chosen frames' unit embeddings."""
    return _unit(np.asarray(feats, dtype=float)[np.asarray(idx)].mean(0))


def retrieve(v: np.ndarray, lib_emb: np.ndarray, k_ret: int):
    """-> (indices of the top ``k_ret`` phrases, their similarities), best first.

    Ties go to the earlier library entry.
    """
    sims = lib_emb @ v
    order = np.lexsort((np.arange(len(sims)), -sims))[:k_ret]
    return order, sims[order]


def passes_self_check(v: np.ndarray, best_sim: float, neutral_emb: np.ndarray) -> bool:
    """True when the best phrase beats every neutral phrase for this camera."""
    return bool(best_sim > float(np.max(neutral_emb @ v)))


def describe(feats: np.ndarray, base_scores: Sequence[float], lib_emb: np.ndarray,
             neutral_emb: np.ndarray, n_warm: int, k_calm: int, k_ret: int) -> dict:
    """The whole retrieval step for one camera.

    feats / base_scores: the camera's warm-up video, one row / value per
    sampled frame. Returns the chosen phrase indices (empty when the
    self-check fails) and everything needed to audit the choice.
    """
    calm = calm_indices(base_scores, n_warm, k_calm)
    v = camera_vector(feats, calm)
    idx, sims = retrieve(v, lib_emb, k_ret)
    ok = passes_self_check(v, float(sims[0]), neutral_emb)
    return {"calm": calm.tolist(), "phrases": idx.tolist() if ok else [],
            "top": idx.tolist(), "sims": [float(x) for x in sims],
            "neutral_max": float(np.max(neutral_emb @ v)), "fallback": not ok}
