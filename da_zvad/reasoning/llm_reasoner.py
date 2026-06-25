"""M4 - LLM reasoning head (interface + stub).

For frames flagged as anomalous, an MLLM (e.g. LLaVA) produces a natural-language
explanation of *why* the frame is anomalous, grounded in the verbalized domain
context.

STATUS: interface defined; ``StubReasoner`` returns a templated placeholder so the
full pipeline runs end-to-end today. The real MLLM-backed reasoner (the next piece
to implement) will subclass ``LLMReasoner`` and override ``explain`` -- nothing else
in the pipeline changes.
"""
from __future__ import annotations

from typing import Any


class LLMReasoner:
    """Abstract interface for the reasoning module."""

    def explain(self, frame: Any, context: str, score: float) -> str:
        raise NotImplementedError(
            "Real MLLM reasoner not implemented yet. Use StubReasoner, or "
            "subclass LLMReasoner and override explain()."
        )


class StubReasoner(LLMReasoner):
    """Placeholder that keeps the pipeline runnable without loading an MLLM.

    Clearly returns a templated string -- this is plumbing, not a real explanation.
    """

    def explain(self, frame: Any, context: str, score: float) -> str:
        ctx = context or "the scene"
        return (
            f"[stub explanation] Anomalous frame (score={score:.2f}) in context "
            f"'{ctx}'. A real MLLM explanation will be generated here."
        )

    def __repr__(self) -> str:
        return "StubReasoner(<not a real explanation>)"
