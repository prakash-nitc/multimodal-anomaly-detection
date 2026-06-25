"""M3 - Verbalized domain context.

The core domain-adaptation mechanism: instead of retraining, the operator supplies
a natural-language description of the deployment environment (e.g. "a dimly lit
warehouse with conveyor belts"). That description is woven into the prompt
ensembles so that "what counts as normal" is grounded in the scene -- with zero
parameter updates.
"""
from __future__ import annotations

from typing import List, Tuple


class VerbalizedContext:
    def __init__(self, description: str = "a generic scene"):
        self.description = description.strip().rstrip(".")

    def ground(
        self, normal_prompts: List[str], abnormal_prompts: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Inject the domain description into both prompt ensembles.

        Returns new (normal, abnormal) ensembles = the originals plus
        context-grounded variants. This is what makes the model's notion of
        "normal" environment-specific.
        """
        ctx = self.description
        grounded_normal = list(normal_prompts) + [
            f"{ctx}, everything is normal",
            f"{ctx}, a usual and safe moment",
        ]
        grounded_abnormal = list(abnormal_prompts) + [
            f"{ctx}, but something abnormal is happening",
            f"{ctx}, but a dangerous or unexpected event occurs",
        ]
        return grounded_normal, grounded_abnormal

    def __repr__(self) -> str:
        return f"VerbalizedContext(description={self.description!r})"
