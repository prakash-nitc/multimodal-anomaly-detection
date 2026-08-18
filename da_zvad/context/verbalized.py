"""M3 - Verbalized domain context.

The core domain-adaptation mechanism: instead of retraining, the operator supplies
a natural-language description of the deployment environment (e.g. "a dimly lit
warehouse with conveyor belts"). That description is woven into the prompt
ensembles so that "what counts as normal" is grounded in the scene -- with zero
parameter updates.

Two fusion modes, and the difference between them is an empirical finding rather
than a style choice:

``both``
    Append the description to BOTH ensembles. This was the original design and
    it measurably HURTS (ShanghaiTech, 2026-08-14: matched context 0.463 micro
    AUROC vs 0.493 with no context, and a deliberately WRONG description scored
    best at 0.504).

    The mechanism is prototype dilution. Each ensemble is mean-pooled into one
    embedding, so text added to both sides becomes a common component of both
    prototypes, pulling them toward each other and shrinking the margin that
    carries the decision. The better the description matches the imagery, the
    more strongly it aligns with every frame, and the more of the contrast it
    swamps -- which is why an ACCURATE description was the most damaging one.

``normal``
    Ground only the normal ensemble; leave the abnormal ensemble alone. This
    follows the premise of the method: the deployment environment defines what
    NORMAL looks like there, and an anomaly is a departure from it. Because the
    added text is no longer common to both prototypes, it shifts the boundary
    instead of collapsing it.

``both`` is retained so the negative result above stays reproducible.
"""
from __future__ import annotations

from typing import List, Tuple

MODES = ("normal", "both")


class VerbalizedContext:
    def __init__(self, description: str = "a generic scene", mode: str = "normal"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.description = description.strip().rstrip(".")
        self.mode = mode

    def ground(
        self, normal_prompts: List[str], abnormal_prompts: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Inject the domain description into the prompt ensembles.

        Returns new (normal, abnormal) ensembles. Which ensembles are touched
        depends on ``mode`` -- see the module docstring.
        """
        ctx = self.description
        grounded_normal = list(normal_prompts) + [
            f"{ctx}, everything is normal",
            f"{ctx}, a usual and safe moment",
        ]
        if self.mode == "normal":
            return grounded_normal, list(abnormal_prompts)

        grounded_abnormal = list(abnormal_prompts) + [
            f"{ctx}, but something abnormal is happening",
            f"{ctx}, but a dangerous or unexpected event occurs",
        ]
        return grounded_normal, grounded_abnormal

    def __repr__(self) -> str:
        return (f"VerbalizedContext(description={self.description!r}, "
                f"mode={self.mode!r})")
