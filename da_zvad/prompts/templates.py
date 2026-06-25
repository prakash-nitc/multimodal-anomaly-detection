"""Per-domain prompt ensembles.

Anomaly scoring contrasts an ensemble of "normal" prompts against an ensemble of
"abnormal" prompts. Keeping these in one place makes the prompt-design ablation
(generic vs. domain-specific vs. ensembled) a one-line change.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# domain -> (normal_prompts, abnormal_prompts)
DOMAINS: Dict[str, Tuple[List[str], List[str]]] = {
    "generic": (
        [
            "a normal scene",
            "a typical everyday scene with nothing unusual",
            "a calm and ordinary situation",
        ],
        [
            "an abnormal scene",
            "an unusual or unexpected event",
            "something is clearly wrong",
        ],
    ),
    "industrial": (
        [
            "a photo of a flawless product",
            "a perfect product without any defects",
            "a normal item passing quality inspection",
        ],
        [
            "a photo of a defective product",
            "a product with a crack, scratch or damage",
            "an item that failed quality inspection",
        ],
    ),
    "surveillance": (
        [
            "a normal day with people behaving ordinarily",
            "a calm and safe public area",
            "nothing dangerous or abnormal is happening",
        ],
        [
            "a dangerous or violent event",
            "a fight, robbery or accident in progress",
            "an abnormal and unsafe situation",
        ],
    ),
}


def get_prompts(domain: str) -> Tuple[List[str], List[str]]:
    """Return (normal_prompts, abnormal_prompts) for a domain.

    Falls back to the generic ensemble for unknown domains.
    """
    return DOMAINS.get(domain, DOMAINS["generic"])
