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
    # ShanghaiTech / Avenue style campus footage.
    #
    # The "surveillance" ensemble above encodes a CRIME notion of anomaly
    # (violence, robbery, accidents) inherited from datasets like UCF-Crime.
    # Campus benchmarks do not contain those events. Their documented anomaly
    # classes are wheeled vehicles intruding on pedestrian walkways plus abrupt
    # human motion (Liu et al., 2018). Asked whether a frame shows "a robbery
    # in progress", a frozen CLIP correctly answers "no" for every frame of the
    # benchmark, anomalous ones included -- which is chance performance by
    # construction, not an encoder failure.
    #
    # These prompts come from the dataset's PUBLISHED anomaly taxonomy, not
    # from inspecting test labels. In DA-ZVAD terms that taxonomy is exactly
    # what verbalized domain context is supposed to carry: an operator
    # deploying to a campus states what counts as out of place there.
    "campus": (
        [
            "pedestrians walking on a campus walkway",
            "people strolling along a paved outdoor path",
            "students walking on foot on a university campus",
            "an ordinary walkway with people going about on foot",
        ],
        [
            "a person riding a bicycle on a pedestrian walkway",
            "a person riding a skateboard or scooter among pedestrians",
            "a car or motor vehicle driving on a pedestrian path",
            "a person running or chasing on a walkway",
            "people fighting or pushing each other on a walkway",
            "a person jumping, falling or brawling on a walkway",
        ],
    ),
}


def get_prompts(domain: str) -> Tuple[List[str], List[str]]:
    """Return (normal_prompts, abnormal_prompts) for a domain.

    Falls back to the generic ensemble for unknown domains.
    """
    return DOMAINS.get(domain, DOMAINS["generic"])
