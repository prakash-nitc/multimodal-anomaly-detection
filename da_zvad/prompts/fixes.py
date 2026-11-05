"""Phrases and published constants for the fixes experiment.

Pre-registered in docs/10_phase3/PREREG_fixes.md. Kept apart from banks.py so
that file stays byte-identical to the run that used it.

Unlike banks.NEAR_MISS, nothing here depends on a dataset or its anomaly classes:
the same list goes to every benchmark and encoder, and no phrase looks like an
anomaly. NOVA (arXiv 2609.06360, sec. 3.2 and supp. F.3) forbids look-alike
actions from the normal side and prefers static, low-motion descriptions.
"""
from __future__ import annotations

from typing import List

from da_zvad.prompts.banks import CALM_ANCHORS

CALM_PEOPLE: List[str] = [
    "a person standing still", "a person sitting quietly",
    "two people talking calmly", "a person walking slowly",
]
CALM_NORMAL: List[str] = CALM_ANCHORS + CALM_PEOPLE

# NOVA's text prefixes (p.6).
KEY_NORMAL = "Normal scene: "
KEY_ANOMALOUS = "Anomalous scene: "

# Published anomaly-side weights: Flashback fixes 0.95 (p.9), NOVA fixes 0.90 (p.6).
ALPHA_FLASHBACK = 0.95
ALPHA_NOVA = 0.90

# NOVA's anchor sits in a top-50% mean over 20 text prompts plus itself, so it
# carries one part in eleven (eq. 4, p.5; M = 20, p.6).
ANCHOR_WEIGHT = 1.0 / 11.0
ANCHOR_WEIGHT_HEAVY = 1.0 / 4.0
