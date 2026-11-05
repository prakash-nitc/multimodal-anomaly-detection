"""Phrases for the one-at-a-time improvement round.

Pre-registered in docs/10_phase3/PREREG_ideas.md. Kept apart from banks.py and
fixes.py so those files stay byte-identical to the runs that used them.
"""
from __future__ import annotations

from typing import List

# A second wording of fix C, written before it was scored: four quiet places and
# four calm people, as in fixes.CALM_NORMAL, with no phrase in common.
CALM_SECOND_PLACES: List[str] = [
    "a deserted pavement", "an unoccupied hallway", "a vacant plaza",
    "a still courtyard with no one in it",
]
CALM_SECOND_PEOPLE: List[str] = [
    "someone waiting patiently", "a person resting on a bench",
    "a few people chatting quietly", "someone strolling at an easy pace",
]
CALM_SECOND: List[str] = CALM_SECOND_PLACES + CALM_SECOND_PEOPLE
