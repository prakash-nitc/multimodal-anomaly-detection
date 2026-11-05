"""Phrases for the one-at-a-time improvement round.

Pre-registered in docs/10_phase3/PREREG_ideas.md. Kept apart from banks.py and
fixes.py so those files stay byte-identical to the runs that used them.
"""
from __future__ import annotations

from typing import Dict, List

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

# Step 3: one abnormal bank for every dataset and encoder, written once by an LLM
# (Claude) before it was scored, then frozen. Concrete actions, because ASK-Hint
# finds them work better than abstract labels. Twelve types of four phrases; the
# author knew the benchmarks' anomaly classes, as the pre-registration states.
BANK: Dict[str, List[str]] = {
    "fighting": [
        "two people punching each other", "a person kicking someone on the ground",
        "a group of people brawling", "a person violently shoving another person",
    ],
    "weapon": [
        "a person pointing a gun at someone", "a person firing a gun",
        "a person threatening someone with a knife", "a person stabbing someone",
    ],
    "theft": [
        "a person snatching a bag from someone", "a robber holding up a cashier",
        "a person stealing goods from a shop", "a burglar climbing through a window",
    ],
    "vandalism": [
        "a person smashing a window", "a person kicking a door open",
        "a person damaging a parked car", "a person spraying graffiti on a wall",
    ],
    "road accident": [
        "two cars crashing into each other", "a car hitting a pedestrian",
        "a motorcycle skidding across the road", "a vehicle overturned on the road",
    ],
    "fire or explosion": [
        "flames and thick smoke rising from a building", "a car on fire",
        "an explosion with debris flying", "a blast throwing people to the ground",
    ],
    "riot or arrest": [
        "a crowd of rioters throwing objects", "a mob clashing with police",
        "police wrestling a person to the ground", "a person being handcuffed and arrested",
    ],
    "abuse": [
        "an adult hitting a child", "a person dragging another person by force",
        "a person being beaten while lying down", "a person being grabbed and restrained",
    ],
    "falling": [
        "a person collapsing to the ground", "a person lying motionless on the ground",
        "a person falling from a height", "a person tripping and falling hard",
    ],
    "vehicle among pedestrians": [
        "a cyclist riding through a crowd of pedestrians", "a car driving on a pedestrian walkway",
        "a motorbike riding on a footpath", "a person on a skateboard weaving between pedestrians",
    ],
    "sudden running": [
        "a person sprinting through a crowd", "a person chasing someone",
        "people running away in panic", "a person jumping over a barrier",
    ],
    "trespass": [
        "a person climbing over a fence", "a person forcing entry into a restricted area",
        "a person leaving a suspicious bag behind", "a person hiding and watching people",
    ],
}
BANK_PHRASES: List[str] = [p for v in BANK.values() for p in v]

# Flashback's repulsive prompting (Fig. 2, p.4): different yaml wrappers per side,
# written out here so the text does not depend on a yaml library's line wrapping.
GENERIC_CATEGORY = "anomalous event"


def template_normal(text: str) -> str:
    return "Normal Scene:\n- %s\n" % text


def template_anomalous(text: str, category: str = GENERIC_CATEGORY) -> str:
    return "Anomalous Scene:\n    Action Category: %s\n    Scene Description: %s\n" % (category, text)
