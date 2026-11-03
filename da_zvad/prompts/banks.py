"""Prompt banks and the normal-phrase library (Phase 4, normal side).

Pre-registered in docs/10_phase3/PREREG_normal_side.md. Every list below was
written before any score under it existed; changing one after the run needs a
dated addendum there.

Three kinds of text live here:

``WORRIES``  (abnormal side, one list per dataset)
    The anomaly classes each benchmark's own paper publishes, phrased as events.
    They play the part of an operator's "what worries you here" list. Only the
    DATASET-level taxonomy is used, never a per-scene or per-video list (NWPU
    Campus publishes which anomalies occur in each scene's test videos; using
    that would tell the detector what to look for in each camera).

    The phrases describe the EVENT and avoid naming the place. Scene words on
    the abnormal side are what caused prototype dilution (context/verbalized.py):
    text common to both ensembles pulls them together. The August campus prompts
    ("... on a pedestrian walkway") scored 0.486 on ShanghaiTech and are the
    reason for this rule.

``NEAR_MISS`` and ``CALM_ANCHORS``  (normal side)
    NEAR_MISS lists everyday actions that LOOK like a listed anomaly but are
    normal (NOVA's "confusing verbs": hugging for fighting, paying at a counter
    for robbery). They are derived from the class names alone. CALM_ANCHORS are
    four quiet, empty scenes shared by every dataset.

``library()``  (the retrieval library)
    Short normal descriptions, composed as "<place> with <activity>", that the
    retrieval step chooses from for each camera. Nothing in it may contain a
    banned word, so retrieval cannot describe an anomaly as normal -- the
    failure LLaVA showed on XD-Violence.

Sources for WORRIES (PDFs in docs/07_reference_papers/):
    ucf_crime     Sultani et al. 2018, Sec. 3.1: 13 classes
    xd_violence   Wu et al. 2020, Sec. 3: 6 classes
    nwpu_campus   Cao et al. 2023 supplementary, Table 2: 28 classes
    avenue        Liu et al. 2018, Sec. 4.1: "throwing objects, loitering and running"
    shanghaitech  the 'campus' ensemble in prompts/templates.py (Aug 2026), with
                  scene words removed. NOT yet checked against the dataset's own
                  paper (Luo et al. 2017) -- flagged in the pre-registration.
"""
from __future__ import annotations

import re
from typing import Dict, List

# ---------------------------------------------------------------- the filter
# Union of every anomaly class across the five benchmarks plus the violent and
# accident vocabulary a caption could use. Matched as whole words (and simple
# inflections), case-insensitive.
BANNED = (
    "abuse abused abusing arrest arrested arson assault assaulted attack attacked "
    "battering beat beating beaten blast blood brawl brawling burglar burglary "
    "burning chase chasing chased climb climbing collision crash crashed crashing "
    "dog dogs explode exploded explosion fall falling fallen fell fight fighting "
    "fire flames gun guns gunpoint hit hitting injured jaywalk jaywalking jump "
    "jumping kick kicking knife litter littering loiter loitering mug mugging "
    "photograph photographing police protest protesting punch punching riot rioting "
    "rob robbed robber robbery run running runs scuffle shoot shooting shoplift "
    "shoplifting smash smashed smoke snatch snatching stab stabbing steal stealing "
    "stolen theft thief threat throw throwing thrown truck trucks u-turn vandal "
    "vandalism vandalising vandalizing violence violent weapon wrong bicycle "
    "bicycles bike bikes biking cycling cyclist cyclists skateboard skateboarding "
    "scooter motorcycle motorbike illegal illegally conflict"
).split()
_BANNED_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in BANNED) + r")\b", re.I)


def banned_words(text: str) -> List[str]:
    """Every banned word in ``text`` (empty list = clean)."""
    return [m.group(0).lower() for m in _BANNED_RE.finditer(text)]


def is_clean(text: str) -> bool:
    return not banned_words(text)


# ---------------------------------------------------------------- abnormal side
WORRIES: Dict[str, List[str]] = {
    "ucf_crime": [
        "a person being abused", "a person being arrested", "a building on fire",
        "a person being assaulted", "a burglar breaking into a building",
        "an explosion", "people fighting", "a road accident with crashed cars",
        "a robbery with a weapon", "a person shooting a gun",
        "a person shoplifting goods", "a person stealing something",
        "a person vandalising property",
    ],
    "xd_violence": [
        "a person being abused", "a car accident", "an explosion",
        "people fighting", "a violent riot", "a person shooting a gun",
    ],
    "shanghaitech": [
        "a person riding a bicycle among pedestrians",
        "a person riding a skateboard or scooter among pedestrians",
        "a car or motor vehicle driving among pedestrians",
        "a person running or chasing someone",
        "people fighting or pushing each other",
        "a person jumping, falling or brawling",
    ],
    "avenue": [
        "a person throwing an object", "a person loitering", "a person running",
    ],
    "nwpu_campus": [
        "a person climbing a fence", "a car crossing a pedestrian square",
        "a person cycling on a footpath", "a person kicking a trash can",
        "a person jaywalking across a road", "a person snatching a bag",
        "a person walking across a lawn", "a vehicle making a wrong turn",
        "a person cycling across a square", "a person chasing someone",
        "a person loitering", "people in a scuffle", "a person littering",
        "a backpack left behind on the ground", "a vehicle making a u-turn",
        "a person battering someone", "a car driving on the wrong side of the road",
        "a person falling down",
        "a cyclist suddenly stopping in the middle of the road",
        "a group conflict between people", "a person climbing a tree",
        "a person stealing something", "a car parked illegally", "a truck",
        "a protest", "a person playing with water",
        "a person photographing a restricted area", "a dog",
    ],
}

# ---------------------------------------------------------------- normal side
# One or more look-alike normal actions per published class (NOVA-style),
# written from the class names only. Classes with no safe look-alike (one that
# avoids every banned word) are skipped rather than forced.
NEAR_MISS: Dict[str, List[str]] = {
    "ucf_crime": [
        "a parent holding a child's hand",            # abuse
        "a security guard talking calmly with a visitor",  # arrest
        "a street lamp glowing at night",             # arson
        "two people greeting each other with a hug",  # assault
        "a resident unlocking their own front door",  # burglary
        "sunlight reflecting brightly off a window",   # explosion
        "friends patting each other on the back",     # fighting
        "cars stopped in traffic",                    # road accident
        "a customer handing money to a cashier",      # robbery
        "a person pointing at something in the distance",  # shooting
        "a shopper putting items into a basket",      # shoplifting
        "a person loading bags into their own car",   # stealing
        "a worker cleaning a wall",                   # vandalism
    ],
    "xd_violence": [
        "a parent holding a child's hand",            # abuse
        "cars stopped in traffic",                    # car accident
        "bright stage lights flashing",               # explosion
        "people dancing close together",              # fighting
        "a large crowd enjoying a concert",           # riot
        "a person holding a microphone",              # shooting
    ],
    "shanghaitech": [
        "a person walking quickly",                   # running / chasing
        "people walking side by side",                # chasing
        "friends standing close together and talking",  # fighting / pushing
        "a person stepping down from a curb",         # jumping
        "a person bending down to tie a shoe",        # falling
        "a person waving to a friend",                # brawling
    ],
    "avenue": [
        "a person handing something to a friend",     # throwing
        "a person standing and waiting for someone",  # loitering
        "a person walking quickly",                   # running
        "a person adjusting a backpack",              # throwing a bag
    ],
    "nwpu_campus": [
        "a person leaning on a railing",              # climbing fence
        "cars driving on the road beside a square",   # car crossing square
        "a person walking on a footpath",             # cycling on footpath
        "a person putting rubbish into a bin",        # kicking trash can / littering
        "pedestrians crossing at a crosswalk",        # jaywalking
        "a person handing a bag to a friend",         # snatching bag
        "people walking along a path beside the grass",  # crossing lawn
        "cars turning at an intersection",            # wrong turn / u-turn
        "a person waiting for a friend",              # loitering
        "friends talking in a group",                 # scuffle / group conflict
        "a person walking quickly",                   # chasing
        "a person carrying a backpack",               # forgetting backpack
        "cars driving on the right side of the road",  # driving on wrong side
        "a person sitting down on a bench",           # falling
        "a person standing under a tree",             # climbing tree
        "a person picking up their own bag",          # stealing
        "cars parked in a parking area",              # illegal parking
        "students walking together in a group",       # protest
        "a fountain in a square",                     # playing with water
    ],
}

CALM_ANCHORS: List[str] = [
    "an empty walkway", "an empty street", "an empty room",
    "a quiet open area with nobody around",
]

# The self-check compares the best retrieved phrase against these: a phrase
# that matches the camera no better than "a photo of a place" carries nothing.
NEUTRAL: List[str] = ["a photo of a place", "a photo of a scene", "a photo"]

# ---------------------------------------------------------------- the library
# place -> (type, tags). Types pick the activities that make sense there; the
# 'campus' tag marks what the remove-a-domain test deletes.
PLACES: Dict[str, tuple] = {
    # outdoor pedestrian
    "a university campus walkway": ("walk", ("campus",)),
    "a paved path between university buildings": ("walk", ("campus",)),
    "a college campus square": ("square", ("campus",)),
    "a school courtyard": ("square", ("campus",)),
    "a campus road beside lecture halls": ("road", ("campus",)),
    "a footpath beside a lawn on campus": ("walk", ("campus",)),
    "a pedestrian street in a town": ("walk", ()),
    "a sidewalk beside a busy road": ("walk", ()),
    "a riverside promenade": ("walk", ()),
    "a city plaza": ("square", ()),
    "a public park": ("park", ()),
    "a tree-lined garden path": ("park", ()),
    # roads
    "a city street": ("road", ()),
    "a road intersection controlled by traffic lights": ("road", ()),
    "a highway": ("road", ()),
    "a quiet residential street": ("road", ()),
    "a parking lot": ("parking", ()),
    "an underground car park": ("parking", ()),
    "a petrol station forecourt": ("parking", ()),
    # indoor public
    "a building entrance": ("indoor", ()),
    "an office lobby": ("indoor", ()),
    "a long indoor corridor": ("indoor", ()),
    "a shopping mall corridor": ("indoor", ()),
    "a train station hall": ("transit", ()),
    "a subway platform": ("transit", ()),
    "a bus stop": ("transit", ()),
    "an airport terminal": ("transit", ()),
    "a hospital corridor": ("indoor", ()),
    "an elevator lobby": ("indoor", ()),
    # shops and services
    "a convenience store": ("shop", ()),
    "a supermarket aisle": ("shop", ()),
    "a shop checkout counter": ("shop", ()),
    "a bank lobby": ("shop", ()),
    "a restaurant": ("venue", ()),
    "a cafe": ("venue", ()),
    "a bar": ("venue", ()),
    # homes and workplaces
    "the front of a house": ("home", ()),
    "a living room": ("room", ()),
    "a kitchen": ("room", ()),
    "an apartment hallway": ("indoor", ()),
    "an open-plan office": ("work", ()),
    "a warehouse": ("work", ()),
    "a factory floor": ("work", ()),
    # film, television and sport (XD-Violence is not surveillance footage)
    "a television news studio": ("venue", ()),
    "a sports stadium": ("stadium", ()),
    "a concert stage": ("stadium", ()),
    "a classroom": ("room", ("campus",)),
}

ACTIVITIES: Dict[str, List[str]] = {
    "walk": ["people walking", "a few people walking slowly",
             "people standing and talking", "people walking in both directions",
             "people carrying bags", "nobody around"],
    "square": ["people walking", "people standing and talking",
               "people sitting on benches", "a few people crossing", "nobody around"],
    "park": ["people walking", "people sitting on benches", "people relaxing",
             "nobody around"],
    "road": ["cars driving slowly", "light traffic", "cars waiting at a red light",
             "cars parked along the side", "pedestrians crossing at a crosswalk",
             "nobody around"],
    "parking": ["cars parked in rows", "a person walking to a parked car",
                "a car slowly leaving a space", "nobody around"],
    "indoor": ["people walking", "people standing and waiting",
               "people carrying bags", "nobody around"],
    "transit": ["people waiting", "people boarding", "people walking",
                "nobody around"],
    "shop": ["customers browsing shelves", "a customer paying at the counter",
             "a shop assistant at the counter", "nobody around"],
    "venue": ["people sitting at tables", "people talking",
              "staff serving customers", "nobody around"],
    "home": ["nobody around", "a person walking to the door",
             "a car parked in the driveway"],
    "room": ["people sitting and talking", "a person reading",
             "nobody around"],
    "work": ["workers walking", "workers handling boxes", "machines operating",
             "nobody around"],
    "stadium": ["spectators watching", "a crowd cheering", "nobody around"],
}


def library(exclude_tags=()) -> List[dict]:
    """All "<place> with <activity>" phrases, minus places carrying any of
    ``exclude_tags``. Deterministic order (dict order above)."""
    out = []
    for place, (kind, tags) in PLACES.items():
        if set(tags) & set(exclude_tags):
            continue
        for act in ACTIVITIES[kind]:
            out.append({"text": "%s with %s" % (place, act), "place": place,
                        "type": kind, "tags": list(tags)})
    return out


def bank(dataset: str, normal_base: List[str], abnormal_base: List[str],
         near_miss: bool = True, worries: bool = True):
    """-> (normal phrases, abnormal phrases) for one dataset.

    The base ensembles are always kept, so every bank contains the legacy
    prompts and adds to them.
    """
    normal = list(normal_base)
    if near_miss:
        normal += NEAR_MISS[dataset] + CALM_ANCHORS
    abnormal = list(abnormal_base) + (WORRIES[dataset] if worries else [])
    return normal, abnormal


def check_all() -> List[str]:
    """Problems with the banks (empty list = fine). Run by the self-test."""
    problems = []
    for ds, phrases in NEAR_MISS.items():
        for p in phrases + CALM_ANCHORS:
            if not is_clean(p):
                problems.append("near-miss %s: %r has %s" % (ds, p, banned_words(p)))
    for e in library():
        if not is_clean(e["text"]):
            problems.append("library: %r has %s" % (e["text"], banned_words(e["text"])))
    texts = [e["text"] for e in library()]
    if len(set(texts)) != len(texts):
        problems.append("library has duplicate phrases")
    if set(WORRIES) != set(NEAR_MISS):
        problems.append("WORRIES and NEAR_MISS cover different datasets")
    return problems
