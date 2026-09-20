"""What the game itself believes each terrain is: a water/land family and a
climate, read from descape/terrain_classes.json (regenerate with
tools/gen_terrain_classes.py). Qt-free and library-free, in the shape of
terrain_palette.py.

This exists so the auto-beach edge can pick a shoreline by DATA rather than
by guessing from enum names, which gets real cases wrong in both directions:
BEACH_NON_NAVIGABLE and its three siblings are family `land` despite the
name, and FOREST_MANGROVE, FOREST_REEDS_SHALLOWS and the five RICE_FARM
terrains are family `shallows` with no water word in sight.

Deliberately NOT the same question batch_api.is_water() answers. That one is
a name-keyword classifier with its own pinned assertions, and its documented
intent ("would a bulk edit treat this as water") excludes ice on purpose.
The two disagree on exactly eight ids; see tests/test_terrain_classes.py,
which pins the divergence rather than letting a later reader unify them.

Unknown ids return "unknown" rather than raising, unlike batch_api.is_water:
this gets called once per tile inside a paint loop, where a raise mid-stroke
would leave EditHistory's snapshot uncleared (fill_tools.py's own docstring
covers that trap).
"""

from __future__ import annotations

import json
from functools import cache, lru_cache
from pathlib import Path

_CLASSES_PATH = Path(__file__).resolve().parent / "terrain_classes.json"

UNKNOWN = "unknown"

# The four families the game counts as navigable water. `shallows` is IN,
# and that is a real decision rather than an oversight: SHALLOWS and
# SHALLOWS_AZURE are the two most natural AoE2 shoreline waters, and
# excluding them would leave the feature's main use case with no auto-beach
# at all. The cost is that picking a rice paddy or a mangrove also counts as
# water, which is odd but harmless -- it takes deliberately choosing one.
WATER_FAMILIES = frozenset({"medium_water", "deep_water", "shallow_water", "shallows"})

# The hardcoded last resort of auto_beach_for(). Unlike the two rules above
# it, this one is NOT derived from the table -- there are eight temperate
# beaches and nothing in the data says which one a given shoreline wants.
DEFAULT_BEACH_ID = 2  # BEACH


@lru_cache(maxsize=1)
def _classes() -> dict[int, dict]:
    data = json.loads(_CLASSES_PATH.read_text())["classes"]
    return {int(k): v for k, v in data.items()}


def terrain_family(terrain_id: int) -> str:
    """One of medium_water/deep_water/shallow_water/shallows/beach/land/
    obsolete, or "unknown" for an id the game's table has no enabled slot
    for."""
    entry = _classes().get(terrain_id)
    return entry["family"] if entry else UNKNOWN


def terrain_climate(terrain_id: int) -> str:
    """temperate/ice/snow, or "unknown"."""
    entry = _classes().get(terrain_id)
    return entry["climate"] if entry else UNKNOWN


def is_water_family(terrain_id: int) -> bool:
    return terrain_family(terrain_id) in WATER_FAMILIES


def is_beach_family(terrain_id: int) -> bool:
    return terrain_family(terrain_id) == "beach"


@lru_cache(maxsize=1)
def beach_terrains() -> tuple[int, ...]:
    """Every beach-family id, ascending -- what the Beach combo offers."""
    return tuple(sorted(tid for tid in _classes() if is_beach_family(tid)))


@lru_cache(maxsize=1)
def water_terrains() -> tuple[int, ...]:
    return tuple(sorted(tid for tid in _classes() if is_water_family(tid)))


@cache
def _beaches_in_climate(climate: str) -> tuple[int, ...]:
    return tuple(tid for tid in beach_terrains() if terrain_climate(tid) == climate)


def auto_beach_for(water_id: int, land_id: int | None = None) -> int:
    """The beach terrain to lay between `water_id` and the land tile being
    converted.

    Keyed on the LAND tile first, not the water, and per tile rather than per
    stroke -- which costs nothing and is strictly better: one stroke crossing
    from ice into grass gets BEACH_ICE on the ice side and BEACH on the grass
    side.

    1. If the land tile's climate has exactly one beach, use it (ice does:
       BEACH_ICE).
    2. Else if the water's climate does, use that -- this is what catches
       snow land against ICE_NAVIGABLE, which a land-only rule would miss,
       since snow has no beach of its own.
    3. Else DEFAULT_BEACH_ID. A hardcoded default, not a derivation: the
       temperate climate has eight beaches and nothing chooses between them.

    Known gap, with no data to fix it: snow land against ordinary water falls
    to step 3 and gets a sand beach. The .dat has no snow beach at all.
    """
    for terrain_id in (land_id, water_id):
        if terrain_id is None:
            continue
        candidates = _beaches_in_climate(terrain_climate(terrain_id))
        if len(candidates) == 1:
            return candidates[0]
    return DEFAULT_BEACH_ID
