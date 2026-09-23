"""Which unit_consts are walls (and gates), which are eye candy, and which are
invisible -- the three const sets behind the Filters popup's Show Walls /
Show Eye Candy / Show Invisible Objects.

Filtering these exists for the same reason show_trees does: scale. Measured
2026-09-20 across the 20-file examples/ corpus, the 18 files that carry units
at all hold 99-2004 wall placements and 224-3093 eye-candy placements each, on
top of the trees show_trees already handles. Walls hide what is behind and
inside them; eye candy covers the tiles underneath it. (The GH #65 plan quotes
narrower ranges, 95-1929 and 1030-3093; those were measured over a subset.)

Both sets are DERIVED at load time from the two committed tables
(object_catalog.json's .dat `class`/`type`, unit_graphic_map.json's
`angle_count`) rather than transcribed, so they cannot drift from the .dat.
Stdlib only, no library import and no configured install, the same shape
unit_rotation.py and gate_orientation.py use for the same reason: the default
test tier covers it, and unit_filter.py stays a leaf module (view_layers.py:6-8
records why that matters).

Deliberately NOT importing unit_sprites, which pulls numpy, asset_source and
sld_decoder and would take unit_filter.py off the leaf list. The 8-wall
frozenset it already holds as _ROTATION_VARIANT_CONSTS is re-derived here
rather than copied; tests/test_unit_kind.py pins the two against each other.

## wall_consts(): 105 consts, 9 walls plus 96 gates

Raw `class == 27` is the wrong basis, and AGENTS.md says so ("excluded by
const set, never by `unit.class_`"). Class 27 holds 36 consts and 27 of them
are invisible scaffolding: Empty building x5, Sheep annex1/2, Empty TC annex,
Mole annex 1-4, Thin blocker spawner A/B, Gaia transition building, Relic
building. Adding `angle_count == 5` -- the five stored SHAPES that make a wall
a wall, per AGENTS.md's own hard rule -- narrows it to exactly 72 WALL,
117 WALL2, 119 WALL4, 155 WALL3, 231 Aqueduct, 370 CWAL, 788 SWAL, 1062 FENCE,
2678 WALL5. 208 TWAL is class 27 with no unit_graphic_map.json entry at all
and zero corpus placements; it stays out. Corpus: the derived set covers 8193
of the 8204 class-27 placements (99.87%), dropping only Empty TC annex (10) and
Mole under construction (1), neither of which is a wall.

Gates are in, so a hidden wall line isn't left with gate-shaped lumps in it.

**The one member beyond unit_sprites.wall_connector_consts() is Aqueduct (231),
and that divergence must not be "fixed" in either direction.**
tests/test_wall_connectivity.py pins Aqueduct absent from the connector set and
AGENTS.md carries an open [NEEDS DECISION] on it, but that question is about
connector membership, where an extra const gets a real wall's stored index
re-derived from a wall-calibrated neighbour mask -- a write-path correctness
bug. Here it only hides an aqueduct when you asked to hide walls, which is what
you want. test_unit_kind.py pins the delta as exactly {231} so it stays
asserted rather than accidental.

## eye_candy_consts(): 387 consts

`type == 10` is the .dat's own EyeCandy object type, 556 consts. Subtracted:

- terrain_palette.TREE_UNIT_IDS (53). show_trees owns those.
- _RESOURCE_CLASSES: ocean/deep-sea/shore fish, berry bush, stone mine, gold
  mine, ore mine, salvage pile, resource pile, and DE's class 63 holding
  Oysters/Whale. Corpus placements this drops: GOLDM 1098, STONM 560,
  FORAG 426. These are what the user is looking *for*, the same reasoning that
  split show_trees off show_gaia.
- class 34, the 96 cliff consts (all of them type 10), owned by the cliff
  tooling.

Class 30 (Flag) is not excluded, and that is the one arguable consequence: its
only type-10 members are FLARE (112) and FLR_R (201), and the class-11 flare
family (274/332/697/1689/1785) comes along too. Flares are cosmetic markers, so
hiding them under "eye candy" is right, and hand-carving them out would break
the measured-not-judged property the set is built on. The real flags
(1150/1151/1307 and the rest) are type 20 and never in the set at all.

Owner-independent, like show_trees: eye candy is usually GAIA but a real player
can own it, and an eye-candy object is eye candy either way.

Not to be confused with the Terrain brush's persisted "Eye candy" checkbox
(settings.get_paint_eye_candy, viewer.py's paint_eye_candy_check), which is a
different question over a different set -- which doodads a terrain stroke
PLACES, versus which consts a filter HIDES.

Blockers are also type 10, and they are subtracted too: they belong to
invisible_consts() below, so the two sets stay disjoint (GH #53, by decision
2026-09-21). The consts that moved: BLOCKER 1776, Blocker 1x3 2423, Blocker
3x1 2424, plus the hidden-in-editor Terrain blocker 1613, Thin blocker 2435,
Buildable Blocker 1x3/3x1 2429/2430 and OLD-FISH3 260.

## invisible_consts(): the .dat's `standing_graphic[0] == -1`, 84 consts

object_catalog.json carries `no_graphic: true` on exactly these (GH #53). The
game's own "Erase Invisible Objects" is engine-side, with no data file listing
its members, so the set is derived from the .dat instead.

The basis is "the .dat has no standing graphic", NOT "missing from
unit_graphic_map.json". Absence from the map also catches objects that have
real art in the game but no map entry yet: FARM 50 (242 corpus placements),
FLARE4 697 (230), FLAME1-4, WFALL, SMOKE, BUTTERFL2. Those are art gaps, not
invisible objects, and none of them is in this set.

Members that are not hidden in the editor: Invisible Object A-E (1291, 2551,
2553, 2555, 2563, class 38), Map Revealer / Medium / Giant (837, 1774, 1775,
class 30), BLOCKER 1776 and Blocker 1x3/3x1 2423/2424 (class 14), and
OLD-ACADEMY 0 / OLD_EXPLORER 127. The rest are hidden-in-editor scaffolding,
e.g. class 27's Empty TC annex 890, and stay in: if a file places one it still
draws as a box and is still invisible in the game. Corpus placements (20 files,
measured 2026-09-21): REVEAL 837 459 across 8 files, 1774 50, 1775 3, BLOCKER
1776 36, Blocker 3x1 21, Invisible Object A 17 across 9 files, Blocker 1x3 16,
Empty TC annex 10.

Decisions made with the user: Map Revealers are in, and blockers move out of
eye candy (above). Owner-blind like the other const gates: Invisible Object A
exists to be player-owned ("keeps player alive").

## Two documented non-changes

- render.wall_variant_rotation_overrides is built over ALL units ignoring the
  filter (documented at its own definition), so a hidden wall still shapes its
  visible neighbours' connectors. That is correct: the stored shape is a
  property of the file, not of what is on screen. Do not "fix" it to honour
  these gates.
- No clear_caches() here, unlike object_catalog.py's name resolution: neither
  set touches a configured install, so nothing they read can change at runtime.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from descape import gate_orientation
from descape.terrain_palette import TREE_UNIT_IDS

_CATALOG_PATH = Path(__file__).resolve().parent / "object_catalog.json"
_GRAPHIC_MAP_PATH = Path(__file__).resolve().parent / "unit_graphic_map.json"

# object_catalog.json's `objects[<const>]["class"]` -- the .dat's unit.class_.
_WALL_CLASS = 27
_CLIFF_CLASS = 34

# The five stored SHAPES that make a wall a wall, per AGENTS.md's hard rule.
_WALL_ANGLE_COUNT = 5

# object_catalog.json's `objects[<const>]["type"]` -- the .dat's unit.type.
_EYE_CANDY_TYPE = 10

# Ocean / deep-sea / shore fish, berry bush, stone mine, gold mine, ore mine,
# salvage pile, resource pile, and DE's class 63 (Oysters/Whale).
_RESOURCE_CLASSES = frozenset({5, 7, 8, 31, 32, 33, 40, 41, 48, 63})


@lru_cache(maxsize=1)
def _objects() -> dict[int, dict]:
    """unit_const -> its committed catalog entry, keyed by int."""
    data = json.loads(_CATALOG_PATH.read_text())["objects"]
    return {int(const): entry for const, entry in data.items()}


@lru_cache(maxsize=1)
def _angle_counts() -> dict[int, int]:
    """unit_const -> its standing graphic's angle_count. Same table
    unit_sprites.graphic_map() reads, loaded independently to keep this module
    free of that one's install-dependent imports (unit_rotation.py:78 reads it
    the same way, for the same reason)."""
    data = json.loads(_GRAPHIC_MAP_PATH.read_text())["graphics"]
    return {int(const): int(entry["angle_count"]) for const, entry in data.items()}


@lru_cache(maxsize=1)
def wall_consts() -> frozenset[int]:
    """Every const Show Walls hides: real walls plus all four orientations of
    every gate. See the module docstring for why class 27 alone is not it."""
    angle_counts = _angle_counts()
    walls = {
        const
        for const, entry in _objects().items()
        if entry.get("class") == _WALL_CLASS and angle_counts.get(const) == _WALL_ANGLE_COUNT
    }
    gates = {const for group in gate_orientation.groups().values() for const in group}
    return frozenset(walls | gates)


@lru_cache(maxsize=1)
def invisible_consts() -> frozenset[int]:
    """Every const Show Invisible Objects hides: the .dat gives it no standing
    graphic. See the module docstring for why that and not "no map entry"."""
    return frozenset(const for const, entry in _objects().items() if entry.get("no_graphic"))


# .dat class -> descape/editor_markers.py category (GH #53 Part B); any other invisible const is "other".
_INVISIBLE_CATEGORY_BY_CLASS = {38: "invisible", 30: "revealer", 14: "blocker"}


def invisible_category(unit_const: int) -> str | None:
    """Which editor-only marker an invisible_consts() member draws as, or None
    for any const outside that set."""
    if unit_const not in invisible_consts():
        return None
    return _INVISIBLE_CATEGORY_BY_CLASS.get(_objects()[unit_const].get("class"), "other")


@lru_cache(maxsize=1)
def eye_candy_consts() -> frozenset[int]:
    """Every const Show Eye Candy hides: the .dat's own EyeCandy type, minus
    the trees show_trees owns, the resources the user is looking for, the
    cliffs the cliff tooling owns, and the invisible objects (blockers)."""
    invisible = invisible_consts()
    return frozenset(
        const
        for const, entry in _objects().items()
        if entry.get("type") == _EYE_CANDY_TYPE
        and const not in TREE_UNIT_IDS
        and entry.get("class") not in _RESOURCE_CLASSES
        and entry.get("class") != _CLIFF_CLASS
        and const not in invisible
    )
