"""Which unit_consts are walls (and gates), which are passable eye candy, which
are obstacles, which are invisible, and which are buildings -- the five const
sets behind the Filters popup's Show Walls / Show Eye Candy / Show Obstacles /
Show Invisible Objects / Show Buildings.

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
sld_decoder and would take unit_filter.py off the leaf list. The 9-wall
frozenset it already holds as _ROTATION_VARIANT_CONSTS is re-derived here
rather than copied; tests/test_unit_kind.py pins the two against each other.

## wall_consts(): 105 consts, 9 walls plus 96 gates

Raw `class == 27` is the wrong basis, and AGENTS.md says so ("excluded by
const set, never by `unit.class_`"). Class 27 holds 39 consts and 30 of them
are invisible scaffolding: Empty building x5, Sheep annex1/2, Empty TC annex,
Mole annex 1-4, Thin blocker spawner A/B, Gaia transition building, Relic
building, Empty Castle Annex and the rest. Adding `angle_count == 5` -- the five stored SHAPES that make a wall
a wall, per AGENTS.md's own hard rule -- narrows it to exactly 72 WALL,
117 WALL2, 119 WALL4, 155 WALL3, 231 Aqueduct, 370 CWAL, 788 SWAL, 1062 FENCE,
2678 WALL5. 208 TWAL is class 27 with no unit_graphic_map.json entry at all
and zero corpus placements; it stays out. Corpus: the derived set covers 8193
of the 8204 class-27 placements (99.87%), dropping only Empty TC annex (10) and
Mole under construction (1), neither of which is a wall.

Gates are in, so a hidden wall line isn't left with gate-shaped lumps in it.

**This set equals unit_sprites.wall_connector_consts(), with no divergence.**
Aqueduct (231) used to be the one member beyond it, while its connector
membership was an open decision. On 2026-09-26 (GH #110) the user decided to
treat Aqueduct as a full wall, so it joined the connector set and the delta is
now empty. test_unit_kind.py pins that equality in both directions.

## eye_candy_consts(): 389 consts

`type == 10` is the .dat's own EyeCandy object type, 569 consts. Subtracted:

- terrain_palette.TREE_UNIT_IDS (56). show_trees owns those.
- _RESOURCE_CLASSES: ocean/deep-sea/shore fish, berry bush, stone mine, gold
  mine, ore mine, salvage pile, resource pile, and DE's class 63 holding
  Oysters/Whale. Corpus placements this drops: GOLDM 1098, STONM 560,
  FORAG 426. These are what the user is looking *for*, the same reasoning that
  split show_trees off show_gaia.
- class 34, the 96 cliff consts (all of them type 10), owned by the cliff
  tooling.

Class 30 (Flag) is not excluded. Its only type-10 members are FLARE (112) and
FLR_R (201). FLARE and the class-11 flare family (274/332/697/1689/1785) stand
on the .dat's BLANK graphic, so they are invisible_consts() and subtracted from
here like blockers; FLR_R has its own graphic and stays eye candy. The real
flags (1150/1151/1307 and the rest) are type 20 and never in the set at all.

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

eye_candy_consts() is the umbrella: player_stats still counts all of it as
"Eye candy". The Filters popup splits it into the two disjoint sets below.

## obstacle_consts() / passable_eye_candy_consts(): 131 / 258 (GH #149)

Source: the .dat's `unit.obstruction_type`, committed as object_catalog.json's
`obstruction` key (omitted when 0). The 389 eye-candy consts by
obstruction_type, measured 2026-10-07:

| type | consts | what | corpus placements |
|---|---|---|---|
| 0 | 258 | grass, plants, stumps, rubble, skeletons, barrels, rugs, signs | 15,722 |
| 2 | 70 | ROCKX/ROCKF*/ROCKJ1, ruins, pagodas, burned buildings, Theatre | 1,933 |
| 3 | 44 | statues, LUMBER/GOODS/QUARRY piles, LOOT, Stonehenge, stalls | 180 |
| 10 | 11 | MNTN1-11 mountains | 133 |
| 4 | 6 | "Rock ... Hover" variants | 3 (old-allies only) |

Cross-checks: trees are all 2 or 3, buildings mostly 2, type-70 units 5,
cliffs 2. So 0 is the passable value. **Rule: obstacle_consts() is the
eye_candy_consts() members with a non-zero obstruction_type**; the rest are
passable_eye_candy_consts().

Two calls go beyond the measurement and are pending an in-game check (the
GH #149 collision probe, not yet run):

- The six Hover rocks (type 4) are obstacles. Type 4 is also on Thin blocker
  spawners and Mole annexes, which block.
- BARRELS/RUGS/SIGN/CRATR have obstruction 0 but a 0.2-0.5 `collision_size`.
  They stay passable: obstruction_type, not collision_size, is the gate.

A mismatch there moves that const across the rule with an explicit exception
list here, not a change of basis.

**Known limitation:** wall_run.blocked_tiles() treats an obstacle as
occupying only its 1x1 anchor tile, the footprint every non-building gets
(tile_span(..., NON_BUILDING_SPAN)). About 25 of the 131 are bigger: 3x3
mountains, the 2.5-tile Theatre and Stonehenge, the 2x2 CastleRuins and MRKT,
the 1.5-tile ruins and burned buildings. A wall run can pass through their
outer tiles until obstacles get real spans (gen_unit_render_data.py's span
tables, which would also change drawing and picking).

## invisible_consts(): no .dat standing graphic, 115 consts

object_catalog.json carries `no_graphic: true` on exactly these (GH #53):
`standing_graphic[0]` is -1 or the .dat's `BLANK` graphic, which has no art
file. The 2026-09-22 game patch moved 20 of these consts (Empty TC annex and
the other class-27 scaffolding) from -1 to BLANK, and on 2026-09-26 the user
decided BLANK counts as no graphic everywhere, flares included. The
game's own "Erase Invisible Objects" is engine-side, with no data file listing
its members, so the set is derived from the .dat instead.

The basis is "the .dat has no standing graphic", NOT "missing from
unit_graphic_map.json". Absence from the map also catches objects that have
real art in the game but no map entry yet: FARM 50 (242 corpus placements),
FLAME1-4, WFALL, SMOKE, BUTTERFL2. Those are art gaps, not invisible objects,
and none of them is in this set.

Members that are not hidden in the editor: Invisible Object A-E (1291, 2551,
2553, 2555, 2563, class 38), Map Revealer / Medium / Giant (837, 1774, 1775,
class 30), BLOCKER 1776 and Blocker 1x3/3x1 2423/2424 (class 14), the flares
FLARE2 274 and FLARE5 1689/1785, and OLD-ACADEMY 0 / OLD_EXPLORER 127. The rest are hidden-in-editor scaffolding,
e.g. class 27's Empty TC annex 890, and stay in: if a file places one it still
draws as a box and is still invisible in the game. Corpus placements (21 files,
measured 2026-09-26): REVEAL 837 459 across 8 files, FLARE4 697 230 (all in
R4_LeLoi_4), 1774 50, 1775 3, BLOCKER 1776 36, Invisible Object A 25 across
10 files, Blocker 3x1 21, Blocker 1x3 16, Empty TC annex 10.

Decisions made with the user: Map Revealers are in, and blockers move out of
eye candy (above). Owner-blind like the other const gates: Invisible Object A
exists to be player-owned ("keeps player alive").

## building_consts(): 375 consts (GH #120)

The basis is terrain_palette.BUILDING_TILE_SPANS (491 consts), the same rule
player_stats, unit_pick.footprint_entries() and render._unit_color already use
for "is a building", not BuildingInfo's 235. Three subtractions:

- wall_consts() (105, a strict subset of BUILDING_TILE_SPANS). Show Walls
  owns those.
- .dat classes 51 and 54, the mobile packed/unpacked siege (10 consts:
  TREBU 42/1690, PTREB 331/1691, PMANG 479, TOWN_CENTER_PACKED 444 and the
  campaign HBNB/HGOS 682/683/729/730; 32 corpus placements). The .dat types
  them as buildings, but a trebuchet is a unit to the person looking at the
  map, so it stays visible (decided with the user 2026-09-26).
- class 39, which after walls leaves only 1192 GTAC2: a hidden, graphic-less
  duplicate gate code gate_orientation drops on purpose. Subtracting the class
  keeps Show Buildings off everything gate-like; 1192 ends up under neither
  toggle, which is fine with no graphic and no corpus placements.

Measured 2026-09-26 after the .dat regeneration, by .dat class: 215 class 3
(ordinary buildings), 102 class 14, 30 class 27 scaffolding (Empty TC annex and the like), 11 towers
(class 52), 9 farms (class 49), 6 dead-building remnants (class 11), 2 class 1
(KOH-FLAG, RCHURCH). 5,191 corpus placements across the 21 examples/
files, from 0 (blank_map) to 1,113 (Dos Pilas).

46 consts overlap invisible_consts() (Empty TC annex 890 and others). That
overlap is allowed: UnitFilter ANDs its gates, so either toggle hides them,
and the walls / eye-candy / invisible disjointness is not extended to this
set. Nor is it one of player_stats' four partition buckets, so the partition
invariant is untouched.

Deliberately NOT GH #5's Buildings count (player_stats), which is raw
BUILDING_TILE_SPANS membership with walls as a sub-count: net of walls, the two
differ by exactly the 10 mobile-siege consts plus 1192. GH #143's "exclude
buildings" should reuse this set.

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
from descape.terrain_palette import BUILDING_TILE_SPANS, TREE_UNIT_IDS

_CATALOG_PATH = Path(__file__).resolve().parent / "object_catalog.json"
_GRAPHIC_MAP_PATH = Path(__file__).resolve().parent / "unit_graphic_map.json"

# object_catalog.json's `objects[<const>]["class"]` -- the .dat's unit.class_.
_WALL_CLASS = 27
_CLIFF_CLASS = 34
# Gate class; after wall_consts() only 1192 GTAC2 is left in it (building_consts()).
_GATE_CLASS = 39
# Packed/unpacked mobile siege (trebuchets, packed TC): .dat buildings that stay units here.
_MOBILE_SIEGE_CLASSES = frozenset({51, 54})

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
def building_consts() -> frozenset[int]:
    """Every const Show Buildings hides: BUILDING_TILE_SPANS minus walls and
    gates, mobile siege and the leftover gate class. See the module docstring."""
    walls = wall_consts()
    objects = _objects()
    return frozenset(
        const
        for const in BUILDING_TILE_SPANS
        if const not in walls
        and objects.get(const, {}).get("class") not in _MOBILE_SIEGE_CLASSES
        and objects.get(const, {}).get("class") != _GATE_CLASS
    )


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


@lru_cache(maxsize=1)
def obstacle_consts() -> frozenset[int]:
    """Every const Show Obstacles hides: eye candy whose .dat obstruction_type
    is non-zero (rocks, ruins, statues, mountains). See the module docstring."""
    objects = _objects()
    return frozenset(const for const in eye_candy_consts() if objects[const].get("obstruction", 0) != 0)


@lru_cache(maxsize=1)
def passable_eye_candy_consts() -> frozenset[int]:
    """Every const Show Eye Candy hides: eye candy units can walk through."""
    return eye_candy_consts() - obstacle_consts()
