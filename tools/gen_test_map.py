#!/usr/bin/env python3
"""Generates tests/fixtures/test_map_240x240.aoe2scenario, a 240x240 v1.59 map
holding everything the app can place or author, plus the edge cases most
likely to break a future change.

Built through the app's own edit models (UnitEditModel, TriggerEditModel,
OptionsEditModel, MessagesEditModel), its terrain and elevation tools and its
planners (cliff_chain, wall_run, terrain_units, beach_edges), then written by
scenario_write.write_scenario(). That is the opposite of the library-built
fixtures (tools/gen_units_fixture.py and friends), on purpose: those stay the
round-trip oracles, while cliff frames, wall indexes and gate siblings only
exist correctly through the app's tools. The donor is the shipped 120x120
template, resized by scenario_new.load_blank_scenario(). A test input, not a
byte oracle: regenerating it is expected whenever a serialization path changes.

The map is laid out in REGIONS, fixed tile rectangles with a one-line "why"
each, at least 2 tiles of plain grass apart. Every write is checked against
its region's rectangle, elevation ramps included, so the regions cannot leak
into each other. Content that is not on the map (players, diplomacy, map
options, messages, disables, triggers, odd values) sits in SECTIONS. Every
region and section is a build flag; build(flags) builds only those, which is
how the bisect set under build/test_map_bisect/ puts each risky one alone on a
blank 240x240 for the in-game check.

The output must not depend on the machine. main() hides every install
(AOE2DE_INSTALL_PATH points at a path that does not exist, settings are
isolated) and build() reads only committed tables: every catalog is iterated
by id, never in objects() display order, which sorts by install strings.
Two tool results still depend on a visible install, and are kept out:

- Cliff frame limits. cliff_catalog.frame_count() reads the real .sld frame
  count, which differs from the .dat fallback for most families (24 against
  25 for Default, Desert, Snow, Limestone and Terrace). The shape table caps
  a limit at 24, so only a limit under 24 changes a frame: Marble
  (2178-2186, 23 against 25) and the three Short families (2199-2225, 24
  against 23). Shaped strokes (runs, corners, loops, joins) use only the six
  families whose capped limit is the same either way, Short Marble 2190
  being 23 both ways; tests/test_test_map_fixture.py pins that. The other
  36 consts appear as one-tile strokes, a lone piece at the picker's frame,
  which no table lookup touches.
- Names. Labels come from install strings when one is visible; nothing here
  reads a display name, and cliff suffixes were measured equal both ways.

What the app's models cannot author, and what was done instead:

- An effect with an unknown type id (91, as in R4_LeLoi_4): the library's
  _add_effect() refuses an id outside EffectId, so it is added as a known
  type and its effect_type is then set raw.
- Every condition and effect field is set to a non-default value except the
  inert quantity-cluster slots (trigger_fields.apply_quantity_cluster_rule),
  which the property form keeps read-only for that entry's live slot.
- Long strings stop at 32000 bytes: a message of 32768 bytes or more passes
  MessagesEditModel (its limit is 65535) but writes a file that no longer
  loads (TASK-397).
- Facings and variants cover the non-hidden consts; hidden ones appear once,
  at their first facing or variant, in objects_hidden.
- Player names: tribe_name is a read-only row in the Players panel.
- Resources and pop limit "at the field maximum" use 2147483520, the largest
  s32 that is exact as f32: the model refuses 2**31 - 1 because the mirrored
  f32 copy would desync.
- All 8 players are active (decided 2026-10-17): Number of Players sets a
  prefix, so "P7 inactive" would have taken P8 with it.
- The custom AI for P7 is embedded from a script this generator writes into a
  temporary content root, so no install or profile is needed.
- A trigger version 4.4 section with no exec-order byte (F7_3_York) needs a
  byte-patched file, and game-written empty strings need a game save; neither
  is attempted here.

Regenerate with `.venv/bin/python3 tools/gen_test_map.py` and commit the
result; tests/test_test_map_fixture.py pins it.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from AoE2ScenarioParser.datasets.object_support import Civilization
from AoE2ScenarioParser.datasets.techs import TechInfo

from descape import (
    ai_scripts,
    asset_source,
    beach_edges,
    cliff_chain,
    content_roots,
    garrison,
    gate_orientation,
    library_compat,
    object_catalog,
    shape_tools,
    terrain_catalog,
    terrain_classes,
    terrain_palette,
    terrain_target,
    terrain_units,
    trigger_fields,
    trigger_organize,
    unit_kind,
    unit_rotation,
    unit_sprites,
    unit_variant,
    wall_run,
)
from descape.batch_api import set_layer
from descape.edit_history import EditHistory
from descape.elevation_tools import set_tile_elevation
from descape.messages_model import MessagesEditModel
from descape.options_model import OptionsEditModel
from descape.place_rotation import PlaceRotation, rotation_for
from descape.render import occupied_tiles_for, span_anchor, unit_tile_bounds
from descape.scenario_io import is_workshop_subscribed, load_map_and_units, parse_triggers
from descape.scenario_new import load_blank_scenario
from descape.scenario_write import WriteBlockedError, write_scenario
from descape.trigger_model import TriggerEditModel, move_entries
from descape.unit_model import UnitEditModel
from testkit.zlib_runtime import warn_if_not_stock

FIXTURE_PATH = ROOT / "tests" / "fixtures" / "test_map_240x240.aoe2scenario"
BISECT_DIR = ROOT / "build" / "test_map_bisect"
REGEN_COMMAND = ".venv/bin/python3 tools/gen_test_map.py"
MAP_SIZE = 240
SEED = 240
NO_INSTALL = "/nonexistent/descape-test-map-no-install"

GAIA = 0
ACTIVE_PLAYERS = (1, 2, 3, 4, 5, 6, 7, 8)
GRASS = 0
WATER = 23
DEEP_WATER = 22
LAYER_TERRAIN = 100  # GRASS_DRY
TOTAL_TRIGGERS = 600
# Below the largest corpus id (1,140,139); the game crashes on 2,000,000,000 (a SIZE_MAX allocation).
LARGE_REFERENCE_ID = 1_000_000
REFERENCE_GAP = 7

# Per-player set: Town Center, Castle, House, then Villager, Militia, Archer, Scout.
TOWN_CENTER, CASTLE, HOUSE = 109, 82, 70
VILLAGER, MILITIA, ARCHER, SCOUT = 83, 74, 4, 448
CROSSBOWMAN = 24
TRANSPORT_SHIP = 545
KILL_OBJECT = 14
STONE_WALL, PALISADE_WALL = 117, 72
FLARE4, HOUSE_RUBBLE, PLANTS, FLOWER, GRASS_TUFT = 697, 1434, 818, 1366, 1358
INVISIBLE_OBJECT, REVEALER = 1291, 837
MOUNTAINS = (310, 311, 744)
CASTLE_RUINS, ROCKB1, THEATRE = 1837, 1148, 2530
OAK = 349
MOVEMENT_SPEED = 5  # an ObjectAttribute stored as float
CUSTOM_AI_NAME = "TestMapAi.Ai"

# Shaped strokes only on families whose frame limit, capped at 24, is install-independent (see the docstring).
SHAPED_CLIFF_FAMILIES = (264, 1849, 1858, 2069, 2190, 2651)
E, W, S, N = (1, 0), (-1, 0), (0, 1), (0, -1)

MULTIBYTE = "Śląsk 漢字 ✓ Ωmega"
# 32000 bytes: a message of 32768+ bytes writes a file that no longer loads (TASK-397, see the docstring).
LONG_TEXT = "Long text near the field limit. " * 1000
LINE_BREAKS = ("cr\rbreak", "lf\nbreak", "crlf\r\nbreak")


@dataclass(frozen=True)
class Region:
    """A half-open tile rectangle [x0, x1) x [y0, y1) and why it exists."""

    x0: int
    y0: int
    x1: int
    y1: int
    why: str
    risky: bool = False
    elevated: bool = False

    def holds_tile(self, x: int, y: int) -> bool:
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1

    def holds_bounds(self, bounds: tuple[int, int, int, int]) -> bool:
        x0, x1, y0, y1 = bounds
        return self.x0 <= x0 and x1 <= self.x1 and self.y0 <= y0 and y1 <= self.y1

    def tiles(self):
        for y in range(self.y0, self.y1):
            for x in range(self.x0, self.x1):
                yield x, y


REGIONS: dict[str, Region] = {
    "corner_nw": Region(0, 0, 10, 10, "elevation 7 at the corner tile, a unit at x/y 0", elevated=True),
    "corner_ne": Region(230, 0, 240, 10, "elevation 7 at the corner tile, a unit at x 239.99", elevated=True),
    "corner_sw": Region(0, 230, 10, 240, "elevation 7 at the corner tile, a unit at y 239.99", elevated=True),
    "corner_se": Region(230, 230, 240, 240, "elevation 7 at the corner tile, a unit at x/y 239.99", elevated=True),
    "edge_n": Region(12, 0, 72, 10, "a building flush with the north edge, a wall run into it, units at y 0"),
    "edge_s": Region(12, 230, 72, 240, "water touching the south edge, a building flush with it, units at y 239.99"),
    "edge_w": Region(0, 12, 10, 72, "a cliff run into the west edge, a building flush with it, units at x 0"),
    "edge_e": Region(230, 12, 240, 72, "a wall run into the east edge, a building flush with it, units at x 239.99"),
    "objects_buildings": Region(12, 12, 72, 80, "every non-hidden Buildings catalog object, by id, owned by P1, on layered ground"),
    "objects_others": Region(12, 82, 72, 132, "every non-hidden Others catalog object, by id, on layered ground"),
    "objects_units": Region(12, 134, 72, 170, "every non-hidden Units catalog object, by id, ships on water, on layered ground"),
    "objects_heroes": Region(12, 172, 72, 194, "every non-hidden Heroes catalog object, by id, on layered ground"),
    "objects_hidden": Region(12, 196, 46, 228, "every editor-hidden catalog object, by id", risky=True),
    "objects_no_graphic": Region(48, 196, 62, 206, "the non-hidden objects with no graphic (invisible objects, revealers)", risky=True),
    "cameras": Region(64, 196, 72, 204, "five players' initial camera on one shared tile (York's five-on-one)"),
    "terrain_listed": Region(74, 12, 134, 62, "every editor-listed terrain as a 4x4 block in id order, forests auto-treed"),
    "mosaic": Region(74, 64, 134, 124, "a seeded 60x60 mosaic of 45 terrains with a layer on every other tile (blend load)"),
    "facings": Region(74, 126, 134, 164, "every non-hidden ANGLE const at its middle and last facing"),
    "variants": Region(74, 166, 134, 228, "every non-hidden cyclable const at its middle and last variant"),
    "elevation": Region(136, 12, 178, 52, "staircase 0..7, peak, pit, saddle, diagonal step, ridge, buildings/units/cliff on slopes",
                        elevated=True),
    "stacks": Region(136, 54, 178, 86, "about 160 stacked tiles of mixed kinds, three with 70-80 units, part on a slope",
                     elevated=True),
    "transitions": Region(136, 88, 178, 120, "checkerboard, 1-tile island and channel, every water by land, beach rings, farms"),
    "walls": Region(136, 122, 178, 178, "every wall const in every shape 0..4: x, y and diagonal runs, L, T and + junctions"),
    "gates": Region(136, 180, 168, 228, "all 24 visible gate consts, each placed over a wall run"),
    "cliffs": Region(180, 12, 228, 80, "every cliff const: runs both axes and facings, corners, a loop, a join, lone pieces"),
    "scenery": Region(180, 82, 228, 104, "mountains, CastleRuins, ROCKB1 and Theatre with units and trees inside, eye candy block"),
    "terrain_hidden": Region(180, 106, 212, 132, "the editor-hidden and .dat-only terrains as 4x4 blocks", risky=True),
    "trigger_targets": Region(214, 106, 228, 132, "units in several players that the vocabulary triggers name"),
    "players": Region(180, 134, 202, 166, "per active player: Town Center, Castle, House, Villager, Militia, Archer, Scout"),
    "garrisons": Region(204, 134, 228, 150, "a castle filled to capacity, a ship carrying units, occupants of a host elsewhere"),
    "field_spread": Region(204, 152, 228, 166, "106 distinct z heights, one unit with status 0, the largest variant index in use"),
    "placement": Region(180, 168, 208, 186, "stacks, off-centre points, building on a unit, shoreline, units on cliffs and walls"),
    "player_eye_candy": Region(210, 168, 228, 186, "P1-owned FLARE4 and rubble, other players' eye candy, invisible objects",
                               risky=True),
    "refs": Region(180, 188, 196, 200, "a reference_id gap and one large reference_id, added last", risky=True),
    "strings": Region(198, 188, 228, 200, "unit captions: multibyte, empty, long, and lone CR, LF and CRLF"),
}

SECTIONS: dict[str, str] = {
    "players": "8 slots with distinct base-game civs and colours, resources, pop limits, all 8 active, P7 with a custom AI",
    "diplomacy": "every stance value in an asymmetric matrix, allied victory on for some",
    "map_options": "a custom victory condition with its custom fields, the score sentinel and the other map options set",
    "messages": "every message slot set, '\\r\\r' instructions, string ids on hints and scouts",
    "disables": "some techs, units and buildings disabled per player",
    "trigger_vocab": "every 1.59 condition and effect with every field non-default, plus an unknown effect type",
    "trigger_structure": "leading run, both divider styles, tags, custom display and entry orders, exec order 1, 600 triggers",
    "odd_values": "resources at 0 and the maximum, pop limit bounds, GAIA starting age 4294967295, negative quantities",
}

# A flag needs these built first; only trigger_vocab's targets matter for the bisect set.
DEPENDS = {"trigger_vocab": ("trigger_targets",), "garrisons": ("players",)}

RISKY = tuple(sorted([*(name for name, r in REGIONS.items() if r.risky), "trigger_vocab", "odd_values"]))


# -- catalog reads, all by id -------------------------------------------------


def _catalog() -> list[object_catalog.CatalogEntry]:
    return sorted(object_catalog.objects(), key=lambda e: e.id)


def category_consts(category: str) -> list[int]:
    invisible = unit_kind.invisible_consts()
    return [e.id for e in _catalog() if e.category == category and not e.hidden and e.id not in invisible]


def hidden_consts() -> list[int]:
    return [e.id for e in _catalog() if e.hidden]


def no_graphic_consts() -> list[int]:
    invisible = unit_kind.invisible_consts()
    return [e.id for e in _catalog() if not e.hidden and e.id in invisible]


def angle_consts() -> list[int]:
    return [e.id for e in _catalog() if not e.hidden and unit_rotation.rotation_is_angle(e.id)]


def cyclable_consts() -> list[int]:
    return [e.id for e in _catalog() if not e.hidden and unit_variant.is_cyclable(e.id)]


def listed_terrains() -> list[int]:
    return [t.id for t in terrain_catalog.terrains() if not t.hidden and not terrain_palette.is_dat_only_terrain(t.id)]


def unlisted_terrains() -> list[int]:
    return [t.id for t in terrain_catalog.terrains() if t.hidden or terrain_palette.is_dat_only_terrain(t.id)]


def visible_gate_consts() -> list[int]:
    hidden = {e.id: e.hidden for e in _catalog()}
    return sorted(
        const
        for group in gate_orientation.groups().values()
        if all(hidden.get(c) is False for c in group)
        for const in group
    )


def middle_and_last(count: int) -> list[int]:
    return sorted({count // 2, count - 1} - {0}) or [count - 1]


def _ground(const: int) -> int:
    """The terrain a const's .dat restriction allows, preferring grass, then water."""
    restriction = object_catalog.dat_objects().get(const, {}).get("restriction")
    if restriction is None:
        return GRASS
    allowed = object_catalog.dat_terrain_restrictions().get(restriction, frozenset())
    for terrain in (GRASS, WATER, DEEP_WATER, 1, 4, 2):
        if terrain in allowed:
            return terrain
    return min(allowed) if allowed else GRASS


def _owner(const: int, category: str) -> int:
    return unit_kind.placement_owner(const, GAIA if category == "Others" else 1)


def _span(const: int) -> tuple[int, int]:
    return terrain_palette.tile_span(const, (1, 1))


# -- the build ----------------------------------------------------------------


class _Map:
    """One build: the blank scenario, its four edit models and what each region added."""

    def __init__(self, flags: frozenset[str]):
        self.flags = flags
        self.scenario = load_blank_scenario(MAP_SIZE)
        self.mm = self.scenario.map_manager
        self.history = EditHistory()
        self.units = UnitEditModel(self.scenario)
        self._triggers: TriggerEditModel | None = None
        self._options: OptionsEditModel | None = None
        self._messages: MessagesEditModel | None = None
        self._region: tuple[str, Region] | None = None
        # Reference ids and areas later builders name: placed units, the gap id, the trigger area.
        self.targets: dict[str, object] = {}

    @property
    def triggers(self) -> TriggerEditModel:
        if self._triggers is None:
            self._triggers = TriggerEditModel(self.scenario)
        return self._triggers

    @property
    def options(self) -> OptionsEditModel:
        if self._options is None:
            self._options = OptionsEditModel(self.scenario)
        return self._options

    @property
    def messages(self) -> MessagesEditModel:
        if self._messages is None:
            self._messages = MessagesEditModel(self.scenario)
        return self._messages

    # -- region-checked writes --

    @property
    def region(self) -> Region:
        assert self._region is not None, "a map write outside any region"
        return self._region[1]

    def tile(self, x: int, y: int):
        if not self.region.holds_tile(x, y):
            raise AssertionError(f"{self._region[0]}: tile ({x}, {y}) is outside its rectangle")
        return self.mm.terrain[y * MAP_SIZE + x]

    def paint(self, x: int, y: int, terrain_id: int) -> None:
        terrain_target.write_tile(self.tile(x, y), terrain_target.TERRAIN, terrain_id)

    def paint_rect(self, x0: int, y0: int, x1: int, y1: int, terrain_id: int) -> None:
        for y in range(y0, y1):
            for x in range(x0, x1):
                self.paint(x, y, terrain_id)

    def layer(self, x: int, y: int, terrain_id: int) -> None:
        set_layer(self.tile(x, y), terrain_id)

    def elevate(self, x: int, y: int, height: int) -> None:
        self.tile(x, y)
        for index in set_tile_elevation(self.mm, x, y, height):
            self.tile(index % MAP_SIZE, index // MAP_SIZE)

    def add(self, player: int, const: int, x: float, y: float, **fields):
        return self.units.add(player, const, x, y, **fields)

    def add_on(self, player: int, const: int, tx: int, ty: int, **fields):
        """add() with the const's footprint anchored on tile (tx, ty), as Place Unit snaps it."""
        return self.add(player, const, *span_anchor(tx, ty, *_span(const)), **fields)

    def owned(self, player: int, const: int, x: float, y: float, **fields):
        """add() as the placement owner, then reassign() as the Units panel's owner change does."""
        unit = self.add(unit_kind.placement_owner(const, player), const, x, y, **fields)
        self.units.reassign(unit, player)
        return unit

    @contextmanager
    def building(self, name: str):
        """One region: every write inside is checked against its rectangle, as one unit edit."""
        region = REGIONS[name]
        self._region = (name, region)
        first_id = self.units.next_unit_id
        self.units.begin_unit_edit(list(range(9)))
        try:
            with self.units.batch_adds():
                yield region
        finally:
            self.units.commit_unit_edit(f"Build {name}", self.history)
            self._region = None
        for unit in (u for units in self.scenario.unit_manager.units for u in units if u.reference_id >= first_id):
            bounds = unit_tile_bounds(unit, MAP_SIZE, MAP_SIZE)
            if bounds is None or not region.holds_bounds(bounds):
                raise AssertionError(f"{name}: unit {unit.unit_const} at ({unit.x}, {unit.y}) is outside its rectangle")

    @contextmanager
    def trigger_edit(self, label: str):
        """One recorded bracket over every trigger, the way a batch script edits."""
        model = self.triggers
        model.begin_trigger_edit(content_touched=range(model.trigger_count))
        try:
            yield model
        finally:
            model.commit_trigger_edit(label, self.history)

    def add_triggers(self, names: Iterable[str]) -> list[int]:
        names = list(names)
        model = self.triggers
        start = model.trigger_count
        model.begin_trigger_edit()
        model.structural_edit(lambda m: [m.add_trigger(name) for name in names])
        model.commit_trigger_edit(f"Add {len(names)} triggers", self.history)
        return list(range(start, start + len(names)))


def _pack(m: _Map, region: Region, items, *, gap: int = 1, row_gap: int = 1, layer: bool = False, cycle: bool = False) -> None:
    """Shelf-packs `items` ((const, owner, rotations), side by side per item) by ground, then by height.

    Each ground group gets its own rows, painted with that ground; land tiles take a layer when asked.
    Each unit is placed as Place Unit does; with `cycle` it is placed at variant 0 and then Cycle Variant
    writes its rotation."""
    groups: dict[int, list] = defaultdict(list)
    for const, owner, rotations in items:
        groups[_ground(const)].append((const, owner, rotations))
    y = region.y0
    painted: set[tuple[int, int]] = set()
    for ground in sorted(groups):
        group = sorted(groups[ground], key=lambda item: (-_span(item[0])[1], item[0]))
        x, row_top, row_h = region.x0, y, 0
        placements = []
        for const, owner, rotations in group:
            sx, sy = _span(const)
            width = sx * len(rotations)
            if x + width > region.x1:
                x, row_top, row_h = region.x0, row_top + row_h + row_gap, 0
            placements.append((const, owner, rotations, x, row_top))
            x += width + gap
            row_h = max(row_h, sy)
        bottom = row_top + row_h
        if bottom > region.y1:
            raise AssertionError(f"packing overflows its region by {bottom - region.y1} rows")
        if ground != GRASS:
            m.paint_rect(region.x0, y, region.x1, bottom, ground)
            painted.update((px, py) for py in range(y, bottom) for px in range(region.x0, region.x1))
        for const, owner, rotations, px, py in placements:
            sx, _sy = _span(const)
            for k, rotation in enumerate(rotations):
                if cycle:
                    m.units.set_variant(m.add_on(owner, const, px + k * sx, py, rotation=_place_rotation(const, owner)), rotation)
                else:
                    m.add_on(owner, const, px + k * sx, py, rotation=rotation)
        y = bottom + row_gap + 1
    if layer:
        for x, y in region.tiles():
            if (x, y) not in painted:
                m.layer(x, y, LAYER_TERRAIN)


def _place_rotation(const: int, owner: int, *, facing: int | None = None, variant: int | None = None) -> float:
    if facing is not None:
        count = unit_rotation.angle_count_for(const)
        return rotation_for(PlaceRotation(facing=unit_rotation.facing_to_rotation(facing, count)), const, owner)
    if variant is not None:
        return rotation_for(PlaceRotation(variant=variant, variant_const=const), const, owner)
    return rotation_for(PlaceRotation(), const, owner)


# -- completeness regions -----------------------------------------------------


def _objects(category: str):
    def build(m: _Map, region: Region) -> None:
        items = []
        for const in category_consts(category):
            owner = _owner(const, category)
            items.append((const, owner, (_place_rotation(const, owner),)))
        _pack(m, region, items, layer=True)

    return build


def _build_objects_hidden(m: _Map, region: Region) -> None:
    category = {e.id: e.category for e in _catalog()}
    items = []
    for const in hidden_consts():
        owner = _owner(const, category[const])
        items.append((const, owner, (_place_rotation(const, owner),)))
    _pack(m, region, items)


def _build_objects_no_graphic(m: _Map, region: Region) -> None:
    _pack(m, region, [(const, unit_kind.placement_owner(const, 1), (0.0,)) for const in no_graphic_consts()])


def _build_facings(m: _Map, region: Region) -> None:
    items = []
    for const in angle_consts():
        owner = unit_kind.placement_owner(const, 1)
        facings = middle_and_last(unit_rotation.angle_count_for(const))
        items.append((const, owner, tuple(_place_rotation(const, owner, facing=f) for f in facings)))
    _pack(m, region, items, row_gap=0)


def _build_variants(m: _Map, region: Region) -> None:
    items = []
    for const in cyclable_consts():
        owner = unit_kind.placement_owner(const, GAIA)
        variants = middle_and_last(unit_variant.variant_count_for(const))
        items.append((const, owner, tuple(_place_rotation(const, owner, variant=v) for v in variants)))
    _pack(m, region, items, cycle=True)


def _terrain_blocks(m: _Map, region: Region, terrain_ids: list[int], per_row: int) -> None:
    """4x4 blocks with a 1-tile gap, forests auto-treed through terrain_units with a fixed seed."""
    rng = random.Random(SEED)
    for k, terrain_id in enumerate(terrain_ids):
        bx, by = region.x0 + (k % per_row) * 5, region.y0 + (k // per_row) * 5
        changes, tiles_by_index = [], {}
        for y in range(by, by + 4):
            for x in range(bx, bx + 4):
                tile = m.tile(x, y)
                old = (tile.terrain_id, tile.elevation, tile.layer)
                m.paint(x, y, terrain_id)
                index = y * MAP_SIZE + x
                changes.append((index, old, (terrain_id, tile.elevation, tile.layer)))
                tiles_by_index[index] = (x, y)
        plan = terrain_units.plan_terrain_units(changes, tiles_by_index, {}, trees=True, doodads=True, rng=rng)
        if plan.adds:
            m.units.add_many(GAIA, plan.adds)


def _build_terrain_listed(m: _Map, region: Region) -> None:
    _terrain_blocks(m, region, listed_terrains(), 12)


def _build_terrain_hidden(m: _Map, region: Region) -> None:
    _terrain_blocks(m, region, unlisted_terrains(), 6)


def _build_mosaic(m: _Map, region: Region) -> None:
    rng = random.Random(SEED + 1)
    palette = rng.sample(listed_terrains(), 45)
    for x, y in region.tiles():
        m.paint(x, y, rng.choice(palette))
    for x, y in region.tiles():
        if (x + y) % 2 == 0:
            m.layer(x, y, rng.choice(palette))


def _stroke(m: _Map, const: int, press: tuple[int, int], steps, frame: int = 0, *, truncates: bool = False) -> list:
    """A Cliff tool stroke, committed as viewer._commit_cliff_stroke() does: joins rewritten, then nodes added.

    Every path node must be placed, and resolved unless it is a lone piece (which keeps the picker's
    piece, never resolved), so a stroke laid into another cannot silently lose a node; only `truncates`
    (a run drawn off the map edge on purpose) may come up short."""
    existing = cliff_chain.existing_cliffs(m.scenario, MAP_SIZE, MAP_SIZE)
    stroke = cliff_chain.ChainStroke(
        map_width=MAP_SIZE, map_height=MAP_SIZE, existing=existing, fallback_const=const, fallback_frame=frame
    )
    stroke.set_path(press, list(steps))
    nodes = stroke.nodes()
    if not truncates:
        expected = len(steps) + (stroke.anchor is None)
        short = [n.index for n in nodes if not n.resolved] if steps else []
        assert len(nodes) == expected and not short, (
            f"stroke {const} at {press}: {len(nodes)} of {expected} nodes placed, unresolved {short}"
        )
    # Rewrites first, as the viewer commits them: a raise there must precede every add.
    for rewrite in stroke.rewrites():
        m.units.set_cliff_frame(existing[rewrite.existing_index].unit, rewrite.rotation)
    for node in nodes:
        m.tile(*node.bounds[::2])
        m.tile(node.bounds[1] - 1, node.bounds[3] - 1)
        m.add(GAIA, node.unit_const, node.x, node.y, rotation=float(node.rotation), initial_animation_frame=node.rotation)
    return nodes


def _build_cliffs(m: _Map, region: Region) -> None:
    x0, y0 = region.x0, region.y0
    default, desert, snow, limestone, short_marble, terrace = SHAPED_CLIFF_FAMILIES
    # Straight runs both axes, each in both drag directions (the two facings, tall and thin).
    _stroke(m, default, (x0 + 4, y0 + 2), [E] * 4)
    _stroke(m, default, (x0 + 20, y0 + 6), [W] * 4)
    _stroke(m, terrace, (x0 + 2, y0 + 10), [S] * 4)
    _stroke(m, terrace, (x0 + 6, y0 + 26), [N] * 4)
    # Both corner directions, a closed loop, and a run continued off an existing end cap (a join).
    # The snow corner's north run stops 2 tiles short of the desert run's foot.
    _stroke(m, desert, (x0 + 12, y0 + 12), [E, E, E, S, S, S])
    _stroke(m, snow, (x0 + 12, y0 + 35), [E, E, E, N, N, N])
    # One drag closing on its own first node.
    _stroke(m, limestone, (x0 + 26, y0 + 12), [E, E, E, S, S, S, W, W, W, N, N])
    capped = _stroke(m, short_marble, (x0 + 26, y0 + 30), [E] * 3)
    head = capped[-1].bounds
    _stroke(m, short_marble, (head[0], head[2]), [E] * 2)
    # Every const no stroke produced, as a lone piece: a one-tile stroke at the picker's frame.
    placed = {u.unit_const for u in _units_in(m)}
    x, y, row_h = x0 + 1, y0 + 40, 0
    for const in sorted(unit_sprites.cliff_consts() - placed, key=lambda c: (-_span(c)[1], c)):
        sx, sy = _span(const)
        if x + sx > region.x1 - 1:
            x, y, row_h = x0 + 1, y + row_h + 1, 0
        _stroke(m, const, (x + sx // 2, y + sy // 2), [])
        x, row_h = x + sx + 1, max(row_h, sy)


def _units_in(m: _Map) -> list:
    region = m.region
    return [
        u for units in m.scenario.unit_manager.units for u in units if region.holds_tile(int(u.x), int(u.y))
    ]


def _wall_run(m: _Map, owner: int, const: int, tiles) -> None:
    existing_tiles, existing_walls = wall_run.wall_scene(m.scenario, MAP_SIZE, MAP_SIZE)
    for tile in tiles:
        m.tile(*tile)
    plan = wall_run.plan_wall_run(tiles, unit_const=const, existing_tiles=existing_tiles, existing_walls=existing_walls)
    wall_run.apply_wall_plan(m.units, owner, plan)


def _build_walls(m: _Map, region: Region) -> None:
    for row, const in enumerate(unit_sprites.wall_family_consts()):
        x, y = region.x0, region.y0 + row * 6

        def path(ax: int, ay: int, bx: int, by: int, ox: int, oy: int) -> list[tuple[int, int]]:
            return shape_tools.wall_path_tiles(ox + ax, oy + ay, ox + bx, oy + by, MAP_SIZE, MAP_SIZE)

        _wall_run(m, 1, const, path(0, 2, 4, 2, x, y))
        _wall_run(m, 1, const, path(0, 0, 0, 4, x + 6, y))
        _wall_run(m, 1, const, path(0, 0, 4, 4, x + 8, y))
        _wall_run(m, 1, const, path(4, 0, 0, 4, x + 14, y))
        _wall_run(m, 1, const, path(0, 0, 0, 3, x + 20, y) + path(1, 3, 3, 3, x + 20, y))
        _wall_run(m, 1, const, path(0, 0, 4, 0, x + 25, y) + path(2, 1, 2, 4, x + 25, y))
        _wall_run(m, 1, const, path(0, 2, 4, 2, x + 31, y) + path(2, 0, 2, 4, x + 31, y))
        # A lone piece: the tower shape, index 2.
        _wall_run(m, 1, const, [(x + 37, y + 2)])


def _build_gates(m: _Map, region: Region) -> None:
    gates = visible_gate_consts()
    by_group: dict[tuple, list[int]] = defaultdict(list)
    for key, group in gate_orientation.groups().items():
        if group[0] in gates:
            by_group[key] = list(group)
    for row, key in enumerate(sorted(by_group)):
        wall = PALISADE_WALL if key[0] == "P" else STONE_WALL
        for col, gate in enumerate(by_group[key]):
            cx, cy = region.x0 + col * 8 + 2, region.y0 + row * 8 + 2
            x, y = span_anchor(cx, cy, *_span(gate))
            footprint = occupied_tiles_for(gate, x, y, MAP_SIZE, MAP_SIZE)
            line, step = _gate_line(footprint)
            run = [(lx - step[0], ly - step[1]) for lx, ly in line[:1]] + line + [
                (line[-1][0] + step[0], line[-1][1] + step[1])
            ]
            _wall_run(m, 1, wall, run)
            tiles, walls = wall_run.wall_scene(m.scenario, MAP_SIZE, MAP_SIZE)
            plan = wall_run.plan_gate_over_walls(footprint, existing_tiles=tiles, existing_walls=walls)
            for tile in footprint:
                m.tile(*tile)
            wall_run.apply_gate_plan(m.units, 1, gate, x, y, plan)


def _gate_line(footprint) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    found = wall_run.gate_run_line(footprint)
    assert found is not None, f"no run line through gate footprint {footprint}"
    line, step = found
    return sorted(line, key=lambda t: (t[0] * step[0] + t[1] * step[1])), step


def _build_players(m: _Map, region: Region) -> None:
    for row, player in enumerate(ACTIVE_PLAYERS):
        y = region.y0 + row * 4
        m.add_on(player, TOWN_CENTER, region.x0, y)
        castle = m.add_on(player, CASTLE, region.x0 + 5, y)
        m.add_on(player, HOUSE, region.x0 + 10, y)
        for k, const in enumerate((VILLAGER, MILITIA, ARCHER, SCOUT)):
            m.add_on(player, const, region.x0 + 13 + (k % 2) * 2, y + (k // 2) * 2, rotation=_place_rotation(const, player))
        m.targets[f"castle:{player}"] = castle.reference_id


# -- bug-bait regions ---------------------------------------------------------


def _corner(dx: int, dy: int):
    def build(m: _Map, region: Region) -> None:
        cx = region.x0 if dx < 0 else region.x1 - 1
        cy = region.y0 if dy < 0 else region.y1 - 1
        m.elevate(cx, cy, 7)
        m.add(1, ARCHER, 0.0 if dx < 0 else 239.99, 0.0 if dy < 0 else 239.99)

    return build


def _build_edge_n(m: _Map, region: Region) -> None:
    m.add_on(1, HOUSE, region.x0 + 2, 0)
    _wall_run(m, 1, STONE_WALL, shape_tools.wall_path_tiles(region.x0 + 8, 8, region.x0 + 8, 0, MAP_SIZE, MAP_SIZE))
    for k in range(3):
        m.add(1, MILITIA, region.x0 + 12.5 + k, 0.0)


def _build_edge_s(m: _Map, region: Region) -> None:
    m.paint_rect(region.x0, region.y0 + 4, region.x0 + 20, region.y1, WATER)
    m.add_on(1, HOUSE, region.x0 + 24, region.y1 - 2)
    for k in range(3):
        m.add(1, MILITIA, region.x0 + 30.5 + k, 239.99)


def _build_edge_w(m: _Map, region: Region) -> None:
    # Pressed so the second piece lands on x 0: a run truncates at its first off-map node.
    _stroke(m, SHAPED_CLIFF_FAMILIES[0], (4, region.y0 + 4), [W] * 4, truncates=True)
    m.add_on(1, HOUSE, 0, region.y0 + 12)
    for k in range(3):
        m.add(1, MILITIA, 0.0, region.y0 + 18.5 + k)


def _build_edge_e(m: _Map, region: Region) -> None:
    _wall_run(m, 1, STONE_WALL, shape_tools.wall_path_tiles(232, region.y0 + 4, 239, region.y0 + 4, MAP_SIZE, MAP_SIZE))
    m.add_on(1, HOUSE, 238, region.y0 + 12)
    for k in range(3):
        m.add(1, MILITIA, 239.99, region.y0 + 18.5 + k)


def _build_elevation(m: _Map, region: Region) -> None:
    x0, y0 = region.x0, region.y0
    # Staircase 0..7 along x; its pyramid ramps 7 tiles out, all inside the rectangle.
    for k in range(8):
        m.elevate(x0 + 8 + k, y0 + 8, k)
    # Small shapes on flat ground, 6 apart: peak, pit in a plateau, saddle, diagonal-only step, ridge.
    by = y0 + 26
    m.elevate(x0 + 3, by, 1)
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            m.elevate(x0 + 10 + dx, by + dy, 1)
    m.elevate(x0 + 10, by, 0)
    # Saddle: 2 two tiles north and south, so the centre is 1 between them and above 0 east and west.
    for dy in (-2, 2):
        m.elevate(x0 + 18, by + dy, 2)
    m.elevate(x0 + 24, by, 1)
    m.elevate(x0 + 25, by + 1, 1)
    for k in range(5):
        m.elevate(x0 + 30 + k, by, 2)
    # A hill with a building across its slope, units on the slope and a cliff on top.
    hx, hy = x0 + 32, y0 + 10
    m.elevate(hx, hy, 4)
    m.add_on(1, HOUSE, hx - 3, hy + 1)
    for k in range(4):
        m.add(1, MILITIA, hx - 2.5 + k, hy - 2.5)
    _stroke(m, SHAPED_CLIFF_FAMILIES[0], (hx, hy), [])


def _build_transitions(m: _Map, region: Region) -> None:
    x0, y0 = region.x0, region.y0
    checker = (GRASS, 14, 3, 32, 11, 24)
    for dy in range(8):
        for dx in range(8):
            m.paint(x0 + dx, y0 + dy, checker[(dx + dy) % 2 + 2 * ((dx // 2 + dy // 2) % 3)])
    # A 1-tile island in water, and a 1-tile channel through land.
    m.paint_rect(x0 + 10, y0, x0 + 15, y0 + 5, WATER)
    m.paint(x0 + 12, y0 + 2, GRASS)
    for dy in range(6):
        m.paint(x0 + 17, y0 + dy, WATER)
    # Every water terrain in a row of 2x2 blocks touching each other, beside land.
    waters = sorted(terrain_classes.water_terrains())
    for k, water in enumerate(waters):
        m.paint_rect(x0 + 1 + (k % 20) * 2, y0 + 9 + (k // 20) * 3, x0 + 3 + (k % 20) * 2, y0 + 11 + (k // 20) * 3, water)
    # Beach rings: a straight shore and a concave (L-shaped) one, through apply_beach_ring().
    lake = [(x, y) for x in range(x0 + 4, x0 + 12) for y in range(y0 + 19, y0 + 23)]
    bay = [(x, y) for x in range(x0 + 18, x0 + 26) for y in range(y0 + 19, y0 + 29) if not (x >= x0 + 22 and y >= y0 + 23)]
    for core in (lake, bay):
        for x, y in core:
            m.paint(x, y, WATER)
        for index in beach_edges.apply_beach_ring(m.mm, core, WATER, None, 2):
            m.tile(index % MAP_SIZE, index // MAP_SIZE)
    # Farm terrain under and beside farms (Farm 50, 3x3).
    m.paint_rect(x0 + 30, y0 + 19, x0 + 40, y0 + 25, 7)
    m.add_on(1, 50, x0 + 30, y0 + 19)
    m.add_on(1, 50, x0 + 34, y0 + 19)


def _build_placement(m: _Map, region: Region) -> None:
    x0, y0 = region.x0, region.y0
    for _ in range(4):
        m.add(1, MILITIA, x0 + 1.5, y0 + 1.5)
    for k, (fx, fy) in enumerate(((0.13, 0.87), (0.99, 0.01), (0.5, 0.25))):
        m.add(1, ARCHER, x0 + 4 + k + fx, y0 + 1 + fy)
    m.add(1, VILLAGER, x0 + 9.5, y0 + 1.5)
    m.add_on(1, HOUSE, x0 + 9, y0 + 1)
    m.add_on(1, HOUSE, x0 + 13, y0 + 1)
    m.add_on(1, HOUSE, x0 + 14, y0 + 2)
    # Water on the right half: ships on it, land units on the shoreline tiles.
    m.paint_rect(x0 + 18, y0, region.x1, y0 + 10, WATER)
    for k in range(3):
        m.add_on(1, 539, x0 + 20 + k * 2, y0 + 3)
        m.add(1, MILITIA, x0 + 17.5, y0 + 2.5 + k)
    # Units standing on a cliff and on a wall run.
    _stroke(m, SHAPED_CLIFF_FAMILIES[0], (x0 + 3, y0 + 12), [])
    for k in range(3):
        m.add(1, ARCHER, x0 + 2.5 + k, y0 + 12.5)
    _wall_run(m, 1, STONE_WALL, shape_tools.wall_path_tiles(x0 + 8, y0 + 13, x0 + 14, y0 + 13, MAP_SIZE, MAP_SIZE))
    for k in range(3):
        m.add(1, MILITIA, x0 + 9.5 + k * 2, y0 + 13.5)


def _build_refs(m: _Map, region: Region) -> None:
    x0, y0 = region.x0, region.y0
    before = m.add(1, ARCHER, x0 + 1.5, y0 + 1.5)
    m.units._next_unit_id += REFERENCE_GAP
    m.targets["missing"] = before.reference_id + 1
    m.add(1, ARCHER, x0 + 3.5, y0 + 1.5)
    m.add(2, ARCHER, x0 + 5.5, y0 + 1.5)
    # Last of the whole build: every later add would get an id above it.
    m.units._next_unit_id = LARGE_REFERENCE_ID
    m.targets["large"] = m.add(3, ARCHER, x0 + 7.5, y0 + 1.5).reference_id


def _build_garrisons(m: _Map, region: Region) -> None:
    x0, y0 = region.x0, region.y0
    castle = m.add_on(1, CASTLE, x0 + 1, y0 + 1)
    occupants = [m.add(1, (MILITIA, ARCHER)[k % 2], castle.x, castle.y) for k in range(garrison.capacity(CASTLE))]
    _garrison(m, castle, occupants)
    m.paint_rect(x0 + 8, y0, x0 + 16, y0 + 8, WATER)
    ship = m.add_on(1, TRANSPORT_SHIP, x0 + 11, y0 + 3)
    _garrison(m, ship, [m.add(1, (VILLAGER, MILITIA)[k % 2], ship.x, ship.y) for k in range(5)])
    host = m.targets.get("castle:2")
    if host is not None:
        remote = [m.add(2, MILITIA, x0 + 18.5 + k, y0 + 2.5) for k in range(3)]
        _garrison(m, next(u for u in m.scenario.unit_manager.units[2] if u.reference_id == host), remote)


def _garrison(m: _Map, host, occupants) -> None:
    refusal = garrison.refusal(host.unit_const, str(host.unit_const), 0, [u.unit_const for u in occupants])
    assert refusal is None, refusal
    for unit in occupants:
        m.units.set_garrisoned_in(unit, host.reference_id)


def _build_strings(m: _Map, region: Region) -> None:
    captions = (MULTIBYTE, "", LONG_TEXT[:4000], *LINE_BREAKS)
    for k, caption in enumerate(captions):
        m.add(1, ARCHER, region.x0 + 1.5 + k * 2, region.y0 + 1.5, caption_string=caption)


def _build_trigger_targets(m: _Map, region: Region) -> None:
    for row, player in enumerate((GAIA, *ACTIVE_PLAYERS)):
        y = region.y0 + row * 3
        m.targets[f"unit:{player}"] = m.add(player, ARCHER, region.x0 + 1.5, y + 0.5).reference_id
        m.targets[f"house:{player}"] = m.add_on(player, HOUSE, region.x0 + 4, y).reference_id
    m.targets["area"] = (region.x0 + 7, region.y0 + 1, region.x1 - 2, region.y1 - 2)


# -- corpus-derived regions ---------------------------------------------------


def _build_stacks(m: _Map, region: Region) -> None:
    x0, y0 = region.x0, region.y0
    m.elevate(x0 + 30, y0 + 16, 3)
    kinds = (MILITIA, ARCHER, VILLAGER, SCOUT, CROSSBOWMAN)
    for k, count in enumerate((75, 78, 80)):
        for n in range(count):
            m.add((n % 3) + 1, kinds[n % len(kinds)], x0 + 2.5 + k * 3, y0 + 2.5)
    # Mixed small stacks on a 2-spaced grid: identical points, eye candy over units, near-identical points.
    stacked = 0
    for gy in range(y0 + 6, y0 + 30, 2):
        for gx in range(x0 + 1, x0 + 40, 2):
            kind = ((gx - x0) // 2 + (gy - y0) // 2) % 4
            cx, cy = gx + 0.5, gy + 0.5
            if kind == 0:
                m.add(1, MILITIA, cx, cy)
                m.add(2, MILITIA, cx, cy)
            elif kind == 1:
                m.add(GAIA, GRASS_TUFT, cx, cy)
                m.add(GAIA, FLOWER, cx, cy)
                m.add(1, ARCHER, cx, cy)
            elif kind == 2:
                m.add(1, VILLAGER, cx + 0.2, cy - 0.1)
                m.add(1, SCOUT, cx - 0.2, cy + 0.1)
            else:
                m.add(2, ARCHER, cx, cy)
                m.add(3, ARCHER, cx, cy)
            stacked += 1
            if stacked >= 150:
                break
        if stacked >= 150:
            break
    m.add_on(1, HOUSE, x0 + 12, y0 + 1)
    m.add(1, VILLAGER, x0 + 12.5, y0 + 1.5)
    m.add_on(GAIA, ROCKB1, x0 + 16, y0 + 1)
    for k in range(3):
        m.add(1, MILITIA, x0 + 16.5 + (k % 2), y0 + 1.5 + (k // 2))


def _build_scenery(m: _Map, region: Region) -> None:
    x0, y0 = region.x0, region.y0
    hosts = [(const, x0 + 1 + k * 8, y0 + 1) for k, const in enumerate(MOUNTAINS)]
    hosts += [(CASTLE_RUINS, x0 + 25, y0 + 1), (THEATRE, x0 + 31, y0 + 1), (ROCKB1, x0 + 38, y0 + 1)]
    for const, tx, ty in hosts:
        m.add_on(GAIA, const, tx, ty)
        m.add(1, MILITIA, tx + 0.5, ty + 0.5)
        m.add(2, ARCHER, tx + 1.5, ty + 0.5)
        m.units.set_variant(m.add(GAIA, OAK, tx + 0.5, ty + 1.5), 3.0)
    solid = sorted(c for c in unit_kind.obstacle_consts() if _span(c) == (1, 1) and c in set(category_consts("Others")))
    passable = sorted(c for c in unit_kind.passable_eye_candy_consts() if _span(c) == (1, 1) and c in set(category_consts("Others")))
    for k, const in enumerate(solid[:20] + passable[:20]):
        m.add_on(GAIA, const, x0 + 1 + k % 20, y0 + 9 + (k // 20) * 2)


def _build_player_eye_candy(m: _Map, region: Region) -> None:
    x0, y0 = region.x0, region.y0
    for k in range(24):
        m.owned(1, FLARE4, x0 + 1.5 + k % 6, y0 + 1.5 + k // 6)
    for k in range(4):
        m.owned(1, HOUSE_RUBBLE, *span_anchor(x0 + 8 + k * 2, y0 + 1, *_span(HOUSE_RUBBLE)))
    for k, (player, const) in enumerate(((2, PLANTS), (2, FLOWER), (3, PLANTS), (3, GRASS_TUFT))):
        m.owned(player, const, x0 + 1.5 + k * 2, y0 + 7.5)
    m.owned(1, INVISIBLE_OBJECT, x0 + 1.5, y0 + 10.5)
    m.owned(2, INVISIBLE_OBJECT, x0 + 3.5, y0 + 10.5)
    for k in range(3):
        m.owned(k + 1, REVEALER, x0 + 6.5 + k * 2, y0 + 10.5)


def _build_cameras(m: _Map, region: Region) -> None:
    tx, ty = (region.x0 + region.x1) // 2, (region.y0 + region.y1) // 2
    for player in range(1, 6):
        m.options.set_value(f"player:initial_view_x:{player}", tx)
        m.options.set_value(f"player:initial_view_y:{player}", ty)


def _build_field_spread(m: _Map, region: Region) -> None:
    for k in range(106):
        unit = m.add(1, MILITIA, region.x0 + 0.5 + (k % 12) * 2, region.y0 + 0.5 + k // 12, z=k * 0.25)
        if k == 0:
            m.units.set_status(unit, 0)
    widest = max(cyclable_consts(), key=lambda c: (unit_variant.variant_count_for(c), -c))
    tree = m.add_on(GAIA, widest, region.x1 - 2, region.y1 - 2, rotation=_place_rotation(widest, GAIA))
    m.units.set_variant(tree, float(unit_variant.variant_count_for(widest) - 1))


# -- sections -----------------------------------------------------------------


_CIVS = ("BRITONS", "FRANKS", "GOTHS", "TEUTONS", "JAPANESE", "CHINESE", "BYZANTINES", "PERSIANS")


def _build_players_section(m: _Map) -> None:
    o = m.options
    o.set_value("player_count", len(ACTIVE_PLAYERS))
    colours = (3, 5, 0, 7, 1, 6, 2, 4)
    for player in range(1, 9):
        civ = Civilization[_CIVS[player - 1]].value
        o.set_value(f"player:civilization:{player}", civ)
        o.set_value(f"player:architecture:{player}", civ)
        o.set_value(f"player:color:{player}", colours[player - 1])
        for k, resource in enumerate(("food", "wood", "stone", "gold")):
            o.set_value(f"player:{resource}:{player}", 100 * player + 10 * k + 5)
        o.set_value(f"player:pop_limit:{player}", 50 + 25 * player)
        o.set_value(f"player:starting_age:{player}", 2 + player % 5)
        o.set_value(f"player:base_priority:{player}", player)
        o.set_value(f"player:lock_civilization:{player}", player % 2)
    o.set_value("player:human:2", 0)
    o.set_value("player:lock_personality:7", 1)
    with tempfile.TemporaryDirectory(prefix="descape-test-map-ai-") as tmp:
        ai_dir = Path(tmp) / "resources" / "_common" / "ai"
        ai_dir.mkdir(parents=True)
        (ai_dir / CUSTOM_AI_NAME).write_bytes(b"")
        (ai_dir / "TestMapAi.per").write_bytes(b'(load "TestMapLib")\r\n(defrule\r\n\t(true)\r\n=>\r\n\t(disable-self))\r\n')
        (ai_dir / "TestMapLib.per").write_bytes(b"(defconst test-map-lib 1)\r\n")
        root = content_roots.ContentRoot(content_roots.KIND_PROFILE, "Test map", Path(tmp), profile_id="0")
        choice = next(c for c in ai_scripts.scan_ai_choices([root]) if c.stored_name == CUSTOM_AI_NAME)
        o.set_personality(7, ai_scripts.resolve(choice, [root]))
    o.set_personality(3, ai_scripts.NONE)


def _build_diplomacy(m: _Map) -> None:
    stances = (0, 1, 3)
    for row in range(1, 9):
        for col in range(1, 9):
            if row != col:
                m.options.set_value(f"stance:{row}:{col}", stances[(row * 3 + col) % 3])
    for player in (1, 3, 4):
        m.options.set_value(f"allied_victory:{player}", 1)


def _build_map_options(m: _Map) -> None:
    # victory_score holds the 0xFFFFFFFF "unset" sentinel the 2_Joan_coop files carry, shown as 14000.
    # script_name's file is tests/fixtures/test_map_script.xs; the game errors at load without it.
    for field_id, value in (
        ("victory_condition", 4), ("custom_conquest", 1), ("custom_explored_percent", 60), ("custom_relics", 3),
        ("custom_all_conditions", 1), ("victory_score", 0xFFFFFFFF), ("victory_years", 9000), ("lock_teams", 1),
        ("allow_players_choose_teams", 0), ("random_start_points", 1), ("max_number_of_teams", 4),
        ("collide_and_correct", 1), ("villager_force_drop", 1), ("lock_coop_alliances", 1), ("secondary_game_modes", 2),
        ("no_waves_on_shore", 1), ("all_techs", 1), ("ai_map_type", 9), ("script_name", "test_map_script"),
    ):
        m.options.set_value(field_id, value)


def _build_messages(m: _Map) -> None:
    for field_id, text in (
        ("instructions", "\r\r"), ("hints", MULTIBYTE), ("victory", LINE_BREAKS[2]), ("loss", LINE_BREAKS[1]),
        ("history", LONG_TEXT), ("scouts", LINE_BREAKS[0]),
    ):
        m.messages.set_value(field_id, text)
    m.messages.set_value("hints_id", 21000)
    m.messages.set_value("scouts_id", 21001)


def _build_disables(m: _Map) -> None:
    for player in range(1, 9):
        m.options.set_value(f"disabled:techs:{player}", (TechInfo.LOOM.ID, TechInfo.WHEELBARROW.ID)[: 1 + player % 2])
        m.options.set_value(f"disabled:units:{player}", (ARCHER, MILITIA, SCOUT)[: 1 + player % 3])
        m.options.set_value(f"disabled:buildings:{player}", (12, 87)[: 1 + (player + 1) % 2])


def _build_odd_values(m: _Map) -> None:
    o = m.options
    for resource in ("food", "wood", "stone", "gold"):
        o.set_value(f"player:{resource}:4", 0)
        o.set_value(f"player:{resource}:5", 2147483520)
    o.set_value("player:pop_limit:4", 0)
    o.set_value("player:pop_limit:5", 2147483520)
    o.set_value("player:starting_age:0", 4294967295)
    if "trigger_vocab" not in m.flags and "trigger_structure" not in m.flags:
        return
    (index,) = m.add_triggers(["Odd: negative and split quantities"])
    vocab = library_compat.load_vocabulary("1.59")
    with m.trigger_edit("Odd quantities") as model:
        trigger = model.manager().triggers[index]
        # An int quantity (Damage Object) beside a float one (Modify Attribute on Movement Speed).
        for type_id, values in ((24, {"quantity": -5}), (51, {"object_attributes": MOVEMENT_SPEED, "quantity": -0.375})):
            trigger._add_effect(type_id)
            effect = trigger.effects[-1]
            for name, value in values.items():
                assert name in vocab.effects[type_id].attributes, (type_id, name)
                setattr(effect, name, value)
        trigger._add_condition(10)
        trigger.conditions[-1].quantity = -3


# -- triggers -----------------------------------------------------------------


_VARIABLES = ("tm_counter", "tm_ünïcode", "tm_flag", "tm_last")


def _area(m: _Map) -> tuple[int, int, int, int]:
    return m.targets.get("area", (1, 1, 9, 9))


def _unit_refs(m: _Map) -> list[int]:
    refs = [m.targets[key] for key in sorted(m.targets) if key.startswith("unit:")]
    return refs or [0]


def _field_value(m: _Map, spec: trigger_fields.FieldSpec, default, k: int, trigger_ids: list[int]):
    """A value the spec allows that differs from the vocabulary default, real references where it names one."""
    name, kind, presentation = spec.name, spec.kind, spec.presentation
    ax1, ay1, ax2, ay2 = _area(m)
    coords = {"area_x1": ax1, "area_y1": ay1, "area_x2": ax2, "area_y2": ay2, "location_x": ax1 + 2, "location_y": ay1 + 3,
              "wall_x1": ax1, "wall_y1": ay1, "wall_x2": ax1 + 4, "wall_y2": ay1}
    refs = _unit_refs(m)
    if name in coords:
        return coords[name]
    if kind == trigger_fields.BOOL:
        return 0 if default == 1 else 1
    if kind == trigger_fields.ENUM:
        values = [value for _label, value in spec.choices if value != default and value != trigger_fields.UNSET]
        return values[k % len(values)]
    if kind == trigger_fields.REFERENCE:
        if presentation == "Unit":
            return refs[k % len(refs)]
        if presentation == "Unit[]":
            return refs[: 2 + k % 3]
        if presentation == "TriggerId":
            return trigger_ids[k % len(trigger_ids)]
        if presentation == "VariableId":
            ids = [i for i in range(len(_VARIABLES)) if i != default]
            return ids[k % len(ids)]
        if presentation == "TechInfo":
            return (TechInfo.LOOM.ID, TechInfo.BALLISTICS.ID, TechInfo.WHEELBARROW.ID)[k % 3]
        if presentation == "BuildingInfo":
            return HOUSE
        return (ARCHER, MILITIA, VILLAGER)[k % 3]
    if kind == trigger_fields.INT_LIST:
        if presentation == "Unit[]":
            return refs[: 2 + k % 3]
        return [1, 2] if presentation else [3, 5, 8]
    if kind == trigger_fields.STR:
        if spec.multiline == trigger_fields.XS:
            # The game concatenates every Script Call's text into default0.xs, so names must be unique.
            if name == "xs_function":
                return f"bool testMap{k}() {{\r   return (false);\r}}"
            return f"int tm{k} = 1;\rvoid testMap{k}() {{\r}}"
        texts = (MULTIBYTE, *LINE_BREAKS, "plain text")
        return texts[k % len(texts)]
    if kind == trigger_fields.FLOAT:
        return 2.5
    return 3 + k % 7 if default != 3 + k % 7 else 11


def _author(m: _Map, entry, definition, presentation_map, type_attribute: str, k: int, trigger_ids: list[int]) -> None:
    """Every field of one condition or effect set, one at a time, as the property form writes them."""
    specs = trigger_fields.field_specs(definition, presentation_map, type_attribute)
    is_effect = type_attribute == "effect_type"
    for index, spec in enumerate(specs):
        live = trigger_fields.apply_quantity_cluster_rule(specs, entry) if is_effect else specs
        if live[index].read_only:
            continue
        before = trigger_fields.live_quantity_slot(entry) if is_effect else ""
        setattr(entry, spec.attribute, _field_value(m, spec, definition.default_attributes.get(spec.name), k + index, trigger_ids))
        if is_effect:
            for name, value in trigger_fields.cluster_fixup(
                entry, before, trigger_fields.live_quantity_slot(entry), definition.attributes
            ):
                setattr(entry, name, value)


def _build_trigger_vocab(m: _Map) -> None:
    vocab = library_compat.load_vocabulary("1.59")
    names = [f"Vocab C{type_id}: {d.name}" for type_id, d in sorted(vocab.conditions.items())]
    names += [f"Vocab E{type_id}: {d.name}" for type_id, d in sorted(vocab.effects.items())]
    names += ["Vocab E91: unknown effect type", "Names a deleted unit"]
    indices = m.add_triggers(names)
    model = m.triggers
    model.begin_trigger_edit()
    model.structural_edit(lambda mgr: [mgr.add_variable(name) for name in _VARIABLES])
    model.commit_trigger_edit("Add variables", m.history)
    with m.trigger_edit("Author the vocabulary") as model:
        triggers = model.manager().triggers
        trigger_ids = [triggers[i].trigger_id for i in indices[:3]]
        entries = [("condition", t, vocab.conditions[t]) for t in sorted(vocab.conditions)]
        entries += [("effect", t, vocab.effects[t]) for t in sorted(vocab.effects)]
        for k, (kind, type_id, definition) in enumerate(entries):
            trigger = triggers[indices[k]]
            if kind == "condition":
                trigger._add_condition(type_id)
                _author(m, trigger.conditions[-1], definition, vocab.condition_presentation, "condition_type", k, trigger_ids)
            else:
                trigger._add_effect(type_id)
                _author(m, trigger.effects[-1], definition, vocab.effect_presentation, "effect_type", k, trigger_ids)
            for spec in trigger_fields.trigger_specs("1.59"):
                if spec.kind == trigger_fields.BOOL and spec.name != "enabled":
                    setattr(trigger, spec.attribute, k % 2)
            trigger.short_description = MULTIBYTE
            trigger.description = LINE_BREAKS[k % 3]
            trigger.description_order = k
            trigger.short_description_stid = 30000 + k
            trigger.description_stid = 31000 + k
        unknown = triggers[indices[-2]]
        unknown._add_effect(1)
        unknown.effects[-1].effect_type = 91
        # The refs region's gap id (a unit deleted in the editor) beside live units of two players.
        dangling = triggers[indices[-1]]
        dangling._add_effect(KILL_OBJECT)
        missing = m.targets.get("missing", m.units.next_unit_id + 1)
        dangling.effects[-1].selected_object_ids = [missing, *_unit_refs(m)[1:3]]


def _build_trigger_structure(m: _Map) -> None:
    model = m.triggers
    vocab_count = model.trigger_count
    leading = m.add_triggers(["Intro timer", "[intro] Tagged start", "(intro) Second tag", "<intro> Third tag"])
    spaced = trigger_organize.DividerFormat()
    tight = trigger_organize.DividerFormat(spaced=False)
    dividers = m.add_triggers(
        [trigger_organize.format_divider("Vocabulary", tight), trigger_organize.format_divider("Structure", spaced),
         trigger_organize.format_divider("Padding", spaced)]
    )
    body = m.add_triggers(
        ["Disabled trigger", "Looping trigger", "OR short-circuit", "Objective header", "Disabled objective",
         "Whole-map area", "Corner tile area", "", LONG_TEXT[:2000], MULTIBYTE, *LINE_BREAKS]
    )
    padding = m.add_triggers(f"Timer {k:03d}" for k in range(max(0, TOTAL_TRIGGERS - model.trigger_count)))
    with m.trigger_edit("Structure") as model:
        manager = model.manager()
        triggers = manager.triggers
        disabled, looping, short_circuit, header, disabled_header, whole, corner = (triggers[i] for i in body[:7])
        disabled.enabled = 0
        looping.looping = 1
        short_circuit._add_condition(10)
        short_circuit._add_condition(vocab_or_condition())
        short_circuit._add_condition(10)
        for k in range(3):
            short_circuit._add_effect(26)
            short_circuit.effects[-1].message = f"entry {k}"
        # Non-identity condition and effect orders, moved as the panel's Move buttons do; OR stays in the middle.
        move_entries(short_circuit, [("condition", 2)], -1, to_edge=True)
        move_entries(short_circuit, [("condition", 0)], 1)
        move_entries(short_circuit, [("effect", 2)], -1)
        for objective in (header, disabled_header):
            objective.display_as_objective = 1
            objective.header = 1
            objective.short_description = "Objectives"
        disabled_header.enabled = 0
        for trigger, area in ((whole, (0, 0, MAP_SIZE - 1, MAP_SIZE - 1)), (corner, (MAP_SIZE - 1,) * 4)):
            trigger._add_effect(2)
            effect = trigger.effects[-1]
            effect.area_x1, effect.area_y1, effect.area_x2, effect.area_y2 = area
        for k, i in enumerate(leading + padding):
            triggers[i]._add_condition(10)
            triggers[i].conditions[-1].timer = 1 + k % 60
            triggers[i]._add_effect(26)
            triggers[i].effects[-1].message = f"tick {k}"
        vocab = list(range(vocab_count))
        order = [*leading, dividers[0], *vocab, dividers[1], *body, dividers[2], *reversed(padding)]
        rest = [i for i in range(len(triggers)) if i not in set(order)]
        manager.trigger_display_order = order + rest
        model.set_exec_order(1)


def vocab_or_condition() -> int:
    vocab = library_compat.load_vocabulary("1.59")
    return next(type_id for type_id, d in vocab.conditions.items() if d.name == "or")


# -- assembly -----------------------------------------------------------------


REGION_BUILDERS = {
    "corner_nw": _corner(-1, -1),
    "corner_ne": _corner(1, -1),
    "corner_sw": _corner(-1, 1),
    "corner_se": _corner(1, 1),
    "edge_n": _build_edge_n,
    "edge_s": _build_edge_s,
    "edge_w": _build_edge_w,
    "edge_e": _build_edge_e,
    "objects_buildings": _objects("Buildings"),
    "objects_others": _objects("Others"),
    "objects_units": _objects("Units"),
    "objects_heroes": _objects("Heroes"),
    "objects_hidden": _build_objects_hidden,
    "objects_no_graphic": _build_objects_no_graphic,
    "cameras": _build_cameras,
    "terrain_listed": _build_terrain_listed,
    "mosaic": _build_mosaic,
    "facings": _build_facings,
    "variants": _build_variants,
    "elevation": _build_elevation,
    "stacks": _build_stacks,
    "transitions": _build_transitions,
    "walls": _build_walls,
    "gates": _build_gates,
    "cliffs": _build_cliffs,
    "scenery": _build_scenery,
    "terrain_hidden": _build_terrain_hidden,
    "trigger_targets": _build_trigger_targets,
    "players": _build_players,
    "garrisons": _build_garrisons,
    "field_spread": _build_field_spread,
    "placement": _build_placement,
    "player_eye_candy": _build_player_eye_candy,
    "strings": _build_strings,
    "refs": _build_refs,
}

SECTION_BUILDERS = {
    "players": _build_players_section,
    "diplomacy": _build_diplomacy,
    "map_options": _build_map_options,
    "messages": _build_messages,
    "disables": _build_disables,
    "odd_values": _build_odd_values,
    "trigger_vocab": _build_trigger_vocab,
    "trigger_structure": _build_trigger_structure,
}

# "players" names both a region and a section: one flag builds both.
ALL_FLAGS = frozenset(REGIONS) | frozenset(SECTIONS)


def _resolve_flags(flags: Iterable[str] | None) -> frozenset[str]:
    selected = set(ALL_FLAGS if flags is None else flags)
    unknown = selected - ALL_FLAGS
    if unknown:
        raise ValueError(f"unknown build flags: {sorted(unknown)}")
    pending = list(selected)
    while pending:
        for dependency in DEPENDS.get(pending.pop(), ()):
            if dependency not in selected:
                selected.add(dependency)
                pending.append(dependency)
    return frozenset(selected)


def check_layout() -> None:
    """Every region on the map, and no two closer than 2 tiles on both axes."""
    regions = list(REGIONS.items())
    for i, (a_name, a) in enumerate(regions):
        if not (0 <= a.x0 < a.x1 <= MAP_SIZE and 0 <= a.y0 < a.y1 <= MAP_SIZE):
            raise AssertionError(f"{a_name} is not on the map")
        for b_name, b in regions[i + 1 :]:
            if not (a.x1 + 2 <= b.x0 or b.x1 + 2 <= a.x0 or a.y1 + 2 <= b.y0 or b.y1 + 2 <= a.y0):
                raise AssertionError(f"{a_name} and {b_name} are closer than 2 tiles")


def build_map(flags: Iterable[str] | None = None) -> _Map:
    """Every selected region and section applied to a blank 240x240, nothing written yet."""
    check_layout()
    m = _Map(_resolve_flags(flags))
    for name, build in REGION_BUILDERS.items():
        if name in m.flags:
            with m.building(name) as region:
                build(m, region)
    for name, build in SECTION_BUILDERS.items():
        if name in m.flags:
            build(m)
    return m


def _serialize(m: _Map) -> bytes:
    """A built map's bytes, through write_scenario() exactly as a Save does."""
    with tempfile.TemporaryDirectory(prefix="descape-test-map-") as tmp:
        out = Path(tmp) / "test_map.aoe2scenario"
        write_scenario(
            m.scenario, out, triggers=m._triggers, options=m._options, units=m.units, messages=m._messages, backup=False
        )
        return out.read_bytes()


def build(flags: Iterable[str] | None = None) -> bytes:
    """The fixture's bytes."""
    return _serialize(build_map(flags))


def region_counts(unit_manager, flags: Iterable[str]) -> dict[str, int]:
    """Units per selected region, by the tile each unit's point is on."""
    counts = {name: 0 for name in REGIONS if name in flags}
    for unit in (u for player_units in unit_manager.units for u in player_units):
        for name in counts:
            if REGIONS[name].holds_tile(int(unit.x), int(unit.y)):
                counts[name] += 1
    return counts


def verify(path: Path, flags: Iterable[str] | None = None, expected: dict[str, int] | None = None) -> dict[str, int]:
    """Reloads `path` through the real loader and checks what every selected flag placed; returns units per region.

    `expected` is the built map's own region_counts(): every region must read back exactly that many."""
    flags = _resolve_flags(flags)
    loaded = load_map_and_units(path)
    if not (loaded.units_write_supported and loaded.terrain_write_supported):
        raise AssertionError(f"{path}: a write gate failed after reload")
    units = [u for player_units in loaded.unit_manager.units for u in player_units]
    counts = region_counts(loaded.unit_manager, flags)
    empty = [name for name, count in counts.items() if count == 0 and name not in ("cameras", "mosaic")]
    if empty:
        raise AssertionError(f"{path}: no units read back in {empty}")
    if expected is not None and counts != expected:
        drift = {name: (expected.get(name), count) for name, count in counts.items() if expected.get(name) != count}
        raise AssertionError(f"{path}: units per region (built, read back) differ: {drift}")
    consts = {u.unit_const for u in units}
    expected: set[int] = set()
    for name, consts_of in (
        ("objects_buildings", lambda: category_consts("Buildings")), ("objects_others", lambda: category_consts("Others")),
        ("objects_units", lambda: category_consts("Units")), ("objects_heroes", lambda: category_consts("Heroes")),
        ("objects_hidden", hidden_consts), ("objects_no_graphic", no_graphic_consts), ("gates", visible_gate_consts),
        ("cliffs", lambda: sorted(unit_sprites.cliff_consts())), ("walls", unit_sprites.wall_family_consts),
    ):
        if name in flags:
            expected |= set(consts_of())
    missing = sorted(expected - consts)
    if missing:
        raise AssertionError(f"{path}: consts missing after reload: {missing}")
    if "walls" in flags:
        shapes = {(u.unit_const, int(u.rotation)) for u in units if REGIONS["walls"].holds_tile(int(u.x), int(u.y))}
        lacking = [(c, i) for c in unit_sprites.wall_family_consts() for i in range(5) if (c, i) not in shapes]
        if lacking:
            raise AssertionError(f"{path}: wall (const, index) pairs missing: {lacking}")
    manager = parse_triggers(loaded)
    if "trigger_structure" in flags and len(manager.triggers) != TOTAL_TRIGGERS:
        raise AssertionError(f"{path}: {len(manager.triggers)} triggers, expected {TOTAL_TRIGGERS}")
    if "trigger_vocab" in flags:
        vocab = library_compat.load_vocabulary("1.59")
        conditions = {c.condition_type for t in manager.triggers for c in t.conditions}
        effects = {e.effect_type for t in manager.triggers for e in t.effects}
        lacking = sorted(set(vocab.conditions) - conditions) + sorted(set(vocab.effects) - effects)
        if lacking:
            raise AssertionError(f"{path}: vocabulary entries missing: {lacking}")
    return counts


def _hide_install(config_dir: Path) -> None:
    """main()'s isolation: no config, no install, so the output cannot depend on this machine."""
    from testkit.settings_isolation import isolate_settings

    isolate_settings(config_dir)
    os.environ["AOE2DE_INSTALL_PATH"] = NO_INSTALL
    asset_source.clear_install_caches()


def _write_and_verify(path: Path, flags: Iterable[str] | None = None) -> tuple[bytes, dict[str, int]]:
    m = build_map(flags)
    expected = region_counts(m.scenario.unit_manager, m.flags)
    data = _serialize(m)
    _write(path, data)
    return data, verify(path, flags, expected)


def _write(path: Path, data: bytes) -> None:
    if is_workshop_subscribed(path):
        raise WriteBlockedError(f"Refusing to write into a subscribed-mods (mods/subscribed/) folder: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=FIXTURE_PATH, help="Destination file")
    parser.add_argument("--no-bisect", action="store_true", help="Skip the build/test_map_bisect/ set")
    args = parser.parse_args()
    # Refused before the build, so a refused run touches neither the settings nor the environment.
    if is_workshop_subscribed(args.out):
        raise WriteBlockedError(f"Refusing to write into a subscribed-mods (mods/subscribed/) folder: {args.out}")
    with tempfile.TemporaryDirectory(prefix="descape-test-map-config-") as config_dir:
        _hide_install(Path(config_dir))
        data, counts = _write_and_verify(args.out)
        warn_if_not_stock(str(args.out))
        print(f"{args.out}: {len(data)} bytes, {sum(counts.values())} units in {len(counts)} regions")
        if args.no_bisect:
            return
        for flag in RISKY:
            _write_and_verify(BISECT_DIR / f"{flag}.aoe2scenario", {flag})
        _write_and_verify(BISECT_DIR / "all_safe.aoe2scenario", ALL_FLAGS - set(RISKY))
        print(f"{BISECT_DIR}: {len(RISKY)} risky flags alone, plus all_safe.aoe2scenario")


if __name__ == "__main__":
    main()
