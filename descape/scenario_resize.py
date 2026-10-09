"""Resize an existing map (GH #45): grow or shrink its tile dimensions from
any of nine anchors, carrying terrain, elevation, units, trigger coordinates
and cameras onto the new origin. Pure, no Qt, the same relationship to
descape/scenario_write.py that descape/scenario_new.py has.

Square-only for now. A W x H map does not load through the pinned library
yet, and whether AoE2:DE accepts one at all waits on an in-game check, so any
request that would produce a non-square map, or start from one, is refused.
Every anchor is still available for a square-to-square resize.

The pipeline, and why each step sits where it does:

1. scenario_write.build_patched() builds the document's *current* body,
   every pending terrain/option/unit/trigger/message edit included, so a
   resize never silently drops unsaved work.
2. resize_body() splices that body: camera coordinates are clamped in place
   (fixed width, so nothing shifts), then the terrain block is replaced by a
   remapped one at the new size. The terrain remap copies whole 7-byte
   structs, so terrain, elevation, layer and the unused bytes all carry
   verbatim. A newly created tile is never zero-filled: it takes terrain 0
   and layer -1 with the elevation of the nearest old edge tile, since an
   elevation jump between neighbours crashes the game on load, and the old
   map already satisfied the adjacent-step rule along its edge.
3. The result is reloaded through the real loader (offsets are never
   carried across a length-changing splice), and only then are units and
   trigger coordinates remapped, through edit models built on the reloaded
   document, so they write at fresh offsets.

Units translate by (dx, dy) through UnitEditModel.set_position(), which
leaves `z` and `rotation` verbatim. A unit whose new tile is off the map is
deleted, as is one whose on-map footprint the cut leaves partly off it (a
4x4 building straddling the new edge), and so is everything garrisoned in
it (the garrison closure), which is the count the confirm dialog shows.
Trigger areas, locations and wall runs translate and clamp through
descape/trigger_geometry.py's own field groups, written back through
TriggerShape.fields, only when the file's Triggers section is writable;
otherwise they are left as they were and the plan says so.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from descape import library_compat, player_fields, render, trigger_geometry
from descape.messages_model import MessagesEditModel
from descape.options_model import OptionsEditModel
from descape.scenario_io import (
    TERRAIN_STRUCT_SIZE,
    LoadedScenario,
    load_map_and_units_from_bytes,
    parse_triggers,
)
from descape.scenario_new import (
    BLANK_TERRAIN_STRUCT,
    MAX_MAP_TILES,
    MIN_MAP_TILES,
    MapSizeError,
    validate_map_size,
)
from descape.scenario_write import PatchedBody, WriteBlockedError, _compress_bytes, build_patched
from descape.trigger_model import TriggerEditModel
from descape.unit_model import UnitEditModel

_SIZE_STRUCT = struct.Struct("<ii")
_CAMERA_STRUCTS = {
    "f32": struct.Struct("<f"),
    "s16": struct.Struct("<h"),
    "s32": struct.Struct("<i"),
}


class Anchor(Enum):
    """Which part of the old map stays put, as (column, row) on a 3x3 grid:
    0 is the x = 0 / y = 0 edge, 2 the far edge, 1 the middle. TOP_LEFT is
    the tile (0, 0) corner, the west tip in the isometric view."""

    TOP_LEFT = (0, 0)
    TOP = (1, 0)
    TOP_RIGHT = (2, 0)
    LEFT = (0, 1)
    CENTER = (1, 1)
    RIGHT = (2, 1)
    BOTTOM_LEFT = (0, 2)
    BOTTOM = (1, 2)
    BOTTOM_RIGHT = (2, 2)

    @property
    def label(self) -> str:
        return self.name.replace("_", " ").lower()


class ResizeRefusedError(ValueError):
    """Raised by resize_scenario() for a request resize_plan() refuses."""


TRIGGERS_REMAPPED = "remapped"
TRIGGERS_STALE = "stale"
TRIGGERS_NONE = "none"


def _axis_offset(old: int, new: int, position: int) -> int:
    if position == 0:
        return 0
    if position == 2:
        return new - old
    return (new - old) // 2  # floors: an odd delta biases toward (0, 0)


def origin_offset(old_w: int, old_h: int, new_w: int, new_h: int, anchor: Anchor) -> tuple[int, int]:
    """(dx, dy): where the old map's (0, 0) lands in the new map's
    coordinates. CENTER floors, so an odd delta biases toward (0, 0)."""
    column, row = anchor.value
    return _axis_offset(old_w, new_w, column), _axis_offset(old_h, new_h, row)


def _overlap(old: int, new: int, d: int) -> int:
    return max(0, min(old + d, new) - max(d, 0))


def _off_map(unit, dx: int, dy: int, w: int, h: int, old_w: int, old_h: int) -> bool:
    tx, ty = math.floor(unit.x + dx), math.floor(unit.y + dy)
    if not (0 <= tx < w and 0 <= ty < h):
        return True
    # The cut also takes a footprint it leaves partly off (replace_refusal's state); old-edge overhang is kept.
    bounds = render.unit_tile_bounds(unit, old_w, old_h)
    if bounds is None:
        return False
    x0, x1, y0, y1 = bounds
    return x0 + dx < 0 or x1 + dx > w or y0 + dy < 0 or y1 + dy > h


def removal_closure(
    units_by_player: Iterable[Iterable], dx: int, dy: int, w: int, h: int, *, old_w: int, old_h: int
) -> list:
    """Every unit a resize deletes: those whose translated anchor tile is off
    the new map or whose footprint (render.unit_tile_bounds() on the old
    `old_w` x `old_h` map) the cut leaves partly off it, plus everything
    garrisoned (transitively) in one of them."""
    units = [u for player_units in units_by_player for u in player_units]
    removed = {id(u): u for u in units if _off_map(u, dx, dy, w, h, old_w, old_h)}
    by_host: dict[int, list] = {}
    for u in units:
        if u.garrisoned_in_id != -1 and u.garrisoned_in_id != u.reference_id:
            by_host.setdefault(u.garrisoned_in_id, []).append(u)
    pending = [u.reference_id for u in removed.values() if u.reference_id != -1]
    while pending:
        for occupant in by_host.pop(pending.pop(), ()):
            if id(occupant) not in removed:
                removed[id(occupant)] = occupant
                if occupant.reference_id != -1:
                    pending.append(occupant.reference_id)
    return list(removed.values())


@dataclass(frozen=True)
class ResizePlan:
    """The dry run the dialog reads, computed on the document's current
    state. `refusal` is None when the resize can go ahead."""

    old_w: int
    old_h: int
    new_w: int
    new_h: int
    anchor: Anchor
    dx: int
    dy: int
    tiles_gained: int
    tiles_lost: int
    units_moved: int
    units_deleted: int
    triggers: str  # TRIGGERS_REMAPPED, TRIGGERS_STALE or TRIGGERS_NONE
    trigger_note: str
    refusal: str | None

    @property
    def destructive(self) -> bool:
        return self.tiles_lost > 0 or self.units_deleted > 0

    def summary(self) -> str:
        if self.refusal is not None:
            return self.refusal
        parts = [
            f"{self.old_w}×{self.old_h} to {self.new_w}×{self.new_h}, anchored {self.anchor.label}."
        ]
        if self.tiles_lost:
            parts.append(f"{self.tiles_lost:,} tiles will be cut off.")
        if self.units_deleted:
            noun = "object" if self.units_deleted == 1 else "objects"
            parts.append(f"{self.units_deleted:,} {noun} will be deleted.")
        parts.append(self.trigger_note)
        parts.append("This cannot be undone.")
        return " ".join(p for p in parts if p)


def _unit_count(scenario: LoadedScenario) -> int:
    return sum(len(units) for units in scenario.unit_manager.units)


def _trigger_state(scenario: LoadedScenario) -> tuple[str, str]:
    manager = parse_triggers(scenario)
    if manager is None:
        return TRIGGERS_STALE, (
            "Triggers can't be read in this file, so trigger areas and locations are left "
            "as they are and may point at the wrong ground."
        )
    if not scenario.trigger_write_supported or not library_compat.vocabulary_is_available(
        scenario.scenario_version
    ):
        return TRIGGERS_STALE, (
            "Triggers can't be written back in this file, so trigger areas and locations "
            "are left as they are and may point at the wrong ground."
        )
    if not manager.triggers:
        return TRIGGERS_NONE, ""
    return TRIGGERS_REMAPPED, "Trigger areas and locations will be shifted with the map."


def _refusal(scenario: LoadedScenario, new_w: int, new_h: int, dx: int, dy: int, deleted: int) -> str | None:
    mm = scenario.map_manager
    old_w, old_h = mm.map_width, mm.map_height
    try:
        validate_map_size(new_w, new_h)
    except MapSizeError as e:
        return f"{e}."
    if old_w != old_h or new_w != new_h:
        return (
            "Only square maps can be resized for now: non-square maps wait on an "
            "in-game check that AoE2:DE accepts them."
        )
    if (new_w, new_h) == (old_w, old_h):
        return f"The map is already {old_w}×{old_h}."
    if not scenario.terrain_write_supported:
        return "This file's terrain block failed load-time verification, so it can't be resized."
    if scenario.terrain_struct_size != TERRAIN_STRUCT_SIZE or not scenario.terrain_has_layer:
        return "This file's terrain tiles have an unfamiliar layout, so it can't be resized."
    needs_unit_edits = _unit_count(scenario) > 0 and ((dx, dy) != (0, 0) or deleted > 0)
    if needs_unit_edits and (not scenario.units_write_supported or scenario.number_of_unit_sections != 9):
        return (
            "This file's units can't be edited, so it can only be grown from the "
            "top-left anchor with every object kept where it is."
        )
    if player_fields.camera_fields(scenario) is None:
        return "This file's camera positions could not be located, so it can't be resized safely."
    return None


def resize_plan(scenario: LoadedScenario, new_w: int, new_h: int, anchor: Anchor) -> ResizePlan:
    """What resize_scenario() would do, without doing it."""
    mm = scenario.map_manager
    old_w, old_h = mm.map_width, mm.map_height
    dx, dy = origin_offset(old_w, old_h, new_w, new_h, anchor)
    in_range = min(new_w, new_h) >= MIN_MAP_TILES and max(new_w, new_h) <= MAX_MAP_TILES
    units = scenario.unit_manager.units
    deleted = len(removal_closure(units, dx, dy, new_w, new_h, old_w=old_w, old_h=old_h)) if in_range else 0
    overlap = _overlap(old_w, new_w, dx) * _overlap(old_h, new_h, dy)
    triggers, note = _trigger_state(scenario)
    return ResizePlan(
        old_w=old_w,
        old_h=old_h,
        new_w=new_w,
        new_h=new_h,
        anchor=anchor,
        dx=dx,
        dy=dy,
        tiles_gained=new_w * new_h - overlap,
        tiles_lost=old_w * old_h - overlap,
        units_moved=_unit_count(scenario) - deleted,
        units_deleted=deleted,
        triggers=triggers,
        trigger_note=note,
        refusal=_refusal(scenario, new_w, new_h, dx, dy, deleted),
    )


def _filler(elevation: int) -> bytes:
    return BLANK_TERRAIN_STRUCT[:1] + bytes((elevation,)) + BLANK_TERRAIN_STRUCT[2:]


def remap_terrain(block: bytes, old_w: int, old_h: int, new_w: int, new_h: int, dx: int, dy: int) -> bytes:
    """The new terrain block: old tile (x, y) moves to (x + dx, y + dy),
    copied as whole structs row by row; a tile with no old source takes
    BLANK_TERRAIN_STRUCT with the elevation of the nearest old tile
    (clamp-to-edge, so corners extend from the corner tile)."""
    s = TERRAIN_STRUCT_SIZE
    assert len(block) == s * old_w * old_h
    fillers: dict[int, bytes] = {}

    def fill(ox: int, oy: int) -> bytes:
        elevation = block[(oy * old_w + ox) * s + 1]
        f = fillers.get(elevation)
        if f is None:
            f = fillers[elevation] = _filler(elevation)
        return f

    x_lo, x_hi = max(dx, 0), min(old_w + dx, new_w)  # new-x span with an old source
    out = bytearray()
    for ny in range(new_h):
        oy = ny - dy
        src_y = min(max(oy, 0), old_h - 1)
        if not 0 <= oy < old_h or x_lo >= x_hi:
            out += b"".join(fill(min(max(nx - dx, 0), old_w - 1), src_y) for nx in range(new_w))
            continue
        row = (oy * old_w) * s
        out += fill(0, oy) * x_lo
        out += block[row + (x_lo - dx) * s : row + (x_hi - dx) * s]
        out += fill(old_w - 1, oy) * (new_w - x_hi)
    assert len(out) == s * new_w * new_h
    return bytes(out)


def _clamp_cameras(body: bytearray, scenario: LoadedScenario, shift: int, new_w: int, new_h: int) -> int:
    """Clamps every located camera coordinate above the new map's last tile
    down to it, in place, reading the *current* value from `body` (a Players
    mode edit may have changed it). Values below 0 (the unset -1) stay as
    they are. Returns how many were changed."""
    fields = player_fields.camera_fields(scenario)
    if fields is None:
        raise WriteBlockedError("This file's camera positions could not be located.")
    changed = 0
    for field in fields:
        t = field.target
        codec = _CAMERA_STRUCTS[t.codec]
        offset = t.offset + shift
        (value,) = codec.unpack_from(body, offset)
        bound = (new_w if field.axis == "x" else new_h) - 1
        if value > bound:
            codec.pack_into(body, offset, float(bound) if t.codec == "f32" else bound)
            changed += 1
    return changed


def resize_body(scenario: LoadedScenario, new_w: int, new_h: int, anchor: Anchor, patched: PatchedBody) -> bytes:
    """`patched.body` (build_patched()'s output for `scenario`) at the new
    size: cameras clamped, terrain block remapped and its size pair
    rewritten. Everything else is carried byte-for-byte."""
    mm = scenario.map_manager
    old_w, old_h = mm.map_width, mm.map_height
    dx, dy = origin_offset(old_w, old_h, new_w, new_h, anchor)
    body = bytearray(patched.body)
    t = patched.terrain_offset
    size_at = t - _SIZE_STRUCT.size
    if _SIZE_STRUCT.unpack_from(body, size_at) != (old_w, old_h):
        raise WriteBlockedError(f"The size pair before the terrain block is not ({old_w}, {old_h}).")
    # Map (initial_player_views) and Units.player_data_3 both sit after every
    # upstream splice and before the terrain splice below, so one shift fits both.
    _clamp_cameras(body, scenario, t - scenario.terrain_block_offset, new_w, new_h)
    old_end = t + TERRAIN_STRUCT_SIZE * old_w * old_h
    block = remap_terrain(bytes(body[t:old_end]), old_w, old_h, new_w, new_h, dx, dy)
    return bytes(body[:size_at]) + _SIZE_STRUCT.pack(new_w, new_h) + block + bytes(body[old_end:])


def _axis_delta(field: str, dx: int, dy: int) -> int:
    if "_x" in field:
        return dx
    if "_y" in field:
        return dy
    raise ValueError(f"{field!r} is not a coordinate field")


def remap_triggers(loaded: LoadedScenario, dx: int, dy: int, new_w: int, new_h: int) -> TriggerEditModel | None:
    """Translates and clamps every fully-set coordinate group on the reloaded
    document, through trigger_geometry's own field list, and marks each
    changed trigger dirty. None when nothing changed."""
    model = TriggerEditModel(loaded)
    manager = model.manager()
    vocabulary = library_compat.load_vocabulary(loaded.scenario_version)
    for index, trigger in enumerate(manager.triggers):
        changed = False
        for shape in trigger_geometry.shapes_for_trigger(trigger, vocabulary, trigger_index=index):
            entries = trigger.conditions if shape.entry_kind == trigger_geometry.ENTRY_CONDITION else trigger.effects
            entry = entries[shape.entry_index]
            for field, value in zip(shape.fields, shape.coords, strict=True):
                bound = (new_w if "_x" in field else new_h) - 1
                new_value = min(max(value + _axis_delta(field, dx, dy), 0), bound)
                if new_value != value:
                    setattr(entry, field, new_value)
                    changed = True
        if changed:
            model.mark_dirty(index)
    return model if model.has_edits else None


def remap_units(
    loaded: LoadedScenario, dx: int, dy: int, new_w: int, new_h: int, *, old_w: int, old_h: int
) -> tuple[UnitEditModel | None, int]:
    """Deletes the removal closure, then translates every survivor by
    (dx, dy) through set_position() (z and rotation untouched). Returns the
    model (None when no unit needed an edit) and the number deleted.
    `old_w` x `old_h` is the pre-resize size the units' coordinates are in."""
    removed = removal_closure(loaded.unit_manager.units, dx, dy, new_w, new_h, old_w=old_w, old_h=old_h)
    if _unit_count(loaded) == 0 or ((dx, dy) == (0, 0) and not removed):
        return None, 0
    model = UnitEditModel(loaded)
    model.remove_many(removed)
    if (dx, dy) != (0, 0):
        for units in loaded.unit_manager.units:
            for unit in list(units):
                model.set_position(unit, unit.x + dx, unit.y + dy, unit.z)
    return model, len(removed)


@dataclass
class ResizeResult:
    """The resized document, reloaded, with the edit models holding its unit
    and trigger remaps (None where nothing needed one). Saving it means
    write_scenario(loaded, path, units=unit_edits, triggers=trigger_edits)."""

    loaded: LoadedScenario
    data: bytes  # the reloaded bytes, before the unit/trigger remap
    plan: ResizePlan
    unit_edits: UnitEditModel | None
    trigger_edits: TriggerEditModel | None
    units_deleted: int


def resize_scenario(
    scenario: LoadedScenario,
    new_w: int,
    new_h: int,
    anchor: Anchor,
    *,
    triggers: TriggerEditModel | None = None,
    options: OptionsEditModel | None = None,
    units: UnitEditModel | None = None,
    messages: MessagesEditModel | None = None,
) -> ResizeResult:
    """Resizes `scenario` with its pending edits (the same models
    write_scenario() takes). Raises ResizeRefusedError for anything
    resize_plan() refuses. `scenario` itself is left untouched."""
    plan = resize_plan(scenario, new_w, new_h, anchor)
    if plan.refusal is not None:
        raise ResizeRefusedError(plan.refusal)
    patched = build_patched(scenario, triggers, options, units, messages)
    body = resize_body(scenario, new_w, new_h, anchor, patched)
    data = patched.header + _compress_bytes(body)
    loaded = load_map_and_units_from_bytes(data, scenario.path)
    mm = loaded.map_manager
    if (mm.map_width, mm.map_height) != (new_w, new_h) or not loaded.terrain_write_supported:
        raise WriteBlockedError(
            f"The resized map reloaded as {mm.map_width}x{mm.map_height} "
            f"(terrain verified: {loaded.terrain_write_supported})."
        )
    unit_edits, deleted = remap_units(loaded, plan.dx, plan.dy, new_w, new_h, old_w=plan.old_w, old_h=plan.old_h)
    if deleted != plan.units_deleted:
        raise WriteBlockedError(f"Deleted {deleted} objects, but the plan counted {plan.units_deleted}.")
    trigger_edits = None
    if plan.triggers == TRIGGERS_REMAPPED:
        if parse_triggers(loaded) is None or not loaded.trigger_write_supported:
            raise WriteBlockedError("The resized map's Triggers section no longer verifies.")
        trigger_edits = remap_triggers(loaded, plan.dx, plan.dy, new_w, new_h)
    return ResizeResult(loaded, data, plan, unit_edits, trigger_edits, deleted)


__all__ = [
    "TRIGGERS_NONE",
    "TRIGGERS_REMAPPED",
    "TRIGGERS_STALE",
    "Anchor",
    "ResizePlan",
    "ResizeRefusedError",
    "ResizeResult",
    "origin_offset",
    "remap_terrain",
    "remap_triggers",
    "remap_units",
    "removal_closure",
    "resize_body",
    "resize_plan",
    "resize_scenario",
]
