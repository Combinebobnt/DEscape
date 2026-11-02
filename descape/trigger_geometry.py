"""Qt-free reader for the map coordinates a trigger's conditions and effects
carry: areas, locations and (v1.57+ build_object only) wall runs.

The one shared accessor for trigger geometry. The trigger map overlay (GH #41)
reads through it, and map resize is meant to write back through the same
records, so the two can never disagree about which fields exist in which
scenario version.

Rules, each with its reason:

- Which fields an entry has comes from the vocabulary (the library's per-version
  JSON), never a hardcoded type list: the area-bearing effect set grows from 22
  types (v1.36) to 44 (v1.58), and wall_* exists only from v1.57.
- None reads exactly as -1 (trigger_fields.UNSET). After a depoison() a field
  this scenario version lacks reads as None instead of raising.
- A group is valid only when every coordinate in it is >= 0. A half-set group
  yields no shape.
- Nothing is re-normalized. The library's validate_coords() already copied and
  swapped at parse time; doing it again here would double-apply.
- Shapes are frozen plain ints and hold no library object: a later parse of
  another scenario disables fields on the same library classes, so a live entry
  must not outlive the call that read it.
- Given a unit_references.ReferenceIndex, a Unit/Unit[] field (by presentation)
  also yields one "unit" shape per id that resolves to an on-map unit: the
  footprint as inclusive corners, like an area. Its coords are derived, not
  stored, so a writer (map resize) must skip "unit" shapes.
- created_units_for_trigger() reads the units a Create Object or Place
  Foundation effect would place (GH #167), matched on the vocabulary's effect
  name, never a type number. The map's trigger ghosts draw them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from . import ruler, unit_references
from .trigger_fields import UNSET

AREA_FIELDS = ("area_x1", "area_y1", "area_x2", "area_y2")
LOCATION_FIELDS = ("location_x", "location_y")
WALL_FIELDS = ("wall_x1", "wall_y1", "wall_x2", "wall_y2")

SHAPE_AREA = "area"
SHAPE_LOCATION = "location"
SHAPE_WALL = "wall"
SHAPE_UNIT = "unit"

# The two effect fields a run line reads unit shapes from.
SELECTED_OBJECTS_FIELD = "selected_object_ids"
LOCATION_OBJECT_FIELD = "location_object_reference"

_GROUPS = ((SHAPE_AREA, AREA_FIELDS), (SHAPE_LOCATION, LOCATION_FIELDS), (SHAPE_WALL, WALL_FIELDS))
_SHAPE_FOR_FIELDS = {fields: shape for shape, fields in _GROUPS}

ENTRY_CONDITION = "condition"
ENTRY_EFFECT = "effect"
_KINDS = (
    (ENTRY_CONDITION, "conditions", "condition_type"),
    (ENTRY_EFFECT, "effects", "effect_type"),
)

# GH #167: the effects that put a new unit of their const on their location.
CREATING_EFFECTS = ("create_object", "place_foundation")
CREATED_UNIT_FIELD = "object_list_unit_id"
CREATED_PLAYER_FIELD = "source_player"
FACET_FIELD = "facet"


@dataclass(frozen=True)
class TriggerShape:
    """One coordinate group of one condition or effect, as plain ints.

    `fields` names the source attributes in the same order as `coords`, which
    is what lets a writer (map resize) put a remapped value back where it came
    from. Areas are inclusive on both corners, the way the game stores them.
    """

    trigger_index: int
    entry_kind: str
    entry_index: int
    shape: str
    fields: tuple[str, ...]
    coords: tuple[int, ...]
    # "unit" shapes only: the referenced unit's own tile and id.
    anchor: tuple[int, int] | None = None
    reference_id: int = UNSET

    @property
    def entry_ref(self) -> tuple[str, int]:
        """(entry_kind, entry_index), the panel's own entry reference shape."""
        return (self.entry_kind, self.entry_index)


@dataclass(frozen=True)
class CreatedUnit:
    """One unit a Create Object or Place Foundation effect would place, as
    plain ints (GH #167). `effect` is the vocabulary name; `player` may be
    UNSET; `facet` is UNSET when unset or when the effect has no facet."""

    trigger_index: int
    entry_index: int
    effect: str
    unit_const: int
    player: int
    tile: tuple[int, int]
    facet: int = UNSET

    @property
    def entry_ref(self) -> tuple[str, int]:
        """(entry_kind, entry_index), as TriggerShape's: always an effect."""
        return (ENTRY_EFFECT, self.entry_index)

    @property
    def payload(self) -> tuple:
        """Everything but the two indices: what the ghost looks like, so the
        map's resolve memo keys on it and survives an entry renumbering."""
        return (self.effect, self.unit_const, self.player, self.tile, self.facet)


def _read(obj: Any, attribute: str) -> Any:
    try:
        return getattr(obj, attribute, None)
    except Exception:  # noqa: BLE001 -- the library's version-gated properties raise
        return None


def coordinate(entry: Any, attribute: str) -> int:
    """One coordinate as an int, UNSET for None, a raising read or a non-number.
    Public so trigger_range reads an area exactly as the overlay does."""
    value = _read(entry, attribute)
    if value is None:
        return UNSET
    try:
        return int(value)
    except (TypeError, ValueError):
        return UNSET


def coordinate_fields(definition: Any) -> tuple[tuple[str, ...], ...]:
    """The coordinate groups a VocabularyEntry displays, in AREA, LOCATION,
    WALL order. A group counts only when all of its fields are listed; no
    shipped version lists part of one."""
    if definition is None:
        return ()
    attributes = set(definition.attributes)
    return tuple(fields for _shape, fields in _GROUPS if all(f in attributes for f in fields))


def _unit_corners(ref: unit_references.UnitRef) -> tuple[int, int, int, int]:
    """render.unit_tile_bounds()' half-open (x0, x1, y0, y1) as inclusive
    (x1, y1, x2, y2) corners, the order and form an area uses. A diagonal
    gate gets its bounding rectangle, not its sparse footprint (accepted)."""
    x0, x1, y0, y1 = ref.bounds
    return (x0, y0, x1 - 1, y1 - 1)


def shapes_for_entry(
    entry: Any,
    definition: Any,
    *,
    trigger_index: int,
    entry_kind: str,
    entry_index: int,
    references: unit_references.ReferenceIndex | None = None,
    presentation: Mapping[str, str] | None = None,
) -> list[TriggerShape]:
    """Every fully-set coordinate group of one condition or effect, then,
    given `references` and the kind's `presentation` map, one "unit" shape
    per referenced id that is placed on the map. Dangling and off-map ids
    yield nothing."""
    shapes = []
    for fields in coordinate_fields(definition):
        coords = tuple(coordinate(entry, f) for f in fields)
        if any(c < 0 for c in coords):
            continue
        shapes.append(
            TriggerShape(trigger_index, entry_kind, entry_index, _SHAPE_FOR_FIELDS[fields], fields, coords)
        )
    if references is None or presentation is None:
        return shapes
    for field, ids in unit_references.references_in(entry, definition, presentation).items():
        for ref_id in ids:
            ref = references.get(ref_id)
            if ref is None or ref.bounds is None:
                continue
            shapes.append(
                TriggerShape(
                    trigger_index, entry_kind, entry_index, SHAPE_UNIT, (field,), _unit_corners(ref),
                    anchor=ref.own_tile, reference_id=ref_id,
                )
            )
    return shapes


def shapes_for_trigger(
    trigger: Any,
    vocabulary: Any,
    *,
    trigger_index: int,
    references: unit_references.ReferenceIndex | None = None,
) -> list[TriggerShape]:
    """Every shape of one trigger, conditions first, each in list order.
    Unit shapes only when `references` is given."""
    shapes: list[TriggerShape] = []
    for kind, list_name, type_attribute in _KINDS:
        if kind == ENTRY_CONDITION:
            definitions, presentation = vocabulary.conditions, vocabulary.condition_presentation
        else:
            definitions, presentation = vocabulary.effects, vocabulary.effect_presentation
        for entry_index, entry in enumerate(_read(trigger, list_name) or ()):
            definition = definitions.get(_read(entry, type_attribute))
            shapes.extend(
                shapes_for_entry(
                    entry,
                    definition,
                    trigger_index=trigger_index,
                    entry_kind=kind,
                    entry_index=entry_index,
                    references=references,
                    presentation=presentation,
                )
            )
    return shapes


def created_units_for_trigger(trigger: Any, vocabulary: Any, *, trigger_index: int) -> list[CreatedUnit]:
    """The units one trigger's Create Object and Place Foundation effects
    would place, in list order. An unset const and an unset or half-set
    location yield nothing; the facet is read only where the type lists it."""
    records: list[CreatedUnit] = []
    for entry_index, entry in enumerate(_read(trigger, "effects") or ()):
        definition = vocabulary.effects.get(_read(entry, "effect_type"))
        if definition is None or definition.name not in CREATING_EFFECTS:
            continue
        const = coordinate(entry, CREATED_UNIT_FIELD)
        x, y = (coordinate(entry, f) for f in LOCATION_FIELDS)
        if const < 0 or x < 0 or y < 0:
            continue
        facet = coordinate(entry, FACET_FIELD) if FACET_FIELD in definition.attributes else UNSET
        player = coordinate(entry, CREATED_PLAYER_FIELD)
        records.append(CreatedUnit(trigger_index, entry_index, definition.name, const, player, (x, y), facet))
    return records


def area_centre(coords: Sequence[int]) -> tuple[int, int]:
    """An area's integer midpoint, so a run's number is reproducible."""
    x1, y1, x2, y2 = coords
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def tile_centroid(tiles: Sequence[tuple[int, int]]) -> tuple[int, int] | None:
    """The integer centroid of some units' own tiles, or None for none: a
    run's origin and Go to Objects (GH #138) share it."""
    if not tiles:
        return None
    return (sum(x for x, _y in tiles) // len(tiles), sum(y for _x, y in tiles) // len(tiles))


def location_superseded(shapes: Iterable[TriggerShape]) -> bool:
    """Whether ONE entry's location is only an authoring-time copy: a resolved
    location_object_reference is the real destination (60 of 61 corpus
    effects carrying both store that unit's own tile as the location)."""
    return any(s.shape == SHAPE_UNIT and s.fields == (LOCATION_OBJECT_FIELD,) for s in shapes)


def run_origin(shapes: Iterable[TriggerShape]) -> tuple[int, int] | None:
    """Where ONE entry's run starts: the area centre, else the integer
    centroid of the selected units' own tiles, else None."""
    selected: list[tuple[int, int]] = []
    for shape in shapes:
        if shape.shape == SHAPE_AREA:
            return area_centre(shape.coords)
        if shape.shape == SHAPE_UNIT and shape.anchor is not None and shape.fields == (SELECTED_OBJECTS_FIELD,):
            selected.append(shape.anchor)
    return tile_centroid(selected)


def run_for_entry(shapes: Iterable[TriggerShape]) -> ruler.Measurement | None:
    """Origin to destination, for ONE entry's shapes, or None unless the
    entry has both. Origin: run_origin(). Destination: the location object's
    own tile, else the location. Not special-cased to Patrol: Attack Move,
    Task Object, Teleport and the rest share the same shape."""
    shapes = list(shapes)
    origin = run_origin(shapes)
    if origin is None:
        return None
    location = target = None
    for shape in shapes:
        if shape.shape == SHAPE_LOCATION and location is None:
            location = shape
        elif (
            shape.shape == SHAPE_UNIT
            and shape.anchor is not None
            and shape.fields == (LOCATION_OBJECT_FIELD,)
            and target is None
        ):
            target = shape.anchor
    if target is not None:
        destination = target
    elif location is not None:
        destination = (location.coords[0], location.coords[1])
    else:
        return None
    return ruler.measure(origin, destination)


# GH #229's drag zones: the middle, then edges and corners named for the
# side they grab in map axes (n/s the y-min/y-max side, w/e the x-min/x-max).
ZONE_MOVE = "move"
AREA_ZONES = (ZONE_MOVE, "n", "s", "w", "e", "nw", "ne", "sw", "se")
# GH #41's location mark, dragged by the same machine. Not an area zone.
ZONE_LOCATION = "location"


def area_hit_zone(
    coords: Sequence[int], point: tuple[float, float], band: float, tile_borders: bool = False
) -> str | None:
    """Which part of an inclusive area (x1, y1, x2, y2) the fractional map
    point is over, or None outside it. Each edge band is `band` tiles wide,
    capped at a third of the span so a 1x1 area keeps a middle to grab.
    `tile_borders` widens it to at least one tile, only on an axis 3 or more
    tiles long, so the area still keeps a middle tile."""
    x1, y1, x2, y2 = coords
    lo_x, hi_x = min(x1, x2), max(x1, x2) + 1
    lo_y, hi_y = min(y1, y2), max(y1, y2) + 1
    px, py = point
    if not (lo_x <= px < hi_x and lo_y <= py < hi_y):
        return None
    bx, by = (
        min(max(1.0, band) if tile_borders and span >= 3 else band, span / 3)
        for span in (hi_x - lo_x, hi_y - lo_y)
    )
    ns = "n" if py - lo_y < by else "s" if hi_y - py < by else ""
    we = "w" if px - lo_x < bx else "e" if hi_x - px < bx else ""
    return ns + we or ZONE_MOVE


def location_hit(tile: Sequence[int], point: tuple[float, float], reach: float) -> bool:
    """Whether the fractional map point is within `reach` tiles of the tile's
    centre on both axes."""
    return abs(point[0] - tile[0] - 0.5) <= reach and abs(point[1] - tile[1] - 0.5) <= reach


def _clamp(value: int, low: int, high: int) -> int:
    return min(max(value, low), high)


def dragged_area(
    coords: Sequence[int],
    zone: str,
    start_tile: tuple[int, int],
    end_tile: tuple[int, int],
    map_w: int,
    map_h: int,
) -> tuple[int, int, int, int]:
    """The min/max area a drag from start_tile to end_tile makes of `coords`.
    A move keeps the size and shifts the whole area no further than the map
    edge; a resize shifts the grabbed sides by the same delta, clamped to the
    map, and may cross the opposite side. A location drags as a 1x1 rect."""
    x1, x2 = sorted((coords[0], coords[2]))
    y1, y2 = sorted((coords[1], coords[3]))
    dx, dy = end_tile[0] - start_tile[0], end_tile[1] - start_tile[1]
    if zone in (ZONE_MOVE, ZONE_LOCATION):
        dx = _clamp(dx, -x1, map_w - 1 - x2)
        dy = _clamp(dy, -y1, map_h - 1 - y2)
        return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
    if "w" in zone:
        x1 = _clamp(x1 + dx, 0, map_w - 1)
    elif "e" in zone:
        x2 = _clamp(x2 + dx, 0, map_w - 1)
    if "n" in zone:
        y1 = _clamp(y1 + dy, 0, map_h - 1)
    elif "s" in zone:
        y2 = _clamp(y2 + dy, 0, map_h - 1)
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
