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


def _read(obj: Any, attribute: str) -> Any:
    try:
        return getattr(obj, attribute, None)
    except Exception:  # noqa: BLE001 -- the library's version-gated properties raise
        return None


def _coordinate(entry: Any, attribute: str) -> int:
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
        coords = tuple(_coordinate(entry, f) for f in fields)
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


def area_centre(coords: Sequence[int]) -> tuple[int, int]:
    """An area's integer midpoint, so a run's number is reproducible."""
    x1, y1, x2, y2 = coords
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def location_superseded(shapes: Iterable[TriggerShape]) -> bool:
    """Whether ONE entry's location is only an authoring-time copy: a resolved
    location_object_reference is the real destination (60 of 61 corpus
    effects carrying both store that unit's own tile as the location)."""
    return any(s.shape == SHAPE_UNIT and s.fields == (LOCATION_OBJECT_FIELD,) for s in shapes)


def run_for_entry(shapes: Iterable[TriggerShape]) -> ruler.Measurement | None:
    """Origin to destination, for ONE entry's shapes, or None unless the
    entry has both. Origin: the area centre, else the integer centroid of the
    selected units' own tiles. Destination: the location object's own tile,
    else the location. Not special-cased to Patrol: Attack Move, Task Object,
    Teleport and the rest share the same shape."""
    area = location = target = None
    selected: list[tuple[int, int]] = []
    for shape in shapes:
        if shape.shape == SHAPE_AREA and area is None:
            area = shape
        elif shape.shape == SHAPE_LOCATION and location is None:
            location = shape
        elif shape.shape == SHAPE_UNIT and shape.anchor is not None:
            if shape.fields == (LOCATION_OBJECT_FIELD,) and target is None:
                target = shape.anchor
            elif shape.fields == (SELECTED_OBJECTS_FIELD,):
                selected.append(shape.anchor)
    if area is not None:
        origin = area_centre(area.coords)
    elif selected:
        origin = (sum(x for x, _y in selected) // len(selected), sum(y for _x, y in selected) // len(selected))
    else:
        return None
    if target is not None:
        destination = target
    elif location is not None:
        destination = (location.coords[0], location.coords[1])
    else:
        return None
    return ruler.measure(origin, destination)
