"""Qt-free resolver from a trigger field's placed-unit reference_id to the
unit it names: owner, const, own tile and footprint.

Which fields hold such ids comes from the vocabulary's presentation map
("Unit" or "Unit[]"), never a hardcoded field list: condition `unit_object`
and `next_object` carry presentation "" before v1.44, so they are not
references there. map_analysis.check_unit_references() and the trigger
panel's labels both walk through here, so the dangling-reference check and
the labels cannot disagree.

Rules, each with its reason:

- One pass over every placed unit (all_units()), never unit_pick.build_index(),
  which drops filtered and off-map units by design and is keyed
  (player, ref). A reference names a unit whatever the Filters menu says.
- Plain ints only. A UnitRef holds no library object: a later parse of another
  scenario disables fields on the same library classes.
- A reference_id placed twice: the first unit wins and the id goes in
  `duplicates`, so describe() can say so.
- None and -1 both mean unset.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from . import trigger_fields
from .render import unit_tile_bounds

UNIT_REFERENCE_PRESENTATIONS = frozenset({"Unit", "Unit[]"})
LIST_PRESENTATION = "Unit[]"


@dataclass(frozen=True)
class UnitRef:
    reference_id: int
    player_id: int
    unit_const: int
    own_tile: tuple[int, int]  # (int(x), int(y))
    bounds: tuple[int, int, int, int] | None  # render.unit_tile_bounds(); None = off-map


@dataclass(frozen=True)
class ReferenceIndex:
    by_id: Mapping[int, UnitRef]
    duplicates: frozenset[int]

    def get(self, ref_id: Any) -> UnitRef | None:
        return self.by_id.get(ref_id) if isinstance(ref_id, int) and not isinstance(ref_id, bool) else None


EMPTY_INDEX = ReferenceIndex(MappingProxyType({}), frozenset())


def all_units(loaded) -> Iterator[tuple[int, Any]]:
    """(player_id, unit) over the raw nine lists, index 0 = GAIA."""
    for player_id, units in enumerate(loaded.unit_manager.units):
        for unit in units:
            yield player_id, unit


def build_reference_index(loaded) -> ReferenceIndex:
    mm = loaded.map_manager
    width, height = mm.map_width, mm.map_height
    by_id: dict[int, UnitRef] = {}
    duplicates: set[int] = set()
    for player_id, unit in all_units(loaded):
        ref_id = int(unit.reference_id)
        if ref_id in by_id:
            duplicates.add(ref_id)
            continue
        by_id[ref_id] = UnitRef(
            reference_id=ref_id,
            player_id=player_id,
            unit_const=int(unit.unit_const),
            own_tile=(int(unit.x), int(unit.y)),
            bounds=unit_tile_bounds(unit, width, height),
        )
    return ReferenceIndex(MappingProxyType(by_id), frozenset(duplicates))


def unit_reference_fields(definition: Any, presentation: Mapping[str, str]) -> tuple[str, ...]:
    """The attributes of one condition/effect type that hold placed-unit ids."""
    if definition is None:
        return ()
    return tuple(a for a in definition.attributes if presentation.get(a) in UNIT_REFERENCE_PRESENTATIONS)


def _read(obj: Any, attribute: str) -> Any:
    try:
        return getattr(obj, attribute, None)
    except Exception:  # noqa: BLE001 -- the library's version-gated properties raise
        return None


def _ids(value: Any) -> tuple[int, ...]:
    values = value if isinstance(value, (list, tuple)) else (value,)
    return tuple(
        v for v in values if isinstance(v, int) and not isinstance(v, bool) and v != trigger_fields.UNSET
    )


def references_in(entry: Any, definition: Any, presentation: Mapping[str, str]) -> dict[str, tuple[int, ...]]:
    """field -> the set ids it holds, in stored order. Fields holding nothing are omitted."""
    found = {}
    for attribute in unit_reference_fields(definition, presentation):
        ids = _ids(_read(entry, attribute))
        if ids:
            found[attribute] = ids
    return found


def describe(index: ReferenceIndex, ref_id: Any) -> str:
    """`Name (const) [P1, X12, Y34]`, the library's own shape with the unit's
    own tile, or why the id does not resolve. "" for unset."""
    if not _ids(ref_id):
        return ""
    ref = index.get(ref_id)
    if ref is None:
        return f"{ref_id}: not placed on the map"
    name = trigger_fields.resolve_reference("UnitInfo", ref.unit_const) or "Unknown object"
    text = f"{name} ({ref.unit_const}) [P{ref.player_id}"
    text += ", off the map]" if ref.bounds is None else f", X{ref.own_tile[0]}, Y{ref.own_tile[1]}]"
    if ref.reference_id in index.duplicates:
        text += " (id placed more than once)"
    return text
