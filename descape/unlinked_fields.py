"""Carries struct fields the installed AoE2ScenarioParser class doesn't link,
so they follow their unit or condition through edits instead of a list slot.

Scenario version 1.59 added `UnitStruct.capture_flag` and
`ConditionStruct.allow_in_fog`. DEscape reads 1.59 through a vendored
structure (descape/versions/DE/v1.59/), but the pinned 0.8.3 `Unit` and
`Condition` classes have no RetrieverObjectLink for either field. On save the
library commits objects into list slots by index and shortens or pads the list
from its end, so an unlinked value stays with its slot: delete a unit, move
its list neighbour, and the neighbour saves with the deleted unit's flag.

The fix, without monkeypatching the library's link lists:

- pull(): at load, copy each slot's value onto its object as a plain instance
  attribute of the same name. AoE2Object.__deepcopy__ copies `__dict__`, so
  undo snapshots and trigger copy/paste carry it along.
- push(): after the library's commit and before any slot is serialized, write
  each object's attribute back into its current slot. An object that never
  had one (a new unit or condition) writes UNLINKED_FIELDS' default.

Dormant by construction once the pin moves to a library that links a field:
active_fields() skips any name the class links, so nothing is double-handled.
tests/test_v159_load_path.py asserts UNLINKED_FIELDS equals "UnitStruct/
ConditionStruct retriever names minus the class's link names" for every repo
structure, so a new vendored version with another such field fails there.

No Unit RetrieverObjectLink is added instead: UnitManager finds the caption
links by position (unit_manager.py:155,159).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from AoE2ScenarioParser.objects.data_objects.condition import Condition
from AoE2ScenarioParser.objects.data_objects.unit import Unit

from descape import library_compat

# class -> {field: default for an object that never carried one}. -1 is the
# vendored structure's own default for both.
UNLINKED_FIELDS: Mapping[type, Mapping[str, Any]] = {
    Unit: {"capture_flag": -1},
    Condition: {"allow_in_fog": -1},
}


def link_names(cls: type) -> frozenset[str]:
    """Every retriever name `cls`'s _link_list links (the `link=` target, or
    the attribute name when that is the retriever's own name)."""
    names = set()
    for link in library_compat._iter_links(cls._link_list):
        target = getattr(link, "link", None) or link.name
        names.add(target.split("[")[0])
    return frozenset(names)


def active_fields(cls: type, retriever_map: Mapping[str, Any]) -> list[str]:
    """The UNLINKED_FIELDS names for `cls` this file's struct has and the
    installed class does not link. Empty for v1.58 and older, and after a pin
    bump that links them."""
    linked = link_names(cls)
    return [name for name in UNLINKED_FIELDS.get(cls, {}) if name in retriever_map and name not in linked]


def pull(cls: type, objects: Iterable[Any], entries: Iterable[Any]) -> None:
    """Copies each entry's unlinked values onto the object built from it.
    `objects` and `entries` are parallel, in struct order."""
    fields = None
    for obj, entry in zip(objects, entries, strict=True):
        if fields is None:
            fields = active_fields(cls, entry.retriever_map)
            if not fields:
                return
        for name in fields:
            setattr(obj, name, entry.retriever_map[name].data)


def push(cls: type, objects: Iterable[Any], entries: Iterable[Any]) -> None:
    """Writes each object's carried value into the slot it now occupies.
    Call after the library's commit (which re-slots objects) and before the
    slots are serialized."""
    defaults = UNLINKED_FIELDS.get(cls, {})
    fields = None
    for obj, entry in zip(objects, entries, strict=True):
        if fields is None:
            fields = active_fields(cls, entry.retriever_map)
            if not fields:
                return
        for name in fields:
            entry.retriever_map[name].set_data(getattr(obj, name, defaults[name]), affect_dirty=False)
