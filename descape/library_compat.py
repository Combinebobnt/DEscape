"""
AoE2ScenarioParser compatibility shim for reading the Triggers and Units
sections.

Kept separate from scenario_io.py so that module stays about the Map/Units load
path itself. Nothing in here parses a scenario; it makes parsing one safe and
supplies the condition/effect vocabulary the trigger UI is built from. Named
for the library it wraps, not for one section: Unit joined
Trigger/Condition/Effect/Variable as a poisoned class once the Units write
path needed the same fix -- Unit is parsed eagerly rather than lazily like
Triggers, so depoisoning it after the fact is too late for objects already
parsed while poisoned; see depoison()'s own docstring below.

Two library behaviours make a shim necessary.

1. Class-level poisoning across versions. Each of Trigger/Condition/Effect/
   Variable/Unit carries a class-level `_link_list` of RetrieverObjectLink
   objects, some gated by a `Support(since=...)` range. On load, any link whose
   range excludes the file's scenario version is "disabled": the library calls
   setattr(cls, name, property(...)) to replace that attribute with one that
   raises UnsupportedAttributeError, and sets link.disabled = True so it is
   never reconsidered. Both effects are global to the class and permanent, and
   the error message captures the version that was current when it fired. Load
   a 1.41 file and then a 1.55 file in the same process and reading
   Condition.local_technology on the *1.55* file raises "supported since: 1.55.
   Current version: 1.41". The library knows: retriever_object_link.py:81 has a
   standing "Todo: Doesn't work properly when reading an older scenario first,
   and a newer one later. Properties don't get reset!". depoison() is that
   reset.

2. Version-dependent module state. _initialise_version_dependencies() rewrites
   the module-level conditions.attributes / effects.attributes dicts on every
   load, so with two files open they describe whichever was loaded last.
   load_vocabulary() reads the per-version JSON off disk instead, which also
   means the UI can populate a condition/effect picker before any file is open.

**Import ordering is load-bearing.** PRISTINE_CLASS_STATE snapshots every
POISONED_CLASSES class at import time, and that snapshot is only clean if nothing has loaded a
scenario yet. descape/scenario_io.py imports this module at its own top for
exactly that reason: it is the only module that loads scenarios, so importing
it can never come first. Do not make that import lazy.

Reaches into AoE2ScenarioParser private API, acceptable on the same terms as
scenario_io.py: the install is pinned to a known commit (v0.8.3,
b763e2e37006bedab50c7d349b3ce24b9c2497f6). See tests/test_private_api_guard.py.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

import AoE2ScenarioParser
from AoE2ScenarioParser.objects.data_objects.condition import Condition
from AoE2ScenarioParser.objects.data_objects.effect import Effect
from AoE2ScenarioParser.objects.data_objects.terrain_tile import TerrainTile
from AoE2ScenarioParser.objects.data_objects.trigger import Trigger
from AoE2ScenarioParser.objects.data_objects.unit import Unit
from AoE2ScenarioParser.objects.data_objects.variable import Variable
from AoE2ScenarioParser.objects.managers.map_manager import MapManager
from AoE2ScenarioParser.sections.retrievers.retriever_object_link import RetrieverObjectLink
from AoE2ScenarioParser.sections.retrievers.retriever_object_link_group import (
    RetrieverObjectLinkGroup,
)

# Every class whose class-level state a load can change. Trigger/Condition/
# Effect/Variable/Unit are poisoned by the library itself; MapManager and
# TerrainTile are re-linked by adapt_map_links() below for a structure that
# lacks fields they pull. depoison() restores all seven the same way.
POISONED_CLASSES = (Trigger, Condition, Effect, Variable, Unit, MapManager, TerrainTile)

# Names never touched by a restore: they are descriptor slots owned by the type
# machinery, not library state, and assigning them raises.
_UNRESTORABLE = frozenset({"__dict__", "__weakref__"})

VERSIONS_DIR = Path(AoE2ScenarioParser.__file__).resolve().parent / "versions" / "DE"

# The Triggers section's first field is an f64 trigger_version, and trigger_tail
# starts at the Triggers section's first byte. See scenario_io.load_map_and_units.
_TRIGGER_VERSION_STRUCT = struct.Struct("<d")


def _iter_links(links) -> Iterator[Any]:
    """Flatten a _link_list. Entries are RetrieverObjectLink, or a
    RetrieverObjectLinkGroup whose own `.group` holds the real links; only the
    leaves carry `support`/`disabled`."""
    for link in links:
        if isinstance(link, RetrieverObjectLinkGroup):
            yield from _iter_links(link.group)
        else:
            yield link


def _snapshot() -> dict[type, dict[str, Any]]:
    return {cls: dict(vars(cls)) for cls in POISONED_CLASSES}


# Captured at import, before any scenario has been loaded -- see the module
# docstring on why that ordering is guaranteed rather than merely hoped for.
PRISTINE_CLASS_STATE: dict[type, dict[str, Any]] = _snapshot()


def class_state_delta(cls: type) -> tuple[list[str], list[str]]:
    """(added, overwritten) attribute names on `cls` versus its pristine
    snapshot. Empty lists mean the class is clean. Exposed for tests and for
    diagnosing a load that poisoned something depoison() does not cover."""
    pristine = PRISTINE_CLASS_STATE[cls]
    current = dict(vars(cls))
    added = sorted(set(current) - set(pristine))
    overwritten = sorted(
        name for name, val in pristine.items() if name in current and current[name] is not val
    )
    return added, overwritten


def depoison() -> None:
    """Restore every POISONED_CLASSES class to its pre-load state.

    Call before every Triggers parse, and before the Map/Units section walk in
    scenario_io._load_map_and_units() -- Unit is parsed there, not lazily like
    Triggers, so depoisoning it after the fact is a silent no-op: an object
    already parsed while poisoned keeps its poisoned properties regardless of
    what happens to the class afterward. Three steps, all of them required:
    delete attributes the library added, restore the ones it overwrote, and
    clear `disabled` on every link so the next load re-evaluates support
    ranges against its own scenario version.

    Restoring overwritten *values* is not redundant with deleting added names.
    The library overwrites hand-written properties the classes already define
    (Effect._quantity_float is the one that bites), so those names exist in the
    pristine snapshot and deleting-added-names alone leaves them poisoned.
    """
    for cls in POISONED_CLASSES:
        pristine = PRISTINE_CLASS_STATE[cls]
        for name in sorted(set(vars(cls)) - set(pristine)):
            if name not in _UNRESTORABLE:
                delattr(cls, name)
        for name, value in pristine.items():
            if name not in _UNRESTORABLE and vars(cls).get(name) is not value:
                setattr(cls, name, value)
        for link in _iter_links(cls._link_list):
            link.disabled = False


class _AbsentFieldLink(RetrieverObjectLink):
    """Stands in for a link whose retriever this file's structure does not
    have: pulls a fixed value, pushes nothing. Used where `support=` cannot
    be, because the library would pass None to an __init__ that rejects it."""

    def __init__(self, variable_name: str, value: Any) -> None:
        super().__init__(variable_name)
        self._value = value

    def pull_from_link(self, uuid=None, number_hist=None, host_obj=None, progress=None) -> Any:
        return self._value

    def push_to_link(self, uuid=None, number_hist=None, host_obj=None, progress=None) -> None:
        return None


def _without_link(group: RetrieverObjectLinkGroup, name: str, replacement=None) -> RetrieverObjectLinkGroup:
    """A new group equal to `group` minus the link called `name`, or with it
    swapped for `replacement`. Never mutates `group`: it belongs to the
    pristine snapshot. Raises KeyError if no such link, so a library bump
    that renames it fails loudly."""
    names = [link.name for link in group.group]
    if name not in names:
        raise KeyError(f"{name!r} is not a link of {group.section_name}/{group.link}: {names}")
    links = []
    for link in group.group:
        if link.name != name:
            links.append(link)
        elif replacement is not None:
            links.append(replacement)
    new_group = RetrieverObjectLinkGroup(group.section_name, group.link, group=links)
    # The constructor re-parents its children; point them back at the
    # pristine group so that group object is left exactly as it was.
    for link in group.group:
        link.parent = group
    return new_group


def _relinked(cls: type, name: str, replacement=None) -> list:
    """cls._link_list with the first group holding `name` rebuilt by
    _without_link(). A new list, so depoison()'s value-restore step puts the
    pristine one back."""
    new_list = list(cls._link_list)
    for i, entry in enumerate(new_list):
        if isinstance(entry, RetrieverObjectLinkGroup) and any(link.name == name for link in entry.group):
            new_list[i] = _without_link(entry, name, replacement)
            return new_list
    raise KeyError(f"{cls.__name__}._link_list has no group holding {name!r}")


def adapt_map_links(map_section: Any) -> None:
    """Re-link MapManager/TerrainTile for a Map section lacking fields they
    pull unconditionally (v1.21: no map_color_mood, no TerrainStruct.layer).

    Presence-derived, never keyed on scenario version. Call after
    depoison() and after the Map section has parsed, before
    MapManager.construct(). A no-op for every structure that has both
    fields; the next depoison() undoes it. `layer` is dropped rather than
    gated, so TerrainTile.__init__'s -1 default stands and the eight
    in-memory `.layer` sites keep working.
    """
    retriever_map = map_section.retriever_map
    if "map_color_mood" not in retriever_map:
        MapManager._link_list = _relinked(MapManager, "_map_color_mood", _AbsentFieldLink("_map_color_mood", ""))
    tiles = retriever_map["terrain_data"].data or []
    if tiles and "layer" not in tiles[0].retriever_map:
        TerrainTile._link_list = _relinked(TerrainTile, "layer")


def trigger_version(trigger_tail: bytes) -> float:
    """The Triggers section's f64 trigger_version, read straight off the front
    of `trigger_tail`. Distinct from scenario version, and the thing the
    library's 1.54-only load gate actually tests."""
    if len(trigger_tail) < _TRIGGER_VERSION_STRUCT.size:
        raise ValueError(f"trigger_tail too short to hold a trigger version: {len(trigger_tail)} bytes")
    return _TRIGGER_VERSION_STRUCT.unpack_from(trigger_tail, 0)[0]


@dataclass(frozen=True)
class VocabularyEntry:
    """One condition or effect type, as the library's per-version JSON defines it."""

    id: int
    name: str
    attributes: tuple[str, ...]  # the fields this type actually displays
    default_attributes: Mapping[str, Any]  # every field, with its unset value


@dataclass(frozen=True)
class TriggerVocabulary:
    """The condition/effect vocabulary for one scenario version.

    `*_presentation` maps a field name to how its value should be rendered
    (an enum name like "PlayerId", "bool", or "" for a plain number), from the
    JSON's own "-1" pseudo-entry. AoE2ScenarioParser's
    objects/support/attr_presentation.py turns those into display strings.
    """

    scenario_version: str
    conditions: Mapping[int, VocabularyEntry]
    effects: Mapping[int, VocabularyEntry]
    condition_presentation: Mapping[str, str]
    effect_presentation: Mapping[str, str]


def _parse_vocabulary_json(path: Path) -> tuple[Mapping[int, VocabularyEntry], Mapping[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    # Key "-1" is not a real type: it holds the shared attribute-presentation map.
    presentation = dict(raw.get("-1", {}).get("attribute_presentation", {}))
    entries = {}
    for key, value in raw.items():
        if key == "-1":
            continue
        entry_id = int(key)
        entries[entry_id] = VocabularyEntry(
            id=entry_id,
            name=value["name"],
            attributes=tuple(value.get("attributes", ())),
            default_attributes=MappingProxyType(dict(value.get("default_attributes", {}))),
        )
    return MappingProxyType(entries), MappingProxyType(presentation)


def vocabulary_is_available(scenario_version: str) -> bool:
    """False for a scenario version the installed library ships no definition
    for. The census found 71 such files (DE 1.21/1.32/1.35). v1.21 files now
    load from this repo's own structure (scenario_io.structure_is_available())
    yet still have no vocabulary, so this stays the trigger-specific probe."""
    return (VERSIONS_DIR / f"v{scenario_version}" / "conditions.json").is_file()


@cache
def load_vocabulary(scenario_version: str) -> TriggerVocabulary:
    """Condition/effect vocabulary for `scenario_version`, read from the
    library's own versions/DE/v<version>/ JSON.

    Deliberately not the module-level conditions.attributes/effects.attributes
    dicts: those are rewritten per load and describe the wrong version once two
    files have been opened. Reading the JSON needs no loaded scenario at all.
    Cached because the UI asks per repaint and the files never change on disk.
    """
    version_dir = VERSIONS_DIR / f"v{scenario_version}"
    conditions_path = version_dir / "conditions.json"
    effects_path = version_dir / "effects.json"
    if not conditions_path.is_file() or not effects_path.is_file():
        raise FileNotFoundError(
            f"AoE2ScenarioParser ships no condition/effect definition for scenario "
            f"version {scenario_version} (looked in {version_dir})"
        )
    conditions, condition_presentation = _parse_vocabulary_json(conditions_path)
    effects, effect_presentation = _parse_vocabulary_json(effects_path)
    return TriggerVocabulary(
        scenario_version=scenario_version,
        conditions=conditions,
        effects=effects,
        condition_presentation=condition_presentation,
        effect_presentation=effect_presentation,
    )
