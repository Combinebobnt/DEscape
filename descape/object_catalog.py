"""Qt-free catalog of pickable id-dataset entries, plus document-scoped
reference resolvers, backing the trigger property form's reference-field
pickers.

Two different problems live here:

1. **Id datasets** (`UnitInfo`/`BuildingInfo`/`HeroInfo`/`OtherInfo` combined
   as one object catalog, plus `TechInfo`) -- library-backed by default,
   `tools/gen_object_catalog.py`'s committed `object_catalog.json` swaps in
   real in-game display names, class grouping, and the editor-hidden flag
   from the user's own install without this module's callers changing.
2. **Document-scoped references** (`TriggerId`/`VariableId`) -- resolved
   against the open document's own trigger/variable list, which no dataset
   covers.

Lazy and lru_cache'd throughout: importing this module must never touch a
configured install. `clear_caches()` is called from
asset_source.set_install_path_override(), so a name resolved before a fresh
install is configured never survives past that point.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from AoE2ScenarioParser.datasets.buildings import BuildingInfo
from AoE2ScenarioParser.datasets.heroes import HeroInfo
from AoE2ScenarioParser.datasets.other import OtherInfo
from AoE2ScenarioParser.datasets.techs import TechInfo
from AoE2ScenarioParser.datasets.units import UnitInfo

from descape import asset_source

# The in-game editor's own four object-placing menus
# (docs/INGAME_EDITOR_REFERENCE.md:87) -- a grouping recoverable only from
# which library dataset a member sits in, never from the .dat itself (see the
# plan's slice 4 note on `category`). Order matters: it is also the merge
# priority for combined_object_name()/objects(), first dataset wins on an id
# collision.
_OBJECT_DATASETS = (
    (UnitInfo, "Units"),
    (BuildingInfo, "Buildings"),
    (HeroInfo, "Heroes"),
    (OtherInfo, "Others"),
)

# tools/gen_object_catalog.py's output: integers and short .dat codes only,
# never generated at runtime and never touching an install by itself --
# resolve_name() below is what reaches into the configured install, through
# asset_source.resource_string().
_CATALOG_JSON_PATH = Path(__file__).resolve().parent / "object_catalog.json"


@lru_cache(maxsize=1)
def _dat_table() -> dict:
    """The committed .dat-derived table, or an empty one if it has not been
    generated yet -- every reader below already degrades to the
    library-only name on a miss, so a missing file behaves exactly like the
    library-only catalog this module shipped with before slice 4."""
    if not _CATALOG_JSON_PATH.is_file():
        return {"objects": {}, "techs": {}}
    return json.loads(_CATALOG_JSON_PATH.read_text())


def _dat_entry(section: str, id_: int) -> dict | None:
    return _dat_table()[section].get(str(id_))


def resolve_name(id_: int, library_name: str, dat_entry: dict | None) -> str:
    """install string -> library enum name -> .dat short code -> "UNKNOWN_
    <id>", each step falling through to the next. Degrades to library_name
    alone with no install configured or no .dat entry for id_ at all --
    what keeps the default (install-free) test tier on the exact format
    resolve_reference()/the catalog always had before slice 4."""
    if dat_entry is not None:
        install_name = asset_source.resource_string(dat_entry["string_id"])
        if install_name:
            return install_name
    if library_name:
        return library_name
    if dat_entry is not None and dat_entry.get("code"):
        return dat_entry["code"]
    return f"UNKNOWN_{id_}"


def clear_caches() -> None:
    """Drops every cache this module builds from a configured install's
    resolved names, so asset_source.set_install_path_override() can call
    this the same way it clears its own caches and unit_sprites'."""
    _combined_object_names.cache_clear()
    objects.cache_clear()
    techs.cache_clear()


@lru_cache(maxsize=1)
def _combined_object_names() -> dict[int, str]:
    """object_const -> display name, merged across the four object datasets.

    First dataset wins on an id collision, so the result is stable regardless
    of iteration order. Install-resolved (e.g. "Town Center") once slice 4's
    object_catalog.json is generated and an install is configured; the exact
    ALL-CAPS library name (e.g. "TOWN CENTER") otherwise, which is what
    trigger_fields.resolve_reference's own pinned tests depend on.
    """
    names: dict[int, str] = {}
    for dataset, _category in _OBJECT_DATASETS:
        for member in dataset:
            if member.ID in names:
                continue
            dat_entry = _dat_entry("objects", member.ID)
            names[member.ID] = resolve_name(member.ID, member.name.replace("_", " "), dat_entry)
    return names


def combined_object_name(object_const: int) -> str:
    """The merged display name for object_const, or "" if no dataset covers
    it. A function rather than exposing the cached dict, so a caller can't
    mutate the shared cache."""
    return _combined_object_names().get(object_const, "")


@dataclass(frozen=True)
class CatalogEntry:
    """One pickable row of an id-dataset catalog.

    `class_id`/`hidden` come from object_catalog.json (techs carry neither --
    the plan's `category` note applies to objects only) -- None/False for
    every entry, and every tech entry, until that file is generated.
    """

    id: int
    name: str
    category: str
    class_id: int | None
    hidden: bool
    search_text: str


def _entry(id_: int, name: str, category: str, dat_entry: dict | None = None) -> CatalogEntry:
    return CatalogEntry(
        id=id_,
        name=name,
        category=category,
        class_id=dat_entry.get("class") if dat_entry else None,
        hidden=bool(dat_entry.get("hidden")) if dat_entry else False,
        search_text=f"{name.lower()} {id_}",
    )


@lru_cache(maxsize=1)
def objects() -> tuple[CatalogEntry, ...]:
    """Every object across the four datasets, categorized by which one it
    came from, alphabetical by name. An id already claimed by an earlier
    dataset in _OBJECT_DATASETS is not repeated under a later category."""
    seen: set[int] = set()
    entries = []
    for dataset, category in _OBJECT_DATASETS:
        for member in dataset:
            if member.ID in seen:
                continue
            seen.add(member.ID)
            dat_entry = _dat_entry("objects", member.ID)
            name = resolve_name(member.ID, member.name.replace("_", " "), dat_entry)
            entries.append(_entry(member.ID, name, category, dat_entry))
    return tuple(sorted(entries, key=lambda e: e.name))


@lru_cache(maxsize=1)
def techs() -> tuple[CatalogEntry, ...]:
    """Every tech in TechInfo, alphabetical by name."""
    entries = []
    for member in TechInfo:
        dat_entry = _dat_entry("techs", member.ID)
        name = resolve_name(member.ID, member.name.replace("_", " "), dat_entry)
        entries.append(_entry(member.ID, name, "Techs"))
    return tuple(sorted(entries, key=lambda e: e.name))


def entry(catalog: Sequence[CatalogEntry], id_: int) -> CatalogEntry | None:
    """The catalog row for id_, or None if it is not in this catalog."""
    for candidate in catalog:
        if candidate.id == id_:
            return candidate
    return None


def name_for(catalog: Sequence[CatalogEntry], id_: int) -> str:
    """entry(catalog, id_)'s name, or "" if id_ is not in this catalog."""
    found = entry(catalog, id_)
    return found.name if found is not None else ""


def object_name(id_: int) -> str:
    """Display name for any object id, including ones the library-backed
    `objects()` dataset omits -- e.g. cliffs (class 34), which
    AoE2ScenarioParser's UnitInfo/BuildingInfo/HeroInfo/OtherInfo enums don't
    cover at all but object_catalog.json still carries a .dat "code" for.
    Goes through resolve_name()'s own install-name -> code -> UNKNOWN_<id>
    fallback chain directly, skipping the library-name step objects() would
    otherwise supply.
    """
    return resolve_name(id_, "", _dat_entry("objects", id_))


# -- document-scoped references: TriggerId, VariableId -----------------------
#
# Neither dataset above covers these -- a trigger_id or variable_id only means
# anything relative to the open document's own trigger/variable list, read
# straight off the parsed TriggerManager rather than cached, since a manager
# held across another file's open is only safe to read fresh (see
# TriggerPanel.variable_rows()'s own reasoning).


def trigger_choices(manager) -> tuple[tuple[str, int], ...]:
    """(label, trigger_id) pairs for manager's trigger list, in list order --
    what a TriggerId combo box populates from."""
    return tuple((t.name or f"Trigger {t.trigger_id}", t.trigger_id) for t in manager.triggers)


def trigger_name_for(manager, trigger_id: int) -> str:
    """The name of the trigger with this trigger_id, or "" if manager's
    trigger list does not carry one. Trigger ids are renumbered by
    remove_triggers()/reorder_triggers(), so this always re-reads manager
    rather than caching by id."""
    for t in manager.triggers:
        if t.trigger_id == trigger_id:
            return t.name or ""
    return ""


def variable_choices(manager) -> tuple[tuple[str, int], ...]:
    """(label, variable_id) pairs for manager's variable list."""
    return tuple((v.name or f"Variable {v.variable_id}", v.variable_id) for v in manager.variables)


def variable_name_for(manager, variable_id: int) -> str:
    """The name of the variable with this variable_id, or "" if manager's
    variable list does not carry one."""
    for v in manager.variables:
        if v.variable_id == variable_id:
            return v.name or ""
    return ""
