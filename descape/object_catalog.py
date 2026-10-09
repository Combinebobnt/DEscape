"""Qt-free catalog of pickable id-dataset entries, plus document-scoped
reference resolvers, backing the trigger property form's reference-field
pickers.

Two different problems live here:

1. **Id datasets** (`UnitInfo`/`BuildingInfo`/`HeroInfo`/`OtherInfo` combined
   as one object catalog, plus `TechInfo`) -- library-backed by default,
   `tools/gen_object_catalog.py`'s committed `object_catalog.json` swaps in
   real in-game display names, class grouping, and the editor-hidden flag
   from the user's own install without this module's callers changing.
   `objects()` also lists every .dat-table object no dataset covers and the
   .dat does not hide from the editor, under `derived_category()`'s
   .dat-derived category. `combined_object_name()` stays dataset-only: the
   trigger field formatter's ALL-CAPS contract depends on it.
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
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

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
        return {"objects": {}, "techs": {}, "terrains": {}}
    return json.loads(_CATALOG_JSON_PATH.read_text())


def _dat_entry(section: str, id_: int) -> dict | None:
    return _dat_table()[section].get(str(id_))


def dat_objects() -> dict[int, dict]:
    """unit_const -> the committed .dat row for every object, or {} with no
    generated table. A fresh dict of fresh rows, as dat_terrains()."""
    return {int(k): dict(v) for k, v in _dat_table()["objects"].items()}


def dat_terrains() -> dict[int, dict]:
    """terrain_id -> the committed `{string_id, code, hidden}` row for every
    enabled .dat terrain, or {} with no generated table. A fresh dict, so a
    caller can't mutate the cached table."""
    return {int(k): dict(v) for k, v in _dat_table().get("terrains", {}).items()}


# Type 80 classes that the library files under Units/Heroes, not Buildings:
# packed (51) and unpacked (54) siege.
_SIEGE_BUILDING_CLASSES = frozenset({51, 54})
_RELIC_CLASS = 42


def derived_category(dat_entry: dict) -> str:
    """The picker category for an object no library dataset covers, from its
    .dat fields alone. Re-measured against every library id by
    tests/test_object_catalog.py (1245 of 1246 editor-visible ids agree)."""
    type_ = dat_entry.get("type")
    class_ = dat_entry.get("class")
    if type_ == 70 and class_ == _RELIC_CLASS:
        return "Others"
    if type_ == 80 and class_ not in _SIEGE_BUILDING_CLASSES:
        return "Buildings"
    if type_ in (70, 80):
        return "Heroes" if dat_entry.get("hero_mode", 0) & 1 else "Units"
    return "Others"


def icon_for(object_id: int) -> int | None:
    """The .dat's icon index for an object (GH #140's unit portrait), or None
    for -1 or an id the committed catalog lacks."""
    entry = _dat_entry("objects", object_id)
    icon = None if entry is None else entry.get("icon")
    return icon if isinstance(icon, int) and icon >= 0 else None


def resolve_name(id_: int, library_name: str, dat_entry: dict | None, lang: str | None = None) -> str:
    """install string -> library enum name -> .dat short code -> "UNKNOWN_
    <id>", each step falling through to the next. Degrades to library_name
    alone with no install configured or no .dat entry for id_ at all --
    what keeps the default (install-free) test tier on the exact format
    resolve_reference()/the catalog always had before slice 4.

    lang defaults to asset_source.get_language(), which parses config.yaml on
    every call; a bulk builder resolves it once and passes it in."""
    if dat_entry is not None:
        install_name = asset_source.resource_string(dat_entry["string_id"], lang)
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
    lang = asset_source.get_language()
    for dataset, _category in _OBJECT_DATASETS:
        for member in dataset:
            if member.ID in names:
                continue
            dat_entry = _dat_entry("objects", member.ID)
            names[member.ID] = resolve_name(member.ID, member.name.replace("_", " "), dat_entry, lang)
    return names


def combined_object_name(object_const: int) -> str:
    """The merged display name for object_const, or "" if no dataset covers
    it. A function rather than exposing the cached dict, so a caller can't
    mutate the shared cache."""
    return _combined_object_names().get(object_const, "")


def is_known_object(object_const: int) -> bool:
    """Whether any library dataset or the committed .dat table covers this const."""
    return bool(combined_object_name(object_const)) or _dat_entry("objects", object_const) is not None


def display_name(unit_const: int) -> str:
    """Mirrors terrain_palette.name_for_terrain_id's shape, UNKNOWN_<id>
    fallback included: a scenario can legitimately reference a unit_const no
    dataset covers, and that must read as a known gap rather than a crash.

    combined_object_name() supplies the four-dataset merge (~1,355 members
    total, first dataset wins on an ID collision); the title-casing is this
    call site's own display convention, not shared with
    trigger_fields.resolve_reference's ALL CAPS one. An id no dataset covers
    falls through to object_name() (install string -> .dat code), so a
    dat-only const, editor-hidden or not, reads e.g. "Jarl" rather than
    UNKNOWN_<id>; only an id the .dat table lacks too reads UNKNOWN_<id>.
    """
    name = combined_object_name(unit_const)
    if name:
        return name.title()
    # Install strings are already cased, so no .title() past the library path.
    return object_name(unit_const)


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
    dataset in _OBJECT_DATASETS is not repeated under a later category.

    Then every .dat-table id no dataset covers and the .dat does not hide
    from the editor (the Sept 2026 patch's Jarl, Longhouse, Spruce, ...),
    named install string -> .dat code and categorized by derived_category().
    Editor-hidden dat-only ids stay out entirely (user decision 2026-09-27)."""
    seen: set[int] = set()
    entries = []
    lang = asset_source.get_language()
    for dataset, category in _OBJECT_DATASETS:
        for member in dataset:
            if member.ID in seen:
                continue
            seen.add(member.ID)
            dat_entry = _dat_entry("objects", member.ID)
            name = resolve_name(member.ID, member.name.replace("_", " "), dat_entry, lang)
            entries.append(_entry(member.ID, name, category, dat_entry))
    for key, dat_entry in _dat_table()["objects"].items():
        id_ = int(key)
        if id_ in seen or dat_entry.get("hidden"):
            continue
        name = resolve_name(id_, "", dat_entry, lang)
        entries.append(_entry(id_, name, derived_category(dat_entry), dat_entry))
    return tuple(sorted(entries, key=lambda e: e.name))


@lru_cache(maxsize=1)
def techs() -> tuple[CatalogEntry, ...]:
    """Every tech in TechInfo, alphabetical by name."""
    entries = []
    lang = asset_source.get_language()
    for member in TechInfo:
        dat_entry = _dat_entry("techs", member.ID)
        name = resolve_name(member.ID, member.name.replace("_", " "), dat_entry, lang)
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


def object_name(id_: int, lang: str | None = None) -> str:
    """Display name for any object id, including ones the library-backed
    `objects()` dataset omits -- e.g. cliffs (class 34), which
    AoE2ScenarioParser's UnitInfo/BuildingInfo/HeroInfo/OtherInfo enums don't
    cover at all but object_catalog.json still carries a .dat "code" for.
    Goes through resolve_name()'s own install-name -> code -> UNKNOWN_<id>
    fallback chain directly, skipping the library-name step objects() would
    otherwise supply. A caller naming many ids passes lang, as resolve_name()
    says.
    """
    return resolve_name(id_, "", _dat_entry("objects", id_), lang)


def tech_name(id_: int, lang: str | None = None) -> str:
    """Display name for any tech id, including ones `techs()`' TechInfo enum
    omits -- the tech-side mirror of object_name(), and for the same reason:
    a scenario's disable lists can legitimately carry an id no enum covers,
    and that must read as a known gap rather than blank.

    Object ids and tech ids are separate id spaces that collide freely on the
    same number, which is why this is its own function rather than a
    `section` argument to object_name(): resolving a tech through the objects
    table would name it after an unrelated unit. lang as object_name().
    """
    return resolve_name(id_, "", _dat_entry("techs", id_), lang)


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
