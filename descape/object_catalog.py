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
   .dat-derived category, plus `_EDITOR_LISTED_DAT_ONLY` (the Gunpowder Keg),
   which the .dat hides but the official editor lists. `combined_object_name()` stays dataset-only
   in AoE2: the trigger field formatter's ALL-CAPS contract depends on it. Under
   Return of Rome it names `objects()`' rows, the AoE2-shared hidden ones included.
2. **Document-scoped references** (`TriggerId`/`VariableId`) -- resolved
   against the open document's own trigger/variable list, which no dataset
   covers.

Lazy and lru_cache'd throughout: importing this module must never touch a
configured install. `clear_caches()` is called from
asset_source.set_install_path_override(), so a name resolved before a fresh
install is configured never survives past that point.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from AoE2ScenarioParser.datasets.buildings import BuildingInfo
from AoE2ScenarioParser.datasets.heroes import HeroInfo
from AoE2ScenarioParser.datasets.other import OtherInfo
from AoE2ScenarioParser.datasets.techs import TechInfo
from AoE2ScenarioParser.datasets.units import UnitInfo

from descape import asset_source, game_dataset

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
# Every objects() category, in the in-game Object List Type's own option order (GH #41).
OBJECT_CATEGORIES = tuple(category for _dataset, category in _OBJECT_DATASETS)

# tools/gen_object_catalog.py's output: integers and short .dat codes only,
# never generated at runtime and never touching an install by itself --
# resolve_name() below is what reaches into the configured install, through
# asset_source.resource_string().
_CATALOG_JSON_PATH = Path(__file__).resolve().parent / "object_catalog.json"


@game_dataset.cache(maxsize=1)
def _dat_table() -> dict:
    """The committed .dat-derived table, or an empty one if it has not been
    generated yet -- every reader below already degrades to the
    library-only name on a miss, so a missing file behaves exactly like the
    library-only catalog this module shipped with before slice 4."""
    path = game_dataset.table_path(_CATALOG_JSON_PATH)
    if not path.is_file():
        return {"objects": {}, "techs": {}, "terrains": {}}
    return json.loads(path.read_text())


@functools.cache
def _aoe2_dat_objects() -> dict:
    """The AoE2 table's object rows whatever dataset is active (GH #67's shared hidden objects)."""
    path = game_dataset.table_path(_CATALOG_JSON_PATH, game_dataset.AOE2)
    if not path.is_file():
        return {}
    return json.loads(path.read_text())["objects"]


def _ror_shared_hidden() -> dict[int, str]:
    """id -> category for every object the RoR editor hides that is the same
    object as one AoE2's objects() lists (its editor's, plus the library
    members it hides): same .dat code and class in both
    tables (543 on 2026-10-17, the Castle among them). Listed as hidden, so
    "Show editor-hidden objects" offers them; an emptied RoR slot (WALL5 2678)
    or a reused one (93, Spearman there, Hoplite here) differs in code."""
    aoe2 = _aoe2_dat_objects()
    candidates: dict[int, str] = {}
    for dataset, category in _OBJECT_DATASETS:
        for member in dataset:
            candidates.setdefault(member.ID, category)
    for key, row in aoe2.items():
        if not row.get("hidden"):
            candidates.setdefault(int(key), derived_category(row))
    for id_, category in _EDITOR_LISTED_DAT_ONLY.items():
        candidates.setdefault(id_, category)
    shared: dict[int, str] = {}
    for id_, category in candidates.items():
        old, new = aoe2.get(str(id_)), _dat_entry("objects", id_)
        if old and new and new.get("hidden") and (old.get("code"), old.get("class")) == (new.get("code"), new.get("class")):
            shared[id_] = category
    return shared


def same_object_in_aoe2(id_: int) -> bool:
    """Whether id_ is the same object in both games' tables: same .dat code and
    class (GH #67). A pick keeps only such a const across a dataset switch."""
    old, new = _aoe2_dat_objects().get(str(id_)), _dat_table()["objects"].get(str(id_))
    return bool(old and new) and (old.get("code"), old.get("class")) == (new.get("code"), new.get("class"))


def _dat_entry(section: str, id_: int) -> dict | None:
    return _dat_table()[section].get(str(id_))


def dat_objects() -> dict[int, dict]:
    """unit_const -> the committed .dat row for every object, or {} with no
    generated table. A fresh dict of fresh rows, as dat_terrains()."""
    return {int(k): dict(v) for k, v in _dat_table()["objects"].items()}


def dat_techs() -> dict[int, dict]:
    """tech_id -> the committed .dat row for every tech, or {} with no
    generated table. A fresh dict of fresh rows, as dat_objects()."""
    return {int(k): dict(v) for k, v in _dat_table().get("techs", {}).items()}


def dat_terrains() -> dict[int, dict]:
    """terrain_id -> the committed `{string_id, code, hidden}` row for every
    enabled .dat terrain, or {} with no generated table. A fresh dict, so a
    caller can't mutate the cached table."""
    return {int(k): dict(v) for k, v in _dat_table().get("terrains", {}).items()}


def dat_terrain_restrictions() -> dict[int, frozenset[int]]:
    """restriction id -> the terrain ids that .dat restriction row passes, for
    every id an object's `restriction` key names plus the Beach row 2
    (GH #176), or {} with no generated table. An object with no `restriction`
    key passes everywhere."""
    return {int(k): frozenset(v) for k, v in _dat_table().get("terrain_restrictions", {}).items()}


# Type 80 classes that the library files under Units/Heroes, not Buildings:
# packed (51) and unpacked (54) siege.
_SIEGE_BUILDING_CLASSES = frozenset({51, 54})
_RELIC_CLASS = 42

# Dat-only ids the .dat hides but the official editor lists, by its category: Gunpowder Keg (GH #208)
# and Invisible Spawner A-J, which its 1.59 Other tab lists for a player (GH #32, as RoR's below).
_EDITOR_LISTED_DAT_ONLY = dict.fromkeys((2607, 2616, 2617, 2618, 2619, 2620, 2621, 2645, 2646, 2647, 2648), "Others")
# The same for Return of Rome, from its in-game editor's Other tab: Invisible Spawner A-J (GH #67).
_ROR_EDITOR_LISTED_DAT_ONLY = dict.fromkeys((2616, 2617, 2618, 2619, 2620, 2621, 2645, 2646, 2647, 2648), "Others")
# The reverse: unhidden RoR .dat ids its editor leaves out, a nameless wolf and pyramid (measured, GH #67).
_ROR_EDITOR_UNLISTED = frozenset({89, 689})
# Unhidden AoE2 ids its 1.59 editor lists for neither Player 1 nor GAIA, listed here as
# editor-hidden: CUMAN DISABLED (measured 2026-10-21, GH #32).
_EDITOR_UNLISTED = frozenset({1261})


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


def is_building(object_id: int) -> bool:
    """derived_category()'s "Buildings" rule from the .dat row alone. GH #148:
    which portrait set an icon index points into (the game's own rule is not in the data)."""
    entry = _dat_entry("objects", object_id)
    return entry is not None and derived_category(entry) == "Buildings"


# Other-tab objects whose .dat icon index points into the building portraits: Granary
# (eye-candy type 10, same id in both games), which both editors draw with a building icon (GH #32).
_BUILDING_PORTRAIT_CONSTS = frozenset({1089})


def portrait_for(object_id: int) -> tuple[int | None, bool]:
    """(icon_for(), is_building() or a _BUILDING_PORTRAIT_CONSTS member):
    asset_source.get_unit_icon()'s arguments."""
    return icon_for(object_id), is_building(object_id) or object_id in _BUILDING_PORTRAIT_CONSTS


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
    _aoe2_dat_objects.cache_clear()


@game_dataset.cache(maxsize=1)
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
    if game_dataset.is_ror():
        # The library's enums are AoE2's: under Return of Rome objects()' rows stand in (GH #67).
        return {entry.id: entry.name for entry in objects()}
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


# GH #58: what a picker row for a unit only the active data mod's .dat defines ends in.
DATA_MOD_TAG = "(data mod)"


def _entry(id_: int, name: str, category: str, dat_entry: dict | None = None) -> CatalogEntry:
    return CatalogEntry(
        id=id_,
        name=name,
        category=category,
        class_id=dat_entry.get("class") if dat_entry else None,
        hidden=bool(dat_entry.get("hidden")) if dat_entry else False,
        search_text=f"{name.lower()} {id_}",
    )


@game_dataset.cache(maxsize=1)
def objects() -> tuple[CatalogEntry, ...]:
    """Every object across the four datasets, categorized by which one it
    came from, alphabetical by name. An id already claimed by an earlier
    dataset in _OBJECT_DATASETS is not repeated under a later category.

    Then every .dat-table id no dataset covers and the .dat does not hide
    from the editor (the Sept 2026 patch's Jarl, Longhouse, Spruce, ...),
    named install string -> .dat code and categorized by derived_category().
    Editor-hidden dat-only ids stay out (user decision 2026-09-27), except
    _EDITOR_LISTED_DAT_ONLY (_ROR_EDITOR_LISTED_DAT_ONLY under Return of
    Rome), listed as not hidden under the official editor's category (GH #208).
    Under Return of Rome, also _ror_shared_hidden(), as hidden. A dat-only id
    the active data mod adds (GH #58) keeps its derived_category(), the tab the
    game's editor lists it under, and its row name ends in DATA_MOD_TAG."""
    from descape import data_mods

    seen: set[int] = set()
    entries = []
    lang = asset_source.get_language()
    mod_only = data_mods.mod_only_ids()
    # Under Return of Rome the AoE2 enums and the AoE2 Gunpowder Keg id do not apply (GH #67).
    ror = game_dataset.is_ror()
    editor_listed = _ROR_EDITOR_LISTED_DAT_ONLY if ror else _EDITOR_LISTED_DAT_ONLY
    for dataset, category in () if ror else _OBJECT_DATASETS:
        for member in dataset:
            if member.ID in seen:
                continue
            seen.add(member.ID)
            dat_entry = _dat_entry("objects", member.ID)
            if dat_entry is not None and member.ID in _EDITOR_UNLISTED:
                dat_entry = {**dat_entry, "hidden": True}
            name = resolve_name(member.ID, member.name.replace("_", " "), dat_entry, lang)
            entries.append(_entry(member.ID, name, category, dat_entry))
    for key, raw_entry in _dat_table()["objects"].items():
        id_ = int(key)
        category = editor_listed.get(id_)
        if id_ in seen or (raw_entry.get("hidden") and category is None) or (ror and id_ in _ROR_EDITOR_UNLISTED):
            continue
        dat_entry = raw_entry if category is None else {**raw_entry, "hidden": False}
        if not ror and id_ in _EDITOR_UNLISTED:
            dat_entry = {**dat_entry, "hidden": True}
        name = resolve_name(id_, "", dat_entry, lang)
        if id_ in mod_only:
            name = f"{name} {DATA_MOD_TAG}"
        entries.append(_entry(id_, name, category or derived_category(dat_entry), dat_entry))
        seen.add(id_)
    if ror:
        for id_, category in _ror_shared_hidden().items():
            if id_ not in seen and id_ not in _ROR_EDITOR_UNLISTED:
                dat_entry = _dat_entry("objects", id_)
                entries.append(_entry(id_, resolve_name(id_, "", dat_entry, lang), category, dat_entry))
    return tuple(sorted(entries, key=lambda e: e.name))


@game_dataset.cache(maxsize=1)
def techs() -> tuple[CatalogEntry, ...]:
    """Every tech in TechInfo, alphabetical by name. Under Return of Rome,
    every named tech of its own .dat instead (GH #67)."""
    entries = []
    lang = asset_source.get_language()
    if game_dataset.is_ror():
        for key, dat_entry in _dat_table().get("techs", {}).items():
            entries.append(_entry(int(key), resolve_name(int(key), "", dat_entry, lang), "Techs"))
        return tuple(sorted(entries, key=lambda e: e.name))
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
