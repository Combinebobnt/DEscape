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
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from AoE2ScenarioParser.datasets.buildings import BuildingInfo
from AoE2ScenarioParser.datasets.heroes import HeroInfo
from AoE2ScenarioParser.datasets.other import OtherInfo
from AoE2ScenarioParser.datasets.techs import TechInfo
from AoE2ScenarioParser.datasets.units import UnitInfo

from descape import asset_source, game_dataset, games

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

# The trigger tech picker's internal rows, hidden there until "Show internal
# techs" is ticked: placeholders and the .dat's own helper techs, which no
# scenario has a reason to research. Matched on the .dat code name, never the
# display label ("BLANK TECHNOLOGY 0", "Technology Placeholder 01"). A hand-
# written list, not a .dat fact, so it lives here rather than in
# tools/gen_object_catalog.py. Measured 2026-10-29: 74 of TechInfo's 587 rows.
INTERNAL_TECH_FAMILIES: Mapping[str, re.Pattern[str]] = {
    "Blank Technology": re.compile(r"^Blank Technology \d+$"),
    "Placeholder": re.compile(r"^Placeholder \d+$"),
    "C-Bonus": re.compile(r"^C-Bonus, "),
    "(make avail)": re.compile(r" \(make avail\)$", re.IGNORECASE),
    "Auto upgrade": re.compile(r"^Auto upgrade "),
    "requirement": re.compile(r" requirement$"),
    "Change TC constr time": re.compile(r"^Change TC constr time\d*$"),
    "Static Bonuses": re.compile(r" Static Bonuses$"),
}


def internal_tech_family(code: str) -> str:
    """The INTERNAL_TECH_FAMILIES key a tech's .dat code name falls in, or ""."""
    return next((family for family, pattern in INTERNAL_TECH_FAMILIES.items() if pattern.search(code)), "")


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
def _read_table(path: Path) -> dict:
    if not path.is_file():
        return {"objects": {}, "techs": {}, "terrains": {}}
    return json.loads(path.read_text())


def _committed_table(game_id: str) -> dict:
    """`game_id`'s own committed table whatever game or data mod is active, or an empty one like _dat_table()."""
    return _read_table(game_dataset.table_path(_CATALOG_JSON_PATH, game_id))


def _default_dat_table() -> dict:
    """The default game's committed table (AoE2's), whatever game or data mod
    is active: what a data mod's rows and a pick kept across a switch compare with."""
    return _committed_table(games.DEFAULT_ID)


def _ror_shared_hidden() -> dict[int, str]:
    """id -> category for every object the RoR editor hides that is the same
    object as one its fallback game's objects() lists (AoE2's editor's, plus
    the library members it hides): same .dat code and class in both
    tables (543 on 2026-10-17, the Castle among them). Listed as hidden, so
    "Show editor-hidden objects" offers them; an emptied RoR slot (WALL5 2678)
    or a reused one (93, Spearman there, Hoplite here) differs in code."""
    fallback = game_dataset.game().fallback
    if fallback is None:
        return {}
    base = _committed_table(fallback)["objects"]
    candidates: dict[int, str] = {}
    for dataset, category in _OBJECT_DATASETS:
        for member in dataset:
            candidates.setdefault(member.ID, category)
    for key, row in base.items():
        if not row.get("hidden"):
            candidates.setdefault(int(key), derived_category(row))
    for id_, category in _EDITOR_LISTED_BY_GAME.for_game(games.get(fallback)).items():
        candidates.setdefault(id_, category)
    shared: dict[int, str] = {}
    for id_, category in candidates.items():
        old, new = base.get(str(id_)), _dat_entry("objects", id_)
        if old and new and new.get("hidden") and (old.get("code"), old.get("class")) == (new.get("code"), new.get("class")):
            shared[id_] = category
    return shared


def same_object_in_aoe2(id_: int) -> bool:
    """Whether id_ is the same object in the default game's committed table
    and the one in effect: same .dat code and class (GH #67). A pick keeps only
    such a const across a game or data-mod switch."""
    old, new = _default_dat_table()["objects"].get(str(id_)), _dat_table()["objects"].get(str(id_))
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


def is_aoe2_terrain(terrain_id: int) -> bool:
    """Whether the committed AoE2 table has terrain_id, whatever dataset or data mod is in effect."""
    return str(terrain_id) in _default_dat_table().get("terrains", {})


@game_dataset.cache(maxsize=1)
def data_mod_terrain_ids() -> frozenset[int]:
    """The active data mod's own terrains (GH #271): ids the committed AoE2
    table lacks, or whose row differs from it in code or string_id, or whose
    string_id the mod's strings files give a text other than the install's.
    Empty with no data mod."""
    if game_dataset.data_mod() is None:
        return frozenset()
    from descape import asset_source

    aoe2 = _default_dat_table().get("terrains", {})
    renamed = asset_source.data_mod_changed_string_ids()
    def changed(key: str, row: dict) -> bool:
        old = aoe2.get(key)
        return old is None or (old.get("code"), old.get("string_id")) != (row.get("code"), row.get("string_id"))

    rows = _dat_table().get("terrains", {})
    return frozenset(int(key) for key, row in rows.items() if changed(key, row) or row.get("string_id") in renamed)


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

_EDITOR_LISTED_BY_GAME = games.PerGame(
    "object_catalog: dat-hidden ids the game's editor lists, by category",
    {"aoe2": _EDITOR_LISTED_DAT_ONLY, "ror": _ROR_EDITOR_LISTED_DAT_ONLY},
)
_EDITOR_HIDDEN_BY_GAME = games.PerGame(
    "object_catalog: unhidden ids the game's editor lists for no one, offered as editor-hidden",
    {"aoe2": _EDITOR_UNLISTED, "ror": frozenset()},
)
_EDITOR_LEFT_OUT_BY_GAME = games.PerGame(
    "object_catalog: unhidden ids the game's editor leaves out, left out of objects()",
    {"aoe2": frozenset(), "ror": _ROR_EDITOR_UNLISTED},
)
_SHARED_HIDDEN_BY_GAME = games.PerGame(
    "object_catalog: editor-hidden objects shared with the fallback game, offered as hidden",
    {"aoe2": dict, "ror": _ror_shared_hidden},
)


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


def icon_for(object_id: int, civ: int | None = None) -> int | None:
    """The .dat's icon index for an object (GH #140's unit portrait), civ's own
    copy's when its row has one (GH #372's `civ_icons`), or None for -1 or an
    id the committed catalog lacks."""
    entry = _dat_entry("objects", object_id)
    if entry is None:
        return None
    icon = entry.get("icon")
    if civ is not None:
        icon = entry.get("civ_icons", {}).get(str(civ), icon)
    return icon if isinstance(icon, int) and icon >= 0 else None


def portrait_civ_key(object_id: int, civ: int | None) -> int | None:
    """civ when object_id's row has its own icon for it, else None: two civs
    showing the same portrait share one cache entry (GH #372)."""
    entry = _dat_entry("objects", object_id)
    return civ if civ is not None and entry is not None and str(civ) in entry.get("civ_icons", {}) else None


def is_building(object_id: int) -> bool:
    """derived_category()'s "Buildings" rule from the .dat row alone. GH #148:
    which portrait set an icon index points into (the game's own rule is not in the data)."""
    entry = _dat_entry("objects", object_id)
    return entry is not None and derived_category(entry) == "Buildings"


# Other-tab objects whose .dat icon index points into the building portraits: Granary
# (eye-candy type 10, same id in both games), which both editors draw with a building icon (GH #32).
_BUILDING_PORTRAIT_CONSTS = frozenset({1089})


def portrait_for(object_id: int, civ: int | None = None) -> tuple[int | None, bool]:
    """(icon_for(object_id, civ), is_building() or a _BUILDING_PORTRAIT_CONSTS
    member): asset_source.get_unit_icon()'s arguments."""
    return icon_for(object_id, civ), is_building(object_id) or object_id in _BUILDING_PORTRAIT_CONSTS


def _takes_install_name(section: str, id_: int, dat_entry: dict, lang: str) -> bool:
    # TASK-507: with no strings file of its own a game reads its fallback's, which fit only a row the fallback shares.
    fallback = game_dataset.game().fallback
    if fallback is None or game_dataset.has_own_strings(lang):
        return True
    base_row = _committed_table(fallback)[section].get(str(id_))
    if base_row is None:
        return False
    return (base_row.get("code"), base_row.get("string_id")) == (dat_entry.get("code"), dat_entry.get("string_id"))


def resolve_name(id_: int, library_name: str, dat_entry: dict | None, lang: str | None = None, *, section: str) -> str:
    """install string -> library enum name -> .dat short code -> "UNKNOWN_
    <id>", each step falling through to the next. Degrades to library_name
    alone with no install configured or no .dat entry for id_ at all --
    what keeps the default (install-free) test tier on the exact format
    resolve_reference()/the catalog always had before slice 4.

    Under a game with a fallback (Return of Rome) and no strings file of its
    own for lang, a row takes an install string only if the fallback's
    `section` row has the same id, code and string id.

    lang defaults to asset_source.get_language(), which parses config.yaml on
    every call; a bulk builder resolves it once and passes it in."""
    if dat_entry is not None:
        lang = lang or asset_source.get_language()
        install_name = asset_source.resource_string(dat_entry["string_id"], lang)
        if install_name and _takes_install_name(section, id_, dat_entry, lang):
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
    _read_table.cache_clear()


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
    if not game_dataset.game().library_enums:
        # The library's enums are AoE2's: under another game (Return of Rome) objects()' rows stand in (GH #67).
        return {entry.id: entry.name for entry in objects()}
    for dataset, _category in _OBJECT_DATASETS:
        for member in dataset:
            if member.ID in names:
                continue
            dat_entry = _dat_entry("objects", member.ID)
            names[member.ID] = resolve_name(member.ID, member.name.replace("_", " "), dat_entry, lang, section="objects")
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
    `internal` is a tech whose .dat code falls in INTERNAL_TECH_FAMILIES;
    only a picker that asks for it hides such a row.
    """

    id: int
    name: str
    category: str
    class_id: int | None
    hidden: bool
    search_text: str
    internal: bool = False


# GH #58: what a picker row for a unit only the active data mod's .dat defines ends in.
DATA_MOD_TAG = "(data mod)"


def _entry(id_: int, name: str, category: str, dat_entry: dict | None = None, internal: bool = False) -> CatalogEntry:
    return CatalogEntry(
        id=id_,
        name=name,
        category=category,
        class_id=dat_entry.get("class") if dat_entry else None,
        hidden=bool(dat_entry.get("hidden")) if dat_entry else False,
        search_text=f"{name.lower()} {id_}",
        internal=internal,
    )


def _internal_tech(dat_entry: dict | None) -> bool:
    # Fail open: no committed row, no code, nothing hidden.
    return bool(dat_entry) and bool(internal_tech_family(str(dat_entry.get("code") or "")))


@game_dataset.cache(maxsize=1)
def objects() -> tuple[CatalogEntry, ...]:
    """Every object across the four datasets, categorized by which one it
    came from, alphabetical by name. An id already claimed by an earlier
    dataset in _OBJECT_DATASETS is not repeated under a later category.

    Then every .dat-table id no dataset covers and the .dat does not hide
    from the editor (the Sept 2026 patch's Jarl, Longhouse, Spruce, ...),
    named install string -> .dat code and categorized by derived_category().
    Editor-hidden dat-only ids stay out (user decision 2026-09-27), except
    the game's _EDITOR_LISTED_BY_GAME entry, listed as not hidden under the
    official editor's category (GH #208). The game's _EDITOR_HIDDEN_BY_GAME ids
    are listed as hidden and its _EDITOR_LEFT_OUT_BY_GAME ids not at all; its
    _SHARED_HIDDEN_BY_GAME rows (Return of Rome's) join as hidden. Without
    library enums (Return of Rome) only the .dat rows are listed. A dat-only id
    the active data mod adds (GH #58) keeps its derived_category(), the tab the
    game's editor lists it under, and its row name ends in DATA_MOD_TAG while
    Settings > Mods shows that label (GH #265)."""
    from descape import data_mods, settings

    seen: set[int] = set()
    entries = []
    lang = asset_source.get_language()
    mod_only = data_mods.mod_only_ids() if settings.get_data_mod_label_shown() else frozenset()
    # Under Return of Rome the AoE2 enums and the AoE2 Gunpowder Keg id do not apply (GH #67).
    game = game_dataset.game()
    editor_listed = _EDITOR_LISTED_BY_GAME.for_game(game)
    editor_hidden = _EDITOR_HIDDEN_BY_GAME.for_game(game)
    left_out = _EDITOR_LEFT_OUT_BY_GAME.for_game(game)
    for dataset, category in _OBJECT_DATASETS if game.library_enums else ():
        for member in dataset:
            if member.ID in seen:
                continue
            seen.add(member.ID)
            dat_entry = _dat_entry("objects", member.ID)
            if dat_entry is not None and member.ID in editor_hidden:
                dat_entry = {**dat_entry, "hidden": True}
            name = resolve_name(member.ID, member.name.replace("_", " "), dat_entry, lang, section="objects")
            entries.append(_entry(member.ID, name, category, dat_entry))
    for key, raw_entry in _dat_table()["objects"].items():
        id_ = int(key)
        category = editor_listed.get(id_)
        if id_ in seen or (raw_entry.get("hidden") and category is None) or id_ in left_out:
            continue
        dat_entry = raw_entry if category is None else {**raw_entry, "hidden": False}
        if id_ in editor_hidden:
            dat_entry = {**dat_entry, "hidden": True}
        name = resolve_name(id_, "", dat_entry, lang, section="objects")
        if id_ in mod_only:
            name = f"{name} {DATA_MOD_TAG}"
        entries.append(_entry(id_, name, category or derived_category(dat_entry), dat_entry))
        seen.add(id_)
    for id_, category in _SHARED_HIDDEN_BY_GAME.for_game(game)().items():
        if id_ not in seen and id_ not in left_out:
            dat_entry = _dat_entry("objects", id_)
            entries.append(_entry(id_, resolve_name(id_, "", dat_entry, lang, section="objects"), category, dat_entry))
    return tuple(sorted(entries, key=lambda e: e.name))


@game_dataset.cache(maxsize=1)
def techs() -> tuple[CatalogEntry, ...]:
    """Every tech in TechInfo, alphabetical by name. Under a game without
    library enums (Return of Rome), every named tech of its own .dat instead (GH #67)."""
    entries = []
    lang = asset_source.get_language()
    if not game_dataset.game().library_enums:
        for key, dat_entry in _dat_table().get("techs", {}).items():
            name = resolve_name(int(key), "", dat_entry, lang, section="techs")
            entries.append(_entry(int(key), name, "Techs", internal=_internal_tech(dat_entry)))
        return tuple(sorted(entries, key=lambda e: e.name))
    for member in TechInfo:
        dat_entry = _dat_entry("techs", member.ID)
        name = resolve_name(member.ID, member.name.replace("_", " "), dat_entry, lang, section="techs")
        entries.append(_entry(member.ID, name, "Techs", internal=_internal_tech(dat_entry)))
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
    return resolve_name(id_, "", _dat_entry("objects", id_), lang, section="objects")


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
    return resolve_name(id_, "", _dat_entry("techs", id_), lang, section="techs")


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
