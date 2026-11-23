"""Loads real AoE2DE terrain textures from a configured game install, falling
back gracefully to terrain_palette's flat colors when no install is configured
or a given texture can't be found.

Proprietary game assets are never bundled in this repo -- only the (factual,
non-asset) terrain_id -> filename table in terrain_texture_map.json is
committed. This module is the one place that reaches into an actual install
directory to read the real .dds files at runtime.

Configure the install path via, in priority order:
  1. a runtime override set with set_install_path_override() -- the viewer's
     "Load" button uses this, so a path picked in the GUI takes effect
     immediately without restarting
  2. the AOE2DE_INSTALL_PATH environment variable
  3. an "aoe2de_install" key in config.yaml, at the OS-standard per-user
     config location (see CONFIG_PATH below). set_install_path_override()
     also persists here on success, so a path loaded once in the GUI is
     remembered next run.
"""

from __future__ import annotations

import codecs
import json
import locale
import os
import re
import shutil
import stat
import sys
import threading
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import platformdirs
import yaml

from descape import debug_log, game_dataset, games, perf_trace

APP_NAME = "DEscape"


def _default_config_path() -> Path:
    return Path(platformdirs.user_config_dir(APP_NAME, appauthor=False, roaming=True)) / "config.yaml"


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = _default_config_path()
LEGACY_CONFIG_PATH = REPO_ROOT / "config.yaml"

# Legacy migration is a source-distribution concept: under a freeze,
# LEGACY_CONFIG_PATH resolves to _MEIPASS/config.yaml, which never exists, so
# migrate_legacy_config() would silently no-op forever anyway -- IS_FROZEN
# makes that explicit rather than accidental.
IS_FROZEN = getattr(sys, "frozen", False)
TERRAIN_TEXTURE_SUBPATH = "resources/_common/terrain/textures/2x"

# GH #326: one terrain texture covers TERRAIN_TEXTURE_SPAN x TERRAIN_TEXTURE_SPAN tiles at
# every zoom, in every game. Measured 10 on an in-game farm, where the .dat's
# terrain_dimensions says 6; taken as every terrain's span (user decision 2026-11-20).
TERRAIN_TEXTURE_SPAN = 10
# The level get_terrain_texture_array() serves: render.SMALL_MAP_TILE_PIXELS, a small map's reference level.
TEXTURE_REFERENCE_TILE_PIXELS = 64


def terrain_texture_size(tile_px: int) -> int:
    """The side of the texture a level of `tile_px` crops its tiles from:
    TERRAIN_TEXTURE_SPAN tiles' worth, so a ground pattern covers the same
    tiles at every zoom (GH #326). Each level's texture is resized from the
    full source file, so it shows as much detail as the level has pixels
    (a 512 or 1024 px source is enlarged at the finer levels). Always a
    multiple of tile_px."""
    return TERRAIN_TEXTURE_SPAN * tile_px


def crop_pad(size: int, tile_px: int) -> int:
    """Rows and columns crop_source() wrap-pads a `size` square by: 0 where
    tile_px divides it, as it always does for terrain_texture_size(), else
    tile_px (a stand-in texture of its own size at the native tiers' 96 and
    192, TASK-599 and TASK-601). Exactly tile_px: the native kernel checks a
    layer mask's padded size against it, so each tier keeps its own copy."""
    return 0 if size % tile_px == 0 else tile_px


# (id(source), pad) -> (weakref to source, padded copy), for crop_source().
_wrap_padded: dict = {}
# Reentrant: the weakref callback below can run under GC inside a locked section.
_wrap_lock = threading.RLock()


def clear_wrap_padded() -> None:
    with _wrap_lock:
        _wrap_padded.clear()


def crop_source(arr, tile_px: int):
    """`arr` (a square texture or mask, period arr.shape[0]) as the array a
    tile_px crop at render._crop_offset()'s origin slices: itself where
    tile_px divides the period, else a read-only copy extended by tile_px
    rows and columns of its own wrap, so the crop never runs off the edge.
    Ground textures tile at their edges, so the wrap is seamless. Only a
    stand-in texture (tests, review packs) still pads: terrain_texture_size()
    always divides. Cached per source array while it lives, so a stand-in
    pads its own pixels."""
    pad = crop_pad(arr.shape[0], tile_px)
    if pad == 0:
        return arr
    key = (id(arr), pad)
    with _wrap_lock:
        hit = _wrap_padded.get(key)
        if hit is not None and hit[0]() is arr:
            return hit[1]
    import weakref

    import numpy as np

    padded = np.pad(arr, ((0, pad), (0, pad)) + ((0, 0),) * (arr.ndim - 2), mode="wrap")
    padded.flags.writeable = False

    def _drop(ref, key=key):
        with _wrap_lock:
            if _wrap_padded.get(key, (None,))[0] is ref:
                del _wrap_padded[key]

    with _wrap_lock:
        _wrap_padded[key] = (weakref.ref(arr, _drop), padded)
    return padded

# TASK-143 terrain blending. blendomatic_x1.dat holds the neighbour-transition
# alpha masks; terrain/masks/*.png the layer ("Layering Mode") overlay masks.
BLENDOMATIC_SUBPATH = "resources/_common/dat/blendomatic_x1.dat"
TERRAIN_MASK_SUBPATH = "resources/_common/terrain/masks"
# Each blendomatic tile is a 97x49 diamond stored row by row, row r holding
# 4*min(r, 48 - r) + 1 pixels centred in the 97 columns: 2353 in all.
BLENDOMATIC_ROWS = 49
BLENDOMATIC_WIDTH = 97
BLENDOMATIC_TILE_SIZE = 2353

# key-value-strings-utf8.txt is a flat "<int key> \"<value>\"" list per line
# (comments and blank lines interspersed) -- the same key space
# gen_object_catalog.py commits as string_id. Backslash-escaped quotes are
# the only escape this file actually uses (verified against the real file:
# 109 of ~19k parseable lines carry one, e.g. Sent to \"%s\":).
_STRING_LINE = re.compile(r'^(\d+)\s+"((?:[^"\\]|\\.)*)"')
STRINGS_SUBPATH_TEMPLATE = "resources/{lang}/strings/key-value/key-value-strings-utf8.txt"
# GH #67: Return of Rome's own strings file under its root; _string_table() finds it through games.ROR.strings_files.
ROR_STRINGS_SUBPATH_TEMPLATE = "resources/{lang}/strings/key-value/key-value-pompeii-strings-utf8.txt"
DEFAULT_LANGUAGE = "en"

# GH #140: a unit's in-game portrait, by object_catalog.icon_for() index. Uncompressed RGBA or DXT5, 256 px.
UNIT_ICON_DIR = "widgetui/textures/ingame/units"
UNIT_ICON_NAME = "{icon:03d}_50730.dds"
# GH #148: buildings index a separate set, named "{icon:03d}_<name>.dds" (007_castle.DDS).
BUILDING_ICON_DIR = "widgetui/textures/ingame/buildings"
# GH #67: Return of Rome renumbers portraits. Its icons.json maps an index to an atlas RefName, which
# its morematerials.json (RoR's own files) or the install's materials.json (AoE2's) turns into a file.
UI_DIR = "widgetui"
ICON_TABLE_NAME = "icons.json"
ROR_MATERIALS_NAME = "morematerials.json"
MATERIALS_NAME = "materials.json"
# GH #230: tech icons, named "{icon:03d}_<name>.DDS" like buildings (006_loom.DDS).
TECH_ICON_DIR = "widgetui/textures/ingame/tech"
# GH #148: the stat icons the game's editorbottomunitspanel.json uses, as plain 66 px RGBA PNGs.
STAT_ICON_DIR = "widgetui/textures/ingame/staticons"
# GH #20: particle effects, one <name>.json each. An effect's AtlasFile is relative to this folder.
PARTICLE_SUBPATH = "resources/_common/particles"

_override_path: Path | None = None

# The content layers the file-, path- and install-keyed caches do not read: a switch naming only these keeps them (TASK-478).
FILE_KEYED_SURVIVES = frozenset({game_dataset.TABLES})


def _terrain_texture_map() -> dict[int, str]:
    return _terrain_texture_map_for(game_dataset.table_path(REPO_ROOT / "descape" / "terrain_texture_map.json"))


@lru_cache(maxsize=2)
def _terrain_texture_map_for(path: Path) -> dict[int, str]:
    data = json.loads(path.read_text())
    return {int(k): v for k, v in data["mapping"].items()}


@lru_cache(maxsize=1)
def get_install_path() -> Path | None:
    if _override_path is not None:
        return _override_path

    env_path = os.environ.get("AOE2DE_INSTALL_PATH")
    if env_path:
        p = Path(env_path)
        return p if p.is_dir() else None

    if CONFIG_PATH.is_file():
        try:
            config = yaml.safe_load(read_config_text(CONFIG_PATH)) or {}
        except (yaml.YAMLError, OSError):
            return None
        raw = config.get("aoe2de_install")
        if raw:
            p = Path(raw)
            return p if p.is_dir() else None

    return None


def is_available() -> bool:
    return get_install_path() is not None


def validate_install_path(path: Path) -> tuple[bool, str]:
    """Checks whether `path` looks like a real AoE2DE install (specifically,
    that it has the terrain textures this tool actually uses). Returns
    (ok, message) -- message is a human-readable success/error description,
    meant to go straight into a status label."""
    if not path.is_dir():
        return False, f"Not a directory: {path}"

    tex_dir = path / TERRAIN_TEXTURE_SUBPATH
    if not tex_dir.is_dir():
        return False, f"Not an AoE2DE install -- missing {TERRAIN_TEXTURE_SUBPATH}"

    count = sum(1 for _ in tex_dir.glob("*.dds"))
    if count == 0:
        return False, f"Found {TERRAIN_TEXTURE_SUBPATH}, but no .dds textures in it"

    return True, f"Loaded: {count} terrain textures found"


def set_install_path_override(path: Path | None) -> None:
    """Sets the runtime install path (or clears it, with None) and drops every
    cache downstream of it, so the next lookup reflects the change immediately
    instead of returning stale pre-override results."""
    global _override_path
    _override_path = path
    clear_install_caches()


def clear_art_caches(changed: frozenset[str] | None = None) -> None:
    """Drops every cache holding sprite, terrain-texture or portrait art, for a
    change in which files draw it (mod art toggled or opted out, TASK-378; a
    data mod's portraits, GH #363), and
    bumps install_generation() so ThumbLoader-keyed thumbnails reload.
    `changed` names the content layers that moved (game_dataset.TABLES,
    ROOTS; None is everything). TABLES alone (a data-mod switch, TASK-478)
    drops only what a terrain id or const keys, keeps the file-keyed art and
    leaves install_generation()."""
    keeps_files = game_dataset.only_changed(changed, FILE_KEYED_SURVIVES)
    get_terrain_texture_path.cache_clear()
    get_terrain_average_color.cache_clear()
    # Only file-, path- or install-keyed caches belong in this block (TASK-478).
    if not keeps_files:
        # Path-keyed rather than id-keyed, so a changed install that resolves to
        # a same-named file under a different root would otherwise keep serving
        # the old swatch or texture.
        _terrain_thumbnail_for_path.cache_clear()
        _forget_prefetches()
    # Imported here, not at module scope: unit_sprites imports this module, so
    # a top-level import would be a cycle. Its caches remember MISSES as well
    # as sprites -- deliberately, since re-deriving one costs a whole file walk
    # -- so without this, configuring an install for the first time would leave
    # every unit drawn as a coloured dot until the app was restarted.
    from descape import terrain_blend, unit_sprites

    unit_sprites.clear_caches(changed)
    # Blended blocks are built from the textures above.
    terrain_blend.clear_caches()
    # GH #363: with a data mod, it and Mod Art's mods supply portraits, so these drop under TABLES too.
    _layered_icon_files.cache_clear()
    get_unit_icon.cache_clear()


def clear_install_caches(changed: frozenset[str] | None = None) -> None:
    """Drops every cache downstream of the install path, leaving the override
    as it is. The test suite calls this before each test (see tests/conftest.py).
    `changed` is the layers a game_dataset switch moved; None (an install or
    profile change) is everything. A data-mod switch names TABLES alone
    (TASK-478): the art and other file- or install-keyed caches stay,
    everything keyed through the tables drops."""
    keeps_files = game_dataset.only_changed(changed, FILE_KEYED_SURVIVES)
    # Only file-, path- or install-keyed caches belong in these blocks (TASK-478).
    if not keeps_files:
        get_install_path.cache_clear()
        from descape import mod_graphics

        # Its palettes come from the install, and its mod folders from the profile derived from it.
        mod_graphics.clear()
    clear_art_caches(changed)
    if not keeps_files:
        get_blendomatic.cache_clear()
        get_blend_mask.cache_clear()
        get_terrain_mask_array.cache_clear()
        _sized_terrain_mask.cache_clear()
        _masks_asked.clear()
        clear_wrap_padded()
        _icon_table_files.cache_clear()
        get_unit_icon.cache_clear()
        get_tech_icon.cache_clear()
        get_stat_icon.cache_clear()
        # GH #140's instruction text colours, read from the install's UIColors.json.
        from descape import message_markup

        message_markup.clear_caches()
    _string_table.cache_clear()
    _data_mod_string_ids.cache_clear()
    _data_mod_changed_string_ids.cache_clear()
    game_dataset.clear_caches()
    # object_catalog.py's name resolution reads resource_string() above --
    # same reasoning, a different module. Imported here for the same
    # cycle-avoidance reason (object_catalog.py imports this module).
    from descape import object_catalog

    object_catalog.clear_caches()


def get_language() -> str:
    """config.yaml's "language" key -- an install language folder name
    under resources/ (e.g. "en", "de", "fr") -- or DEFAULT_LANGUAGE if unset
    or unreadable. Not cached, unlike get_install_path(): caching it here would
    add another cache set_install_path_override() has no real reason to know
    about clearing. Each call is a full YAML parse (~3 ms on a real config),
    so object_catalog.py's bulk builders call it once per build and pass the
    result down rather than once per row."""
    if CONFIG_PATH.is_file():
        try:
            config = yaml.safe_load(read_config_text(CONFIG_PATH)) or {}
        except (yaml.YAMLError, OSError):
            return DEFAULT_LANGUAGE
        raw = config.get("language")
        if raw:
            return str(raw)
    return DEFAULT_LANGUAGE


def _unescape_string_value(text: str) -> str:
    return text.replace('\\"', '"').replace("\\\\", "\\")


@game_dataset.cache(maxsize=4)
def _string_table(lang: str) -> dict[int, str]:
    """The strings for lang, merged key by key along the content stack
    (game_dataset.ContentStack.strings_paths()): the install's, under Return
    of Rome its own file over them (GH #67), and under a data mod the mod's
    files over those (GH #58). Empty without an install."""
    install = get_install_path()
    if install is None:
        return {}
    table: dict[int, str] = {}
    for path in game_dataset.stack().strings_paths(install, lang):
        for match in _string_lines(path):
            table[int(match.group(1))] = _unescape_string_value(match.group(2))
    return table


def _data_mod_strings_paths(lang: str) -> list[Path]:
    """The active data mod's strings files for lang (GH #58), existing or not; [] with no mod."""
    return game_dataset.stack().mod_strings_paths(lang)


def _string_lines(path: Path) -> list[re.Match]:
    """Every _STRING_LINE match in a strings file; [] when it does not exist."""
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return [match for match in map(_STRING_LINE.match, lines) if match is not None]


@game_dataset.cache(maxsize=4)
def _data_mod_string_ids(lang: str) -> frozenset[int]:
    return frozenset(int(match.group(1)) for path in _data_mod_strings_paths(lang) for match in _string_lines(path))


def data_mod_string_ids(lang: str | None = None) -> frozenset[int]:
    """The string keys the active data mod's own strings files define (GH #265),
    install or not; empty with no data mod. lang defaults to get_language()."""
    return _data_mod_string_ids(lang or get_language())


@game_dataset.cache(maxsize=4)
def _data_mod_changed_string_ids(lang: str) -> frozenset[int]:
    install = get_install_path()
    base = {} if install is None else {
        int(m.group(1)): m.group(2) for m in _string_lines(install / STRINGS_SUBPATH_TEMPLATE.format(lang=lang))
    }
    mod = {int(m.group(1)): m.group(2) for path in _data_mod_strings_paths(lang) for m in _string_lines(path)}
    return frozenset(key for key, text in mod.items() if base.get(key) != text)


def data_mod_changed_string_ids(lang: str | None = None) -> frozenset[int]:
    """data_mod_string_ids() minus the keys whose text equals the install's
    (GH #271): a mod shipping a full copy of the strings file renames only
    what it changed. Every mod key without an install."""
    return _data_mod_changed_string_ids(lang or get_language())


def resource_string(key: int, lang: str | None = None) -> str | None:
    """The install's own display string for key (a language_dll_name value
    from empires2_x2_p1.dat), or None if unavailable -- no install
    configured, no strings file for this language, or no entry for this key.
    lang defaults to get_language()."""
    return _string_table(lang or get_language()).get(key)


def _find_file(folder: Path, match) -> Path | None:
    try:
        return next((p for p in sorted(folder.iterdir()) if match(p.name.lower())), None)
    except OSError:
        return None


def unit_icon_path(icon: int | None, building: bool = False) -> Path | None:
    """The portrait file for a catalog icon index, or None. A building looks
    in BUILDING_ICON_DIR by its "NNN_" prefix first, then falls back to the
    unit set. Extension case varies, so names are matched loosely. With a data
    mod in effect, its portrait_layers() come before the install (GH #363)."""
    return unit_icon_path_in(get_install_path(), icon, building)


def portrait_layers() -> tuple[Path, ...]:
    """The folders a portrait is looked up in before the install (GH #363): the
    data mod's own, then the ones Mod Art draws, in its order; () with no data
    mod in effect (Return of Rome included). GUI thread only."""
    mod = game_dataset.data_mod()
    if mod is None:
        return ()
    from descape import mod_graphics

    return (mod.root, *(root for root in mod_graphics.drawn_roots() if root != mod.root))


def unit_icon_path_in(
    install: Path | None, icon: int | None, building: bool = False, layers: tuple[Path, ...] | None = None
) -> Path | None:
    """unit_icon_path() under an install the caller passes, never read here:
    the Units catalog's thumbnail worker resolves it on the GUI thread (GH #68).
    layers are portrait_layers()'s folders, read here when None (GH #363)."""
    if install is None or icon is None or icon < 0:
        return None
    section = "Buildings" if building else "Units"
    stack = game_dataset.stack()
    if stack.game.icon_materials is not None:
        path = _icon_table_files(install, stack.game).get((section, icon))
        if path is not None:
            return path
    layers = portrait_layers() if layers is None else tuple(layers)
    path = _layered_icon_files(layers, install).get((section, icon)) if layers else None
    if path is not None:
        return path
    # GH #67: without its icon table, the game's own icon set first, then each fallback's.
    # GH #363: a data mod's and its art mods' loose portraits beat those by name.
    roots = [*layers, *stack.roots(install)]
    if building:
        prefix = f"{icon:03d}_"
        for root in roots:
            path = _find_file(root / BUILDING_ICON_DIR, lambda name: name.startswith(prefix) and name.endswith(".dds"))
            if path is not None:
                return path
    wanted = UNIT_ICON_NAME.format(icon=icon)
    for root in roots:
        folder = root / UNIT_ICON_DIR
        path = folder / wanted
        path = path if path.is_file() else _find_file(folder, lambda name: name == wanted)
        if path is not None:
            return path
    return None


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _atlas_files(materials: dict) -> dict[str, str]:
    """RefName -> FileName (relative to widgetui/) for every atlas texture a materials file defines.
    Wrong-shaped entries are skipped: a data mod's file is untrusted (GH #363)."""
    files = {}
    atlases = materials.get("AtlasTextures")
    for atlas in atlases if isinstance(atlases, list) else ():
        definition = atlas.get("AtlasDef") if isinstance(atlas, dict) else None
        textures = definition.get("Textures") if isinstance(definition, dict) else None
        for texture in textures if isinstance(textures, list) else ():
            if isinstance(texture, dict) and isinstance(texture.get("RefName"), str) and isinstance(texture.get("FileName"), str):
                files[texture["RefName"]] = texture["FileName"]
    return files


@lru_cache(maxsize=2)
def _icon_table_files(install: Path, game: games.Game) -> dict[tuple[str, int], Path]:
    """("Units", "Buildings" or "Techs", icon index) -> its icon file, through `game`'s own icons.json (GH #67).
    A RefName the game's Game.icon_materials file defines is one of its own files, else the first
    fallback whose materials.json defines it has the file. Only existing files; {} without the icon table."""
    own_ui = game.root(install) / UI_DIR
    table = _read_json(own_ui / ICON_TABLE_NAME)
    if not table:
        return {}
    sources = [(own_ui, _atlas_files(_read_json(own_ui / game.icon_materials)))]
    for of in games.chain(game)[1:]:
        ui = of.root(install) / UI_DIR
        sources.append((ui, _atlas_files(_read_json(ui / MATERIALS_NAME))))
    files = {}
    for section in ("Units", "Buildings", "Techs"):
        for key, ref in (table.get(section) or {}).items():
            if not (key.isascii() and key.isdigit()):
                continue
            for root, refs in sources:
                if ref in refs:
                    path = root / refs[ref]
                    wanted = path.name.lower()
                    path = path if path.is_file() else _find_file(path.parent, lambda name, w=wanted: name == w)
                    if path is not None:
                        files[(section, int(key))] = path
                    break
    return files


@lru_cache(maxsize=2)
def _layered_icon_files(layers: tuple[Path, ...], install: Path) -> dict[tuple[str, int], Path]:
    """("Units" or "Buildings", icon index) -> its portrait file through a data mod's icon table (GH #363).
    As in the game's file overlay, icons.json, morematerials.json and materials.json each come whole
    from the first of `layers` with a readable one (materials from the install last); a RefName resolves
    through morematerials, then materials, and its file comes from the first folder that has it, the
    install last. widgetui/ and file names match case-insensitively, as the side_key fingerprint does,
    and wrong-shaped entries are skipped. Only existing files; {} when no layer has an icon table."""
    from descape import content_roots

    layer_uis = [ui for ui in (content_roots.child_dir(root, UI_DIR) for root in layers) if ui is not None]
    install_ui = content_roots.child_dir(install, UI_DIR)
    uis = [*layer_uis, *([install_ui] if install_ui is not None else [])]
    listings: dict[Path, dict[str, Path]] = {}

    def first(folders, name: str) -> dict:
        """The first folder's readable `name`, whole; {} when none has one."""
        for folder in folders:
            path = _ui_file(folder, name, listings)
            data = _read_json(path) if path is not None else {}
            if data:
                return data
        return {}

    table = first(layer_uis, ICON_TABLE_NAME)
    if not table:
        return {}
    # morematerials second, so its RefNames win.
    refs = {**_atlas_files(first(uis, MATERIALS_NAME)), **_atlas_files(first(uis, ROR_MATERIALS_NAME))}
    files = {}
    for section in ("Units", "Buildings"):
        entries = table.get(section)
        for key, ref in entries.items() if isinstance(entries, dict) else ():
            if not (key.isascii() and key.isdigit() and isinstance(ref, str) and ref in refs):
                continue
            path = next((p for p in (_ui_file(ui, refs[ref], listings) for ui in uis) if p is not None), None)
            if path is not None:
                files[(section, int(key))] = path
    return files


def _ui_file(ui: Path, rel: str, listings: dict[Path, dict[str, Path]]) -> Path | None:
    """ui/<rel>, each part matched case-insensitively (listings memoizes each folder's names,
    first by sorted name), or None. A rel that is absolute or climbs out with ".." is None."""
    parts = Path(rel).parts
    if not parts or Path(rel).is_absolute() or ".." in parts:
        return None
    found: Path | None = ui
    for part in parts:
        names = listings.get(found)
        if names is None:
            names = listings[found] = {}
            try:
                for entry in sorted(found.iterdir()):
                    names.setdefault(entry.name.lower(), entry)
            except OSError:
                pass
        found = names.get(part.lower())
        if found is None:
            return None
    return found if found.is_file() else None


def read_unit_icon_in(
    install: Path | None,
    icon: int | None,
    building: bool = False,
    px: int | None = None,
    layers: tuple[Path, ...] | None = None,
):
    """get_unit_icon() under a passed install and portrait layers, uncached and
    thread-safe when layers is given, downscaled to fit px x px when px is given; or None.
    Colour and alpha are downscaled apart: an RGBA resize premultiplies, which
    would lose the shading DE keeps under a portrait's transparent player-colour
    region (catalog_thumbs.tinted(), GH #372)."""
    arr = _read_rgba(unit_icon_path_in(install, icon, building, layers))
    if arr is None or px is None:
        return arr
    import numpy as np
    from PIL import Image

    rgb = Image.fromarray(np.ascontiguousarray(arr[..., :3]), "RGB")
    alpha = Image.fromarray(np.ascontiguousarray(arr[..., 3]), "L")
    rgb.thumbnail((px, px), Image.LANCZOS)
    alpha.thumbnail((px, px), Image.LANCZOS)
    return np.dstack((np.array(rgb), np.array(alpha)))


# A byte mask of an uncompressed DDS pixel format -> the byte it selects.
_DDS_BYTE_MASKS = {0xFF: 0, 0xFF00: 1, 0xFF0000: 2, 0xFF000000: 3}
_DDPF_ALPHAPIXELS = 0x1
_DDPF_RGB = 0x40
_DDS_HEADER_BYTES = 128


def _read_dds_rgba_fast(path: Path):
    """An uncompressed 32-bit DDS as RGBA through numpy, or None for any other
    layout (Pillow decodes those). Pillow reads this layout with a pure-Python
    decoder, about 60 ms per 256 px portrait and holding the GIL the whole
    time, so a decode on the catalog's worker thread froze the GUI anyway
    (measured, GH #68). Two thirds of the install's portraits are this layout."""
    import struct

    import numpy as np

    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) < _DDS_HEADER_BYTES or data[:4] != b"DDS ":
        return None
    height, width = struct.unpack_from("<2I", data, 12)
    flags, fourcc, bits, *masks = struct.unpack_from("<I4s5I", data, 80)
    if not flags & _DDPF_RGB or bits != 32 or fourcc != b"\0\0\0\0":
        return None
    count = width * height * 4
    if width == 0 or height == 0 or len(data) < _DDS_HEADER_BYTES + count:
        return None
    channels = [_DDS_BYTE_MASKS.get(mask) for mask in masks[:3]]
    if None in channels:
        return None
    pixels = np.frombuffer(data, np.uint8, count, _DDS_HEADER_BYTES).reshape(height, width, 4)
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    for out, byte in enumerate(channels):
        rgba[..., out] = pixels[..., byte]
    alpha = _DDS_BYTE_MASKS.get(masks[3]) if flags & _DDPF_ALPHAPIXELS else None
    rgba[..., 3] = 255 if alpha is None else pixels[..., alpha]
    return rgba


def _read_rgba(path: Path | None):
    """path decoded as a read-only (h, w, 4) uint8 RGBA array, or None."""
    if path is None:
        return None
    import numpy as np
    from PIL import Image

    if path.suffix.lower() == ".dds":
        arr = _read_dds_rgba_fast(path)
        if arr is not None:
            arr.flags.writeable = False
            return arr
    try:
        with Image.open(path) as img:
            arr = np.array(img.convert("RGBA"))
    except (OSError, ValueError):
        return None
    arr.flags.writeable = False
    return arr


@lru_cache(maxsize=64)
def get_unit_icon(icon: int | None, building: bool = False):
    """The portrait for this catalog icon index (unit_icon_path(), a data mod's
    first, GH #363) as a read-only (h, w, 4) uint8 RGBA array, or None (no
    install, no file, a decode failure)."""
    return _read_rgba(unit_icon_path(icon, building))


def tech_icon_path(icon: int | None) -> Path | None:
    """The tech icon file for a catalog tech icon index, or None: through the
    active game's icon table as unit_icon_path_in() does, then by its "NNN_"
    prefix in the game's own root and each fallback's (GH #67)."""
    install = get_install_path()
    if install is None or icon is None or icon < 0:
        return None
    stack = game_dataset.stack()
    if stack.game.icon_materials is not None:
        path = _icon_table_files(install, stack.game).get(("Techs", icon))
        if path is not None:
            return path
    prefix = f"{icon:03d}_"
    for root in stack.roots(install):
        path = _find_file(root / TECH_ICON_DIR, lambda name: name.startswith(prefix) and name.endswith(".dds"))
        if path is not None:
            return path
    return None


@lru_cache(maxsize=64)
def get_tech_icon(icon: int | None):
    """tech_icon_path() as get_unit_icon()'s read-only RGBA array, or None."""
    return _read_rgba(tech_icon_path(icon))


@lru_cache(maxsize=16)
def get_stat_icon(name: str):
    """STAT_ICON_DIR/<name>.png ("hp", "damage", ...) as get_unit_icon()'s
    read-only RGBA array, or None."""
    folder = game_dataset.resource_path(STAT_ICON_DIR)
    if folder is None:
        return None
    wanted = f"{name}.png".lower()
    path = folder / f"{name}.png"
    return _read_rgba(path if path.is_file() else _find_file(folder, lambda n: n == wanted))


def read_config_text(path: Path) -> str:
    """A config file's text as UTF-8, a leading BOM dropped, whatever the
    locale: Windows' cp1252 would mangle a hand-edited non-ASCII path. A file
    that is not UTF-8 (an older Notepad's ANSI save) reads as the locale's,
    or as cp1252 under a UTF-8 locale, so a later save keeps its characters."""
    return decode_config_bytes(path.read_bytes())


def decode_config_bytes(data: bytes) -> str:
    """read_config_text() for bytes already read."""
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        encoding = locale.getpreferredencoding(False)
        if codecs.lookup(encoding).name == "utf-8":
            encoding = "cp1252"
        return data.decode(encoding, errors="replace")


def write_config_file(path: Path, config: dict) -> None:
    """Atomically writes `config`, keeping a symlink's target and its mode,
    and creating missing parents. Takes the path explicitly rather than
    reading this module's CONFIG_PATH, so settings.py can pass its own
    (separately monkeypatched in tests) global through."""
    # yaml_file_io imports scenario_io, which reaches this module through content_roots.
    from descape.yaml_file_io import atomic_write

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        mode = None
    atomic_write(path, yaml.safe_dump(config, default_flow_style=False, sort_keys=False), mode=mode, create_mode=0o666)


def migrate_legacy_config() -> Path | None:
    """One-time copy of a pre-relocation repo-root config.yaml to the
    OS-standard location, idempotent: returns the new path only when it
    actually copied something, None otherwise (including when there was
    nothing to migrate). The legacy file is copied, not moved -- it is the
    user's data, and deleting it is destructive for no gain; it is simply
    ignored from then on. Must be called explicitly at the GUI entry point,
    never at import time (see main()'s own call site for why)."""
    if IS_FROZEN:
        return None
    if CONFIG_PATH.exists() or not LEGACY_CONFIG_PATH.is_file():
        return None
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(LEGACY_CONFIG_PATH, CONFIG_PATH)
    except OSError:
        return None  # read-only home, permissions: degrade to defaults, never crash
    return CONFIG_PATH


def save_install_path_to_config(path: Path | None) -> None:
    """Persists `path` to config.yaml (read-modify-write, so any other keys
    a future version adds there survive); None removes the key."""
    config = {}
    if CONFIG_PATH.is_file():
        try:
            config = yaml.safe_load(read_config_text(CONFIG_PATH)) or {}
        except (yaml.YAMLError, OSError):
            config = {}
    if path is None:
        config.pop("aoe2de_install", None)
    else:
        config["aoe2de_install"] = str(path)
    write_config_file(CONFIG_PATH, config)


def save_profile_path_to_config(path: Path | None) -> None:
    """Persists the AoE2:DE profile folder (content_roots.PROFILE_CONFIG_KEY) the
    same read-modify-write way; None removes the key, back to auto-detect."""
    config = {}
    if CONFIG_PATH.is_file():
        try:
            config = yaml.safe_load(read_config_text(CONFIG_PATH)) or {}
        except (yaml.YAMLError, OSError):
            config = {}
    if path is None:
        config.pop("aoe2de_profile", None)
    else:
        config["aoe2de_profile"] = str(path)
    write_config_file(CONFIG_PATH, config)


def undrawable_terrain_textures() -> list[str]:
    """Every texture file the terrain table in effect names that neither the
    install's terrain/textures/2x (one folder listing) nor an active mod
    provides, sorted: those terrains draw their flat colour (GH #272's status
    line). [] without an install or a folder that will not list."""
    install = get_install_path()
    if install is None:
        return []
    from descape import mod_graphics

    try:
        with os.scandir(game_dataset.resource_path_in(install, TERRAIN_TEXTURE_SUBPATH)) as entries:
            have = {entry.name.casefold() for entry in entries}
    except OSError:
        return []
    names = set(_terrain_texture_map().values())
    return sorted(n for n in names if n.casefold() not in have and mod_graphics.resolve_terrain_texture(n) is None)


@lru_cache(maxsize=256)
def get_terrain_texture_path(terrain_id: int) -> Path | None:
    """Returns the real .dds path for a terrain_id, or None if unavailable --
    either no install is configured, or this terrain_id has no known texture.
    An active mod's texture of the same name wins (mod_graphics, TASK-378)."""
    install = get_install_path()
    if install is None:
        return None
    filename = _terrain_texture_map().get(terrain_id)
    if filename is None:
        return None
    from descape import mod_graphics

    mod_path = mod_graphics.resolve_terrain_texture(filename)
    if mod_path is not None:
        return mod_path
    path = game_dataset.resource_path_in(install, f"{TERRAIN_TEXTURE_SUBPATH}/{filename}")
    return path if path.is_file() else None


@lru_cache(maxsize=256)
def get_terrain_average_color(terrain_id: int) -> tuple[int, int, int] | None:
    """Average color of the real texture for terrain_id, or None if unavailable.
    Cheap way to get authentic-ish colors without a full texture-blitting
    renderer.

    One definition, _average_rgb() of the full-size decode. The textured
    renderer's first level load of a file records it (_averages), so a Copy
    Region thumbnail of a terrain on screen costs no decode; otherwise
    (textures off) this decodes the file once, ~15 ms. A terrain with no file
    averages get_terrain_texture_array()'s stand-in, if a test serves one."""
    from PIL import Image

    path = get_terrain_texture_path(terrain_id)
    if path is None:
        arr = get_terrain_texture_array(terrain_id)
        return None if arr is None else _average_rgb(Image.fromarray(arr))
    with _prefetch_lock:
        known = _averages.get(path)
        generation = _generation
    if known is not None:
        return known

    def decode(file: Path):
        with Image.open(file) as img:
            return _average_rgb(img.convert("RGB"))

    rgb = _decode_with_install_fallback(path, decode)
    if rgb is not None:
        with _prefetch_lock:
            if generation == _generation:
                _averages.setdefault(path, rgb)
    return rgb


def _average_rgb(image) -> tuple[int, int, int]:
    """A PIL RGB image's average colour: resized to 32x32 (sampling is plenty
    for an average), then the integer floor mean per channel."""
    import numpy as np

    pixels = np.asarray(image.resize((32, 32))).reshape(-1, 3)
    r, g, b = (int(total) // len(pixels) for total in pixels.sum(axis=0, dtype=np.int64))
    return (r, g, b)


# A browser swatch is one tile at default graphics quality, so it matches
# render.SMALL_MAP_TILE_PIXELS. Not imported from render.py: this module is
# below it in the import order and has no other reason to depend on it.
TERRAIN_THUMBNAIL_PX = 64


@lru_cache(maxsize=256)
def _terrain_thumbnail_for_path(path: Path, px: int):
    """The open+resize half of get_terrain_thumbnail, keyed on the FILE, not
    on a terrain_id. 134 terrains share only 85 textures -- g_for.dds alone
    backs 14 of them -- so an id-keyed cache would decode the same .dds a
    dozen times over."""
    import numpy as np
    from PIL import Image

    def decode(file: Path):
        with Image.open(file) as img:
            return np.array(img.convert("RGB").resize((px, px)))

    return _decode_with_install_fallback(path, decode)


# A mod .dds Pillow cannot read: an unsupported DXGI/FourCC format, or a truncated download.
_TEXTURE_DECODE_ERRORS = (OSError, ValueError, NotImplementedError, SyntaxError)


def _decode_with_install_fallback(path: Path, decode):
    """decode(path), or for a mod texture that will not decode (TASK-378), the
    install's file of the same name, logged. With no install file of that name
    (a data mod's own texture, GH #272) None, logged, so the terrain draws its
    flat colour. An install file's error raises as before."""
    try:
        return decode(path)
    except _TEXTURE_DECODE_ERRORS as exc:
        install = get_install_path()
        # The active dataset's copy (the RoR root's under Return of Rome), so an install file is never mistaken for a mod's.
        fallback = game_dataset.resource_path_in(install, f"{TERRAIN_TEXTURE_SUBPATH}/{path.name}")
        if fallback == path:
            raise
        if fallback is None or not fallback.is_file():
            debug_log.log(f"mod art: texture {path} unreadable ({exc!r}); the install has none of that name", level=debug_log.WARNING)
            return None
        debug_log.log(f"mod art: texture {path} unreadable ({exc!r}); drawing the install's", level=debug_log.WARNING)
        return decode(fallback)


def get_terrain_thumbnail(terrain_id: int, px: int = TERRAIN_THUMBNAIL_PX):
    """A (px, px, 3) uint8 array of terrain_id's real texture, or None when
    no install is configured or the id has no mapped texture -- the same
    degradation contract every other function here has, so a caller falls
    back to terrain_palette.color_for_terrain_id.

    Deliberately NOT built from get_terrain_texture_array(): that cache
    holds a level-sized array per file (640 px at the reference level), so
    filling a browser would warm ~100MB in the renderer's own cache for
    textures the open map may never touch. At 64px the whole catalog is
    ~1.6MB, and only the 85 distinct files are ever decoded."""
    path = get_terrain_texture_path(terrain_id)
    if path is None:
        return None
    return _terrain_thumbnail_for_path(path, px)


def get_terrain_texture_array(terrain_id: int):
    """terrain_id's texture at the reference level, terrain_texture_size(
    TEXTURE_REFERENCE_TILE_PIXELS), as a read-only (side, side, 3) uint8
    array, or None when no install is configured or the id has no texture.
    Also the stand-in seam: tests and review packs replace it to serve a
    texture with no file behind it, which terrain_texture_at() then serves
    at the stand-in's own size."""
    path = get_terrain_texture_path(terrain_id)
    if path is None:
        return None
    return _texture_for_path(path, terrain_texture_size(TEXTURE_REFERENCE_TILE_PIXELS))


def terrain_texture_at(terrain_id: int, size: int):
    """terrain_id's texture at `size` (terrain_texture_size()), resized from
    the full source file, or None. A terrain with no file gets whatever
    get_terrain_texture_array() serves: None, or a stand-in at its own size."""
    path = get_terrain_texture_path(terrain_id)
    if path is None:
        return get_terrain_texture_array(terrain_id)
    return _texture_for_path(path, size)


def _texture_for_path(path: Path, size: int):
    """One file's texture at `size`, keyed on the FILE like
    _terrain_thumbnail_for_path: a map's 37 ids resolve to 26 files. Read-only,
    since ids sharing a file share the array; consumers only crop views.
    Takes a prefetched load's result (waiting if it is still running) rather
    than decoding the file a second time; a worker's exception raises here.
    Cached in _textures under TEXTURE_CACHE_BUDGET_BYTES (_keep_texture())."""
    key = (path, size)
    with _prefetch_lock:
        if key in _textures:
            _textures.move_to_end(key)
            return _textures[key]
        entry = _pending.pop(key, None)
        generation = _generation
    if entry is not None and entry[0] == generation:
        with perf_trace.texture_wait():
            arr, average = entry[1].result()
    else:
        with perf_trace.texture_load():
            arr, average = _load_texture(path, size)
    with _prefetch_lock:
        if generation == _generation:
            _keep_texture(key, arr)
            if average is not None:
                _averages.setdefault(path, average)
    return arr


def _keep_texture(key: tuple[Path, int], arr) -> None:
    """Caches arr (None for a file that will not decode, so it is not retried)
    as the most recently used entry, then drops least recently used entries
    until the cache is back within TEXTURE_CACHE_BUDGET_BYTES, never the new
    one (GH #326: ~500 MB if a big map visits every level). Under _prefetch_lock."""
    global _texture_bytes
    old = _textures.pop(key, None)
    _texture_bytes -= 0 if old is None else old.nbytes
    _textures[key] = arr
    _texture_bytes += 0 if arr is None else arr.nbytes
    while _texture_bytes > TEXTURE_CACHE_BUDGET_BYTES and len(_textures) > 1:
        dropped = _textures.pop(next(iter(_textures)))
        _texture_bytes -= 0 if dropped is None else dropped.nbytes


def texture_cache_stats() -> tuple[int, int]:
    """(entries, bytes) the terrain texture cache holds: tests and benches."""
    with _prefetch_lock:
        return len(_textures), _texture_bytes


def terrain_average_color_if_loaded(terrain_id: int) -> tuple[int, int, int] | None:
    """get_terrain_average_color() if terrain_id's file has been decoded,
    else None, never decoding or waiting on a prefetch: the minimap's terrain
    colour (GH #54)."""
    path = get_terrain_texture_path(terrain_id)
    if path is None or path not in _averages:
        return None
    return get_terrain_average_color(terrain_id)


def loaded_texture_count() -> int:
    """How many texture files have been decoded: the minimap re-colours while this grows (GH #54)."""
    return len(_averages)


# PIL's reducing_gap for a texture downscale (GH #326: 3.6 vs 17 ms at 160 px, max channel diff 2); None is a plain resize.
TEXTURE_REDUCING_GAP: float | None = 3.0


def _load_texture(path: Path, size: int):
    """Decode one texture file and resize it to (size, size): (read-only
    array, the file's _average_rgb() or None once known). (None, None) for a
    mod file that will not decode. Any thread: PIL releases the GIL."""
    decoded = _decode_texture(path)
    if decoded is None:
        return None, None
    return _sized_texture(decoded[0], size), decoded[1]


def _decode_texture(path: Path):
    """(the file's PIL RGB image, its _average_rgb() or None once known), or
    None for a mod file that will not decode. Decoded once per prefetch job,
    whatever sizes it resizes to."""
    from PIL import Image

    def decode(file: Path):
        with Image.open(file) as img:
            rgb = img.convert("RGB")
        return rgb, None if path in _averages else _average_rgb(rgb)

    return _decode_with_install_fallback(path, decode)


def _sized_texture(rgb, size: int):
    """A decoded texture resized to (size, size), as a read-only array."""
    import numpy as np

    if rgb.size != (size, size):
        rgb = rgb.resize((size, size), reducing_gap=TEXTURE_REDUCING_GAP)
    arr = np.array(rgb)
    arr.flags.writeable = False
    return arr


def _parse_blendomatic(data: bytes):
    """blendomatic.dat's per-pixel alpha bytes as a (modes, tiles,
    BLENDOMATIC_TILE_SIZE) uint8 array, or None if the bytes do not parse
    exactly to EOF or hold an alpha past 128. Layout (openage doc/media/blendomatic.md): a u32 mode
    count and u32 tile count, then per mode a u32 tile size, one flag byte
    per tile, 32 one-bit masks of tile_size bits, and one alpha byte per
    pixel per tile. The bit masks are skipped: the alpha bytes carry them."""
    import struct

    import numpy as np

    if len(data) < 8:
        return None
    n_modes, n_tiles = struct.unpack_from("<II", data, 0)
    offset, modes = 8, []
    for _ in range(n_modes):
        if offset + 4 > len(data):
            return None
        (tile_size,) = struct.unpack_from("<I", data, offset)
        if tile_size != BLENDOMATIC_TILE_SIZE:
            return None
        offset += 4 + n_tiles + 4 * tile_size
        end = offset + n_tiles * tile_size
        if end > len(data):
            return None
        modes.append(np.frombuffer(data, np.uint8, n_tiles * tile_size, offset).reshape(n_tiles, tile_size))
        offset = end
    if offset != len(data) or not modes:
        return None
    arr = np.stack(modes)
    # terrain_blend's integer compose, numpy and native, is exact only for 0..128.
    if int(arr.max()) > 128:
        return None
    arr.flags.writeable = False
    return arr


@lru_cache(maxsize=1)
def get_blendomatic():
    """The install's blendomatic_x1.dat (_parse_blendomatic()), alpha 0..128
    per pixel, where the alpha is the tile's OWN weight and the bleeding
    neighbour's is 128 - alpha (measured, see terrain_blend). None without an
    install or when the file is missing or no longer parses."""
    path = game_dataset.resource_path(BLENDOMATIC_SUBPATH)
    if path is None:
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return _parse_blendomatic(data)


@lru_cache(maxsize=8)
def blendomatic_square_gather(tile_px: int):
    """(tile_px, tile_px) indices into one blendomatic tile's flat pixel list:
    square texel (row b, column a) takes the diamond pixel nearest its centre,
    mapped by iso_geometry.square_to_diamond_local() (the inverse of the
    sampling Stepped and Sloped draw with). A point between two raster rows'
    spans clamps to its own row's nearest pixel, so every texel is defined."""
    import numpy as np

    from descape import iso_geometry

    widths = np.array([4 * min(r, BLENDOMATIC_ROWS - 1 - r) + 1 for r in range(BLENDOMATIC_ROWS)])
    starts = (BLENDOMATIC_WIDTH - widths) // 2
    row_offsets = np.concatenate([[0], np.cumsum(widths)[:-1]])
    b, a = np.mgrid[0:tile_px, 0:tile_px]
    u, v = iso_geometry.square_to_diamond_local((a + 0.5) / tile_px, (b + 0.5) / tile_px)
    row = np.clip(np.floor((v + 1.0) / 2.0 * BLENDOMATIC_ROWS), 0, BLENDOMATIC_ROWS - 1).astype(np.int64)
    col = np.floor((u + 1.0) / 2.0 * BLENDOMATIC_WIDTH).astype(np.int64)
    col = np.clip(col, starts[row], starts[row] + widths[row] - 1)
    gather = row_offsets[row] + col - starts[row]
    gather.flags.writeable = False
    return gather


# Covers the whole key space (TASK-031.85 D6): 9 modes x 31 tiles x the 7 tile sizes of every quality's ladders is 1,953.
@lru_cache(maxsize=2048)
def get_blend_mask(mode: int, tile_index: int, tile_px: int):
    """Blendomatic tile (mode, tile_index) resampled onto the square
    tile_px x tile_px block every terrain style starts from, as a read-only
    uint8 array of own-weight alpha 0..128, faded to 0 along its bleeding
    edges (terrain_blend.fade_bleeding_edges()) and with every other edge
    conformed to its vertex profile (terrain_blend.conform_edges()). None
    without get_blendomatic()."""
    from descape import terrain_blend

    tiles = get_blendomatic()
    if tiles is None:
        return None
    mask = terrain_blend.fade_bleeding_edges(tiles[mode, tile_index][blendomatic_square_gather(tile_px)], tile_index)
    mask = terrain_blend.conform_edges(mask, tile_index)
    mask.flags.writeable = False
    return mask


# (name, size) pairs terrain_mask_at() has returned since the last clear_install_caches().
_masks_asked: set[tuple[str, int]] = set()


def terrain_mask_at(name: str, size: int):
    """terrain/masks/<name> at `size` (terrain_texture_size()), the side of
    the texture it modulates, so it crops at that texture's offset. With no
    file, whatever get_terrain_mask_array() serves: None, or a stand-in."""
    mask = _sized_terrain_mask(name, size)
    if mask is None:
        mask = get_terrain_mask_array(name)
    _masks_asked.add((name, size))
    return mask


def terrain_mask_loaded(name: str, size: int) -> bool:
    """Whether terrain_mask_at(name, size) has run since the last
    clear_install_caches(), without loading anything: native_composite.
    prime_walk()'s filter (TASK-031.85). An lru eviction since leaves it
    True, which costs the walk a few unneeded blend masks, never a load."""
    return (name, size) in _masks_asked


@lru_cache(maxsize=32)
def get_terrain_mask_array(name: str):
    """terrain/masks/<name>'s red channel at the reference level's texture
    side as a read-only uint8 array, 0..255 = the layer terrain's weight, or
    None. The stand-in seam for masks, as get_terrain_texture_array() is for
    textures."""
    return _sized_terrain_mask(name, terrain_texture_size(TEXTURE_REFERENCE_TILE_PIXELS))


@lru_cache(maxsize=128)  # every mask at each ladder size
def _sized_terrain_mask(name: str, size: int):
    """terrain/masks/<name> resized from the file (512 px) to (size, size).
    The files mix L and RGB modes, so they go through RGB first. Only the red
    band is resized: byte-identical to resizing RGB, a third of the cost at
    1920 px (33 -> 11 ms, TASK-031.26)."""
    path = game_dataset.resource_path(f"{TERRAIN_MASK_SUBPATH}/{name}") if name else None
    if path is None or not path.is_file():
        return None
    import numpy as np
    from PIL import Image

    try:
        with Image.open(path) as img:
            red = img.convert("RGB").getchannel("R")
            if red.size != (size, size):
                red = red.resize((size, size))
            arr = np.ascontiguousarray(np.array(red))
    except (OSError, ValueError):
        return None
    arr.flags.writeable = False
    return arr


# Texture prefetch pool, sized like margin_warm.WORKERS (not imported: that
# module pulls in Qt). Created on first use; tests swap in an inline executor.
PREFETCH_WORKERS = min(4, max(1, (os.cpu_count() or 2) - 1))
_prefetch_executor = None
# (path, size) -> (generation, Future) for prefetched loads no get has taken yet.
_pending: dict[tuple[Path, int], tuple[int, Future]] = {}
# Queued prefetch jobs, (path, [(size, Future)]), the next to run first; each has one pool slot.
_jobs: deque[tuple[Path, list[tuple[int, Future]]]] = deque()
# (path, size) -> texture (or None), least recently used first; _texture_bytes is their total nbytes.
_textures: OrderedDict[tuple[Path, int], object] = OrderedDict()
_texture_bytes = 0
# The user's ~512 MB (2026-11-20): above it the least recently used level's file reloads on its next visit.
TEXTURE_CACHE_BUDGET_BYTES = 512 * 2**20
# path -> _average_rgb() of the full file, recorded by the first decode (GH #326).
_averages: dict[Path, tuple[int, int, int]] = {}
_prefetch_lock = threading.Lock()
# Bumped by clear_install_caches() (not a data-mod switch's), so a load from before it is never served.
_generation = 0


def install_generation() -> int:
    """How many times clear_art_caches() has dropped the file-keyed art (clear_install_caches() included):
    a key that changes with the install or the mod art drawn (GH #148's portraits, TASK-378).
    A data-mod switch (changed is TABLES alone, TASK-478) does not move it: pair it with game_dataset.generation()."""
    return _generation


def _prefetch_pool():
    global _prefetch_executor
    if _prefetch_executor is None:
        _prefetch_executor = ThreadPoolExecutor(max_workers=PREFETCH_WORKERS, thread_name_prefix="texture-prefetch")
    return _prefetch_executor


def _run_prefetch_job() -> None:
    """One pool slot: takes the job at the front of _jobs, decodes its file
    once and resolves each size's Future as soon as that size is resized, in
    the caller's order. A size whose Future was cancelled is skipped."""
    with _prefetch_lock:
        if not _jobs:
            return
        path, futures = _jobs.popleft()
    wanted = [(size, future) for size, future in futures if future.set_running_or_notify_cancel()]
    if not wanted:
        return
    with perf_trace.texture_prefetch():
        try:
            decoded = _decode_texture(path)
            for size, future in wanted:
                future.set_result((None, None) if decoded is None else (_sized_texture(decoded[0], size), decoded[1]))
        except Exception as exc:  # noqa: BLE001 - raised again at the get
            for _size, future in wanted:
                if not future.done():
                    future.set_exception(exc)


def prefetch_terrain_textures(terrain_ids, tile_pxs, back: bool = False) -> int:
    """Starts loading the texture files behind terrain_ids, at the texture
    size of each level in tile_pxs, on a small thread pool, so a render that
    needs them later waits on the load already in flight instead of decoding
    each on the GUI thread. One job per file decodes it once for all its
    sizes, resized in tile_pxs' order (the caller's anchor level first). A
    call's jobs go ahead of every earlier call's still-queued ones: the latest
    caller is the current view (a level warm's anchor ahead of the open's
    reference level). A size this call asks for that an earlier call still
    has queued moves into this call's job (a re-anchor's level, queued as a
    neighbour, goes to the front); the earlier job keeps its other sizes.
    `back` instead queues behind everything and moves nothing: the whole
    ladder after a load settles (GH #326). Loads already cached, running or
    taken by a get are skipped, as are ids with no texture. Returns the
    number of (file, size) loads submitted."""
    sizes = list(dict.fromkeys(terrain_texture_size(px) for px in tile_pxs))
    paths = dict.fromkeys(p for p in map(get_terrain_texture_path, map(int, terrain_ids)) if p is not None)
    jobs, submitted = [], 0
    with _prefetch_lock:
        moved: dict[Path, dict[int, Future]] = {}
        if not back and _jobs:
            wanted, kept = set(sizes), []
            for path, queued in _jobs:
                rest = queued
                if path in paths:
                    # A cancelled one (an install clear) is dropped, so its size is loaded afresh.
                    moved.setdefault(path, {}).update((s, f) for s, f in queued if s in wanted and not f.cancelled())
                    rest = [(s, f) for s, f in queued if s not in wanted]
                if rest:
                    kept.append((path, rest))
            _jobs.clear()
            _jobs.extend(kept)
        for path in paths:
            earlier = moved.get(path, {})
            futures = []
            for size in sizes:
                key = (path, size)
                if size in earlier:
                    futures.append((size, earlier[size]))
                elif key not in _textures and key not in _pending:
                    future = Future()
                    _pending[key] = (_generation, future)
                    futures.append((size, future))
                    submitted += 1
            if futures:
                jobs.append((path, futures))
        if back:
            _jobs.extend(jobs)
        else:
            _jobs.extendleft(reversed(jobs))
    # Outside the lock: a test's inline executor runs the slot inside submit(). A job emptied
    # above leaves its slot behind, which finds nothing or the next job.
    pool = _prefetch_pool()
    for _job in jobs:
        pool.submit(_run_prefetch_job)
    return submitted


def _forget_prefetches() -> None:
    """clear_art_caches()'s half for the texture cache and prefetch: cached
    textures and pending loads are dropped (a queued one cancelled, a running
    one finishes unserved) and the generation moves on, in one locked step.
    _jobs stays: a get may be waiting on one of its Futures, and its job skips the cancelled ones."""
    global _generation, _texture_bytes
    with _prefetch_lock:
        _generation += 1
        for _gen, future in _pending.values():
            future.cancel()
        _pending.clear()
        _textures.clear()
        _texture_bytes = 0
        _averages.clear()


def shutdown_prefetch_pool() -> None:
    """For app exit: cancels the queued loads and stops the pool without
    waiting on a running one. Not an atexit hook: concurrent.futures' own exit
    hook runs first and would decode every queued file before it."""
    global _prefetch_executor
    with _prefetch_lock:
        for _gen, future in _pending.values():
            future.cancel()
        _pending.clear()
        # A get may hold a queued job's Future already: cancelled, it raises rather than waiting on a dead slot.
        for _path, futures in _jobs:
            for _size, future in futures:
                future.cancel()
        _jobs.clear()
        executor, _prefetch_executor = _prefetch_executor, None
    if executor is not None:
        executor.shutdown(wait=False, cancel_futures=True)
