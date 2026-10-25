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
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import platformdirs
import yaml

from descape import debug_log, game_dataset, perf_trace

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

# Real source textures are 512-2048px square; crops taken from them for tile
# blitting are tiny (render.tile_pixels_for_map(), 32-64px depending on map
# size), so there's no need to keep the full-resolution image cached per
# terrain_id -- downsampling
# once at load time keeps this cache's memory footprint reasonable across the
# ~85 distinct texture files a real install has, while still leaving plenty
# of room to pick varied non-repeating crop origins.
LOADED_TEXTURE_SIZE = 512
# GH #67: the tile size at which a tile crops its share of a LOADED_TEXTURE_SIZE texture,
# for terrain_texture_size(): render.SMALL_MAP_TILE_PIXELS, a small map's reference level.
TEXTURE_REFERENCE_TILE_PIXELS = 64


def terrain_texture_size(tile_px: int) -> int:
    """The texture size a level of `tile_px` crops its tiles from. Under
    Return of Rome (GH #67) a zoomed-out level reads textures shrunk by its
    ratio to TEXTURE_REFERENCE_TILE_PIXELS, so a tile keeps the texture share
    it has at the reference and a pattern shrinks with the map: RoR's farm
    rows are twice as wide as AoE2's, and at one size for every level they
    grew to a few coarse rows per farm. AoE2 keeps LOADED_TEXTURE_SIZE at
    every level (user decision 2026-10-17). Always a multiple of tile_px."""
    if not game_dataset.is_ror() or tile_px >= TEXTURE_REFERENCE_TILE_PIXELS:
        return LOADED_TEXTURE_SIZE
    return LOADED_TEXTURE_SIZE * tile_px // TEXTURE_REFERENCE_TILE_PIXELS

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
# GH #67: Return of Rome's own strings, a separate file under the RoR root, layered over the base file.
ROR_STRINGS_SUBPATH_TEMPLATE = "resources/{lang}/strings/key-value/key-value-pompeii-strings-utf8.txt"
DEFAULT_LANGUAGE = "en"
# GH #58: a data mod's own strings, in this order over the install's (one mod ships both).
MOD_STRINGS_FILES = ("key-value-strings-utf8.txt", "key-value-modded-strings-utf8.txt")

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


def clear_art_caches() -> None:
    """Drops every cache holding sprite or terrain-texture art, for a change
    in which files draw it (mod art toggled or opted out, TASK-378), and
    bumps install_generation() so ThumbLoader-keyed thumbnails reload."""
    get_terrain_texture_path.cache_clear()
    get_terrain_average_color.cache_clear()
    # Path-keyed rather than id-keyed, so a changed install that resolves to
    # a same-named file under a different root would otherwise keep serving
    # the old swatch or texture.
    _terrain_thumbnail_for_path.cache_clear()
    _terrain_texture_for_path.cache_clear()
    _shrunk_texture.cache_clear()
    _forget_prefetches()
    # Imported here, not at module scope: unit_sprites imports this module, so
    # a top-level import would be a cycle. Its caches remember MISSES as well
    # as sprites -- deliberately, since re-deriving one costs a whole file walk
    # -- so without this, configuring an install for the first time would leave
    # every unit drawn as a coloured dot until the app was restarted.
    from descape import terrain_blend, unit_sprites

    unit_sprites.clear_caches()
    # Blended blocks are built from the textures above.
    terrain_blend.clear_caches()


def clear_install_caches() -> None:
    """Drops every cache downstream of the install path, leaving the override
    as it is. The test suite calls this before each test (see tests/conftest.py)."""
    get_install_path.cache_clear()
    from descape import mod_graphics

    # Its palettes come from the install, and its mod folders from the profile derived from it.
    mod_graphics.clear()
    clear_art_caches()
    get_blendomatic.cache_clear()
    get_blend_mask.cache_clear()
    get_terrain_mask_array.cache_clear()
    _sized_terrain_mask.cache_clear()
    _string_table.cache_clear()
    _ror_icon_files.cache_clear()
    get_unit_icon.cache_clear()
    get_tech_icon.cache_clear()
    get_stat_icon.cache_clear()
    # object_catalog.py's name resolution reads resource_string() above --
    # same reasoning, a different module. Imported here for the same
    # cycle-avoidance reason (object_catalog.py imports this module).
    from descape import object_catalog

    object_catalog.clear_caches()
    # GH #140's instruction text colours, read from the install's UIColors.json.
    from descape import message_markup

    message_markup.clear_caches()


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
    """The install's strings for lang; under Return of Rome its own strings
    file read over them, a RoR string winning on a shared key (GH #67), and
    under a data mod the mod's strings files likewise, key by key (GH #58)."""
    install = get_install_path()
    if install is None:
        return {}
    paths = [install / STRINGS_SUBPATH_TEMPLATE.format(lang=lang)]
    if game_dataset.is_ror():
        paths.append(game_dataset.ror_root(install) / ROR_STRINGS_SUBPATH_TEMPLATE.format(lang=lang))
    mod = game_dataset.data_mod()
    if mod is not None:
        from descape import content_roots

        folder = content_roots.child_dir(mod.root, "resources", lang, "strings", "key-value")
        if folder is not None:
            paths.extend(folder / name for name in MOD_STRINGS_FILES)
    table: dict[int, str] = {}
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = _STRING_LINE.match(line)
            if match is None:
                continue
            table[int(match.group(1))] = _unescape_string_value(match.group(2))
    return table


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
    unit set. Extension case varies, so names are matched loosely."""
    return unit_icon_path_in(get_install_path(), icon, building)


def unit_icon_path_in(install: Path | None, icon: int | None, building: bool = False) -> Path | None:
    """unit_icon_path() under an install the caller passes, never read here:
    the Units catalog's thumbnail worker resolves it on the GUI thread (GH #68)."""
    if install is None or icon is None or icon < 0:
        return None
    if game_dataset.is_ror():
        path = _ror_icon_files(install).get(("Buildings" if building else "Units", icon))
        if path is not None:
            return path
    # GH #67: under Return of Rome without its icon table, its own icon set first, then the install's.
    roots = [game_dataset.ror_root(install), install] if game_dataset.is_ror() else [install]
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
    """RefName -> FileName (relative to widgetui/) for every atlas texture a materials file defines."""
    files = {}
    for atlas in materials.get("AtlasTextures", ()):
        for texture in atlas.get("AtlasDef", {}).get("Textures", ()):
            if "RefName" in texture and "FileName" in texture:
                files[texture["RefName"]] = texture["FileName"]
    return files


@lru_cache(maxsize=2)
def _ror_icon_files(install: Path) -> dict[tuple[str, int], Path]:
    """("Units" or "Buildings", RoR icon index) -> its portrait file, through RoR's icons.json (GH #67).
    A RefName RoR's morematerials.json defines is a RoR file, else one the install's materials.json
    defines is AoE2's. Only existing files; {} without the RoR icon table."""
    ror_ui = game_dataset.ror_root(install) / UI_DIR
    table = _read_json(ror_ui / ICON_TABLE_NAME)
    if not table:
        return {}
    sources = [
        (ror_ui, _atlas_files(_read_json(ror_ui / ROR_MATERIALS_NAME))),
        (install / UI_DIR, _atlas_files(_read_json(install / UI_DIR / MATERIALS_NAME))),
    ]
    files = {}
    for section in ("Units", "Buildings"):
        for key, ref in (table.get(section) or {}).items():
            if not key.isdigit():
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


def read_unit_icon_in(install: Path | None, icon: int | None, building: bool = False, px: int | None = None):
    """get_unit_icon() under a passed install, uncached and thread-safe,
    downscaled to fit px x px when px is given; or None."""
    arr = _read_rgba(unit_icon_path_in(install, icon, building))
    if arr is None or px is None:
        return arr
    import numpy as np
    from PIL import Image

    image = Image.fromarray(np.asarray(arr))
    image.thumbnail((px, px), Image.LANCZOS)
    return np.array(image)


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
    """The install's portrait for this catalog icon index (unit_icon_path())
    as a read-only (h, w, 4) uint8 RGBA array, or None (no install, no file,
    a decode failure)."""
    return _read_rgba(unit_icon_path(icon, building))


def tech_icon_path(icon: int | None) -> Path | None:
    """The tech icon file for a catalog tech icon index, by its "NNN_" prefix, or None."""
    install = get_install_path()
    if install is None or icon is None or icon < 0:
        return None
    prefix = f"{icon:03d}_"
    return _find_file(install / TECH_ICON_DIR, lambda name: name.startswith(prefix) and name.endswith(".dds"))


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
    data = path.read_bytes()
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        encoding = locale.getpreferredencoding(False)
        if codecs.lookup(encoding).name == "utf-8":
            encoding = "cp1252"
        return data.decode(encoding, errors="replace")


def write_config_file(path: Path, config: dict) -> None:
    """Creates `path`'s parent directory if needed, then writes `config` to
    it. Takes the path explicitly -- rather than reading this module's own
    CONFIG_PATH -- so settings.py can pass its own (separately monkeypatched
    in tests) global through; a version that read asset_source.CONFIG_PATH
    internally would silently make settings.CONFIG_PATH decorative."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, default_flow_style=False, sort_keys=False), encoding="utf-8")


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


def save_install_path_to_config(path: Path) -> None:
    """Persists `path` to config.yaml (read-modify-write, so any other keys
    a future version adds there survive)."""
    config = {}
    if CONFIG_PATH.is_file():
        try:
            config = yaml.safe_load(read_config_text(CONFIG_PATH)) or {}
        except (yaml.YAMLError, OSError):
            config = {}
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

    Derived from get_terrain_texture_array()'s cached 512x512 array, not the
    2048x2048 .dds: the textured renderer has already decoded every terrain on
    screen, so a Copy Region thumbnail costs a 32x32 resize instead of ~20 ms of
    decode per terrain. One definition whatever is cached. Measured against the
    .dds-derived average: 133 of 134 terrains identical, terrain 23 (g_wt3.dds)
    one step lower on red. The cost: textures-off rendering now loads that 512²
    array (~0.8 MB) for each terrain on the map; with textures on it is resident
    anyway."""
    arr = get_terrain_texture_array(terrain_id)
    if arr is None:
        return None
    import numpy as np
    from PIL import Image

    # Sampling is plenty for an average. Integer floor mean per channel.
    pixels = np.asarray(Image.fromarray(arr).resize((32, 32))).reshape(-1, 3)
    r, g, b = (int(total) // len(pixels) for total in pixels.sum(axis=0, dtype=np.int64))
    return (r, g, b)


# A browser swatch is one tile at default graphics quality, so it matches
# render.SMALL_MAP_TILE_PIXELS. Not imported from render.py: this module is
# below it in the import order and has no other reason to depend on it.
TERRAIN_THUMBNAIL_PX = 64


@lru_cache(maxsize=256)
def _terrain_thumbnail_for_path(path: Path, px: int):
    """The open+resize half of get_terrain_thumbnail, keyed on the FILE, not
    on a terrain_id. 131 terrains share only 85 textures -- g_for.dds alone
    backs 12 of them -- so an id-keyed cache would decode the same .dds a
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
    install's file of the same name, logged; an install file's error raises as before."""
    try:
        return decode(path)
    except _TEXTURE_DECODE_ERRORS as exc:
        install = get_install_path()
        fallback = None if install is None else install / TERRAIN_TEXTURE_SUBPATH / path.name
        if fallback is None or fallback == path or not fallback.is_file():
            raise
        debug_log.log(f"mod art: texture {path} unreadable ({exc!r}); drawing the install's")
        return decode(fallback)


def get_terrain_thumbnail(terrain_id: int, px: int = TERRAIN_THUMBNAIL_PX):
    """A (px, px, 3) uint8 array of terrain_id's real texture, or None when
    no install is configured or the id has no mapped texture -- the same
    degradation contract every other function here has, so a caller falls
    back to terrain_palette.color_for_terrain_id.

    Deliberately NOT built from get_terrain_texture_array(): that cache
    holds a 512x512x3 array per file and evicts nothing at 85 < 256, so
    filling a browser would warm ~67MB in the renderer's own cache for
    textures the open map may never touch. At 64px the whole catalog is
    ~1.6MB, and only the 85 distinct files are ever decoded."""
    path = get_terrain_texture_path(terrain_id)
    if path is None:
        return None
    return _terrain_thumbnail_for_path(path, px)


def get_terrain_texture_array(terrain_id: int):
    """The real texture for terrain_id as a read-only (LOADED_TEXTURE_SIZE,
    LOADED_TEXTURE_SIZE, 3) uint8 numpy array, or None if unavailable --
    either no install is configured, or this terrain_id has no known texture.
    Backs real per-tile texture blitting in render.py; see
    get_terrain_average_color for the cheaper single-color equivalent."""
    path = get_terrain_texture_path(terrain_id)
    if path is None:
        return None
    return _terrain_texture_for_path(path)


def terrain_texture_at(terrain_id: int, size: int):
    """get_terrain_texture_array() at `size` (terrain_texture_size()): the
    array itself at LOADED_TEXTURE_SIZE, else that array shrunk (GH #67)."""
    if size == LOADED_TEXTURE_SIZE:
        return get_terrain_texture_array(terrain_id)
    path = get_terrain_texture_path(terrain_id)
    return None if path is None else _shrunk_texture(path, size)


@lru_cache(maxsize=256)
def _shrunk_texture(path: Path, size: int):
    """_terrain_texture_for_path(path) box-filtered to (size, size), read-only."""
    import numpy as np
    from PIL import Image

    arr = np.array(Image.fromarray(_terrain_texture_for_path(path)).resize((size, size), Image.BOX))
    arr.flags.writeable = False
    return arr


@lru_cache(maxsize=256)
def _terrain_texture_for_path(path: Path):
    """get_terrain_texture_array's load, keyed on the FILE like
    _terrain_thumbnail_for_path: a map's 37 ids resolve to 26 files. Read-only,
    since ids sharing a file share the array; consumers only crop views.
    Takes a prefetched load's result (waiting if it is still running) rather
    than decoding the file a second time; a worker's exception raises here."""
    with _prefetch_lock:
        entry = _pending.pop(path, None)
        generation = _generation
    if entry is not None and entry[0] == generation:
        with perf_trace.texture_wait():
            arr = entry[1].result()
    else:
        with perf_trace.texture_load():
            arr = _load_texture(path)
    with _prefetch_lock:
        if generation == _generation:
            _loaded.add(path)
    return arr


def _load_texture(path: Path):
    """Decode and resize one texture file. Any thread: PIL releases the GIL."""
    import numpy as np
    from PIL import Image

    def decode(file: Path):
        with Image.open(file) as img:
            rgb = img.convert("RGB").resize((LOADED_TEXTURE_SIZE, LOADED_TEXTURE_SIZE))
            return np.array(rgb)

    arr = _decode_with_install_fallback(path, decode)
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


@lru_cache(maxsize=512)
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


def terrain_mask_at(name: str, size: int):
    """get_terrain_mask_array() at `size`, as terrain_texture_at() (GH #67)."""
    if size == LOADED_TEXTURE_SIZE:
        return get_terrain_mask_array(name)
    return _sized_terrain_mask(name, size)


@lru_cache(maxsize=32)
def get_terrain_mask_array(name: str):
    """terrain/masks/<name>'s red channel as a read-only (LOADED_TEXTURE_SIZE,
    LOADED_TEXTURE_SIZE) uint8 array, 0..255 = the layer terrain's weight, or
    None. The files mix L and RGB modes, so they go through RGB first; they
    are 512 px already, the same grid get_terrain_texture_array() crops."""
    return _sized_terrain_mask(name, LOADED_TEXTURE_SIZE)


@lru_cache(maxsize=96)  # every mask at up to three sizes under Return of Rome
def _sized_terrain_mask(name: str, size: int):
    """get_terrain_mask_array()'s load at `size`, box-filtered below LOADED_TEXTURE_SIZE."""
    path = game_dataset.resource_path(f"{TERRAIN_MASK_SUBPATH}/{name}") if name else None
    if path is None or not path.is_file():
        return None
    import numpy as np
    from PIL import Image

    try:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            if rgb.size != (LOADED_TEXTURE_SIZE, LOADED_TEXTURE_SIZE):
                rgb = rgb.resize((LOADED_TEXTURE_SIZE, LOADED_TEXTURE_SIZE))
            if size != LOADED_TEXTURE_SIZE:
                rgb = rgb.resize((size, size), Image.BOX)
            arr = np.ascontiguousarray(np.array(rgb)[:, :, 0])
    except (OSError, ValueError):
        return None
    arr.flags.writeable = False
    return arr


# Texture prefetch pool, sized like margin_warm.WORKERS (not imported: that
# module pulls in Qt). Created on first use; tests swap in an inline executor.
PREFETCH_WORKERS = min(4, max(1, (os.cpu_count() or 2) - 1))
_prefetch_executor = None
# path -> (generation, Future) for prefetched loads no get has taken yet.
_pending: dict[Path, tuple[int, Future]] = {}
# Paths _terrain_texture_for_path has cached: lru_cache has no membership test.
_loaded: set[Path] = set()
_prefetch_lock = threading.Lock()
# Bumped by clear_install_caches(), so a load from before it is never served.
_generation = 0


def install_generation() -> int:
    """How many times clear_art_caches() has run (clear_install_caches() included): a key
    that changes with the install or the mod art drawn (GH #148's portraits, TASK-378)."""
    return _generation


def _prefetch_pool():
    global _prefetch_executor
    if _prefetch_executor is None:
        _prefetch_executor = ThreadPoolExecutor(max_workers=PREFETCH_WORKERS, thread_name_prefix="texture-prefetch")
    return _prefetch_executor


def _prefetch_load(path: Path):
    with perf_trace.texture_prefetch():
        return _load_texture(path)


def prefetch_terrain_textures(terrain_ids) -> int:
    """Starts loading the texture files behind terrain_ids on a small thread
    pool, so a render that needs them later waits on the load already in
    flight instead of decoding each on the GUI thread. Files already cached or
    already pending are skipped, as are ids with no texture. Returns the
    number of loads submitted."""
    submitted = 0
    for terrain_id in terrain_ids:
        path = get_terrain_texture_path(int(terrain_id))
        if path is None:
            continue
        with _prefetch_lock:
            if path in _loaded or path in _pending:
                continue
            _pending[path] = (_generation, _prefetch_pool().submit(_prefetch_load, path))
        submitted += 1
    return submitted


def _forget_prefetches() -> None:
    """clear_install_caches()'s half for the prefetch: pending loads are
    dropped (a queued one cancelled, a running one finishes unserved) and the
    generation moves on."""
    global _generation
    with _prefetch_lock:
        _generation += 1
        for _gen, future in _pending.values():
            future.cancel()
        _pending.clear()
        _loaded.clear()


def shutdown_prefetch_pool() -> None:
    """For app exit: cancels the queued loads and stops the pool without
    waiting on a running one. Not an atexit hook: concurrent.futures' own exit
    hook runs first and would decode every queued file before it."""
    global _prefetch_executor
    with _prefetch_lock:
        for _gen, future in _pending.values():
            future.cancel()
        _pending.clear()
        executor, _prefetch_executor = _prefetch_executor, None
    if executor is not None:
        executor.shutdown(wait=False, cancel_futures=True)
