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

import json
import os
import re
import shutil
from functools import lru_cache
from pathlib import Path

import platformdirs
import yaml

APP_NAME = "DEscape"


def _default_config_path() -> Path:
    return Path(platformdirs.user_config_dir(APP_NAME, appauthor=False, roaming=True)) / "config.yaml"


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = _default_config_path()
LEGACY_CONFIG_PATH = REPO_ROOT / "config.yaml"
TERRAIN_TEXTURE_SUBPATH = "resources/_common/terrain/textures/2x"

# Real source textures are 512-2048px square; crops taken from them for tile
# blitting are tiny (render.tile_pixels_for_map(), 32-64px depending on map
# size), so there's no need to keep the full-resolution image cached per
# terrain_id -- downsampling
# once at load time keeps this cache's memory footprint reasonable across the
# ~85 distinct texture files a real install has, while still leaving plenty
# of room to pick varied non-repeating crop origins.
LOADED_TEXTURE_SIZE = 512

# key-value-strings-utf8.txt is a flat "<int key> \"<value>\"" list per line
# (comments and blank lines interspersed) -- the same key space
# gen_object_catalog.py commits as string_id. Backslash-escaped quotes are
# the only escape this file actually uses (verified against the real file:
# 109 of ~19k parseable lines carry one, e.g. Sent to \"%s\":).
_STRING_LINE = re.compile(r'^(\d+)\s+"((?:[^"\\]|\\.)*)"')
STRINGS_SUBPATH_TEMPLATE = "resources/{lang}/strings/key-value/key-value-strings-utf8.txt"
DEFAULT_LANGUAGE = "en"

_override_path: Path | None = None


@lru_cache(maxsize=1)
def _terrain_texture_map() -> dict[int, str]:
    data = json.loads((REPO_ROOT / "descape" / "terrain_texture_map.json").read_text())
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
            config = yaml.safe_load(CONFIG_PATH.read_text()) or {}
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
    get_install_path.cache_clear()
    get_terrain_texture_path.cache_clear()
    get_terrain_average_color.cache_clear()
    get_terrain_texture_array.cache_clear()
    _string_table.cache_clear()
    # Imported here, not at module scope: unit_sprites imports this module, so
    # a top-level import would be a cycle. Its caches remember MISSES as well
    # as sprites -- deliberately, since re-deriving one costs a whole file walk
    # -- so without this, configuring an install for the first time would leave
    # every unit drawn as a coloured dot until the app was restarted.
    from descape import unit_sprites

    unit_sprites.clear_caches()
    # object_catalog.py's name resolution reads resource_string() above --
    # same reasoning, a different module. Imported here for the same
    # cycle-avoidance reason (object_catalog.py imports this module).
    from descape import object_catalog

    object_catalog.clear_caches()


def get_language() -> str:
    """config.yaml's "language" key -- an install language folder name
    under resources/ (e.g. "en", "de", "fr") -- or DEFAULT_LANGUAGE if unset
    or unreadable. Not cached, unlike get_install_path(): it is a small file
    read only when a catalog name is (re)resolved, which object_catalog.py's
    own caching already makes infrequent, and caching it here too would add
    another cache set_install_path_override() has no real reason to know
    about clearing."""
    if CONFIG_PATH.is_file():
        try:
            config = yaml.safe_load(CONFIG_PATH.read_text()) or {}
        except (yaml.YAMLError, OSError):
            return DEFAULT_LANGUAGE
        raw = config.get("language")
        if raw:
            return str(raw)
    return DEFAULT_LANGUAGE


def _unescape_string_value(text: str) -> str:
    return text.replace('\\"', '"').replace("\\\\", "\\")


@lru_cache(maxsize=4)
def _string_table(lang: str) -> dict[int, str]:
    install = get_install_path()
    if install is None:
        return {}
    path = install / STRINGS_SUBPATH_TEMPLATE.format(lang=lang)
    if not path.is_file():
        return {}
    table: dict[int, str] = {}
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


def write_config_file(path: Path, config: dict) -> None:
    """Creates `path`'s parent directory if needed, then writes `config` to
    it. Takes the path explicitly -- rather than reading this module's own
    CONFIG_PATH -- so settings.py can pass its own (separately monkeypatched
    in tests) global through; a version that read asset_source.CONFIG_PATH
    internally would silently make settings.CONFIG_PATH decorative."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, default_flow_style=False, sort_keys=False))


def migrate_legacy_config() -> Path | None:
    """One-time copy of a pre-relocation repo-root config.yaml to the
    OS-standard location, idempotent: returns the new path only when it
    actually copied something, None otherwise (including when there was
    nothing to migrate). The legacy file is copied, not moved -- it is the
    user's data, and deleting it is destructive for no gain; it is simply
    ignored from then on. Must be called explicitly at the GUI entry point,
    never at import time (see main()'s own call site for why)."""
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
            config = yaml.safe_load(CONFIG_PATH.read_text()) or {}
        except (yaml.YAMLError, OSError):
            config = {}
    config["aoe2de_install"] = str(path)
    write_config_file(CONFIG_PATH, config)


@lru_cache(maxsize=256)
def get_terrain_texture_path(terrain_id: int) -> Path | None:
    """Returns the real .dds path for a terrain_id, or None if unavailable --
    either no install is configured, or this terrain_id has no known texture."""
    install = get_install_path()
    if install is None:
        return None
    filename = _terrain_texture_map().get(terrain_id)
    if filename is None:
        return None
    path = install / TERRAIN_TEXTURE_SUBPATH / filename
    return path if path.is_file() else None


@lru_cache(maxsize=256)
def get_terrain_average_color(terrain_id: int) -> tuple[int, int, int] | None:
    """Average color of the real texture for terrain_id, or None if unavailable.
    Cheap way to get authentic-ish colors without a full texture-blitting
    renderer."""
    path = get_terrain_texture_path(terrain_id)
    if path is None:
        return None
    from PIL import Image

    with Image.open(path) as img:
        img = img.convert("RGB")
        # Sampling is plenty for an average -- these textures are large (2048x2048).
        small = img.resize((32, 32))
        pixels = list(small.getdata())
    n = len(pixels)
    r = sum(p[0] for p in pixels) // n
    g = sum(p[1] for p in pixels) // n
    b = sum(p[2] for p in pixels) // n
    return (r, g, b)


@lru_cache(maxsize=256)
def get_terrain_texture_array(terrain_id: int):
    """The real texture for terrain_id as an (LOADED_TEXTURE_SIZE,
    LOADED_TEXTURE_SIZE, 3) uint8 numpy array, or None if unavailable --
    either no install is configured, or this terrain_id has no known texture.
    Backs real per-tile texture blitting in render.py; see
    get_terrain_average_color for the cheaper single-color equivalent."""
    path = get_terrain_texture_path(terrain_id)
    if path is None:
        return None

    import numpy as np
    from PIL import Image

    with Image.open(path) as img:
        img = img.convert("RGB").resize((LOADED_TEXTURE_SIZE, LOADED_TEXTURE_SIZE))
        return np.array(img)
