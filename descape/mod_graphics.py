"""Which file draws a sprite or terrain texture when the player's enabled
cosmetic mods override the install's: per file stem, the highest-priority
active mod that ships one, else the install (None here).

Qt-free module state, read by unit_sprites (sprites), asset_source (terrain
textures) and the viewer (the View > Layers "Mod Art" row, Settings > General's
opt-out list, the status line):

- **Off by module default.** The viewer turns it on from the persisted layer,
  so tests and tools/ never pick up the developer's mods by accident.
- **Active mods**: the profile id whose mods/mod-status.json is newest
  (content_roots.active_profile_id(); every profile when none has one), its
  enabled mods in ascending Priority (1, the in-game list's top, wins), then
  mods the status file does not list, by folder name, minus the Settings
  exclusions (`status_path` strings).
- **Inside one mod, .sld beats .smx**; stems and folder names match
  case-insensitively. `_x1` names are asked for, and their `_x2` copies at
  Graphics Quality Enhanced (GH #241).
- generation() moves on every change, for caches keyed on the active art
  (render's sprite memo); asset_source.clear_art_caches() is the one call
  that drops the art-derived caches when it does.

Strictly read-only, like content_roots: mod folders under mods/subscribed/ are
re-synced by the game. tests/test_content_roots.py scans this module's source
for write calls. Imports of content_roots and asset_source are local, inside
functions, since asset_source imports this module back.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from descape import debug_log

SPRITE_SUFFIXES = (".sld", ".smx")  # preference order inside one mod
GRAPHICS_PARTS = ("drs", "graphics")
TEXTURE_PARTS = ("terrain", "textures", "2x")
PALETTE_PARTS = ("resources", "_common", "palettes")
PALETTE_CONF = "palettes.conf"
WHITE_PLAYER_PALETTE = "playercolor_white.pal"


@dataclass(frozen=True)
class ModArt:
    """One mod that ships art, for the Settings list and the status line."""

    status_path: str
    title: str
    priority: int | None
    sld: int
    smx: int
    textures: int
    excluded: bool


@dataclass
class _Index:
    roots: list  # every candidate mod root, resolution order, exclusions included
    sprites: dict[str, list[tuple[str, Path, str]]]  # stem -> [(status_path, path, format)] in order
    textures: dict[str, list[tuple[str, Path]]]
    counts: dict[str, tuple[int, int, int]]


_enabled = False
_excluded: frozenset[str] = frozenset()
_generation = 0
_index: _Index | None = None
_palettes: _Palettes | None = None
_lock = threading.RLock()


def generation() -> int:
    return _generation


def enabled() -> bool:
    return _enabled


def _bump() -> None:
    global _generation
    _generation += 1


def set_enabled(flag: bool) -> bool:
    """Turns mod art on or off; True when that changed anything."""
    global _enabled
    flag = bool(flag)
    if flag == _enabled:
        return False
    _enabled = flag
    _bump()
    return True


def set_excluded(status_paths) -> bool:
    """The Settings opt-outs, as mod-status `Path` strings; True when changed."""
    global _excluded
    new = frozenset(str(p) for p in status_paths)
    if new == _excluded:
        return False
    _excluded = new
    _bump()
    return True


def clear() -> None:
    """Forget the scanned mod folders and palettes (an install or profile change)."""
    global _index, _palettes
    with _lock:
        _index = None
        _palettes = None
    _bump()


def reset() -> None:
    """Module defaults: off, nothing excluded, nothing scanned. For tests."""
    global _enabled, _excluded
    _enabled = False
    _excluded = frozenset()
    clear()


def _candidate_roots() -> list:
    from descape import content_roots

    profile = content_roots.profile_path()
    if profile is None:
        return []
    active = content_roots.active_profile_id(profile)
    roots = [
        r for r in content_roots.content_roots(install=None, profile=profile)
        if r.kind in (content_roots.KIND_MOD_SUBSCRIBED, content_roots.KIND_MOD_LOCAL)
        and (active is None or r.profile_id == active)
    ]
    return sorted(roots, key=lambda r: (r.priority is None, r.priority or 0, r.mod_id.casefold(), r.profile_id or ""))


def _listing(folder: Path | None) -> list[Path]:
    if folder is None:
        return []
    try:
        return sorted(p for p in folder.iterdir() if p.is_file())
    except OSError as exc:
        debug_log.log(f"mod art: cannot list {folder} ({exc!r})")
        return []


def _build_index() -> _Index:
    from descape import content_roots

    roots = _candidate_roots()
    sprites: dict[str, list[tuple[str, Path, str]]] = {}
    textures: dict[str, list[tuple[str, Path]]] = {}
    counts: dict[str, tuple[int, int, int]] = {}
    for root in roots:
        common = root.common
        by_stem: dict[str, tuple[Path, str]] = {}
        sld = smx = 0
        for path in _listing(content_roots.child_dir(common, *GRAPHICS_PARTS)):
            suffix = path.suffix.casefold()
            if suffix not in SPRITE_SUFFIXES:
                continue
            sld += suffix == ".sld"
            smx += suffix == ".smx"
            stem = path.stem.casefold()
            held = by_stem.get(stem)
            if held is None or SPRITE_SUFFIXES.index(suffix) < SPRITE_SUFFIXES.index("." + held[1]):
                by_stem[stem] = (path, suffix[1:])
        for stem, (path, fmt) in by_stem.items():
            sprites.setdefault(stem, []).append((root.status_path, path, fmt))
        texture_files = _listing(content_roots.child_dir(common, *TEXTURE_PARTS))
        for path in texture_files:
            textures.setdefault(path.name.casefold(), []).append((root.status_path, path))
        counts[root.status_path] = (sld, smx, len(texture_files))
    return _Index(roots, sprites, textures, counts)


def _get_index() -> _Index:
    global _index
    with _lock:
        if _index is None:
            _index = _build_index()
        return _index


def mods_with_art() -> list[ModArt]:
    """Every candidate mod shipping sprites or terrain textures, in resolution
    order, excluded ones included (flagged) so Settings can list them."""
    index = _get_index()
    found = []
    for root in index.roots:
        sld, smx, tex = index.counts.get(root.status_path, (0, 0, 0))
        if sld or smx or tex:
            found.append(ModArt(root.status_path, root.mod_id, root.priority, sld, smx, tex, root.status_path in _excluded))
    return found


def resolve_sprite(file_name: str) -> tuple[Path, str] | None:
    """(path, "sld" | "smx") of the highest-priority active mod's file for
    this stem, or None: off, or no active mod ships it (the install draws)."""
    if not _enabled:
        return None
    for status_path, path, fmt in _get_index().sprites.get(file_name.casefold(), ()):
        if status_path not in _excluded:
            return path, fmt
    return None


def resolve_terrain_texture(file_name: str) -> Path | None:
    """The highest-priority active mod's terrain/textures/2x/<file_name>, or None."""
    if not _enabled:
        return None
    for status_path, path in _get_index().textures.get(file_name.casefold(), ()):
        if status_path not in _excluded:
            return path
    return None


class _Palettes:
    """palettes.conf's numbered palettes, each read from the install on first use."""

    def __init__(self, folder: Path | None):
        self.folder = folder
        self.files: dict[int, str] = {}
        self.loaded: dict[int, np.ndarray | None] = {}
        self.shades = None
        if folder is None:
            return
        try:
            text = (folder / PALETTE_CONF).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            debug_log.log(f"mod art: no {PALETTE_CONF} ({exc!r})")
            return
        for raw in text.splitlines():
            number, _, name = raw.split("//", 1)[0].strip().partition(",")
            if name and number.strip().isdigit():
                self.files[int(number)] = name.strip()

    def get(self, number: int, default=None):
        if number not in self.loaded:
            self.loaded[number] = self._load(self.files.get(number))
        found = self.loaded[number]
        return default if found is None else found

    def _load(self, name: str | None) -> np.ndarray | None:
        from descape import slp_decoder, smx_decoder

        if name is None or self.folder is None:
            return None
        try:
            return smx_decoder.palette_array(slp_decoder.load_palette_rgba(self.folder / name))
        except (OSError, ValueError, IndexError) as exc:
            debug_log.log(f"mod art: palette {name} unreadable ({exc!r})")
            return None

    def player_shades(self) -> np.ndarray:
        from descape import slp_decoder, smx_decoder

        if self.shades is None:
            try:
                white = slp_decoder.load_palette(self.folder / WHITE_PLAYER_PALETTE) if self.folder else []
            except (OSError, ValueError, IndexError):
                white = []
            self.shades = smx_decoder.player_shades(white)
        return self.shades


def _get_palettes() -> _Palettes:
    global _palettes
    from descape import asset_source, content_roots

    with _lock:
        if _palettes is None:
            install = asset_source.get_install_path()
            folder = None if install is None else content_roots.child_dir(install, *PALETTE_PARTS)
            _palettes = _Palettes(folder)
        return _palettes


def palettes() -> _Palettes:
    """The install's palettes by number, a mapping with .get() (smx_decoder's input)."""
    return _get_palettes()


def player_shades() -> np.ndarray:
    return _get_palettes().player_shades()
