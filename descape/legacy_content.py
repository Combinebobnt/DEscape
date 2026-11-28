"""The 1999 game's derived root: its install unpacked once into DE's layout (TASK-430, D6).

Every DEscape lookup is a path along a game's root chain, so rather than read
the Age of Kings / The Conquerors archives live, build_root() writes, once per
install, a folder that mirrors an AoE2:DE install's layout under
CACHE_ROOT/<key>/, and game_dataset.install_root() serves it as the `aoc`
game's content root:

- each graphic SLP, the first archive in drs_archive.AOC_GRAPHICS_ORDER
  holding its id winning, as resources/_common/drs/graphics/slp_<id>.slp,
  bytes as stored;
- blendomatic.dat, copied to resources/_common/dat/blendomatic_x1.dat;
- interfac.drs' palette (bina PALETTE_ID) under resources/_common/palettes/,
  with a palettes.conf naming it and playercolor_white.pal, a neutral
  player-colour ramp (the grey player's, GREY_RAMP_BASE) in the file the
  SMX model reads (mod_graphics.WHITE_PLAYER_PALETTE);
- the string tables of language.dll, then language_x1.dll and
  language_x1_p1.dll (a later one winning), as
  resources/<lang>/strings/key-value/key-value-strings-utf8.txt, <lang> from
  the tables' resource language (Game.strings_any_language reads it
  whatever DEscape's own language is).

Terrain textures and portraits come in later stages. `key` hashes the install
path, each source file's name, size and mtime, and ROOT_FORMAT, which any
change to what the builder writes bumps. A root is built in a staging folder
beside its target and renamed into place whole, so a complete root never
changes under a reader. As dat_cache does, nothing here deletes: a root an
install no longer keys to, and a failed build's staging folder, stay until
deleted by hand, which is safe while the 1999 game is not shown.

The builder only reads the install, and its output is the user's own derived
data, never committed or shipped. Qt-free.

HD Edition's root is a second recipe of the same shape (descape/hd_content.py);
recipe() gives the module for a Game.install kind, and callers reach either
through it.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import struct
from collections.abc import Callable
from pathlib import Path

import platformdirs
import yaml

from descape import drs_archive, games

# Bump on any change to what build_root() writes: every user's root is rebuilt once.
ROOT_FORMAT = 1
CACHE_ROOT = Path(platformdirs.user_cache_dir("DEscape", appauthor=False)) / "game_roots" / "aoc"
META_NAME = "root.json"
INSTALL_ENV = "AOC_INSTALL_PATH"
# config.yaml's key, spelled out where it is read and written (tests/test_settings_file_keys.py discovers it there).
INSTALL_KEY = "aoc_install"
PLACEHOLDER = "The folder holding language.dll and Data/"

DATA_DIR = "Data"
# What validate_install() asks of an install, besides one of games.AOC.dat_subpaths.
REQUIRED_FILES = ("Data/graphics.drs", "Data/terrain.drs", "Data/interfac.drs", "language.dll")
STRINGS_DLLS = ("language.dll", "language_x1.dll", "language_x1_p1.dll")
BLENDOMATIC = "Data/blendomatic.dat"

GRAPHICS_SUBPATH = "resources/_common/drs/graphics"
BLENDOMATIC_SUBPATH = "resources/_common/dat/blendomatic_x1.dat"
PALETTE_SUBPATH = "resources/_common/palettes"
STRINGS_SUBPATH = "resources/{lang}/strings/key-value/key-value-strings-utf8.txt"
PALETTE_ID = 50500
PALETTE_NAME = f"interfac_{PALETTE_ID}.pal"
PALETTES_CONF = "palettes.conf"
WHITE_PLAYER_PALETTE = "playercolor_white.pal"
# Player 7's player_color_base in the AoC .dat: palette 128..135 is a grey ramp. Every base is a multiple of 8.
GREY_RAMP_BASE = 128
RAMP_LENGTH = 8

# A resource language id (LANGID) to DE's folder under resources/: whole ids first, then the primary language.
_LANG_FOLDERS = {0x0404: "tw", 0x080A: "mx", 0x0416: "br"}
_PRIMARY_LANG_FOLDERS = {
    0x04: "zh", 0x07: "de", 0x09: "en", 0x0A: "es", 0x0C: "fr", 0x10: "it", 0x11: "jp",
    0x12: "ko", 0x15: "pl", 0x16: "br", 0x19: "ru", 0x1F: "tr", 0x2A: "vi",
}
DEFAULT_LANGUAGE = "en"

# A failed build per root key, so the game counts as unconfigured until the install changes.
_failures: dict[str, str] = {}

Progress = Callable[[int, str], None]


class RootBuildError(RuntimeError):
    """An install that validated but could not be unpacked."""


# Each Game.install kind with a derived root, to the module building it; each has this module's public functions.
_RECIPES = {games.AOC_INSTALL: "descape.legacy_content", games.HD_INSTALL: "descape.hd_content"}


def recipe(install: str):
    """The module that builds a Game.derived_root game's root from install kind `install`. ValueError for another kind."""
    import importlib

    try:
        return importlib.import_module(_RECIPES[install])
    except KeyError:
        raise ValueError(f"install kind {install!r} has no derived root") from None


def _find(install: Path, rel: str) -> Path | None:
    """`rel` under `install`, each part matched case-insensitively (a copy from Windows keeps its own case)."""
    here = Path(install)
    for part in rel.split("/"):
        exact = here / part
        if exact.exists():
            here = exact
            continue
        try:
            here = next(p for p in here.iterdir() if p.name.casefold() == part.casefold())
        except (OSError, StopIteration):
            return None
    return here


def _dat(install: Path) -> Path | None:
    return next((p for rel in games.AOC.dat_subpaths if (p := _find(install, rel)) is not None and p.is_file()), None)


def install_path() -> Path | None:
    """The configured 1999 install: INSTALL_ENV, else config.yaml's INSTALL_KEY. None when unset or not a folder."""
    raw = os.environ.get(INSTALL_ENV)
    if not raw:
        from descape import asset_source

        try:
            config = yaml.safe_load(asset_source.read_config_text(asset_source.CONFIG_PATH)) or {}
        except (yaml.YAMLError, OSError):
            return None
        raw = config.get("aoc_install") if isinstance(config, dict) else None
    if not raw:
        return None
    path = Path(str(raw))
    return path if path.is_dir() else None


def save_install_path(path: Path | None) -> None:
    """Persists the 1999 install to config.yaml's INSTALL_KEY; None removes it."""
    from descape import asset_source

    asset_source.save_aoc_install_to_config(path)


def browse_start() -> Path | None:
    """Where Browse opens with no path typed: nowhere in particular, since the game is not on Steam."""
    return None


def validate_install(path: Path) -> tuple[bool, str]:
    """Whether `path` is an Age of Kings / The Conquerors install, with a status line either way."""
    if not Path(path).is_dir():
        return False, f"Not a folder: {path}"
    missing = [rel for rel in REQUIRED_FILES if (found := _find(path, rel)) is None or not found.is_file()]
    if missing:
        return False, f"Not an Age of Kings / The Conquerors install: missing {', '.join(missing)}"
    if _dat(path) is None:
        return False, f"No Conquerors data here ({' or '.join(games.AOC.dat_subpaths)}): an Age of Kings-only install is not supported"
    return True, "The Conquerors install found."


def source_files(install: Path) -> list[Path]:
    """Every install file build_root() reads, the ones present."""
    rels = [f"{DATA_DIR}/{name}" for name in drs_archive.AOC_GRAPHICS_ORDER]
    rels += ["Data/interfac.drs", "Data/terrain.drs", BLENDOMATIC, *STRINGS_DLLS]
    return [p for rel in rels if (p := _find(install, rel)) is not None and p.is_file()]


def root_key(install: Path) -> str:
    """The derived root's folder name: ROOT_FORMAT, the install path, each source file's name, size and mtime."""
    files = []
    for path in source_files(install):
        stat = path.stat()
        files.append([path.name.casefold(), stat.st_size, stat.st_mtime_ns])
    payload = {"format": ROOT_FORMAT, "install": str(Path(install).resolve()), "files": files}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]


def built_root(install: Path | None) -> Path | None:
    """The current derived root of `install`, or None while it is not built. Never builds."""
    if install is None:
        return None
    root = CACHE_ROOT / root_key(install)
    return root if (root / META_NAME).is_file() else None


def build_failed(install: Path) -> str | None:
    """Why the last build of `install`'s current root failed in this process, or None."""
    return _failures.get(root_key(install))


def forget_failures() -> None:
    """For a changed install setting: the next activation tries the build again."""
    _failures.clear()


def ensure_root(install: Path | None = None, progress: Progress | None = None) -> Path | None:
    """The derived root of `install` (default the configured one), built first
    when it is not current. None without a valid install. Raises
    RootBuildError when the build fails, and records it (build_failed())."""
    install = install_path() if install is None else Path(install)
    if install is None or not validate_install(install)[0]:
        return None
    key = str(install)
    try:
        key = root_key(install)
        root = built_root(install)
        return root if root is not None else build_root(install, progress)
    except Exception as exc:  # any failure on a user's own files falls back to the stand-in
        message = f"Could not unpack the Age of Kings / The Conquerors install: {type(exc).__name__}: {exc}"
        _failures[key] = message
        raise RootBuildError(message) from exc


def build_root(install: Path, progress: Progress | None = None) -> Path:
    """Unpacks `install` into CACHE_ROOT/<root_key()>/ and returns it. A complete root another build placed first is used."""
    step = progress or (lambda _percent, _caption: None)
    install = Path(install)
    key = root_key(install)
    target = CACHE_ROOT / key
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    # What a bad file can fail is read first, so a failed build leaves no 90 MB staging folder behind.
    step(0, "reading the game's palette and names")
    palette_bytes, ramp = _read_palette(install)
    strings, lang = _read_strings(install)
    staging = CACHE_ROOT / f"{key}.partial-{os.getpid()}-{secrets.token_hex(4)}"
    staging.mkdir()
    step(5, "unpacking the unit art")
    count = _write_graphics(install, staging, lambda share: step(5 + round(85 * share), "unpacking the unit art"))
    step(90, "copying the palette, terrain blends and names")
    write_palettes(staging, palette_bytes, ramp)
    blend = _find(install, BLENDOMATIC)
    if blend is not None and blend.is_file():
        (staging / BLENDOMATIC_SUBPATH).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(blend, staging / BLENDOMATIC_SUBPATH)
    _write_strings(staging, strings, lang)
    meta = {"format": ROOT_FORMAT, "key": key, "install": str(install), "language": lang, "graphics": count}
    (staging / META_NAME).write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    step(100, "done")
    return _place(staging, target)


def _place(staging: Path, target: Path) -> Path:
    try:
        os.rename(staging, target)
    except OSError:
        if (target / META_NAME).is_file():
            return target  # another build got there first; this staging folder stays behind
        os.rename(target, target.with_name(f"{target.name}.stale-{os.getpid()}-{secrets.token_hex(4)}"))
        os.rename(staging, target)
    return target


def _write_graphics(install: Path, out: Path, share: Callable[[float], None]) -> int:
    data_dir = _find(install, DATA_DIR)
    stack = drs_archive.aoc_graphics(data_dir)
    folder = out / GRAPHICS_SUBPATH
    folder.mkdir(parents=True)
    ids = stack.ids("slp")
    for i, slp_id in enumerate(ids):
        (folder / f"slp_{slp_id}.slp").write_bytes(stack.read("slp", slp_id))
        if i % 64 == 0:
            share(i / len(ids))
    return len(ids)


def _read_palette(install: Path) -> tuple[bytes, list[tuple[int, int, int]]]:
    """interfac.drs' JASC palette as stored, and the neutral player ramp from it. ValueError for a bad or short one."""
    raw = drs_archive.DrsArchive(_find(install, "Data/interfac.drs")).read("bina", PALETTE_ID)
    return raw, palette_ramp(raw, "interfac.drs")


def palette_ramp(raw: bytes, where: str) -> list[tuple[int, int, int]]:
    """The neutral player ramp from JASC palette `raw` (named `where` in errors). ValueError for a bad or short one."""
    lines = raw.decode("latin-1").splitlines()
    if len(lines) < 3 or lines[0].strip() != "JASC-PAL":
        raise ValueError(f"{where} {PALETTE_ID} is not a JASC-PAL palette")
    palette = [tuple(int(v) for v in line.split()[:3]) for line in lines[3:3 + int(lines[2])] if line.strip()]
    if len(palette) < GREY_RAMP_BASE + RAMP_LENGTH:
        raise ValueError(f"{where} palette {PALETTE_ID} holds {len(palette)} colours, not 256")
    return [palette[GREY_RAMP_BASE + i % RAMP_LENGTH] for i in range(256)]


def write_palettes(out: Path, raw: bytes, ramp) -> None:
    """The palette files the SMX model reads, under `out`'s PALETTE_SUBPATH."""
    folder = out / PALETTE_SUBPATH
    folder.mkdir(parents=True)
    (folder / PALETTE_NAME).write_bytes(raw)
    (folder / PALETTES_CONF).write_text(f"{PALETTE_ID},{PALETTE_NAME}\n", encoding="utf-8")
    (folder / WHITE_PLAYER_PALETTE).write_text(jasc_palette(ramp), encoding="latin-1")


def jasc_palette(colours) -> str:
    rows = "".join(f"{r} {g} {b}\r\n" for r, g, b in colours)
    return f"JASC-PAL\r\n0100\r\n{len(colours)}\r\n{rows}"


def _read_strings(install: Path) -> tuple[dict[int, str], str]:
    merged: dict[int, str] = {}
    langs: list[int] = []
    for name in STRINGS_DLLS:
        path = _find(install, name)
        if path is None or not path.is_file():
            continue
        table, lang_id = read_string_table(path.read_bytes())
        merged.update(table)
        if lang_id is not None:
            langs.append(lang_id)
    return merged, language_folder(langs[0]) if langs else DEFAULT_LANGUAGE


def _write_strings(out: Path, merged: dict[int, str], lang: str) -> None:
    path = out / STRINGS_SUBPATH.format(lang=lang)
    path.parent.mkdir(parents=True)
    path.write_text("".join(f'{key} "{_escape(text)}"\n' for key, text in sorted(merged.items())), encoding="utf-8")


def _escape(text: str) -> str:
    # DE's key-value files: a line break is a literal \n, a quote \" (asset_source._unescape_string_value()).
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "")


def language_folder(lang_id: int) -> str:
    """DE's resources/ folder name for a Windows resource language id, DEFAULT_LANGUAGE for one it does not know."""
    return _LANG_FOLDERS.get(lang_id) or _PRIMARY_LANG_FOLDERS.get(lang_id & 0x3FF, DEFAULT_LANGUAGE)


# -- PE string tables -----------------------------------------------------------

_RT_STRING = 6
_RESOURCE_DIRECTORY = 2  # the data directory index
_MAX_DIRECTORIES = 8192
_MAX_ENTRIES = 4096


def read_string_table(data: bytes) -> tuple[dict[int, str], int | None]:
    """A PE file's RT_STRING resources as {string id: text} (empty strings
    left out), and the first block's resource language id. Each block holds
    16 length-prefixed UTF-16 strings; string id = (block id - 1) * 16 + index.
    Raises ValueError for a file that is not a PE image."""
    if data[:2] != b"MZ":
        raise ValueError("not a PE file (no MZ header)")
    (pe,) = struct.unpack_from("<I", data, 0x3C)
    if data[pe:pe + 4] != b"PE\0\0":
        raise ValueError("not a PE file (no PE signature)")
    sections, optional_size = struct.unpack_from("<H12xH", data, pe + 6)
    optional = pe + 24
    (magic,) = struct.unpack_from("<H", data, optional)
    directories = optional + {0x10B: 96, 0x20B: 112}.get(magic, -1)
    if directories < optional:
        raise ValueError(f"unknown PE optional header magic {magic:#x}")
    rsrc_rva, _rsrc_size = struct.unpack_from("<II", data, directories + 8 * _RESOURCE_DIRECTORY)
    spans = []
    for i in range(sections):
        _vsize, vaddr, raw_size, raw_ptr = struct.unpack_from("<IIII", data, optional + optional_size + 40 * i + 8)
        spans.append((vaddr, min(raw_size, _vsize) or raw_size, raw_ptr))

    def offset_of(rva: int) -> int:
        for vaddr, size, raw_ptr in spans:
            if vaddr <= rva < vaddr + size:
                return rva - vaddr + raw_ptr
        raise ValueError(f"RVA {rva:#x} lies in no section")

    if rsrc_rva == 0:
        return {}, None
    base = offset_of(rsrc_rva)

    visited: set[int] = set()

    def entries(offset: int) -> list[tuple[int, int]]:
        # A directory seen twice is a cycle; a real table lists far fewer entries than the cap.
        if offset in visited or len(visited) > _MAX_DIRECTORIES:
            raise ValueError(f"resource directory at {offset:#x} repeats or there are too many")
        visited.add(offset)
        named, numbered = struct.unpack_from("<HH", data, base + offset + 12)
        if named + numbered > _MAX_ENTRIES:
            raise ValueError(f"resource directory at {offset:#x} lists {named + numbered} entries")
        start = base + offset + 16
        return [struct.unpack_from("<II", data, start + 8 * i) for i in range(named + numbered)]

    strings: dict[int, str] = {}
    language = None
    for type_id, type_offset in entries(0):
        if type_id != _RT_STRING or not type_offset & 0x80000000:
            continue
        for block_id, block_offset in entries(type_offset & 0x7FFFFFFF):
            if not block_offset & 0x80000000:
                continue
            for lang_id, leaf in entries(block_offset & 0x7FFFFFFF):
                if leaf & 0x80000000:
                    continue
                data_rva, size = struct.unpack_from("<II", data, base + leaf)
                if language is None:
                    language = lang_id
                if lang_id == language:
                    _read_block(data, offset_of(data_rva), size, block_id, strings)
    return strings, language


def _read_block(data: bytes, at: int, size: int, block_id: int, into: dict[int, str]) -> None:
    end = at + size
    for index in range(16):
        if at + 2 > end:
            break
        (length,) = struct.unpack_from("<H", data, at)
        at += 2
        text = data[at:at + 2 * length].decode("utf-16-le", errors="replace")
        at += 2 * length
        if text:
            into[(block_id - 1) * 16 + index] = text
