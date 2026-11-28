"""HD Edition's derived root: its install's loose files gathered once into DE's layout (TASK-431, H7).

The second recipe beside descape/legacy_content.py's, with the same public
functions, key rule, staging-then-rename build and cache folder shape
(CACHE_ROOT/<key>/), so game_dataset.install_root() serves either the same way.
HD Edition already keeps loose files, so the builder mostly links:

- each `<id>.slp` of drs/gamedata_x2, gamedata_x1 and graphics, the first
  folder in GRAPHICS_ORDER holding the id winning, as
  resources/_common/drs/graphics/slp_<id>.slp;
- terrain/textures/<name>_00_color.png (any case of the suffix) as
  resources/_common/terrain/textures/2x/<name>_00_color.png;
- dat/blendomatic_x1.dat at the same path;
- every language folder's strings files (Game.strings_files) at the same paths;
- drs/interface/50500.bina as the palette, palettes.conf and neutral player
  ramp, by legacy_content's code (HD's grey player keeps ramp base 128).

"Links" is a hardlink, else a symlink, else a copy: the first that works
on this machine is kept for the rest of the build (a cache on another
volume than the Steam library refuses hardlinks with EXDEV). A symlinked
root goes stale if the install moves or updates; the key (root_key(): the
art names and Steam's build id) names another root then. As
legacy_content, nothing here deletes. Qt-free.

GRAPHICS_ORDER: AoK HD.exe pairs each data set with its own folder
(gamedataset.cpp's strings: empires2_x2_p1.dat with gamedata_x2), searched
before the base drs/graphics; gamedata_x1's six SLPs are a byte-identical
subset of gamedata_x2's, so its place in the order changes nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
from collections.abc import Callable
from pathlib import Path

import platformdirs
import yaml

from descape import games, legacy_content
from descape.legacy_content import META_NAME, Progress, RootBuildError

# Bump on any change to what build_root() writes: every user's root is rebuilt once.
ROOT_FORMAT = 1
CACHE_ROOT = Path(platformdirs.user_cache_dir("DEscape", appauthor=False)) / "game_roots" / "hd"
INSTALL_ENV = "HD_INSTALL_PATH"
# config.yaml's key, spelled out where it is read and written (tests/test_settings_file_keys.py discovers it there).
INSTALL_KEY = "hd_install"
PLACEHOLDER = "The Age2HD folder holding resources/"

DRS_DIR = "resources/_common/drs"
# tools/_drs.HD_GRAPHICS_ORDER, which tests/test_hd_content.py pins this to.
GRAPHICS_ORDER = ("gamedata_x2", "gamedata_x1", "graphics")
PALETTE = f"{DRS_DIR}/interface/{legacy_content.PALETTE_ID}.bina"
TEXTURES_DIR = "resources/_common/terrain/textures"
TEXTURE_SUFFIX = "_00_color.png"
BLENDOMATIC = "resources/_common/dat/blendomatic_x1.dat"
STRINGS_DIR = "resources/{lang}/strings/key-value"
TEXTURES_SUBPATH = "resources/_common/terrain/textures/2x"
# Steam's folder name for HD Edition under <library>/steamapps/common/.
STEAM_FOLDER = "Age2HD"
STEAM_MANIFEST = "appmanifest_221380.acf"

_find = legacy_content._find
_failures: dict[str, str] = {}


def install_path() -> Path | None:
    """The configured HD Edition install: INSTALL_ENV, else config.yaml's INSTALL_KEY. None when unset or not a folder."""
    raw = os.environ.get(INSTALL_ENV)
    if not raw:
        from descape import asset_source

        try:
            config = yaml.safe_load(asset_source.read_config_text(asset_source.CONFIG_PATH)) or {}
        except (yaml.YAMLError, OSError):
            return None
        raw = config.get("hd_install") if isinstance(config, dict) else None
    if not raw:
        return None
    path = Path(str(raw))
    return path if path.is_dir() else None


def save_install_path(path: Path | None) -> None:
    """Persists the HD Edition install to config.yaml's INSTALL_KEY; None removes it."""
    from descape import asset_source

    asset_source.save_hd_install_to_config(path)


def browse_start() -> Path | None:
    """<steamapps>/common/Age2HD beside the AoE2:DE install's Steam library, when that folder exists."""
    from descape import asset_source

    de = asset_source.get_de_install_path()
    if de is None:
        return None
    steamapps = next((p for p in (de, *de.parents) if p.name.casefold() == "steamapps"), None)
    candidate = None if steamapps is None else steamapps / "common" / STEAM_FOLDER
    return candidate if candidate is not None and candidate.is_dir() else None


def _files(folder: Path | None, matches) -> list[os.DirEntry]:
    if folder is None or not folder.is_dir():
        return []
    with os.scandir(folder) as entries:
        return sorted((e for e in entries if matches(e.name) and e.is_file()), key=lambda e: e.name)


def _is_slp(name: str) -> bool:
    stem, dot, ext = name.rpartition(".")
    return bool(dot) and ext.casefold() == "slp" and stem.isdigit()


def _is_texture(name: str) -> bool:
    return name.casefold().endswith(TEXTURE_SUFFIX)


def validate_install(path: Path) -> tuple[bool, str]:
    """Whether `path` is an HD Edition install, with a status line either way. An AoE2:DE folder
    shares the .dat, graphics SLPs and palette; gamedata_x1/ and the PNG textures refuse it."""
    path = Path(path)
    if not path.is_dir():
        return False, f"Not a folder: {path}"
    missing = [rel for rel in games.HD.dat_subpaths if (found := _find(path, rel)) is None or not found.is_file()]
    if (found := _find(path, f"{DRS_DIR}/gamedata_x1")) is None or not found.is_dir():
        missing.append(f"{DRS_DIR}/gamedata_x1/")
    if not _files(_find(path, f"{DRS_DIR}/graphics"), _is_slp):
        missing.append(f"{DRS_DIR}/graphics/*.slp")
    if (found := _find(path, PALETTE)) is None or not found.is_file():
        missing.append(PALETTE)
    if not _files(_find(path, TEXTURES_DIR), _is_texture):
        missing.append(f"{TEXTURES_DIR}/*{TEXTURE_SUFFIX}")
    if missing:
        return False, f"Not an Age of Empires II: HD Edition install: missing {', '.join(missing)}"
    return True, "HD Edition install found."


def _sources(install: Path) -> dict[str, list[tuple[Path, str]]]:
    """Every install file build_root() reads, by kind, as (source, path under the root)."""
    graphics: dict[str, tuple[Path, str]] = {}
    for name in GRAPHICS_ORDER:
        for entry in _files(_find(install, f"{DRS_DIR}/{name}"), _is_slp):
            stem = entry.name.rpartition(".")[0]
            graphics.setdefault(str(int(stem)), (Path(entry.path), f"{legacy_content.GRAPHICS_SUBPATH}/slp_{int(stem)}.slp"))
    textures: dict[str, tuple[Path, str]] = {}
    for entry in _files(_find(install, TEXTURES_DIR), _is_texture):
        rel = f"{TEXTURES_SUBPATH}/{entry.name[: -len(TEXTURE_SUFFIX)]}{TEXTURE_SUFFIX}"
        textures.setdefault(rel.casefold(), (Path(entry.path), rel))  # x_00_color.png and x_00_COLOR.png: one target
    return {"graphics": list(graphics.values()), "textures": list(textures.values()), "other": _other_sources(install)}


def _other_sources(install: Path) -> list[tuple[Path, str]]:
    """blendomatic and every language folder's strings files, as (source, path under the root)."""
    other = []
    if (blend := _find(install, BLENDOMATIC)) is not None and blend.is_file():
        other.append((blend, BLENDOMATIC))
    resources = _find(install, "resources")
    for lang in sorted(p.name for p in resources.iterdir() if p.is_dir()) if resources is not None else ():
        folder = _find(install, STRINGS_DIR.format(lang=lang))
        other.extend(
            (path, f"{STRINGS_DIR.format(lang=lang)}/{name}")
            for name in games.HD.strings_files
            if folder is not None and (path := _find(folder, name)) is not None and path.is_file()
        )
    return other


def source_files(install: Path) -> list[Path]:
    """Every install file build_root() reads, the ones present (root_key() stats their folders instead)."""
    found = [source for group in _sources(install).values() for source, _ in group]
    palette = _find(install, PALETTE)
    return [*found, palette] if palette is not None and palette.is_file() else found


def _key_folders(install: Path) -> list[tuple[Path, Callable[[str], bool]]]:
    rels = [(f"{DRS_DIR}/{name}", _is_slp) for name in GRAPHICS_ORDER] + [(TEXTURES_DIR, _is_texture)]
    return [(p, matches) for rel, matches in rels if (p := _find(install, rel)) is not None and p.is_dir()]


def _key_files(install: Path) -> list[Path]:
    files = [source for source, _ in _other_sources(install)]
    palette = _find(install, PALETTE)
    return [*files, palette] if palette is not None and palette.is_file() else files


def _steam_buildid(install: Path) -> str | None:
    """The build id in `<steamapps>/appmanifest_221380.acf` beside a Steam install, else None."""
    manifest = install.parent.parent / STEAM_MANIFEST
    try:
        text = manifest.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    match = re.search(r'"buildid"\s+"(\d+)"', text)
    return match.group(1) if match else None


def root_key(install: Path) -> str:
    """The derived root's folder name: ROOT_FORMAT, the install path, the art and texture
    names each folder holds, the other source files' size and mtime, and Steam's build id.
    Stat-ing every SLP took about 41 ms a call, several per open; the folders' mtimes also
    move for the scenarios, sounds and AI files saved beside the art, so neither is keyed."""
    install = Path(install)
    entries = []
    for folder, matches in _key_folders(install):
        with os.scandir(folder) as found:
            names = sorted(e.name for e in found if matches(e.name))
        entries.append([str(folder.relative_to(install)).casefold(), hashlib.sha256("\n".join(names).encode()).hexdigest()])
    for path in _key_files(install):
        stat = path.stat()
        entries.append([str(path.relative_to(install)).casefold(), stat.st_size, stat.st_mtime_ns])
    payload = {"format": ROOT_FORMAT, "install": str(install.resolve()), "files": entries, "build": _steam_buildid(install)}
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
        message = f"Could not gather the HD Edition install's files: {type(exc).__name__}: {exc}"
        _failures[key] = message
        raise RootBuildError(message) from exc


class _Linker:
    """Places a file by the first of hardlink, symlink and copy that works, then keeps to it."""

    MODES = ("hardlink", "symlink", "copy")

    def __init__(self) -> None:
        self.mode = self.MODES[0]

    def place(self, source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                if self.mode == "hardlink":
                    os.link(source, target)
                elif self.mode == "symlink":
                    os.symlink(source.resolve(), target)
                else:
                    shutil.copyfile(source, target)
                return
            except OSError:
                if self.mode == "copy":
                    raise
                self.mode = self.MODES[self.MODES.index(self.mode) + 1]


def build_root(install: Path, progress: Progress | None = None) -> Path:
    """Gathers `install` into CACHE_ROOT/<root_key()>/ and returns it. A complete root another build placed first is used."""
    step = progress or (lambda _percent, _caption: None)
    install = Path(install)
    key = root_key(install)
    target = CACHE_ROOT / key
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    step(0, "reading the game's palette")
    raw = _find(install, PALETTE).read_bytes()
    ramp = legacy_content.palette_ramp(raw, PALETTE)
    sources = _sources(install)
    staging = CACHE_ROOT / f"{key}.partial-{os.getpid()}-{secrets.token_hex(4)}"
    staging.mkdir()
    linker = _Linker()
    graphics = sources["graphics"]
    for i, (source, rel) in enumerate(graphics):
        linker.place(source, staging / rel)
        if i % 64 == 0:
            step(5 + round(80 * i / len(graphics)), "linking the unit art")
    step(85, "linking the terrain textures, blends and names")
    for source, rel in (*sources["textures"], *sources["other"]):
        linker.place(source, staging / rel)
    legacy_content.write_palettes(staging, raw, ramp)
    meta = {
        "format": ROOT_FORMAT, "key": key, "install": str(install), "graphics": len(graphics),
        "textures": len(sources["textures"]), "placed_by": linker.mode,
    }
    (staging / META_NAME).write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    step(100, "done")
    return legacy_content._place(staging, target)
