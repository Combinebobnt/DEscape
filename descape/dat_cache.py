"""The on-disk cache of a data mod's derived tables (GH #58).

Deriving a mod's tables parses its .dat (5-18 s, pure Python), so the result is
kept under CACHE_ROOT/<key>/: one JSON file per dat_derive.TABLE_NAMES name,
plus META_NAME. The key hashes every input, not just the mod .dat: the mod
.dat's sha256, the install's fingerprint (install_fingerprint(), the files the
generators read besides the .dat), dat_derive.source_digest() (the generators'
own source) and the genieutils-py version, so a game patch, a generator edit or
a parser bump misses the cache rather than serving stale tables.

An entry is never written in place. build_entry() parses the mod .dat first, so
an unreadable one (most failures) fails before any folder exists, then derives
into a staging folder of its own (<key>.partial-<pid>-<random>) and renames it
to <key> whole. A complete entry therefore never changes under a viewer reading
it, and two builds of one key cannot interleave: the second rename finds the
first's entry and uses it. A build that dies mid-derive, or loses that race,
leaves its staging folder behind, never a half entry under the real name.
Nothing here deletes: old entries and leftover staging folders stay until the
folder is deleted by hand, which is safe while no data mod is active.

meta.json also carries `mod_only_ids`, the unit consts the mod's .dat defines and
the install's does not, which the catalog tags (object_catalog.DATA_MOD_TAG).
The install's id set is parsed once per install fingerprint and kept beside the
entries.

Writes only under the folder it is given, CACHE_ROOT for the viewer, DEscape's
own user cache folder; write_refusal() refuses a destination inside the game
profile or a mods/subscribed/ folder. Reading a mod .dat from there is fine.
Qt-free.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import importlib.metadata
import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path

import platformdirs

from descape import dat_derive
from descape.scenario_io import is_in_game_profile, is_workshop_subscribed

CACHE_ROOT = Path(platformdirs.user_cache_dir("DEscape", appauthor=False)) / "dat_tables"
META_NAME = "meta.json"
INSTALL_IDS_DIR = "install_ids"
DAT_SUBPATH = Path("resources") / "_common" / "dat" / "empires2_x2_p1.dat"
# Install files the generators read besides the .dat, by size and mtime (folders by mtime).
_FINGERPRINT_PATHS = (
    DAT_SUBPATH,
    Path("resources") / "_common" / "palettes" / "original.pal",
    Path("resources") / "_common" / "drs" / "graphics",
    Path("resources") / "_common" / "particles",
    Path("resources") / "_common" / "terrain" / "textures" / "2x",
)


@dataclass(frozen=True)
class Entry:
    """A complete cache entry: its key, its folder and the mod-only unit consts."""

    key: str
    path: Path
    mod_only_ids: frozenset[int]


def write_refusal(path: Path) -> str | None:
    """Why `path` may not be written to, or None. The cache never writes into
    the game's profile (Proton compatdata/ included) or a mods/subscribed/ folder."""
    try:
        if is_workshop_subscribed(path) or is_in_game_profile(path):
            return f"{path} is inside the game's profile or mods folder"
    except (OSError, RuntimeError):
        return f"{path} cannot be resolved"
    return None


@functools.lru_cache(maxsize=64)
def _sha256_of(path: str, size: int, mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    """The file's sha256, hashed once per path, size and mtime (a mod .dat is 4-12 MB)."""
    stat = Path(path).stat()
    return _sha256_of(str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns)


def install_fingerprint(install: Path) -> dict:
    """The install path plus each _FINGERPRINT_PATHS entry's size (files) and mtime, None when absent."""
    found: dict[str, object] = {"install": str(Path(install).resolve())}
    for sub in _FINGERPRINT_PATHS:
        path = Path(install) / sub
        try:
            stat = path.stat()
        except OSError:
            found[sub.as_posix()] = None
            continue
        found[sub.as_posix()] = [stat.st_size if path.is_file() else None, stat.st_mtime_ns]
    return found


def genieutils_version() -> str | None:
    try:
        return importlib.metadata.version("genieutils-py")
    except importlib.metadata.PackageNotFoundError:
        return None


def _digest(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:32]


def entry_inputs(mod_dat: Path, install: Path) -> dict:
    """Everything an entry is derived from, as recorded in its meta.json."""
    return {
        "mod_dat": str(mod_dat),
        "mod_dat_sha256": file_sha256(mod_dat),
        "install": install_fingerprint(install),
        "source_digest": dat_derive.source_digest(),
        "genieutils": genieutils_version(),
    }


def entry_key(inputs: dict) -> str:
    """The cache folder name for entry_inputs(); the mod .dat's path is not part of it."""
    return _digest({key: value for key, value in inputs.items() if key != "mod_dat"})


def _read_entry(folder: Path, key: str) -> Entry | None:
    try:
        meta = json.loads((folder / META_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not all((folder / name).is_file() for name in dat_derive.TABLE_NAMES):
        return None
    ids = meta.get("mod_only_ids") if isinstance(meta, dict) else None
    if not isinstance(ids, list):
        return None
    return Entry(key, folder, frozenset(int(i) for i in ids))


def lookup(key: str, root: Path | None = None) -> Entry | None:
    """The complete entry named `key`, or None."""
    root = CACHE_ROOT if root is None else root
    return _read_entry(root / key, key)


def _unique_sibling(path: Path, label: str) -> Path:
    return path.with_name(f"{path.name}.{label}-{os.getpid()}-{secrets.token_hex(4)}")


def _write_json(path: Path, payload: dict) -> None:
    tmp = _unique_sibling(path, "tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _install_ids(install: Path, root: Path, parsed: dict) -> set[int]:
    """The install .dat's unit consts, kept per install fingerprint and derivation source under `root`."""
    name = _digest({"install": install_fingerprint(install), "source_digest": dat_derive.source_digest()})
    path = root / INSTALL_IDS_DIR / f"{name}.json"
    with contextlib.suppress(OSError, ValueError, KeyError, TypeError):
        return {int(i) for i in json.loads(path.read_text(encoding="utf-8"))["ids"]}
    ids = dat_derive.parse_unit_ids(Path(install) / DAT_SUBPATH, parsed)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, {"ids": sorted(ids)})
    return ids


def _place(staging: Path, out_dir: Path) -> Entry:
    """Renames the finished `staging` to `out_dir`. When another build got there
    first its entry is used; an incomplete folder in the way is renamed aside."""
    try:
        os.rename(staging, out_dir)
    except OSError:
        existing = _read_entry(out_dir, out_dir.name)
        if existing is not None:
            return existing
        os.rename(out_dir, _unique_sibling(out_dir, "stale"))
        os.rename(staging, out_dir)
    entry = _read_entry(out_dir, out_dir.name)
    if entry is None:
        raise RuntimeError(f"{out_dir} is incomplete after deriving")
    return entry


def build_entry(mod_dat: Path, install: Path, out_dir: Path) -> Entry:
    """Derives `mod_dat`'s tables and places them at `out_dir` (CACHE_ROOT/<key>
    as the viewer calls it), returning the entry. Raises on any failure, which
    never leaves a complete-looking entry at `out_dir`."""
    out_dir = Path(out_dir)
    refusal = write_refusal(out_dir)
    if refusal is not None:
        raise ValueError(f"refusing to write the table cache: {refusal}")
    inputs = entry_inputs(mod_dat, install)
    parsed: dict = {}
    # First, so a .dat genieutils cannot read fails before any folder exists.
    mod_ids = dat_derive.parse_unit_ids(Path(mod_dat), parsed)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = _unique_sibling(out_dir, "partial")
    dat_derive.derive(Path(mod_dat), Path(install), staging, parsed=parsed)
    # The mod's parse (~1 GB resident) goes before the install's is made.
    parsed.clear()
    install_ids = _install_ids(Path(install), out_dir.parent, parsed)
    _write_json(staging / META_NAME, {**inputs, "mod_only_ids": sorted(mod_ids - install_ids)})
    return _place(staging, out_dir)
