"""Map prefabs: clipboard blocks saved for good (Qt-free).

A prefab is one RegionBlock in its own file in prefab_dir(), beside
config.yaml. Using one pushes its block onto the session clipboard, so paste
and every paste refusal stay exactly as for a copy: nothing here writes a
scenario.

The file is gzip of two JSON lines. Line 1 is the header, which the list
reads alone, so a folder of whole-map prefabs lists without parsing tiles.
Line 2 is the body: the tile arrays and one object per unit. Every finite
float is a plain JSON number (shortest repr round-trips an f32 exactly); a
non-finite z or rotation is {"f32": "<8 hex digits>"}, its little-endian
bytes, because JSON would lose a NaN's sign and payload.

Loading validates, never clamps: ranges are the storage types', since a copy
is verbatim and a prefab must carry whatever a scenario can. Any bad field
refuses the whole file. Unknown keys are ignored and a newer version is
refused. save_block() validates its own encoding the same way, so it never
writes a file that would not load back. The session-only fields
(RegionUnit.cut_reference_id, RegionBlock.cut_doc_id) are never saved.
"""

from __future__ import annotations

import base64
import datetime
import gzip
import io
import json
import math
import re
import struct
import unicodedata
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from descape import asset_source, clipboard_history, games
from descape.region_clipboard import RegionBlock, RegionUnit
from descape.scenario_io import is_workshop_subscribed
from descape.yaml_file_io import atomic_write, is_int

FORMAT_ID = "descape-prefab"
VERSION = 1
SUFFIX = ".descape-prefab"
PREFAB_DIRNAME = "prefabs"
HEADER_MAX_BYTES = 256 * 1024
BODY_MAX_BYTES = 64 * 1024 * 1024
NAME_MAX = 80
SLUG_MAX = 48
THUMBNAIL_MAX = clipboard_history.THUMBNAIL_MAX_TILES
_F32_MAX = struct.unpack("<f", b"\xff\xff\x7f\x7f")[0]
_WINDOWS_DEVICES = frozenset(["con", "prn", "aux", "nul"] + [f"com{i}" for i in range(1, 10)] + [f"lpt{i}" for i in range(1, 10)])

# The saved RegionUnit fields, in body order. A new RegionUnit field is saved
# only once it is added here (tests/test_prefabs.py fails until it is placed).
UNIT_INT_FIELDS = {
    "player": (0, 8),
    "unit_const": (0, 0xFFFF),
    "status": (0, 0xFF),
    "initial_animation_frame": (0, 0xFFFF),
    "caption_string_id": (-(2**31), 2**31 - 1),
    "garrison_slot": (-1, 2**31 - 1),
    "capture_flag": (-128, 127),
}
UNIT_FLOAT_FIELDS = ("dx", "dy")
UNIT_F32_FIELDS = ("z", "rotation")  # may be non-finite
UNIT_STR_FIELDS = ("caption_string",)
UNIT_SESSION_ONLY = frozenset({"cut_reference_id"})
BLOCK_SESSION_ONLY = frozenset({"cut_doc_id"})
_MIN_UNIT_BYTES = len(
    json.dumps(
        dict.fromkeys((*UNIT_INT_FIELDS, *UNIT_FLOAT_FIELDS, *UNIT_F32_FIELDS), 0) | dict.fromkeys(UNIT_STR_FIELDS, ""),
        separators=(",", ":"),
    )
)
_TILE_RANGES = {"terrain_ids": (0, 0xFF), "elevations": (0, 0xFF), "layers": (-(2**15), 2**15 - 1)}


class PrefabRefused(Exception):
    """A prefab file that cannot be read, or a save or rename that is refused."""


class PrefabNameTaken(PrefabRefused):
    """Another prefab already has the name; `path` is that file."""

    def __init__(self, name: str, path: Path):
        super().__init__(f'a prefab named "{name}" already exists')
        self.path = path


class PrefabWriteBlocked(PrefabRefused):
    """The prefab folder is somewhere DEscape never writes (mods/subscribed/)."""


@dataclass(frozen=True)
class PrefabInfo:
    """One prefab file's header, or `error` (with the stem as its name) when it can't be listed."""

    path: Path
    name: str
    saved: str = ""
    width: int = 0
    height: int = 0
    has_tiles: bool = False
    unit_count: int = 0
    dataset: str = games.DEFAULT_ID
    legacy: bool = False
    thumbnail: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 3), dtype=np.uint8), compare=False)
    error: str | None = None


def prefab_dir() -> Path:
    """Resolved at call time, as autosave.autosave_dir() is, so tests' redirected CONFIG_PATH applies."""
    return asset_source.CONFIG_PATH.parent / PREFAB_DIRNAME


# -- names and slugs -------------------------------------------------------


def name_problem(name) -> str | None:
    """Why `name` (already stripped) can't name a prefab, or None."""
    if not isinstance(name, str) or not name:
        return "a prefab needs a name"
    if len(name) > NAME_MAX:
        return f"a prefab name is at most {NAME_MAX} characters"
    if name != name.strip():
        return "a prefab name may not start or end with a space"
    if any(unicodedata.category(c) == "Cc" for c in name):
        return "a prefab name may not contain control characters"
    if not _encodable(name):
        return "a prefab name may not contain unpaired surrogates"
    return None


def _encodable(text: str) -> bool:
    """False for a lone surrogate, which JSON can carry but neither a prefab nor a scenario can save."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _checked_name(name: str) -> str:
    name = name.strip() if isinstance(name, str) else name
    problem = name_problem(name)
    if problem:
        raise PrefabRefused(problem)
    return name


def slug_for(name: str) -> str:
    """The file stem a name gets before any collision suffix."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name).strip("-")[:SLUG_MAX].strip("-") or "prefab"
    return f"{slug}-prefab" if slug in _WINDOWS_DEVICES else slug


def _free_path(folder: Path, name: str) -> Path:
    base = slug_for(name)
    candidate, n = folder / f"{base}{SUFFIX}", 1
    while candidate.exists():
        n += 1
        candidate = folder / f"{base}-{n}{SUFFIX}"
    return candidate


def _same(a: Path, b: Path) -> bool:
    try:
        return a.exists() and b.exists() and a.samefile(b)
    except OSError:
        return False


# -- encoding --------------------------------------------------------------


def _f32_value(value: float):
    if math.isfinite(value):
        return value
    return {"f32": struct.pack("<f", value).hex()}


def _encode(block: RegionBlock, name: str, saved: str) -> bytes:
    thumb = clipboard_history.thumbnail_rgb(block)
    h, w = thumb.shape[:2]
    header = {
        "format": FORMAT_ID,
        "version": VERSION,
        "name": name,
        "saved": saved,
        "width": block.width,
        "height": block.height,
        "has_tiles": block.has_tiles,
        "unit_count": len(block.units),
        "dataset": block.dataset,
        "legacy": block.legacy,
        "thumbnail": {"w": w, "h": h, "rgb": base64.b64encode(thumb.tobytes()).decode("ascii")},
    }
    units = []
    for unit in block.units:
        obj = {key: getattr(unit, key) for key in UNIT_INT_FIELDS}
        obj.update({key: getattr(unit, key) for key in UNIT_FLOAT_FIELDS})
        obj.update({key: _f32_value(getattr(unit, key)) for key in UNIT_F32_FIELDS})
        obj.update({key: getattr(unit, key) for key in UNIT_STR_FIELDS})
        units.append(obj)
    body = {
        "terrain_ids": list(block.terrain_ids),
        "elevations": list(block.elevations),
        "layers": list(block.layers),
        "units": units,
    }
    try:
        lines = [json.dumps(part, ensure_ascii=False, allow_nan=False, separators=(",", ":")) for part in (header, body)]
        data = ("\n".join(lines) + "\n").encode("utf-8")
    except ValueError as e:  # a non-finite dx or dy, or a lone surrogate (UnicodeEncodeError)
        raise PrefabRefused(f"the block cannot be saved as a prefab ({e})") from e
    return gzip.compress(data, compresslevel=6, mtime=0)


# -- decoding --------------------------------------------------------------


def _stream_read(read, *args) -> bytes:
    try:
        return read(*args)
    except (EOFError, zlib.error, OSError) as e:  # gzip.BadGzipFile is an OSError
        raise PrefabRefused(f"not a readable prefab file ({e or type(e).__name__})") from e


def _read_raw(stream, limit: int, what: str) -> bytes:
    raw = _stream_read(stream.readline, limit + 1)
    if len(raw) > limit:
        raise PrefabRefused(f"the {what} is over the {limit}-byte limit for a prefab file")
    if not raw:
        raise PrefabRefused(f"the file has no {what}")
    return raw


_STRING = re.compile(rb'"(?:[^"\\]|\\.)*"')
_SHAPE_SLACK = 256


def _check_body_shape(raw: bytes, info: PrefabInfo) -> None:
    """Bounds what json.loads() may build before it runs: a flood of empty
    objects, lists or strings costs about 20x its bytes in memory. A unit
    object can't be shorter than _MIN_UNIT_BYTES, so its count is bounded by
    the bytes as well as by the header's claim."""
    rest, strings = _STRING.subn(b"", raw)
    units = min(info.unit_count, len(raw) // _MIN_UNIT_BYTES)
    for found, allowed, what in (
        (rest.count(b"{"), 1 + 3 * units, "objects"),
        (rest.count(b"["), 4, "lists"),
        (strings, 4 + 17 * units, "strings"),
    ):
        if found > allowed + _SHAPE_SLACK:
            raise PrefabRefused(f"the body holds more {what} than {info.unit_count} units can ({found})")


def _decode(raw: bytes, what: str) -> dict:
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as e:
        raise PrefabRefused(f"the {what} is not valid JSON") from e
    except ValueError as e:  # an integer over Python's 4300-digit conversion limit
        raise PrefabRefused(f"the {what} is not valid JSON ({e})") from e
    if not isinstance(obj, dict):
        raise PrefabRefused(f"the {what} is not a JSON object")
    return obj


def _require(obj: dict, key: str, what: str):
    if key not in obj:
        raise PrefabRefused(f"the {what} has no {key!r}")
    return obj[key]


def _positive_int(obj: dict, key: str) -> int:
    value = _require(obj, key, "header")
    if not is_int(value) or not 1 <= value <= 2**31 - 1:
        raise PrefabRefused(f"the header's {key} is not a positive whole number in range")
    return value


def _thumbnail(obj) -> np.ndarray:
    if not isinstance(obj, dict):
        raise PrefabRefused("the header's thumbnail is not an object")
    w, h, rgb = obj.get("w"), obj.get("h"), obj.get("rgb")
    if not (is_int(w) and is_int(h) and 0 <= w <= THUMBNAIL_MAX and 0 <= h <= THUMBNAIL_MAX) or not isinstance(rgb, str):
        raise PrefabRefused("the header's thumbnail size is not valid")
    try:
        data = base64.b64decode(rgb.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as e:
        raise PrefabRefused("the header's thumbnail is not valid base64") from e
    if len(data) != w * h * 3:
        raise PrefabRefused("the header's thumbnail does not match its size")
    return np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3)).copy()


def _header(obj: dict, path: Path) -> PrefabInfo:
    found = obj.get("format")
    if found != FORMAT_ID:
        raise PrefabRefused("not a DEscape prefab file")
    version = obj.get("version")
    if not is_int(version) or version < 1:
        raise PrefabRefused("unreadable prefab file version")
    if version > VERSION:
        raise PrefabRefused(f"prefab file version {version} is newer than this DEscape reads ({VERSION})")
    name = _require(obj, "name", "header")
    problem = name_problem(name)
    if problem:
        raise PrefabRefused(f"the header's name is not valid: {problem}")
    saved = _require(obj, "saved", "header")
    if not isinstance(saved, str):
        raise PrefabRefused("the header's saved time is not text")
    has_tiles, legacy = _require(obj, "has_tiles", "header"), _require(obj, "legacy", "header")
    if not isinstance(has_tiles, bool) or not isinstance(legacy, bool):
        raise PrefabRefused("the header's has_tiles or legacy is not true or false")
    dataset = _require(obj, "dataset", "header")
    if not isinstance(dataset, str) or dataset not in games.ids():
        raise PrefabRefused("the header names a game this DEscape does not know")
    unit_count = _require(obj, "unit_count", "header")
    if not is_int(unit_count) or unit_count < 0:
        raise PrefabRefused("the header's unit count is not valid")
    return PrefabInfo(
        path=path,
        name=name,
        saved=saved,
        width=_positive_int(obj, "width"),
        height=_positive_int(obj, "height"),
        has_tiles=has_tiles,
        unit_count=unit_count,
        dataset=dataset,
        legacy=legacy,
        thumbnail=_thumbnail(_require(obj, "thumbnail", "header")),
    )


def _int_list(body: dict, key: str, lo: int, hi: int) -> tuple[int, ...]:
    values = _require(body, key, "body")
    if not isinstance(values, list) or not all(type(v) is int for v in values):
        raise PrefabRefused(f"the body's {key} is not a list of whole numbers")
    if values and (min(values) < lo or max(values) > hi):
        raise PrefabRefused(f"the body's {key} has a value outside {lo}..{hi}")
    return tuple(values)


def _finite(value, key: str, index: int) -> float:
    # abs() first: isfinite() of an int past float range raises OverflowError.
    if type(value) not in (int, float) or abs(value) > _F32_MAX or not math.isfinite(value):
        raise PrefabRefused(f"unit {index}'s {key} is not a finite number in range")
    return float(value)


def _f32(value, key: str, index: int) -> float:
    if isinstance(value, dict):
        text = value.get("f32")
        if len(value) == 1 and isinstance(text, str) and re.fullmatch(r"[0-9a-fA-F]{8}", text):
            return struct.unpack("<f", bytes.fromhex(text))[0]
        raise PrefabRefused(f"unit {index}'s {key} is not a valid f32 object")
    return _finite(value, key, index)


def _unit(obj, index: int, count: int) -> RegionUnit:
    if not isinstance(obj, dict):
        raise PrefabRefused(f"unit {index} is not a JSON object")
    values = {}
    for key, (lo, hi) in UNIT_INT_FIELDS.items():
        value = _require(obj, key, f"unit {index}")
        if type(value) is not int or not lo <= value <= hi:
            raise PrefabRefused(f"unit {index}'s {key} is not a whole number in {lo}..{hi}")
        values[key] = value
    for key in UNIT_FLOAT_FIELDS:
        values[key] = _finite(_require(obj, key, f"unit {index}"), key, index)
    for key in UNIT_F32_FIELDS:
        values[key] = _f32(_require(obj, key, f"unit {index}"), key, index)
    for key in UNIT_STR_FIELDS:
        value = _require(obj, key, f"unit {index}")
        if not isinstance(value, str) or not _encodable(value):
            raise PrefabRefused(f"unit {index}'s {key} is not text")
        values[key] = value
    slot = values["garrison_slot"]
    if slot != -1 and (slot >= count or slot == index):
        raise PrefabRefused(f"unit {index}'s garrison_slot names no other unit in the prefab")
    return RegionUnit(**values)


def _block(info: PrefabInfo, body: dict) -> RegionBlock:
    tiles = {key: _int_list(body, key, lo, hi) for key, (lo, hi) in _TILE_RANGES.items()}
    expected = info.width * info.height if info.has_tiles else 0
    if any(len(values) != expected for values in tiles.values()):
        raise PrefabRefused(f"the body's tile lists do not hold {expected} tiles each")
    units = _require(body, "units", "body")
    if not isinstance(units, list):
        raise PrefabRefused("the body's units is not a list")
    if len(units) != info.unit_count:
        raise PrefabRefused(f"the body holds {len(units)} units, the header says {info.unit_count}")
    return RegionBlock(
        width=info.width,
        height=info.height,
        units=tuple(_unit(obj, i, len(units)) for i, obj in enumerate(units)),
        dataset=info.dataset,
        legacy=info.legacy,
        **tiles,
    )


def _parse(raw, path: Path, *, body: bool) -> tuple[PrefabInfo, RegionBlock | None]:
    with gzip.GzipFile(fileobj=raw, mode="rb") as stream:
        info = _header(_decode(_read_raw(stream, HEADER_MAX_BYTES, "header"), "header"), path)
        if not body:
            return info, None
        raw_body = _read_raw(stream, BODY_MAX_BYTES, "body")
        _check_body_shape(raw_body, info)
        block = _block(info, _decode(raw_body, "body"))
        # Reading to the end is what makes gzip check its CRC and length trailer.
        if _stream_read(stream.read, 1):
            raise PrefabRefused("the file has data after its body")
        return info, block


def _open(path: Path, *, body: bool) -> tuple[PrefabInfo, RegionBlock | None]:
    try:
        with open(path, "rb") as raw:
            return _parse(raw, path, body=body)
    except OSError as e:
        raise PrefabRefused(f"cannot read the file: {e}") from e


def read_info(path: Path) -> PrefabInfo:
    """The header alone, raising PrefabRefused."""
    return _open(Path(path), body=False)[0]


def load(path: Path) -> tuple[PrefabInfo, RegionBlock]:
    """The header and the block, raising PrefabRefused."""
    return _open(Path(path), body=True)


def load_block(path: Path) -> RegionBlock:
    return load(path)[1]


def list_prefabs(folder: Path | None = None) -> list[PrefabInfo]:
    """Every prefab file in the folder, by name case-insensitively. An unreadable
    one is listed with its reason, never dropped and never raised."""
    folder = prefab_dir() if folder is None else folder
    try:
        paths = [p for p in folder.iterdir() if p.suffix == SUFFIX and p.is_file()]
    except OSError:
        return []
    out = []
    for path in paths:
        try:
            out.append(read_info(path))
        except PrefabRefused as e:
            out.append(PrefabInfo(path=path, name=path.stem, error=str(e)))
    return sorted(out, key=lambda info: (info.name.casefold(), info.path.name))


def find_by_name(name: str, folder: Path | None = None) -> Path | None:
    key = name.strip().casefold()
    return next((i.path for i in list_prefabs(folder) if i.error is None and i.name.casefold() == key), None)


# -- writing ---------------------------------------------------------------


def _guard(folder: Path) -> None:
    try:
        blocked = is_workshop_subscribed(folder)
    except (OSError, RuntimeError) as e:  # a symlink loop raises on every Python version
        raise PrefabWriteBlocked(f"refusing to write a prefab to a folder that cannot be resolved ({e}): {folder}") from e
    if blocked:
        raise PrefabWriteBlocked(f"refusing to write a prefab into a subscribed-mods (mods/subscribed/) folder: {folder}")


def ensure_dir(folder: Path | None = None) -> Path:
    """The prefab folder, created after the write guard; raises PrefabRefused."""
    folder = prefab_dir() if folder is None else folder
    _guard(folder)
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise PrefabRefused(f"cannot create the prefab folder: {e}") from e
    return folder


def _write(block: RegionBlock, name: str, saved: str, path: Path) -> None:
    data = _encode(block, name, saved)
    _parse(io.BytesIO(data), path, body=True)  # never write what would not load back
    ensure_dir(path.parent)
    try:
        atomic_write(path, data)
    except OSError as e:
        raise PrefabRefused(f"cannot write the prefab: {e}") from e


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def save_block(block: RegionBlock, name: str, *, replace_path: Path | None = None, folder: Path | None = None) -> Path:
    """Saves `block` as a new prefab, or over `replace_path` (the file the
    name is taken by, after the user agreed). Raises PrefabNameTaken for a
    taken name without it, and PrefabRefused for anything else refused."""
    name = _checked_name(name)
    folder = prefab_dir() if folder is None else folder
    _guard(folder)
    taken = find_by_name(name, folder)
    if taken is not None and (replace_path is None or not _same(taken, Path(replace_path))):
        raise PrefabNameTaken(name, taken)
    path = Path(replace_path) if replace_path is not None else _free_path(folder, name)
    _write(block, name, _now(), path)
    return path


def rename(path: Path, name: str) -> Path:
    """Renames the prefab at `path`, keeping its saved time. The new file is
    written before the old one goes, unless the new name's slug is the old
    file's own, which replaces it in place. Returns the new path."""
    path = Path(path)
    name = _checked_name(name)
    info, block = load(path)
    folder = path.parent
    _guard(folder)
    taken = find_by_name(name, folder)
    if taken is not None and not _same(taken, path):
        raise PrefabNameTaken(name, taken)
    target = path if _slug_owner(path, name) else _free_path(folder, name)
    _write(block, name, info.saved, target)
    if not _same(target, path):
        try:
            path.unlink()
        except OSError as e:
            raise PrefabRefused(f"saved the renamed prefab, but could not remove the old file: {e}") from e
    return target


def _slug_owner(path: Path, name: str) -> bool:
    """Whether `path`'s stem is `name`'s slug, with or without a collision suffix."""
    base = slug_for(name)
    stem = path.stem
    if stem == base:
        return True
    if re.fullmatch(re.escape(base) + r"-\d+", stem) is None:
        return False
    # A -N suffix is the slug's own only where _free_path() would have added it.
    base_file = path.with_name(base + SUFFIX)
    return base_file.exists() and not _same(base_file, path)


def delete(path: Path) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError as e:
        raise PrefabRefused(f"cannot delete the prefab: {e}") from e
