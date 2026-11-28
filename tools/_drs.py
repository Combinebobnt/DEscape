"""Read-only access to Ensemble's DRS archives (version `1.00`) and the SLP headers inside them.

Stdlib only, so a frozen build can bundle it on its own. A DRS file is a
64-byte header (copyright text, version, type tag, table count, first-member
offset), then one 12-byte entry per member type (a reversed, space-padded
extension such as `" pls"` for "slp", its table's offset and member count),
then each table's 12-byte (id, offset, size) rows. `DrsArchive` reads those
tables once and serves member bytes on demand.

`AOC_GRAPHICS_ORDER` is The Conquerors 1.0c archives in the engine's own lookup
order, which `_game_root.classic_graphics()` stacks as a `LayeredDrs`. Measured from that game's `age2_x1.exe`: it opens `sounds_x1`,
`gamedata_x1`, `gamedata_x1_p1`, `sounds`, `graphics`, `terrain`, `interfac`
and `gamedata` in that order, appends each to the tail of one resource list,
and a lookup walks that list from the head and takes the first archive that
holds the id. `AOC_GRAPHICS_ORDER` keeps that relative order for the four archives
graphics come from (terrain.drs and interfac.drs are separate lookups here).
On the shipped files the only overlaps are `gamedata_x1_p1.drs` over six
`graphics.drs` SLPs and `gamedata_x1.drs` over 46 `gamedata.drs` bina members.

HD Edition keeps no archives: its SLPs are loose `<id>.slp` files in three
folders under `resources/_common/drs/`, which `LooseSlpStack` serves with
`LayeredDrs`'s interface. `HD_GRAPHICS_ORDER` is the expected lookup order
(the expansions over the base game), not yet measured from `AoK HD.exe`: of
the 16 ids `gamedata_x2` shares with `graphics`, all differ in bytes.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

DRS_VERSION = "1.00"

_HEADER = struct.Struct("<40s4s12sii")
_TABLE = struct.Struct("<4sii")
_ROW = struct.Struct("<iii")
_SLP_HEADER = struct.Struct("<4si24s")
_SLP_FRAME = struct.Struct("<IIIIiiii")

# The engine's lookup order for these four; see the module docstring.
AOC_GRAPHICS_ORDER = ("gamedata_x1.drs", "gamedata_x1_p1.drs", "graphics.drs", "gamedata.drs")
# HD Edition's loose SLP folders under resources/_common/drs/, first match winning; see the module docstring.
HD_GRAPHICS_ORDER = ("gamedata_x2", "gamedata_x1", "graphics")


def _kind(tag: bytes) -> str:
    return tag[::-1].decode("latin-1").strip()


class DrsArchive:
    """One DRS file. `members[kind][id]` is `(offset, size)` in the file."""

    def __init__(self, path: str | PathLike[str]) -> None:
        self.path = Path(path)
        with self.path.open("rb") as fh:
            head = fh.read(_HEADER.size)
            if len(head) < _HEADER.size:
                raise ValueError(f"{self.path}: too short for a DRS header")
            copyright_, version, file_type, table_count, _first = _HEADER.unpack(head)
            self.version = version.decode("latin-1")
            if self.version != DRS_VERSION:
                raise ValueError(f"{self.path}: DRS version {self.version!r}, expected {DRS_VERSION!r}")
            if not 0 <= table_count < 256:
                raise ValueError(f"{self.path}: implausible DRS table count {table_count}")
            self.copyright = copyright_.split(b"\0", 1)[0].decode("latin-1").rstrip("\x1a")
            self.file_type = file_type.split(b"\0", 1)[0].decode("latin-1")
            tables = [_TABLE.unpack(fh.read(_TABLE.size)) for _ in range(table_count)]
            self.size = self.path.stat().st_size
            self.members: dict[str, dict[int, tuple[int, int]]] = {}
            for tag, offset, count in tables:
                fh.seek(offset)
                raw = fh.read(_ROW.size * count)
                if len(raw) != _ROW.size * count:
                    raise ValueError(f"{self.path}: the {_kind(tag)!r} table runs past the end of the file")
                rows = {}
                for member_id, member_offset, size in _ROW.iter_unpack(raw):
                    if member_offset < 0 or size < 0 or member_offset + size > self.size:
                        raise ValueError(f"{self.path}: {_kind(tag)} {member_id} lies outside the file")
                    rows[member_id] = (member_offset, size)
                self.members[_kind(tag)] = rows

    def ids(self, kind: str) -> list[int]:
        return sorted(self.members.get(kind, ()))

    def has(self, kind: str, member_id: int) -> bool:
        return member_id in self.members.get(kind, ())

    def read(self, kind: str, member_id: int) -> bytes:
        try:
            offset, size = self.members[kind][member_id]
        except KeyError:
            raise KeyError(f"{self.path.name} holds no {kind} {member_id}") from None
        with self.path.open("rb") as fh:
            fh.seek(offset)
            return fh.read(size)

    def __repr__(self) -> str:
        counts = ", ".join(f"{k}={len(v)}" for k, v in self.members.items())
        return f"DrsArchive({self.path.name!r}, {counts})"


class LayeredDrs:
    """Several archives searched in order: the FIRST one holding an id wins."""

    def __init__(self, *archives: DrsArchive) -> None:
        self.archives = list(archives)

    def source(self, kind: str, member_id: int) -> DrsArchive | None:
        return next((a for a in self.archives if a.has(kind, member_id)), None)

    def has(self, kind: str, member_id: int) -> bool:
        return self.source(kind, member_id) is not None

    def ids(self, kind: str) -> list[int]:
        return sorted({i for a in self.archives for i in a.members.get(kind, ())})

    def read(self, kind: str, member_id: int) -> bytes:
        archive = self.source(kind, member_id)
        if archive is None:
            names = ", ".join(a.path.name for a in self.archives)
            raise KeyError(f"no archive holds {kind} {member_id} (searched {names})")
        return archive.read(kind, member_id)


class LooseSlpStack:
    """Folders of loose `<id>.slp` files searched in order, the FIRST holding an id winning,
    with LayeredDrs's has/read/ids for kind "slp" (every other kind is empty)."""

    def __init__(self, *folders: str | PathLike[str]) -> None:
        self.folders = [Path(folder) for folder in folders]
        self.members: list[dict[int, Path]] = []
        for folder in self.folders:
            found: dict[int, Path] = {}
            if folder.is_dir():
                for path in folder.iterdir():
                    if path.suffix.lower() == ".slp" and path.stem.isdigit():
                        found.setdefault(int(path.stem), path)
            self.members.append(found)

    def source(self, kind: str, member_id: int) -> Path | None:
        if kind != "slp":
            return None
        return next((members[member_id] for members in self.members if member_id in members), None)

    def has(self, kind: str, member_id: int) -> bool:
        return self.source(kind, member_id) is not None

    def ids(self, kind: str) -> list[int]:
        return sorted({i for members in self.members for i in members}) if kind == "slp" else []

    def read(self, kind: str, member_id: int) -> bytes:
        path = self.source(kind, member_id)
        if path is None:
            names = ", ".join(folder.name for folder in self.folders)
            raise KeyError(f"no folder holds {kind} {member_id} (searched {names})")
        return path.read_bytes()


@dataclass(slots=True)
class SlpFrame:
    cmd_table_offset: int
    outline_table_offset: int
    palette_offset: int
    properties: int
    width: int
    height: int
    hotspot_x: int
    hotspot_y: int


@dataclass(slots=True)
class SlpHeader:
    version: str
    comment: str
    frames: list[SlpFrame]


def slp_header(data: bytes) -> SlpHeader:
    """The 32-byte SLP header and its per-frame info rows (no pixel decoding)."""
    if len(data) < _SLP_HEADER.size:
        raise ValueError("too short for an SLP header")
    version, frame_count, comment = _SLP_HEADER.unpack_from(data)
    if not 0 <= frame_count <= (len(data) - _SLP_HEADER.size) // _SLP_FRAME.size:
        raise ValueError(f"SLP frame count {frame_count} does not fit in {len(data)} bytes")
    frames = [SlpFrame(*_SLP_FRAME.unpack_from(data, _SLP_HEADER.size + i * _SLP_FRAME.size)) for i in range(frame_count)]
    return SlpHeader(version.decode("latin-1"), comment.split(b"\0", 1)[0].decode("latin-1"), frames)


def read_jasc_palette(data: bytes) -> list[tuple[int, int, int]]:
    """A JASC-PAL text palette (e.g. interfac.drs bina 50500) as (r, g, b) tuples."""
    lines = data.decode("latin-1").split()
    if len(lines) < 3 or lines[0] != "JASC-PAL":
        raise ValueError("not a JASC-PAL palette")
    count = int(lines[2])
    values = lines[3:3 + 3 * count]
    if len(values) != 3 * count:
        raise ValueError(f"JASC-PAL says {count} colours but holds {len(values) // 3}")
    nums = [int(v) for v in values]
    if any(not 0 <= n <= 255 for n in nums):
        raise ValueError("JASC-PAL colour component outside 0..255")
    return [(nums[i], nums[i + 1], nums[i + 2]) for i in range(0, len(nums), 3)]
