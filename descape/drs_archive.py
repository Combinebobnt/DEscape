"""Read-only access to Ensemble's DRS archives (version `1.00`), for building the 1999 game's derived root.

The runtime twin of tools/_drs.py, which the regeneration tools read at regen
time (tools/ is not importable from the app): the same archive layout and the
same The Conquerors 1.0c lookup order, tested against each other on one packed
archive. A DRS file is a 64-byte header (copyright text, version, type tag,
table count, first-member offset), one 12-byte entry per member type (a
reversed, space-padded extension such as `" pls"` for "slp", its table's
offset and member count), then each table's 12-byte (id, offset, size) rows.
Qt-free and stdlib only.
"""

from __future__ import annotations

import struct
from os import PathLike
from pathlib import Path

DRS_VERSION = "1.00"

_HEADER = struct.Struct("<40s4s12sii")
_TABLE = struct.Struct("<4sii")
_ROW = struct.Struct("<iii")

# First match wins, as in age2_x1.exe (tools/_drs.py's docstring has the measurement).
AOC_GRAPHICS_ORDER = ("gamedata_x1.drs", "gamedata_x1_p1.drs", "graphics.drs", "gamedata.drs")


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
            _copyright, version, _file_type, table_count, _first = _HEADER.unpack(head)
            if version.decode("latin-1") != DRS_VERSION:
                raise ValueError(f"{self.path}: DRS version {version!r}, expected {DRS_VERSION!r}")
            if not 0 <= table_count < 256:
                raise ValueError(f"{self.path}: implausible DRS table count {table_count}")
            tables = [_TABLE.unpack(fh.read(_TABLE.size)) for _ in range(table_count)]
            size = self.path.stat().st_size
            self.members: dict[str, dict[int, tuple[int, int]]] = {}
            for tag, offset, count in tables:
                if count < 0 or offset < 0 or offset + _ROW.size * count > size:
                    raise ValueError(f"{self.path}: the {_kind(tag)!r} table runs past the end of the file")
                fh.seek(offset)
                raw = fh.read(_ROW.size * count)
                if len(raw) != _ROW.size * count:
                    raise ValueError(f"{self.path}: the {_kind(tag)!r} table runs past the end of the file")
                rows = {}
                for member_id, member_offset, member_size in _ROW.iter_unpack(raw):
                    if member_offset < 0 or member_size < 0 or member_offset + member_size > size:
                        raise ValueError(f"{self.path}: {_kind(tag)} {member_id} lies outside the file")
                    rows[member_id] = (member_offset, member_size)
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


class LayeredDrs:
    """Several archives searched in order: the FIRST one holding an id wins."""

    def __init__(self, *archives: DrsArchive) -> None:
        self.archives = list(archives)

    def source(self, kind: str, member_id: int) -> DrsArchive | None:
        return next((a for a in self.archives if a.has(kind, member_id)), None)

    def ids(self, kind: str) -> list[int]:
        return sorted({i for a in self.archives for i in a.members.get(kind, ())})

    def read(self, kind: str, member_id: int) -> bytes:
        archive = self.source(kind, member_id)
        if archive is None:
            raise KeyError(f"no archive holds {kind} {member_id}")
        return archive.read(kind, member_id)


def aoc_graphics(data_dir: str | PathLike[str]) -> LayeredDrs:
    """The Conquerors 1.0c graphic lookup from its `Data/` folder; absent
    archives are skipped, except `graphics.drs`, which must be there. Names
    match case-insensitively, as on Windows (a copied install keeps its own case)."""
    root = Path(data_dir)
    files = {p.name.casefold(): p for p in root.iterdir() if p.is_file()} if root.is_dir() else {}
    if "graphics.drs" not in files:
        raise FileNotFoundError(f"{root}: no graphics.drs")
    return LayeredDrs(*(DrsArchive(files[name]) for name in AOC_GRAPHICS_ORDER if name in files))
