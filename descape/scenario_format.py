"""Which on-disk scenario format a file is in, read off its first bytes.

A scenario's file version string ("1.21") does not identify its layout: HD
Edition's `.aoe2scenario` and The Conquerors' `.scx` both say 1.21. What
does is the FileHeader version (the s32 at byte 8) plus, for a header-2
file, the DataHeader version (the f32 4 bytes into the inflated body).
Header 2 is every pre-DE format (Age of Kings, The Conquerors, HD Edition
`.scx`); `scenario_io` reads those through `library_compat.LEGACY_VERSIONS_DIR`,
keyed by data version, and they save back only the edit families
`legacy_write.ALLOWED` gives their band. Header 3 and later is HD's
`.aoe2scenario` and every DE file, read as before.

Qt-free and independent of `scenario_io`, which imports this.
"""

from __future__ import annotations

import struct
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

LEGACY_HEADER_VERSION = 2
_PREFIX = struct.Struct("<4sIi")  # file version, header_length, header version
_DATA_VERSION = struct.Struct("<f")
_DATA_VERSION_OFFSET = 4  # DataHeader: u32 next_unit_id, then f32 version

# ScenarioFormat.game's labels. Prefixed so none equals a games.py id: a game claims labels by Game.formats.
AOK = "scn-aok"
AOC = "scn-aoc"
HD = "scn-hd"  # header-2 bands only HD Edition writes
UP = "scn-up"  # band 1.23: UserPatch .scx2 mostly, HD Edition too; the bytes cannot tell which
HD_DE = "hd-aoe2scenario"  # HD Edition's header-3 1.21 .aoe2scenario: no game claims it, so it opens as AoE2:DE's
AOE1 = "aoe1-1.15"  # Age of Empires and Rise of Rome share this one layout: the format cannot tell them apart
# Header-2 bands by data version, as measured on the 2026-10-28 corpus (1.15: 2026-11-23, 64 AoE1 files).
_LEGACY_GAMES = {"1.15": AOE1, "1.20": AOK, "1.22": AOC, "1.21": HD, "1.23": UP, "1.24": HD, "1.26": HD}
_DISPLAY_NAMES = {
    AOE1: "Age of Empires",
    AOK: "Age of Kings",
    AOC: "The Conquerors",
    HD: "HD Edition",
    UP: "UserPatch / HD Edition .scx/.scx2",
    HD_DE: "HD Edition",
}
_BAND_NAMES = {
    AOE1: "Age of Empires / Rise of Rome .scn/.scx",
    AOK: "Age of Kings .scn",
    AOC: "The Conquerors .scn/.scx",
    HD: "HD Edition .scx",
    UP: "UserPatch / HD Edition .scx/.scx2",
}
# The UnitStruct fields a band does not store, by data version, and the value its units carry
# (TASK-515): the band's documents are read-only, so no write path encodes one.
_ABSENT_UNIT_FIELDS = {"1.15": MappingProxyType({"initial_animation_frame": 0, "garrisoned_in_id": -1})}
_NO_ABSENT_FIELDS: Mapping[str, int] = MappingProxyType({})
# Below this data version a header-2 file is Age of Empires (1997), not AoE2: DEscape does not convert it.
_AOE2_FIRST_DATA_VERSION = 1.18


@dataclass(frozen=True)
class ScenarioFormat:
    file_version: str
    header_version: int | None
    data_version: str | None  # "1.20" etc. for a header-2 file, else None
    game: str | None  # a label above (HD_DE for header 3); None for DE or an unknown band. Not a games.py id.
    legacy: bool  # header 2: saves per legacy_write.ALLOWED, trigger 1.6 read through legacy_triggers

    @property
    def display_name(self) -> str:
        """The game's name for messages, e.g. "The Conquerors"."""
        fallback = "older-format .scn/.scx/.scx2" if self.legacy else "Age of Empires II: Definitive Edition"
        return _DISPLAY_NAMES.get(self.game or "", fallback)


DE_FORMAT = ScenarioFormat(file_version="", header_version=None, data_version=None, game=None, legacy=False)


def read_format(raw: bytes) -> tuple[ScenarioFormat, bytes | None]:
    """The file's format, plus its inflated body for a header-2 file (None
    otherwise, or when the body does not inflate), so the loader inflates once.
    A file too short for the 12-byte prefix reads as DE: today's loader then
    fails on it its own way."""
    if len(raw) < _PREFIX.size:
        return DE_FORMAT, None
    version_bytes, header_length, header_version = _PREFIX.unpack_from(raw)
    file_version = version_bytes.decode("latin-1")
    if header_version != LEGACY_HEADER_VERSION:
        game = HD_DE if file_version == "1.21" and header_version == 3 else None
        return ScenarioFormat(file_version, header_version, None, game, False), None
    body = None
    data_version = None
    try:
        body = zlib.decompress(raw[8 + header_length :], -zlib.MAX_WBITS)
        if len(body) >= _DATA_VERSION_OFFSET + _DATA_VERSION.size:
            (value,) = _DATA_VERSION.unpack_from(body, _DATA_VERSION_OFFSET)
            data_version = f"{value:.2f}"
    except zlib.error:
        body = None
    game = _LEGACY_GAMES.get(data_version or "")
    return ScenarioFormat(file_version, header_version, data_version, game, True), body


def detect_format(raw: bytes) -> ScenarioFormat:
    return read_format(raw)[0]


def body_offset(raw: bytes) -> int:
    """Where a header-2 file's compressed body starts: past the 8-byte prefix
    and `header_length` more bytes. The FileHeader parse must end exactly here."""
    return 8 + _PREFIX.unpack_from(raw)[1]


def absent_unit_fields(fmt: ScenarioFormat) -> Mapping[str, int]:
    """Unit fields `fmt`'s band does not store, each with the value its units read as."""
    return _ABSENT_UNIT_FIELDS.get(fmt.data_version or "", _NO_ABSENT_FIELDS) if fmt.legacy else _NO_ABSENT_FIELDS


def is_aoe1_era(fmt: ScenarioFormat) -> bool:
    """A header-2 file of Age of Empires (1997) or Rise of Rome: its data
    version is below AoE2's first. DEscape offers no conversion for one."""
    if not fmt.legacy or fmt.data_version is None:
        return False
    try:
        return float(fmt.data_version) < _AOE2_FIRST_DATA_VERSION
    except ValueError:
        return False


def converts(fmt: ScenarioFormat) -> bool:
    """Whether File > Convert to .aoe2scenario takes a document of this format."""
    return fmt.legacy and not is_aoe1_era(fmt)


def legacy_unsupported_message(fmt: ScenarioFormat) -> str:
    """The "Failed to load" modal's body for a header-2 band with no structure."""
    band = _BAND_NAMES.get(fmt.game or "", "older .scn/.scx/.scx2")
    what = f"data version {fmt.data_version}" if fmt.data_version else "data that does not decompress"
    if is_aoe1_era(fmt):
        return f"DEscape can't open this file.\n\nAge of Empires scenario {what} ({band}) isn't supported yet."
    return (
        "DEscape can't open this file.\n\n"
        f"Age of Empires II scenario {what} ({band}) isn't supported yet. "
        "To work on it, open it in AoE2:DE's editor and save it as .aoe2scenario, "
        "then open that copy here."
    )


def read_only_reason(fmt: ScenarioFormat) -> str:
    """The status-bar and Save-refusal text for a legacy document."""
    if not converts(fmt):
        return f"Read-only: {fmt.display_name} files open for viewing only."
    return (
        f"Read-only: {fmt.display_name} files open for viewing only. To edit one, "
        "use File > Convert to .aoe2scenario, which makes an editable AoE2:DE copy."
    )
