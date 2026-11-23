"""Which on-disk scenario format a file is in, read off its first bytes.

A scenario's file version string ("1.21") does not identify its layout: HD
Edition's `.aoe2scenario` and The Conquerors' `.scx` both say 1.21. What
does is the FileHeader version (the s32 at byte 8) plus, for a header-2
file, the DataHeader version (the f32 4 bytes into the inflated body).
Header 2 is every pre-DE format (Age of Kings, The Conquerors, HD Edition
`.scx`); `scenario_io` reads those through `library_compat.LEGACY_VERSIONS_DIR`,
keyed by data version, and they open read-only. Header 3 and later is HD's
`.aoe2scenario` and every DE file, read as before.

Qt-free and independent of `scenario_io`, which imports this.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

LEGACY_HEADER_VERSION = 2
_PREFIX = struct.Struct("<4sIi")  # file version, header_length, header version
_DATA_VERSION = struct.Struct("<f")
_DATA_VERSION_OFFSET = 4  # DataHeader: u32 next_unit_id, then f32 version

# Header-2 bands by data version, as measured on the 2026-10-28 corpus.
_LEGACY_GAMES = {"1.20": "aok", "1.22": "aoc", "1.21": "hd", "1.23": "hd", "1.24": "hd", "1.26": "hd"}
_DISPLAY_NAMES = {"aok": "Age of Kings", "aoc": "The Conquerors", "hd": "HD Edition .scx"}
_BAND_NAMES = {"aok": "Age of Kings .scn", "aoc": "The Conquerors .scn/.scx", "hd": "UserPatch / HD Edition .scx"}


@dataclass(frozen=True)
class ScenarioFormat:
    file_version: str
    header_version: int | None
    data_version: str | None  # "1.20" etc. for a header-2 file, else None
    game: str | None  # "aok", "aoc" or "hd"; None for DE or an unknown band. A label, not a games.py id.
    legacy: bool  # header 2: read-only, triggers unread

    @property
    def display_name(self) -> str:
        """The game's name for messages, e.g. "The Conquerors"."""
        fallback = "older-format .scn/.scx" if self.legacy else "Age of Empires II: Definitive Edition"
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
        game = "hd" if file_version == "1.21" and header_version == 3 else None
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


def legacy_unsupported_message(fmt: ScenarioFormat) -> str:
    """The "Failed to load" modal's body for a header-2 band with no structure."""
    band = _BAND_NAMES.get(fmt.game or "", "older .scn/.scx")
    what = f"data version {fmt.data_version}" if fmt.data_version else "data that does not decompress"
    return (
        "DEscape can't open this file.\n\n"
        f"Age of Empires II scenario {what} ({band}) isn't supported yet. "
        "To work on it, open it in AoE2:DE's editor and save it as .aoe2scenario, "
        "then open that copy here."
    )


def read_only_reason(fmt: ScenarioFormat) -> str:
    """The status-bar and Save-refusal text for a legacy document."""
    return (
        f"Read-only: {fmt.display_name} files open for viewing only. To edit one, "
        "open it in AoE2:DE's editor and save it as .aoe2scenario."
    )
