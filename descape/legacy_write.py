"""Which edits a pre-DE (header-2) scenario may save back to its own format, Qt-free.

A legacy `.scn`/`.scx` saves through the same byte splice as a DE file
(scenario_write.build_patched()), but only the edit families its band's own
game has been shown to open: ALLOWED, keyed by data version. A band gains a
family only once that family's oracle pair (tools/legacy_oracle_pairs.py)
opened in the band's game. build_patched() asks pending_kinds() what a save
carries and refusal() whether the band allows all of it, so every kind of
edit outside the four families below (units, other options, string-table
ids, triggers) is refused by default.

Text in a legacy file is one byte per character in the game's code page,
never UTF-8: decode()/encode() are cp1252 with its five undefined bytes
mapped to their latin-1 code points, a bijection over all 256 byte values,
so every stored string (a DBCS one included) round-trips byte for byte.
"""

from __future__ import annotations

import contextlib
import struct
from collections.abc import Iterator

from descape import scenario_format

TERRAIN = "terrain"
ELEVATION = "elevation"
MESSAGES = "messages"
DIPLOMACY = "diplomacy"
FAMILIES = (TERRAIN, ELEVATION, MESSAGES, DIPLOMACY)
ANY = "any"  # a lock_reason() family: whether any family is allowed (Save, autosave)

# The families each band's game has opened. Empty until a saved pair opens in that game itself.
ALLOWED: dict[str, frozenset[str]] = {
    "1.20": frozenset(),
    "1.22": frozenset(),
    "1.26": frozenset(),
}

# Every terrain id and the highest elevation the band's corpus holds
# (measured 2026-11-26 over 156 AoK, 382 AoC and 9 HD 1.26 files).
LEGAL_TERRAIN: dict[str, frozenset[int]] = {
    "1.20": frozenset({0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28}),
    "1.22": frozenset(
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28,
         29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40}
    ),
    "1.26": frozenset(
        {0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22, 23, 25, 26, 32, 34, 37, 39, 41,
         42, 44, 47}
    ),
}
BAND_MAX_ELEVATION: dict[str, int] = {"1.20": 7, "1.22": 8, "1.26": 6}

_SUFFIX = {scenario_format.AOK: ".scn"}  # every other band saves .scx; a file's own suffix wins (save_suffix())
_HEADER_LENGTH = struct.Struct("<I")


def is_legacy(scenario) -> bool:
    fmt = getattr(scenario, "source_format", None)
    return bool(fmt is not None and fmt.legacy)


def allowed(scenario) -> frozenset[str]:
    """The families `scenario`'s band may save. Raises ValueError for a DE document."""
    if not is_legacy(scenario):
        raise ValueError("not a pre-DE document")
    return ALLOWED.get(scenario.source_format.data_version or "", frozenset())


def lock_reason(scenario, family: str | None = None) -> str | None:
    """Why `family` may not be edited on a legacy document, None when it may.
    None (every unannotated caller) is always refused; ANY asks for any family."""
    families = allowed(scenario)
    if family is not None and (family in families if family != ANY else bool(families)):
        return None
    fmt = scenario.source_format
    if not families:
        return scenario_format.read_only_reason(fmt)
    names = " and ".join(f for f in FAMILIES if f in families)
    return (
        f"Older-format file ({fmt.display_name}): only {names} edits can be saved back to it so far. "
        "To edit the rest, use File > Convert to .aoe2scenario, which makes an editable AoE2:DE copy."
    )


def legal_terrain(scenario) -> frozenset[int]:
    return LEGAL_TERRAIN.get(scenario.source_format.data_version or "", frozenset())


def max_elevation(scenario, default: int) -> int:
    """The highest elevation a tool may set: the band's own on a legacy document, else `default`."""
    if not is_legacy(scenario):
        return default
    return BAND_MAX_ELEVATION.get(scenario.source_format.data_version or "", 0)


def save_suffix(scenario) -> str:
    """The suffix Save As offers: the file's own, else its band's."""
    suffix = scenario.path.suffix.lower()
    if suffix in (".scn", ".scx", ".scx2"):
        return suffix
    return _SUFFIX.get(scenario.source_format.game or "", ".scx")


def terrain_kinds(scenario, body: bytes, terrain_offset: int) -> set[str]:
    """TERRAIN and/or ELEVATION, whichever tile byte differs between the
    original terrain block and `body`'s (at `terrain_offset`)."""
    stride = scenario.terrain_struct_size
    n = len(scenario.map_manager.terrain)
    old = scenario.decompressed_body[scenario.terrain_block_offset : scenario.terrain_block_offset + stride * n]
    new = body[terrain_offset : terrain_offset + stride * n]
    kinds = set()
    if old[0::stride] != new[0::stride]:
        kinds.add(TERRAIN)
    if old[1::stride] != new[1::stride]:
        kinds.add(ELEVATION)
    return kinds


def model_kinds(options=None, units=None, messages=None, triggers=None) -> set[str]:
    """Every kind of edit the edit models hold: DIPLOMACY, MESSAGES, or a named
    kind ("units", "map options", "string ids", ...) for anything else."""
    kinds: set[str] = set()
    if options is not None and options.has_edits:
        kinds |= options.pending_kinds()
    if messages is not None and messages.has_edits:
        kinds |= messages.pending_kinds()
    if units is not None and units.has_edits:
        kinds.add("units")
    if triggers is not None and triggers.has_edits:
        kinds.add("triggers")
    return kinds


def pending_kinds(scenario, body: bytes, terrain_offset: int, options=None, units=None, messages=None,
                  triggers=None) -> set[str]:
    """Every kind of edit a save of `body` carries: model_kinds() plus TERRAIN/ELEVATION."""
    return terrain_kinds(scenario, body, terrain_offset) | model_kinds(options, units, messages, triggers)


def illegal_tile(scenario, body: bytes, terrain_offset: int) -> str | None:
    """The first tile whose changed terrain id or elevation the band's game
    does not hold, described for a refusal, else None."""
    band = scenario.source_format.data_version or ""
    legal, top = LEGAL_TERRAIN.get(band, frozenset()), BAND_MAX_ELEVATION.get(band, -1)
    stride, start = scenario.terrain_struct_size, scenario.terrain_block_offset
    width = scenario.map_manager.map_width
    old = scenario.decompressed_body
    for i in range(len(scenario.map_manager.terrain)):
        o, p = start + i * stride, terrain_offset + i * stride
        tid, elev = body[p], body[p + 1]
        if tid != old[o] and tid not in legal:
            return f"tile ({i % width}, {i // width}) has terrain id {tid}, which {scenario.source_format.display_name} does not have"
        if elev != old[o + 1] and elev > top:
            return f"tile ({i % width}, {i // width}) has elevation {elev}, above this format's {top}"
    return None


def tile_findings(scenario, changes, width: int) -> list[str]:
    """A script run's tile changes (TileDiffRecord.changes) a pre-DE save
    would refuse: a family the band does not allow, or a value it lacks."""
    families = allowed(scenario)
    band = scenario.source_format.data_version or ""
    legal, top = LEGAL_TERRAIN.get(band, frozenset()), BAND_MAX_ELEVATION.get(band, -1)
    name = scenario.source_format.display_name
    refused: set[str] = set()
    findings: list[str] = []
    for index, (old_terrain, old_elevation, _l), (terrain, elevation, _layer) in changes:
        y, x = divmod(index, width)
        if terrain != old_terrain:
            if TERRAIN not in families:
                refused.add(TERRAIN)
            elif terrain not in legal:
                findings.append(f"tile ({x}, {y}) terrain_id {terrain!r} is not a {name} terrain")
        if elevation != old_elevation:
            if ELEVATION not in families:
                refused.add(ELEVATION)
            elif not isinstance(elevation, int) or elevation > top:
                findings.append(f"tile ({x}, {y}) elevation {elevation!r} is above this format's {top}")
    return [f"{name} files can't save {k} edits yet" for k in sorted(refused)] + findings


def refusal(scenario, kinds: set[str], body: bytes | None = None, terrain_offset: int = -1) -> str | None:
    """Why a legacy save carrying `kinds` may not be written, None when it may.
    `body` is needed only when `kinds` holds TERRAIN or ELEVATION."""
    families = allowed(scenario)
    refused = sorted(kinds - families)
    if refused:
        names = " and ".join(f for f in FAMILIES if f in families) or "no"
        return (
            f"{scenario.source_format.display_name} files can't save {', '.join(refused)} edits yet "
            f"({names} edits can be saved back to this format so far). "
            + ("Undo them, or use File > Convert to .aoe2scenario." if scenario_format.converts(scenario.source_format) else "Undo them.")
        )
    if kinds & {TERRAIN, ELEVATION} and body is not None:
        return illegal_tile(scenario, body, terrain_offset)
    return None


def fix_header_length(header: bytes) -> bytes:
    """`header` with its FileHeader.header_length (u32 at byte 4) set to what
    follows it, which a header-2 reader skips by. DE stores 0 there instead."""
    return header[:4] + _HEADER_LENGTH.pack(len(header) - 8) + header[8:]


@contextlib.contextmanager
def _allow_for_oracle(data_version: str, families=FAMILIES) -> Iterator[None]:
    """Widens one band's ALLOWED entry for the oracle pair tool and tests only."""
    before = ALLOWED.get(data_version)
    ALLOWED[data_version] = frozenset(families)
    try:
        yield
    finally:
        if before is None:
            ALLOWED.pop(data_version, None)
        else:
            ALLOWED[data_version] = before


# -- text --------------------------------------------------------------------

_UNDEFINED = frozenset({0x81, 0x8D, 0x8F, 0x90, 0x9D})  # cp1252 leaves these unassigned
_DECODE = {b: bytes([b]).decode("cp1252") for b in range(0x80, 0xA0) if b not in _UNDEFINED}
_TO_CP1252 = str.maketrans(dict(_DECODE))
_FROM_CP1252 = str.maketrans({ch: b for b, ch in _DECODE.items()})


def decode(raw: bytes) -> str:
    """A stored legacy string's text: cp1252, its undefined bytes as latin-1."""
    return raw.decode("latin-1").translate(_TO_CP1252)


def encode(text: str) -> bytes:
    """decode()'s inverse. Raises ValueError naming the first character the
    format cannot store (or a NUL, which ends a string in the old games)."""
    for ch in text:
        if ch == "\x00" or (0x80 <= ord(ch) < 0xA0 and ord(ch) not in _UNDEFINED):
            raise ValueError(f"{_name(ch)} can't be stored in an older-format scenario")
    mapped = text.translate(_FROM_CP1252)
    try:
        return mapped.encode("latin-1")
    except UnicodeEncodeError as e:
        raise ValueError(f"{_name(text[e.start])} can't be stored in an older-format scenario") from None


def _name(ch: str) -> str:
    return f"The character {ch!r} (U+{ord(ch):04X})"
