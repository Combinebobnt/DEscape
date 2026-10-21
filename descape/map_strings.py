"""The Map section's leading strings, read and spliced from raw bytes
(TASK-333). Qt-free.

The section starts at options_section_end with a `60 0a` marker and a str16
for water_definition, the same for map_color_mood and, from scenario 1.40,
for script_name. Every Map retriever after them, up to terrain_block_offset,
is fixed-width.

Each str16 is measured from its raw signed int16 prefix, never from
retriever_length(): the library strips one trailing NUL from
water_definition on parse and does not add it back (the field is in its
no-trailing-NUL list), so a file storing `Preset_Main\\x00` reports 13 bytes
for 14. A backward walk across this field lands one byte off, which is why
this forward walk is the only reader of these spans.

Only water_definition is editable. The splice region runs from the first
marker through the end of map_color_mood, the strings this module owns.
script_name is walked for the gate but never rebuilt here: GH #216's own
splice (options_model.serialize_script_name_resize()) owns it, and runs
before this one because it sits at the higher offset.
"""

from __future__ import annotations

import struct
from collections.abc import Mapping
from dataclasses import dataclass

from descape.scenario_io import LoadedScenario, retriever_length

WATER_DEFINITION = "water_definition"
# The strings this module may rebuild; map_color_mood is copied verbatim today.
OWNED = ("water_definition", "map_color_mood")
EDITABLE = frozenset({WATER_DEFINITION})

_MARKER = b"\x60\x0a"
_STR16_PREFIX = struct.Struct("<h")
STR16_MAX_PAYLOAD = 0x7FFF
# Map's tail retriever, verified on its own path (scenario_io._verify_terrain_block()).
_TAIL = ("terrain_data",)


@dataclass(frozen=True)
class FrontItem:
    name: str
    offset: int  # within decompressed_body, load time
    length: int  # bytes on disk, a string's 2-byte prefix included
    is_string: bool


@dataclass(frozen=True)
class MapStringsLayout:
    start: int  # the splice region: the front's first marker ...
    end: int  # ... through the end of its last OWNED string
    items: tuple[FrontItem, ...]  # every walked item, script_name included
    front_end: int  # where the walk stopped: the first fixed-width retriever


def _is_marker(retriever) -> bool:
    return retriever.datatype.type == "data" and retriever.datatype.var == "2"


def _is_str16(retriever) -> bool:
    return retriever.datatype.type == "str" and retriever.datatype.var == "str16"


def _payload_matches(payload: bytes, value) -> bool:
    if not isinstance(value, str) or "\x00" in value:
        return False
    encoded = value.encode("utf-8")
    return payload in (encoded, encoded + b"\x00")


def layout(loaded: LoadedScenario) -> MapStringsLayout | None:
    """The verified front, or None (fail closed). See the module docstring
    for the walk; each check below is one way a misframed parse would show."""
    start, terrain = loaded.options_section_end, loaded.terrain_block_offset
    if start < 0 or terrain < 0:
        return None
    section = loaded._scenario.sections.get("Map")
    if section is None or WATER_DEFINITION not in section.retriever_map:
        return None
    retriever_map = section.retriever_map
    names = [name for name in retriever_map if name not in _TAIL]
    body = loaded.decompressed_body
    cursor = start
    items: list[FrontItem] = []
    for name in names:
        retriever = retriever_map[name]
        if _is_marker(retriever):
            if bytes(body[cursor : cursor + 2]) != _MARKER or cursor + 2 > terrain:
                return None
            items.append(FrontItem(name, cursor, 2, False))
            cursor += 2
        elif _is_str16(retriever):
            if cursor + 2 > terrain:
                return None
            (length,) = _STR16_PREFIX.unpack_from(body, cursor)
            end = cursor + 2 + length
            if length < 0 or end > terrain:
                return None
            items.append(FrontItem(name, cursor, 2 + length, True))
            cursor = end
        else:
            break
    rest = names[len(items) :]
    if any(retriever_map[name].datatype.type == "str" for name in rest):
        return None
    if cursor + sum(retriever_length(retriever_map[name]) for name in rest) != terrain:
        return None

    strings = [item for item in items if item.is_string]
    if WATER_DEFINITION not in {item.name for item in strings}:
        return None
    for item in strings:
        if item.name in EDITABLE:
            payload = bytes(body[item.offset + 2 : item.offset + item.length])
            if not _payload_matches(payload, retriever_map[item.name].data):
                return None
    owned = [i for i, item in enumerate(items) if item.is_string and item.name in OWNED]
    last = owned[-1]
    # A string another splice owns must not sit inside this region.
    if any(item.is_string and item.name not in OWNED for item in items[: last + 1]):
        return None
    region_end = items[last].offset + items[last].length
    return MapStringsLayout(start, region_end, tuple(items), cursor)


def write_supported(loaded: LoadedScenario) -> bool:
    """The water row's own gate, never folded into options_write_supported()."""
    return layout(loaded) is not None


def current_value(loaded: LoadedScenario, name: str) -> str:
    """The file's own value for `name`, as the library parsed it (one
    trailing NUL already stripped). The one reader: the panel's as-stored row
    and the model's original both come from here."""
    value = loaded._scenario.sections["Map"].retriever_map[name].data
    if not isinstance(value, str):
        raise TypeError(f"Map.{name} resolved to a {type(value).__name__}, not a str")
    return value


def encode_str16(value) -> bytes:
    """`value` as the game writes it: signed int16 length + UTF-8, no NUL."""
    if not isinstance(value, str):
        raise ValueError(f"{value!r} is not text")
    if "\x00" in value:
        raise ValueError(f"{value!r} contains a NUL character")
    payload = value.encode("utf-8")
    if len(payload) > STR16_MAX_PAYLOAD:
        raise ValueError(f"{len(payload)} bytes is over the str16 limit of {STR16_MAX_PAYLOAD}")
    return _STR16_PREFIX.pack(len(payload)) + payload


def splice(loaded: LoadedScenario, edits: Mapping[str, str]) -> tuple[int, int, bytes]:
    """(start, end, replacement) in load-time offsets for the region, every
    unedited item copied byte for byte from decompressed_body."""
    found = layout(loaded)
    if found is None:
        raise ValueError("this file's Map strings do not verify")
    unknown = set(edits) - EDITABLE
    if unknown:
        raise KeyError(f"not an editable Map string: {sorted(unknown)}")
    body = loaded.decompressed_body
    parts: list[bytes] = []
    for item in found.items:
        if item.offset >= found.end:
            break
        if item.is_string and item.name in edits:
            parts.append(encode_str16(edits[item.name]))
        else:
            parts.append(bytes(body[item.offset : item.offset + item.length]))
    return found.start, found.end, b"".join(parts)
