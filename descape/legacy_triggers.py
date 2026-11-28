"""Pre-DE (trigger version 1.6) Triggers sections, rewritten into the DE-shaped
layout the legacy structures declare so the library can parse them (TASK-522).

A 1.6 effect or condition stores its own field count, and the fields present
are the leading prefix of DE's: the rest read as -1, the unset sentinel.
`docs/SCENARIO_IO.md` "Older formats" has the layout.
"""

from __future__ import annotations

import struct

TRIGGER_VERSION = 1.6
EFFECT_FIELDS = 46  # DE's EffectStruct s32 fields, static_value_46
CONDITION_FIELDS = 21  # DE's ConditionStruct s32 fields, static_value_21
ES_ONLY_BYTES = 396


class LegacyTriggerFormatError(ValueError):
    """A 1.6 Triggers section that does not walk to the exact end of its bytes."""


class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def take(self, fmt: str) -> int | float:
        try:
            (value,) = struct.unpack_from("<" + fmt, self.data, self.pos)
        except struct.error as e:
            raise LegacyTriggerFormatError(f"trigger section ends early at byte {self.pos}") from e
        self.pos += struct.calcsize("<" + fmt)
        return value

    def ints(self, count: int) -> list[int]:
        if count < 0:
            raise LegacyTriggerFormatError(f"negative count {count} at byte {self.pos}")
        return [self.take("i") for _ in range(count)]

    def str32(self) -> bytes:
        size = self.take("i")
        if size < 0 or self.pos + size > len(self.data):
            raise LegacyTriggerFormatError(f"string of {size} bytes overruns at byte {self.pos}")
        value = self.data[self.pos : self.pos + size]
        self.pos += size
        return value


def _p(fmt: str, *values) -> bytes:
    return struct.pack("<" + fmt, *values)


def _str32(value: bytes) -> bytes:
    return _p("i", len(value)) + value


def _padded(values: list[int], width: int) -> bytes:
    if len(values) > width:
        raise LegacyTriggerFormatError(f"{len(values)} fields, more than DE's {width}")
    return b"".join(_p("i", v) for v in (values + [-1] * width)[:width])


def _effect(r: _Reader) -> bytes:
    effect_type = r.take("i")
    values = r.ints(r.take("i"))
    message = r.str32()
    sound = r.str32()
    selected = r.ints(max(values[4], 0)) if len(values) > 4 else []
    out = _p("i", effect_type) + _p("i", EFFECT_FIELDS) + _padded(values, EFFECT_FIELDS)
    return out + _str32(message) + _str32(sound) + b"".join(_p("i", v) for v in selected)


def _condition(r: _Reader) -> bytes:
    condition_type = r.take("i")
    values = r.ints(r.take("i"))
    return _p("i", condition_type) + _p("i", CONDITION_FIELDS) + _padded(values, CONDITION_FIELDS)


def _trigger(r: _Reader) -> bytes:
    out = bytearray()
    out += _p("I", r.take("I"))  # enabled
    out += _p("b", r.take("b"))  # looping
    out += _p("i", r.take("i"))  # description_string_table_id
    out += _p("B", r.take("B"))  # display_as_objective
    out += _p("I", r.take("I"))  # objective_description_order
    r.take("i")  # 4 unknown bytes, DE has no field for them
    # make_header, short_description_string_table_id, display_on_screen, unknown (5), mute_objectives
    out += _p("B", 0) + _p("i", -1) + _p("B", 0) + bytes(5) + _p("B", 0)
    description = r.str32()
    name = r.str32()
    out += _str32(description) + _str32(name) + _str32(b"")
    for entry in (_effect, _condition):
        count = r.take("i")
        if count < 0:
            raise LegacyTriggerFormatError(f"negative entry count {count} at byte {r.pos}")
        out += _p("i", count)
        for _ in range(count):
            out += entry(r)
        out += b"".join(_p("i", v) for v in r.ints(count))  # display order
    return bytes(out)


def _skip_ai_files(r: _Reader) -> None:
    include_ai_files = r.take("I")
    if r.take("I"):  # es_only_data
        if r.pos + ES_ONLY_BYTES > len(r.data):
            raise LegacyTriggerFormatError("es_only_data overruns the section")
        r.pos += ES_ONLY_BYTES
    if include_ai_files:
        for _ in range(r.take("I")):
            r.str32()
            r.str32()


def normalize(tail: bytes) -> bytes:
    """`tail` (a 1.6 Triggers section plus its AI-files trailer, to the end of
    the body) as the DE-shaped Triggers bytes the legacy structures declare.
    Raises LegacyTriggerFormatError unless the walk ends exactly at len(tail)."""
    r = _Reader(tail)
    out = bytearray()
    version = r.take("d")
    if version != TRIGGER_VERSION:
        raise LegacyTriggerFormatError(f"trigger version {version} is not {TRIGGER_VERSION}")
    out += _p("d", version)
    out += _p("b", r.take("b"))  # instruction_start
    count = r.take("i")
    if count < 0:
        raise LegacyTriggerFormatError(f"negative trigger count {count}")
    out += _p("i", count)
    for _ in range(count):
        out += _trigger(r)
    out += b"".join(_p("I", v & 0xFFFFFFFF) for v in r.ints(count))  # trigger display order
    _skip_ai_files(r)
    if r.pos != len(tail):
        raise LegacyTriggerFormatError(f"trigger section ends at byte {r.pos} of {len(tail)}")
    out += bytes(1028) + _p("I", 0)  # unknown_bytes, number_of_variables
    return bytes(out)
