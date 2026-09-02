"""Shared game-style serialization helper for tools/gen_trigger_fixture.py and
tools/gen_units_fixture.py.

Kept in tools/, not descape/: this is generator machinery, and importing it
from the code under test would make the one test that regenerates a section
circular (see either generator's own module docstring). Both fixtures are
built by adding real objects through AoE2ScenarioParser's own manager API,
committing, then re-serializing through this helper rather than through
descape.trigger_model / descape.unit_model.

The one deliberate departure from the library's own serializer, without which
both fixtures would be useless: an empty str field is written the way the
game writes it (length 0) rather than the way the library writes it (length 1
holding a NUL). See _game_style_bytes() for the measurement.
"""

from __future__ import annotations


class GenerationVerificationError(Exception):
    """Raised instead of leaving a fixture file on disk that failed its
    reload-through-the-real-loader check."""


def _game_style_bytes(section) -> bytes:
    """AoE2ScenarioParser's own serialization of a section (or of one struct
    entry from a struct-typed field -- anything exposing `.retriever_map`),
    with one deliberate difference: an empty str field is written as length 0
    rather than as length 1 holding a lone NUL.

    This is the single point where a fixture stops being a library export and
    starts being a game-shaped one, and it is the reason either fixture is
    worth having at all.

    Measured on the corpus: AoE2:DE writes an empty trigger/effect/condition/
    unit-caption string as a bare length-0 field, while the library
    unconditionally appends a NUL trail (helper/bytes_conversions.py's
    add_str_trail, from which only a fixed list of scenario-level message
    fields is exempt). That one-byte difference per empty string is the bulk
    of the Triggers-section drift measured for the trigger fixture, and the
    exact mechanism behind the Units-section caption_string drift measured for
    the units fixture (both write paths' plans record the corpus figures).

    Without this, a fixture built through the library round-trips through the
    library perfectly -- it came out of the library, so it is already in the
    library's normal form -- and the default tier cannot tell a byte-blob
    write path apart from whole-section re-serialization. Verified by
    mutation on the trigger fixture: with this applied, disabling the blob
    splice fails tests/test_trigger_write_path.py; without it, that mutation
    passes every test.
    """
    parts: list[bytes] = []
    for retriever in section.retriever_map.values():
        if retriever.datatype.type == "struct":
            parts.append(b"".join(_game_style_bytes(entry) for entry in (retriever.data or [])))
            continue
        data = retriever.get_data_as_bytes()
        var_type, var_len = retriever.datatype.type_and_length
        if var_type == "str" and data == _library_empty_string(var_len):
            data = _game_empty_string(var_len)
        parts.append(data)
    return b"".join(parts)


def _library_empty_string(prefix_size: int) -> bytes:
    """A length-1 string holding one NUL: what add_str_trail() produces."""
    return (1).to_bytes(prefix_size, "little", signed=True) + b"\x00"


def _game_empty_string(prefix_size: int) -> bytes:
    return (0).to_bytes(prefix_size, "little", signed=True)
