"""TASK-333: descape/map_strings.py, the Map string front's gate and splice.

Every shipped fixture stores an empty water_definition, so the other shapes
are built by splicing raw bytes into a fixture's body and reloading it
through the real loader (_variant()). A length running past the terrain
block makes the loader raise, so that one case swaps a loaded fixture's
body instead."""

from __future__ import annotations

import dataclasses
import struct
from pathlib import Path

import pytest

from descape import map_strings, option_fields
from descape.options_model import field_offsets, options_write_supported
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units, load_map_and_units_from_bytes
from descape.scenario_write import _compress_bytes

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TRIGGER_FIXTURE = FIXTURES / "triggers_120x120.aoe2scenario"
V159_FIXTURE = FIXTURES / "real_blank_120x120_v159.aoe2scenario"


def _item(loaded, name: str) -> map_strings.FrontItem:
    found = map_strings.layout(loaded)
    assert found is not None
    return next(item for item in found.items if item.name == name)


def _variant_body(path: Path, prefix: int, payload: bytes):
    donor = load_map_and_units(path)
    water = _item(donor, map_strings.WATER_DEFINITION)
    body = donor.decompressed_body
    new = body[: water.offset] + struct.pack("<h", prefix) + payload + body[water.offset + water.length :]
    return donor, new


def _variant(payload: bytes, path: Path = TRIGGER_FIXTURE, prefix: int | None = None):
    """`path` reloaded with water_definition's raw payload (and, optionally,
    its raw length prefix) replaced."""
    donor, body = _variant_body(path, len(payload) if prefix is None else prefix, payload)
    return load_map_and_units_from_bytes(donor.header_bytes + _compress_bytes(body), path.name)


def _scalars_ok(loaded) -> bool:
    return options_write_supported(loaded, option_fields.specs_for(loaded))


# -- the gate passes -------------------------------------------------------------


@pytest.mark.parametrize("path", [BLANK_TEMPLATE_PATH, TRIGGER_FIXTURE, V159_FIXTURE], ids=["blank", "triggers", "v159"])
def test_the_gate_passes_on_the_shipped_files(path) -> None:
    loaded = load_map_and_units(path)
    found = map_strings.layout(loaded)
    assert found is not None
    names = [item.name for item in found.items if item.is_string]
    assert names == ["water_definition", "map_color_mood", "script_name"]
    assert found.start == loaded.options_section_end
    assert map_strings.current_value(loaded, map_strings.WATER_DEFINITION) == ""


@pytest.mark.parametrize("payload", [b"Default", b"FE1", b"Preset_Main\x00"], ids=["default", "fe1", "preset-main-nul"])
def test_the_gate_passes_on_stored_names(payload) -> None:
    loaded = _variant(payload)
    assert map_strings.write_supported(loaded)
    assert _item(loaded, map_strings.WATER_DEFINITION).length == 2 + len(payload)


def test_a_trailing_nul_is_read_without_it_and_measured_with_it() -> None:
    """28 corpus files store Preset_Main plus a NUL the library strips, so its
    retriever_length() is one byte short of the 14 bytes on disk."""
    loaded = _variant(b"Preset_Main\x00")
    assert map_strings.current_value(loaded, map_strings.WATER_DEFINITION) == "Preset_Main"
    assert _item(loaded, map_strings.WATER_DEFINITION).length == 14


def test_the_region_stops_before_script_name() -> None:
    """GH #216's own splice owns script_name, so this region ends at map_color_mood."""
    loaded = load_map_and_units(TRIGGER_FIXTURE)
    found = map_strings.layout(loaded)
    mood = _item(loaded, "map_color_mood")
    script = _item(loaded, "script_name")
    assert found.end == mood.offset + mood.length
    assert found.end <= script.offset
    assert found.front_end == script.offset + script.length


# -- the gate fails closed ---------------------------------------------------------


def test_a_wrong_marker_fails_the_gate_only() -> None:
    donor = load_map_and_units(TRIGGER_FIXTURE)
    body = donor.decompressed_body
    start = donor.options_section_end
    forged = body[:start] + b"\x61\x0a" + body[start + 2 :]
    loaded = load_map_and_units_from_bytes(donor.header_bytes + _compress_bytes(forged), "marker.aoe2scenario")
    assert not map_strings.write_supported(loaded)
    assert _scalars_ok(loaded)


def test_a_negative_length_fails_the_gate_only() -> None:
    loaded = _variant(b"", prefix=-1)
    assert not map_strings.write_supported(loaded)
    assert _scalars_ok(loaded)


def test_a_length_past_the_terrain_block_fails_the_gate_only() -> None:
    _donor, body = _variant_body(TRIGGER_FIXTURE, 30_000, b"")
    loaded = dataclasses.replace(load_map_and_units(TRIGGER_FIXTURE), decompressed_body=body)
    assert not map_strings.write_supported(loaded)
    assert _scalars_ok(loaded)


def test_a_nul_that_survives_the_strip_fails_the_gate_only() -> None:
    loaded = _variant(b"X\x00\x00")
    assert map_strings.current_value(loaded, map_strings.WATER_DEFINITION) == "X\x00"
    assert not map_strings.write_supported(loaded)
    assert _scalars_ok(loaded)


def test_a_parsed_name_that_disagrees_with_its_bytes_fails_the_gate() -> None:
    loaded = load_map_and_units(TRIGGER_FIXTURE)
    loaded._scenario.sections["Map"].retriever_map["water_definition"].data = "Arctic"
    assert not map_strings.write_supported(loaded)


def test_an_untrusted_anchor_fails_the_gate() -> None:
    loaded = dataclasses.replace(load_map_and_units(TRIGGER_FIXTURE), terrain_block_offset=-1)
    assert map_strings.layout(loaded) is None


def test_a_missing_retriever_fails_the_gate() -> None:
    loaded = load_map_and_units(TRIGGER_FIXTURE)
    del loaded._scenario.sections["Map"].retriever_map["water_definition"]
    assert map_strings.layout(loaded) is None


# -- the scalar walk stays blind to the strings ------------------------------------


@pytest.mark.parametrize("payload", [b"", b"Preset_Main\x00"], ids=["empty", "preset-main-nul"])
def test_field_offsets_ignore_the_water_spec(payload) -> None:
    """A backward walk across water_definition lands one byte off on the NUL
    variant, so the scalar walk must never include it."""
    loaded = _variant(payload)
    specs = option_fields.specs_for(loaded)
    assert any(s.kind == option_fields.STRING_COMBO for s in specs)
    without = [s for s in specs if s.kind != option_fields.STRING_COMBO]
    assert field_offsets(loaded, specs) == field_offsets(loaded, without)
    assert _scalars_ok(loaded)


# -- encode and splice ---------------------------------------------------------------


def test_encode_str16_writes_no_trailing_nul() -> None:
    assert map_strings.encode_str16("Red Tide") == b"\x08\x00Red Tide"
    assert map_strings.encode_str16("") == b"\x00\x00"


@pytest.mark.parametrize(
    "value", [None, 5, b"Arctic", "a\x00b", "x" * (map_strings.STR16_MAX_PAYLOAD + 1)], ids=["none", "int", "bytes", "nul", "too-long"]
)
def test_encode_str16_refuses(value) -> None:
    with pytest.raises(ValueError):
        map_strings.encode_str16(value)


def test_the_splice_substitutes_only_the_edited_string() -> None:
    loaded = _variant(b"Preset_Main\x00")
    start, end, replacement = map_strings.splice(loaded, {"water_definition": "Noon"})
    body = loaded.decompressed_body
    water = _item(loaded, map_strings.WATER_DEFINITION)
    assert (start, end) == (map_strings.layout(loaded).start, map_strings.layout(loaded).end)
    expected = body[start : water.offset] + b"\x04\x00Noon" + body[water.offset + water.length : end]
    assert replacement == expected


def test_an_empty_splice_copies_the_region_verbatim() -> None:
    loaded = _variant(b"Preset_Main\x00")
    start, end, replacement = map_strings.splice(loaded, {})
    assert replacement == loaded.decompressed_body[start:end]


def test_the_splice_refuses_a_string_it_does_not_own() -> None:
    loaded = load_map_and_units(TRIGGER_FIXTURE)
    with pytest.raises(KeyError):
        map_strings.splice(loaded, {"script_name": "probe"})
    with pytest.raises(KeyError):
        map_strings.splice(loaded, {"map_color_mood": "Winter"})
