"""Age of Empires (1997) and Rise of Rome 1.15 scenarios (TASK-515): read,
shown with Return of Rome data, view-only. No third-party bytes: the file
below is packed field by field in agescx's documented order with the
>= 1.18 fields left out, so its body half is a regression pin, not
independent evidence. Real files are tests/test_aoe1_corpus.py's."""

from __future__ import annotations

import json
import struct
import zlib

import pytest

from descape import game_dataset, games, legacy_convert, library_compat, scenario_format, scenario_io
from descape.scenario_write import LegacyFormatReadOnly, build_patched, refuse_legacy

_SEPARATOR = 0xFFFFFF9D
_MAP = 4
# (const, x, y) per player section: GAIA first, then player 1. Town Center and Bowman.
_UNITS = ([(109, 1.5, 2.5)], [(4, 3.25, 0.75)])
# Instructions, hints, victory, loss, history: no scouts field, the last two NUL-terminated.
_MESSAGES = (b"Hold the ford", b"", b"Won\rwell", b"Lost\x00", b"Long ago\x00")


def _str16(text: bytes) -> bytes:
    return struct.pack("<H", len(text)) + text


def _header() -> bytes:
    """The header-2 FileHeader, as tests/test_legacy_format.py packs it, file version 1.11."""
    rest = struct.pack("<iI", 2, 1234) + struct.pack("<I", 2) + b"Hi" + struct.pack("<II", 0, 2)
    return b"1.11" + struct.pack("<I", len(rest)) + rest


def _body() -> bytes:
    out = [struct.pack("<If", 7, 1.15), b"\x00" * 256 * 16]
    out += [struct.pack("<4I", 1, i == 0, 1, 4) for i in range(16)]  # active, human, civilization, unknown
    out += [b"\x00" * 9, _str16(b"tiny.scx")]
    out += [_str16(text) for text in _MESSAGES]  # no string-table ids before 1.18
    out += [_str16(b" <None> ") for _ in range(3)]  # cinematics
    out += [_str16(b" <None> "), struct.pack("<IIih", 0, 0, 0, 1)]  # background, no bitmap
    out += [_str16(b"") for _ in range(48)]  # player strings, AI names
    out += [b"\x00" * 8 + struct.pack("<i", 0) for _ in range(16)]  # AI files; no AI type byte
    out += [struct.pack("<I", _SEPARATOR)]
    out += [struct.pack("<4i", 0, 200, 200, 150) for _ in range(16)]  # gold, wood, food, stone
    out += [struct.pack("<I10i", _SEPARATOR, 1, 0, 0, 0, 0, 0, 0, 0, 900, 9000)]
    out += [b"\x00" * 16 * 16 * 4, b"\x00" * 11520, struct.pack("<I", _SEPARATOR), b"\x00" * 64]
    out += [struct.pack("<320I", *[1] * 320), struct.pack("<3I", 0, 0, 0), struct.pack("<16I", *[0] * 16)]
    out += [struct.pack("<Iii", _SEPARATOR, _MAP, _MAP)]
    out += [struct.pack("<BBB", (i % 3) * 2, i % 2, 0) for i in range(_MAP * _MAP)]
    out += [struct.pack("<I", 9), struct.pack("<32f", *[100.0] * 32)]
    ref = 0
    for section in range(9):
        units = _UNITS[section] if section < len(_UNITS) else []
        out.append(struct.pack("<I", len(units)))
        for const, x, y in units:
            out.append(struct.pack("<fffiHBf", x, y, 0.0, ref, const, 2, 0.0))  # 23 bytes
            ref += 1
    out.append(struct.pack("<I", 9))
    for player in range(8):
        # constant_name, cameras, allied victory, 9 stances, 9 AI stances, victory version 1.0, 13 bytes
        out.append(_str16(b"Player %d\x00" % (player + 1)) + struct.pack("<ffhhBH", 72.0, 72.0, 72, 72, 0, 9))
        out.append(bytes(9) + struct.pack("<9I", *[4] * 9) + struct.pack("<f", 1.0) + bytes(13))
    return b"".join(out)


def _raw() -> bytes:
    deflate = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
    return _header() + deflate.compress(_body()) + deflate.flush()


@pytest.fixture
def aoe1():
    loaded = scenario_io.load_map_and_units_from_bytes(_raw(), "tiny.scx")
    yield loaded


def test_the_band_is_detected_and_named():
    fmt = scenario_format.detect_format(_raw())
    assert (fmt.data_version, fmt.game, fmt.legacy) == ("1.15", scenario_format.AOE1, True)
    assert fmt.display_name == "Age of Empires" and scenario_format.is_aoe1_era(fmt)
    assert not scenario_format.converts(fmt)
    reason = scenario_format.read_only_reason(fmt)
    assert reason == "Read-only: Age of Empires files open for viewing only.", "no AoE2:DE route offered"
    assert dict(scenario_format.absent_unit_fields(fmt)) == {"initial_animation_frame": 0, "garrisoned_in_id": -1}
    assert not scenario_format.absent_unit_fields(scenario_format.DE_FORMAT)


def test_a_1_15_file_loads_its_map_and_units(aoe1):
    mm = aoe1.map_manager
    assert (mm.map_width, mm.map_height) == (_MAP, _MAP)
    assert [(t.terrain_id, t.elevation) for t in mm.terrain[:4]] == [(0, 0), (2, 1), (4, 0), (0, 1)]
    units = aoe1.unit_manager.units
    assert [len(p) for p in units] == [1, 1] + [0] * 7
    got = [(u.unit_const, u.x, u.y, u.initial_animation_frame, u.garrisoned_in_id) for p in units for u in p]
    assert got == [(109, 1.5, 2.5, 0, -1), (4, 3.25, 0.75, 0, -1)]
    assert aoe1.trigger_tail == b"", "the body ends with Units"
    assert aoe1.terrain_block_offset >= 0 and aoe1.units_block_offset >= 0


def test_a_1_15_file_is_return_of_romes_and_view_only(aoe1):
    assert games.for_scenario(aoe1) is games.ROR
    assert game_dataset.file_game(aoe1) is games.ROR
    assert not scenario_io.unit_field_supported(aoe1, "garrisoned_in_id")
    assert not scenario_io.unit_field_supported(aoe1, "initial_animation_frame")
    assert scenario_io.unit_field_supported(aoe1, "rotation")
    with pytest.raises(LegacyFormatReadOnly, match="Age of Empires files open for viewing only"):
        refuse_legacy(aoe1)
    patched = build_patched(aoe1)
    assert (patched.header, patched.body) == (aoe1.header_bytes, aoe1.decompressed_body)
    with pytest.raises(ValueError, match="do not convert"):
        legacy_convert.convert(aoe1)


def test_the_library_walk_is_refused_by_name():
    """The library's Unit links both absent fields with no support range, so its walk can never read one."""
    with pytest.raises(scenario_io.UnsupportedStructureVersion, match="fast unit reader is off"):
        scenario_io.load_map_and_units_from_bytes(_raw(), "tiny.scx", fast_units=False)


def test_the_measured_deltas_from_aok():
    """Measured 2026-11-26 on the 64 corpus files: v1.20 less these, and no Triggers section."""
    v115, v120 = (json.loads(library_compat.legacy_structure_path(v).read_text(encoding="utf-8")) for v in ("1.15", "1.20"))
    assert list(v115) == [name for name in v120 if name != "Triggers"]
    units = v115["Units"]["structs"]
    unit = units["PlayerUnitsStruct"]["structs"]["UnitStruct"]["retrievers"]
    assert list(unit) == ["x", "y", "z", "reference_id", "unit_const", "status", "rotation"]
    assert len(units["PlayerDataFourStruct"]["retrievers"]) == 4 and "color" not in units["PlayerDataThreeStruct"]["retrievers"]
    two = v115["PlayerDataTwo"]
    assert "ai_type" not in two["retrievers"] and len(two["structs"]["ResourcesStruct"]["retrievers"]) == 4
    assert list(two["structs"]["AIStruct"]["retrievers"])[:3] == ["ai_file_length", "city_plan_length", "personality_length"]
    options = v115["Options"]["retrievers"]
    assert options["unknown_per_player_flags"]["repeat"] == 320
    assert list(options)[1:] == ["combat_mode", "naval_mode", "all_techs", "per_player_starting_age"]
    assert "player_1_camera_x" not in v115["Map"]["retrievers"]
    image = v115["BackgroundImage"]["structs"]["BitMapInfoStruct"]["retrievers"]["image"]
    assert image["dependencies"]["on_construct"]["target"] == "self:image_size", "rows pad to 4 bytes"


def test_a_1_15_file_shows_its_messages_without_string_ids(aoe1):
    """1.15 stores the five texts with no string-table ids (GH #312's Somin1 showed "0 fields")."""
    from descape.messages_fields import present_fields, stored_text
    from descape.messages_model import MessagesEditModel

    assert aoe1.messages_write_supported, "the id-less walk verifies"
    retrievers = aoe1._scenario.sections["Messages"].retriever_map
    fields = present_fields(retrievers)
    assert [s.field_id for s in fields] == ["instructions", "hints", "victory", "loss", "history"]
    assert [stored_text(aoe1, s) for s in fields] == ["Hold the ford", "", "Won\rwell", "Lost", "Long ago"]
    model = MessagesEditModel(aoe1)
    assert model.current_value("victory") == "Won\nwell"
    start, end = model.section_span()
    assert model.serialize() == aoe1.decompressed_body[start:end], "no ids written, texts verbatim"
    with pytest.raises(KeyError):
        model.set_value("loss_id", 7)


def test_the_messages_panel_lists_the_five_fields(aoe1):
    pytest.importorskip("PyQt5")
    from descape.messages_panel import MessagesPanel

    import conftest

    conftest.ensure_qapp()
    panel = MessagesPanel()
    panel.show_scenario(aoe1, editable=False)
    assert [s.field_id for s in panel._fields] == ["instructions", "hints", "victory", "loss", "history"]
    assert panel.status.text().startswith("5 fields")
    assert not panel._warning_labels, "no string id to warn about"


def test_the_triggers_panel_says_the_file_has_none(aoe1):
    pytest.importorskip("PyQt5")
    from descape.trigger_panel import TriggerPanel

    key = TriggerPanel._unsupported_key(aoe1)
    assert key == "legacy_no_triggers" and "Convert" not in TriggerPanel._UNSUPPORTED[key]


def test_no_write_path_encodes_a_1_15_unit(aoe1, tmp_path):
    """The band saves no family, so a unit edit (which would encode 23 bytes, dropping the garrison) is refused."""
    from pathlib import Path

    from descape import legacy_write
    from descape.scenario_write import LegacyEditNotAllowed, check_destination
    from descape.unit_model import UnitEditModel

    assert not legacy_write.allowed(aoe1) and "1.15" not in legacy_write.ALLOWED
    model = UnitEditModel(aoe1)
    unit = aoe1.unit_manager.units[1][0]
    model.set_position(unit, unit.x + 1, unit.y, unit.z)
    with pytest.raises(LegacyEditNotAllowed, match=r"Undo them\.$"):
        build_patched(aoe1, units=model)
    with pytest.raises(LegacyFormatReadOnly) as caught:
        check_destination(aoe1, Path(tmp_path / "x.aoe2scenario"))
    assert "Convert" not in str(caught.value)
