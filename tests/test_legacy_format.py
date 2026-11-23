"""Pre-DE scenario formats (GH #308): detection, the three legacy structures,
and the read-only guard. No third-party bytes: real files are the corpus
tier's (tests/test_legacy_corpus.py). The FileHeader parse is the one
independent check here; the field pins only guard against regressions."""

from __future__ import annotations

import json
import shutil
import struct
import sys
import zlib
from pathlib import Path

import pytest
from AoE2ScenarioParser.helper.incremental_generator import IncrementalGenerator
from AoE2ScenarioParser.scenarios.aoe2_de_scenario import AoE2DEScenario

from descape import batch_api, library_compat, scenario_format, scenario_io
from descape.messages_fields import MESSAGE_FIELDS, present_fields
from descape.scenario_format import ScenarioFormat
from descape.scenario_resize import Anchor, resize_scenario
from descape.scenario_write import LegacyFormatReadOnly, WriteBlockedError, build_patched, write_scenario

import conftest

_GUI = [pytest.mark.gui, pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")]
_LEGACY = library_compat.LEGACY_VERSIONS_DIR
_HD = library_compat.REPO_VERSIONS_DIR / "v1.21" / "structure.json"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
AOC = ScenarioFormat("1.21", 2, "1.22", "aoc", True)


def _header2(file_version: bytes, instructions: bytes = b"Hi", player_count: int = 3) -> bytes:
    """A header-2 FileHeader packed by hand: c4, u32 header_length, s32 2,
    u32 timestamp, str32 instructions, u32 individual_victories_used, u32 player_count."""
    rest = struct.pack("<iI", 2, 1234) + struct.pack("<I", len(instructions)) + instructions + struct.pack("<II", 0, player_count)
    return file_version + struct.pack("<I", len(rest)) + rest


def _file(data_version: float, file_version: bytes = b"1.21") -> bytes:
    body = struct.pack("<If", 7, data_version) + b"\x00" * 16
    deflate = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
    return _header2(file_version) + deflate.compress(body) + deflate.flush()


def test_detect_format_keys_header_2_by_data_version():
    for version, game in ((1.20, "aok"), (1.22, "aoc"), (1.23, "hd"), (1.26, "hd"), (1.15, None)):
        fmt, body = scenario_format.read_format(_file(version))
        assert (fmt.header_version, fmt.legacy, fmt.game) == (2, True, game), version
        assert fmt.data_version == f"{version:.2f}", "two decimals: 1.20, never 1.2"
        assert body[:4] == struct.pack("<I", 7)
    hd3 = b"1.21" + struct.pack("<Ii", 4, 3) + b"\x00" * 8
    assert scenario_format.detect_format(hd3) == ScenarioFormat("1.21", 3, None, "hd", False)
    de = scenario_io.BLANK_TEMPLATE_PATH.read_bytes()
    fmt = scenario_format.detect_format(de)
    assert not fmt.legacy and fmt.game is None and fmt.header_version != 2
    assert scenario_format.detect_format(b"1.2") == scenario_format.DE_FORMAT
    broken = _header2(b"1.21") + b"not deflate"
    assert scenario_format.read_format(broken) == (ScenarioFormat("1.21", 2, None, None, True), None)


def test_a_header_2_band_without_a_structure_is_refused_by_name():
    for raw, label in (
        (_file(1.23), "1.21 (data 1.23)"),
        (_file(1.21), "1.21 (data 1.21)"),
        (_header2(b"1.21") + b"not deflate", "1.21"),
    ):
        with pytest.raises(scenario_io.UnsupportedStructureVersion) as caught:
            scenario_io.load_map_and_units_from_bytes(raw, "old.scx")
        assert caught.value.scenario_version == label
        message = str(caught.value)
        assert "isn't supported yet" in message and "own definition" not in message, "never the HD 1.21 path"
    assert "data version 1.23 (UserPatch / HD Edition .scx)" in str(
        scenario_io.UnsupportedStructureVersion.for_format(scenario_format.detect_format(_file(1.23)))
    )


@pytest.mark.parametrize("data_version", ["1.20", "1.22", "1.26"])
def test_the_header_2_file_header_parses_under_each_legacy_structure(data_version):
    raw = _header2(b"1.18", instructions=b"Win it", player_count=5)
    scenario = AoE2DEScenario("DE", "1.18", source_location="x", name="", variant=None)
    scenario.structure = json.loads(library_compat.legacy_structure_path(data_version).read_text(encoding="utf-8"))
    igen = IncrementalGenerator("x", raw + b"body")
    scenario._load_header_section(igen)
    header = scenario.sections["FileHeader"].retriever_map
    assert igen.progress == scenario_format.body_offset(raw) == len(raw)
    assert (header["savable"].data, header["timestamp_of_last_save"].data) == (2, 1234)
    assert (header["scenario_instructions"].data, header["player_count"].data) == ("Win it", 5)
    assert "unknown_numbers" not in header, "header 2 has no DLC list"


def _structure(name: str) -> dict:
    path = _HD if name == "hd" else _LEGACY / f"v{name}" / "structure.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _has(node, key: str) -> bool:
    if isinstance(node, dict):
        return key in node or any(_has(v, key) for v in node.values())
    return isinstance(node, list) and any(_has(v, key) for v in node)


def test_the_measured_deltas_from_hd():
    hd = _structure("hd")
    for version in ("1.20", "1.22", "1.26"):
        assert list(_structure(version)) == list(hd), f"v{version}: section order"
    aoc, aok = _structure("1.22"), _structure("1.20")
    for s in (aoc, aok):
        options = s["Options"]["retrievers"]
        assert options["disabled_building_ids_player_1"]["repeat"] == 20
        # Measured 2026-11-22 by the GlobalVictory/Diplomacy separators: dropping the
        # age and keeping the colour also sums right, but misplaces every field between.
        assert list(options)[-2:] == ["all_techs", "per_player_starting_age"]
        assert "player_color" not in s["PlayerDataTwo"]["structs"]["ResourcesStruct"]["retrievers"]
        assert "lock_teams" not in s["Diplomacy"]["retrievers"]
        assert "unknown_2" not in s["Map"]["retrievers"]
    assert "unknown_1" in aoc["Map"]["retrievers"] and "scouts" in aoc["Messages"]["retrievers"]
    assert "unknown_1" not in aok["Map"]["retrievers"]
    assert "scouts" not in aok["Messages"]["retrievers"] and "ascii_scouts" not in aok["Messages"]["retrievers"]
    assert _has(aoc, "population_limit") and not _has(aok, "population_limit"), "AoK: 6 floats per player, not 7"
    h126 = _structure("1.26")
    assert {k: v for k, v in h126.items() if k != "FileHeader"} == {k: v for k, v in hd.items() if k != "FileHeader"}


def _legacy(path: Path = scenario_io.BLANK_TEMPLATE_PATH) -> scenario_io.LoadedScenario:
    loaded = scenario_io.load_map_and_units(path)
    loaded.source_format, loaded.structure_source = AOC, "legacy"  # what a real header-2 load sets
    return loaded


def test_every_write_path_refuses_a_legacy_document(tmp_path):
    out = tmp_path / "out.aoe2scenario"
    loaded = _legacy()
    for call in (
        lambda: write_scenario(loaded, out),
        lambda: build_patched(loaded),
        lambda: resize_scenario(loaded, 240, 240, Anchor.TOP_LEFT),
        lambda: batch_api.resize_map(loaded, 240, 240),
    ):
        with pytest.raises(LegacyFormatReadOnly, match="The Conquerors files open for viewing only"):
            call()
    assert issubclass(LegacyFormatReadOnly, WriteBlockedError)
    assert not out.exists()


def test_write_scenario_refuses_a_legacy_document_before_patching(tmp_path, monkeypatch):
    """Its own guard, not only build_patched()'s, which resize reaches directly."""
    from descape import scenario_write

    def reached(*_a, **_k):
        raise AssertionError("write_scenario() reached build_patched()")

    monkeypatch.setattr(scenario_write, "build_patched", reached)
    with pytest.raises(LegacyFormatReadOnly):
        write_scenario(_legacy(), tmp_path / "out.aoe2scenario")


def test_strip_units_refuses_a_legacy_file_before_touching_any(tmp_path, monkeypatch):
    first, second = tmp_path / "a.aoe2scenario", tmp_path / "b.scx"
    shutil.copyfile(scenario_io.BLANK_TEMPLATE_PATH, first)
    shutil.copyfile(FIXTURES / "units_120x120.aoe2scenario", second)
    before, second_bytes = first.read_bytes(), second.read_bytes()
    real = scenario_format.detect_format
    monkeypatch.setattr(scenario_format, "detect_format", lambda raw: AOC if raw == second_bytes else real(raw))
    module = conftest.load_verify_module("strip_units")
    monkeypatch.setattr(sys, "argv", ["strip_units", str(first), str(second)])
    with pytest.raises(LegacyFormatReadOnly):
        module.main()
    assert first.read_bytes() == before


def test_messages_walk_only_the_fields_the_file_has():
    loaded = scenario_io.load_map_and_units(scenario_io.BLANK_TEMPLATE_PATH)
    full = loaded._scenario.sections["Messages"].retriever_map
    assert present_fields(full) == MESSAGE_FIELDS
    aok = {k: v for k, v in full.items() if k not in ("scouts", "ascii_scouts")}
    assert [s.field_id for s in present_fields(aok)] == ["instructions", "hints", "victory", "loss", "history"]
    scouts = scenario_io.retriever_length(full["scouts"]) + scenario_io.retriever_length(full["ascii_scouts"])
    start, end = loaded.messages_section_start, loaded.messages_section_end
    assert scenario_io._verify_messages_block(loaded.decompressed_body, start, end, full)
    assert scenario_io._verify_messages_block(loaded.decompressed_body, start, end - scouts, aok)
    assert not scenario_io._verify_messages_block(loaded.decompressed_body, start, end, aok)


def test_the_messages_model_and_panel_take_a_scouts_less_file(monkeypatch):
    from descape.messages_model import MessagesEditModel

    loaded = scenario_io.load_map_and_units(scenario_io.BLANK_TEMPLATE_PATH)
    section = loaded._scenario.sections["Messages"]
    aok = {k: v for k, v in section.retriever_map.items() if k not in ("scouts", "ascii_scouts")}
    monkeypatch.setattr(section, "retriever_map", aok)
    model = MessagesEditModel(loaded)
    assert [s.field_id for s in model.fields] == ["instructions", "hints", "victory", "loss", "history"]
    assert len(model.serialize()) == 5 * 4 + sum(2 + len(model._encoded(s.field_id)) for s in model.fields)
    if conftest.PYQT5_AVAILABLE:
        conftest.ensure_qapp()
        from descape.messages_panel import MessagesPanel

        panel = MessagesPanel()
        panel.show_scenario(loaded)
        assert panel.status.text().startswith("5 fields")


@_GUI[0]
@_GUI[1]
def test_the_window_shows_a_legacy_document_read_only(monkeypatch):
    from PyQt5.QtWidgets import QMessageBox

    shown: list[str] = []
    monkeypatch.setattr(QMessageBox, "information", lambda _parent, _title, text, *a, **k: shown.append(text))
    window = conftest.blank_window(load=False)
    try:
        loaded = _legacy()
        window.load_scenario(Path("old.scx"), loaded=loaded)
        reason = window.game_data_lock_reason()
        assert reason.startswith("Read-only: The Conquerors files open for viewing only")
        assert not window.save_action.isEnabled() and not window.save_as_action.isEnabled()
        assert window._refuse_locked_save() and shown == [reason]
        assert window.trigger_panel._unsupported_key(loaded) == "legacy"
        assert scenario_io.parse_triggers(loaded) is None and loaded.trigger_read_supported is False
    finally:
        conftest.close_window(window)


@_GUI[0]
@_GUI[1]
def test_a_legacy_clipboard_block_is_refused_on_a_de_document():
    from descape import region_clipboard, trigger_clipboard

    window = conftest.blank_window(load=False)
    try:
        window.load_scenario(Path("old.scx"), loaded=_legacy())
        loaded = window.scenario
        block = window._with_file_game(
            region_clipboard.copy_region(loaded.map_manager, loaded.unit_manager, 0, 0, 2, 2)
        )
        assert block.legacy
        window.load_scenario(FIXTURES / "triggers_120x120.aoe2scenario")
        assert "older-format (.scn/.scx)" in window._cross_game_refusal(block)
        loaded = window.scenario
        de_block = window._with_file_game(
            region_clipboard.copy_region(loaded.map_manager, loaded.unit_manager, 0, 0, 2, 2)
        )
        assert not de_block.legacy and window._cross_game_refusal(de_block) == ""
        manager = scenario_io.parse_triggers(loaded)
        copied = window._with_file_game(
            trigger_clipboard.copy_block(manager, [0], doc_id=window._doc_id, scenario_version=loaded.scenario_version)
        )
        assert window._cross_game_refusal(copied) == "", "a trigger block has no legacy field and pastes"
    finally:
        conftest.close_window(window)
