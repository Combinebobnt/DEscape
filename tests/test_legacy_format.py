"""Pre-DE scenario formats (GH #308): detection, the legacy structures,
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

from descape import batch_api, games, library_compat, scenario_format, scenario_io
from descape.messages_fields import MESSAGE_FIELDS, present_fields
from descape.scenario_format import ScenarioFormat
from descape.scenario_resize import Anchor, resize_scenario
from descape.scenario_write import (
    LegacyEditNotAllowed,
    LegacyFormatReadOnly,
    WriteBlockedError,
    build_patched,
    write_scenario,
)

import conftest

_GUI = [pytest.mark.gui, pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")]
_LEGACY = library_compat.LEGACY_VERSIONS_DIR
_HD = library_compat.REPO_VERSIONS_DIR / "v1.21" / "structure.json"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
AOC = ScenarioFormat("1.21", 2, "1.22", scenario_format.AOC, True)


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
    aok, aoc, hd, aoe1 = scenario_format.AOK, scenario_format.AOC, scenario_format.HD, scenario_format.AOE1
    up, hd_de = scenario_format.UP, scenario_format.HD_DE
    assert games.ids() and not {aok, aoc, hd, aoe1, up, hd_de} & set(games.ids()), "a label is never a game id"
    bands = ((1.20, aok), (1.22, aoc), (1.21, hd), (1.23, up), (1.24, hd), (1.26, hd), (1.15, aoe1), (1.10, None))
    for version, game in bands:
        fmt, body = scenario_format.read_format(_file(version))
        assert (fmt.header_version, fmt.legacy, fmt.game) == (2, True, game), version
        assert fmt.data_version == f"{version:.2f}", "two decimals: 1.20, never 1.2"
        assert body[:4] == struct.pack("<I", 7)
    hd3 = b"1.21" + struct.pack("<Ii", 4, 3) + b"\x00" * 8
    assert scenario_format.detect_format(hd3) == ScenarioFormat("1.21", 3, None, hd_de, False)
    de = scenario_io.BLANK_TEMPLATE_PATH.read_bytes()
    fmt = scenario_format.detect_format(de)
    assert not fmt.legacy and fmt.game is None and fmt.header_version != 2
    assert scenario_format.detect_format(b"1.2") == scenario_format.DE_FORMAT
    broken = _header2(b"1.21") + b"not deflate"
    assert scenario_format.read_format(broken) == (ScenarioFormat("1.21", 2, None, None, True), None)


def test_a_header_2_band_without_a_structure_is_refused_by_name():
    for raw, label in (
        (_file(1.25), "1.21 (data 1.25)"),
        (_file(1.10, b"1.11"), "1.11 (data 1.10)"),
        (_header2(b"1.21") + b"not deflate", "1.21"),
    ):
        with pytest.raises(scenario_io.UnsupportedStructureVersion) as caught:
            scenario_io.load_map_and_units_from_bytes(raw, "old.scx")
        assert caught.value.scenario_version == label
        message = str(caught.value)
        assert "isn't supported yet" in message and "own definition" not in message, "never the HD 1.21 path"
    assert "data version 1.25 (older .scn/.scx/.scx2)" in str(
        scenario_io.UnsupportedStructureVersion.for_format(scenario_format.detect_format(_file(1.25)))
    )
    # AoE2:DE neither imports nor converts an Age of Empires (1997) file, so no message offers it.
    aoe1 = str(scenario_io.UnsupportedStructureVersion.for_format(scenario_format.detect_format(_file(1.10))))
    assert "Age of Empires scenario data version 1.10" in aoe1 and "AoE2" not in aoe1


_BANDS = ("1.20", "1.21", "1.22", "1.23", "1.24", "1.26")


@pytest.mark.parametrize("data_version", (*_BANDS, "1.15"))
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
    for version in _BANDS:
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
    # Measured 2026-11-26 (TASK-675): each HD/UserPatch band is a neighbour with one section changed.
    for version, base, section, donor in (("1.21", aoc, "Messages", aok), ("1.24", h126, "Options", aoc)):
        assert _structure(version) == dict(base, **{section: donor[section]}), f"v{version}"
    v123 = _structure("1.23")
    assert {k: v for k, v in v123.items() if k != "Diplomacy"} == {k: v for k, v in aoc.items() if k != "Diplomacy"}
    # Only byte 0 of the 4 is ever nonzero, so these are not HD's team fields.
    assert list(v123["Diplomacy"]["retrievers"]) == [*aoc["Diplomacy"]["retrievers"], "unknown_team_bytes"]


def _legacy(path: Path = scenario_io.BLANK_TEMPLATE_PATH) -> scenario_io.LoadedScenario:
    loaded = conftest.legacy_loaded(path)  # what a real header-2 load sets
    assert loaded.source_format == AOC
    return loaded


def test_a_legacy_document_saves_only_its_bands_allowed_edits(tmp_path):
    """TASK-520: resize stays refused, .aoe2scenario is refused, a no-edit build
    succeeds byte-identically and an edit outside ALLOWED raises LegacyEditNotAllowed."""
    out = tmp_path / "out.aoe2scenario"
    loaded = _legacy()
    for call in (
        lambda: resize_scenario(loaded, 240, 240, Anchor.TOP_LEFT),
        lambda: batch_api.resize_map(loaded, 240, 240),
    ):
        with pytest.raises(LegacyFormatReadOnly, match="The Conquerors files open for viewing only"):
            call()
    with pytest.raises(LegacyFormatReadOnly, match=r"not as \.aoe2scenario"):
        write_scenario(loaded, out)
    patched = build_patched(loaded)
    assert (patched.header, patched.body) == (loaded.header_bytes, loaded.decompressed_body)
    loaded.map_manager.terrain[0].terrain_id = 1 if loaded.map_manager.terrain[0].terrain_id != 1 else 2
    with pytest.raises(LegacyEditNotAllowed, match="can't save terrain edits"):
        build_patched(loaded)
    assert issubclass(LegacyFormatReadOnly, WriteBlockedError) and issubclass(LegacyEditNotAllowed, WriteBlockedError)
    assert not out.exists()


def test_write_scenario_refuses_a_legacy_document_before_patching(tmp_path, monkeypatch):
    """An .aoe2scenario name is refused by write_scenario()'s destination guard, before any patch runs."""
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
def test_the_window_opens_only_an_allowed_familys_tools_on_a_legacy_document(tmp_path, monkeypatch):
    """TASK-520 with terrain and elevation allowed: their tools and Save enable, every unit,
    cliff, mirror, resize, planting and Teams control stays off, Save backs up with backups
    switched off, Save As keeps the format, and Return of Rome data still locks it."""
    from PyQt5.QtWidgets import QFileDialog

    from descape import game_dataset, legacy_write, settings

    settings.set_backups_enabled(False)
    folder = tmp_path / "maps"
    folder.mkdir()
    path = folder / "old.scx"
    shutil.copyfile(scenario_io.BLANK_TEMPLATE_PATH, path)
    window = conftest.blank_window(load=False)
    try:
        with legacy_write._allow_for_oracle("1.22", {legacy_write.TERRAIN, legacy_write.ELEVATION}):
            window.load_scenario(path, loaded=_legacy(path))
            window.mode_combo.setCurrentText("Terrain")  # a hidden action never reads as enabled
            on = ("draw_action", "fill_action", "draw_line_action", "draw_rect_action", "elevation_action",
                  "set_level_action", "save_action", "save_as_action")
            off = ("cliff_action", "mirror_action", "resize_action", "select_action", "eraser_action",
                   "disables_action")
            assert [n for n in on if not getattr(window, n).isEnabled()] == []
            assert [n for n in off if getattr(window, n).isEnabled()] == []
            assert not window.paint_trees_check.isEnabled() and window._terrain_unit_boxes() == (False, False)
            assert window._refuse_locked_edit() and window._refuse_locked_edit(legacy_write.DIPLOMACY)
            assert window._ensure_option_edits() is None, "Teams rows and other options stay locked"
            window.mode_combo.setCurrentText("Units")
            assert window.mode == "units"
            assert not window.place_unit_action.isEnabled() and not window.move_action.isEnabled()
            window.mode_combo.setCurrentText("Terrain")

            tiles = window.scenario.map_manager.terrain
            window.edit_history.apply("paint", tiles, lambda: setattr(tiles[0], "terrain_id", 1 if tiles[0].terrain_id != 1 else 2))
            window.save()
            assert sorted(p.name for p in folder.iterdir()) == ["old.scx", "old.scx.bak", "old.scx.orig"]
            assert window._save_as_filter() == "The Conquerors scenario (*.scx)"
            window._untitled = True  # a recovered autosave slot
            assert Path(window._save_as_start_path()).name == "Untitled.scx"
            window._untitled = False
            target = folder / "copy.scx"
            monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(target), "")))
            window.save_as()
            assert target.exists() and window.scenario.path == target

            window.set_game_data(game_dataset.ROR)
            assert window.terrain_lock_reason(legacy_write.TERRAIN) is not None
            assert not window.draw_action.isEnabled() and not window.save_action.isEnabled()
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
        assert "older-format (.scn/.scx/.scx2)" in window._cross_game_refusal(block)
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


@_GUI[0]
@_GUI[1]
def test_a_legacy_band_with_diplomacy_opens_only_its_stance_cells():
    from descape import diplomacy_fields, legacy_write

    window = conftest.blank_window(load=False)
    try:
        with legacy_write._allow_for_oracle("1.22", {legacy_write.DIPLOMACY}):
            window.load_scenario(Path("old.scx"), loaded=_legacy())
            editable = window._editable_diplomacy_fields()
            allied = set(diplomacy_fields.allied_victory_offsets(window.scenario))
            assert set(diplomacy_fields.stance_offsets(window.scenario)) <= editable and not allied & editable
            reasons = window._diplomacy_read_only_reasons()
            assert allied <= set(reasons) and all(reasons[c].startswith("Older-format file") for c in allied)
    finally:
        conftest.close_window(window)


@_GUI[0]
@_GUI[1]
def test_a_region_paste_is_refused_on_a_legacy_document_even_with_terrain_allowed():
    from descape import legacy_write

    window = conftest.blank_window(load=False)
    try:
        window.load_scenario(FIXTURES / "triggers_120x120.aoe2scenario")
        window._on_tool_selected("select")
        window.on_region_selected((0, 0, 2, 2))
        window.copy_region()
        assert window._clipboard_history.active_block is not None
        with legacy_write._allow_for_oracle("1.22", {legacy_write.TERRAIN, legacy_write.ELEVATION}):
            window.load_scenario(Path("old.scx"), loaded=_legacy())
            assert window._clipboard_history.active_block is not None
            before = [(t.terrain_id, t.elevation) for t in window.scenario.map_manager.terrain]
            window.paste_terrain_check.setChecked(True)
            window.on_hover((10, 10))
            window.paste_region()
            assert [(t.terrain_id, t.elevation) for t in window.scenario.map_manager.terrain] == before
            assert "Paste refused: Older-format file" in window.status_log.toPlainText()
    finally:
        conftest.close_window(window)
