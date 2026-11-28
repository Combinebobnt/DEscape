"""TASK-057 in the window: a scenario newer than every known structure opens
read-only in compatibility mode. The newer file is simulated by rewriting a
known file's version c4 to 1.60, and scenario_compat's test seam picks the
base and the donor. The donors carry distinctive strings written through the
real write path, so a panel that shows a donor's value as the file's is caught."""

from __future__ import annotations

import pytest
from test_scenario_compat import BLANK_V158, V159, _rebuilt, c4

from descape import library_compat, scenario_compat, scenario_io, settings
from descape.messages_model import MessagesEditModel
from descape.options_model import OptionsEditModel
from descape.scenario_write import write_scenario
from testkit.viewer_family import set_viewer_global

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

DONOR_TRIBE = "Qzdonortribe"
DONOR_TEXT = "Qzdonorinstructions"


def _marked_donor(src, tmp_path) -> bytes:
    """`src` with player 1's name and the instructions set to the donor marks, through the write path."""
    loaded = scenario_io.load_map_and_units(src)
    options, messages = OptionsEditModel(loaded), MessagesEditModel(loaded)
    options.set_value("player:tribe_name:1", DONOR_TRIBE)
    messages.set_value("instructions", DONOR_TEXT)
    out = tmp_path / f"donor_{src.parent.name}_{src.name}"
    write_scenario(loaded, out, options=options, messages=messages, backup=False)
    return out.read_bytes()


@pytest.fixture()
def window():
    window = conftest.blank_window(load=False)
    yield window
    conftest.close_window(window)


def _open(window, monkeypatch, tmp_path, raw: bytes, base: str | None = None, donor: bytes | None = None):
    monkeypatch.setattr(scenario_compat, "TEST_BASE", base)
    monkeypatch.setattr(scenario_compat, "TEST_DONOR", donor)
    path = tmp_path / "newer.aoe2scenario"
    path.write_bytes(raw)
    window.load_scenario(path)
    assert window.scenario is not None and window.scenario.compat is not None
    return window.scenario.compat


@pytest.fixture()
def units_unknown(window, monkeypatch, tmp_path):
    """The v159 fixture as 1.60 under base 1.58: Settings trusted, Units Unknown."""
    return _open(window, monkeypatch, tmp_path, c4(V159.read_bytes()), "1.58", _marked_donor(BLANK_V158, tmp_path))


@pytest.fixture()
def settings_unknown(window, monkeypatch, tmp_path):
    """The library's 1.56 default as 1.60 under base 1.55: Settings Unknown (the wrong-offset pair)."""
    raw = (library_compat.VERSIONS_DIR / "v1.56" / "default.aoe2scenario").read_bytes()
    donor = _marked_donor(library_compat.VERSIONS_DIR / "v1.55" / "default.aoe2scenario", tmp_path)
    return _open(window, monkeypatch, tmp_path, c4(raw), "1.55", donor)


@pytest.fixture()
def all_trusted(window, monkeypatch, tmp_path):
    """The v159 fixture as 1.60 under the newest structure with the shipped template."""
    return _open(window, monkeypatch, tmp_path, c4(V159.read_bytes()))


def _widget_text(window) -> str:
    from PyQt5.QtWidgets import (
        QAbstractButton,
        QComboBox,
        QLabel,
        QLineEdit,
        QListWidget,
        QPlainTextEdit,
        QTableWidget,
        QTextEdit,
        QTreeWidget,
        QTreeWidgetItemIterator,
        QWidget,
    )

    parts = []
    for w in window.findChildren(QWidget):
        parts.append(w.toolTip())
        if isinstance(w, (QLabel, QLineEdit, QAbstractButton)):
            parts.append(w.text())
        elif isinstance(w, (QTextEdit, QPlainTextEdit)):
            parts.append(w.toPlainText())
        elif isinstance(w, QComboBox):
            parts += [w.itemText(i) for i in range(w.count())]
        elif isinstance(w, QListWidget):
            parts += [w.item(i).text() for i in range(w.count())]
        elif isinstance(w, QTableWidget):
            parts += [it.text() for r in range(w.rowCount()) for c in range(w.columnCount()) if (it := w.item(r, c))]
        elif isinstance(w, QTreeWidget):
            it = QTreeWidgetItemIterator(w)
            while it.value():
                parts += [it.value().text(c) for c in range(w.columnCount())]
                it += 1
    return "\n".join(parts)


def _visit_every_mode(window) -> None:
    from PyQt5.QtWidgets import QApplication

    for label in settings.MODE_LABELS.values():
        window.mode_combo.setCurrentText(label)
        QApplication.processEvents()


def _unit_count(window) -> int:
    return sum(map(len, window.scenario.unit_manager.units))


def _logged(window, monkeypatch) -> list[str]:
    lines: list[str] = []
    real = window._log_status
    monkeypatch.setattr(window, "_log_status", lambda message, **kwargs: (lines.append(message), real(message, **kwargs)))
    return lines


def test_the_banner_names_what_was_read(window, units_unknown):
    assert not window.game_data_banner.isHidden()
    text = window.game_data_banner.text()
    assert "scenario version 1.60 is newer" in text
    assert "units Unknown; players and map settings shown; triggers Unknown" in text


def test_all_trusted_draws_the_units_and_says_so(window, all_trusted):
    assert all_trusted.settings_trusted and all_trusted.units_trusted
    text = window.game_data_banner.text()
    assert "scenario version 1.60 is newer" in text and "Terrain and elevation can be edited" in text
    assert _unit_count(window) > 0


def test_units_unknown_draws_none_and_the_stats_say_unknown(window, units_unknown):
    assert not units_unknown.units_trusted
    assert _unit_count(window) == 0
    assert window.player_stats_rows == [("Units", "", scenario_compat.UNKNOWN, "")]


def test_the_info_page_names_the_version_and_hides_the_tail(window, units_unknown):
    info = window._info_text()
    assert "Scenario version: 1.60 (compatibility mode, read as 1.58)" in info
    for row in ("Trigger tail", "XS script file", "Embedded XS"):
        assert f"{row}: {scenario_compat.UNKNOWN}" in info


def test_the_trigger_panel_and_map_analysis(window, units_unknown):
    window.mode_combo.setCurrentText("Triggers")
    assert window.trigger_panel.status.text() == "Triggers in scenario version 1.60 can't be read yet; they are kept as they are."
    assert not window.analysis_action.isEnabled()
    assert window.analysis_action.toolTip() == window.game_data_lock_reason()


def test_settings_unknown_shows_placeholders_and_no_donor_value(window, settings_unknown):
    assert not settings_unknown.settings_trusted
    assert "players and map settings Unknown" in window.game_data_banner.text()
    for mode, panel in (
        ("Map Options", window.map_options_panel),
        ("Players", window.players_panel),
        ("Diplomacy", window.diplomacy_panel),
        ("Messages", window.messages_panel),
    ):
        window.mode_combo.setCurrentText(mode)
        assert panel.status.text() == f"{scenario_compat.UNKNOWN}.", mode
    _visit_every_mode(window)
    text = _widget_text(window)
    assert DONOR_TRIBE not in text and DONOR_TEXT not in text


def test_the_settings_trusted_panels_show_the_files_values(window, units_unknown):
    """The control for the donor sweep: the marks are in the donor, not the file, so they never show."""
    window.mode_combo.setCurrentText("Players")
    assert window.players_panel.status.text() != f"{scenario_compat.UNKNOWN}."
    _visit_every_mode(window)
    text = _widget_text(window)
    assert DONOR_TRIBE not in text and DONOR_TEXT not in text


TERRAIN_TOOLS = (
    "draw_action", "spray_action", "fill_action", "draw_line_action", "draw_rect_action", "replace_brush_action",
    "clone_action", "eyedropper_action", "elevation_action", "set_level_action", "save_action", "save_as_action",
)  # fmt: skip
LOCKED_TOOLS = ("cliff_action", "mirror_action", "resize_action", "select_action", "eraser_action")
UNIT_TOOLS = ("place_unit_action", "move_action", "unit_line_action", "unit_shape_action", "unit_brush_action")


@pytest.mark.parametrize("case", ["units_unknown", "settings_unknown"])
def test_every_edit_is_refused(window, case, request, monkeypatch):
    from PyQt5.QtWidgets import QFileDialog, QMessageBox

    request.getfixturevalue(case)
    window.mode_combo.setCurrentText("Terrain")
    assert "once DEscape can read" in window.terrain_lock_reason()
    for action in (*TERRAIN_TOOLS, *LOCKED_TOOLS):
        assert not getattr(window, action).isEnabled(), action
    window.mode_combo.setCurrentText("Units")
    assert not window.place_unit_action.isEnabled()
    assert window._ensure_unit_edits() is None
    assert window._editable_player_fields() == frozenset()

    from descape import autosave

    writes = []
    set_viewer_global(monkeypatch, "write_scenario", lambda *args, **kwargs: writes.append(args))
    monkeypatch.setattr(scenario_compat, "write_compat", lambda *args, **kwargs: writes.append(args))
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: None)
    asked = []
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: asked.append(args) or ("", ""))
    lines = _logged(window, monkeypatch)
    window.save()
    window.save_as()
    assert not asked, "Save As refuses before asking for a path"
    monkeypatch.setattr(settings, "get_autosave_enabled", lambda: True)
    tiles = window.scenario.map_manager.terrain
    window.edit_history.apply("paint", tiles, lambda: setattr(tiles[0], "terrain_id", 15))
    window._autosave_tick()
    assert writes == []
    assert not list(autosave.autosave_dir().glob(f"*{autosave.AUTOSAVE_SUFFIX}"))
    assert sum(line.startswith("Save refused.") for line in lines) == 2

    window._region = (0, 0, 3, 3)
    window.copy_region()
    assert not window.copy_units(entries=[])
    assert sum(line.startswith("Copy refused.") for line in lines) == 2
    assert window._clipboard_history.entries == []


# -- B2: terrain writes when Settings and Units both read ----------------------


def _paint_one(window) -> int:
    """A real tile edit through the history; the new terrain id."""
    tiles = window.scenario.map_manager.terrain
    new = 15 if tiles[0].terrain_id != 15 else 2
    window.edit_history.apply("paint", tiles, lambda: setattr(tiles[0], "terrain_id", new))
    return new


def test_all_trusted_opens_the_terrain_tools_only(window, all_trusted):
    assert window.terrain_lock_reason() is None and window.game_data_lock_reason() is not None
    window.mode_combo.setCurrentText("Terrain")
    for action in TERRAIN_TOOLS:
        assert getattr(window, action).isEnabled(), action
    for action in LOCKED_TOOLS:
        assert not getattr(window, action).isEnabled(), action
    for action in (
        "convert_version_action", "disables_action", "swap_players_action", "convert_action",
        "wall_pick_action", "variant_next_action", "variant_prev_action", "variant_random_action",
    ):  # fmt: skip
        assert not getattr(window, action).isEnabled(), action
    window.draw_action.trigger()
    assert not window.paint_trees_param_action.isVisible() and not window.paint_eye_candy_param_action.isVisible()
    window.mode_combo.setCurrentText("Units")
    for action in UNIT_TOOLS:
        assert not getattr(window, action).isEnabled(), action
    assert window._ensure_unit_edits() is None


def test_the_clone_stamp_clones_terrain_only_and_plants_nothing(window, all_trusted):
    """GH #285: its Elevation box takes Paste Region's elevation_ok, which a compat
    document never has; Trees and Eye candy stay hidden as for Draw."""
    window.mode_combo.setCurrentText("Terrain")
    window.clone_action.trigger()
    assert window._current_tool == "clone" and window.scenario.map_is_square
    assert window.clone_terrain_check.isEnabled() and not window.clone_elevation_check.isEnabled()
    assert window.clone_elevation_check.isChecked() and not window._clone_elevation_ok()
    assert not window.paint_trees_param_action.isVisible() and not window.paint_eye_candy_param_action.isVisible()


def test_a_ticked_tree_box_plants_nothing_and_logs_no_refusal(window, all_trusted, monkeypatch):
    lines = _logged(window, monkeypatch)
    window.paint_trees_check.setChecked(True)
    window.paint_eye_candy_check.setChecked(True)
    tiles = window.scenario.map_manager.terrain
    window.edit_history.begin_stroke(tiles)
    for tile in tiles[:400]:
        tile.terrain_id = 10  # forest, where trees would plant
    record = window.edit_history.build_stroke_record("paint", tiles)
    window.edit_history.abort_stroke()
    assert record is not None
    units = _unit_count(window)
    assert window._apply_terrain_unit_plan(record, window.scenario.map_manager) == (record, None)
    assert _unit_count(window) == units
    assert lines == []


def test_save_writes_the_files_own_bytes_with_the_tile(window, all_trusted, tmp_path):
    path = window.scenario.path
    raw = path.read_bytes()
    new = _paint_one(window)
    window.save()
    assert not window.edit_history.is_dirty
    written = path.read_bytes()
    assert written[: all_trusted.body.start] == raw[: all_trusted.body.start] and written[:4] == b"1.60"
    reread = scenario_io.load_map_and_units_from_bytes(c4(written, "1.59"), "reread.aoe2scenario")
    assert reread.map_manager.terrain[0].terrain_id == new
    assert sum(map(len, reread.unit_manager.units)) == _unit_count(window) > 0


def test_save_in_the_game_profile_backs_up_with_backups_off(window, all_trusted, monkeypatch):
    path = window.scenario.path
    raw = path.read_bytes()
    monkeypatch.setattr(settings, "get_backups_enabled", lambda: False)
    set_viewer_global(monkeypatch, "is_in_game_profile", lambda _path: True)
    _paint_one(window)
    window.save()
    assert not window.edit_history.is_dirty and path.read_bytes() != raw
    backups = sorted(p for p in path.parent.iterdir() if p.suffix in (".bak", ".orig"))
    assert [p.suffix for p in backups] == [".bak", ".orig"]
    assert all(p.read_bytes() == raw for p in backups)


def test_the_open_logs_why_each_area_reads_or_not(window, monkeypatch, tmp_path):
    lines = _logged(window, monkeypatch)
    plan = _open(window, monkeypatch, tmp_path, c4(V159.read_bytes()), "1.58", BLANK_V158.read_bytes())
    logged = [line for line in lines if line.startswith("Compatibility mode (1.60 read as 1.58): ")]
    assert len(logged) == 1
    assert plan.reasons["units"] != "read" and f"units: {plan.reasons['units']}" in logged[0]
    assert "settings: read" in logged[0] and "triggers: " in logged[0]


def test_autosave_writes_a_slot_that_reopens_in_compatibility_mode(window, all_trusted, monkeypatch):
    from descape import autosave

    monkeypatch.setattr(settings, "get_autosave_enabled", lambda: True)
    new = _paint_one(window)
    window._autosave_tick()
    slots = sorted(autosave.autosave_dir().glob(f"*{autosave.AUTOSAVE_SUFFIX}"))
    assert len(slots) == 1 and window.edit_history.is_dirty
    window.edit_history.mark_saved()
    window.load_scenario(slots[0], untitled=True)
    assert window.scenario.compat is not None and window.scenario.compat.file_version == "1.60"
    assert window.scenario.map_manager.terrain[0].terrain_id == new


def test_a_refusal_shows_the_compat_paragraph_and_keeps_the_open_document(window, monkeypatch, tmp_path):
    from PyQt5.QtWidgets import QMessageBox

    window.load_scenario(scenario_io.BLANK_TEMPLATE_PATH)

    def corrupt(body, grid):
        body[grid.tiles_start + 1] = 200  # an elevation no map has
        return bytes(body)

    path = tmp_path / "broken.aoe2scenario"
    path.write_bytes(_rebuilt(c4(V159.read_bytes()), corrupt))
    shown = []
    monkeypatch.setattr(QMessageBox, "critical", lambda _parent, title, text: shown.append((title, text)))
    window.load_scenario(path)
    assert len(shown) == 1 and shown[0][0] == "Failed to load"
    assert "Compatibility mode could not read it either: the terrain could not be found" in shown[0][1]
    assert window.scenario.path == scenario_io.BLANK_TEMPLATE_PATH
    assert window.game_data_lock_reason() is None
