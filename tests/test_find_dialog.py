"""Edit > Find and Replace (GH #144): FindDialog and its ViewerWindow wiring.

The dialog is modeless, so nothing here calls exec_(). ViewerWindow._warn and
_confirm are stubbed in every test (a real box hangs an xdist worker), and
every window closes through conftest.close_window(), which marks the history
saved first.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

UNITS_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"

_REF_OAK = 100
_REF_PINE = 101
_REF_WALL = 102
_REF_HOUSE = 200
_REF_ARCHER_P1 = 201
_REF_VILLAGER_P1 = 203
_REF_ARCHER_P2 = 300
_REF_VILLAGER_P2 = 301
_BARRACKS = 12
_OAK = 349
_STONE_WALL = 117

_close = conftest.close_window


@pytest.fixture(autouse=True)
def _no_install_env(monkeypatch):
    from descape import asset_source

    monkeypatch.delenv("AOE2DE_INSTALL_PATH", raising=False)
    asset_source.clear_install_caches()


class _Boxes:
    """Records every _warn/_confirm call; `answer` is what _confirm returns."""

    def __init__(self):
        self.warned: list[tuple[str, str]] = []
        self.confirmed: list[tuple] = []
        self.answer: dict | None = {}

    def warn(self, title, text):
        self.warned.append((title, text))

    def confirm(self, title, text, choices=(), actions=()):
        self.confirmed.append((title, text, tuple(choices), tuple(actions)))
        return self.answer


def _window(style: str | None = None):
    from PyQt5.QtWidgets import QApplication

    window = conftest.shown_window(1200, 800)
    boxes = _Boxes()
    window._warn = boxes.warn
    window._confirm = boxes.confirm
    window.load_scenario(UNITS_FIXTURE)
    assert window.scenario is not None
    if style is not None:
        window.terrain_style_combo.setCurrentText(style)
    QApplication.processEvents()
    return window, boxes


def _dialog(window):
    window._show_find_dialog()
    return window._find_dialog


def _settle():
    from PyQt5.QtWidgets import QApplication

    QApplication.processEvents()
    QApplication.processEvents()


def _rows(dialog):
    return {row.reference_id: row for row in dialog.objects_tab.rows}


def _check_only(dialog, refs) -> None:
    tab = dialog.objects_tab
    tab._set_all_checked(False)
    from PyQt5.QtCore import Qt

    for i in range(tab.tree.topLevelItemCount()):
        item = tab.tree.topLevelItem(i)
        if item.data(0, Qt.UserRole).reference_id in refs:
            item.setCheckState(0, Qt.Checked)
    assert {key[1] for key in tab.checked} == set(refs)


def _unit(window, ref):
    return next(u for u in window.scenario.unit_manager.get_all_units() if u.reference_id == ref)


def _owner(window, ref) -> int:
    for player, units in enumerate(window.scenario.unit_manager.units):
        if any(u.reference_id == ref for u in units):
            return player
    raise AssertionError(f"{ref} not placed")


# -- wiring --------------------------------------------------------------------


def test_action_is_rebindable_unbound_and_enabled_without_building_the_unit_model() -> None:
    from descape import settings

    window, _boxes = _window()
    try:
        row = next(r for r in settings.REBINDABLE_ACTIONS if r[0] == "edit_find_replace")
        assert row[2] == ""
        assert window._keybind_actions["edit_find_replace"] is window.find_action
        assert window.find_action.shortcut().isEmpty()
        assert window.find_action.isEnabled()
        dialog = _dialog(window)
        assert not dialog.isModal()
        dialog.objects_tab.run_find()
        assert len(dialog.objects_tab.rows) == 8
        assert window.unit_edits is None, "opening Find or searching must not build the unit model"
    finally:
        _close(window)


def test_action_is_disabled_with_no_document() -> None:
    window = conftest.shown_window(1200, 800)
    try:
        assert not window.find_action.isEnabled()
        window._show_find_dialog()
        assert window._find_dialog is None
    finally:
        _close(window)


def test_closing_or_reopening_the_file_closes_the_dialog() -> None:
    window, _boxes = _window()
    try:
        _dialog(window)
        window.edit_history.mark_saved()
        window.load_scenario(UNITS_FIXTURE)
        assert window._find_dialog is None
        _dialog(window)
        window.close_scenario()
        assert window._find_dialog is None
    finally:
        _close(window)


def test_region_area_follows_the_selection() -> None:
    from descape.find_dialog import AREA_REGION

    window, _boxes = _window()
    try:
        dialog = _dialog(window)
        tab = dialog.objects_tab
        item = tab.area_combo.model().item(AREA_REGION)
        assert not item.isEnabled()
        window.on_region_selected((10, 10, 13, 13))
        assert item.isEnabled()
        tab.area_combo.setCurrentIndex(AREA_REGION)
        tab.run_find()
        assert sorted(_rows(dialog)) == [_REF_HOUSE, _REF_ARCHER_P1, _REF_VILLAGER_P1]
        window.on_region_selected(None)
        assert not item.isEnabled()
    finally:
        _close(window)


def test_invalid_regex_disables_find_inline() -> None:
    window, _boxes = _window()
    try:
        tab = _dialog(window).objects_tab
        tab.regex_box.setChecked(True)
        tab.name_edit.setText("(open")
        assert not tab.find_button.isEnabled()
        assert tab.pattern_error.isVisibleTo(tab)
        tab.name_edit.setText("arch")
        assert tab.find_button.isEnabled()
        tab.run_find()
        assert {r.unit_const for r in tab.rows} == {4}
    finally:
        _close(window)


# -- operations ----------------------------------------------------------------


def test_delete_is_one_undo_step_and_results_refresh_after_undo_from_terrain_mode() -> None:
    window, boxes = _window()
    try:
        dialog = _dialog(window)
        dialog.objects_tab.run_find()
        _check_only(dialog, {_REF_HOUSE, _REF_ARCHER_P2})
        before = window.edit_history.cursor
        dialog.objects_tab.delete_button.click()
        assert window.edit_history.cursor == before + 1
        assert "Delete 2 objects" in boxes.confirmed[-1][1]
        _settle()
        # The house's villager went with it.
        assert sorted(_rows(dialog)) == [_REF_OAK, _REF_PINE, _REF_WALL, _REF_ARCHER_P1, _REF_VILLAGER_P2]
        window.mode_combo.setCurrentText("Terrain")
        window.undo()
        _settle()
        assert len(dialog.objects_tab.rows) == 8
        window.redo()
        _settle()
        assert len(dialog.objects_tab.rows) == 5
    finally:
        _close(window)


def test_a_cancelled_confirm_changes_nothing() -> None:
    window, boxes = _window()
    try:
        dialog = _dialog(window)
        dialog.objects_tab.run_find()
        boxes.answer = None
        dialog.objects_tab.delete_button.click()
        assert window.edit_history.cursor == 0
        assert len(window.scenario.unit_manager.get_all_units()) == 8
    finally:
        _close(window)


def test_replace_is_one_undo_step_and_refused_rows_are_skipped_and_marked() -> None:
    from descape.find_dialog import _COL_NOTE

    window, boxes = _window()
    try:
        dialog = _dialog(window)
        tab = dialog.objects_tab
        tab.run_find()
        _check_only(dialog, {_REF_HOUSE, _REF_OAK, _REF_ARCHER_P2})
        tab.replace_edit.set_value(_BARRACKS)
        before = window.edit_history.cursor
        tab.replace_button.click()
        assert window.edit_history.cursor == before + 1
        assert "Replace 2 objects" in boxes.confirmed[-1][1]
        assert _unit(window, _REF_HOUSE).unit_const == 70, "a Barracks cannot hold the house's villager"
        assert _unit(window, _REF_OAK).unit_const == _BARRACKS
        assert _unit(window, _REF_ARCHER_P2).unit_const == _BARRACKS
        _settle()
        house_item = next(
            tab.tree.topLevelItem(i)
            for i in range(tab.tree.topLevelItemCount())
            if tab.tree.topLevelItem(i).text(1) == str(_REF_HOUSE)
        )
        assert "Garrison" in house_item.text(_COL_NOTE)
        assert "Replaced 2; 1 refused" in window.status_log.toPlainText()
        window.undo()
        assert _unit(window, _REF_OAK).unit_const == _OAK
        assert (_unit(window, _REF_OAK).x, _unit(window, _REF_OAK).y) == (5.5, 5.5)
    finally:
        _close(window)


def test_a_replaced_host_carries_its_garrison_to_the_new_anchor() -> None:
    window, _boxes = _window()
    try:
        dialog = _dialog(window)
        tab = dialog.objects_tab
        tab.run_find()
        _check_only(dialog, {_REF_HOUSE})
        tab.replace_edit.set_value(82)  # Castle: 4x4, so the 2x2 house's anchor moves
        tab.replace_button.click()
        house, villager = _unit(window, _REF_HOUSE), _unit(window, _REF_VILLAGER_P1)
        assert house.unit_const == 82
        assert (villager.x, villager.y) == (house.x, house.y)
        assert villager.garrisoned_in_id == _REF_HOUSE
    finally:
        _close(window)


def test_a_host_refused_only_for_its_dropped_occupants_type_is_replaced() -> None:
    """House + villager -> Watch Tower: the villager cannot sit in a tower,
    and once it drops the house can become one holding it (TASK-208 b)."""
    window, _boxes = _window()
    try:
        keys = [(1, _REF_HOUSE), (1, _REF_VILLAGER_P1)]
        refused = window._find_replace_preflight(keys, 79)
        assert refused == {(1, _REF_VILLAGER_P1): "Garrison: Watch Tower cannot go inside Watch Tower"}
        replaced, refused = window._find_replace_keys(keys, 79)
        assert replaced == 1 and set(refused) == {(1, _REF_VILLAGER_P1)}
        assert (_unit(window, _REF_HOUSE).unit_const, _unit(window, _REF_VILLAGER_P1).unit_const) == (79, 83)
        assert _unit(window, _REF_VILLAGER_P1).garrisoned_in_id == _REF_HOUSE
    finally:
        _close(window)


def test_change_owner_carries_occupants_and_refuses_an_orphan_occupant_row() -> None:
    window, _boxes = _window()
    try:
        dialog = _dialog(window)
        tab = dialog.objects_tab
        tab.run_find()
        _check_only(dialog, {_REF_VILLAGER_P1})
        tab.owner_combo.setCurrentIndex(3)
        tab.owner_apply.click()
        assert window.edit_history.cursor == 0, "an occupant without its host is refused"
        assert _owner(window, _REF_VILLAGER_P1) == 1
        assert "1 refused" in window.status_log.toPlainText()

        _check_only(dialog, {_REF_HOUSE})
        tab.owner_apply.click()
        assert window.edit_history.cursor == 1
        assert _owner(window, _REF_HOUSE) == 3
        assert _owner(window, _REF_VILLAGER_P1) == 3
        assert tab.checked == {(3, _REF_HOUSE)}, "the checked key follows the owner change"
        _settle()
        assert _rows(dialog)[_REF_HOUSE].player == 3
    finally:
        _close(window)


def test_select_on_map_reports_the_hidden_count() -> None:
    window, _boxes = _window()
    try:
        assert not window.show_garrisoned_action.isChecked()
        dialog = _dialog(window)
        dialog.objects_tab.run_find()
        _check_only(dialog, {_REF_ARCHER_P1, _REF_VILLAGER_P1})
        dialog.objects_tab.select_button.click()
        assert window.mode == "units"
        assert window._selection == [(1, _REF_ARCHER_P1)]
        assert "Selected 1 of 2; 1 hidden by filters / garrisoned / off-map" in window.status_log.toPlainText()
    finally:
        _close(window)


def test_double_click_centres_on_the_row(monkeypatch) -> None:
    window, _boxes = _window()
    try:
        dialog = _dialog(window)
        dialog.objects_tab.run_find()
        centred = []
        monkeypatch.setattr(window.map_view, "center_on_tile", lambda x, y: centred.append((x, y)))
        tab = dialog.objects_tab
        item = next(
            tab.tree.topLevelItem(i)
            for i in range(tab.tree.topLevelItemCount())
            if tab.tree.topLevelItem(i).text(1) == str(_REF_ARCHER_P2)
        )
        tab.tree.itemDoubleClicked.emit(item, 0)
        assert centred == [(20, 20)]
    finally:
        _close(window)


def test_export_csv_writes_checked_rows_and_refuses_compatdata(tmp_path) -> None:
    window, boxes = _window()
    try:
        dialog = _dialog(window)
        tab = dialog.objects_tab
        tab.run_find()
        _check_only(dialog, {_REF_OAK, _REF_ARCHER_P2})
        out = tmp_path / "out.csv"
        tab.ask_csv_path = lambda: str(out)
        tab.export_button.click()
        assert len(out.read_text(encoding="utf-8").splitlines()) == 3
        blocked = tmp_path / "compatdata" / "out.csv"
        blocked.parent.mkdir()
        tab.ask_csv_path = lambda: str(blocked)
        tab.export_button.click()
        assert not blocked.exists()
        assert boxes.warned and "compatdata" in boxes.warned[-1][1]
    finally:
        _close(window)


def test_a_mutating_op_is_refused_mid_stroke(monkeypatch) -> None:
    window, _boxes = _window()
    try:
        dialog = _dialog(window)
        dialog.objects_tab.run_find()
        monkeypatch.setattr(window, "_stroke_in_progress", lambda: True)
        dialog.objects_tab.delete_button.click()
        assert window.edit_history.cursor == 0
        assert len(window.scenario.unit_manager.get_all_units()) == 8
    finally:
        _close(window)


# -- render: a wall's neighbour after Replace, undo and redo ----------------------


@pytest.fixture
def wall_art(tmp_path, monkeypatch):
    """A synthetic install drawing the stone wall's five shapes in five colours,
    so a neighbour drawn with a stale shape is a pixel mismatch."""
    from descape import asset_source, unit_sprites

    from test_unit_sprites import build_sld

    graphics = tmp_path / unit_sprites.GRAPHICS_SUBPATH
    graphics.mkdir(parents=True)
    name = "t_wall_stone_x1"
    (graphics / f"{name}.sld").write_bytes(build_sld(5, canvas=unit_sprites.NATIVE_TILE_W, playercolor=False))
    table = {
        _STONE_WALL: {
            "graphic_id": 1,
            "file_name": name,
            "angle_count": 5,
            "mirroring_mode": 0,
            "frame_count": 1,
            # Else a GAIA wall's stored rotation reads as 0.0 (render.stored_rotation).
            "rotation_is_variant": True,
        }
    }
    monkeypatch.setattr(unit_sprites, "graphic_map", lambda: table)
    asset_source.set_install_path_override(tmp_path)
    unit_sprites.clear_caches()
    yield
    asset_source.set_install_path_override(None)
    unit_sprites.clear_caches()


def _fresh_canvas(window, style: str, sprites: bool = True) -> np.ndarray:
    from descape import render

    canvas_w, canvas_h = window._cache.canvas_dims(0)
    if style == "Stepped":
        full = render.render_terrain_iso_with_proj(window.scenario, with_units=True, with_sprites=sprites)[0]
    else:
        full = render.render_terrain_sloped_with_proj(window.scenario, with_units=True, with_sprites=sprites)[0]
    return full[:canvas_h, :canvas_w]


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_replacing_a_wall_beside_a_wall_repaints_like_a_fresh_render(style, wall_art) -> None:
    from descape import render

    window, _boxes = _window(style)
    try:
        # The fresh render's default filter shows the garrisoned villager.
        window.show_garrisoned_action.setChecked(True)
        assert window.show_sprites_action.isChecked()
        model = window._ensure_unit_edits()
        # A run at tiles 7..9: the middle wall is a straight run (index 0) only
        # while the fixture's wall at 7 stays a wall, then a tower (2).
        with window._unit_edit(model, "Add walls", [0]):
            model.add(0, _STONE_WALL, 8.5, 5.5, rotation=2.5132741928100586)
            model.add(0, _STONE_WALL, 9.5, 5.5, rotation=2.5132741928100586)
        assert render.wall_variant_rotation_overrides(window.scenario)[(0, 3)] == 0.0
        canvas_w, canvas_h = window._cache.canvas_dims(0)
        start = window._cache.render_rect(0, 0, canvas_w, canvas_h).copy()
        assert np.array_equal(start, _fresh_canvas(window, style))

        dialog = _dialog(window)
        tab = dialog.objects_tab
        tab.run_find()
        _check_only(dialog, {_REF_WALL})
        tab.replace_edit.set_value(_OAK)
        tab.replace_button.click()
        assert _unit(window, _REF_WALL).unit_const == _OAK
        assert render.wall_variant_rotation_overrides(window.scenario)[(0, 3)] == 2.0
        replaced =window._cache.render_rect(0, 0, canvas_w, canvas_h).copy()
        assert np.array_equal(replaced, _fresh_canvas(window, style)), "stale pixels after Replace"
        assert not np.array_equal(replaced, start)

        window._cache.render_rect(0, 0, canvas_w, canvas_h)
        window.undo()
        undone = window._cache.render_rect(0, 0, canvas_w, canvas_h).copy()
        assert np.array_equal(undone, _fresh_canvas(window, style)), "stale pixels after undo"
        assert np.array_equal(undone, start)

        window._cache.render_rect(0, 0, canvas_w, canvas_h)
        window.redo()
        redone = window._cache.render_rect(0, 0, canvas_w, canvas_h).copy()
        assert np.array_equal(redone, _fresh_canvas(window, style)), "stale pixels after redo"
        assert np.array_equal(redone, replaced)
    finally:
        _close(window)


# == Part B: the Triggers tab =========================================================

FIND_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "find_triggers_120x120.aoe2scenario"
_KNIGHT = 38
_ARCHER = 4
_ARCHER_A = 500


def _trigger_window():
    from PyQt5.QtWidgets import QApplication

    window = conftest.shown_window(1200, 800)
    boxes = _Boxes()
    window._warn = boxes.warn
    window._confirm = boxes.confirm
    window.load_scenario(FIND_FIXTURE)
    assert window.scenario is not None
    QApplication.processEvents()
    return window, boxes


def _ttab(window):
    dialog = _dialog(window)
    dialog.tabs.setCurrentWidget(dialog.triggers_tab)
    assert dialog.triggers_tab.initialized
    return dialog.triggers_tab


def _tfind(tab, text="", **kwargs):
    tab.text_edit.setText(text)
    for key, box in tab.scope_boxes.items():
        box.setChecked(key in kwargs.get("scopes", ("names", "descriptions", "messages")))
    tab.run_find()


def _check_children(tab, predicate) -> None:
    from PyQt5.QtCore import Qt

    tab._set_all_checked(False)
    for i in range(tab.tree.topLevelItemCount()):
        parent = tab.tree.topLevelItem(i)
        for j in range(parent.childCount()):
            hit = parent.child(j).data(0, Qt.UserRole)
            if predicate(hit):
                parent.child(j).setCheckState(0, Qt.Checked)


def _check_parents(tab, indices) -> None:
    from PyQt5.QtCore import Qt

    tab._set_all_checked(False)
    for i in range(tab.tree.topLevelItemCount()):
        parent = tab.tree.topLevelItem(i)
        if parent.data(0, Qt.UserRole)[1] in indices:
            parent.setCheckState(0, Qt.Checked)
    assert tab.checked_parents == set(indices)


def _triggers(window):
    from descape.scenario_io import parse_triggers

    return parse_triggers(window.scenario).triggers


def test_text_replace_is_one_step_declaring_only_the_changed_trigger() -> None:
    window, boxes = _trigger_window()
    try:
        tab = _ttab(window)
        _tfind(tab, "bridge")
        assert len(tab.hits) == 1
        tab.replace_text_edit.setText("gate")
        before = window.edit_history.cursor
        tab.replace_text_button.click()
        assert "Replace in 1 fields across 1 triggers" in boxes.confirmed[-1][1]
        assert window.edit_history.cursor == before + 1
        assert window.edit_history.records[-1].touched == [0]
        assert _triggers(window)[0].effects[0].message == "Hold the gate.\r\nArchers to the walls."
        _settle()
        assert tab.hits == [], "the re-run no longer finds the replaced text"
    finally:
        _close(window)


def test_type_replace_enable_disable_and_delete_are_one_step_each() -> None:
    window, _boxes = _trigger_window()
    try:
        tab = _ttab(window)
        tab.add_type_const(_ARCHER)
        _tfind(tab)
        _check_children(tab, lambda h: h.attribute == "object_list")
        tab.replace_type_edit.set_value(_KNIGHT)
        tab.replace_type_button.click()
        assert window.edit_history.cursor == 1
        assert window.edit_history.records[-1].touched == [2]
        assert _triggers(window)[2].conditions[0].object_list == _KNIGHT
        assert _triggers(window)[2].effects[0].object_list_unit_id == _ARCHER, "an unchecked hit is not written"

        tab.clear_types()
        _tfind(tab)
        _check_parents(tab, {0, 3})
        tab.disable_button.click()
        assert window.edit_history.cursor == 2
        assert window.edit_history.records[-1].touched == [0], "trigger 3 was already disabled"
        _settle()
        assert tab.checked_parents == {0, 3}, "the dialog's own enable/disable keeps its checks"
        tab.enable_button.click()
        assert window.edit_history.cursor == 3
        assert window.edit_history.records[-1].touched == [0, 3]

        _settle()
        _check_parents(tab, {1})
        tab.delete_button.click()
        assert window.edit_history.cursor == 4
        assert [t.name for t in _triggers(window)] == ["[Find] Messages", "Retarget", "Find: disabled"]
        _settle()
        assert tab.checked_parents == set(), "a delete clears the checks"
    finally:
        _close(window)


def test_an_object_replace_following_a_paired_effect_is_one_composite_step() -> None:
    from descape.edit_history import CompositeDiffRecord

    window, boxes = _trigger_window()
    try:
        dialog = _dialog(window)
        tab = dialog.objects_tab
        tab.run_find()
        _check_only(dialog, {_ARCHER_A})
        tab.replace_edit.set_value(_KNIGHT)
        boxes.answer = {"paired": True, "filter_only": False}
        tab.replace_button.click()
        _title, _text, choices, actions = boxes.confirmed[-1]
        assert [c[0] for c in choices] == ["paired", "filter_only"]
        assert {a[0] for a in actions} == {"Show skipped", "Show others"}
        assert window.edit_history.cursor == 1
        record = window.edit_history.records[-1]
        assert isinstance(record, CompositeDiffRecord)
        assert record.kinds() == {"unit", "trigger"}
        assert _unit(window, _ARCHER_A).unit_const == _KNIGHT
        effects = _triggers(window)[2].effects
        assert effects[0].object_list_unit_id == _KNIGHT
        assert effects[1].object_list_unit_id == _ARCHER, "tier (b) stays off by default"
        window.undo()
        assert _unit(window, _ARCHER_A).unit_const == _ARCHER
        assert _triggers(window)[2].effects[0].object_list_unit_id == _ARCHER
        window.redo()
        assert _unit(window, _ARCHER_A).unit_const == _KNIGHT
        assert _triggers(window)[2].effects[0].object_list_unit_id == _KNIGHT
    finally:
        _close(window)


def test_a_replace_with_no_trigger_model_omits_the_follow_up(monkeypatch) -> None:
    from descape.edit_history import UnitDiffRecord

    window, boxes = _trigger_window()
    try:
        monkeypatch.setattr(window, "_ensure_trigger_edits", lambda: None)
        dialog = _dialog(window)
        dialog.objects_tab.run_find()
        _check_only(dialog, {_ARCHER_A})
        dialog.objects_tab.replace_edit.set_value(_KNIGHT)
        dialog.objects_tab.replace_button.click()
        _title, text, choices, _actions = boxes.confirmed[-1]
        assert choices == () and "not followed" in text
        assert isinstance(window.edit_history.records[-1], UnitDiffRecord)
        assert _triggers(window)[2].effects[0].object_list_unit_id == _ARCHER
    finally:
        _close(window)


def test_jump_lands_on_the_entry_even_with_a_panel_filter() -> None:
    window, _boxes = _trigger_window()
    try:
        window.mode_combo.setCurrentText("Triggers")
        window.trigger_panel.filter_edit.setText("Messages")
        tab = _ttab(window)
        tab.set_object_refs([_ARCHER_A])
        _tfind(tab)
        window.mode_combo.setCurrentText("Units")
        parent = next(
            tab.tree.topLevelItem(i) for i in range(tab.tree.topLevelItemCount()) if tab.tree.topLevelItem(i).text(0) == "2"
        )
        target = next(parent.child(j) for j in range(parent.childCount()) if parent.child(j).text(1).startswith("Effect #3"))
        tab.tree.itemDoubleClicked.emit(target, 0)
        panel = window.trigger_panel
        assert window.mode == "triggers"
        assert panel.filter_edit.text() == ""
        assert panel.current_trigger_index() == 2
        assert ("effect", 3) in panel.selected_entry_refs()
    finally:
        _close(window)


def test_results_refresh_after_trigger_undo_from_units_mode_and_a_panel_delete() -> None:
    window, _boxes = _trigger_window()
    try:
        tab = _ttab(window)
        _tfind(tab, "bridge")
        tab.replace_text_edit.setText("gate")
        tab.replace_text_button.click()
        _settle()
        assert tab.hits == []
        window.mode_combo.setCurrentText("Units")
        window.undo()
        _settle()
        assert len(tab.hits) == 1 and "bridge" in tab.hits[0].value_display

        _tfind(tab)
        assert len({h.trigger_index for h in tab.hits}) == 4
        window.trigger_structural_edit("delete", [1])
        _settle()
        assert [h.trigger_name for h in tab.hits] == ["[Find] Messages", "Retarget", "Find: disabled"]
    finally:
        _close(window)


def test_a_history_jump_across_delete_and_new_clears_checks_and_stale_hits_are_refused() -> None:
    window, _boxes = _trigger_window()
    try:
        tab = _ttab(window)
        _tfind(tab)
        assert tab.checked_parents == {0, 1, 2, 3}
        window.trigger_structural_edit("delete", [3])
        window.trigger_structural_edit("new", [])
        _settle()
        assert tab.checked_parents == set(), "a trigger change the dialog didn't make clears every check"

        _tfind(tab, "bridge")
        hits = list(tab.hits)
        model = window._ensure_trigger_edits()
        with window._trigger_edit(model, "Retype", content_touched=[0]) as m:
            m.manager().triggers[0].effects[0].message = "Hold the bridge now.\r\nArchers to the walls."
        before = window.edit_history.cursor
        notes = window._find_text_replace(hits, "bridge", False, False, "gate")
        assert window.edit_history.cursor == before, "a stale hit is refused, never written"
        assert list(notes.values()) == ["changed since Find; re-run Find"]
        assert _triggers(window)[0].effects[0].message.startswith("Hold the bridge now.")
    finally:
        _close(window)


def test_objects_triggers_column_is_blank_until_triggers_parse_then_follows_edits() -> None:
    window, _boxes = _trigger_window()
    try:
        dialog = _dialog(window)
        dialog.objects_tab.run_find()
        assert all(r.trigger_refs is None for r in dialog.objects_tab.rows)
        assert window.scenario._trigger_manager is None, "the Objects tab must not parse triggers"
        dialog.tabs.setCurrentWidget(dialog.triggers_tab)
        assert _rows(dialog)[_ARCHER_A].trigger_refs == 5
        assert _rows(dialog)[502].trigger_refs == 0
        window.trigger_structural_edit("delete", [2])
        _settle()
        assert _rows(dialog)[_ARCHER_A].trigger_refs == 0
    finally:
        _close(window)


def test_a_half_typed_panel_name_is_committed_before_an_op_which_then_refuses_the_stale_row() -> None:
    """Pins what happens: the Find op commits the focused panel field first (as
    a focus-out would), so the typed text is never lost, and the trigger it
    renamed no longer matches its Find row, so the op refuses it."""
    window, _boxes = _trigger_window()
    try:
        window.mode_combo.setCurrentText("Triggers")
        panel = window.trigger_panel
        panel.select_trigger(2)
        editor = next(w for spec, _k, _i, w in panel._rows if spec.name == "name")
        editor.setFocus()
        editor.setText("Retargeted")
        assert window.focusWidget() is editor
        tab = _ttab(window)
        _tfind(tab, "Retarget", scopes=("names",))
        _check_parents(tab, {2})
        before = window.edit_history.cursor
        tab.disable_button.click()
        assert _triggers(window)[2].name == "Retargeted"
        assert window.edit_history.cursor == before + 1, "only the name commit was recorded"
        assert _triggers(window)[2].enabled
        assert "results changed since Find" in window.status_log.toPlainText()
    finally:
        _close(window)


def test_the_tab_is_disabled_when_triggers_do_not_parse(monkeypatch) -> None:
    from descape import viewer as viewer_module

    window, _boxes = _trigger_window()
    try:
        monkeypatch.setattr(viewer_module, "parse_triggers", lambda loaded: None)
        dialog = _dialog(window)
        dialog.tabs.setCurrentWidget(dialog.triggers_tab)
        tab = dialog.triggers_tab
        assert not tab.body.isEnabled()
        assert tab.unsupported.isVisibleTo(tab) and "can't be read" in tab.unsupported.text()
    finally:
        _close(window)


def test_a_read_only_trigger_file_finds_but_cannot_mutate() -> None:
    from descape.scenario_io import parse_triggers

    window, _boxes = _trigger_window()
    try:
        parse_triggers(window.scenario)
        window.scenario.trigger_write_supported = False
        tab = _ttab(window)
        _tfind(tab, "bridge")
        assert len(tab.hits) == 1
        for button in (tab.replace_text_button, tab.delete_button, tab.enable_button, tab.disable_button):
            assert not button.isEnabled()
            assert button.toolTip() == "Triggers are read-only for this file"
    finally:
        _close(window)


def test_save_reload_keeps_every_untouched_triggers_bytes(tmp_path) -> None:
    from descape.scenario_io import load_map_and_units, parse_triggers
    from descape.scenario_write import write_scenario
    from descape.trigger_model import TriggerEditModel

    window, _boxes = _trigger_window()
    try:
        original = load_map_and_units(FIND_FIXTURE)
        parse_triggers(original)
        original_blobs = TriggerEditModel(original)._blobs
        tab = _ttab(window)
        _tfind(tab, "bridge")
        tab.replace_text_edit.setText("gate")
        tab.replace_text_button.click()
        out = tmp_path / "replaced.aoe2scenario"
        write_scenario(window.scenario, out, **window._edit_model_kwargs())
        again = load_map_and_units(out)
        assert parse_triggers(again).triggers[0].effects[0].message.startswith("Hold the gate.")
        blobs = TriggerEditModel(again)._blobs
        assert blobs[1:] == original_blobs[1:]
        assert blobs[0] != original_blobs[0]
    finally:
        _close(window)
