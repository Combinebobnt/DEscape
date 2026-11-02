"""File > New Map: generates a blank tiles x tiles map in memory from the
shipped donor template (descape.scenario_new.blank_scenario_bytes) and loads
the result through the normal load path, marking the document untitled --
rather than building a scenario from scratch through AoE2ScenarioParser's own
(unverified, non-byte-stable) serializer. See scenario_new.py's docstring and
descape/viewer.py's new_map()/new_map_dialog().

Non-GUI tests pin the donor template asset's own shape (dimensions, version,
emptiness) and the write-guard that keeps Save As from ever overwriting a
shipped template -- both load-bearing for tests/test_write_path.py and
tests/test_lazy_viewport.py, which assume this same file's geometry. The byte
production itself -- reproducing real game exports at other sizes, bounds
checking, precondition guards -- is tests/test_scenario_new.py's job, not
this module's; GUI tests here create only a few of the seven standard sizes
to keep this file's own runtime down, on the understanding that the
generator underneath every size is already covered there.

GUI tests exercise the actual QAction/dialog wiring and the untitled-state
bookkeeping in descape/viewer.py (ViewerWindow._untitled, UNTITLED_PATH,
save_as() adoption). Every ViewerWindow() constructed here must end its test
with edit_history.mark_saved() before close() -- close() runs
_confirm_discard_changes(), which pops a modal QMessageBox on a dirty
document and would hang an offscreen run (see tests/test_lazy_viewport.py's
_check_exposed_rect_smaller() for the same pattern).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from descape.scenario_io import (
    BLANK_TEMPLATE_PATH,
    BLANK_TEMPLATE_SIZES,
    BLANK_TEMPLATE_TILES,
    TEMPLATE_DIR,
    blank_template_path,
    load_map_and_units,
)
from descape.scenario_new import LARGE_MAP_CONFIRM_TILES
from descape.scenario_write import WriteBlockedError, write_scenario

import conftest

_CORNER_TERRAIN = 2  # BEACH, distinct from a new map's terrain_id 0


@pytest.mark.parametrize("tiles", BLANK_TEMPLATE_SIZES)
def test_blank_template_is_shipped_and_loads(tiles: int) -> None:
    path = blank_template_path(tiles)
    assert path.is_file()
    s = load_map_and_units(path)
    mm = s.map_manager
    assert mm.map_width == mm.map_height == tiles
    assert s.map_is_square
    assert s.terrain_write_supported
    assert s.scenario_version == "1.59"
    assert all(t.terrain_id == 0 and t.elevation == 0 for t in mm.terrain)
    assert sum(len(units) for units in s.unit_manager.units) == 0


def test_write_scenario_refuses_to_write_into_template_dir() -> None:
    s = load_map_and_units(BLANK_TEMPLATE_PATH)
    original_bytes = BLANK_TEMPLATE_PATH.read_bytes()
    with pytest.raises(WriteBlockedError):
        write_scenario(s, BLANK_TEMPLATE_PATH)
    # A future sibling template must be covered by the same guard, without
    # needing to exist on disk first -- the check is on the destination's
    # parent directory, not on colliding with an existing file. Deliberately
    # not one of BLANK_TEMPLATE_SIZES (which now really are shipped files) --
    # this is testing the guard's blanket directory coverage, not colliding
    # with a real one.
    with pytest.raises(WriteBlockedError):
        write_scenario(s, TEMPLATE_DIR / "blank_960x960.aoe2scenario")
    assert BLANK_TEMPLATE_PATH.read_bytes() == original_bytes


def test_write_scenario_still_writes_elsewhere(tmp_path) -> None:
    s = load_map_and_units(BLANK_TEMPLATE_PATH)
    dest = tmp_path / "copy.aoe2scenario"
    write_scenario(s, dest)
    assert dest.is_file()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_new_map_action_creates_untitled_document() -> None:
    """file_new_default's action: a 120x120 Grass 1 map at elevation 0 with
    no dialog (NewMapDialog.exec_in unpatched, so opening it would block)."""
    conftest.ensure_qapp()
    from PyQt5.QtGui import QKeySequence

    from descape.viewer import UNTITLED_NAME, ViewerWindow

    window = ViewerWindow()
    try:
        # Trigger the QAction itself, not window.new_map() directly, so the
        # wiring is covered too.
        window.new_action.trigger()

        assert window.scenario is not None
        assert window._untitled is True
        assert window.scenario.path.name == UNTITLED_NAME
        assert window.scenario.path != BLANK_TEMPLATE_PATH
        assert window.windowTitle() == f"{UNTITLED_NAME} — DEscape"
        assert window.edit_history.is_dirty is False
        assert window.save_as_action.isEnabled()
        assert window.close_action.isEnabled()

        mm = window.scenario.map_manager
        assert mm.map_width == mm.map_height == BLANK_TEMPLATE_TILES
        assert all(t.terrain_id == 0 and t.elevation == 0 for t in mm.terrain)

        assert window.new_action.isEnabled()
        assert window.new_action.shortcut().isEmpty()
        assert window.new_action in window.actions()  # its keybind fires with no menu entry
        assert window.new_map_action.shortcut() == QKeySequence("Ctrl+N")
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_file_menu_has_new_map_dialog_action_and_no_size_submenu() -> None:
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    try:
        file_menu = window.menuBar().actions()[0].menu()
        actions = file_menu.actions()
        assert actions[0] is window.new_map_action
        assert window.new_map_action.text() == "&New Map…"
        assert window.new_action not in actions
        submenus = [a.text() for a in actions if a.menu() is not None]
        assert "&New Map" not in submenus
        assert window._keybind_actions["file_new"] is window.new_map_action
        assert window._keybind_actions["file_new_default"] is window.new_action
    finally:
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_default_action_ignores_the_stored_dialog_choice() -> None:
    """file_new_default stays 120x120 Grass 1 at 0 whatever New Map… last saved,
    and plants nothing: Grass 1 carries eye candy, so the saved boxes would show."""
    conftest.ensure_qapp()
    from descape import game_dataset, settings
    from descape.viewer import ViewerWindow

    settings.set_new_map_choice(
        {"tiles": 144, "game_data": "ror", "terrain_id": _CORNER_TERRAIN, "elevation": 5, "trees": True, "eye_candy": True}
    )
    window = ViewerWindow()
    try:
        window.new_action.trigger()
        mm = window.scenario.map_manager
        assert mm.map_width == mm.map_height == BLANK_TEMPLATE_TILES
        assert all(t.terrain_id == 0 and t.elevation == 0 for t in mm.terrain)
        assert game_dataset.from_variant(window.scenario.game_variant) == game_dataset.AOE2
        assert sum(len(units) for units in window.scenario.unit_manager.units) == 0
    finally:
        window.edit_history.mark_saved()
        window.close()


def _patch_new_map_dialog(
    monkeypatch, *, tiles=None, terrain_id=None, elevation=None, trees=None, eye_candy=None, accept=True, seen=None
):
    """NewMapDialog.exec_in without its event loop: records the dialog's
    initial choice into `seen`, sets the given widgets, then accepts or
    rejects. The size goes through the dialog's own Size/Custom logic."""
    from PyQt5.QtWidgets import QDialog

    from descape.new_map_dialog import NewMapDialog

    def fake_exec(dialog, window):
        if seen is not None:
            seen.append(dialog.current_choice())
        if not accept:
            dialog.reject()
            return QDialog.Rejected
        if tiles is not None:
            dialog._set_size(tiles)
        if terrain_id is not None:
            index = dialog.terrain_combo.findData(terrain_id)
            assert index >= 0, terrain_id
            dialog.terrain_combo.setCurrentIndex(index)
        if elevation is not None:
            dialog.elevation_spin.setValue(elevation)
        if trees is not None:
            dialog.trees_check.setChecked(trees)
        if eye_candy is not None:
            dialog.eye_candy_check.setChecked(eye_candy)
        dialog.accept()
        return QDialog.Accepted

    monkeypatch.setattr(NewMapDialog, "exec_in", fake_exec)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_new_map_dialog_creates_the_choice_and_remembers_it(monkeypatch) -> None:
    """Ctrl+N's action through a patched dialog: the map gets the picked
    size, terrain and elevation on every tile, the choice is saved to
    config.yaml, and the next open starts from it."""
    conftest.ensure_qapp()
    from descape import settings
    from descape.new_map_dialog import NewMapChoice
    from descape.viewer import ViewerWindow

    seen: list = []
    _patch_new_map_dialog(monkeypatch, tiles=144, terrain_id=_CORNER_TERRAIN, elevation=3, seen=seen)

    window = ViewerWindow()
    try:
        window.new_map_action.trigger()
        mm = window.scenario.map_manager
        assert mm.map_width == mm.map_height == 144
        assert all(t.terrain_id == _CORNER_TERRAIN and t.elevation == 3 for t in mm.terrain)
        assert window._untitled is True
        assert seen == [NewMapChoice()]

        expected = {
            "tiles": 144,
            "game_data": "aoe2",
            "terrain_id": _CORNER_TERRAIN,
            "elevation": 3,
            "trees": settings.NEW_MAP_DEFAULT_TREES,
            "eye_candy": settings.NEW_MAP_DEFAULT_EYE_CANDY,
        }
        assert settings.get_new_map_choice() == expected
        monkeypatch.setattr(settings, "_new_map", None)  # re-read from disk
        assert settings.get_new_map_choice() == expected

        window.edit_history.mark_saved()
        _patch_new_map_dialog(monkeypatch, accept=False, seen=seen)
        window.new_map_action.trigger()
        assert seen[-1] == NewMapChoice.from_dict(expected)
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_the_panel_is_built_once_and_rebuilt_for_other_game_data(monkeypatch) -> None:
    """Reopening reuses the panel (its swatches stay decoded); a game data
    switch, whose art the swatches come from, builds a new one."""
    conftest.ensure_qapp()
    from PyQt5.QtWidgets import QDialog

    from descape import game_dataset
    from descape.new_map_dialog import NewMapDialog
    from descape.viewer import ViewerWindow

    panels = []
    monkeypatch.setattr(NewMapDialog, "exec_in", lambda dialog, window: panels.append(dialog) or QDialog.Rejected)

    window = ViewerWindow()
    try:
        window.new_map_action.trigger()
        window.new_map_action.trigger()
        assert len(panels) == 2 and panels[0] is panels[1]
        game_dataset.set_active(game_dataset.ROR)
        window.new_map_action.trigger()
        assert panels[2] is not panels[0]
        assert panels[2].built_for[0] == game_dataset.ROR
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_new_map_while_the_panel_is_open_does_nothing(monkeypatch) -> None:
    conftest.ensure_qapp()
    from PyQt5.QtWidgets import QDialog

    from descape.new_map_dialog import NewMapDialog
    from descape.viewer import ViewerWindow

    opened = []

    def reenter(dialog, window):
        opened.append(dialog)
        dialog.show()  # as exec_in leaves it while running
        window.new_map_dialog()
        dialog.hide()
        return QDialog.Rejected

    monkeypatch.setattr(NewMapDialog, "exec_in", reenter)
    window = ViewerWindow()
    try:
        window.new_map_action.trigger()
        assert len(opened) == 1
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_preset_480_does_not_confirm(monkeypatch) -> None:
    """Pins the deliberate asymmetry in new_map_dialog()'s docstring: the
    labelled 480 preset never pops the large-map confirmation for its size,
    only a typed Custom size above LARGE_MAP_CONFIRM_TILES does (a Ludicrous
    forest with Trees ticked asks about its trees instead; Grass 1 here has
    none). Fails loudly (instead of hanging on an unclickable modal) if a
    future 'consistency fix' routes presets through the same confirm."""
    conftest.ensure_qapp()
    import descape.viewer as viewer_module
    from descape.viewer import ViewerWindow

    def fail_if_called(*args, **kwargs):
        raise AssertionError("QMessageBox.question must not be called for the 480 preset")

    monkeypatch.setattr(viewer_module.QMessageBox, "question", staticmethod(fail_if_called))
    _patch_new_map_dialog(monkeypatch, tiles=480)

    window = ViewerWindow()
    try:
        window.new_map_action.trigger()
        mm = window.scenario.map_manager
        assert mm.map_width == mm.map_height == 480
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_custom_size_creates_that_size(monkeypatch) -> None:
    """170 <= LARGE_MAP_CONFIRM_TILES, so this needs no QMessageBox patch at
    all -- if new_map_dialog() ever popped a confirm here, the unpatched
    QMessageBox.question would hang the offscreen run rather than silently
    pass, which is the point."""
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    _patch_new_map_dialog(monkeypatch, tiles=170)

    window = ViewerWindow()
    try:
        window.new_map_action.trigger()
        mm = window.scenario.map_manager
        assert mm.map_width == mm.map_height == 170
        assert window._untitled is True
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_new_map_dialog_on_dirty_document_confirms_discard_exactly_once(monkeypatch) -> None:
    """new_map_dialog() checks _confirm_discard_changes() itself and then
    calls ViewerWindow._create_new_map() directly, not new_map() -- if it
    called new_map() instead, a dirty document would trigger the discard
    prompt twice, after the user already answered the dialog."""
    conftest.ensure_qapp()
    from PyQt5.QtWidgets import QMessageBox

    import descape.viewer as viewer_module
    from descape.viewer import ViewerWindow

    _patch_new_map_dialog(monkeypatch, tiles=168)

    window = ViewerWindow()
    try:
        window.new_map()
        # Force is_dirty with no real edit: a lost save point reads dirty (a bare cursor bump no longer does, GH #108).
        window.edit_history.saved_at_cursor = None
        assert window.edit_history.is_dirty

        discard_calls = []

        def fake_discard_prompt(*args, **kwargs):
            discard_calls.append(args)
            return QMessageBox.Discard

        monkeypatch.setattr(viewer_module.QMessageBox, "question", staticmethod(fake_discard_prompt))
        window.new_map_dialog()

        assert len(discard_calls) == 1
        mm = window.scenario.map_manager
        assert mm.map_width == mm.map_height == 168
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_new_map_dialog_cancelled_leaves_document_and_settings_untouched(monkeypatch) -> None:
    conftest.ensure_qapp()
    from descape import settings
    from descape.viewer import ViewerWindow

    before = settings.get_new_map_choice()
    _patch_new_map_dialog(monkeypatch, accept=False)

    window = ViewerWindow()
    try:
        assert window.scenario is None
        window.new_map_action.trigger()
        assert window.scenario is None
        assert settings.get_new_map_choice() == before
        assert "new_map" not in settings._load_config()
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_custom_size_above_threshold_confirms_before_creating(monkeypatch) -> None:
    conftest.ensure_qapp()
    from PyQt5.QtWidgets import QMessageBox

    import descape.viewer as viewer_module
    from descape.viewer import ViewerWindow

    tiles = LARGE_MAP_CONFIRM_TILES + 60
    _patch_new_map_dialog(monkeypatch, tiles=tiles)

    window = ViewerWindow()
    try:
        calls = []

        def cancel(*args, **kwargs):
            calls.append(args)
            return QMessageBox.Cancel

        monkeypatch.setattr(viewer_module.QMessageBox, "question", staticmethod(cancel))
        window.new_map_action.trigger()
        assert len(calls) == 1
        assert window.scenario is None  # cancelled -- nothing created

        monkeypatch.setattr(
            viewer_module.QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes)
        )
        window.new_map_action.trigger()
        mm = window.scenario.map_manager
        assert mm.map_width == mm.map_height == tiles
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_custom_set_to_a_preset_size_does_not_confirm(monkeypatch) -> None:
    """Custom 480 is the Ludicrous size, so it gets the preset's no-prompt
    treatment, and the saved 480 reopens on the Ludicrous row."""
    conftest.ensure_qapp()
    import descape.viewer as viewer_module
    from descape.viewer import ViewerWindow

    def fail_if_called(*args, **kwargs):
        raise AssertionError("QMessageBox.question must not be called for a preset size")

    def custom_480(dialog, window):
        dialog.size_combo.setCurrentIndex(dialog.size_combo.count() - 1)
        dialog.custom_spin.setValue(480)
        assert dialog.is_custom
        dialog.accept()
        return viewer_module.QDialog.Accepted

    from descape.new_map_dialog import NewMapDialog

    monkeypatch.setattr(viewer_module.QMessageBox, "question", staticmethod(fail_if_called))
    monkeypatch.setattr(NewMapDialog, "exec_in", custom_480)

    window = ViewerWindow()
    try:
        window.new_map_action.trigger()
        mm = window.scenario.map_manager
        assert mm.map_width == mm.map_height == 480
        reopened: list = []
        window.edit_history.mark_saved()
        monkeypatch.setattr(
            NewMapDialog, "exec_in", lambda dialog, window: reopened.append(dialog.is_custom) or viewer_module.QDialog.Rejected
        )
        window.new_map_action.trigger()
        assert reopened == [False]
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_choice_is_saved_even_when_the_large_map_question_is_cancelled(monkeypatch) -> None:
    conftest.ensure_qapp()
    from PyQt5.QtWidgets import QMessageBox

    import descape.viewer as viewer_module
    from descape import settings
    from descape.viewer import ViewerWindow

    _patch_new_map_dialog(monkeypatch, tiles=300, elevation=4)
    monkeypatch.setattr(viewer_module.QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Cancel))

    window = ViewerWindow()
    try:
        window.new_map_action.trigger()
        assert window.scenario is None
        assert settings.get_new_map_choice()["tiles"] == 300
        assert settings.get_new_map_choice()["elevation"] == 4
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_create_new_map_reports_an_out_of_range_fill(monkeypatch) -> None:
    conftest.ensure_qapp()
    import descape.viewer as viewer_module
    from descape.viewer import ViewerWindow

    errors = []
    monkeypatch.setattr(viewer_module.QMessageBox, "critical", staticmethod(lambda *a, **k: errors.append(a)))

    window = ViewerWindow()
    try:
        window._create_new_map(144, terrain_id=300)
        assert window.scenario is None
        assert len(errors) == 1
        assert "terrain id 300" in errors[0][2]
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_return_of_rome_choice_creates_a_ror_map_and_switches_back(monkeypatch) -> None:
    """A Return of Rome choice builds from the RoR donor, so the new map's
    header says RoR and the load switches the active dataset to it; a later
    AoE2 choice switches back."""
    conftest.ensure_qapp()
    from descape import game_dataset
    from descape.new_map_dialog import NewMapDialog
    from descape.viewer import ViewerWindow

    def pick(game_data, tiles):
        def fake_exec(dialog, window):
            dialog.game_data_combo.setCurrentIndex(dialog.game_data_combo.findData(game_data))
            dialog._set_size(tiles)
            dialog.accept()
            return dialog.Accepted

        monkeypatch.setattr(NewMapDialog, "exec_in", fake_exec)

    window = ViewerWindow()
    try:
        pick(game_dataset.ROR, 144)
        window.new_map_action.trigger()
        mm = window.scenario.map_manager
        assert mm.map_width == mm.map_height == 144
        assert game_dataset.from_variant(window.scenario.game_variant) == game_dataset.ROR
        assert game_dataset.active() == game_dataset.ROR
        assert "[Return of Rome]" in window.windowTitle()
        assert window.game_data_lock_reason() is None
        assert sum(len(units) for units in window.scenario.unit_manager.units) == 0

        window.edit_history.mark_saved()
        pick(game_dataset.AOE2, 120)
        window.new_map_action.trigger()
        assert game_dataset.from_variant(window.scenario.game_variant) == game_dataset.AOE2
        assert game_dataset.active() == game_dataset.AOE2
        assert "[Return of Rome]" not in window.windowTitle()
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_a_new_return_of_rome_map_saves_and_reloads(tmp_path, monkeypatch) -> None:
    """A RoR map from New Map… goes through Save As and reloads as a RoR 1.59 file of the picked size, terrain and height."""
    conftest.ensure_qapp()
    import descape.viewer as viewer_module
    from descape import game_dataset
    from descape.new_map_dialog import NewMapDialog
    from descape.viewer import ViewerWindow

    def fake_exec(dialog, window):
        dialog.game_data_combo.setCurrentIndex(dialog.game_data_combo.findData(game_dataset.ROR))
        dialog._set_size(144)
        dialog.terrain_combo.setCurrentIndex(dialog.terrain_combo.findData(_CORNER_TERRAIN))
        dialog.elevation_spin.setValue(4)
        dialog.accept()
        return dialog.Accepted

    monkeypatch.setattr(NewMapDialog, "exec_in", fake_exec)
    dest = tmp_path / "ror_new.aoe2scenario"
    monkeypatch.setattr(
        viewer_module.QFileDialog, "getSaveFileName", staticmethod(lambda *args, **kwargs: (str(dest), ""))
    )
    window = ViewerWindow()
    try:
        window.new_map_action.trigger()
        window.save_as()
        reloaded = load_map_and_units(dest)
        assert game_dataset.from_variant(reloaded.game_variant) == game_dataset.ROR
        assert reloaded.scenario_version == "1.59"
        mm = reloaded.map_manager
        assert mm.map_width == mm.map_height == 144
        assert all(t.terrain_id == _CORNER_TERRAIN and t.elevation == 4 for t in mm.terrain)
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_new_map_does_not_touch_disk() -> None:
    """Insurance for the in-memory generation decision (descape.scenario_io.
    load_map_and_units_from_bytes) -- File > New Map must never write to
    descape/templates/, since nothing in the design needs it to."""
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    before = {p: p.read_bytes() for p in TEMPLATE_DIR.glob("*.aoe2scenario")}
    window = ViewerWindow()
    try:
        window.new_map(240)
        after = {p: p.read_bytes() for p in TEMPLATE_DIR.glob("*.aoe2scenario")}
        assert after == before
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_save_as_default_never_points_at_the_template() -> None:
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    try:
        window.new_map()
        start_path = window._save_as_start_path()
        assert Path(start_path).is_absolute()
        assert Path(start_path).parent.resolve() != TEMPLATE_DIR.resolve()
        assert "blank_" not in Path(start_path).name
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
@pytest.mark.parametrize("tiles", (BLANK_TEMPLATE_TILES, 168, 220))
def test_save_as_on_new_map_adopts_path(tmp_path, monkeypatch, tiles: int) -> None:
    """168, alongside the donor's own 120, proves Save As works end-to-end on
    a *generated* (not shipped-file) size too. 220 is GH #94's size: its far
    corner is painted, and must read back at the same index."""
    conftest.ensure_qapp()
    import descape.viewer as viewer_module
    from descape.viewer import ViewerWindow

    dest = tmp_path / "mymap.aoe2scenario"
    monkeypatch.setattr(
        viewer_module.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *args, **kwargs: (str(dest), "")),
    )

    window = ViewerWindow()
    try:
        window.new_map(tiles)
        window.mode_combo.setCurrentText("Terrain")
        window._on_tool_selected("draw")
        window.terrain_panel.set_terrain(_CORNER_TERRAIN)
        window.paint_trees_check.setChecked(False)
        window.paint_eye_candy_check.setChecked(False)
        window.brush_size_spin.setValue(1)
        window.on_edit_stroke_start()
        window.on_edit_stroke_tile(tiles - 1, tiles - 1, 0)
        window.on_edit_stroke_end()
        window.save_as()

        assert dest.is_file()
        reloaded = load_map_and_units(dest)
        mm = reloaded.map_manager
        assert mm.map_width == mm.map_height == tiles
        assert mm.terrain[tiles * tiles - 1].terrain_id == _CORNER_TERRAIN
        assert mm.terrain[tiles * tiles - 2].terrain_id == 0

        assert window._untitled is False
        assert window.scenario.path == dest
        assert window.windowTitle() == f"{dest.name} — DEscape"
        assert BLANK_TEMPLATE_PATH.read_bytes()  # donor template itself untouched (still readable/non-empty)
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_open_after_new_clears_untitled() -> None:
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    try:
        window.new_map()
        assert window._untitled is True

        window.load_scenario(BLANK_TEMPLATE_PATH)  # a normal open, no untitled=

        assert window._untitled is False
        assert window.scenario.path == BLANK_TEMPLATE_PATH
        assert window.windowTitle() == f"{BLANK_TEMPLATE_PATH.name} — DEscape"
    finally:
        window.edit_history.mark_saved()
        window.close()


# --- Trees and Eye candy (GH #243 follow-up) ---

_GRASS_1, _FOREST_OAK = 0, 10


def _unit_counts(window) -> tuple[int, int]:
    units = window.scenario.unit_manager.units
    return len(units[0]), sum(len(player_units) for player_units in units[1:])


def _no_question(monkeypatch, *, ask=True) -> None:
    """Fails on the plain large-map question, and on _ask_new_map_confirm unless `ask`."""
    import descape.viewer as viewer_module

    def fail_question(*args, **kwargs):
        raise AssertionError("QMessageBox.question must not be called here")

    monkeypatch.setattr(viewer_module.QMessageBox, "question", staticmethod(fail_question))
    if not ask:

        def fail_ask(self, *args, **kwargs):
            raise AssertionError("_ask_new_map_confirm must not be called here")

        monkeypatch.setattr(viewer_module.ViewerWindow, "_ask_new_map_confirm", fail_ask)


def _stub_ask(monkeypatch, answer: str) -> list:
    from descape.viewer import ViewerWindow

    asked: list = []

    def ask(self, text, **kwargs):
        asked.append((text, kwargs))
        return answer

    monkeypatch.setattr(ViewerWindow, "_ask_new_map_confirm", ask)
    return asked


def _spy_create(monkeypatch) -> list:
    """Records _create_new_map's arguments instead of generating."""
    from descape.viewer import ViewerWindow

    created: list = []
    monkeypatch.setattr(ViewerWindow, "_create_new_map", lambda self, tiles, **kwargs: created.append((tiles, kwargs)))
    return created


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_a_forest_choice_opens_with_its_trees_unmodified_and_nothing_to_undo(monkeypatch) -> None:
    conftest.ensure_qapp()
    from PyQt5.QtWidgets import QApplication

    from descape import terrain_units
    from descape.viewer import ViewerWindow

    _no_question(monkeypatch, ask=False)
    _patch_new_map_dialog(monkeypatch, tiles=120, terrain_id=_FOREST_OAK, trees=True, eye_candy=False)
    window = ViewerWindow()
    try:
        window.new_map_action.trigger()
        assert _unit_counts(window) == (120 * 120, 0)
        assert {u.unit_const for u in window.scenario.unit_manager.units[0]} <= terrain_units.TREE_CONSTS
        assert window._untitled is True
        assert window.edit_history.is_dirty is False
        assert window.edit_history.can_undo is False
        assert not window.undo_action.isEnabled()
        assert window.isEnabled()
        assert QApplication.overrideCursor() is None
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_trees_on_a_terrain_with_none_plant_nothing(monkeypatch) -> None:
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    _no_question(monkeypatch, ask=False)
    _patch_new_map_dialog(monkeypatch, tiles=120, terrain_id=_GRASS_1, trees=True, eye_candy=False)
    window = ViewerWindow()
    try:
        window.new_map_action.trigger()
        assert _unit_counts(window) == (0, 0)
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_a_ludicrous_forest_asks_and_cancel_creates_nothing(monkeypatch) -> None:
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    _no_question(monkeypatch)
    asked = _stub_ask(monkeypatch, "cancel")
    _patch_new_map_dialog(monkeypatch, tiles=480, terrain_id=_FOREST_OAK, trees=True)
    window = ViewerWindow()
    try:
        window.new_map_action.trigger()
        assert window.scenario is None
        ((text, kwargs),) = asked
        assert "230,400 trees" in text and "1 GB" in text
        assert "larger than" not in text  # 480 is a preset: no size reason
        assert kwargs == {"bare_label": "Create without trees"}
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_create_passes_the_kinds_the_terrain_can_place_on(monkeypatch) -> None:
    """Forest Oak has no eye candy, so a ticked Eye candy box plants nothing there."""
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    _no_question(monkeypatch)
    asked = _stub_ask(monkeypatch, "create")
    created = _spy_create(monkeypatch)
    _patch_new_map_dialog(monkeypatch, tiles=480, terrain_id=_FOREST_OAK, trees=True, eye_candy=True)
    window = ViewerWindow()
    try:
        window.new_map_action.trigger()
        assert len(asked) == 1
        assert created == [(480, {"terrain_id": _FOREST_OAK, "elevation": 0, "game_data": "aoe2", "trees": True, "doodads": False})]
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_bare_creates_a_bare_map_and_keeps_the_saved_boxes(monkeypatch) -> None:
    conftest.ensure_qapp()
    from descape import settings
    from descape.viewer import ViewerWindow

    _no_question(monkeypatch)
    asked = _stub_ask(monkeypatch, "bare")
    _patch_new_map_dialog(monkeypatch, tiles=480, terrain_id=_FOREST_OAK, trees=True, eye_candy=True)
    window = ViewerWindow()
    try:
        window.new_map_action.trigger()
        assert len(asked) == 1
        assert window.scenario.map_manager.map_width == 480
        assert all(t.terrain_id == _FOREST_OAK for t in window.scenario.map_manager.terrain)
        assert _unit_counts(window) == (0, 0)
        saved = settings.get_new_map_choice()
        assert (saved["trees"], saved["eye_candy"]) == (True, True)
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_a_ludicrous_forest_with_trees_off_asks_nothing(monkeypatch) -> None:
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    _no_question(monkeypatch, ask=False)
    created = _spy_create(monkeypatch)
    _patch_new_map_dialog(monkeypatch, tiles=480, terrain_id=_FOREST_OAK, trees=False, eye_candy=True)
    window = ViewerWindow()
    try:
        window.new_map_action.trigger()
        assert created == [(480, {"terrain_id": _FOREST_OAK, "elevation": 0, "game_data": "aoe2", "trees": False, "doodads": False})]
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
@pytest.mark.parametrize(
    ("tiles", "terrain_id", "trees", "eye_candy", "both", "reasons", "bare_label"),
    [
        (400, _FOREST_OAK, True, False, False, ("larger than", "160,000 trees"), "Create without trees"),
        (300, _FOREST_OAK, True, False, False, ("larger than",), "Create without trees"),
        (300, _GRASS_1, True, True, False, ("larger than",), "Create without eye candy"),
        (300, _FOREST_OAK, True, True, True, ("larger than",), "Create without trees or eye candy"),
    ],
)
def test_a_large_custom_size_that_plants_asks_once_naming_each_reason(
    monkeypatch, tiles, terrain_id, trees, eye_candy, both, reasons, bare_label
) -> None:
    """The size question and the trees question merge into one, never two.
    No terrain in either game's tables plants both kinds, so `both` gives Forest Oak eye candy too."""
    conftest.ensure_qapp()
    from descape import terrain_units
    from descape.viewer import ViewerWindow

    if both:
        real = terrain_units.is_doodad_terrain
        monkeypatch.setattr(terrain_units, "is_doodad_terrain", lambda tid: tid == terrain_id or real(tid))
    _no_question(monkeypatch)
    asked = _stub_ask(monkeypatch, "cancel")
    _patch_new_map_dialog(monkeypatch, tiles=tiles, terrain_id=terrain_id, trees=trees, eye_candy=eye_candy)
    window = ViewerWindow()
    try:
        window.new_map_action.trigger()
        ((text, kwargs),) = asked
        assert all(reason in text for reason in reasons), text
        assert ("It gets about" in text) is (len(reasons) == 2)  # the count only past the cutoff
        assert kwargs["bare_label"] == bare_label
        assert window.scenario is None
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_a_planting_failure_reaches_the_error_box_and_frees_the_window(monkeypatch) -> None:
    conftest.ensure_qapp()
    from PyQt5.QtWidgets import QApplication

    import descape.viewer as viewer_module
    from descape import scenario_new
    from descape.unit_model import UnitEditsUnavailableError
    from descape.viewer import ViewerWindow

    def refuse(*args, **kwargs):
        raise UnitEditsUnavailableError("refused")

    errors = []
    monkeypatch.setattr(scenario_new, "UnitEditModel", refuse)
    monkeypatch.setattr(viewer_module.QMessageBox, "critical", staticmethod(lambda *a, **k: errors.append(a)))
    window = ViewerWindow()
    try:
        window._create_new_map(120, terrain_id=_FOREST_OAK, trees=True)
        assert window.scenario is None
        assert len(errors) == 1 and "refused" in errors[0][2]
        assert window.isEnabled()
        assert QApplication.overrideCursor() is None
        assert window.statusBar().currentMessage() == ""
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
@pytest.mark.parametrize("game_data", ("aoe2", "ror"))
def test_planting_runs_disabled_under_a_wait_cursor_and_frees_the_window_before_the_load(monkeypatch, game_data) -> None:
    """The busy state covers generation only; the warms are dropped before
    another game's tables come in, and not for the same game."""
    conftest.ensure_qapp()
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    import descape.viewer as viewer_module
    from descape.viewer import ViewerWindow

    events: list = []
    window = ViewerWindow()
    real_generate = viewer_module.blank_scenario_bytes
    real_load = ViewerWindow.load_scenario
    real_cancel = ViewerWindow._cancel_warms

    def generate(*args, **kwargs):
        cursor = QApplication.overrideCursor()
        events.append(("generate", window.isEnabled(), cursor and cursor.shape(), window.statusBar().currentMessage()))
        return real_generate(*args, **kwargs)

    def load(self, *args, **kwargs):
        events.append(("load", self.isEnabled(), QApplication.overrideCursor()))
        return real_load(self, *args, **kwargs)

    def cancel(self):
        events.append(("cancel",))
        return real_cancel(self)

    monkeypatch.setattr(viewer_module, "blank_scenario_bytes", generate)
    monkeypatch.setattr(ViewerWindow, "load_scenario", load)
    monkeypatch.setattr(ViewerWindow, "_cancel_warms", cancel)
    try:
        window._create_new_map(120, terrain_id=_FOREST_OAK, game_data=game_data, trees=True)
        start = events.index(("load", True, None))
        before = events[:start]
        assert before[-1] == ("generate", False, Qt.WaitCursor, "Planting trees…")
        assert (("cancel",) in before) is (game_data == "ror")
        assert _unit_counts(window) == (120 * 120, 0)
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
@pytest.mark.parametrize(("click", "expected"), (("Create", "create"), ("Create without trees", "bare"), ("Cancel", "cancel")))
def test_the_planting_question_has_three_buttons_and_maps_each(monkeypatch, click, expected) -> None:
    conftest.ensure_qapp()
    import descape.viewer as viewer_module
    from descape.viewer import ViewerWindow

    seen = []

    def exec_(box):
        seen.append(([b.text() for b in box.buttons()], box.defaultButton().text(), box.text()))
        next(b for b in box.buttons() if b.text() == click).click()
        return 0

    monkeypatch.setattr(viewer_module.QMessageBox, "exec_", exec_)
    window = ViewerWindow()
    try:
        assert window._ask_new_map_confirm("Plants a lot.", bare_label="Create without trees") == expected
        ((texts, default, text),) = seen
        assert sorted(texts) == sorted(["Create", "Create without trees", "Cancel"])
        assert default == "Cancel" and text == "Plants a lot."
    finally:
        window.close()
