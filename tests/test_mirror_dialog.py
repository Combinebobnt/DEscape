"""Map mirroring's Qt wiring (Stage 1: terrain + elevation) -- driven through
a real offscreen ViewerWindow, the same technique/conventions
tests/test_fill_tool.py established (QT_QPA_PLATFORM=offscreen, one shared
QApplication via conftest.ensure_qapp(), qtbot used nowhere in this repo).
tests/test_mirror_tools.py already covers the Qt-free algorithm; this module
covers only what needs a real ViewerWindow: QAction/keybind registration,
_update_tool_enabled() gating, MirrorDialog's Preview/Apply/Cancel state
machine, and the Keybinds tab's Map section header.

Every ViewerWindow() constructed here calls edit_history.mark_saved() before
close() -- close() runs _confirm_discard_changes(), which pops a modal
QMessageBox on a dirty document and would hang an offscreen run.
"""

from __future__ import annotations

import pytest

import conftest
from descape.mirror_tools import plan_mirror
from descape.scenario_io import BLANK_TEMPLATE_PATH

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_OTHER_TERRAIN = 2  # BEACH -- distinct from the blank template's own terrain_id=0


def _mirror_window():
    """Loads the blank template and switches to Terrain mode. Caller must
    edit_history.mark_saved() + close()."""
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.load_scenario(BLANK_TEMPLATE_PATH)
    assert window.scenario is not None, "blank template failed to load"
    window.mode_combo.setCurrentText("Terrain")
    return window


def _paint_asymmetric_tiles(window) -> None:
    """Directly mutates a handful of scattered tiles (not through a stroke)
    so the map is no longer symmetric under any mode, then records it as one
    undo step -- mirrors mirror_tools tests' own FakeTile-mutation shape,
    just against the real MapManager."""
    mm = window.scenario.map_manager

    def _mutate() -> None:
        for x, y in [(0, 0), (5, 3), (40, 90)]:
            mm.terrain[y * mm.map_width + x].terrain_id = _OTHER_TERRAIN

    window.edit_history.apply("Paint asymmetric tiles", mm.terrain, _mutate)


def test_mirror_registered_as_keybind_action() -> None:
    from descape import settings
    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    assert ("map_mirror", "Mirror Map…", "") in settings.REBINDABLE_ACTIONS

    window = ViewerWindow()
    try:
        assert window._keybind_actions["map_mirror"] is window.mirror_action
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_mirror_enable_gating() -> None:
    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    try:
        assert not window.mirror_action.isEnabled()  # no map loaded yet

        window.load_scenario(BLANK_TEMPLATE_PATH)
        window.mode_combo.setCurrentText("Terrain")
        assert window.mirror_action.isEnabled()
        # Same gate as Elevate/Set Level (write_ok and map_is_square), not
        # Draw/Paint Can's (write_ok alone) -- plan_mirror's elevation half
        # raw-assigns tile.elevation the same way MapManager.set_elevation
        # does, so it needs the same squareness guarantee.
        assert window.mirror_action.isEnabled() == window.elevation_action.isEnabled()

        # No real non-square fixture in this tier (the app cannot open one
        # today -- scenario_io.py) -- LoadedScenario is a plain dataclass,
        # so flip the flag directly and re-run the same gating pass load/
        # close call, exactly as it would for a real non-square file.
        window.scenario.map_is_square = False
        window._update_tool_enabled()
        assert not window.mirror_action.isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_mirror_end_to_end_and_undo() -> None:
    """Paints a few asymmetric tiles, mirrors under 8-way (the most
    demanding mode -- exercises every transform), asserts full D4 symmetry,
    then undoes the mirror and asserts the grid is byte-identical to the
    pre-mirror (post-paint) state -- one record, one undo."""
    from descape.edit_history import tile_state
    from descape.mirror_tools import TRANSFORMS

    window = _mirror_window()
    try:
        mm = window.scenario.map_manager
        _paint_asymmetric_tiles(window)
        assert len(window.edit_history.records) == 1
        before_mirror = [tile_state(t) for t in mm.terrain]

        plan = plan_mirror(mm, 9, 0, do_terrain=True, do_elevation=True)
        dirty = window.on_mirror(plan)
        assert dirty  # the painted tiles broke symmetry, so this is a real change
        assert len(window.edit_history.records) == 2
        assert window.edit_history.records[-1].label == "Mirror Map"
        assert window.edit_history.can_undo

        n = mm.map_width
        for x in range(n):
            for y in range(n):
                base = tile_state(mm.terrain[y * n + x])
                for name in ("id", "r", "r2", "r3", "mx", "my", "d", "a"):
                    gx, gy = TRANSFORMS[name](x, y, n)
                    assert tile_state(mm.terrain[gy * n + gx]) == base

        window.undo()
        assert [tile_state(t) for t in mm.terrain] == before_mirror
        # EditHistory is a cursor into a list, not a pop-stack -- undo()
        # moves the cursor back rather than deleting the record, so the
        # paint record is still the (now current) undo tip and the mirror
        # record is still there, past the cursor, for redo.
        assert window.edit_history.can_undo
        assert window.edit_history.peek_undo().label == "Paint asymmetric tiles"
        assert window.edit_history.can_redo
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_preview_auto_undo_on_mode_change() -> None:
    from descape.viewer import MirrorDialog

    window = _mirror_window()
    try:
        _paint_asymmetric_tiles(window)
        assert len(window.edit_history.records) == 1

        dialog = MirrorDialog(window)
        try:
            dialog._on_preview()
            assert len(window.edit_history.records) == 2
            assert dialog._preview_record is not None

            other_index = 1 if dialog.mode_combo.currentIndex() == 0 else 0
            dialog.mode_combo.setCurrentIndex(other_index)

            # EditHistory is a cursor into a list, not a pop-stack --
            # undo() moves the cursor back rather than deleting the record
            # (the mirror record is still there, past the cursor, for
            # redo), so "gone" means "no longer the undo tip", not "removed
            # from the list".
            assert len(window.edit_history.records) == 2
            assert window.edit_history.cursor == 1  # exactly one record still applied
            assert window.edit_history.peek_undo().label == "Paint asymmetric tiles"
            assert dialog._preview_record is None
        finally:
            dialog.close()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_preview_auto_undo_declines_when_not_top_record() -> None:
    """If some other edit lands on top of the preview's own record before
    the dialog tries to auto-undo it, EditHistory.peek_undo() no longer
    matches -- the guard must leave both records alone rather than eating
    the unrelated edit."""
    from descape.viewer import MirrorDialog

    window = _mirror_window()
    try:
        mm = window.scenario.map_manager
        _paint_asymmetric_tiles(window)

        dialog = MirrorDialog(window)
        try:
            dialog._on_preview()
            assert len(window.edit_history.records) == 2
            preview_record = dialog._preview_record
            assert preview_record is not None

            def _other_edit() -> None:
                mm.terrain[50].elevation = 2

            window.edit_history.apply("Other edit", mm.terrain, _other_edit)
            assert len(window.edit_history.records) == 3
            assert window.edit_history.peek_undo() is not preview_record

            other_index = 1 if dialog.mode_combo.currentIndex() == 0 else 0
            dialog.mode_combo.setCurrentIndex(other_index)

            # Declined: nothing was undone, all three records remain.
            assert len(window.edit_history.records) == 3
            assert window.edit_history.records[-1].label == "Other edit"
            # dialog._preview_record is still reset to None on any option
            # change regardless -- it no longer describes the current
            # mode/slice selection either way.
            assert dialog._preview_record is None
        finally:
            dialog.close()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_stepped_full_rerender_fallback_fires_for_mirror(monkeypatch) -> None:
    """Mirrors test_fill_tool.py's own
    test_stepped_full_rerender_fallback_actually_fires -- lowers the
    threshold rather than mirroring a 480x480 map, to stay in the default
    tier."""
    from PyQt5.QtWidgets import QApplication

    import descape.viewer as viewer_module

    monkeypatch.setattr(viewer_module, "STEPPED_FULL_RERENDER_THRESHOLD", 100)

    window = _mirror_window()
    try:
        window.terrain_style_combo.setCurrentText("Stepped")
        mm = window.scenario.map_manager
        n = mm.map_width

        def _paint_quadrant() -> None:
            for y in range(n // 2):
                for x in range(n // 2):
                    mm.terrain[y * n + x].terrain_id = _OTHER_TERRAIN

        window.edit_history.apply("Paint quadrant", mm.terrain, _paint_quadrant)

        plan = plan_mirror(mm, 9, 0, do_terrain=True, do_elevation=False)
        dirty = window.on_mirror(plan)
        assert dirty  # well past the patched threshold

        assert "re-rendered full map instead of patching" in window.status_log.toPlainText()
        assert QApplication.overrideCursor() is None
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_mirror_refuses_on_elevation_violations(monkeypatch) -> None:
    """plan_mirror's own seam check is covered exhaustively in
    tests/test_mirror_tools.py -- this only pins that on_mirror() actually
    refuses (no record pushed) and logs a message when handed a plan whose
    elevation_violations is non-empty, without needing a real geometry that
    triggers one."""
    from descape.mirror_tools import MirrorPlan

    window = _mirror_window()
    try:
        before_count = len(window.edit_history.records)
        fake_plan = MirrorPlan(changes=[(0, (5, 0, -1))], source_indices=frozenset(), elevation_violations=[(0, 1)])
        dirty = window.on_mirror(fake_plan)
        assert dirty is None
        assert len(window.edit_history.records) == before_count
        assert "refused" in window.status_log.toPlainText().lower()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_keybinds_tab_renders_map_section_header() -> None:
    from PyQt5.QtWidgets import QLabel

    from descape.viewer import SettingsDialog, ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    try:
        dialog = SettingsDialog(window)
        try:
            headers = {
                label.text().replace("<b>", "").replace("</b>", "")
                for label in dialog.findChildren(QLabel)
                if label.text().startswith("<b>")
            }
            assert "Map" in headers
        finally:
            dialog.close()
    finally:
        window.edit_history.mark_saved()
        window.close()
