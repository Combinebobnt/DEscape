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

from descape.mirror_tools import plan_mirror
from descape.scenario_io import BLANK_TEMPLATE_PATH

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_OTHER_TERRAIN = 2  # BEACH -- distinct from the blank template's own terrain_id=0


def _mirror_window():
    """Loads the blank template and switches to Terrain mode. Caller must
    edit_history.mark_saved() + close()."""
    return conftest.terrain_edit_window()


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


def test_preview_then_cancel_restores_the_pre_preview_state() -> None:
    """GH #77: Cancel after a Preview leaves tiles, cursor and undo tip exactly
    as they were before the dialog opened."""
    from descape.edit_history import tile_state
    from descape.viewer import MirrorDialog

    window = _mirror_window()
    try:
        mm = window.scenario.map_manager
        _paint_asymmetric_tiles(window)
        before_tiles = [tile_state(t) for t in mm.terrain]
        before_cursor = window.edit_history.cursor
        before_tip = window.edit_history.peek_undo()

        dialog = MirrorDialog(window)
        dialog._on_preview()
        assert window.edit_history.cursor == before_cursor + 1, "the preview applied nothing"
        assert [tile_state(t) for t in mm.terrain] != before_tiles

        dialog.reject()
        assert [tile_state(t) for t in mm.terrain] == before_tiles
        assert window.edit_history.cursor == before_cursor
        assert window.edit_history.peek_undo() is before_tip
    finally:
        window.edit_history.mark_saved()
        window.close()


def _mirror_overlay_items(window) -> list:
    z = window.map_view.MIRROR_OVERLAY_Z
    return [item for item in window.map_view.scene().items() if item.zValue() == z]


@pytest.mark.parametrize("finish", ["cancel", "apply"])
def test_the_source_slice_overlay_shows_on_open_and_clears_on_finish(finish: str) -> None:
    """GH #77 step 2: the shaded slice is up as soon as the dialog opens and
    gone once it closes by either button."""
    from descape.viewer import MirrorDialog

    window = _mirror_window()
    try:
        _paint_asymmetric_tiles(window)
        assert _mirror_overlay_items(window) == []
        dialog = MirrorDialog(window)
        # Outline and fill; the axis lines are Flat-only and this window is Stepped.
        assert len(_mirror_overlay_items(window)) == 2
        if finish == "cancel":
            dialog.reject()
        else:
            dialog._on_apply()
            assert window.edit_history.peek_undo().label == "Mirror Map"
        assert _mirror_overlay_items(window) == []
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

        assert "refreshed the full map in place instead of patching" in window.status_log.toPlainText()
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


def test_mirror_overlay_every_subpath_is_closed() -> None:
    """Same bug class as test_footprint_outlines.test_every_subpath_is_closed:
    show_mirror_overlay() built one addPolygon() per tile with no
    closeSubpath(), so a stroked tile diamond drew only 3 of its 4 edges."""
    from PyQt5.QtGui import QPainterPath

    window = _mirror_window()
    try:
        window.map_view.show_mirror_overlay([(0, 0), (5, 3), (2, 2)], [])
        path = window.map_view._mirror_overlay_outline_item.path()
        moves = sum(
            1 for i in range(path.elementCount()) if path.elementAt(i).type == QPainterPath.MoveToElement
        )
        assert moves > 0
        assert path.elementCount() == 5 * moves
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


# --- Stage 2: units -----------------------------------------------------------


def _place(window, unit_const: int, x: float, y: float, player: int = 1):
    model = window._ensure_unit_edits()
    model.begin_unit_edit([player])
    unit = model.add(player=player, unit_const=unit_const, x=x, y=y)
    model.commit_unit_edit("Place unit", window.edit_history)
    window._after_unit_mutation()
    return unit


def _units(window):
    return list(window.scenario.unit_manager.get_all_units())


def test_units_checkbox_is_available_and_off_by_default() -> None:
    from descape.viewer import MirrorDialog

    window = _mirror_window()
    try:
        dialog = MirrorDialog(window)
        assert dialog.units_checkbox.isEnabled()
        assert not dialog.units_checkbox.isChecked()
        assert not dialog.ownership_checkbox.isChecked()
        assert not dialog.ownership_spin.isEnabled()
        dialog.ownership_checkbox.setChecked(True)
        assert dialog.ownership_spin.isEnabled()
        dialog.reject()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_mirroring_units_copies_them_into_every_slice_as_one_undo_record() -> None:
    from descape.mirror_tools import plan_mirror_units

    window = _mirror_window()
    try:
        mm = window.scenario.map_manager
        _place(window, 349, 10.5, 12.5, player=0)  # an oak, well inside mode 1's source slice
        before = len(_units(window))
        records = len(window.edit_history.records)

        plan = plan_mirror(mm, 1, 0, do_terrain=True, do_elevation=False)
        unit_plan = plan_mirror_units(
            mm, 1, 0, window.scenario.unit_manager.units, plan.source_indices,
            referencing=window.unit_edits.referencing,
        )
        window.on_mirror(plan, unit_plan)

        assert len(window.edit_history.records) == records + 1
        assert len(_units(window)) == before + 1
        image = max(_units(window), key=lambda u: u.reference_id)
        assert (image.x, image.y) == (mm.map_width - 12.5, mm.map_width - 10.5)

        window.undo()
        assert len(_units(window)) == before
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_mirroring_units_refuses_a_straddling_building() -> None:
    from descape.mirror_tools import plan_mirror_units

    window = _mirror_window()
    try:
        mm = window.scenario.map_manager
        n = mm.map_width
        _place(window, 82, n / 2 - 1.0, n / 2 - 1.0)  # a castle across mode 1's axis
        before = len(_units(window))

        plan = plan_mirror(mm, 1, 0, do_terrain=True, do_elevation=False)
        unit_plan = plan_mirror_units(
            mm, 1, 0, window.scenario.unit_manager.units, plan.source_indices
        )
        assert unit_plan.blocked
        assert window.on_mirror(plan, unit_plan) is None
        assert len(_units(window)) == before
        assert "straddle" in window.status_log.toPlainText()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_refusal_names_the_spilling_building_and_its_tile_not_the_tree() -> None:
    """TASK-043: under mode 5 an oak's copy lands on the one tile a castle
    spills past the boundary. The refusal and the summary name the castle and
    that tile; the oak, which also straddles, is not named."""
    from descape.mirror_tools import plan_mirror_units
    from descape.object_catalog import display_name

    window = _mirror_window()
    try:
        mm = window.scenario.map_manager
        n = mm.map_width
        _place(window, 349, 21.3, n - 23 + 0.6, player=0)
        _place(window, 82, n - 23.5, 20.5)
        plan = plan_mirror(mm, 5, 0, do_terrain=True, do_elevation=False)
        unit_plan = plan_mirror_units(mm, 5, 0, window.scenario.unit_manager.units, plan.source_indices)
        assert len(unit_plan.straddling) == 2, "the oak no longer straddles, so this proves nothing"
        before_log = window.status_log.toPlainText()
        assert window.on_mirror(plan, unit_plan) is None
        log = _log_since(window, before_log)
        assert "1 unit(s) straddle the symmetry axis or a slice boundary" in log
        assert f"(e.g. Castle at ({n - 24}, 20), tile ({n - 22}, 22))" in log
        assert display_name(349) not in log

        dialog = _units_dialog(window, 5)
        assert "BLOCKED: 1 unit(s) straddle the symmetry axis or a slice boundary" in dialog.summary_label.text()
        dialog.reject()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_dialogs_own_units_path_previews_and_auto_undoes() -> None:
    """The dialog's own _compute_unit_plan(): a preview with Units ticked
    applies terrain and units as one composite, and changing an option undoes
    it again."""
    from descape.edit_history import CompositeDiffRecord
    from descape.viewer import MirrorDialog

    window = _mirror_window()
    try:
        _paint_asymmetric_tiles(window)
        _place(window, 349, 10.5, 12.5, player=0)
        before = len(_units(window))

        dialog = MirrorDialog(window)
        dialog.mode_combo.setCurrentIndex(0)  # mode 1
        dialog.units_checkbox.setChecked(True)
        dialog._on_preview()

        assert isinstance(window.edit_history.peek_undo(), CompositeDiffRecord)
        assert len(_units(window)) == before + 1

        dialog.elevation_checkbox.setChecked(not dialog.elevation_checkbox.isChecked())
        assert len(_units(window)) == before, "changing an option must undo the preview"
        dialog.reject()
    finally:
        window.edit_history.mark_saved()
        window.close()


# --- TASK-250: the overlap policy ---------------------------------------------


def _place_axis_castle(window):
    """test_mirroring_units_refuses_a_straddling_building's castle: across
    mode 1's axis, and across mode 5's centre too."""
    n = window.scenario.map_manager.map_width
    return _place(window, 82, n / 2 - 1.0, n / 2 - 1.0)


def _units_dialog(window, mode_id: int, policy: str | None = None):
    from descape.viewer import MirrorDialog

    dialog = MirrorDialog(window)
    dialog.mode_combo.setCurrentIndex(dialog.mode_combo.findData(mode_id))
    dialog.units_checkbox.setChecked(True)
    if policy is not None:
        dialog.overlaps_combo.setCurrentIndex(dialog.overlaps_combo.findData(policy))
    return dialog


def _log_since(window, before: str) -> str:
    text = window.status_log.toPlainText()
    assert text.startswith(before)
    return text[len(before):]


def test_overlap_control_follows_the_units_box() -> None:
    from descape.mirror_tools import OVERLAP_POLICIES, OVERLAPS_PLACE, OVERLAPS_REFUSE
    from descape.viewer import MirrorDialog

    window = _mirror_window()
    try:
        dialog = MirrorDialog(window)
        combo = dialog.overlaps_combo
        assert dialog.overlaps_label.text() == "Overlapping copies:"
        assert [combo.itemText(i) for i in range(combo.count())] == ["Refuse", "Skip them", "Place anyway"]
        assert [combo.itemData(i) for i in range(combo.count())] == list(OVERLAP_POLICIES)
        assert combo.currentData() == OVERLAPS_REFUSE
        assert "Elevation seams and garrisoned units outside the source slice still refuse" in combo.toolTip()
        dialog.layout().activate()
        assert dialog.overlaps_label.geometry().x() > dialog.units_checkbox.geometry().x(), "not indented"

        assert not combo.isEnabled() and not dialog.overlaps_label.isEnabled()
        dialog.units_checkbox.setChecked(True)
        assert combo.isEnabled() and dialog.overlaps_label.isEnabled()
        dialog.units_checkbox.setChecked(False)
        assert not combo.isEnabled() and not dialog.overlaps_label.isEnabled()

        combo.setCurrentIndex(combo.findData(OVERLAPS_PLACE))
        dialog.reject()
        reopened = MirrorDialog(window, last=dialog.state())
        assert reopened.overlaps_combo.currentData() == OVERLAPS_PLACE, "TASK-514: the pick is restored"
        assert not reopened.overlaps_combo.isEnabled() and not reopened.overlaps_label.isEnabled()
        reopened.reject()
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize(("policy", "fate", "added"), [("skip", "skipped", 0), ("place", "placed overlapping", 1)])
def test_skip_and_place_mirror_past_a_straddling_building(policy, fate, added) -> None:
    from PyQt5.QtWidgets import QDialog

    from descape.edit_history import tile_state

    window = _mirror_window()
    try:
        mm = window.scenario.map_manager
        _paint_asymmetric_tiles(window)
        castle = _place_axis_castle(window)
        before_tiles = [tile_state(t) for t in mm.terrain]
        before_units = len(_units(window))
        before_cursor = window.edit_history.cursor
        before_log = window.status_log.toPlainText()

        dialog = _units_dialog(window, 1, policy)
        (overlap,) = dialog._compute_unit_plan(dialog._compute_plan()).overlapping
        dialog._on_apply()

        assert dialog.result() == QDialog.Accepted
        assert window.edit_history.cursor == before_cursor + 1
        assert window.edit_history.peek_undo().label == "Mirror Map"
        assert len(_units(window)) == before_units + added
        if added:
            placed = max(_units(window), key=lambda u: u.reference_id)
            assert (placed.unit_const, placed.x, placed.y) == (castle.unit_const, overlap.x, overlap.y)
        log = _log_since(window, before_log)
        assert "Mirrored" in log
        assert f"; 1 overlapping cop(ies) {fate} (e.g. ({int(overlap.x)}, {int(overlap.y)}))" in log

        window.undo()
        assert len(_units(window)) == before_units
        assert [tile_state(t) for t in mm.terrain] == before_tiles
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("policy", ["refuse", "skip", "place"])
def test_no_overlap_policy_gets_past_an_elevation_seam(policy) -> None:
    """D3: the policy rides on the unit plan only, so with straddlers present
    every policy still meets on_mirror()'s elevation refusal. Under refuse
    the unit check fires first, so that arm asserts no message text."""
    from descape.mirror_tools import OVERLAP_POLICIES, MirrorPlan, plan_mirror_units

    assert policy in OVERLAP_POLICIES
    window = _mirror_window()
    try:
        mm = window.scenario.map_manager
        _place_axis_castle(window)
        source = plan_mirror(mm, 1, 0, do_terrain=True, do_elevation=False).source_indices
        unit_plan = plan_mirror_units(mm, 1, 0, window.scenario.unit_manager.units, source, overlaps=policy)
        assert unit_plan.straddling and unit_plan.overlapping, "no straddler, so the policy is never exercised"
        seam = MirrorPlan(changes=[(0, (5, 0, -1))], source_indices=source, elevation_violations=[(0, 1)])
        before_records = len(window.edit_history.records)
        before_units = len(_units(window))
        before_log = window.status_log.toPlainText()

        assert window.on_mirror(seam, unit_plan) is None
        assert len(window.edit_history.records) == before_records
        assert len(_units(window)) == before_units
        log = _log_since(window, before_log)
        assert "refused" in log
        if policy != "refuse":
            assert "elevation seam" in log
            assert "straddle" not in log
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_summary_under_skip_is_not_blocked_by_overlaps_alone() -> None:
    window = _mirror_window()
    try:
        _place_axis_castle(window)
        dialog = _units_dialog(window, 1)
        text = dialog.summary_label.text()
        assert "BLOCKED" in text and "straddle" in text  # refuse, the default
        assert dialog.summary_label.styleSheet() != ""

        dialog.overlaps_combo.setCurrentIndex(dialog.overlaps_combo.findData("skip"))
        text = dialog.summary_label.text()
        assert "BLOCKED" not in text and "straddle" not in text
        assert "1 unit cop(ies) overlap the source and will be skipped" in text
        assert dialog.summary_label.styleSheet() == ""

        dialog.overlaps_combo.setCurrentIndex(dialog.overlaps_combo.findData("place"))
        assert "1 unit cop(ies) overlap the source and will be placed overlapping" in dialog.summary_label.text()
        dialog.reject()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_summary_under_skip_is_blocked_by_an_elevation_seam() -> None:
    """The reverse: a real seam (test_mirror_tools' ramp, under mode 5) keeps
    the summary BLOCKED and Apply refusing, whatever the overlap pick."""
    window = _mirror_window()
    try:
        mm = window.scenario.map_manager
        n = mm.map_width

        def _ramp() -> None:
            for y in range(n):
                for x in range(n):
                    mm.terrain[y * n + x].elevation = 0 if x < n // 2 else min(x - n // 2, 3)

        window.edit_history.apply("Ramp", mm.terrain, _ramp)
        _place_axis_castle(window)
        assert plan_mirror(mm, 5, 0, do_terrain=True, do_elevation=True).elevation_violations
        before_records = len(window.edit_history.records)

        dialog = _units_dialog(window, 5, "skip")
        assert dialog.elevation_checkbox.isChecked()
        assert dialog._compute_unit_plan(dialog._compute_plan()).straddling
        text = dialog.summary_label.text()
        assert "BLOCKED" in text and "elevation seam" in text
        assert "straddle" not in text
        assert dialog.summary_label.styleSheet() != ""

        dialog._on_apply()
        assert dialog.result() != dialog.Accepted
        assert len(window.edit_history.records) == before_records
        dialog.reject()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_changing_the_overlap_policy_undoes_a_live_preview() -> None:
    """D4: the combo goes through _on_option_changed() like every other option."""
    window = _mirror_window()
    try:
        _paint_asymmetric_tiles(window)
        _place_axis_castle(window)
        before = len(_units(window))
        dialog = _units_dialog(window, 1, "place")
        dialog._on_preview()
        assert len(_units(window)) == before + 1
        dialog.overlaps_combo.setCurrentIndex(dialog.overlaps_combo.findData("skip"))
        assert len(_units(window)) == before, "changing the policy must undo the preview"
        assert dialog._preview_record is None
        dialog.reject()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_garrison_block_under_skip_refuses_without_naming_the_straddlers() -> None:
    """D2: skip waives the straddlers, never a garrison, and the refusal names
    only what actually blocks."""
    from descape.mirror_tools import plan_mirror_units

    window = _mirror_window()
    try:
        mm = window.scenario.map_manager
        n = mm.map_width
        oak = _place(window, 349, 10.5, 12.5, player=0)
        holder = _place(window, 70, n - 10.0, n - 10.0)  # a house in the destination slice
        _place_axis_castle(window)
        plan = plan_mirror(mm, 1, 0, do_terrain=True, do_elevation=False)
        unit_plan = plan_mirror_units(
            mm, 1, 0, window.scenario.unit_manager.units, plan.source_indices,
            referencing=lambda u: [oak] if u is holder else [], overlaps="skip",
        )
        assert unit_plan.garrisoned_blockers == [holder] and unit_plan.straddling
        before_log = window.status_log.toPlainText()
        assert window.on_mirror(plan, unit_plan) is None
        log = _log_since(window, before_log)
        assert "garrison" in log and "straddle" not in log
    finally:
        window.edit_history.mark_saved()
        window.close()


# --- angular modes (6-way / 3-fold) ------------------------------------------


def test_the_angular_modes_are_offered_after_a_disabled_header() -> None:
    """The silent-drop trap: angular modes have an empty `group`, so the
    length buckets alone would never add them."""
    from descape.viewer import MirrorDialog

    window = _mirror_window()
    try:
        dialog = MirrorDialog(window)
        combo = dialog.mode_combo
        ids = [combo.itemData(i) for i in range(combo.count()) if combo.itemData(i) is not None]
        assert ids == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        header = next(i for i in range(combo.count()) if combo.itemText(i).startswith("Approximate"))
        assert not combo.model().item(header).isEnabled()
        assert combo.itemData(header) is None
        assert combo.currentData() == 1
        dialog.reject()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_mode_12_draws_three_mirror_axes_and_the_rotations_none() -> None:
    import math

    from descape.mirror_tools import MODE_BY_ID
    from descape.viewer import MirrorDialog

    window = _mirror_window()
    try:
        dialog = MirrorDialog(window)
        window.map_view._terrain_style = "flat"
        n = window.scenario.map_manager.map_width
        size = n * window.map_view._tile_pixels
        assert dialog._axis_lines(MODE_BY_ID[10], n) == []
        assert dialog._axis_lines(MODE_BY_ID[11], n) == []
        lines = dialog._axis_lines(MODE_BY_ID[12], n)
        assert len(lines) == 3
        exact_a = dialog._axis_lines(MODE_BY_ID[1], n)[0]
        assert any((p.x(), p.y(), q.x(), q.y()) == (exact_a[0].x(), exact_a[0].y(), exact_a[1].x(), exact_a[1].y())
                   for p, q in lines)
        angles = []
        for p, q in lines:
            assert abs((p.x() + q.x()) / 2 - size / 2) < 1e-6 and abs((p.y() + q.y()) / 2 - size / 2) < 1e-6
            for point in (p, q):  # clipped onto the square's own boundary
                assert min(point.x(), point.y(), size - point.x(), size - point.y()) == pytest.approx(0, abs=1e-6)
            angles.append(round(math.degrees(math.atan2(q.y() - p.y(), q.x() - p.x())) % 180, 6))
        assert sorted(angles) == [15.0, 75.0, 135.0]
        dialog.reject()
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize(
    ("style", "iso", "render_style", "axes"),
    [("Flat", True, "stepped", 0), ("Stepped", False, "stepped", 0), ("Flat", False, "flat", 3)],
    ids=["flat_isometric", "stepped", "flat_square"],
)
def test_mode_12_in_the_scene_shades_its_slice_and_draws_axes_in_flat_square_only(style, iso, render_style, axes) -> None:
    """TASK-046 step 7, in the scene the user sees rather than in
    _axis_lines' return value. Pass conditions, with mode 12 picked:
    - the view really renders in the style named (Flat + Isometric View
      builds Stepped geometry, so MapView's own style is "stepped");
    - the shaded slice is up: its outline and fill path items sit in the
      scene at the overlay's z;
    - Flat + Isometric View and Stepped add no line item at that z;
    - Flat square adds exactly the three axis line items, the ones
      _axis_lines gives for mode 12;
    - closing the dialog takes every overlay item away again."""
    from PyQt5.QtWidgets import QGraphicsLineItem, QGraphicsPathItem

    from descape.mirror_tools import MODE_BY_ID
    from descape.viewer import MirrorDialog

    window = _mirror_window()
    try:
        window.terrain_style_combo.setCurrentText(style)
        window.iso_action.setChecked(iso)
        assert window._terrain_style == style.lower()
        assert window.map_view._terrain_style == render_style
        dialog = MirrorDialog(window)
        dialog.mode_combo.setCurrentIndex(dialog.mode_combo.findData(12))
        assert dialog._current_mode() is MODE_BY_ID[12]
        items = _mirror_overlay_items(window)
        paths = [item for item in items if isinstance(item, QGraphicsPathItem)]
        lines = [item for item in items if isinstance(item, QGraphicsLineItem)]
        assert len(paths) == 2
        assert not paths[0].path().isEmpty() and paths[0].path() == paths[1].path()
        assert len(lines) == axes
        assert len(items) == 2 + axes
        expected = dialog._axis_lines(MODE_BY_ID[12], window.scenario.map_manager.map_width)
        assert sorted((ln.line().x1(), ln.line().y1(), ln.line().x2(), ln.line().y2()) for ln in lines) == sorted(
            (p.x(), p.y(), q.x(), q.y()) for p, q in expected
        )
        dialog.reject()
        assert _mirror_overlay_items(window) == []
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_angular_modes_default_elevation_off_and_report_unreachable_corners() -> None:
    from descape.viewer import MirrorDialog

    window = _mirror_window()
    try:
        dialog = MirrorDialog(window)
        assert dialog.elevation_checkbox.isChecked()
        dialog.mode_combo.setCurrentIndex(dialog.mode_combo.findData(11))
        assert not dialog.elevation_checkbox.isChecked()
        plan = plan_mirror(window.scenario.map_manager, 11, 0, True, False)
        assert plan.unreachable
        assert f"{len(plan.unreachable)} corner tile(s)" in dialog.summary_label.text()
        dialog.elevation_checkbox.setChecked(True)
        dialog.mode_combo.setCurrentIndex(dialog.mode_combo.findData(12))
        assert dialog.elevation_checkbox.isChecked(), "a user choice survives a switch within angular modes"
        dialog.mode_combo.setCurrentIndex(dialog.mode_combo.findData(1))
        assert dialog.elevation_checkbox.isChecked()
        assert "corner tile" not in dialog.summary_label.text()
        dialog.reject()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_six_way_mirror_with_units_is_one_undo_step() -> None:
    from descape.edit_history import CompositeDiffRecord, tile_state
    from descape.viewer import MirrorDialog

    window = _mirror_window()
    try:
        mm = window.scenario.map_manager
        _paint_asymmetric_tiles(window)
        _place(window, 349, 77.5, 70.5, player=0)  # an oak inside mode 11's wedge 0
        before_tiles = [tile_state(t) for t in mm.terrain]
        before_units = len(_units(window))

        dialog = MirrorDialog(window)
        dialog.mode_combo.setCurrentIndex(dialog.mode_combo.findData(11))
        dialog.units_checkbox.setChecked(True)
        dialog._on_apply()
        assert isinstance(window.edit_history.peek_undo(), CompositeDiffRecord)
        assert len(_units(window)) == before_units + 5
        window.undo()
        assert [tile_state(t) for t in mm.terrain] == before_tiles
        assert len(_units(window)) == before_units
    finally:
        window.edit_history.mark_saved()
        window.close()


# --- TASK-450: a pre-1.55 file's units have no readable caption ---------------


@pytest.mark.parametrize("error", ["UnsupportedAttributeError", "AttributeError"])
def test_mirroring_units_survives_an_unreadable_caption(error) -> None:
    """A pre-1.55 load leaves Unit's caption fields raising
    UnsupportedAttributeError (test_unit_model's _poison_unit_caption_fields
    shape); a unit parsed then reads as plain AttributeError once add() has
    depoisoned the class (test_region_clipboard's poisoned-parse case). The
    copy falls back to -1/"" only where the read raises: add() depoisons, so
    the second source's real caption still copies."""
    from AoE2ScenarioParser.exceptions.asp_exceptions import UnsupportedAttributeError
    from AoE2ScenarioParser.objects.data_objects.unit import Unit
    from PyQt5.QtWidgets import QDialog

    from descape import library_compat

    exc = {"UnsupportedAttributeError": UnsupportedAttributeError, "AttributeError": AttributeError}[error]

    def _raise(self_, val=None):
        raise exc("synthetic pre-1.55 caption read")

    window = _mirror_window()
    try:
        model = window._ensure_unit_edits()
        model.begin_unit_edit([0])
        for x, y in [(10.5, 12.5), (14.5, 20.5)]:  # two oaks inside mode 1's source slice
            model.add(player=0, unit_const=349, x=x, y=y, caption_string_id=7, caption_string="Hi")
        model.commit_unit_edit("Place oaks", window.edit_history)
        window._after_unit_mutation()
        before_units = len(_units(window))
        before_cursor = window.edit_history.cursor
        dialog = _units_dialog(window, 1)

        Unit.caption_string_id = property(_raise, _raise)
        Unit.caption_string = property(_raise, _raise)
        try:
            dialog._on_apply()
        finally:
            library_compat.depoison()

        assert dialog.result() == QDialog.Accepted
        assert window.edit_history.cursor == before_cursor + 1
        copies = sorted(_units(window), key=lambda u: u.reference_id)[before_units:]
        assert [(u.caption_string_id, u.caption_string) for u in copies] == [(-1, ""), (7, "Hi")]
        window.undo()
        assert len(_units(window)) == before_units
    finally:
        library_compat.depoison()
        window.edit_history.mark_saved()
        window.close()


# --- TASK-514: the options are session-sticky ---------------------------------


def _off_default(dialog) -> None:
    """Every option off its default: an angular mode, slice 3, Terrain off,
    Elevation on, Units on with Skip them, ownership on at 3."""
    from descape.mirror_tools import OVERLAPS_SKIP

    dialog.mode_combo.setCurrentIndex(dialog.mode_combo.findData(11))
    dialog.slice_combo.setCurrentIndex(3)
    dialog.terrain_checkbox.setChecked(False)
    dialog.elevation_checkbox.setChecked(True)
    dialog.units_checkbox.setChecked(True)
    dialog.overlaps_combo.setCurrentIndex(dialog.overlaps_combo.findData(OVERLAPS_SKIP))
    dialog.ownership_checkbox.setChecked(True)
    dialog.ownership_spin.setValue(3)


def test_a_restored_dialog_shows_every_option_and_its_enabled_states() -> None:
    from descape.mirror_tools import OVERLAPS_SKIP
    from descape.viewer import MirrorDialog

    window = _mirror_window()
    try:
        dialog = MirrorDialog(window)
        _off_default(dialog)
        state = dialog.state()
        dialog.reject()
        assert state == {
            "mode_id": 11,
            "slice": 3,
            "terrain": False,
            "elevation": True,
            "units": True,
            "overlaps": OVERLAPS_SKIP,
            "rotate_ownership": True,
            "ownership_steps": 3,
        }

        restored = MirrorDialog(window, last=state)
        assert restored.mode_combo.currentData() == 11
        assert restored.slice_combo.count() == 6 and restored.slice_combo.currentData() == 3
        assert not restored.terrain_checkbox.isChecked()
        assert restored.elevation_checkbox.isChecked()
        assert restored.units_checkbox.isChecked()
        assert restored.overlaps_combo.currentData() == OVERLAPS_SKIP
        assert restored.overlaps_combo.isEnabled() and restored.overlaps_label.isEnabled()
        assert restored.ownership_checkbox.isChecked() and restored.ownership_spin.value() == 3
        assert restored.ownership_spin.isEnabled()
        assert restored.state() == state
        restored.mode_combo.setCurrentIndex(restored.mode_combo.findData(12))
        assert restored.slice_combo.currentData() == 0, "a later mode change repopulates from 0"
        restored.reject()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_restored_slice_is_clamped_to_the_modes_count() -> None:
    from descape.viewer import MirrorDialog

    window = _mirror_window()
    try:
        dialog = MirrorDialog(window, last={"mode_id": 1, "slice": 5})
        assert dialog.slice_combo.count() == 2 and dialog.slice_combo.currentData() == 1
        dialog.reject()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_restored_elevation_wins_over_the_mode_kinds_default() -> None:
    """The trap: _elevation_default_kind starts at "lattice", so a restored
    angular mode must not see a kind switch on the first _on_mode_changed()."""
    from descape.viewer import MirrorDialog

    window = _mirror_window()
    try:
        dialog = MirrorDialog(window, last={"mode_id": 11, "elevation": True})
        assert dialog.elevation_checkbox.isChecked(), "a restored angular mode's Elevation must stay on"
        dialog.elevation_checkbox.setChecked(False)
        dialog.mode_combo.setCurrentIndex(dialog.mode_combo.findData(1))
        assert dialog.elevation_checkbox.isChecked(), "a switch to lattice still re-defaults it on"
        dialog.mode_combo.setCurrentIndex(dialog.mode_combo.findData(11))
        assert not dialog.elevation_checkbox.isChecked(), "and back to angular re-defaults it off"
        dialog.reject()

        lattice = MirrorDialog(window, last={"mode_id": 7, "elevation": False})
        assert not lattice.elevation_checkbox.isChecked()
        lattice.reject()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_full_restore_still_plans_once_per_open(monkeypatch) -> None:
    import descape.viewer as viewer_module
    from descape.viewer import MirrorDialog

    window = _mirror_window()
    try:
        donor = MirrorDialog(window)
        _off_default(donor)
        state = donor.state()
        donor.reject()

        calls = []
        real = viewer_module.plan_mirror
        monkeypatch.setattr(viewer_module, "plan_mirror", lambda *a, **k: calls.append(1) or real(*a, **k))
        MirrorDialog(window).reject()
        bare = len(calls)
        calls.clear()
        MirrorDialog(window, last=state).reject()
        assert bare >= 1 and len(calls) == bare
    finally:
        window.edit_history.mark_saved()
        window.close()


def _arm_mirror(monkeypatch, configure=None, accept: bool = True) -> list:
    """Replaces MirrorDialog.exec_: records the state each dialog opened with,
    runs `configure`, then accepts or rejects through the dialog's own slots."""
    from PyQt5.QtWidgets import QDialog

    from descape.viewer import MirrorDialog

    opened = []

    def exec_(self):
        opened.append(self.state())
        if configure is not None:
            configure(self)
        if accept:
            self.accept()
            return QDialog.Accepted
        self.reject()
        return QDialog.Rejected

    monkeypatch.setattr(MirrorDialog, "exec_", exec_)
    return opened


def _seven_way_two(dialog) -> None:
    dialog.mode_combo.setCurrentIndex(dialog.mode_combo.findData(7))
    dialog.slice_combo.setCurrentIndex(2)
    dialog.units_checkbox.setChecked(True)
    dialog.ownership_checkbox.setChecked(True)
    dialog.ownership_spin.setValue(2)


def test_an_applied_dialogs_options_open_next_time_even_after_another_file(monkeypatch, tmp_path) -> None:
    import shutil
    from pathlib import Path

    from descape.viewer import MirrorDialog

    other = tmp_path / "units.aoe2scenario"
    shutil.copyfile(Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario", other)

    window = _mirror_window()
    try:
        _arm_mirror(monkeypatch, _seven_way_two)
        window._show_mirror_dialog()
        applied = window._mirror_defaults
        assert applied["mode_id"] == 7 and applied["slice"] == 2 and applied["ownership_steps"] == 2

        opened = _arm_mirror(monkeypatch, accept=False)
        window._show_mirror_dialog()
        assert opened == [applied]

        window.edit_history.mark_saved()
        window.load_scenario(other)
        assert window.scenario.path == other
        window._show_mirror_dialog()
        assert opened == [applied, applied], "another scenario keeps the remembered options"
        assert MirrorDialog(window).state() != applied
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_cancelled_dialog_leaves_the_remembered_options(monkeypatch) -> None:
    from descape.viewer import MirrorDialog

    window = _mirror_window()
    try:
        defaults = MirrorDialog(window)
        default_state = defaults.state()
        defaults.reject()

        _arm_mirror(monkeypatch, _seven_way_two, accept=False)
        window._show_mirror_dialog()
        opened = _arm_mirror(monkeypatch, accept=False)
        window._show_mirror_dialog()
        assert opened == [default_state], "nothing applied yet: a cancel leaves the defaults"

        _arm_mirror(monkeypatch, _seven_way_two)
        window._show_mirror_dialog()
        applied = window._mirror_defaults
        _arm_mirror(monkeypatch, _off_default, accept=False)
        window._show_mirror_dialog()
        opened = _arm_mirror(monkeypatch, accept=False)
        window._show_mirror_dialog()
        assert opened == [applied], "the cancelled options must not be kept"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_read_only_units_untick_but_are_remembered(monkeypatch) -> None:
    from descape.viewer import MirrorDialog

    window = _mirror_window()
    try:
        monkeypatch.setattr(window, "_ensure_unit_edits", lambda: None)
        dialog = MirrorDialog(window, last={"units": True})
        assert not dialog.units_checkbox.isChecked()
        assert not dialog.overlaps_combo.isEnabled()
        assert dialog.state()["units"] is True, "the next writable file gets Units back"
        dialog.reject()
    finally:
        window.edit_history.mark_saved()
        window.close()
