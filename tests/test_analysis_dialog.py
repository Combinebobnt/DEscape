"""Tools > Map Analysis: AnalysisDialog, its viewer wiring, and
MapView.center_on_tile().

The dialog is modeless, so nothing here may call exec_(). Every window is
closed through conftest.close_window(), which marks the history saved first.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

UNITS_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"


def _report():
    from descape.map_analysis import AnalysisReport, CheckResult, Finding

    located = Finding("unit off somewhere", "warning", tile=(3, 4))
    unlocated = Finding("no place", "info")
    return (
        AnalysisReport(
            (
                CheckResult("Clean check"),
                CheckResult("Two findings", (located, unlocated)),
                CheckResult("Skipped check", unavailable="triggers could not be parsed"),
            )
        ),
        located,
        unlocated,
    )


def _window():
    from PyQt5.QtWidgets import QApplication

    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    window.resize(1200, 800)
    window.show()
    window.load_scenario(UNITS_FIXTURE)
    QApplication.processEvents()
    return window


def test_dialog_groups_results_and_navigates_only_located_findings() -> None:
    conftest.ensure_qapp()
    from descape.analysis_dialog import AnalysisDialog

    report, located, _unlocated = _report()
    navigated = []
    dialog = AnalysisDialog(on_navigate=navigated.append)
    dialog.set_report(report, "fixture.aoe2scenario")

    tree = dialog.tree
    assert tree.topLevelItemCount() == 3
    assert tree.topLevelItem(0).text(0) == "Clean check (clean)"
    assert tree.topLevelItem(1).text(0) == "Two findings (2)"
    assert tree.topLevelItem(2).text(0) == "Skipped check (unavailable: triggers could not be parsed)"
    assert dialog.headline.text() == f"fixture.aoe2scenario: {report.headline}"

    group = tree.topLevelItem(1)
    assert (group.child(1).text(0), group.child(1).text(1)) == ("Info", "no place")
    assert group.child(1).toolTip(1) == "No map location for this finding"
    assert group.child(0).toolTip(1) == ""
    tree.itemDoubleClicked.emit(group.child(1), 0)
    tree.itemDoubleClicked.emit(group, 0)
    assert navigated == []
    tree.itemDoubleClicked.emit(group.child(0), 0)
    assert navigated == [located]
    dialog.deleteLater()


def test_set_report_replaces_previous_rows() -> None:
    conftest.ensure_qapp()
    from descape.analysis_dialog import AnalysisDialog

    report, _, _ = _report()
    dialog = AnalysisDialog()
    dialog.set_report(report)
    dialog.set_report(report)
    assert dialog.tree.topLevelItemCount() == 3
    dialog.deleteLater()


# --- Refresh (GH #209) ----------------------------------------------------------


def _rows(dialog, label: str):
    """The finding rows of the check labelled `label`, top to bottom."""
    tree = dialog.tree
    group = next(tree.topLevelItem(i) for i in range(tree.topLevelItemCount()) if tree.topLevelItem(i).text(0).startswith(label))
    return group, [group.child(j) for j in range(group.childCount())]


def test_refresh_button_sits_in_the_close_row_and_calls_on_refresh() -> None:
    conftest.ensure_qapp()
    from PyQt5.QtWidgets import QDialogButtonBox

    from descape.analysis_dialog import AnalysisDialog

    calls = []
    dialog = AnalysisDialog(on_refresh=lambda: calls.append("refresh"))
    box = dialog.findChild(QDialogButtonBox)
    assert dialog.refresh_button.text() == "Refresh"
    assert box.buttonRole(dialog.refresh_button) == QDialogButtonBox.ActionRole
    assert dialog.refresh_button.toolTip()
    assert not dialog.refresh_button.autoDefault(), "Enter must not start a re-run"
    dialog.refresh_button.click()
    assert calls == ["refresh"]
    dialog.deleteLater()

    bare = AnalysisDialog()
    bare.refresh_button.click()
    bare.deleteLater()


def test_a_refill_keeps_the_current_finding_by_identity_not_by_row() -> None:
    conftest.ensure_qapp()
    from PyQt5.QtCore import Qt

    from descape.analysis_dialog import AnalysisDialog
    from descape.map_analysis import AnalysisReport, CheckResult, Finding

    report, located, unlocated = _report()
    selected = []
    dialog = AnalysisDialog(on_select=selected.append)
    dialog.set_report(report)
    _group, rows = _rows(dialog, "Two findings")
    dialog.tree.setCurrentItem(rows[1])
    assert selected[-1] == unlocated

    # A new finding above it: the same row number now holds another finding.
    earlier = Finding("new, sorts first", "error", tile=(1, 1))
    grown = AnalysisReport((CheckResult("Clean check"), CheckResult("Two findings", (earlier, located, unlocated))))
    dialog.set_report(grown)
    group, rows = _rows(dialog, "Two findings")
    assert dialog.tree.currentItem() is rows[2]
    assert dialog.tree.currentItem().data(0, Qt.UserRole) == unlocated
    assert selected[-1] == unlocated, "the window hears about the kept row again, for its marker"

    # The check's own row is kept by its label, though its text carries the count.
    dialog.tree.setCurrentItem(group)
    dialog.set_report(report)
    group, rows = _rows(dialog, "Two findings")
    assert dialog.tree.currentItem() is group and group.text(0) == "Two findings (2)"

    # A fixed finding is gone: nothing is current, and nothing else is picked for it.
    dialog.tree.setCurrentItem(rows[0])
    dialog.set_report(AnalysisReport((CheckResult("Clean check"), CheckResult("Two findings", (unlocated,)))))
    assert dialog.tree.currentItem() is None
    assert selected[-1] is None
    dialog.deleteLater()


def test_a_refill_follows_a_moved_unit_and_leaves_a_collapsed_check_collapsed() -> None:
    conftest.ensure_qapp()
    from PyQt5.QtCore import Qt

    from descape.analysis_dialog import AnalysisDialog
    from descape.map_analysis import AnalysisReport, CheckResult, Finding

    other = Finding("Oak at (5.5, 5.5)", "warning", unit_key=(0, 100))
    before = Finding("Archer at (11.3, 10.5)", "warning", unit_key=(1, 201))
    after = Finding("Archer at (14.3, 10.5)", "warning", unit_key=(1, 201))
    noise = CheckResult("Elevation", (Finding("a", "error", tile=(1, 1)), Finding("b", "error", tile=(2, 2))))
    dialog = AnalysisDialog()
    dialog.set_report(AnalysisReport((noise, CheckResult("Off-grid units", (other, before)))))
    elevation, _rows_e = _rows(dialog, "Elevation")
    elevation.setExpanded(False)
    dialog.tree.setCurrentItem(_rows(dialog, "Off-grid units")[1][1])

    # Same unit in the same check, in another check (not kept) it would be a different finding.
    dialog.set_report(AnalysisReport((noise, CheckResult("Off-grid units", (after, other)))))
    group, rows = _rows(dialog, "Off-grid units")
    assert dialog.tree.currentItem() is rows[0]
    assert dialog.tree.currentItem().data(0, Qt.UserRole) == after
    assert not _rows(dialog, "Elevation")[0].isExpanded(), "the collapsed check re-opened"
    assert group.isExpanded()

    dialog.set_report(AnalysisReport((noise, CheckResult("Other check", (after,)))))
    assert dialog.tree.currentItem() is None
    dialog.deleteLater()


def test_a_refill_keeps_the_scroll_position_and_the_kept_row_in_view() -> None:
    conftest.ensure_qapp()
    from PyQt5.QtWidgets import QApplication

    from descape.analysis_dialog import AnalysisDialog
    from descape.map_analysis import AnalysisReport, CheckResult, Finding

    findings = tuple(Finding(f"finding {i}", "warning", tile=(i, 0)) for i in range(80))
    dialog = AnalysisDialog()
    try:
        dialog.set_report(AnalysisReport((CheckResult("Long check", findings),)))
        dialog.show()
        QApplication.processEvents()
        tree, bar = dialog.tree, dialog.tree.verticalScrollBar()
        assert bar.maximum() > 40, "the fixture must scroll"
        bar.setValue(40)
        tree.setCurrentItem(_rows(dialog, "Long check")[1][42])
        assert bar.value() == 40, "row 42 is already in view at 40"

        dialog.set_report(AnalysisReport((CheckResult("Long check", findings),)))
        assert bar.value() == 40
        assert tree.currentItem().text(1) == "finding 42"

        # Nothing current: the position alone is kept.
        tree.setCurrentItem(None)
        dialog.set_report(AnalysisReport((CheckResult("Long check", findings),)))
        assert bar.value() == 40 and tree.currentItem() is None

        # Thirty rows above the kept one are fixed: the old position would hide it.
        tree.setCurrentItem(_rows(dialog, "Long check")[1][42])
        dialog.set_report(AnalysisReport((CheckResult("Long check", findings[30:]),)))
        kept = tree.currentItem()
        assert kept.text(1) == "finding 42"
        assert tree.viewport().rect().contains(tree.visualItemRect(kept).center())
    finally:
        dialog.close()
        dialog.deleteLater()


def test_refresh_reruns_on_the_edited_map_and_keeps_the_row_its_ring_and_the_focus() -> None:
    from PyQt5.QtCore import Qt

    window = _marker_window()
    try:
        view = window.map_view
        window.analysis_action.trigger()
        dialog = window._analysis_dialog
        _group, rows = _rows(dialog, "Illegal elevation steps")
        before = len(rows)
        kept = rows[-1].data(0, Qt.UserRole)
        dialog.tree.setCurrentItem(rows[-1])
        assert view._analysis_focus == kept.tile and len(view._analysis_markers) == 5

        # A second bump, earlier in the scan: its findings land above the kept row.
        mm = window.scenario.map_manager
        mm.terrain[5 * mm.map_width + 5].elevation = 2
        dialog.refresh_button.setFocus()
        dialog.refresh_button.click()

        assert window._analysis_dialog is dialog and dialog.isVisible()
        _group, rows = _rows(dialog, "Illegal elevation steps")
        assert len(rows) == 2 * before
        assert view._analysis_markers == _expected_anchors(window) and len(view._analysis_markers) == 10
        current = dialog.tree.currentItem()
        assert current.data(0, Qt.UserRole) == kept and current is not rows[before - 1]
        assert view._analysis_focus == kept.tile, "the kept row's marker lost its ring"
        assert view.analysis_marker_item()._focus == _top_vertex(view, kept.tile)

        # The window disabled itself, and the dialog with it, for the run.
        assert window.isEnabled() and not window._busy
        assert dialog.isEnabled() and dialog.refresh_button.isEnabled()
        assert dialog.focusWidget() is dialog.refresh_button
        assert not window.edit_history.is_dirty
    finally:
        conftest.close_window(window)


def test_refresh_with_no_map_loaded_changes_nothing(monkeypatch) -> None:
    window = _marker_window()
    try:
        window.analysis_action.trigger()
        dialog = window._analysis_dialog
        headline = dialog.headline.text()
        count = dialog.tree.topLevelItemCount()
        # Not reachable from the UI (closing the map closes the dialog); the callback must cope anyway.
        monkeypatch.setattr(window, "scenario", None)
        dialog.refresh_button.click()
        assert dialog.headline.text() == headline and dialog.tree.topLevelItemCount() == count
        assert dialog.isEnabled() and window.isEnabled() and not window._busy
    finally:
        conftest.close_window(window)


def test_menu_action_is_gated_on_a_loaded_map_and_registered_as_a_keybind() -> None:
    from descape import settings
    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    try:
        assert not window.analysis_action.isEnabled()
        assert window._keybind_actions["analysis_run"] is window.analysis_action
        assert "analysis_run" in {a for a, _, _ in settings.REBINDABLE_ACTIONS}
        menus = [a.text() for a in window.menuBar().actions()]
        assert menus.index("&Tools") == menus.index("&View") + 1
        assert menus.index("&Help") == menus.index("&Tools") + 1
        window.load_scenario(UNITS_FIXTURE)
        assert window.analysis_action.isEnabled()
    finally:
        conftest.close_window(window)


def test_show_analysis_opens_a_modeless_reused_dialog_that_closes_with_the_map() -> None:
    window = _window()
    try:
        window.analysis_action.trigger()
        dialog = window._analysis_dialog
        assert dialog is not None and dialog.isVisible()
        assert not dialog.isModal()
        assert "units_120x120.aoe2scenario: Clean on" in dialog.headline.text()
        assert dialog.tree.topLevelItemCount() > 0

        window.analysis_action.trigger()
        assert window._analysis_dialog is dialog

        window.close_scenario()
        assert window._analysis_dialog is None
    finally:
        conftest.close_window(window)


def test_running_and_browsing_an_analysis_leaves_the_document_clean() -> None:
    """GH #90 also-check: analysis is read-only, so no "*" in the title."""
    window = _marker_window()
    try:
        window._update_title()
        title = window.windowTitle()
        assert "*" not in title
        window.analysis_action.trigger()
        tree = window._analysis_dialog.tree
        group = next(tree.topLevelItem(i) for i in range(tree.topLevelItemCount()) if tree.topLevelItem(i).childCount())
        tree.setCurrentItem(group.child(0))
        tree.itemDoubleClicked.emit(group.child(0), 0)

        assert not window.edit_history.is_dirty
        assert window.windowTitle() == title
    finally:
        conftest.close_window(window)


def test_navigating_to_a_unit_finding_switches_to_units_mode_and_selects_it() -> None:
    from descape.map_analysis import Finding

    window = _window()
    try:
        assert window.mode_combo.currentText() != "Units"
        window._navigate_to_finding(Finding("x", "warning", unit_key=(1, 201)))
        assert window.mode_combo.currentText() == "Units"
        assert window._selection == [(1, 201)]
    finally:
        conftest.close_window(window)


def test_the_invalid_position_rows_list_the_fixture_archer_and_select_it() -> None:
    """GH #173: three group rows, the overlap one holding Archer 201 inside House 200 by name."""
    window = _window()
    try:
        window.analysis_action.trigger()
        dialog = window._analysis_dialog
        for label in ("Units on terrain they can't stand on", "Buildings on terrain they can't stand on"):
            group, rows = _rows(dialog, label)
            assert group.text(0).endswith("(clean)") and rows == []
        group, rows = _rows(dialog, "Units overlapping another object")
        assert group.text(0) == "Units overlapping another object (1)"
        assert (rows[0].text(0), rows[0].text(1)) == (
            "Info",
            "Player 1 unit 4 (id 201) at (11.5, 10.5) overlaps House (Player 1) at (10.5, 10.5)",
        )
        dialog.tree.itemDoubleClicked.emit(rows[0], 0)
        assert window.mode_combo.currentText() == "Units"
        assert window._selection == [(1, 201)]
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("style", ["Flat top-down", "Flat", "Stepped", "Sloped"])
def test_center_on_tile_centres_the_tile_in_every_terrain_style(style: str) -> None:
    from PyQt5.QtWidgets import QApplication

    window = _window()
    try:
        window.terrain_style_combo.setCurrentText(style.split(maxsplit=1)[0])
        if style.startswith("Flat"):
            window.iso_action.setChecked(style == "Flat")
        QApplication.processEvents()
        view = window.map_view
        # Zoomed in: at fit-to-window the whole map is visible and there is nothing to scroll.
        view.scale(8.0, 8.0)
        QApplication.processEvents()
        # Within 3 screen pixels: centerOn() rounds through integer scrollbar values.
        tolerance = 3.0 / view.transform().m11()
        for tile in [(100, 15), (20, 90)]:
            view.center_on_tile(*tile)
            QApplication.processEvents()
            poly_center = view._tile_polygon(*tile).boundingRect().center()
            view_center = view.mapToScene(view.viewport().rect().center())
            assert abs(view_center.x() - poly_center.x()) < tolerance, (style, tile, view_center, poly_center)
            assert abs(view_center.y() - poly_center.y()) < tolerance, (style, tile, view_center, poly_center)
    finally:
        conftest.close_window(window)


# --- on-map markers ---------------------------------------------------------

_BUMP = (20, 12)


def _marker_window():
    """The blank template with one tile raised to 2 among 0s: 8 elevation
    errors on 5 distinct tiles, and nothing else located."""
    from descape.scenario_io import BLANK_TEMPLATE_PATH

    window = conftest.shown_window(900, 700)
    window.load_scenario(BLANK_TEMPLATE_PATH)
    assert window.scenario is not None
    mm = window.scenario.map_manager
    mm.terrain[_BUMP[1] * mm.map_width + _BUMP[0]].elevation = 2
    return window


def _expected_anchors(window):
    from descape import map_analysis

    mm = window.scenario.map_manager
    return map_analysis.marker_anchors(map_analysis.analyze(window.scenario), mm.map_width, mm.map_height)


def _top_vertex(view, tile):
    from PyQt5.QtCore import QPointF

    rect = view._tile_polygon(*tile).boundingRect()
    return QPointF(rect.center().x(), rect.top())


def test_show_analysis_installs_one_marker_per_located_tile() -> None:
    window = _marker_window()
    try:
        window.analysis_action.trigger()
        item = window.map_view.analysis_marker_item()
        expected = _expected_anchors(window)
        assert len(expected) == 5
        assert item is not None and item.zValue() == window.map_view.ANALYSIS_MARKER_Z
        assert [(s, n) for _p, s, n in item.marker_points()] == [(s, n) for _t, s, n in expected]
        assert window.map_view._analysis_markers == expected
    finally:
        conftest.close_window(window)


def test_closing_the_dialog_removes_the_markers_and_a_rerun_replaces_them() -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest

    window = _marker_window()
    try:
        view = window.map_view
        window.analysis_action.trigger()
        window.analysis_action.trigger()
        assert len(view.analysis_marker_item().marker_points()) == 5, "a re-run appended"

        window._analysis_dialog.close()
        assert view.analysis_marker_item() is None
        assert view._analysis_markers == []

        # Escape reaches reject() without a closeEvent; the markers must still go.
        window.analysis_action.trigger()
        assert view.analysis_marker_item() is not None
        QTest.keyClick(window._analysis_dialog, Qt.Key_Escape)
        assert not window._analysis_dialog.isVisible()
        assert view.analysis_marker_item() is None
        assert not [i for i in view.scene().items() if type(i).__name__ == "AnalysisMarkerItem"]
    finally:
        conftest.close_window(window)


def test_close_and_load_leave_no_markers() -> None:
    from descape.scenario_io import BLANK_TEMPLATE_PATH

    window = _marker_window()
    try:
        view = window.map_view
        window.analysis_action.trigger()
        window.close_scenario()
        assert view.analysis_marker_item() is None and view._analysis_markers == []

        window.load_scenario(BLANK_TEMPLATE_PATH)
        mm = window.scenario.map_manager
        mm.terrain[_BUMP[1] * mm.map_width + _BUMP[0]].elevation = 2
        window.analysis_action.trigger()
        assert view.analysis_marker_item() is not None
        window.load_scenario(BLANK_TEMPLATE_PATH)
        assert view.analysis_marker_item() is None and view._analysis_markers == []
    finally:
        conftest.close_window(window)


def test_markers_survive_a_style_switch_at_each_styles_top_vertex() -> None:
    window = _marker_window()
    try:
        view = window.map_view
        window.analysis_action.trigger()
        tiles = [t for t, _s, _n in view._analysis_markers]
        for style in ("Stepped", "Sloped", "Flat"):
            window.terrain_style_combo.setCurrentText(style)
            item = view.analysis_marker_item()
            assert item is not None, style
            points = [p for p, _s, _n in item.marker_points()]
            assert points == [_top_vertex(view, t) for t in tiles], style
        window.iso_action.setChecked(False)
        points = [p for p, _s, _n in view.analysis_marker_item().marker_points()]
        assert points == [_top_vertex(view, t) for t in tiles], "Flat top-down"
    finally:
        conftest.close_window(window)


def test_an_elevation_edit_moves_a_marker_in_stepped() -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    window = _marker_window()
    try:
        view = window.map_view
        window.terrain_style_combo.setCurrentText("Stepped")
        window.analysis_action.trigger()
        tile = view._analysis_markers[-1][0]
        index = [t for t, _s, _n in view._analysis_markers].index(tile)
        before = view.analysis_marker_item().marker_points()[index][0]
        window.mode_combo.setCurrentText("Terrain")
        window._on_tool_selected("elevation")
        window.on_edit_stroke_start()
        window.on_edit_stroke_tile(tile[0], tile[1], Qt.NoModifier)
        window.on_edit_stroke_end()
        QApplication.processEvents()
        after = view.analysis_marker_item().marker_points()[index][0]
        assert after == _top_vertex(view, tile)
        assert after != before, "the marker stayed at the pre-edit ground height"
    finally:
        conftest.close_window(window)


def test_selecting_a_row_focuses_its_marker_and_a_group_row_clears_it() -> None:
    window = _marker_window()
    try:
        view = window.map_view
        window.analysis_action.trigger()
        tree = window._analysis_dialog.tree
        group = next(tree.topLevelItem(i) for i in range(tree.topLevelItemCount()) if tree.topLevelItem(i).childCount())
        row = group.child(0)
        tree.setCurrentItem(row)
        from PyQt5.QtCore import Qt

        tile = row.data(0, Qt.UserRole).tile
        assert view._analysis_focus == tile
        assert view.analysis_marker_item()._focus == _top_vertex(view, tile)
        tree.setCurrentItem(group)
        assert view._analysis_focus is None
        assert view.analysis_marker_item()._focus is None
    finally:
        conftest.close_window(window)


def _colour_hits(array, rgb, tolerance=40):
    import numpy as np

    return (np.abs(array.astype(np.int16) - np.array(rgb, dtype=np.int16)) <= tolerance).all(axis=2)


def test_the_marker_colour_setting_paints_and_repaints(tmp_path, monkeypatch) -> None:
    """The ruler-colour pixel test's shape: pin a distinctive colour before
    the window opens, check the capture near the marker, then live-apply a
    second colour and check it repaints. The focus ring gives solid pixels."""
    from PyQt5.QtGui import QImage
    from PyQt5.QtWidgets import QApplication

    from descape import settings
    from testkit.qt_capture import qimage_rgb888_to_array

    (tmp_path / "config.yaml").write_text("overlay_colors:\n  analysis_marker: '#ff00ff'\n")
    monkeypatch.setattr(settings, "_overlay_colors", None)
    window = _marker_window()
    try:
        view = window.map_view
        window.analysis_action.trigger()
        view.set_analysis_marker_focus(_BUMP)
        view.center_on_tile(*_BUMP)
        QApplication.processEvents()

        def capture():
            QApplication.processEvents()
            image = view.viewport().grab().toImage().convertToFormat(QImage.Format_RGB888)
            return qimage_rgb888_to_array(image)

        def near_marker(hits):
            anchor = view.mapFromScene(_top_vertex(view, _BUMP))
            y0, x0 = max(0, anchor.y() - 40), max(0, anchor.x() - 50)
            return hits[y0 : anchor.y() + 5, x0 : anchor.x() + 50]

        array = capture()
        default = tuple(int(settings.get_default_overlay_color("analysis_marker")[i : i + 2], 16) for i in (1, 3, 5))
        assert near_marker(_colour_hits(array, (255, 0, 255))).sum() >= 20, "configured colour missing"
        assert not _colour_hits(array, default).any(), "default marker colour still showing"

        settings.set_overlay_color("analysis_marker", "#00ffff")
        view.apply_overlay_colors()
        array = capture()
        assert near_marker(_colour_hits(array, (0, 255, 255))).sum() >= 20, "live colour change did not repaint"
        assert not _colour_hits(array, (255, 0, 255), tolerance=8).any(), "old colour still showing"
    finally:
        conftest.close_window(window)


# --- the owed in-app passes, offscreen (TASK-097) ------------------------------


def _marker_boxes_at_scale(window, monkeypatch, linear_scale: float) -> list:
    """Zoom MapView to `linear_scale` on the bump, repaint, and return the device-space marker
    boxes AnalysisMarkerItem.paint() drew (its own drawRoundedRect calls, focus ring excluded)."""
    from PyQt5.QtGui import QPainter
    from PyQt5.QtWidgets import QApplication

    from descape.viewer_canvas import AnalysisMarkerItem

    view = window.map_view
    current = abs(view.transform().determinant()) ** 0.5
    view.scale(linear_scale / current, linear_scale / current)
    view.center_on_tile(*_BUMP)
    QApplication.processEvents()
    boxes: list = []
    inside = {"on": False}
    real_round = QPainter.drawRoundedRect
    real_paint = AnalysisMarkerItem.paint

    def spy_round(painter, rect, *args):
        if inside["on"]:
            boxes.append((round(rect.width(), 3), round(rect.height(), 3)))
        return real_round(painter, rect, *args)

    def spy_paint(item, painter, option, widget=None):
        inside["on"] = True
        try:
            return real_paint(item, painter, option, widget)
        finally:
            inside["on"] = False

    with monkeypatch.context() as patch:
        patch.setattr(QPainter, "drawRoundedRect", spy_round)
        patch.setattr(AnalysisMarkerItem, "paint", spy_paint)
        view.viewport().grab()
    assert abs(abs(view.transform().determinant()) ** 0.5 - linear_scale) < 1e-6 * linear_scale
    return boxes


def test_markers_keep_one_device_size_from_the_furthest_zoom_out_to_the_furthest_zoom_in(monkeypatch) -> None:
    """TASK-097 step 3, at MapView's own zoom limits (more than 100x apart), no focus ring.

    Pass conditions:
    - markers are drawn at both limits;
    - every box drawn at the furthest zoom in has a size that a box at the furthest zoom out
      also has, and every box is one text line tall at the marker font (font height + padding).
    """
    from PyQt5.QtGui import QFontMetricsF

    from descape.viewer_canvas import AnalysisMarkerItem

    window = _marker_window()
    try:
        view = window.map_view
        window.analysis_action.trigger()
        item = view.analysis_marker_item()
        lo, hi = view._min_linear_scale, view._max_linear_scale
        assert hi / lo > 100
        far = _marker_boxes_at_scale(window, monkeypatch, lo)
        assert item.markers_drawn >= 1 and far
        near = _marker_boxes_at_scale(window, monkeypatch, hi)
        assert item.markers_drawn >= 1 and near
        assert set(near) <= set(far), (near, far)
        height = QFontMetricsF(item._font).height() + AnalysisMarkerItem.PAD_PX
        assert all(abs(h - height) < 1.0 for _w, h in far + near), (height, far, near)
    finally:
        conftest.close_window(window)


def test_the_map_analysis_marker_swatch_recolours_the_live_markers(monkeypatch) -> None:
    """TASK-097 step 7 through Settings > Appearance's own "Map Analysis markers" swatch.
    Pass conditions: picking #00ffff persists it and the live marker item draws in it at once;
    the row's Default button puts back the shipped colour on both."""
    from PyQt5.QtGui import QColor
    from PyQt5.QtWidgets import QColorDialog

    from descape import settings
    from descape.viewer import SettingsDialog

    window = _marker_window()
    dialog = SettingsDialog(window)
    try:
        window.analysis_action.trigger()
        item = window.map_view.analysis_marker_item()
        default = settings.get_default_overlay_color("analysis_marker")
        assert item.color().name() == default
        monkeypatch.setattr(QColorDialog, "getColor", staticmethod(lambda *a, **k: QColor("#00ffff")))
        swatch = dialog._overlay_swatches["analysis_marker"]
        swatch.click()
        assert settings.get_overlay_color("analysis_marker") == "#00ffff"
        assert window.map_view.analysis_marker_item() is item
        assert item.color().name() == "#00ffff"
        grid = swatch.parentWidget().layout()
        row, column, _rs, _cs = grid.getItemPosition(grid.indexOf(swatch))
        grid.itemAtPosition(row, column + 1).widget().click()
        assert settings.get_overlay_color("analysis_marker") == default
        assert item.color().name() == default
    finally:
        dialog.close()
        conftest.close_window(window)
