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
