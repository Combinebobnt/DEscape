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
