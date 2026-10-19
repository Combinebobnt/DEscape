"""GH #142: entering a panel mode from a narrow pane widens it to the panel's
floor, and at that floor nothing the user needs is cut off.

The `font_sensitive` tests here also run at 144 DPI (Windows 150% scaling)
through tests/test_font_dpi_sweep.py; `DESCAPE_TEST_FONT_DPI=144` runs them at
that DPI directly. tests/test_view_info_floor.py holds View's own widen.
"""

from __future__ import annotations

import pytest
from test_unit_edit_viewer import _no_install_env  # noqa: F401 -- autouse fixture

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_NARROW_PX = 150


def _panel_class(mode: str):
    from descape.diplomacy_panel import DiplomacyPanel
    from descape.messages_panel import MessagesPanel
    from descape.players_panel import PlayersPanel
    from descape.units_panel import UnitsPanel

    return {"Units": UnitsPanel, "Players": PlayersPanel, "Diplomacy": DiplomacyPanel, "Messages": MessagesPanel}[mode]


def _set_left(window, px: int) -> None:
    from PyQt5.QtWidgets import QApplication

    total = sum(window.content_splitter.sizes())
    window.content_splitter.setSizes([px, total - px])
    QApplication.processEvents()
    assert window.content_splitter.sizes()[0] == px, "the pane setup did not take"


def _enter(window, mode: str, *, from_px: int, from_mode: str = "View") -> int:
    """From `from_mode` with the left pane at `from_px`, switch to `mode`; returns the pane width after."""
    from PyQt5.QtWidgets import QApplication

    assert from_mode != mode
    window.mode_combo.setCurrentText(from_mode)
    _set_left(window, from_px)
    window.mode_combo.setCurrentText(mode)
    QApplication.processEvents()
    assert window.mode_combo.currentText() == mode
    return window.content_splitter.sizes()[0]


def _blank_window():
    from descape.scenario_io import BLANK_TEMPLATE_PATH

    window = conftest.shown_window(1500, 900)
    window.load_scenario(BLANK_TEMPLATE_PATH)
    assert window.scenario is not None
    return window


@pytest.mark.parametrize(("mode", "base_px"), [("Units", 380), ("Players", 320), ("Diplomacy", 430), ("Messages", 300)])
def test_entering_a_panel_mode_from_a_narrow_pane_widens_it_to_the_floor(mode: str, base_px: int) -> None:
    """Pass conditions, at the pinned test font: the panel's floor is its GH #142 base width,
    and entering the mode from a 150 px pane leaves the pane at exactly that width."""
    window = _blank_window()
    try:
        assert base_px == _panel_class(mode).MIN_USEFUL_WIDTH
        assert _enter(window, mode, from_px=_NARROW_PX) == base_px
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("mode", ["Units", "Players", "Diplomacy", "Messages"])
def test_entering_a_panel_mode_leaves_a_wider_pane_alone(mode: str) -> None:
    """Pass condition: a 600 px pane (wider than every floor) is still 600 px after entering."""
    window = _blank_window()
    try:
        assert _panel_class(mode).MIN_USEFUL_WIDTH < 600
        assert _enter(window, mode, from_px=600) == 600
    finally:
        conftest.close_window(window)


def _check_messages_fit_the_floor(*, string_ids: bool) -> None:
    """Pass conditions, at the pane width entering Messages from a narrow pane gives:
    - the form's minimum width fits the scroll area's viewport, and no horizontal scroll bar shows;
    - every field editor (and, with `string_ids`, every "Use my text" button) lies wholly inside
      the viewport at no less than its own minimum width.
    """
    from PyQt5.QtCore import QPoint
    from PyQt5.QtWidgets import QApplication

    from descape.messages_fields import MESSAGE_FIELDS

    window = _blank_window()
    try:
        assert _enter(window, "Messages", from_px=_NARROW_PX) == _panel_class("Messages").MIN_USEFUL_WIDTH
        panel = window.messages_panel
        if string_ids:
            ids = {f"{spec.field_id}_id": 123456 for spec in MESSAGE_FIELDS}
            panel.show_scenario(panel._loaded, values=ids, editable=True)
            QApplication.processEvents()
            assert len(panel._clear_id_buttons) == len(MESSAGE_FIELDS)
        viewport = panel.area.viewport()
        for spec in MESSAGE_FIELDS:
            widgets = [panel.widget_for(spec.field_id)]
            if string_ids:
                widgets.append(panel._clear_id_buttons[spec.field_id])
            for widget in widgets:
                left = widget.mapTo(viewport, QPoint(0, 0)).x()
                assert left >= 0 and left + widget.width() <= viewport.width(), (spec.field_id, left, widget.width())
                assert widget.width() >= widget.minimumSizeHint().width(), (spec.field_id, widget)
        assert panel.host.minimumSizeHint().width() <= viewport.width()
        assert not panel.area.horizontalScrollBar().isVisible()
    finally:
        conftest.close_window(window)


@pytest.mark.font_sensitive
def test_messages_fits_the_floor_with_no_horizontal_scroll() -> None:
    """The six text fields as a file with no language-file string ids shows them."""
    _check_messages_fit_the_floor(string_ids=False)


@pytest.mark.xfail(
    strict=True,
    reason='DEFECT (TASK-149 sweep): at the 300 px Messages floor a field with a language-file string id '
    'set pushes its "Use my text (clear string ID)" button past the viewport, with no horizontal scroll bar',
)
def test_messages_with_a_string_id_set_fits_the_floor_with_no_horizontal_scroll() -> None:
    """Every field showing its widest row: the language-file warning beside "Use my text"."""
    _check_messages_fit_the_floor(string_ids=True)


@pytest.mark.xfail(
    strict=True,
    reason="DEFECT (TASK-149 sweep): at the 210 px View floor the stats table's Stat column (180 px at "
    "96 DPI) leaves the Value column mostly past the table's viewport",
)
def test_the_view_stats_value_column_shows_in_full_at_the_floor() -> None:
    """The blank template's Player 1 breakdown, at the pinned 96 DPI.

    Pass conditions, after entering View from a narrow pane:
    - the stats table has value rows;
    - the Value column ends inside the table's viewport;
    - every value's text fits the Value column's width.
    """
    from PyQt5.QtWidgets import QApplication

    window = _blank_window()
    try:
        assert _enter(window, "View", from_px=_NARROW_PX, from_mode="Units") == window.INFO_PAGE_MIN_USEFUL_WIDTH
        combo = window.stats_player_combo
        combo.setCurrentIndex(combo.findData(1))
        QApplication.processEvents()
        tree = window.info_tree
        header = tree.header()
        values = [count for _label, count, _note, _parent in window.player_stats_rows if count]
        assert values, "no value rows in the stats table"
        right = header.sectionViewportPosition(1) + header.sectionSize(1)
        assert right <= tree.viewport().width(), f"Value column ends at {right} px of {tree.viewport().width()}"
        metrics = tree.fontMetrics()
        widest = max(metrics.horizontalAdvance(v) for v in values)
        assert widest <= header.sectionSize(1), f"widest value {widest} px, column {header.sectionSize(1)} px"
    finally:
        conftest.close_window(window)


@pytest.mark.font_sensitive
def test_the_garrison_buttons_show_in_full_with_a_tower_selected() -> None:
    """Units mode entered from a narrow pane, a Watch Tower holding two units selected.

    Pass conditions: the Garrison block is shown; every visible button in its button rows lies
    wholly inside the inspector's viewport at no less than its own minimum width; no
    horizontal scroll bar shows.
    """
    import test_unit_edit_viewer as viewer_tests
    from PyQt5.QtCore import QPoint
    from PyQt5.QtWidgets import QApplication, QPushButton

    window = viewer_tests._window()
    try:
        window.resize(1500, 900)
        window.show()
        QApplication.processEvents()
        assert _enter(window, "Units", from_px=_NARROW_PX) == _panel_class("Units").MIN_USEFUL_WIDTH
        tower, _occupants = viewer_tests._tower_with_two_inside(window)
        viewer_tests._hide_garrisoned(window)
        window._selection = [(1, tower.reference_id)]
        window._refresh_selection_view()
        QApplication.processEvents()
        panel = window.units_panel
        assert panel.garrison_buttons.isVisible()
        viewport = panel.inspector_area.viewport()
        buttons = [b for row in panel.garrison_button_rows for b in row.findChildren(QPushButton) if b.isVisible()]
        assert len(buttons) >= 3
        for button in buttons:
            left = button.mapTo(viewport, QPoint(0, 0)).x()
            assert left >= 0 and left + button.width() <= viewport.width(), (button.text(), left, button.width())
            assert button.width() >= button.minimumSizeHint().width(), button.text()
        assert not panel.inspector_area.horizontalScrollBar().isVisible()
    finally:
        viewer_tests._close(window)
