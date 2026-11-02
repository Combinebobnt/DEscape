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
# The View tests start from a narrow pane in a mode with the plain MIN_SPLIT_PANE drag floor:
# Units (TASK-362) and Terrain (TASK-551) have their pages' own.
_PLAIN_FLOOR_MODE = "Players"


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


@pytest.mark.font_sensitive
def test_messages_with_a_string_id_set_fits_the_floor_with_no_horizontal_scroll() -> None:
    """Every field showing its widest row: the language-file warning beside "Use my text"."""
    _check_messages_fit_the_floor(string_ids=True)


@pytest.mark.font_sensitive
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
        assert _enter(window, "View", from_px=_NARROW_PX, from_mode=_PLAIN_FLOOR_MODE) == window.INFO_PAGE_MIN_USEFUL_WIDTH
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


def _show_view_player(window, player_id: int) -> None:
    from PyQt5.QtWidgets import QApplication

    combo = window.stats_player_combo
    combo.setCurrentIndex(combo.findData(player_id))
    QApplication.processEvents()


def _stat_rows(tree) -> list:
    """Every three-column row: not a section header, not free text, not the combo's row."""
    from PyQt5.QtWidgets import QTreeWidgetItemIterator

    rows = []
    iterator = QTreeWidgetItemIterator(tree)
    while iterator.value() is not None:
        item = iterator.value()
        if not item.isFirstColumnSpanned() and tree.itemWidget(item, 0) is None:
            rows.append(item)
        iterator += 1
    return rows


@pytest.mark.font_sensitive
@pytest.mark.parametrize("player_id", [1, 0])
def test_the_view_table_fits_the_floor_with_note_hidden_and_full_labels_in_tooltips(player_id: int) -> None:
    """TASK-407, Player 1 and GAIA at View's floor. Pass conditions:
    - no horizontal scroll bar, and the Note column is hidden;
    - every Stat row's column 0 tooltip is its full label, plus its note when it has one, so an
      elided label ("Scenario version" at 96 DPI) and a hidden note stay readable.
    """
    window = _blank_window()
    try:
        assert _enter(window, "View", from_px=_NARROW_PX, from_mode=_PLAIN_FLOOR_MODE) == window.INFO_PAGE_MIN_USEFUL_WIDTH
        _show_view_player(window, player_id)
        tree = window.info_tree
        assert not tree.horizontalScrollBar().isVisible()
        assert tree.header().isSectionHidden(2)
        rows = _stat_rows(tree)
        assert rows
        for item in rows:
            note = item.text(2)
            assert item.toolTip(0) == (f"{item.text(0)}: {note}" if note else item.text(0)), item.text(0)
    finally:
        conftest.close_window(window)


@pytest.mark.font_sensitive
def test_a_wide_view_pane_shows_note_and_every_stat_label_in_full() -> None:
    """TASK-407 at a 600 px pane, Player 1. Pass conditions: Note shown; Stat at its content
    width and Value at no less than its own; no horizontal scroll bar."""
    window = _blank_window()
    try:
        window.mode_combo.setCurrentText("View")
        _set_left(window, 600)
        _show_view_player(window, 1)
        tree = window.info_tree
        assert not tree.header().isSectionHidden(2)
        assert tree.columnWidth(0) == tree.sizeHintForColumn(0)
        assert tree.columnWidth(1) >= tree.sizeHintForColumn(1)
        assert not tree.horizontalScrollBar().isVisible()
    finally:
        conftest.close_window(window)


@pytest.mark.font_sensitive
def test_dragging_the_view_pane_narrow_wide_narrow_hides_shows_and_hides_note() -> None:
    """TASK-407: the Note column's hiding follows the pane width, never sticks.
    Pass conditions: at the floor, 600 px, then the floor again, Note is hidden, shown, hidden,
    with no horizontal scroll bar at any of the three."""
    window = _blank_window()
    try:
        floor = _enter(window, "View", from_px=_NARROW_PX, from_mode=_PLAIN_FLOOR_MODE)
        _show_view_player(window, 1)
        tree = window.info_tree
        seen = []
        for px in (floor, 600, floor):
            _set_left(window, px)
            seen.append(tree.header().isSectionHidden(2))
            assert not tree.horizontalScrollBar().isVisible(), px
        assert seen == [True, False, True]
    finally:
        conftest.close_window(window)


@pytest.mark.font_sensitive
def test_a_bigger_view_table_font_still_fits_the_floor() -> None:
    """TASK-407: a 1.25x tree font at View's floor (Player 1) leaves no horizontal scroll bar.
    The floor follows the app font only (FontScaledWidth), so this checks fit_columns() re-fits on
    a tree font change, including after a vertical scroll bar it brings narrows the viewport
    (TASK-409). 1.5x left 5-12 px at 96-120 DPI on Linux; 1.25x leaves 30+ px at every swept DPI."""
    from PyQt5.QtWidgets import QApplication

    window = _blank_window()
    try:
        _enter(window, "View", from_px=_NARROW_PX, from_mode=_PLAIN_FLOOR_MODE)
        _show_view_player(window, 1)
        tree = window.info_tree
        font = tree.font()
        font.setPointSizeF(font.pointSizeF() * 1.25)
        tree.setFont(font)
        QApplication.processEvents()
        room = tree.viewport().width() - tree.sizeHintForColumn(1)
        least = 3 * tree.indentation() + tree._STAT_MIN_CHARS * tree.fontMetrics().averageCharWidth()
        # The column sum, not only the bar: Linux paints the bar a turn after the overflow (TASK-409).
        shown = sum(tree.columnWidth(c) for c in range(3) if not tree.header().isSectionHidden(c))
        assert shown <= tree.viewport().width(), f"columns {shown} px in a {tree.viewport().width()} px viewport"
        assert not tree.horizontalScrollBar().isVisible(), f"Stat room {room} px, its minimum {least} px"
    finally:
        conftest.close_window(window)


@pytest.mark.font_sensitive
def test_a_long_player_label_elides_inside_the_view_floor() -> None:
    """TASK-407: Player 1 labelled with 40+ characters, at View's floor. Pass conditions: the
    combo lies wholly inside the table's viewport with no horizontal scroll bar; it paints an
    elided label; its popup's item text is the full label."""
    from PyQt5.QtCore import QPoint
    from PyQt5.QtWidgets import QApplication

    from descape import player_labels

    long = "P1 - The Very Long Tribe Name Of The North Coast"
    assert len(long) >= 40
    window = _blank_window()
    try:
        base = player_labels.DEFAULT_LABELS
        window._player_labels = player_labels.PlayerLabels(
            (base.text[0], long, *base.text[2:]), (base.full[0], long, *base.full[2:])
        )
        window._relabel_stats_players()
        _enter(window, "View", from_px=_NARROW_PX, from_mode=_PLAIN_FLOOR_MODE)
        _show_view_player(window, 1)
        QApplication.processEvents()
        tree = window.info_tree
        combo = window.stats_player_combo
        viewport = tree.viewport()
        left = combo.mapTo(viewport, QPoint(0, 0)).x()
        assert left >= 0 and left + combo.width() <= viewport.width(), (left, combo.width(), viewport.width())
        assert not tree.horizontalScrollBar().isVisible()
        shown = combo.displayed_text()
        assert shown.endswith("…") and len(shown) < len(long), shown
        assert combo.view().model().index(combo.currentIndex(), 0).data() == long
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


# --- TASK-551: Terrain mode's drag floor ---------------------------------------


def _filter_placeholder_image(window):
    """The filter field's placeholder strip, unfocused (no caret): what a cut placeholder changes."""
    from PyQt5.QtCore import QRect
    from PyQt5.QtWidgets import QApplication

    edit = window.terrain_panel.view.filter_edit
    edit.clearFocus()
    QApplication.processEvents()
    advance = edit.fontMetrics().horizontalAdvance(edit.placeholderText())
    # The frame and text margins sit left of the text, so the strip runs a little past its advance.
    return edit.grab().toImage().copy(QRect(0, 0, advance + edit.height() // 2, edit.height()))


def check_terrain_fits_its_drag_floor(window) -> int:
    """Terrain mode with the pane dragged as narrow as the splitter allows, every group expanded
    and the unused terrains shown. Pass conditions:
    - the pane stops above MIN_SPLIT_PANE;
    - every heading and terrain name shows in full: the name column fits the tree's viewport and
      there is nothing to scroll horizontally;
    - the show-unused box and the List/Grid buttons lie wholly inside the page at no less than
      their size hints;
    - the filter's placeholder paints exactly as in a pane 200 px wider, so it is not elided.
    Returns the narrowest pane width."""
    from PyQt5.QtCore import QPoint
    from PyQt5.QtWidgets import QApplication

    from descape import settings

    window.mode_combo.setCurrentText("Terrain")
    # Relative, not a fixed 600 px: at 168 DPI the floor itself passes 600.
    _set_left(window, window.terrain_panel.drag_floor() + 200)
    wide_placeholder = _filter_placeholder_image(window)
    total = sum(window.content_splitter.sizes())
    window.content_splitter.setSizes([settings.MIN_SPLIT_PANE, total - settings.MIN_SPLIT_PANE])
    QApplication.processEvents()
    pane = window.content_splitter.sizes()[0]
    assert pane > settings.MIN_SPLIT_PANE, "Terrain has no drag floor"

    panel = window.terrain_panel
    view = panel.view
    view.show_hidden_checkbox.setChecked(True)
    view.tree.expandAll()
    QApplication.processEvents()
    tree = view.tree
    assert tree.header().sectionSize(0) <= tree.viewport().width(), (tree.header().sectionSize(0), tree.viewport().width())
    assert tree.horizontalScrollBar().maximum() == 0, "a name is cut at the floor"
    for widget in (view.show_hidden_checkbox, *view.mode_buttons.values()):
        left = widget.mapTo(panel, QPoint(0, 0)).x()
        assert left >= 0 and left + widget.width() <= panel.width(), (widget.text(), left, widget.width(), panel.width())
        assert widget.width() >= widget.sizeHint().width(), widget.text()
    assert _filter_placeholder_image(window) == wide_placeholder, "the filter's placeholder is cut at the floor"
    return pane


@pytest.mark.font_sensitive
def test_terrain_mode_cannot_be_dragged_below_its_names_and_controls() -> None:
    """GH #142 follow-up (TASK-551), at the pinned font: check_terrain_fits_its_drag_floor's
    conditions; leaving Terrain gives the pane back its plain floor."""
    from PyQt5.QtWidgets import QApplication

    from descape import settings

    window = _blank_window()
    try:
        floor = check_terrain_fits_its_drag_floor(window)
        assert floor < window.terrain_panel.MIN_USEFUL_WIDTH, "the drag floor should stay below Terrain's entry widen"
        window.mode_combo.setCurrentText("View")
        QApplication.processEvents()
        total = sum(window.content_splitter.sizes())
        window.content_splitter.setSizes([settings.MIN_SPLIT_PANE, total - settings.MIN_SPLIT_PANE])
        QApplication.processEvents()
        assert window.content_splitter.sizes()[0] == settings.MIN_SPLIT_PANE
    finally:
        conftest.close_window(window)
