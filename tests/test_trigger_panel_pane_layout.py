"""Trigger panel: the button rows, minimum widths and the tree/detail splits."""

from __future__ import annotations

import pytest
from trigger_panel_helpers import (
    TRIGGER_FIXTURE,
    _no_install_env,  # noqa: F401 -- autouse fixture
    _row_widgets,
    _sectioned_window,
)

from testkit.qt_window import set_panel_width

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


# -- TASK-212, TASK-330: two button rows below the tree -----------------------


def _bottom_rows(panel) -> list:
    """The layout items after the tree in the trigger pane, in order."""
    pane_layout = panel.tree.parentWidget().layout()
    at = pane_layout.indexOf(panel.tree)
    return [pane_layout.itemAt(i) for i in range(at + 1, pane_layout.count())]


def test_two_button_rows_sit_below_the_trigger_tree() -> None:
    window = _sectioned_window()
    try:
        panel = window.trigger_panel
        rows = _bottom_rows(panel)
        assert len(rows) == 2, "TASK-212: five rows left the tree ~3 rows tall"
        assert [row.layout() for row in rows] == [panel.trigger_buttons, panel.reorder_buttons]
        reorder = [panel.trigger_copy_button, panel.trigger_move_up_button, panel.trigger_move_down_button,
                   panel.section_up_button, panel.section_down_button]
        assert _row_widgets(panel.reorder_buttons) == reorder
        assert [b.text() for b in reorder[1:]] == ["▲", "▼", "⇈", "⇊"]
        assert [b.accessibleName() for b in reorder[1:]] == ["Move Up", "Move Down", "Section Up", "Section Down"]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_every_trigger_verb_is_still_reachable(monkeypatch) -> None:
    """Each verb is a button in the pane or an action in the Sections or Tags
    menu, and triggering it reaches its slot."""
    window = _sectioned_window()
    try:
        panel = window.trigger_panel
        ops: list[str] = []
        monkeypatch.setattr(panel, "_request_trigger_op", ops.append)
        buttons = {
            "new": panel.trigger_new_button,
            "clipboard_copy": panel.trigger_clipboard_copy_button,
            "paste": panel.trigger_paste_button,
            "delete": panel.trigger_delete_button,
            "copy": panel.trigger_copy_button,
            "move_up": panel.trigger_move_up_button,
            "move_down": panel.trigger_move_down_button,
            "section_up": panel.section_up_button,
            "section_down": panel.section_down_button,
        }
        pane = panel.tree.parentWidget()
        for op, button in buttons.items():
            assert pane.isAncestorOf(button), op
            button.setEnabled(True)
            button.click()
            assert ops[-1] == op

        called: list[str] = []
        for slot in ("request_tag_add", "request_tag_remove_selected", "request_tag_rename", "request_tag_remove",
                     "request_new_section", "request_section_rename", "collapse_all_sections",
                     "expand_all_sections"):
            monkeypatch.setattr(panel, slot, lambda slot=slot: called.append(slot))
        assert pane.isAncestorOf(panel.tags_button)
        assert panel.tags_button.menu() is panel.tags_menu
        assert [a.text() for a in panel.tags_menu.actions() if not a.isSeparator()] == [
            "Add Tag to Selection…", "Remove Tag from Selection…", "Rename Tag…", "Remove Tag Everywhere",
        ]
        menu_actions = panel.sections_menu.actions() + panel.tags_menu.actions()
        for action in (panel.tag_add_action, panel.tag_remove_selected_action, panel.tag_rename_action,
                       panel.tag_remove_action, panel.section_new_action, panel.section_rename_action,
                       panel.collapse_all_action, panel.expand_all_action):
            assert action in menu_actions, action.text()
            action.setEnabled(True)
            action.trigger()
        assert called == ["request_tag_add", "request_tag_remove_selected", "request_tag_rename",
                          "request_tag_remove", "request_new_section", "request_section_rename",
                          "collapse_all_sections", "expand_all_sections"]

        assert pane.isAncestorOf(panel.sections_button)
        assert panel.sections_button.menu() is panel.sections_menu
        assert panel.section_move_action in panel.sections_menu.actions()
        assert panel.section_move_action.menu() is panel.section_move_menu
        panel.select_trigger(0)
        panel.section_move_menu.aboutToShow.emit()
        assert panel.section_move_menu.actions(), "the submenu lists the sections"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_no_trigger_pane_button_clips_at_the_minimum_width() -> None:
    from PyQt5.QtWidgets import QApplication

    from descape.trigger_panel import TriggerPanel

    window = _sectioned_window()
    try:
        panel = window.trigger_panel
        set_panel_width(window, TriggerPanel.MIN_USEFUL_WIDTH)
        QApplication.processEvents()
        assert panel.width() <= TriggerPanel.MIN_USEFUL_WIDTH + 10, "fixture assumption: the pane is at its minimum"
        tag_row = panel.tag_combo.parentWidget().layout().itemAt(1).layout()
        widgets = [w for row in _bottom_rows(panel) for w in _row_widgets(row.layout())] + _row_widgets(tag_row)
        assert len(widgets) == 12
        for widget in widgets:
            if widget is panel.tag_combo:
                # Elides by design, but must keep a usable width (188 px measured).
                assert widget.width() >= 170, f"the tag filter gets {widget.width()} px"
                continue
            assert widget.isVisible(), widget.text()
            assert widget.width() >= widget.sizeHint().width(), (
                f"{widget.text() or widget.accessibleName()!r} gets {widget.width()} px of "
                f"its {widget.sizeHint().width()} at {panel.width()} px"
            )

        # GH #213: the entry row. Its push buttons sit below Fusion's 80 px hint
        # on purpose, so the bar is their own fitted minimum, and the label inside it.
        entry_row = [*_row_widgets(panel.entry_buttons), panel.entry_move_up_button, panel.entry_move_down_button]
        assert [w.text() for w in entry_row] == ["New", "Duplicate", "Delete", "Type…", "▲", "▼"]
        right = 0
        for widget in entry_row:
            assert widget.isVisible(), widget.text()
            needed = widget.minimumSizeHint().width()
            assert widget.width() >= needed, f"{widget.text()!r} gets {widget.width()} px of its {needed}"
            assert widget.width() > widget.fontMetrics().horizontalAdvance(widget.text()), widget.text()
            left = widget.mapTo(panel, widget.rect().topLeft()).x()
            assert left >= right, f"{widget.text()!r} overlaps the button before it"
            right = left + widget.width()
        assert right <= panel.width(), f"the entry row runs to {right} px in a {panel.width()} px pane"
    finally:
        window.edit_history.mark_saved()
        window.close()


def _placeholder_ink_width(edit) -> int:
    """Columns of the empty line edit's inside (its 3 px frame band skipped)
    holding any pixel off its Base colour: the drawn placeholder's width."""
    from PyQt5.QtGui import QColor, QPalette

    assert edit.text() == "" and edit.placeholderText(), "fixture assumption: the placeholder is showing"
    image = edit.grab().toImage()
    base = edit.palette().color(QPalette.Base).rgb() & 0xFFFFFF
    inked = [
        x
        for x in range(3, image.width() - 3)
        if any(QColor(image.pixel(x, y)).rgb() & 0xFFFFFF != base for y in range(3, image.height() - 3))
    ]
    assert inked, "no placeholder ink drawn"
    return inked[-1] - inked[0] + 1


def test_the_filter_row_does_not_clip_at_the_minimum_width() -> None:
    """TASK-068's filter row, the sibling of the tag row check above, at
    MIN_USEFUL_WIDTH. Passes when: the row holds exactly filter_edit and
    sort_combo, both visible; the sort combo gets its full size hint, so
    neither mode's label is elided; the filter box gets at least its minimum
    size hint; the two sit side by side inside the pane; and the empty box's
    placeholder draws the same ink width at this width as at a 700 px pane,
    so it is not elided."""
    from PyQt5.QtWidgets import QApplication

    from descape.trigger_panel import TriggerPanel

    window = _sectioned_window()
    try:
        panel = window.trigger_panel
        set_panel_width(window, TriggerPanel.MIN_USEFUL_WIDTH)
        QApplication.processEvents()
        assert panel.width() <= TriggerPanel.MIN_USEFUL_WIDTH + 10, "fixture assumption: the pane is at its minimum"
        filter_row = panel.filter_edit.parentWidget().layout().itemAt(0).layout()
        assert _row_widgets(filter_row) == [panel.filter_edit, panel.sort_combo]

        combo = panel.sort_combo
        assert combo.isVisible() and panel.filter_edit.isVisible()
        assert combo.width() >= combo.sizeHint().width(), (
            f"the sort combo gets {combo.width()} px of its {combo.sizeHint().width()} at {panel.width()} px"
        )
        edit = panel.filter_edit
        assert edit.width() >= edit.minimumSizeHint().width(), (
            f"the filter box gets {edit.width()} px of its {edit.minimumSizeHint().width()} minimum"
        )
        edit_right = edit.mapTo(panel, edit.rect().topRight()).x()
        combo_left = combo.mapTo(panel, combo.rect().topLeft()).x()
        assert edit_right < combo_left, "the filter box overlaps the sort combo"
        assert combo_left + combo.width() <= panel.width(), "the sort combo runs past the pane"

        # The placeholder, measured: its ink at this width against the same
        # string drawn with room to spare. QLineEdit elides a placeholder that
        # does not fit, which would draw narrower ink.
        narrow = _placeholder_ink_width(edit)
        set_panel_width(window, 700)
        QApplication.processEvents()
        assert edit.width() > 300, "fixture assumption: the wide pane widens the filter box"
        wide = _placeholder_ink_width(edit)
        assert narrow == wide, f"the placeholder draws {narrow} px of its {wide} at {TriggerPanel.MIN_USEFUL_WIDTH} px"
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- TASK-285: the trigger list / entry list bar ----------------------------------
# The split's starting share and its minimums are tests/test_trigger_detail_dock.py's.


def _drag_handle(handle, dy: int) -> None:
    """Press a splitter handle, drag it `dy` px down and let go, on real mouse events."""
    from PyQt5.QtCore import QEvent, QPoint, Qt
    from PyQt5.QtGui import QMouseEvent
    from PyQt5.QtWidgets import QApplication

    start = handle.rect().center()

    def send(kind, pos, button, buttons) -> None:
        event = QMouseEvent(kind, pos, handle.mapToGlobal(pos), button, buttons, Qt.NoModifier)
        QApplication.sendEvent(handle, event)

    send(QEvent.MouseButtonPress, start, Qt.LeftButton, Qt.LeftButton)
    send(QEvent.MouseMove, start + QPoint(0, dy), Qt.NoButton, Qt.LeftButton)
    send(QEvent.MouseButtonRelease, start + QPoint(0, dy), Qt.LeftButton, Qt.NoButton)
    QApplication.processEvents()


def test_dragging_the_tree_detail_handle_still_moves_and_keeps_the_split() -> None:
    from PyQt5.QtWidgets import QApplication

    window = conftest.shown_window(1500, 900)
    window.load_scenario(TRIGGER_FIXTURE)
    window.mode_combo.setCurrentText("Triggers")
    QApplication.processEvents()
    try:
        panel = window.trigger_panel
        top, bottom = panel.splitter.sizes()

        _drag_handle(panel.splitter.handle(1), 40)
        assert panel.splitter.sizes() == [top + 40, bottom - 40]

        # A later resize keeps the dragged proportion; it does not go back to the starting split.
        window.resize(window.width(), window.height() - 100)
        QApplication.processEvents()
        shrunk_top, shrunk_bottom = panel.splitter.sizes()
        assert shrunk_top + shrunk_bottom < top + bottom
        start, dragged = top / (top + bottom), (top + 40) / (top + bottom)
        now = shrunk_top / (shrunk_top + shrunk_bottom)
        assert abs(now - dragged) < abs(now - start), f"{shrunk_top}:{shrunk_bottom} from {top}:{bottom}"
    finally:
        conftest.close_window(window)
