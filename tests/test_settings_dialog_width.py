"""TASK-498: Settings opens at its coded size (SettingsDialog's resize()) rather
than at a wider minimum, with no tab scrolling sideways and wrapped option
labels kept to two lines on Tools and three on Keybinds.

Not font_sensitive: the budget is for the pinned 96 DPI test font, and a larger
font may widen the dialog (TASK-210).
"""

from __future__ import annotations

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


def _settle() -> None:
    from PyQt5.QtWidgets import QApplication

    for _ in range(4):
        QApplication.processEvents()


def _each_tab(dialog):
    """Shows each tab in turn and yields (title, page widget)."""
    from PyQt5.QtWidgets import QTabWidget

    tabs = dialog.findChild(QTabWidget)
    for i in range(tabs.count()):
        tabs.setCurrentIndex(i)
        _settle()
        yield tabs.tabText(i), tabs.widget(i)


def test_settings_minimum_width_fits_its_opening_width_and_no_tab_scrolls_sideways() -> None:
    from PyQt5.QtWidgets import QScrollArea

    dialog, window = conftest.dialog_and_window()
    try:
        opening = dialog.width()
        minimum = dialog.minimumSizeHint().width()
        assert minimum <= opening, f"Settings' minimum width is {minimum} px; it opens at {opening}"
        dialog.resize(minimum, dialog.height())
        dialog.show()
        scrolling = []
        for title, page in _each_tab(dialog):
            areas = ([page] if isinstance(page, QScrollArea) else []) + page.findChildren(QScrollArea)
            scrolling.extend(
                f"{title}: {type(area).__name__} by {area.horizontalScrollBar().maximum()} px"
                for area in areas
                if area.isVisible() and area.horizontalScrollBar().maximum() > 0
            )
        assert not scrolling, scrolling
    finally:
        dialog.close()
        conftest.close_window(window)


@pytest.fixture
def _font_baseline():
    """Puts the shared QApplication's font and apply_ui_font()'s cached default back
    (tests/test_ui_font.py's fixture: a font left applied poisons the worker)."""
    conftest.ensure_qapp()
    from PyQt5.QtWidgets import QApplication

    from descape import gc_hold, viewer_dialogs

    app = QApplication.instance()
    saved_font = app.font()
    saved_default = viewer_dialogs._DEFAULT_FONT
    viewer_dialogs._DEFAULT_FONT = None
    yield
    viewer_dialogs._DEFAULT_FONT = saved_default
    with gc_hold.paused():
        app.setFont(saved_font)


def _tab_bar_cuts(dialog) -> list[str]:
    """What keeps a tab title out of view at the dialog's opening width: Qt's scroll arrows, or a
    tab running past the bar's right edge."""
    from PyQt5.QtWidgets import QTabWidget, QToolButton

    dialog.show()
    _settle()
    bar = dialog.findChild(QTabWidget).tabBar()
    cuts = [f"scroll arrow {b.arrowType()}" for b in bar.findChildren(QToolButton) if b.isVisible()]
    cuts += [f"{bar.tabText(i)} ends at {bar.tabRect(i).right()} of {bar.width()} px" for i in range(bar.count())
             if bar.tabRect(i).right() >= bar.width()]
    return cuts


@pytest.mark.parametrize("font_size", [None, 14], ids=["test font", "14 pt UI font"])
def test_settings_opens_wide_enough_for_every_tab_title(font_size, _font_baseline) -> None:
    """TASK-500 (decided 2026-10-29): Settings opens at the tab bar's own width hint when that is
    wider than its coded 520 px, so no tab scrolls out of view at any UI font. 14 pt still fits
    the 800 px offscreen screen; the next test covers a font that does not."""
    from PyQt5.QtWidgets import QApplication

    from descape.viewer_dialogs import apply_ui_font

    if font_size is not None:
        apply_ui_font(QApplication.instance(), "", font_size)
    dialog, window = conftest.dialog_and_window()
    try:
        assert dialog.width() >= 520
        cuts = _tab_bar_cuts(dialog)
        assert not cuts, f"at {dialog.width()} px: {cuts}"
    finally:
        dialog.close()
        conftest.close_window(window)


def test_settings_never_opens_wider_than_the_screen(_font_baseline) -> None:
    """TASK-500: at the largest UI font the tab titles want more than the 800 px offscreen screen,
    and the dialog stops at the screen's width (or its own minimum) instead."""
    from PyQt5.QtWidgets import QApplication, QTabWidget

    from descape import settings
    from descape.viewer_dialogs import apply_ui_font

    apply_ui_font(QApplication.instance(), "", settings.UI_FONT_SIZE_MAX)
    dialog, window = conftest.dialog_and_window()
    try:
        screen = QApplication.primaryScreen().availableGeometry().width()
        assert dialog.findChild(QTabWidget).tabBar().sizeHint().width() > screen, "the precondition: titles outgrow the screen"
        assert dialog.width() == max(screen, dialog.minimumSizeHint().width())
    finally:
        dialog.close()
        conftest.close_window(window)


def test_tools_and_keybinds_labels_wrap_to_few_lines_at_the_opening_width() -> None:
    """Word-wrapped, and not squeezed below one line by their scroll area (the GOTCHAS trap).
    Tools takes two lines at most. Keybinds three: its key box and the Default and Clear
    buttons leave the longest action labels too little room for two (user, 2026-10-28)."""
    dialog, window = conftest.dialog_and_window()
    try:
        dialog.show()
        checked = 0
        failures = []
        for title, _page in _each_tab(dialog):
            if title == "Tools":
                labels, most = list(dialog.tool_defaults_page.labels.values()), 2
            elif title == "Keybinds":
                labels, most = [widgets[0] for _d, _h, rows in dialog._keybind_sections for _id, widgets in rows], 3
            else:
                continue
            for label in labels:
                checked += 1
                metrics = label.fontMetrics()
                if not (label.isVisible() and metrics.height() <= label.height() < (most + 1) * metrics.lineSpacing()):
                    failures.append(f"{title}: {label.text()!r} is {label.height()} px tall")
        assert checked > 0
        assert not failures, failures
    finally:
        dialog.close()
        conftest.close_window(window)


def test_font_picker_minimum_ignores_long_family_names_and_shows_their_start() -> None:
    """The closed box cuts a long name at its end; the open list widens past the box for it."""
    from PyQt5.QtCore import QEvent, Qt
    from PyQt5.QtGui import QFocusEvent
    from PyQt5.QtWidgets import QApplication

    dialog, window = conftest.dialog_and_window()
    try:
        picker_type = type(dialog.ui_font_combo)
        plain = picker_type()
        long_named = picker_type()
        # Longer than any family here, short enough for the open list to fit an 800 px offscreen screen.
        long_name = "Noto Sans Devanagari UI ExtraCondensed ExtraLight Italic"
        long_named.model().insertRows(0, 1)
        long_named.model().setData(long_named.model().index(0, 0), long_name)
        assert long_named.minimumSizeHint().width() == plain.minimumSizeHint().width()

        long_named.resize(long_named.minimumSizeHint())
        long_named.show()
        QApplication.processEvents()
        line_edit = long_named.lineEdit()
        line_edit.clearFocus()
        long_named.setCurrentIndex(0)
        assert line_edit.cursorPosition() == 0, "the unfocused box scrolls to a cut name's end"
        line_edit.end(False)
        # Offscreen delivers no focus events to an inactive window, so leaving the box is sent by hand,
        # to the combo: it is its line edit's focus proxy and gets the real ones.
        QApplication.sendEvent(long_named, QFocusEvent(QEvent.FocusOut, Qt.TabFocusReason))
        QApplication.processEvents()
        assert line_edit.cursorPosition() == 0, "leaving the box leaves a cut name's end in view"
        long_named.showPopup()
        QApplication.processEvents()
        view = long_named.view()
        # The delegate's width hint leaves out the icon before each name, about a row high.
        need = view.sizeHintForColumn(0) + view.sizeHintForRow(0)
        assert view.viewport().width() >= need, "the open list cuts a family name"
    finally:
        for picker in (plain, long_named):
            picker.hidePopup()
            picker.close()
        dialog.close()
        conftest.close_window(window)
