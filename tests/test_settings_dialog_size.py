"""TASK-210: the Settings dialog fits a 768 px screen.

Appearance grew past it (minimumSizeHint height 927 under the pinned test font
once the Trigger status group landed), so that tab scrolls vertically. The
scroll area must not trade that for a sideways scroll: at the dialog's own
default size every row still fits the viewport's width.
"""

from __future__ import annotations

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

# A 768 px screen less a title bar and a taskbar.
MAX_MIN_HEIGHT = 700


def _tab(dialog, title: str):
    from PyQt5.QtWidgets import QTabWidget

    tabs = dialog.findChild(QTabWidget)
    index = next(i for i in range(tabs.count()) if tabs.tabText(i) == title)
    return tabs, index


def test_the_dialog_minimum_height_fits_a_768_px_screen() -> None:
    dialog, window = conftest.dialog_and_window()
    try:
        assert dialog.minimumSizeHint().height() < MAX_MIN_HEIGHT
        assert dialog.height() < MAX_MIN_HEIGHT
    finally:
        dialog.close()
        conftest.close_window(window)


@pytest.fixture
def font_baseline():
    """tests/test_ui_font.py's baseline: restore the session-wide app font and
    viewer_dialogs._DEFAULT_FONT, or a bumped font poisons every later test."""
    conftest.ensure_qapp()
    from PyQt5.QtWidgets import QApplication

    from descape import viewer_dialogs

    app = QApplication.instance()
    saved_font = app.font()
    saved_default = viewer_dialogs._DEFAULT_FONT
    viewer_dialogs._DEFAULT_FONT = None
    yield
    viewer_dialogs._DEFAULT_FONT = saved_default
    app.setFont(saved_font)


def _pump() -> None:
    from PyQt5.QtWidgets import QApplication

    for _ in range(4):
        QApplication.processEvents()


def _assert_no_sideways_scroll(scroll) -> None:
    from PyQt5.QtCore import QPoint
    from PyQt5.QtWidgets import QWidget

    content, viewport = scroll.widget(), scroll.viewport()
    assert scroll.horizontalScrollBar().maximum() == 0
    assert content.width() <= viewport.width()
    right_edges = [
        child.mapTo(content, QPoint(child.width(), 0)).x()
        for child in content.findChildren(QWidget)
        if child.isVisibleTo(content)
    ]
    assert right_edges and max(right_edges) <= viewport.width()


def _shown_appearance(dialog):
    from PyQt5.QtWidgets import QScrollArea

    tabs, index = _tab(dialog, "Appearance")
    scroll = tabs.widget(index)
    assert isinstance(scroll, QScrollArea)
    dialog.show()
    tabs.setCurrentIndex(index)
    _pump()
    return scroll


def test_appearance_scrolls_vertically_but_never_sideways() -> None:
    dialog, window = conftest.dialog_and_window()
    try:
        scroll = _shown_appearance(dialog)
        _assert_no_sideways_scroll(scroll)
        # The tab really is taller than its viewport, so the scroll is doing work.
        assert scroll.verticalScrollBar().maximum() > 0
    finally:
        dialog.close()
        conftest.close_window(window)


def test_a_live_ui_font_bump_widens_the_dialog_not_the_page(font_baseline) -> None:
    """The UI font spin lives on this very tab, and a bigger font widens the
    page; the dialog must grow with it rather than scroll sideways."""
    from descape import settings

    dialog, window = conftest.dialog_and_window()
    try:
        scroll = _shown_appearance(dialog)
        width_before = dialog.width()
        dialog.ui_font_size_spin.setValue(settings.UI_FONT_SIZE_MAX)
        dialog._ui_font_apply_timer.stop()
        dialog._apply_ui_font()
        _pump()
        assert dialog.width() > width_before
        _assert_no_sideways_scroll(scroll)
    finally:
        dialog.close()
        conftest.close_window(window)
