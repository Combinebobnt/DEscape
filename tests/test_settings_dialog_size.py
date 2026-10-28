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

    from descape import gc_hold, viewer_dialogs

    app = QApplication.instance()
    saved_font = app.font()
    saved_default = viewer_dialogs._DEFAULT_FONT
    viewer_dialogs._DEFAULT_FONT = None
    yield
    viewer_dialogs._DEFAULT_FONT = saved_default
    with gc_hold.paused():
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


def test_keybinds_never_scrolls_sideways() -> None:
    """The Mouse Wheel rows' dropdowns hold long action names and a wrapped
    note; neither may widen the rows past the dialog's default width."""
    from PyQt5.QtWidgets import QScrollArea

    dialog, window = conftest.dialog_and_window()
    try:
        tabs, index = _tab(dialog, "Keybinds")
        dialog.show()
        tabs.setCurrentIndex(index)
        _pump()
        scroll = tabs.widget(index).findChild(QScrollArea)
        assert scroll is not None and dialog._wheel_note.isVisibleTo(scroll.widget())
        _assert_no_sideways_scroll(scroll)
        assert scroll.verticalScrollBar().maximum() > 0
    finally:
        dialog.close()
        conftest.close_window(window)


def test_tools_scrolls_vertically_but_never_sideways() -> None:
    """TASK-272: one row per tool option outgrows the dialog's height, never its width."""
    from PyQt5.QtWidgets import QScrollArea

    dialog, window = conftest.dialog_and_window()
    try:
        tabs, index = _tab(dialog, "Tools")
        scroll = tabs.widget(index)
        assert isinstance(scroll, QScrollArea)
        dialog.show()
        tabs.setCurrentIndex(index)
        _pump()
        _assert_no_sideways_scroll(scroll)
        assert scroll.verticalScrollBar().maximum() > 0
        assert dialog.height() < MAX_MIN_HEIGHT
    finally:
        dialog.close()
        conftest.close_window(window)


def test_mods_never_scrolls_sideways(monkeypatch, tmp_path) -> None:
    """GH #244: long scanned paths and mod titles scroll inside their own lists,
    never widening the Mods page past the dialog's default width."""
    from PyQt5.QtWidgets import QScrollArea

    from descape import content_roots, mod_graphics

    long_path = tmp_path.joinpath(*(["a_rather_long_folder_name"] * 12))
    root = content_roots.ContentRoot(content_roots.KIND_PROFILE, "Profile 0", long_path, profile_id="0")
    monkeypatch.setattr(content_roots, "content_roots", lambda: [root])
    title = "A cosmetic mod with a very long workshop title " * 6
    mod = mod_graphics.ModArt("subscribed//1_Long", title, 1, 3, 0, 0, False)
    monkeypatch.setattr(mod_graphics, "mods_with_art", lambda: [mod])
    dialog, window = conftest.dialog_and_window()
    try:
        tabs, index = _tab(dialog, "Mods")
        dialog.show()
        tabs.setCurrentIndex(index)
        _pump()
        scroll = tabs.widget(index)
        assert isinstance(scroll, QScrollArea)
        assert dialog.mod_art_list.item(0).text() == title
        for view in (dialog.profile_roots_list, dialog.mod_art_list):
            assert view.horizontalScrollBar().maximum() > 0, "the long row scrolls inside its list"
        _assert_no_sideways_scroll(scroll)
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
