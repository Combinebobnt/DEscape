"""GH #205: viewer_common.KeepOpenMenu stays open after a pick on a checkable
row or a row opted in with keep_open(), and the viewer's toggle menus
(Filters, View and its submenus, Sources, Targets) are built from it.

The class tests drive real mouse and key events at a bare popup menu, which
does show under QT_QPA_PLATFORM=offscreen. Every menu a test pops up is
closed again by the `menus` fixture: a popup left open would swallow the
mouse events of every later test on the same xdist worker.

Every ViewerWindow() constructed here must call edit_history.mark_saved()
before close(), as tests/test_fill_tool.py's module docstring explains.
"""

from __future__ import annotations

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


def _process() -> None:
    from PyQt5.QtWidgets import QApplication

    QApplication.processEvents()


@pytest.fixture
def menus():
    """Call with a menu class to get an instance that is closed at teardown."""
    conftest.ensure_qapp()
    made = []

    def make(cls, *args):
        menu = cls(*args)
        made.append(menu)
        return menu

    yield make
    for menu in reversed(made):
        menu.close()
    _process()
    for menu in made:
        menu.deleteLater()
    _process()


def _keep_open_menu(menus):
    from descape.viewer_common import KeepOpenMenu

    return menus(KeepOpenMenu)


def _checkable(menu, text: str, *, checked: bool = False):
    action = menu.addAction(text)
    action.setCheckable(True)
    action.setChecked(checked)
    return action


def _counted(menu, text: str):
    """A plain row and the list its `triggered` appends to."""
    fired: list[bool] = []
    action = menu.addAction(text)
    action.triggered.connect(fired.append)
    return action, fired


def _popup(menu) -> None:
    from PyQt5.QtCore import QPoint
    from PyQt5.QtTest import QTest

    menu.popup(QPoint(40, 40))
    _process()
    assert menu.isVisible(), "the popup menu did not show"
    # QMenu ignores the first move after a popup, and Qt drops a move that
    # repeats the last position: two spots, so one always lands and is spent.
    for spot in (QPoint(1, 1), QPoint(2, 2)):
        QTest.mouseMove(menu.windowHandle(), spot)
    _process()


def _row(menu, action):
    return menu.actionGeometry(action).center()


def _hover(menu, action) -> None:
    """A button-less move, sent to the QWindow: QTest's QWidget overload moves
    the cursor instead of sending an event."""
    from PyQt5.QtTest import QTest

    QTest.mouseMove(menu.windowHandle(), _row(menu, action))
    _process()


def _click(menu, action) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest

    _hover(menu, action)
    QTest.mouseClick(menu, Qt.LeftButton, Qt.NoModifier, _row(menu, action))
    _process()


def _release_only(menu, action) -> None:
    """A left-button release with no press before it."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest

    QTest.mouseRelease(menu, Qt.LeftButton, Qt.NoModifier, _row(menu, action))
    _process()


def test_click_on_a_checkable_row_toggles_it_and_keeps_the_menu_open(menus) -> None:
    menu = _keep_open_menu(menus)
    row = _checkable(menu, "Show Trees")
    _popup(menu)

    _click(menu, row)
    assert row.isChecked()
    assert menu.isVisible()

    _click(menu, row)
    assert not row.isChecked()
    assert menu.isVisible()


def test_exclusive_group_moves_the_radio_and_keeps_the_menu_open(menus) -> None:
    from PyQt5.QtWidgets import QActionGroup

    menu = _keep_open_menu(menus)
    group = QActionGroup(menu)
    first = group.addAction(_checkable(menu, "Every 10", checked=True))
    second = group.addAction(_checkable(menu, "Every 20"))
    _popup(menu)

    _click(menu, second)
    assert second.isChecked()
    assert not first.isChecked()
    assert menu.isVisible()

    # Clicking the checked member of an exclusive group must not clear it.
    _click(menu, second)
    assert second.isChecked()
    assert not first.isChecked()
    assert menu.isVisible()


def test_plain_row_closes_the_menu_unless_opted_in(menus) -> None:
    menu = _keep_open_menu(menus)
    row, fired = _counted(menu, "Show All")
    _popup(menu)

    _click(menu, row)
    assert len(fired) == 1
    assert not menu.isVisible()

    menu.keep_open(row)
    _popup(menu)
    _click(menu, row)
    assert len(fired) == 2
    assert menu.isVisible()


def test_disabled_checkable_row_is_not_toggled(menus) -> None:
    menu = _keep_open_menu(menus)
    row = _checkable(menu, "Show Trees")
    row.setEnabled(False)
    _popup(menu)

    _click(menu, row)
    assert not row.isChecked()
    assert not row.isEnabled()


def test_release_without_a_press_does_not_toggle_again(menus) -> None:
    menu = _keep_open_menu(menus)
    row = _checkable(menu, "Show Trees")
    _popup(menu)
    _click(menu, row)
    assert row.isChecked()

    # Sent directly after the handled click: after a fall-through click this passes for the wrong reason.
    _release_only(menu, row)
    assert row.isChecked()
    assert menu.isVisible()


def test_release_without_a_press_does_not_fire_a_plain_row(menus) -> None:
    menu = _keep_open_menu(menus)
    row = _checkable(menu, "Show Trees")
    plain, fired = _counted(menu, "Open Map")
    _popup(menu)
    _click(menu, row)
    assert row.isChecked()

    # A button-less move makes the plain row current; only a press arms the menu.
    _hover(menu, plain)
    assert menu.activeAction() is plain
    _release_only(menu, plain)
    assert fired == []
    assert menu.isVisible()


def test_drag_onto_a_row_and_release_toggles_once_and_stays_open(menus) -> None:
    """The press landed elsewhere (the menu bar), so only the drag arms the menu."""
    from PyQt5.QtCore import QEvent, QPointF, Qt
    from PyQt5.QtGui import QMouseEvent
    from PyQt5.QtWidgets import QApplication

    menu = _keep_open_menu(menus)
    row = _checkable(menu, "Show Trees")
    _popup(menu)

    spot = _row(menu, row)
    drag = QMouseEvent(
        QEvent.MouseMove, QPointF(spot), QPointF(menu.mapToGlobal(spot)), Qt.NoButton, Qt.LeftButton, Qt.NoModifier
    )
    QApplication.sendEvent(menu, drag)
    _release_only(menu, row)
    assert row.isChecked()
    assert menu.isVisible()


def test_space_toggles_and_stays_open_and_return_toggles_and_closes(menus) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest

    menu = _keep_open_menu(menus)
    row = _checkable(menu, "Show Trees")
    _popup(menu)
    menu.setActiveAction(row)

    QTest.keyClick(menu, Qt.Key_Space)
    _process()
    assert row.isChecked()
    assert menu.isVisible()

    QTest.keyClick(menu, Qt.Key_Return)
    _process()
    assert not row.isChecked()
    assert not menu.isVisible()


def test_space_on_an_opted_in_plain_row_fires_it_and_stays_open(menus) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest

    menu = _keep_open_menu(menus)
    row, fired = _counted(menu, "Show All")
    menu.keep_open(row)
    _popup(menu)
    menu.setActiveAction(row)

    QTest.keyClick(menu, Qt.Key_Space)
    _process()
    assert len(fired) == 1
    assert menu.isVisible()


def test_click_in_a_keep_open_submenu_keeps_both_menus_open(menus) -> None:
    from PyQt5.QtTest import QTest

    from descape.viewer_common import KeepOpenMenu

    menu = _keep_open_menu(menus)
    _checkable(menu, "Isometric View")
    sub = menus(KeepOpenMenu, "Layers", menu)
    menu.addMenu(sub)
    row = _checkable(sub, "Units")
    _popup(menu)

    # A click on the submenu's own row opens it, after Qt's short popup delay.
    _click(menu, sub.menuAction())
    QTest.qWait(250)
    assert sub.isVisible(), "the submenu did not open"
    assert menu.isVisible()

    _click(sub, row)
    assert row.isChecked()
    assert sub.isVisible()
    assert menu.isVisible()


def test_control_the_same_click_closes_a_stock_qmenu(menus) -> None:
    """Without this, the tests above would pass if offscreen never closed a menu."""
    from PyQt5.QtWidgets import QMenu

    menu = menus(QMenu)
    row = _checkable(menu, "Show Trees")
    _popup(menu)

    _click(menu, row)
    assert row.isChecked()
    assert not menu.isVisible()


def _top_menus(window) -> dict[str, object]:
    return {
        action.text().replace("&", ""): action.menu()
        for action in window.menuBar().actions()
        if action.menu() is not None
    }


def test_viewer_toggle_menus_are_keep_open_menus() -> None:
    from descape.viewer_common import KeepOpenMenu

    window = conftest.blank_window()
    try:
        assert isinstance(window.filters_button.menu(), KeepOpenMenu)
        assert isinstance(window.convert_sources_button.menu(), KeepOpenMenu)
        assert isinstance(window.eraser_targets_button.menu(), KeepOpenMenu)

        view_menu = _top_menus(window)["View"]
        assert isinstance(view_menu, KeepOpenMenu)
        submenus = {
            action.text().replace("&", ""): action.menu()
            for action in view_menu.actions()
            if action.menu() is not None
        }
        assert {"Distance Ticks", "Grid", "Footprint Outlines", "Layers"} <= set(submenus)
        not_keep_open = sorted(name for name, sub in submenus.items() if not isinstance(sub, KeepOpenMenu))
        assert not not_keep_open
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_viewer_command_menus_stay_plain_menus() -> None:
    from descape.viewer_common import KeepOpenMenu

    window = conftest.blank_window()
    try:
        top = _top_menus(window)
        assert {"File", "Edit", "Map", "View", "Tools", "Help"} <= set(top)
        keep_open = sorted(name for name, menu in top.items() if isinstance(menu, KeepOpenMenu))
        assert keep_open == ["View"]
        assert not isinstance(window.more_tools_menu, KeepOpenMenu)
        assert not isinstance(window.wall_pick_button.menu(), KeepOpenMenu)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_viewer_bulk_rows_are_opted_in() -> None:
    window = conftest.blank_window()
    try:
        filters_menu = window.filters_button.menu()
        for action in (
            window.filter_all_players_action,
            window.filter_no_players_action,
            window.filter_show_all_action,
            window.filter_hide_all_action,
        ):
            assert not action.isCheckable()
            assert filters_menu._keeps_open(action), action.text()

        targets_menu = window.eraser_targets_button.menu()
        everything = targets_menu.actions()[0]
        assert everything.text() == "Everything"
        assert not everything.isCheckable()
        assert targets_menu._keeps_open(everything)
    finally:
        window.edit_history.mark_saved()
        window.close()
