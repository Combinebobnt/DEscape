"""descape.item_activation: Return and Enter activate a list row on every OS.

Qt 5 on macOS tries edit() on Return and never emits `activated`, so these
views stand in a keyPressEvent that does just that, which Linux's Qt never
would, and the filter must activate the row anyway.
"""

from __future__ import annotations

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


def _mac_list():
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QListWidget

    class MacList(QListWidget):
        def __init__(self) -> None:
            super().__init__()
            self.reached = []

        def keyPressEvent(self, event) -> None:
            # Qt 5's macOS branch, for an item it cannot edit: no `activated`, the key ignored.
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self.reached.append(event.key())
                event.ignore()
                return
            super().keyPressEvent(event)

    return MacList()


def test_return_and_enter_activate_the_current_row_where_qt_would_not() -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest

    from descape import item_activation

    conftest.ensure_qapp()
    view = _mac_list()
    view.addItems(["a", "b"])
    view.setCurrentRow(1)
    got = []
    view.itemActivated.connect(lambda item: got.append(item.text()))
    QTest.keyClick(view, Qt.Key_Return)
    assert got == [], "fixture: the stand-in handler never activates"
    item_activation.install(view)
    QTest.keyClick(view, Qt.Key_Return)
    QTest.keyClick(view, Qt.Key_Enter, Qt.KeypadModifier)
    assert got == ["b", "b"]
    assert view.reached == [Qt.Key_Return], "the filter answers Return itself"
    view.setCurrentRow(-1)
    QTest.keyClick(view, Qt.Key_Return)
    assert got == ["b", "b"], "no current row, nothing to activate"


def test_every_view_whose_enter_rests_on_activated_installs_the_filter() -> None:
    """Off macOS Qt's own handler activates too, so only this catches a dropped install()."""
    from test_catalog_browser import _view

    from descape import item_activation
    from descape.swatch_grid import SwatchGridView
    from descape.value_picker import PickerItem, ValuePickerView

    conftest.ensure_qapp()
    views = {
        "catalog list": _view().list,
        "swatch grid": SwatchGridView([PickerItem("Grass", 7)], swatch=lambda _value: None),
        "value picker tree": ValuePickerView([PickerItem("Grass", 7)]).tree,
    }
    for name, view in views.items():
        assert view.findChildren(item_activation._ReturnActivates), name


def test_a_subclass_signal_named_activated_does_not_shadow_the_view_s() -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest

    from descape.swatch_grid import SwatchGridView
    from descape.value_picker import PickerItem

    conftest.ensure_qapp()
    grid = SwatchGridView([PickerItem("Grass", 7), PickerItem("Sand", 9)], swatch=lambda _value: None)
    got = []
    grid.activated.connect(got.append)
    grid.set_current_value(9)
    QTest.keyClick(grid, Qt.Key_Return)
    assert got == [9]


def test_the_key_still_reaches_the_dialog_s_default_button() -> None:
    """As on Linux and Windows: the row activates, then OK presses."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout

    from descape import item_activation

    conftest.ensure_qapp()
    dialog = QDialog()
    view = _mac_list()
    view.addItems(["a"])
    view.setCurrentRow(0)
    item_activation.install(view)
    buttons = QDialogButtonBox(QDialogButtonBox.Ok)
    buttons.accepted.connect(dialog.accept)
    layout = QVBoxLayout(dialog)
    layout.addWidget(view)
    layout.addWidget(buttons)
    got = []
    view.itemActivated.connect(lambda item: got.append(item.text()))
    dialog.show()
    try:
        QTest.keyClick(view, Qt.Key_Return)
        assert got == ["a"]
        assert dialog.result() == QDialog.Accepted
    finally:
        dialog.close()
