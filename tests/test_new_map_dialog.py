"""descape/new_map_dialog.py: File > New Map…'s dialog (GH #243) on its own,
offscreen. Creating the map and saving the choice is test_new_map.py's job.
"""

from __future__ import annotations

import pytest

from descape import terrain_catalog
from descape.scenario_new import STANDARD_MAP_SIZE_NAMES, STANDARD_MAP_SIZES

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_BEACH = 2


@pytest.fixture
def dialog_for():
    conftest.ensure_qapp()
    from descape.new_map_dialog import NewMapDialog

    dialogs = []

    def make(initial=None):
        dialog = NewMapDialog(None, initial)
        dialogs.append(dialog)
        return dialog

    yield make
    for dialog in dialogs:
        dialog.deleteLater()


def test_size_rows_are_the_presets_in_order_then_custom(dialog_for) -> None:
    dialog = dialog_for()
    combo = dialog.size_combo
    labels = [combo.itemText(i) for i in range(combo.count())]
    assert labels == [f"{STANDARD_MAP_SIZE_NAMES[t]} ({t}×{t})" for t in STANDARD_MAP_SIZES] + ["Custom"]
    assert labels[0] == "Tiny (120×120)"
    assert labels[-2] == "Ludicrous (480×480)"


def test_custom_enables_the_size_spin_and_presets_disable_it(dialog_for) -> None:
    dialog = dialog_for()
    assert not dialog.is_custom
    assert not dialog.custom_spin.isEnabled()
    dialog.size_combo.setCurrentIndex(dialog.size_combo.count() - 1)
    assert dialog.is_custom
    assert dialog.custom_spin.isEnabled()
    dialog.custom_spin.setValue(300)
    assert dialog.tiles == 300
    dialog.size_combo.setCurrentIndex(1)
    assert not dialog.custom_spin.isEnabled()
    assert dialog.tiles == 144


def test_custom_row_shows_the_size_range(dialog_for) -> None:
    from descape.scenario_new import MAX_MAP_TILES, MIN_MAP_TILES

    dialog = dialog_for()
    spin = dialog.custom_spin
    assert (spin.minimum(), spin.maximum()) == (MIN_MAP_TILES, MAX_MAP_TILES)
    from PyQt5.QtWidgets import QFormLayout

    label = dialog.findChild(QFormLayout).labelForField(spin)
    assert label.text() == f"Custom size ({MIN_MAP_TILES}-{MAX_MAP_TILES}):"


def test_both_games_are_listed_and_pickable(dialog_for) -> None:
    from descape.new_map_dialog import AOE2, ROR

    dialog = dialog_for()
    combo = dialog.game_data_combo
    assert [combo.itemData(i) for i in range(combo.count())] == [AOE2, ROR]
    assert [combo.itemText(i) for i in range(combo.count())] == ["Age of Empires II", "Return of Rome"]
    assert all(combo.model().item(i).isEnabled() for i in range(combo.count()))
    assert combo.currentData() == AOE2
    combo.setCurrentIndex(combo.findData(ROR))
    dialog.accept()
    assert dialog.choice.game_data == ROR


def test_an_initial_return_of_rome_choice_is_kept(dialog_for) -> None:
    from descape.new_map_dialog import ROR, NewMapChoice

    assert dialog_for(NewMapChoice(game_data=ROR)).current_choice().game_data == ROR


def test_terrain_list_is_the_visible_catalog_by_name(dialog_for) -> None:
    dialog = dialog_for()
    combo = dialog.terrain_combo
    ids = [combo.itemData(i) for i in range(combo.count())]
    visible = [e for e in terrain_catalog.terrains() if not e.hidden]
    assert sorted(ids) == sorted(e.id for e in visible)
    names = [combo.itemText(i) for i in range(combo.count())]
    assert names == sorted(names)
    assert combo.currentData() == 0


def test_choice_matches_the_widgets_after_accept(dialog_for) -> None:
    from descape.new_map_dialog import AOE2, NewMapChoice

    dialog = dialog_for()
    assert dialog.choice is None
    dialog.size_combo.setCurrentIndex(dialog.size_combo.findData(144))
    dialog.terrain_combo.setCurrentIndex(dialog.terrain_combo.findData(_BEACH))
    dialog.elevation_spin.setValue(3)
    dialog.accept()
    assert dialog.choice == NewMapChoice(tiles=144, game_data=AOE2, terrain_id=_BEACH, elevation=3)


def test_elevation_spin_spans_the_elevation_tool_range(dialog_for) -> None:
    from descape import iso_geometry

    spin = dialog_for().elevation_spin
    assert (spin.minimum(), spin.maximum()) == (iso_geometry.MIN_ELEVATION, iso_geometry.MAX_ELEVATION)


def test_an_initial_custom_size_selects_custom(dialog_for) -> None:
    from descape.new_map_dialog import NewMapChoice

    dialog = dialog_for(NewMapChoice(tiles=300, terrain_id=_BEACH, elevation=5))
    assert dialog.is_custom
    assert dialog.custom_spin.isEnabled()
    assert dialog.custom_spin.value() == 300
    assert dialog.current_choice() == NewMapChoice(tiles=300, terrain_id=_BEACH, elevation=5)


def test_an_initial_preset_selects_that_preset(dialog_for) -> None:
    from descape.new_map_dialog import NewMapChoice

    dialog = dialog_for(NewMapChoice(tiles=240))
    assert not dialog.is_custom
    assert dialog.size_combo.currentText() == "Giant (240×240)"


def test_bad_initial_values_fall_back_to_the_defaults(dialog_for) -> None:
    from descape.new_map_dialog import NewMapChoice

    hidden = next(e.id for e in terrain_catalog.terrains() if e.hidden)
    for initial in (
        NewMapChoice(tiles=5, game_data="aok", terrain_id=hidden, elevation=40),
        NewMapChoice(tiles=9999, game_data="aok", terrain_id=999, elevation=-1),
    ):
        assert dialog_for(initial).current_choice() == NewMapChoice()


def _main_window(width=760, height=540):
    """A QMainWindow with a menu, a shortcut action and a button, inside the
    offscreen platform's 800x600 screen."""
    from PyQt5.QtWidgets import QAction, QApplication, QMainWindow, QPushButton

    conftest.ensure_qapp()
    window = QMainWindow()
    window.setGeometry(20, 20, width, height)
    menu = window.menuBar().addMenu("&File")
    window.fired = []
    window.save_action = QAction("Save", window)
    window.save_action.setShortcut("Ctrl+S")
    window.save_action.triggered.connect(lambda: window.fired.append("save"))
    menu.addAction(window.save_action)
    # Window shortcuts ViewerWindow also has: a single-key tool key, Esc and Return.
    for name, key in (("tool", "S"), ("escape", "Esc"), ("return", "Return")):
        action = QAction(name, window)
        action.setShortcut(key)
        action.triggered.connect(lambda _checked=False, name=name: window.fired.append(name))
        window.addAction(action)
    window.button = QPushButton("behind")
    window.button.clicked.connect(lambda: window.fired.append("click"))
    window.setCentralWidget(window.button)
    window.show()
    QApplication.setActiveWindow(window)
    QApplication.processEvents()
    assert window.isActiveWindow()
    return window


def _run_in(window, dialog, check):
    """exec_in with `check(dialog, modal)` run once the panel is up; check returns the button to press."""
    from PyQt5.QtCore import QTimer

    from descape.new_map_dialog import InWindowModal

    seen = {}

    def step():
        finish = "reject"
        try:
            modal = next(w for w in window.findChildren(InWindowModal) if w.isVisible())
            seen["modal"] = modal
            finish = check(dialog, modal)
        except Exception as e:  # noqa: BLE001 -- re-raised after the loop, where pytest sees it
            seen["error"] = e
        if dialog.isVisible():
            (dialog.accept if finish == "accept" else dialog.reject)()

    def give_up():
        # done() emits finished even when hidden, so this always ends the loop.
        seen.setdefault("error", AssertionError("exec_in still open after 5 s"))
        dialog.reject()

    QTimer.singleShot(0, step)
    # Owned by the dialog and stopped after, so it cannot fire into a later test.
    guard = QTimer(dialog)
    guard.setSingleShot(True)
    guard.timeout.connect(give_up)
    guard.start(5000)
    result = dialog.exec_in(window)
    guard.stop()
    if "error" in seen:
        raise seen["error"]
    return result, seen


def test_the_panel_is_centered_over_the_whole_window_and_follows_a_resize() -> None:
    from PyQt5.QtWidgets import QApplication, QDialog

    from descape.new_map_dialog import NewMapDialog

    window = _main_window()
    dialog = NewMapDialog(window)

    def check(dialog, modal):
        assert modal.geometry() == window.rect()
        assert not dialog.isWindow()
        offset = dialog.geometry().center() - window.rect().center()
        assert abs(offset.x()) <= 1 and abs(offset.y()) <= 1
        window.resize(700, 500)
        QApplication.processEvents()
        QApplication.processEvents()
        assert modal.geometry() == window.rect()
        offset = dialog.geometry().center() - window.rect().center()
        assert abs(offset.x()) <= 1 and abs(offset.y()) <= 1
        return "accept"

    try:
        result, seen = _run_in(window, dialog, check)
        assert result == QDialog.Accepted
        assert dialog.choice is not None
        assert not seen["modal"].isVisible()
    finally:
        window.close()


def test_the_panel_is_fixed_at_its_contents_size() -> None:
    """No resizing: the panel is exactly as big as every row needs (and at
    least MIN_PANEL_SIZE), re-fitted once the terrain icons have loaded."""
    from PyQt5.QtWidgets import QApplication, QFormLayout, QSizeGrip

    from descape.new_map_dialog import MIN_PANEL_SIZE, NewMapDialog

    window = _main_window()
    dialog = NewMapDialog(window)

    def check(dialog, modal):
        assert dialog.findChild(QSizeGrip) is None
        for _ in range(dialog.terrain_combo.count() + 1):
            if not dialog._pending_swatches:
                break
            QApplication.processEvents()
        assert not dialog._pending_swatches
        fitted = dialog.sizeHint().expandedTo(MIN_PANEL_SIZE)
        assert dialog.minimumSize() == dialog.maximumSize() == fitted
        dialog.resize(10, 10)
        QApplication.processEvents()
        assert dialog.size() == fitted
        form = dialog.findChild(QFormLayout)
        for row in range(form.rowCount()):
            for role in (QFormLayout.LabelRole, QFormLayout.FieldRole):
                widget = form.itemAt(row, role).widget()
                assert widget.isVisible()
                assert widget.width() >= widget.minimumSizeHint().width(), widget
        combo = dialog.terrain_combo
        widest = max(combo.fontMetrics().horizontalAdvance(combo.itemText(i)) for i in range(combo.count()))
        assert combo.width() >= widest
        return "reject"

    try:
        _run_in(window, dialog, check)
    finally:
        window.close()


def test_set_initial_reshows_a_choice_and_clears_the_last_one(dialog_for) -> None:
    from descape.new_map_dialog import ROR, NewMapChoice

    dialog = dialog_for()
    dialog.accept()
    assert dialog.choice is not None
    dialog.set_initial(NewMapChoice(tiles=300, game_data=ROR, terrain_id=_BEACH, elevation=6))
    assert dialog.choice is None
    assert dialog.current_choice() == NewMapChoice(tiles=300, game_data=ROR, terrain_id=_BEACH, elevation=6)
    assert not dialog.terrain_combo.itemIcon(dialog.terrain_combo.currentIndex()).isNull()


def test_the_window_behind_the_panel_takes_no_shortcut_click_or_close() -> None:
    from PyQt5.QtCore import QPoint, Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication, QDialog

    from descape.new_map_dialog import NewMapDialog

    window = _main_window()
    QTest.keyClick(window.button, Qt.Key_S, Qt.ControlModifier)
    assert window.fired == ["save"]  # the shortcut does fire with no panel open
    window.fired.clear()
    dialog = NewMapDialog(window)
    file_menu = window.menuBar().actions()[0]

    def check(dialog, modal):
        assert not file_menu.isEnabled()
        QTest.keyClick(dialog.size_combo, Qt.Key_S, Qt.ControlModifier)
        behind = window.button.geometry().topLeft() + QPoint(5, 5)  # on the button, outside the panel
        assert not dialog.geometry().contains(behind)
        assert window.childAt(behind) is modal
        QTest.mouseClick(modal, Qt.LeftButton, pos=behind)
        assert window.close() is False
        QApplication.processEvents()
        assert window.fired == []
        assert window.isVisible()
        return "reject"

    try:
        result, seen = _run_in(window, dialog, check)
        assert result == QDialog.Rejected
        assert dialog.choice is None
        assert file_menu.isEnabled()
        assert not seen["modal"].isVisible()
        QTest.keyClick(window.button, Qt.Key_S, Qt.ControlModifier)
        assert window.fired == ["save"]  # the filter is gone again
    finally:
        window.close()


def test_plain_keys_reach_the_panel_over_window_shortcuts() -> None:
    """S is a single-key window shortcut (a tool key in DEscape), Esc and
    Return are too: inside the panel they type ahead and cancel or create
    instead of firing the window's actions."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QDialog

    from descape.new_map_dialog import NewMapDialog

    window = _main_window()
    dialog = NewMapDialog(window)

    def check(dialog, modal):
        dialog.size_combo.setCurrentIndex(0)
        QTest.keyClick(dialog.size_combo, Qt.Key_S)
        assert dialog.size_combo.currentText() == "Small (144×144)"
        assert window.fired == []
        return "reject"

    try:
        _run_in(window, dialog, check)
        for key, expected in ((Qt.Key_Escape, QDialog.Rejected), (Qt.Key_Return, QDialog.Accepted)):

            def press(dialog, modal, key=key):
                QTest.keyClick(dialog.focusWidget(), key)
                return "accept" if dialog.result() == QDialog.Accepted else "reject"

            result, _seen = _run_in(window, dialog, press)
            assert result == expected
        assert window.fired == []
    finally:
        window.close()


def test_focus_goes_back_where_it_was() -> None:
    from PyQt5.QtWidgets import QApplication

    from descape.new_map_dialog import NewMapDialog

    window = _main_window()
    window.button.setFocus()
    dialog = NewMapDialog(window)

    def check(dialog, modal):
        assert dialog.isAncestorOf(QApplication.focusWidget())
        return "reject"

    try:
        _run_in(window, dialog, check)
        assert QApplication.focusWidget() is window.button
    finally:
        window.close()


def test_other_windows_and_drops_are_blocked_while_open() -> None:
    from PyQt5.QtCore import QMimeData, QPoint, Qt
    from PyQt5.QtGui import QDragEnterEvent
    from PyQt5.QtWidgets import QApplication, QWidget

    from descape.new_map_dialog import NewMapDialog

    window = _main_window()
    other = QWidget()
    other.setGeometry(30, 30, 200, 100)
    other.show()
    dialog = NewMapDialog(window)
    entered = []
    window.dragEnterEvent = entered.append
    window.setAcceptDrops(True)

    def check(dialog, modal):
        assert not other.isEnabled()
        mime = QMimeData()
        mime.setText("x")
        event = QDragEnterEvent(QPoint(5, 5), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        QApplication.sendEvent(window, event)
        assert entered == []
        assert not event.isAccepted()
        return "reject"

    try:
        _run_in(window, dialog, check)
        assert other.isEnabled()
    finally:
        other.close()
        window.close()


def test_the_same_panel_opens_twice_after_the_backdrop_is_deleted() -> None:
    from PyQt5.QtCore import QCoreApplication, QEvent
    from PyQt5.QtWidgets import QDialog

    from descape.new_map_dialog import InWindowModal, NewMapDialog

    window = _main_window()
    dialog = NewMapDialog(window)
    try:
        for finish in ("reject", "accept"):
            result, _seen = _run_in(window, dialog, lambda dialog, modal, finish=finish: finish)
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            assert window.findChildren(InWindowModal) == []
            assert dialog.parentWidget() is window
            assert not dialog.isVisible()
        assert result == QDialog.Accepted
    finally:
        window.close()


def test_only_the_shown_terrain_swatch_is_decoded_before_the_dialog_shows(monkeypatch) -> None:
    """The rest fill in a few per tick after show, so opening the dialog
    never waits on ~112 .dds decodes."""
    from PyQt5.QtCore import QRect

    from descape import terrain_browser

    decoded = []
    real = terrain_browser.swatch_pixmap
    monkeypatch.setattr(terrain_browser, "swatch_pixmap", lambda tid, *a: decoded.append(tid) or real(tid, *a))

    conftest.ensure_qapp()
    from PyQt5.QtWidgets import QApplication

    from descape.new_map_dialog import NewMapDialog

    dialog = NewMapDialog(None)
    try:
        combo = dialog.terrain_combo
        assert decoded == [combo.currentData()]
        assert not combo.itemIcon(combo.currentIndex()).isNull()
        assert sum(not combo.itemIcon(i).isNull() for i in range(combo.count())) == 1
        dialog.setGeometry(QRect(40, 30, 500, 240))
        dialog.show()
        for _ in range(combo.count()):
            QApplication.processEvents()
            if not dialog._pending_swatches:
                break
        assert all(not combo.itemIcon(i).isNull() for i in range(combo.count()))
        assert sorted(decoded) == sorted(combo.itemData(i) for i in range(combo.count()))
    finally:
        dialog.close()
        dialog.deleteLater()
