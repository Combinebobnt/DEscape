"""descape/favorites_dialog.py (GH #195): the Favorite Objects window on its own.
The viewer's wiring is tests/test_favorites_viewer.py."""

from __future__ import annotations

import pytest
from PyQt5.QtCore import QEvent, QPoint, Qt

from descape import asset_source, catalog_browser, object_catalog, place_random, settings
from descape.place_random import RANDOM_VILLAGER

import conftest
from conftest import ManualExecutor

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_ARCHER = 4
_CASTLE = 82
_UNKNOWN = 32000


@pytest.fixture(autouse=True)
def _no_install_from_the_environment(monkeypatch):
    """The no-install cases stay so with AOE2DE_INSTALL_PATH set, which outranks the config conftest hides."""
    monkeypatch.delenv("AOE2DE_INSTALL_PATH", raising=False)
    asset_source.clear_install_caches()


def _entry(value: int):
    if place_random.is_random(value):
        return next(e for e in place_random.catalog_entries() if e.id == value)
    return object_catalog.entry(object_catalog.objects(), value)


def _rows(*values):
    return [(value, None if value == _UNKNOWN else _entry(value)) for value in values]


def _dialog(loader=None, **callbacks):
    from descape.catalog_thumbs import ThumbLoader
    from descape.favorites_dialog import UnitFavoritesDialog

    conftest.ensure_qapp()
    return UnitFavoritesDialog(None, loader=loader or ThumbLoader(executor=ManualExecutor()), **callbacks)


def _texts(dialog) -> list[str]:
    return [dialog.list.item(i).text() for i in range(dialog.list.count())]


def _values(dialog) -> list[int]:
    return [dialog.list.item(i).data(Qt.UserRole) for i in range(dialog.list.count())]


def _press(widget, pos: QPoint, button) -> None:
    from PyQt5.QtGui import QMouseEvent

    widget.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, pos, button, button, Qt.NoModifier))


def test_it_lists_the_rows_in_order_with_names_tooltips_and_stars() -> None:
    dialog = _dialog()
    dialog.set_entries(_rows(_CASTLE, _ARCHER, RANDOM_VILLAGER))
    assert dialog.windowTitle() == "Favorite Objects"
    assert _values(dialog) == dialog.values() == [_CASTLE, _ARCHER, RANDOM_VILLAGER]
    assert _texts(dialog) == [_entry(v).name for v in (_CASTLE, _ARCHER, RANDOM_VILLAGER)]
    assert dialog.list.item(1).toolTip() == f"{_entry(_ARCHER).name} ({_ARCHER}) (favorite)"
    assert all(dialog.list.item(i).data(catalog_browser.FAVORITE_ROLE) for i in range(dialog.list.count()))
    assert dialog.hint.isHidden()


def test_a_click_picks_the_row() -> None:
    picked = []
    dialog = _dialog(on_pick=picked.append)
    dialog.set_entries(_rows(_ARCHER, RANDOM_VILLAGER))
    dialog.list.itemClicked.emit(dialog.list.item(1))
    assert picked == [RANDOM_VILLAGER]


def test_an_unknown_row_is_greyed_selectable_and_never_picks() -> None:
    from PyQt5.QtGui import QPalette

    picked, removed = [], []
    dialog = _dialog(on_pick=picked.append, on_remove=removed.append)
    dialog.set_entries(_rows(_ARCHER, _UNKNOWN))
    item = dialog.list.item(1)
    assert item.text() == f"Unknown object ({_UNKNOWN})"
    assert "cannot be placed" in item.toolTip()
    assert item.foreground().color() == dialog.list.palette().color(QPalette.Disabled, QPalette.Text)
    assert item.flags() & Qt.ItemIsSelectable and item.flags() & Qt.ItemIsEnabled
    dialog.list.setCurrentItem(item)
    dialog.list.itemClicked.emit(item)
    assert picked == []
    dialog.remove_button.click()
    assert removed == [_UNKNOWN], "removable all the same"


def test_remove_the_menu_and_delete_with_focus_all_remove() -> None:
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    removed = []
    dialog = _dialog(on_remove=removed.append)
    dialog.set_entries(_rows(_ARCHER, _CASTLE))
    assert not dialog.remove_button.isEnabled(), "nothing highlighted yet"
    dialog.list.setCurrentRow(1)
    assert dialog.remove_button.isEnabled()
    dialog.remove_button.click()
    menu = dialog._remove_menu(_ARCHER)
    assert [a.text() for a in menu.actions()] == ["Remove from Favorites"]
    menu.actions()[0].trigger()
    dialog.show()
    try:
        dialog.list.setFocus()
        QApplication.processEvents()
        QTest.keyClick(dialog.list, Qt.Key_Delete)
    finally:
        dialog.close()
    assert removed == [_CASTLE, _ARCHER, _CASTLE]


def test_a_delete_that_rebuilds_the_list_under_the_key_event_does_not_crash() -> None:
    from PyQt5.QtTest import QTest

    rows = _rows(_ARCHER, _CASTLE, RANDOM_VILLAGER)
    dialog = _dialog()
    dialog._on_remove = lambda value: dialog.set_entries([row for row in rows if row[0] != value])
    dialog.set_entries(rows)
    dialog.show()
    try:
        dialog.list.setCurrentRow(1)
        QTest.keyClick(dialog.list, Qt.Key_Delete)
        assert _values(dialog) == [_ARCHER, RANDOM_VILLAGER]
    finally:
        dialog.close()


def test_a_right_press_neither_picks_nor_moves_the_highlight() -> None:
    picked = []
    dialog = _dialog(on_pick=picked.append)
    dialog.set_entries(_rows(_ARCHER, _CASTLE))
    dialog.show()
    try:
        dialog.list.setCurrentRow(0)
        spot = dialog.list.visualItemRect(dialog.list.item(1)).center()
        _press(dialog.list, spot, Qt.RightButton)
        assert dialog.list.currentRow() == 0 and picked == []
        _press(dialog.list, spot, Qt.LeftButton)
        assert dialog.list.currentRow() == 1, "a left press still moves it, or this proves nothing"
    finally:
        dialog.close()


def test_the_disabled_list_shows_its_hint_and_turns_remove_and_drag_off() -> None:
    picked, removed = [], []
    dialog = _dialog(on_pick=picked.append, on_remove=removed.append)
    dialog.set_entries(_rows(_ARCHER, _CASTLE))
    dialog.list.setCurrentRow(0)
    assert dialog.remove_button.isEnabled() and dialog.list.dragEnabled()
    dialog.set_pick_enabled(False, "Switch to Units mode to place a favorite.")
    assert not dialog.list.isEnabled()
    assert dialog.hint.text() == "Switch to Units mode to place a favorite." and not dialog.hint.isHidden()
    assert not dialog.remove_button.isEnabled() and not dialog.list.dragEnabled()
    dialog.list.itemClicked.emit(dialog.list.item(0))
    dialog._remove_current()
    assert picked == [] and removed == []
    dialog.set_pick_enabled(True)
    assert dialog.list.isEnabled() and dialog.remove_button.isEnabled() and dialog.hint.isHidden()


def test_the_empty_hint_and_the_sizes() -> None:
    dialog = _dialog()
    assert dialog.hint.text() == "Right-click an object in the Units catalog to add it."
    assert not dialog.list.dragEnabled(), "one row or none: nothing to reorder"
    assert (dialog.width(), dialog.height()) == (300, 470)
    assert (dialog.minimumWidth(), dialog.minimumHeight()) == (200, 280)


def test_without_an_install_it_shows_names_only() -> None:
    settings.set_units_favorites_mode("grid")
    dialog = _dialog()
    dialog.set_entries(_rows(_ARCHER))
    assert dialog.mode() == "names" and dialog.display_combo.currentData() == "names"
    assert not conftest.display_mode_enabled(dialog, "icons") and not conftest.display_mode_enabled(dialog, "grid")
    assert "game install" in dialog.display_combo.toolTip(), "the dropdown says why"
    assert settings.get_units_favorites_mode() == "grid", "the sync never saves over the choice"


def test_it_builds_rows_through_the_catalogs_public_helpers() -> None:
    """One row builder for the catalog and this window, and no reach into catalog_browser's private names."""
    from pathlib import Path

    from descape import favorites_dialog

    source = Path(favorites_dialog.__file__).read_text(encoding="utf-8")
    assert "catalog_browser._" not in source and "fill_rows(" in source


def test_the_dropdown_saves_its_own_mode(fake_install, monkeypatch) -> None:
    from PyQt5.QtWidgets import QListView

    dialog = _dialog()
    dialog.set_entries(_rows(_ARCHER, _CASTLE))
    assert dialog.mode() == settings.UNITS_FAVORITES_MODE_DEFAULT
    conftest.choose_display_mode(dialog, "grid")
    assert dialog.list.viewMode() == QListView.IconMode and _texts(dialog) == ["", ""]
    monkeypatch.setattr(settings, "_units_favorites_mode", None)
    assert settings.get_units_favorites_mode() == "grid"
    assert settings.get_units_catalog_mode() == settings.UNITS_CATALOG_MODE_DEFAULT


def _recording(loader, monkeypatch) -> list[tuple[list[int], object]]:
    asked: list[tuple[list[int], object]] = []
    real = loader.request
    monkeypatch.setattr(loader, "request", lambda consts, *a, **k: (asked.append((list(consts), k.get("civ"))), real(consts, *a, **k)))
    return asked


def test_the_civ_reaches_its_requests_and_an_unchanged_one_asks_nothing(fake_install, monkeypatch) -> None:
    from PyQt5.QtWidgets import QApplication

    dialog = _dialog()
    asked = _recording(dialog._loader, monkeypatch)
    dialog.show()
    try:
        dialog.set_entries(_rows(_ARCHER, _UNKNOWN, RANDOM_VILLAGER))
        QApplication.processEvents()
        assert asked and _UNKNOWN not in asked[-1][0], "an unknown id never reaches the loader"
        assert place_random.preview_const(RANDOM_VILLAGER) in asked[-1][0]
        asked.clear()
        dialog.set_civ(5)
        assert asked and all(civ == 5 for _consts, civ in asked)
        asked.clear()
        dialog.set_civ(5)
        dialog.set_team_color(dialog._team_rgb)
        assert asked == []
        dialog.set_team_color((1, 2, 3))
        assert asked
    finally:
        dialog.close()


def test_an_antiquity_owner_renames_the_rows_and_a_switch_back_restores_them(athenian_bowman) -> None:
    dialog = _dialog()
    dialog.set_entries(_rows(_CASTLE, _ARCHER, _UNKNOWN))
    base = _texts(dialog)
    dialog.list.setCurrentRow(1)
    dialog.set_civ(athenian_bowman)
    assert _texts(dialog) == [base[0], "Bowman", base[2]], "only the Archer has an Athenian name"
    assert dialog.list.item(1).toolTip() == f"Bowman ({_ARCHER}) (favorite)"
    assert dialog.values() == [_CASTLE, _ARCHER, _UNKNOWN]
    assert dialog.list.currentItem().data(Qt.UserRole) == _ARCHER, "the highlight survives the rename"
    dialog._choose_mode("names")
    assert _texts(dialog)[1] == "Bowman", "a mode change keeps the owner's names"
    dialog.set_civ(1)
    assert _texts(dialog) == base


def test_a_renaming_owner_switch_still_asks_for_the_new_civs_portraits(athenian_bowman, fake_install, monkeypatch) -> None:
    from PyQt5.QtWidgets import QApplication

    dialog = _dialog()
    asked = _recording(dialog._loader, monkeypatch)
    dialog.show()
    try:
        dialog.set_entries(_rows(_ARCHER, _CASTLE))
        QApplication.processEvents()
        asked.clear()
        dialog.set_civ(athenian_bowman)
        assert asked and all(civ == athenian_bowman for _consts, civ in asked)
    finally:
        dialog.close()


def test_rows_set_under_an_antiquity_owner_take_its_names(athenian_bowman) -> None:
    dialog = _dialog()
    dialog.set_civ(athenian_bowman)
    dialog.set_entries(_rows(_ARCHER))
    assert _texts(dialog) == ["Bowman"]


def test_a_switch_between_owners_with_the_same_own_names_refills_nothing(athenian_bowman, monkeypatch) -> None:
    dialog = _dialog()
    dialog.set_entries(_rows(_ARCHER, _CASTLE))
    calls = []
    real = dialog._fill
    monkeypatch.setattr(dialog, "_fill", lambda: (calls.append(1), real())[1])
    dialog.set_civ(1)
    dialog.set_civ(2)
    assert calls == [], "two owners without own names"
    dialog.set_civ(athenian_bowman)
    assert calls == [1]
    dialog.set_civ(48)  # the Spartans name the same consts by the same strings
    assert calls == [1]
    assert _texts(dialog)[0] == "Bowman"


def test_the_window_and_the_catalog_both_fill_from_one_loader(fake_install) -> None:
    from PyQt5.QtWidgets import QApplication

    from descape.catalog_browser import CatalogBrowserView
    from descape.catalog_thumbs import ThumbLoader

    conftest.ensure_qapp()
    loader = ThumbLoader()
    view = CatalogBrowserView(loader=loader)
    conftest.choose_display_mode(view, "icons")
    dialog = _dialog(loader)
    view.resize(380, 300)
    view.show()
    dialog.show()
    try:
        QApplication.processEvents()
        dialog.set_entries(_rows(_CASTLE, _ARCHER))
        conftest.settle_thumbs(loader)
        catalog_rows = [v for v in catalog_browser.visible_values(view.list) if v in view._by_id]
        assert len(catalog_rows) > 2
        for value in (*catalog_rows, _CASTLE, _ARCHER):
            assert loader.cached(catalog_browser.thumb_const(value), 32, None)[0], value
    finally:
        dialog.close()
        view.close()


@pytest.mark.parametrize("mode", ["names", "grid"])
def test_a_drop_reports_the_new_order_keeps_every_row_and_picks_nothing(mode, fake_install) -> None:
    from PyQt5.QtCore import QMimeData, QPointF
    from PyQt5.QtGui import QDropEvent
    from PyQt5.QtWidgets import QApplication

    picked, reordered = [], []
    dialog = _dialog(on_pick=picked.append, on_reorder=reordered.append)
    dialog.show()
    try:
        conftest.choose_display_mode(dialog, mode)
        dialog.set_entries(_rows(_ARCHER, _CASTLE, RANDOM_VILLAGER))
        assert dialog.list.dragEnabled()
        dialog.list.setCurrentRow(2)
        first = dialog.list.visualItemRect(dialog.list.item(0))
        before_middle = QPoint(first.left() + 2, first.center().y()) if mode == "grid" else QPoint(first.center().x(), first.top() + 2)

        class Drop(QDropEvent):
            def source(self):
                return dialog.list

        mime = QMimeData()
        event = Drop(QPointF(before_middle), dialog.list.model().supportedDragActions(), mime, Qt.LeftButton, Qt.NoModifier)
        dialog.list.dropEvent(event)
        assert event.dropAction() == Qt.CopyAction
        QApplication.processEvents()
        assert reordered == [[RANDOM_VILLAGER, _ARCHER, _CASTLE]]
        assert sorted(_values(dialog)) == sorted([_ARCHER, _CASTLE, RANDOM_VILLAGER]), "no row removed"
        assert picked == []
    finally:
        dialog.close()


def test_set_entries_keeps_the_highlighted_row_by_id() -> None:
    dialog = _dialog()
    dialog.set_entries(_rows(_ARCHER, _CASTLE))
    dialog.list.setCurrentRow(1)
    dialog.set_entries(_rows(_CASTLE, _ARCHER, RANDOM_VILLAGER))
    assert dialog.list.currentItem().data(Qt.UserRole) == _CASTLE
    dialog.set_entries(_rows(_ARCHER))
    assert dialog.list.currentItem() is None and not dialog.remove_button.isEnabled()


def test_the_viewport_takes_drops_after_every_refill() -> None:
    dialog = _dialog()
    for rows in (_rows(_ARCHER, _CASTLE), _rows(_CASTLE, _ARCHER), _rows(_CASTLE, _ARCHER, RANDOM_VILLAGER)):
        dialog.set_entries(rows)
        assert dialog.list.viewport().acceptDrops()
    dialog.set_entries(_rows(_ARCHER))
    assert not dialog.list.viewport().acceptDrops()


def test_a_closed_window_asks_the_loader_nothing_until_shown(fake_install, monkeypatch) -> None:
    from PyQt5.QtWidgets import QApplication

    dialog = _dialog()
    asked = _recording(dialog._loader, monkeypatch)
    dialog.set_entries(_rows(_ARCHER, _CASTLE))
    dialog.set_civ(5)
    dialog.set_team_color((1, 2, 3))
    assert asked == []
    dialog.show()
    try:
        QApplication.processEvents()
        assert asked and asked[-1][1] == 5
        dialog._loader.request([_CASTLE], 64, client=dialog)
        assert id(dialog) in dialog._loader._pending
    finally:
        dialog.close()
    assert id(dialog) not in dialog._loader._pending, "a closed window keeps nothing queued"
    asked.clear()
    dialog.set_entries(_rows(_CASTLE))
    assert asked == []


def test_the_window_keeps_its_menu_when_the_catalog_toggles(monkeypatch) -> None:
    from PyQt5.QtWidgets import QApplication, QMenu

    settings.set_units_catalog_right_click("toggle")
    opened, removed = [], []
    monkeypatch.setattr(QMenu, "exec_", lambda menu, *_args: opened.append([a.text() for a in menu.actions()]))
    dialog = _dialog(on_remove=removed.append)
    dialog.set_entries(_rows(_ARCHER, _CASTLE))
    dialog.show()
    try:
        QApplication.processEvents()
        dialog._show_remove_menu(dialog.list.visualItemRect(dialog.list.item(1)).center())
        assert opened == [["Remove from Favorites"]] and removed == []
    finally:
        dialog.close()
