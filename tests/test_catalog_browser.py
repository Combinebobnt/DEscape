"""descape/catalog_browser.py (GH #68, TASK-075 S5): the Units panel's
category buttons, class combo, filter and three display modes, standalone."""

from __future__ import annotations

import pytest
from PyQt5.QtCore import Qt

from descape import asset_source, catalog_browser, object_catalog, settings
from descape.catalog_browser import CatalogBrowserView
from descape.place_random import RANDOM_VILLAGER

import conftest
from conftest import ManualExecutor

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

@pytest.fixture(autouse=True)
def _no_install_from_the_environment(monkeypatch):
    """The no-install cases stay so with AOE2DE_INSTALL_PATH set, which outranks the config conftest hides."""
    monkeypatch.delenv("AOE2DE_INSTALL_PATH", raising=False)
    asset_source.clear_install_caches()


_ARCHER = 4
_CASTLE = 82
_BUILDER = 118  # in the catalog, hidden from the in-game editor
_CIVILIAN = 4  # the villagers' class


def _view(executor=None, real_pool=False):
    from descape.catalog_thumbs import ThumbLoader

    conftest.ensure_qapp()
    return CatalogBrowserView(loader=ThumbLoader(executor=None if real_pool else executor or ManualExecutor()))


def _shown(view) -> list[int]:
    return [view.list.item(i).data(Qt.UserRole) for i in range(view.list.count()) if not view.list.item(i).isHidden()]


def _counts(view) -> dict[str, str]:
    return {category: button.text() for category, button in view.category_buttons.items()}


def _visible_count(category: str, needle: str = "") -> int:
    return sum(
        1
        for entry in object_catalog.objects()
        if entry.category == category and not entry.hidden and needle in entry.search_text
    )


def test_it_starts_on_units_with_a_count_on_every_button() -> None:
    view = _view()
    assert view.category_buttons["Units"].isChecked()
    # The Units count includes Villager (Random) (GH #164).
    assert _counts(view) == {
        "Units": f"Units ({_visible_count('Units') + 1})",
        "Buildings": f"Buildings ({_visible_count('Buildings')})",
        "Heroes": f"Heroes ({_visible_count('Heroes')})",
        "Others": f"Other ({_visible_count('Others')})",
    }


def test_the_counts_follow_the_filter_and_hidden_box_but_not_the_class() -> None:
    view = _view()
    before = _counts(view)
    view.class_combo.setCurrentIndex(view.class_combo.findData(_CIVILIAN))
    assert _counts(view) == before, "the class combo narrows only the list"
    view.filter_edit.setText("castle")
    assert _counts(view)["Buildings"] == f"Buildings ({_visible_count('Buildings', 'castle')})"
    assert view.category_buttons["Units"].isChecked(), "a filter hit elsewhere never switches category"
    view.filter_edit.clear()
    view.show_hidden_checkbox.setChecked(True)
    assert _counts(view)["Units"] != before["Units"]


def test_gun_lists_the_gunpowder_keg_under_other() -> None:
    """GH #208: under Other, as in the official editor, with the hidden box unticked."""
    view = _view()
    view.filter_edit.setText("gun")
    assert 2607 not in _shown(view)
    view.category_buttons["Others"].click()
    assert not view.show_hidden_checkbox.isChecked()
    assert 2607 in _shown(view)


def test_the_class_combo_lists_the_categorys_classes_and_resets_on_a_switch() -> None:
    view = _view()
    labels = [view.class_combo.itemText(i) for i in range(view.class_combo.count())]
    assert labels[0].startswith("All classes")
    assert any(label.startswith("Civilian (") for label in labels)
    counts = [int(label.rsplit("(", 1)[1].rstrip(")")) for label in labels[1:]]
    assert counts == sorted(counts, reverse=True)
    view.class_combo.setCurrentIndex(view.class_combo.findData(_CIVILIAN))
    assert set(_shown(view)) >= {83, 293, RANDOM_VILLAGER}
    assert all(object_catalog.entry(object_catalog.objects(), v).class_id == _CIVILIAN for v in _shown(view) if v > 0)
    view.category_buttons["Buildings"].click()
    assert view.class_combo.currentIndex() == 0
    assert view.class_combo.findData(_CIVILIAN) < 0 or view.class_combo.currentData() is None


def test_villager_random_is_a_units_row_under_the_villagers_class() -> None:
    view = _view()
    view.filter_edit.setText("random")
    assert RANDOM_VILLAGER in _shown(view)
    view.filter_edit.setText("villager")
    assert sorted(_shown(view)) == sorted([83, 293, RANDOM_VILLAGER])


def test_select_crosses_category_class_hidden_and_a_hiding_filter() -> None:
    view = _view()
    view.class_combo.setCurrentIndex(view.class_combo.findData(_CIVILIAN))
    view.select(_ARCHER)
    assert view.class_combo.currentData() is None, "a class that excludes it is dropped"
    assert view.current_value() == _ARCHER
    view.select(_CASTLE)
    assert view.category_buttons["Buildings"].isChecked()
    assert view.current_value() == _CASTLE
    view.select(_BUILDER)
    assert view.show_hidden_checkbox.isChecked()
    assert view.current_value() == _BUILDER
    view.filter_edit.setText("this matches nothing at all")
    view.select(_ARCHER)
    assert view.list.currentItem().data(Qt.UserRole) == _ARCHER, "current even though the filter hides it"
    assert view.current_value() is None
    view.select(-12345)
    assert view.list.currentItem().data(Qt.UserRole) == _ARCHER


def test_a_category_switch_never_reports_a_pick() -> None:
    view = _view()
    reported = []
    view.select(_ARCHER)
    view.current_changed.connect(reported.append)
    view.category_buttons["Heroes"].click()
    view.filter_edit.setText("a")
    view.class_combo.setCurrentIndex(1)
    assert [value for value in reported if value is not None] == []


def test_without_an_install_the_icon_modes_are_off_until_refresh(tmp_path) -> None:
    view = _view()
    assert view.mode() == "names"
    assert not view.display_buttons["icons"].isEnabled()
    assert not view.display_buttons["grid"].isEnabled()
    assert view.display_buttons["names"].isChecked()
    asset_source.set_install_path_override(conftest._fake_install(tmp_path))
    try:
        view.refresh_thumbnails()
        assert view.display_buttons["icons"].isEnabled() and view.display_buttons["grid"].isEnabled()
        assert view.mode() == settings.UNITS_CATALOG_MODE_DEFAULT == "icons"
        assert view.display_buttons["icons"].isChecked()
    finally:
        asset_source.set_install_path_override(None)


def test_the_modes_switch_the_view_and_persist(fake_install, monkeypatch) -> None:
    from PyQt5.QtWidgets import QListView

    view = _view()
    view.display_buttons["grid"].click()
    assert view.list.viewMode() == QListView.IconMode
    assert view.list.item(0).text() == ""
    assert view.list.item(0).toolTip()
    # Forget the in-memory value, so the mode is read back from the (isolated) config file.
    monkeypatch.setattr(settings, "_units_catalog_mode", None)
    assert settings.get_units_catalog_mode() == "grid"
    assert _view().mode() == "grid", "a fresh view starts in the saved mode"
    view.display_buttons["names"].click()
    assert view.list.viewMode() == QListView.ListMode
    assert view.list.iconSize().isEmpty()
    view.display_buttons["icons"].click()
    assert view.list.viewMode() == QListView.ListMode
    assert view.list.iconSize().width() == 32
    monkeypatch.setattr(settings, "_units_catalog_mode", None)
    assert settings.get_units_catalog_mode() == "icons"


def test_only_the_rows_on_screen_are_requested(fake_install) -> None:
    from PyQt5.QtWidgets import QApplication

    executor = ManualExecutor()
    view = _view(executor)
    requested = []
    real_request = view._loader.request
    view._loader.request = lambda consts, px, client=None: (requested.append(list(consts)), real_request(consts, px, client))
    view.resize(380, 300)
    view.show()
    try:
        QApplication.processEvents()
        requested.clear()
        view.list.verticalScrollBar().setValue(view.list.verticalScrollBar().maximum())
        QApplication.processEvents()
        assert requested, "a scroll asks for the new rows"
        last = requested[-1]
        viewport = view.list.viewport().rect()
        on_screen = [value for value in _shown(view) if view.list.visualItemRect(view._items[value]).intersects(viewport)]
        assert set(last) <= set(on_screen) | {83}
        assert len(last) <= len(on_screen) < len(_shown(view)) // 4
        assert _shown(view)[0] not in last, "the top row scrolled away"
    finally:
        view.close()


@pytest.mark.parametrize("mode", ["icons", "names"])
def test_the_list_modes_scroll_to_the_widest_name_and_never_clip_it(mode, fake_install) -> None:
    """Plan 075 section 3: the list modes scroll horizontally, never elide or
    clip ("FLAGSHIP OF NEARCHOS DOCKED MOVEABLE" at MIN_USEFUL_WIDTH)."""
    from PyQt5.QtWidgets import QApplication

    from descape.units_panel import UnitsPanel

    view = _view()
    view.display_buttons[mode].click()
    view.resize(UnitsPanel.MIN_USEFUL_WIDTH, 400)
    view.show()
    try:
        overflowed = 0
        for category, _label in catalog_browser.CATEGORIES:
            view.category_buttons[category].click()
            QApplication.processEvents()
            metrics = view.list.fontMetrics()
            items = [view._items[value] for value in _shown(view)]
            widest = max(items, key=lambda item: metrics.horizontalAdvance(item.text()))
            need = metrics.horizontalAdvance(widest.text()) + view.list.iconSize().width()
            viewport = view.list.viewport().width()
            bar = view.list.horizontalScrollBar()
            assert view.list.visualItemRect(widest).width() >= need, (category, widest.text())
            if need <= viewport:
                continue
            overflowed += 1
            assert bar.maximum() > 0, (category, widest.text())
            bar.setValue(bar.maximum())
            QApplication.processEvents()
            assert view.list.visualItemRect(widest).right() <= viewport, (category, widest.text())
            bar.setValue(0)
        assert overflowed, "no name outgrew the pane, so this proves nothing"
    finally:
        view.close()


def test_the_list_takes_focus_only_on_a_click() -> None:
    view = _view()
    assert view.list.focusPolicy() == Qt.ClickFocus
    assert view.class_combo.focusPolicy() == Qt.ClickFocus
    assert all(button.focusPolicy() == Qt.NoFocus for button in view.category_buttons.values())
    assert all(button.focusPolicy() == Qt.NoFocus for button in view.display_buttons.values())


@pytest.mark.parametrize("mode", ["icons", "names", "grid"])
def test_enter_activates_in_every_mode(mode, fake_install) -> None:
    from PyQt5.QtTest import QTest

    view = _view()
    view.display_buttons[mode].click()
    activated = []
    view.activated.connect(activated.append)
    view.select(_ARCHER)
    QTest.keyClick(view.list, Qt.Key_Return)
    assert activated == [_ARCHER]


def test_a_grid_row_with_no_thumbnail_gets_a_glyph(fake_install) -> None:
    view = _view(real_pool=True)
    view.display_buttons["grid"].click()
    no_icon = next(v for v in _shown(view) if object_catalog.icon_for(v) is None and v > 0)
    view._loader.request([no_icon], 48, client="test")
    conftest.settle_thumbs(view._loader)
    assert view._loader.cached(no_icon, 48) == (True, None), "the loader found nothing, so the glyph path ran"
    # Pixels, not isNull(): the transparent placeholder is a non-null icon too.
    image = view._items[no_icon].icon().pixmap(48, 48).toImage()
    drawn = sum(1 for x in range(image.width()) for y in range(image.height()) if image.pixelColor(x, y).alpha() > 0)
    assert drawn > image.width() * image.height() // 2, "a drawn glyph, not the blank placeholder"


def test_the_random_row_shows_its_preview_villager_with_a_badge(fake_install) -> None:
    from PIL import Image

    icon, _building = object_catalog.portrait_for(83)
    portrait = asset_source.get_install_path() / asset_source.UNIT_ICON_DIR / asset_source.UNIT_ICON_NAME.format(icon=icon)
    Image.new("RGBA", (32, 32), conftest._UNIT_RED).save(portrait, format="DDS")
    view = _view(real_pool=True)
    view.select(RANDOM_VILLAGER)
    view._loader.request([83], 32, client="test")
    conftest.settle_thumbs(view._loader)
    image = view._items[RANDOM_VILLAGER].icon().pixmap(32, 32).toImage()
    assert image.pixelColor(8, 8).getRgb() == conftest._UNIT_RED, "the male villager's portrait"
    badge = image.pixelColor(image.width() - 4, image.height() - 7)
    assert badge.red() < 100 or badge.red() == badge.green() == badge.blue(), "a dark or white question-mark badge in the corner"
    plain = view._items[83].icon().pixmap(32, 32).toImage()
    assert plain.pixelColor(image.width() - 4, image.height() - 7).getRgb() == conftest._UNIT_RED


# --- GH #32: GAIA-only objects hidden for a player owner ------------------------

_DEER = 65  # class 9, Prey Animal
_PREY_CLASS = 9


def test_set_excluded_hides_rows_counts_and_an_emptied_class_then_restores_them() -> None:
    view = _view()
    before = _counts(view)
    prey = {e.id for e in object_catalog.objects() if e.class_id == _PREY_CLASS and not e.hidden}
    assert _DEER in prey and _DEER in _shown(view)
    assert view.class_combo.findData(_PREY_CLASS) >= 0

    view.set_excluded(frozenset(prey))

    assert not prey & set(_shown(view))
    assert _counts(view)["Units"] == f"Units ({_visible_count('Units') + 1 - len(prey)})"
    assert _counts(view)["Buildings"] == before["Buildings"]
    assert view.class_combo.findData(_PREY_CLASS) < 0, "a class with every row excluded leaves the combo"
    view.filter_edit.setText("deer")
    assert _DEER not in _shown(view)

    view.set_excluded(frozenset())

    assert _DEER in _shown(view)
    view.filter_edit.clear()
    assert _counts(view) == before
    assert view.class_combo.findData(_PREY_CLASS) >= 0


def test_a_random_pick_is_excluded_only_when_every_choice_is() -> None:
    view = _view()
    view.set_excluded(frozenset({83}))
    assert RANDOM_VILLAGER in _shown(view) and 83 not in _shown(view)
    view.set_excluded(frozenset({83, 293}))
    assert RANDOM_VILLAGER not in _shown(view)


def test_excluded_rows_survive_a_reload_of_the_entries() -> None:
    view = _view()
    view.set_excluded(frozenset({_DEER}))
    view.reload_entries()
    assert _DEER not in _shown(view)


def test_cuman_disabled_is_editor_hidden() -> None:
    """GH #32: the 1.59 editor lists it for neither Player 1 nor GAIA."""
    entry = object_catalog.entry(object_catalog.objects(), 1261)
    assert entry is not None and entry.hidden
    view = _view()
    view.filter_edit.setText("cuman disabled")
    assert 1261 not in _shown(view)
    view.show_hidden_checkbox.setChecked(True)
    assert 1261 in _shown(view)


def test_invisible_spawners_are_listed_under_other_like_the_editor() -> None:
    """GH #32: the 1.59 Other tab lists Invisible Spawner A-J, which the .dat hides."""
    spawners = (2616, 2617, 2618, 2619, 2620, 2621, 2645, 2646, 2647, 2648)
    for const in spawners:
        entry = object_catalog.entry(object_catalog.objects(), const)
        assert entry is not None and entry.category == "Others" and not entry.hidden
    view = _view()
    view.category_buttons["Others"].click()
    assert set(spawners) <= set(_shown(view))


# -- Browse's view: a given catalog (GH #273) ----------------------------------------


def _given_view(rows, default_category="", mode_store=None):
    from descape.catalog_thumbs import ThumbLoader

    conftest.ensure_qapp()
    return CatalogBrowserView(
        ThumbLoader(executor=ManualExecutor()),
        entries=rows,
        default_category=default_category,
        mode_store=mode_store or (lambda: "icons", lambda _mode: None),
    )


def _rows_in(*categories):
    return [entry for entry in object_catalog.objects() if entry.category in categories]


def test_given_entries_carry_no_random_row() -> None:
    assert RANDOM_VILLAGER in _view()._by_id, "the sidebar keeps it, or this proves nothing"
    view = _given_view(object_catalog.objects())
    assert RANDOM_VILLAGER not in view._by_id
    assert RANDOM_VILLAGER not in view._items


@pytest.mark.parametrize("category", ["Buildings", "Others"])
def test_a_one_category_catalog_opens_on_it_with_the_block_hidden(category) -> None:
    view = _given_view(_rows_in(category), default_category="Units")
    assert view.category_buttons[category].isChecked()
    assert view.category_block.isHidden()
    assert _shown(view)


def test_a_catalog_lacking_categories_disables_their_buttons() -> None:
    view = _given_view(_rows_in("Units", "Heroes"))
    assert not view.category_block.isHidden()
    assert {category for category, button in view.category_buttons.items() if button.isEnabled()} == {"Units", "Heroes"}
    assert view.category_buttons["Units"].isChecked()


def test_the_default_category_opens_when_it_has_rows() -> None:
    view = _given_view(object_catalog.objects(), default_category="Buildings")
    assert view.category_buttons["Buildings"].isChecked()
    assert _CASTLE in _shown(view)


def test_the_sidebar_keeps_every_category_live() -> None:
    view = _view()
    assert not view.category_block.isHidden()
    assert all(button.isEnabled() for button in view.category_buttons.values())


def test_a_pinned_hidden_row_shows_with_the_box_unticked_and_no_other_hidden_row() -> None:
    view = _given_view(object_catalog.objects())
    hidden = {entry.id for entry in object_catalog.objects() if entry.hidden}
    assert _BUILDER in hidden
    unpinned_count = view.category_buttons["Units"].text()
    view.pin(_BUILDER)
    view.select(_BUILDER)
    assert view.current_value() == _BUILDER
    assert not view.show_hidden_checkbox.isChecked()
    assert hidden & set(_shown(view)) == {_BUILDER}
    assert view.category_buttons["Units"].text() != unpinned_count
    view.pin(None)
    assert _BUILDER not in _shown(view)


def test_select_still_ticks_the_box_for_a_hidden_row_nobody_pinned() -> None:
    view = _given_view(object_catalog.objects())
    view.select(_BUILDER)
    assert view.show_hidden_checkbox.isChecked()


def test_a_mode_store_writes_its_own_key_and_leaves_the_units_key_alone(fake_install, monkeypatch) -> None:
    store = (settings.get_browse_catalog_mode, settings.set_browse_catalog_mode)
    view = _given_view(_rows_in("Units"), mode_store=store)
    view.display_buttons["grid"].click()
    monkeypatch.setattr(settings, "_browse_catalog_mode", None)
    monkeypatch.setattr(settings, "_units_catalog_mode", None)
    assert settings.get_browse_catalog_mode() == "grid"
    assert settings.get_units_catalog_mode() == settings.UNITS_CATALOG_MODE_DEFAULT
    assert _given_view(_rows_in("Units"), mode_store=store).mode() == "grid"
    assert _view().mode() == settings.UNITS_CATALOG_MODE_DEFAULT


def test_focus_first_row_lands_on_the_first_visible_row() -> None:
    view = _given_view(_rows_in("Units"))
    view.filter_edit.setText("archer")
    view.focus_first_row()
    assert view.list.currentItem().data(Qt.UserRole) == _shown(view)[0]
    assert view.focusWidget() is view.list
    current = view.list.currentItem()
    view.filter_edit.setText("no such object at all")
    view.focus_first_row()
    assert view.list.currentItem() is current, "no visible row is a no-op"


def _ring(image, rect) -> list[int]:
    """Every pixel one in from `rect`'s edge."""
    inset = rect.adjusted(1, 1, -1, -1)
    xs = range(inset.left(), inset.right() + 1)
    ys = range(inset.top(), inset.bottom() + 1)
    across = [image.pixel(x, y) for x in xs for y in (inset.top(), inset.bottom())]
    return across + [image.pixel(x, y) for y in ys for x in (inset.left(), inset.right())]


@pytest.mark.parametrize("sidebar", [True, False], ids=["sidebar", "browse"])
def test_the_grid_frames_the_selected_cell_and_no_other(sidebar, fake_install) -> None:
    from PyQt5.QtGui import QPalette
    from PyQt5.QtWidgets import QApplication

    view = _view() if sidebar else _given_view(_rows_in("Units"))
    view.resize(420, 420)
    view.show()
    try:
        view.display_buttons["grid"].click()
        view.select(_ARCHER)
        QApplication.processEvents()
        image = view.list.viewport().grab().toImage()
        highlight = view.list.palette().color(QPalette.Highlight).rgb()
        area = view.list.viewport().rect()
        selected = view.list.visualItemRect(view._items[_ARCHER])
        assert area.contains(selected), "the selected cell must be on screen, or this proves nothing"
        assert set(_ring(image, selected)) == {highlight}
        others = [item for value, item in view._items.items() if value != _ARCHER and not item.isHidden()]
        on_screen = [rect for rect in map(view.list.visualItemRect, others) if area.contains(rect)]
        assert len(on_screen) > 1
        assert not any(highlight in _ring(image, rect) for rect in on_screen)
    finally:
        view.close()

