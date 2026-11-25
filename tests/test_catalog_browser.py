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
    assert not conftest.display_mode_enabled(view, "icons")
    assert not conftest.display_mode_enabled(view, "grid")
    assert view.display_combo.currentData() == "names"
    asset_source.set_install_path_override(conftest._fake_install(tmp_path))
    try:
        view.refresh_thumbnails()
        assert conftest.display_mode_enabled(view, "icons") and conftest.display_mode_enabled(view, "grid")
        assert view.mode() == settings.UNITS_CATALOG_MODE_DEFAULT == "icons"
        assert view.display_combo.currentData() == "icons"
    finally:
        asset_source.set_install_path_override(None)


def test_the_view_dropdown_says_why_only_names_is_offered_without_an_install(tmp_path) -> None:
    """The Icons/Names/Grid buttons' no-install tooltip, kept on the closed dropdown and its items (GH #195)."""
    from PyQt5.QtCore import Qt

    view = _view()
    combo = view.display_combo
    assert "game install" in combo.toolTip() and combo.toolTip().startswith("Icons and Grid")
    assert "game install" in combo.itemData(combo.findData("grid"), Qt.ToolTipRole)
    asset_source.set_install_path_override(conftest._fake_install(tmp_path))
    try:
        view.refresh_thumbnails()
        assert combo.toolTip() == "Icon and name rows"
        assert combo.itemData(combo.findData("grid"), Qt.ToolTipRole) == "Icons only, name and Unit ID on hover"
    finally:
        asset_source.set_install_path_override(None)


def test_the_modes_switch_the_view_and_persist(fake_install, monkeypatch) -> None:
    from PyQt5.QtWidgets import QListView

    view = _view()
    conftest.choose_display_mode(view, "grid")
    assert view.list.viewMode() == QListView.IconMode
    assert view.list.item(0).text() == ""
    assert view.list.item(0).toolTip()
    # Forget the in-memory value, so the mode is read back from the (isolated) config file.
    monkeypatch.setattr(settings, "_units_catalog_mode", None)
    assert settings.get_units_catalog_mode() == "grid"
    assert _view().mode() == "grid", "a fresh view starts in the saved mode"
    conftest.choose_display_mode(view, "names")
    assert view.list.viewMode() == QListView.ListMode
    assert view.list.iconSize().isEmpty()
    conftest.choose_display_mode(view, "icons")
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
    view._loader.request = lambda consts, px, client=None, **kw: (requested.append(list(consts)), real_request(consts, px, client, **kw))
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
    conftest.choose_display_mode(view, mode)
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
    assert view.display_combo.focusPolicy() == Qt.NoFocus


@pytest.mark.parametrize("mode", ["icons", "names", "grid"])
def test_enter_activates_in_every_mode(mode, fake_install) -> None:
    from PyQt5.QtTest import QTest

    view = _view()
    conftest.choose_display_mode(view, mode)
    activated = []
    view.activated.connect(activated.append)
    view.select(_ARCHER)
    QTest.keyClick(view.list, Qt.Key_Return)
    assert activated == [_ARCHER]


def test_a_grid_row_with_no_thumbnail_gets_a_glyph(fake_install) -> None:
    view = _view(real_pool=True)
    conftest.choose_display_mode(view, "grid")
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


# --- GH #372: the Owner's civ picks the rows' portraits ----------------------------


def test_an_owner_switch_mid_decode_never_puts_the_old_civs_portrait_on_a_row(archer_civ_portrait) -> None:
    from PyQt5.QtWidgets import QApplication

    executor = ManualExecutor()
    view = _view(executor)

    def run_last() -> None:
        executor.run(len(executor.submitted) - 1)
        QApplication.processEvents()

    conftest.choose_display_mode(view, "icons")
    view.filter_edit.setText("archer 4")
    view.set_civ(archer_civ_portrait)
    view.resize(380, 300)
    view.show()
    try:
        QApplication.processEvents()
        assert _ARCHER in _shown(view)
        while executor.icons()[-1] != conftest.CIV_PORTRAIT_ICON:
            run_last()
        view.set_civ(1)  # an owner without an Archer portrait of its own
        run_last()  # the first owner's portrait lands late
        image = view._items[_ARCHER].icon().pixmap(32, 32).toImage()
        assert image.pixelColor(16, 16).getRgb() != conftest.CIV_PORTRAIT_BLUE
        while not view._loader.idle():
            run_last()
        image = view._items[_ARCHER].icon().pixmap(32, 32).toImage()
        assert image.pixelColor(16, 16).getRgb() == conftest._UNIT_RED
        view.set_civ(archer_civ_portrait)
        QApplication.processEvents()
        image = view._items[_ARCHER].icon().pixmap(32, 32).toImage()
        assert image.pixelColor(16, 16).getRgb() == conftest.CIV_PORTRAIT_BLUE, "back from the memo"
    finally:
        view.close()


def test_an_owner_with_no_portrait_never_keeps_the_previous_owners(fake_install, monkeypatch) -> None:
    """An override of -1 is no portrait: the row goes back to the placeholder."""
    from PyQt5.QtWidgets import QApplication

    real = object_catalog._dat_entry

    def entry(section, id_):
        row = real(section, id_)
        return {**row, "civ_icons": {"9": -1}} if section == "objects" and id_ == _ARCHER else row

    monkeypatch.setattr(object_catalog, "_dat_entry", entry)
    monkeypatch.setattr(catalog_browser.catalog_thumbs, "sprite_thumb", lambda *_args: None)
    view = _view(real_pool=True)
    conftest.choose_display_mode(view, "icons")
    view.filter_edit.setText("archer 4")
    view.resize(380, 300)
    view.show()
    try:
        QApplication.processEvents()
        conftest.settle_thumbs(view._loader)
        QApplication.processEvents()
        assert view._items[_ARCHER].icon().pixmap(32, 32).toImage().pixelColor(16, 16).getRgb() == conftest._UNIT_RED
        view.set_civ(9)
        conftest.settle_thumbs(view._loader)
        QApplication.processEvents()
        assert view._items[_ARCHER].icon().pixmap(32, 32).toImage().pixelColor(16, 16).alpha() == 0
    finally:
        view.close()


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
    conftest.choose_display_mode(view, "grid")
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
        conftest.choose_display_mode(view, "grid")
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



# -- GH #195: favorites ------------------------------------------------------------

_UNKNOWN = 32000  # no dataset lists it
_GOLD = catalog_browser._STAR_GOLD.getRgb()


class _Store:
    """An in-memory favorites store: the (getter, setter) pair the sidebar binds to settings."""

    def __init__(self, ids=()):
        self.ids = list(ids)
        self.writes: list[list[int]] = []

    def pair(self):
        return (lambda: list(self.ids), self._set)

    def _set(self, ids) -> None:
        self.ids = list(ids)
        self.writes.append(list(ids))


def _fav_view(store: _Store | None = None, executor=None) -> CatalogBrowserView:
    from descape.catalog_thumbs import ThumbLoader

    conftest.ensure_qapp()
    store = store if store is not None else _Store()
    return CatalogBrowserView(loader=ThumbLoader(executor=executor or ManualExecutor()), favorites=store.pair())


def _other_const() -> int:
    return next(e.id for e in object_catalog.objects() if e.category == "Others" and not e.hidden)


def _recorder(signal) -> list:
    got: list = []
    signal.connect(lambda *args: got.append(args))
    return got


def test_without_a_store_there_is_no_button_menu_or_star() -> None:
    for view in (_view(), _given_view(_rows_in("Units"))):
        assert view.favorites_button is None
        assert view.list.contextMenuPolicy() != Qt.CustomContextMenu
        assert view.favorites() == []
        assert not any(view.list.item(i).data(catalog_browser.FAVORITE_ROLE) for i in range(view.list.count()))
        view.set_favorites([_ARCHER])
        assert view.favorites() == []


def test_the_class_row_holds_the_class_combo_favorites_and_the_view_dropdown() -> None:
    view = _fav_view()
    assert view.favorites_button.text() == "Favorites"
    assert view.favorites_button.isCheckable() and not view.favorites_button.isChecked()
    assert view.favorites_button.focusPolicy() == Qt.NoFocus
    assert view.favorites_button.toolTip() == catalog_browser.FAVORITES_TIP
    row = view.class_combo.parentWidget().layout()
    assert [row.indexOf(w) for w in (view.class_combo, view.favorites_button, view.display_combo)] == sorted(
        row.indexOf(w) for w in (view.class_combo, view.favorites_button, view.display_combo)
    )


def test_the_menu_follows_the_store_and_its_action_toggles_it() -> None:
    store = _Store()
    view = _fav_view(store)
    changed = _recorder(view.favorites_changed)
    menu = view._favorite_menu(_ARCHER)
    assert [a.text() for a in menu.actions()] == ["Add to Favorites"]
    menu.actions()[0].trigger()
    assert store.ids == [_ARCHER] and len(changed) == 1
    menu = view._favorite_menu(_ARCHER)
    assert [a.text() for a in menu.actions()] == ["Remove from Favorites"]
    menu.actions()[0].trigger()
    assert store.ids == [] and len(changed) == 2


def test_an_unknown_id_offers_only_remove() -> None:
    view = _fav_view(_Store([_UNKNOWN]))
    assert [a.text() for a in view._favorite_menu(_UNKNOWN).actions()] == ["Remove from Favorites"]
    view.remove_favorite(_UNKNOWN)
    assert view._favorite_menu(_UNKNOWN).actions() == []


def test_set_and_remove_write_through_the_store_and_emit() -> None:
    store = _Store([_ARCHER])
    view = _fav_view(store)
    changed = _recorder(view.favorites_changed)
    view.set_favorites([_CASTLE, _ARCHER])
    assert store.writes == [[_CASTLE, _ARCHER]] and len(changed) == 1
    view.remove_favorite(_CASTLE)
    assert store.ids == [_ARCHER] and len(changed) == 2
    view.remove_favorite(_CASTLE)
    assert len(store.writes) == 2, "removing a non-favorite writes nothing"
    store.ids = [83]
    assert view.favorites() == [83], "favorites() reads the store, not a copy of its own"


def test_a_right_press_changes_no_current_row() -> None:
    from PyQt5.QtCore import QEvent
    from PyQt5.QtGui import QMouseEvent

    view = _fav_view()
    view.resize(380, 400)
    view.show()
    try:
        view.select(_ARCHER)
        other = next(view.list.item(i) for i in range(view.list.count()) if view.list.item(i).data(Qt.UserRole) != _ARCHER)
        view.list.scrollToItem(other)
        center = view.list.visualItemRect(other).center()
        changes = _recorder(view.current_changed)
        picks = _recorder(view.picked)

        def press(button) -> None:
            view.list.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, center, button, button, Qt.NoModifier))

        press(Qt.RightButton)
        assert view.current_value() == _ARCHER and changes == [] and picks == []
        press(Qt.LeftButton)
        assert view.list.currentItem() is other, "a left press still moves the row, or this proves nothing"
    finally:
        view.close()


def test_a_favorite_carries_the_star_flag_and_a_tooltip_in_both_views() -> None:
    view = _fav_view(_Store([_ARCHER]))
    archer = view._items[_ARCHER]
    entry = view.entry(_ARCHER)
    assert archer.data(catalog_browser.FAVORITE_ROLE)
    assert archer.toolTip().endswith("(favorite)") and archer.text() == entry.name
    others = [item for value, item in view._items.items() if value != _ARCHER]
    assert not any(item.data(catalog_browser.FAVORITE_ROLE) for item in others)
    assert not any("(favorite)" in item.toolTip() for item in others)
    view.add_favorite(83)
    assert view._items[83].data(catalog_browser.FAVORITE_ROLE), "refreshed in place"
    view.set_favorites_view(True)
    assert all(view._items[v].data(catalog_browser.FAVORITE_ROLE) for v in (_ARCHER, 83))
    assert view._items[_ARCHER].toolTip().endswith("(favorite)")


def test_the_filter_finds_a_starred_row_by_name() -> None:
    view = _fav_view(_Store([_ARCHER, _CASTLE]))
    view.filter_edit.setText("archer")
    assert _ARCHER in _shown(view)
    view.set_favorites_view(True)
    assert _ARCHER in _shown(view) and _CASTLE not in _shown(view)


def _favorite_row_spot(view, value: int, grid: bool):
    """The pixel the delegate's star covers for `value`'s row: its centre."""
    from PyQt5.QtCore import QRect

    rect = view.list.visualItemRect(view._items[value])
    radius = catalog_browser._STAR_RADIUS
    if grid:
        disc = 2 * radius + 4
        corner = QRect(rect.right() - disc - 1, rect.top() + 2, disc, disc)
        return corner.center().x(), corner.center().y()
    return round(rect.right() - catalog_browser._STAR_ROOM / 2), rect.center().y()


@pytest.mark.parametrize("mode", ["icons", "names", "grid"])
def test_the_delegate_paints_the_star_on_a_favorite_only(mode, fake_install) -> None:
    from PyQt5.QtGui import QPalette
    from PyQt5.QtWidgets import QApplication

    grid = mode == "grid"
    view = _fav_view(_Store([_ARCHER, _CASTLE]))
    view.resize(420, 420)
    view.show()
    try:
        conftest.choose_display_mode(view, mode)
        view.set_favorites_view(True)
        view.add_favorite(_other_const())
        view.select(_CASTLE)
        QApplication.processEvents()
        image = view.list.viewport().grab().toImage()
        area = view.list.viewport().rect()
        for value in (_ARCHER, _CASTLE):
            assert area.contains(view.list.visualItemRect(view._items[value]))
        assert image.pixelColor(*_favorite_row_spot(view, _ARCHER, grid)).getRgb() == _GOLD
        selected = image.pixelColor(*_favorite_row_spot(view, _CASTLE, grid)).getRgb()
        if grid:
            assert selected == _GOLD, "Grid's corner star stays gold on its dark disc"
        else:
            assert selected == view.list.palette().color(QPalette.HighlightedText).getRgb()
        view.set_favorites_view(False)
        view.category_buttons["Units"].click()
        QApplication.processEvents()
        plain = next(
            v for v, item in view._items.items()
            if v != _ARCHER and not item.isHidden() and area.contains(view.list.visualItemRect(item))
        )
        image = view.list.viewport().grab().toImage()
        assert image.pixelColor(*_favorite_row_spot(view, plain, grid)).getRgb() != _GOLD
    finally:
        view.close()


@pytest.mark.parametrize("mode, extra", [("icons", catalog_browser._STAR_ROOM), ("names", catalog_browser._STAR_ROOM), ("grid", 0)])
def test_the_size_hint_keeps_room_for_the_star_in_the_list_modes(mode, extra, fake_install) -> None:
    from PyQt5.QtWidgets import QStyleOptionViewItem

    view = _fav_view()
    conftest.choose_display_mode(view, mode)
    delegate = view.list.itemDelegate()

    def width() -> int:
        index = view.list.indexFromItem(view._items[_ARCHER])
        option = QStyleOptionViewItem()
        option.initFrom(view.list)
        return delegate.sizeHint(option, index).width()

    before = width()
    view.add_favorite(_ARCHER)
    assert width() == before + extra


def test_the_favorites_view_lists_the_stored_order_across_categories() -> None:
    order = [_CASTLE, _ARCHER, _other_const(), RANDOM_VILLAGER]
    view = _fav_view(_Store(order))
    view.favorites_button.click()
    assert view.in_favorites_view()
    assert _shown(view) == order
    assert not any(button.isChecked() for button in view.category_buttons.values())


def test_the_category_counts_do_not_change_when_the_view_toggles() -> None:
    view = _fav_view(_Store([_BUILDER, _ARCHER, _CASTLE]))
    before = _counts(view)
    view.set_favorites_view(True)
    assert _BUILDER in _shown(view), "the editor-hidden favorite is listed, or this proves nothing"
    assert _counts(view) == before
    view.set_favorites_view(False)
    assert _counts(view) == before


def test_a_category_click_leaves_the_view_and_says_so() -> None:
    view = _fav_view(_Store([_ARCHER]))
    view.category_buttons["Buildings"].click()
    view.favorites_button.click()
    left = _recorder(view.favorites_view_left)
    view.category_buttons["Heroes"].click()
    assert not view.in_favorites_view() and not view.favorites_button.isChecked()
    assert view.category_buttons["Heroes"].isChecked()
    assert len(left) == 1
    view.category_buttons["Units"].click()
    assert len(left) == 1, "a category click outside the view leaves nothing"


def test_the_button_returns_to_the_last_category_and_says_so() -> None:
    view = _fav_view(_Store([_ARCHER]))
    view.category_buttons["Buildings"].click()
    left = _recorder(view.favorites_view_left)
    view.favorites_button.click()
    assert left == []
    view.favorites_button.click()
    assert view.category_buttons["Buildings"].isChecked() and _CASTLE in _shown(view)
    assert len(left) == 1


def test_highlight_lands_only_on_a_listed_shown_row_without_a_signal() -> None:
    view = _fav_view()
    changes = _recorder(view.current_changed)
    view.highlight(_CASTLE)
    assert view.list.currentItem() is None and view.category_buttons["Units"].isChecked(), "no category switch"
    view.highlight(None)
    view.filter_edit.setText("castle")
    view.highlight(_ARCHER)
    assert view.list.currentItem() is None, "a filtered-out row stays unhighlighted"
    view.filter_edit.clear()
    view.highlight(_BUILDER)
    assert view.list.currentItem() is None and not view.show_hidden_checkbox.isChecked(), "no box ticked"
    view.highlight(_ARCHER)
    assert view.current_value() == _ARCHER
    assert changes == []


def test_select_keeps_a_favorite_in_the_view_and_takes_another_value_to_its_category() -> None:
    view = _fav_view(_Store([_ARCHER, _BUILDER]))
    view.set_favorites_view(True)
    view.select(_BUILDER)
    assert view.in_favorites_view() and view.current_value() == _BUILDER
    assert not view.show_hidden_checkbox.isChecked()
    view.select(_CASTLE)
    assert not view.in_favorites_view() and view.category_buttons["Buildings"].isChecked()
    assert view.current_value() == _CASTLE


def test_the_class_combo_counts_the_favorites_and_resets_with_the_view() -> None:
    view = _fav_view(_Store([_ARCHER, 83, _CASTLE, _UNKNOWN]))
    view.class_combo.setCurrentIndex(view.class_combo.findData(_CIVILIAN))
    view.set_favorites_view(True)
    assert view.class_combo.currentIndex() == 0, "a new list starts on All classes"
    assert view.class_combo.itemText(0) == f"{catalog_browser._ALL_CLASSES} (3)", "the unknown id has no class"
    view.class_combo.setCurrentIndex(view.class_combo.findData(_CIVILIAN))
    assert _shown(view) == [83]
    view.set_favorites_view(False)
    assert view.class_combo.currentIndex() == 0


def test_an_excluded_favorite_is_not_listed() -> None:
    view = _fav_view(_Store([_ARCHER, _CASTLE]))
    view.set_excluded(frozenset({_CASTLE}))
    view.set_favorites_view(True)
    assert _shown(view) == [_ARCHER]


def test_an_unknown_id_is_greyed_selectable_matched_by_id_and_never_picked(fake_install) -> None:
    from PyQt5.QtGui import QPalette

    view = _fav_view(_Store([_ARCHER, _UNKNOWN]))
    assert view.entry(_UNKNOWN) is None
    view.set_favorites_view(True)
    item = view._items[_UNKNOWN]
    assert item.text() == f"Unknown object ({_UNKNOWN})"
    assert item.foreground().color() == view.list.palette().color(QPalette.Disabled, QPalette.Text)
    assert item.flags() & Qt.ItemIsSelectable and item.flags() & Qt.ItemIsEnabled
    picks, activations = _recorder(view.picked), _recorder(view.activated)
    view.list.setCurrentItem(item)
    view.list.itemClicked.emit(item)
    view.list.itemActivated.emit(item)
    assert picks == [] and activations == []
    assert view.current_value() is None
    view.filter_edit.setText(str(_UNKNOWN))
    assert _shown(view) == [_UNKNOWN]
    view.filter_edit.setText("archer")
    assert _UNKNOWN not in _shown(view)
    view.filter_edit.clear()
    view.class_combo.setCurrentIndex(1)
    assert _UNKNOWN not in _shown(view), "a class never matches it"


@pytest.mark.parametrize("mode", ["icons", "grid"])
def test_an_unknown_id_never_reaches_the_loader(mode, fake_install, monkeypatch) -> None:
    from PyQt5.QtWidgets import QApplication

    view = _fav_view(_Store([_UNKNOWN, _ARCHER]))
    asked: list[int] = []
    real = view._loader.request
    monkeypatch.setattr(view._loader, "request", lambda consts, *a, **k: (asked.extend(consts), real(consts, *a, **k)))
    view.resize(380, 400)
    view.show()
    try:
        conftest.choose_display_mode(view, mode)
        view.set_favorites_view(True)
        QApplication.processEvents()
        assert _ARCHER in asked and _UNKNOWN not in asked
        view._on_thumb_ready(_UNKNOWN, catalog_browser.icon_px(mode), None, None)
        assert not view._items[_UNKNOWN].icon().isNull(), "it keeps its stand-in"
    finally:
        view.close()


def test_favorites_rows_ask_for_the_owners_civ(fake_install, monkeypatch) -> None:
    from PyQt5.QtWidgets import QApplication

    view = _fav_view(_Store([_CASTLE, _ARCHER]))
    civs: list = []
    real = view._loader.request
    monkeypatch.setattr(view._loader, "request", lambda consts, *a, **k: (civs.append((list(consts), k.get("civ"))), real(consts, *a, **k)))
    view.resize(380, 400)
    view.show()
    try:
        conftest.choose_display_mode(view, "icons")
        view.set_favorites_view(True)
        QApplication.processEvents()
        civs.clear()
        view.set_civ(5)
        assert civs and all(civ == 5 for _consts, civ in civs)
        assert _CASTLE in civs[-1][0]
    finally:
        view.close()


class _SelfDrop:
    """A drop from the list onto itself, offering the actions a real drag of its rows offers
    (the model's): QDropEvent.source() is the list, which a test cannot get from a real drag offscreen."""

    @staticmethod
    def make(widget, pos):
        from PyQt5.QtCore import QMimeData, QPointF
        from PyQt5.QtGui import QDropEvent

        class Event(QDropEvent):
            def source(self):
                return widget

        mime = QMimeData()
        event = Event(QPointF(pos), widget.model().supportedDragActions(), mime, Qt.LeftButton, Qt.NoModifier)
        event.mime = mime  # kept alive with the event
        return event


@pytest.mark.parametrize("mode", ["names", "grid"])
def test_a_drop_reorders_saves_keeps_every_row_and_picks_nothing(mode, fake_install) -> None:
    from PyQt5.QtCore import QPoint
    from PyQt5.QtWidgets import QApplication

    store = _Store([_ARCHER, _CASTLE, 83])
    view = _fav_view(store)
    view.resize(420, 420)
    view.show()
    try:
        conftest.choose_display_mode(view, mode)
        view.set_favorites_view(True)
        view.list.setCurrentItem(view._items[_ARCHER])
        picks, changed = _recorder(view.picked), _recorder(view.favorites_changed)
        last = view.list.visualItemRect(view._items[83])
        past_middle = QPoint(last.right() - 2, last.center().y()) if mode == "grid" else QPoint(last.center().x(), last.bottom() - 2)
        event = _SelfDrop.make(view.list, past_middle)
        view.list.dropEvent(event)
        assert event.isAccepted() and event.dropAction() == Qt.CopyAction, "Copy, so Qt's startDrag removes no row"
        assert store.ids == [_ARCHER, _CASTLE, 83], "saved on the next turn of the event loop"
        QApplication.processEvents()
        assert store.ids == [_CASTLE, 83, _ARCHER]
        assert _shown(view) == [_CASTLE, 83, _ARCHER]
        assert picks == [] and len(changed) == 1
        own = view.list.visualItemRect(view._items[_ARCHER]).center()
        view.list.setCurrentItem(view._items[_ARCHER])
        assert view.list.drop_order(own) is None, "a drop on its own row moves nothing"
    finally:
        view.close()


def test_drag_is_on_only_in_the_favorites_view_with_every_row_shown() -> None:
    from PyQt5.QtWidgets import QAbstractItemView

    view = _fav_view(_Store([_ARCHER, _CASTLE, 83]))

    def drag_on() -> bool:
        on = view.list.dragEnabled()
        assert on == (view.list.dragDropMode() == QAbstractItemView.InternalMove)
        return on

    assert not drag_on(), "never in a category"
    view.set_favorites_view(True)
    assert drag_on()
    view.filter_edit.setText("archer")
    assert not drag_on()
    view.filter_edit.clear()
    assert drag_on()
    view.class_combo.setCurrentIndex(view.class_combo.findData(_CIVILIAN))
    assert not drag_on()
    view.class_combo.setCurrentIndex(0)
    view.set_excluded(frozenset({_CASTLE}))
    assert not drag_on()


def test_reload_entries_keeps_the_favorites_view() -> None:
    view = _fav_view(_Store([_CASTLE, _ARCHER]))
    view.set_favorites_view(True)
    view.reload_entries()
    assert view.in_favorites_view() and view.favorites_button.isChecked()
    assert _shown(view) == [_CASTLE, _ARCHER]


def test_no_install_never_saves_over_the_chosen_mode(monkeypatch) -> None:
    settings.set_units_catalog_mode("grid")
    monkeypatch.setattr(settings, "_units_catalog_mode", None)
    view = _fav_view()
    assert view.mode() == "names" and view.display_combo.currentData() == "names"
    view.refresh_thumbnails()
    monkeypatch.setattr(settings, "_units_catalog_mode", None)
    assert settings.get_units_catalog_mode() == "grid"


def test_the_viewport_takes_drops_after_every_refill_of_the_view() -> None:
    """Static movement turns the viewport's drops off directly, and the view's own setter skips
    an unchanged value: without the viewport's, a second drag would be refused."""
    view = _fav_view(_Store([_ARCHER, _CASTLE, 83]))
    view.set_favorites_view(True)
    assert view.list.viewport().acceptDrops()
    view.set_favorites([_CASTLE, _ARCHER, 83])
    assert view.list.viewport().acceptDrops(), "after a reorder's rebuild"
    view.remove_favorite(83)
    assert view.list.viewport().acceptDrops(), "after a removal's rebuild"
    view.set_favorites_view(False)
    assert not view.list.viewport().acceptDrops()


def test_a_drop_leaves_the_list_out_of_its_dragging_state(fake_install) -> None:
    from PyQt5.QtCore import QPoint
    from PyQt5.QtWidgets import QAbstractItemView

    view = _fav_view(_Store([_ARCHER, _CASTLE, 83]))
    view.set_favorites_view(True)
    view.list.setCurrentItem(view._items[_ARCHER])
    view.list.setState(QAbstractItemView.DraggingState)
    view.list.dropEvent(_SelfDrop.make(view.list, QPoint(2, 2)))
    assert view.list.state() == QAbstractItemView.NoState


def test_a_grid_drop_between_cells_lands_there_and_below_the_last_row_at_the_end(fake_install) -> None:
    from PyQt5.QtCore import QPoint

    order = [_ARCHER, _CASTLE, 83, 293]
    view = _fav_view(_Store(order))
    view.resize(420, 420)
    view.show()
    try:
        conftest.choose_display_mode(view, "grid")
        view.set_favorites_view(True)
        first, second = (view.list.visualItemRect(view._items[v]) for v in order[:2])
        assert first.top() == second.top() and second.left() - first.right() > 1, "cells on one line with a gap"
        view.list.setCurrentItem(view._items[293])
        for x in range(first.right() + 1, second.left()):
            gap = QPoint(x, first.center().y())
            assert view.list.itemAt(gap) is None, "in the gap, or this proves nothing"
            assert view.list.drop_order(gap) == [_ARCHER, 293, _CASTLE, 83], x
        view.list.setCurrentItem(view._items[_ARCHER])
        last = view.list.visualItemRect(view._items[293])
        assert view.list.drop_order(QPoint(last.center().x(), last.bottom() + 4)) == [_CASTLE, 83, 293, _ARCHER]
    finally:
        view.close()


# -- the star's style and the right-click setting (GH #195's amendment) ------------


def _star_pixels(view, value: int, grid: bool) -> list[tuple[int, int, int, int]]:
    """Every pixel in a box around `value`'s star."""
    from PyQt5.QtWidgets import QApplication

    QApplication.processEvents()
    image = view.list.viewport().grab().toImage()
    x, y = _favorite_row_spot(view, value, grid)
    reach = catalog_browser._STAR_RADIUS + 2
    return [image.pixelColor(px, py).getRgb() for px in range(x - reach, x + reach + 1) for py in range(y - reach, y + reach + 1)]


def _near(rgba, rgb, tolerance: int = 30) -> bool:
    return all(abs(a - b) <= tolerance for a, b in zip(rgba[:3], rgb, strict=True))


@pytest.mark.parametrize("mode", ["icons", "names"])
def test_the_gold_star_has_a_dark_outline_and_mono_is_black_or_white(mode, fake_install) -> None:
    view = _fav_view(_Store([_ARCHER, _CASTLE]))
    view.resize(420, 420)
    view.show()
    try:
        conftest.choose_display_mode(view, mode)
        view.set_favorites_view(True)
        outline = catalog_browser._STAR_OUTLINE.getRgb()[:3]
        gold = _star_pixels(view, _ARCHER, grid=False)
        assert gold[len(gold) // 2] == _GOLD, "gold at the centre"
        assert any(_near(px, outline) for px in gold), "the dark outline at the edge"
        settings.set_favorite_star_style("mono")
        mono = _star_pixels(view, _ARCHER, grid=False)
        assert mono[len(mono) // 2] == catalog_browser.mono_star_colour(view.list.palette()).getRgb()
        assert _GOLD not in mono and not any(_near(px, outline, 10) for px in mono)
    finally:
        view.close()


@pytest.mark.parametrize(
    "preset, expected",
    [("light", (0, 0, 0)), ("editor_parchment_red", (0, 0, 0)), ("editor_dark_wood_red", (255, 255, 255)), ("dark", (255, 255, 255))],
)
def test_mono_is_pure_black_or_white_whatever_the_themes_text_colour(preset, expected) -> None:
    """Dark Wood's text colour is a dim cream (#ddc8b1): mono never takes it."""
    from PyQt5.QtWidgets import QApplication

    from descape.viewer_dialogs import build_theme_palette

    conftest.ensure_qapp()
    palette = build_theme_palette(QApplication.instance().style().standardPalette(), preset, {})
    assert catalog_browser.mono_star_colour(palette).getRgb()[:3] == expected


def test_the_grid_star_turns_white_on_its_disc_in_mono(fake_install) -> None:
    view = _fav_view(_Store([_ARCHER, _CASTLE]))
    view.resize(420, 420)
    view.show()
    try:
        conftest.choose_display_mode(view, "grid")
        view.set_favorites_view(True)
        pixels = _star_pixels(view, _ARCHER, grid=True)
        assert pixels[len(pixels) // 2] == _GOLD
        settings.set_favorite_star_style("mono")
        pixels = _star_pixels(view, _ARCHER, grid=True)
        assert pixels[len(pixels) // 2] == (255, 255, 255, 255)
    finally:
        view.close()


def _no_menu(monkeypatch) -> None:
    from PyQt5.QtWidgets import QMenu

    def refuse(*_args):
        raise AssertionError("a toggle right-click opens no menu")

    monkeypatch.setattr(QMenu, "exec_", refuse)


def _right_click(view, value: int) -> None:
    from PyQt5.QtWidgets import QApplication

    QApplication.processEvents()
    view._show_favorite_menu(view.list.visualItemRect(view._items[value]).center())


def test_a_toggle_right_click_adds_and_removes_with_no_menu(monkeypatch) -> None:
    assert settings.get_units_catalog_right_click() == "menu", "the default"
    settings.set_units_catalog_right_click("toggle")
    _no_menu(monkeypatch)
    store = _Store()
    view = _fav_view(store)
    view.resize(380, 400)
    view.show()
    try:
        current = view.current_value()
        changed = _recorder(view.favorites_changed)
        _right_click(view, _ARCHER)
        assert store.ids == [_ARCHER] and len(changed) == 1
        assert view._items[_ARCHER].data(catalog_browser.FAVORITE_ROLE)
        _right_click(view, _ARCHER)
        assert store.ids == [] and len(changed) == 2
        assert view.current_value() == current, "the pick never moves"
    finally:
        view.close()


def test_a_toggle_right_click_removes_an_unknown_favorite(monkeypatch) -> None:
    settings.set_units_catalog_right_click("toggle")
    _no_menu(monkeypatch)
    store = _Store([_ARCHER, _UNKNOWN])
    view = _fav_view(store)
    view.resize(380, 400)
    view.show()
    try:
        view.set_favorites_view(True)
        _right_click(view, _UNKNOWN)
        assert store.ids == [_ARCHER]
    finally:
        view.close()


def test_a_right_click_opens_the_menu_by_default(monkeypatch) -> None:
    from PyQt5.QtWidgets import QMenu

    opened = []
    monkeypatch.setattr(QMenu, "exec_", lambda menu, *_args: opened.append([a.text() for a in menu.actions()]))
    store = _Store()
    view = _fav_view(store)
    view.resize(380, 400)
    view.show()
    try:
        _right_click(view, _ARCHER)
        assert opened == [["Add to Favorites"]] and store.ids == []
    finally:
        view.close()


# -- an antiquity owner's own names (an Athenian Archer reads Bowman) ------------------


def _texts(view) -> list[str]:
    return [view.list.item(i).text() for i in range(view.list.count())]


def test_an_antiquity_owner_renames_sorts_and_still_finds_by_the_standard_name(athenian_bowman) -> None:
    view = _view()
    base = view._items[_ARCHER].text()
    view.set_civ(athenian_bowman)
    item = view._items[_ARCHER]
    assert item.text() == "Bowman"
    assert item.toolTip() == "Bowman (4)"
    assert view.entry(_ARCHER).name == base, "entry() keeps the standard name"
    texts = _texts(view)
    assert texts == sorted(texts), "rows sort by the shown name"
    for needle in ("archer", "bowman"):
        view.filter_edit.setText(needle)
        assert _ARCHER in _shown(view), needle
    view.filter_edit.clear()
    view.set_civ(1)
    assert view._items[_ARCHER].text() == base


def test_a_switch_between_owners_with_the_same_own_names_repopulates_nothing(athenian_bowman, monkeypatch) -> None:
    view = _view()
    calls = []
    real = view._populate
    monkeypatch.setattr(view, "_populate", lambda: (calls.append(1), real())[1])
    view.set_civ(1)
    view.set_civ(2)
    assert calls == [], "two owners without own names"
    view.set_civ(athenian_bowman)
    assert calls == [1]
    view.set_civ(48)  # the Spartans name every one of the same consts
    assert calls == [1]
    assert view._items[_ARCHER].text() == "Bowman"


def test_a_switch_between_two_antiquity_owners_with_different_names_renames(athenian_bowman, monkeypatch) -> None:
    view = _view()
    view.set_civ(athenian_bowman)
    real = object_catalog.civ_name
    spartan = lambda oid, civ, lang=None: "Spartan Bowman" if (oid, civ) == (_ARCHER, 48) else real(oid, civ, lang)  # noqa: E731
    monkeypatch.setattr(object_catalog, "civ_name", spartan)
    view.set_civ(48)  # the same named ids, another name
    assert view._items[_ARCHER].text() == "Spartan Bowman"


def test_the_favorites_view_shows_the_owners_name(athenian_bowman) -> None:
    store = [_ARCHER]
    conftest.ensure_qapp()
    view = CatalogBrowserView(favorites=(lambda: list(store), lambda ids: store.__setitem__(slice(None), ids)))
    view.set_civ(athenian_bowman)
    view.set_favorites_view(True)
    assert view._items[_ARCHER].text() == "Bowman"
    assert view._items[_ARCHER].toolTip() == "Bowman (4) (favorite)"
