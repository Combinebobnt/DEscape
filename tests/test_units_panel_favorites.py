"""GH #195 in descape/units_panel.py: the catalog's favorites store bound to the active
game, the on_favorites_changed and on_placing_art hooks the Favorite Objects window
follows, and pick_object() with the favorites view on. Standalone, as
tests/test_units_panel.py; that module is near its line ceiling."""

from __future__ import annotations

import pytest

from descape import game_dataset, games, settings
from descape.place_random import RANDOM_VILLAGER
from descape.units_panel import UnitsPanel

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_ARCHER = 4
_VILLAGER = 83
_CASTLE = 82
_BUILDER = 118  # in the catalog, hidden from the in-game editor


class _Hooks:
    def __init__(self):
        self.changed = 0
        self.art: list[tuple[tuple[int, int, int], int | None]] = []

    def on_changed(self) -> None:
        self.changed += 1

    def on_art(self, rgb, civ) -> None:
        self.art.append((tuple(rgb), civ))


def _panel(hooks: _Hooks | None = None, **kwargs) -> UnitsPanel:
    conftest.ensure_qapp()
    hooks = hooks or _Hooks()
    return UnitsPanel(on_favorites_changed=hooks.on_changed, on_placing_art=hooks.on_art, **kwargs)


def _favorites_on(panel, *ids) -> None:
    panel.catalog_view.set_favorites(list(ids))
    panel.catalog_view.set_favorites_view(True)


def test_the_store_follows_the_active_game() -> None:
    panel = _panel()
    here = game_dataset.active()
    other = next(game_id for game_id in games.ids() if game_id != here)
    panel.catalog_view.add_favorite(_ARCHER)
    assert settings.get_unit_favorites(here) == [_ARCHER]
    with game_dataset.activated(other):
        assert panel.catalog_view.favorites() == []
        panel.catalog_view.add_favorite(93)
    assert settings.get_unit_favorites(other) == [93]
    assert panel.catalog_view.favorites() == [_ARCHER]


def test_the_favorites_hook_fires_on_a_change_and_every_catalog_refresh() -> None:
    hooks = _Hooks()
    panel = _panel(hooks)
    hooks.changed = 0
    panel.catalog_view._favorite_menu(_ARCHER).actions()[0].trigger()
    assert hooks.changed == 1
    for refresh in (panel.reload_catalog, panel.rename_catalog, panel.refresh_thumbnails):
        before = hooks.changed
        refresh()
        assert hooks.changed == before + 1, refresh.__name__


def test_the_placing_art_hook_follows_the_owner_and_a_civ_refresh() -> None:
    hooks = _Hooks()
    civs = {1: 40, 2: 41}
    panel = _panel(hooks, portrait_civ_for=civs.get)
    assert hooks.art, "__init__ already pushes it"
    combo = panel.owner_combo
    combo.setCurrentIndex(combo.findData(2))
    assert hooks.art[-1] == (panel._team_rgb(2), 41)
    civs[2] = 42
    panel.refresh_portrait_civs()
    assert hooks.art[-1] == (panel._team_rgb(2), 42)


def test_pick_object_takes_the_random_row() -> None:
    panel = _panel()
    assert panel.pick_object(RANDOM_VILLAGER)
    assert panel.catalog_view.current_value() == RANDOM_VILLAGER
    assert panel.selected_object_choices() == (83, 293)


def test_a_favorite_pick_stays_in_the_view_without_ticking_the_box() -> None:
    panel = _panel()
    _favorites_on(panel, _ARCHER, _BUILDER)
    assert panel.pick_object(_BUILDER)
    view = panel.catalog_view
    assert view.in_favorites_view() and view.current_value() == _BUILDER
    assert not view.show_hidden_checkbox.isChecked()


@pytest.mark.parametrize("const", [_VILLAGER, _BUILDER, _CASTLE])
def test_a_non_favorite_pick_keeps_the_view_with_no_row_current(const) -> None:
    panel = _panel()
    _favorites_on(panel, _ARCHER)
    panel.pick_object(_ARCHER)
    assert panel.pick_object(const)
    view = panel.catalog_view
    assert view.in_favorites_view() and view.list.currentItem() is None
    assert not view.show_hidden_checkbox.isChecked()
    assert panel.selected_object_const() == const


def test_leaving_the_view_highlights_the_pick_where_its_category_is_shown() -> None:
    panel = _panel()
    view = panel.catalog_view
    _favorites_on(panel, _ARCHER)
    panel.pick_object(_VILLAGER)
    view.favorites_button.click()
    assert view.category_buttons["Units"].isChecked() and view.current_value() == _VILLAGER
    view.favorites_button.click()
    panel.pick_object(_CASTLE)
    view.favorites_button.click()
    assert view.category_buttons["Units"].isChecked(), "no jump to the pick's category"
    assert view.list.currentItem() is None
    view.favorites_button.click()
    view.category_buttons["Buildings"].click()
    assert view.current_value() == _CASTLE
    assert panel.selected_object_const() == _CASTLE
