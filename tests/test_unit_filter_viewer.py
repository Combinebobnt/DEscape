"""The Filters toolbar popup, driven through a real offscreen ViewerWindow
-- phase 3's P3-b.

Same technique and default-tier rationale as tests/test_toolbar_params.py:
assert on QAction handles and on the live chunk cache's own state, never on
widget isVisible() (which is unconditionally False for a never-shown window
under QT_QPA_PLATFORM=offscreen).

Every ViewerWindow() constructed here calls edit_history.mark_saved() before
close() -- see tests/test_fill_tool.py's module docstring for why.
"""

from __future__ import annotations

import pytest

from descape.scenario_io import BLANK_TEMPLATE_PATH
from descape.unit_filter import UnitFilter

import conftest


# The app's own default menu state is NOT UnitFilter(): GH #42's Show
# Garrisoned Units ships unchecked, so every "nothing has been toggled yet"
# expectation below reads through this helper rather than UnitFilter().
def app_default(**overrides) -> UnitFilter:
    return UnitFilter(show_garrisoned=False, **overrides)


pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


def test_filters_button_needs_a_map_but_not_a_writable_one() -> None:
    """Filtering is read-only and style-independent, so it gates on
    has_map alone -- not write_ok, not the sloped-editable check every paint
    tool needs."""
    window = conftest.blank_window(load=False)
    try:
        assert not window.filters_button.isEnabled()
        window.load_scenario(BLANK_TEMPLATE_PATH)
        assert window.filters_button.isEnabled()
        window.terrain_style_combo.setCurrentText("Sloped")
        assert window.filters_button.isEnabled(), "Sloped filters fine, it just can't pick"
    finally:
        conftest.close_window(window)


def test_default_menu_state_is_the_default_filter() -> None:
    window = conftest.blank_window()
    try:
        assert window._current_unit_filter() == app_default()
        assert UnitFilter().is_default
    finally:
        conftest.close_window(window)


def test_toggling_gaia_and_trees_builds_the_matching_filter() -> None:
    window = conftest.blank_window()
    try:
        window.show_gaia_action.setChecked(False)
        assert window._unit_filter == app_default(show_gaia=False)
        window.show_trees_action.setChecked(False)
        assert window._unit_filter == app_default(show_gaia=False, show_trees=False)
        window.show_gaia_action.setChecked(True)
        window.show_trees_action.setChecked(True)
        assert window._unit_filter == app_default()
    finally:
        conftest.close_window(window)


def test_all_players_checked_yields_none_not_a_full_set() -> None:
    """players must stay None when everything is checked, so an all-checked
    menu compares equal to UnitFilter() -- that equality is what lets
    set_unit_filter() skip a full-canvas eviction it doesn't need."""
    window = conftest.blank_window()
    try:
        assert window._current_unit_filter().players is None
        window.player_actions[3].setChecked(False)
        assert window._current_unit_filter().players == frozenset({1, 2, 4, 5, 6, 7, 8})
        window.player_actions[3].setChecked(True)
        assert window._current_unit_filter().players is None
    finally:
        conftest.close_window(window)


def test_no_players_is_distinguishable_from_all_players() -> None:
    window = conftest.blank_window()
    try:
        window._set_all_players(False)
        assert window._unit_filter.players == frozenset()
        assert not window._unit_filter.is_default
        window._set_all_players(True)
        assert window._unit_filter == app_default()
    finally:
        conftest.close_window(window)


def test_bulk_player_toggle_applies_once_not_once_per_player() -> None:
    """Each apply evicts and recomposites the whole canvas, so the bulk
    actions must block the per-action signals rather than firing eight
    times."""
    window = conftest.blank_window()
    try:
        calls = []
        real = window._cache.set_unit_filter
        window._cache.set_unit_filter = lambda f: (calls.append(f), real(f))[1]

        window._set_all_players(False)
        assert len(calls) == 1, f"expected one apply, got {len(calls)}"
    finally:
        conftest.close_window(window)


def test_hide_all_clears_gaia_trees_and_players_too() -> None:
    """_set_all_filters is the menu's "Show All"/"Hide All" shortcuts --
    unlike _set_all_players it must also reach show_gaia_action and
    show_trees_action, which the per-player bulk actions leave alone."""
    window = conftest.blank_window()
    try:
        window._set_all_filters(False)
        assert not window.show_gaia_action.isChecked()
        assert not window.show_trees_action.isChecked()
        assert window._unit_filter.players == frozenset()
        assert not window._unit_filter.is_default

        window._set_all_filters(True)
        assert window.show_gaia_action.isChecked()
        assert window.show_trees_action.isChecked()
        assert window._unit_filter.is_default
    finally:
        conftest.close_window(window)


def test_show_all_hide_all_applies_once_not_once_per_entry() -> None:
    window = conftest.blank_window()
    try:
        calls = []
        real = window._cache.set_unit_filter
        window._cache.set_unit_filter = lambda f: (calls.append(f), real(f))[1]

        window._set_all_filters(False)
        assert len(calls) == 1, f"expected one apply, got {len(calls)}"
    finally:
        conftest.close_window(window)


def test_filter_reaches_the_live_cache() -> None:
    window = conftest.blank_window()
    try:
        window.show_gaia_action.setChecked(False)
        assert window._cache.unit_filter == app_default(show_gaia=False)
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("style", ["Flat", "Stepped", "Sloped"])
def test_filter_survives_a_terrain_style_switch(style: str) -> None:
    """Switching style rebuilds the cache from scratch, so the current
    filter has to be passed into the NEW cache -- otherwise hiding GAIA and
    then switching to Stepped would silently bring every tree back."""
    window = conftest.blank_window()
    try:
        window.show_trees_action.setChecked(False)
        window.terrain_style_combo.setCurrentText(style)
        assert window._cache.unit_filter == app_default(show_trees=False)
    finally:
        conftest.close_window(window)


def test_filter_is_not_persisted_across_windows() -> None:
    """Deliberately per-session: a persisted filter could present an
    apparently unit-less map on next launch with no visible cause."""
    first = conftest.blank_window()
    try:
        first.show_gaia_action.setChecked(False)
        assert first._unit_filter != app_default()
    finally:
        conftest.close_window(first)

    second = conftest.blank_window()
    try:
        assert second._unit_filter == app_default()
        assert second.show_gaia_action.isChecked()
    finally:
        conftest.close_window(second)


# --- GH #65: Show Walls / Show Eye Candy -------------------------------

_WALL_CONST = 117  # WALL2, a unit_kind.wall_consts() member
_EYE_CANDY_CONST = 1358  # Grass Green


def test_the_two_new_actions_exist_and_ship_checked() -> None:
    window = conftest.blank_window()
    try:
        assert window.show_walls_action.isChecked()
        assert window.show_eye_candy_action.isChecked()
        assert window._current_unit_filter() == app_default()
    finally:
        conftest.close_window(window)


def test_toggling_walls_and_eye_candy_builds_the_matching_filter() -> None:
    window = conftest.blank_window()
    try:
        window.show_walls_action.setChecked(False)
        assert window._unit_filter == app_default(show_walls=False)
        window.show_eye_candy_action.setChecked(False)
        assert window._unit_filter == app_default(show_walls=False, show_eye_candy=False)
        window.show_walls_action.setChecked(True)
        window.show_eye_candy_action.setChecked(True)
        assert window._unit_filter == app_default()
    finally:
        conftest.close_window(window)


def test_the_new_toggles_reach_the_live_cache() -> None:
    window = conftest.blank_window()
    try:
        window.show_walls_action.setChecked(False)
        assert window._cache.unit_filter == app_default(show_walls=False)
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("action_name", ["show_walls_action", "show_eye_candy_action"])
def test_one_toggle_costs_exactly_one_apply(action_name: str) -> None:
    """Each apply evicts and recomposites the whole canvas, so a single
    checkbox must not fan out into several set_unit_filter() calls."""
    window = conftest.blank_window()
    try:
        calls = []
        real = window._cache.set_unit_filter
        window._cache.set_unit_filter = lambda f: (calls.append(f), real(f))[1]

        getattr(window, action_name).setChecked(False)
        assert len(calls) == 1, f"expected one apply, got {len(calls)}"
    finally:
        conftest.close_window(window)


def test_hide_all_and_show_all_reach_the_new_toggles_too() -> None:
    """_set_all_filters is the menu's bulk shortcut; a new kind toggle that
    it misses would leave "Hide All" visibly not hiding everything."""
    window = conftest.blank_window()
    try:
        window._set_all_filters(False)
        assert not window.show_walls_action.isChecked()
        assert not window.show_eye_candy_action.isChecked()

        window._set_all_filters(True)
        assert window.show_walls_action.isChecked()
        assert window.show_eye_candy_action.isChecked()
        assert window._unit_filter.is_default
    finally:
        conftest.close_window(window)


def test_the_new_toggles_appear_in_the_filter_summary() -> None:
    window = conftest.blank_window()
    try:
        window.show_walls_action.setChecked(False)
        assert "walls hidden" in window._filter_summary()
        window.show_eye_candy_action.setChecked(False)
        assert "eye candy hidden" in window._filter_summary()
    finally:
        conftest.close_window(window)


def _place_wall(window, unit_const: int = _WALL_CONST):
    """Places a wall (or unit_const) through the model and returns (unit, its
    on-screen centre). The click point comes from the entry's own footprint polygon
    rather than tile*tile_pixels: the default style is isometric, so a
    top-down position resolves to no tile at all -- and a test built on one
    would then "pass" for the wrong reason the moment walls were hidden."""
    from PyQt5.QtCore import QPointF

    model = window._ensure_unit_edits()
    occupied = {(int(u.x), int(u.y)) for units in window.scenario.unit_manager.units for u in units}
    tile = next((x, y) for x in range(40, 60) for y in range(40, 60) if (x, y) not in occupied)
    with window._unit_edit(model, "Place wall", [1]):
        unit = model.add(1, unit_const, tile[0] + 0.5, tile[1] + 0.5)
    mv = window.map_view
    entry = next(e for e in mv._unit_index.entries if e.unit is unit)
    poly = mv._unit_polygons_for(entry)[0]
    return unit, QPointF(sum(x for x, _y in poly) / len(poly), sum(y for _x, y in poly) / len(poly))


def test_a_hidden_wall_is_unpickable() -> None:
    """The half that makes this a Filters entry rather than a View one: a
    hidden unit is absent from UnitIndex entirely, so it cannot be picked
    through the tile it no longer paints on."""
    window = conftest.blank_window()
    try:
        window.mode_combo.setCurrentText("Units")
        _unit, pos = _place_wall(window)
        mv = window.map_view
        assert mv.pick_unit_at(pos) is not None, "the wall was unpickable to begin with"

        window.show_walls_action.setChecked(False)
        assert mv.pick_unit_at(pos) is None
        window.show_walls_action.setChecked(True)
        assert mv.pick_unit_at(pos) is not None
    finally:
        conftest.close_window(window)


def test_placing_a_unit_the_filter_hides_says_so() -> None:
    """The cliff tool's "(Show GAIA is off...)" cue, generalized to the whole
    filter: placing a wall with Show Walls off looks exactly like the tool
    doing nothing.

    Asserted through on_unit_place()'s own status line rather than through a
    const set, so it also covers eye candy, trees and owner -- and whatever
    gate the next Filters entry adds."""
    from PyQt5.QtCore import Qt

    window = conftest.blank_window()
    try:
        window.mode_combo.setCurrentText("Units")
        _unit, pos = _place_wall(window)
        window.units_panel.select_object(_WALL_CONST)
        assert window.units_panel.selected_object_const() == _WALL_CONST

        window.on_unit_place(pos, Qt.NoModifier)
        assert "Placed" in _last_status(window)
        assert "Filters" not in _last_status(window)

        window.show_walls_action.setChecked(False)
        window.on_unit_place(pos, Qt.NoModifier)
        assert "Filters" in _last_status(window), _last_status(window)
    finally:
        conftest.close_window(window)


def _last_status(window) -> str:
    return window.status_log.toPlainText().splitlines()[-1]


# --- GH #53: Show Invisible Objects -------------------------------------

_INVISIBLE_OBJECT_A = 1291  # a unit_kind.invisible_consts() member, player-owned by design


def test_show_invisible_exists_ships_checked_and_reaches_the_cache() -> None:
    window = conftest.blank_window()
    try:
        assert window.show_invisible_action.isChecked()
        window.show_invisible_action.setChecked(False)
        assert window._unit_filter == app_default(show_invisible=False)
        assert window._cache.unit_filter == app_default(show_invisible=False)
        assert "invisible objects hidden" in window._filter_summary()
        window._set_all_filters(False)
        window._set_all_filters(True)
        assert window.show_invisible_action.isChecked()
        # Show All checks Show Garrisoned too, so this one IS the unfiltered filter.
        assert window._unit_filter.is_default
    finally:
        conftest.close_window(window)


def test_show_invisible_is_bound_as_a_rebindable_action() -> None:
    window = conftest.blank_window()
    try:
        assert window._keybind_actions["filter_show_invisible"] is window.show_invisible_action
    finally:
        conftest.close_window(window)


def test_a_hidden_invisible_object_is_unpickable() -> None:
    window = conftest.blank_window()
    try:
        window.mode_combo.setCurrentText("Units")
        _unit, pos = _place_wall(window, _INVISIBLE_OBJECT_A)
        mv = window.map_view
        assert mv.pick_unit_at(pos) is not None, "the object was unpickable to begin with"

        window.show_invisible_action.setChecked(False)
        assert mv.pick_unit_at(pos) is None
        window.show_invisible_action.setChecked(True)
        assert mv.pick_unit_at(pos) is not None
    finally:
        conftest.close_window(window)


# --- GH #42: Show Garrisoned Units --------------------------------------


def test_show_garrisoned_ships_unchecked_and_is_the_app_default() -> None:
    """The one Filters entry that ships off: a scenario's occupants are drawn
    stacked on their host until it is turned on."""
    window = conftest.blank_window()
    try:
        assert not window.show_garrisoned_action.isChecked()
        assert window._unit_filter == app_default()
        assert window._cache.unit_filter == app_default()
        assert "garrisoned units hidden" in window._filter_summary()
        window.show_garrisoned_action.setChecked(True)
        assert window._unit_filter.is_default
        assert window._cache.unit_filter.is_default
    finally:
        conftest.close_window(window)


def test_show_garrisoned_is_bound_as_a_rebindable_action() -> None:
    window = conftest.blank_window()
    try:
        assert window._keybind_actions["filter_show_garrisoned"] is window.show_garrisoned_action
    finally:
        conftest.close_window(window)


def test_a_garrisoned_unit_is_unpickable_until_the_toggle_is_on() -> None:
    window = conftest.blank_window()
    try:
        window.mode_combo.setCurrentText("Units")
        host, _pos = _place_wall(window, 79)  # Watch Tower
        model = window._ensure_unit_edits()
        with window._unit_edit(model, "Garrison", [1]):
            occupant = model.add(1, 4, host.x, host.y, garrisoned_in_id=host.reference_id)
        window._rebuild_unit_index()
        mv = window.map_view
        assert all(e.unit is not occupant for e in mv._unit_index.entries)

        window.show_garrisoned_action.setChecked(True)
        assert any(e.unit is occupant for e in mv._unit_index.entries)
    finally:
        conftest.close_window(window)
