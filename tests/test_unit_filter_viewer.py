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

import conftest
from descape.scenario_io import BLANK_TEMPLATE_PATH
from descape.unit_filter import UnitFilter

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
        assert window._current_unit_filter() == UnitFilter()
        assert window._current_unit_filter().is_default
    finally:
        conftest.close_window(window)


def test_toggling_gaia_and_trees_builds_the_matching_filter() -> None:
    window = conftest.blank_window()
    try:
        window.show_gaia_action.setChecked(False)
        assert window._unit_filter == UnitFilter(show_gaia=False)
        window.show_trees_action.setChecked(False)
        assert window._unit_filter == UnitFilter(show_gaia=False, show_trees=False)
        window.show_gaia_action.setChecked(True)
        window.show_trees_action.setChecked(True)
        assert window._unit_filter.is_default
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
        assert window._unit_filter.is_default
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
        assert window._cache.unit_filter == UnitFilter(show_gaia=False)
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
        assert window._cache.unit_filter == UnitFilter(show_trees=False)
    finally:
        conftest.close_window(window)


def test_filter_is_not_persisted_across_windows() -> None:
    """Deliberately per-session: a persisted filter could present an
    apparently unit-less map on next launch with no visible cause."""
    first = conftest.blank_window()
    try:
        first.show_gaia_action.setChecked(False)
        assert not first._unit_filter.is_default
    finally:
        conftest.close_window(first)

    second = conftest.blank_window()
    try:
        assert second._unit_filter.is_default
        assert second.show_gaia_action.isChecked()
    finally:
        conftest.close_window(second)
