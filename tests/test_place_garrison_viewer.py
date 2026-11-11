"""TASK-549: Place Unit's click on a garrison host puts the object inside.

The preview half lives in test_place_preview.py, whose window and host
helpers this reuses; both ask the same ViewerWindow pair
(_place_garrison_host(), _place_garrison_refusal()), so a click here and the
tint there cannot disagree.
"""

from __future__ import annotations

import pytest
from PyQt5.QtCore import Qt
from test_place_preview import _ARCHER, _HOUSE, _KNIGHT, _close, _free_tile, _host_entry, _window, add_host

from descape import place_rotation

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


def _tile_pos(window, tile):
    tp = window.map_view._tile_pixels
    from PyQt5.QtCore import QPointF

    return QPointF((tile[0] + 0.5) * tp, (tile[1] + 0.5) * tp)


def _click(window, host_or_tile) -> None:
    tile = host_or_tile if isinstance(host_or_tile, tuple) else (int(host_or_tile.x), int(host_or_tile.y))
    window.on_unit_place(_tile_pos(window, tile), Qt.NoModifier)


def _units(window) -> list:
    return [u for units in window.scenario.unit_manager.units for u in units]


def _inside(window, host) -> list:
    return [u for u in _units(window) if u.garrisoned_in_id == host.reference_id]


def _last_status(window) -> str:
    return window.status_log.toPlainText().splitlines()[-1]


def _setup(const: int = _ARCHER, owner: int = 1):
    window = _window(const=const)
    host = add_host(window)
    window.units_panel.select_owner(owner)
    return window, host


def test_a_click_on_a_host_with_room_puts_the_object_inside_in_one_record() -> None:
    window, host = _setup()
    try:
        before = len(_units(window))
        records = len(window.edit_history.records)
        rotation = place_rotation.rotation_for(window._place_rotation, _ARCHER, 1)

        _click(window, host)

        (unit,) = _inside(window, host)
        assert len(_units(window)) == before + 1
        assert (unit.x, unit.y, unit.z) == (host.x, host.y, host.z)
        assert unit.unit_const == _ARCHER
        assert unit.rotation == rotation
        assert unit in window.scenario.unit_manager.units[1]
        assert len(window.edit_history.records) == records + 1
        assert window.edit_history.peek_undo().label == "Place unit"
        assert window._selection == [(1, host.reference_id)], "the host is selected, not the new unit"
        assert window.units_panel.garrison_tree.topLevelItemCount() == 1
        assert _last_status(window) == "Placed Archer for Player 1 inside Watch Tower (1/5)"

        window.undo()
        assert _inside(window, host) == []
        assert len(_units(window)) == before
        window.redo()
        assert len(_inside(window, host)) == 1
    finally:
        _close(window)


@pytest.mark.parametrize(
    ("case", "text"),
    [
        ("full", "Place Unit: Watch Tower is full (5 places). Tick Free placement to place it anyway."),
        ("wrong_type", "Place Unit: Knight cannot go inside Watch Tower. Tick Free placement to place it anyway."),
    ],
)
def test_a_refused_click_on_a_host_adds_nothing_and_says_why(case, text) -> None:
    window, host = _setup(_ARCHER if case == "full" else _KNIGHT)
    try:
        if case == "full":
            for _ in range(5):
                window._garrison_add_const(_host_entry(window, host), 83)
        before = len(_units(window))
        records = len(window.edit_history.records)

        _click(window, host)

        assert len(_units(window)) == before
        assert len(window.edit_history.records) == records
        assert _last_status(window) == text
    finally:
        _close(window)


def test_with_free_placement_the_object_goes_on_the_map_at_the_host() -> None:
    window, host = _setup()
    try:
        window.unrestricted_check.setChecked(True)
        _click(window, host)
        assert _inside(window, host) == []
        placed = window.map_view._unit_index.entry_for_key(window._selection[0]).unit
        assert placed.unit_const == _ARCHER and placed.garrisoned_in_id == -1
        assert (int(placed.x), int(placed.y)) == (int(host.x), int(host.y))
    finally:
        _close(window)


def test_a_house_at_a_host_follows_the_placement_rules() -> None:
    """A const nothing can hold never enters garrison mode."""
    window, host = _setup(_HOUSE)
    try:
        before = len(_units(window))
        _click(window, host)
        assert len(_units(window)) == before
        status = _last_status(window)
        assert status.startswith("Place Unit: ") and "inside" not in status
        assert status.endswith("Tick Free placement to place it anyway.")
    finally:
        _close(window)


def test_a_click_on_a_visible_occupant_goes_into_its_host() -> None:
    window, host = _setup()
    try:
        window._garrison_add_const(_host_entry(window, host), 83)
        window.show_garrisoned_action.setChecked(True)
        hit = window.map_view.pick_unit_cover_at(_tile_pos(window, (int(host.x), int(host.y))))
        assert hit is not None and hit[0].unit.garrisoned_in_id == host.reference_id, "the occupant must be on top"

        _click(window, host)

        assert sorted(u.unit_const for u in _inside(window, host)) == sorted([83, _ARCHER])
    finally:
        _close(window)


def test_a_player_2_archer_goes_into_a_player_1_tower_as_player_2() -> None:
    """The Owner combo's player, as any placement (user, 2026-11-02)."""
    window, host = _setup(owner=2)
    try:
        _click(window, host)
        (unit,) = _inside(window, host)
        assert unit in window.scenario.unit_manager.units[2]
        assert _last_status(window) == "Placed Archer for Player 2 inside Watch Tower (1/5)"
    finally:
        _close(window)


def test_a_click_on_open_grass_still_places_on_the_map() -> None:
    window, host = _setup()
    try:
        tile = _free_tile(window)
        _click(window, tile)
        placed = window.map_view._unit_index.entry_for_key(window._selection[0]).unit
        assert (int(placed.x), int(placed.y)) == tile and placed.garrisoned_in_id == -1
        assert _inside(window, host) == []
    finally:
        _close(window)
