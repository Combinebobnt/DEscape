"""Turning and flipping a just-pasted region as a block (GH #197), through a
real offscreen ViewerWindow: Edit > Rotate Selection and Flip Selection on
an armed paste, ViewerWindow.on_region_transform() behind them.

Built on tests/test_region_move.py's shape: the blank template, the Select
tool, Copy then Paste at a hovered tile. Every ViewerWindow() here must call
edit_history.mark_saved() before close().
"""

from __future__ import annotations

import pytest
from test_region_move import _drag_region
from test_region_select import _select_window

from descape import region_transform, render

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_A, _B, _C = 2, 15, 1  # BEACH, GRASS_1, WATER: present in every DE version
ARCHER = 4  # ANGLE, 16 facings
VILLAGER = 83
HOUSE = 70  # 2x2
WALL = 117  # five wall shapes
PASTE_AT = (40, 40)


def _close(window) -> None:
    window.edit_history.mark_saved()
    window.close()


def _set_terrain(window, tiles: dict[tuple[int, int], int]) -> None:
    mm = window.scenario.map_manager
    for (x, y), terrain_id in tiles.items():
        mm.terrain[y * mm.map_width + x].terrain_id = terrain_id


def _add(window, placements):
    """[(player, const, x, y, rotation, holder index or None)], one edit; the units in order."""
    model = window._ensure_unit_edits()
    added = []
    with window._unit_edit(model, "Add", sorted({p[0] for p in placements})):
        for player, const, x, y, rotation, holder in placements:
            inside = -1 if holder is None else added[holder].reference_id
            added.append(model.add(player, const, x, y, rotation=rotation, garrisoned_in_id=inside))
    return added


def _setup(window, size=(5, 3), units=True):
    """An asymmetric size[0] x size[1] block at the origin, with units, copied."""
    w, h = size
    _set_terrain(window, {(0, 0): _A, (1, 0): _A, (0, 1): _B, (w - 1, h - 1): _C})
    if units:
        _add(
            window,
            [
                (1, ARCHER, 1.5, 1.5, 1.25, None),
                (1, WALL, 3.5, 0.5, 3.0, None),
                (1, HOUSE, *render.span_anchor(3, 1, 2, 2), 0.0, None),
                (1, VILLAGER, *render.span_anchor(3, 1, 2, 2), 0.0, 2),
            ],
        )
    window.on_region_selected((0, 0, w, h))
    window.copy_region()


def _paste(window, at=PASTE_AT) -> None:
    window.on_hover(at)
    window.paste_region()
    assert window._paste_move is not None, "the paste did not arm"


def _tiles(window, x0=30, y0=30, x1=56, y1=56):
    mm = window.scenario.map_manager
    return [
        (mm.terrain[y * mm.map_width + x].terrain_id, mm.terrain[y * mm.map_width + x].elevation)
        for y in range(y0, y1)
        for x in range(x0, x1)
    ]


def _units(window, x0=30, y0=30, x1=56, y1=56):
    """Every unit field but the re-minted reference_id, plus whether it is garrisoned."""
    return sorted(
        (player, u.unit_const, float(u.x), float(u.y), float(u.z), round(float(u.rotation), 9),
         u.status, u.initial_animation_frame, u.garrisoned_in_id != -1)
        for player, units in enumerate(window.scenario.unit_manager.units)
        for u in units
        if x0 <= u.x < x1 and y0 <= u.y < y1
    )


def _live(window, const):
    return [u for units in window.scenario.unit_manager.units for u in units if u.unit_const == const]


def _last_status(window) -> str:
    return window.status_log.toPlainText().splitlines()[-1]


def test_paste_then_rotate_is_one_undo_record_and_one_undo_restores_the_map():
    window = _select_window()
    try:
        _setup(window)
        before_tiles, before_units = _tiles(window), _units(window)
        depth = len(window.edit_history.records)
        _paste(window)
        window.rotate_cw_action.trigger()
        assert _last_status(window) == "Rotated pasted region ↻"
        assert len(window.edit_history.records) == depth + 1
        assert _tiles(window) != before_tiles
        window.undo()
        assert _tiles(window) == before_tiles
        assert _units(window) == before_units
    finally:
        _close(window)


def test_the_region_turns_to_its_swapped_size_about_the_pivot():
    window = _select_window()
    try:
        _setup(window)
        _paste(window)
        assert window._region == (40, 40, 45, 43)
        window.rotate_cw_action.trigger()
        # pivot2 = (85, 83); a 3 x 5 block centres at ((85-3)//2, (83-5)//2).
        assert window._region == (41, 39, 44, 44)
        assert window._paste_move.pivot2 == (85, 83)
        assert window._paste_move.element == region_transform.ROTATE_CW
    finally:
        _close(window)


@pytest.mark.parametrize(
    "actions",
    [("rotate_cw_action",) * 4, ("rotate_ccw_coarse_action",) * 4, ("flip_horizontal_action",) * 2,
     ("flip_vertical_action",) * 2, ("rotate_cw_action", "flip_horizontal_action", "rotate_cw_action",
                                     "flip_horizontal_action")],
    ids=["four_cw", "four_ccw_coarse", "flip_h_twice", "flip_v_twice", "mixed_identity"],
)
def test_turns_that_compose_to_the_identity_reproduce_the_plain_paste(actions):
    window = _select_window()
    try:
        _setup(window)
        _paste(window)
        plain_tiles, plain_units = _tiles(window), _units(window)
        depth = len(window.edit_history.records)
        for name in actions:
            getattr(window, name).trigger()
        assert window._paste_move.element == "id"
        assert _tiles(window) == plain_tiles
        assert _units(window) == plain_units
        assert len(window.edit_history.records) == depth
        assert window._region == (40, 40, 45, 43)
    finally:
        _close(window)


def test_a_flip_after_a_turn_flips_the_turned_block_on_screen():
    """Each step applies after the ones before it: compose(step, so_far)."""
    from descape.mirror_tools import compose

    window = _select_window()
    try:
        _setup(window, units=False)
        _paste(window)
        source = window._paste_move.block
        window.rotate_cw_action.trigger()
        window.flip_horizontal_action.trigger()
        element = compose(region_transform.FLIP_HORIZONTAL, region_transform.ROTATE_CW)
        assert element != compose(region_transform.ROTATE_CW, region_transform.FLIP_HORIZONTAL)
        expected = region_transform.transform_block(source, element, False).block
        x0, y0 = region_transform.anchor_for((85, 83), expected)
        mm = window.scenario.map_manager
        got = tuple(
            mm.terrain[(y0 + y) * mm.map_width + x0 + x].terrain_id
            for y in range(expected.height)
            for x in range(expected.width)
        )
        assert got == expected.terrain_ids
    finally:
        _close(window)


def test_turn_move_turn_stays_one_record_and_the_move_shifts_the_pivot():
    window = _select_window()
    try:
        _setup(window)
        depth = len(window.edit_history.records)
        before = _tiles(window)
        _paste(window)
        window.rotate_cw_action.trigger()
        anchor = window._paste_move.anchor
        _drag_region(window, (anchor[0] + 1, anchor[1] + 1), (anchor[0] + 4, anchor[1] + 3))
        assert window._paste_move.pivot2 == (85 + 6, 83 + 4)
        window.rotate_ccw_action.trigger()
        assert window._paste_move.element == "id"
        # Back to the plain block, centred on the moved pivot: (40, 40) + (3, 2).
        assert window._paste_move.anchor == (43, 42)
        assert len(window.edit_history.records) == depth + 1
        window.undo()
        assert _tiles(window) == before
    finally:
        _close(window)


def test_a_turn_replays_the_pastes_own_category_flags():
    window = _select_window()
    try:
        _setup(window)
        window.paste_units_check.setChecked(False)
        _paste(window)
        assert _units(window) == []
        window.paste_units_check.setChecked(True)
        window.rotate_cw_action.trigger()
        assert _units(window) == [], "the turn pasted units the paste itself left out"
        assert not window._paste_move.do_units
    finally:
        window.paste_units_check.setChecked(True)
        _close(window)


def test_a_block_clipped_at_the_map_edge_turns_back_whole():
    window = _select_window()
    try:
        _setup(window, units=False)
        _paste(window, at=(116, 40))  # columns 116..120: the last one is off a 120 map
        clipped = _tiles(window, 100, 30, 120, 56)
        window.rotate_cw_action.trigger()
        assert window._region == (117, 39, 120, 44), "the turned 3 x 5 block fits wholly on the map"
        mm = window.scenario.map_manager
        on_map = {mm.terrain[y * mm.map_width + x].terrain_id for x in range(117, 120) for y in range(39, 44)}
        assert {_A, _B, _C} <= on_map, "the column the first paste clipped is back"
        window.rotate_ccw_action.trigger()
        assert _tiles(window, 100, 30, 120, 56) == clipped
    finally:
        _close(window)


def test_a_turn_that_would_land_off_the_map_writes_nothing_and_stays_armed():
    window = _select_window()
    try:
        _setup(window, size=(20, 2), units=False)
        _paste(window, at=(119, 60))
        state = window._paste_move
        snapshot = _tiles(window, 100, 40, 120, 80)
        depth, cursor = len(window.edit_history.records), window.edit_history.cursor
        window.rotate_cw_action.trigger()
        assert _last_status(window) == "Rotate: the turned block would land off the map, move it inward first"
        assert window._paste_move is state
        assert (len(window.edit_history.records), window.edit_history.cursor) == (depth, cursor)
        assert _tiles(window, 100, 40, 120, 80) == snapshot
        assert window.rotate_cw_action.isEnabled()
    finally:
        _close(window)


def test_a_units_only_paste_whose_units_all_turn_off_the_map_comes_back_and_stays_armed():
    window = _select_window()
    try:
        _set_terrain(window, {(0, 0): _A})
        _add(window, [(1, ARCHER, 19.5, 0.5, 0.0, None)])
        window.on_region_selected((0, 0, 20, 2))
        window.copy_region()
        window.paste_terrain_check.setChecked(False)
        window.paste_elevation_check.setChecked(False)
        _paste(window, at=(100, 118))
        assert [(u.x, u.y) for u in _live(window, ARCHER)] == [(19.5, 0.5), (119.5, 118.5)]
        record = window._paste_move.record
        depth = len(window.edit_history.records)
        window.rotate_cw_action.trigger()
        assert _last_status(window) == "Rotate: nothing would land there, left as it was"
        assert [(u.x, u.y) for u in _live(window, ARCHER)] == [(19.5, 0.5), (119.5, 118.5)]
        assert window._paste_move is not None and window._paste_move.record is record
        assert window.edit_history.peek_undo() is record
        assert len(window.edit_history.records) == depth
    finally:
        window.paste_terrain_check.setChecked(True)
        window.paste_elevation_check.setChecked(True)
        _close(window)


def test_a_pasted_holder_and_its_occupant_stay_linked_on_one_point():
    window = _select_window()
    try:
        _setup(window)
        _paste(window)
        for name in ("rotate_cw_action", "flip_vertical_action"):
            getattr(window, name).trigger()
            (house,) = [u for u in _live(window, HOUSE) if u.x >= 30]
            (villager,) = [u for u in _live(window, VILLAGER) if u.x >= 30]
            assert villager.garrisoned_in_id == house.reference_id, name
            assert (villager.x, villager.y) == (house.x, house.y), name
    finally:
        _close(window)


def test_a_turned_wall_and_archer_carry_the_mirror_rules():
    window = _select_window()
    try:
        _setup(window)
        _paste(window)
        window.rotate_cw_action.trigger()
        (wall,) = [u for u in _live(window, WALL) if u.x >= 30]
        (archer,) = [u for u in _live(window, ARCHER) if u.x >= 30]
        assert float(wall.rotation) == 4.0  # a NW-SE diagonal run turns into a NE-SW one
        assert float(archer.rotation) == pytest.approx(1.25 + 3.141592653589793 / 2, abs=1e-6)
    finally:
        _close(window)


@pytest.mark.parametrize(("method", "verb"), [("rotate", "Rotate"), ("flip", "Flip")])
def test_a_lapsed_paste_says_so_by_action_and_writes_nothing(method, verb):
    window = _select_window()
    try:
        _setup(window)
        _paste(window)
        window.on_region_selected(None)  # a new selection lapses the paste
        assert window._paste_move is None
        snapshot = _tiles(window)
        depth = len(window.edit_history.records)
        step = region_transform.ROTATE_CW if method == "rotate" else region_transform.FLIP_HORIZONTAL
        window.on_region_transform(step)
        assert _last_status(window) == f"{verb}: paste a region first, then turn it"
        assert _tiles(window) == snapshot
        assert len(window.edit_history.records) == depth
    finally:
        _close(window)


def test_terrain_mode_menu_enablement_follows_the_paste_then_the_committed_region():
    """TASK-256: a committed region is a target of its own, so the menu stays
    open around the paste; what changes is which target the actions take."""
    window = _select_window()
    try:
        turn = [*window._rotate_actions, *window._flip_actions, *window._rotate_group_actions]
        _setup(window)
        assert window._transform_target() == "region" and window._region_rotate_target()
        assert all(action.isEnabled() for action in turn)
        _paste(window)
        assert window._transform_target() == "paste" and not window._region_rotate_target()
        assert all(action.isEnabled() for action in turn)
        assert not any(action.isVisible() for action in window._rotate_toolbar_actions)
        window.on_hover((80, 80))
        window.paste_region()  # a second paste re-arms on the new one
        assert window._transform_target() == "paste"
        window.undo()  # the undo lapses it; the pasted rectangle stays selected
        assert window._paste_move is None
        assert window._transform_target() == "region" and window._region_rotate_target()
        assert all(action.isEnabled() for action in turn)
        window.deselect()
        assert not any(action.isEnabled() for action in turn)
    finally:
        _close(window)


def test_triggers_mode_turns_nothing():
    window = _select_window()
    try:
        _setup(window)
        _paste(window)
        window.mode_combo.setCurrentText("Triggers")
        assert not any(action.isEnabled() for action in [*window._rotate_actions, *window._flip_actions])
    finally:
        _close(window)


def test_with_a_units_selection_rotate_turns_the_units_and_flip_flips_the_region():
    window = _select_window()
    try:
        (loner,) = _add(window, [(1, ARCHER, 80.5, 80.5, 0.0, None)])
        _setup(window)
        _paste(window)
        plain = _tiles(window)
        window.mode_combo.setCurrentText("Units")
        window._selection = [(1, loner.reference_id)]
        window._refresh_selection_view()
        assert window.flip_horizontal_action.isEnabled()
        assert window.rotate_cw_action.isEnabled()

        def lone_rotation() -> float:
            (unit,) = [u for u in _live(window, ARCHER) if (u.x, u.y) == (80.5, 80.5)]
            return float(unit.rotation)

        window.flip_horizontal_action.trigger()
        flipped = _tiles(window)
        assert flipped != plain
        assert lone_rotation() == 0.0
        assert window._paste_move.element == region_transform.FLIP_HORIZONTAL
        assert window._selection == [(1, loner.reference_id)]

        window.rotate_cw_action.trigger()
        assert lone_rotation() != 0.0, "Rotate left the selected unit alone"
        assert _tiles(window) == flipped, "Rotate turned the region instead of the selection"
    finally:
        _close(window)
