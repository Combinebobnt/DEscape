"""Units mode: off-grid placement, footprint snapping and group drags."""

from __future__ import annotations

import numpy as np
import pytest
from PyQt5.QtCore import QPointF, Qt
from unit_edit_viewer_helpers import (
    _BARRACKS_CONST,
    _HOUSE_CONST,
    _PLACE_CONST,
    _add_units,
    _chunk_for_tile,
    _close,
    _empty_tile,
    _free_tiles,
    _make_stack,
    _no_install_env,  # noqa: F401 -- autouse fixture
    _pos_in_tile,
    _positions,
    _select_keys,
    _window,
)

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


# --- D2's off-grid toggle (off-grid placement, Stage 3) ---------------------


def _place_at(window, pos, modifiers=Qt.NoModifier, const: int = _PLACE_CONST):
    window.units_panel.select_object(const)
    assert window.units_panel.selected_object_const() == const
    window.units_panel.select_owner(1)
    window.place_unit_action.setChecked(True)
    window.on_unit_place(pos, modifiers)
    return window.map_view._unit_index.entry_for_key(window._selection[0])


def test_the_off_grid_checkbox_shows_only_for_place_unit_and_move() -> None:
    """Asserted on the captured QAction handle, never on the widget's own
    isVisible(): offscreen, a never-shown widget reads False unconditionally.
    Move shows it because a unit drag reads it (TASK-012)."""
    window = _window()
    try:
        for tool in ("place_unit", "move"):
            getattr(window, f"{tool}_action").setChecked(True)
            assert window.off_grid_param_action.isVisible(), tool
            assert window.off_grid_check.isEnabled(), tool
            assert window.tool_param_separator_action.isVisible(), tool

        for tool in ("convert", "select", "pan"):
            getattr(window, f"{tool}_action").setChecked(True)
            assert not window.off_grid_param_action.isVisible(), tool
            assert not window.off_grid_check.isEnabled(), tool
    finally:
        _close(window)


def test_the_toggles_start_off_and_a_tick_lasts_the_session_only(monkeypatch) -> None:
    """TASK-272: a fresh window starts snapped and restricted (the schema
    default), and ticking either box writes no default for the next window."""
    from descape import settings

    window = _window()
    try:
        assert not window.off_grid_check.isChecked()
        assert not window.unrestricted_check.isChecked()
        window.off_grid_check.setChecked(True)
        window.unrestricted_check.setChecked(True)
    finally:
        _close(window)
    monkeypatch.setattr(settings, "_tool_option_overrides", None)
    assert settings.get_tool_option_defaults()["off_grid"] is False
    assert settings.get_tool_option_defaults()["free_placement"] is False
    fresh = _window()
    try:
        assert not fresh.off_grid_check.isChecked()
        assert not fresh.unrestricted_check.isChecked()
    finally:
        _close(fresh)


def test_placement_snaps_by_default_and_floats_with_the_toggle() -> None:
    window = _window()
    try:
        window.unrestricted_check.setChecked(True)  # the second unit overlaps the first
        tx, ty = _empty_tile(window)
        entry = _place_at(window, _pos_in_tile(window, tx, ty, 0.15, 0.85))
        assert (entry.unit.x, entry.unit.y) == (tx + 0.5, ty + 0.5)

        window.off_grid_check.setChecked(True)
        entry = _place_at(window, _pos_in_tile(window, tx, ty, 0.15, 0.85))
        assert (entry.unit.x, entry.unit.y) != (tx + 0.5, ty + 0.5)
        # Under the cursor, to within the pixel the click itself quantizes to.
        tp = window.map_view._tile_pixels
        assert abs(entry.unit.x - (tx + 0.15)) <= 1 / tp
        assert abs(entry.unit.y - (ty + 0.85)) <= 1 / tp
    finally:
        _close(window)


def test_held_alt_places_off_grid_for_one_click_without_changing_the_toggle() -> None:
    window = _window()
    try:
        window.unrestricted_check.setChecked(True)  # the second unit overlaps the first
        tx, ty = _empty_tile(window)
        entry = _place_at(window, _pos_in_tile(window, tx, ty, 0.15, 0.85), Qt.AltModifier)
        assert (entry.unit.x, entry.unit.y) != (tx + 0.5, ty + 0.5)
        assert not window.off_grid_check.isChecked(), "the one-off modifier flipped the toggle"

        entry = _place_at(window, _pos_in_tile(window, tx, ty, 0.15, 0.85))
        assert (entry.unit.x, entry.unit.y) == (tx + 0.5, ty + 0.5)
    finally:
        _close(window)


def test_moving_a_unit_takes_the_same_branch_as_placing_one() -> None:
    """The choice lives in _placement_point, so Move and Place cannot
    disagree -- this is what pins that they really share it."""
    window = _window()
    try:
        tx, ty = _empty_tile(window)
        entry = _place_at(window, _pos_in_tile(window, tx, ty, 0.5, 0.5))
        key = window._selection[0]

        window.off_grid_check.setChecked(True)
        window.on_unit_move(key, _pos_in_tile(window, tx + 1, ty + 1, 0.2, 0.7), Qt.NoModifier)
        moved = window.map_view._unit_index.entry_for_key(key)
        assert moved is not None
        assert (moved.unit.x, moved.unit.y) != (tx + 1.5, ty + 1.5)

        window.off_grid_check.setChecked(False)
        window.on_unit_move(key, _pos_in_tile(window, tx + 2, ty + 2, 0.2, 0.7), Qt.NoModifier)
        moved = window.map_view._unit_index.entry_for_key(key)
        assert (moved.unit.x, moved.unit.y) == (tx + 2.5, ty + 2.5)
        assert entry is not None
    finally:
        _close(window)


def test_an_off_grid_unit_paints_where_it_stands() -> None:
    """The end-to-end check the whole feature exists for: model coordinates
    reaching the canvas. A model-only assertion is exactly what b1's own
    postmortem records as having missed the last bug here."""
    window = _window()
    try:
        tx, ty = _empty_tile(window)
        window.off_grid_check.setChecked(True)
        _place_at(window, _pos_in_tile(window, tx, ty, 0.1, 0.1))
        cx, cy = _chunk_for_tile(window, tx, ty)
        low = np.array(window._cache.get_chunk(0, cx, cy), copy=True)

        window.undo()
        window.off_grid_check.setChecked(False)
        _place_at(window, _pos_in_tile(window, tx, ty, 0.1, 0.1))
        snapped = np.array(window._cache.get_chunk(0, cx, cy), copy=True)

        assert not np.array_equal(low, snapped), (
            "an off-grid unit painted the same pixels as a snapped one"
        )
    finally:
        _close(window)


def test_a_refused_inverse_snaps_and_says_so(monkeypatch) -> None:
    """The screen -> map-point inverse can legitimately have no answer for a
    pixel: a degenerate denominator, or a round-trip that misses by more than
    a pixel, both reachable on a steep Sloped ramp. Snapping there is the
    right behaviour, but a SILENT snap is not -- a tester cannot tell it from
    a checkbox that never applied or an Alt the window manager ate.

    Driven by forcing _pick_map_point to refuse rather than by constructing a
    degenerate ramp: this module forces Flat on purpose, and the geometry's
    own degenerate case is pinned in tests/test_screen_to_map_point.py. What
    is untested without this is the viewer's REACTION to a refusal."""
    from descape.viewer_common import OFF_GRID_FALLBACK_MESSAGE

    window = _window()
    try:
        tx, ty = _empty_tile(window)
        window.off_grid_check.setChecked(True)
        monkeypatch.setattr(window.map_view, "_pick_map_point", lambda pos: None)

        entry = _place_at(window, _pos_in_tile(window, tx, ty, 0.15, 0.85))

        assert (entry.unit.x, entry.unit.y) == (tx + 0.5, ty + 0.5)
        assert OFF_GRID_FALLBACK_MESSAGE in window.status_log.toPlainText()
    finally:
        _close(window)


def test_a_successful_off_grid_placement_stays_quiet() -> None:
    """The other half: the fallback line must not fire on every off-grid click,
    or it stops meaning anything."""
    from descape.viewer_common import OFF_GRID_FALLBACK_MESSAGE

    window = _window()
    try:
        tx, ty = _empty_tile(window)
        window.off_grid_check.setChecked(True)
        _place_at(window, _pos_in_tile(window, tx, ty, 0.15, 0.85))
        assert OFF_GRID_FALLBACK_MESSAGE not in window.status_log.toPlainText()
    finally:
        _close(window)


# --- GH #206: snapped buildings land on the game's grid ----------------------


_MARKET_CONST = 84  # 4x4


def test_a_snapped_house_centres_on_the_whole_number_nearest_the_cursor() -> None:
    window = _window()
    try:
        window.unrestricted_check.setChecked(True)  # the second House overlaps the first
        tx, ty = _empty_tile(window)
        entry = _place_at(window, _pos_in_tile(window, tx, ty, 0.2, 0.3), const=_HOUSE_CONST)
        assert (entry.unit.x, entry.unit.y) == (tx, ty)
        entry = _place_at(window, _pos_in_tile(window, tx, ty, 0.8, 0.6), const=_HOUSE_CONST)
        assert (entry.unit.x, entry.unit.y) == (tx + 1, ty + 1)
    finally:
        _close(window)


def test_a_snapped_market_centres_its_4x4_footprint_on_the_cursor() -> None:
    from descape.render_units import unit_tile_bounds

    window = _window()
    try:
        window.unrestricted_check.setChecked(True)  # the second Market overlaps the first
        tx, ty = _empty_tile(window)
        mm = window.scenario.map_manager
        entry = _place_at(window, _pos_in_tile(window, tx, ty, 0.2, 0.3), const=_MARKET_CONST)
        assert (entry.unit.x, entry.unit.y) == (tx, ty)
        assert unit_tile_bounds(entry.unit, mm.map_width, mm.map_height) == (tx - 2, tx + 2, ty - 2, ty + 2)
        entry = _place_at(window, _pos_in_tile(window, tx, ty, 0.8, 0.6), const=_MARKET_CONST)
        assert (entry.unit.x, entry.unit.y) == (tx + 1, ty + 1)
    finally:
        _close(window)


def test_a_snapped_barracks_sits_on_its_tiles_centre() -> None:
    window = _window()
    try:
        tx, ty = _empty_tile(window)
        entry = _place_at(window, _pos_in_tile(window, tx, ty, 0.8, 0.2), const=_BARRACKS_CONST)
        assert (entry.unit.x, entry.unit.y) == (tx + 0.5, ty + 0.5)
    finally:
        _close(window)


def test_a_snapped_house_with_no_map_point_takes_the_far_corner_quietly(monkeypatch) -> None:
    from descape.viewer_common import OFF_GRID_FALLBACK_MESSAGE

    window = _window()
    try:
        tx, ty = _empty_tile(window)
        monkeypatch.setattr(window.map_view, "_pick_map_point", lambda pos: None)
        entry = _place_at(window, _pos_in_tile(window, tx, ty, 0.2, 0.3), const=_HOUSE_CONST)
        assert (entry.unit.x, entry.unit.y) == (tx + 1, ty + 1)
        assert OFF_GRID_FALLBACK_MESSAGE not in window.status_log.toPlainText()
    finally:
        _close(window)


def test_a_snapped_move_puts_an_off_parity_house_on_the_grid() -> None:
    window = _window()
    try:
        _free_tiles(window, [(63, 63), (64, 64)])
        (key,) = _add_units(window, [(_HOUSE_CONST, 63.5, 63.5)])
        window.on_unit_move(key, _pos_in_tile(window, 66, 66, 0.4, 0.7), Qt.NoModifier)
        moved = window.map_view._unit_index.entry_for_key(key)
        assert (moved.unit.x, moved.unit.y) == (66.0, 67.0)
    finally:
        _close(window)


def test_a_snapped_group_move_keeps_an_off_parity_members_half() -> None:
    window = _window()
    try:
        _free_tiles(window, [(63, 63), (64, 64), (66, 62)])
        keys = _add_units(window, [(_HOUSE_CONST, 63.5, 63.5), (_PLACE_CONST, 66.5, 62.5)])
        _select_keys(window, keys)
        before = _positions(window, keys)

        _gesture(window, (63, 63), (66, 65), tool="move")

        assert _positions(window, keys) == [(x + 3, y + 2) for x, y in before]
        assert _positions(window, keys)[0] == (66.5, 65.5)
    finally:
        _close(window)


# --- GH #75: group and stack move --------------------------------------------


def _gesture(window, from_tile, to_tile=None, modifiers=Qt.NoModifier, to_scene=None, *, tool: str) -> None:
    """A real press(-move)-release through MapView under `tool` ("select" or
    "move"); no `to_tile` is a plain click."""
    from PyQt5.QtCore import QEvent
    from PyQt5.QtGui import QMouseEvent

    getattr(window, f"{tool}_action").setChecked(True)
    view = window.map_view
    press = conftest.viewport_pos(view, *from_tile)
    if to_scene is not None:
        end = QPointF(view.mapFromScene(to_scene))
    else:
        end = press if to_tile is None else conftest.viewport_pos(view, *to_tile)
    view.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, press, Qt.LeftButton, Qt.LeftButton, modifiers))
    if end != press:
        view.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, end, Qt.NoButton, Qt.LeftButton, modifiers))
    view.mouseReleaseEvent(QMouseEvent(QEvent.MouseButtonRelease, end, Qt.LeftButton, Qt.NoButton, modifiers))


def test_a_select_click_on_a_group_member_collapses_at_press() -> None:
    """TASK-252: Select only selects, so nothing defers the collapse to the
    release any more (GH #75's old pending collapse is gone)."""
    window = _window()
    try:
        _free_tiles(window, [(62, 62), (64, 62)])
        keys = _add_units(window, [(_PLACE_CONST, 62.5, 62.5), (_PLACE_CONST, 64.5, 62.5)])
        _select_keys(window, keys)
        from PyQt5.QtCore import QEvent
        from PyQt5.QtGui import QMouseEvent

        window.select_action.setChecked(True)
        view = window.map_view
        vp = conftest.viewport_pos(view, 64, 62)
        view.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, vp, Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
        assert window._selection == [keys[1]]
        view.mouseReleaseEvent(QMouseEvent(QEvent.MouseButtonRelease, vp, Qt.LeftButton, Qt.NoButton, Qt.NoModifier))
        assert window._selection == [keys[1]]
    finally:
        _close(window)


def test_a_move_click_on_a_group_member_leaves_the_group() -> None:
    window = _window()
    try:
        _free_tiles(window, [(62, 62), (64, 62)])
        keys = _add_units(window, [(_PLACE_CONST, 62.5, 62.5), (_PLACE_CONST, 64.5, 62.5)])
        _select_keys(window, keys)
        depth = window.edit_history.cursor
        _gesture(window, (64, 62), tool="move")
        assert window._selection == keys
        assert window.edit_history.cursor == depth
    finally:
        _close(window)


def test_a_group_drag_moves_every_member_in_one_undo_record() -> None:
    window = _window()
    try:
        _free_tiles(window, [(62, 62), (64, 63), (66, 62)])
        keys = _add_units(
            window, [(_PLACE_CONST, 62.5, 62.5), (_PLACE_CONST, 64.3, 63.8), (_PLACE_CONST, 66.5, 62.5)]
        )
        _select_keys(window, keys)
        before = _positions(window, keys)
        depth = window.edit_history.cursor

        _gesture(window, (62, 62), (65, 64), tool="move")

        after = _positions(window, keys)
        # Whole-tile (3, 2), and the off-centre member keeps its sub-tile offset.
        assert after == [(x + 3, y + 2) for x, y in before]
        assert window.edit_history.cursor == depth + 1
        assert window._selection == keys, "a group stays selected after the drag"
        window.undo()
        assert _positions(window, keys) == before
        window.redo()
        assert _positions(window, keys) == after
    finally:
        _close(window)


def test_the_edge_clamp_keeps_the_shape_even_on_an_off_map_release() -> None:
    window = _window()
    try:
        w = window.scenario.map_manager.map_width
        _free_tiles(window, [(w - 6, 60), (w - 4, 61)])
        keys = _add_units(window, [(_PLACE_CONST, w - 5.5, 60.5), (_PLACE_CONST, w - 3.5, 61.5)])
        _select_keys(window, keys)
        before = _positions(window, keys)

        tp = window.map_view._tile_pixels
        # Released well past the right edge: clamped, not cancelled.
        _gesture(window, (w - 6, 60), to_scene=QPointF((w + 10) * tp, 60.5 * tp), tool="move")

        after = _positions(window, keys)
        assert after == [(x + 3, y) for x, y in before], "the outer member stops on the last column"
        assert "stopped at the map edge" in window.status_log.toPlainText()
    finally:
        _close(window)


def test_a_fully_clamped_group_drag_writes_nothing() -> None:
    window = _window()
    try:
        w = window.scenario.map_manager.map_width
        _free_tiles(window, [(w - 3, 60), (w - 1, 60)])
        keys = _add_units(window, [(_PLACE_CONST, w - 2.5, 60.5), (_PLACE_CONST, w - 0.5, 60.5)])
        _select_keys(window, keys)
        before = _positions(window, keys)
        depth = window.edit_history.cursor

        _gesture(window, (w - 3, 60), (w - 1, 60), tool="move")

        assert _positions(window, keys) == before
        assert window.edit_history.cursor == depth
    finally:
        _close(window)


def test_grabbing_a_houses_lower_tile_moves_the_group_exactly_one_tile() -> None:
    """A 2x2 house sits at tile + 1.0, so int(house.x) is its UPPER tile. The
    delta runs from the grabbed cover tile, not from that."""
    window = _window()
    try:
        _free_tiles(window, [(62, 62), (63, 62), (62, 63), (63, 63), (66, 62)])
        keys = _add_units(window, [(_HOUSE_CONST, 63.0, 63.0), (_PLACE_CONST, 66.5, 62.5)])
        _select_keys(window, keys)
        before = _positions(window, keys)
        # Zoomed in, so a one-tile drag clears UNIT_DRAG_THRESHOLD_PX.
        window.map_view.scale(4.0, 4.0)
        window.map_view.centerOn(QPointF(63 * window.map_view._tile_pixels, 63 * window.map_view._tile_pixels))

        _gesture(window, (62, 62), (63, 62), tool="move")

        assert _positions(window, keys) == [(x + 1, y) for x, y in before]
    finally:
        _close(window)


def test_an_off_grid_group_drag_keeps_every_offset(monkeypatch) -> None:
    window = _window()
    try:
        _free_tiles(window, [(62, 62), (64, 63)])
        keys = _add_units(window, [(_PLACE_CONST, 62.5, 62.5), (_PLACE_CONST, 64.3, 63.8)])
        _select_keys(window, keys)
        before = _positions(window, keys)
        monkeypatch.setattr(window.map_view, "_pick_map_point", lambda pos: (70.25, 66.75))

        _gesture(window, (62, 62), (70, 66), modifiers=Qt.AltModifier, tool="move")

        dx, dy = 70.25 - before[0][0], 66.75 - before[0][1]
        after = _positions(window, keys)
        assert after[0] == (70.25, 66.75)
        assert after[1] == pytest.approx((before[1][0] + dx, before[1][1] + dy), abs=1e-4)
    finally:
        _close(window)


def test_a_select_click_on_a_stacked_group_member_carries_the_cycle_on_after_the_collapse() -> None:
    window = _window()
    try:
        tile, stack = _make_stack(window)
        other = (tile[0] + 3, tile[1])
        _free_tiles(window, [other])
        (loner,) = _add_units(window, [(_PLACE_CONST, other[0] + 0.5, other[1] + 0.5)])
        _select_keys(window, [stack[0], loner])

        _gesture(window, tile, tool="select")
        assert window._selection == [stack[0]]
        assert "1 of 3 stacked here" in window.status_log.toPlainText()

        _gesture(window, tile, tool="select")
        assert window._selection == [stack[1]]
    finally:
        _close(window)


def test_a_plain_drag_on_a_stack_still_pulls_one_unit_off() -> None:
    window = _window()
    try:
        tile, stack = _make_stack(window)
        before = _positions(window, stack)
        _gesture(window, tile, (tile[0] + 2, tile[1]), tool="move")
        after = _positions(window, stack)
        assert sum(a != b for a, b in zip(after, before, strict=True)) == 1
    finally:
        _close(window)


def test_select_whole_stack_widens_the_selection_and_a_drag_moves_all_of_it() -> None:
    window = _window()
    try:
        tile, stack = _make_stack(window)
        other = (tile[0] + 3, tile[1])
        _free_tiles(window, [other])
        (loner,) = _add_units(window, [(_PLACE_CONST, other[0] + 0.5, other[1] + 0.5)])
        _select_keys(window, [loner, stack[1]])

        window.select_stack_action.trigger()
        assert window._selection == [loner, stack[1], stack[0], stack[2]]
        assert window._stack_cycle is None
        assert "4 units selected (stack)" in window.status_log.toPlainText()

        before = _positions(window, stack)
        _gesture(window, tile, (tile[0], tile[1] + 2), tool="move")
        assert _positions(window, stack) == [(x, y + 2) for x, y in before]
    finally:
        _close(window)


def test_select_whole_stack_says_so_when_nothing_selected_is_stacked() -> None:
    window = _window()
    try:
        tile, _stack = _make_stack(window)
        other = (tile[0] + 3, tile[1])
        _free_tiles(window, [other])
        (loner,) = _add_units(window, [(_PLACE_CONST, other[0] + 0.5, other[1] + 0.5)])
        _select_keys(window, [loner])
        window.on_select_whole_stack()
        assert window._selection == [loner]
        assert "no selected unit is stacked" in window.status_log.toPlainText()
        _select_keys(window, [])
        window.on_select_whole_stack()
        assert "nothing is selected" in window.status_log.toPlainText()
    finally:
        _close(window)


def test_clamp_group_delta_uses_each_members_own_span() -> None:
    from descape.viewer_common import clamp_group_delta
    from testkit import fakes

    house = fakes.SyntheticUnit(x=3.0, y=10.0, unit_const=_HOUSE_CONST)  # covers x 2..3
    villager = fakes.SyntheticUnit(x=15.5, y=10.5, unit_const=_PLACE_CONST)
    assert clamp_group_delta([house, villager], -5, 0, 20, 20, whole_tiles=True) == (-2, 0, True)
    assert clamp_group_delta([house, villager], 9, 0, 20, 20, whole_tiles=True) == (4, 0, True)
    assert clamp_group_delta([house, villager], 1, 1, 20, 20, whole_tiles=True) == (1, 1, False)
    # Free: the house's anchor may reach span/2 from the edge, the villager just short of it.
    dx, _dy, clamped = clamp_group_delta([house, villager], -5.0, 0.0, 20, 20, whole_tiles=False)
    assert (dx, clamped) == (-2.0, True)
    dx, _dy, _ = clamp_group_delta([villager], 9.0, 0.0, 20, 20, whole_tiles=False)
    assert int(villager.x + dx) == 19


def test_an_off_map_member_does_not_limit_the_clamp() -> None:
    from descape.viewer_common import clamp_group_delta
    from testkit import fakes

    stray = fakes.SyntheticUnit(x=-2.0, y=5.5, unit_const=_PLACE_CONST)
    villager = fakes.SyntheticUnit(x=5.5, y=5.5, unit_const=_PLACE_CONST)
    assert clamp_group_delta([stray, villager], -4, 0, 20, 20, whole_tiles=True) == (-4, 0, False)
