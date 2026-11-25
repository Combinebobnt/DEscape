"""The Move drag ghost shows the shape a wall lands as, and the walls the
drop reshapes (TASK-523), through a real offscreen ViewerWindow held mid-drag.

The window is Flat, so `_ghost_sprites_on` is patched True and the render's
unit_sprite_draws_at() is stubbed to record what each ghost asks for:
(x, y, rotation_override), returning no draws. The planner itself is tested
headless in test_wall_run.py; the windows are test_wall_run_viewer's GH #266
and GH #194 ones."""

from __future__ import annotations

import pytest
from test_gate_corner_pieces_viewer import _pieces_window
from test_wall_run_viewer import (  # noqa: F401 -- the two underscored fixtures are autouse
    _HELD,
    _ROW,
    PLAYER,
    VILLAGER,
    WALL,
    _add_unit,
    _close,
    _gate_window,
    _hold_the_synthetic_button,
    _last_status,
    _mouse_event,
    _moved_wall_window,
    _point_pos,
    _select_units,
    _wall_consts_intact,
    _walls_on,
)

from descape import unit_pick, wall_run
from testkit.render_family import set_render_global
from testkit.viewer_family import set_viewer_global

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_SRC = (30.5, _ROW + 0.5)
_DOWN3 = (30.5, _ROW + 3.5)


def _recording(window, monkeypatch) -> list:
    """Patches sprites on and returns the list each ghost resolve appends (x, y, override) to."""
    asked: list = []
    monkeypatch.setattr(window, "_ghost_sprites_on", lambda: True)
    set_render_global(
        monkeypatch, "unit_sprite_draws_at",
        lambda *args, rotation_override=None, **kwargs: (asked.append((args[5].x, args[5].y, rotation_override)), [])[1],
    )
    return asked


def _press(window, point) -> None:
    from PyQt5.QtCore import QEvent, Qt

    _HELD["value"] = True
    window.map_view.mousePressEvent(_mouse_event(QEvent.MouseButtonPress, _point_pos(window, *point), Qt.LeftButton, Qt.LeftButton))


def _move(window, point) -> None:
    from PyQt5.QtCore import QEvent, Qt

    window.map_view.mouseMoveEvent(_mouse_event(QEvent.MouseMove, _point_pos(window, *point), Qt.NoButton, Qt.LeftButton))


def _release(window, point) -> None:
    from PyQt5.QtCore import QEvent, Qt

    try:
        window.map_view.mouseReleaseEvent(
            _mouse_event(QEvent.MouseButtonRelease, _point_pos(window, *point), Qt.LeftButton, Qt.NoButton)
        )
    finally:
        _HELD["value"] = False


def _hold(window, asked, a, b) -> list:
    """Press at `a` and move to `b`, button held: what the last move's ghosts asked for."""
    _press(window, a)
    _move(window, b)
    asked.clear()
    _move(window, b)
    return sorted(asked)


def _ids(entries) -> list[tuple[int, int]]:
    return sorted((id(u), v) for _p, u, v in entries)


def _assert_same_plan(plan, ref) -> None:
    assert plan is not None and ref is not None
    assert (_ids(plan.rewrites), _ids(plan.moved), _ids(plan.landed)) == (_ids(ref.rewrites), _ids(ref.moved), _ids(ref.landed))


def _preview_plan(window):
    """The plan the preview last drew from (_drag_wall_plan()'s memo)."""
    memo = window._drag_plan_memo
    assert memo is not None, "the preview planned nothing"
    return memo[3]


def _entry(window, unit, player_id: int = PLAYER):
    return window.map_view._unit_index.entry_for_key(unit_pick.unit_key(player_id, unit))


_LANDED_RUN = [(29.5, _ROW + 0.5, 2.0), (30.5, _ROW + 3.5, 2.0), (31.5, _ROW + 0.5, 2.0)]


@pytest.mark.parametrize("box", [False, True])
def test_the_screenshot_case_ghosts_the_tower_it_lands_as_and_its_old_neighbours(monkeypatch, box):
    """GH #266's case, mid-drag: nothing is written, the preview's plan is the
    drop's, and the release stores what was ghosted."""
    window, wall, west, east = _moved_wall_window()
    try:
        window.gate_walls_check.setChecked(box)
        asked = _recording(window, monkeypatch)
        records = len(window.edit_history.records)

        assert _hold(window, asked, _SRC, _DOWN3) == _LANDED_RUN
        assert (wall.rotation, west.rotation, east.rotation) == (0.0, 0.0, 0.0)
        assert len(window.edit_history.records) == records
        model = window._ensure_unit_edits()
        _assert_same_plan(_preview_plan(window), window._move_plan(model, [(_entry(window, wall), *_DOWN3)], gate_walls=True))

        _release(window, _DOWN3)
        assert (wall.x, wall.y) == _DOWN3, _last_status(window)
        assert (wall.rotation, west.rotation, east.rotation) == (2.0, 2.0, 2.0)
    finally:
        _close(window)


def test_in_a_radian_file_the_ghosts_ask_for_the_derived_shapes(monkeypatch):
    window, _wall, west, _east = _moved_wall_window(integer=False)
    try:
        west.rotation = 1.256637
        window._after_unit_mutation()
        asked = _recording(window, monkeypatch)
        assert _hold(window, asked, _SRC, _DOWN3) == _LANDED_RUN
    finally:
        _release(window, _DOWN3)
        _close(window)


def test_a_lone_tower_moved_into_the_open_asks_for_the_tower(monkeypatch):
    window, _wall, _west, _east = _moved_wall_window()
    try:
        lone = _add_unit(window, WALL, 60.5, 60.5)
        lone.rotation = 2.0
        window._after_unit_mutation()
        asked = _recording(window, monkeypatch)
        assert _hold(window, asked, (60.5, 60.5), (65.5, 60.5)) == [(65.5, 60.5, 2.0)]
        plan = _preview_plan(window)
        assert (_ids(plan.landed), plan.moved, plan.rewrites) == ([(id(lone), 2)], [], [])
    finally:
        _release(window, (65.5, 60.5))
        _close(window)


def test_a_group_drag_ghosts_both_landings_and_both_new_run_ends(monkeypatch):
    window, wall, _west, east = _moved_wall_window()
    try:
        _select_units(window, [wall, east])
        asked = _recording(window, monkeypatch)
        got = _hold(window, asked, _SRC, _DOWN3)
        assert got == [(29.5, _ROW + 0.5, 2.0), (30.5, _ROW + 3.5, 2.0), (31.5, _ROW + 3.5, 2.0), (32.5, _ROW + 0.5, 2.0)]
        _release(window, _DOWN3)

        set_viewer_global(monkeypatch, "GROUP_GHOST_CAP", 1)
        window.undo()
        assert _hold(window, asked, _SRC, _DOWN3) == [(30.5, _ROW + 3.5, 2.0)], "past the cap: the anchor's shape, no neighbour"
    finally:
        _release(window, _DOWN3)
        _close(window)


def test_a_refused_drop_keeps_the_source_shape_until_free_placement_lifts_it(monkeypatch):
    window, _wall, _west, _east = _moved_wall_window()
    try:
        _add_unit(window, VILLAGER, *_DOWN3)
        asked = _recording(window, monkeypatch)
        assert _hold(window, asked, _SRC, _DOWN3) == [(30.5, _ROW + 3.5, None)]
        window.unrestricted_check.setChecked(True)
        asked.clear()
        _move(window, _DOWN3)
        assert sorted(asked) == _LANDED_RUN
    finally:
        _release(window, _DOWN3)
        _close(window)


class _Hiding:
    """The window's live filter with `hidden` units filtered out."""

    def __init__(self, real, hidden) -> None:
        self._real, self._hidden = real, {id(u) for u in hidden}

    def matches(self, player_id: int, unit) -> bool:
        return id(unit) not in self._hidden and self._real.matches(player_id, unit)

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_a_neighbour_the_live_filter_hides_gets_no_ghost(monkeypatch):
    window, _wall, west, east = _moved_wall_window()
    try:
        asked = _recording(window, monkeypatch)
        monkeypatch.setattr(window, "_unit_filter", _Hiding(window._unit_filter, [west, east]))
        assert _hold(window, asked, _SRC, _DOWN3) == [(30.5, _ROW + 3.5, 2.0)]
    finally:
        _release(window, _DOWN3)
        _close(window)


def _scene_walks(monkeypatch, raising: dict | None = None) -> list:
    """Counts wall_run.wall_scene calls; while `raising["on"]` they raise instead."""
    calls: list = []
    real = wall_run.wall_scene

    def counted(*args, **kwargs):
        if raising is not None and raising["on"]:
            raise AssertionError("the preview walked a wall scene")
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(wall_run, "wall_scene", counted)
    return calls


def test_a_box_on_drag_carrying_a_gate_is_previewed_as_before(monkeypatch):
    """The GH #194 carve-out, as a group: a single gate drag plans nothing anyway."""
    window, gate = _gate_window(box=True)
    try:
        (run_wall,) = _walls_on(window, {(22, _ROW)})
        _select_units(window, [gate, run_wall])
        asked = _recording(window, monkeypatch)
        window._gate_wall_scene()  # the refusal's own memo, warmed before the stub
        raising = {"on": True}
        _scene_walks(monkeypatch, raising)
        got = _hold(window, asked, (22.5, _ROW + 0.5), (22.5, _ROW + 3.5))
        raising["on"] = False
        assert (22.5, _ROW + 3.5, window._ghost_rotation_override(_entry(window, run_wall))) in got
        assert window._drag_plan_memo is None
    finally:
        _release(window, (22.5, _ROW + 3.5))
        _close(window)


def _drop_plan(window, dx: float, dy: float):
    """_move_group()'s plan for the selection at (dx, dy) under the drag's turn, built as it builds it."""
    index = window.map_view._unit_index
    entries = [index.entry_for_key(k) for k in window._selection]
    if window._drag_turn:
        entries = window._without_carried_pieces(entries)
    stand_ins = {id(e.unit): s for e in entries if (s := window._drag_stand_in(e, True)) is not e.unit}
    piece_ins, _pieces = window._piece_stand_ins(entries, stand_ins)
    return window._plan_unit_move(window._ensure_unit_edits(), entries, dx, dy, stand_ins | piece_ins)[1]


def test_a_box_off_group_carrying_a_gate_plans_its_corner_pieces_as_the_drop_does(monkeypatch):
    """TASK-531: the pieces leave the gate's end tiles, so a wall landing just
    north of (28, _ROW) sees it empty, a tower with only (29, _ROW) beside it,
    only when they are among the movers."""
    window, gate, pieces = _pieces_window(box=False)
    try:
        lone = _add_unit(window, WALL, 28.5, _ROW - 3.5)
        window._rebuild_unit_index()
        _select_units(window, [gate, lone])
        asked = _recording(window, monkeypatch)
        _hold(window, asked, (28.5, _ROW - 3.5), (28.5, _ROW - 0.5))
        _assert_same_plan(_preview_plan(window), _drop_plan(window, 0, 3))
        assert _ids(_preview_plan(window).landed) == [(id(lone), 2)]
        assert {id(p) for p in pieces} <= window._drag_scene_memo[2].moved_ids
    finally:
        _release(window, (28.5, _ROW - 0.5))
        _close(window)


def test_a_turn_mid_drag_plans_the_turned_gate_and_pieces_as_the_drop_does(monkeypatch):
    """TASK-545/532: the gate lands by its turned span and its pieces on its
    turned ends, and each Rotate input at the same delta replans."""
    window, gate, _pieces = _pieces_window(box=False, free=True)
    try:
        walls = _walls_on(window, {(29, _ROW), (30, _ROW)})
        _select_units(window, [gate, *walls])
        asked = _recording(window, monkeypatch)
        calls: list = []
        real = wall_run.plan_from_scene
        monkeypatch.setattr(wall_run, "plan_from_scene", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
        _hold(window, asked, (29.5, _ROW + 0.5), (29.5, _ROW + 6.5))
        flat = _preview_plan(window)
        _assert_same_plan(flat, _drop_plan(window, 0, 6))
        for _step in (1, 2):
            before = len(calls)
            window.rotate_cw_action.trigger()
            assert len(calls) == before + 1, "a Rotate input at the same delta replans"
            assert window._drag_plan_memo[1] == (0, 6)
            _assert_same_plan(_preview_plan(window), _drop_plan(window, 0, 6))
        assert _ids(_preview_plan(window).landed) != _ids(flat.landed), "the turned span changes a landing shape"
    finally:
        _release(window, (29.5, _ROW + 6.5))
        _close(window)


def _zoom_in(window) -> None:
    """test_gate_corner_pieces_viewer's: a drag can stay on its own tile past the threshold."""
    from PyQt5.QtCore import QPointF

    view = window.map_view
    view.scale(12, 12)
    view.centerOn(QPointF(_SRC[0] * view._tile_pixels, _SRC[1] * view._tile_pixels))
    assert view.transform().m11() * view._tile_pixels > 2 * view.UNIT_DRAG_THRESHOLD_PX


def test_a_drag_walks_the_scene_once_and_plans_once_per_tile(monkeypatch):
    window, _wall, _west, _east = _moved_wall_window()
    try:
        _recording(window, monkeypatch)
        walks = _scene_walks(monkeypatch)
        calls: list = []
        real = wall_run.plan_from_scene
        monkeypatch.setattr(wall_run, "plan_from_scene", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
        _zoom_in(window)
        _press(window, _SRC)
        for row in range(4):
            for offset in (0.05, 0.95) if row == 0 else (0.25, 0.75):
                _move(window, (30 + offset, _ROW + row + offset))
        assert (len(walks), len(calls)) == (1, 4)
    finally:
        _release(window, (30.5, _ROW + 3.5))
        _close(window)


@pytest.mark.parametrize("what", ["villager", "flat_wall"])
def test_a_villager_drag_or_a_flat_wall_drag_walks_no_scene(monkeypatch, what):
    window, _wall, _west, _east = _moved_wall_window()
    try:
        src = _SRC
        if what == "villager":
            _add_unit(window, VILLAGER, 60.5, 60.5)
            _recording(window, monkeypatch)
            src = (60.5, 60.5)
        raising = {"on": True}
        _scene_walks(monkeypatch, raising)
        _press(window, src)
        _move(window, (src[0], src[1] + 3))
        _move(window, (src[0], src[1] + 4))
        raising["on"] = False
        assert window.map_view._unit_ghost_item is not None, "the drag drew no ghost"
        assert window._drag_scene_memo is None
    finally:
        _release(window, (src[0], src[1] + 4))
        _close(window)
