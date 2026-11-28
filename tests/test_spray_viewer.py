"""The Spray Can (GH #295) through a real offscreen ViewerWindow: MapView's
dab timer (descape/map_view_stroke_tick.py) and the viewer's spray stroke
(descape/viewer_strokes.py). The kernel itself is tests/test_spray.py.

MapView tests patch `_stroke_button_held` True unless they say otherwise:
they press through mousePressEvent directly, which sets no real button state.
Every ViewerWindow() here must call edit_history.mark_saved() before close().
"""

from __future__ import annotations

import pytest
from test_brush_stroke import _edit_window, _press, _release, _shown_flat

from descape import brush, spray

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_TERRAIN = 15  # test_brush_stroke's _edit_window pick, distinct from the template's 0


def _close(window) -> None:
    window.edit_history.mark_saved()
    window.close()


# -- MapView's tick ------------------------------------------------------------


def _tick_window(monkeypatch, tool: str = "spray"):
    """A shown Flat window with `tool` active, the held-button seam True and
    every delivery recorded (and still applied). Returns (window, calls)."""
    window = _edit_window(tool)
    _shown_flat(window)
    map_view = window.map_view
    monkeypatch.setattr(map_view, "_stroke_button_held", lambda: True)
    calls: list[list[tuple[int, int]]] = []
    real = map_view._on_stroke_tiles

    def record(tiles, modifiers) -> None:
        calls.append(list(tiles))
        real(tiles, modifiers)

    monkeypatch.setattr(map_view, "_on_stroke_tiles", record)
    return window, calls


def _move(map_view, tile: tuple[int, int]) -> None:
    from PyQt5.QtCore import QEvent, Qt

    map_view.mouseMoveEvent(
        conftest.mouse_event(QEvent.MouseMove, conftest.polygon_viewport_pos(map_view, *tile), Qt.NoButton, Qt.LeftButton)
    )


def _tile_under_nozzle(map_view):
    return map_view._pick_tile(map_view.mapToScene(map_view._stroke_nozzle_pos))


def test_the_press_dabs_once_and_a_move_only_moves_the_nozzle(monkeypatch) -> None:
    window, calls = _tick_window(monkeypatch)
    try:
        map_view = window.map_view
        _press(map_view, (40, 40))
        assert calls == [[(40, 40)]]
        assert map_view._stroke_tick_timer.isActive()
        assert map_view._stroke_tick_timer.interval() == spray.SPRAY_TICK_MS
        _move(map_view, (44, 40))
        _move(map_view, (48, 41))
        assert calls == [[(40, 40)]], "a move dabbed"
        assert _tile_under_nozzle(map_view) == (48, 41)
    finally:
        _close(window)


def test_a_tick_dabs_the_tile_under_the_nozzle(monkeypatch) -> None:
    window, calls = _tick_window(monkeypatch)
    try:
        map_view = window.map_view
        _press(map_view, (40, 40))
        map_view._on_stroke_tick()
        _move(map_view, (45, 42))
        map_view._on_stroke_tick()
        map_view._on_stroke_tick()
        assert calls == [[(40, 40)], [(40, 40)], [(45, 42)], [(45, 42)]]
    finally:
        _close(window)


def test_a_nozzle_off_the_map_sprays_nothing(monkeypatch) -> None:
    from PyQt5.QtCore import QEvent, QPointF, Qt

    window, calls = _tick_window(monkeypatch)
    try:
        map_view = window.map_view
        _press(map_view, (40, 40))
        corner = map_view.mapFromScene(map_view._map_rect.topLeft()) - QPointF(30, 30).toPoint()
        map_view.mouseMoveEvent(conftest.mouse_event(QEvent.MouseMove, QPointF(corner), Qt.NoButton, Qt.LeftButton))
        assert not map_view._pos_on_map(map_view.mapToScene(map_view._stroke_nozzle_pos))
        map_view._on_stroke_tick()
        assert calls == [[(40, 40)]]
        assert map_view._stroke_active, "an off-map nozzle keeps the stroke open"
    finally:
        _close(window)


def test_a_scroll_mid_hold_sprays_the_new_tile_under_the_same_position(monkeypatch) -> None:
    window, calls = _tick_window(monkeypatch)
    try:
        map_view = window.map_view
        map_view.scale(4, 4)  # zoomed in, so the scrollbars have room
        _press(map_view, (40, 40))
        pos = map_view._stroke_nozzle_pos
        bar = map_view.horizontalScrollBar()
        bar.setValue(bar.value() + 60)
        after = _tile_under_nozzle(map_view)
        assert after is not None and after != (40, 40)
        assert map_view._stroke_nozzle_pos == pos
        map_view._on_stroke_tick()
        assert calls[-1] == [after]
    finally:
        _close(window)


def test_no_dab_while_the_window_is_disabled_or_a_modal_is_up(monkeypatch) -> None:
    from PyQt5.QtWidgets import QApplication, QDialog

    window, calls = _tick_window(monkeypatch)
    try:
        map_view = window.map_view
        _press(map_view, (40, 40))
        window.setEnabled(False)
        map_view._on_stroke_tick()
        window.setEnabled(True)
        assert calls == [[(40, 40)]], "a tick dabbed under a busy span"

        dialog = QDialog(window)
        dialog.setModal(True)
        dialog.show()
        QApplication.processEvents()
        try:
            assert QApplication.activeModalWidget() is dialog
            map_view._on_stroke_tick()
        finally:
            dialog.close()
            dialog.deleteLater()
        assert calls == [[(40, 40)]], "a tick dabbed under a modal"
        assert map_view._stroke_active and map_view._stroke_tick_timer.isActive()

        map_view._on_stroke_tick()
        assert calls[-1] == [(40, 40)]
    finally:
        _close(window)


def test_a_tick_with_the_button_up_ends_the_stroke(monkeypatch) -> None:
    window, calls = _tick_window(monkeypatch)
    try:
        map_view = window.map_view
        _press(map_view, (40, 40))
        monkeypatch.setattr(map_view, "_stroke_button_held", lambda: False)
        map_view._on_stroke_tick()
        assert not map_view._stroke_active
        assert not map_view._stroke_tick_timer.isActive()
        assert not window.edit_history.in_stroke
        assert calls == [[(40, 40)]]
    finally:
        _close(window)


@pytest.mark.parametrize("closer", ["release", "set_tool", "clear_image", "window_close"])
def test_every_stroke_closing_path_stops_the_tick(closer: str, monkeypatch) -> None:
    window, calls = _tick_window(monkeypatch)
    try:
        map_view = window.map_view
        _press(map_view, (40, 40))
        assert map_view._stroke_tick_timer.isActive()
        window.edit_history.mark_saved()
        if closer == "release":
            _release(map_view, (40, 40))
        elif closer == "set_tool":
            map_view.set_tool("draw")
        elif closer == "clear_image":
            map_view.clear_image()
        else:
            window.close()
        assert not map_view._stroke_tick_timer.isActive(), f"{closer} left the tick running"
        assert map_view._stroke_nozzle_pos is None
        n = len(calls)
        map_view._on_stroke_tick()
        assert len(calls) == n, f"a tick after {closer} dabbed"
    finally:
        _close(window)


@pytest.mark.parametrize("tool", ["draw", "elevation", "set_level"])
def test_no_other_stroke_tool_starts_the_tick(tool: str, monkeypatch) -> None:
    window, _calls = _tick_window(monkeypatch, tool)
    try:
        _press(window.map_view, (40, 40))
        assert window.map_view._stroke_active
        assert not window.map_view._stroke_tick_timer.isActive()
        _release(window.map_view, (40, 40))
    finally:
        _close(window)


def test_the_real_timer_delivers_dabs(monkeypatch) -> None:
    """Wiring only: the rate is tools/bench_gui_drag.py's job."""
    import time

    from PyQt5.QtTest import QTest

    window, calls = _tick_window(monkeypatch)
    try:
        window.spray_flow_spin.setValue(100)
        _press(window.map_view, (40, 40))
        deadline = time.monotonic() + 2.0
        while len(calls) < 3 and time.monotonic() < deadline:
            QTest.qWait(10)
        assert len(calls) >= 3, f"{len(calls)} dabs in 2 s"
        _release(window.map_view, (40, 40))
        assert len(window.edit_history.records) == 1
    finally:
        _close(window)


# -- the viewer's spray stroke -------------------------------------------------

_SEED = 0x5EED
_FOREST_OAK = 10  # one tree per tile when painted as a base
_FOREST_OAK_CONST = 411


def predict(seed: int, flow: float, hardness: float, centres, size: int, shape: str, w: int, h: int, thickness: int = 1) -> set:
    """What a stroke of one dab per centre paints, with the viewer's painted-tile filter in front of each dab."""
    stroke = spray.SprayStroke(seed, flow, hardness)
    painted: set = set()
    for cx, cy in centres:
        fresh = [t for t in brush.brush_tiles(cx, cy, size, shape, w, h, thickness) if t not in painted]
        painted.update(spray.dab(stroke, fresh, cx, cy, size, shape, thickness))
    return painted


def _spray_window(monkeypatch, size: int = 9, flow: int = 30, hardness: int = 40, seed: int = _SEED):
    monkeypatch.setattr(spray, "new_seed", lambda: seed)
    window = _edit_window("spray")
    window.brush_size_spin.setValue(size)
    window.spray_flow_spin.setValue(flow)
    window.spray_hardness_spin.setValue(hardness)
    return window


def _dabs(window, centres) -> None:
    window.on_edit_stroke_start()
    for x, y in centres:
        window.on_edit_stroke_tiles([(x, y)], 0)
    window.on_edit_stroke_end()


def _painted(window, terrain_id: int = _TERRAIN) -> set:
    mm = window.scenario.map_manager
    return {(t.x, t.y) for t in mm.terrain if t.terrain_id == terrain_id}


def _predicted(window, flow: float, hardness: float, centres, seed: int = _SEED) -> set:
    mm = window.scenario.map_manager
    shape = window.brush_shape_combo.currentData()
    return predict(seed, flow, hardness, centres, window.brush_size_spin.value(), shape, mm.map_width, mm.map_height)


def test_dabs_at_one_point_paint_what_the_kernel_predicts_as_one_undo_record(monkeypatch) -> None:
    window = _spray_window(monkeypatch)
    try:
        centres = [(40, 40)] * 6
        _dabs(window, centres)
        want = _predicted(window, 0.3, 0.4, centres)
        assert 0 < len(want) < 81
        assert _painted(window) == want
        assert [r.label for r in window.edit_history.records] == ["Spray terrain"]
        assert window._stroke_spray is None
        window.undo()
        assert _painted(window) == set()
        window.redo()
        assert _painted(window) == want
    finally:
        _close(window)


def test_full_flow_and_hardness_paints_a_draw_stroke_in_one_dab(monkeypatch) -> None:
    window = _spray_window(monkeypatch, size=5, flow=100, hardness=100)
    try:
        _dabs(window, [(20, 30)])
        sprayed = _painted(window)
        window.undo()
        window._on_tool_selected("draw")
        _dabs(window, [(20, 30)])
        assert sprayed == _painted(window) and len(sprayed) == 25
    finally:
        _close(window)


def test_more_dabs_paint_a_superset_and_a_soft_brush_fills_its_centre_first(monkeypatch) -> None:
    window = _spray_window(monkeypatch, flow=20, hardness=0)
    try:
        sets = []
        for n in (2, 4, 8):
            _dabs(window, [(40, 40)] * n)
            sets.append(_painted(window))
            window.undo()
        assert sets[0] <= sets[1] <= sets[2] and sets[0] != sets[2]
        # Expected about 0.8 of the centre against 0.2 of the rim after 3 dabs at 30 %, over 6 seeds.
        centre = {(x, y) for x in range(39, 42) for y in range(39, 42)}
        rim = {(x, y) for x in range(36, 45) for y in range(36, 45)} - {(x, y) for x in range(37, 44) for y in range(37, 44)}
        window.spray_flow_spin.setValue(30)
        in_centre = in_rim = 0
        for seed in range(6):
            monkeypatch.setattr(spray, "new_seed", lambda seed=seed: seed)
            _dabs(window, [(40, 40)] * 3)
            in_centre += len(_painted(window) & centre)
            in_rim += len(_painted(window) & rim)
            window.undo()
        assert in_centre / (6 * len(centre)) > 0.6 > 0.35 > in_rim / (6 * len(rim)), (in_centre, in_rim)
    finally:
        _close(window)


def _rim_seed(flow: float) -> int:
    """A seed whose (44, 40) threshold lies between one full-weight dab and that plus its rim dab:
    it paints only if the rim dab's coverage is kept."""
    rim = spray.weights(9, "square", 1, 0.0)[(4, 0)]
    return next(s for s in range(10_000) if flow * rim < flow < spray.tile_threshold(s, 44, 40) < flow * (1 + rim) - 0.02)


def test_a_tile_first_met_at_the_rim_keeps_its_coverage_and_paints_under_the_nozzle(monkeypatch) -> None:
    """Pins dab() before the _stroke_painted update: marking a not-yet-crossed tile painted drops its coverage."""
    seed = _rim_seed(0.5)
    window = _spray_window(monkeypatch, flow=50, hardness=0, seed=seed)
    try:
        window.on_edit_stroke_start()
        window.on_edit_stroke_tiles([(40, 40)], 0)
        assert (44, 40) not in _painted(window)
        window.on_edit_stroke_tiles([(44, 40)], 0)
        assert (44, 40) in _painted(window), "the rim dab's coverage was dropped"
        window.on_edit_stroke_end()
        assert _painted(window) == _predicted(window, 0.5, 0.0, [(40, 40), (44, 40)], seed=seed)
    finally:
        _close(window)


def _set_target(window, target: str) -> None:
    from test_terrain_target_viewer import _INDEX

    window.terrain_target_combo.setCurrentIndex(_INDEX[target])
    assert window._terrain_target() == target


def _layers(window) -> dict:
    mm = window.scenario.map_manager
    return {(t.x, t.y): t.layer for t in mm.terrain if t.layer != -1}


def test_the_target_layer_and_clear_layer_and_their_labels(monkeypatch) -> None:
    from descape.terrain_palette import name_for_terrain_id
    from descape.terrain_target import CLEAR, LAYER, TERRAIN

    window = _spray_window(monkeypatch, size=5)
    try:
        log_before = window.status_log.toPlainText()
        _dabs(window, [(20, 20)] * 3)
        assert window.status_log.toPlainText() == log_before, "a Terrain-target spray logged a line"
        window.undo()

        _set_target(window, LAYER)
        _dabs(window, [(20, 20)] * 3)
        want = _predicted(window, 0.3, 0.4, [(20, 20)] * 3)
        assert _layers(window) == dict.fromkeys(want, _TERRAIN)
        assert _painted(window) == set(), "a layer spray wrote the base"
        assert window.edit_history.records[-1].label == "Spray layer"
        last = window.status_log.toPlainText().splitlines()[-1]
        assert f"Spray layer: {name_for_terrain_id(_TERRAIN)} layered on {len(want)} tiles" in last

        _set_target(window, CLEAR)
        window.spray_flow_spin.setValue(100)
        window.spray_hardness_spin.setValue(100)
        _dabs(window, [(20, 20)])
        assert _layers(window) == {}
        assert window.edit_history.records[-1].label == "Clear layer (spray)"
        assert f"Clear layer (spray): cleared layered terrain on {len(want)} tiles" in window.status_log.toPlainText()
        _set_target(window, TERRAIN)
    finally:
        _close(window)


def test_sprayed_forest_gets_trees_on_its_sprayed_tiles_only_in_one_undo_record(monkeypatch) -> None:
    window = _spray_window(monkeypatch, size=5, flow=50)
    try:
        window.terrain_panel.set_terrain(_FOREST_OAK)
        window.paint_trees_check.setChecked(True)
        _dabs(window, [(20, 20)] * 2)
        sprayed = _painted(window, _FOREST_OAK)
        assert sprayed == _predicted(window, 0.5, 0.4, [(20, 20)] * 2)
        assert 0 < len(sprayed) < 25
        trees = [(int(u.x), int(u.y)) for u in window.scenario.unit_manager.units[0] if u.unit_const == _FOREST_OAK_CONST]
        assert sorted(trees) == sorted(sprayed)
        assert len(window.edit_history.records) == 1
        window.undo()
        assert _painted(window, _FOREST_OAK) == set()
        assert not [u for u in window.scenario.unit_manager.units[0] if u.unit_const == _FOREST_OAK_CONST]
    finally:
        _close(window)


def test_a_flow_change_mid_stroke_does_not_change_the_open_stroke(monkeypatch) -> None:
    window = _spray_window(monkeypatch)
    try:
        window.on_edit_stroke_start()
        window.on_edit_stroke_tiles([(40, 40)], 0)
        window.spray_flow_spin.setValue(100)
        window.spray_hardness_spin.setValue(100)
        assert (window._stroke_spray.flow, window._stroke_spray.hardness) == (0.3, 0.4)
        window.on_edit_stroke_tiles([(40, 40)], 0)
        window.on_edit_stroke_end()
        assert _painted(window) == _predicted(window, 0.3, 0.4, [(40, 40)] * 2)
        assert len(_painted(window)) < 81
    finally:
        _close(window)
