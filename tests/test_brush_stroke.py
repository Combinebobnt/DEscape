"""Brush size/shape wiring for the drag-stroke edit tools (Draw, Elevate,
Set Elevation), driven through a real offscreen ViewerWindow -- same
technique and default-tier rationale as tests/test_fill_tool.py and
tests/test_toolbar_params.py. tests/test_brush.py covers the pure geometry;
tests/test_elevation_tools.py covers the batched-elevation correctness fix;
this module covers the actual stroke wiring in descape/viewer.py (the
footprint loop inside on_edit_stroke_tile, the painted-tile dedupe, and the
hover preview).

Every ViewerWindow() constructed here must call edit_history.mark_saved()
before close() -- see tests/test_fill_tool.py's module docstring for why.
"""

from __future__ import annotations

import pytest

from descape.brush import BRUSH_SHAPE_CIRCLE, BRUSH_SHAPE_SQUARE

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_TERRAIN = 15  # GRASS_1, distinct from the blank template's own terrain_id=0


def _edit_window(tool: str = "draw"):
    window = conftest.terrain_edit_window()
    window._on_tool_selected(tool)
    window.terrain_panel.set_terrain(_TERRAIN)
    # This file tests brush/stroke mechanics, not descape/terrain_units.py --
    # Trees defaults on and would otherwise pop an unpatched large-fill
    # confirm QMessageBox for the whole-map fills below, hanging offscreen.
    window.paint_trees_check.setChecked(False)
    window.paint_eye_candy_check.setChecked(False)
    return window


def _shown_flat(window) -> None:
    from PyQt5.QtWidgets import QApplication

    # Unchecked BEFORE the style switch, while still in Stepped -- iso_action
    # defaults checked (MapView._isometric's own default), so Flat would
    # otherwise render through the Flat+Isometric plan's real-iso path
    # instead of the plain top-down canvas this module means to exercise.
    window.iso_action.setChecked(False)
    window.terrain_style_combo.setCurrentText("Flat")
    window.show()
    QApplication.processEvents()


def _stroke(window, cx: int, cy: int, modifiers: int = 0) -> None:
    """A single-cursor-tile stroke, called directly rather than through real
    mouse events -- matches tests/test_fill_tool.py's
    test_mid_drag_tool_switch...'s own direct-call style for the cases that
    don't need to exercise MapView's own event routing."""
    window.on_edit_stroke_start()
    window.on_edit_stroke_tile(cx, cy, modifiers)
    window.on_edit_stroke_end()


def test_square_brush_paints_whole_footprint_in_one_undo_record() -> None:
    window = _edit_window("draw")
    try:
        window.brush_size_spin.setValue(3)
        window.brush_shape_combo.setCurrentIndex(window.brush_shape_combo.findData(BRUSH_SHAPE_SQUARE))
        mm = window.scenario.map_manager

        _stroke(window, 10, 10)

        painted = [t for t in mm.terrain if t.terrain_id == _TERRAIN]
        assert len(painted) == 9
        assert len(window.edit_history.records) == 1
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_circle_brush_paints_exactly_the_circle_footprint() -> None:
    from descape.brush import brush_tiles

    window = _edit_window("draw")
    try:
        window.brush_size_spin.setValue(5)
        window.brush_shape_combo.setCurrentIndex(window.brush_shape_combo.findData(BRUSH_SHAPE_CIRCLE))
        mm = window.scenario.map_manager

        _stroke(window, 40, 40)

        expected = set(brush_tiles(40, 40, 5, BRUSH_SHAPE_CIRCLE, mm.map_width, mm.map_height))
        painted = {(t.x, t.y) for t in mm.terrain if t.terrain_id == _TERRAIN}
        assert painted == expected
        # The 4 corners of the bounding 5x5 square are excluded by the
        # circle shape -- confirm they're specifically untouched, not just
        # that the count matches.
        for dx, dy in [(-2, -2), (2, -2), (-2, 2), (2, 2)]:
            assert mm.get_tile(40 + dx, 40 + dy).terrain_id != _TERRAIN
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_elevate_drag_raises_each_tile_at_most_once_per_stroke() -> None:
    """The load-bearing regression test. Cursor-tile dedupe
    (MapView._stroke_touched) is NOT the same as painted-tile dedupe
    (ViewerWindow._stroke_painted) once a brush is bigger than one tile: a
    3x3 brush dragged across several cursor tiles has painted tiles that
    fall under more than one cursor position. Without _stroke_painted,
    Elevate's accumulating +1 would raise those overlapping tiles more than
    once in a single stroke. Driven through real QMouseEvents (not direct
    on_edit_stroke_tile calls) so MapView's own _touch_tile dedupe is
    genuinely exercised, matching test_fill_tool.py's
    test_drag_after_click_fills_only_once."""
    from PyQt5.QtCore import QEvent, Qt

    window = _edit_window("elevation")
    try:
        window.brush_size_spin.setValue(3)
        window.brush_shape_combo.setCurrentIndex(window.brush_shape_combo.findData(BRUSH_SHAPE_SQUARE))
        _shown_flat(window)
        map_view = window.map_view
        mm = window.scenario.map_manager

        cx, cy = 10, 10
        press_pos = conftest.viewport_pos(map_view, cx, cy)
        map_view.mousePressEvent(conftest.mouse_event(QEvent.MouseButtonPress, press_pos, Qt.LeftButton, Qt.LeftButton))

        # Drag right one cursor tile at a time -- each step's 3x3 footprint
        # overlaps the previous step's by two columns.
        for step in range(1, 5):
            move_pos = conftest.viewport_pos(map_view, cx + step, cy)
            map_view.mouseMoveEvent(conftest.mouse_event(QEvent.MouseMove, move_pos, Qt.NoButton, Qt.LeftButton))

        map_view.mouseReleaseEvent(
            conftest.mouse_event(QEvent.MouseButtonRelease, press_pos, Qt.LeftButton, Qt.NoButton)
        )

        touched = [
            t.elevation
            for y in range(cy - 2, cy + 3)
            for x in range(cx - 2, cx + 7)
            for t in [mm.get_tile(x, y)]
            if t.elevation != 0
        ]
        assert touched, "expected the drag to have raised at least one tile"
        assert max(touched) == 1, f"a tile was raised more than once in a single stroke: elevations {touched}"
        assert len(window.edit_history.records) == 1
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_footprint_clipped_at_map_corner_does_not_raise() -> None:
    window = _edit_window("draw")
    try:
        window.brush_size_spin.setValue(9)
        window.brush_shape_combo.setCurrentIndex(window.brush_shape_combo.findData(BRUSH_SHAPE_SQUARE))
        mm = window.scenario.map_manager

        _stroke(window, 0, 0)  # no exception -- footprint clips to the map

        painted = [t for t in mm.terrain if t.terrain_id == _TERRAIN]
        assert 0 < len(painted) < 81
        for t in mm.terrain:
            if t.terrain_id == _TERRAIN:
                assert 0 <= t.x < mm.map_width
                assert 0 <= t.y < mm.map_height
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_brush_size_1_matches_pre_brush_single_tile_behavior() -> None:
    window = _edit_window("draw")
    try:
        assert window.brush_size_spin.value() == 1  # the default
        mm = window.scenario.map_manager

        _stroke(window, 20, 20)

        painted = [(t.x, t.y) for t in mm.terrain if t.terrain_id == _TERRAIN]
        assert painted == [(20, 20)]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_multi_step_stroke_hands_each_change_to_apply_dirty_exactly_once() -> None:
    """The incremental dirty-set bookkeeping in on_edit_stroke_tile, pinned
    as a contract rather than as an implementation shape, so a rewrite of it
    (the two passes over the cumulative dirty set became one) has to keep
    behaving identically.

    Set Elevation propagates, so a single tile is re-changed several times
    over an 8-step drag; the non-vacuity assertion below fails if the
    fixture ever stops reproducing that. The three real assertions: nothing
    changed goes unreported, nothing is reported at a state it was already
    shown at, and every index's LAST report carries its final state (the
    drift _stroke_seen_state exists to prevent; see
    tests/test_stroke_elevation_sync.py for what that drift looked like).
    """
    from collections import Counter

    from descape.edit_history import tile_state

    window = _edit_window("set_level")
    try:
        window.brush_size_spin.setValue(5)
        window.brush_shape_combo.setCurrentIndex(window.brush_shape_combo.findData(BRUSH_SHAPE_CIRCLE))
        window.elevation_level_spin.setValue(6)
        mm = window.scenario.map_manager
        start = [tile_state(t) for t in mm.terrain]

        # Snapshotted AT CALL TIME: the terrain keeps mutating for the rest
        # of the drag, so the indices alone say nothing about what was shown.
        reports: list[dict[int, tuple[int, int, int]]] = []
        original = window._apply_dirty

        def recording(dirty_indices) -> None:
            reports.append({i: tile_state(mm.terrain[i]) for i in dirty_indices})
            original(dirty_indices)

        window._apply_dirty = recording
        try:
            window.on_edit_stroke_start()
            for step in range(8):
                window.on_edit_stroke_tile(40 + step, 40, 0)
            window.on_edit_stroke_end()
        finally:
            del window._apply_dirty

        final = [tile_state(t) for t in mm.terrain]
        changed = {i for i, state in enumerate(final) if state != start[i]}
        assert changed, "the drag changed nothing"

        counts = Counter(i for report in reports for i in report)
        assert any(c > 1 for c in counts.values()), (
            "no tile was reported twice, so the fixture no longer reproduces mid-drag "
            "propagation and the dedupe assertions below are vacuous"
        )

        assert changed <= set(counts), "a tile changed by the stroke was never handed to _apply_dirty"

        last_state: dict[int, tuple[int, int, int]] = {}
        for report in reports:
            for i, state in report.items():
                # Start state is reportable only as a revert from something already shown.
                assert state != last_state.get(i, start[i]), f"index {i} reported at a state already shown"
                last_state[i] = state
        for i, shown in last_state.items():
            assert shown == final[i], f"index {i} was last shown at {shown}, ended at {final[i]}"
    finally:
        window.edit_history.mark_saved()
        window.close()


def _record_apply_dirty(window):
    """Wraps window._apply_dirty; returns the list each call appends its
    index set to. Undo with `del window._apply_dirty`."""
    calls: list[set[int]] = []
    original = window._apply_dirty

    def recording(dirty_indices) -> None:
        calls.append(set(dirty_indices))
        original(dirty_indices)

    window._apply_dirty = recording
    return calls


def test_a_tile_returned_to_its_stroke_start_state_is_repainted() -> None:
    """Batch F Track S's intended behaviour change. Raising (40, 40) from 1
    to 2 ramps (41, 40) from 0 to 1; a Shift-lower on (41, 40) then takes
    both back to their start states. The old O(map) scan never re-reported
    a tile once it matched the start again, so both stayed stale on screen."""
    from PyQt5.QtCore import Qt

    from descape.edit_history import tile_state

    window = _edit_window("set_level")
    try:
        mm = window.scenario.map_manager
        window.elevation_level_spin.setValue(1)
        _stroke(window, 40, 40)
        window._on_tool_selected("elevation")
        start = {i: tile_state(mm.terrain[i]) for i in (40 * mm.map_width + 40, 40 * mm.map_width + 41)}
        hill, ramp = start

        calls = _record_apply_dirty(window)
        try:
            window.on_edit_stroke_start()
            window.on_edit_stroke_tile(40, 40, 0)
            assert mm.get_tile(40, 40).elevation == 2 and mm.get_tile(41, 40).elevation == 1
            window.on_edit_stroke_tile(41, 40, Qt.ShiftModifier)
            assert mm.get_tile(40, 40).elevation == 1 and mm.get_tile(41, 40).elevation == 0
            window.on_edit_stroke_end()
        finally:
            del window._apply_dirty

        assert {hill, ramp} <= calls[0]
        assert {hill, ramp} <= calls[1], "the revert to the stroke-start state was not repainted"
        assert {i: tile_state(mm.terrain[i]) for i in start} == start
    finally:
        window.edit_history.mark_saved()
        window.close()


def _full_scan_guard(window, events):
    """Drives one stroke of `events` ((tiles, modifiers) per mouse event) and
    asserts, after each, that _apply_dirty got exactly the full-map oracle:
    every index whose state differs from the last state handed over, or from
    the stroke-start state if never handed over."""
    from descape.edit_history import tile_state

    mm = window.scenario.map_manager
    start = [tile_state(t) for t in mm.terrain]
    ref_seen: dict[int, tuple[int, int, int]] = {}
    calls = _record_apply_dirty(window)
    try:
        window.on_edit_stroke_start()
        for n, (tiles, modifiers) in enumerate(events):
            before = len(calls)
            window.on_edit_stroke_tiles(tiles, modifiers)
            got = set().union(*calls[before:])
            ref = set()
            for i, t in enumerate(mm.terrain):
                state = tile_state(t)
                if state != ref_seen.get(i, start[i]):
                    ref_seen[i] = state
                    ref.add(i)
            assert got == ref, f"event {n}: missing {sorted(ref - got)[:8]}, extra {sorted(got - ref)[:8]}"
        window.on_edit_stroke_end()
    finally:
        del window._apply_dirty
    return calls


def _path(y: int, x0: int, n: int) -> list[tuple[int, int]]:
    return [(x0 + k, y) for k in range(n)]


def _guard_case(case: str):
    from PyQt5.QtCore import Qt

    shift = Qt.ShiftModifier
    if case == "draw":
        window = _edit_window("draw")
        window.brush_size_spin.setValue(3)
        events = [(_path(40, 30 + 3 * k, 3), 0) for k in range(5)]
    elif case == "beach_draw":
        window = _beach_window(beach_id=None, width=2)
        window.brush_size_spin.setValue(3)
        events = [(_path(40, 30 + 3 * k, 3), 0) for k in range(5)]
    else:
        # A level-4 hill first, so Elevate's propagation has slopes to work on.
        window = _edit_window("set_level")
        window.brush_size_spin.setValue(3)
        window.elevation_level_spin.setValue(4)
        _stroke(window, 40, 40)
        if case == "set_level":
            window.elevation_level_spin.setValue(1)
            events = [(_path(40, 34 + 2 * k, 2), 0) for k in range(6)]
        else:
            window._on_tool_selected("elevation")
            if case == "elevate":
                events = [(_path(40, 34 + 2 * k, 2), 0) for k in range(6)]
            elif case == "lower":
                events = [(_path(40, 34 + 2 * k, 2), shift) for k in range(6)]
            else:  # Shift toggled mid-drag, back and forth over the hill
                events = [(_path(40 + (k % 2), 34 + 2 * k, 2), shift if k % 2 else 0) for k in range(6)]
    return window, events


@pytest.mark.parametrize("case", ["draw", "beach_draw", "elevate", "lower", "shift_toggle", "set_level"])
def test_live_stroke_repaint_set_equals_a_full_map_scan(case) -> None:
    window, events = _guard_case(case)
    try:
        calls = _full_scan_guard(window, events)
        assert sum(len(c) for c in calls) > len(events), "the stroke repainted almost nothing; the guard is vacuous"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_full_scan_guard_catches_an_unreported_beach_ring(monkeypatch) -> None:
    """Control: the ring's indices dropped from `touched` must fail the guard."""
    import descape.viewer as viewer_module

    real = viewer_module.apply_beach_ring
    monkeypatch.setattr(viewer_module, "apply_beach_ring", lambda *a, **k: (real(*a, **k), [])[1])
    window, events = _guard_case("beach_draw")
    try:
        with pytest.raises(AssertionError, match="missing"):
            _full_scan_guard(window, events)
    finally:
        if window.edit_history.in_stroke:
            window.on_edit_stroke_end()
        window.edit_history.mark_saved()
        window.close()


def _release_record_guard(window, events):
    """Drives one stroke and asserts the release's tile record (diffed over
    the stroke's reported writes only) equals a full-map diff against the
    stroke-start state. Returns the record's changes."""
    from descape.edit_history import tile_state

    mm = window.scenario.map_manager
    start = [tile_state(t) for t in mm.terrain]
    history = window.edit_history
    real_build = history.build_stroke_record
    built = []

    def build(*args, **kwargs):
        built.append(real_build(*args, **kwargs))
        return built[-1]

    history.build_stroke_record = build
    try:
        window.on_edit_stroke_start()
        for tiles, modifiers in events:
            window.on_edit_stroke_tiles(tiles, modifiers)
        window.on_edit_stroke_end()
    finally:
        del history.build_stroke_record
    oracle = [(i, start[i], tile_state(t)) for i, t in enumerate(mm.terrain) if tile_state(t) != start[i]]
    (record,) = built
    got = record.changes if record is not None else []
    assert got == oracle, f"missing {sorted({c[0] for c in oracle} - {c[0] for c in got})[:8]}"
    return got


@pytest.mark.parametrize("case", ["draw", "beach_draw", "elevate", "lower", "shift_toggle", "set_level"])
def test_the_release_record_equals_a_full_map_diff(case) -> None:
    window, events = _guard_case(case)
    try:
        changes = _release_record_guard(window, events)
        assert len(changes) > len(events), "the stroke changed almost nothing; the guard is vacuous"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_release_record_drops_a_tile_changed_then_restored() -> None:
    """Raise (40, 40) on a level-1 hill (ramping its ring), Shift-lower
    (41, 40) back (restoring both of those), then raise (60, 60): both are
    written during the stroke but must not be recorded."""
    from PyQt5.QtCore import Qt

    window = _edit_window("set_level")
    try:
        mm = window.scenario.map_manager
        window.elevation_level_spin.setValue(1)
        _stroke(window, 40, 40)
        window._on_tool_selected("elevation")
        events = [([(40, 40)], 0), ([(41, 40)], Qt.ShiftModifier), ([(60, 60)], 0)]
        changes = _release_record_guard(window, events)
        recorded = {i for i, _old, _new in changes}
        w = mm.map_width
        assert 60 * w + 60 in recorded and 40 * w + 39 in recorded
        assert not recorded & {40 * w + 40, 40 * w + 41}, "a restored tile was recorded"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_release_record_guard_catches_an_unreported_beach_ring(monkeypatch) -> None:
    """Control: a mutation that under-reports its writes must fail the guard,
    or the touched-only diff could silently drop tiles from undo."""
    import descape.viewer as viewer_module

    real = viewer_module.apply_beach_ring
    monkeypatch.setattr(viewer_module, "apply_beach_ring", lambda *a, **k: (real(*a, **k), [])[1])
    window, events = _guard_case("beach_draw")
    try:
        with pytest.raises(AssertionError, match="missing"):
            _release_record_guard(window, events)
    finally:
        if window.edit_history.in_stroke:
            window.on_edit_stroke_end()
        window.edit_history.mark_saved()
        window.close()


def test_paint_can_ignores_brush_size() -> None:
    window = _edit_window("fill")
    try:
        window.brush_size_spin.setValue(9)  # left over from a prior tool selection
        mm = window.scenario.map_manager

        window.on_fill(0, 0, 0)

        # A flood fill, not a 9x9 patch -- covers the whole uniform map.
        assert all(t.terrain_id == _TERRAIN for t in mm.terrain)
        assert not window.brush_size_spin_action.isVisible()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_hover_preview_matches_the_stroke_footprint() -> None:
    """Checks the actual QPainterPath geometry, not just the memo key --
    Flat mode's axis-aligned _tile_polygon makes tile-center containment a
    reliable, simple check. Also confirms the painted set (after a real
    stroke at the same cursor tile) equals the same expected footprint, so
    the preview and the edit are shown to agree, not just each independently
    match descape.brush's own output."""
    from PyQt5.QtCore import QPointF

    from descape.brush import brush_tiles

    window = _edit_window("draw")
    try:
        window.brush_size_spin.setValue(5)
        window.brush_shape_combo.setCurrentIndex(window.brush_shape_combo.findData(BRUSH_SHAPE_CIRCLE))
        _shown_flat(window)
        mm = window.scenario.map_manager
        map_view = window.map_view
        tp = map_view._tile_pixels

        map_view._update_highlight(40, 40)
        expected = set(brush_tiles(40, 40, 5, BRUSH_SHAPE_CIRCLE, mm.map_width, mm.map_height))
        assert map_view._highlight_key == (40, 40, 5, BRUSH_SHAPE_CIRCLE)
        assert map_view._highlight_outline_item is not None

        path = map_view._highlight_outline_item.path()
        for tx, ty in expected:
            center = QPointF((tx + 0.5) * tp, (ty + 0.5) * tp)
            assert path.contains(center), f"expected highlight to cover tile ({tx}, {ty})"
        for dx, dy in [(-2, -2), (2, -2), (-2, 2), (2, 2)]:  # circle-excluded bounding-box corners
            corner = QPointF((40 + dx + 0.5) * tp, (40 + dy + 0.5) * tp)
            assert not path.contains(corner), f"highlight should exclude the corner at offset ({dx}, {dy})"

        _stroke(window, 40, 40)
        painted = {(t.x, t.y) for t in mm.terrain if t.terrain_id == _TERRAIN}
        assert painted == expected
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("style", ["Flat", "Stepped"])
@pytest.mark.parametrize(
    ("size", "shape"), [(3, BRUSH_SHAPE_SQUARE), (9, BRUSH_SHAPE_SQUARE), (3, BRUSH_SHAPE_CIRCLE), (9, BRUSH_SHAPE_CIRCLE)]
)
def test_every_highlight_subpath_is_closed_so_every_tile_edge_strokes(style: str, size: int, shape: str) -> None:
    """QPainterPath.addPolygon() leaves the subpath OPEN, so the outline pen
    stroked 3 of each tile's 4 edges. Same defect tests/
    test_unit_selection_viewer.py's own subpath check pins for the unit cues,
    found there first; this path survived it because _update_highlight paints
    a fill over the very same path, which hides the missing edge.

    Counted (MoveTo + 3 LineTo + the close = 5 elements per tile) rather than
    eyeballed, and at a brush size > 1 so several subpaths have to close, not
    just the one.
    """
    window = _edit_window("draw")
    try:
        window.brush_size_spin.setValue(size)
        window.brush_shape_combo.setCurrentIndex(window.brush_shape_combo.findData(shape))
        if style == "Flat":
            _shown_flat(window)
        else:
            window.terrain_style_combo.setCurrentText(style)
        map_view = window.map_view
        map_view._update_highlight(40, 40)
        path = map_view._highlight_outline_item.path()
        moves = sum(1 for i in range(path.elementCount()) if path.elementAt(i).type == 0)
        assert moves > 1, f"{style}: expected several subpaths at brush size {size}, got {moves}"
        assert path.elementCount() == 5 * moves, (
            f"{style}: unclosed subpath. {path.elementCount()} elements for {moves} "
            f"tile polygon(s), expected {5 * moves}"
        )
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize(("size", "shape"), [(3, BRUSH_SHAPE_SQUARE), (9, BRUSH_SHAPE_CIRCLE)])
def test_every_sloped_highlight_subpath_returns_to_its_move_to(size: int, shape: str) -> None:
    """GH #87 in Sloped, where a tile outline is not a 4-gon, so 5 elements
    per subpath does not hold. Closure is checked directly instead."""
    from PyQt5.QtCore import QPointF
    from PyQt5.QtGui import QPainterPath
    from PyQt5.QtWidgets import QApplication

    window = _edit_window("draw")
    try:
        window.brush_size_spin.setValue(size)
        window.brush_shape_combo.setCurrentIndex(window.brush_shape_combo.findData(shape))
        window.terrain_style_combo.setCurrentText("Sloped")
        window.show()
        QApplication.processEvents()
        map_view = window.map_view
        map_view._update_highlight(40, 40)
        path = map_view._highlight_outline_item.path()

        starts = [i for i in range(path.elementCount()) if path.elementAt(i).type == QPainterPath.MoveToElement]
        assert len(starts) > 1, f"expected several subpaths at brush size {size}, got {len(starts)}"
        for start, end in zip(starts, [*starts[1:], path.elementCount()], strict=True):
            assert end - start > 2, f"subpath at element {start} is degenerate"
            first, last = path.elementAt(start), path.elementAt(end - 1)
            assert QPointF(last.x, last.y) == QPointF(first.x, first.y), f"subpath at element {start} is left open"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_brush_size_change_refreshes_preview_without_a_mouse_move() -> None:
    window = _edit_window("draw")
    try:
        window.on_hover((40, 40))
        map_view = window.map_view

        map_view._update_highlight(40, 40)
        assert map_view._highlight_key == (40, 40, 1, BRUSH_SHAPE_SQUARE)

        window.brush_size_spin.setValue(5)  # no mouse move in between
        assert map_view._highlight_key == (40, 40, 5, BRUSH_SHAPE_SQUARE)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_brush_resets_to_size_1_square_on_a_fresh_window() -> None:
    """Guards the no-persistence requirement: brush state is session-only,
    with no settings.py config key, so a fresh window always starts at
    size 1 / square regardless of what a previous window in the same
    process left the spinbox/combo at."""
    window = _edit_window("draw")
    try:
        window.brush_size_spin.setValue(9)
        window.brush_shape_combo.setCurrentIndex(window.brush_shape_combo.findData(BRUSH_SHAPE_CIRCLE))
    finally:
        window.edit_history.mark_saved()
        window.close()

    fresh = _edit_window("draw")
    try:
        assert fresh.brush_size_spin.value() == 1
        assert fresh.brush_shape_combo.currentData() == BRUSH_SHAPE_SQUARE
    finally:
        fresh.edit_history.mark_saved()
        fresh.close()


# -- the highlight pulse's stroke gate (perf batch B, step B3) ---------------


def _press(map_view, tile: tuple[int, int]) -> None:
    from PyQt5.QtCore import QEvent, Qt

    map_view.mousePressEvent(
        conftest.mouse_event(QEvent.MouseButtonPress, conftest.polygon_viewport_pos(map_view, *tile), Qt.LeftButton, Qt.LeftButton)
    )


def _release(map_view, tile: tuple[int, int]) -> None:
    from PyQt5.QtCore import QEvent, Qt

    map_view.mouseReleaseEvent(
        conftest.mouse_event(QEvent.MouseButtonRelease, conftest.polygon_viewport_pos(map_view, *tile), Qt.LeftButton, Qt.NoButton)
    )


def test_the_pulse_pauses_for_the_stroke_and_resumes_on_release() -> None:
    """Its 40ms tick dirties the highlight's scene rect, which re-enters the
    canvas repaint. That is pure competition with the edit work while a
    stroke is running, and the cursor isn't resting there to be found."""
    window = _edit_window("draw")
    try:
        _shown_flat(window)
        map_view = window.map_view
        assert map_view._pulse_timer.isActive(), "an edit tool alone should pulse"

        _press(map_view, (40, 40))
        assert map_view._stroke_active
        assert not map_view._pulse_timer.isActive(), "the stroke should pause the pulse"

        _release(map_view, (40, 40))
        assert not map_view._stroke_active
        assert map_view._pulse_timer.isActive(), "the release should resume it"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gc_is_off_for_the_stroke_and_back_on_after_release() -> None:
    """A full collection mid-stroke was a 20-45 ms step spike on June."""
    import gc

    window = _edit_window("draw")
    try:
        _shown_flat(window)
        map_view = window.map_view
        _press(map_view, (40, 40))
        assert not gc.isenabled(), "the stroke should hold gc off"
        _release(map_view, (40, 40))
        assert gc.isenabled(), "the release should turn it back on"
    finally:
        gc.enable()
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("closer", ["leave", "clear_image", "window_close"])
def test_every_stroke_closing_path_turns_gc_back_on(closer: str) -> None:
    import gc

    from PyQt5.QtCore import QEvent

    window = _edit_window("draw")
    try:
        _shown_flat(window)
        map_view = window.map_view
        _press(map_view, (40, 40))
        assert not gc.isenabled()
        window.edit_history.mark_saved()
        if closer == "leave":
            map_view.leaveEvent(QEvent(QEvent.Leave))
        elif closer == "clear_image":
            map_view.clear_image()
        else:
            window.close()
        assert gc.isenabled(), f"{closer} left gc disabled"
    finally:
        gc.enable()
        window.edit_history.mark_saved()
        window.close()


def test_a_stroke_does_not_enable_gc_someone_else_disabled() -> None:
    import gc

    window = _edit_window("draw")
    try:
        _shown_flat(window)
        map_view = window.map_view
        gc.disable()
        _press(map_view, (40, 40))
        _release(map_view, (40, 40))
        assert not gc.isenabled(), "the stroke re-enabled gc it never disabled"
    finally:
        gc.enable()
        window.edit_history.mark_saved()
        window.close()


def test_a_highlight_rebuilt_mid_stroke_is_not_left_fully_opaque() -> None:
    """The pause's own blind spot: _update_highlight() creates a fresh fill
    item at Qt's default opacity of 1.0, and with the pulse stopped nothing
    would bring it back down until the stroke ended. Reached in the app by
    the cursor crossing off-map mid-drag, which clears the highlight."""
    window = _edit_window("draw")
    try:
        _shown_flat(window)
        map_view = window.map_view
        map_view._update_highlight(40, 40)

        _press(map_view, (40, 40))
        map_view._clear_highlight()
        map_view._update_highlight(41, 41)

        opacity = map_view._highlight_fill_item.opacity()
        assert map_view.HIGHLIGHT_PULSE_MIN_ALPHA <= opacity <= map_view.HIGHLIGHT_PULSE_MAX_ALPHA, opacity

        # And the resumed pulse still drives it, rather than the paused
        # value being latched in.
        _release(map_view, (41, 41))
        map_view._pulse_phase_ms = map_view.HIGHLIGHT_PULSE_PERIOD_MS // 4
        map_view._on_pulse_tick()
        assert map_view._highlight_fill_item.opacity() != opacity
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_double_click_on_one_tile_is_two_strokes_but_one_undo_record(monkeypatch) -> None:
    """GH #46 step 11. mouseDoubleClickEvent re-enters the press handler, so
    the second click starts its own stroke; it repaints nothing, so pushes nothing."""
    from PyQt5.QtCore import QEvent, Qt

    window = _edit_window("draw")
    try:
        _shown_flat(window)
        map_view = window.map_view
        starts = []
        real_begin = window.edit_history.begin_stroke
        monkeypatch.setattr(window.edit_history, "begin_stroke", lambda tiles: (starts.append(1), real_begin(tiles))[1])
        before = len(window.edit_history.records)

        _press(map_view, (40, 40))
        _release(map_view, (40, 40))
        pos = conftest.polygon_viewport_pos(map_view, 40, 40)
        map_view.mouseDoubleClickEvent(conftest.mouse_event(QEvent.MouseButtonDblClick, pos, Qt.LeftButton, Qt.LeftButton))
        _release(map_view, (40, 40))

        assert len(starts) == 2
        assert len(window.edit_history.records) == before + 1
        assert window.scenario.map_manager.get_tile(40, 40).terrain_id == _TERRAIN
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- auto beach (2026-08-31 water/beach plan, Stage 4) ------------------------

_WATER_DEEP = 22
_BEACH = 2
_BEACH_WET = 107


def _beach_window(beach_id=_BEACH, width: int = 1, enabled: bool = True):
    """_edit_window with Draw on a water terrain and auto-beach configured."""
    window = _edit_window("draw")
    window.terrain_panel.set_terrain(_WATER_DEEP)
    window.auto_beach_check.setChecked(enabled)
    if beach_id is None:
        window.beach_combo.setCurrentIndex(0)  # "Auto"
    else:
        window.beach_combo.setCurrentIndex(window.beach_combo.findData(beach_id))
    window.beach_width_spin.setValue(max(width, 1))
    if width == 0:
        window.auto_beach_check.setChecked(False)
    return window


def _grid(window):
    mm = window.scenario.map_manager
    return {
        (x, y): mm.get_tile(x, y).terrain_id
        for y in range(mm.map_height)
        for x in range(mm.map_width)
    }


def _tiles_with(window, terrain_id: int) -> set[tuple[int, int]]:
    return {tile for tile, tid in _grid(window).items() if tid == terrain_id}


def test_auto_beach_rings_a_single_touch() -> None:
    window = _beach_window()
    try:
        _stroke(window, 20, 20)
        water = _tiles_with(window, _WATER_DEEP)
        beach = _tiles_with(window, _BEACH)
        assert water == {(20, 20)}
        assert beach == {
            (19, 19), (20, 19), (21, 19),
            (19, 20), (21, 20),
            (19, 21), (20, 21), (21, 21),
        }
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_drag_leaves_no_beach_inside_the_water() -> None:
    """Hazard 1, the highest-value assertion in the plan and otherwise a
    silent bug: if ring tiles ever reached _stroke_painted, a tile beached by
    an earlier touch would be filtered out when the brush advanced over it
    and never become water, stranding beach inside the water body."""
    window = _beach_window()
    try:
        window.on_edit_stroke_start()
        for x in range(18, 24):
            window.on_edit_stroke_tile(x, 20, 0)
        window.on_edit_stroke_end()

        cores = {(x, 20) for x in range(18, 24)}
        grid = _grid(window)
        assert {tile for tile in cores if grid[tile] != _WATER_DEEP} == set()
        # And the shoreline runs the length of the stroke on both sides.
        for x in range(18, 24):
            assert grid[(x, 19)] == _BEACH
            assert grid[(x, 21)] == _BEACH
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_one_auto_beach_drag_is_one_undo_record() -> None:
    window = _beach_window()
    try:
        before = len(window.edit_history.records)
        window.on_edit_stroke_start()
        for x in range(18, 22):
            window.on_edit_stroke_tile(x, 20, 0)
        window.on_edit_stroke_end()
        assert len(window.edit_history.records) == before + 1

        window.undo()
        assert _tiles_with(window, _WATER_DEEP) == set()
        assert _tiles_with(window, _BEACH) == set()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_checkbox_off_reproduces_a_plain_draw_stroke() -> None:
    plain = _edit_window("draw")
    plain.terrain_panel.set_terrain(_WATER_DEEP)
    beached = _beach_window(enabled=False)
    try:
        for window in (plain, beached):
            _stroke(window, 20, 20)
        assert _grid(plain) == _grid(beached)
    finally:
        for window in (plain, beached):
            window.edit_history.mark_saved()
            window.close()


def test_repainting_water_over_water_pushes_no_phantom_undo_step() -> None:
    window = _beach_window()
    try:
        _stroke(window, 20, 20)
        after_first = len(window.edit_history.records)
        _stroke(window, 20, 20)
        assert len(window.edit_history.records) == after_first
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_right_button_stroke_beaches_identically() -> None:
    """_touch_tile ORs ShiftModifier in for right-button drags, so this pins
    that the auto-beach path ignores `modifiers`, exactly as Draw does."""
    from PyQt5.QtCore import Qt

    left = _beach_window()
    right = _beach_window()
    try:
        _stroke(left, 20, 20, 0)
        _stroke(right, 20, 20, Qt.ShiftModifier)
        assert _grid(left) == _grid(right)
    finally:
        for window in (left, right):
            window.edit_history.mark_saved()
            window.close()


def test_toggling_the_checkbox_mid_stroke_does_not_half_apply() -> None:
    """Hazard 2: the checkbox, combo and width are live widgets, so reading
    them per touch would split one undo record across two settings. They are
    snapshotted at on_edit_stroke_start instead."""
    window = _beach_window(enabled=True)
    try:
        window.on_edit_stroke_start()
        window.on_edit_stroke_tile(18, 20, 0)
        window.auto_beach_check.setChecked(False)  # mid-stroke change
        window.on_edit_stroke_tile(19, 20, 0)
        window.on_edit_stroke_end()
        grid = _grid(window)
        # The second touch still beaches: the stroke uses its own snapshot.
        assert grid[(20, 19)] == _BEACH
        assert grid[(20, 21)] == _BEACH
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_an_explicit_beach_terrain_is_used_instead_of_auto() -> None:
    window = _beach_window(beach_id=_BEACH_WET)
    try:
        _stroke(window, 20, 20)
        assert _tiles_with(window, _BEACH_WET)
        assert _tiles_with(window, _BEACH) == set()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_wider_ring_paints_a_wider_shoreline() -> None:
    window = _beach_window(width=3)
    try:
        _stroke(window, 20, 20)
        beach = _tiles_with(window, _BEACH)
        assert len(beach) == 7 * 7 - 1
        assert (17, 17) in beach
        assert (16, 20) not in beach
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_ring_follows_the_brush_footprint() -> None:
    window = _beach_window()
    try:
        window.brush_size_spin.setValue(3)
        _stroke(window, 20, 20)
        water = _tiles_with(window, _WATER_DEEP)
        assert water == {(x, y) for y in range(19, 22) for x in range(19, 22)}
        beach = _tiles_with(window, _BEACH)
        assert (18, 18) in beach
        assert (20, 18) in beach
        assert not (beach & water)
    finally:
        window.edit_history.mark_saved()
        window.close()
