"""The Replace Brush (GH #296) through a real offscreen ViewerWindow: the
stroke in descape/viewer_strokes.py, its Tool Options row and its History
labels. The match itself is tests/test_terrain_replace.py.

Every ViewerWindow here closes through conftest.close_window().
"""

from __future__ import annotations

import pytest
from test_brush_stroke import _shown_flat

from descape import brush, settings
from descape.edit_history import CompositeDiffRecord, TileDiffRecord
from descape.terrain_palette import name_for_terrain_id
from descape.terrain_target import CLEAR, LAYER, TERRAIN

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_BASE = 0  # the blank template's own terrain
_WATER = 1
_PICK = 2  # BEACH
_FOREST = 13  # never the pick: a terrain the brush must leave alone
_FOREST_OAK = 10  # density 1000/1000: one tree per tile when painted as a base
_FOREST_OAK_CONST = 411
_INDEX = {TERRAIN: 0, LAYER: 1, CLEAR: 2}


def _window(pick: int = _PICK, size: int = 3, shape: str = brush.BRUSH_SHAPE_SQUARE):
    window = conftest.terrain_edit_window()
    window.replace_brush_action.setChecked(True)
    assert window._current_tool == "replace_brush"
    assert window.terrain_panel.set_terrain(pick)
    window.paint_trees_check.setChecked(False)
    window.paint_eye_candy_check.setChecked(False)
    window.brush_size_spin.setValue(size)
    window.brush_shape_combo.setCurrentIndex(window.brush_shape_combo.findData(shape))
    return window


def _set(window, tiles, terrain_id: int, layer: int = -1) -> None:
    mm = window.scenario.map_manager
    for x, y in tiles:
        tile = mm.terrain[y * mm.map_width + x]
        tile.terrain_id, tile.layer = terrain_id, layer


def _pairs(window) -> dict[tuple[int, int], tuple[int, int]]:
    """(terrain_id, layer) of every tile that is not the template's (0, -1)."""
    mm = window.scenario.map_manager
    width = mm.map_width
    return {(i % width, i // width): (t.terrain_id, t.layer) for i, t in enumerate(mm.terrain) if (t.terrain_id, t.layer) != (_BASE, -1)}


def _stroke(window, tiles) -> None:
    window.on_edit_stroke_start()
    for x, y in tiles:
        window.on_edit_stroke_tile(x, y, 0)
    window.on_edit_stroke_end()


def _last_status(window) -> str:
    return window.status_log.toPlainText().splitlines()[-1]


def _square(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    return [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]


def test_a_drag_writes_only_matching_tiles_as_one_undo_record() -> None:
    window = _window()
    try:
        column = [(12, y) for y in range(5, 16)]
        _set(window, column, _FOREST)
        _stroke(window, [(x, 10) for x in range(10, 15)])
        covered = set(_square(9, 9, 15, 11))
        want = {t: (_PICK, -1) for t in covered if t not in column}
        want.update(dict.fromkeys(column, (_FOREST, -1)))
        assert _pairs(window) == want
        assert [r.label for r in window.edit_history.records] == ["Replace terrain"]
        assert isinstance(window.edit_history.records[0], TileDiffRecord)
        assert _last_status(window) == (
            f"Replaced {len(covered) - 3} tiles of {name_for_terrain_id(_BASE)} with {name_for_terrain_id(_PICK)}"
        )
        window.undo()
        assert _pairs(window) == dict.fromkeys(column, (_FOREST, -1))
    finally:
        conftest.close_window(window)


def _mouse(map_view, kind: str, tile, button) -> None:
    from PyQt5.QtCore import QEvent, Qt

    pos = conftest.viewport_pos(map_view, *tile)
    if kind == "press":
        map_view.mousePressEvent(conftest.mouse_event(QEvent.MouseButtonPress, pos, button, button))
    elif kind == "move":
        map_view.mouseMoveEvent(conftest.mouse_event(QEvent.MouseMove, pos, Qt.NoButton, button))
    else:
        map_view.mouseReleaseEvent(conftest.mouse_event(QEvent.MouseButtonRelease, pos, button, Qt.NoButton))


@pytest.mark.parametrize("button", ["left", "right"])
def test_the_press_tile_is_the_reference_through_mapview(button: str) -> None:
    """Pressed on the centre of a size-5 square outline, which the footprint
    leaves out: the match is still that tile's Water, not the ring's ground.
    Right-drag does what left-drag does."""
    from PyQt5.QtCore import Qt

    qt_button = Qt.LeftButton if button == "left" else Qt.RightButton
    window = _window(size=5, shape=brush.BRUSH_SHAPE_SQUARE_OUTLINE)
    try:
        _shown_flat(window)
        map_view = window.map_view
        mm = window.scenario.map_manager
        assert (20, 20) not in brush.brush_tiles(20, 20, 5, brush.BRUSH_SHAPE_SQUARE_OUTLINE, mm.map_width, mm.map_height)
        _set(window, [(20, 20), (18, 20), (23, 21)], _WATER)
        _mouse(map_view, "press", (20, 20), qt_button)
        _mouse(map_view, "move", (21, 20), qt_button)
        _mouse(map_view, "release", (21, 20), qt_button)
        # (18, 20) is on the press's ring (not its first or last tile), (23, 21) on the move's; the centre is on neither.
        assert _pairs(window) == {(20, 20): (_WATER, -1), (18, 20): (_PICK, -1), (23, 21): (_PICK, -1)}
        assert [r.label for r in window.edit_history.records] == ["Replace terrain"]
        assert f"Replaced 2 tiles of {name_for_terrain_id(_WATER)} with" in _last_status(window)
    finally:
        conftest.close_window(window)


def test_a_fast_drag_through_mapview_leaves_no_gap() -> None:
    """The cursor jumps six tiles between two events: the tiles between are replaced too."""
    from PyQt5.QtCore import Qt

    window = _window(size=1)
    try:
        _shown_flat(window)
        _mouse(window.map_view, "press", (10, 10), Qt.LeftButton)
        _mouse(window.map_view, "move", (16, 10), Qt.LeftButton)
        _mouse(window.map_view, "release", (16, 10), Qt.LeftButton)
        assert _pairs(window) == {(x, 10): (_PICK, -1) for x in range(10, 17)}
    finally:
        conftest.close_window(window)


def _island(window) -> list[tuple[int, int]]:
    """A Forest box at x 18..24, y 8..14, enclosing ground the press side can't reach. Returns the enclosed tiles."""
    box = set(_square(18, 8, 24, 14))
    inner = _square(19, 9, 23, 13)
    _set(window, sorted(box - set(inner)), _FOREST)
    return inner


def test_contiguous_is_honoured_and_global_crosses_the_gap() -> None:
    window = _window(size=9)
    try:
        inner = _island(window)
        window.replace_flood_combo.setCurrentIndex(window.replace_flood_combo.findData("contiguous"))
        _stroke(window, [(16, 11)])
        assert not any(_pairs(window).get(t) for t in inner), "Contiguous wrote the cut-off patch"
        assert _pairs(window)[(17, 11)] == (_PICK, -1)
        window.undo()

        window.replace_flood_combo.setCurrentIndex(window.replace_flood_combo.findData("global"))
        _stroke(window, [(16, 11)])
        assert _pairs(window)[(20, 11)] == (_PICK, -1)
    finally:
        conftest.close_window(window)


def test_the_flood_is_the_one_at_press_whatever_the_combo_does_mid_stroke() -> None:
    window = _window(size=9)
    try:
        inner = _island(window)
        combo = window.replace_flood_combo
        combo.setCurrentIndex(combo.findData("contiguous"))
        window.on_edit_stroke_start()
        window.on_edit_stroke_tile(14, 11, 0)
        combo.setCurrentIndex(combo.findData("global"))
        window.on_edit_stroke_tile(16, 11, 0)
        window.on_edit_stroke_tile(17, 11, 0)
        window.on_edit_stroke_end()
        assert not any(_pairs(window).get(t) for t in inner), "a mid-stroke Flood change reached the stroke"
        assert len(window.edit_history.records) == 1
    finally:
        conftest.close_window(window)


def test_a_press_off_the_maps_far_edge_writes_nothing() -> None:
    """Flat picks x == map_width on the map rect's far edge; its flat index would wrap onto the next row."""
    window = _window(size=3)
    try:
        width = window.scenario.map_manager.map_width
        _stroke(window, [(width, 5), (width - 1, 5)])
        assert _pairs(window) == {}
        assert window.edit_history.records == []
        assert _last_status(window).endswith("(no change)")
        assert "pressed off the map" in _last_status(window)
    finally:
        conftest.close_window(window)


def test_trees_are_planted_on_the_replaced_tiles_only_in_the_same_record() -> None:
    window = _window(pick=_FOREST_OAK)
    try:
        window.paint_trees_check.setChecked(True)
        _set(window, [(10, 10), (8, 9)], _WATER)
        _stroke(window, [(9, 10)])
        replaced = {t for t, pair in _pairs(window).items() if pair == (_FOREST_OAK, -1)}
        assert replaced == set(_square(8, 9, 10, 11)) - {(10, 10), (8, 9)}
        trees = [u for u in window.scenario.unit_manager.units[0] if u.unit_const == _FOREST_OAK_CONST]
        assert {(int(u.x), int(u.y)) for u in trees} == replaced
        [record] = window.edit_history.records
        assert isinstance(record, CompositeDiffRecord) and record.label == "Replace terrain"
        window.undo()
        assert not [u for u in window.scenario.unit_manager.units[0] if u.unit_const == _FOREST_OAK_CONST]
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize(
    ("target", "label", "status"),
    [
        (LAYER, "Replace layer", f"Replace layer: {name_for_terrain_id(_PICK)} layered on 3 tiles"),
        (CLEAR, "Clear layer (replace)", "Clear layer (replace): cleared layered terrain on 3 tiles"),
    ],
)
def test_history_labels_and_status_off_the_terrain_target(target: str, label: str, status: str) -> None:
    window = _window()
    try:
        # A row of nine: three plain, three layered Water, three Forest. The press is a plain tile for Layer.
        _set(window, [(x, 5) for x in range(13, 16)], _BASE, _WATER)
        _set(window, [(x, 5) for x in range(16, 19)], _FOREST)
        window.brush_size_spin.setValue(1)
        window.terrain_target_combo.setCurrentIndex(_INDEX[target])
        assert window._terrain_target() == target
        start = 13 if target == CLEAR else 10
        _stroke(window, [(x, 5) for x in range(start, 19)])
        assert [r.label for r in window.edit_history.records] == [label]
        assert _last_status(window).startswith(status)
        pairs = _pairs(window)
        if target == LAYER:
            assert {t: pairs[t] for t in [(x, 5) for x in range(10, 13)]} == dict.fromkeys([(x, 5) for x in range(10, 13)], (_BASE, _PICK))
            assert all(pairs[(x, 5)] == (_BASE, _WATER) for x in range(13, 16)), "a different layer was overwritten"
        else:
            assert all((x, 5) not in pairs for x in range(13, 16))
        assert all(pairs[(x, 5)] == (_FOREST, -1) for x in range(16, 19))
    finally:
        conftest.close_window(window)


def test_the_row_shows_flood_for_this_tool_only_and_no_auto_beach() -> None:
    window = _window()
    try:
        flood = (window.replace_flood_label_action, window.replace_flood_param_action)
        assert all(a.isVisible() for a in flood) and window.replace_flood_combo.isEnabled()
        assert [window.replace_flood_combo.itemText(i) for i in range(2)] == ["Global", "Contiguous"]
        assert window.replace_flood_combo.currentData() == "global"
        assert not window.auto_beach_param_action.isVisible()
        for action in (
            window.terrain_target_param_action, window.paint_trees_param_action,
            window.paint_eye_candy_param_action, window.tool_param_separator_action,
        ):
            assert action.isVisible()
        assert "only over tiles matching the one you press on" in window.replace_brush_action.toolTip()
        for tool in settings.TOOLS:
            if tool.tool_id != "replace_brush" and "terrain" in tool.modes:
                window._on_tool_selected(tool.tool_id)
                assert not any(a.isVisible() for a in flood), tool.tool_id
                assert not window.replace_flood_combo.isEnabled(), tool.tool_id
        window.mode_combo.setCurrentText("View")
        assert not any(a.isVisible() for a in flood)
    finally:
        conftest.close_window(window)


def test_it_greys_with_draw_on_a_locked_document(monkeypatch) -> None:
    window = _window()
    try:
        monkeypatch.setattr(window, "terrain_lock_reason", lambda family=None: "locked")
        window._update_tool_enabled()
        assert not window.draw_action.isEnabled()
        assert not window.replace_brush_action.isEnabled()
        assert not window.replace_flood_combo.isEnabled()
    finally:
        conftest.close_window(window)


def test_the_settings_default_reaches_the_combo() -> None:
    window = _window()
    try:
        assert window.apply_tool_option_default("replace_flood", "contiguous")
        assert window.replace_flood_combo.currentData() == "contiguous"
        assert window.tool_option_values()["replace_flood"] == "contiguous"
        assert not window.apply_tool_option_default("replace_flood", "sideways")
    finally:
        conftest.close_window(window)
