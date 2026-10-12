"""The Tool Options row's Target combo, driven through a real offscreen
ViewerWindow: which of a tile's two terrains (its base, or the layered
terrain drawn over it) Draw, Draw Line, Draw Rectangle and Paint Can write.
The Qt-free half is tests/test_terrain_target.py and tests/test_fill_tools.py.

Same offscreen technique tests/test_fill_tool.py documents; every
ViewerWindow() here must call edit_history.mark_saved() before close().
"""

from __future__ import annotations

import numpy as np
import pytest
from test_sprite_toggle_viewer import sprite_install  # noqa: F401 -- fixture, requested by name

# blend_install is requested by name and requests the next two itself;
# blend_keys_match_the_mirrors is autouse, so every composite here checks the cache's blend key map.
from test_terrain_blend_render import (  # noqa: F401
    blend_install,
    blend_keys_match_the_mirrors,
    fake_farm,
    synthetic_textures,
)
from test_terrain_units_viewer import _canvas, _fresh

from descape.edit_history import TileDiffRecord
from descape.terrain_palette import name_for_terrain_id
from descape.terrain_target import CLEAR, LAYER, TERRAIN

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_BASE = 0  # the blank template's own terrain, GRASS_1
_PICK = 2  # BEACH
_OTHER = 9  # GRASS_3
_FOREST_OAK = 10  # density 1000/1000: one tree per tile when painted as a base
_FOREST_OAK_CONST = 411
_WATER = 1  # WATER_SHALLOW, a water-family terrain
_INDEX = {TERRAIN: 0, LAYER: 1, CLEAR: 2}


def _window(tool: str = "draw", pick: int = _PICK):
    """The blank template, Terrain mode, `tool` active, `pick` selected, Trees
    and Eye candy off. Caller must conftest.close_window()."""
    window = conftest.terrain_edit_window()
    window._on_tool_selected(tool)
    assert window.terrain_panel.set_terrain(pick)
    window.paint_trees_check.setChecked(False)
    window.paint_eye_candy_check.setChecked(False)
    return window


def _set_target(window, target: str) -> None:
    window.terrain_target_combo.setCurrentIndex(_INDEX[target])
    assert window._terrain_target() == target


def _stroke(window, tiles) -> None:
    window.on_edit_stroke_start()
    for x, y in tiles:
        window.on_edit_stroke_tile(x, y, 0)
    window.on_edit_stroke_end()


def _pairs(window) -> dict[tuple[int, int], tuple[int, int]]:
    """(terrain_id, layer) of every tile that is not the untouched template's (0, -1)."""
    mm = window.scenario.map_manager
    width = mm.map_width
    return {
        (i % width, i // width): (t.terrain_id, t.layer)
        for i, t in enumerate(mm.terrain)
        if (t.terrain_id, t.layer) != (_BASE, -1)
    }


def _last_status(window) -> str:
    return window.status_log.toPlainText().splitlines()[-1]


def _fail_if_called(*args, **kwargs):
    raise AssertionError("must not be called off the Terrain target")


# -- the combo and its gating -------------------------------------------------


def test_the_combo_starts_on_terrain_and_carries_the_three_targets() -> None:
    window = _window()
    try:
        combo = window.terrain_target_combo
        assert [combo.itemText(i) for i in range(combo.count())] == ["Terrain", "Layer", "Clear layer"]
        assert [combo.itemData(i) for i in range(combo.count())] == [TERRAIN, LAYER, CLEAR]
        assert combo.currentIndex() == 0
        assert window._terrain_target() == TERRAIN
        assert window.terrain_target_param_action.isVisible()
        assert "layered terrain" in combo.toolTip()
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("target", [LAYER, CLEAR])
def test_trees_eye_candy_and_auto_beach_hide_off_the_terrain_target(target: str) -> None:
    window = _window("draw", pick=_WATER)
    try:
        window.auto_beach_check.setChecked(True)
        shown = (
            window.paint_trees_param_action, window.paint_eye_candy_param_action,
            window.auto_beach_param_action, window.beach_param_action, window.beach_width_param_action,
        )
        live = (
            window.paint_trees_check, window.paint_eye_candy_check, window.auto_beach_check,
            window.beach_combo, window.beach_width_spin,
        )
        assert all(action.isVisible() for action in shown)
        assert all(widget.isEnabled() for widget in live)

        _set_target(window, target)
        assert not any(action.isVisible() for action in shown)
        assert not any(widget.isEnabled() for widget in live)
        # The combo itself, the brush and the separator stay.
        assert window.terrain_target_param_action.isVisible()
        assert window.brush_size_spin_action.isVisible()
        assert window.tool_param_separator_action.isVisible()

        _set_target(window, TERRAIN)
        assert all(action.isVisible() for action in shown)
        assert all(widget.isEnabled() for widget in live)
    finally:
        conftest.close_window(window)


def test_a_file_with_no_layer_field_hides_and_resets_the_combo_and_writes_no_layer() -> None:
    window = _window("draw")
    try:
        _set_target(window, LAYER)
        logged = window.status_log.toPlainText()
        window.scenario.terrain_has_layer = False  # as a v1.21 document
        window._update_tool_enabled()

        assert not window.terrain_target_param_action.isVisible()
        assert not window.terrain_target_label_action.isVisible()
        assert not window.terrain_target_combo.isEnabled()
        assert window.terrain_target_combo.currentIndex() == 0
        assert window._terrain_target() == TERRAIN
        assert window.status_log.toPlainText() == logged  # the reset is silent
        assert window.paint_trees_param_action.isVisible()  # the Terrain target's row is back

        _stroke(window, [(5, 5)])
        window._on_tool_selected("draw_rect")
        window.on_shape_commit([(8, 8), (9, 8)])
        window._on_tool_selected("fill")
        window.on_fill(30, 30, 0)
        mm = window.scenario.map_manager
        assert all(t.layer == -1 for t in mm.terrain)
        assert all(t.terrain_id == _PICK for t in mm.terrain)  # every tool painted the base
        assert [r.label for r in window.edit_history.records] == ["Paint terrain", "Draw rectangle", "Fill terrain"]
    finally:
        conftest.close_window(window)


def test_a_forced_combo_index_on_a_file_with_no_layer_field_still_writes_the_base() -> None:
    """The gate is _terrain_target(), not the combo: an index left on Layer
    (set before the next re-gate) must not reach a tile."""
    window = _window("draw")
    try:
        window.scenario.terrain_has_layer = False
        window.terrain_target_combo.blockSignals(True)
        window.terrain_target_combo.setCurrentIndex(1)
        window.terrain_target_combo.blockSignals(False)
        assert window.terrain_target_combo.currentData() == LAYER
        assert window._terrain_target() == TERRAIN

        _stroke(window, [(5, 5)])
        assert _pairs(window) == {(5, 5): (_PICK, -1)}
        assert window.edit_history.records[-1].label == "Paint terrain"
    finally:
        conftest.close_window(window)


# -- Draw ---------------------------------------------------------------------


def test_a_draw_layer_stroke_is_one_record_that_undoes_and_redoes() -> None:
    window = _window("draw")
    try:
        _set_target(window, LAYER)
        tiles = [(5, 5), (6, 5), (7, 6)]
        _stroke(window, tiles)

        want = dict.fromkeys(tiles, (_BASE, _PICK))
        assert _pairs(window) == want
        assert len(window.edit_history.records) == 1
        record = window.edit_history.records[-1]
        assert isinstance(record, TileDiffRecord)
        assert record.label == "Paint layer"
        assert f"Paint layer: {name_for_terrain_id(_PICK)} layered on 3 tiles" in _last_status(window)

        window.undo()
        assert _pairs(window) == {}
        window.redo()
        assert _pairs(window) == want
    finally:
        conftest.close_window(window)


def test_a_draw_clear_layer_stroke_is_one_record_that_undoes_and_redoes() -> None:
    window = _window("draw")
    try:
        _set_target(window, LAYER)
        _stroke(window, [(5, 5), (6, 5), (7, 6)])
        _set_target(window, CLEAR)
        _stroke(window, [(6, 5), (7, 6), (20, 20)])  # (20, 20) never had a layer

        assert _pairs(window) == {(5, 5): (_BASE, _PICK)}
        assert len(window.edit_history.records) == 2
        record = window.edit_history.records[-1]
        assert record.label == "Clear layer"
        assert len(record.changes) == 2
        assert "Clear layer: cleared layered terrain on 2 tiles" in _last_status(window)

        window.undo()
        assert _pairs(window) == {(5, 5): (_BASE, _PICK), (6, 5): (_BASE, _PICK), (7, 6): (_BASE, _PICK)}
        window.redo()
        assert _pairs(window) == {(5, 5): (_BASE, _PICK)}
    finally:
        conftest.close_window(window)


def test_a_layer_stroke_of_the_tiles_own_base_writes_no_layer_and_no_record() -> None:
    window = _window("draw", pick=_PICK)
    try:
        _stroke(window, [(5, 5)])  # Terrain target: (5, 5) is BEACH now
        _set_target(window, LAYER)
        _stroke(window, [(5, 5)])
        assert _pairs(window) == {(5, 5): (_PICK, -1)}
        assert len(window.edit_history.records) == 1
        assert "layered on 0 tiles" in _last_status(window)
    finally:
        conftest.close_window(window)


def test_a_target_change_mid_stroke_does_not_half_apply() -> None:
    """The Target is snapshotted at stroke start, like Auto beach: a combo
    change while the button is down must not split one undo record in two."""
    window = _window("draw")
    try:
        _set_target(window, LAYER)
        window.on_edit_stroke_start()
        window.on_edit_stroke_tile(5, 5, 0)
        window.terrain_target_combo.setCurrentIndex(_INDEX[TERRAIN])
        window.on_edit_stroke_tile(6, 5, 0)
        window.terrain_target_combo.setCurrentIndex(_INDEX[CLEAR])
        window.on_edit_stroke_tile(7, 5, 0)
        window.on_edit_stroke_end()

        assert _pairs(window) == {(5, 5): (_BASE, _PICK), (6, 5): (_BASE, _PICK), (7, 5): (_BASE, _PICK)}
        assert len(window.edit_history.records) == 1
        assert window.edit_history.records[-1].label == "Paint layer"

        # The next stroke takes the combo's new state.
        _stroke(window, [(5, 5)])
        assert _pairs(window) == {(6, 5): (_BASE, _PICK), (7, 5): (_BASE, _PICK)}
        assert window.edit_history.records[-1].label == "Clear layer"
    finally:
        conftest.close_window(window)


def test_a_water_layer_stroke_lays_no_beach_ring() -> None:
    window = _window("draw", pick=_WATER)
    try:
        window.auto_beach_check.setChecked(True)
        assert window.auto_beach_check.isEnabled()
        _set_target(window, LAYER)
        _stroke(window, [(10, 10)])
        # One layered tile, and no neighbour turned into a beach.
        assert _pairs(window) == {(10, 10): (_BASE, _WATER)}

        _set_target(window, TERRAIN)
        assert window.auto_beach_check.isChecked()  # the tick survived the round trip
        _stroke(window, [(30, 30)])
        ring = {tile for tile, (terrain_id, _layer) in _pairs(window).items() if terrain_id not in (_BASE, _WATER)}
        assert ring, "the Terrain stroke laid no ring, so the Layer case above proved nothing"
    finally:
        conftest.close_window(window)


# -- nothing is planted, and nothing asks, off the Terrain target --------------


def _gaia_trees(window) -> list:
    return [u for u in window.scenario.unit_manager.units[0] if u.unit_const == _FOREST_OAK_CONST]


@pytest.mark.parametrize("target", [LAYER, CLEAR])
def test_off_terrain_no_tool_reaches_the_unit_plan_or_a_confirm(target: str, monkeypatch) -> None:
    """A spy, not a "plants nothing" assertion: terrain_units already ignores
    a tile whose terrain_id did not change, so that would pass without the
    skip. Trees is ticked, the pick is a forest and the confirm threshold is
    1, so on the Terrain target every one of these would plan and ask."""
    import descape.viewer as viewer_module

    window = _window("draw", pick=_FOREST_OAK)
    try:
        mm = window.scenario.map_manager
        # A layered patch for the Clear layer gestures to act on.
        for i in range(mm.map_width * 3):
            mm.terrain[i].layer = _OTHER
        window.paint_trees_check.setChecked(True)
        window.paint_eye_candy_check.setChecked(True)
        _set_target(window, target)
        monkeypatch.setattr(viewer_module, "TERRAIN_UNIT_CONFIRM_THRESHOLD", 1)
        monkeypatch.setattr(viewer_module.QMessageBox, "question", staticmethod(_fail_if_called))
        monkeypatch.setattr(window, "_apply_terrain_unit_plan", _fail_if_called)

        _stroke(window, [(1, 1), (2, 1)])
        window._on_tool_selected("draw_line")
        window.on_shape_commit([(x, 0) for x in range(10, 40)])
        window._on_tool_selected("draw_rect")
        window.on_shape_commit([(x, y) for y in range(3) for x in range(50, 110)])  # 180 tiles
        window._on_tool_selected("fill")
        window.on_fill(45, 2, 0)  # what is left of the layered patch

        labels = {
            LAYER: ["Paint layer", "Draw layer line", "Draw layer rectangle", "Fill layer"],
            CLEAR: ["Clear layer", "Clear layer (line)", "Clear layer (rectangle)", "Clear layer (fill)"],
        }[target]
        assert [r.label for r in window.edit_history.records] == labels
        assert all(isinstance(r, TileDiffRecord) for r in window.edit_history.records)
        assert _gaia_trees(window) == []
        assert all(t.terrain_id == _BASE for t in mm.terrain)
    finally:
        conftest.close_window(window)


def test_the_same_gestures_on_the_terrain_target_do_reach_the_unit_plan() -> None:
    """The spy test's own control: with the Target on Terrain the same window
    setup plants a tree per painted tile."""
    window = _window("draw", pick=_FOREST_OAK)
    try:
        window.paint_trees_check.setChecked(True)
        _stroke(window, [(1, 1), (2, 1)])
        assert len(_gaia_trees(window)) == 2
    finally:
        conftest.close_window(window)


# -- Draw Line, Draw Rectangle and Paint Can per target ------------------------


@pytest.mark.parametrize(
    ("tool", "layer_label", "clear_label"),
    [
        ("draw_line", "Draw layer line", "Clear layer (line)"),
        ("draw_rect", "Draw layer rectangle", "Clear layer (rectangle)"),
    ],
)
def test_a_shape_writes_the_layer_then_clears_it(tool: str, layer_label: str, clear_label: str) -> None:
    window = _window(tool)
    try:
        tiles = [(x, 7) for x in range(4, 10)]
        name = name_for_terrain_id(_PICK)
        _set_target(window, LAYER)
        window.on_shape_commit(tiles)
        assert _pairs(window) == dict.fromkeys(tiles, (_BASE, _PICK))
        assert window.edit_history.records[-1].label == layer_label
        assert f"{layer_label}: {name} layered on 6 of 6 tiles" in _last_status(window)

        # Again over the same tiles plus two new ones: only the new ones change.
        window.on_shape_commit([*tiles, (10, 7), (11, 7)])
        assert f"{layer_label}: {name} layered on 2 of 8 tiles" in _last_status(window)
        assert len(window.edit_history.records) == 2

        _set_target(window, CLEAR)
        window.on_shape_commit([(x, 7) for x in range(8)])  # 4 unlayered, then 4 layered
        assert _pairs(window) == {(x, 7): (_BASE, _PICK) for x in range(8, 12)}
        assert window.edit_history.records[-1].label == clear_label
        assert f"{clear_label}: cleared layered terrain on 4 of 8 tiles" in _last_status(window)

        window.undo()
        assert _pairs(window) == {(x, 7): (_BASE, _PICK) for x in range(4, 12)}
        window.undo()
        window.undo()
        assert _pairs(window) == {}
    finally:
        conftest.close_window(window)


def test_a_shape_on_the_terrain_target_keeps_its_wording_and_clears_layers() -> None:
    window = _window("draw_rect")
    try:
        mm = window.scenario.map_manager
        mm.terrain[7 * mm.map_width + 4].layer = _OTHER
        window.on_shape_commit([(4, 7), (5, 7)])
        assert _pairs(window) == {(4, 7): (_PICK, -1), (5, 7): (_PICK, -1)}
        assert window.edit_history.records[-1].label == "Draw rectangle"
        assert _last_status(window) == f"Draw rectangle: 2 tiles set to {name_for_terrain_id(_PICK)}"
    finally:
        conftest.close_window(window)


def _two_patch_window():
    """Paint Can active over a 6x4 BEACH rectangle at (10, 10) whose west half
    carries a GRASS_3 layer, on the template's GRASS_1."""
    window = _window("draw_rect", pick=_PICK)
    window.on_shape_commit([(x, y) for y in range(10, 14) for x in range(10, 16)])
    _set_target(window, LAYER)
    window.terrain_panel.set_terrain(_OTHER)
    window.on_shape_commit([(x, y) for y in range(10, 14) for x in range(10, 13)])
    window._on_tool_selected("fill")
    return window


def _patch(x0: int, x1: int, pair: tuple[int, int]) -> dict:
    return {(x, y): pair for y in range(10, 14) for x in range(x0, x1)}


def test_a_layer_fill_covers_the_clicked_pair_region_only() -> None:
    window = _two_patch_window()
    try:
        before = len(window.edit_history.records)
        window.terrain_panel.set_terrain(_FOREST_OAK)
        window.on_fill(14, 11, 0)  # the unlayered east half
        assert _pairs(window) == {**_patch(10, 13, (_PICK, _OTHER)), **_patch(13, 16, (_PICK, _FOREST_OAK))}
        assert len(window.edit_history.records) == before + 1
        assert window.edit_history.records[-1].label == "Fill layer"
        assert f"Filled 12 tiles with layer {name_for_terrain_id(_FOREST_OAK)} from (14, 11)" in _last_status(window)

        window.on_fill(14, 11, 0)  # the same layer again
        assert len(window.edit_history.records) == before + 1
        assert _last_status(window) == (
            f"Fill at (14, 11): the layered terrain is already {name_for_terrain_id(_FOREST_OAK)} (no change)"
        )

        window.undo()
        assert _pairs(window) == {**_patch(10, 13, (_PICK, _OTHER)), **_patch(13, 16, (_PICK, -1))}
    finally:
        conftest.close_window(window)


def test_a_layer_fill_with_the_regions_own_base_clears_or_does_nothing() -> None:
    window = _two_patch_window()
    try:
        before = len(window.edit_history.records)
        window.terrain_panel.set_terrain(_PICK)  # the rectangle's own base
        window.on_fill(14, 11, 0)  # unlayered: nothing to write
        assert len(window.edit_history.records) == before
        assert "is the base terrain here" in _last_status(window)
        assert "(no change)" in _last_status(window)

        window.on_fill(11, 11, 0)  # the layered half: a layer equal to the base is none
        assert _pairs(window) == _patch(10, 16, (_PICK, -1))
        assert window.edit_history.records[-1].label == "Clear layer (fill)"
        assert "Cleared layered terrain on 12 tiles from (11, 11)" in _last_status(window)
    finally:
        conftest.close_window(window)


def test_a_clear_layer_fill_removes_one_patch_and_ignores_the_pick() -> None:
    window = _two_patch_window()
    try:
        before = len(window.edit_history.records)
        _set_target(window, CLEAR)
        window.terrain_panel.set_terrain(_FOREST_OAK)  # ignored
        window.on_fill(14, 11, 0)  # no layer here
        assert len(window.edit_history.records) == before
        assert _last_status(window) == "Fill at (14, 11): no layered terrain here (no change)"

        window.on_fill(11, 11, 0)
        assert _pairs(window) == _patch(10, 16, (_PICK, -1))
        assert len(window.edit_history.records) == before + 1
        assert window.edit_history.records[-1].label == "Clear layer (fill)"
        assert "Cleared layered terrain on 12 tiles from (11, 11)" in _last_status(window)

        window.undo()
        assert _pairs(window) == {**_patch(10, 13, (_PICK, _OTHER)), **_patch(13, 16, (_PICK, -1))}
    finally:
        conftest.close_window(window)


# -- the visibility note ------------------------------------------------------


def test_the_note_is_logged_while_layered_terrain_is_not_drawn(monkeypatch) -> None:
    from descape import asset_source

    no_blendomatic = lambda: None  # noqa: E731
    no_blendomatic.cache_clear = lambda: None
    monkeypatch.setattr(asset_source, "get_blendomatic", no_blendomatic)
    window = _window("draw")
    try:
        install_note = "not drawn without an AoE2:DE install configured"
        assert window._layer_visibility_note() == install_note
        _set_target(window, LAYER)
        assert _last_status(window) == f"Target: Layer (layered terrain is {install_note})"
        assert install_note in window.terrain_target_combo.toolTip()
        _stroke(window, [(5, 5)])
        assert _last_status(window) == f"Paint layer: {name_for_terrain_id(_PICK)} layered on 1 tiles ({install_note})"
        window._on_tool_selected("fill")
        window.on_fill(5, 5, 0)
        assert "(no change)" in _last_status(window)
        window.on_fill(0, 0, 0)
        assert _last_status(window).endswith(f"({install_note})")

        _set_target(window, TERRAIN)
        assert _last_status(window) == "Target: Terrain"

        window._on_tool_selected("draw")
        _set_target(window, CLEAR)
        window._on_layer_toggled("terrain_blend", False)
        blend_note = "not drawn while View > Layers > Terrain Blending is off"
        assert window._layer_visibility_note() == blend_note
        assert blend_note in window.terrain_target_combo.toolTip()  # the toggle re-gated the row
        window._on_layer_toggled("terrain_textures", False)
        assert window._layer_visibility_note() == "not drawn while View > Layers > Terrain Textures is off"
    finally:
        conftest.close_window(window)


def test_the_note_is_absent_while_blending_is_effective(blend_install) -> None:  # noqa: F811
    window = _window("draw")
    try:
        assert window._layer_visibility_note() == ""
        _set_target(window, LAYER)
        assert _last_status(window) == "Target: Layer"
        assert "Right now" not in window.terrain_target_combo.toolTip()
        _stroke(window, [(5, 5)])
        assert _last_status(window) == f"Paint layer: {name_for_terrain_id(_PICK)} layered on 1 tiles"
    finally:
        conftest.close_window(window)


# -- render: every layer edit patches to what a fresh render draws --------------

_RENDER_STEPS = (
    "draw layer", "undo draw", "fill layer (patched)", "undo fill", "fill layer (past the threshold)",
    "undo big fill", "clear layer rectangle", "undo rectangle",
)


def _render_window(style: str):
    window = _window("draw", pick=_OTHER)
    if style == "Flat":
        # Top-down Flat, not the Flat+Isometric path (see test_fill_tool's _shown_flat).
        window.iso_action.setChecked(False)
    window.terrain_style_combo.setCurrentText(style)
    # A BEACH base rectangle, so the fills below have a small pair region to find.
    window.terrain_panel.set_terrain(_PICK)
    window._on_tool_selected("draw_rect")
    window.on_shape_commit([(x, y) for y in range(20, 26) for x in range(20, 28)])
    window.terrain_panel.set_terrain(_OTHER)
    _set_target(window, LAYER)
    return window


def _render_history(window, style: str, monkeypatch, before_step=None) -> None:
    """A Draw Layer stroke, a small Layer fill, a Layer fill past the
    full-refresh threshold and a Clear layer rectangle, each followed by its
    Undo. The whole canvas is rendered right before each step (so a stale
    chunk cannot be hidden by an eviction) and compared with a fresh render
    right after it, failing as `<step>: ...`."""
    import descape.viewer as viewer_module

    def draw():
        window._on_tool_selected("draw")
        _stroke(window, [(40, 40), (41, 40), (41, 41), (60, 12)])

    def fill():
        window._on_tool_selected("fill")
        window.on_fill(22, 22, 0)  # the 48-tile BEACH rectangle

    def big_fill():
        # 14,352 tiles of GRASS_1: past the lowered threshold, so Stepped and Sloped refresh wholesale.
        monkeypatch.setattr(viewer_module, "STEPPED_FULL_RERENDER_THRESHOLD", 100)
        window.on_fill(0, 0, 0)

    def big_undo():
        window.undo()
        monkeypatch.setattr(viewer_module, "STEPPED_FULL_RERENDER_THRESHOLD", 20_000)

    def rectangle():
        fill()  # something for the rectangle to clear
        _canvas(window)
        _set_target(window, CLEAR)
        window._on_tool_selected("draw_rect")
        window.on_shape_commit([(x, y) for y in range(18, 23) for x in range(18, 24)])

    acts = (draw, window.undo, fill, window.undo, big_fill, big_undo, rectangle, window.undo)
    for step, act in zip(_RENDER_STEPS, acts, strict=True):
        before = _canvas(window)  # every chunk resident
        if before_step is not None:
            before_step(step)
        act()
        after = _canvas(window)
        assert np.array_equal(after, _fresh(window, style)), f"{step}: the canvas differs from a fresh render"
        assert not np.array_equal(after, before), f"{step}: no pixel changed, so the check above proved nothing"


@pytest.mark.parametrize("style", ["Flat", "Stepped", "Sloped"])
def test_layer_edits_and_their_undo_match_a_fresh_render(style: str, blend_install, monkeypatch) -> None:  # noqa: F811
    from descape import render

    window = _render_window(style)
    try:
        assert render.blend_effective(window._cache.layers), "blending is off, so the case would prove nothing"
        _render_history(window, style, monkeypatch)
        if style != "Flat":
            assert "refreshed the full map in place instead of patching" in window.status_log.toPlainText()
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("broken", ["draw layer", "undo fill", "clear layer rectangle"])
@pytest.mark.parametrize("style", ["Flat", "Stepped", "Sloped"])
def test_the_render_check_names_the_step_whose_layer_mirror_went_stale(
    style: str, broken: str, blend_install, monkeypatch  # noqa: F811
) -> None:
    """Control (a step-k break fails naming step k): on step `broken` only,
    nothing re-reads the terrain mirrors, so the cache composites the old layer."""
    window = _render_window(style)
    try:
        cache = window._cache
        real = cache._refresh_terrain_ids
        current = {}
        monkeypatch.setattr(
            cache, "_refresh_terrain_ids", lambda *a, **k: None if current.get("step") == broken else real(*a, **k)
        )
        with pytest.raises(AssertionError, match=f"{broken}: the canvas differs"):
            _render_history(window, style, monkeypatch, before_step=lambda step: current.update(step=step))
    finally:
        conftest.close_window(window)
