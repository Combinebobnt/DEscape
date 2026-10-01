"""Place Unit's hover ghost (GH #128): a translucent copy of the picked object,
owned by the picked owner, that follows the cursor at exactly the point and in
exactly the look a click would place it.

Layered like tests/test_unit_drag_preview.py, whose ghost item and helpers
this feature reuses:

- geometry: the ghost equals the unit a click at the same pixel then places,
  snapped, free, and for a 4x4 building clamped at the map edge;
- the sprite tier, against a synthetic install (test_view_layers_viewer.py's
  sprite_install pattern), since conftest hides the real one;
- the scene item's lifecycle: hover creates it, leave/tool/mode/wall clear it;
- refresh with no mouse move: catalog pick, owner, checkbox, held Alt;
- the design intent: hovering builds no unit model, adds no unit and
  invalidates no cache.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PyQt5.QtCore import QEvent, QPointF, QRectF, Qt

from descape import asset_source, render, unit_sprites

import conftest
from test_unit_sprites import CONST as SPRITE_CONST
from test_unit_sprites import FILE_NAME, build_sld

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"

_UNIT_CONST = 83  # a 1x1 unit coloured by its owner
_BUILDING_CONST = 33  # 4x4, for the map-edge clamp


def _window(style: str = "Flat", const: int | None = _UNIT_CONST):
    conftest.ensure_qapp()
    from PyQt5.QtWidgets import QApplication

    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.load_scenario(FIXTURE_PATH)
    assert window.scenario is not None, "fixture failed to load"
    window.iso_action.setChecked(False)
    window.terrain_style_combo.setCurrentText(style)
    window.mode_combo.setCurrentText("Units")
    window.show()
    QApplication.processEvents()
    if const is not None:
        window.units_panel.select_object(const)
        assert window.units_panel.selected_object_const() == const
    window.units_panel.select_owner(1)
    window.place_unit_action.setChecked(True)
    assert window.map_view._tool == "place_unit"
    return window


def _close(window) -> None:
    conftest.close_window(window)


def _scene_pos(window, tile_x: float, tile_y: float) -> QPointF:
    """Scene point at a (possibly fractional) Flat tile coordinate."""
    tp = window.map_view._tile_pixels
    return QPointF(tile_x * tp, tile_y * tp)


def _hover_scene(window, scene: QPointF, modifiers=Qt.NoModifier) -> QPointF:
    """A real mouse move to `scene`; returns the scene pos MapView resolved."""
    from PyQt5.QtGui import QMouseEvent

    view = window.map_view
    viewport = QPointF(view.mapFromScene(scene))
    view.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, viewport, Qt.NoButton, Qt.NoButton, modifiers))
    return view.mapToScene(viewport.toPoint())


def _hover_tile(window, tile_x: int, tile_y: int) -> QPointF:
    view = window.map_view
    viewport = conftest.viewport_pos(view, tile_x, tile_y)
    view.mouseMoveEvent(conftest.mouse_event(QEvent.MouseMove, viewport, Qt.NoButton, Qt.NoButton))
    return view.mapToScene(viewport.toPoint())


def _mark_polygons(item) -> list[list[tuple[float, float]]]:
    return [[(p.x(), p.y()) for p in polygon] for polygon, _brush, _pen in item._marks]


def _mark_colors(item) -> set[tuple[int, int, int]]:
    return {brush.color().getRgb()[:3] for _polygon, brush, _pen in item._marks}


def _place_at(window, scene: QPointF, modifiers=Qt.NoModifier):
    """Commits a placement at `scene` the way a click does; returns the entry."""
    window.on_unit_place(scene, modifiers)
    return window.map_view._unit_index.entry_for_key(window._selection[0])


def _expected_mark(window, entry):
    view = window.map_view
    polygons = [[(float(x), float(y)) for x, y in pts] for pts in view._unit_polygons_for(entry)]
    color = render.unit_mark_color(entry.unit, window.scenario.player_colors[entry.player_id])
    return polygons, color


# --- Geometry: the ghost is the unit a click places ------------------------


def test_a_snapped_ghost_is_exactly_the_unit_the_click_places() -> None:
    window = _window()
    try:
        scene = _hover_scene(window, _scene_pos(window, 40.3, 40.7))
        item = window.map_view._unit_ghost_item
        assert item is not None
        ghost_polygons, ghost_colors = _mark_polygons(item), _mark_colors(item)

        entry = _place_at(window, scene)
        assert (entry.unit.x, entry.unit.y) == (40.5, 40.5)
        polygons, color = _expected_mark(window, entry)
        assert ghost_polygons == polygons
        assert ghost_colors == {color}
    finally:
        _close(window)


def test_a_free_ghost_is_exactly_the_unit_the_click_places() -> None:
    window = _window()
    try:
        window.free_place_check.setChecked(True)
        scene = _hover_scene(window, _scene_pos(window, 40.3, 40.7))
        item = window.map_view._unit_ghost_item
        assert item is not None
        ghost_polygons = _mark_polygons(item)

        entry = _place_at(window, scene)
        assert (entry.unit.x, entry.unit.y) != (40.5, 40.5), "free placement must not snap"
        assert ghost_polygons == _expected_mark(window, entry)[0]
    finally:
        _close(window)


def test_a_building_ghost_clamps_at_the_map_edge_like_the_placed_building() -> None:
    window = _window(const=_BUILDING_CONST)
    try:
        w = window.scenario.map_manager.map_width
        h = window.scenario.map_manager.map_height
        scene = _hover_scene(window, _scene_pos(window, w - 0.5, h - 0.5))
        item = window.map_view._unit_ghost_item
        assert item is not None
        ghost_polygons = _mark_polygons(item)
        tp = window.map_view._tile_pixels
        assert item.sceneBoundingRect().right() == w * tp, "the 4x4 footprint is clamped to the edge"

        entry = _place_at(window, scene)
        assert entry.unit.unit_const == _BUILDING_CONST
        assert ghost_polygons == _expected_mark(window, entry)[0]
    finally:
        _close(window)


# --- Sprite tier, against a synthetic install --------------------------------


@pytest.fixture
def sprite_install(tmp_path, monkeypatch):
    """test_view_layers_viewer.py's fixture of the same name: one small
    synthetic sprite, so Show sprites resolves real art in-session.

    Plus one trap of its own: every hover asks viewer._wall_consts(), which is
    functools.cache'd. First reached under the patched graphic_map it caches
    an empty set, and later wall tests in the same worker fail. Warmed first."""
    from descape import viewer

    viewer._wall_consts.cache_clear()
    assert viewer._wall_consts(), "the real wall set must be cached before the patch"
    graphics = tmp_path / unit_sprites.GRAPHICS_SUBPATH
    graphics.mkdir(parents=True)
    (graphics / f"{FILE_NAME}.sld").write_bytes(build_sld(4))
    monkeypatch.setattr(
        unit_sprites, "graphic_map",
        lambda: {SPRITE_CONST: {"graphic_id": 1, "file_name": FILE_NAME, "angle_count": 4,
                                "mirroring_mode": 6, "frame_count": 1}},
    )
    asset_source.set_install_path_override(tmp_path)
    yield tmp_path
    asset_source.set_install_path_override(None)


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_an_elevated_ghost_draws_the_placed_units_own_sprite(sprite_install, style) -> None:
    window = _window(style, const=None)
    try:
        window.show_sprites_action.setChecked(True)
        assert window._sprites_enabled
        window.units_panel._set_pending_object(SPRITE_CONST)
        view = window.map_view
        assert view._terrain_style == style.lower()
        scene = _hover_tile(window, 40, 40)
        item = view._unit_ghost_item
        assert item is not None
        assert item._images, "the sprite must resolve, or this test proves nothing"
        assert not item._marks
        got = [(x, y, arr.shape, arr.tobytes()) for (_img, x, y), arr in zip(item._images, item._arrays, strict=True)]

        entry = _place_at(window, scene)
        cache = view._sloped_cache()
        draws = render.unit_sprite_draws_at(
            window.scenario,
            view._iso_proj,
            view._iso_elevations,
            None if cache is None else cache.corner_rise,
            entry.player_id,
            entry.unit,
            tree_scale=window._layers.tree_scale,
            hero_glow=window._layers.hero_glow,
        )
        want = []
        for draw, ax, ay in draws:
            rgba = np.ascontiguousarray(draw.rgba)
            want.append((float(ax - draw.hotspot_x), float(ay - draw.hotspot_y), rgba.shape, rgba.tobytes()))
        assert got == want
    finally:
        _close(window)


# --- The scene item's lifecycle ----------------------------------------------


def test_hovering_creates_the_ghost_and_clears_the_hover_outline() -> None:
    window = _window()
    try:
        view = window.map_view
        entry = view._unit_index.entries[0]
        _hover_tile(window, int(entry.unit.x), int(entry.unit.y))
        assert view._unit_ghost_item is not None
        assert view._unit_hover_item is None, "a place click never selects, so no cyan outline"
    finally:
        _close(window)


def test_leaving_the_view_clears_the_ghost() -> None:
    window = _window()
    try:
        view = window.map_view
        _hover_tile(window, 40, 40)
        assert view._unit_ghost_item is not None
        view.leaveEvent(QEvent(QEvent.Leave))
        assert view._unit_ghost_item is None
        view.refresh_place_ghost()
        assert view._unit_ghost_item is None, "a refresh after leaving must not resurrect it"
    finally:
        _close(window)


def test_switching_tool_clears_the_ghost() -> None:
    window = _window()
    try:
        view = window.map_view
        _hover_tile(window, 40, 40)
        assert view._unit_ghost_item is not None
        window.pan_action.setChecked(True)
        assert view._tool == "pan"
        assert view._unit_ghost_item is None
    finally:
        _close(window)


def test_switching_mode_clears_the_ghost() -> None:
    window = _window()
    try:
        view = window.map_view
        _hover_tile(window, 40, 40)
        assert view._unit_ghost_item is not None
        window.mode_combo.setCurrentText("View")
        assert view._unit_ghost_item is None
    finally:
        _close(window)


def test_no_object_picked_shows_no_ghost() -> None:
    window = _window(const=None)
    try:
        assert window.units_panel.selected_object_const() is None
        _hover_tile(window, 40, 40)
        assert window.map_view._unit_ghost_item is None
    finally:
        _close(window)


def _a_wall_const() -> int:
    from descape.viewer import _wall_consts

    return min(_wall_consts())


def test_a_wall_pick_shows_no_ghost() -> None:
    """A wall makes Place Unit a shape drag, whose branch in mouseMoveEvent
    runs first and keeps today's brush cue."""
    window = _window(const=None)
    try:
        window.units_panel._set_pending_object(_a_wall_const())
        _hover_tile(window, 40, 40)
        assert window.map_view._unit_ghost_item is None
    finally:
        _close(window)


def test_picking_a_wall_under_a_live_ghost_clears_it() -> None:
    """The shape-drag branch never touches the unit ghost, so the pick's own
    refresh is what removes a ghost left from the previous object."""
    window = _window()
    try:
        view = window.map_view
        _hover_tile(window, 40, 40)
        assert view._unit_ghost_item is not None
        window.units_panel._set_pending_object(_a_wall_const())
        assert view._unit_ghost_item is None
        _hover_tile(window, 41, 41)
        assert view._unit_ghost_item is None
    finally:
        _close(window)


def test_leaving_and_reentering_the_same_tile_shows_the_ghost_again() -> None:
    """The memo returns early on an unchanged key, so it must notice that
    MapView dropped the item, or the ghost never comes back on that tile."""
    window = _window()
    try:
        view = window.map_view
        _hover_tile(window, 40, 40)
        assert view._unit_ghost_item is not None
        view.leaveEvent(QEvent(QEvent.Leave))
        assert view._unit_ghost_item is None
        _hover_tile(window, 40, 40)
        assert view._unit_ghost_item is not None
    finally:
        _close(window)


@pytest.mark.parametrize("on_unit", [True, False], ids=["unit_drag", "marquee"])
def test_a_keybound_switch_mid_drag_keeps_the_drag_or_marquee(on_unit) -> None:
    """Place Unit can be bound to a key, so the tool can change under a held
    button; the next move must stay with the drag or marquee it began."""
    window = _window()
    try:
        view = window.map_view
        window.pan_action.setChecked(True)
        if on_unit:
            entry = view._unit_index.entries[0]
            tile = (int(entry.unit.x), int(entry.unit.y))
        else:
            anchors = [(e.unit.x, e.unit.y) for e in view._unit_index.entries]
            tile = next(
                (x, y) for x in range(10, 110) for y in range(10, 110)
                if all(abs(x - ux) > 4 or abs(y - uy) > 4 for ux, uy in anchors)
            )
        press = conftest.viewport_pos(view, *tile)
        view.mousePressEvent(conftest.mouse_event(QEvent.MouseButtonPress, press, Qt.LeftButton, Qt.LeftButton))
        assert (view._unit_drag_key is not None) is on_unit
        assert (view._marquee_start_pos is not None) is not on_unit
        window.place_unit_action.setChecked(True)
        move = conftest.viewport_pos(view, tile[0] + 3, tile[1] + 3)
        view.mouseMoveEvent(conftest.mouse_event(QEvent.MouseMove, move, Qt.NoButton, Qt.LeftButton))
        assert view._place_hover is None
        if not on_unit:
            assert view._unit_ghost_item is None
    finally:
        _close(window)


def test_an_elevation_edit_under_a_still_cursor_drops_the_memo() -> None:
    """The memo keys on id(_iso_elevations), which an in-place elevation edit
    or undo keeps, so _apply_dirty has to drop it."""
    window = _window()
    try:
        _hover_tile(window, 40, 40)
        assert window._place_ghost_memo is not None
        window._apply_dirty([40 * window.scenario.map_manager.map_width + 40])
        assert window._place_ghost_memo is None
    finally:
        _close(window)


def test_a_place_click_leaves_the_ghost_for_the_next_placement() -> None:
    window = _window()
    try:
        view = window.map_view
        scene = _hover_tile(window, 40, 40)
        _place_at(window, scene)
        assert view._unit_ghost_item is not None
    finally:
        _close(window)


# --- Refresh with no mouse move ------------------------------------------------


def test_a_catalog_pick_updates_the_ghost_without_a_move() -> None:
    window = _window()
    try:
        view = window.map_view
        _hover_tile(window, 40, 40)
        before = view._unit_ghost_item.sceneBoundingRect()
        window.units_panel.select_object(_BUILDING_CONST)
        assert view._unit_ghost_item is not None
        after = view._unit_ghost_item.sceneBoundingRect()
        tp = view._tile_pixels
        assert before == QRectF(40 * tp, 40 * tp, tp, tp)
        assert after.width() == 4 * tp and after.height() == 4 * tp
    finally:
        _close(window)


def test_an_owner_change_updates_the_ghost_colour_without_a_move() -> None:
    window = _window()
    try:
        view = window.map_view
        colors = window.scenario.player_colors
        assert colors[1] != colors[2]
        _hover_tile(window, 40, 40)
        assert _mark_colors(view._unit_ghost_item) == {tuple(colors[1])}
        window.units_panel.select_owner(2)
        assert _mark_colors(view._unit_ghost_item) == {tuple(colors[2])}
    finally:
        _close(window)


def test_the_free_placement_box_updates_the_ghost_without_a_move() -> None:
    window = _window()
    try:
        view = window.map_view
        _hover_scene(window, _scene_pos(window, 40.3, 40.7))
        snapped = view._unit_ghost_item.sceneBoundingRect()
        window.free_place_check.setChecked(True)
        free = view._unit_ghost_item.sceneBoundingRect()
        assert free != snapped
        window.free_place_check.setChecked(False)
        assert view._unit_ghost_item.sceneBoundingRect() == snapped
    finally:
        _close(window)


def test_held_alt_frees_the_ghost_and_its_release_snaps_it_back() -> None:
    from PyQt5.QtGui import QKeyEvent

    window = _window()
    try:
        view = window.map_view
        _hover_scene(window, _scene_pos(window, 40.3, 40.7))
        snapped = view._unit_ghost_item.sceneBoundingRect()
        view.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Alt, Qt.AltModifier))
        assert view._unit_ghost_item.sceneBoundingRect() != snapped
        view.keyReleaseEvent(QKeyEvent(QEvent.KeyRelease, Qt.Key_Alt, Qt.NoModifier))
        assert view._unit_ghost_item.sceneBoundingRect() == snapped
    finally:
        _close(window)


# --- The design intent, pinned -------------------------------------------------


def test_hovering_builds_no_unit_model_adds_no_unit_and_invalidates_no_cache() -> None:
    """Hovering must never pay for the lazy unit-model build (a multi-hundred
    ms scan on a large file) nor mutate anything: the ghost is an overlay."""
    window = _window()
    try:
        assert window.unit_edits is None, "the fixture must start with the model unbuilt"
        counts = [len(units) for units in window.scenario.unit_manager.units]
        calls = {"units": 0, "region": 0}
        cache = window._cache
        real_units, real_region = cache.invalidate_units, cache.invalidate_region

        def counted_units(*a, **k):
            calls["units"] += 1
            return real_units(*a, **k)

        def counted_region(*a, **k):
            calls["region"] += 1
            return real_region(*a, **k)

        cache.invalidate_units = counted_units
        cache.invalidate_region = counted_region
        for step in range(8):
            _hover_scene(window, _scene_pos(window, 30.2 + step * 1.3, 30.6 + step))
        window.units_panel.select_owner(3)
        window.free_place_check.setChecked(True)
        assert window.map_view._unit_ghost_item is not None

        assert window.unit_edits is None
        assert [len(units) for units in window.scenario.unit_manager.units] == counts
        assert calls == {"units": 0, "region": 0}
    finally:
        _close(window)
