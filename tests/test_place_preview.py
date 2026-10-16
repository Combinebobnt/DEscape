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


def test_an_off_grid_ghost_is_exactly_the_unit_the_click_places() -> None:
    window = _window()
    try:
        window.off_grid_check.setChecked(True)
        scene = _hover_scene(window, _scene_pos(window, 40.3, 40.7))
        item = window.map_view._unit_ghost_item
        assert item is not None
        ghost_polygons = _mark_polygons(item)

        entry = _place_at(window, scene)
        assert (entry.unit.x, entry.unit.y) != (40.5, 40.5), "off-grid placement must not snap"
        assert ghost_polygons == _expected_mark(window, entry)[0]
    finally:
        _close(window)


_HOUSE_CONST = 70  # 2x2: the snapped anchor takes the whole number nearest the cursor


@pytest.mark.parametrize(
    ("off_grid", "modifiers"),
    [(False, Qt.NoModifier), (True, Qt.NoModifier), (False, Qt.AltModifier)],
    ids=["snapped", "off_grid", "alt"],
)
def test_a_house_ghost_is_exactly_the_house_the_click_places(off_grid, modifiers) -> None:
    """GH #206: the ghost and the click resolve one point for a building too."""
    window = _window(const=_HOUSE_CONST)
    try:
        window.off_grid_check.setChecked(off_grid)
        scene = _hover_scene(window, _scene_pos(window, 40.3, 40.7), modifiers)
        item = window.map_view._unit_ghost_item
        assert item is not None
        ghost_polygons = _mark_polygons(item)

        entry = _place_at(window, scene, modifiers)
        tp = window.map_view._tile_pixels
        if off_grid or modifiers != Qt.NoModifier:
            # The resolved scene pixel, which the zoomed-out viewport quantizes coarsely.
            assert abs(entry.unit.x - scene.x() / tp) <= 1 / tp and abs(entry.unit.y - scene.y() / tp) <= 1 / tp
            assert (entry.unit.x, entry.unit.y) != (40.0, 41.0), "off-grid placement must not snap"
        else:
            assert (entry.unit.x, entry.unit.y) == (40.0, 41.0)
        assert ghost_polygons == _expected_mark(window, entry)[0]
    finally:
        _close(window)


def test_a_building_ghost_clamps_at_the_map_edge_like_the_placed_building() -> None:
    window = _window(const=_BUILDING_CONST)
    try:
        w = window.scenario.map_manager.map_width
        h = window.scenario.map_manager.map_height
        scene = _hover_scene(window, _scene_pos(window, w - 0.2, h - 0.2))
        item = window.map_view._unit_ghost_item
        assert item is not None
        ghost_polygons = _mark_polygons(item)
        tp = window.map_view._tile_pixels
        assert item.sceneBoundingRect().right() == w * tp, "the 4x4 footprint is clamped to the edge"

        entry = _place_at(window, scene)
        assert entry.unit.unit_const == _BUILDING_CONST
        # The snapped 4x4 anchors on a whole number clamped to size - span/2 (GH #206).
        assert (entry.unit.x, entry.unit.y) == (w - 2, h - 2)
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
    button; the next move must stay with the drag or marquee it began. A
    drag starts under Move, a marquee under Select (TASK-252)."""
    window = _window()
    try:
        view = window.map_view
        (window.move_action if on_unit else window.select_action).setChecked(True)
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


def test_the_off_grid_box_updates_the_ghost_without_a_move() -> None:
    window = _window()
    try:
        view = window.map_view
        _hover_scene(window, _scene_pos(window, 40.3, 40.7))
        snapped = view._unit_ghost_item.sceneBoundingRect()
        window.off_grid_check.setChecked(True)
        free = view._unit_ghost_item.sceneBoundingRect()
        assert free != snapped
        window.off_grid_check.setChecked(False)
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
        window.off_grid_check.setChecked(True)
        assert window.map_view._unit_ghost_item is not None

        assert window.unit_edits is None
        assert [len(units) for units in window.scenario.unit_manager.units] == counts
        assert calls == {"units": 0, "region": 0}
    finally:
        _close(window)


# --- GH #172: turning the ghost before placing ----------------------------------

_OAK = 349  # cyclable, 42 variants
_STONE_GATE = 64
_HOUSE = 33  # INERT: angle_count 1
_FEMALE_VILLAGER = 293  # _UNIT_CONST's sibling under Villager (Random), GH #164


@pytest.fixture
def rotating_install(tmp_path, monkeypatch):
    """A synthetic install where a turn is visible: the villager stores one
    differently coloured frame per facing, the oak one per variant, and each
    stone gate orientation its own colour. Warms _wall_consts first, for the
    reason sprite_install records."""
    from descape import viewer
    from descape.gate_orientation import orientation_siblings

    viewer._wall_consts.cache_clear()
    assert viewer._wall_consts(), "the real wall set must be cached before the patch"
    graphics = tmp_path / unit_sprites.GRAPHICS_SUBPATH
    graphics.mkdir(parents=True)
    table: dict[int, dict] = {}

    def add(const, frames, colour_base=0, **extra):
        name = f"t_rotate_{const}_x1"
        # No team-colour mask: it would paint over the per-frame colour that tells frames apart.
        art = build_sld(frames, canvas=16, colour_base=colour_base, playercolor=False)
        (graphics / f"{name}.sld").write_bytes(art)
        table[const] = {"graphic_id": const, "file_name": name, "angle_count": frames,
                        "mirroring_mode": 0, "frame_count": 1, **extra}

    add(_UNIT_CONST, 16)
    add(_FEMALE_VILLAGER, 16, colour_base=11)
    add(_OAK, 42, colour_base=3, rotation_is_variant=True)
    for i, const in enumerate(orientation_siblings(_STONE_GATE)):
        add(const, 1, colour_base=5 * i + 1)
    monkeypatch.setattr(unit_sprites, "graphic_map", lambda: table)
    asset_source.set_install_path_override(tmp_path)
    unit_sprites.clear_caches()
    yield tmp_path
    asset_source.set_install_path_override(None)
    unit_sprites.clear_caches()


def _sprite_window(const: int, owner: int = 1):
    window = _window("Stepped", const=None)
    window.show_sprites_action.setChecked(True)
    assert window._sprites_enabled
    window.units_panel.select_owner(owner)
    window.units_panel._set_pending_object(const)
    return window


def _ghost_draws(view) -> list:
    item = view._unit_ghost_item
    assert item is not None
    assert item._images and not item._marks, "the sprite must resolve, or this proves nothing"
    return [(x, y, arr.shape, arr.tobytes()) for (_img, x, y), arr in zip(item._images, item._arrays, strict=True)]


def _placed_draws(window, entry) -> list:
    view = window.map_view
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
    out = []
    for draw, ax, ay in draws:
        rgba = np.ascontiguousarray(draw.rgba)
        out.append((float(ax - draw.hotspot_x), float(ay - draw.hotspot_y), rgba.shape, rgba.tobytes()))
    return out


def _last_status(window) -> str:
    return window.status_log.toPlainText().splitlines()[-1]


def test_a_turned_ghost_is_the_unit_the_click_places_and_the_turn_persists(rotating_install) -> None:
    from descape import place_rotation

    window = _sprite_window(_UNIT_CONST)
    try:
        view = window.map_view
        scene = _hover_tile(window, 40, 40)
        before = _ghost_draws(view)
        window.rotate_place_ghost(steps=6)
        ghost = _ghost_draws(view)
        assert ghost != before, "the turn must change the ghost"
        assert "facing 6/16" in _last_status(window)

        entry = _place_at(window, scene)
        want = place_rotation.rotation_for(window._place_rotation, _UNIT_CONST, 1)
        assert entry.unit.rotation == want == pytest.approx(6 * 2 * 3.141592653589793 / 16)
        assert ghost == _placed_draws(window, entry)
        assert "facing 6/16" in _last_status(window)

        scene = _hover_tile(window, 44, 44)
        second = _place_at(window, scene)
        assert second.unit.rotation == want, "the facing carries into the next placement"
    finally:
        _close(window)


def _ghost_const(window) -> int:
    """The const Place Unit's current ghost was built with (its memo key)."""
    memo = window._place_ghost_memo
    assert memo is not None and memo[1] is window.map_view._unit_ghost_item
    return memo[0][1]


def test_villager_random_ghost_is_each_clicks_villager_and_the_facing_survives_a_reroll(rotating_install) -> None:
    """GH #164: the panel pre-rolls, so the ghost before each click is the
    villager it places, and a re-roll after the click shows at once."""
    import random

    from descape.place_random import RANDOM_VILLAGER

    window = _sprite_window(_UNIT_CONST)
    try:
        panel = window.units_panel
        panel._rng = random.Random(164)
        panel.catalog_view.select(RANDOM_VILLAGER)
        view = window.map_view
        _hover_tile(window, 30, 40)
        window.rotate_place_ghost(steps=6)
        facing = 6 * 2 * 3.141592653589793 / 16
        placed, ghosts = [], {}
        for i in range(10):
            scene = _hover_tile(window, 30 + 2 * i, 40)
            const, ghost = _ghost_const(window), _ghost_draws(view)
            entry = _place_at(window, scene)
            assert entry.unit.unit_const == const
            assert ghost == _placed_draws(window, entry)
            assert entry.unit.rotation == pytest.approx(facing), "the facing carries across a re-roll"
            # Re-rolled with no mouse move: the ghost already shows the next pick.
            assert _ghost_const(window) == panel.selected_object_const()
            placed.append(entry.unit.unit_const)
            ghosts[const] = ghost
        assert set(placed) == {_UNIT_CONST, _FEMALE_VILLAGER}
        assert ghosts[_UNIT_CONST] != ghosts[_FEMALE_VILLAGER], "the two sexes must look different here"
        assert panel.catalog_view.current_value() == RANDOM_VILLAGER
    finally:
        _close(window)


def test_a_cycled_tree_places_the_ghosts_variant_as_an_integer(rotating_install) -> None:
    window = _sprite_window(_OAK, owner=0)
    try:
        view = window.map_view
        scene = _hover_tile(window, 40, 40)
        before = _ghost_draws(view)
        window.rotate_place_ghost(steps=5)
        ghost = _ghost_draws(view)
        assert ghost != before
        assert "variant 5/42" in _last_status(window)

        entry = _place_at(window, scene)
        assert entry.unit.rotation == 5.0 and entry.unit.rotation == int(entry.unit.rotation)
        assert ghost == _placed_draws(window, entry)
    finally:
        _close(window)


def test_a_gate_turn_changes_the_catalog_pick_and_the_placed_const(rotating_install) -> None:
    from descape import gate_orientation

    window = _sprite_window(_STONE_GATE)
    try:
        view = window.map_view
        scene = _hover_tile(window, 40, 40)
        before = _ghost_draws(view)
        window.rotate_place_ghost(steps=1)
        sibling = gate_orientation.cycle_const(_STONE_GATE, 1)
        assert window.units_panel.selected_object_const() == sibling
        ghost = _ghost_draws(view)
        assert ghost != before

        entry = _place_at(window, scene)
        assert entry.unit.unit_const == sibling
        assert entry.unit.rotation == 0.0
        assert ghost == _placed_draws(window, entry)
    finally:
        _close(window)


def test_a_gate_turn_clears_a_catalog_filter_hiding_its_sibling() -> None:
    from descape import gate_orientation

    window = _window(const=_STONE_GATE)
    try:
        _hover_tile(window, 40, 40)
        catalog = window.units_panel.catalog_view
        catalog.filter_edit.setText(window.units_panel.placing_label.toolTip())
        assert window.units_panel.selected_object_const() == _STONE_GATE
        window.rotate_place_ghost(steps=1)
        sibling = gate_orientation.cycle_const(_STONE_GATE, 1)
        assert window.units_panel.selected_object_const() == sibling
        assert not catalog.filter_edit.text()
        assert "cleared the catalog filter" in _last_status(window)
    finally:
        _close(window)


def test_an_inert_pick_does_not_turn_and_says_why() -> None:
    window = _window(const=_HOUSE)
    try:
        scene = _hover_tile(window, 40, 40)
        window.rotate_place_ghost(steps=1)
        assert "has no facing or variant to rotate" in _last_status(window)
        assert window._place_rotation == type(window._place_rotation)()
        assert _place_at(window, scene).unit.rotation == 0.0
    finally:
        _close(window)


def test_gaia_angle_turns_and_draws_its_facing_and_a_gaia_tree_still_cycles(rotating_install) -> None:
    """GH #217: a GAIA creatable turns like a player's, and the placed unit draws the turned ghost."""
    window = _sprite_window(_UNIT_CONST, owner=0)
    try:
        view = window.map_view
        scene = _hover_tile(window, 40, 40)
        before = _ghost_draws(view)
        window.rotate_place_ghost(steps=3)
        ghost = _ghost_draws(view)
        assert ghost != before, "the turn must change the GAIA ghost"
        assert "facing 3/16" in _last_status(window)
        entry = _place_at(window, scene)
        assert entry.unit.rotation == pytest.approx(3 * 2 * 3.141592653589793 / 16)
        assert ghost == _placed_draws(window, entry)

        window.units_panel._set_pending_object(_OAK)
        scene = _hover_tile(window, 44, 44)
        window.rotate_place_ghost(steps=2)
        assert "variant 2/42" in _last_status(window)
        assert _place_at(window, scene).unit.rotation == 2.0
    finally:
        _close(window)


def test_loading_a_scenario_resets_the_pending_rotation() -> None:
    window = _window()
    try:
        _hover_tile(window, 40, 40)
        window.rotate_place_ghost(steps=3)
        assert window._place_rotation.facing != 0.0
        window.load_scenario(FIXTURE_PATH)
        assert window._place_rotation == type(window._place_rotation)()
    finally:
        _close(window)


# --- GH #172: the Rotate actions route to the ghost --------------------------------


def test_a_rotate_key_over_the_ghost_turns_it_not_the_just_placed_unit() -> None:
    window = _window()
    try:
        view = window.map_view
        scene = _hover_tile(window, 40, 40)
        entry = _place_at(window, scene)
        assert window._selection, "the placed unit is selected"
        records = len(window.edit_history.records)
        assert view._unit_ghost_item is not None
        assert window.rotate_cw_action.isEnabled()

        window.rotate_cw_action.trigger()

        assert entry.unit.rotation == 0.0
        assert len(window.edit_history.records) == records, "no undo record"
        assert "facing 1/16" in _last_status(window)
        window.rotate_cw_coarse_action.trigger()
        assert "facing 5/16" in _last_status(window)
        assert entry.unit.rotation == 0.0
    finally:
        _close(window)


def test_a_rotate_key_off_the_map_rotates_the_selection_as_before() -> None:
    from descape import unit_rotation

    window = _window()
    try:
        view = window.map_view
        entry = _place_at(window, _hover_tile(window, 40, 40))
        view.leaveEvent(QEvent(QEvent.Leave))
        assert view._unit_ghost_item is None
        records = len(window.edit_history.records)

        window.rotate_cw_action.trigger()

        assert entry.unit.rotation == pytest.approx(unit_rotation.rotate_step(0.0, 16, 1))
        assert len(window.edit_history.records) == records + 1
        assert window._place_rotation == type(window._place_rotation)()
    finally:
        _close(window)


def test_the_rotate_actions_are_enabled_with_place_unit_armed_and_no_selection() -> None:
    window = _window()
    try:
        assert not window._selection
        actions = window._rotate_actions + window._rotate_toolbar_actions
        assert all(a.isEnabled() for a in actions)
        assert not any(a.isEnabled() for a in window._variant_actions), "Cycle Variant is out of scope"

        window.rotate_cw_action.trigger()  # cursor never entered the map
        assert "nothing to rotate" in _last_status(window)

        window.units_panel._set_pending_object(_a_wall_const())
        assert not any(a.isEnabled() for a in actions), "a wall pick has no ghost to turn"
        window.units_panel._set_pending_object(_UNIT_CONST)
        assert all(a.isEnabled() for a in actions)
        window.pan_action.setChecked(True)
        assert not any(a.isEnabled() for a in actions)
    finally:
        _close(window)


def test_a_coarse_rotate_over_the_ghost_turns_a_quarter_of_the_picks_own_count() -> None:
    """The coarse step comes from the pick, not from a selection: an oak's
    quarter is 11 of its 42 variants."""
    window = _window(const=_OAK)
    try:
        _hover_tile(window, 40, 40)
        window.rotate_cw_coarse_action.trigger()
        assert "variant 11/42" in _last_status(window)
    finally:
        _close(window)


# --- TASK-288: a GAIA-only pick ghosts for GAIA ------------------------------

_BERRY_BUSH = 59  # FORAG, GAIA-only whatever the owner combo says


@pytest.fixture
def bush_install(tmp_path, monkeypatch):
    """The Forage Bush on a full team-colour mask, so the ghost's pixels are
    its owner's colour. Warms _wall_consts first, for sprite_install's reason."""
    from descape import viewer

    viewer._wall_consts.cache_clear()
    assert viewer._wall_consts(), "the real wall set must be cached before the patch"
    graphics = tmp_path / unit_sprites.GRAPHICS_SUBPATH
    graphics.mkdir(parents=True)
    (graphics / f"{FILE_NAME}.sld").write_bytes(build_sld(1, canvas=16))
    table = {_BERRY_BUSH: {"graphic_id": 1, "file_name": FILE_NAME, "angle_count": 1,
                           "mirroring_mode": 0, "frame_count": 1}}
    monkeypatch.setattr(unit_sprites, "graphic_map", lambda: table)
    asset_source.set_install_path_override(tmp_path)
    unit_sprites.clear_caches()
    yield tmp_path
    asset_source.set_install_path_override(None)
    unit_sprites.clear_caches()


def test_a_gaia_only_ghost_is_drawn_for_gaia_whatever_the_owner_combo(bush_install) -> None:
    import dataclasses

    from descape.unit_filter import GAIA_PLAYER_ID

    window = _sprite_window(_BERRY_BUSH, owner=1)
    try:
        colors = window.scenario.player_colors
        assert colors[GAIA_PLAYER_ID] != colors[1]
        scene = _hover_tile(window, 40, 40)
        ghost = _ghost_draws(window.map_view)

        entry = _place_at(window, scene)
        assert entry.player_id == GAIA_PLAYER_ID
        assert ghost == _placed_draws(window, entry)
        as_player_1 = _placed_draws(window, dataclasses.replace(entry, player_id=1))
        assert ghost != as_player_1, "the team-colour mask must tell GAIA from Player 1"
        assert window.units_panel.owner_id() == 1
    finally:
        _close(window)


# --- The Eyedropper in Units mode: a picked unit is the next placement -------

_ARCHER = 4
_ARCHER_KEY = (2, 300)  # Player 2's archer on tile (20, 20)
_OAK_KEY = (0, 100)  # GAIA oak on (5, 5), stored variant 7
_WALL_KEY = (0, 102)  # GAIA stone wall on (7, 5): a shape index, radian-encoded
_FIXTURE_HOUSE_KEY = (1, 200)  # Player 1's house on (10, 10): one frame, nothing to turn
_OFF_GRID_FACING = 2.4  # between frames 6 and 7 of the archer's 16


def _eyedrop_window(const: int = _HOUSE):
    """_window() with the Eyedropper active: `const` pending, owner Player 1."""
    window = _window(const=const)
    window.eyedropper_action.setChecked(True)
    assert window.map_view._tool == "eyedropper"
    return window


def _fixture_entry(window, key):
    entry = window.map_view._unit_index.entry_for_key(key)
    assert entry is not None, f"the fixture has no unit {key}"
    return entry


def _stale_rotation(window):
    """A pending rotation no fixture unit would produce, to see it survive a pick."""
    from descape import place_rotation

    window._place_rotation = place_rotation.PlaceRotation(facing=1.25, variant=3, variant_const=_OAK)
    return window._place_rotation


def test_an_eyedropped_unit_is_what_the_next_click_places() -> None:
    import math

    from descape import object_catalog, place_rotation

    window = _eyedrop_window()
    try:
        entry = _fixture_entry(window, _ARCHER_KEY)
        entry.unit.rotation = _OFF_GRID_FACING
        records = len(window.edit_history.records)

        window.pick_unit_value(entry, (20, 20), Qt.NoModifier)

        panel = window.units_panel
        assert panel.selected_object_const() == _ARCHER
        assert panel.owner_id() == 2
        assert window._place_rotation.facing == _OFF_GRID_FACING
        name = object_catalog.display_name(_ARCHER)
        assert f"Picked {name}, Player 2, facing 6/16 from (20, 20)" in _last_status(window)
        assert len(window.edit_history.records) == records, "a pick is not an edit"
        assert entry.unit.rotation == _OFF_GRID_FACING, "a pick reads the unit, it never rewrites it"

        window.place_unit_action.setChecked(True)
        placed = _place_at(window, _hover_tile(window, 40, 40))
        assert (placed.unit.unit_const, placed.player_id) == (_ARCHER, 2)
        assert placed.unit.rotation == place_rotation.rotation_for(window._place_rotation, _ARCHER, 2)
        assert placed.unit.rotation == pytest.approx(6 * 2 * math.pi / 16), "snapped to the archer's own frames"
    finally:
        _close(window)


def test_an_eyedropped_tree_carries_its_variant_into_the_next_placement() -> None:
    window = _eyedrop_window()
    try:
        entry = _fixture_entry(window, _OAK_KEY)
        assert entry.unit.rotation == 7.0

        window.pick_unit_value(entry, (5, 5), Qt.NoModifier)

        assert window.units_panel.selected_object_const() == _OAK
        assert window.units_panel.owner_id() == 0
        state = window._place_rotation
        assert (state.variant, state.variant_const, state.facing) == (7, _OAK, 0.0)
        assert "GAIA, variant 7/42 from (5, 5)" in _last_status(window)

        window.place_unit_action.setChecked(True)
        placed = _place_at(window, _hover_tile(window, 40, 40))
        assert (placed.unit.unit_const, placed.player_id) == (_OAK, 0)
        assert placed.unit.rotation == 7.0 and placed.unit.rotation == int(placed.unit.rotation)
    finally:
        _close(window)


def test_an_eyedropped_gaia_unit_places_its_snapped_facing() -> None:
    """The pick stores the facing; rotation_for() alone decides what a click writes, GAIA included (GH #217)."""
    import math
    from types import SimpleNamespace

    window = _eyedrop_window()
    try:
        gaia_archer = SimpleNamespace(player_id=0, unit=SimpleNamespace(unit_const=_ARCHER, rotation=_OFF_GRID_FACING))

        window.pick_unit_value(gaia_archer, (3, 4), Qt.NoModifier)

        assert window.units_panel.owner_id() == 0
        assert window._place_rotation.facing == _OFF_GRID_FACING
        window.place_unit_action.setChecked(True)
        placed = _place_at(window, _hover_tile(window, 40, 40))
        assert (placed.unit.unit_const, placed.player_id) == (_ARCHER, 0)
        assert placed.unit.rotation == pytest.approx(6 * 2 * math.pi / 16), "snapped to the archer's 16-frame grid"

        window.units_panel.select_owner(1)
        turned = _place_at(window, _hover_tile(window, 44, 44))
        assert turned.unit.rotation == placed.unit.rotation, "the owner does not change the facing"
    finally:
        _close(window)


@pytest.mark.parametrize(
    ("key", "tile"), [(_WALL_KEY, (7, 5)), (_FIXTURE_HOUSE_KEY, (10, 10))], ids=["wall", "single_frame"]
)
def test_an_eyedropped_wall_or_single_frame_object_leaves_the_pending_rotation_alone(key, tile) -> None:
    window = _eyedrop_window(const=_UNIT_CONST)
    try:
        window.units_panel.select_owner(2)
        before = _stale_rotation(window)
        entry = _fixture_entry(window, key)

        window.pick_unit_value(entry, tile, Qt.NoModifier)

        assert window.units_panel.selected_object_const() == entry.unit.unit_const
        assert window.units_panel.owner_id() == entry.player_id
        assert window._place_rotation == before
        status = _last_status(window)
        assert "Picked" in status and "facing" not in status and "variant" not in status
    finally:
        _close(window)


def test_a_shift_eyedrop_copies_the_type_and_nothing_else() -> None:
    from descape import object_catalog

    window = _eyedrop_window()
    try:
        before = _stale_rotation(window)
        entry = _fixture_entry(window, _ARCHER_KEY)
        entry.unit.rotation = _OFF_GRID_FACING

        window.pick_unit_value(entry, (20, 20), Qt.ShiftModifier)

        assert window.units_panel.selected_object_const() == _ARCHER
        assert window.units_panel.owner_id() == 1, "Shift keeps the owner"
        assert window._place_rotation == before, "Shift keeps the pending rotation"
        assert f"Picked {object_catalog.display_name(_ARCHER)} from (20, 20)" in _last_status(window)
    finally:
        _close(window)


def test_eyedropping_a_type_the_catalog_lacks_changes_nothing() -> None:
    from types import SimpleNamespace

    window = _eyedrop_window()
    try:
        before = _stale_rotation(window)
        stranger = SimpleNamespace(player_id=2, unit=SimpleNamespace(unit_const=-12345, rotation=2.0))

        window.pick_unit_value(stranger, (3, 4), Qt.NoModifier)

        assert window.units_panel.selected_object_const() == _HOUSE
        assert window.units_panel.owner_id() == 1
        assert window._place_rotation == before
        assert "from (3, 4) (not in the catalog, unchanged)" in _last_status(window)
    finally:
        _close(window)


def test_the_units_eyedropper_works_on_a_file_whose_terrain_failed_verification() -> None:
    """TASK-356: a Units-mode pick writes UI state only, so the terrain gate
    is not its own; in Terrain mode the tool greys and hands over to Pan."""
    window = _window(const=_HOUSE)
    try:
        assert window.mode == "units"
        window.scenario.terrain_write_supported = False
        window._update_tool_enabled()
        assert window.eyedropper_action.isEnabled()
        window.eyedropper_action.setChecked(True)
        assert window.map_view._tool == "eyedropper"

        view = window.map_view
        pos = conftest.viewport_pos(view, 20, 20)
        view.mousePressEvent(conftest.mouse_event(QEvent.MouseButtonPress, pos, Qt.LeftButton, Qt.LeftButton))
        assert (window.units_panel.selected_object_const(), window.units_panel.owner_id()) == (_ARCHER, 2)

        window.mode_combo.setCurrentText("Terrain")
        assert not window.eyedropper_action.isEnabled()
        assert window._current_tool == "pan"
    finally:
        _close(window)


def test_a_refused_eyedrop_leaves_the_catalog_filter_as_it_was() -> None:
    from types import SimpleNamespace

    window = _eyedrop_window()
    try:
        view = window.units_panel.catalog_view
        view.filter_edit.setText("house")
        stranger = SimpleNamespace(player_id=2, unit=SimpleNamespace(unit_const=-12345, rotation=2.0))

        window.pick_unit_value(stranger, (3, 4), Qt.NoModifier)

        assert "(not in the catalog, unchanged)" in _last_status(window)
        assert view.filter_edit.text() == "house"
        assert not view.show_hidden_checkbox.isChecked()
    finally:
        _close(window)


@pytest.mark.parametrize("modifiers", [Qt.NoModifier, Qt.ShiftModifier], ids=["plain", "shift"])
def test_an_eyedrop_that_reveals_its_row_says_so(modifiers) -> None:
    """TASK-356, as rotate_place_ghost's gate turn does: a pick that cleared the
    filter or ticked the editor-hidden box names what it changed."""
    from types import SimpleNamespace

    from descape import constant_picker, object_catalog

    window = _eyedrop_window()
    try:
        view = window.units_panel.catalog_view
        view.filter_edit.setText("this matches nothing at all")
        window.pick_unit_value(_fixture_entry(window, _ARCHER_KEY), (20, 20), modifiers)
        assert window.units_panel.selected_object_const() == _ARCHER
        assert _last_status(window).endswith("(cleared the catalog filter to show it)")

        builder = SimpleNamespace(player_id=1, unit=SimpleNamespace(unit_const=118, rotation=0.0))
        assert not view.show_hidden_checkbox.isChecked()
        window.pick_unit_value(builder, (3, 4), modifiers)
        assert window.units_panel.selected_object_const() == 118
        assert _last_status(window).endswith(f'(ticked "{constant_picker.HIDDEN_LABEL}" to show it)')

        # Neither changes on a second pick of a row already shown, so no note.
        window.pick_unit_value(_fixture_entry(window, _ARCHER_KEY), (20, 20), modifiers)
        assert "to show it" not in _last_status(window)

        # A class filter that excludes the row is reset by select(), and said so.
        archer_class = object_catalog.entry(object_catalog.objects(), _ARCHER).class_id
        combo = view.class_combo
        other = next(
            i for i in range(combo.count()) if combo.itemData(i) is not None and combo.itemData(i) != archer_class
        )
        combo.setCurrentIndex(other)
        window.pick_unit_value(_fixture_entry(window, _ARCHER_KEY), (20, 20), modifiers)
        assert window.units_panel.selected_object_const() == _ARCHER
        assert combo.currentData() is None
        assert _last_status(window).endswith("(reset the class filter to show it)")
    finally:
        _close(window)


def test_repeat_eyedropper_clicks_walk_down_a_stack() -> None:
    """Real presses through MapView, so this is also the window's wiring."""
    window = _eyedrop_window()
    try:
        view = window.map_view
        tile = (40, 40)
        model = window._ensure_unit_edits()
        with window._unit_edit(model, "Stack", [1, 2]):
            model.add(1, _UNIT_CONST, 40.5, 40.5)
            model.add(2, _ARCHER, 40.5, 40.5)
            model.add(1, _ARCHER, 40.5, 40.5)
        want = [(member.unit.unit_const, member.player_id) for member in view.stack_group_at(tile)]
        assert len(set(want)) == 3, "each member must be tellable from the others"
        units_before = sum(len(units) for units in window.scenario.unit_manager.units)
        pos = conftest.viewport_pos(view, *tile)

        got = []
        for _ in range(4):
            view.mousePressEvent(conftest.mouse_event(QEvent.MouseButtonPress, pos, Qt.LeftButton, Qt.LeftButton))
            got.append((window.units_panel.selected_object_const(), window.units_panel.owner_id()))

        assert got == [*want, want[0]]
        assert view._eyedrop_cycle == (tile, 0)
        assert view._picker_cycle is None, "Pick from map keeps a stack position of its own"
        assert sum(len(units) for units in window.scenario.unit_manager.units) == units_before, "nothing placed"

        # And the other way round: Pick from map starts at the top and leaves the Eyedropper's position.
        picked: list = []
        view.on_unit_pick = picked.append
        scene = view.mapToScene(pos.toPoint())
        view._picker_click(scene)
        view._picker_click(scene)
        keys = [(member.player_id, member.unit.reference_id) for member in view.stack_group_at(tile)]
        assert picked == keys[:2]
        assert view._eyedrop_cycle == (tile, 0)
    finally:
        _close(window)


@pytest.mark.parametrize("reset", ["tool", "mode", "unit_index"])
def test_a_tool_mode_or_index_switch_restarts_the_eyedropper_at_the_top_of_a_stack(reset) -> None:
    """TASK-356: the stack position outlived all three, so the next click picked
    the unit under the one the hover outline shows."""
    window = _eyedrop_window()
    try:
        view = window.map_view
        tile = (40, 40)
        model = window._ensure_unit_edits()
        with window._unit_edit(model, "Stack", [1, 2]):
            model.add(1, _UNIT_CONST, 40.5, 40.5)
            model.add(2, _ARCHER, 40.5, 40.5)
        top = view.stack_group_at(tile)[0]
        pos = conftest.viewport_pos(view, *tile)

        def click():
            view.mousePressEvent(conftest.mouse_event(QEvent.MouseButtonPress, pos, Qt.LeftButton, Qt.LeftButton))
            return window.units_panel.selected_object_const(), window.units_panel.owner_id()

        assert click() == (top.unit.unit_const, top.player_id)
        assert view._eyedrop_cycle == (tile, 0)
        if reset == "tool":
            view.set_tool("pan")
            view.set_tool("eyedropper")
        elif reset == "mode":
            view.set_mode("terrain")
            view.set_mode("units")
        else:
            view.set_unit_index(view._unit_index)
        assert view._eyedrop_cycle is None
        assert click() == (top.unit.unit_const, top.player_id), "the next click starts at the top again"
    finally:
        _close(window)
