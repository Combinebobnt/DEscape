"""A unit edit's repaint sized from the edited units' real sprite extents, not
render._sprite_reach_px() (maintainer plan 2026-09-26 sprite-bbox, Steps 1-3,
as amended on 2026-09-27: the tight bbox covers the VISIBLE level only, and
every other resident level evicts today's reach-padded bbox).

Every oracle here compares the visible level, the one the tight bbox repaints,
against a fresh cache built from the same scenario and View > Layers state: a
chunk the tight bbox should have repaired but left resident shows as a pixel
mismatch. The whole visible level is rendered (so made resident) right before
each checked edit, since an evicted chunk recomposites fresh and would hide a
stale one. The synthetic art is sized inside the real MAX_SPRITE_REACH_*, so
the reach fallback and the non-visible reach eviction are correct by
construction too.

Stepped runs at mip 0 and mip -1, so the level-to-reference conversion runs;
Sloped has only mip 0.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from descape import asset_source, render, render_cache, unit_rotation, unit_sprites
from descape.elevation_tools import set_tile_elevation
from descape.render_cache import REACH_FALLBACK, FlatChunkCache, IsoChunkCache, SlopedChunkCache, UnitSplice
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units

import conftest
from test_unit_sprites import build_sld

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

T = unit_sprites.NATIVE_TILE_W
FOREST_OAK, FOREST_PALM = 10, 13
TREE_OAK, TREE_PALM = 411, 351
CASTLE = 82  # 4x4, drawn as a two-piece composite below
DOCK = 45  # 3x3 with no foundation terrain, given no art: it resolves to its coloured mark
VILLAGER = 83
HERO = 160
STONE_WALL = 117
GATE_NE = 64
TOWER_PIECE = 9001

STROKE_TILES = [(20 + i, 30) for i in range(8)]
SHIFTED_TILES = [(24 + i, 30) for i in range(8)]
STYLE_MIPS = [("Stepped", 0), ("Stepped", -1), ("Sloped", 0)]


@pytest.fixture
def art(tmp_path, monkeypatch):
    """A synthetic install drawing every const these tests edit, each a solid
    square in its own colour per frame, hotspot at the centre. No canvas passes
    6 * NATIVE_TILE_W, so no reach passes 288 native px: inside every real
    MAX_SPRITE_REACH_* (DOWN, 316, is the smallest). The castle's second piece
    carries its own depth slot and sits 300 native px up, so the castle
    spans two anchor tiles and reaches 396 px up."""
    graphics = tmp_path / unit_sprites.GRAPHICS_SUBPATH
    graphics.mkdir(parents=True)
    table: dict[int, dict] = {}

    def add(const, canvas, angles=1, colour_base=0, name=None, playercolor=False):
        name = name or f"t_bbox_{const}_x1"
        art = build_sld(angles, canvas=canvas, colour_base=colour_base, playercolor=playercolor)
        (graphics / f"{name}.sld").write_bytes(art)
        entry = {"graphic_id": const, "file_name": name, "angle_count": angles, "mirroring_mode": 0,
                 "frame_count": 1}
        table[const] = entry
        return entry

    add(TREE_OAK, 4 * T, colour_base=3)
    add(TREE_PALM, 4 * T, colour_base=7)
    # The real .dat angle count, so one Rotate step lands on the next stored frame.
    add(VILLAGER, 2 * T, angles=unit_rotation.angle_count_for(VILLAGER), colour_base=11)
    # A full-coverage team-colour mask, so Convert changes its pixels.
    add(HERO, 2 * T, colour_base=19, playercolor=True)
    # Large, so a neighbour's reshaped frame reaches past the placed wall's own extent.
    # Team-coloured, like the gates, so a Convert of one changes its pixels.
    add(STONE_WALL, 6 * T, angles=5, colour_base=23, playercolor=True)
    for i, const in enumerate((64, 88, 659, 667)):
        add(const, T, colour_base=5 * i + 1, playercolor=True)
    main = add(CASTLE, 6 * T, colour_base=13)
    tower = add(TOWER_PIECE, 2 * T, colour_base=17, name="t_bbox_tower_x1")
    table.pop(TOWER_PIECE)
    main["pieces"] = [
        {"unit_id": CASTLE, **{k: main[k] for k in ("file_name", "angle_count", "frame_count")},
         "dx": 0, "dy": 0, "parent": True},
        {"unit_id": TOWER_PIECE, **{k: tower[k] for k in ("file_name", "angle_count", "frame_count")},
         "dx": 0, "dy": -300, "slot": [0, 0]},
    ]
    monkeypatch.setattr(unit_sprites, "graphic_map", lambda: table)
    asset_source.set_install_path_override(tmp_path)
    unit_sprites.clear_caches()
    yield
    asset_source.set_install_path_override(None)
    unit_sprites.clear_caches()


# --- helpers ---------------------------------------------------------------


def _window(style: str, slope: bool = False):
    """The blank template in `style`, Units mode, sprites on. slope raises a
    cone at (40, 40) so a footprint there spans several elevations."""
    window = conftest.blank_window()
    if slope:
        set_tile_elevation(window.scenario.map_manager, 40, 40, 3)
    window.terrain_style_combo.setCurrentText(style)
    if slope:
        window._render_current()
    window.mode_combo.setCurrentText("Units")
    assert window._cache.sprites_enabled
    assert isinstance(window._cache, IsoChunkCache if style == "Stepped" else SlopedChunkCache)
    return window


def _show_level(window, monkeypatch, mip: int) -> None:
    """Pins the level the viewer treats as on screen, with the whole level in view."""
    cache = window._cache
    _cx0, _cy0, cx1, cy1 = cache.chunk_index_range(mip, 0, 0, *cache.canvas_dims(mip))
    monkeypatch.setattr(window.map_view, "viewport_chunk_target", lambda: (mip, 0, 0, cx1, cy1))


def _level(window, mip: int) -> np.ndarray:
    """The level as the cache serves it; every chunk is resident afterwards."""
    w, h = window._cache.canvas_dims(mip)
    return window._cache.render_rect(0, 0, w, h, mip=mip).copy()


def _fresh(window, mip: int, rect=None) -> np.ndarray:
    """A fresh cache's render of level `mip` (or a level-pixel rect of it)
    from the window's current scenario, filter, layers and grid."""
    cache = window._cache
    common = {"unit_filter": cache.unit_filter, "sprites": True, "layers": cache.layers}
    if isinstance(cache, SlopedChunkCache):
        fresh = SlopedChunkCache(
            window.scenario, cache.elevations.copy(), cache.corner_rise.copy(), cache.proj, cache.tile_px, **common
        )
    else:
        fresh = IsoChunkCache(window.scenario, cache.elevations.copy(), cache.proj, cache.tile_px, **common)
    fresh.set_grid(cache.grid)
    rect = rect or (0, 0, *fresh.canvas_dims(mip))
    return fresh.render_rect(*rect, mip=mip)


def _check(window, mip: int, what: str) -> np.ndarray:
    now = _level(window, mip)
    fresh = _fresh(window, mip)
    if not np.array_equal(now, fresh):
        ys, xs = np.nonzero((now != fresh).any(axis=2))
        pytest.fail(f"{what}: {len(xs)} stale px at level {mip}, in x {xs.min()}..{xs.max()}, y {ys.min()}..{ys.max()}")
    return now


def _contains(outer, inner) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


class _Spy:
    """Records the cache's repaint calls with their `levels`, what
    sprite_extent_before() answered, and each tight bbox beside today's."""

    def __init__(self, window, monkeypatch):
        cache = window._cache
        self.calls: list[tuple[str, tuple, tuple | None]] = []
        self.before: list = []
        self.tight: list[tuple[tuple, tuple]] = []
        # Every reach-padded bbox sized, tight path or not.
        self.reach: list[tuple] = []
        # Where the last tight repaint's own calls start; terrain patches come before it.
        self.mark = 0
        real_patch, real_evict = cache.patch, cache.invalidate_region
        real_before, real_bbox = cache.sprite_extent_before, window._unit_edit_bbox

        def patch(bbox, elevation_changed=None, levels=None, **kwargs):
            self.calls.append(("patch", bbox, None if levels is None else tuple(levels)))
            return real_patch(bbox, elevation_changed, levels, **kwargs)

        def evict(bbox, levels=None, **kwargs):
            self.calls.append(("evict", bbox, None if levels is None else tuple(levels)))
            return real_evict(bbox, levels, **kwargs)

        def before(changed, mip):
            result = real_before(changed, mip)
            self.before.append(result)
            return result

        def unit_edit_bbox(changed, pre=REACH_FALLBACK, post=REACH_FALLBACK):
            out = real_bbox(changed, pre, post)
            if pre is not REACH_FALLBACK and post is not REACH_FALLBACK:
                self.tight.append((out[0], real_bbox(changed)[0]))
                self.mark = len(self.calls)
            else:
                self.reach.append(out[0])
            return out

        monkeypatch.setattr(cache, "patch", patch)
        monkeypatch.setattr(cache, "invalidate_region", evict)
        monkeypatch.setattr(cache, "sprite_extent_before", before)
        monkeypatch.setattr(window, "_unit_edit_bbox", unit_edit_bbox)

    def reset(self) -> None:
        self.calls.clear()
        self.before.clear()
        self.tight.clear()
        self.reach.clear()
        self.mark = 0

    def assert_tight(self, mip: int) -> None:
        """The visible level was repainted with a bbox strictly inside
        today's, and nothing repainted it with anything else."""
        assert self.tight, "the tight path never ran"
        assert REACH_FALLBACK not in self.before
        tight, reach = self.tight[-1]
        assert _contains(reach, tight) and tight != reach, (tight, reach)
        repaint = self.calls[self.mark :]
        visible = [c for c in repaint if c[2] == (mip,)]
        assert visible and all(c[1] == tight for c in visible), repaint
        assert all(c[2] is not None for c in repaint), repaint

    def assert_fallback(self, mip: int) -> None:
        """Today's reach bbox, split like the tight one: the visible level
        patched or evicted with it, every other level evicted with it, and
        nothing repainted at levels None (every resident level)."""
        assert self.before and self.before[-1] is REACH_FALLBACK
        assert not self.tight
        assert self.reach, "the reach bbox was never sized"
        reach = self.reach[-1]
        assert self.calls and all(c[1] == reach and c[2] is not None for c in self.calls), self.calls
        assert any(c[2] == (mip,) for c in self.calls), self.calls
        assert all(c[0] == "evict" for c in self.calls if mip not in c[2]), self.calls


def _add(window, player: int, const: int, x: float, y: float):
    """A setup edit (wholesale repaint, not under test)."""
    model = window._ensure_unit_edits()
    with window._unit_edit(model, "Add", [player]):
        return model.add(player, const, x, y)


def _select(window, unit):
    (entry,) = [e for e in window.map_view._unit_index.entries if e.unit is unit]
    window._selection = [(entry.player_id, entry.unit.reference_id)]
    window._refresh_selection_view()
    return entry


def _place(window, monkeypatch, player: int, const: int, x: float, y: float):
    """Place Unit's real handler, with the click already resolved to (x, y)."""
    from PyQt5.QtCore import Qt

    monkeypatch.setattr(window, "_placement_point", lambda _pos, _mods: (x, y))
    monkeypatch.setattr(window.units_panel, "selected_object_const", lambda: const)
    window.units_panel.select_owner(player)
    window.on_unit_place(None, Qt.NoModifier)
    placed = [u for u in window.scenario.unit_manager.units[player] if (u.unit_const, u.x, u.y) == (const, x, y)]
    return placed[-1]


def _set_x(window, unit, x: float) -> None:
    entry = _select(window, unit)
    window.units_panel.show_unit(entry)
    window.units_panel.unit_field_editors["x"].setValue(x)
    assert unit.x == x


def _delete(window, unit) -> None:
    from PyQt5.QtCore import Qt

    _select(window, unit)
    window.on_unit_delete(Qt.NoModifier)


def _draw(window, terrain_id: int, tiles) -> None:
    window.terrain_panel.set_terrain(terrain_id)
    window.on_edit_stroke_start()
    for x, y in tiles:
        window.on_edit_stroke_tile(x, y, 0)
    window.on_edit_stroke_end()


def _ratio(monkeypatch, repaint: str) -> None:
    """Forces a batch's outcome: patch everything, or evict everything."""
    import descape.viewer as viewer_module

    value = float("inf") if repaint == "patch" else 0.0
    monkeypatch.setattr(viewer_module, "_TIGHT_PATCH_AREA_RATIO", {"stepped": value, "sloped": value})


# --- the fresh-render oracles ----------------------------------------------


@pytest.mark.parametrize("a_path", ["wholesale", "component"])
@pytest.mark.parametrize("repaint", ["patch", "evict"])
@pytest.mark.parametrize(("style", "mip"), STYLE_MIPS)
def test_draw_strokes_over_contested_trees_undo_and_redo(style, mip, repaint, a_path, art, monkeypatch) -> None:
    """Stroke A crosses a villager's tile. It splices that shared tile's
    component, or with the component cap at 0 the cache rebuilds its sources
    wholesale and the post side must resolve a level with no post-edit layer.
    Stroke B replaces half of A's oaks with palms and plants the rest on
    grass: a splice. Undo removes palms with no replacement, which only the
    pre side covers. Each is checked patched and evicted."""
    _ratio(monkeypatch, repaint)
    real_cap = render_cache._COMPONENT_SPLICE_MAX_UNITS
    if a_path == "wholesale":
        monkeypatch.setattr(render_cache, "_COMPONENT_SPLICE_MAX_UNITS", 0)
    window = _window(style)
    try:
        _add(window, 1, VILLAGER, STROKE_TILES[3][0] + 0.5, STROKE_TILES[3][1] + 0.5)
        window.mode_combo.setCurrentText("Terrain")
        window._on_tool_selected("draw")
        window.paint_trees_check.setChecked(True)
        window.paint_eye_candy_check.setChecked(False)
        _show_level(window, monkeypatch, mip)
        spy = _Spy(window, monkeypatch)
        spliced = []
        real_plan = window._cache._unit_splice_plan
        # The plan invalidate_units() consumes, so each call is one splice-or-rebuild verdict.
        monkeypatch.setattr(
            window._cache, "_unit_splice_plan",
            lambda c: (plan := real_plan(c), spliced.append(plan is not None))[0],
        )
        _level(window, mip)

        _draw(window, FOREST_OAK, STROKE_TILES)
        spy.assert_tight(mip)
        after_a = _check(window, mip, "stroke A")
        # Back to the real cap, so stroke B and the undo/redo run as they would.
        monkeypatch.setattr(render_cache, "_COMPONENT_SPLICE_MAX_UNITS", real_cap)
        spy.reset()
        _draw(window, FOREST_PALM, SHIFTED_TILES)
        spy.assert_tight(mip)
        after_b = _check(window, mip, "stroke B")
        assert not np.array_equal(after_a, after_b)
        assert spliced == [a_path == "component", True], f"stroke A must take the {a_path} path, stroke B splice"

        spy.reset()
        window.undo()
        spy.assert_tight(mip)
        assert np.array_equal(_check(window, mip, "undo of B"), after_a)
        spy.reset()
        window.redo()
        spy.assert_tight(mip)
        assert np.array_equal(_check(window, mip, "redo of B"), after_b)
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("const", [CASTLE, TREE_OAK], ids=["castle", "tree"])
@pytest.mark.parametrize(("style", "mip"), STYLE_MIPS)
def test_place_move_and_delete_a_tall_unit(style, mip, const, art, monkeypatch) -> None:
    """Place has only a post side, Delete only a pre side, Move both. The
    castle is multi-anchor (its tower piece keys a second footprint tile)."""
    player = 1 if const == CASTLE else 0
    window = _window(style)
    try:
        _show_level(window, monkeypatch, mip)
        spy = _Spy(window, monkeypatch)
        start = _level(window, mip)

        unit = _place(window, monkeypatch, player, const, 60.0, 60.0)
        cache = window._cache
        contribution = render._resolve_unit_sprite(
            window.scenario, cache.proj, cache.elevations, cache.unit_filter, None, {}, True, player,
            window.scenario.unit_manager.units[player].index(unit), unit,
        )
        assert len(contribution.bboxes) == (2 if const == CASTLE else 1)
        spy.assert_tight(mip)
        placed = _check(window, mip, "place")
        assert not np.array_equal(placed, start)

        spy.reset()
        _set_x(window, unit, unit.x + 6)
        spy.assert_tight(mip)
        _check(window, mip, "move")

        spy.reset()
        _delete(window, unit)
        spy.assert_tight(mip)
        assert np.array_equal(_check(window, mip, "delete"), start)
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize(("style", "mip"), STYLE_MIPS)
def test_moving_a_mark_only_building_across_an_elevation_step(style, mip, art, monkeypatch) -> None:
    """The dock has no art, so no sprite term covers it: only the building
    bbox (its mark, drawn at its own tile's height) does."""
    window = _window(style, slope=True)
    try:
        dock = _add(window, 1, DOCK, 39.5, 37.5)
        assert render._resolve_unit_sprite(
            window.scenario, window._cache.proj, window._cache.elevations, window._cache.unit_filter, None, {},
            True, 1, 0, dock,
        ) is None
        tiles = render.unit_occupied_tiles(dock, 120, 120)
        assert len({int(window._cache.elevations[y, x]) for x, y in tiles}) > 1, "the footprint is flat"
        _show_level(window, monkeypatch, mip)
        spy = _Spy(window, monkeypatch)
        _level(window, mip)
        _set_x(window, dock, 41.5)
        spy.assert_tight(mip)
        _check(window, mip, "mark move onto the slope")
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_a_visible_level_with_no_resident_chunks_takes_the_reach_fallback(style, art, monkeypatch) -> None:
    """Stepped: mip 0 is resident but the view shows mip -1, which holds
    nothing. Sloped: nothing is resident at all. Either way today's bbox runs,
    on the visible level, and mip 0 evicts it."""
    window = _window(style)
    try:
        villager = _add(window, 1, VILLAGER, 60.5, 60.5)
        visible = -1 if style == "Stepped" else 0
        _show_level(window, monkeypatch, visible)
        if style == "Stepped":
            _level(window, 0)
        spy = _Spy(window, monkeypatch)
        _set_x(window, villager, 64.5)
        spy.assert_fallback(visible)
        _check(window, 0, "move under the fallback")
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize(("tight", "expected"), [(0.0, "evict"), (float("inf"), "patch")])
def test_the_tight_split_prices_its_batch_on_the_tight_ratio(tight, expected, art, monkeypatch) -> None:
    """The tight split's patch-or-evict choice reads _TIGHT_PATCH_AREA_RATIO."""
    import descape.viewer as viewer_module

    monkeypatch.setattr(viewer_module, "_TIGHT_PATCH_AREA_RATIO", {"stepped": tight, "sloped": tight})
    window = _window("Stepped")
    try:
        window.mode_combo.setCurrentText("Terrain")
        window._on_tool_selected("draw")
        window.paint_trees_check.setChecked(True)
        window.paint_eye_candy_check.setChecked(False)
        _show_level(window, monkeypatch, 0)
        spy = _Spy(window, monkeypatch)
        _level(window, 0)
        _draw(window, FOREST_OAK, STROKE_TILES)
        spy.assert_tight(0)
        visible = [c[0] for c in spy.calls[spy.mark :] if c[2] == (0,)]
        assert visible == [expected], spy.calls[spy.mark :]
        _check(window, 0, "stroke")
    finally:
        conftest.close_window(window)


def test_the_area_check_answers_false_with_no_viewport_target(art, monkeypatch) -> None:
    """Even at a zero ratio: with no target there is no visible area to price
    an eviction against, which is why the no-target reach patch never asks."""
    import descape.viewer as viewer_module

    monkeypatch.setattr(viewer_module, "_TIGHT_PATCH_AREA_RATIO", {"stepped": 0.0, "sloped": 0.0})
    window = _window("Stepped")
    try:
        _level(window, 0)
        bbox = (0, 0, window._cache.chunk_px, window._cache.chunk_px)
        assert window._patch_area_exceeds_viewport(bbox, levels=(0,)), "a zero ratio did not evict with a target"
        monkeypatch.setattr(window.map_view, "viewport_chunk_target", lambda: None)
        assert not window._patch_area_exceeds_viewport(bbox, levels=(0,))
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize(("style", "mip"), STYLE_MIPS)
def test_placing_a_wall_beside_walls_takes_the_reach_fallback(style, mip, art, monkeypatch) -> None:
    """A wall's neighbours change shape with it, so only today's reach bbox
    covers them. The fallback must fire, and the neighbours must repaint.
    Both setup walls store a radian-encoded index, which makes the file
    radian-encoded, so every wall's shape is derived from its neighbours."""
    window = _window(style)
    try:
        walls = [_add(window, 1, STONE_WALL, x, 50.5) for x in (50.5, 52.5)]
        for wall in walls:
            wall.rotation = 2 * math.pi / 5
        window._after_unit_mutation()
        assert render.wall_variant_rotation_overrides(window.scenario) == {}, "isolated walls keep their own shape"
        _show_level(window, monkeypatch, mip)
        spy = _Spy(window, monkeypatch)
        before = _level(window, mip)
        _place(window, monkeypatch, 1, STONE_WALL, 51.5, 50.5)
        assert len(render.wall_variant_rotation_overrides(window.scenario)) == 3, "the run did not reshape"
        assert not np.array_equal(_check(window, mip, "wall place"), before)
        spy.assert_fallback(mip)
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize(("style", "mip"), STYLE_MIPS)
def test_a_gate_orientation_change_takes_the_reach_fallback(style, mip, art, monkeypatch) -> None:
    window = _window(style)
    try:
        gate = _add(window, 1, GATE_NE, 60.0, 60.5)
        _select(window, gate)
        _show_level(window, monkeypatch, mip)
        spy = _Spy(window, monkeypatch)
        _level(window, mip)
        window.on_unit_rotate(1)
        assert gate.unit_const != GATE_NE
        _check(window, mip, "gate orientation")
        spy.assert_fallback(mip)
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize(("style", "mip"), STYLE_MIPS)
def test_a_convert_stroke_repaints_the_new_team_colour(style, mip, art, monkeypatch) -> None:
    window = _window(style)
    try:
        heroes = [_add(window, 1, HERO, 60.5 + 3 * i, 60.5) for i in range(2)]
        window.units_panel.select_owner(2)
        window.brush_size_spin.setValue(1)
        _show_level(window, monkeypatch, mip)
        spy = _Spy(window, monkeypatch)
        before = _level(window, mip)
        window._begin_convert_stroke()
        for unit in heroes:
            window._convert_stroke_tile(int(unit.x), int(unit.y))
        window._end_convert_stroke()
        assert all(u in window.scenario.unit_manager.units[2] for u in heroes)
        spy.assert_tight(mip)
        assert not np.array_equal(_check(window, mip, "convert"), before)
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize(("style", "mip"), STYLE_MIPS)
def test_a_convert_of_a_reshaped_wall_and_a_gate_is_sized_tight(style, mip, art, monkeypatch) -> None:
    """A reassign moves nothing, so no wall's neighbour mask changes: the
    middle wall of a reshaped run (its override read at its new key) and a
    gate repaint from their real extents, not the reach bbox."""
    window = _window(style)
    try:
        walls = [_add(window, 1, STONE_WALL, x, 50.5) for x in (50.5, 51.5, 52.5)]
        for wall in walls:
            wall.rotation = 2 * math.pi / 5
        gate = _add(window, 1, GATE_NE, 60.0, 60.5)
        window._after_unit_mutation()
        shapes = render.wall_variant_rotation_overrides(window.scenario)
        assert len(shapes) == 3, "the run did not reshape -- the overrides path is vacuous"
        window.units_panel.select_owner(2)
        window.brush_size_spin.setValue(1)
        _show_level(window, monkeypatch, mip)
        spy = _Spy(window, monkeypatch)
        # The synthetic frames share one size, so pixels can't show which frame the
        # post extent resolved: record the overrides it passes for the wall instead.
        seen: list[dict] = []
        inside = [False]
        real_after, real_resolve = window._cache.sprite_extent_after, render._resolve_unit_sprite

        def after(changed, level):
            inside[0] = True
            try:
                return real_after(changed, level)
            finally:
                inside[0] = False

        def resolve(*args, **kwargs):
            if inside[0] and args[9] is walls[1]:
                seen.append(args[5])
            return real_resolve(*args, **kwargs)

        monkeypatch.setattr(window._cache, "sprite_extent_after", after)
        monkeypatch.setattr(render, "_resolve_unit_sprite", resolve)
        before = _level(window, mip)
        window._begin_convert_stroke()
        for unit in (walls[1], gate):
            window._convert_stroke_tile(int(unit.x), int(unit.y))
        window._end_convert_stroke()
        assert walls[1] in window.scenario.unit_manager.units[2] and gate in window.scenario.unit_manager.units[2]
        now = render.wall_variant_rotation_overrides(window.scenario)
        assert sorted(now.values()) == sorted(shapes.values()), "the convert reshaped the run"
        key = (2, window.scenario.unit_manager.units[2].index(walls[1]))
        assert seen and all(o.get(key) == now[key] for o in seen), "the post extent resolved without the real overrides"
        spy.assert_tight(mip)
        assert not np.array_equal(_check(window, mip, "wall and gate convert"), before)
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize(("style", "mip"), STYLE_MIPS)
def test_rotating_a_unit_repaints_its_new_frame(style, mip, art, monkeypatch) -> None:
    window = _window(style)
    try:
        villager = _add(window, 1, VILLAGER, 60.5, 60.5)
        _select(window, villager)
        _show_level(window, monkeypatch, mip)
        spy = _Spy(window, monkeypatch)
        before = _level(window, mip)
        window.on_unit_rotate(1)
        spy.assert_tight(mip)
        assert not np.array_equal(_check(window, mip, "rotate"), before)
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize(("layer", "player", "const"), [("hero_glow", 1, HERO), ("small_trees", 0, TREE_OAK)])
@pytest.mark.parametrize(("style", "mip"), STYLE_MIPS)
def test_a_build_time_layer_reaches_the_resolved_extent(style, mip, layer, player, const, art, monkeypatch) -> None:
    """Hero Glow grows the hero's art and Small Trees shrinks the tree's, so
    the post side must resolve with the cache's own layer values."""
    window = _window(style)
    try:
        unit = _add(window, player, const, 60.5, 60.5)
        window.layer_actions[layer].setChecked(True)
        assert getattr(window._cache.layers, layer)
        _show_level(window, monkeypatch, mip)
        spy = _Spy(window, monkeypatch)
        _level(window, mip)
        _set_x(window, unit, 63.5)
        spy.assert_tight(mip)
        _check(window, mip, f"move under {layer}")
    finally:
        conftest.close_window(window)


def test_a_stale_second_level_evicts_the_reach_bbox_and_is_not_recomposited(art, monkeypatch) -> None:
    """Stepped, mip -1 on screen and a region of mip 0 resident, both levels
    stale (a gen bump with nothing changed, so every chunk is still right).
    The visible level reads its stale layer for the pre side. Mip 0 loses its
    chunks in today's bbox and is neither patched nor rebuilt."""
    window = _window("Stepped")
    try:
        cache = window._cache
        castle = _add(window, 1, CASTLE, 60.0, 60.0)
        _show_level(window, monkeypatch, -1)
        _level(window, -1)
        region = (2048, 1024, 5632, 3584)
        cache.render_rect(*region, mip=0)
        cache.invalidate_units()
        assert not cache.is_level_resident(-1) and not cache.is_level_resident(0)
        spy = _Spy(window, monkeypatch)

        _set_x(window, castle, 64.0)

        spy.assert_tight(-1)
        _tight, reach = spy.tight[-1]
        assert ("evict", reach, (0,)) in spy.calls
        assert not cache.is_level_resident(0), "mip 0 was rebuilt by the edit"
        lx0, ly0, lx1, ly1 = cache._bbox_to_level(0, reach)
        cx0, cy0, cx1, cy1 = cache.chunk_index_range(0, lx0, ly0, lx1, ly1)
        assert not any(cache.has_chunk(0, cx, cy) for cx in range(cx0, cx1 + 1) for cy in range(cy0, cy1 + 1))
        assert any(key[0] == 0 for key in cache._cache), "the eviction took more than today's bbox"
        _check(window, -1, "visible level")
        assert np.array_equal(cache.render_rect(*region, mip=0), _fresh(window, 0, region))
    finally:
        conftest.close_window(window)


def _wall_run(window, xs=(50.5, 51.5, 52.5), y: float = 50.5):
    """Stone walls in a row, radian-encoded, so every one's shape is derived from its neighbours."""
    walls = [_add(window, 1, STONE_WALL, x, y) for x in xs]
    for wall in walls:
        wall.rotation = 2 * math.pi / 5
    window._after_unit_mutation()
    return walls


def test_a_wholesale_wall_move_rebuilds_no_off_screen_level(art, monkeypatch) -> None:
    """Group-move-wall plan Step 1: a wall Move whose sources rebuild wholesale
    (the component cap at 0 keeps it wholesale now that moved walls splice),
    mips -2/-1/0 resident and 0 on screen. The reach bbox patches mip 0 alone
    and evicts itself from -2 and -1, which stay stale for LevelWarmer instead
    of rebuilding inside the handler. A forced read then matches a fresh cache."""
    monkeypatch.setattr(render_cache, "_COMPONENT_SPLICE_MAX_UNITS", 0)
    mips = (-2, -1, 0)
    window = _window("Stepped")
    try:
        cache = window._cache
        assert set(mips) <= set(cache.mip_levels()), cache.mip_levels()
        walls = _wall_run(window)
        assert len(render.wall_variant_rotation_overrides(window.scenario)) == 3, "the run did not reshape"
        _show_level(window, monkeypatch, 0)
        # Whole coarse levels, but only a region of mip 0: the whole of it alone fills the chunk budget.
        cache.render_rect(2048, 1024, 5632, 3584, mip=0)
        for mip in (-1, -2):
            _level(window, mip)
        assert cache.resident_levels() == list(mips)
        assert all(cache._levels[m].gen == cache._source_gen for m in mips)
        spy = _Spy(window, monkeypatch)
        plans: list = []
        real_plan = cache._unit_splice_plan
        monkeypatch.setattr(cache, "_unit_splice_plan", lambda c: (plan := real_plan(c), plans.append(plan))[0])

        entries = [e for e in window.map_view._unit_index.entries if e.unit is walls[0] or e.unit is walls[2]]
        window._move_units(window._ensure_unit_edits(), entries, 0.0, 3.0, "Move 2 units")

        assert plans == [None], "the Move did not take the wholesale path"
        rebuilt = [m for m in (-2, -1) if cache._levels[m].gen == cache._source_gen]
        assert not rebuilt, f"off-screen levels {rebuilt} were rebuilt inside the handler"
        spy.assert_fallback(0)
        reach = spy.reach[-1]
        assert ("evict", reach, (-2, -1)) in spy.calls, spy.calls
        for mip in (-2, -1):
            cx0, cy0, cx1, cy1 = cache.chunk_index_range(mip, *cache._bbox_to_level(mip, reach))
            assert not any(cache.has_chunk(mip, cx, cy) for cx in range(cx0, cx1 + 1) for cy in range(cy0, cy1 + 1))
        for mip in mips:
            cache._level(mip)
            _check(window, mip, f"level {mip} after the wall Move")
    finally:
        conftest.close_window(window)


def test_with_no_viewport_target_the_reach_bbox_patches_every_resident_level(art, monkeypatch) -> None:
    window = _window("Stepped")
    try:
        villager = _add(window, 1, VILLAGER, 60.5, 60.5)
        monkeypatch.setattr(window.map_view, "viewport_chunk_target", lambda: None)
        _level(window, 0)
        spy = _Spy(window, monkeypatch)
        _set_x(window, villager, 64.5)
        assert spy.before and spy.before[-1] is REACH_FALLBACK
        assert spy.calls and all(c == ("patch", spy.reach[-1], None) for c in spy.calls), spy.calls
        _check(window, 0, "move with no viewport target")
    finally:
        conftest.close_window(window)


# --- unit level ------------------------------------------------------------


class _Unit:
    """The attributes the resolver and the tile walks read, duck-typed the way
    tests/test_sprite_edit_bbox.py's own Unit is."""

    def __init__(self, x: float, y: float, unit_const: int):
        self.x, self.y, self.unit_const, self.rotation = x, y, unit_const, 0.0


def _bare_cache(*units: _Unit, sprites: bool = True) -> IsoChunkCache:
    """An IsoChunkCache over the blank template holding `units` for GAIA."""
    scenario = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    scenario.unit_manager.units[0].extend(units)
    elevations, proj = render.elevations_and_proj(scenario)
    return IsoChunkCache(scenario, elevations, proj, proj.tile_px, sprites=sprites)


def test_level_bbox_to_reference_round_trips_as_a_superset_at_every_mip() -> None:
    cache = _bare_cache(sprites=False)
    rng = np.random.default_rng(7)
    assert len(cache.mip_levels()) > 2
    for mip in cache.mip_levels():
        w, h = cache.canvas_dims(mip)
        for _ in range(200):
            x0, x1 = sorted(int(v) for v in rng.integers(0, w, 2))
            y0, y1 = sorted(int(v) for v in rng.integers(0, h, 2))
            box = (x0, y0, x1 + 1, y1 + 1)
            ref = cache._level_bbox_to_reference(mip, box)
            assert _contains(cache._bbox_to_level(mip, ref), box), (mip, box, ref)
            if mip == 0:
                assert ref == box


@pytest.mark.parametrize("mip", [0, -1])
def test_sprite_extents_of_a_tree_equal_its_contribution_converted(mip, art) -> None:
    """A centred 1x1 tree has no building bbox of its own, so each side is
    exactly its contribution's bbox at the level, converted to reference."""
    tree = _Unit(30.5, 30.5, TREE_OAK)
    cache = _bare_cache(tree)
    scenario = cache.scenario
    w, h = cache.canvas_dims(mip)
    cache.render_rect(0, 0, w, h, mip=mip)
    proj = cache._levels[mip].proj
    index = scenario.unit_manager.units[0].index(tree)

    def contribution_box():
        c = render._resolve_unit_sprite(
            scenario, proj, cache.elevations, cache.unit_filter, None, {}, True, 0, index, tree
        )
        (box,) = c.bboxes.values()
        return cache._level_bbox_to_reference(mip, box)

    old = contribution_box()
    tree.x = 33.5
    changed = [UnitSplice(0, index, tree, (30, 30), (33, 30), ((30, 30),), ((33, 30),))]
    assert cache.sprite_extent_before(changed, mip) == old
    cache.invalidate_units(changed)
    new = contribution_box()
    assert cache.sprite_extent_after(changed, mip) == new != old


def test_sprite_extents_fall_back_for_a_wall_or_a_level_with_nothing_resident(art) -> None:
    cache = _bare_cache()
    w, h = cache.canvas_dims(0)
    cache.render_rect(0, 0, w // 4, h // 4, mip=0)
    wall, tree = _Unit(10.5, 10.5, STONE_WALL), _Unit(10.5, 10.5, TREE_OAK)
    wall_splice = [UnitSplice(0, 0, wall, None, (10, 10), (), ((10, 10),))]
    tree_splice = [UnitSplice(0, 0, tree, None, (10, 10), (), ((10, 10),))]
    assert cache.sprite_extent_before(wall_splice, 0) is REACH_FALLBACK
    assert cache.sprite_extent_after(wall_splice, 0) is REACH_FALLBACK
    assert cache.sprite_extent_before(tree_splice, -1) is REACH_FALLBACK
    assert cache.sprite_extent_before(tree_splice, None) is REACH_FALLBACK
    assert cache.sprite_extent_before(tree_splice, 0) is None


def test_flat_has_no_unit_edit_sprite_extent() -> None:
    scenario = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    mm = scenario.map_manager
    cache = FlatChunkCache(scenario, render.tile_pixels_for_map(mm.map_width, mm.map_height))
    cache.render_rect(0, 0, 64, 64, mip=0)
    unit = _Unit(1.5, 1.5, TREE_OAK)
    changed = [UnitSplice(0, 0, unit, None, (1, 1), (), ((1, 1),))]
    for extent, mip in ((cache.sprite_extent_before, 0), (cache.sprite_extent_after, 0), (cache.sprite_extent_before, -1)):
        with pytest.raises(NotImplementedError):
            extent(changed, mip)


def test_the_reach_fallback_sentinel_is_one_named_object() -> None:
    assert render_cache.REACH_FALLBACK is REACH_FALLBACK
    assert repr(REACH_FALLBACK) == "REACH_FALLBACK"
