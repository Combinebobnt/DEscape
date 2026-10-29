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
Sloped at mip 0, -1 and -2 (its coarser ladder, TASK-031.60) and +1 (TASK-146).
A +1 case renders and checks only the rect over its edits (_show_level()'s
focus), with the LRU raised so all of it stays resident: a whole +1 level of
the template is 4x the budget.
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
PALISADE = 72
GATE_NE = 64
TOWER_PIECE = 9001

STROKE_TILES = [(20 + i, 30) for i in range(8)]
SHIFTED_TILES = [(24 + i, 30) for i in range(8)]
STYLE_MIPS = [("Stepped", 0), ("Stepped", -1), ("Sloped", 0), ("Sloped", -1), ("Sloped", -2), ("Sloped", 1)]
# Tile boxes (x0, y0, x1, y1) a +1 case renders and checks instead of the whole level, which is
# 118M px on the template, 4x the LRU (TASK-146). _show_level() takes one; MAIN is the default.
FOCUS_MAIN = (49, 49, 67, 62)  # every unit placed, moved or recoloured on rows 50 to 61
FOCUS_DRAW = (19, 29, 32, 31)  # STROKE_TILES and SHIFTED_TILES
FOCUS_DOCK = (36, 35, 44, 44)  # the dock's move onto the cone at (40, 40)


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
    # A wall drawing each stored shape in its own colour (no team mask, frames picked by
    # variant_index()), so a stale shape shows (GH #266).
    add(PALISADE, 6 * T, angles=5, colour_base=29)["rotation_is_variant"] = True
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


_STYLE_CACHES = {"Stepped": IsoChunkCache, "Sloped": SlopedChunkCache, "Flat": FlatChunkCache}


def _window(style: str, slope: bool = False):
    """The blank template in `style` (Flat: top-down, Isometric View off),
    Units mode, sprites on. slope raises a cone at (40, 40) so a footprint
    there spans several elevations."""
    window = conftest.blank_window()
    if slope:
        set_tile_elevation(window.scenario.map_manager, 40, 40, 3)
    if style == "Flat":
        window.iso_action.setChecked(False)
    window.terrain_style_combo.setCurrentText(style)
    if slope:
        window._render_current()
    window.mode_combo.setCurrentText("Units")
    assert window._cache.sprites_enabled
    assert isinstance(window._cache, _STYLE_CACHES[style])
    return window


def _focus_rect(cache, mip: int, tiles) -> tuple[int, int, int, int]:
    """Level `mip`'s rect over the tile box `tiles`: its corners' swept bounds
    at level 0, padded by the sprite reach, converted and clamped."""
    from descape import iso_geometry

    x0, y0, x1, y1 = tiles
    corners = ((x0, y0), (x1, y0), (x0, y1), (x1, y1))
    boxes = np.array([iso_geometry.tile_screen_bounds_swept(x, y, cache.proj) for x, y in corners])
    left, up, right, down = render._sprite_reach_px(cache.proj)
    ref = (int(boxes[:, 0].min()) - left, int(boxes[:, 1].min()) - up, int(boxes[:, 2].max()) + right, int(boxes[:, 3].max()) + down)
    lx0, ly0, lx1, ly1 = cache._bbox_to_level(mip, ref)
    w, h = cache.canvas_dims(mip)
    return max(0, lx0), max(0, ly0), min(w, lx1), min(h, ly1)


def _level_rect(window, mip: int) -> tuple[int, int, int, int]:
    """The whole level, or at +1 the rect over _show_level()'s focus tiles."""
    if mip > 0:
        return _focus_rect(window._cache, mip, window._test_focus)
    return (0, 0, *window._cache.canvas_dims(mip))


def _show_level(window, monkeypatch, mip: int, focus=FOCUS_MAIN) -> None:
    """Pins the level the viewer treats as on screen, with the whole level in
    view; at +1 only the `focus` tiles' rect, every chunk of it kept resident."""
    cache = window._cache
    window._test_focus = focus
    if mip > 0:
        cache.max_bytes *= 8
    cx0, cy0, cx1, cy1 = cache.chunk_index_range(mip, *_level_rect(window, mip))
    monkeypatch.setattr(window.map_view, "viewport_chunk_target", lambda: (mip, cx0, cy0, cx1, cy1))


def _level(window, mip: int) -> np.ndarray:
    """The level (+1: its focus rect) as the cache serves it; every chunk is resident afterwards."""
    return window._cache.render_rect(*_level_rect(window, mip), mip=mip).copy()


def _fresh(window, mip: int, rect=None) -> np.ndarray:
    """A fresh cache's render of level `mip` (or a level-pixel rect of it)
    from the window's current scenario, filter, layers and grid."""
    cache = window._cache
    common = {"unit_filter": cache.unit_filter, "sprites": True, "layers": cache.layers}
    if isinstance(cache, FlatChunkCache):
        fresh = FlatChunkCache(window.scenario, cache.tile_px, **common)
    elif isinstance(cache, SlopedChunkCache):
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
    fresh = _fresh(window, mip, _level_rect(window, mip))
    if not np.array_equal(now, fresh):
        ys, xs = np.nonzero((now != fresh).any(axis=2))
        pytest.fail(f"{what}: {len(xs)} stale px at level {mip}, in x {xs.min()}..{xs.max()}, y {ys.min()}..{ys.max()}")
    return now


def _contains(outer, inner) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def _covers(outer, inner) -> bool:
    """_contains() for extents, where None paints nothing."""
    return inner is None or (outer is not None and _contains(outer, inner))


def _shares_a_tile(cache, changed: list[UnitSplice]) -> bool:
    """Whether a unit outside `changed` stands on one of its new footprint tiles."""
    ours = {id(s.unit) for s in changed}
    return any(id(u) not in ours for s in changed for t in s.new_tiles for u, _c in cache.units_by_tile.get(t, ()))


@pytest.fixture(autouse=True)
def layer_extents(monkeypatch) -> list:
    """TASK-031.61 c2x S3's oracle on every extent case here: an answer
    sprite_extent_after() read from a current layer contains the per-unit
    resolve, and equals it where no other unit shares a new tile. Returns
    each (mip, units resolved while answering) it checked."""
    checked = []
    resolves = [0]
    real, real_resolve = render_cache._ChunkCacheBase.sprite_extent_after, render._resolve_unit_sprite

    def resolve(*args, **kwargs):
        resolves[0] += 1
        return real_resolve(*args, **kwargs)

    def after(self, changed, mip):
        resolves[0] = 0
        box = real(self, changed, mip)
        if box is not REACH_FALLBACK and self._layer_current(mip):
            checked.append((mip, resolves[0]))
            resolved = self._resolved_extent(changed, mip)
            assert _covers(box, resolved), f"layer {box} misses the resolved {resolved}"
            assert _shares_a_tile(self, changed) or box == resolved, f"layer {box} != resolved {resolved}"
        return box

    monkeypatch.setattr(render, "_resolve_unit_sprite", resolve)
    monkeypatch.setattr(render_cache._ChunkCacheBase, "sprite_extent_after", after)
    return checked


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
        # Every patch_composite_rects() call: (rects, levels).
        self.rects: list[tuple[list, tuple | None]] = []
        real_patch, real_evict, real_rects = cache.patch, cache.invalidate_region, cache.patch_composite_rects
        real_before, real_bbox = cache.sprite_extent_before, window._unit_edit_bbox

        def patch(bbox, elevation_changed=None, levels=None, **kwargs):
            self.calls.append(("patch", bbox, None if levels is None else tuple(levels)))
            return real_patch(bbox, elevation_changed, levels, **kwargs)

        def evict(bbox, levels=None, **kwargs):
            self.calls.append(("evict", bbox, None if levels is None else tuple(levels)))
            return real_evict(bbox, levels, **kwargs)

        def patch_rects(rects, levels=None, **kwargs):
            self.rects.append((list(rects), None if levels is None else tuple(levels)))
            return real_rects(rects, levels=levels, **kwargs)

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
        monkeypatch.setattr(cache, "patch_composite_rects", patch_rects)
        monkeypatch.setattr(cache, "sprite_extent_before", before)
        monkeypatch.setattr(window, "_unit_edit_bbox", unit_edit_bbox)

    def reset(self) -> None:
        self.calls.clear()
        self.before.clear()
        self.tight.clear()
        self.reach.clear()
        self.rects.clear()
        self.mark = 0

    def assert_rects(self, mip: int) -> None:
        """The visible level was patched by per-chunk rects, each inside the
        tight bbox, and neither patched nor evicted any other way."""
        assert self.tight, "the tight path never ran"
        assert REACH_FALLBACK not in self.before
        tight, _reach = self.tight[-1]
        ((rects, levels),) = self.rects
        assert levels == (mip,) and rects and all(_contains(tight, rect) for rect in rects), (tight, rects)
        repaint = self.calls[self.mark :]
        assert all(c[2] is not None and mip not in c[2] for c in repaint), repaint

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

    monkeypatch.setattr(window, "_placement_point", lambda _pos, _mods, unit_const=None: (x, y))
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
    """Forces a batch's outcome: patch everything, evict everything, or
    (`rects`, Stepped) patch per-chunk rects in place of that eviction."""
    import descape.viewer as viewer_module

    value = float("inf") if repaint == "patch" else 0.0
    monkeypatch.setattr(viewer_module, "_TIGHT_PATCH_AREA_RATIO", {"stepped": value, "sloped": value})
    monkeypatch.setattr(viewer_module, "_RECT_PATCH_AREA_RATIO", float("inf") if repaint == "rects" else 0.0)


# --- the fresh-render oracles ----------------------------------------------


# Stepped also patches per-chunk rects (TASK-031.61 C1), at both of its levels.
CONTESTED = [
    (style, mip, repaint)
    for style, mip in STYLE_MIPS
    for repaint in ("patch", "evict", *(("rects",) if style == "Stepped" else ()))
]


@pytest.mark.parametrize("a_path", ["wholesale", "component"])
@pytest.mark.parametrize(("style", "mip", "repaint"), CONTESTED)
def test_draw_strokes_over_contested_trees_undo_and_redo(style, mip, repaint, a_path, art, monkeypatch) -> None:
    """Stroke A crosses a villager's tile. It splices that shared tile's
    component, or with the component cap at 0 the cache rebuilds its sources
    wholesale and the post side must resolve a level with no post-edit layer.
    Stroke B replaces half of A's oaks with palms and plants the rest on
    grass: a splice. Undo removes palms with no replacement, which only the
    pre side covers. Each is checked patched and evicted, and on Stepped
    patched by per-chunk rects."""
    _ratio(monkeypatch, repaint)
    real_cap = render_cache._COMPONENT_SPLICE_MAX_UNITS
    real_visible = render_cache.VISIBLE_SPLICE_MAX_UNITS
    if a_path == "wholesale":
        monkeypatch.setattr(render_cache, "_COMPONENT_SPLICE_MAX_UNITS", 0)
        monkeypatch.setattr(render_cache, "VISIBLE_SPLICE_MAX_UNITS", 0)
    window = _window(style)
    try:
        _add(window, 1, VILLAGER, STROKE_TILES[3][0] + 0.5, STROKE_TILES[3][1] + 0.5)
        window.mode_combo.setCurrentText("Terrain")
        window._on_tool_selected("draw")
        window.paint_trees_check.setChecked(True)
        window.paint_eye_candy_check.setChecked(False)
        _show_level(window, monkeypatch, mip, FOCUS_DRAW)
        spy = _Spy(window, monkeypatch)
        repainted = spy.assert_rects if repaint == "rects" else spy.assert_tight
        spliced = []
        real_plan = window._cache._unit_splice_plan
        # The plan invalidate_units() consumes, so each call is one splice-or-rebuild verdict.
        monkeypatch.setattr(
            window._cache, "_unit_splice_plan",
            lambda c: (plan := real_plan(c), spliced.append(plan is not None))[0],
        )
        _level(window, mip)

        _draw(window, FOREST_OAK, STROKE_TILES)
        repainted(mip)
        after_a = _check(window, mip, "stroke A")
        # Back to the real cap, so stroke B and the undo/redo run as they would.
        monkeypatch.setattr(render_cache, "_COMPONENT_SPLICE_MAX_UNITS", real_cap)
        monkeypatch.setattr(render_cache, "VISIBLE_SPLICE_MAX_UNITS", real_visible)
        spy.reset()
        _draw(window, FOREST_PALM, SHIFTED_TILES)
        repainted(mip)
        after_b = _check(window, mip, "stroke B")
        assert not np.array_equal(after_a, after_b)
        assert spliced == [a_path == "component", True], f"stroke A must take the {a_path} path, stroke B splice"

        spy.reset()
        window.undo()
        repainted(mip)
        assert np.array_equal(_check(window, mip, "undo of B"), after_a)
        spy.reset()
        window.redo()
        repainted(mip)
        assert np.array_equal(_check(window, mip, "redo of B"), after_b)
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("const", [CASTLE, TREE_OAK], ids=["castle", "tree"])
@pytest.mark.parametrize(("style", "mip"), STYLE_MIPS)
def test_place_move_and_delete_a_tall_unit(style, mip, const, art, monkeypatch) -> None:
    """Place has only a post side, Delete only a pre side, Move both. The
    castle is multi-anchor: its main piece's column slices key every tile on
    its two front faces (TASK-221), the tower's slot (the back corner on the
    left face) among them."""
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
        mm = window.scenario.map_manager
        tiles = render.unit_occupied_tiles(unit, mm.map_width, mm.map_height)
        x0, y1 = min(x for x, _ in tiles), max(y for _, y in tiles)
        faces = {t for t in tiles if t[0] == x0 or t[1] == y1}
        assert set(contribution.bboxes) == (faces if const == CASTLE else set(tiles))
        assert len(contribution.bboxes) == (7 if const == CASTLE else 1)
        if const == CASTLE:
            # The tower paints unmoved at its slot; the main piece's slices there are all moved ones.
            y0 = min(y for _, y in tiles)
            assert any(type(d) is not unit_sprites.MovedSlice for d, *_ in contribution.by_anchor[(x0, y0)])
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
        _show_level(window, monkeypatch, mip, FOCUS_DOCK)
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
    # No per-chunk rects (TASK-031.61 C1): past the ratio this evicts, as pinned.
    monkeypatch.setattr(viewer_module, "_RECT_PATCH_AREA_RATIO", 0.0)
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


def _owners_in_index(window, units) -> list[int]:
    """Each unit's owner as the pick index reports it."""
    index = window.map_view._unit_index
    return [next((e.player_id for e in index.entries if e.unit is unit), None) for unit in units]


def _check_numpy(window, step: str) -> np.ndarray:
    """Level 0 as the cache serves it, equal to a fresh numpy cache's."""
    import os

    from descape import composite_backend

    if os.environ.get("DESCAPE_REQUIRE_NATIVE") == "1":
        assert composite_backend.active_backend() == "native", composite_backend.unavailable_reason
    now = _level(window, 0)
    with composite_backend.use_backend("numpy"):
        fresh = _fresh(window, 0)
    if not np.array_equal(now, fresh):
        ys, xs = np.nonzero((now != fresh).any(axis=2))
        pytest.fail(f"{step}: {len(xs)} px differ from a fresh numpy render, x {xs.min()}..{xs.max()}, y {ys.min()}..{ys.max()}")
    return now


@pytest.mark.parametrize("style", ["Stepped", "Sloped", "Flat"])
def test_a_convert_stroke_its_undo_and_redo_match_a_fresh_numpy_render(style, art, monkeypatch) -> None:
    """TASK-052: a Convert drag across three team-coloured heroes and the
    tiles between them, through the real stroke handlers, at mip 0 in view.
    - before release, after the first hero and the throttled refresh (the timer's handler), the level changed
      and equals a fresh numpy render
    - the stroke records one undo step
    - after the stroke, its undo and its redo, every hero's owner in the scenario and in the pick index is
      2, 1 and 2, and the level equals a fresh numpy render
    - the stroke and the redo change the level from its pre-stroke pixels, and the undo restores them"""
    window = _window(style)
    try:
        heroes = [_add(window, 1, HERO, 60.5 + 3 * i, 60.5) for i in range(3)]
        window.units_panel.select_owner(2)
        window.brush_size_spin.setValue(1)
        _show_level(window, monkeypatch, 0)
        records = len(window.edit_history.records)
        start = _level(window, 0)

        window._begin_convert_stroke()
        window._convert_stroke_tile(60, 60)
        assert heroes[0] in window.scenario.unit_manager.units[2]
        window._flush_unit_stroke_refresh()
        mid = _check_numpy(window, "mid-stroke")
        assert not np.array_equal(mid, start), "the first hero's recolour was not on screen before release"
        for x in range(61, 67):
            window._convert_stroke_tile(x, 60)
        window._end_convert_stroke()
        assert len(window.edit_history.records) == records + 1

        stroked = None
        for step, act, owner in (("stroke", None, 2), ("undo", window.undo, 1), ("redo", window.redo, 2)):
            if act is not None:
                act()
            assert all(u in window.scenario.unit_manager.units[owner] for u in heroes), f"{step}: scenario owners"
            assert _owners_in_index(window, heroes) == [owner] * 3, f"{step}: pick index owners"
            now = _check_numpy(window, step)
            assert np.array_equal(now, start) is (owner == 1), f"{step}: pixels vs the pre-stroke level"
            if stroked is None:
                stroked = now
            assert owner == 1 or np.array_equal(now, stroked), f"{step}: differs from the stroke's pixels"
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


def _wall_run(window, xs=(50.5, 51.5, 52.5), y: float = 50.5, variant: int = 2, const: int = STONE_WALL):
    """Walls in a row, radian-encoded, so every one's shape is derived from its neighbours.
    Index 2 by default: what each derives once a move leaves it alone, so such a move writes no
    shape (GH #266) and keeps the path a test is about."""
    walls = [_add(window, 1, const, x, y) for x in xs]
    for wall in walls:
        wall.rotation = variant * 2 * math.pi / 5
    window._after_unit_mutation()
    return walls


def _literal_walls(window, shapes: dict) -> dict:
    """A Palisade on each tile of `shapes`, storing its literal index:
    with no radian value in the file, each draws its stored shape, and
    every shape draws differently (the `art` fixture)."""
    walls = {tile: _add(window, 1, PALISADE, tile[0] + 0.5, tile[1] + 0.5) for tile in shapes}
    for tile, wall in walls.items():
        wall.rotation = float(shapes[tile])
    window._after_unit_mutation()
    return walls


def _file_is_radian(window) -> bool:
    units = window.scenario.unit_manager.units
    return unit_sprites.file_is_radian(
        [render.stored_rotation(p, u) for p, us in enumerate(units) for u in us if unit_sprites.rotation_variant_eligible(u.unit_const)],
        5,
    )


def _drop(window, monkeypatch, unit, x: float, y: float) -> None:
    """Move's real release of `unit` alone, resolved to (x, y). Shift keeps
    it a plain move, so no garrison hit test reads the absent cursor."""
    from PyQt5.QtCore import Qt

    entry = _select(window, unit)
    monkeypatch.setattr(window, "_placement_point", lambda _pos, _mods, unit_const=None: (x, y))
    window.on_unit_move((entry.player_id, unit.reference_id), None, Qt.ShiftModifier)


@pytest.mark.parametrize(("style", "mip"), STYLE_MIPS)
def test_a_wall_dragged_out_of_an_integer_run_repaints_every_reshaped_wall(style, mip, art, monkeypatch) -> None:
    """GH #266 in an integer-encoded file: (52, 50) leaves its run for
    (52, 53), which continues the NW-SE diagonal (53, 54)-(54, 55). The run
    ends either side of the gap go 0 to 2 and (53, 54) goes 2 to 3. That one
    is no orthogonal neighbour of either tile, so only its own-equals-own
    splice repaints it."""
    window = _window(style)
    try:
        run = {(50, 50): 2, (51, 50): 0, (52, 50): 0, (53, 50): 0, (54, 50): 2}
        walls = _literal_walls(window, run | {(53, 54): 2, (54, 55): 2})
        _show_level(window, monkeypatch, mip)
        _level(window, mip)

        _drop(window, monkeypatch, walls[(52, 50)], 52.5, 53.5)

        reshaped = [walls[t].rotation for t in ((52, 50), (51, 50), (53, 50), (53, 54), (54, 55))]
        assert reshaped == [2.0, 2.0, 2.0, 3.0, 2.0]
        assert window.edit_history.peek_undo().unit_field_entries is not None, "not a fields-only record"
        _check(window, mip, "wall drag")
        window.undo()
        assert walls[(53, 54)].rotation == 2.0
        _check(window, mip, "undo")
        window.redo()
        _check(window, mip, "redo")
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize(("style", "mip"), STYLE_MIPS)
def test_a_move_that_overwrites_the_last_radian_walls_goes_wholesale(style, mip, art, monkeypatch) -> None:
    """GH #266: carrying the radian run's two ends off leaves all three walls
    towers, written as literals over the file's only radian values, so it
    now reads integer-encoded. The pair at (60, 56) stores 1: the radian
    file's overrides draw them as the run ends they are (2), an integer one
    as stored. They are outside the edit, so only the wholesale refresh
    repaints them. Undo flips the file back."""
    window = _window(style)
    try:
        walls = _wall_run(window, variant=1, const=PALISADE)
        _literal_walls(window, {(60, 56): 1, (61, 56): 1})
        assert _file_is_radian(window)
        _show_level(window, monkeypatch, mip)
        before = _level(window, mip)

        entries = [e for e in window.map_view._unit_index.entries if e.unit is walls[0] or e.unit is walls[2]]
        window._move_units(window._ensure_unit_edits(), entries, 0.0, 3.0, "Move 2 units")

        _check(window, mip, "radian flip")
        assert [w.rotation for w in walls] == [2.0, 2.0, 2.0]
        assert not _file_is_radian(window)
        assert window.edit_history.peek_undo().unit_field_entries is None, "the flip did not go wholesale"
        window.undo()
        assert _file_is_radian(window)
        assert np.array_equal(_check(window, mip, "undo"), before)
        window.redo()
        _check(window, mip, "redo")
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("how", ["typed_x", "scatter"])
def test_a_typed_x_or_scatter_that_overwrites_the_last_radian_walls_goes_wholesale(how, art, monkeypatch) -> None:
    """The case above through the Inspector's typed X and Scatter's move
    mode, which plan with no gate part: the radian run's middle wall leaves
    for (56, 50), so all three walls are towers written as literals over the
    file's only radian values. The pair at (60, 56) repaints only through
    the wholesale refresh. Undo flips the file back."""
    from descape.scatter_dialog import ScatterParams

    window = _window("Stepped")
    try:
        walls = _wall_run(window, variant=1, const=PALISADE)
        _literal_walls(window, {(60, 56): 1, (61, 56): 1})
        assert _file_is_radian(window)
        _show_level(window, monkeypatch, 0)
        before = _level(window, 0)
        entry = _select(window, walls[1])

        if how == "typed_x":
            window.units_panel.unit_field_editors["x"].setValue(56.5)
        else:
            params = ScatterParams(
                tiles=[(56, 50)], unit_const=None, player=1, count=None, density=None, seed=7, jitter=0.0,
                min_spacing=0, move_selected=True,
            )
            window._scatter_selected_units([entry], params)

        assert (int(walls[1].x), int(walls[1].y)) == (56, 50)
        assert [w.rotation for w in walls] == [2.0, 2.0, 2.0]
        assert not _file_is_radian(window)
        assert window.edit_history.peek_undo().unit_field_entries is None, "the flip did not go wholesale"
        _check(window, 0, how)
        window.undo()
        assert _file_is_radian(window)
        assert np.array_equal(_check(window, 0, "undo"), before)
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize(("style", "mip"), STYLE_MIPS)
def test_a_gate_stepping_off_a_run_reshapes_it_by_splice(style, mip, art, monkeypatch) -> None:
    """GH #194 with "Moving a gate updates walls" ticked, and only reshapes:
    the gate steps from beside the run's middle to the row beyond, so
    (52..55, 50) go 2 to 0. A plan with rewrites only is a fields-only
    splice edit since GH #266."""
    from PyQt5.QtCore import Qt

    window = _window(style)
    try:
        walls = _literal_walls(window, {(x, 50): 2 if x in (50, 57) or 52 <= x <= 55 else 0 for x in range(50, 58)})
        gate = _add(window, 1, GATE_NE, 54.0, 51.5)
        window.gate_walls_check.setChecked(True)
        _show_level(window, monkeypatch, mip)
        _level(window, mip)
        _select(window, gate)

        window.on_unit_nudge(0, 1, Qt.ShiftModifier)

        assert gate.y == 52.5
        assert [walls[(x, 50)].rotation for x in range(50, 58)] == [2.0] + [0.0] * 6 + [2.0]
        assert window.status_log.toPlainText().splitlines()[-1].endswith(", reshaping 4 adjacent")
        assert window.edit_history.peek_undo().unit_field_entries is not None, "not a fields-only record"
        _check(window, mip, "gate step")
        window.undo()
        _check(window, mip, "undo")
    finally:
        conftest.close_window(window)


def test_a_wholesale_wall_move_rebuilds_no_off_screen_level(art, monkeypatch) -> None:
    """Group-move-wall plan Step 1: a wall Move whose sources rebuild wholesale
    (the component cap at 0 keeps it wholesale now that moved walls splice),
    mips -2/-1/0 resident and 0 on screen. The reach bbox patches mip 0 alone
    and evicts itself from -2 and -1, which stay stale for LevelWarmer instead
    of rebuilding inside the handler. A forced read then matches a fresh cache."""
    monkeypatch.setattr(render_cache, "_COMPONENT_SPLICE_MAX_UNITS", 0)
    monkeypatch.setattr(render_cache, "VISIBLE_SPLICE_MAX_UNITS", 0)
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


def _moved_tree(mip: int):
    """_bare_cache() with one oak, level `mip` and level 0 rendered whole,
    then the oak moved three tiles east: (cache, its splice)."""
    tree = _Unit(30.5, 30.5, TREE_OAK)
    cache = _bare_cache(tree)
    for level in {mip, 0}:
        cache.render_rect(0, 0, *cache.canvas_dims(level), mip=level)
    tree.x = 33.5
    return cache, [UnitSplice(0, 0, tree, (30, 30), (33, 30), ((30, 30),), ((33, 30),))]


@pytest.mark.parametrize("mip", [0, -1])
def test_a_spliced_level_answers_the_after_extent_from_its_layer(mip, art, layer_extents) -> None:
    cache, changed = _moved_tree(mip)
    cache.invalidate_units(changed)
    assert cache.sprite_extent_after(changed, mip) is not None
    assert layer_extents == [(mip, 0)], "the after extent resolved the unit instead of reading the layer"


def test_a_stale_level_still_resolves_the_after_extent(art, layer_extents) -> None:
    cache, changed = _moved_tree(0)
    cache.invalidate_units()  # wholesale: every level stale
    assert cache._levels[0].gen != cache._source_gen
    assert cache.sprite_extent_after(changed, 0) == cache._resolved_extent(changed, 0) is not None
    assert layer_extents == []


def test_a_pending_level_still_takes_the_reach_fallback(art, layer_extents) -> None:
    """Level 0 defers an Elevate under the oak (only -1 splices it), so its
    layer is behind when the oak then moves."""
    from descape.elevation_tools import set_tiles_elevation

    tree = _Unit(30.5, 30.5, TREE_OAK)
    cache = _bare_cache(tree)
    for level in (0, -1):
        cache.render_rect(0, 0, *cache.canvas_dims(level), mip=level)
    scenario = cache.scenario
    mm = scenario.map_manager
    set_tiles_elevation(mm, [(30, 30, mm.get_tile(30, 30).elevation + 1)])
    elevation_changed: set = set()
    bbox = render.dirty_screen_bbox_iso(
        scenario, [30 * mm.map_width + 30], cache.elevations, cache.proj, with_units=True, with_sprites=True,
        elevation_changed=elevation_changed,
    )
    cache.patch(bbox, elevation_changed=elevation_changed, rebuild_levels=(-1,))
    assert cache._levels[0].pending_elev, "level 0 did not defer the raise"
    tree.x = 33.5
    changed = [UnitSplice(0, 0, tree, (30, 30), (33, 30), ((30, 30),), ((33, 30),))]
    cache.invalidate_units(changed)
    assert cache.sprite_extent_after(changed, 0) is REACH_FALLBACK
    assert layer_extents == []


def test_the_layer_extent_oracle_fails_on_a_layer_read_before_its_splice(art) -> None:
    """Control: the moved oak read from level 0's layer before the edit
    reached it (sprite_extent_after() forced onto the layer of a stale level)
    misses the resolved extent, and the autouse oracle says so."""
    cache, changed = _moved_tree(0)
    cache.invalidate_units()
    cache._layer_current = lambda mip: True
    with pytest.raises(AssertionError, match="misses the resolved"):
        cache.sprite_extent_after(changed, 0)


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


FAR_OAK, BARE_LAYERED = (80, 60), (60, 90)


class _NoMarks(set):
    """A stroke's pending_stale_levels that ignores _erase_units()'s mark."""

    def update(self, *_):
        pass


@pytest.mark.parametrize("mark", [True, False], ids=["mark", "no_mark"])
def test_an_erase_flush_over_a_stale_level_still_repaints_its_erased_sprites(art, monkeypatch, mark) -> None:
    """Two flushes of one Eraser drag, Trees and Terrain Overlay ticked. The
    first is over the splice cap and evicts, which leaves the visible level
    stale. The second takes one far oak and clears one layer. The oak is
    erased over that stale level, so _erase_units()'s mark sizes its batch
    by reach (TASK-318). The `no_mark` case pins the flush's units-first
    order: tiles first, the cleared tile's _apply_dirty() rebuilds the
    level without the oak before the unit step reads it, and the oak stays
    in its resident chunk."""
    monkeypatch.setattr(render_cache, "UNIT_SPLICE_MAX_UNITS", 4)
    monkeypatch.setattr(render_cache, "VISIBLE_SPLICE_MAX_UNITS", 0)  # the stale level comes from the gen bump
    _ratio(monkeypatch, "evict")
    window = _window("Stepped")
    try:
        window.mode_combo.setCurrentText("Terrain")
        window._on_tool_selected("draw")
        window.paint_trees_check.setChecked(True)
        window.paint_eye_candy_check.setChecked(False)
        _show_level(window, monkeypatch, 0)
        _draw(window, FOREST_OAK, STROKE_TILES)
        _draw(window, FOREST_OAK, [FAR_OAK])
        mm = window.scenario.map_manager
        bare = BARE_LAYERED[1] * mm.map_width + BARE_LAYERED[0]
        mm.terrain[bare].layer = FOREST_PALM
        window._apply_dirty([bare])
        _check(window, 0, "setup")
        trees = len(window.scenario.unit_manager.units[0])

        window._on_tool_selected("eraser")
        window.brush_size_spin.setValue(1)
        for name, action in window.eraser_target_actions.items():
            action.setChecked(name in ("trees", "overlay"))
        window.on_edit_stroke_start()
        assert window._erase is not None and window._erase.overlay
        window.on_edit_stroke_tiles(STROKE_TILES[:5], 0)
        window._erase_refresh_timer.stop()
        window._flush_erase_refresh()
        _ratio(monkeypatch, "patch")
        if not mark:
            window._erase.pending_stale_levels = _NoMarks()
        window.on_edit_stroke_tiles([BARE_LAYERED, FAR_OAK], 0)
        window._erase_refresh_timer.stop()
        assert bool(window._erase.pending_stale_levels) == mark
        window._flush_erase_refresh()
        assert mm.terrain[bare].layer == -1 and len(window.scenario.unit_manager.units[0]) == trees - 6
        _check(window, 0, "second flush")
        window.on_edit_stroke_end()
        _check(window, 0, "erase end")
    finally:
        conftest.close_window(window)


def _stale_erase_window(style: str, monkeypatch, overlay: bool, extra_level: int | None = None):
    """The flush test's setup up to its first, over-cap flush (TASK-318):
    trees on STROKE_TILES and FAR_OAK, BARE_LAYERED layered when `overlay`,
    mip 0 on screen. Leaves an Eraser stroke open with the visible level
    stale and the patch ratio in force. extra_level is rendered whole before
    the drag, so it is resident and goes stale too."""
    monkeypatch.setattr(render_cache, "UNIT_SPLICE_MAX_UNITS", 4)
    monkeypatch.setattr(render_cache, "VISIBLE_SPLICE_MAX_UNITS", 0)
    _ratio(monkeypatch, "evict")
    window = _window(style)
    try:
        window.mode_combo.setCurrentText("Terrain")
        window._on_tool_selected("draw")
        window.paint_trees_check.setChecked(True)
        window.paint_eye_candy_check.setChecked(False)
        _show_level(window, monkeypatch, 0)
        _draw(window, FOREST_OAK, STROKE_TILES)
        _draw(window, FOREST_OAK, [FAR_OAK])
        if overlay:
            mm = window.scenario.map_manager
            bare = BARE_LAYERED[1] * mm.map_width + BARE_LAYERED[0]
            mm.terrain[bare].layer = FOREST_PALM
            window._apply_dirty([bare])
        _check(window, 0, "setup")
        if extra_level is not None:
            _level(window, extra_level)
        window._on_tool_selected("eraser")
        window.brush_size_spin.setValue(1)
        targets = ("trees", "overlay") if overlay else ("trees",)
        for name, action in window.eraser_target_actions.items():
            action.setChecked(name in targets)
        window.on_edit_stroke_start()
        window.on_edit_stroke_tiles(STROKE_TILES[:5], 0)
        window._erase_refresh_timer.stop()
        window._flush_erase_refresh()
        _ratio(monkeypatch, "patch")
        assert not window._cache.is_level_resident(0), "the over-cap flush left the visible level current"
    except BaseException:
        conftest.close_window(window)
        raise
    return window


def _assert_stale_with_oak_resident(window, mip: int) -> None:
    """The precondition of every route: level `mip` is stale and FAR_OAK's
    chunks there are still cached, so its rebuild can drop the oak under them."""
    cache = window._cache
    assert not cache.is_level_resident(mip), f"level {mip} is current"
    box = cache._levels[mip].building_bboxes.get(FAR_OAK)
    assert box is not None, f"level {mip}'s layer has no oak"
    cx0, cy0, cx1, cy1 = cache.chunk_index_range(mip, *box)
    assert any(cache.has_chunk(mip, cx, cy) for cx in range(cx0, cx1 + 1) for cy in range(cy0, cy1 + 1)), (
        f"the oak's chunks at level {mip} were evicted"
    )


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_an_erase_then_a_paint_over_a_stale_level_repaints_the_erased_sprite(style, art, monkeypatch) -> None:
    """TASK-318's paint route: a composite between an erase and its flush
    rebuilds the stale visible level from the live scenario, so its layer no
    longer holds the erased oak while the oak's chunk still shows it."""
    window = _stale_erase_window(style, monkeypatch, overlay=False)
    try:
        window.on_edit_stroke_tiles([FAR_OAK], 0)
        window._erase_refresh_timer.stop()
        _assert_stale_with_oak_resident(window, 0)
        _level(window, 0)
        window._flush_erase_refresh()
        _check(window, 0, "flush after a paint")
        window.on_edit_stroke_end()
        _check(window, 0, "erase end")
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_an_erase_released_with_its_splices_pending_repaints_the_erased_sprite(style, art, monkeypatch) -> None:
    """TASK-318's release route: the stroke end with the oak's splice still
    pending and a layer cleared, so the end's tile patch could rebuild the
    stale visible level before the unit step reads the oak's extent."""
    window = _stale_erase_window(style, monkeypatch, overlay=True)
    try:
        window.on_edit_stroke_tiles([BARE_LAYERED, FAR_OAK], 0)
        window._erase_refresh_timer.stop()
        assert window._erase.pending_splices and window._erase.pending_layer_tiles
        _assert_stale_with_oak_resident(window, 0)
        window.on_edit_stroke_end()
        _check(window, 0, "erase end")
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_an_erase_then_a_zoom_onto_a_stale_level_repaints_the_erased_sprite(style, art, monkeypatch) -> None:
    """TASK-318's zoom route: level -1 resident before the drag goes stale
    with mip 0, becomes the visible level inside the erase-to-flush window,
    and its paint rebuilds it without the oak."""
    window = _stale_erase_window(style, monkeypatch, overlay=False, extra_level=-1)
    try:
        window.on_edit_stroke_tiles([FAR_OAK], 0)
        window._erase_refresh_timer.stop()
        _show_level(window, monkeypatch, -1)
        _assert_stale_with_oak_resident(window, -1)
        _level(window, -1)
        window._flush_erase_refresh()
        _check(window, -1, "flush after a zoom")
    finally:
        conftest.close_window(window)
