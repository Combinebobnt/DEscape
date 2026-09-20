"""Batch D's D4: invalidate_units(changed) splices units_by_tile/
building_bboxes/sprites for a single-unit edit instead of paying
_refresh_source_caches()'s full O(all units) rebuild -- see
render_cache.UnitSplice/_splice_eligible's own docstrings for the guard
this file exercises.

Fixture posture matches tests/test_patch_unit_sources.py: BLANK_TEMPLATE_PATH
(tracked, 120x120) with duck-typed units appended directly, not corpus.

Tests are ordered cheapest-first, oracle last per style: the non-vacuity
tests (shared-tile / move-onto-occupied fallback) matter more read early,
since a guard that always fires would make every splice test pass for the
wrong reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from test_sprite_edit_bbox import sprite_install  # noqa: F401 -- pytest fixture, imported for its name

from descape import render, unit_sprites
from descape.render import (
    elevations_and_proj,
    render_terrain_iso_with_proj,
    render_terrain_sloped_with_proj,
    sloped_elevations_and_proj,
    tile_pixels_for_map,
)
from descape.render_cache import IsoChunkCache, SlopedChunkCache, UnitSplice, _splice_eligible
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
from descape.terrain_palette import BUILDING_TILE_SPANS

from test_unit_sprites import CONST as SPRITE_CONST

MILL_CONST = 68  # BUILDING_TILE_SPANS[68] == (2, 2) -- a real multi-tile building
WALL_CONST = 63  # a real unit_sprites.wall_connector_consts() member, span (4, 1)
assert BUILDING_TILE_SPANS[MILL_CONST] == (2, 2)
assert WALL_CONST in unit_sprites.wall_connector_consts()

BASE_ELEVATION = 5
MILL_TILE = (60.0, 60.0)
ELSEWHERE_TILE = (10.0, 10.0)
MOVED_TILE = (61.0, 60.0)


@dataclass
class Unit:
    """The attributes _units_by_tile/_building_bboxes_iso/sprite_draws_by_anchor
    actually read -- duck-typed, matching tests/test_patch_unit_sources.py's
    own fixture."""

    x: float
    y: float
    unit_const: int
    rotation: float = 0.0


def _scenario():
    scenario = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for t in scenario.map_manager.terrain:
        t.elevation = BASE_ELEVATION
    return scenario


def _own_tile(unit) -> tuple[int, int]:
    return (int(unit.x), int(unit.y))


def _occupied(scenario, unit) -> tuple[tuple[int, int], ...]:
    mm = scenario.map_manager
    tiles = render.unit_occupied_tiles(unit, mm.map_width, mm.map_height)
    return tuple(tiles) if tiles is not None else ()


def _place(scenario, player_id: int, unit_const: int, x: float, y: float) -> Unit:
    unit = Unit(x, y, unit_const)
    scenario.unit_manager.units[player_id].append(unit)
    return unit


def _move_splice(scenario, player_id: int, index: int, unit, new_x: float, new_y: float) -> UnitSplice:
    old_own, old_tiles = _own_tile(unit), _occupied(scenario, unit)
    unit.x, unit.y = new_x, new_y
    return UnitSplice(player_id, index, unit, old_own, _own_tile(unit), old_tiles, _occupied(scenario, unit))


def _add_splice(scenario, player_id: int, unit) -> UnitSplice:
    index = scenario.unit_manager.units[player_id].index(unit)
    return UnitSplice(player_id, index, unit, None, _own_tile(unit), (), _occupied(scenario, unit))


def _delete_splice(scenario, player_id: int, index: int, unit) -> UnitSplice:
    old_own, old_tiles = _own_tile(unit), _occupied(scenario, unit)
    del scenario.unit_manager.units[player_id][index]
    return UnitSplice(player_id, index, unit, old_own, None, old_tiles, ())


def _make_cache(style: str, scenario, sprites: bool = False):
    mm = scenario.map_manager
    tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
    if style == "stepped":
        elevations, proj = elevations_and_proj(scenario)
        cache = IsoChunkCache(scenario, elevations, proj, tile_px, sprites=sprites)
    else:
        elevations, corner_rise, proj = sloped_elevations_and_proj(scenario)
        cache = SlopedChunkCache(scenario, elevations, corner_rise, proj, tile_px, sprites=sprites)
    cache.render_rect(0, 0, *cache.canvas_dims(0), mip=0)
    return cache


def _oracle(style: str, scenario, sprites: bool = False) -> np.ndarray:
    if style == "stepped":
        full, _elev, _proj = render_terrain_iso_with_proj(scenario, with_units=True, with_sprites=sprites)
        return full
    full, _elev, _corner_rise, _proj = render_terrain_sloped_with_proj(
        scenario, with_units=True, with_sprites=sprites
    )
    return full


def _call_counts(monkeypatch):
    """Patches the two expensive wholesale walks and returns a dict this
    test can read after the fact -- a real _refresh_source_caches() must
    call both at least once, so zero on either after invalidate_units()
    is the splice-fired signal."""
    counts = {"building_bboxes": 0, "sprites": 0}
    real_bboxes = render._building_bboxes_iso
    real_sprites = render.sprite_draws_by_anchor

    def counted_bboxes(*args, **kwargs):
        counts["building_bboxes"] += 1
        return real_bboxes(*args, **kwargs)

    def counted_sprites(*args, **kwargs):
        counts["sprites"] += 1
        return real_sprites(*args, **kwargs)

    monkeypatch.setattr(render, "_building_bboxes_iso", counted_bboxes)
    monkeypatch.setattr(render, "sprite_draws_by_anchor", counted_sprites)
    return counts


def _apply(cache, splice: UnitSplice) -> None:
    """invalidate_units() alone only refreshes SOURCE state -- units are
    baked into already-cached chunk PIXELS (see _ChunkCacheBase.
    set_unit_filter's own docstring for the same trap), so a real caller
    (_after_unit_mutation) always pairs it with invalidate_region() over
    the whole canvas. Mirrored here so this file's pixel oracles test the
    splice, not a stale warm chunk."""
    cache.invalidate_units([splice])
    cache.invalidate_region((0, 0, *cache.canvas_dims(0)))


@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_wall_const_always_falls_back(style, monkeypatch):
    """A wall/connector const's override is a function of its neighbours --
    _splice_eligible() must reject it regardless of whether any tile is
    actually shared."""
    scenario = _scenario()
    unit = _place(scenario, 1, WALL_CONST, *ELSEWHERE_TILE)
    cache = _make_cache(style, scenario)
    canvas_w, canvas_h = cache.canvas_dims(0)
    counts = _call_counts(monkeypatch)

    splice = _move_splice(scenario, 1, 0, unit, *MOVED_TILE)
    assert not _splice_eligible(cache.units_by_tile, splice)
    _apply(cache, splice)

    # IsoChunkCache's wholesale path is lazy (its own _source_gen-gated
    # _level()): _building_bboxes_iso doesn't run until the next composite,
    # so the count assertion below must follow a repaint, not precede it.
    stitched = cache.render_rect(0, 0, canvas_w, canvas_h, mip=0)
    assert counts["building_bboxes"] >= 1, "wall const should have taken the wholesale fallback"
    full = _oracle(style, scenario)
    assert np.array_equal(stitched, full[:canvas_h, :canvas_w])


@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_shared_own_tile_falls_back(style, monkeypatch):
    """Two Mills stacked on the same tile -- a real corpus shape (5 of 16
    example files carry >1 unit sharing an own-tile). Moving one must not
    take the splice path, since building_bboxes/by_anchor union both
    Mills' data at that key with no way to subtract just one back out."""
    scenario = _scenario()
    _place(scenario, 1, MILL_CONST, *MILL_TILE)  # stays put, index 0
    mover = _place(scenario, 1, MILL_CONST, *MILL_TILE)  # index 1, same tile
    cache = _make_cache(style, scenario)
    canvas_w, canvas_h = cache.canvas_dims(0)
    counts = _call_counts(monkeypatch)

    splice = _move_splice(scenario, 1, 1, mover, *MOVED_TILE)
    assert not _splice_eligible(cache.units_by_tile, splice)
    _apply(cache, splice)
    cache.render_rect(0, 0, canvas_w, canvas_h, mip=0)  # IsoChunkCache's fallback is lazy, see sibling test

    assert counts["building_bboxes"] >= 1, "a shared own-tile should have taken the wholesale fallback"


@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_move_onto_occupied_tile_falls_back(style, monkeypatch):
    """The other direction of the same guard: moving ONTO a tile another
    unit already occupies must also fall back, not just moving away from a
    shared one."""
    scenario = _scenario()
    _place(scenario, 1, MILL_CONST, *MILL_TILE)  # index 0, stationary target
    mover = _place(scenario, 1, MILL_CONST, *ELSEWHERE_TILE)  # index 1
    cache = _make_cache(style, scenario)
    canvas_w, canvas_h = cache.canvas_dims(0)
    counts = _call_counts(monkeypatch)

    splice = _move_splice(scenario, 1, 1, mover, *MILL_TILE)
    assert not _splice_eligible(cache.units_by_tile, splice)
    _apply(cache, splice)
    cache.render_rect(0, 0, canvas_w, canvas_h, mip=0)  # IsoChunkCache's fallback is lazy, see sibling test

    assert counts["building_bboxes"] >= 1, "moving onto an occupied tile should have fallen back"


@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_lone_building_move_splices_and_matches_a_fresh_render(style, monkeypatch):
    scenario = _scenario()
    unit = _place(scenario, 1, MILL_CONST, *ELSEWHERE_TILE)
    cache = _make_cache(style, scenario)
    canvas_w, canvas_h = cache.canvas_dims(0)
    counts = _call_counts(monkeypatch)

    splice = _move_splice(scenario, 1, 0, unit, *MOVED_TILE)
    assert _splice_eligible(cache.units_by_tile, splice), "fixture is not testing the splice path"
    _apply(cache, splice)

    assert counts == {"building_bboxes": 0, "sprites": 0}, (
        "a splice-eligible lone move should never call the wholesale rebuilds"
    )
    stitched = cache.render_rect(0, 0, canvas_w, canvas_h, mip=0)
    assert counts == {"building_bboxes": 0, "sprites": 0}, (
        "a resident, already-spliced level should not re-rebuild on repaint"
    )
    full = _oracle(style, scenario)
    assert np.array_equal(stitched, full[:canvas_h, :canvas_w])


@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_single_unit_add_splices_and_matches_a_fresh_render(style, monkeypatch):
    scenario = _scenario()
    _place(scenario, 1, MILL_CONST, *ELSEWHERE_TILE)  # unrelated occupant, far away
    cache = _make_cache(style, scenario)
    canvas_w, canvas_h = cache.canvas_dims(0)
    counts = _call_counts(monkeypatch)

    new_unit = _place(scenario, 2, MILL_CONST, *MILL_TILE)
    splice = _add_splice(scenario, 2, new_unit)
    assert _splice_eligible(cache.units_by_tile, splice)
    _apply(cache, splice)

    assert counts == {"building_bboxes": 0, "sprites": 0}
    stitched = cache.render_rect(0, 0, canvas_w, canvas_h, mip=0)
    full = _oracle(style, scenario)
    assert np.array_equal(stitched, full[:canvas_h, :canvas_w])


@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_single_unit_delete_splices_and_matches_a_fresh_render(style, monkeypatch):
    scenario = _scenario()
    _place(scenario, 1, MILL_CONST, *ELSEWHERE_TILE)  # unrelated occupant, far away
    doomed = _place(scenario, 2, MILL_CONST, *MILL_TILE)
    cache = _make_cache(style, scenario)
    canvas_w, canvas_h = cache.canvas_dims(0)
    counts = _call_counts(monkeypatch)

    index = scenario.unit_manager.units[2].index(doomed)
    splice = _delete_splice(scenario, 2, index, doomed)
    assert _splice_eligible(cache.units_by_tile, splice)
    _apply(cache, splice)

    assert counts == {"building_bboxes": 0, "sprites": 0}
    stitched = cache.render_rect(0, 0, canvas_w, canvas_h, mip=0)
    full = _oracle(style, scenario)
    assert np.array_equal(stitched, full[:canvas_h, :canvas_w])


def test_units_by_tile_splice_removes_and_adds_exactly_this_unit():
    """A direct check on the spliced dict's shape (not just the rendered
    pixels), so a regression that happens to be pixel-invisible (e.g. a
    stray duplicate entry that would only matter for depth_order tie-
    breaking on a future add) still fails loudly."""
    scenario = _scenario()
    unit = _place(scenario, 1, MILL_CONST, *ELSEWHERE_TILE)
    cache = _make_cache("stepped", scenario)
    old_tiles = _occupied(scenario, unit)

    splice = _move_splice(scenario, 1, 0, unit, *MOVED_TILE)
    _apply(cache, splice)

    for tile in old_tiles:
        assert tile not in cache.units_by_tile or unit not in (e[0] for e in cache.units_by_tile[tile])
    for tile in splice.new_tiles:
        assert any(e[0] is unit for e in cache.units_by_tile[tile])


@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_sprite_bearing_unit_move_splices_and_matches_a_fresh_render(style, sprite_install, monkeypatch):  # noqa: F811
    """Every test above uses sprites=False, which never exercises
    _splice_building_and_sprites()'s `sprites is not None` branch --
    by_anchor/bboxes/skip_ids splicing, the piece of D4 that most needed
    the sprite_anchor_tile()-vs-own-tile reconciliation this file's docstring
    (render_cache._splice_building_and_sprites) describes. A real (if tiny,
    synthetic) .sld via test_sprite_edit_bbox's own fixture is what lets
    this unit actually resolve to sprite pieces instead of a coloured mark."""
    scenario = _scenario()
    unit = _place(scenario, 1, SPRITE_CONST, *ELSEWHERE_TILE)
    cache = _make_cache(style, scenario, sprites=True)
    canvas_w, canvas_h = cache.canvas_dims(0)
    counts = _call_counts(monkeypatch)

    splice = _move_splice(scenario, 1, 0, unit, *MOVED_TILE)
    assert _splice_eligible(cache.units_by_tile, splice), "fixture is not testing the splice path"
    _apply(cache, splice)

    assert counts == {"building_bboxes": 0, "sprites": 0}
    stitched = cache.render_rect(0, 0, canvas_w, canvas_h, mip=0)
    assert counts == {"building_bboxes": 0, "sprites": 0}, (
        "a resident, already-spliced level should not re-rebuild on repaint"
    )
    full = _oracle(style, scenario, sprites=True)
    assert np.array_equal(stitched, full[:canvas_h, :canvas_w])


def _sprite_layer(cache):
    """The cache's live SpriteLayer -- per mip level on Stepped, one on Sloped."""
    return cache._level(0).sprites if hasattr(cache, "_level") else cache.sprites


@pytest.fixture
def slotted_composite_install(tmp_path, monkeypatch):
    """MILL_CONST as a two-piece composite whose pieces carry their own depth
    slots, so one spliced unit owns TWO by_anchor keys rather than one. Same
    oversized canvas and pinned reach constants as sprite_install, and for the
    same reasons (that fixture's own docstring)."""
    from test_sprite_edit_bbox import REACH_NAMES, SPRITE_CANVAS

    graphics = tmp_path / unit_sprites.GRAPHICS_SUBPATH
    graphics.mkdir(parents=True)
    from test_unit_sprites import FILE_NAME, build_sld

    (graphics / f"{FILE_NAME}.sld").write_bytes(build_sld(1, canvas=SPRITE_CANVAS))

    def piece(slot, dy):
        return {"unit_id": MILL_CONST, "file_name": FILE_NAME, "angle_count": 1,
                "frame_count": 1, "dx": 0, "dy": dy, "parent": slot == [1, 0], "slot": slot}

    monkeypatch.setattr(
        unit_sprites, "graphic_map",
        lambda: {MILL_CONST: {
            "graphic_id": 1, "file_name": FILE_NAME, "angle_count": 1,
            "mirroring_mode": 0, "frame_count": 1,
            "pieces": [piece([1, 0], -96), piece([0, 1], 96)],
        }},
    )
    for name in REACH_NAMES:
        monkeypatch.setattr(unit_sprites, name, SPRITE_CANVAS // 2)
    from descape import asset_source

    asset_source.set_install_path_override(tmp_path)
    unit_sprites.clear_caches()
    yield
    asset_source.set_install_path_override(None)
    unit_sprites.clear_caches()


@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_a_slotted_composite_move_splices_every_one_of_its_anchors(
    style, slotted_composite_install, monkeypatch
):
    """A composite contributes to one by_anchor key per depth slot, all inside
    its footprint. Clearing only sprite_anchor_tile(old_tiles) would leave the
    other slot's pieces stranded on the pre-move tile -- visible as a ghost of
    half the building, and invisible to every sprites=False splice test."""
    scenario = _scenario()
    unit = _place(scenario, 1, MILL_CONST, *ELSEWHERE_TILE)
    cache = _make_cache(style, scenario, sprites=True)
    canvas_w, canvas_h = cache.canvas_dims(0)
    layer = _sprite_layer(cache)
    assert len(layer.by_anchor) == 2, "the fixture is not producing two slots"
    counts = _call_counts(monkeypatch)

    splice = _move_splice(scenario, 1, 0, unit, *MOVED_TILE)
    assert _splice_eligible(cache.units_by_tile, splice), "fixture is not testing the splice path"
    _apply(cache, splice)

    assert counts == {"building_bboxes": 0, "sprites": 0}
    assert len(_sprite_layer(cache).by_anchor) == 2, "the moved composite left a stale anchor behind"
    stitched = cache.render_rect(0, 0, canvas_w, canvas_h, mip=0)
    full = _oracle(style, scenario, sprites=True)
    assert np.array_equal(stitched, full[:canvas_h, :canvas_w])
