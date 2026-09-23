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
from descape.render_cache import FlatChunkCache, IsoChunkCache, SlopedChunkCache, UnitSplice, _splice_eligible
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
from descape.terrain_palette import BUILDING_TILE_SPANS
from descape.unit_filter import UnitFilter

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


def _make_cache(style: str, scenario, sprites: bool = False, unit_filter: UnitFilter = UnitFilter()):
    mm = scenario.map_manager
    tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
    if style == "stepped":
        elevations, proj = elevations_and_proj(scenario)
        cache = IsoChunkCache(scenario, elevations, proj, tile_px, sprites=sprites, unit_filter=unit_filter)
    else:
        elevations, corner_rise, proj = sloped_elevations_and_proj(scenario)
        cache = SlopedChunkCache(
            scenario, elevations, corner_rise, proj, tile_px, sprites=sprites, unit_filter=unit_filter
        )
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


# --- Convert's reassign splice (convert-and-flat-unit-splice plan, Step 1) ---


def _reassign_splice(scenario, source: int, destination: int, unit) -> UnitSplice:
    """What UnitEditModel.reassign() does to the lists (delete by identity,
    append to the destination), plus the splice the Convert stroke builds."""
    own, tiles = _own_tile(unit), _occupied(scenario, unit)
    units = scenario.unit_manager.units
    index = next(i for i, u in enumerate(units[source]) if u is unit)
    del units[source][index]
    units[destination].append(unit)
    return UnitSplice(
        destination, len(units[destination]) - 1, unit, own, own, tiles, tiles, old_player_id=source
    )


def _fresh_render(style: str, scenario, sprites: bool, unit_filter: UnitFilter) -> np.ndarray:
    cache = _make_cache(style, scenario, sprites=sprites, unit_filter=unit_filter)
    return cache.render_rect(0, 0, *cache.canvas_dims(0), mip=0)


@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_lone_building_reassign_splices_and_recolours(style, monkeypatch):
    scenario = _scenario()
    _place(scenario, 1, MILL_CONST, *MILL_TILE)  # stays behind, so the source list reorders
    unit = _place(scenario, 1, MILL_CONST, *ELSEWHERE_TILE)
    cache = _make_cache(style, scenario)
    canvas_w, canvas_h = cache.canvas_dims(0)
    before = cache.render_rect(0, 0, canvas_w, canvas_h, mip=0).copy()
    counts = _call_counts(monkeypatch)

    splice = _reassign_splice(scenario, 1, 2, unit)
    assert _splice_eligible(cache.units_by_tile, splice), "fixture is not testing the splice path"
    _apply(cache, splice)

    assert counts == {"building_bboxes": 0, "sprites": 0}
    stitched = cache.render_rect(0, 0, canvas_w, canvas_h, mip=0)
    assert not np.array_equal(stitched, before), "the reassign did not recolour anything"
    full = _oracle(style, scenario)
    assert np.array_equal(stitched, full[:canvas_h, :canvas_w])


@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_sprite_bearing_reassign_retints_and_matches_a_fresh_render(style, sprite_install, monkeypatch):  # noqa: F811
    """The sprite is team-tinted, so a reassign really dirties it: the splice
    re-resolves it under the destination's team index."""
    scenario = _scenario()
    unit = _place(scenario, 1, SPRITE_CONST, *ELSEWHERE_TILE)
    assert scenario.team_indices[1] != scenario.team_indices[2], "players 1 and 2 share a tint"
    cache = _make_cache(style, scenario, sprites=True)
    canvas_w, canvas_h = cache.canvas_dims(0)
    before = cache.render_rect(0, 0, canvas_w, canvas_h, mip=0).copy()
    counts = _call_counts(monkeypatch)

    splice = _reassign_splice(scenario, 1, 2, unit)
    assert _splice_eligible(cache.units_by_tile, splice)
    _apply(cache, splice)

    assert counts == {"building_bboxes": 0, "sprites": 0}
    stitched = cache.render_rect(0, 0, canvas_w, canvas_h, mip=0)
    assert not np.array_equal(stitched, before), "the reassign did not retint the sprite"
    full = _oracle(style, scenario, sprites=True)
    assert np.array_equal(stitched, full[:canvas_h, :canvas_w])


@pytest.mark.parametrize("style", ["stepped", "sloped"])
@pytest.mark.parametrize(("shown", "visible_after"), [({1}, False), ({2}, True)])
def test_a_reassign_across_the_player_filter_matches_a_fresh_cache(style, shown, visible_after, monkeypatch):
    """Destination hidden (the unit vanishes) and source hidden (it appears)."""
    scenario = _scenario()
    unit = _place(scenario, 1, MILL_CONST, *ELSEWHERE_TILE)
    unit_filter = UnitFilter(players=frozenset(shown))
    cache = _make_cache(style, scenario, unit_filter=unit_filter)
    canvas_w, canvas_h = cache.canvas_dims(0)
    counts = _call_counts(monkeypatch)

    splice = _reassign_splice(scenario, 1, 2, unit)
    assert _splice_eligible(cache.units_by_tile, splice)
    _apply(cache, splice)

    assert counts == {"building_bboxes": 0, "sprites": 0}
    in_tiles = any(e[0] is unit for t in splice.new_tiles for e in cache.units_by_tile.get(t, ()))
    assert in_tiles == visible_after
    stitched = cache.render_rect(0, 0, canvas_w, canvas_h, mip=0)
    assert np.array_equal(stitched, _fresh_render(style, scenario, False, unit_filter))


# --- Flat's row splice (convert-and-flat-unit-splice plan, Step 2) ----------

FLAT_MIPS = (0, 1)


def _flat_cache(scenario, sprites: bool = False, unit_filter: UnitFilter = UnitFilter()) -> FlatChunkCache:
    """A Flat cache with FLAT_MIPS resident: every level painted once, so each
    holds draws (and icons, with sprites) the splice must keep in step."""
    mm = scenario.map_manager
    cache = FlatChunkCache(
        scenario, tile_pixels_for_map(mm.map_width, mm.map_height), sprites=sprites, unit_filter=unit_filter
    )
    _flat_renders(cache)
    return cache


def _flat_counts(monkeypatch):
    """Counts the two wholesale Flat walks. _flat_unit_draws and
    _flat_icon_layer both go through these, so zero means no rebuild at all."""
    counts = {"rows": 0, "icons": 0}
    real_rows, real_icons = render._flat_unit_rows, render._flat_icon_layer_sliced

    def counted_rows(*args, **kwargs):
        counts["rows"] += 1
        return real_rows(*args, **kwargs)

    def counted_icons(*args, **kwargs):
        counts["icons"] += 1
        return real_icons(*args, **kwargs)

    monkeypatch.setattr(render, "_flat_unit_rows", counted_rows)
    monkeypatch.setattr(render, "_flat_icon_layer_sliced", counted_icons)
    return counts


FLAT_VIEW_TILES = (8, 48)  # every Flat fixture unit sits inside this square of tiles


def _flat_renders(cache) -> list[np.ndarray]:
    lo, hi = FLAT_VIEW_TILES
    renders = []
    for mip in FLAT_MIPS:
        tp = cache.mip_tile_px(mip)
        renders.append(cache.render_rect(lo * tp, lo * tp, hi * tp, hi * tp, mip=mip))
    return renders


def _assert_flat_matches_fresh(cache, scenario, sprites: bool, unit_filter: UnitFilter = UnitFilter()) -> None:
    fresh = _flat_cache(scenario, sprites=sprites, unit_filter=unit_filter)
    assert np.array_equal(cache._row_uid, fresh._row_uid)
    assert np.array_equal(cache._row_player, fresh._row_player)
    for mip, got, want in zip(FLAT_MIPS, _flat_renders(cache), _flat_renders(fresh), strict=True):
        assert np.array_equal(got, want), f"mip {mip} differs from a fresh cache"


def _flat_scenario(const: int):
    """Three player-1 units, one each for GAIA and players 2 and 3, all on
    distinct tiles, so every block has neighbours on both sides."""
    scenario = _scenario()
    for i, player in enumerate((0, 1, 1, 1, 2, 3)):
        _place(scenario, player, const, 12.5 + 4 * i, 30.5)
    return scenario


def _flat_edit(op: str, scenario, const: int) -> list[UnitSplice]:
    units = scenario.unit_manager.units
    target = units[1][1]
    if op == "move":
        return [_move_splice(scenario, 1, 1, target, 40.5, 44.5)]
    if op == "add":
        return [_add_splice(scenario, 2, _place(scenario, 2, const, 44.5, 20.5))]
    if op == "delete":
        return [_delete_splice(scenario, 1, 1, target)]
    # To player 2, not 3: the synthetic sprite's team 1 and 3 tints are pixel-identical.
    return [_reassign_splice(scenario, 1, 2, target)]


@pytest.mark.parametrize("sprites", [False, True])
@pytest.mark.parametrize("op", ["move", "add", "delete", "reassign"])
def test_a_flat_row_splice_matches_a_fresh_cache_at_every_resident_level(op, sprites, request, monkeypatch):
    if sprites:
        request.getfixturevalue("sprite_install")
    const = SPRITE_CONST if sprites else MILL_CONST
    scenario = _flat_scenario(const)
    cache = _flat_cache(scenario, sprites=sprites)
    before = [r.copy() for r in _flat_renders(cache)]
    if sprites:
        assert all(len(cache._level_icon_layers[mip]) == 6 for mip in FLAT_MIPS), "icons are not resolving"
    counts = _flat_counts(monkeypatch)

    splices = _flat_edit(op, scenario, const)
    assert cache._flat_splice_eligible(splices), "fixture is not testing the splice path"
    _apply(cache, splices[0])

    after = _flat_renders(cache)
    assert counts == {"rows": 0, "icons": 0}, "the splice path rebuilt a whole walk"
    assert not np.array_equal(after[0], before[0]), "the edit changed nothing on screen"
    _assert_flat_matches_fresh(cache, scenario, sprites)


@pytest.mark.parametrize("sprites", [False, True])
def test_a_multi_player_reassign_batch_keeps_the_row_table_exact(sprites, request, monkeypatch):
    """Destinations both below and above their sources, several per block, one
    unit hidden by the filter on each side: the side table and every level's
    draws must equal a fresh walk exactly, row for row."""
    if sprites:
        request.getfixturevalue("sprite_install")
    const = SPRITE_CONST if sprites else MILL_CONST
    scenario = _scenario()
    for player in (1, 3, 5, 6):
        for i in range(3):
            _place(scenario, player, const, 10.5 + 4 * player, 10.5 + 4 * i)
    unit_filter = UnitFilter(players=frozenset({1, 3, 5}))
    cache = _flat_cache(scenario, sprites=sprites, unit_filter=unit_filter)
    counts = _flat_counts(monkeypatch)
    units = scenario.unit_manager.units

    splices = [
        _reassign_splice(scenario, 3, 1, units[3][0]),   # down
        _reassign_splice(scenario, 3, 5, units[3][0]),   # up, and 3's list shifted under it
        _reassign_splice(scenario, 1, 5, units[1][1]),
        _reassign_splice(scenario, 5, 1, units[5][0]),
        _reassign_splice(scenario, 6, 1, units[6][2]),   # hidden source, visible destination
        _reassign_splice(scenario, 1, 6, units[1][0]),   # visible source, hidden destination
    ]
    assert cache._flat_splice_eligible(splices)
    cache.invalidate_units(splices)
    cache.invalidate_region((0, 0, *cache.canvas_dims(0)))  # as _apply() does
    assert counts == {"rows": 0, "icons": 0}

    for mip in FLAT_MIPS:
        bboxes, colors, row_uid, row_player = render._flat_unit_rows(scenario, cache.mip_tile_px(mip), unit_filter)
        got_bboxes, got_colors = cache._level_unit_draws(mip)
        assert np.array_equal(got_bboxes, bboxes) and np.array_equal(got_colors, colors), f"mip {mip}"
        assert np.array_equal(cache._row_uid, row_uid) and np.array_equal(cache._row_player, row_player)
        if sprites:
            icons, rows = render._flat_icon_layer(scenario, cache.mip_tile_px(mip), unit_filter)
            assert sorted(cache._level_icon_layers[mip]) == sorted(icons) and rows == len(row_uid)
    _assert_flat_matches_fresh(cache, scenario, sprites, unit_filter)


def test_a_flat_splice_refuses_an_in_flight_warm(sprite_install, monkeypatch):  # noqa: F811
    scenario = _flat_scenario(SPRITE_CONST)
    mm = scenario.map_manager
    cache = FlatChunkCache(scenario, tile_pixels_for_map(mm.map_width, mm.map_height), sprites=True)
    cache.render_rect(0, 0, *cache.canvas_dims(0), mip=0)
    job = cache.level_warm_job(1)
    assert job is not None

    splices = _flat_edit("move", scenario, SPRITE_CONST)
    assert cache._flat_splice_eligible(splices)
    cache.invalidate_units(splices)

    assert job.install(render._drain(job.gen)) is False
    assert not cache.is_level_resident(1)


@pytest.mark.parametrize("case", ["wall", "duplicate", "off_map"])
def test_ineligible_flat_batches_fall_back_to_wholesale(case, monkeypatch):
    scenario = _flat_scenario(MILL_CONST)
    units = scenario.unit_manager.units
    if case == "wall":
        _place(scenario, 1, WALL_CONST, 14.5, 40.5)
    cache = _flat_cache(scenario)
    counts = _flat_counts(monkeypatch)

    if case == "wall":
        splices = [_move_splice(scenario, 1, 3, units[1][3], 20.5, 40.5)]
    elif case == "duplicate":
        first = _move_splice(scenario, 1, 1, units[1][1], 40.5, 44.5)
        splices = [first, _move_splice(scenario, 1, 1, units[1][1], 41.5, 44.5)]
    else:
        splices = [_move_splice(scenario, 1, 1, units[1][1], -40.0, -40.0)]
    assert not cache._flat_splice_eligible(splices)
    cache.invalidate_units(splices)
    cache.invalidate_region((0, 0, *cache.canvas_dims(0)))

    assert counts["rows"] >= 1, "an ineligible batch should have rebuilt the rows"
    _assert_flat_matches_fresh(cache, scenario, False)
