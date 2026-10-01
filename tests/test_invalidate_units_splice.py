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

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
from test_bystander_grid_patch import grid_state
from test_sprite_edit_bbox import sprite_install  # noqa: F401 -- pytest fixture, imported for its name

from descape import asset_source, debug_log, perf_trace, render, render_cache, unit_sprites
from descape.render import (
    elevations_and_proj,
    render_terrain_iso_with_proj,
    render_terrain_sloped_with_proj,
    sloped_elevations_and_proj,
    tile_pixels_for_map,
)
from descape.render_cache import (
    FlatChunkCache,
    IsoChunkCache,
    SlopedChunkCache,
    UnitSplice,
    _batch_splice_eligible,
    _splice_eligible,
)
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
from descape.terrain_palette import BUILDING_TILE_SPANS
from descape.unit_filter import UnitFilter

import conftest
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
    own fixture. `player` is the list the unit sits in, as UnitManager keeps
    it (the in-place units_by_tile update reads it); _place() sets it."""

    x: float
    y: float
    unit_const: int
    rotation: float = 0.0
    player: int = 0


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
    unit = Unit(x, y, unit_const, player=player_id)
    scenario.unit_manager.units[player_id].append(unit)
    # As UnitEditModel does, so the own-tile index a component splice reads is fresh.
    scenario.unit_gen += 1
    return unit


def _move_splice(scenario, player_id: int, index: int, unit, new_x: float, new_y: float) -> UnitSplice:
    old_own, old_tiles = _own_tile(unit), _occupied(scenario, unit)
    unit.x, unit.y = new_x, new_y
    scenario.unit_gen += 1
    return UnitSplice(player_id, index, unit, old_own, _own_tile(unit), old_tiles, _occupied(scenario, unit))


def _add_splice(scenario, player_id: int, unit) -> UnitSplice:
    # By identity: the dataclass's == would match an equal unit placed earlier.
    index = next(i for i, u in enumerate(scenario.unit_manager.units[player_id]) if u is unit)
    return UnitSplice(player_id, index, unit, None, _own_tile(unit), (), _occupied(scenario, unit))


def _delete_splice(scenario, player_id: int, index: int, unit) -> UnitSplice:
    old_own, old_tiles = _own_tile(unit), _occupied(scenario, unit)
    del scenario.unit_manager.units[player_id][index]
    scenario.unit_gen += 1
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
    # The sliced generator: every build reaches it, drained (_building_bboxes_iso,
    # a paint's _level()) or stepped by a warm job (_assemble_level).
    real_bboxes = render._building_bboxes_iso_sliced
    real_sprites = render.sprite_draws_by_anchor

    def counted_bboxes(*args, **kwargs):
        counts["building_bboxes"] += 1
        return real_bboxes(*args, **kwargs)

    def counted_sprites(*args, **kwargs):
        counts["sprites"] += 1
        return real_sprites(*args, **kwargs)

    monkeypatch.setattr(render, "_building_bboxes_iso_sliced", counted_bboxes)
    monkeypatch.setattr(render, "sprite_draws_by_anchor", counted_sprites)
    return counts


def _apply(cache, splice: UnitSplice) -> None:
    """invalidate_units() alone only refreshes SOURCE state -- units are
    baked into already-cached chunk PIXELS (see _ChunkCacheBase.
    set_unit_filter's own docstring for the same trap), so a real caller
    (_after_unit_mutation) always pairs it with invalidate_region() over
    the whole canvas. Mirrored here so this file's pixel oracles test the
    splice, not a stale warm chunk."""
    _apply_batch(cache, [splice])


def _assert_grid_current(cache) -> None:
    """Every current level's bystander grid against a full build of that
    level's own dict. Not a fresh cache: its wholesale walks would bump the
    _call_counts() this file's splice-fired checks read. Pixels cover the dict."""
    if isinstance(cache, IsoChunkCache):
        levels = [lvl for lvl in cache._levels.values() if lvl.gen == cache._source_gen]
    else:
        levels = [cache]
    for lvl in levels:
        want = render.build_bystander_grid(lvl.building_bboxes, cache.chunk_px)
        assert grid_state(lvl.bystander_grid) == grid_state(want), "the patched grid differs from a full build"


@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_a_moved_connector_is_refused_per_splice_and_splices_its_component(style, monkeypatch):
    """A wall/connector const's override is a function of its neighbours --
    _splice_eligible() must reject it regardless of whether any tile is
    actually shared. The membership component takes the move instead
    (group-move-wall plan, Step 2), here with no neighbour to re-derive."""
    scenario = _scenario()
    unit = _place(scenario, 1, WALL_CONST, *ELSEWHERE_TILE)
    cache = _make_cache(style, scenario)
    canvas_w, canvas_h = cache.canvas_dims(0)
    counts = _call_counts(monkeypatch)

    splice = _move_splice(scenario, 1, 0, unit, *MOVED_TILE)
    assert not _splice_eligible(cache.units_by_tile, splice)
    assert render_cache._batch_splice_refusal(cache.units_by_tile, [splice]) == "const:move"
    _apply(cache, splice)

    # IsoChunkCache's wholesale path is lazy (its own _source_gen-gated
    # _level()): _building_bboxes_iso doesn't run until the next composite,
    # so the count assertion below must follow a repaint, not precede it.
    stitched = cache.render_rect(0, 0, canvas_w, canvas_h, mip=0)
    assert counts["building_bboxes"] == 0, "the moved connector took the wholesale fallback"
    full = _oracle(style, scenario)
    assert np.array_equal(stitched, full[:canvas_h, :canvas_w])


@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_moving_off_a_shared_own_tile_splices_its_component(style, monkeypatch):
    """Two Mills stacked on the same tile -- a real corpus shape (5 of 16
    example files carry >1 unit sharing an own-tile). The per-splice guard
    refuses a move of one, since building_bboxes/by_anchor union both Mills'
    data at that key; the component splice re-derives the one that stays."""
    scenario = _scenario()
    _place(scenario, 1, MILL_CONST, *MILL_TILE)  # stays put, index 0
    mover = _place(scenario, 1, MILL_CONST, *MILL_TILE)  # index 1, same tile
    cache = _make_cache(style, scenario)
    canvas_w, canvas_h = cache.canvas_dims(0)
    counts = _call_counts(monkeypatch)

    splice = _move_splice(scenario, 1, 1, mover, *MOVED_TILE)
    assert not _splice_eligible(cache.units_by_tile, splice)
    _apply(cache, splice)
    stitched = cache.render_rect(0, 0, canvas_w, canvas_h, mip=0)  # IsoChunkCache's fallback is lazy

    assert counts["building_bboxes"] == 0, "the shared own-tile move fell back to the wholesale rebuild"
    assert np.array_equal(stitched, _oracle(style, scenario)[:canvas_h, :canvas_w])


@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_moving_onto_an_occupied_tile_splices_its_component(style, monkeypatch):
    """The other direction of the same guard: moving ONTO a tile another
    unit already occupies is refused per splice too, and splices the
    component the same way."""
    scenario = _scenario()
    _place(scenario, 1, MILL_CONST, *MILL_TILE)  # index 0, stationary target
    mover = _place(scenario, 1, MILL_CONST, *ELSEWHERE_TILE)  # index 1
    cache = _make_cache(style, scenario)
    canvas_w, canvas_h = cache.canvas_dims(0)
    counts = _call_counts(monkeypatch)

    splice = _move_splice(scenario, 1, 1, mover, *MILL_TILE)
    assert not _splice_eligible(cache.units_by_tile, splice)
    _apply(cache, splice)
    stitched = cache.render_rect(0, 0, canvas_w, canvas_h, mip=0)  # IsoChunkCache's fallback is lazy

    assert counts["building_bboxes"] == 0, "moving onto an occupied tile fell back to the wholesale rebuild"
    assert np.array_equal(stitched, _oracle(style, scenario)[:canvas_h, :canvas_w])


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
    append to the destination, bump unit_gen so the wall-override and own-tile
    memos see it), plus the splice the Convert stroke builds."""
    own, tiles = _own_tile(unit), _occupied(scenario, unit)
    units = scenario.unit_manager.units
    index = next(i for i, u in enumerate(units[source]) if u is unit)
    del units[source][index]
    units[destination].append(unit)
    unit.player = destination
    scenario.unit_gen += 1
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


# --- Draw stroke-end batches (stroke-end repaint splice plan, Steps 1-2) ----


def _remove_splice(scenario, player_id: int, unit) -> UnitSplice:
    """A remove_many() entry: deleted by identity, like the model does."""
    index = next(i for i, u in enumerate(scenario.unit_manager.units[player_id]) if u is unit)
    return _delete_splice(scenario, player_id, index, unit)


def _apply_batch(cache, splices: list[UnitSplice]) -> None:
    cache.invalidate_units(splices)
    if not isinstance(cache, FlatChunkCache):
        _assert_grid_current(cache)
    cache.invalidate_region((0, 0, *cache.canvas_dims(0)))


@pytest.mark.parametrize("sprites", [False, True])
@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_a_remove_and_add_on_one_tile_splices_and_matches_a_fresh_render(style, sprites, request, monkeypatch):
    """Draw's forest-over-forest shape: the old unit goes, a new one lands on
    the same tile. The add is listed FIRST: applying it before the removal
    would pop the add's own sprite slot, so this also pins removals-first."""
    if sprites:
        request.getfixturevalue("sprite_install")
    const = SPRITE_CONST if sprites else MILL_CONST
    scenario = _scenario()
    _place(scenario, 3, MILL_CONST, *ELSEWHERE_TILE)  # unrelated occupant, far away
    old = _place(scenario, 1, const, *MILL_TILE)
    assert scenario.team_indices[1] != scenario.team_indices[2], "players 1 and 2 share a tint"
    cache = _make_cache(style, scenario, sprites=sprites)
    canvas_w, canvas_h = cache.canvas_dims(0)
    before = cache.render_rect(0, 0, canvas_w, canvas_h, mip=0).copy()
    counts = _call_counts(monkeypatch)

    removal = _remove_splice(scenario, 1, old)
    # Another player, so the swap is visible on screen.
    add = _add_splice(scenario, 2, _place(scenario, 2, const, *MILL_TILE))
    batch = [add, removal]
    assert not _splice_eligible(cache.units_by_tile, add), "the per-splice guard should reject this"
    assert _batch_splice_eligible(cache.units_by_tile, batch), "fixture is not testing the splice path"
    _apply_batch(cache, batch)

    assert counts == {"building_bboxes": 0, "sprites": 0}
    stitched = cache.render_rect(0, 0, canvas_w, canvas_h, mip=0)
    assert counts == {"building_bboxes": 0, "sprites": 0}
    assert not np.array_equal(stitched, before), "the swap changed nothing on screen"
    full = _oracle(style, scenario, sprites=sprites)
    assert np.array_equal(stitched, full[:canvas_h, :canvas_w])


@pytest.mark.parametrize("sprites", [False, True])
@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_two_removals_sharing_a_tile_splice_and_match_a_fresh_render(style, sprites, request, monkeypatch):
    if sprites:
        request.getfixturevalue("sprite_install")
    const = SPRITE_CONST if sprites else MILL_CONST
    scenario = _scenario()
    first = _place(scenario, 0, const, *MILL_TILE)
    second = _place(scenario, 0, const, *MILL_TILE)
    cache = _make_cache(style, scenario, sprites=sprites)
    canvas_w, canvas_h = cache.canvas_dims(0)
    before = cache.render_rect(0, 0, canvas_w, canvas_h, mip=0).copy()
    counts = _call_counts(monkeypatch)

    batch = [_remove_splice(scenario, 0, first), _remove_splice(scenario, 0, second)]
    assert _batch_splice_eligible(cache.units_by_tile, batch)
    _apply_batch(cache, batch)

    assert counts == {"building_bboxes": 0, "sprites": 0}
    stitched = cache.render_rect(0, 0, canvas_w, canvas_h, mip=0)
    assert not np.array_equal(stitched, before), "the removals changed nothing on screen"
    full = _oracle(style, scenario, sprites=sprites)
    assert np.array_equal(stitched, full[:canvas_h, :canvas_w])


@pytest.mark.parametrize("with_add", [False, True])
@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_a_removal_beside_a_non_batch_occupant_splices_its_component(style, with_add, monkeypatch):
    """A removal pops every slot on its tile, so the batch guard refuses a
    unit on that tile the batch doesn't name; the component splice re-derives
    that unit instead of going wholesale."""
    scenario = _scenario()
    _place(scenario, 1, MILL_CONST, *MILL_TILE)  # stays, not in the batch
    doomed = _place(scenario, 0, MILL_CONST, *MILL_TILE)
    cache = _make_cache(style, scenario)
    canvas_w, canvas_h = cache.canvas_dims(0)
    counts = _call_counts(monkeypatch)

    batch = [_remove_splice(scenario, 0, doomed)]
    if with_add:
        batch.append(_add_splice(scenario, 0, _place(scenario, 0, MILL_CONST, *MILL_TILE)))
    assert not _batch_splice_eligible(cache.units_by_tile, batch)
    _apply_batch(cache, batch)
    stitched = cache.render_rect(0, 0, canvas_w, canvas_h, mip=0)  # Iso's fallback is lazy

    assert counts["building_bboxes"] == 0, "the shared tile fell back to the wholesale rebuild"
    full = _oracle(style, scenario)
    assert np.array_equal(stitched, full[:canvas_h, :canvas_w])


@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_two_adds_onto_one_tile_fall_back(style):
    scenario = _scenario()
    cache = _make_cache(style, scenario)
    batch = [_add_splice(scenario, 0, _place(scenario, 0, MILL_CONST, *MILL_TILE)) for _ in range(2)]
    assert not _batch_splice_eligible(cache.units_by_tile, batch)


@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_a_batch_over_the_cap_falls_back(style, monkeypatch):
    scenario = _scenario()
    doomed = [_place(scenario, 0, MILL_CONST, 20.0 + 4 * i, 20.0) for i in range(3)]
    cache = _make_cache(style, scenario)
    canvas_w, canvas_h = cache.canvas_dims(0)
    batch = [_remove_splice(scenario, 0, u) for u in doomed]
    assert _batch_splice_eligible(cache.units_by_tile, batch)
    monkeypatch.setattr(render_cache, "UNIT_SPLICE_MAX_UNITS", 2)
    assert not _batch_splice_eligible(cache.units_by_tile, batch)
    counts = _call_counts(monkeypatch)

    _apply_batch(cache, batch)
    stitched = cache.render_rect(0, 0, canvas_w, canvas_h, mip=0)

    assert counts["building_bboxes"] >= 1, "an over-cap batch should have taken the wholesale fallback"
    full = _oracle(style, scenario)
    assert np.array_equal(stitched, full[:canvas_h, :canvas_w])


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


@pytest.mark.parametrize("sprites", [False, True])
def test_a_flat_remove_and_tail_add_on_one_tile_splices(sprites, request, monkeypatch):
    """Draw's forward stroke: remove_many() then add_many(), which appends."""
    if sprites:
        request.getfixturevalue("sprite_install")
    const = SPRITE_CONST if sprites else MILL_CONST
    scenario = _flat_scenario(const)
    extra = _place(scenario, 0, const, 20.5, 40.5)  # GAIA's block now has two rows
    cache = _flat_cache(scenario, sprites=sprites)
    counts = _flat_counts(monkeypatch)

    removal = _remove_splice(scenario, 0, scenario.unit_manager.units[0][0])
    add = _add_splice(scenario, 0, _place(scenario, 0, const, extra.x, extra.y))
    batch = [removal, _remove_splice(scenario, 0, extra), add]
    assert cache._flat_splice_eligible(batch), "fixture is not testing the splice path"
    _apply_batch(cache, batch)

    _flat_renders(cache)
    assert counts == {"rows": 0, "icons": 0}, "the splice path rebuilt a whole walk"
    _assert_flat_matches_fresh(cache, scenario, sprites)


def test_a_flat_mid_block_insert_falls_back(monkeypatch):
    """Undo puts a removed unit back at its original index, not the block's
    end, which _splice_rows() cannot express."""
    scenario = _flat_scenario(MILL_CONST)
    _place(scenario, 0, MILL_CONST, 20.5, 40.5)
    cache = _flat_cache(scenario)
    counts = _flat_counts(monkeypatch)

    restored = Unit(30.5, 40.5, MILL_CONST)
    scenario.unit_manager.units[0].insert(0, restored)
    batch = [UnitSplice(0, 0, restored, None, _own_tile(restored), (), _occupied(scenario, restored))]
    assert not cache._flat_splice_eligible(batch)
    _apply_batch(cache, batch)

    _flat_renders(cache)
    assert counts["rows"] >= 1, "a mid-block insert should have rebuilt the rows"
    _assert_flat_matches_fresh(cache, scenario, False)


# --- GH #75: a group drag of walls through the window ----------------------

RUN_WALL_CONST = 72  # a real _ROTATION_VARIANT_CONSTS member, span (1, 1)
RUN_WALL_RADIAN = 2 * (2 * np.pi / 5)  # variant 2, radian-encoded, so the file is radian
UNITS_FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"


@pytest.fixture
def run_wall_install(tmp_path, monkeypatch):
    """RUN_WALL_CONST as a 5-variant graphic whose frames differ in colour, so
    a wall drawn at the wrong variant is a pixel mismatch. A copy of
    test_elevation_unit_splice's wall_install, which cannot be imported here
    because that module imports this one."""
    from test_sprite_edit_bbox import REACH_NAMES

    from test_unit_sprites import FILE_NAME, build_sld

    canvas = 4 * unit_sprites.NATIVE_TILE_W
    graphics = tmp_path / unit_sprites.GRAPHICS_SUBPATH
    graphics.mkdir(parents=True)
    (graphics / f"{FILE_NAME}.sld").write_bytes(build_sld(5, canvas=canvas, playercolor=False))
    monkeypatch.setattr(
        unit_sprites, "graphic_map",
        lambda: {RUN_WALL_CONST: {"graphic_id": 1, "file_name": FILE_NAME, "angle_count": 5,
                                  "mirroring_mode": 0, "frame_count": 1, "rotation_is_variant": True}},
    )
    for name in REACH_NAMES:
        monkeypatch.setattr(unit_sprites, name, canvas // 2)
    asset_source.set_install_path_override(tmp_path)
    unit_sprites.clear_caches()
    yield
    asset_source.set_install_path_override(None)
    unit_sprites.clear_caches()


def _derived_variant(scenario, player_id: int, unit) -> int:
    """The shape the wall-connectivity override gives `unit` right now."""
    index = next(i for i, u in enumerate(scenario.unit_manager.units[player_id]) if u is unit)
    overrides = render.wall_variant_rotation_overrides(scenario)
    return unit_sprites.variant_index(overrides[(player_id, index)], 5)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_a_group_wall_drag_reshapes_the_stationary_neighbour_like_a_fresh_render(style, run_wall_install):
    """GH #75 step 8. Two walls either side of a stationary one are dragged two
    tiles east, so the middle wall goes from a run piece (0) to an end piece (2)
    without moving. The group-move-wall plan's Step 2 splices that as a
    membership component holding the stationary wall, under the window's
    scoped repaint, and undo and redo take the same path (both hand
    _after_unit_mutation() move-shaped splices)."""
    from PyQt5.QtCore import Qt

    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    try:
        window.load_scenario(UNITS_FIXTURE_PATH)
        window.terrain_style_combo.setCurrentText(style.capitalize())
        window.show_garrisoned_action.setChecked(True)  # the fixture's garrisoned villager, as the oracle draws it
        window.mode_combo.setCurrentText("Units")
        assert isinstance(window._cache, IsoChunkCache if style == "stepped" else SlopedChunkCache)
        assert window.show_sprites_action.isChecked()
        scenario = window.scenario

        model = window._ensure_unit_edits()
        with window._unit_edit(model, "Add walls", [1]):
            west, middle, east = [model.add(1, RUN_WALL_CONST, x + 0.5, 40.5, rotation=RUN_WALL_RADIAN)
                                  for x in (39, 40, 41)]
        assert _derived_variant(scenario, 1, middle) == 0
        canvas_w, canvas_h = window._cache.canvas_dims(0)
        before = window._cache.render_rect(0, 0, canvas_w, canvas_h).copy()
        assert np.array_equal(before, _oracle(style, scenario, sprites=True)[:canvas_h, :canvas_w])
        plans: list = []
        real_plan = window._cache._unit_splice_plan
        window._cache._unit_splice_plan = lambda c: (plan := real_plan(c), plans.append(plan))[0]

        window._selection = [(1, west.reference_id), (1, east.reference_id)]
        window._refresh_selection_view()
        target = window.map_view._tile_polygon(41, 40).boundingRect().center()
        window.on_unit_move((1, west.reference_id), target, Qt.NoModifier)

        assert "Moved 2 units by (2, 0)" in window.status_log.toPlainText()
        assert [(u.x, u.y) for u in (west, middle, east)] == [(41.5, 40.5), (40.5, 40.5), (43.5, 40.5)]
        assert _derived_variant(scenario, 1, middle) == 2, "the stationary wall did not reshape"
        after = window._cache.render_rect(0, 0, canvas_w, canvas_h).copy()
        assert np.array_equal(after, _oracle(style, scenario, sprites=True)[:canvas_h, :canvas_w])
        window.undo()
        assert _derived_variant(scenario, 1, middle) == 0
        assert np.array_equal(window._cache.render_rect(0, 0, canvas_w, canvas_h), before), "undo"
        window.redo()
        assert np.array_equal(window._cache.render_rect(0, 0, canvas_w, canvas_h), after), "redo"
        assert len(plans) == 3 and all(any(s.unit is middle for s in p) for p in plans if p), plans
        assert all(p is not None for p in plans), "the move, its undo or its redo went wholesale"
    finally:
        window.edit_history.mark_saved()
        window.close()


# --- Convert splices walls and gates (2026-09-27 plan, Step A) ---------------

STYLES = ["stepped", "sloped"]
STATE_MIPS = (0, 1)  # the Stepped levels each Convert test below keeps current
RUN_WALL_INTEGER = 2.0  # variant 2 as a literal index, so the file is integer-encoded
GATE_CONST = 64  # an axis-aligned stone gate, span (4, 1)
assert GATE_CONST in unit_sprites.wall_connector_consts()
assert render.tile_span(GATE_CONST, render.NON_BUILDING_SPAN) == (4, 1)


def _warm_mips(cache) -> None:
    """Builds every STATE_MIPS level past mip 0 (Stepped only); one small rect
    is enough to make a level current."""
    if isinstance(cache, IsoChunkCache):
        for mip in STATE_MIPS[1:]:
            cache.render_rect(0, 0, 256, 256, mip=mip)


def _layer_state(sprites):
    """A SpriteLayer as comparable values, by_anchor lists kept in order. Draws
    compare by identity: both caches get them from unit_sprites' own memo."""
    if sprites is None:
        return None
    by_anchor = {k: [(id(d), px, py) for d, px, py in v] for k, v in sprites.by_anchor.items()}
    return by_anchor, dict(sprites.bboxes), sprites.skip_ids, dict(sprites.farm_by_tile)


def _unit_state(cache, spliced: bool):
    """units_by_tile (bucket order kept) plus every level's building_bboxes and
    sprite layer. A spliced cache's levels are read without _level(), which
    would silently rebuild a stale one and hide a fallback."""
    by_tile = {t: [(id(u), color) for u, color in bucket] for t, bucket in cache.units_by_tile.items()}
    levels = {}
    if isinstance(cache, IsoChunkCache):
        for mip in STATE_MIPS:
            lvl = cache._levels[mip] if spliced else cache._level(mip)
            assert not spliced or lvl.gen == cache._source_gen, f"mip {mip} went stale, so it was not spliced"
            levels[mip] = (dict(lvl.building_bboxes), _layer_state(lvl.sprites))
    else:
        levels[0] = (dict(cache.building_bboxes), _layer_state(cache.sprites))
    return by_tile, levels


def _make_convert_cache(style: str, scenario, sprites: bool, unit_filter: UnitFilter = UnitFilter()):
    cache = _make_cache(style, scenario, sprites=sprites, unit_filter=unit_filter)
    _warm_mips(cache)
    return cache


def _assert_spliced_like_a_fresh_cache(style, cache, scenario, sprites, counts, unit_filter=UnitFilter()) -> None:
    """Zero wholesale rebuilds, then every level's unit state and the mip-0
    render equal a fresh cache's. counts is read before the fresh cache
    exists, since building one calls the counted walks itself."""
    canvas_w, canvas_h = cache.canvas_dims(0)
    stitched = cache.render_rect(0, 0, canvas_w, canvas_h, mip=0)
    state = _unit_state(cache, spliced=True)
    assert counts["building_bboxes"] == 0, "the Convert fell back to the wholesale rebuild"
    fresh = _make_convert_cache(style, scenario, sprites, unit_filter)
    fresh_state = _unit_state(fresh, spliced=False)
    assert state[0] == fresh_state[0], "units_by_tile differs from a fresh cache"
    for mip, got in state[1].items():
        assert got[0] == fresh_state[1][mip][0], f"mip {mip} building_bboxes differ from a fresh cache"
        assert got[1] == fresh_state[1][mip][1], f"mip {mip} sprite layer differs from a fresh cache"
    assert np.array_equal(stitched, fresh.render_rect(0, 0, canvas_w, canvas_h, mip=0))
    if unit_filter == UnitFilter():
        assert np.array_equal(stitched, _oracle(style, scenario, sprites=sprites)[:canvas_h, :canvas_w])


def _wall_run(scenario, player_id: int, rotation: float, xs=(39, 40, 41), y: int = 40) -> list[Unit]:
    walls = [_place(scenario, player_id, RUN_WALL_CONST, x + 0.5, y + 0.5) for x in xs]
    for wall in walls:
        wall.rotation = rotation
    return walls


def _assert_override_disagrees(scenario, splice: UnitSplice, stored: int) -> None:
    """Control 1's non-vacuity: the reassigned wall's derived shape differs
    from its stored one, so resolving it with {} would draw the wrong frame."""
    overrides = render.wall_variant_rotation_overrides(scenario)
    derived = overrides.get((splice.player_id, splice.index))
    assert derived is not None, "the wall has no override, so {} and the real dict coincide"
    assert unit_sprites.variant_index(derived, 5) != stored, "the override equals the stored variant"


def test_is_reassign_needs_an_owner_change_and_nothing_else():
    unit = Unit(10.5, 10.5, RUN_WALL_CONST)
    own, tiles = (10, 10), ((10, 10),)
    assert UnitSplice(2, 0, unit, own, own, tiles, tiles, old_player_id=1).is_reassign
    assert not UnitSplice(2, 0, unit, own, own, tiles, tiles).is_reassign, "an elevation re-anchor"
    assert not UnitSplice(2, 0, unit, own, (11, 10), tiles, ((11, 10),), old_player_id=1).is_reassign
    assert not UnitSplice(2, 0, unit, None, own, (), tiles, old_player_id=1).is_reassign


@pytest.mark.parametrize("encoding", ["radian", "integer"])
@pytest.mark.parametrize("sprites", [False, True])
@pytest.mark.parametrize("style", STYLES)
def test_a_reassigned_wall_in_a_run_splices_with_its_neighbour_derived_shape(
    style, sprites, encoding, run_wall_install, monkeypatch
):
    """The middle wall of a run: in the radian file its stored variant (2)
    disagrees with its neighbour mask, so only the real override draws it
    right. In the integer file the stored index is trusted and there is none."""
    scenario = _scenario()
    _place(scenario, 1, MILL_CONST, *ELSEWHERE_TILE)  # a non-wall left behind in the source list
    _west, middle, _east = _wall_run(scenario, 1, RUN_WALL_RADIAN if encoding == "radian" else RUN_WALL_INTEGER)
    cache = _make_convert_cache(style, scenario, sprites)
    counts = _call_counts(monkeypatch)

    splice = _reassign_splice(scenario, 1, 2, middle)
    if encoding == "radian":
        _assert_override_disagrees(scenario, splice, 2)
    else:
        assert not render.wall_variant_rotation_overrides(scenario), "an integer file should trust its indices"
    assert _batch_splice_eligible(cache.units_by_tile, [splice]), "fixture is not testing the splice path"
    _apply(cache, splice)

    _assert_spliced_like_a_fresh_cache(style, cache, scenario, sprites, counts)


@pytest.mark.parametrize("sprites", [False, True])
@pytest.mark.parametrize("style", STYLES)
def test_a_reassigned_gate_between_walls_splices(style, sprites, run_wall_install, monkeypatch):
    """A gate is a connector for its neighbours' masks but reads no override
    itself, and its own tile sits inside its span-4 footprint."""
    scenario = _scenario()
    gate = _place(scenario, 1, GATE_CONST, 37.0, 40.5)  # footprint x 35..38
    _wall_run(scenario, 1, RUN_WALL_RADIAN, xs=(33, 34, 39, 40))
    cache = _make_convert_cache(style, scenario, sprites)
    before = cache.render_rect(0, 0, *cache.canvas_dims(0), mip=0).copy()
    counts = _call_counts(monkeypatch)

    splice = _reassign_splice(scenario, 1, 2, gate)
    assert len(splice.new_tiles) == 4
    assert _batch_splice_eligible(cache.units_by_tile, [splice]), "fixture is not testing the splice path"
    _apply(cache, splice)

    assert not np.array_equal(cache.render_rect(0, 0, *cache.canvas_dims(0), mip=0), before), "nothing recoloured"
    _assert_spliced_like_a_fresh_cache(style, cache, scenario, sprites, counts)


@pytest.mark.parametrize(("source", "destination"), [(1, 0), (0, 1)])
@pytest.mark.parametrize("sprites", [False, True])
@pytest.mark.parametrize("style", STYLES)
def test_walls_and_a_gate_reassigned_to_and_from_gaia_splice(style, sprites, source, destination, run_wall_install, monkeypatch):
    """GAIA is the one owner stored_rotation() treats differently. Every
    rotation-variant const keeps its stored value there too, so neither the
    file's radian verdict nor any wall's shape moves."""
    scenario = _scenario()
    gate = _place(scenario, source, GATE_CONST, 37.0, 40.5)
    walls = _wall_run(scenario, source, RUN_WALL_RADIAN, xs=(33, 34, 39, 40, 41))
    _place(scenario, source, MILL_CONST, *ELSEWHERE_TILE)
    cache = _make_convert_cache(style, scenario, sprites)
    counts = _call_counts(monkeypatch)

    batch = [_reassign_splice(scenario, source, destination, u) for u in (walls[3], gate, walls[0])]
    _assert_override_disagrees(scenario, batch[0], 2)
    assert _batch_splice_eligible(cache.units_by_tile, batch), "fixture is not testing the splice path"
    _apply_batch(cache, batch)

    _assert_spliced_like_a_fresh_cache(style, cache, scenario, sprites, counts)


@pytest.mark.parametrize("style", STYLES)
def test_a_wall_moved_with_an_owner_change_falls_back(style, run_wall_install, monkeypatch):
    """No edit builds this shape today, and a plain move of the same wall
    splices (the group-move-wall tests below), so the batch guard names it
    `const:other` and the cache rebuilds wholesale."""
    scenario = _scenario()
    _west, middle, _east = _wall_run(scenario, 1, RUN_WALL_RADIAN)
    cache = _make_cache(style, scenario, sprites=True)
    canvas_w, canvas_h = cache.canvas_dims(0)
    counts = _call_counts(monkeypatch)

    splice = replace(_move_splice(scenario, 1, 1, middle, 40.5, 44.5), old_player_id=2)
    assert render_cache._batch_splice_refusal(cache.units_by_tile, [splice]) == "const:other"
    assert render_cache._splice_plan_or_refusal(scenario, cache.units_by_tile, UnitFilter(), [splice]) == (
        None, "const:other",
    )
    _apply(cache, splice)
    stitched = cache.render_rect(0, 0, canvas_w, canvas_h, mip=0)

    assert counts["building_bboxes"] >= 1, "an owner change plus a move should have taken the wholesale fallback"
    assert np.array_equal(stitched, _oracle(style, scenario, sprites=True)[:canvas_h, :canvas_w])


# --- Flat: a Convert of walls and gates takes the row splice ----------------


@pytest.mark.parametrize("encoding", ["radian", "integer"])
@pytest.mark.parametrize("sprites", [False, True])
def test_a_flat_reassigned_wall_in_a_run_splices_with_its_neighbour_derived_shape(
    sprites, encoding, run_wall_install, monkeypatch
):
    """Flat's row splice for the middle wall of a run: in the radian file only
    the real override draws its icon at the right variant."""
    scenario = _scenario()
    _place(scenario, 1, MILL_CONST, *ELSEWHERE_TILE)
    _west, middle, _east = _wall_run(scenario, 1, RUN_WALL_RADIAN if encoding == "radian" else RUN_WALL_INTEGER)
    cache = _flat_cache(scenario, sprites=sprites)
    counts = _flat_counts(monkeypatch)

    splice = _reassign_splice(scenario, 1, 2, middle)
    if encoding == "radian":
        _assert_override_disagrees(scenario, splice, 2)
    assert cache._flat_splice_eligible([splice]), "fixture is not testing the splice path"
    _apply(cache, splice)

    assert counts == {"rows": 0, "icons": 0}, "the Flat Convert fell back to the wholesale rebuild"
    _assert_flat_matches_fresh(cache, scenario, sprites)


@pytest.mark.parametrize(("source", "destination"), [(1, 2), (1, 0), (0, 1)])
@pytest.mark.parametrize("sprites", [False, True])
def test_a_flat_convert_of_walls_and_a_gate_splices(sprites, source, destination, run_wall_install, monkeypatch):
    """GAIA included: stored_rotation() treats it differently, but keeps every
    rotation-variant const's stored value, so no wall's shape moves."""
    scenario = _scenario()
    gate = _place(scenario, source, GATE_CONST, 37.0, 40.5)
    walls = _wall_run(scenario, source, RUN_WALL_RADIAN, xs=(33, 34, 39, 40, 41))
    cache = _flat_cache(scenario, sprites=sprites)
    before = [r.copy() for r in _flat_renders(cache)]
    counts = _flat_counts(monkeypatch)

    batch = [_reassign_splice(scenario, source, destination, u) for u in (walls[3], gate, walls[0])]
    _assert_override_disagrees(scenario, batch[0], 2)
    assert cache._flat_splice_eligible(batch), "fixture is not testing the splice path"
    cache.invalidate_units(batch)
    cache.invalidate_region((0, 0, *cache.canvas_dims(0)))

    assert counts == {"rows": 0, "icons": 0}, "the Flat Convert fell back to the wholesale rebuild"
    assert any(not np.array_equal(a, b) for a, b in zip(before, _flat_renders(cache), strict=True)), "nothing changed"
    _assert_flat_matches_fresh(cache, scenario, sprites)


def test_a_flat_moved_wall_in_a_run_still_falls_back(run_wall_install, monkeypatch):
    scenario = _scenario()
    _west, middle, _east = _wall_run(scenario, 1, RUN_WALL_RADIAN)
    cache = _flat_cache(scenario, sprites=True)
    splice = _move_splice(scenario, 1, 1, middle, 40.5, 44.5)
    assert not cache._flat_splice_eligible([splice])
    assert not cache._flat_splice_eligible([replace(splice, old_player_id=2)])


# --- Convert splices a shared tile's whole component (2026-09-27 plan, Step B)

DECOR_TILE = (59.5, 60.5)  # the Mill at MILL_TILE's sprite anchor tile, (59, 60)


@pytest.fixture
def shared_art_install(tmp_path, monkeypatch):
    """MILL_CONST and SPRITE_CONST both drawn with one team-tinted graphic, so
    a unit standing on the Mill's anchor tile shares its by_anchor key and the
    two overlap on screen: paint order is visible as pixels."""
    from test_sprite_edit_bbox import REACH_NAMES

    from test_unit_sprites import FILE_NAME, build_sld

    canvas = 4 * unit_sprites.NATIVE_TILE_W
    graphics = tmp_path / unit_sprites.GRAPHICS_SUBPATH
    graphics.mkdir(parents=True)
    (graphics / f"{FILE_NAME}.sld").write_bytes(build_sld(1, canvas=canvas))
    entry = {"graphic_id": 1, "file_name": FILE_NAME, "angle_count": 1, "mirroring_mode": 0, "frame_count": 1}
    monkeypatch.setattr(unit_sprites, "graphic_map", lambda: {MILL_CONST: entry, SPRITE_CONST: entry})
    for name in REACH_NAMES:
        monkeypatch.setattr(unit_sprites, name, canvas // 2)
    asset_source.set_install_path_override(tmp_path)
    unit_sprites.clear_caches()
    yield
    asset_source.set_install_path_override(None)
    unit_sprites.clear_caches()


def _assert_takes_the_component_path(cache, scenario, batch: list[UnitSplice]) -> list[UnitSplice]:
    """The batch guard refuses it for a shared tile, and the component path
    takes it instead. Returns the component, for size checks."""
    assert not _batch_splice_eligible(cache.units_by_tile, batch), "fixture shares no tile"
    component, why = render_cache._reassign_component_splices(scenario, cache.units_by_tile, cache.unit_filter, batch)
    assert component is not None, f"the component path refused the batch: {why}"
    return component


def _decorated_mill(scenario, mill_owner: int, decor_owner: int):
    """A Mill with a 1x1 unit standing on its sprite anchor tile, placed so the
    decoration's list position differs from the Mill's."""
    decor = _place(scenario, decor_owner, SPRITE_CONST, *DECOR_TILE)
    mill = _place(scenario, mill_owner, MILL_CONST, *MILL_TILE)
    anchor = unit_sprites.sprite_anchor_tile(list(_occupied(scenario, mill)))
    assert anchor == _own_tile(decor), "the decoration is not on the Mill's anchor tile"
    return mill, decor


@pytest.mark.parametrize(("source", "destination"), [(1, 2), (1, 4), (5, 2)])
@pytest.mark.parametrize("sprites", [False, True])
@pytest.mark.parametrize("style", STYLES)
def test_a_decorated_building_reassign_keeps_the_fresh_draw_order(
    style, sprites, source, destination, shared_art_install, monkeypatch
):
    """The decoration's owner (3) sorts between source and destination in two
    of the three cases, so the Mill's place in the shared bucket and in
    by_anchor[anchor] flips; in the first it stays put."""
    scenario = _scenario()
    _place(scenario, source, SPRITE_CONST, *ELSEWHERE_TILE)  # the source list reorders
    mill, decor = _decorated_mill(scenario, source, 3)
    cache = _make_convert_cache(style, scenario, sprites)
    anchor = _own_tile(decor)
    first = cache.units_by_tile[anchor][0][0]
    before = cache.render_rect(0, 0, *cache.canvas_dims(0), mip=0).copy()
    counts = _call_counts(monkeypatch)

    splice = _reassign_splice(scenario, source, destination, mill)
    component = _assert_takes_the_component_path(cache, scenario, [splice])
    assert {id(s.unit) for s in component} == {id(mill), id(decor)}
    _apply(cache, splice)

    flips = (source < 3) != (destination < 3)
    assert (cache.units_by_tile[anchor][0][0] is first) != flips, "the bucket order is not the case under test"
    assert not np.array_equal(cache.render_rect(0, 0, *cache.canvas_dims(0), mip=0), before), "nothing recoloured"
    _assert_spliced_like_a_fresh_cache(style, cache, scenario, sprites, counts)


@pytest.mark.parametrize("sprites", [False, True])
@pytest.mark.parametrize("style", STYLES)
def test_two_batch_units_on_one_tile_splice_in_fresh_order(style, sprites, shared_art_install, monkeypatch):
    """Both occupants convert in one batch, listed against the walk's order."""
    scenario = _scenario()
    low = _place(scenario, 1, SPRITE_CONST, *DECOR_TILE)
    high = _place(scenario, 3, SPRITE_CONST, *DECOR_TILE)
    _place(scenario, 4, SPRITE_CONST, *ELSEWHERE_TILE)
    cache = _make_convert_cache(style, scenario, sprites)
    counts = _call_counts(monkeypatch)

    batch = [_reassign_splice(scenario, 3, 2, high), _reassign_splice(scenario, 1, 4, low)]
    component = _assert_takes_the_component_path(cache, scenario, batch)
    assert [id(s.unit) for s in component] == [id(high), id(low)], "the component is not in walk order"
    _apply_batch(cache, batch)

    _assert_spliced_like_a_fresh_cache(style, cache, scenario, sprites, counts)


@pytest.mark.parametrize("sprites", [False, True])
@pytest.mark.parametrize("style", STYLES)
def test_a_chain_component_re_derives_units_the_batch_never_touches(style, sprites, shared_art_install, monkeypatch):
    """Mills A-B-C overlapping in a row: only A converts, and only B shares a
    tile with it, but B's other tile is shared with C, so all three re-derive."""
    scenario = _scenario()
    a = _place(scenario, 5, MILL_CONST, 60.0, 60.0)  # x 59..60
    b = _place(scenario, 3, MILL_CONST, 61.0, 60.0)  # x 60..61
    c = _place(scenario, 1, MILL_CONST, 62.0, 60.0)  # x 61..62
    _place(scenario, 3, MILL_CONST, 30.0, 30.0)  # far away, outside the component
    cache = _make_convert_cache(style, scenario, sprites)
    counts = _call_counts(monkeypatch)

    splice = _reassign_splice(scenario, 5, 2, a)
    component = _assert_takes_the_component_path(cache, scenario, [splice])
    # Walk order after the reassign: C (player 1), A (now 2), B (3); the far Mill stays out.
    assert [id(s.unit) for s in component] == [id(c), id(a), id(b)]
    assert [s.old_player_id for s in component] == [None, 5, None]
    _apply(cache, splice)

    _assert_spliced_like_a_fresh_cache(style, cache, scenario, sprites, counts)


@pytest.mark.parametrize(("shown", "visible_after"), [({1, 3}, False), ({3, 4}, True)])
@pytest.mark.parametrize("sprites", [False, True])
@pytest.mark.parametrize("style", STYLES)
def test_a_filter_flip_inside_a_shared_tile_matches_a_fresh_cache(
    style, sprites, shown, visible_after, shared_art_install, monkeypatch
):
    """Destination hidden (the Mill vanishes from under its decoration) and
    source hidden (it appears under it)."""
    scenario = _scenario()
    mill, decor = _decorated_mill(scenario, 1, 3)
    unit_filter = UnitFilter(players=frozenset(shown))
    cache = _make_convert_cache(style, scenario, sprites, unit_filter)
    counts = _call_counts(monkeypatch)

    splice = _reassign_splice(scenario, 1, 4, mill)
    _assert_takes_the_component_path(cache, scenario, [splice])
    _apply(cache, splice)

    anchor = _own_tile(decor)
    assert any(e[0] is mill for e in cache.units_by_tile[anchor]) == visible_after
    _assert_spliced_like_a_fresh_cache(style, cache, scenario, sprites, counts, unit_filter)


@pytest.mark.parametrize("style", STYLES)
def test_a_component_over_the_cap_falls_back(style, shared_art_install, monkeypatch):
    scenario = _scenario()
    a = _place(scenario, 5, MILL_CONST, 60.0, 60.0)
    _place(scenario, 3, MILL_CONST, 61.0, 60.0)
    _place(scenario, 1, MILL_CONST, 62.0, 60.0)
    cache = _make_cache(style, scenario, sprites=True)
    canvas_w, canvas_h = cache.canvas_dims(0)
    monkeypatch.setattr(render_cache, "_COMPONENT_SPLICE_MAX_UNITS", 2)
    counts = _call_counts(monkeypatch)

    splice = _reassign_splice(scenario, 5, 2, a)
    assert not cache.can_splice([splice]), "a three-unit component should be over a cap of 2"
    _apply(cache, splice)
    stitched = cache.render_rect(0, 0, canvas_w, canvas_h, mip=0)

    assert counts["building_bboxes"] >= 1, "an over-cap component should have taken the wholesale fallback"
    assert np.array_equal(stitched, _oracle(style, scenario, sprites=True)[:canvas_h, :canvas_w])


@pytest.mark.parametrize("sprites", [False, True])
@pytest.mark.parametrize("style", STYLES)
def test_two_buildings_on_one_own_tile_keep_their_union_bbox(style, sprites, shared_art_install, monkeypatch):
    """Two Mills with one own tile but different footprints, so building_bboxes
    holds a real union there. Converting one must keep the other's half, with
    sprites off too, where the building part is all the key holds."""
    scenario = _scenario()
    stays = _place(scenario, 3, MILL_CONST, *MILL_TILE)
    converts = _place(scenario, 1, MILL_CONST, MILL_TILE[0] + 0.9, MILL_TILE[1] + 0.9)
    assert _own_tile(stays) == _own_tile(converts) and _occupied(scenario, stays) != _occupied(scenario, converts)
    cache = _make_convert_cache(style, scenario, sprites)
    counts = _call_counts(monkeypatch)

    splice = _reassign_splice(scenario, 1, 4, converts)
    _assert_takes_the_component_path(cache, scenario, [splice])
    _apply(cache, splice)

    _assert_spliced_like_a_fresh_cache(style, cache, scenario, sprites, counts)


# --- Why a unit-edit batch went wholesale (unit-edit component splice plan, Step 0) ---


def _index_of(scenario, player_id: int, unit) -> int:
    return next(i for i, u in enumerate(scenario.unit_manager.units[player_id]) if u is unit)


def _group_move(scenario, player_id: int, units, dx: float, dy: float) -> list[UnitSplice]:
    """_move_units()' batch: every unit shifted by one delta, each at its list index."""
    return [
        _move_splice(scenario, player_id, _index_of(scenario, player_id, u), u, u.x + dx, u.y + dy) for u in units
    ]


def test_the_refusal_names_the_clause_that_refused():
    scenario = _scenario()
    target = _place(scenario, 1, MILL_CONST, *MILL_TILE)
    mover = _place(scenario, 1, MILL_CONST, *ELSEWHERE_TILE)
    lead = _place(scenario, 2, MILL_CONST, 30.0, 30.0)  # x 29..30
    trail = _place(scenario, 2, MILL_CONST, 32.0, 30.0)  # x 31..32
    wall = _place(scenario, 3, WALL_CONST, 90.0, 90.0)
    by_tile = render._units_by_tile(scenario)
    refusal = render_cache._batch_splice_refusal

    assert refusal(by_tile, [_move_splice(scenario, 1, 1, mover, 90.0, 20.0)]) is None
    assert refusal(by_tile, [_move_splice(scenario, 1, 1, mover, *MILL_TILE)]) == "occupant:foreign"
    # lead's new footprint (x 30..31) takes trail's old tile 31; trail is moving, not removed.
    assert refusal(by_tile, _group_move(scenario, 2, [lead, trail], 1.0, 0.0)) == "occupant:batch"
    wall_move = _move_splice(scenario, 3, 0, wall, 91.0, 90.0)
    assert refusal(by_tile, [wall_move]) == "const:move"
    # const:move outranks an earlier splice's occupant, and the component splice takes both.
    onto_target = _move_splice(scenario, 1, 1, mover, MILL_TILE[0] + 1, MILL_TILE[1])
    assert refusal(by_tile, [onto_target, wall_move]) == "const:move"
    assert render_cache._splice_plan(scenario, by_tile, UnitFilter(), [onto_target, wall_move]) is not None
    # An added or removed wall anywhere in the batch outranks a moved one, and is final.
    wall_add = _add_splice(scenario, 3, _place(scenario, 3, WALL_CONST, 80.0, 80.0))
    assert refusal(by_tile, [wall_move, wall_add]) == "const:add"
    assert render_cache._splice_plan_or_refusal(scenario, by_tile, UnitFilter(), [wall_move, wall_add]) == (
        None, "const:add",
    )
    assert refusal(by_tile, [wall_move, _remove_splice(scenario, 3, wall_add.unit)]) == "const:remove"
    adds = [_add_splice(scenario, 4, _place(scenario, 4, MILL_CONST, 100.0, 100.0)) for _ in range(2)]
    assert refusal(by_tile, adds) == "claimed"
    assert render_cache._batch_splice_eligible(by_tile, [_remove_splice(scenario, 1, target)])
    assert refusal({}, adds[:1] * (render_cache.UNIT_SPLICE_MAX_UNITS + 1)) == "cap"


@pytest.fixture
def traced(monkeypatch):
    """Perf Trace on, with fresh module state restored afterwards."""
    debug_log.clear()
    for name, value in perf_trace._fresh_state().items():
        monkeypatch.setattr(perf_trace, name, value)
    monkeypatch.setattr(perf_trace, "_idle_scheduler", None)
    monkeypatch.setattr(perf_trace, "_sprite_counter", lambda: (0, 0))
    monkeypatch.setattr(perf_trace, "_enabled", True)
    yield
    debug_log.clear()


@pytest.mark.parametrize("style", STYLES)
def test_a_wholesale_unit_edit_says_why_on_its_perf_line(style, traced):
    scenario = _scenario()
    lone = _place(scenario, 1, MILL_CONST, 90.0, 90.0)
    cache = _make_cache(style, scenario)

    with perf_trace.op("Place unit"):
        wall = _place(scenario, 1, WALL_CONST, *ELSEWHERE_TILE)
        cache.invalidate_units([_add_splice(scenario, 1, wall)])
    with perf_trace.op("Move unit"):
        cache.invalidate_units([_move_splice(scenario, 1, 1, wall, *MOVED_TILE)])
    with perf_trace.op("Nudge"):
        cache.invalidate_units([_move_splice(scenario, 1, 0, lone, 91.0, 90.0)])
    perf_trace.flush_idle()

    lines = debug_log.get_log_text().splitlines()
    place = next(line for line in lines if "perf op place-unit" in line)
    move = next(line for line in lines if "perf op move-unit" in line)
    nudge = next(line for line in lines if "perf op nudge" in line)
    assert "splice_refused=const:add" in place
    assert "splice_refused" not in move, "a moved wall should splice its component"
    assert "splice_refused" not in nudge, "a spliced edit reported a refusal"


# --- Unit-edit batches on shared tiles (unit-edit component splice plan, Step 1) ---


def _assert_takes_the_membership_path(cache, scenario, batch: list[UnitSplice]) -> list[UnitSplice]:
    """The batch guard refuses it for a shared tile and the membership
    component takes it instead. Returns the component."""
    assert render_cache._batch_splice_refusal(cache.units_by_tile, batch) in (
        "occupant:foreign", "occupant:batch", "claimed",
    ), "fixture is not testing the component path"
    component, why = render_cache._membership_component_splices(scenario, cache.units_by_tile, cache.unit_filter, batch)
    assert component is not None, f"the component path refused the batch: {why}"
    return component


def _source_gen(cache):
    return cache._source_gen if isinstance(cache, IsoChunkCache) else None


def _apply_and_check(style, cache, scenario, sprites, counts, batch) -> list[UnitSplice]:
    """One membership batch through invalidate_units(): no gen bump, no
    wholesale walk, every level equal to a fresh cache's."""
    gen = _source_gen(cache)
    component = _assert_takes_the_membership_path(cache, scenario, batch)
    counts.update(building_bboxes=0, sprites=0)
    _apply_batch(cache, batch)
    assert _source_gen(cache) == gen, "the batch bumped _source_gen, so it went wholesale"
    _assert_spliced_like_a_fresh_cache(style, cache, scenario, sprites, counts)
    return component


@pytest.mark.parametrize("occupant", ["gaia", "building"])
@pytest.mark.parametrize("sprites", [False, True])
@pytest.mark.parametrize("style", STYLES)
def test_a_tree_added_and_undone_on_an_occupied_tile_splices(style, sprites, occupant, shared_art_install, monkeypatch):
    """Draw's stroke-end shape: a GAIA unit lands on a tile another unit
    holds, then its undo removes it again. The GAIA occupant has the lower
    list index, so a naive batch-then-re-anchor order would put the new unit
    first in the shared bucket; a player building's footprint tile is the
    other foreign-occupant shape."""
    scenario = _scenario()
    if occupant == "gaia":
        other = _place(scenario, 0, SPRITE_CONST, *DECOR_TILE)
    else:
        other = _place(scenario, 3, MILL_CONST, *MILL_TILE)
    _place(scenario, 0, SPRITE_CONST, *ELSEWHERE_TILE)
    cache = _make_convert_cache(style, scenario, sprites)
    counts = _call_counts(monkeypatch)

    tree = _place(scenario, 0, SPRITE_CONST, *DECOR_TILE)
    component = _apply_and_check(style, cache, scenario, sprites, counts, [_add_splice(scenario, 0, tree)])
    assert {id(s.unit) for s in component} == {id(other), id(tree)}
    if occupant == "gaia":
        assert [e[0] for e in cache.units_by_tile[_own_tile(tree)]] == [other, tree]

    _apply_and_check(style, cache, scenario, sprites, counts, [_remove_splice(scenario, 0, tree)])


@pytest.mark.parametrize("sprites", [False, True])
@pytest.mark.parametrize("style", STYLES)
def test_a_tree_removed_and_restored_beside_an_occupant_splices(style, sprites, shared_art_install, monkeypatch):
    """A removal on a shared tile, and its undo putting the unit back at its
    old list index (_gaia_membership_diff()'s arriving units), in front of
    the occupant it shares the tile with."""
    scenario = _scenario()
    tree = _place(scenario, 0, SPRITE_CONST, *DECOR_TILE)
    rock = _place(scenario, 0, SPRITE_CONST, *DECOR_TILE)
    _place(scenario, 3, MILL_CONST, 30.0, 30.0)
    cache = _make_convert_cache(style, scenario, sprites)
    counts = _call_counts(monkeypatch)

    _apply_and_check(style, cache, scenario, sprites, counts, [_remove_splice(scenario, 0, tree)])

    scenario.unit_manager.units[0].insert(0, tree)
    scenario.unit_gen += 1
    _apply_and_check(style, cache, scenario, sprites, counts, [_add_splice(scenario, 0, tree)])
    assert [e[0] for e in cache.units_by_tile[_own_tile(tree)]] == [tree, rock]


@pytest.mark.parametrize("foreign", [False, True], ids=["alone", "beside-foreign"])
@pytest.mark.parametrize("sprites", [False, True])
@pytest.mark.parametrize("style", STYLES)
def test_two_adds_onto_one_tile_splice_in_walk_order(style, sprites, foreign, shared_art_install, monkeypatch):
    """Two batch units landing on one tile (claimed), listed against the
    walk's order, alone and on a tile a foreign unit already holds."""
    scenario = _scenario()
    other = _place(scenario, 0, SPRITE_CONST, *DECOR_TILE) if foreign else None
    _place(scenario, 3, SPRITE_CONST, *ELSEWHERE_TILE)
    cache = _make_convert_cache(style, scenario, sprites)
    counts = _call_counts(monkeypatch)

    first = _place(scenario, 0, SPRITE_CONST, *DECOR_TILE)
    second = _place(scenario, 0, SPRITE_CONST, *DECOR_TILE)
    batch = [_add_splice(scenario, 0, second), _add_splice(scenario, 0, first)]
    want = "occupant:foreign" if foreign else "claimed"
    assert render_cache._batch_splice_refusal(cache.units_by_tile, batch) == want
    _apply_and_check(style, cache, scenario, sprites, counts, batch)
    assert [e[0] for e in cache.units_by_tile[_own_tile(first)]] == [u for u in (other, first, second) if u]


@pytest.mark.parametrize("sprites", [False, True])
@pytest.mark.parametrize("style", STYLES)
def test_a_group_move_onto_its_own_old_tiles_splices(style, sprites, shared_art_install, monkeypatch):
    """_move_units()' batch where one mover's new footprint is another
    mover's old one (occupant:batch)."""
    scenario = _scenario()
    lead = _place(scenario, 2, MILL_CONST, 30.0, 30.0)  # x 29..30
    trail = _place(scenario, 2, MILL_CONST, 32.0, 30.0)  # x 31..32
    cache = _make_convert_cache(style, scenario, sprites)
    counts = _call_counts(monkeypatch)

    batch = _group_move(scenario, 2, [lead, trail], 1.0, 0.0)
    assert render_cache._batch_splice_refusal(cache.units_by_tile, batch) == "occupant:batch"
    _apply_and_check(style, cache, scenario, sprites, counts, batch)


@pytest.mark.parametrize("sprites", [False, True])
@pytest.mark.parametrize("style", STYLES)
def test_a_group_move_onto_a_foreign_unit_splices_in_fresh_order(style, sprites, shared_art_install, monkeypatch):
    """A player-3 unit moved onto a GAIA unit's tile, beside a second mover:
    the walk puts the stationary GAIA unit first, so the mover's append must
    land after it, a different order from the batch's own."""
    scenario = _scenario()
    gaia = _place(scenario, 0, SPRITE_CONST, *DECOR_TILE)
    mover = _place(scenario, 3, SPRITE_CONST, DECOR_TILE[0] - 20, DECOR_TILE[1])
    other = _place(scenario, 3, SPRITE_CONST, DECOR_TILE[0] - 20, DECOR_TILE[1] + 5)
    cache = _make_convert_cache(style, scenario, sprites)
    before = cache.render_rect(0, 0, *cache.canvas_dims(0), mip=0).copy()
    counts = _call_counts(monkeypatch)

    component = _apply_and_check(style, cache, scenario, sprites, counts, _group_move(scenario, 3, [mover, other], 20.0, 0.0))
    assert [id(s.unit) for s in component] == [id(gaia), id(mover), id(other)]
    assert [e[0] for e in cache.units_by_tile[_own_tile(gaia)]] == [gaia, mover]
    assert not np.array_equal(cache.render_rect(0, 0, *cache.canvas_dims(0), mip=0), before)


@pytest.mark.parametrize("style", STYLES)
def test_a_membership_component_over_the_cap_falls_back(style, shared_art_install, monkeypatch):
    scenario = _scenario()
    _place(scenario, 3, MILL_CONST, *MILL_TILE)
    cache = _make_cache(style, scenario, sprites=True)
    canvas_w, canvas_h = cache.canvas_dims(0)
    gen = _source_gen(cache)
    counts = _call_counts(monkeypatch)
    tree = _place(scenario, 0, SPRITE_CONST, *DECOR_TILE)
    batch = [_add_splice(scenario, 0, tree)]
    _assert_takes_the_membership_path(cache, scenario, batch)
    monkeypatch.setattr(render_cache, "_COMPONENT_SPLICE_MAX_UNITS", 1)

    assert render_cache._membership_component_splices(scenario, cache.units_by_tile, cache.unit_filter, batch) == (
        None, "cap",
    )
    assert not cache.can_splice(batch), "a two-unit component should be over a cap of 1"
    _apply_batch(cache, batch)
    stitched = cache.render_rect(0, 0, canvas_w, canvas_h, mip=0)

    assert gen is None or _source_gen(cache) != gen, "an over-cap component did not bump the gen"
    assert counts["building_bboxes"] >= 1, "an over-cap component should have taken the wholesale fallback"
    assert np.array_equal(stitched, _oracle(style, scenario, sprites=True)[:canvas_h, :canvas_w])


# --- A group Move holding a wall/connector splices (group-move-wall plan, Step 2) ---


def _variant(k: int) -> float:
    """Wall variant k, radian-encoded, so the file is radian and every wall with a neighbour is re-derived."""
    return k * 2 * np.pi / 5


def _walls(scenario, owners_and_tiles, rotation: float) -> list[Unit]:
    walls = [_place(scenario, owner, RUN_WALL_CONST, x + 0.5, y + 0.5) for owner, (x, y) in owners_and_tiles]
    for wall in walls:
        wall.rotation = rotation
    return walls


def _shapes(scenario, walls) -> list[int | None]:
    """Each wall's derived variant now, None where the override leaves its stored one."""
    overrides = render.wall_variant_rotation_overrides(scenario)
    out = []
    for wall in walls:
        player = next(p for p, units in enumerate(scenario.unit_manager.units) if any(u is wall for u in units))
        derived = overrides.get((player, _index_of(scenario, player, wall)))
        out.append(None if derived is None else unit_sprites.variant_index(derived, 5))
    return out


def _assert_packs_like_a_fresh_cache(style, cache, scenario, sprites, unit_filter) -> None:
    """Every checked level's native unit pack against a fresh cache's (native backend only)."""
    from test_convert_splice_corpus import _pack_state

    from descape import composite_backend

    if composite_backend.native is None:
        return
    fresh = _make_convert_cache(style, scenario, sprites, unit_filter)
    keys: dict = {}
    for mip in STATE_MIPS if isinstance(cache, IsoChunkCache) else (0,):
        pack = cache._unit_pack_of(mip, create=False)
        assert pack is not None, f"mip {mip} lost its unit pack instead of refreshing it"
        assert _pack_state(pack, keys) == _pack_state(fresh._unit_pack_of(mip, create=True), keys), f"mip {mip} pack"


def _apply_move_and_check(style, cache, scenario, sprites, counts, batch, unit_filter=UnitFilter()):
    """A Move batch holding a wall/connector through invalidate_units(): refused
    per batch as `const:move`, spliced as a membership component with no gen
    bump, then every level's units_by_tile, building_bboxes, sprite layer,
    grid and pack equal to a fresh cache's. Returns the plan."""
    assert render_cache._batch_splice_refusal(cache.units_by_tile, batch) == "const:move"
    plan, why = render_cache._splice_plan_or_refusal(scenario, cache.units_by_tile, cache.unit_filter, batch)
    assert plan is not None, f"the Move went wholesale: {why}"
    gen = _source_gen(cache)
    counts.update(building_bboxes=0, sprites=0)
    _apply_batch(cache, batch)
    assert _source_gen(cache) == gen, "the Move bumped _source_gen, so it went wholesale"
    _assert_spliced_like_a_fresh_cache(style, cache, scenario, sprites, counts, unit_filter)
    _assert_packs_like_a_fresh_cache(style, cache, scenario, sprites, unit_filter)
    return plan


def _in_plan(plan, *units) -> bool:
    return all(any(s.unit is u for s in plan) for u in units)


@pytest.mark.parametrize("encoding", ["radian", "integer"])
@pytest.mark.parametrize("sprites", [False, True])
@pytest.mark.parametrize("style", STYLES)
def test_a_wall_moved_out_of_a_run_reshapes_its_stationary_neighbours(
    style, sprites, encoding, run_wall_install, monkeypatch
):
    """The middle of a five-wall run moves away: its two neighbours go from
    run pieces (0) to ends (2). In the integer file every stored index is
    trusted, so there is nothing to re-derive: that is why the other cases
    here are radian (a control drops the neighbour search and only the
    radian row goes red)."""
    scenario = _scenario()
    rotation = RUN_WALL_RADIAN if encoding == "radian" else RUN_WALL_INTEGER
    run = _walls(scenario, [(1, (x, 40)) for x in (38, 39, 40, 41, 42)], rotation)
    far = _walls(scenario, [(1, (80, 80))], rotation)[0]
    cache = _make_convert_cache(style, scenario, sprites)
    counts = _call_counts(monkeypatch)
    before = _shapes(scenario, run[1:2] + run[3:4])

    plan = _apply_move_and_check(style, cache, scenario, sprites, counts, _group_move(scenario, 1, [run[2]], 0.0, 4.0))

    if encoding == "radian":
        assert (before, _shapes(scenario, run[1:2] + run[3:4])) == ([0, 0], [2, 2]), "the neighbours did not reshape"
        assert _in_plan(plan, run[1], run[3]), "the reshaped neighbours are not in the plan"
    assert not _in_plan(plan, far), "a wall nowhere near the move joined the plan"


@pytest.mark.parametrize("sprites", [False, True])
@pytest.mark.parametrize("style", STYLES)
def test_a_wall_moved_next_to_a_lone_wall_reshapes_it(style, sprites, run_wall_install, monkeypatch):
    """The lone wall stores variant 1 and has no neighbour, so it draws 1.
    The mover lands east of it and the neighbour bit makes it an end (2)."""
    scenario = _scenario()
    lone = _walls(scenario, [(1, (50, 40))], _variant(1))[0]
    mover = _walls(scenario, [(1, (55, 45))], _variant(2))[0]
    cache = _make_convert_cache(style, scenario, sprites)
    counts = _call_counts(monkeypatch)
    assert _shapes(scenario, [lone]) == [None]

    plan = _apply_move_and_check(style, cache, scenario, sprites, counts, _group_move(scenario, 1, [mover], -4.0, -5.0))

    assert _shapes(scenario, [lone, mover]) == [2, 2]
    assert _in_plan(plan, lone)


@pytest.mark.parametrize("sprites", [False, True])
@pytest.mark.parametrize("style", STYLES)
def test_a_gate_moved_beside_a_wall_reshapes_the_wall(style, sprites, run_wall_install, monkeypatch):
    """A gate is a connector for the mask but reads no override itself: the
    plan must pass the real overrides for the wall it reshapes."""
    scenario = _scenario()
    wall = _walls(scenario, [(1, (40, 40))], _variant(1))[0]
    gate = _place(scenario, 1, GATE_CONST, 38.0, 50.5)
    cache = _make_convert_cache(style, scenario, sprites)
    counts = _call_counts(monkeypatch)

    batch = _group_move(scenario, 1, [gate], 0.0, -10.0)
    assert (39, 40) in _occupied(scenario, gate), "the gate did not land beside the wall"
    plan = _apply_move_and_check(style, cache, scenario, sprites, counts, batch)

    assert _shapes(scenario, [wall]) == [2]
    assert _in_plan(plan, wall)


@pytest.mark.parametrize("sprites", [False, True])
@pytest.mark.parametrize("style", STYLES)
def test_a_wall_moved_into_a_gap_changes_its_own_shape_and_its_neighbours(style, sprites, run_wall_install, monkeypatch):
    scenario = _scenario()
    run = _walls(scenario, [(1, (x, 40)) for x in (38, 39, 41, 42)], _variant(1))
    mover = _walls(scenario, [(1, (40, 46))], _variant(1))[0]
    cache = _make_convert_cache(style, scenario, sprites)
    counts = _call_counts(monkeypatch)
    assert _shapes(scenario, [mover, run[1], run[2]]) == [None, 2, 2]

    _apply_move_and_check(style, cache, scenario, sprites, counts, _group_move(scenario, 1, [mover], 0.0, -6.0))

    assert _shapes(scenario, [mover, run[1], run[2]]) == [0, 0, 0]


@pytest.mark.parametrize("sprites", [False, True])
@pytest.mark.parametrize("style", STYLES)
def test_a_mixed_group_move_of_walls_a_gate_and_shared_tiles_splices(style, sprites, run_wall_install, monkeypatch):
    """_move_units()' shape at small scale: a wall run and the gate on its
    end, a Mill that lands on another mover's old tile, a Mill leaving a tile
    it shares with a stationary Mill, a stationary wall under the run's first
    piece and another player's wall past the gate, both reshaped."""
    scenario = _scenario()
    run = _walls(scenario, [(1, (x, 30)) for x in (30, 31, 32, 33, 34)], _variant(1))
    gate = _place(scenario, 1, GATE_CONST, 37.0, 30.5)  # x 35..38
    under = _walls(scenario, [(1, (30, 31))], _variant(1))[0]
    past = _walls(scenario, [(2, (40, 30))], _variant(1))[0]
    lead = _place(scenario, 1, MILL_CONST, 60.0, 60.0)  # x 59..60
    trail = _place(scenario, 1, MILL_CONST, 62.0, 60.0)  # x 61..62
    stays = _place(scenario, 3, MILL_CONST, 45.0, 45.0)
    leaves = _place(scenario, 1, MILL_CONST, 45.0, 45.0)
    cache = _make_convert_cache(style, scenario, sprites)
    counts = _call_counts(monkeypatch)
    assert _shapes(scenario, [under, past]) == [2, None]

    batch = _group_move(scenario, 1, [*run, gate, lead, trail, leaves], 1.0, 0.0)
    plan = _apply_move_and_check(style, cache, scenario, sprites, counts, batch)

    assert _shapes(scenario, [under, past]) == [None, 2], "the stationary walls did not reshape"
    assert _in_plan(plan, under, past, stays)


@pytest.mark.parametrize("hidden", ["neighbours", "mover"])
@pytest.mark.parametrize("sprites", [False, True])
@pytest.mark.parametrize("style", STYLES)
def test_a_filter_hidden_wall_on_either_side_of_the_move_matches_a_fresh_cache(
    style, sprites, hidden, run_wall_install, monkeypatch
):
    """Player 2 is hidden. `neighbours`: the reshaped walls are player 2's, so
    they hold no key and are skipped. `mover`: player 2's hidden wall moves,
    and the connector mask ignores the filter, so player 1's visible walls
    beside it reshape all the same."""
    scenario = _scenario()
    owners = (1, 2, 1, 2, 1) if hidden == "neighbours" else (1, 1, 2, 1, 1)
    run = _walls(scenario, [(o, (x, 40)) for o, x in zip(owners, (38, 39, 40, 41, 42), strict=True)], RUN_WALL_RADIAN)
    unit_filter = UnitFilter(players=frozenset({1}))
    cache = _make_convert_cache(style, scenario, sprites, unit_filter)
    counts = _call_counts(monkeypatch)

    batch = _group_move(scenario, owners[2], [run[2]], 0.0, 4.0)
    plan = _apply_move_and_check(style, cache, scenario, sprites, counts, batch, unit_filter)

    assert _shapes(scenario, [run[1], run[3]]) == [2, 2]
    assert _in_plan(plan, run[1], run[3]) is (hidden == "mover")


@pytest.mark.parametrize("style", STYLES)
def test_a_wall_neighbourhood_past_the_cap_falls_back(style, run_wall_install, monkeypatch):
    """The reshaped neighbours count toward _COMPONENT_SPLICE_MAX_UNITS: the
    mover plus its two neighbours fit a cap of 3, not 2."""
    scenario = _scenario()
    run = _walls(scenario, [(1, (x, 40)) for x in (38, 39, 40, 41, 42)], RUN_WALL_RADIAN)
    cache = _make_cache(style, scenario, sprites=True)
    canvas_w, canvas_h = cache.canvas_dims(0)
    gen = _source_gen(cache)
    counts = _call_counts(monkeypatch)
    batch = _group_move(scenario, 1, [run[2]], 0.0, 4.0)

    monkeypatch.setattr(render_cache, "_COMPONENT_SPLICE_MAX_UNITS", 3)
    assert cache.can_splice(batch)
    monkeypatch.setattr(render_cache, "_COMPONENT_SPLICE_MAX_UNITS", 2)
    assert render_cache._splice_plan_or_refusal(scenario, cache.units_by_tile, cache.unit_filter, batch) == (
        None, "const:move/component:cap",
    )
    _apply_batch(cache, batch)
    stitched = cache.render_rect(0, 0, canvas_w, canvas_h, mip=0)

    assert gen is None or _source_gen(cache) != gen, "an over-cap neighbourhood did not bump the gen"
    assert counts["building_bboxes"] >= 1, "an over-cap neighbourhood should have taken the wholesale fallback"
    assert np.array_equal(stitched, _oracle(style, scenario, sprites=True)[:canvas_h, :canvas_w])


@pytest.mark.parametrize("const", sorted(unit_sprites._ROTATION_VARIANT_CONSTS))
def test_every_rotation_variant_const_is_1x1_and_never_rotated_by_a_fields_only_edit(const):
    """_connector_neighbours() looks a wall up by own tile and keys the mask on
    its bounds' low corner, which must be the same tile. And UnitSplice.is_move
    admits every fields-only edit of a wall as a move, which is only safe while
    no such edit can write its rotation (Rotate is ANGLE-only, Cycle Variant
    cyclable-only)."""
    from descape import unit_rotation, unit_variant

    for x, y in ((10.5, 10.5), (10.0, 10.0), (10.99, 10.01)):
        unit = Unit(x, y, const)
        bounds = render.unit_tile_bounds(unit, 120, 120)
        assert (bounds[0], bounds[2]) == _own_tile(unit) and bounds[1] - bounds[0] == bounds[3] - bounds[2] == 1
        assert render.unit_occupied_tiles(unit, 120, 120) == [_own_tile(unit)]
    assert not unit_rotation.rotation_is_angle(const)
    assert not unit_variant.is_cyclable(const)


def test_is_move_needs_both_own_tiles_and_no_owner_change():
    unit = Unit(10.5, 10.5, RUN_WALL_CONST)
    own, tiles = (10, 10), ((10, 10),)
    assert UnitSplice(1, 0, unit, own, (11, 10), tiles, ((11, 10),)).is_move
    assert UnitSplice(1, 0, unit, own, own, tiles, tiles).is_move, "an in-place edit is move-shaped"
    assert not UnitSplice(1, 0, unit, None, own, (), tiles).is_move
    assert not UnitSplice(1, 0, unit, own, None, tiles, ()).is_move
    assert not UnitSplice(2, 0, unit, own, own, tiles, tiles, old_player_id=1).is_move


# --- In-place units_by_tile update (paste-undo-membership plan, Step 2) ------

SHARED_TILE = (40.0, 40.0)  # every unit here is a mill at this anchor, so all share four buckets


def _bucket_ids(units_by_tile) -> dict:
    """units_by_tile by identity and colour, bucket order kept: the fixture's
    dataclass == would match two equal units in swapped slots."""
    return {tile: [(id(u), color) for u, color in bucket] for tile, bucket in units_by_tile.items()}


def _in_place_batch(case: str, scenario) -> tuple[list[UnitSplice], UnitFilter]:
    """Players 0, 1 and 3 already hold the shared tile, so each tail add lands
    between a lower and a higher player's entries. Returns the post-edit batch
    in the order a paste (adds) or its undo (removals) builds it."""
    old = {p: _place(scenario, p, MILL_CONST, *SHARED_TILE) for p in (0, 1, 3)}
    _place(scenario, 1, MILL_CONST, 70.0, 70.0)  # a non-shared player-1 neighbour in the list
    flt = UnitFilter(players=frozenset({2, 3})) if case == "filter" else UnitFilter()
    if case in ("filter", "players", "off_map"):
        added = [_place(scenario, p, MILL_CONST, *SHARED_TILE) for p in (0, 1, 2)]
        if case == "off_map":
            added.append(_place(scenario, 2, MILL_CONST, 500.0, 500.0))
        return [_add_splice(scenario, int(u.player), u) for u in added], flt
    if case == "removals":
        # Player 3's unit stays, so the adds still land before a higher player.
        batch = [_remove_splice(scenario, 0, old[0]), _remove_splice(scenario, 1, scenario.unit_manager.units[1][-1])]
        added = [_place(scenario, p, MILL_CONST, *SHARED_TILE) for p in (1, 2)]
        return batch + [_add_splice(scenario, p, u) for p, u in zip((1, 2), added, strict=True)], flt
    assert case == "same_player"
    batch = [_remove_splice(scenario, 1, old[1])]
    added = [_place(scenario, 1, MILL_CONST, *SHARED_TILE), _place(scenario, 1, MILL_CONST, 41.0, 40.0)]
    return batch + [_add_splice(scenario, 1, u) for u in added], flt


def _count_units_by_tile(monkeypatch) -> list:
    calls = []
    real = render._units_by_tile

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(render, "_units_by_tile", counted)
    return calls


@pytest.mark.parametrize("via", ["skipped", "cap"])
@pytest.mark.parametrize("case", ["filter", "off_map", "players", "removals", "same_player"])
@pytest.mark.parametrize("style", STYLES)
def test_an_in_place_membership_update_equals_a_fresh_build(style, case, via, monkeypatch):
    """A refused (cap) or skipped (splice_levels=False) membership batch patches
    units_by_tile in place, never walking every unit, to exactly a fresh
    build's buckets, bumps _units_version, and (Stepped) the gen; the render
    then equals a fresh cache's."""
    scenario = _scenario()
    batch, flt = _in_place_batch(case, scenario)
    # The cache is built pre-edit: undo the batch's list changes, build, redo them.
    units = scenario.unit_manager.units
    snapshot = [list(us) for us in units]
    for s in batch:
        if s.new_own_tile is not None:
            units[s.player_id][:] = [u for u in units[s.player_id] if u is not s.unit]
    for s in sorted((s for s in batch if s.new_own_tile is None), key=lambda s: (s.player_id, s.index)):
        units[s.player_id].insert(s.index, s.unit)
    scenario.unit_gen += 1
    cache = _make_cache(style, scenario, unit_filter=flt)
    for p, us in enumerate(snapshot):
        units[p][:] = us
    scenario.unit_gen += 1
    if via == "cap":
        monkeypatch.setattr(render_cache, "UNIT_SPLICE_MAX_UNITS", 0)
    version, gen = cache._units_version, getattr(cache, "_source_gen", None)
    walks = _count_units_by_tile(monkeypatch)

    cache.invalidate_units(batch, splice_levels=via != "skipped")

    assert not walks, "the batch walked every unit instead of updating in place"
    assert cache._units_version > version
    if gen is not None:
        assert cache._source_gen == gen + 1, "an in-place update must leave every level stale"
    fresh = _make_cache(style, scenario, unit_filter=flt)
    assert _bucket_ids(cache.units_by_tile) == _bucket_ids(fresh.units_by_tile)
    cache.invalidate_region((0, 0, *cache.canvas_dims(0)))
    canvas_w, canvas_h = cache.canvas_dims(0)
    assert np.array_equal(
        cache.render_rect(0, 0, canvas_w, canvas_h, mip=0), fresh.render_rect(0, 0, canvas_w, canvas_h, mip=0)
    )


def test_the_in_place_insert_is_ordered_not_appended():
    """Pins the parity case's non-vacuity: a player-1 add onto a tile a
    player-3 unit holds must land before it, so a plain append would differ."""
    scenario = _scenario()
    batch, _flt = _in_place_batch("players", scenario)
    tile = _own_tile(batch[0].unit)
    players = [int(u.player) for u, _c in render._units_by_tile(scenario)[tile]]
    assert players == [0, 0, 1, 1, 2, 3]


@pytest.mark.parametrize("shape", ["non_tail", "move", "both_sides"])
@pytest.mark.parametrize("style", STYLES)
def test_a_batch_the_in_place_update_cannot_express_falls_back_to_wholesale(style, shape, monkeypatch):
    scenario = _scenario()
    first = _place(scenario, 1, MILL_CONST, *SHARED_TILE)
    _place(scenario, 1, MILL_CONST, 70.0, 70.0)
    cache = _make_cache(style, scenario)
    if shape == "non_tail":
        restored = Unit(30.5, 40.5, MILL_CONST, player=1)
        scenario.unit_manager.units[1].insert(0, restored)
        scenario.unit_gen += 1
        batch = [_add_splice(scenario, 1, restored)]
    elif shape == "move":
        batch = [_move_splice(scenario, 1, 0, first, 50.0, 50.0)]
    else:
        # Undo of a Convert of the tail unit: it leaves player 1 and arrives at player 2's tail.
        moved = scenario.unit_manager.units[1][-1]
        gone = _remove_splice(scenario, 1, moved)
        scenario.unit_manager.units[2].append(moved)
        moved.player = 2
        batch = [gone, _add_splice(scenario, 2, moved)]
    version = cache._units_version
    assert not cache._update_units_by_tile_in_place(batch)
    assert cache._units_version == version, "a refused in-place update must touch nothing"
    walks = _count_units_by_tile(monkeypatch)

    cache.invalidate_units(batch, splice_levels=False)

    assert walks, "expected the wholesale units_by_tile rebuild"
    canvas_w, canvas_h = cache.canvas_dims(0)
    cache.invalidate_region((0, 0, canvas_w, canvas_h))
    assert np.array_equal(
        cache.render_rect(0, 0, canvas_w, canvas_h, mip=0), _oracle(style, scenario)[:canvas_h, :canvas_w]
    )


@pytest.mark.parametrize("style", STYLES)
def test_an_in_place_update_says_so_on_its_perf_line(style, traced):
    scenario = _scenario()
    cache = _make_cache(style, scenario)
    with perf_trace.op("Paste Region"):
        mill = _place(scenario, 2, MILL_CONST, *SHARED_TILE)
        cache.invalidate_units([_add_splice(scenario, 2, mill)], splice_levels=False)
    perf_trace.flush_idle()
    line = next(line for line in debug_log.get_log_text().splitlines() if "perf op paste-region" in line)
    assert "splice_refused=skipped" in line
    assert "units_in_place" in line
