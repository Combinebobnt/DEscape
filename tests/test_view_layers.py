"""View > Layers: the registry, its availability matrix, and the threading
of LayerState through render.py and the three chunk caches.

The load-bearing check here is the byte-identity gate, the same bar
tests/test_unit_filter.py sets for the unit filter: a DEFAULT LayerState must
change nothing at all, at every mip level, through every compositor. Threading
a parameter through the leaf tile painters is exactly the kind of change that
can perturb pixels while every behavioural assertion still passes.

The textures half needs a REAL texture source to discriminate at all: under
pytest no install is configured, so asset_source.get_terrain_texture_array()
already returns None and "textures off" would be vacuously equal to "textures
on". Every textures test here installs deterministic synthetic textures first
(the fixture shape tests/test_farm_terrain.py and tests/test_sloped_render.py
already use), and one of them pins the actual design claim: textures off is
byte-identical to a render with no texture source at all, i.e. it provably
reuses the existing no-install fallback rather than being a second code path.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass

import numpy as np
import pytest

from descape import asset_source, render, render_cache, settings, unit_sprites
from descape.render import tile_pixels_for_map
from descape.render_cache import FlatChunkCache, IsoChunkCache, SlopedChunkCache
from descape.terrain_palette import TREE_UNIT_IDS
from descape.view_layers import DEFAULT_LAYERS, LAYERS, SMALL_TREE_SCALE, LayerState, availability
from testkit.fakes import FakeScenario, SyntheticTile

from test_unit_sprites import build_sld

MAP_W = MAP_H = 16
BACKGROUND_TERRAIN = 1
FAKE_FARM_CONST = 999001
FAKE_FARM_SPAN = (3, 3)
FAKE_FARM_TERRAIN = 7
TEXTURE_COLORS = {BACKGROUND_TERRAIN: (10, 20, 30), FAKE_FARM_TERRAIN: (200, 150, 40)}

# A real member of the set Filters' Show Trees owns, since that membership is
# exactly what render._resolve_unit_sprite() tests -- a synthetic const would
# make the whole small-trees half of this file vacuous.
TREE_CONST = min(TREE_UNIT_IDS)
NON_TREE_CONST = 999002
SPRITE_FILE = "t_layer_synthetic_x1"
SPRITE_CANVAS = 32


@dataclass
class PlacedUnit:
    """testkit's SyntheticUnit carries no `rotation`, which the sprite
    resolver reads -- the farm and small-trees tests need real sprite
    resolution, so they place this instead."""

    x: float
    y: float
    unit_const: int
    rotation: float = 0.0
    reference_id: int = 1


def _scenario(*, with_farm: bool = False, with_trees: bool = False) -> FakeScenario:
    tiles = [
        SyntheticTile(x=x, y=y, elevation=(x + y) % 3, terrain_id=BACKGROUND_TERRAIN)
        for y in range(MAP_H)
        for x in range(MAP_W)
    ]
    units_by_player = [[] for _ in range(9)]
    if with_farm:
        units_by_player[1] = [PlacedUnit(x=5.0, y=5.0, unit_const=FAKE_FARM_CONST)]
    if with_trees:
        # Well apart, so each owns its own anchor tile and its own bbox.
        units_by_player[0] = [
            PlacedUnit(x=4.5, y=4.5, unit_const=TREE_CONST, reference_id=1),
            PlacedUnit(x=11.5, y=11.5, unit_const=NON_TREE_CONST, reference_id=2),
        ]
    return FakeScenario(MAP_W, MAP_H, tiles, units_by_player)


@pytest.fixture
def fake_farm(monkeypatch):
    """A synthetic const that resolves to no .sld but does carry a
    foundation terrain, so the farm-overlay path runs with no install --
    tests/test_farm_terrain.py's own fixture, kept local to this file the
    way that file keeps its scenario helpers local."""
    monkeypatch.setitem(render.BUILDING_TILE_SPANS, FAKE_FARM_CONST, FAKE_FARM_SPAN)
    monkeypatch.setitem(render.FOUNDATION_TERRAIN, FAKE_FARM_CONST, FAKE_FARM_TERRAIN)


@pytest.fixture
def fake_trees(tmp_path, monkeypatch):
    """One synthetic sprite, registered for a real TREE_UNIT_IDS const AND a
    non-tree one. Both resolve through the same file, so the only thing that
    can make them render differently is the const test in
    render._resolve_unit_sprite() -- which is what makes the non-tree unit a
    real scoping control rather than a second tree."""
    assert NON_TREE_CONST not in TREE_UNIT_IDS, "the scoping control is a tree"
    graphics = tmp_path / unit_sprites.GRAPHICS_SUBPATH
    graphics.mkdir(parents=True)
    (graphics / f"{SPRITE_FILE}.sld").write_bytes(build_sld(1, canvas=SPRITE_CANVAS))
    entry = {"graphic_id": 1, "file_name": SPRITE_FILE, "angle_count": 1,
             "mirroring_mode": 6, "frame_count": 1}
    monkeypatch.setattr(
        unit_sprites, "graphic_map", lambda: {TREE_CONST: entry, NON_TREE_CONST: entry}
    )
    asset_source.set_install_path_override(tmp_path)
    unit_sprites.clear_caches()
    yield
    asset_source.set_install_path_override(None)
    unit_sprites.clear_caches()


# A multiple of every tile_px in the mip ladders these fixtures build (16
# through 128 at MAP_W x MAP_H) -- _crop_offset asserts texture_size is an
# exact multiple of tile_px, and the ladder reaches ABOVE level 0's tile_px,
# so sizing this off tile_pixels_for_map() is not enough.
FAKE_TEXTURE_SIZE = 256


def _install_textures(monkeypatch) -> None:
    """Deterministic solid-colour textures, so the oracle stays exact rather
    than depending on install-specific art -- the fixture shape
    tests/test_farm_terrain.py and tests/test_sloped_render.py already use.
    Solid colour also makes the crop offset irrelevant, so a real mip ladder
    needs no per-level texture."""
    textures = {
        tid: np.full((FAKE_TEXTURE_SIZE, FAKE_TEXTURE_SIZE, 3), color, dtype=np.uint8)
        for tid, color in TEXTURE_COLORS.items()
    }
    monkeypatch.setattr(asset_source, "get_terrain_texture_array", textures.get)
    # The flat colour is the texture's average, which for a solid fake IS the texture:
    # pinned to the palette (no average), as when these fakes had no .dds behind them.
    monkeypatch.setattr(asset_source, "get_terrain_average_color", lambda _tid: None)


def _iso_cache(scn, **kwargs) -> IsoChunkCache:
    elevations, proj = render.elevations_and_proj(scn)
    return IsoChunkCache(scn, elevations, proj, tile_pixels_for_map(MAP_W, MAP_H), chunk_px=128, **kwargs)


def _sloped_cache(scn, **kwargs) -> SlopedChunkCache:
    elevations, corner_rise, proj = render.sloped_elevations_and_proj(scn)
    return SlopedChunkCache(
        scn, elevations, corner_rise, proj, tile_pixels_for_map(MAP_W, MAP_H), chunk_px=128, **kwargs
    )


def _flat_cache(scn, **kwargs) -> FlatChunkCache:
    return FlatChunkCache(scn, tile_pixels_for_map(MAP_W, MAP_H), chunk_px=128, **kwargs)


def _whole_canvas(cache) -> np.ndarray:
    w, h = cache.canvas_dims(0)
    return cache.render_rect(0, 0, w, h)


def _level_canvas(cache, mip: int) -> np.ndarray:
    w, h = cache.canvas_dims(mip)
    return cache.render_rect(0, 0, w, h, mip=mip)


# --- the state object ---------------------------------------------------


def test_the_default_state_matches_every_registry_rows_own_default() -> None:
    """Not "everything on" any more: small_trees is the first row to default
    off, since it draws the map away from what the game itself shows rather
    than restoring detail. The registry row and the dataclass field are two
    places to say that, so this pins them equal."""
    for spec in LAYERS:
        assert getattr(LayerState(), spec.layer_id) == spec.default, spec.layer_id
    assert LayerState() == DEFAULT_LAYERS


def test_state_is_frozen_and_compares_by_value() -> None:
    """Frozen and value-comparing is what makes set_layers()' no-op guard
    and its eviction decision correct -- a mutable state edited in place
    would skip both."""
    state = LayerState()
    with pytest.raises(FrozenInstanceError):
        state.terrain_textures = False
    assert LayerState(farm_overlay=False) == LayerState(farm_overlay=False)
    assert LayerState(farm_overlay=False) != LayerState()
    assert len({LayerState(), LayerState()}) == 1


def test_every_registry_row_has_a_unique_id_and_a_state_field() -> None:
    ids = [spec.layer_id for spec in LAYERS]
    assert len(ids) == len(set(ids))
    for layer_id in ids:
        assert hasattr(LayerState(), layer_id), f"{layer_id} has no LayerState field"


def test_every_registry_row_has_a_keybind_row_kept_with_the_other_view_rows() -> None:
    """Contiguity is not cosmetic: _build_keybinds_tab titles a section by
    prefix change, so a view_layer_* row separated from the view_* run emits
    a second "View" header."""
    action_ids = [action_id for action_id, _label, _default in settings.REBINDABLE_ACTIONS]
    for spec in LAYERS:
        assert f"view_layer_{spec.layer_id}" in action_ids
    view_positions = [i for i, aid in enumerate(action_ids) if aid.startswith("view_")]
    assert view_positions == list(range(min(view_positions), max(view_positions) + 1))


# --- the availability matrix -------------------------------------------


@pytest.mark.parametrize("style", ["flat", "stepped", "sloped"])
@pytest.mark.parametrize("sprites_enabled", [True, False])
def test_no_layer_is_available_without_a_map(style, sprites_enabled) -> None:
    for spec in LAYERS:
        enabled, tip = availability(spec, style=style, sprites_enabled=sprites_enabled, has_map=False)
        assert not enabled
        assert tip


@pytest.mark.parametrize("style", ["flat", "stepped", "sloped"])
@pytest.mark.parametrize("sprites_enabled", [True, False])
def test_terrain_textures_is_available_in_every_style_and_sprite_state(style, sprites_enabled) -> None:
    spec = next(s for s in LAYERS if s.layer_id == "terrain_textures")
    enabled, tip = availability(spec, style=style, sprites_enabled=sprites_enabled, has_map=True)
    assert enabled
    assert tip == spec.tooltip


@pytest.mark.parametrize(
    ("style", "sprites_enabled", "expected"),
    [
        ("stepped", True, True),
        ("sloped", True, True),
        ("flat", True, True),
        ("stepped", False, False),
        ("sloped", False, False),
        ("flat", False, False),
    ],
)
def test_farm_overlay_needs_sprites_in_every_style(style, sprites_enabled, expected) -> None:
    """A fact about the render, not policy: every cache builds its drape
    only with sprites on (top-down Flat too, since GH #150)."""
    spec = next(s for s in LAYERS if s.layer_id == "farm_overlay")
    enabled, tip = availability(spec, style=style, sprites_enabled=sprites_enabled, has_map=True)
    assert enabled is expected
    assert tip
    if not expected:
        assert tip != spec.tooltip, "a greyed row must explain itself, not repeat the normal tooltip"


# --- byte identity, the gate -------------------------------------------


def test_default_layers_is_byte_identical_to_omitting_the_argument(monkeypatch) -> None:
    """Every compositor, with real textures installed so the terrain path
    is genuinely exercised rather than collapsing to the fallback."""
    _install_textures(monkeypatch)
    scn = _scenario()
    tile_px = tile_pixels_for_map(MAP_W, MAP_H)
    elevations, proj = render.elevations_and_proj(scn)
    _, corner_rise, sproj = render.sloped_elevations_and_proj(scn)

    omitted = render.composite_rect_iso(scn, 0, 0, 128, 128, elevations, proj, tile_px, {}, {})
    passed = render.composite_rect_iso(
        scn, 0, 0, 128, 128, elevations, proj, tile_px, {}, {}, layers=LayerState()
    )
    assert np.array_equal(omitted, passed)

    omitted = render.composite_rect_sloped(scn, 0, 0, 128, 128, corner_rise, sproj, tile_px, {}, {})
    passed = render.composite_rect_sloped(
        scn, 0, 0, 128, 128, corner_rise, sproj, tile_px, {}, {}, layers=LayerState()
    )
    assert np.array_equal(omitted, passed)

    omitted = render.composite_rect_flat(scn, 0, 0, 128, 128, tile_px)
    passed = render.composite_rect_flat(scn, 0, 0, 128, 128, tile_px, layers=LayerState())
    assert np.array_equal(omitted, passed)


@pytest.mark.parametrize("builder", [_iso_cache, _sloped_cache, _flat_cache])
def test_textures_off_changes_pixels_and_round_trips(builder, monkeypatch) -> None:
    _install_textures(monkeypatch)
    scn = _scenario()
    cache = builder(scn)
    before = _whole_canvas(cache).copy()

    cache.set_layers(LayerState(terrain_textures=False))
    flat_coloured = _whole_canvas(cache).copy()
    assert not np.array_equal(before, flat_coloured), "the textures layer must actually repaint"

    cache.set_layers(LayerState())
    assert np.array_equal(before, _whole_canvas(cache))


@pytest.mark.parametrize("builder", [_iso_cache, _sloped_cache, _flat_cache])
def test_textures_off_is_exactly_the_no_install_render(builder, monkeypatch) -> None:
    """The design claim: this layer adds no rendering, it forces the branch
    an unconfigured install already takes. If the two ever diverge, the
    toggle has grown a second terrain path that can drift."""
    _install_textures(monkeypatch)
    off = _whole_canvas(builder(_scenario(), layers=LayerState(terrain_textures=False))).copy()

    monkeypatch.setattr(asset_source, "get_terrain_texture_array", lambda _tid: None)
    no_install = _whole_canvas(builder(_scenario()))
    assert np.array_equal(off, no_install)


@pytest.mark.parametrize("builder", [_iso_cache, _flat_cache])
def test_textures_apply_at_every_mip_level_not_just_the_resident_one(builder, monkeypatch) -> None:
    """A level built lazily AFTER the change must honour it, and a level
    already resident must be rebuilt rather than reused -- a single-level
    assertion cannot see either failure."""
    _install_textures(monkeypatch)
    cache = builder(_scenario())
    levels = sorted(cache._mip_tile_px)
    assert len(levels) > 1, f"{type(cache).__name__} enumerated only {levels} -- test would be vacuous"
    fine, coarse = min(levels), max(levels)

    fine_before = _level_canvas(cache, fine).copy()
    cache.set_layers(LayerState(terrain_textures=False))
    coarse_off = _level_canvas(cache, coarse).copy()
    assert not np.array_equal(fine_before, _level_canvas(cache, fine)), (
        f"{type(cache).__name__} level {fine} kept stale pre-toggle pixels"
    )

    cache.set_layers(LayerState())
    assert np.array_equal(fine_before, _level_canvas(cache, fine))
    assert not np.array_equal(coarse_off, _level_canvas(cache, coarse)), (
        f"{type(cache).__name__} level {coarse} ignored the layer being restored"
    )


def test_set_layers_stores_the_state_and_an_equal_state_is_a_no_op(monkeypatch) -> None:
    """Evicting the whole canvas is the most expensive thing a cache can be
    asked to do, so an unchanged state must not trigger it."""
    _install_textures(monkeypatch)
    cache = _flat_cache(_scenario())
    _whole_canvas(cache)
    cached_before = len(cache._cache)
    assert cached_before > 0

    cache.set_layers(LayerState())
    assert len(cache._cache) == cached_before

    state = LayerState(terrain_textures=False)
    cache.set_layers(state)
    assert cache.layers == state


# --- the farm layer -----------------------------------------------------


def _farm_sprites(cache):
    """The live SpriteLayer a cache would composite with right now."""
    if isinstance(cache, IsoChunkCache):
        return cache._level(0).sprites
    return cache.sprites


@pytest.mark.parametrize("builder", [_iso_cache, _sloped_cache])
def test_farm_overlay_off_drops_the_terrain_override_and_the_paint_skip(
    builder, fake_farm, monkeypatch
) -> None:
    """Both halves, because suppressing only the override would leave the
    farm skipped at paint time and so invisible: with the layer off the
    tile keeps its own terrain AND the unit is no longer in skip_ids, so it
    draws as an ordinary coloured mark."""
    _install_textures(monkeypatch)
    cache = builder(_scenario(with_farm=True), sprites=True)
    _whole_canvas(cache)
    on = _farm_sprites(cache)
    assert on is not None and on.farm_by_tile, "fixture built no farm -- test would be vacuous"
    assert on.skip_ids

    cache.set_layers(LayerState(farm_overlay=False))
    _whole_canvas(cache)
    off = _farm_sprites(cache)
    assert off is not None
    assert not off.farm_by_tile
    assert not off.skip_ids


@pytest.mark.parametrize("builder", [_iso_cache, _sloped_cache, _flat_cache])
def test_farm_overlay_round_trips_byte_identically(builder, fake_farm, monkeypatch) -> None:
    """The trap this covers is specific to the farm half: it is decided
    where the SpriteLayer is BUILT, so set_layers() has to bump the source
    generation as well as evict, or an already-composited level keeps
    serving the pre-flip layer."""
    _install_textures(monkeypatch)
    cache = builder(_scenario(with_farm=True), sprites=True)
    before = _whole_canvas(cache).copy()

    cache.set_layers(LayerState(farm_overlay=False))
    hidden = _whole_canvas(cache).copy()
    assert not np.array_equal(before, hidden), "the farm layer must actually repaint"

    cache.set_layers(LayerState())
    assert np.array_equal(before, _whole_canvas(cache))


def test_farm_overlay_survives_a_unit_splice(fake_farm, monkeypatch) -> None:
    """D4's splice re-resolves the edited unit, so a hardcoded with_farms
    there would quietly re-admit the override on the next single-unit edit
    after the layer was turned off."""
    _install_textures(monkeypatch)
    scn = _scenario(with_farm=True)
    cache = _iso_cache(scn, sprites=True)
    _whole_canvas(cache)
    cache.set_layers(LayerState(farm_overlay=False))
    _whole_canvas(cache)

    unit = scn.unit_manager.units[1][0]
    splice = render_cache.UnitSplice(
        player_id=1, index=0, unit=unit,
        old_own_tile=(5, 5), new_own_tile=(5, 5),
        old_tiles=frozenset({(x, y) for x in (4, 5, 6) for y in (4, 5, 6)}),
        new_tiles=frozenset({(x, y) for x in (4, 5, 6) for y in (4, 5, 6)}),
    )
    cache.invalidate_units([splice])
    _whole_canvas(cache)
    assert not _farm_sprites(cache).farm_by_tile


def test_flat_farm_overlay_off_drops_the_drape_and_restores_the_rect(fake_farm, monkeypatch) -> None:
    """Flat's leg of the off test above (GH #150): with the layer off there
    is no drape and the farm is its plain owner-coloured rect again."""
    _install_textures(monkeypatch)
    scn = _scenario(with_farm=True)
    cache = _flat_cache(scn, sprites=True)
    _whole_canvas(cache)
    assert cache._farm_tiles, "fixture built no Flat drape -- test would be vacuous"

    cache.set_layers(LayerState(farm_overlay=False))
    off = _whole_canvas(cache)
    assert cache._farm_tiles is None
    rect = render.overlay_units(render.render_terrain(scn), scn, with_sprites=True)
    assert np.array_equal(off, rect)


@pytest.mark.parametrize("op", ["move", "add"])
def test_flat_farm_overlay_off_survives_a_unit_splice(fake_farm, monkeypatch, op) -> None:
    """Flat's twin of test_farm_overlay_survives_a_unit_splice: the row
    splice re-resolves the edited farm's icon (a rewritten row, or an
    inserted one), and a hardcoded with_farms there would give it
    FLAT_DRAPED, dropping its rect with no drape under it."""
    _install_textures(monkeypatch)
    scn = _scenario(with_farm=True)
    cache = _flat_cache(scn, sprites=True)
    cache.set_layers(LayerState(farm_overlay=False))
    _whole_canvas(cache)
    assert 0 in cache._level_icon_layers, "level 0's icons were never built, so the splice patches nothing"

    if op == "move":
        unit = scn.unit_manager.units[1][0]
        tiles = tuple(render.unit_occupied_tiles(unit, MAP_W, MAP_H))
        own = (int(unit.x), int(unit.y))
        splice = render_cache.UnitSplice(1, 0, unit, own, own, tiles, tiles)
    else:
        unit = PlacedUnit(x=10.0, y=10.0, unit_const=FAKE_FARM_CONST, reference_id=9)
        scn.unit_manager.units[2].append(unit)
        tiles = tuple(render.unit_occupied_tiles(unit, MAP_W, MAP_H))
        splice = render_cache.UnitSplice(2, 0, unit, None, (int(unit.x), int(unit.y)), (), tiles)
    assert cache.can_splice([splice]), "fixture is not testing the splice path"
    cache.invalidate_units([splice])
    cache.patch_rects(render.flat_tile_rects(splice.changed_tiles, tile_pixels_for_map(MAP_W, MAP_H)))

    assert cache._farm_tiles is None
    assert not any(draw is render.FLAT_DRAPED for draw in cache._level_icon_layers[0].values())
    rect = render.overlay_units(render.render_terrain(scn), scn, with_sprites=True)
    assert np.array_equal(_whole_canvas(cache), rect)


# --- the small-trees layer ----------------------------------------------
# A SIZE row rather than a visibility one, and the only build-time field
# whose effect lands inside the sprite itself, so these mirror the farm
# tests above class for class (build-time invalidation, the splice thread,
# the mip ladder, Flat) plus the one thing the farm layer has no analogue
# for: scoping to TREE_UNIT_IDS.


def _sprite_shapes(scn, *, tree_scale: float) -> dict[tuple[int, int], tuple[int, int]]:
    """Each anchor tile's first resolved sprite's (h, w), straight off the
    real SpriteLayer the caches composite."""
    elevations, proj = render.elevations_and_proj(scn)
    layer = render.sprite_draws_by_anchor(scn, proj, elevations, tree_scale=tree_scale)
    return {tile: draws[0][0].rgba.shape[:2] for tile, draws in layer.by_anchor.items()}


def test_small_trees_shrinks_a_tree_and_leaves_every_other_unit_alone(fake_trees) -> None:
    """**The scoping assertion.** Both units resolve through the same
    synthetic file, so a non-tree that shrinks here means the const test is
    missing or reaching the wrong set."""
    scn = _scenario(with_trees=True)
    full = _sprite_shapes(scn, tree_scale=1.0)
    small = _sprite_shapes(scn, tree_scale=SMALL_TREE_SCALE)

    tree_tile, other_tile = (4, 4), (11, 11)
    assert set(full) == set(small) == {tree_tile, other_tile}, full
    assert small[tree_tile][0] < full[tree_tile][0]
    assert small[tree_tile][1] < full[tree_tile][1]
    assert small[other_tile] == full[other_tile]


def test_tree_scale_of_one_is_byte_identical_to_omitting_the_argument(fake_trees) -> None:
    """The gate, at the render level this time: threading a parameter through
    the resolver must not perturb a default render."""
    scn = _scenario(with_trees=True)
    elevations, proj = render.elevations_and_proj(scn)
    omitted = render.sprite_draws_by_anchor(scn, proj, elevations)
    passed = render.sprite_draws_by_anchor(scn, proj, elevations, tree_scale=1.0)
    assert omitted.bboxes == passed.bboxes
    for tile, draws in omitted.by_anchor.items():
        for (draw, px, py, _rise), (pdraw, ppx, ppy, _prise) in zip(draws, passed.by_anchor[tile], strict=True):
            assert (px, py) == (ppx, ppy)
            assert np.array_equal(draw.rgba, pdraw.rgba)


@pytest.mark.parametrize("builder", [_iso_cache, _sloped_cache])
def test_small_trees_round_trips_byte_identically(builder, fake_trees) -> None:
    """Same build-time trap the farm layer documents: the factor is baked
    into the SpriteLayer when it is BUILT, so set_layers() has to rebuild the
    unit sources as well as evict, or an already-composited level keeps
    serving full-size art.

    No synthetic textures here, unlike the farm tests: a farm's layer shows
    up as TERRAIN, so it needs a texture source to discriminate at all,
    where this one is a difference in the sprite's own pixels."""
    cache = builder(_scenario(with_trees=True), sprites=True)
    before = _whole_canvas(cache).copy()

    cache.set_layers(LayerState(small_trees=True))
    shrunk = _whole_canvas(cache).copy()
    assert not np.array_equal(before, shrunk), "the small-trees layer must actually repaint"

    cache.set_layers(LayerState())
    assert np.array_equal(before, _whole_canvas(cache))


def test_small_trees_survives_a_unit_splice(fake_trees) -> None:
    """The splice re-resolves the edited unit on its own, so a site left on
    the default 1.0 would re-inflate a tree the first time it is moved --
    and only then, which is why this cannot ride the round-trip test."""
    scn = _scenario(with_trees=True)
    cache = _iso_cache(scn, sprites=True)
    _whole_canvas(cache)
    cache.set_layers(LayerState(small_trees=True))
    _whole_canvas(cache)
    shrunk = _sprite_shapes(scn, tree_scale=SMALL_TREE_SCALE)[(4, 4)]

    tree = scn.unit_manager.units[0][0]
    tree.x, tree.y = 6.5, 6.5
    splice = render_cache.UnitSplice(
        player_id=0, index=0, unit=tree,
        old_own_tile=(4, 4), new_own_tile=(6, 6),
        old_tiles=frozenset({(4, 4)}), new_tiles=frozenset({(6, 6)}),
    )
    cache.invalidate_units([splice])
    _whole_canvas(cache)

    level = cache._level(0)
    assert (4, 4) not in level.sprites.by_anchor
    assert level.sprites.by_anchor[(6, 6)][0][0].rgba.shape[:2] == shrunk


@pytest.mark.parametrize("build", [_iso_cache, _sloped_cache], ids=["Stepped", "Sloped"])
def test_small_trees_applies_at_every_mip_level_not_just_the_resident_one(fake_trees, build) -> None:
    """The factor multiplies into sprite_scale(half_w), so it has to follow
    the mip ladder rather than being applied once at level 0.

    Stepped and Sloped, whose ladder is Stepped's coarser levels
    (TASK-031.60); Flat does not reach this layer at all."""
    cache = build(_scenario(with_trees=True), sprites=True)
    levels = sorted(cache._mip_tile_px)
    assert len(levels) > 1, f"enumerated only {levels} -- test would be vacuous"
    fine, coarse = min(levels), max(levels)

    fine_before = _level_canvas(cache, fine).copy()
    cache.set_layers(LayerState(small_trees=True))
    coarse_small = _level_canvas(cache, coarse).copy()
    assert not np.array_equal(fine_before, _level_canvas(cache, fine)), (
        f"level {fine} kept stale full-size trees"
    )

    cache.set_layers(LayerState())
    assert np.array_equal(fine_before, _level_canvas(cache, fine))
    assert not np.array_equal(coarse_small, _level_canvas(cache, coarse)), (
        f"level {coarse} ignored the layer being restored"
    )


def test_flat_ignores_the_small_trees_layer_rather_than_pretending_to_apply_it(fake_trees) -> None:
    """Flat draws units through icon_for(), which contain-fits every unit's
    art into its own footprint rect -- a tree there cannot overflow onto a
    neighbour, so there is nothing behind it to reveal. That is why the
    registry greys the row in Flat rather than letting it be a silent no-op.
    Output-identical is the bar, not work-skipped: the flip still rebuilds
    (the base class routes it through invalidate_units), it just repaints to
    the same pixels."""
    cache = _flat_cache(_scenario(with_trees=True), sprites=True)
    before = _whole_canvas(cache).copy()
    cache.set_layers(LayerState(small_trees=True))
    assert np.array_equal(before, _whole_canvas(cache))


def test_the_drag_ghost_path_takes_the_layer(fake_trees) -> None:
    """unit_sprite_draws_at() is the ghost's own entry point, and it does not
    go through a chunk cache -- so a default left there is the one bug that
    shows as a tree dragging full size and snapping small on drop."""
    scn = _scenario(with_trees=True)
    elevations, proj = render.elevations_and_proj(scn)
    tree = scn.unit_manager.units[0][0]

    full = render.unit_sprite_draws_at(scn, proj, elevations, None, 0, tree)
    small = render.unit_sprite_draws_at(
        scn, proj, elevations, None, 0, tree, tree_scale=SMALL_TREE_SCALE
    )
    assert full and small
    assert small[0][0].rgba.shape[:2] < full[0][0].rgba.shape[:2]


# --- Translucent Walls (GH #35) ------------------------------------------
# The Small Trees tests above, class for class, scoped to
# unit_kind.wall_consts() rather than TREE_UNIT_IDS, with a gate in the set.

WALL_CONST = 117  # stone wall
GATE_CONST = 64  # a stone gate orientation, span (4, 1)
WALL_TILE, NON_WALL_TILE = (4, 4), (12, 12)
TRANSLUCENT = LayerState(translucent_walls=True)


def _wall_scenario() -> FakeScenario:
    tiles = [
        SyntheticTile(x=x, y=y, elevation=(x + y) % 3, terrain_id=BACKGROUND_TERRAIN)
        for y in range(MAP_H)
        for x in range(MAP_W)
    ]
    units_by_player = [[] for _ in range(9)]
    units_by_player[1] = [
        PlacedUnit(x=4.5, y=4.5, unit_const=WALL_CONST, reference_id=1),
        PlacedUnit(x=9.0, y=8.5, unit_const=GATE_CONST, reference_id=2),
        PlacedUnit(x=12.5, y=12.5, unit_const=NON_TREE_CONST, reference_id=3),
    ]
    return FakeScenario(MAP_W, MAP_H, tiles, units_by_player)


@pytest.fixture
def fake_walls(tmp_path, monkeypatch):
    """fake_trees' shape: one synthetic sprite behind a real wall, a real
    gate and a non-wall const, so only the wall-set test in
    render._resolve_unit_sprite() can tell them apart."""
    from descape import unit_kind

    assert {WALL_CONST, GATE_CONST} <= unit_kind.wall_consts()
    assert NON_TREE_CONST not in unit_kind.wall_consts(), "the scoping control is a wall"
    graphics = tmp_path / unit_sprites.GRAPHICS_SUBPATH
    graphics.mkdir(parents=True)
    (graphics / f"{SPRITE_FILE}.sld").write_bytes(build_sld(1, canvas=SPRITE_CANVAS))
    entry = {"graphic_id": 1, "file_name": SPRITE_FILE, "angle_count": 1,
             "mirroring_mode": 6, "frame_count": 1}
    monkeypatch.setattr(
        unit_sprites, "graphic_map",
        lambda: {WALL_CONST: entry, GATE_CONST: entry, NON_TREE_CONST: entry},
    )
    asset_source.set_install_path_override(tmp_path)
    unit_sprites.clear_caches()
    yield
    asset_source.set_install_path_override(None)
    unit_sprites.clear_caches()


def _wall_layer(scn, **kwargs) -> render.SpriteLayer:
    elevations, proj = render.elevations_and_proj(scn)
    return render.sprite_draws_by_anchor(scn, proj, elevations, **kwargs)


def _draws_equal(a, b) -> bool:
    return len(a) == len(b) and all(
        (pa[1:], pa[0].hotspot_x, pa[0].hotspot_y) == (pb[1:], pb[0].hotspot_x, pb[0].hotspot_y)
        and np.array_equal(pa[0].rgba, pb[0].rgba)
        for pa, pb in zip(a, b, strict=True)
    )


def test_translucent_walls_fades_walls_and_gates_and_leaves_every_other_unit_alone(fake_walls) -> None:
    scn = _wall_scenario()
    plain = _wall_layer(scn)
    styled = _wall_layer(scn, wall_alpha=0.5)
    assert set(plain.by_anchor) == set(styled.by_anchor)
    gate_tiles = set(plain.by_anchor) - {WALL_TILE, NON_WALL_TILE}
    assert len(gate_tiles) == 1, plain.by_anchor.keys()

    assert _draws_equal(plain.by_anchor[NON_WALL_TILE], styled.by_anchor[NON_WALL_TILE])
    for tile in (WALL_TILE, *gate_tiles):
        draw, styled_draw = plain.by_anchor[tile][0][0], styled.by_anchor[tile][0][0]
        assert styled_draw.rgba.shape == draw.rgba.shape, tile
        assert int(styled_draw.rgba[..., 3].max()) == 128, tile


def test_wall_alpha_of_one_is_byte_identical_to_omitting_it(fake_walls) -> None:
    scn = _wall_scenario()
    omitted = _wall_layer(scn)
    passed = _wall_layer(scn, wall_alpha=1.0)
    assert omitted.bboxes == passed.bboxes
    for tile, draws in omitted.by_anchor.items():
        assert _draws_equal(draws, passed.by_anchor[tile]), tile


@pytest.mark.parametrize("builder", [_iso_cache, _sloped_cache])
def test_translucent_walls_round_trips_byte_identically(builder, fake_walls) -> None:
    cache = builder(_wall_scenario(), sprites=True)
    before = _whole_canvas(cache).copy()

    cache.set_layers(TRANSLUCENT)
    assert not np.array_equal(before, _whole_canvas(cache)), "the wall layer must actually repaint"

    cache.set_layers(LayerState())
    assert np.array_equal(before, _whole_canvas(cache))


def test_translucent_walls_survives_a_unit_splice(fake_walls) -> None:
    """A splice re-resolves the moved wall alone, so a site left on the 1.0
    default would draw it opaque the first time it moves."""
    scn = _wall_scenario()
    cache = _iso_cache(scn, sprites=True)
    _whole_canvas(cache)
    cache.set_layers(TRANSLUCENT)
    _whole_canvas(cache)
    expected = _wall_layer(scn, wall_alpha=TRANSLUCENT.wall_alpha)
    styled = expected.by_anchor[WALL_TILE][0][0]

    wall = scn.unit_manager.units[1][0]
    wall.x, wall.y = 6.5, 6.5
    splice = render_cache.UnitSplice(
        player_id=1, index=0, unit=wall,
        old_own_tile=WALL_TILE, new_own_tile=(6, 6),
        old_tiles=frozenset({WALL_TILE}), new_tiles=frozenset({(6, 6)}),
    )
    cache.invalidate_units([splice])
    _whole_canvas(cache)

    level = cache._level(0)
    assert WALL_TILE not in level.sprites.by_anchor
    assert np.array_equal(level.sprites.by_anchor[(6, 6)][0][0].rgba, styled.rgba)


@pytest.mark.parametrize("build", [_iso_cache, _sloped_cache], ids=["Stepped", "Sloped"])
def test_translucent_walls_applies_at_every_mip_level(fake_walls, build) -> None:
    cache = build(_wall_scenario(), sprites=True)
    levels = sorted(cache._mip_tile_px)
    assert len(levels) > 1, f"enumerated only {levels} -- test would be vacuous"
    fine, coarse = min(levels), max(levels)

    fine_before = _level_canvas(cache, fine).copy()
    cache.set_layers(TRANSLUCENT)
    coarse_styled = _level_canvas(cache, coarse).copy()
    assert not np.array_equal(fine_before, _level_canvas(cache, fine)), f"level {fine} kept stale walls"

    cache.set_layers(LayerState())
    assert np.array_equal(fine_before, _level_canvas(cache, fine))
    assert not np.array_equal(coarse_styled, _level_canvas(cache, coarse)), (
        f"level {coarse} ignored the layer being restored"
    )


def test_flat_ignores_translucent_walls(fake_walls) -> None:
    cache = _flat_cache(_wall_scenario(), sprites=True)
    before = _whole_canvas(cache).copy()
    cache.set_layers(TRANSLUCENT)
    assert np.array_equal(before, _whole_canvas(cache))


@pytest.mark.parametrize("unit_index", [0, 1], ids=["wall", "gate"])
def test_the_drag_ghost_path_takes_wall_alpha(fake_walls, unit_index) -> None:
    scn = _wall_scenario()
    elevations, proj = render.elevations_and_proj(scn)
    unit = scn.unit_manager.units[1][unit_index]

    full = render.unit_sprite_draws_at(scn, proj, elevations, None, 1, unit)
    styled = render.unit_sprite_draws_at(scn, proj, elevations, None, 1, unit, wall_alpha=0.5)
    assert full and len(styled) == 1
    assert int(styled[0][0].rgba[..., 3].max()) == 128


def test_an_opacity_change_rebuilds_sprites_only_while_translucent_walls_is_on(fake_walls, monkeypatch) -> None:
    """The caches compare the derived wall_alpha, not the raw pct, so the
    slider moved with the row off costs no sprite rebuild."""
    cache = _iso_cache(_wall_scenario(), sprites=True)
    _whole_canvas(cache)
    rebuilds = []
    real = cache._refresh_unit_sources
    monkeypatch.setattr(cache, "_refresh_unit_sources", lambda: (rebuilds.append(1), real())[1])

    cache.set_layers(LayerState(wall_opacity_pct=30))
    assert rebuilds == []

    cache.set_layers(LayerState(translucent_walls=True, wall_opacity_pct=30))
    faded = _whole_canvas(cache).copy()
    assert len(rebuilds) == 1
    cache.set_layers(LayerState(translucent_walls=True, wall_opacity_pct=70))
    assert len(rebuilds) == 2
    assert not np.array_equal(faded, _whole_canvas(cache))


def test_wall_alpha_maps_the_row_and_the_opacity_to_a_factor() -> None:
    assert LayerState().wall_alpha == 1.0
    assert LayerState(wall_opacity_pct=30).wall_alpha == 1.0
    assert LayerState(translucent_walls=True, wall_opacity_pct=30).wall_alpha == 0.3


def test_mod_art_greys_only_without_a_mod_that_ships_art() -> None:
    """TASK-378: on by default, in every style and with sprites off (it also
    swaps terrain textures), greyed only when no scanned mod ships art."""
    spec = next(s for s in LAYERS if s.layer_id == "mod_art")
    assert spec.default is True and DEFAULT_LAYERS.mod_art is True
    assert spec.styles is None and not spec.requires_sprites
    assert availability(spec, style="flat", sprites_enabled=False, has_map=True) == (True, spec.tooltip)
    enabled, tip = availability(spec, style="stepped", sprites_enabled=True, has_map=True, has_mod_art=False)
    assert not enabled and tip.startswith("No enabled mod ships art")
    assert availability(spec, style="stepped", sprites_enabled=True, has_map=False, has_mod_art=False)[1] == (
        "Open a map first"
    )
    other = next(s for s in LAYERS if s.layer_id == "hero_glow")
    assert availability(other, style="stepped", sprites_enabled=True, has_map=True, has_mod_art=False)[0]
