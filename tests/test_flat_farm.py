"""GH #150: in top-down Flat a farm paints its footprint as its farm terrain
with a 1px owner-colour perimeter, as Stepped and Sloped do, under the same
gates (units shown, Show sprites, View > Layers > Farm Terrain Overlay).

The oracle is the independent full render, render_terrain(farm_tiles=...)
plus overlay_units(with_farms=True), never composite_rect_flat() itself. That
full render exists at level 0 only, so the other mips are checked against a
single whole-canvas composite (seams) and by counting outline pixels.

A synthetic FAKE_FARM_CONST (no .sld, a foundation terrain) and solid
synthetic textures throughout, as tests/test_farm_terrain.py does, so no
install is needed outside the corpus-marked check at the bottom.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from descape import asset_source, grid_overlay, render, unit_pick, unit_sprites
from descape.render_cache import FlatChunkCache, UnitSplice
from descape.terrain_palette import PLAYER_COLORS
from descape.unit_filter import UnitFilter
from descape.view_layers import LayerState

from test_unit_sprites import FILE_NAME, build_sld

MAP_W = MAP_H = 12
CHUNK = 128
BACKGROUND_TERRAIN = 1
HAND_PAINTED_TERRAIN = 2
FAKE_FARM_CONST = 999001
FAKE_FARM_SPAN = (3, 3)
FAKE_FARM_TERRAIN = 7
FAKE_PASTURE_CONST = 999003
FAKE_PASTURE_SPAN = (4, 4)
PLAIN_CONST = 999002  # no .sld, no foundation terrain: a 1x1 coloured rect
TEXTURE_SIZE = 256  # a multiple of every mip's tile_px (16..128)
TEXTURE_COLORS = {
    BACKGROUND_TERRAIN: (10, 20, 30),
    HAND_PAINTED_TERRAIN: (90, 90, 90),
    FAKE_FARM_TERRAIN: (200, 150, 40),
}
# 3x3 footprint tiles 1..3, canvas pixels 64..256 at tile_px 64: across the 128 seam.
FARM_XY = (2.5, 2.5)
GRID = grid_overlay.grid_bake(True, -100, 1)


@dataclass
class Tile:
    x: int
    y: int
    elevation: int = 0
    terrain_id: int = BACKGROUND_TERRAIN
    layer: int = -1


@dataclass
class Unit:
    x: float
    y: float
    unit_const: int
    rotation: float = 0.0
    reference_id: int = 1


class _MapManager:
    def __init__(self, tiles):
        self.map_width, self.map_height = MAP_W, MAP_H
        self.terrain = tiles
        self._by_xy = {(t.x, t.y): t for t in tiles}

    def get_tile(self, x, y):
        return self._by_xy[(x, y)]


class _UnitManager:
    def __init__(self, units_by_player):
        self.units = units_by_player


class _Scenario:
    def __init__(self, tiles, units_by_player):
        self.map_manager = _MapManager(tiles)
        self.unit_manager = _UnitManager(units_by_player)
        self.player_colors = tuple(PLAYER_COLORS)
        self.team_indices = tuple(range(len(PLAYER_COLORS)))
        self.unit_gen = 0


def _scenario(*player_units) -> _Scenario:
    """player_units: (player_id, Unit) pairs, appended in order."""
    tiles = [Tile(x, y) for y in range(MAP_H) for x in range(MAP_W)]
    units_by_player = [[] for _ in range(9)]
    for player_id, unit in player_units:
        units_by_player[player_id].append(unit)
    return _Scenario(tiles, units_by_player)


def _farm(ref: int = 1, xy=FARM_XY) -> Unit:
    return Unit(*xy, FAKE_FARM_CONST, reference_id=ref)


@pytest.fixture
def fake_farm(monkeypatch):
    """A const with no .sld and a foundation terrain, plus solid textures."""
    assert FAKE_FARM_CONST not in unit_sprites.graphic_map(), "picked const collides with a real one"
    monkeypatch.setitem(render.BUILDING_TILE_SPANS, FAKE_FARM_CONST, FAKE_FARM_SPAN)
    monkeypatch.setitem(render.FOUNDATION_TERRAIN, FAKE_FARM_CONST, FAKE_FARM_TERRAIN)
    textures = {
        tid: np.full((TEXTURE_SIZE, TEXTURE_SIZE, 3), color, dtype=np.uint8) for tid, color in TEXTURE_COLORS.items()
    }
    monkeypatch.setattr(asset_source, "get_terrain_texture_array", textures.get)


def _tile_px() -> int:
    return render.tile_pixels_for_map(MAP_W, MAP_H)


def _cache(scn, **kwargs) -> FlatChunkCache:
    kwargs.setdefault("sprites", True)
    return FlatChunkCache(scn, _tile_px(), chunk_px=CHUNK, **kwargs)


def _mips(cache) -> list[int]:
    return sorted(cache._mip_tile_px)


def _stitched(cache, mip: int = 0) -> np.ndarray:
    w, h = cache.canvas_dims(mip)
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for y0 in range(0, h, CHUNK):
        for x0 in range(0, w, CHUNK):
            x1, y1 = min(x0 + CHUNK, w), min(y0 + CHUNK, h)
            out[y0:y1, x0:x1] = cache._composite_rect(mip, x0, y0, x1, y1)
    return out


def _oracle(scn, grid=grid_overlay.DEFAULT_GRID) -> np.ndarray:
    farm_tiles, _ids = render.flat_farm_tiles(scn)
    terrain = render.render_terrain(scn, grid=grid, farm_tiles=farm_tiles)
    return render.overlay_units(terrain, scn, with_sprites=True, with_farms=True)


def _rect_render(scn) -> np.ndarray:
    """Today's Flat render: every unit its own coloured rect or icon, no drape."""
    return render.overlay_units(render.render_terrain(scn), scn, with_sprites=True)


def _owner(unit, player_id: int = 1) -> np.ndarray:
    return np.array(render._unit_color(unit, PLAYER_COLORS[player_id]), dtype=np.uint8)


def _bounds(unit) -> tuple[int, int, int, int]:
    return render.unit_tile_bounds(unit, MAP_W, MAP_H)


# --- the drape's tiles ------------------------------------------------------------


def test_flat_farm_tiles_cover_the_footprint_with_a_closed_perimeter(fake_farm) -> None:
    farm = _farm()
    by_tile, unit_ids = render.flat_farm_tiles(_scenario((1, farm)))
    x0, x1, y0, y1 = _bounds(farm)
    footprint = {(x, y) for x in range(x0, x1) for y in range(y0, y1)}
    assert len(footprint) == 9 and set(by_tile) == footprint
    assert unit_ids == {id(farm)}
    corners = {(x0, y0), (x0, y1 - 1), (x1 - 1, y0), (x1 - 1, y1 - 1)}
    for tile, (terrain_id, color, mask) in by_tile.items():
        assert terrain_id == FAKE_FARM_TERRAIN and np.array_equal(color, _owner(farm)), tile
        bits = bin(mask).count("1")
        assert bits == (0 if tile == (x0 + 1, y0 + 1) else 2 if tile in corners else 1), (tile, mask)
    # The union of the outer bits is all four sides: a closed perimeter.
    union = 0
    for _tid, _c, mask in by_tile.values():
        union |= mask
    assert union == render.EDGE_LEFT | render.EDGE_RIGHT | render.EDGE_UP_LEFT | render.EDGE_UP_RIGHT

    hidden, hidden_ids = render.flat_farm_tiles(_scenario((1, farm)), UnitFilter(players=frozenset({2})))
    assert hidden == {} and hidden_ids == set()


# --- pixels -----------------------------------------------------------------------


def test_an_interior_farm_tile_equals_hand_painted_farm_terrain_at_every_mip(fake_farm) -> None:
    farm = _farm()
    x0, _x1, y0, _y1 = _bounds(farm)
    centre = (x0 + 1, y0 + 1)
    farm_cache = _cache(_scenario((1, farm)))
    painted = _scenario()
    painted.map_manager.get_tile(*centre).terrain_id = FAKE_FARM_TERRAIN
    painted_cache = _cache(painted)
    for mip in _mips(farm_cache):
        tp = farm_cache._mip_tile_px[mip]
        rect = (centre[0] * tp, centre[1] * tp, (centre[0] + 1) * tp, (centre[1] + 1) * tp)
        got = farm_cache._composite_rect(mip, *rect)
        assert np.all(got == TEXTURE_COLORS[FAKE_FARM_TERRAIN]), f"mip {mip}: the interior is not farm terrain"
        assert np.array_equal(got, painted_cache._composite_rect(mip, *rect)), mip


def test_the_outline_is_one_owner_colour_pixel_on_outer_edges_only_at_every_mip(fake_farm) -> None:
    farm = _farm()
    x0, x1, y0, y1 = _bounds(farm)
    cache = _cache(_scenario((1, farm)))
    owner = _owner(farm)
    for mip in _mips(cache):
        tp = cache._mip_tile_px[mip]
        canvas = cache._composite_rect(mip, 0, 0, *cache.canvas_dims(mip))
        is_owner = np.all(canvas == owner, axis=2)
        side_x, side_y = (x1 - x0) * tp, (y1 - y0) * tp
        assert is_owner.sum() == 2 * (side_x + side_y) - 4, f"mip {mip}: outline pixel count"
        expected = np.zeros_like(is_owner)
        box = expected[y0 * tp : y1 * tp, x0 * tp : x1 * tp]
        box[0, :] = box[-1, :] = box[:, 0] = box[:, -1] = True
        assert np.array_equal(is_owner, expected), f"mip {mip}: an outline pixel off the outer edge"


@pytest.mark.parametrize("grid", [grid_overlay.DEFAULT_GRID, GRID], ids=["grid-off", "grid-on"])
def test_stitched_chunks_match_the_full_render(fake_farm, grid) -> None:
    """Level 0 against the independent full render; every level's chunks
    against one whole-canvas composite at that level, which a seam-dependent
    drape or outline would fail."""
    farm = _farm()
    scn = _scenario((1, farm), (1, Unit(7.5, 7.5, PLAIN_CONST, reference_id=2)))
    x0, x1, _y0, _y1 = _bounds(farm)
    assert x0 * _tile_px() < CHUNK < x1 * _tile_px(), "the farm no longer straddles a chunk seam"
    cache = _cache(scn)
    cache.set_grid(grid)
    assert cache._farm_tiles, "no drape built -- test would be vacuous"
    full = _oracle(scn, grid)
    assert not np.array_equal(full, _rect_render(scn)), "the drape changed nothing -- vacuous"
    assert np.array_equal(_stitched(cache), full)
    for mip in _mips(cache):
        assert np.array_equal(_stitched(cache, mip), cache._composite_rect(mip, 0, 0, *cache.canvas_dims(mip))), mip


@pytest.mark.parametrize("farm_first", [True, False], ids=["farm-first", "farm-last"])
def test_a_1x1_unit_on_a_farm_tile_paints_over_the_drape(fake_farm, farm_first) -> None:
    farm = _farm()
    x0, _x1, y0, _y1 = _bounds(farm)
    unit = Unit(x0 + 1.5, y0 + 1.5, PLAIN_CONST, reference_id=2)
    pairs = [(1, farm), (1, unit)] if farm_first else [(1, unit), (1, farm)]
    scn = _scenario(*pairs)
    canvas = _stitched(_cache(scn))
    tp = _tile_px()
    tile = canvas[(y0 + 1) * tp : (y0 + 2) * tp, (x0 + 1) * tp : (x0 + 2) * tp]
    assert np.all(tile == _owner(unit)), "the unit's rect is not on top of the drape"
    assert np.array_equal(canvas, _oracle(scn))


def test_sprites_off_or_the_layer_off_is_todays_rect_render(fake_farm) -> None:
    scn = _scenario((1, _farm()))
    off = _cache(scn, sprites=False)
    assert off._farm_tiles is None
    assert np.array_equal(_stitched(off), render.overlay_units(render.render_terrain(scn), scn))
    layer_off = _cache(scn, layers=LayerState(farm_overlay=False))
    assert layer_off._farm_tiles is None
    assert np.array_equal(_stitched(layer_off), _rect_render(scn))


def _whole(cache) -> np.ndarray:
    return cache.render_rect(0, 0, *cache.canvas_dims(0)).copy()


def test_live_layer_and_sprite_round_trips_return_the_original_bytes(fake_farm) -> None:
    scn = _scenario((1, _farm()))
    cache = _cache(scn)
    before = _whole(cache)
    cache.set_layers(LayerState(farm_overlay=False))
    assert not np.array_equal(before, _whole(cache)), "the layer flip repainted nothing"
    cache.set_layers(LayerState())
    assert np.array_equal(before, _whole(cache))
    cache.set_sprites_enabled(False)
    assert not np.array_equal(before, _whole(cache)), "the sprite flip repainted nothing"
    cache.set_sprites_enabled(True)
    assert np.array_equal(before, _whole(cache))


# --- splices ------------------------------------------------------------------------

SPLICE_MIPS = (0, 1)


def _own(unit) -> tuple[int, int]:
    return (int(unit.x), int(unit.y))


def _occupied(unit) -> tuple[tuple[int, int], ...]:
    return tuple(render.unit_occupied_tiles(unit, MAP_W, MAP_H) or ())


def _renders(cache) -> list[np.ndarray]:
    return [cache.render_rect(0, 0, *cache.canvas_dims(mip), mip=mip).copy() for mip in SPLICE_MIPS]


def _edit(op: str, scn) -> UnitSplice:
    units = scn.unit_manager.units
    farm, villager = units[1][0], units[1][1]
    scn.unit_gen += 1
    if op == "move-farm":
        old_own, old_tiles = _own(farm), _occupied(farm)
        farm.x, farm.y = 7.5, 7.5
        return UnitSplice(1, 0, farm, old_own, _own(farm), old_tiles, _occupied(farm))
    if op == "add-farm":
        new = _farm(ref=9, xy=(7.5, 3.5))
        units[2].append(new)
        return UnitSplice(2, len(units[2]) - 1, new, None, _own(new), (), _occupied(new))
    if op == "delete-farm":
        old_own, old_tiles = _own(farm), _occupied(farm)
        del units[1][0]
        return UnitSplice(1, 0, farm, old_own, None, old_tiles, ())
    if op == "convert-farm":
        own, tiles = _own(farm), _occupied(farm)
        del units[1][0]
        units[2].append(farm)
        return UnitSplice(2, len(units[2]) - 1, farm, own, own, tiles, tiles, old_player_id=1)
    if op in ("replace-farm", "replace-to-farm"):
        # Find and Replace's replace_type(): the const changes and a changed span
        # re-anchors on the footprint's low corner, in one row rewrite.
        unit, const = (farm, PLAIN_CONST) if op == "replace-farm" else (villager, FAKE_FARM_CONST)
        index = units[1].index(unit)
        old_own, old_tiles = _own(unit), _occupied(unit)
        x0, _x1, y0, _y1 = _bounds(unit)
        half_x, half_y = (0.5, 0.5) if const == PLAIN_CONST else (FAKE_FARM_SPAN[0] / 2, FAKE_FARM_SPAN[1] / 2)
        unit.unit_const, unit.x, unit.y = const, x0 + half_x, y0 + half_y
        return UnitSplice(1, index, unit, old_own, _own(unit), old_tiles, _occupied(unit))
    assert op == "move-unit-off-farm"
    old_own, old_tiles = _own(villager), _occupied(villager)
    villager.x, villager.y = 9.5, 9.5
    return UnitSplice(1, 1, villager, old_own, _own(villager), old_tiles, _occupied(villager))


@pytest.mark.parametrize(
    "op", ["move-farm", "add-farm", "delete-farm", "convert-farm", "replace-farm", "replace-to-farm", "move-unit-off-farm"]
)
def test_a_spliced_edit_patches_to_a_fresh_cache(fake_farm, op, monkeypatch) -> None:
    """Only the touched tiles are patched, as the viewer's Flat branch does, so
    a stale _farm_tiles would leave the old drape (or miss the new one)."""
    farm = _farm()
    x0, _x1, y0, _y1 = _bounds(farm)
    scn = _scenario((1, farm), (1, Unit(x0 + 1.5, y0 + 1.5, PLAIN_CONST, reference_id=2)))
    cache = _cache(scn)
    before = _renders(cache)
    builds = []
    real = render.flat_farm_tiles
    monkeypatch.setattr(render, "flat_farm_tiles", lambda *a, **k: (builds.append(1), real(*a, **k))[1])

    splice = _edit(op, scn)
    assert cache.can_splice([splice]), "fixture is not testing the splice path"
    cache.invalidate_units([splice])
    cache.patch_rects(render.flat_tile_rects(splice.changed_tiles, _tile_px()))

    after = _renders(cache)
    assert not np.array_equal(after[0], before[0]), "the edit changed nothing on screen"
    assert builds == ([] if op == "move-unit-off-farm" else [1]), "the drape rebuild trigger"
    fresh = _cache(scn)
    assert cache._farm_tiles == fresh._farm_tiles
    for mip, got, want in zip(SPLICE_MIPS, after, _renders(fresh), strict=True):
        assert np.array_equal(got, want), f"mip {mip} differs from a fresh cache"


# --- a draping const with art (a pasture) -------------------------------------------


@pytest.fixture
def fake_pasture(fake_farm, tmp_path, monkeypatch):
    """A DRAPED_SPRITE_CONSTS member with a registered synthetic graphic whose
    ink is 1:2, so contain-fit leaves the drape visible down both sides."""
    graphics = tmp_path / unit_sprites.GRAPHICS_SUBPATH
    graphics.mkdir(parents=True)
    (graphics / f"{FILE_NAME}.sld").write_bytes(build_sld(4, canvas=unit_sprites.NATIVE_TILE_W))
    piece = {"unit_id": FAKE_PASTURE_CONST, "file_name": FILE_NAME, "angle_count": 4,
             "frame_count": 1, "dx": 0, "dy": 0, "parent": True}
    below = {**piece, "unit_id": FAKE_PASTURE_CONST + 100, "dy": unit_sprites.NATIVE_TILE_W}
    del below["parent"]
    entry = {"graphic_id": 1, "file_name": FILE_NAME, "angle_count": 4,
             "mirroring_mode": 6, "frame_count": 1, "pieces": [piece, below]}
    monkeypatch.setattr(unit_sprites, "graphic_map", lambda: {FAKE_PASTURE_CONST: entry})
    monkeypatch.setitem(render.BUILDING_TILE_SPANS, FAKE_PASTURE_CONST, FAKE_PASTURE_SPAN)
    monkeypatch.setitem(render.FOUNDATION_TERRAIN, FAKE_PASTURE_CONST, FAKE_FARM_TERRAIN)
    monkeypatch.setattr(render, "DRAPED_SPRITE_CONSTS", frozenset({FAKE_PASTURE_CONST}))
    asset_source.set_install_path_override(tmp_path)
    unit_sprites.clear_caches()
    yield
    asset_source.set_install_path_override(None)
    unit_sprites.clear_caches()


def test_a_draping_const_with_art_paints_drape_and_icon(fake_pasture) -> None:
    pasture = Unit(6.0, 6.0, FAKE_PASTURE_CONST)
    scn = _scenario((1, pasture))
    cache = _cache(scn)
    assert len(cache._farm_tiles) == 16, "the pasture did not drape"
    icon = cache._level_icons(0)[0]
    assert icon is not render.FLAT_DRAPED and icon.rgba.size, "the pasture lost its icon"
    canvas = _stitched(cache)
    assert np.array_equal(canvas, _oracle(scn))
    x0, x1, y0, y1 = _bounds(pasture)
    tp = _tile_px()
    block = canvas[y0 * tp : y1 * tp, x0 * tp : x1 * tp]
    drape = np.all(block == TEXTURE_COLORS[FAKE_FARM_TERRAIN], axis=2)
    outline = np.all(block == _owner(pasture), axis=2)
    assert drape.any() and outline.any(), "no drape shows beside the icon"
    assert (~drape & ~outline).any(), "no icon ink on the drape"


# --- pick -------------------------------------------------------------------------


@pytest.mark.parametrize("farm_first", [True, False], ids=["unit-after-farm", "farm-after-unit"])
def test_flat_pick_returns_a_unit_on_a_draped_farm_in_either_index_order(fake_farm, farm_first) -> None:
    """GH #84's Flat twin: the farm paints in the terrain pass, so the unit
    on its tile wins the pick; farms_draped=False keeps last-drawn-wins."""
    farm = _farm()
    x0, _x1, y0, _y1 = _bounds(farm)
    unit = Unit(x0 + 1.5, y0 + 1.5, PLAIN_CONST, reference_id=2)
    scn = _scenario(*([(1, farm), (1, unit)] if farm_first else [(1, unit), (1, farm)]))
    index = unit_pick.build_index(scn)
    tp = _tile_px()
    sx, sy = (x0 + 1) * tp + tp // 2, (y0 + 1) * tp + tp // 2
    got = unit_pick.pick_unit(index, "flat", sx, sy, tp, MAP_W, MAP_H, farms_draped=True)
    assert got is not None and got.unit is unit
    plain = unit_pick.pick_unit(index, "flat", sx, sy, tp, MAP_W, MAP_H, farms_draped=False)
    assert plain is not None and plain.unit is (unit if farm_first else farm)
    # Off the unit's tile, the farm is still pickable.
    edge = unit_pick.pick_unit(index, "flat", x0 * tp + 2, y0 * tp + 2, tp, MAP_W, MAP_H, farms_draped=True)
    assert edge is not None and edge.unit is farm


# --- real file ----------------------------------------------------------------------

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.mark.corpus
def test_every_farms_interior_tile_is_its_farm_terrain_on_a_real_file() -> None:
    if asset_source.get_install_path() is None:
        pytest.skip("no AoE2:DE install visible -- set AOE2DE_INSTALL_PATH (conftest hides config.yaml)")
    path = EXAMPLES / "old-allies-final-v2.aoe2scenario"
    if not path.exists():
        pytest.skip(f"{path.name} is not in examples/")
    from descape.scenario_io import load_map_and_units

    scn = load_map_and_units(path)
    mm = scn.map_manager
    tp = render.tile_pixels_for_map(mm.map_width, mm.map_height)
    cache = FlatChunkCache(scn, tp, sprites=True)
    index = unit_pick.build_index(scn)
    checked = 0
    for units in scn.unit_manager.units:
        for unit in units:
            terrain_id = render._terrain_overlay_for(unit.unit_const)
            if terrain_id is None or unit.unit_const in render.DRAPED_SPRITE_CONSTS:
                continue
            x0, x1, y0, y1 = render.unit_tile_bounds(unit, mm.map_width, mm.map_height)
            if (x1 - x0, y1 - y0) != (3, 3):
                continue
            cx, cy = x0 + 1, y0 + 1
            if len(index.by_tile.get((cx, cy), ())) != 1 or cache._farm_tiles.get((cx, cy), (None,))[0] != terrain_id:
                continue  # a unit on it, or an overlapping farm's drape
            got = cache._composite_rect(0, cx * tp, cy * tp, (cx + 1) * tp, (cy + 1) * tp)
            want = render._top_block(mm.get_tile(cx, cy), tp, True, None, terrain_override=terrain_id)
            assert np.array_equal(got, want), (unit.unit_const, cx, cy)
            checked += 1
    assert checked > 0, "no farm interior checked -- the file has no farms in reach"
