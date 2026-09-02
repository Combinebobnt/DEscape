"""Farm-terrain rendering: a placed Farm and its RFarm/Pasture family
paint their real crop terrain across their footprint instead of a
coloured mark, with a player-colour perimeter
outline so a placed unit stays distinguishable from hand-painted farm
terrain. Gated on the same Show sprites toggle / SpriteLayer as real unit
sprites -- see render._terrain_overlay_for and SpriteLayer.farm_by_tile.

In-game, a farm's crop is a TERRAIN TILE BLEND, not a unit sprite --
Unit.building.foundation_terrain_id in the real .dat gives the exact
const -> terrain mapping directly, measured against a real install (see the
plan doc). The first three tests pin that measurement against the
committed unit_render_data.json/unit_graphic_map.json tables, no install
needed. The rest use a synthetic FAKE_FARM_CONST (not a real unit_const),
so the layer/oracle/chunk checks don't depend on genieutils or an install
either -- only the real-asset guard below does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from descape import asset_source, iso_geometry, render, unit_sprites
from descape.terrain_palette import PLAYER_COLORS

MAP_W = MAP_H = 12
FAKE_FARM_CONST = 999001
FAKE_FARM_SPAN = (3, 3)
FAKE_FARM_TERRAIN = 7
BACKGROUND_TERRAIN = 1


def _require_install():
    if asset_source.get_install_path() is None:
        pytest.skip(
            "no AoE2:DE install visible -- set AOE2DE_INSTALL_PATH to one. The configured "
            "config.yaml install is deliberately hidden by conftest._isolated_settings."
        )


# --- data pin, against the real committed tables (no install needed) ----

# const -> terrain id, from Unit.building.foundation_terrain_id in the real
# .dat, measured directly against a real AoE2DE install -- not re-derived
# here, since the whole point of this test is to pin the generated tables
# against drift.
FARM_FAMILY_TERRAIN = {
    50: 7,  # FARM
    357: 8,  # FARM_D
    1187: 63,  # RFARM
    1188: 64,  # RFARM_D
    1893: 63,  # PASTURE_MANGROVE
    1894: 64,  # PASTURE_D (MANGROVE)
    1897: 117,  # PASTURE
    1898: 118,  # PASTURE_D
}


def test_terrain_overlay_for_pins_the_farm_family():
    for const, terrain_id in FARM_FAMILY_TERRAIN.items():
        assert render._terrain_overlay_for(const) == terrain_id, const


def test_terrain_overlay_for_none_for_dropsite_helpers():
    """FARMDROP/FARMSTACK/RFARMDROP are invisible internal helpers with
    foundation_terrain_id == -1 in the real .dat -- absent from
    FOUNDATION_TERRAIN entirely, unlike the family above, so they correctly
    keep the coloured mark rather than becoming an invisible terrain patch."""
    for const in (1193, 1194, 1195):
        assert render._terrain_overlay_for(const) is None, const


def test_terrain_overlay_for_none_when_a_real_sprite_exists():
    """House (70) has both a foundation terrain (purely an in-game
    construction-outline hint) and a real .sld -- the sprite must win."""
    assert 70 in render.FOUNDATION_TERRAIN, "fixture assumption broke: House lost its foundation terrain"
    assert 70 in unit_sprites.graphic_map(), "fixture assumption broke: House lost its .sld entry"
    assert render._terrain_overlay_for(70) is None


@pytest.mark.parametrize("terrain_id", sorted(set(FARM_FAMILY_TERRAIN.values())))
def test_farm_terrain_ids_resolve_on_a_real_install(terrain_id):
    """The 'farms silently stopped painting' tripwire -- if any of these six
    terrain ids stops resolving to a real texture (a moved/renamed .dds, a
    stale terrain_texture_map.json), a farm silently falls back to the
    default-color square instead of raising. Deliberately not corpus-marked,
    same reasoning as tests/test_wall_variants_real_assets.py: this is one
    cheap lookup per id, not a render, so it earns its place in the default
    tier rather than only firing in the ~27-minute one."""
    _require_install()
    texture = asset_source.get_terrain_texture_array(terrain_id)
    assert texture is not None, terrain_id


# --- synthetic scenario, for the layer/oracle/chunk checks ---------------
#
# Same duck-typing tests/test_sprite_chunks.py and friends use for synthetic
# maps; this module keeps its own copy rather than importing one, matching
# that file's own stated convention.


@dataclass
class Tile:
    x: int
    y: int
    elevation: int
    terrain_id: int = BACKGROUND_TERRAIN
    layer: int = -1


@dataclass
class Unit:
    x: float
    y: float
    unit_const: int
    rotation: float = 0.0


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
        # Identity default -- no synthetic scenario here stores a color
        # override.
        self.player_colors = tuple(PLAYER_COLORS)
        self.team_indices = tuple(range(len(PLAYER_COLORS)))


def _scenario(units_by_player, *, terrain_id=BACKGROUND_TERRAIN, elevation=0):
    tiles = [Tile(x, y, elevation, terrain_id=terrain_id) for y in range(MAP_H) for x in range(MAP_W)]
    return _Scenario(tiles, units_by_player)


@pytest.fixture
def fake_farm(monkeypatch):
    """FAKE_FARM_CONST doesn't exist in the real .dat, so it resolves
    nowhere in the real graphic map -- a synthetic stand-in for the real
    Farm family that gives sprite_for() a guaranteed None without needing a
    real install, while still letting _terrain_overlay_for see a foundation
    terrain via this fixture's own BUILDING_TILE_SPANS/FOUNDATION_TERRAIN
    entries."""
    assert FAKE_FARM_CONST not in unit_sprites.graphic_map(), "picked const collides with a real one"
    monkeypatch.setitem(render.BUILDING_TILE_SPANS, FAKE_FARM_CONST, FAKE_FARM_SPAN)
    monkeypatch.setitem(render.FOUNDATION_TERRAIN, FAKE_FARM_CONST, FAKE_FARM_TERRAIN)


TEXTURE_COLORS = {BACKGROUND_TERRAIN: (10, 20, 30), FAKE_FARM_TERRAIN: (200, 150, 40)}


def _synthetic_textures(monkeypatch) -> int:
    """Deterministic solid-color textures, one per terrain id used here
    (TEXTURE_COLORS), sized exactly tile_px so _crop_offset's modulo is
    always (0, 0) -- the same fixture shape tests/test_sloped_render.py
    uses to keep the oracle exact rather than dependent on install-specific
    art. Returns tile_px."""
    tile_px = render.tile_pixels_for_map(MAP_W, MAP_H)
    textures = {
        tid: np.full((tile_px, tile_px, 3), color, dtype=np.uint8)
        for tid, color in TEXTURE_COLORS.items()
    }
    monkeypatch.setattr(asset_source, "get_terrain_texture_array", lambda tid: textures.get(tid))
    return tile_px


def test_farm_layer_covers_the_whole_footprint_with_a_closed_perimeter(fake_farm):
    """A 3x3 farm at (5.0, 5.0) spans tiles x,y in {4, 5, 6} -- 9 entries,
    the corner tiles get two boundary edges, the four edge-midpoint tiles
    get one, and the center tile gets none (an interior farm tile has no
    neighbour outside the footprint to draw a boundary against)."""
    unit = Unit(5.0, 5.0, FAKE_FARM_CONST)
    scn = _scenario([[], [unit]])
    _, elevations, proj = render.render_terrain_iso_with_proj(scn, with_units=True, with_sprites=True)
    sprites = render.sprite_draws_by_anchor(scn, proj, elevations)

    assert set(sprites.farm_by_tile) == {(x, y) for x in (4, 5, 6) for y in (4, 5, 6)}
    assert id(unit) in sprites.skip_ids

    corners = {(4, 4), (4, 6), (6, 4), (6, 6)}
    for (x, y), (terrain_id, _color, mask) in sprites.farm_by_tile.items():
        assert terrain_id == FAKE_FARM_TERRAIN, (x, y)
        bits = bin(mask).count("1")
        if (x, y) == (5, 5):
            assert mask == 0, (x, y, mask)
        elif (x, y) in corners:
            assert bits == 2, (x, y, mask)
        else:
            assert bits == 1, (x, y, mask)


def test_perimeter_stroke_paints_the_outer_edge_and_not_the_internal_one(monkeypatch, fake_farm):
    """The mask-bit checks above never look at a pixel -- a transposed side
    mapping or a silently-empty edge array would still pass them. Checks the
    actual stroke: (4, 5)'s "left" edge faces (3, 5), OUTSIDE the footprint,
    and must carry the owning player's colour; (5, 5)'s "left" edge faces
    (4, 5), an INTERNAL boundary between two farm tiles, and must not."""
    tile_px = _synthetic_textures(monkeypatch)
    unit = Unit(5.0, 5.0, FAKE_FARM_CONST)
    scn = _scenario([[], [unit]])
    img, _elevations, proj = render.render_terrain_iso_with_proj(scn, with_units=True, with_sprites=True)

    outline_color = np.array(render.PLAYER_COLORS[1 % len(render.PLAYER_COLORS)], dtype=np.uint8)
    dst_y, dst_x = iso_geometry.tile_edge_indices(tile_px, "left")
    assert dst_y.size, "fixture produced an empty edge -- proves nothing"

    outer_x, outer_y = iso_geometry.tile_screen_origin(4, 5, 0, proj)
    outer_pixels = img[outer_y + dst_y, outer_x + dst_x]
    assert np.all(outer_pixels == outline_color), "the footprint's own outer edge was not stroked"

    inner_x, inner_y = iso_geometry.tile_screen_origin(5, 5, 0, proj)
    inner_pixels = img[inner_y + dst_y, inner_x + dst_x]
    assert not np.any(np.all(inner_pixels == outline_color, axis=-1)), (
        "an internal boundary between two farm tiles was stroked -- edge_mask leaked a bit "
        "it shouldn't have set"
    )


def _tile_diamond_pixels(img: np.ndarray, x: int, y: int, elevation: int, tile_px: int, proj) -> np.ndarray:
    dst_y, dst_x, _src_y, _src_x = iso_geometry.diamond_indices(tile_px)
    base_x, base_y = iso_geometry.tile_screen_origin(x, y, elevation, proj)
    return img[base_y + dst_y, base_x + dst_x]


def test_farm_interior_tile_matches_hand_painted_terrain(monkeypatch, fake_farm):
    """Non-vacuous oracle: the farm's own CENTER tile (no perimeter stroke,
    per the layer test above) must be pixel-identical to a second scenario
    with the same terrain hand-painted there and no unit at all."""
    tile_px = _synthetic_textures(monkeypatch)

    farm_scn = _scenario([[], [Unit(5.0, 5.0, FAKE_FARM_CONST)]])
    farm_img, _elevations, proj = render.render_terrain_iso_with_proj(
        farm_scn, with_units=True, with_sprites=True
    )

    painted_scn = _scenario([[]])
    for tile in painted_scn.map_manager.terrain:
        if 4 <= tile.x < 7 and 4 <= tile.y < 7:
            tile.terrain_id = FAKE_FARM_TERRAIN
    painted_img, _painted_elevations, painted_proj = render.render_terrain_iso_with_proj(
        painted_scn, with_units=True
    )

    got = _tile_diamond_pixels(farm_img, 5, 5, 0, tile_px, proj)
    expected = _tile_diamond_pixels(painted_img, 5, 5, 0, tile_px, painted_proj)
    assert np.array_equal(got, expected)


def test_without_a_foundation_terrain_the_interior_tile_diverges(monkeypatch):
    """Mutation half: give FAKE_FARM_CONST a footprint but deliberately NO
    FOUNDATION_TERRAIN entry (unlike the fake_farm fixture). The unit then
    falls back to the coloured mark and the center tile keeps
    BACKGROUND_TERRAIN, so the comparison above must FAIL here -- otherwise
    it is measuring nothing, the same trap the rotation fix fell into by
    shipping green while palisades were invisible."""
    tile_px = _synthetic_textures(monkeypatch)
    monkeypatch.setitem(render.BUILDING_TILE_SPANS, FAKE_FARM_CONST, FAKE_FARM_SPAN)
    assert FAKE_FARM_CONST not in render.FOUNDATION_TERRAIN

    farm_scn = _scenario([[], [Unit(5.0, 5.0, FAKE_FARM_CONST)]])
    farm_img, _elevations, proj = render.render_terrain_iso_with_proj(
        farm_scn, with_units=True, with_sprites=True
    )

    painted_scn = _scenario([[]])
    for tile in painted_scn.map_manager.terrain:
        if 4 <= tile.x < 7 and 4 <= tile.y < 7:
            tile.terrain_id = FAKE_FARM_TERRAIN
    painted_img, _painted_elevations, painted_proj = render.render_terrain_iso_with_proj(
        painted_scn, with_units=True
    )

    got = _tile_diamond_pixels(farm_img, 5, 5, 0, tile_px, proj)
    expected = _tile_diamond_pixels(painted_img, 5, 5, 0, tile_px, painted_proj)
    assert not np.array_equal(got, expected)


RAISED_ELEV = 2


def test_elevated_farm_tile_skirt_carries_crop_not_background(monkeypatch, fake_farm):
    """The whole reason this plan chose a terrain-id OVERRIDE at
    _render_tile_iso over compositing an overlay on top: skirts sample from
    that same function's own top_block, so a raised farm tile's cliff face
    must carry crop texture, not the background terrain an on-top overlay
    would have left showing. Isolates the SKIRT strip specifically (the
    drop_px rows below the diamond's own bottom edge) -- the perimeter
    outline never reaches there, it only strokes the diamond's own boundary
    row -- so this can't be mistaken for the outline stroke."""
    tile_px = _synthetic_textures(monkeypatch)

    farm_scn = _scenario([[], [Unit(5.0, 5.0, FAKE_FARM_CONST)]])
    for tile in farm_scn.map_manager.terrain:
        if (tile.x, tile.y) == (4, 4):
            tile.elevation = RAISED_ELEV
    farm_img, _elevations, proj = render.render_terrain_iso_with_proj(
        farm_scn, with_units=True, with_sprites=True
    )

    painted_scn = _scenario([[]])
    for tile in painted_scn.map_manager.terrain:
        if 4 <= tile.x < 7 and 4 <= tile.y < 7:
            tile.terrain_id = FAKE_FARM_TERRAIN
        if (tile.x, tile.y) == (4, 4):
            tile.elevation = RAISED_ELEV
    painted_img, _painted_elevations, painted_proj = render.render_terrain_iso_with_proj(
        painted_scn, with_units=True
    )

    assert proj.elev_step == painted_proj.elev_step, "proj drifted between the two renders"
    drop_px = RAISED_ELEV * proj.elev_step
    dst_y, dst_x, _src_y, _src_x = iso_geometry.skirt_quad_indices(tile_px, drop_px, "left")
    assert dst_y.size, "fixture produced an empty skirt -- proves nothing"

    base_x, base_y = iso_geometry.tile_screen_origin(4, 4, RAISED_ELEV, proj)
    got = farm_img[base_y + dst_y, base_x + dst_x]
    base_x2, base_y2 = iso_geometry.tile_screen_origin(4, 4, RAISED_ELEV, painted_proj)
    expected = painted_img[base_y2 + dst_y, base_x2 + dst_x]
    assert np.array_equal(got, expected)

    # Non-vacuity: the skirt must not incidentally read as background color
    # regardless of the override -- confirms this could actually have failed.
    background_color = np.array(TEXTURE_COLORS[BACKGROUND_TERRAIN], dtype=np.uint8)
    assert not np.all(got == background_color), (
        "skirt pixels came out as the background color -- the override isn't reaching the skirt"
    )


CHUNK = 128


def _stitched(scn, elevations, proj, sprites) -> np.ndarray:
    """Every CHUNK-sized rect composited in isolation, stitched back
    together -- the same shape tests/test_sprite_chunks.py's own
    ``_chunked`` uses for the sprite case."""
    mm = scn.map_manager
    tile_px = render.tile_pixels_for_map(mm.map_width, mm.map_height)
    units_by_tile = render._units_by_tile(scn)
    bboxes = render._building_bboxes_iso(units_by_tile, mm.map_width, mm.map_height, proj, elevations)
    merged = render.merge_sprite_bboxes(bboxes, sprites)
    out = np.zeros((proj.canvas_h + (iso_geometry.MAX_ELEVATION - iso_geometry.MIN_ELEVATION) * proj.elev_step,
                     proj.canvas_w, 3), dtype=np.uint8)
    h, w = out.shape[:2]
    for y0 in range(0, h, CHUNK):
        for x0 in range(0, w, CHUNK):
            x1, y1 = min(x0 + CHUNK, w), min(y0 + CHUNK, h)
            out[y0:y1, x0:x1] = render.composite_rect_iso(
                scn, x0, y0, x1, y1, elevations, proj, tile_px, units_by_tile, merged, sprites=sprites,
            )
    return out


def test_stitched_chunks_match_the_full_farm_render(monkeypatch, fake_farm):
    """The layer reaches both the full-canvas and the chunked path
    identically -- the farm counterpart to
    tests/test_sprite_chunks.py's own stitched-chunk check."""
    _synthetic_textures(monkeypatch)
    scn = _scenario([[], [Unit(5.0, 5.0, FAKE_FARM_CONST), Unit(9.0, 2.0, FAKE_FARM_CONST)]])
    full, elevations, proj = render.render_terrain_iso_with_proj(scn, with_units=True, with_sprites=True)
    sprites = render.sprite_draws_by_anchor(scn, proj, elevations)
    assert sprites.farm_by_tile, "the fixture resolved no farm tiles, so this proves nothing"

    got = _stitched(scn, elevations, proj, sprites)
    assert np.array_equal(got, full)


def test_stitched_chunks_match_on_elevated_farm_terrain(monkeypatch, fake_farm):
    """Flat terrain is the configuration where the bystander/candidate logic
    (merge_sprite_bboxes, the skirt) can't go wrong -- an elevation
    difference is what actually exercises it, so this repeats the check
    above with raised tiles inside both footprints."""
    _synthetic_textures(monkeypatch)
    scn = _scenario([[], [Unit(5.0, 5.0, FAKE_FARM_CONST), Unit(9.0, 2.0, FAKE_FARM_CONST)]])
    for tile in scn.map_manager.terrain:
        if (tile.x, tile.y) in {(4, 4), (5, 5), (9, 2)}:
            tile.elevation = RAISED_ELEV
    full, elevations, proj = render.render_terrain_iso_with_proj(scn, with_units=True, with_sprites=True)
    sprites = render.sprite_draws_by_anchor(scn, proj, elevations)
    assert sprites.farm_by_tile, "the fixture resolved no farm tiles, so this proves nothing"

    got = _stitched(scn, elevations, proj, sprites)
    assert np.array_equal(got, full)
