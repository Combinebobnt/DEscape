"""GH #66: a placed Pasture (1897/1893) draws its annex tree -- hut, four posts,
twenty fences -- over its pasture-terrain drape, each post and fence picking a
shape variant seeded by the placed unit's reference_id.

Install-free: frame dispatch is checked through _frame_key() (resolved frame
indices, no decode), and the drape/sprite coexistence through a stubbed
sprite_pieces_for(). The committed piece table itself is pinned in
tests/test_unit_graphic_map.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from descape import render, unit_sprites
from descape.terrain_palette import PLAYER_COLORS

PASTURE = 1897
PASTURE_TERRAIN = 117


def test_seeded_variant_is_deterministic_and_in_range():
    for seed in (0, 1, 9322, 2**31 - 1, -1):
        for index in range(25):
            for count in (6, 8):
                first = unit_sprites.seeded_variant(seed, index, count)
                assert 0 <= first < count
                assert first == unit_sprites.seeded_variant(seed, index, count)


def test_seeded_variant_varies_within_and_across_seeds():
    within = {unit_sprites.seeded_variant(9322, index, 6) for index in range(20)}
    assert len(within) >= 4, within
    across = {unit_sprites.seeded_variant(seed, 3, 6) for seed in range(9000, 9040)}
    assert len(across) == 6, across


def _pasture_entry():
    return unit_sprites.graphic_map()[PASTURE]


def test_frame_key_is_stable_per_seed_and_differs_across_seeds():
    entry = _pasture_entry()
    a = unit_sprites._frame_key(PASTURE, entry, 7.0, seed=9322)
    assert a == unit_sprites._frame_key(PASTURE, entry, 7.0, seed=9322)
    # Rotation no longer reaches a seeded piece.
    assert a == unit_sprites._frame_key(PASTURE, entry, 0.0, seed=9322)
    keys = {unit_sprites._frame_key(PASTURE, entry, 7.0, seed=s) for s in range(9000, 9010)}
    assert len(keys) == 10


def test_seeded_frames_stay_in_each_pieces_own_range_and_the_hut_is_fixed():
    entry = _pasture_entry()
    for seed in range(100):
        frames = unit_sprites._frame_key(PASTURE, entry, 7.0, seed=seed)
        for piece, frame in zip(entry["pieces"], frames, strict=True):
            if piece.get("parent"):
                assert frame == 0
            else:
                assert 0 <= frame < piece["angle_count"] * piece["frame_count"], (piece, frame)


def test_seed_none_keeps_the_rotation_dispatch():
    entry = _pasture_entry()
    unseeded = unit_sprites._frame_key(PASTURE, entry, 3.0)
    by_rotation = tuple(
        unit_sprites._frame_for(p["unit_id"], p, 3.0) for p in entry["pieces"]
    )
    assert unseeded == by_rotation


@pytest.mark.parametrize("unit_const", [1889, 109, 71, 64, 487, 72, 1888])
def test_a_seed_never_reaches_an_unseeded_graphic(unit_const):
    """1889, town centres, gates and walls carry no seeded piece, so a seed
    must leave every resolved frame (and so every render) unchanged."""
    entry = unit_sprites.graphic_map()[unit_const]
    for rotation in (0.0, 1.0, 3.0, 7.0):
        assert unit_sprites._frame_key(unit_const, entry, rotation, seed=12345) == (
            unit_sprites._frame_key(unit_const, entry, rotation)
        )


# --- drape + sprite coexistence, synthetic scenario ------------------------

MAP_W = MAP_H = 12


@dataclass
class Tile:
    x: int
    y: int
    elevation: int = 0
    terrain_id: int = 1
    layer: int = -1


@dataclass
class Unit:
    x: float
    y: float
    unit_const: int
    rotation: float = 7.0
    reference_id: int = 9322


@dataclass
class BareUnit:
    x: float
    y: float
    unit_const: int
    rotation: float = 7.0


class _MapManager:
    def __init__(self):
        self.map_width, self.map_height = MAP_W, MAP_H
        self.terrain = [Tile(x, y) for y in range(MAP_H) for x in range(MAP_W)]
        self._by_xy = {(t.x, t.y): t for t in self.terrain}

    def get_tile(self, x, y):
        return self._by_xy[(x, y)]


class _UnitManager:
    def __init__(self, units):
        self.units = units


class _Scenario:
    def __init__(self, units_by_player):
        self.map_manager = _MapManager()
        self.unit_manager = _UnitManager(units_by_player)
        self.player_colors = tuple(PLAYER_COLORS)
        self.team_indices = tuple(range(len(PLAYER_COLORS)))


@pytest.fixture
def stub_pieces(monkeypatch):
    seen = []

    def fake(unit_const, rotation, team_index, half_w, tree_scale=1.0, seed=None, hero_glow=False):
        seen.append((unit_const, seed))
        draw = unit_sprites.SpriteDraw(rgba=np.full((4, 4, 4), 255, np.uint8), hotspot_x=2, hotspot_y=2)
        return [unit_sprites.SpritePiece(draw=draw, dx=0, dy=0, slot=(1, 2))]

    monkeypatch.setattr(unit_sprites, "sprite_pieces_for", fake)
    return seen


def _layer(scn):
    _, elevations, proj = render.render_terrain_iso_with_proj(scn, with_units=False)
    return render.sprite_draws_by_anchor(scn, proj, elevations)


def test_a_pasture_keeps_its_drape_under_its_sprite(stub_pieces):
    unit = Unit(6.0, 6.0, PASTURE)
    sprites = _layer(_Scenario([[], [unit]]))
    assert set(sprites.farm_by_tile) == {(x, y) for x in range(4, 8) for y in range(4, 8)}
    assert {t[0] for t in sprites.farm_by_tile.values()} == {PASTURE_TERRAIN}
    assert sprites.by_anchor, "the pasture's pieces must still paint"
    assert (5, 6) in sprites.by_anchor  # slot (1, 2) from footprint low corner (4, 4)
    assert id(unit) in sprites.skip_ids
    assert stub_pieces == [(PASTURE, 9322)]


def test_a_unit_without_reference_id_resolves_unseeded(stub_pieces):
    _layer(_Scenario([[], [BareUnit(6.0, 6.0, PASTURE)]]))
    assert stub_pieces == [(PASTURE, None)]


def test_an_ordinary_sprite_building_still_gets_no_drape(stub_pieces):
    """House (70) has a foundation terrain too; its sprite still wins outright."""
    assert 70 in render.FOUNDATION_TERRAIN
    sprites = _layer(_Scenario([[], [Unit(6.0, 6.0, 70)]]))
    assert sprites.farm_by_tile == {}
    assert sprites.by_anchor
