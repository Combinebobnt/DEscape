"""The wall render rule (unit_sprites.wall_drawn_shape()) re-measured on the
corpus, install-free, rather than trusted from the figures in its docstring.

- Integer-encoded files: every wall with at most one neighbour among its 8
  stores the tower already, so the shipped overrides are empty and no
  game-saved file draws differently. That is the premise of the lone-wall
  rule: AoE2:DE saves such a wall as a tower.
- Radian-encoded files: every on-map wall draws its 8-bit derivation (a lone
  wall the tower), diagonal runs included.
"""

from __future__ import annotations

import pytest

from descape import render, scenario_io, unit_sprites

pytestmark = pytest.mark.corpus

ELCID = "C2_ElCid_coop_1_v0_16.aoe2scenario"


def _walls(scenario):
    """(player_id, i, stored rotation, 8-bit mask or None off-map) per eligible wall, plus the file's encoding."""
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height
    connectors = unit_sprites.wall_connector_consts()
    tiles: set[tuple[int, int]] = set()
    for units in scenario.unit_manager.units:
        for unit in units:
            if unit.unit_const in connectors:
                occupied = render.unit_occupied_tiles(unit, w, h)
                if occupied is not None:
                    tiles.update(occupied)
    walls = []
    for player_id, units in enumerate(scenario.unit_manager.units):
        for i, unit in enumerate(units):
            if not unit_sprites.rotation_variant_eligible(unit.unit_const):
                continue
            bounds = render.unit_tile_bounds(unit, w, h)
            mask = None if bounds is None else unit_sprites.neighbour_mask(bounds[0], bounds[2], tiles, diagonals=True)
            walls.append((player_id, i, render.stored_rotation(player_id, unit), mask))
    radian = unit_sprites.file_is_radian([rotation for _p, _i, rotation, _m in walls], 5)
    return walls, radian


def _loaded(corpus_files):
    for path in corpus_files:
        scenario = scenario_io.load_map_and_units(path)
        walls, radian = _walls(scenario)
        if walls:
            yield path, scenario, walls, radian


def test_integer_files_store_a_tower_wherever_a_wall_has_at_most_one_neighbour(corpus_files):
    files = lone = 0
    bad = []
    for path, scenario, walls, radian in _loaded(corpus_files):
        if radian:
            continue
        files += 1
        assert render.wall_variant_rotation_overrides(scenario) == {}, path.name
        for player_id, i, rotation, mask in walls:
            if mask is not None and mask & (mask - 1) == 0:
                lone += 1
                if unit_sprites.variant_index(rotation, 5) != unit_sprites.WALL_TOWER:
                    bad.append((path.name, player_id, i, rotation, mask))
    if not files:
        pytest.skip("no integer-encoded file with walls in this corpus")
    assert lone, "no wall with at most one neighbour, so this proves nothing"
    assert bad == []


def test_radian_files_draw_every_wall_by_the_8_bit_table(corpus_files):
    files = 0
    bad = []
    elcid_shapes: set[float] = set()
    for path, scenario, walls, radian in _loaded(corpus_files):
        if not radian:
            continue
        files += 1
        overrides = render.wall_variant_rotation_overrides(scenario)
        for player_id, i, _rotation, mask in walls:
            if mask is None:
                if (player_id, i) in overrides:
                    bad.append((path.name, player_id, i, "off-map wall has an entry"))
                continue
            derived = unit_sprites.wall_variant_from_neighbours8(mask)
            want = float(unit_sprites.WALL_TOWER if derived is None else derived)
            if overrides.get((player_id, i)) != want:
                bad.append((path.name, player_id, i, mask, overrides.get((player_id, i)), want))
            if path.name == ELCID and mask in (unit_sprites.NE | unit_sprites.SW, unit_sprites.NW | unit_sprites.SE):
                elcid_shapes.add(overrides.get((player_id, i)))
    if not files:
        pytest.skip("no radian-encoded file with walls in this corpus")
    assert bad == [], bad[:10]
    if ELCID in {p.name for p in corpus_files}:
        assert elcid_shapes == {4.0, 3.0}, elcid_shapes
