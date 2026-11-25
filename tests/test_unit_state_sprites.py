"""A unit whose Unit State is not Alive draws faded, measured on the real game art.

render_sprites._resolve_unit_sprite() is the one sprite seam Stepped and
Sloped share (every walk, splice and drag preview goes through it), so each
check here reads unit_sprite_draws_at(), the thin public call into it.
Top-down Flat's icon takes the same alpha in render_flat.ghosted_icon().

Needs AOE2DE_INSTALL_PATH rather than the configured install, for the reason
tests/test_sprite_variant14_real_assets.py's docstring spells out. Not
`corpus`-marked: a handful of frame decodes from files on disk.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from descape import asset_source, iso_geometry, render_flat, render_iso, render_sprites, unit_sprites, unit_state
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units

OAK = 411
STONE_WALL = 117
WATCH_TOWER = 79
VILLAGER = 83
EMPTY_TC_ANNEX = 890  # an invisible object: drawn as DEscape's own editor marker
REMOVE = 7
FOUNDATION = 0
POINT = (40.5, 40.5)


@dataclass
class _Unit:
    x: float
    y: float
    unit_const: int
    status: int = unit_state.ALIVE
    rotation: float = 0.0
    reference_id: int = 1
    garrisoned_in_id: int = -1
    z: float = 0.0


@pytest.fixture
def scene():
    """(scenario, proj, elevations) over the blank template, with the real install."""
    if asset_source.get_install_path() is None:
        pytest.skip(
            "no AoE2:DE install visible. Set AOE2DE_INSTALL_PATH to one. The configured "
            "config.yaml install is deliberately hidden by conftest._isolated_settings."
        )
    unit_sprites.clear_caches()
    scenario = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    elevations, proj = render_iso.elevations_and_proj(scenario)
    yield scenario, proj, elevations
    unit_sprites.clear_caches()


def _place(scenario, player: int, *units: _Unit) -> None:
    scenario.unit_manager.units[player][:] = list(units)
    scenario.unit_gen += 1


def _draws(scene, unit: _Unit, player: int = 1, **kwargs) -> list[tuple[object, int, int]]:
    scenario, proj, elevations = scene
    _place(scenario, player, unit)
    return render_sprites.unit_sprite_draws_at(scenario, proj, elevations, None, player, unit, **kwargs)


def _one(draws) -> tuple[object, int, int]:
    assert len(draws) == 1, f"expected one flattened draw, got {len(draws)}"
    return draws[0]


def test_a_ghosted_tree_draws_its_alive_art_at_ghost_alpha(scene):
    alive_draw, ax, ay = _one(_draws(scene, _Unit(*POINT, OAK)))
    ghost_draw, gx, gy = _one(_draws(scene, _Unit(*POINT, OAK, status=REMOVE)))
    assert (gx, gy) == (ax, ay)
    assert (ghost_draw.hotspot_x, ghost_draw.hotspot_y) == (alive_draw.hotspot_x, alive_draw.hotspot_y)
    assert ghost_draw.rgba.shape == alive_draw.rgba.shape
    alpha = alive_draw.rgba[..., 3]
    assert int(alpha.max()) == 255, "the tree drew no opaque ink"
    expected = np.rint(alpha * unit_state.GHOST_ALPHA).astype(np.uint8)
    assert np.array_equal(ghost_draw.rgba[..., 3], expected)
    ink = alpha > 0
    assert np.array_equal(ghost_draw.rgba[..., :3][ink], alive_draw.rgba[..., :3][ink])


@pytest.mark.parametrize("status", sorted(unit_state.GHOSTED_STATES))
def test_every_ghosted_state_fades_a_villager(scene, status):
    alive = _one(_draws(scene, _Unit(*POINT, VILLAGER)))[0]
    ghost = _one(_draws(scene, _Unit(*POINT, VILLAGER, status=status)))[0]
    assert int(ghost.rgba[..., 3].max()) == round(int(alive.rgba[..., 3].max()) * unit_state.GHOST_ALPHA)


def test_a_ghosted_tree_keeps_the_small_trees_extent(scene):
    small = 0.6
    alive_full = _one(_draws(scene, _Unit(*POINT, OAK)))[0]
    alive_small = _one(_draws(scene, _Unit(*POINT, OAK), tree_scale=small))[0]
    ghost_small = _one(_draws(scene, _Unit(*POINT, OAK, status=REMOVE), tree_scale=small))[0]
    assert alive_small.rgba.shape != alive_full.rgba.shape, "Small Trees did not shrink the fixture"
    assert ghost_small.rgba.shape == alive_small.rgba.shape
    assert (ghost_small.hotspot_x, ghost_small.hotspot_y) == (alive_small.hotspot_x, alive_small.hotspot_y)


def test_a_ghosted_wall_under_translucent_walls_multiplies_both_alphas(scene):
    wall_alpha = 0.5
    opaque = _draws(scene, _Unit(*POINT, STONE_WALL))
    faded, fx, fy = _one(_draws(scene, _Unit(*POINT, STONE_WALL, status=REMOVE), wall_alpha=wall_alpha))
    translucent, tx, ty = _one(_draws(scene, _Unit(*POINT, STONE_WALL), wall_alpha=wall_alpha))
    assert (fx, fy) == (tx, ty)
    assert faded.rgba.shape == translucent.rgba.shape
    pieces = [unit_sprites.SpritePiece(draw=d, dx=px - fx, dy=py - fy) for d, px, py in opaque]
    flat, _ox, _oy = unit_sprites._assembled_with_origin(pieces)
    expected = np.rint(flat[..., 3] * (wall_alpha * unit_state.GHOST_ALPHA)).astype(np.uint8)
    assert np.array_equal(faded.rgba[..., 3], expected)
    assert int(faded.rgba[..., 3].max()) < int(translucent.rgba[..., 3].max())


def test_an_editor_marker_is_not_faded(scene):
    alive = _draws(scene, _Unit(*POINT, EMPTY_TC_ANNEX))
    ghost = _draws(scene, _Unit(*POINT, EMPTY_TC_ANNEX, status=FOUNDATION))
    assert alive, "the marker drew nothing"
    assert len(ghost) == len(alive)
    for (a, ax, ay), (g, gx, gy) in zip(alive, ghost, strict=True):
        assert (ax, ay, a.hotspot_x, a.hotspot_y) == (gx, gy, g.hotspot_x, g.hotspot_y)
        assert np.array_equal(a.rgba, g.rgba)


def test_a_ghosted_host_draws_no_garrison_flag(scene):
    scenario, proj, elevations = scene
    host = _Unit(*POINT, WATCH_TOWER, reference_id=500)
    occupant = _Unit(*POINT, VILLAGER, reference_id=600, garrisoned_in_id=500)
    _place(scenario, 1, host, occupant)
    flagged = render_sprites.unit_sprite_draws_at(scenario, proj, elevations, None, 1, host)
    bare = unit_sprites.sprite_pieces_for(WATCH_TOWER, 0.0, scenario.team_indices[1], proj.half_w)
    assert len(flagged) > len(bare), "the live host drew no flag, so the check below is vacuous"
    host.status = REMOVE
    scenario.unit_gen += 1
    assert len(render_sprites.unit_sprite_draws_at(scenario, proj, elevations, None, 1, host)) == 1


def test_a_ghosted_sloped_unit_fades_too(scene):
    scenario, proj, elevations = scene
    corner_rise = iso_geometry.corner_rise_px(elevations, proj)
    fold = iso_geometry.tile_folds(elevations)
    unit = _Unit(*POINT, OAK)
    _place(scenario, 0, unit)
    alive = _one(render_sprites.unit_sprite_draws_at(scenario, proj, elevations, corner_rise, 0, unit, fold=fold))[0]
    unit.status = REMOVE
    ghost = _one(render_sprites.unit_sprite_draws_at(scenario, proj, elevations, corner_rise, 0, unit, fold=fold))[0]
    assert np.array_equal(ghost.rgba[..., 3], np.rint(alive.rgba[..., 3] * unit_state.GHOST_ALPHA).astype(np.uint8))


@pytest.mark.parametrize("const", [OAK, VILLAGER])
def test_a_ghosted_flat_icon_fades_and_the_cached_icon_does_not(scene, const):
    scenario, _proj, _elevations = scene
    tile_px = 64
    bounds = (40, 41, 40, 41)
    alive_unit, ghost_unit = _Unit(*POINT, const), _Unit(*POINT, const, status=REMOVE)
    alive = render_flat._flat_unit_icon(alive_unit, 1, 0, {}, scenario.team_indices[1], bounds, tile_px)
    ghost = render_flat._flat_unit_icon(ghost_unit, 1, 0, {}, scenario.team_indices[1], bounds, tile_px)
    assert alive is not None and ghost is not None
    assert ghost is not alive
    assert np.array_equal(ghost.rgba[..., 3], np.rint(alive.rgba[..., 3] * unit_state.GHOST_ALPHA).astype(np.uint8))
    again = render_flat._flat_unit_icon(alive_unit, 1, 0, {}, scenario.team_indices[1], bounds, tile_px)
    assert np.array_equal(again.rgba, alive.rgba), "the fade leaked into the shared icon cache"


def test_a_flat_editor_marker_icon_is_not_faded(scene):
    bounds = (40, 41, 40, 41)
    alive = render_flat._flat_unit_icon(_Unit(*POINT, EMPTY_TC_ANNEX), 1, 0, {}, 1, bounds, 64)
    ghost = render_flat._flat_unit_icon(_Unit(*POINT, EMPTY_TC_ANNEX, status=FOUNDATION), 1, 0, {}, 1, bounds, 64)
    assert alive is not None and ghost is not None
    assert int(alive.rgba[..., 3].max()) > 0, "the marker drew nothing"
    assert np.array_equal(ghost.rgba, alive.rgba)


@pytest.mark.gui
@pytest.mark.parametrize("const", [OAK, WATCH_TOWER])
def test_a_state_edit_in_the_inspector_repaints_the_sprite_as_a_rebuild_would(scene, const):
    """The inspector's Unit State pick splices the unit (its existing
    UnitSplice), and the spliced pixels must equal a wholesale rebuild's."""
    import conftest

    if not conftest.PYQT5_AVAILABLE:
        pytest.skip("PyQt5 not importable")
    from descape import unit_fields, unit_pick

    window = conftest.stepped_window(BLANK_TEMPLATE_PATH)
    try:
        window.show_sprites_action.setChecked(True)
        assert window._cache.sprites_enabled
        window.mode_combo.setCurrentText("Units")
        model = window._ensure_unit_edits()
        with window._unit_edit(model, "Place", [1]):
            unit = model.add(1, const, *POINT)
        entry = next(e for e in window.map_view._unit_index.entries if e.unit is unit)
        xs, ys = zip(*window.map_view._unit_polygons_for(entry)[0], strict=True)
        x, y = int(sum(xs) / len(xs)), int(sum(ys) / len(ys))
        box = (x - 160, y - 260, x + 160, y + 80)
        before = window._cache.render_rect(*box).copy()

        window._selection = [unit_pick.unit_key(1, unit)]
        window._refresh_selection_view()
        window._on_unit_field_changed(unit_fields.FIELDS_BY_ID["status"], REMOVE)
        spliced = window._cache.render_rect(*box).copy()
        assert not np.array_equal(spliced, before), "the state edit repainted nothing"
        window._render_current(reset_view=False)
        assert np.array_equal(window._cache.render_rect(*box), spliced)
    finally:
        conftest.close_window(window)
