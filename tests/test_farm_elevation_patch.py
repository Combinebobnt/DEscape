"""A farm reshapes when the ground under it is elevated.

Raising any tile of a draped farm's footprint, its own tile or another one,
must repaint the farm on the new ground in Stepped and Sloped alike. The
edit runs through the calls a viewer Elevate stroke makes: set_tiles_elevation,
the style's dirty bbox, then cache.patch(). Nothing is invalidated wholesale
afterwards, so the dirty bbox has to cover the reshape as well. The oracle is
a fresh full render with sprites on, which is what drapes the farm.

Pixels only: which units the patch re-anchors differs by style on purpose
(Stepped reads a unit's own tile, Sloped its 3x3), so no cache internals are
compared here.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_elevation_unit_splice import BUMP_TILE, _make_cache, _oracle
from test_sloped_sprites import FARM_CONST, _flat_scenario_with_unit, sprite_install  # noqa: F401  # pytest fixture

from descape import iso_geometry as ig
from descape.elevation_tools import set_tiles_elevation
from descape.render import dirty_screen_bbox_iso, dirty_screen_bbox_sloped

BASE_ELEVATION = 2
FARM_XY = (50.5, 50.5)  # a 3x3 farm on tiles 49..51, own tile (50, 50)
OWN_TILE = (50, 50)
OTHER_TILE = (51, 51)  # a footprint corner: outside Stepped's radius 0 around the own tile


def _farm_scenario():
    scenario = _flat_scenario_with_unit(BASE_ELEVATION, *FARM_XY, unit_const=FARM_CONST)
    # A far bump already at the raised height keeps Sloped's headroom and
    # Stepped's cached elevation range unchanged, so the patch path runs.
    scenario.map_manager.get_tile(*BUMP_TILE).elevation = BASE_ELEVATION + 1
    return scenario


def _elevate(style: str, cache, scenario, tile) -> None:
    """The Elevate tool's +1 on one tile, as ViewerWindow._apply_dirty_render
    patches it: the bbox call mutates cache.elevations and fills the set."""
    mm = scenario.map_manager
    before = [t.elevation for t in mm.terrain]
    set_tiles_elevation(mm, [(*tile, mm.get_tile(*tile).elevation + 1)])
    dirty = [i for i, t in enumerate(mm.terrain) if t.elevation != before[i]]
    changed: set = set()
    bbox_fn = dirty_screen_bbox_iso if style == "stepped" else dirty_screen_bbox_sloped
    bbox = bbox_fn(
        scenario, dirty, cache.elevations, cache.proj, with_units=True,
        with_sprites=cache.sprites_enabled, elevation_changed=changed,
    )
    assert bbox is not None and tile in changed, "the edit never reached the patch path"
    cache.patch(bbox, elevation_changed=changed)


def _canvas(cache) -> np.ndarray:
    return cache.render_rect(0, 0, *cache.canvas_dims(0), mip=0)


def _tile_rect(cache, tile) -> tuple[slice, slice]:
    """The edited tile's diamond box at its new height: a farm tile."""
    proj = cache.proj
    x0, y0 = ig.tile_screen_origin(*tile, BASE_ELEVATION + 1, proj)
    return slice(y0, y0 + 2 * proj.half_h), slice(x0, x0 + 2 * proj.half_w)


@pytest.mark.parametrize("tile", [OWN_TILE, OTHER_TILE], ids=["own_tile", "other_footprint_tile"])
@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_an_elevate_under_a_farm_patches_to_a_fresh_render(style, tile, sprite_install):  # noqa: F811
    scenario = _farm_scenario()
    cache = _make_cache(style, scenario, sprites=True)
    before = _canvas(cache)
    fresh_before = _oracle(style, scenario, sprites=True)
    h, w = before.shape[:2]
    assert np.array_equal(before, fresh_before[:h, :w])

    _elevate(style, cache, scenario, tile)

    after = _canvas(cache)
    fresh_after = _oracle(style, scenario, sprites=True)
    rows, cols = _tile_rect(cache, tile)
    assert not np.array_equal(fresh_after[rows, cols], fresh_before[rows, cols]), (
        "the fresh render shows no change on the raised farm tile, so this proves nothing"
    )
    undraped = _oracle(style, scenario, sprites=False)
    assert not np.array_equal(fresh_after[rows, cols], undraped[rows, cols]), (
        "the raised tile shows no farm drape, so this only checks bare terrain"
    )
    assert not np.array_equal(after[rows, cols], before[rows, cols]), "the farm did not reshape"
    assert np.array_equal(after, fresh_after[:h, :w])
