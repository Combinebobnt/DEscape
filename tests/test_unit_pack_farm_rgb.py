"""UnitPack.refresh() clears a tile's farm colour along with its farm mask.

The kernel reads farm_rgb only under farm_mask, so a stale colour draws no
pixel; it does make a refreshed pack differ from a fresh one byte for byte,
which is the oracle the splice corpus tests (`_pack_state`) compare.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_farm_terrain import FAKE_FARM_CONST, fake_farm  # noqa: F401 -- fake_farm is a fixture
from test_native_composite import (  # noqa: F401 -- native_kernel, synthetic_textures are fixtures
    FARM_X,
    FARM_Y,
    _edit_cache,
    _pack_of,
    _sloped_scenario,
    native_kernel,
    synthetic_textures,
)
from test_sprite_edit_bbox import sprite_install  # noqa: F401 -- a fixture

from descape import composite_backend, render
from descape.render_cache import UnitSplice


@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_a_moved_farm_leaves_no_colour_behind_its_cleared_mask(
    native_kernel, sprite_install, fake_farm, synthetic_textures, style,  # noqa: F811
) -> None:
    scenario = _sloped_scenario()
    mm = scenario.map_manager
    farm = scenario.unit_manager.units[2][-1]
    assert farm.unit_const == FAKE_FARM_CONST
    everything = np.arange(mm.map_width * mm.map_height)
    with composite_backend.use_backend("native"):
        cache = _edit_cache(style, scenario, sprites=True)
        cache.render_rect(0, 0, *cache.canvas_dims(0), mip=0)
        pack = _pack_of(cache, style)
        pack.ensure(everything)
        was_farm = pack.farm_mask.copy() != 0
        assert was_farm.any(), "no farm tile derived -- vacuous"

        index = scenario.unit_manager.units[2].index(farm)
        old_own = (int(farm.x), int(farm.y))
        old_tiles = tuple(render.unit_occupied_tiles(farm, mm.map_width, mm.map_height))
        farm.x, farm.y = FARM_X + 4.5, FARM_Y - 3.5
        scenario.unit_gen += 1
        splice = UnitSplice(
            2, index, farm, old_own, (int(farm.x), int(farm.y)), old_tiles,
            tuple(render.unit_occupied_tiles(farm, mm.map_width, mm.map_height)),
        )
        assert cache.can_splice([splice])
        cache.invalidate_units([splice])
        assert _pack_of(cache, style) is pack, "the move rebuilt the pack instead of refreshing it -- vacuous"
        pack.ensure(everything)

        left = was_farm & (pack.farm_mask == 0)
        assert left.any(), "the move left no farm tile behind -- vacuous"
        assert not pack.farm_rgb.reshape(-1, 3)[left].any(), "a cleared farm tile kept its colour"
        fresh_cache = _edit_cache(style, scenario, sprites=True)
        fresh_cache.render_rect(0, 0, *fresh_cache.canvas_dims(0), mip=0)
        fresh = _pack_of(fresh_cache, style)
        fresh.ensure(everything)
        assert np.array_equal(pack.farm_mask, fresh.farm_mask)
        assert np.array_equal(pack.farm_rgb, fresh.farm_rgb)
