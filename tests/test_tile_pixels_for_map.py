"""Pins descape.render.tile_pixels_for_map()'s LARGE_MAP_TILE_THRESHOLD=240
branch against every size descape.scenario_new.STANDARD_MAP_SIZES makes
reachable via File > New Map, so a future threshold edit has to be a
deliberate, evidenced change rather than an accidental one.

No render.py change was needed to support the four newly-reachable standard
sizes (144/168/200/220): all four are <= 240, so they take the same
SMALL_MAP_TILE_PIXELS branch the existing 120/240 presets already did, and
custom sizes in 241..480 take the LARGE_MAP_TILE_PIXELS branch bounded by the
already-shipping, already-benchmarked 480 case (see the module comment above
tile_pixels_for_map() for the memory measurement that branch is sized
against)."""

from __future__ import annotations

import pytest

from descape.render import LARGE_MAP_TILE_PIXELS, SMALL_MAP_TILE_PIXELS, tile_pixels_for_map
from descape.scenario_new import STANDARD_MAP_SIZES


@pytest.mark.parametrize("tiles", STANDARD_MAP_SIZES)
def test_tile_pixels_for_standard_sizes(tiles: int) -> None:
    expected = SMALL_MAP_TILE_PIXELS if tiles <= 240 else LARGE_MAP_TILE_PIXELS
    assert tile_pixels_for_map(tiles, tiles) == expected
