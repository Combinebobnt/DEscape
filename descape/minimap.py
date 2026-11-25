"""The preview minimap's image (GH #54): one pixel per tile, terrain colour
with each unit's mark painted over its footprint.

Qt-free like clipboard_history.thumbnail_rgb(): an (h, w, 3) uint8 array,
pixel (y, x) being tile (x, y), which minimap_panel projects into the main
view's orientation at paint time.

It never decodes a texture or waits on a prefetch. A terrain takes its
texture's average colour only once that texture is decoded
(asset_source.terrain_average_color_if_loaded()), and its palette colour
until then, so the viewer rebuilds while textures load and the two converge.
color_for_terrain_id() would decode on the GUI thread instead.

Units go GAIA first, then P1 to P8, each player's single-tile units before
its larger footprints, so a building covers what stands on it. A Python loop
costs about 2.3 us per unit, so single-tile units are written with one
fancy-indexed assignment per player, and the filter and colour rules run once
per (player, const) rather than once per unit.
"""

from __future__ import annotations

from dataclasses import replace
from operator import attrgetter

import numpy as np

from descape import asset_source, unit_state
from descape.render_paint import tile_field_mirror
from descape.render_units import NON_BUILDING_SPAN, kind_mark_color, unit_mark_color, unit_tile_bounds
from descape.terrain_palette import palette_color_for_terrain_id, tile_span
from descape.unit_filter import UnitFilter

_get_x = attrgetter("x")
_get_y = attrgetter("y")
_get_const = attrgetter("unit_const")


def minimap_rgb(scenario, unit_filter: UnitFilter) -> tuple[np.ndarray, bool]:
    """(rgb, used_fallback): the (h, w, 3) image, and whether any terrain drew
    its palette colour while its texture was not decoded yet."""
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height
    rgb, used_fallback = _terrain_rgb(scenario, w, h)
    for player_id, units in enumerate(scenario.unit_manager.units):
        if units:
            _paint_player(rgb, units, player_id, scenario.player_colors[player_id], unit_filter, w, h)
    return rgb, used_fallback


def _terrain_rgb(scenario, w: int, h: int) -> tuple[np.ndarray, bool]:
    ids = tile_field_mirror(scenario, "terrain_id")
    if ids.size == 0:
        return np.zeros((h, w, 3), dtype=np.uint8), False
    # Ids are a file's u8, so a bincount finds the ones present without np.unique's sort.
    lut = np.zeros((int(ids.max()) + 1, 3), dtype=np.uint8)
    used_fallback = False
    for terrain_id in np.flatnonzero(np.bincount(ids)).tolist():
        color = asset_source.terrain_average_color_if_loaded(terrain_id)
        if color is None:
            color = palette_color_for_terrain_id(terrain_id)
            used_fallback = used_fallback or asset_source.get_terrain_texture_path(terrain_id) is not None
        lut[terrain_id] = color
    return lut[ids].reshape(h, w, 3), used_fallback


def _paint_player(rgb: np.ndarray, units, player_id: int, player_color, unit_filter: UnitFilter, w: int, h: int) -> None:
    # A plain list: UuidList's pure-Python __getitem__ costs about 1 us per index.
    units = list(units)
    n = len(units)
    consts = np.fromiter(map(_get_const, units), dtype=np.int64, count=n)
    unique, first, inverse = np.unique(consts, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    samples = [units[i] for i in first.tolist()]
    # Every gate but the garrison and Unit State ones is per (player, const): one verdict per const.
    per_const = replace(unit_filter, show_garrisoned=True, show_inactive=True)
    keep = np.array([per_const.matches(player_id, u) for u in samples], dtype=bool)[inverse]
    colors = np.array([kind_mark_color(c, player_color) for c in unique.tolist()], dtype=np.uint8)[inverse]
    ghosted = np.fromiter(map(unit_state.is_ghosted, units), dtype=bool, count=n)
    for i in np.flatnonzero(ghosted).tolist():
        colors[i] = unit_mark_color(units[i], player_color)
    if not unit_filter.show_inactive:
        keep &= ~ghosted
    single = np.array([tile_span(c, NON_BUILDING_SPAN) == (1, 1) for c in unique.tolist()], dtype=bool)[inverse]
    if not unit_filter.show_garrisoned:
        hosts = np.fromiter((getattr(u, "garrisoned_in_id", -1) for u in units), dtype=np.int64, count=n)
        for i in np.flatnonzero(keep & (hosts != -1)).tolist():
            keep[i] = unit_filter.matches(player_id, units[i])

    one = np.flatnonzero(keep & single)
    if one.size:
        picked = [units[i] for i in one.tolist()]
        xs = np.trunc(np.fromiter(map(_get_x, picked), dtype=np.float64, count=one.size)).astype(np.int64)
        ys = np.trunc(np.fromiter(map(_get_y, picked), dtype=np.float64, count=one.size)).astype(np.int64)
        on_map = np.flatnonzero((xs >= 0) & (xs < w) & (ys >= 0) & (ys < h))
        # The last unit on each tile only: numpy leaves a repeated index's winner unspecified.
        tiles = (ys * w + xs)[on_map][::-1]
        _, last = np.unique(tiles, return_index=True)
        winners = on_map[::-1][last]
        rgb[ys[winners], xs[winners]] = colors[one[winners]]
    for i in np.flatnonzero(keep & ~single).tolist():
        bounds = unit_tile_bounds(units[i], w, h)
        if bounds is not None:
            x0, x1, y0, y1 = bounds
            rgb[y0:y1, x0:x1] = colors[i]
