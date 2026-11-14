"""Renders a loaded scenario's Map + Units to an RGB numpy array.

Pure function, no Qt dependency -- backs both the headless PNG export and the
PyQt5 viewer's canvas.

This module keeps only the whole-image entry points, render_scenario() and
save_png(). The rest of the renderer is one module per area:

- render_paint.py: tile-pixel tiers, top and tile blocks, blend keys, Flat
  patch rects, clipped paints and ink planes, grid stamps, ASSEMBLY_SLICE.
- render_units.py: unit spans, anchors, tile bounds and occupied tiles,
  stored_rotation, the per-generation memos, wall overrides, unit colour and
  marks, player_art_for.
- render_sprites.py: SpriteLayer, draped and farm tiles,
  sprite_draws_by_anchor*, sprite resolution, SpriteMemo, sprite bbox merges,
  unit_sprite_draws_at.
- render_bbox.py: building bboxes, dirty screen bboxes, BystanderGrid, rect
  candidates, prepare_rect_native.
- render_iso.py: Stepped: skirt, seam and contact shade, render_terrain_iso*,
  elevations_and_proj, iso candidates, composite_rect_iso, refresh_region_iso.
- render_sloped.py: Sloped: slope shade, tile quads, pick ids, sloped marks
  and candidates, render_terrain_sloped*, composite_rect_sloped,
  composite_ids_rect_sloped.
- render_flat.py: Flat: render_tile, render_terrain, refresh_tiles, Flat unit
  rows and icons, composite_rect_flat, overlay_units, refresh_units_over.

Rules of the split (tests/test_render_split.py checks the first):

- A render module imports only the render modules below it: render_paint
  imports none; render_units imports paint; render_sprites and render_bbox
  import paint and units; render_iso and render_sloped import those four;
  render_flat imports paint, units and sprites. No render module imports
  this one.
- Tests and tools set a render module's global only through
  testkit.render_family, which sets it in every module that binds it.
"""

from __future__ import annotations

import numpy as np

from descape import (
    terrain_style,
)
from descape.grid_overlay import DEFAULT_GRID, GridBake
from descape.render_flat import overlay_units, render_terrain
from descape.render_iso import render_terrain_iso
from descape.render_sloped import render_terrain_sloped
from descape.scenario_io import LoadedScenario


def render_scenario(
    scenario: LoadedScenario,
    with_units: bool = True,
    isometric: bool = False,
    style: str | None = None,
    grid: GridBake = DEFAULT_GRID,
    blend: bool = True,
) -> np.ndarray:
    """isometric=True renders Stepped mode (render_terrain_iso) instead of
    Flat (render_terrain). with_units applies in every mode now (Phase 5 for
    Stepped, Phase 6 for Sloped): each draws units at their own tile's
    elevation, interleaved with terrain in depth order via that mode's own
    with_units parameter -- see _paint_tile_and_units_iso()'s/
    _paint_tile_and_units_sloped()'s docstrings for why that has to be
    interleaved rather than a separate overlay pass, the way Flat mode's
    overlay_units() draws over an already-finished terrain image.

    style, if given ("flat"/"stepped"/"sloped"), selects the render and
    overrides isometric -- Phase 6 (Sloped)'s own entry point, added
    without changing isometric's existing meaning or any existing call
    site: dump_scenario.py's --iso flag and save_png()'s own isometric
    passthrough both keep working exactly as before when style is left at
    its None default. isometric stays a valid way to select Stepped
    (style="stepped" is equivalent, not required). blend=False keeps the
    pre-blending render in every style (blend_effective())."""
    if style is not None:
        if style not in terrain_style.TERRAIN_STYLES:
            expected = ", ".join(repr(s) for s in terrain_style.TERRAIN_STYLES)
            raise ValueError(f"style must be one of {expected}, got {style!r}")
        if style == "sloped":
            return render_terrain_sloped(scenario, with_units=with_units, grid=grid, blend=blend)
        isometric = style == "stepped"
    if isometric:
        return render_terrain_iso(scenario, with_units=with_units, grid=grid, blend=blend)
    img = render_terrain(scenario, grid=grid, blend=blend)
    if with_units:
        img = overlay_units(img, scenario)
    return img


def save_png(
    scenario: LoadedScenario,
    out_path: str,
    with_units: bool = True,
    scale: int = 1,
    isometric: bool = False,
    style: str | None = None,
) -> None:
    from PIL import Image

    img = render_scenario(scenario, with_units=with_units, isometric=isometric, style=style)
    pil_img = Image.fromarray(img, mode="RGB")
    if scale != 1:
        pil_img = pil_img.resize(
            (pil_img.width * scale, pil_img.height * scale), Image.NEAREST
        )
    pil_img.save(out_path)
