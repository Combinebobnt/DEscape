"""Bench-only: a unit edit's repaint bbox sized from the edited units' real
sprite and mark extents rather than render._sprite_reach_px(), computed from
outside the caches so tools/bench_stroke_end.py and tools/bench_unit_edit.py
can price it before any descape/ change (the 2026-09-26 sprite-bbox plan's
Step 0). Step 1 moves this logic onto the caches themselves.

All bboxes are REFERENCE canvas pixels unless a name says level. The caller
reads pre_extent() before cache.invalidate_units(changed) and post_extent()
after it, the order the plan's Design section fixes."""

from __future__ import annotations

from descape import render
from descape.render_cache import _const_splice_eligible

NONE, WALL, STALE, VISIBLE = "none", "wall", "stale", "visible"


def union(*bboxes):
    """Union of the non-None bboxes, or None."""
    real = [b for b in bboxes if b is not None]
    if not real:
        return None
    return (min(b[0] for b in real), min(b[1] for b in real), max(b[2] for b in real), max(b[3] for b in real))


def clamp(bbox, dims):
    if bbox is None:
        return None
    w, h = dims
    return (max(0, bbox[0]), max(0, bbox[1]), min(w, bbox[2]), min(h, bbox[3]))


def contains(outer, inner) -> bool:
    if inner is None:
        return True
    if outer is None:
        return False
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def resident_levels(cache) -> list[int]:
    return sorted({key[0] for key in cache._cache})


def level_to_reference(cache, mip: int, bbox):
    """The inverse of _bbox_to_level(): floor low, ceil high by t_ref/t_lvl, so
    converting the result back to the level is a superset of `bbox`."""
    t_ref, t_lvl = cache.mip_tile_px(0), cache.mip_tile_px(mip)
    if t_ref == t_lvl:
        return bbox
    x0, y0, x1, y1 = bbox
    ref = ((x0 * t_ref) // t_lvl, (y0 * t_ref) // t_lvl, -((-x1 * t_ref) // t_lvl), -((-y1 * t_ref) // t_lvl))
    assert contains(cache._bbox_to_level(mip, ref), bbox), (mip, bbox, ref)
    return ref


def stale_levels(cache) -> list[int]:
    """Resident levels whose layer is older than the cache's sources (Stepped)."""
    return [mip for mip in resident_levels(cache) if not cache.is_level_resident(mip)]


def fallback_reason(cache, changed, visible_mip: int | None, relax_stale: bool = False) -> str:
    """Which of the plan's reach-padded fallbacks this edit takes, read at pre
    time (before invalidate_units): a wall/connector const, a resident Stepped
    level whose layer is stale, or a visible mip holding no resident chunks.

    relax_stale (bench-only variant): a stale level that isn't the visible one
    no longer forces the fallback. The caller then sizes the tight bbox over
    the current levels only and evicts the stale levels' chunks in today's
    bbox (evict_levels), so they recomposite fresh if the user zooms back."""
    if any(not _const_splice_eligible(s.unit) for s in changed):
        return WALL
    resident = resident_levels(cache)
    stale = stale_levels(cache)
    if stale and (not relax_stale or visible_mip in stale):
        return STALE
    if visible_mip is not None and visible_mip not in resident:
        return VISIBLE
    return NONE


def evict_levels(cache, bbox, mips) -> None:
    """invalidate_region(bbox), restricted to the given levels."""
    for mip in mips:
        lx0, ly0, lx1, ly1 = cache._bbox_to_level(mip, bbox)
        if lx1 <= lx0 or ly1 <= ly0:
            continue
        cx0, cy0, cx1, cy1 = cache.chunk_index_range(mip, lx0, ly0, lx1, ly1)
        for key in [k for k in cache._cache if k[0] == mip and cx0 <= k[1] <= cx1 and cy0 <= k[2] <= cy1]:
            cache._cache_bytes -= cache._cache[key].nbytes
            del cache._cache[key]


def _level_state(cache, mip: int):
    """(proj, sprites, building_bboxes, corner_rise, extra_top_px) for one level."""
    if cache.style == "sloped":
        return cache.proj, cache.sprites, cache.building_bboxes, cache.corner_rise, cache._headroom
    lvl = cache._levels[mip]
    return lvl.proj, lvl.sprites, lvl.building_bboxes, None, 0


def pre_extent(cache, changed, levels):
    """What the edited units currently paint, from each level's own layer: its
    sprite bboxes on every old footprint tile, plus its building bbox at the
    old own tile (the mark, which merge_sprite_bboxes already widened)."""
    out = None
    for mip in levels:
        _proj, sprites, building_bboxes, _rise, _top = _level_state(cache, mip)
        level_box = None
        for s in changed:
            if sprites is not None:
                level_box = union(level_box, *(sprites.bboxes.get(k) for k in s.old_tiles))
            if building_bboxes is not None and s.old_own_tile is not None:
                level_box = union(level_box, building_bboxes.get(s.old_own_tile))
        if level_box is not None:
            out = union(out, level_to_reference(cache, mip, level_box))
    return out


def post_extent(cache, changed, levels):
    """What the edited units paint after the edit, resolved per level with the
    same arguments render_cache._reanchor_units() passes, so it holds whether
    or not the level's layer has been rebuilt yet."""
    scenario = cache.scenario
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height
    layers = cache.layers
    out = None
    for mip in levels:
        proj, _sprites, _bb, corner_rise, extra_top = _level_state(cache, mip)
        level_box = None
        for s in changed:
            if s.new_own_tile is None:
                continue
            if cache.sprites_enabled:
                contribution = render._resolve_unit_sprite(
                    scenario, proj, cache.elevations, cache.unit_filter, corner_rise, {}, layers.farm_overlay,
                    s.player_id, s.index, s.unit, layers.tree_scale, layers.hero_glow,
                )
                if contribution is not None:
                    level_box = union(level_box, *contribution.bboxes.values())
            if cache.unit_filter.matches(s.player_id, s.unit):
                level_box = union(
                    level_box, render._building_bbox_for(s.unit, w, h, proj, cache.elevations, extra_top)
                )
        if level_box is not None:
            out = union(out, level_to_reference(cache, mip, level_box))
    return out


def dirty_bbox(cache, changed, with_sprites: bool, flatten: bool = False):
    """ViewerWindow._unit_edit_bbox()'s call, with with_sprites chosen by the
    caller: True is today's reach-padded bbox, False the tile-only term."""
    scenario = cache.scenario
    old_tiles = {t for s in changed for t in s.old_tiles}
    touched = old_tiles | {t for s in changed for t in s.new_tiles}
    width = scenario.map_manager.map_width
    dirty = [y * width + x for x, y in touched]
    if cache.style == "stepped":
        return render.dirty_screen_bbox_iso(
            scenario, dirty, cache.elevations, cache._levels[0].proj, with_units=True,
            with_sprites=with_sprites, elevation_changed=set(), flatten_elevations=flatten,
            extra_anchor_tiles=old_tiles,
        )
    return render.dirty_screen_bbox_sloped(
        scenario, dirty, cache.elevations, cache.proj, with_units=True,
        with_sprites=with_sprites, elevation_changed=set(), extra_anchor_tiles=old_tiles,
    )


def tight_bbox(cache, changed, pre, post, flatten: bool = False):
    """The plan's bbox: tile term unioned with the pre and post extents,
    clamped to the reference canvas the way today's bbox is."""
    tiles = dirty_bbox(cache, changed, with_sprites=False, flatten=flatten)
    return clamp(union(tiles, pre, post), cache.canvas_dims(0))


def chunks_touched(cache, bbox) -> dict[int, int]:
    """Resident chunks per level invalidate_region(bbox) would evict."""
    out: dict[int, int] = {}
    if bbox is None:
        return out
    for mip in resident_levels(cache):
        lx0, ly0, lx1, ly1 = cache._bbox_to_level(mip, bbox)
        if lx1 <= lx0 or ly1 <= ly0:
            continue
        cx0, cy0, cx1, cy1 = cache.chunk_index_range(mip, lx0, ly0, lx1, ly1)
        out[mip] = sum(
            1 for m, cx, cy in cache._cache if m == mip and cx0 <= cx <= cx1 and cy0 <= cy <= cy1
        )
    return out


def patch_area(cache, bbox) -> int:
    """cache.patch_area(), bypassing any instance-level bench wrapper."""
    return 0 if bbox is None else type(cache).patch_area(cache, bbox)


def describe(cache, bbox) -> str:
    if bbox is None:
        return "-"
    chunks = ",".join(f"{mip}:{n}" for mip, n in chunks_touched(cache, bbox).items())
    return f"{bbox[2] - bbox[0]}x{bbox[3] - bbox[1]} {patch_area(cache, bbox) / 1000:.0f}kpx ch[{chunks}]"
