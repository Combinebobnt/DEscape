"""The per-tile early reject in render._render_tile_iso, _render_tile_sloped
and _render_tile_sloped_ids (Batch C).

The Stepped reject skips a candidate tile when iso_geometry.iso_tile_extent's
union box is wholly outside img; Sloped's single paint extent is already the
tile's whole reach. Its correctness burden is "the union is
complete", not "a reach bound is right": if the union misses one sub-paint,
a tile that paints real pixels gets skipped and the pixels silently vanish.

Three bars here. The union contains every contributor's own index_extent
box, including the two seam producers the helper deliberately leaves out
(asserted, not assumed: Batch A shipped three "obviously a subset" claims
that were wrong). Rendering each candidate tile with the reject on and
forced off gives identical bytes over many random rects. And the reject
provably fires, so the equality cannot pass vacuously.

MUTATION PROBE (Stepped), run 2026-09-12 and not left in: shrink
iso_tile_extent's returned box by k px on every side, so the reject fires
on tiles that do paint. This file fails at k=1 (both the containment and
the agreement tests). The existing oracles are much weaker here, which is
why this file exists rather than leaning on them:
tests/test_sprite_chunks.py::test_stitched_chunks_match_the_full_sprite_render
still passes at k=1..7 and first fails at k=8 (a flat 12x12 map whose
128px chunks align exactly with tile bboxes, so no partial overlap reaches
a diamond tip), and tools/verify_iso_chunks.py passed all 7 checks at k=1,
600 random rects included: a 1px band of a diamond's bbox holds only its
tip pixels, which a random rect rarely lands on.
"""

import numpy as np
import pytest

from descape import asset_source, iso_geometry, render

TILE_PX = (8, 16, 32, 64, 128)
ELEV_STEP_PCT = (25, 50, 100, 200)
LEVELS = range(4)


def _elev_steps(tile_px: int) -> list[int]:
    """The real elev_step values canvas_size_and_origin derives for this tile_px."""
    return sorted({
        iso_geometry.canvas_size_and_origin(2, 2, tile_px, 0, 3, elev_step_pct=pct).elev_step
        for pct in ELEV_STEP_PCT
    })


def _contains(outer, inner) -> bool:
    return outer[0] <= inner[0] and inner[1] <= outer[1] and outer[2] <= inner[2] and inner[3] <= outer[3]


@pytest.mark.parametrize("tile_px", TILE_PX)
def test_seam_boxes_are_inside_the_diamond_box(tile_px):
    """Why iso_tile_extent may omit both seam producers."""
    diamond = iso_geometry.index_extent(iso_geometry.diamond_indices, tile_px)
    for side in ("up_left", "up_right"):
        seam = iso_geometry.index_extent(iso_geometry.seam_edge_indices, tile_px, side)
        assert seam is not None and _contains(diamond, seam), (side, seam, diamond)
    apex = iso_geometry.index_extent(iso_geometry.seam_apex_indices, tile_px)
    assert apex is not None and _contains(diamond, apex), (apex, diamond)


@pytest.mark.parametrize("tile_px", TILE_PX)
def test_union_contains_every_contributor(tile_px):
    ext = iso_geometry.index_extent
    empty_seen = False
    for step in _elev_steps(tile_px):
        px = [lvl * step for lvl in LEVELS]
        for dl in px:
            for dr in px:
                for ul in px:
                    for ur in px:
                        for diag in px:
                            union = iso_geometry.iso_tile_extent(tile_px, dl, dr, ul, ur, diag)
                            boxes = [ext(iso_geometry.diamond_indices, tile_px)]
                            boxes.append(ext(iso_geometry.seam_edge_indices, tile_px, "up_left"))
                            boxes.append(ext(iso_geometry.seam_edge_indices, tile_px, "up_right"))
                            boxes.append(ext(iso_geometry.seam_apex_indices, tile_px))
                            if dl:
                                boxes.append(ext(iso_geometry.skirt_quad_indices, tile_px, dl, "left"))
                            if dr:
                                boxes.append(ext(iso_geometry.skirt_quad_indices, tile_px, dr, "right"))
                            if ul:
                                boxes.append(ext(iso_geometry.shadow_quad_indices, tile_px, ul, "up_left"))
                            if ur:
                                boxes.append(ext(iso_geometry.shadow_quad_indices, tile_px, ur, "up_right"))
                            if diag:
                                boxes.append(ext(iso_geometry.shadow_apex_indices, tile_px, diag))
                            if ul:
                                boxes.append(ext(iso_geometry.shadow_tip_indices, tile_px, ul, "up_left"))
                            if ur:
                                boxes.append(ext(iso_geometry.shadow_tip_indices, tile_px, ur, "up_right"))
                            for box in boxes:
                                if box is None:
                                    empty_seen = True
                                    continue
                                assert _contains(union, box), (tile_px, dl, dr, ul, ur, diag, box, union)
    if tile_px <= 16:
        assert empty_seen, "no empty shadow band in the grid, so the None branch went untested"


class _Tile:
    __slots__ = ("terrain_id", "x", "y")

    def __init__(self, x, y, terrain_id):
        self.x, self.y, self.terrain_id = x, y, terrain_id


MAP_W, MAP_H = 9, 8


def _synthetic_map(seed: int):
    rng = np.random.default_rng(seed)
    elevations = rng.integers(0, 4, size=(MAP_H, MAP_W)).astype(np.int64)
    elevations[:3, :3] = 2  # a flat patch, so the no-delta branch is covered too
    tiles = {(x, y): _Tile(x, y, int(rng.integers(0, 40))) for y in range(MAP_H) for x in range(MAP_W)}
    return elevations, tiles


def _random_rects(rng, proj, n):
    canvas_h = proj.canvas_h + (iso_geometry.MAX_ELEVATION - iso_geometry.MIN_ELEVATION) * proj.elev_step
    for _ in range(n):
        rw, rh = int(rng.integers(1, 160)), int(rng.integers(1, 160))
        x0 = int(rng.integers(0, max(1, proj.canvas_w - 1)))
        y0 = int(rng.integers(0, max(1, canvas_h - 1)))
        yield x0, y0, x0 + rw, y0 + rh


@pytest.mark.parametrize("tile_px", (16, 32, 64))
@pytest.mark.parametrize("pct", (25, 100, 200))
def test_reject_never_drops_a_pixel_and_actually_fires(tile_px, pct, monkeypatch):
    """Each candidate tile renders into two copies of the same noise canvas,
    reject on and reject forced off (iso_tile_extent returning None, the "do
    not reject" contract). Noise rather than zeros, so a skipped darken is
    visible too. A rejected tile must also leave the forced-off canvas
    untouched, which is the direct statement that it painted nothing."""
    elevations, tiles = _synthetic_map(seed=tile_px * 1000 + pct)
    proj = iso_geometry.canvas_size_and_origin(
        MAP_W, MAP_H, tile_px, iso_geometry.MIN_ELEVATION, iso_geometry.MAX_ELEVATION, elev_step_pct=pct
    )
    rng = np.random.default_rng(pct)
    real_extent = iso_geometry.iso_tile_extent
    real_texture = asset_source.get_terrain_texture_array
    reached = {"n": 0}

    def counting_texture(terrain_id):
        reached["n"] += 1
        return real_texture(terrain_id)

    monkeypatch.setattr(asset_source, "get_terrain_texture_array", counting_texture)

    candidates_total = rejected = 0
    for x0, y0, x1, y1 in _random_rects(rng, proj, 40):
        candidates = iso_geometry.tiles_in_screen_rect(x0, y0, x1, y1, MAP_W, MAP_H, proj)
        for cx, cy in candidates.tolist():
            noise = rng.integers(1, 256, size=(y1 - y0, x1 - x0, 3), dtype=np.uint8)
            on, off = noise.copy(), noise.copy()

            before = reached["n"]
            render._render_tile_iso(on, tiles[(cx, cy)], tile_px, proj, elevations, MAP_W, MAP_H, offset=(x0, y0))
            was_rejected = reached["n"] == before

            monkeypatch.setattr(iso_geometry, "iso_tile_extent", lambda *key: None)
            render._render_tile_iso(off, tiles[(cx, cy)], tile_px, proj, elevations, MAP_W, MAP_H, offset=(x0, y0))
            monkeypatch.setattr(iso_geometry, "iso_tile_extent", real_extent)

            assert np.array_equal(on, off), (tile_px, pct, (x0, y0, x1, y1), (cx, cy))
            candidates_total += 1
            if was_rejected:
                rejected += 1
                assert np.array_equal(off, noise), ("rejected a tile that paints", (x0, y0, x1, y1), (cx, cy))

    assert candidates_total > 0
    assert rejected > 0, "the reject never fired, so the equality above proves nothing"


@pytest.mark.parametrize("tile_px", (16, 32, 64))
@pytest.mark.parametrize("pct", (25, 100, 200))
def test_sloped_reject_never_drops_a_pixel_or_an_id_and_actually_fires(tile_px, pct, monkeypatch):
    """The Sloped counterpart, colour pass and pick plane both. Reject forced
    off means index_extent returning None, which also drops _clipped_paint
    to its mask path; tests/test_index_extent.py pins that path byte-identical
    to the fast one, so the only thing that can differ here is the reject.

    MUTATION PROBE (Sloped), run 2026-09-13 and not left in: shrink only the
    reject predicate's box by k px, in both functions or in one. This test
    fails at k=1 in every variant (both, colour only, ids only).
    tests/test_sloped_edit.py::test_patch_after_edit_matches_a_fresh_full_render
    and tests/test_sloped_pick.py both pass at k=1..7 and first fail at k=8,
    and each sees only its own side: colour-only k=8 fails the edit oracle
    but not the pick tests, ids-only k=8 the reverse."""
    elevations, tiles = _synthetic_map(seed=tile_px * 1000 + pct + 7)
    proj = iso_geometry.canvas_size_and_origin(
        MAP_W, MAP_H, tile_px, iso_geometry.MIN_ELEVATION, iso_geometry.MAX_ELEVATION,
        elev_step_pct=pct, corner_headroom_steps=1,
    )
    corner_rise = iso_geometry.corner_rise_px(elevations, proj, rule=render.SLOPE_CORNER_RULE)
    rng = np.random.default_rng(pct + 7)
    real_extent = iso_geometry.index_extent
    real_texture = asset_source.get_terrain_texture_array
    reached = {"n": 0}

    def counting_texture(terrain_id):
        reached["n"] += 1
        return real_texture(terrain_id)

    monkeypatch.setattr(asset_source, "get_terrain_texture_array", counting_texture)

    rejected = 0
    for x0, y0, x1, y1 in _random_rects(rng, proj, 40):
        candidates = iso_geometry.tiles_in_screen_rect(x0, y0, x1, y1, MAP_W, MAP_H, proj)
        for cx, cy in candidates.tolist():
            tile = tiles[(cx, cy)]
            noise = rng.integers(1, 256, size=(y1 - y0, x1 - x0, 3), dtype=np.uint8)
            ids = np.full((y1 - y0, x1 - x0), render.PICK_ID_NONE, dtype=np.int32)
            on, off, ids_on, ids_off = noise.copy(), noise.copy(), ids.copy(), ids.copy()

            before = reached["n"]
            render._render_tile_sloped(on, tile, tile_px, proj, corner_rise, offset=(x0, y0))
            render._render_tile_sloped_ids(ids_on, tile, tile_px, proj, corner_rise, MAP_W, offset=(x0, y0))
            was_rejected = reached["n"] == before

            monkeypatch.setattr(iso_geometry, "index_extent", lambda *key: None)
            render._render_tile_sloped(off, tile, tile_px, proj, corner_rise, offset=(x0, y0))
            render._render_tile_sloped_ids(ids_off, tile, tile_px, proj, corner_rise, MAP_W, offset=(x0, y0))
            monkeypatch.setattr(iso_geometry, "index_extent", real_extent)

            assert np.array_equal(on, off), (tile_px, pct, (x0, y0, x1, y1), (cx, cy))
            assert np.array_equal(ids_on, ids_off), (tile_px, pct, (x0, y0, x1, y1), (cx, cy))
            # Subset, not equality: a painted colour can coincide with its noise byte.
            assert np.all((ids_on != ids) | ~(on != noise).any(axis=2)), "colour painted where the id pass did not"
            if was_rejected:
                rejected += 1
                assert np.array_equal(off, noise), ("rejected a tile that paints", (x0, y0, x1, y1), (cx, cy))
                assert np.array_equal(ids_on, ids), "id pass painted a tile the colour pass rejected"

    assert rejected > 0, "the reject never fired, so the equality above proves nothing"
