#!/usr/bin/env python3
"""Review pack for ground textures at the near and far ends of the zoom ladder
(GH #326): a texture enlarged past its source's detail at 128 and 192 px per
tile (Ice Beach's 512 px source, the 1024 px forest floor), grass shrunk
to 16 and 32 px, and grass at 192 px posed blind as an A/B against the same
render with the angled prefilter off (TASK-691's softness question).

Same protocol as tools/gen_review_pack.py (see tools/REVIEW_PACK.md). Reuses
that tool's ground_texture map and scene loader. Needs the examples/ corpus
and a real AoE2:DE install; renders off-engine, Stepped plus two Flat
reference frames, no units.

Run:
    tools/gen_ground_zoom_review_pack.py
    tools/gen_ground_zoom_review_pack.py --inject blocky
    tools/gen_ground_zoom_review_pack.py --replica 1     # 2nd clean copy
    tools/gen_ground_zoom_review_pack.py --check         # writes nothing

Writes build/review_pack/ground_views/<opaque token>/, gitignored.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gen_review_pack as grp
import gen_seam_eyeball as seam_eyeball
import numpy as np

from descape import asset_source, render_flat, render_iso, render_paint, terrain_blend
from descape import iso_geometry as ig
from testkit import review_pack as rp

PACK_ID = "ground_views"
GRASS, ICE_BEACH, FOREST = 0, 37, 10  # sources 2048, 512 and 1024 px
# The 18x18 level grass at (122, 92) of grp.GROUND_MAP; a repainted frame sets every GRASS tile.
CENTRE = (131, 101)

# frame id -> (style, tile_px, crop (w, h), repaint terrain or None). --check asserts each crop
# shows only its own terrain and no elevation feature.
_FRAMES = {
    "ice_close": ("stepped", 192, (720, 360), ICE_BEACH),
    "ice_mid": ("stepped", 128, (720, 360), ICE_BEACH),
    "forest_close": ("stepped", 192, (720, 360), FOREST),
    "grass_mid": ("stepped", 128, (720, 360), None),
    "grass_far": ("stepped", 32, (384, 192), None),
    "grass_far_overhead": ("flat", 32, (384, 192), None),
    "grass_farthest": ("stepped", 16, (256, 128), None),
    "grass_farthest_overhead": ("flat", 16, (256, 128), None),
    "grass_close_a": ("stepped", 192, (720, 360), None),
    "grass_close_b": ("stepped", 192, (720, 360), None),
}
# One of the pair renders with asset_source.angled_prefilter off (_reference_frame()), no inject
# reaches it, and the run key alone records which.
_PAIR = ("grass_close_a", "grass_close_b")
_CLOSE = ("ice_close", "ice_mid", "forest_close")
_FAR = ("grass_far", "grass_farthest")
# Same texture at the same zoom, blitted 1:1 with no resample: far_grain's clean twin, which
# no inject is applied to (as ground_texture's overhead frame).
_OVERHEAD = ("grass_far_overhead", "grass_farthest_overhead")
# A 2048 px source is only ever shrunk at these levels, so nothing enlarges there.
_SPECIFICITY = "grass_mid"
_ONE_CELL_FRAME = "ice_close"
# unfiltered reaches only the far levels' sizes.
_FAR_MAX_SIZE = asset_source.terrain_texture_size(32)

# Equal horizontal neighbours at 1:1 (measured 2026-11-21): clean close frames 0.002-0.037,
# blocky 0.37-0.67. Speckle as grp._speckle_rate: clean far grass 0.04-0.10%, unfiltered ~11.6%
# (2026-11-24, angled prefilter: clean 0.00%).
CLEAN_RUNS_MAX = 0.10
BLOCKY_RUNS_MIN = 0.30
FAR_SPECKLE_MAX = 0.015
UNFILTERED_SPECKLE_MIN = 0.05
# _edge_strength of the prefiltered pair frame over its reference (TASK-691 probe, grass 192 px):
# [1,2,1] 0.84, soft's Gaussian 0.57. Clean must stay SAME-able, soft clearly softer.
CLEAN_EDGE_MIN = 0.75
SOFT_EDGE_MAX = 0.65
SOFT_SIGMA = 1.5

_REAL_TEXTURE_AT = asset_source.terrain_texture_at
# Point-sampled at the size asked, enlarging or shrinking (grp's memo).
_UNFILTERED = grp._texture_variant("unfiltered", 0)


def _render(frame_id: str, scenario=None, elevations=None) -> np.ndarray:
    style, tile_px, (cw, ch), repaint = _FRAMES[frame_id]
    if scenario is None:
        scenario, elevations = grp._ground_scene(repaint)
    cx, cy = CENTRE
    if style == "flat":
        x0, y0 = (2 * cx + 1) * tile_px // 2 - cw // 2, (2 * cy + 1) * tile_px // 2 - ch // 2
        return render_flat.composite_rect_flat(scenario, x0, y0, x0 + cw, y0 + ch, tile_px, with_units=False)
    mm = scenario.map_manager
    proj = ig.canvas_size_and_origin(
        mm.map_width, mm.map_height, tile_px, ig.MIN_ELEVATION, ig.MAX_ELEVATION,
        elev_step_pct=ig.ELEV_STEP_DEFAULT_PCT, corner_headroom_steps=0,
    )
    sx, sy = ig.tile_screen_origin(cx, cy, int(elevations[cy, cx]), proj)
    x0, y0 = sx + proj.half_w - cw // 2, sy + proj.half_h - ch // 2
    return render_iso.composite_rect_iso(
        scenario, x0, y0, x0 + cw, y0 + ch, elevations, proj, tile_px, {}, {}, with_units=False
    )


def _nearest(terrain_id: int, size: int, angled: bool = False) -> np.ndarray:
    """The real texture (the variant asked for), except a source smaller than
    `size` is enlarged nearest-neighbour instead of bicubic, unfiltered."""
    from PIL import Image

    with Image.open(asset_source.get_terrain_texture_path(terrain_id)) as img:
        enlarged = img.size[0] < size
    return _UNFILTERED(terrain_id, size) if enlarged else _REAL_TEXTURE_AT(terrain_id, size, angled=angled)


def _blocky_at(terrain_id: int, size: int, angled: bool = False):
    """COARSE. Every enlarged source point-sampled: the close frames only."""
    if asset_source.get_terrain_texture_path(terrain_id) is None:
        return _REAL_TEXTURE_AT(terrain_id, size, angled=angled)
    return _nearest(terrain_id, size, angled)


def _one_cell_at(terrain_id: int, size: int, angled: bool = False):
    """LOCALIZED. Only the centre cell's crop of Ice Beach at 192 px is
    point-sampled: one diamond in ice_close (its crop repeats every 10 tiles,
    outside the frame)."""
    real = _REAL_TEXTURE_AT(terrain_id, size, angled=angled)
    tile_px = _FRAMES[_ONE_CELL_FRAME][1]
    if terrain_id != ICE_BEACH or size != asset_source.terrain_texture_size(tile_px):
        return real
    ox, oy = render_paint._crop_offset(*CENTRE, size, tile_px)
    arr = np.array(real)
    arr[oy:oy + tile_px, ox:ox + tile_px] = _nearest(terrain_id, size, angled)[oy:oy + tile_px, ox:ox + tile_px]
    arr.flags.writeable = False
    return arr


def _unfiltered_at(terrain_id: int, size: int, angled: bool = False):
    """COARSE. A shrunk source point-sampled with no prefilter, at the far levels only."""
    return _UNFILTERED(terrain_id, size) if size <= _FAR_MAX_SIZE else _REAL_TEXTURE_AT(terrain_id, size, angled=angled)


def _soft_prefilter(arr):
    """MATCHING (the pair's softness check). A wrap-padded Gaussian, sigma
    SOFT_SIGMA, in place of the angled prefilter: clearly softer patch edges."""
    radius = int(np.ceil(3 * SOFT_SIGMA))
    taps = np.exp(-(np.arange(-radius, radius + 1) ** 2) / (2 * SOFT_SIGMA**2))
    taps /= taps.sum()
    n = arr.shape[0]
    pad = ((radius, radius), (radius, radius)) + ((0, 0),) * (arr.ndim - 2)
    padded = np.pad(arr.astype(np.float32), pad, mode="wrap")
    rows = sum(w * padded[i : i + n] for i, w in enumerate(taps))
    both = sum(w * rows[:, i : i + n] for i, w in enumerate(taps))
    out = np.clip(np.rint(both), 0, 255).astype(np.uint8)
    out.flags.writeable = False
    return out


TEXTURE_INJECTS = {"blocky": _blocky_at, "one-cell": _one_cell_at, "unfiltered": _unfiltered_at}
# Replaces asset_source.angled_prefilter on the pair's prefiltered frame only.
PREFILTER_INJECTS = {"soft": _soft_prefilter}
INJECTS = (*TEXTURE_INJECTS, *PREFILTER_INJECTS)
COARSE_INJECT = "blocky"


def _reference_frame(token: str, replica: int = 0) -> str:
    """The pair frame rendered with the prefilter off: blind (from the run token), and flipped on
    odd replicas so the clean copies do not all share one order (a position bias would replicate)."""
    return _PAIR[(int(hashlib.sha256(token.encode()).hexdigest(), 16) + replica) % 2]


def _clear_caches() -> None:
    """grp's texture-derived memos, blended blocks (keyed without the inject) and the
    loaded textures (a patched prefilter is baked into them)."""
    grp._clear_texture_caches()
    terrain_blend.clear_caches()
    asset_source._forget_prefetches()


@contextlib.contextmanager
def _patched(frame_id: str, inject: str | None, reference: str):
    import pytest

    with pytest.MonkeyPatch.context() as mp:
        if frame_id == reference:
            mp.setattr(asset_source, "angled_prefilter", lambda arr: arr)
        elif inject in TEXTURE_INJECTS and frame_id not in _OVERHEAD:
            mp.setattr(asset_source, "terrain_texture_at", TEXTURE_INJECTS[inject])
        elif inject in PREFILTER_INJECTS and frame_id in _PAIR:
            mp.setattr(asset_source, "angled_prefilter", PREFILTER_INJECTS[inject])
        _clear_caches()
        try:
            yield
        finally:
            _clear_caches()


def _frame_images(inject: str | None, reference: str) -> dict[str, np.ndarray]:
    images = {}
    for frame_id in _FRAMES:
        with _patched(frame_id, inject, reference):
            images[frame_id] = _render(frame_id)
    return images


def _blocks_question(frames: str) -> str:
    return (
        f"Look at {frames}. These show ground textures zoomed in further than the texture's own detail "
        "goes, so some softness is expected. First, in your own words, describe the fine detail: the "
        "edges of the light and dark patches, the grain, and any pixel structure you can see.\n\n"
        "Then the closed question. Is the ground texture anywhere built from visible flat-coloured "
        "blocks: small squares or diamonds of one solid colour, each clearly bigger than the 2x2 frame "
        "pixels one original pixel becomes, so the edges of the light and dark patches run in jagged "
        "stair-steps, as when a small image is enlarged without smoothing? One such patch anywhere "
        "counts, even if the rest of the frame is smooth.\n"
        "- NOT this: (1) blur or softness, smooth gradients with no hard steps; (2) the frame's own "
        "whole-number enlargement, which shows every original pixel as a 2x2 square; (3) the texture's "
        "own specks, cracks and pebbles.\n\n"
        "Answer BLOCKY if such blocks are visible, SMOOTH if not."
    )


def _spec() -> rp.PackSpec:
    """Each check's options are written against the injects' measured masks:
    blocky makes runs of equal neighbours (0.37-0.67 of pairs, against
    0.04 at most clean), one-cell does that in one diamond, unfiltered
    is the isolated-pixel rate --check measures, and soft is the pair's
    edge-strength ratio (_edge_strength(), speckle excluded as the question does)."""
    captions = {
        "ice_close": "Icy shoreline ground on level ground, angled view, zoomed in close.",
        "ice_mid": "The same icy ground, angled view, zoomed in a step less.",
        "forest_close": "Forest floor on level ground (the trees are left out), angled view, zoomed in close.",
        "grass_mid": "Grassland on level ground, angled view, at the same zoom as the second frame.",
        "grass_far": "The same grassland, angled view, zoomed out.",
        "grass_far_overhead": (
            "The same grassland seen straight from above, at the same zoom as the zoomed-out angled frame. "
            "Here each map cell is a square."
        ),
        "grass_farthest": "The same grassland, angled view, zoomed out further.",
        "grass_farthest_overhead": (
            "The same grassland seen straight from above, at the same zoom as the further zoomed-out angled "
            "frame. Here each map cell is a square."
        ),
        "grass_close_a": "Frame A: grassland on level ground, angled view, zoomed in close.",
        "grass_close_b": "Frame B: the same grassland at the same place and zoom as frame A.",
    }
    frames = tuple(rp.Frame(fid, f"{fid}.png", "raw", caption=captions[fid]) for fid in _FRAMES)
    checks = (
        rp.Check(
            id="close_detail",
            question=_blocks_question("the icy ground frames and the forest floor frame"),
            frames=_CLOSE,
            answers=("BLOCKY", "SMOOTH", "CANNOT-TELL"),
        ),
        rp.Check(
            id="far_grain",
            question=(
                "Compare each zoomed-out angled grassland frame with the overhead frame at the same zoom, "
                "which shows the same grass texture at the same zoom seen straight from above. First, in "
                "your own words, describe the fine-scale look of each angled frame against its overhead "
                "twin: grain size, contrast between neighbouring pixels, and any isolated bright or dark "
                "single pixels, glitter or one-pixel streaks.\n\n"
                "Then the closed question. Does either angled frame carry pixel-level noise its overhead "
                "twin does not: neighbouring pixels jumping between light and dark, isolated bright and dark "
                "single pixels sprinkled densely all over, so it reads as glittery or salt-and-pepper next "
                "to the overhead? An occasional stray pixel does NOT count, and neither does a grain that is "
                "finer overall, turned diagonally and flattened vertically, which is expected because the "
                "angled view shrinks and turns the ground.\n\n"
                "Answer SPECKLED if such dense pixel noise is present, SMOOTH if each angled frame reads as "
                "cleanly as its overhead twin apart from that expected shrink and turn."
            ),
            frames=("grass_far", "grass_far_overhead", "grass_farthest", "grass_farthest_overhead"),
            answers=("SPECKLED", "SMOOTH", "CANNOT-TELL"),
        ),
        rp.Check(
            id="mid_detail",
            question=_blocks_question("only the grassland frame at the closer zoom"),
            frames=(_SPECIFICITY,),
            answers=("BLOCKY", "SMOOTH", "CANNOT-TELL"),
        ),
        rp.Check(
            id="close_pair",
            question=(
                "Look at the two close grassland frames, A and B. They show the same patch of ground at the "
                "same zoom, drawn two slightly different ways. First, in your own words, describe the edges "
                "of the larger light and dark patches in each frame: how crisp they are, and how wide the "
                "band of in-between shades across an edge is.\n\n"
                "Then the closed question. Judge only the edges of the larger light and dark patches, several "
                "pixels across or more. Is one frame's ground clearly softer than the other's: those edges "
                "spread over a wider band of in-between shades, so its patches look slightly out of focus "
                "next to the other frame's?\n"
                "- NOT this: (1) isolated single bright or dark pixels, glitter or salt-and-pepper sprinkled "
                "over the texture: ignore them completely in both frames, even if one frame has more of "
                "them; (2) the frame's own whole-number enlargement; (3) a small overall difference in "
                "brightness or contrast with the patch edges equally crisp.\n\n"
                "Answer A-SOFTER if frame A's patch edges are clearly softer, B-SOFTER if frame B's are, "
                "SAME if neither is clearly softer."
            ),
            frames=_PAIR,
            answers=("A-SOFTER", "B-SOFTER", "SAME", "CANNOT-TELL"),
        ),
    )
    return rp.PackSpec(
        id=PACK_ID,
        title="Ground review",
        frames=frames,
        checks=checks,
        preamble=(
            "These are offscreen renders of bare ground (no units or objects) from a map editor for a strategy "
            "game. Each frame is a crop, enlarged by a whole-number factor with no smoothing, so a one-pixel "
            "feature in the original appears that many pixels wide here. Judge only what you can actually see."
        ),
        injects=tuple(INJECTS),
    )


def generate(out_dir: Path, inject: str | None, replica: int = 0) -> list[Path]:
    spec = _spec()
    rp.validate_spec(spec)
    token = grp._run_token(PACK_ID, inject, replica)
    pack_dir = out_dir / PACK_ID / token
    pack_dir.mkdir(parents=True, exist_ok=True)
    images = _frame_images(inject, _reference_frame(token, replica))
    written, rendered = [], {}
    for frame in spec.frames:
        scaled, factor = rp.frame_to_band(images[frame.id])
        path = pack_dir / frame.filename
        seam_eyeball._save_png(scaled, path)
        written.append(path)
        rendered[frame.id] = (scaled.shape[1], scaled.shape[0], factor)
    review = pack_dir / "REVIEW.md"
    review.write_text(grp._review_markdown(spec, rendered, pack_dir))
    written.append(review)
    return written


def _equal_runs(img: np.ndarray) -> float:
    """Share of horizontally adjacent pixel pairs that are exactly equal."""
    return float(np.all(img[:, 1:] == img[:, :-1], axis=-1).mean())


def _edge_strength(img: np.ndarray) -> float:
    """Mean |luma step| over both axes after a 3x3 median: patch-edge sharpness
    with isolated-pixel speckle removed, which would otherwise read as crispness."""
    luma = img[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    h, w = luma.shape
    padded = np.pad(luma, 1, mode="edge")
    med = np.median(np.stack([padded[dy : dy + h, dx : dx + w] for dy in range(3) for dx in range(3)]), axis=0)
    return float((np.abs(np.diff(med, axis=0)).mean() + np.abs(np.diff(med, axis=1)).mean()) / 2)


def _purity_failures(clean: dict[str, np.ndarray], reference: str) -> list[str]:
    failures = []
    for frame_id, (_style, _tile_px, _crop, repaint) in _FRAMES.items():
        base_scn, base_elev = grp._ground_scene(repaint)
        own = GRASS if repaint is None else repaint
        scn = grp._ground_scenario(repaint)
        for tile in scn.map_manager.terrain:
            if tile.terrain_id != own:
                tile.terrain_id = 6 if tile.terrain_id != 6 else 14
        level = np.full_like(base_elev, base_elev[CENTRE[1], CENTRE[0]])
        with _patched(frame_id, None, reference):
            other_terrain = _render(frame_id, scn, base_elev)
            levelled = _render(frame_id, base_scn, level)
        if not np.array_equal(other_terrain, clean[frame_id]):
            failures.append(f"frame {frame_id}: a terrain other than {own} reaches the crop")
        if not np.array_equal(levelled, clean[frame_id]):
            failures.append(f"frame {frame_id}: an elevation feature reaches the crop")
    return failures


def check() -> None:
    """Spec coupling, frame band, crop purity, each inject's reach (printed)
    and scope, the specificity frame untouched, one-cell bounded to one
    diamond and under blocky's largest delta, and the run and speckle rates
    the options rest on."""
    reason = grp._ground_prerequisites()
    if reason:
        raise SystemExit(f"{PACK_ID}: {reason}")
    failures: list[str] = []
    spec = _spec()
    try:
        rp.validate_spec(spec)
    except rp.PackError as exc:
        failures.append(f"spec: {exc}")
    failures += [f"leak: {leak}" for leak in rp.leaks(spec)]
    # Either order measures the same; generate() picks it per run token.
    filtered, reference = _PAIR
    clean = _frame_images(None, reference)
    for frame in spec.frames:
        try:
            scaled, factor = rp.frame_to_band(clean[frame.id])
            rp.validate_frame_size(scaled.shape[1], scaled.shape[0], frame.id)
        except rp.PackError as exc:
            failures.append(str(exc))
            continue
        # The blocks question says "2x2 frame pixels".
        if frame.id in (*_CLOSE, _SPECIFICITY) and factor != 2:
            failures.append(f"frame {frame.id}: enlarged {factor}x, the question says 2x2")
    failures += _purity_failures(clean, reference)

    scopes = {"blocky": _CLOSE, "one-cell": (_ONE_CELL_FRAME,), "unfiltered": _FAR, "soft": (filtered,)}
    diamond = ig.diamond_indices(_FRAMES[_ONE_CELL_FRAME][1])[0].size
    one_cell = rp.LocalizedBound(max_changed_fraction=0.05, max_component_px=diamond)
    injected, reach = {}, {}
    for name in INJECTS:
        images = injected[name] = _frame_images(name, reference)
        touched = {}
        for frame_id in _FRAMES:
            mask = rp.changed_pixels(clean[frame_id], images[frame_id])
            if mask.any():
                touched[frame_id] = (int(np.count_nonzero(mask)), rp.largest_component_px(mask))
            if name == "one-cell" and frame_id == _ONE_CELL_FRAME:
                failures += one_cell.failures(mask, f"inject {name} on {frame_id}")
                if mask.any():
                    ys, xs = np.nonzero(mask)
                    cell_runs = _equal_runs(images[frame_id][ys.min() : ys.max() + 1, xs.min() : xs.max() + 1])
                    print(f"one-cell equal-neighbour runs over its bbox: {cell_runs:.3f}")
                    if cell_runs < BLOCKY_RUNS_MIN:
                        failures.append(f"one-cell: equal-neighbour runs {cell_runs:.3f} under {BLOCKY_RUNS_MIN}")
        missed = [fid for fid in scopes[name] if fid not in touched]
        strays = [fid for fid in touched if fid not in scopes[name]]
        if missed:
            failures.append(f"inject {name}: changed nothing in {', '.join(missed)}")
        if strays:
            failures.append(f"inject {name}: reached {', '.join(strays)}, outside its frames")
        reach[name] = touched
        shown = ", ".join(f"{fid} {px}px/{comp}max" for fid, (px, comp) in sorted(touched.items()))
        print(f"reach: {name} -> {shown or 'nothing'}")
    small, big = reach["one-cell"].get(_ONE_CELL_FRAME), reach[COARSE_INJECT].get(_ONE_CELL_FRAME)
    if small and big and small[1] >= big[1]:
        failures.append(f"one-cell's largest delta {small[1]} px is not under blocky's {big[1]} px")

    runs = []
    for frame_id in _CLOSE:
        c, b = _equal_runs(clean[frame_id]), _equal_runs(injected["blocky"][frame_id])
        runs.append(f"{frame_id} {c:.3f}/{b:.3f}")
        if c > CLEAN_RUNS_MAX:
            failures.append(f"{frame_id}: clean equal-neighbour runs {c:.3f} over {CLEAN_RUNS_MAX}")
        if b < BLOCKY_RUNS_MIN:
            failures.append(f"{frame_id}: blocky equal-neighbour runs {b:.3f} under {BLOCKY_RUNS_MIN}")
    print(f"equal-neighbour runs (clean/blocky): {', '.join(runs)}")
    rates = []
    for frame_id, twin in zip(_FAR, _OVERHEAD, strict=True):
        c, u = grp._speckle_rate(clean[frame_id]), grp._speckle_rate(injected["unfiltered"][frame_id])
        o = grp._speckle_rate(clean[twin])
        rates.append(f"{frame_id} {c:.2%}/{u:.2%} (overhead {o:.2%})")
        # far_grain judges the angled resample against its twin; the shrink both share closes here.
        for fid, rate in ((frame_id, c), (twin, o)):
            if rate > FAR_SPECKLE_MAX:
                failures.append(f"{fid}: clean speckle {rate:.2%} over {FAR_SPECKLE_MAX:.1%}")
        if u < UNFILTERED_SPECKLE_MIN:
            failures.append(f"{frame_id}: unfiltered speckle {u:.2%} under {UNFILTERED_SPECKLE_MIN:.0%}")
    print(f"speckle (clean/unfiltered): {', '.join(rates)}")
    if np.array_equal(clean[filtered], clean[reference]):
        failures.append(f"{filtered} and {reference} are identical: the prefilter does not reach the pair")
    base = _edge_strength(clean[reference])
    clean_ratio = _edge_strength(clean[filtered]) / base
    soft_ratio = _edge_strength(injected["soft"][filtered]) / base
    print(f"pair edge strength over the reference (clean/soft): {clean_ratio:.2f}/{soft_ratio:.2f}")
    if clean_ratio < CLEAN_EDGE_MIN:
        failures.append(f"{filtered}: clean edge ratio {clean_ratio:.2f} under {CLEAN_EDGE_MIN}")
    if soft_ratio > SOFT_EDGE_MAX:
        failures.append(f"{filtered}: soft edge ratio {soft_ratio:.2f} over {SOFT_EDGE_MAX}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(f"{len(failures)} check(s) failed")
    print(
        f"OK: {PACK_ID} frames couple to checks, land in band and show only their own terrain; each inject "
        f"bites only its frames, none reaches {_SPECIFICITY}; one-cell is one bounded diamond under blocky"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=grp.OUT_DIR)
    parser.add_argument("--inject", choices=sorted(INJECTS))
    parser.add_argument("--replica", type=int, default=0, help="Clean replica index (1, 2) for the k=3 rule")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
        return
    if reason := grp._ground_prerequisites():
        raise SystemExit(f"{PACK_ID}: {reason}")
    for path in generate(args.out_dir, args.inject, args.replica):
        print(f"wrote {path}")
    token = grp._run_token(PACK_ID, args.inject, args.replica)
    print(f"\nrun `{args.inject or 'clean'}` is directory {token}; prefilter off in {_reference_frame(token, args.replica)}")


if __name__ == "__main__":
    main()
