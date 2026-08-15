"""Qt-free isometric projection geometry primitives for the "Stepped"
terrain rendering mode (real per-tile elevation displacement) -- Phase 1 of
the real-isometric-Z-height plan. Mirrors render.py's no-Qt contract
deliberately: zero PyQt5 imports, so tools/verify_iso_geometry.py (and any
future headless PNG export path) can exercise this module exactly as easily
as the viewer eventually will.

This module has NO visible effect on its own -- nothing in the running app
calls it yet. It exists purely as tested, correct geometry for Phase 2's
compositor (render_terrain_iso) and Phase 3's viewer wiring to build on.

Coordinate convention, verified against the existing Flat isometric view's
real transform (viewer.py's MapView.set_isometric:
QTransform().scale(1, 0.5).rotate(45-90)), not re-derived from scratch:
screen_x grows with (x+y), screen_y grows with (y-x), with +y down. The
(y-x) term (not the more usual x+y-based tutorial formula) is a direct
consequence of the extra -90deg baked into that transform's rotation --
anyone re-deriving this from memory will get the sign backwards; see the
parent plan's decision #4 for the empirical derivation this matches.

Diamond magnitude is a free choice, not one the real transform pins down:
Flat mode gets its exact 0.707/0.354-fraction diamond size for free by
rotating an *already-rendered* flat pixmap with no rescale; Stepped mode
instead composites a diamond onto canvas pixels directly, so nothing forces
that same magnitude here -- only the (x+y)/(y-x) sign structure and the 2:1
width:height aspect (ISO_HALF_W_FRACTION below) are actually load-bearing,
both of which this module preserves exactly regardless of the magnitude
chosen. See ISO_HALF_W_FRACTION's own comment for why 0.5 was chosen over
the geometrically "exact" 0.707: canvas memory, checked against this
project's real Phase 0 benchmark numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from functools import lru_cache

import numpy as np

# Diamond half-width as a fraction of tile_px (half-height is always exactly
# half of half-width -- see half_dims -- giving the standard 2:1 isometric
# diamond aspect regardless of this fraction's value).
#
# 0.5 was chosen over the "geometrically exact" sqrt(2)/2 (~0.707, what a
# literal 45deg-rotated tile_px x tile_px square's bounding box would be)
# specifically for canvas memory, measured directly against a real file
# (2_Joan_coop_2, 200x200@64px/tile, elevation 0-6): at 0.5 the Stepped
# canvas is ~238MB, comfortably under Flat's measured ~491MB ("memory gets
# better, not worse"); at 0.707 it's ~450MB -- still
# technically under Flat's number for this one file, but only by ~8%, not
# a safe margin, and one a larger elevation range or a differently-shaped
# map could plausibly erase. The real tradeoff either way is texel density
# (a tile_px x tile_px source block covers a smaller diamond, so
# nearest-neighbor sampling loses more detail) -- revisit only after Phase
# 2/3 real output is visible to eyeball, not before; changing this one
# constant is the only thing that would need to change.
#
# At this fraction, half_dims(tile_px) works out so a diamond's bounding
# box is exactly (tile_px, tile_px // 2) for tile_px divisible by 4 (true
# for every tile_px this project actually uses: 8, 16, 32, 64, 128 as of
# Phase B-D's mip levels, MIP_MIN/MAX_TILE_PIXELS below -- previously just
# 64, 32, 16 pre-mips) -- i.e. the diamond keeps its source texture's full
# width and halves its height, the standard "square texture warped to an
# isometric diamond" convention.
ISO_HALF_W_FRACTION = 0.5

# How many elevation levels it takes to raise a tile by one full half_h --
# i.e. ELEV_STEP = half_h // ELEV_STEP_DIVISOR. A first pass shipped
# ELEV_STEP_DIVISOR = 1 (one elevation level = one full half_h step),
# matching Phase 0's own tile_px // 4 prototype value under this module's
# half_dims convention -- but that value turned out to be a real, measured
# mistake, not a stylistic choice: at divisor=1, a tile one elevation level
# higher than an adjacent neighbor (the ±1 delta this project's own
# elevation-invariant confirms is essentially the *only* delta real maps
# ever use) is displaced upward by exactly the same half_h the depth-order
# layout already staggers adjacent tiles by, so the two tiles' screen_y
# positions coincide exactly -- a full depth-layer collapse, not a partial
# occlusion. Measured directly across all 16 real example files: divisor=1
# put 3.13% of tiles' own center pixels behind a different, "occluding"
# neighbor (found via tools/verify_iso_geometry.py's occlusion-aware
# round-trip check); divisor=2 brought that to exactly 0% on every file.
# Phase 2's own verify plan calls this kind of *top-face* pixel disagreement
# unacceptable (only *skirt*-face disagreement is expected/accepted, per
# Risk #5) -- so divisor=2 is not a cosmetic tuning knob, it's the value
# that keeps Phase 3's hit-testing correct at the elevation delta real maps
# actually use. Still revisit visually once Phase 2/3 output exists --
# a bigger step reads as a taller, more dramatic hill on screen -- but any
# increase needs re-measuring against this same occlusion check, not just
# an eyeball pass.
#
# The 0% result is contingent on the ±1 invariant, not unconditional --
# confirmed directly: a synthetic Δelev=2 seam (still divisor=2) brings
# center-pixel occlusion back to ~11%, the same collapse mechanism one
# elevation level further out. Per Risk #2, elevations aren't validated
# (a hand-edited or third-party file could have a steeper seam), and this
# module must degrade gracefully rather than crash when that happens --
# it does (screen_to_tile still returns the correct topmost occluding
# tile, never a wrong answer or an exception), so a steeper-than-invariant
# file just means real occlusion returns, matching what would visibly
# render as a taller cliff, not new breakage.
ELEV_STEP_DIVISOR = 2

# canvas_size_and_origin's elev_step_pct default, expressed as "percent of
# half_h" so it can be driven by a plain QSlider (int, no float round-trip
# through YAML) -- 50 reproduces ELEV_STEP_DIVISOR's measured baseline
# exactly (half_h // 2 == half_h * 50 // 100). Settings > Appearance's
# "Stepped elevation height" slider is allowed to go well above this value
# on purpose: unlike a single hard step, AoE2's real elevation transitions
# are smooth multi-tile ramps, so the occlusion measurement that justified
# capping the old divisor-based control at this baseline doesn't carry over
# the same way to a taller value (per the user, 2026-08-06) -- this default
# is just where the dial starts, not a ceiling.
ELEV_STEP_DEFAULT_PCT = 100 // ELEV_STEP_DIVISOR

# The real legal elevation range -- 0..15, DE's own editor authoring ceiling
# (the scenario format stores elevation as a u8, so a generator-made file
# could in principle exceed 15, but no editor-authored file ever will).
# viewer.py's Set Elevation spinbox is capped the same way. Phase 2/3 sized
# the Stepped canvas from each file's own *observed* min/max elevation
# instead of this fixed range, which is exactly what makes Phase 4's
# incremental redraw unsafe: raising a tile above a file's observed max
# moves it outside the canvas canvas_size_and_origin() already allocated,
# and tile_screen_origin()'s resulting negative screen_y silently wraps via
# numpy fancy indexing rather than raising -- confirmed as a real reachable
# bug (the Elevate tool has no upper clamp), not a hypothetical one. Sizing
# the canvas to this fixed range instead (see render.py's
# render_terrain_iso_with_proj) closes the gap outright: the canvas shape
# becomes a pure function of the map's width/height, stable across every
# elevation edit, which is also a precondition for Phase 4's own
# byte-identical-to-a-full-recomposite acceptance bar -- that bar is
# meaningless if an edit could change the canvas shape out from under the
# comparison. Measured cost of the full 0..15 range across the corpus: a
# 240x240@64px map (largest corpus file) gains 64 canvas rows (+0.8%); a
# 480x480@32px map (largest fixture) gains 32 rows (+0.4%); a 120x120@64px
# blank template gains 64 rows (+1.6%). screen_to_tile's O(range) scan goes
# from 8 to 16 iterations of cheap float ops -- negligible. A file with any
# tile above this ceiling still degrades gracefully rather than corrupting:
# refresh_region_iso / dirty_screen_bbox_iso (render.py) decline any edit
# touching an out-of-range tile and viewer.py's _apply_dirty falls back to
# a full re-render, which is the deliberate backstop for that case, not a
# bug.
MIN_ELEVATION = 0
MAX_ELEVATION = 15

# Mip level ladder's content floor and ceiling, in tile_px (Phase B-D).
#
# MIP_MIN_TILE_PIXELS is a CONTENT floor, not a sharpness one: below 8,
# unit dots stop being drawable at all, and a 4x4 window of a noisy
# 512-square texture (render.py's _tile_block slices at 1:1 texel density)
# is an arbitrary point sample, so adjacent tiles get UNCORRELATED colors
# -- coarse levels would get noisier, the opposite of a mip. Runner-up 16:
# visually safer, gives up a level at the default elev_step_pct and 4x the
# fit-to-view memory. Fall back to 16 if a real eyeball pass (Phase B-D-d)
# finds 8 too noisy.
#
# MIP_MAX_TILE_PIXELS = 128 is the real texel ceiling minus one octave:
# asset_source.LOADED_TEXTURE_SIZE = 512, and render.py's _crop_offset
# requires tile_px to divide it evenly -- 128 is the largest power of two
# under that ceiling with real headroom left for a wraparound-free crop.
MIP_MIN_TILE_PIXELS = 8
MIP_MAX_TILE_PIXELS = 128


def half_dims(tile_px: int) -> tuple[int, int]:
    """(half_w, half_h) for one tile's diamond footprint at this tile_px.

    half_h is derived first and half_w doubled *from* it -- never each
    rounded independently -- so the 2:1 aspect ratio holds exactly. The
    gap-free diamond-tiling proof this module relies on (see
    diamond_indices and tools/verify_iso_geometry.py's partition check)
    depends on that exactness structurally, not just by convention: a
    stray rounding difference between half_w and 2*half_h would open a
    one-pixel seam or overlap between every pair of adjacent tiles."""
    half_h = max(1, int(tile_px * ISO_HALF_W_FRACTION / 2))
    return 2 * half_h, half_h


@dataclass(frozen=True)
class IsoProjection:
    """Bundles everything tile_screen_origin() and screen_to_tile() need,
    so the forward and inverse directions can never drift out of sync with
    each other -- construct only via canvas_size_and_origin(), never by
    hand. (This is exactly the failure mode the parent plan's Risk #6
    warns about for Phase 3's viewer coupling; a shared, immutable record
    makes it structurally impossible one phase early, rather than relying
    on a comment.)"""

    tile_px: int
    half_w: int
    half_h: int
    elev_step: int
    origin_x: int
    origin_y: int
    canvas_w: int
    canvas_h: int
    min_elev: int
    max_elev: int
    # Extra up-screen headroom (canvas pixels) tile_screen_bounds_swept()
    # widens by, on top of its own elev_span term -- Stepped/Flat leave this
    # at 0 (default, so every existing construction site is unchanged) since
    # a tile's footprint there is fully described by its own elevation.
    # Sloped's per-corner blend (iso_geometry.corner_rise_px) can put a
    # corner up to one elev_step above this tile's own elevation (it's
    # blended with neighbors, and real maps only ever have a +-1 elevation
    # delta between neighbors), so a sloped tile's rendered footprint can
    # reach one elev_step higher than tile_screen_bounds_swept's existing
    # per-elevation sweep alone would predict. Deliberately NOT folded into
    # canvas_size_and_origin's own canvas_h/origin_y math above -- see this
    # field's own construction site (canvas_size_and_origin's
    # corner_headroom_steps param) for why keeping it out of canvas sizing
    # matters: a Sloped-only canvas-shape change would make
    # render_terrain_sloped's flat-map byte-identity oracle against
    # render_terrain_iso unreachable for a trivial reason (mismatched array
    # shapes) rather than a real one.
    corner_headroom_px: int = 0


def canvas_size_and_origin(
    w: int,
    h: int,
    tile_px: int,
    min_elev: int,
    max_elev: int,
    elev_step_pct: int = ELEV_STEP_DEFAULT_PCT,
    corner_headroom_steps: int = 0,
) -> IsoProjection:
    """Canvas dimensions and the (origin_x, origin_y) screen offset needed
    so every tile of a w x h map, at any elevation actually observed in
    [min_elev, max_elev], lands at a non-negative pixel position -- for the
    placement formula in tile_screen_origin() below.

    Sized from the *observed* elevation range, not assumed to start at 0:
    several of this project's real example files have a minimum elevation
    of 1, not 0. Deliberately does NOT normalize elevations down to a
    0-based range first -- the (harmless) extra canvas headroom if
    min_elev > 0 is a smaller risk than a second place to get an offset
    wrong; min_elev only affects how much of that headroom is needed (via
    canvas_h), max_elev affects both canvas_h and origin_y (since a tile at
    max_elev is the one that rises furthest, and must not go negative).

    Derivation (both axes are linear in x, y, and elevation independently,
    so their extremes fully characterize the min/max the canvas must
    cover):
    - screen_x = (x+y)*half_w, x in [0,w), y in [0,h): ranges [0, (w+h-2)*half_w].
      Already non-negative -- origin_x = 0. Canvas width is that max plus
      one diamond's own width (2*half_w): (w+h)*half_w.
    - screen_y before origin = (y-x)*half_h - elevation*elev_step: most
      negative at y=0, x=w-1, elevation=max_elev -> -((w-1)*half_h +
      max_elev*elev_step), which sets origin_y. Least negative (tallest
      final position) at y=h-1, x=0, elevation=min_elev; adding one
      diamond's own height (2*half_h) gives canvas_h below.

    elev_step_pct is elev_step expressed as a percent of half_h -- defaults
    to ELEV_STEP_DEFAULT_PCT (see that constant's own comment for why 50 is
    the baseline, not a cap). A caller-supplied value below 1 is clamped to
    1 rather than raising, matching this function's existing max(1, ...)
    floor on the resulting elev_step.

    corner_headroom_steps sets IsoProjection.corner_headroom_px (in elev_step
    units, not raw pixels -- elev_step is a value this function computes, not
    one the caller has in hand yet) -- see that field's own comment. It
    deliberately does NOT affect canvas_h/origin_y/canvas_w below: it's
    consumed only by tile_screen_bounds_swept(), never by this function's own
    canvas-sizing math, so Sloped's canvas shape stays identical to Stepped's
    for the same map -- required for render_terrain_sloped's flat-map
    byte-identity oracle against render_terrain_iso to even be comparable.

    Does NOT reserve any headroom for skirt_quad_indices() output -- a
    skirt hangs drop_px pixels *below* its tile's own bbox (see that
    function's docstring), and how large a drop can occur is a Phase 2
    compositing-policy question (which of two neighboring tiles at
    differing elevations "owns" a given drop, bounded by the map's real
    max elevation delta), not something this geometry-only function knows.
    Phase 2 must account for it explicitly when sizing/placing skirts, or
    a tall enough drop near the bottom of the canvas will index past
    canvas_h.
    """
    half_w, half_h = half_dims(tile_px)
    elev_step = max(1, round(half_h * max(1, elev_step_pct) / 100))
    origin_x = 0
    origin_y = (w - 1) * half_h + max_elev * elev_step
    canvas_w = (w + h) * half_w
    canvas_h = (w + h) * half_h + (max_elev - min_elev) * elev_step
    return IsoProjection(
        tile_px=tile_px,
        half_w=half_w,
        half_h=half_h,
        elev_step=elev_step,
        origin_x=origin_x,
        origin_y=origin_y,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        min_elev=min_elev,
        max_elev=max_elev,
        corner_headroom_px=corner_headroom_steps * elev_step,
    )


# Fields is_exact_mip() treats as EQUAL rather than SCALED -- absolute
# elevation indices, not pixels, so a projection sized for a different
# elevation range must never pass as an exact mip even if every pixel
# field happens to cross-multiply cleanly.
_MIP_EQUAL_FIELDS = ("min_elev", "max_elev")


def is_exact_mip(base: IsoProjection, other: IsoProjection) -> bool:
    """True iff `other` is an exact integer rescaling of `base` -- i.e.
    every pixel-valued field is in the exact ratio other.tile_px :
    base.tile_px, and min_elev/max_elev (absolute elevation indices, not
    pixels) are identical.

    Deliberately iterates dataclasses.fields(base) DYNAMICALLY rather than
    a hand-maintained field list: IsoProjection gained a new field
    (corner_headroom_px, for Phase 6/Sloped) during this very mip-level
    work's own development, which is exactly the kind of drift a
    hardcoded list would silently stop checking the day a new field
    landed. tile_px itself is skipped (it's the ratio's own basis, and
    other.tile_px != base.tile_px is the whole point of calling this).

    Pure integer cross-multiplication (`getattr(other, f) * base.tile_px
    == getattr(base, f) * other.tile_px`) for every scaled field -- no
    float, no epsilon. CONSTRUCT AND COMPARE, never predict: a closed-form
    shortcut like `half_h * elev_step_pct % 100 == 0` is sufficient but
    not necessary and is wrong in both directions at low elev_step_pct
    values (measured directly against canvas_size_and_origin)."""
    for f in fields(base):
        name = f.name
        if name == "tile_px":
            continue
        base_val, other_val = getattr(base, name), getattr(other, name)
        if name in _MIP_EQUAL_FIELDS:
            if base_val != other_val:
                return False
        elif other_val * base.tile_px != base_val * other.tile_px:
            return False
    return True


def mip_projection(
    w: int,
    h: int,
    tile_px: int,
    base: IsoProjection,
    elev_step_pct: int,
    corner_headroom_steps: int = 0,
) -> IsoProjection | None:
    """`base` re-derived at `tile_px`, or None if the result is not an
    exact mip of it (see is_exact_mip). canvas_size_and_origin() stays the
    ONLY IsoProjection constructor -- this calls it again with a different
    tile_px and the SAME elev_step_pct, taking min_elev/max_elev from
    `base` itself so the two can never disagree about the elevation range.
    Returning None rather than raising is what lets a caller's level set
    degrade to "shallower" off an elev_step_pct that yields fewer exact
    neighbors, instead of crashing."""
    candidate = canvas_size_and_origin(
        w, h, tile_px, base.min_elev, base.max_elev, elev_step_pct=elev_step_pct,
        corner_headroom_steps=corner_headroom_steps,
    )
    return candidate if is_exact_mip(base, candidate) else None


def mip_tile_px_candidates(base_tile_px: int) -> dict[int, int]:
    """Level index -> tile_px, for every power-of-two tile_px in
    [MIP_MIN_TILE_PIXELS, MIP_MAX_TILE_PIXELS] reachable from
    base_tile_px by a power-of-two ratio (tile_px = base_tile_px * 2**L,
    L any integer). Level 0 (tile_px == base_tile_px) is always present,
    even when base_tile_px itself falls outside [MIN, MAX] -- D2: scene
    space is pinned to the reference level regardless of where it falls
    on the mip ladder.

    Pure integer arithmetic throughout -- no float, no log2. Geometry-free:
    Flat has no projection (canvas_dims() is a bare multiply, exact at
    every tile_px, no elev_step term to break exactness), so
    FlatChunkCache uses this ladder UNFILTERED; IsoChunkCache instead
    filters it through mip_projection()'s real exactness check, since
    Stepped's elev_step term can and does break exactness at some
    elev_step_pct values (see mip_projections_for)."""
    out = {0: base_tile_px}
    tile_px, level = base_tile_px, 1
    while True:
        tile_px *= 2
        if tile_px > MIP_MAX_TILE_PIXELS:
            break
        out[level] = tile_px
        level += 1
    tile_px, level = base_tile_px, -1
    while tile_px % 2 == 0:
        tile_px //= 2
        if tile_px < MIP_MIN_TILE_PIXELS:
            break
        out[level] = tile_px
        level -= 1
    return out


def mip_projections_for(
    w: int, h: int, base: IsoProjection, elev_step_pct: int, corner_headroom_steps: int = 0
) -> dict[int, IsoProjection]:
    """Level index -> IsoProjection, for every exact mip of `base` among
    mip_tile_px_candidates(base.tile_px) (see is_exact_mip). Level L means
    tile_px = base.tile_px * 2**L: POSITIVE L is FINER (mip-up), NEGATIVE L
    is COARSER (mip-down) -- the scene-space scale factor a caller needs is
    base.tile_px / level.tile_px == 2**-L.

    Level 0 is always present and IS `base` itself (not a reconstructed
    copy) -- and doubles as the self-check on elev_step_pct: the identity
    level is built exactly like every other candidate and REQUIRED to come
    back field-equal to `base`, via an assert, so a caller passing an
    elev_step_pct that did not produce `base` fails loudly here rather
    than silently yielding a wrong level set. Measured on 480x480 with
    base built at tile_px=32/elev_step_pct=50: passing 25 or 75 here
    raises (25's own exact set doesn't even contain 32; 75's is empty),
    while passing 55 -- inside the same round() band at this half_h --
    reproduces base exactly and merely yields a SHALLOWER set ({8,16,32}
    instead of {8,16,32,64,128}). A genuinely wrong elev_step_pct fails
    loudly; one inside the rounding band degrades safely, never silently
    wrong.

    Returns a dict of length >= 1. A length-1 result ({0: base}) is a
    normal answer, not a failure: base.tile_px=16 at elev_step_pct=10
    (reachable via render.tile_pixels_for_map at Potato/Potatest quality)
    has NO exact neighbor at all -- callers must handle mip_levels() == [0].

    The exactness matrix, re-measured against canvas_size_and_origin
    itself for 480x480, 120x120 and 200x144 (identical in all three) at
    base tile_px 32 and 64:

        elev_step_pct 10                -> {32, 64}
        elev_step_pct 25, 75, 125, 175  -> {16, 32, 64, 128}
        elev_step_pct 50, 150           -> {8, 16, 32, 64, 128}
        elev_step_pct 100, 200          -> {4, 8, 16, 32, 64, 128}, 4 excluded by MIP_MIN

    The set is BASE-RELATIVE, not absolute: at elev_step_pct=10 with base
    16 it is the singleton {16}."""
    out: dict[int, IsoProjection] = {}
    for level, tile_px in mip_tile_px_candidates(base.tile_px).items():
        if level == 0:
            identity = canvas_size_and_origin(
                w, h, tile_px, base.min_elev, base.max_elev, elev_step_pct=elev_step_pct,
                corner_headroom_steps=corner_headroom_steps,
            )
            assert is_exact_mip(base, identity), (
                f"mip_projections_for's identity level (tile_px={tile_px}) did not reproduce "
                f"`base` exactly -- elev_step_pct={elev_step_pct} is probably not the value "
                f"`base` was itself built with"
            )
            out[0] = base
            continue
        candidate = mip_projection(w, h, tile_px, base, elev_step_pct, corner_headroom_steps=corner_headroom_steps)
        if candidate is not None:
            out[level] = candidate
    return out


def tile_screen_origin(x: int, y: int, elevation: int, proj: IsoProjection) -> tuple[int, int]:
    """Top-left pixel of tile (x, y)'s (2*half_w, 2*half_h) diamond
    bounding box at the given elevation, in canvas pixel space -- exact
    integer arithmetic throughout (x, y, elevation, and every IsoProjection
    field are ints), no rounding involved. See canvas_size_and_origin's
    docstring for the derivation this is the forward half of."""
    sx = proj.origin_x + (x + y) * proj.half_w
    sy = proj.origin_y + (y - x) * proj.half_h - elevation * proj.elev_step
    return sx, sy


def _diamond_membership(local_x, local_y, half_w: int, half_h: int):
    """Pixel-center rhombus membership test: is the pixel at local (x, y)
    (relative to a tile's own (2*half_w, 2*half_h) bounding box) inside its
    diamond? All-integer arithmetic on purpose (pixel *centers* are always
    at a half-integer coordinate -- local_x+0.5, local_y+0.5 -- so scaling
    the whole inequality by 2 keeps everything exact), which keeps this
    deterministic and avoids float boundary ties between adjacent tiles'
    diamonds that a float version could hit right at a shared edge.
    Works elementwise on numpy arrays (diamond_indices) or plain Python
    ints (screen_to_tile) via the same expression."""
    lhs = np.abs(2 * local_x + 1 - 2 * half_w) * half_h + np.abs(2 * local_y + 1 - 2 * half_h) * half_w
    return lhs <= 2 * half_w * half_h


def _inverse_sample(dst_x, dst_y, half_w: int, half_h: int, tile_px: int):
    """Maps destination pixels local to a tile's (2*half_w, 2*half_h)
    diamond bounding box back to source pixels in its tile_px x tile_px
    square texture block -- the per-pixel counterpart to the whole-tile
    analytic inverse in screen_to_tile(), same change-of-basis structure
    (u, v diamond-local coords -> a, b source-axis coords), just applied
    within one tile instead of across the whole map. Nearest-neighbor only
    (floor, not interpolated) -- accepted aliasing, see the parent plan's
    Risk #3."""
    u = (dst_x + 0.5 - half_w) / half_w
    v = (dst_y + 0.5 - half_h) / half_h
    a = (u + v) / 2.0
    b = (v - u) / 2.0
    src_x = np.clip(np.floor((a + 1.0) / 2.0 * tile_px), 0, tile_px - 1).astype(np.int64)
    src_y = np.clip(np.floor((b + 1.0) / 2.0 * tile_px), 0, tile_px - 1).astype(np.int64)
    return src_y, src_x


@lru_cache(maxsize=8)
def diamond_indices(tile_px: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precomputed nearest-neighbor gather indices warping a tile_px x
    tile_px square source texture block into its isometric diamond shape --
    the "rotate 45, squash 0.5" warp Flat mode gets for free by
    transforming an already-rendered whole flat pixmap, done here per-tile
    instead (Stepped mode composites straight onto canvas pixels, so
    there's no single whole-image transform to reuse).

    Returns (dst_y, dst_x, src_y, src_x): four same-length 1-D int64 arrays,
    one entry per destination pixel that lies inside the diamond -- ready
    for `dest[base_y+dst_y, base_x+dst_x] = source[src_y, src_x]` vectorized
    fancy indexing, where (base_x, base_y) = tile_screen_origin(...).

    lru_cache is safe and bounded here: tile_px only ever takes one of a
    handful of discrete values in this project (SMALL/LARGE_MAP_TILE_PIXELS,
    each optionally halved again by potato mode -- 64, 32, 16 pre-mips, see
    render.py's tile_pixels_for_map; Phase B-D's mip levels make the full
    live set MIP_MIN_TILE_PIXELS..MIP_MAX_TILE_PIXELS, i.e. 8..128, still
    only 5 values), so this cache (maxsize=8) can't grow unbounded."""
    half_w, half_h = half_dims(tile_px)
    dy, dx = np.mgrid[0 : 2 * half_h, 0 : 2 * half_w]
    inside = _diamond_membership(dx, dy, half_w, half_h)
    dst_y = dy[inside].astype(np.int64)
    dst_x = dx[inside].astype(np.int64)
    src_y, src_x = _inverse_sample(dst_x, dst_y, half_w, half_h, tile_px)
    return dst_y, dst_x, src_y, src_x


@lru_cache(maxsize=8)
def _diamond_column_edges(tile_px: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(tops, bottoms, used) -- per-column first/last diamond row, derived
    from the SAME membership mask diamond_indices() itself uses, not a
    hand-rolled edge walk. `used[c]` is False for columns 0 and
    2*half_w - 1, which contain no diamond pixels at all (the diamond's two
    side tips land mid-column, not at the bounding box's own edge columns)
    -- callers MUST filter on `used`, since `np.argmax`/reverse-argmax
    return 0 for an all-False column, which would otherwise silently
    produce a spurious edge at row 0.

    This is what skirt_quad_indices() and shadow_quad_indices() below build
    their per-column edges from, replacing skirt_quad_indices' own former
    hand-rolled `edge_x = half_w -+ 2*t` walk -- confirmed, not assumed,
    to only ever visit every OTHER column (t steps x by 2), leaving the
    skirt as a period-2 comb with unpainted gaps between columns. Measured
    directly on a 5x5 synthetic map with one tile raised by 3: 480 extra
    black (unpainted) pixels appear inside the skirt's own footprint versus
    a flat render of the same map, 0 extra with this derived version."""
    half_w, half_h = half_dims(tile_px)
    dy, dx = np.mgrid[0 : 2 * half_h, 0 : 2 * half_w]
    inside = _diamond_membership(dx, dy, half_w, half_h)
    used = inside.any(axis=0)
    tops = np.argmax(inside, axis=0)
    # reverse-argmax for the last True row per column
    bottoms = (2 * half_h - 1) - np.argmax(inside[::-1, :], axis=0)
    return tops, bottoms, used


@lru_cache(maxsize=256)
def skirt_quad_indices(
    tile_px: int, drop_px: int, side: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pure geometry for one vertical "skirt" face -- the drop_px-tall strip
    hanging below a tile's diamond, along either its lower-left ("left") or
    lower-right ("right") edge (the only two edges of a diamond that can
    ever face the camera as a visible drop -- see the parent plan's decision
    #5: no cliff/wall texture asset exists anywhere in this game's asset
    tree, so a skirt reuses the tile's own top texture; which specific
    source pixels get sampled and any darkening are the compositor's job in
    Phase 2, not this function's -- this is geometry only, not policy about
    which of two neighboring tiles "owns" a given drop.

    Returns (dst_y, dst_x, src_y, src_x) like diamond_indices, using the
    SAME (base_x, base_y) = tile_screen_origin(...) origin diamond_indices
    does -- but dst_x/dst_y are NOT confined to that (2*half_w, 2*half_h)
    box the way diamond_indices' own output is: dst_x spans [0, 2*half_w)
    (same horizontal range, since a skirt never widens past its tile), but
    dst_y spans [0, 2*half_h + drop_px) -- a skirt hangs *below* the box by
    construction (that's what "drop" means), so Phase 2 must size any
    canvas headroom for the largest drop it will ever composite itself;
    canvas_size_and_origin() does not reserve any (see its own docstring).
    Each used column (per _diamond_column_edges -- EVERY column the
    diamond's bottom edge actually occupies, not a stepped subsample)
    repeats straight down for drop_px rows, starting one row below that
    column's own last diamond row (bottoms[c] + 1) -- so the skirt's own
    top row sits immediately adjacent to the diamond's true bottom edge at
    every column, with no gap and no overlap. Source pixels are sampled at
    that same row via the same inverse mapping diamond_indices uses, so a
    skirt's top row always matches the top diamond's own edge pixel with
    no visible seam.

    Formerly built via a hand-rolled `edge_x = half_w -+ 2*t` walk that
    only ever advanced by 2 columns per step -- confirmed to visit every
    OTHER column, leaving the skirt as a period-2 comb (480 extra
    unpainted pixels measured on a synthetic 5x5 map with one tile raised
    by 3, 0 on a flat render of the same map -- see
    _diamond_column_edges' own docstring). Rebuilt on that helper instead,
    which derives columns from the diamond's real membership mask."""
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    if drop_px <= 0:
        raise ValueError(f"drop_px must be positive, got {drop_px}")
    half_w, half_h = half_dims(tile_px)
    _tops, bottoms, used = _diamond_column_edges(tile_px)
    cols = np.arange(2 * half_w)
    in_side = (cols <= half_w) if side == "left" else (cols >= half_w)
    edge_x = cols[used & in_side]
    edge_y = bottoms[edge_x] + 1
    src_y, src_x = _inverse_sample(edge_x, edge_y, half_w, half_h, tile_px)

    drops = np.arange(drop_px)
    n_edge = edge_x.size
    dst_y = (edge_y[:, None] + drops[None, :]).ravel().astype(np.int64)
    dst_x = np.broadcast_to(edge_x[:, None], (n_edge, drop_px)).ravel().astype(np.int64)
    src_y_full = np.broadcast_to(src_y[:, None], (n_edge, drop_px)).ravel()
    src_x_full = np.broadcast_to(src_x[:, None], (n_edge, drop_px)).ravel()
    return dst_y, dst_x, src_y_full, src_x_full


@lru_cache(maxsize=256)
def shadow_quad_indices(
    tile_px: int, rise_px: int, side: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pure geometry for one contact-shadow band -- the WEDGE of the back
    neighbor still visible above a raised column's own two BACK-facing
    diamond edges, onto whatever terrain is already painted behind it (an
    earlier, smaller-d tile in depth_order -- see render.py's
    _render_tile_iso for how a caller turns this into a real darkened
    composite). The up-screen counterpart to skirt_quad_indices'
    down-screen drop, for the two diamond edges a skirt never touches:
    "up_left" (back neighbor (x, y-1), up-LEFT on screen since
    screen_x=(x+y)*half_w) and "up_right" (back neighbor (x+1, y),
    up-RIGHT).

    A WEDGE, not a constant rise_px-tall strip -- this is the whole point,
    and the earlier constant-height version was a real bug (measured: only
    61.3% of darkened pixels landed on the intended neighbor at
    tile_px=64, the rest on tiles never tested for an elevation delta,
    usually the diagonal (x+1, y-1) at the SAME elevation as the caster,
    i.e. a shadow drawn across flat ground -- see maintainer/docs/
    PLAN_CONTACT_SHADOW.md). The caster's back half-edge and the
    neighbor's overlapping half-edge run at OPPOSITE slopes and converge
    at the caster's apex column, so the neighbor's exposed sliver tapers
    to zero there. Each column gets its own height, `avail`, and the band
    is by construction a SUBSET of the neighbor's own diamond.

    Column split is a strict, non-overlapping partition of the diamond's
    real columns (per _diamond_column_edges' `used` mask): up_left gets
    every used column c < half_w, up_right gets c >= half_w -- unlike
    skirt_quad_indices' left/right split, which deliberately shares the
    apex column (both sides legitimately reach the same bottom tip). The
    shadow's two sides never should: each of a tile's two back neighbors
    owns a disjoint half of the top edge, and a shared column here would
    double-darken that column when both neighbors are lower.

    Returns (dst_y, dst_x, depth, span) -- FOUR int64 arrays. No
    src_y/src_x: this paints no texture, only darkens whatever's already
    there, so there's nothing to sample. depth is a 0-based row distance
    from the contact row (0 at the row touching the diamond's own top
    edge); span is that column's own total band height, repeated for every
    one of its rows, so `0 <= depth < span` holds elementwise.

    depth and span are returned as separate INTEGERS rather than a
    pre-divided float ratio on purpose: depth/span is a normalisation
    choice, i.e. the first step of falloff POLICY, and this module's
    documented split keeps policy in render.py (SKIRT_SHADE/CONTACT_SHADE
    live there, not here -- the same split skirt_quad_indices already
    keeps). It also keeps _shadow_factors' float32 pinning a render.py
    decision rather than a cross-module dtype coupling, and lets
    tools/verify_iso_geometry.py assert 0 <= depth < span elementwise.

    Documented extent -- dst_x in [1, 2*half_w - 2] (NOT the looser
    [0, 2*half_w) skirt_quad_indices documents: admitting columns 0 or
    2*half_w - 1 here would be exactly the _diamond_column_edges `used`-
    filter bug Step 0 fixed for skirts, recurring), and additionally never
    exactly half_w - 1 or half_w -- the two apex-adjacent columns, where
    the wedge has tapered to zero. dst_y in [rise_px - half_h + 1,
    half_h - 2] -- still goes NEGATIVE (the whole point is reaching above
    the diamond's own row 0), so callers MUST still clip (see render.py's
    _clipped_darken), but by at most half_h - 2 rather than by rise_px,
    and the reach SHRINKS as rise_px grows. depth in [0, span).

    EMPTY BAND: once rise_px >= 2*half_h - 2 the caster fully hides its
    neighbor and all four arrays come back empty (measured first-empty
    rise: 6 / 14 / 30 for tile_px 16 / 32 / 64). Nothing is drawn -- no
    1px floor, no clamp. Callers must tolerate size-0 arrays.

    NOT a mirror of skirt_quad_indices' geometry -- don't read it as one.
    A skirt bridges the real gap between two tiles' diamonds (its own
    neighbor's elevation sets drop_px exactly). This doesn't bridge
    anything -- it SUBSETS: the band is by construction confined to the
    part of the back neighbor's own diamond the caster doesn't already
    cover, so a larger delta shades LESS (and eventually nothing), where a
    larger skirt drop covers more. The two functions answer different
    geometric questions and happen to share a construction technique, not
    a shape."""
    if side not in ("up_left", "up_right"):
        raise ValueError(f"side must be 'up_left' or 'up_right', got {side!r}")
    if rise_px <= 0:
        raise ValueError(f"rise_px must be positive, got {rise_px}")
    half_w, half_h = half_dims(tile_px)
    tops, _bottoms, used = _diamond_column_edges(tile_px)
    cols = np.arange(2 * half_w)
    in_side = (cols < half_w) if side == "up_left" else (cols >= half_w)
    edge_x = cols[used & in_side]
    # The neighbor's column overlapping this one: its diamond sits at
    # (-half_w, -half_h + rise_px) from ours for up_left, (+half_w, same)
    # for up_right -- so its row r maps to our r - half_h + rise_px.
    # in_side is applied first, so partner is always in range; a defensive
    # clip here would only mask an indexing bug.
    partner = edge_x + (half_w if side == "up_left" else -half_w)

    # used[partner] is REQUIRED, not belt-and-braces: tops is argmax over
    # the membership mask, which returns 0 for the two UNUSED columns (0
    # and 2*half_w - 1). The apex-adjacent caster column pairs with an
    # unused partner, so without this mask the formula reads tops == 0 and
    # invents a half_h - rise_px tall band exactly where the diagonal
    # (x+1, y-1) sits -- the mechanism behind the flat-ground shadows.
    avail = np.where(used[partner], tops[edge_x] - (tops[partner] - half_h + rise_px), 0)
    avail = np.maximum(avail, 0)
    keep = avail > 0
    edge_x = edge_x[keep]
    avail = avail[keep]

    # cumsum(avail) - avail, NOT concatenate(([0], cumsum(avail)[:-1])):
    # the latter is length 1 when avail is empty and np.repeat then
    # raises. This form makes the empty case fall out with no branch.
    starts = np.cumsum(avail) - avail
    depth = np.arange(int(avail.sum())) - np.repeat(starts, avail)
    dst_y = (np.repeat(tops[edge_x] - 1, avail) - depth).astype(np.int64)
    dst_x = np.repeat(edge_x, avail).astype(np.int64)
    span = np.repeat(avail, avail).astype(np.int64)
    return dst_y, dst_x, depth.astype(np.int64), span


def corner_rise_px(elevations: np.ndarray, proj: IsoProjection, rule: str = "average") -> np.ndarray:
    """Phase 6 (Sloped)'s per-corner height field: a (h+1, w+1) int64 array
    of canvas-pixel rise, corner_rise[cy, cx] blending the up-to-4 real
    tiles whose own grid footprint touches grid vertex (cx, cy) -- tiles
    (cx-1, cy-1)/(cx, cy-1)/(cx-1, cy)/(cx, cy) ["NW"/"NE"/"SW"/"SE" from
    the corner's own point of view], clipped at the map edge (a border
    vertex has as few as 1 touching tile, a corner one has exactly 1, an
    interior vertex has all 4).

    rule picks how those 1-4 touching elevations combine into one corner
    value -- "max"/"min" (steepest reasonable readings of a convex/concave
    corner) or "average" (today's default, per the 2026-08-09 preliminary
    screenshot read -- see docs/PLAN_V2_6.md's Track C. A constant, not a
    structural choice: switching it never touches sloped_quad_indices or
    anything downstream). "average" is computed as
    `sum(touching elevations) * elev_step // count`, NOT
    `mean(touching elevations * elev_step)` -- the two differ whenever
    count doesn't evenly divide the sum, and only the sum-first form
    guarantees a FLAT map (every touching tile at the same elevation e)
    yields exactly `e * elev_step` at every corner, integer, no rounding --
    the property render_terrain_sloped's flat-map oracle against
    render_terrain_iso depends on.

    Whole-array, not per-tile: called once per render/patch (like
    render.py's _terrain_grid_and_elevations), not once per tile -- a
    tile's own 4 corner values are then plain (cy, cx) lookups into this
    array's output, shared with its neighbors by construction (the same
    array index for the same physical grid vertex from every tile that
    touches it), which is what keeps neighboring tiles' sloped_quad_indices
    calls seeing IDENTICAL corner values at a shared vertex -- see that
    function's own docstring for why that identity is what keeps adjacent
    tiles' warps meeting exactly rather than merely approximately."""
    if rule not in ("max", "min", "average"):
        raise ValueError(f"rule must be 'max', 'min', or 'average', got {rule!r}")
    h, w = elevations.shape
    elev_step = proj.elev_step
    sums = np.zeros((h + 1, w + 1), dtype=np.int64)
    counts = np.zeros((h + 1, w + 1), dtype=np.int64)
    maxes = np.full((h + 1, w + 1), -1, dtype=np.int64)
    mins = np.full((h + 1, w + 1), np.iinfo(np.int64).max, dtype=np.int64)
    # (rows, cols) is where THIS tile's elevation lands in the (h+1, w+1)
    # corner grid -- e.g. rows=slice(1, h+1), cols=slice(1, w+1) places tile
    # (x, y)'s elevation at corner (x+1, y+1), its own SE corner, which is
    # simultaneously corner (cx, cy)'s "NW-quadrant tile" from that corner's
    # point of view. Four placements, one per quadrant, each just the same
    # elevations array shifted by one row and/or column.
    for rows, cols in (
        (slice(1, h + 1), slice(1, w + 1)),  # this tile is corner's NW-quadrant tile
        (slice(1, h + 1), slice(0, w)),  # NE-quadrant tile
        (slice(0, h), slice(1, w + 1)),  # SW-quadrant tile
        (slice(0, h), slice(0, w)),  # SE-quadrant tile
    ):
        sums[rows, cols] += elevations
        counts[rows, cols] += 1
        maxes[rows, cols] = np.maximum(maxes[rows, cols], elevations)
        mins[rows, cols] = np.minimum(mins[rows, cols], elevations)
    if rule == "max":
        return maxes * elev_step
    if rule == "min":
        return mins * elev_step
    return (sums * elev_step) // counts


@lru_cache(maxsize=8)
def tile_uv_fractions(tile_px: int) -> tuple[np.ndarray, np.ndarray]:
    """(fp, fq), one entry per diamond_indices(tile_px) destination pixel in
    the SAME order (zip against diamond_indices' own dst_y/dst_x to know
    which physical pixel each value belongs to) -- each in [0, 1],
    independently reaching 0/1 exactly at the diamond's own 4 tips.

    Derived from each destination pixel's (u, v) diamond-local coordinate
    -- the same change of basis _inverse_sample uses for texture sampling,
    reused here for a different purpose. The identity
    |u|+|v| == max(|u+v|, |v-u|) (a 45-degree rotation) turns the diamond
    |u|+|v|<=1 into the full unit square in (p, q) = (u+v, v-u) space, so
    (fp, fq) = ((p+1)/2, (q+1)/2) parameterize that square directly --
    which is what lets Phase 6 (Sloped)'s _corner_weights() do a plain
    bilinear blend instead of a general barycentric triangle split, and
    what lets render.py's slope shading take a closed-form gradient of
    that same bilinear patch.

    Tip -> grid-corner correspondence (derived directly from the continuous
    isometric formula screen_x=(mapx+mapy)*half_w, screen_y=(mapy-mapx)*
    half_h relative to a tile's own center -- not guessed, and confirmed by
    tests/test_sloped_geometry.py's partition check): the diamond's
    screen-TOP tip (fp=0, fq=0) is grid corner (x+1, y) ["NE"],
    screen-LEFT (fp=0, fq=1) is (x, y) ["NW"], screen-BOTTOM (fp=1, fq=1)
    is (x, y+1) ["SW"], screen-RIGHT (fp=1, fq=0) is (x+1, y+1) ["SE"].
    Equivalently, in AXIS-ALIGNED tile-fraction terms (fx = mapx - x,
    fy = mapy - y, both in [0, 1]): fx = 1 - fq, fy = fp.

    lru_cache is safe/bounded here for the same reason diamond_indices'
    own cache is: tile_px only ever takes a handful of discrete values."""
    dst_y, dst_x, _src_y, _src_x = diamond_indices(tile_px)
    half_w, half_h = half_dims(tile_px)
    u = (dst_x.astype(np.float64) + 0.5 - half_w) / half_w
    v = (dst_y.astype(np.float64) + 0.5 - half_h) / half_h
    fp = (u + v + 1.0) / 2.0
    fq = (v - u + 1.0) / 2.0
    return fp, fq


@lru_cache(maxsize=8)
def _corner_weights(tile_px: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(w_nw, w_ne, w_sw, w_se) bilinear corner-blend weights, one entry per
    diamond_indices(tile_px) destination pixel, in the SAME order as
    tile_uv_fractions() (see that function for the (fp, fq) derivation and
    the tip <-> grid-corner correspondence this directly encodes)."""
    fp, fq = tile_uv_fractions(tile_px)
    w_ne = (1 - fp) * (1 - fq)
    w_nw = (1 - fp) * fq
    w_sw = fp * fq
    w_se = fp * (1 - fq)
    return w_nw, w_ne, w_sw, w_se


@lru_cache(maxsize=2048)
def sloped_quad_indices(
    tile_px: int, d_nw: int, d_ne: int, d_sw: int, d_se: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sloped mode's counterpart to diamond_indices -- same destination
    PIXEL SET and same source sampling (this never changes which pixels a
    tile owns or which texel each one reads, only how far up-screen each
    pixel sits), with dst_y additionally warped by the tile's 4 corner
    pixel-rises (corner_rise_px()'s output at this tile's own 4 grid
    corners: (x, y), (x+1, y), (x, y+1), (x+1, y+1) for d_nw/d_ne/d_sw/d_se
    respectively) instead of diamond_indices' single uniform per-tile shift.

    Normalizes against min(d_nw, d_ne, d_sw, d_se) before computing anything
    -- dst_y's own smallest value stays 0, matching diamond_indices'
    convention, and the cache key collapses every "all four corners equal"
    case (any absolute elevation) to the SAME normalized (0, 0, 0, 0) key.
    The caller MUST subtract that same minimum from tile_screen_origin()'s
    own base_y before painting -- this function folds the remainder into
    dst_y, not into a returned offset, the same "derived pixel quantity
    folded into base_y, not returned separately" pattern render.py's
    skirt/shadow callers already use. See render.py's _render_tile_sloped.

    Delegates to diamond_indices(tile_px) verbatim when all four corners
    are equal (always (0, 0, 0, 0) after normalization) -- not an
    approximation of that case, the EXACT SAME arrays, floating-point warp
    math never runs. That's what makes render_terrain_sloped's flat-map
    output assertable byte-identical to render_terrain_iso's, rather than
    merely close: an independently-derived bilinear result at equal
    corners could differ from diamond_indices' by a stray floating-point
    ULP at a floor() boundary, which byte-identity would catch as a
    (spurious) failure.

    The warp is applied as ONE rounded shift per screen column (that
    column's mean rise, rounded once), not per pixel. Rounding each pixel's
    rise independently is what the first version did, and it left thin
    unpainted seams inside every sloped tile: within a column, rise varies
    over its full corner-to-corner range, so wherever it DECREASED down the
    column, round() ticked down by 1 across some row -- that -1 cancelled
    the +1 step dst_y already takes, two source pixels collapsed onto one
    destination row, and the row between them was never written at all.
    Against a zeroed canvas that read on screen as nested dark arcs
    following the bilinear iso-contour. A rigid per-column translation of
    an already-contiguous run (diamond_indices guarantees each column's
    un-warped dst_y values are contiguous, by construction of a real
    diamond) can produce neither a gap nor a duplicate -- see
    tests/test_sloped_geometry.py's per-column contiguity check.

    What that gives up, stated plainly because it is a geometric
    approximation and not a rounding detail: a rigid shift carries no
    intra-column COMPRESSION, so a tile keeps its full unwarped 2*half_h
    vertical extent instead of foreshortening with the slope. At
    tile_px=64 (elev_step=8), a one-level north-south ramp paints 32 rows
    where the true sloped silhouette is 24. This is forced, not chosen:
    under this function's fixed pixel count, a per-column mapping that is
    contiguous and order-preserving can ONLY be a rigid translation, so
    real compression means a variable-length resample -- a different return
    contract, which would also break _slope_shade's positional alignment
    and the whole-map partition counts. That is the thing to revisit
    alongside the known tile-to-tile boundary seams, not separately.

    maxsize=2048, not skirt_quad_indices'/shadow_quad_indices' 256: this
    key is a 4-tuple under an averaging rule, not those functions' 3-tuple,
    so the reachable key space is larger for the same real maps -- see
    tests/test_sloped_geometry.py's corpus-tier cardinality measurement,
    which this constant should be revisited against if it ever undersizes
    in practice (a bounded lru_cache degrades to slower re-computation on a
    miss, never incorrect output, so undersizing is a perf regression, not
    a correctness one)."""
    d_min = min(d_nw, d_ne, d_sw, d_se)
    nw, ne, sw, se = d_nw - d_min, d_ne - d_min, d_sw - d_min, d_se - d_min
    if nw == 0 and ne == 0 and sw == 0 and se == 0:
        return diamond_indices(tile_px)
    dst_y, dst_x, src_y, src_x = diamond_indices(tile_px)
    w_nw, w_ne, w_sw, w_se = _corner_weights(tile_px)
    rise = w_nw * nw + w_ne * ne + w_sw * sw + w_se * se
    half_w, _half_h = half_dims(tile_px)
    col_sum = np.bincount(dst_x, weights=rise, minlength=2 * half_w)
    col_count = np.bincount(dst_x, minlength=2 * half_w)
    col_shift = np.round(col_sum / np.maximum(col_count, 1)).astype(np.int64)
    sloped_dst_y = dst_y - col_shift[dst_x]
    return sloped_dst_y, dst_x, src_y, src_x


def ground_outline_corners(w: int, h: int, proj: IsoProjection) -> tuple[tuple[int, int], ...]:
    """(west, north, east, south) screen points -- the four grid-corner
    tiles' own outward diamond tips at elevation 0, in that compass order.
    A Qt-free helper for Phase 3's viewer map-extent outline (the whole
    grid's diamond silhouette), not used by the compositor itself.

    West/north/east/south here are about which grid corner is extremal in
    screen space, not compass direction on the map: screen_x = (x+y)*half_w
    is minimized at (0,0) and maximized at (w-1,h-1); screen_y = (y-x)*half_h
    is minimized at (w-1,0) and maximized at (0,h-1) -- see
    canvas_size_and_origin's own derivation docstring for the same
    extremes. Each tile's own outward corner (its local diamond tip facing
    away from the grid center) is added to that tile's tile_screen_origin.

    Deliberately evaluated at elevation 0 regardless of each corner tile's
    real elevation -- a ground-plane reference showing the underlying grid's
    shape, not a tight bounding silhouette of the actual rendered terrain.
    A tall corner tile's own diamond can legitimately extend above (and, via
    a skirt, below) this outline; that's expected, not a bug this function
    needs to account for."""
    half_w, half_h = proj.half_w, proj.half_h

    def corner(x: int, y: int, dx: int, dy: int) -> tuple[int, int]:
        ox, oy = tile_screen_origin(x, y, 0, proj)
        return (ox + dx, oy + dy)

    west = corner(0, 0, 0, half_h)
    north = corner(w - 1, 0, half_w, 0)
    east = corner(w - 1, h - 1, 2 * half_w, half_h)
    south = corner(0, h - 1, half_w, 2 * half_h)
    return (west, north, east, south)


def depth_order(w: int, h: int) -> np.ndarray:
    """Tile (x, y) pairs for a w x h map, in the order they must be painted
    (back to front) to composite with correct occlusion -- ascending
    d = y - x, per the parent plan's decision #4 (verified against the real
    Flat-mode QTransform, not the more usual x+y -- the extra -90deg baked
    into that transform's rotation is why). Tiles sharing a d never overlap
    on screen (proven in the plan: they're spaced a full diamond-width
    apart in screen_x), so any stable order among same-d tiles is correct;
    x ascending is used only for determinism.

    Returns an (w*h, 2) int64 array of (x, y) pairs, not a list of tuples --
    the largest real map in this project's example set is 480x480
    (230,400 tiles), and Phase 2's compositor iterates this every full
    re-render."""
    xs, ys = np.meshgrid(np.arange(w, dtype=np.int64), np.arange(h, dtype=np.int64), indexing="xy")
    xs = xs.ravel()
    ys = ys.ravel()
    d = ys - xs
    order = np.lexsort((xs, d))  # primary key is the LAST arg: d, then xs
    return np.stack([xs[order], ys[order]], axis=1)


def tile_screen_bounds_swept(x, y, proj: IsoProjection):
    """Screen-space bbox (x0, y0, x1, y1) tile (x, y)'s diamond -- plus any
    skirt that could hang beneath it -- occupies at ANY elevation in
    [proj.min_elev, proj.max_elev], not just its current one. Elementwise:
    x, y may be plain ints or same-shaped numpy arrays, in which case each
    return value is an array too (ordinary numpy broadcasting through
    tile_screen_origin's arithmetic, nothing array-specific about this
    function itself).

    This is the swept-bbox formula render.py's refresh_region_iso used to
    carry as two independent inlined copies (once as a per-seed-tile Python
    loop, once vectorized over the whole grid) -- extracted here so Phase
    6's sloped ramps (whose corner heights will widen this bbox) only ever
    need to change it in one place. x-extent is elevation-independent
    (screen_x doesn't depend on elevation at all -- see tile_screen_origin);
    y-extent is tallest at max_elev, MINUS the contact-shadow headroom a
    tile could cast even further up-screen onto whatever's behind it
    (shadow_quad_indices' own dst_y still goes negative, though now by at
    most half_h - 2 rather than by rise_px -- see that function's
    docstring), and lowest (plus a full skirt-headroom drop) at min_elev,
    matching canvas_size_and_origin's own derivation.
    Callers that don't need the full sweep (e.g. a single tile at its own
    real elevation) should use tile_screen_origin directly instead -- this
    is deliberately looser than that, by construction.

    The shadow-headroom widening is why this bbox is what
    render.py's IsoChunkCache-backed compositing (composite_rect_iso() via
    tiles_in_screen_rect(), which calls this as its own tight per-candidate
    "keep" filter) as well as the incremental-patch path
    (dirty_screen_bbox_iso()) both need to stay correct: a caster tile can
    sit just outside a target rect/chunk and still paint (darken) pixels
    inside it. Without this widening, a chunk composited standalone would
    silently miss a shadow whose caster fell just outside that chunk's own
    rect -- exactly what tools/verify_iso_chunks.py's byte-identity-with-a-
    full-render bar exists to catch. tiles_in_screen_rect()'s own d_hi
    candidate-enumeration bound needs the SAME widening independently (see
    its own comment) -- this function's widening only tightens the final
    per-candidate filter, it doesn't loosen the enumeration step that finds
    candidates in the first place."""
    half_w, half_h = proj.half_w, proj.half_h
    elev_span = (proj.max_elev - proj.min_elev) * proj.elev_step
    x0, y0 = tile_screen_origin(x, y, proj.max_elev, proj)
    x1 = x0 + 2 * half_w
    # y at min_elev is exactly y0 + elev_span (sy is linear in elevation --
    # see tile_screen_origin) -- computed this way instead of a second
    # tile_screen_origin call so the (elevation-independent) sx half of that
    # call isn't redundantly computed and thrown away, which mattered enough
    # to measure: the whole-grid caller below builds and discards a
    # 230,400-entry array for it on this project's largest real map.
    y1 = y0 + 2 * half_h + 2 * elev_span
    # Contact-shadow headroom above the top edge. The band is a wedge whose
    # rows run from the caster's own top edge up to at most half_h - 2 above
    # it (dst_y bottoms out at rise_px - half_h + 1, so the reach SHRINKS as
    # rise_px grows and is independent of max_elev) -- half_w/half_h scale,
    # not elev_span scale, which is why half_h is the term that actually
    # covers it. The elev_span term stays only as retained slack: it was the
    # old (pre-wedge) bound and dropping it would tighten the candidate set
    # for no correctness gain, while a looser bbox only ever costs a few
    # extra rejected candidates -- never a missed one.
    #
    # corner_headroom_px (0 for Stepped/Flat, see IsoProjection's own
    # comment) widens the same top edge further still -- Sloped's per-corner
    # blend can put a corner up to one elev_step above this tile's own
    # elevation, which shadow/skirt reach alone doesn't account for.
    y0 = y0 - elev_span - half_h - proj.corner_headroom_px
    return x0, y0, x1, y1


def tiles_in_screen_rect(x0: int, y0: int, x1: int, y1: int, w: int, h: int, proj: IsoProjection) -> np.ndarray:
    """Every (x, y) tile in a w x h map whose swept bbox
    (tile_screen_bounds_swept, i.e. at ANY elevation in
    [proj.min_elev, proj.max_elev]) intersects the half-open screen rect
    [x0, x1) x [y0, y1) -- same candidate SET and same depth_order as
    filtering depth_order(w, h) through a per-tile tile_screen_bounds_swept
    intersection test (what render.py's refresh_region_iso used to do via
    an O(w*h) vectorized scan over the whole grid), but found by directly
    solving for the s = x+y / d = y-x ranges the rect can reach instead of
    visiting every tile -- O(candidates near the rect), not O(w*h). See
    tools/verify_iso_rect_candidates.py for the exact-order equivalence
    proof against that O(w*h) scan (order matters here, not just set
    membership: two tiles at the same screen position but different depth
    paint in a specific order for correct occlusion).

    Returns an (n, 2) int64 array of (x, y) pairs in depth_order (ascending
    d = y-x, x ascending as tiebreak -- see depth_order's own docstring for
    why that tiebreak is arbitrary-but-fixed rather than load-bearing).

    Why s/d instead of x/y directly: screen_x = origin_x + (x+y)*half_w
    depends only on s = x+y, and (independently) the swept screen_y range
    depends only on d = y-x -- see tile_screen_origin and
    canvas_size_and_origin's own derivation. So the rect's x-window bounds
    s directly, and its y-window bounds d directly, with no cross term --
    two independent O(1) interval solves, each then walked over its own
    (typically small, rect-sized) range rather than the whole grid."""
    half_w, half_h = proj.half_w, proj.half_h
    max_drop = (proj.max_elev - proj.min_elev) * proj.elev_step

    # tile_x0(s) = origin_x + s*half_w, tile_x1(s) = tile_x0(s) + 2*half_w.
    # Padded by 1 diagonal on each side -- cheap insurance against the
    # float floor/ceil boundary, not load-bearing for correctness: the
    # "keep" mask below re-derives the exact bbox test per candidate tile,
    # so a looser s/d range here only costs a few extra rejected candidates,
    # never a missed one.
    s_lo = math.floor((x0 - proj.origin_x) / half_w) - 1
    s_hi = math.ceil((x1 - proj.origin_x) / half_w) + 1

    # tile_y0(d) = origin_y + d*half_h - max_elev*elev_step (tallest sweep
    # position); tile_y1(d) = origin_y + d*half_h - min_elev*elev_step +
    # 2*half_h + max_drop (lowest sweep position, plus full skirt headroom).
    #
    # d_lo (governed by tile_y1, the DOWNWARD reach) needs no shadow term --
    # a contact shadow only ever reaches up-screen, never down, so it can
    # never make a tile whose downward reach doesn't touch the rect newly
    # relevant. d_hi (governed by tile_y0, the UPWARD reach) does: a tile's
    # effective upward reach is tile_y0(d) - max_drop - half_h once its own
    # contact shadow is counted, so the threshold against the rect's bottom
    # edge (y1) has to allow for that extra reach too, or a caster tile
    # sitting just past this rect's far edge -- close enough for its shadow
    # to land inside, too far for its own diamond to -- gets enumerated out
    # before the "keep" filter (which DOES already account for it, via
    # tile_screen_bounds_swept) ever sees it. Exactly the systematic
    # per-chunk-boundary miss tools/verify_iso_chunks.py's byte-identity
    # check would catch.
    #
    # Both terms MUST match tile_screen_bounds_swept's own y0 widening
    # exactly (max_drop is its elev_span, half_h its half_h -- the wedge
    # band's real reach, see that function's own comment): this enumeration
    # bound and that per-candidate "keep" filter are proven equivalent by
    # tools/verify_iso_rect_candidates.py, so widening one without the other
    # breaks that equivalence. Do NOT lean on the trailing +1 pad below,
    # whose own comment declares it not load-bearing.
    d_lo = math.floor((y0 - proj.origin_y + proj.min_elev * proj.elev_step - 2 * half_h - max_drop) / half_h) - 1
    d_hi = math.ceil((y1 - proj.origin_y + proj.max_elev * proj.elev_step + max_drop + half_h) / half_h) + 1

    s_vals = np.arange(s_lo, s_hi + 1)
    d_vals = np.arange(d_lo, d_hi + 1)
    s_grid, d_grid = np.meshgrid(s_vals, d_vals, indexing="xy")
    s_grid = s_grid.ravel()
    d_grid = d_grid.ravel()

    # x = (s-d)/2, y = (s+d)/2 are only integers when s and d share parity
    # (s = x+y, d = y-x -> s+d = 2y, s-d = 2x, both always even) -- filter
    # before dividing so the division is always exact.
    same_parity = (s_grid + d_grid) % 2 == 0
    s_grid, d_grid = s_grid[same_parity], d_grid[same_parity]
    xs = (s_grid - d_grid) // 2
    ys = (s_grid + d_grid) // 2

    in_bounds = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs, ys = xs[in_bounds], ys[in_bounds]

    tx0, ty0, tx1, ty1 = tile_screen_bounds_swept(xs, ys, proj)
    keep = (tx0 < x1) & (tx1 > x0) & (ty0 < y1) & (ty1 > y0)
    xs, ys = xs[keep], ys[keep]

    d = ys - xs
    order = np.lexsort((xs, d))  # primary key is the LAST arg: d, then xs -- matches depth_order()
    return np.stack([xs[order], ys[order]], axis=1)


def screen_to_tile(sx: int, sy: int, elevations: np.ndarray, proj: IsoProjection) -> tuple[int, int] | None:
    """Analytic inverse of tile_screen_origin() -- the parent plan's
    decision #3: an O(max_elev - min_elev + 1) check, no full-resolution
    pick-map buffer. elevations is a (h, w) int array, elevations[y, x]
    giving that tile's real elevation (must be the exact array
    canvas_size_and_origin(...)'s min_elev/max_elev were computed from, or
    results are meaningless).

    For each candidate elevation e in [proj.min_elev, proj.max_elev],
    inverts the flat (x+y)/(y-x) placement as if e were correct -- relative
    to each tile's *center*, not its bbox corner: tile_screen_origin() gives
    the top-left of a (2*half_w, 2*half_h) box, so (sx, sy) is shifted by
    (-half_w, -half_h) before solving, putting it on the same corner-free
    footing as the center point (x+y)*half_w, (y-x)*half_h actually sits on
    -- skipping that shift recovers the wrong tile for any pixel that isn't
    dead-center, which is most of them -- then checks the 2x2 neighborhood
    of integer tiles surrounding that continuous solve (not just the
    single nearest one: a plain round() disagrees with the discrete
    pixel-center diamond mask right at a diamond's sharp corner pixels,
    landing one tile off even when the true owner is directly adjacent).
    Each candidate is kept only if that tile's *actual* elevation is
    really e (per Risk #2, elevations aren't assumed contiguous or
    validated -- most candidate e values reject immediately) and (sx, sy)
    genuinely falls inside that tile's own diamond (not just its bounding
    box). Among any surviving candidates -- possible either from the 2x2
    neighborhood check or from genuine cross-elevation overlap, e.g. a
    taller tile's diamond visibly overlapping a shorter neighbor's, exactly
    the intended occlusion effect -- returns the one with the largest
    d = y - x, the tile painted last by depth_order(), i.e. topmost/
    frontmost, matching what's actually visible on screen at that pixel.

    Returns None if no candidate elevation yields a genuine match -- e.g.
    (sx, sy) lands on a vertical skirt face rather than any top diamond
    (per Risk #5, skirt pixels have no exact analytic inverse; Phase 2's
    verify script measures how often this happens on real renders, not
    this one)."""
    h, w = elevations.shape
    best: tuple[int, int] | None = None
    best_d: int | None = None
    for e in range(proj.min_elev, proj.max_elev + 1):
        # Shifted by (-half_w, -half_h) to land on the same center-relative
        # footing as (x+y)*half_w / (y-x)*half_h -- see the docstring above.
        u = sx - proj.origin_x - proj.half_w  # == (x+y) * half_w, independent of elevation
        v = (sy - proj.origin_y - proj.half_h) + e * proj.elev_step  # == (y-x) * half_h, for this candidate e
        cx = (u / proj.half_w - v / proj.half_h) / 2
        cy = (u / proj.half_w + v / proj.half_h) / 2
        # Check both integers surrounding the continuous solve, not just the
        # nearest one: the discrete pixel-center diamond mask (integer
        # arithmetic, see _diamond_membership) and this continuous solve
        # (plain float division) don't perfectly agree right at a diamond's
        # sharp top/bottom corner pixels, where naive round() can land one
        # tile off even though the true owning tile is directly adjacent.
        for x in (math.floor(cx), math.floor(cx) + 1):
            for y in (math.floor(cy), math.floor(cy) + 1):
                if not (0 <= x < w and 0 <= y < h):
                    continue
                if int(elevations[y, x]) != e:
                    continue
                origin_sx, origin_sy = tile_screen_origin(x, y, e, proj)
                local_x, local_y = sx - origin_sx, sy - origin_sy
                if not bool(_diamond_membership(local_x, local_y, proj.half_w, proj.half_h)):
                    continue
                d = y - x
                if best is None or d > best_d:
                    best, best_d = (x, y), d
    return best
