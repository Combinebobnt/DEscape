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
from fractions import Fraction
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
# MIP_MIN_TILE_PIXELS is a CONTENT floor, not a sharpness one: below 16,
# a 4x4 window of a noisy 512-square texture (render.py's _tile_block
# slices at 1:1 texel density) is an arbitrary point sample, so adjacent
# tiles get UNCORRELATED colors -- coarse levels would get noisier, the
# opposite of a mip. B-D-d's 2026-08-15 eyeball pass provisionally accepted
# 8 (judged moot: the map is "unreadably small" at max zoom-out regardless
# of rasterization) but that judgment didn't hold up under further use --
# the blockiness right before a mip-level switch (see viewer.py's
# _select_mip: selection allowed up to ~2x magnification of the current
# level before switching) was visible well before the floor level's own
# extreme zoom-out band. Raised to 16 (2026-08-27) on that basis, costing
# one level of the ladder and ~4x the fit-to-view memory at the coarsest
# reachable level.
#
# 2026-08-28: that pre-switch-magnification reason is now MOOT -- mip_
# for_scale's rule flipped from floor(log2(scale)) to ceil, which never
# magnifies at all (see that method's own docstring), so the specific
# blockiness this raise was chasing no longer occurs regardless of this
# constant's value. Stays 16 anyway: the CONTENT-floor argument two
# paragraphs up is independent of the selection rule and holds on its own.
# Whether to revert to 8 now that the other reason is gone is a separate,
# not-yet-decided question, deliberately not folded into this change.
#
# MIP_MAX_TILE_PIXELS = 128 is the real texel ceiling minus one octave:
# asset_source.LOADED_TEXTURE_SIZE = 512, and render.py's _crop_offset
# requires tile_px to divide it evenly -- 128 is the largest power of two
# under that ceiling with real headroom left for a wraparound-free crop.
MIP_MIN_TILE_PIXELS = 16
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


def diamond_membership(local_x, local_y, half_w: int, half_h: int):
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
    inside = diamond_membership(dx, dy, half_w, half_h)
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
    inside = diamond_membership(dx, dy, half_w, half_h)
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
    i.e. a shadow drawn across flat ground). The caster's back half-edge and the
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


@lru_cache(maxsize=256)
def shadow_apex_indices(tile_px: int, rise_px: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The contact-shadow band's two APEX columns (half_w - 1 and half_w),
    as their own once-per-tile pass, in shadow_quad_indices' own frame and
    four-array (dst_y, dst_x, depth, span) format.

    WHY THIS IS A SEPARATE FUNCTION AND NOT A WIDENED shadow_quad_indices.
    Those two columns are structurally unreachable by either side of the
    band. up_left's column half_w - 1 pairs with partner 2*half_w - 1 and
    up_right's column half_w pairs with partner 0, and both partners are
    exactly the two columns _diamond_column_edges marks unused, so the
    `used[partner]` mask forces avail to 0 there. That mask is correct and
    must stay: dropping it is what painted a half_h - rise_px tall band
    across flat ground (the v0.2 spill bug). Measured across tile_px
    {8,16,32,64,128} at every rise_px below the empty threshold, neither
    apex column is ever claimed, 0 exceptions. So the gap is not a
    partition mistake this can be folded into. It is a real hole with a
    different owner.

    The owner is the DIAGONAL back neighbour (x+1, y-1), which sits at
    screen offset (0, -2*half_h + rise_px), i.e. directly above the caster.
    Neither direct back neighbour reaches these columns at all, which is
    the same fact the partner mask expresses from the other side.

    WHY THIS DOES NOT REOPEN THE REGRESS A PRIOR DESIGN PASS RULED OUT.
    That ruling forbids sizing a band from the diagonal's own
    EXPOSED region, because that region depends on (x, y-1) and (x+1, y)
    too and so tapers to zero in its turn, one iteration further out. This
    function's extent depends only on the caster's own diamond and rise_px.
    Nothing here can taper. The diagonal's elevation is the CALLER's gate
    (draw or do not draw), never an input to the size, and that distinction
    is the whole design. See render.py's _render_tile_iso for the gate.

    Height is 2*half_h - rise_px, the full exposure: the diagonal's diamond
    occupies caster rows rise_px - 2*half_h through rise_px - 1 in these
    columns, and the caster's own top face covers everything from row 0
    down, leaving exactly that many rows visible above the apex. Returning
    the full exposure rather than a pre-capped strip keeps this module to
    geometry and leaves falloff to render.py's CONTACT_RAMP_DIVISOR, the
    same split shadow_quad_indices already documents. In practice the ramp
    reaches an exact 1.0 no-op after a few rows, so most of a tall wedge is
    an inert multiply rather than visible darkening.

    NOTE THE EMPTY THRESHOLDS DIFFER, deliberately. The band empties once
    rise_px >= 2*half_h - 2 (the caster hides its direct neighbours); this
    empties only at rise_px >= 2*half_h, because the diagonal is a further
    half_h up-screen and stays visible slightly longer. So at the tallest
    elev_step_pct stops this draws a thin cue where the band draws nothing
    at all. That is intended, not an overrun.

    depth is 0 at the row touching the caster's apex and grows upward;
    0 <= depth < span holds elementwise, as for the band. dst_y is always
    negative (above the caster's own diamond), so callers MUST clip, same
    as the band."""
    if rise_px <= 0:
        raise ValueError(f"rise_px must be positive, got {rise_px}")
    _half_w, half_h = half_dims(tile_px)
    avail = 2 * half_h - rise_px
    if avail <= 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty.copy(), empty.copy(), empty.copy()

    tops, _bottoms, used = _diamond_column_edges(tile_px)
    cols = np.arange(2 * _half_w)
    # The columns where the DIAGONAL is the tile immediately above the
    # caster's own top edge, rather than a direct back neighbor. Derived
    # by comparing the two tiles' own edges: the diagonal's diamond ends
    # at caster row rise_px - 1 - tops[c] and the caster's begins at
    # tops[c], so the diagonal reaches the caster's own top edge exactly
    # where rise_px >= 2*tops[c]. The bound is INCLUSIVE, and that is not
    # cosmetic: at `<` the two columns where 2*tops[c] == rise_px are
    # claimed by neither this pass nor the band, leaving a 2-column hole
    # on each flank of the wedge (measured at tile_px=64, rise_px=8:
    # columns 23/24 and 39/40 blank). That set is still disjoint from the
    # band's own claimed columns, which need tops[c] > rise_px, so the two
    # passes cannot overlap and cannot double-darken.
    adjacent = used & (2 * tops <= rise_px)
    edge_x = cols[adjacent]
    if edge_x.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty.copy(), empty.copy(), empty.copy()

    depth = np.tile(np.arange(avail, dtype=np.int64), edge_x.size)
    dst_x = np.repeat(edge_x, avail).astype(np.int64)
    dst_y = (np.repeat(tops[edge_x] - 1, avail) - depth).astype(np.int64)
    span = np.full(dst_x.size, avail, dtype=np.int64)
    return dst_y, dst_x, depth, span


@lru_cache(maxsize=256)
def seam_edge_indices(tile_px: int, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Pure geometry for one 1px SEAM line -- the topmost pixel of every
    column on one of a tile's two up-screen diamond edges, i.e. the tile's
    OWN silhouette against whatever sits behind it. "up_left" is the edge
    facing back neighbor (x, y-1), "up_right" the edge facing (x+1, y) --
    the same two sides shadow_quad_indices names, and a caller applies the
    same per-side `delta > 0` test before drawing either.

    Returns (dst_y, dst_x) -- TWO int64 arrays, same (base_x, base_y) =
    tile_screen_origin(...) origin diamond_indices uses. No src_y/src_x
    (nothing is sampled, this only darkens) and no depth/span (the line is
    1px, so there is no falloff to normalise).

    Why this exists ALONGSIDE shadow_quad_indices rather than replacing
    it. The band's length is a property of the caster AND its neighbor:
    the two tiles' back edges converge at the caster's apex, so the
    neighbor's exposed sliver tapers to zero there and the band stops
    short -- measured coverage of a tile's back edge at tile_px=64, one
    level: 83.9% at elev_step_pct=25, 71.0% at 50, 45.2% at 100, 0% at
    200. Each terrace edge therefore rendered as a DASHED line. This
    function's extent is a property of the CASTER ALONE -- every used
    column on that side, always -- so it has no taper, and it is non-empty
    at every elev_step_pct including 200, where the band is empty by
    design and there was previously no up-screen elevation cue at all.
    (This docstring claimed "no taper and no gap" until 2026-08-16. The
    no-gap half was an overclaim: true within one tile, false across a
    run of them until seam_apex_indices below existed -- see its
    docstring for the measurement.)

    Extending the band onto the diagonal neighbor (x+1, y-1) instead is
    ruled out: that tile's own exposed region depends on (x, y-1) and
    (x+1, y) as well, so
    it would taper to zero in its turn -- the same gap one iteration
    further out.

    Unlike the band, these pixels are ON the tile's own diamond (row
    tops[c] for each column c), so a caller must draw this AFTER the top
    face, not before. That also makes it strictly safer than the band for
    units: a tile's own units are drawn after its terrain, so a seam can
    never darken one, where a band reaching an earlier-painted tile can.

    `used` is REQUIRED, not defensive -- same trap shadow_quad_indices
    documents: tops is argmax over the diamond membership mask and returns
    0 for the two columns that contain no diamond pixels at all (0 and
    2*half_w - 1), which would put a stray seam pixel at row 0.

    Column split EXCLUDES the two apex columns (half_w - 1 and half_w),
    which seam_apex_indices owns and a caller draws as its own once-per-
    tile pass -- up_left gets used columns c < half_w - 1, up_right gets
    c > half_w. See that function for why the apex cannot belong to either
    side: this was shadow_quad_indices' strict partition (c < half_w /
    c >= half_w) until 2026-08-16, and that split left a one-column hole
    at every tile apex on any run where only one side qualified.

    Not skirt_quad_indices' split either -- that one deliberately SHARES
    the apex column between its two sides, which is right for a skirt
    (both sides really do reach the same bottom tip) and wrong here: a
    shared column would be darkened twice whenever both back neighbors are
    lower.

    Documented extent -- dst_x in [1, 2*half_w - 2] and dst_y in
    [0, half_h - 1], both strictly inside the diamond's own bounding box,
    so unlike the band this never needs clipping on the full canvas. The
    two sides are disjoint from each other and from the apex pair; all
    three together cover every used column exactly once."""
    if side not in ("up_left", "up_right"):
        raise ValueError(f"side must be 'up_left' or 'up_right', got {side!r}")
    half_w, _half_h = half_dims(tile_px)
    tops, _bottoms, used = _diamond_column_edges(tile_px)
    cols = np.arange(2 * half_w)
    in_side = (cols < half_w - 1) if side == "up_left" else (cols > half_w)
    edge_x = cols[used & in_side]
    return tops[edge_x].astype(np.int64), edge_x.astype(np.int64)


@lru_cache(maxsize=256)
def seam_apex_indices(tile_px: int) -> tuple[np.ndarray, np.ndarray]:
    """The two apex columns (half_w - 1 and half_w) of the same 1px seam
    line seam_edge_indices draws, as their own pass -- returns (dst_y,
    dst_x) in that function's frame and format.

    Why the apex is separate rather than part of whichever side qualifies.
    The seam's extent is a property of the caster alone, but a terrace
    edge is drawn by a RUN of tiles, and adjacent tiles sit exactly
    half_w canvas px apart. Under the old strict partition (up_left took
    1 .. half_w-1, up_right took half_w .. 2*half_w-2) a run where only
    ONE side qualified left a hole every half_w px -- measured at
    tile_px=64 on tests/test_seam_line.py's own pyramid: 558 seam px in 15
    isolated 8-connected components, one per tile side. Coverage of a
    tile's half-edge when only one side draws was 75% at tile_px=8, 88% at
    16, 94% at 32, 97% at 64, 98% at 128 -- which is why a 1-in-32 dash at
    64 read as continuous to both an A/B render and the eye, while the
    coarsest mip (8, MIP_MIN_TILE_PIXELS, exactly the zoom for reading
    terrain shape) was missing a quarter of the line.

    Both directions verified at tile_px=64: an up_left-only run held at
    canvas col half_w, which HAS diamond pixels (it is up_right's first
    column, so the caster can claim it); an up_right-only run held at
    2*half_w - 1, which has NO diamond pixels of its own -- but that same
    canvas column is the NEXT tile's half_w - 1, which does. So the
    uniform rule is that both apex columns get drawn whenever either side
    qualifies, and both lie inside the caster's own diamond, so nothing
    spills onto a neighbor (this does not reopen the v0.2 spill bug).

    Drawn as a separate once-per-tile pass rather than by letting the
    qualifying side extend into it: that is what makes double-darkening
    structurally impossible when BOTH back neighbors are lower. The apex
    is the most visible column on the tile, and SEAM_SHADE squared there
    is precisely the artifact the strict partition existed to prevent --
    so the fix keeps that guarantee instead of trading it away.

    half_w >= 2 always (half_dims doubles half_h >= 1), so both apex
    columns fall inside the used range [1, 2*half_w - 2] at every tile_px
    and this is never empty. `used` is still applied, for the same reason
    seam_edge_indices applies it -- tops is argmax over the membership
    mask and returns 0 for a column with no diamond pixels.

    Takes no `side`, and "apex" is NOT a side any geometry function here
    accepts -- seam_edge_indices(tile_px, "apex") raises. It exists as a
    string only inside render._seam_factors, as that cache's key for this
    pass's factor array."""
    half_w, _half_h = half_dims(tile_px)
    tops, _bottoms, used = _diamond_column_edges(tile_px)
    cols = np.array([half_w - 1, half_w])
    edge_x = cols[used[cols]]
    return tops[edge_x].astype(np.int64), edge_x.astype(np.int64)


@lru_cache(maxsize=256)
def tile_edge_indices(tile_px: int, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Pure geometry for one 1px OUTLINE edge along a tile's diamond
    silhouette -- all four sides, unlike seam_edge_indices (up_left/
    up_right only, and darkens rather than paints) or skirt_quad_indices
    (left/right only, and a drop_px-tall hanging strip rather than 1px on
    the diamond itself). Used by the farm-terrain perimeter outline
    (render.py's SpriteLayer.farm_by_tile), which needs to trace a whole
    footprint's boundary regardless of which side of the diamond that
    boundary falls on.

    Same `side` vocabulary and grid-neighbor pairing as skirt_quad_indices
    and seam_edge_indices -- "left" is the edge facing grid neighbor
    (x-1, y), "right" faces (x, y+1) (both from `bottoms`, the down-screen
    edges a skirt hangs off), "up_left" faces (x, y-1) and "up_right"
    faces (x+1, y) (both from `tops`, the up-screen edges a seam traces) --
    so a caller translating a footprint boundary into which edges to draw
    can reuse exactly the same four (side, nx, ny) tuples _render_tile_iso
    already builds for skirts and seams, rather than a fifth vocabulary.

    Deliberately NOT built by sharing skirt_quad_indices' or
    seam_edge_indices' own implementations: both are verified paths with
    apex/depth/falloff handling a flat 1px outline doesn't need. Adjacent
    tiles' diamond_indices destination sets are disjoint at every tile_px
    in this project's mip ladder (measured directly, all four grid
    neighbors, tile_px 8/16/32/64/128 -- zero intersection), so unlike a
    seam or shadow this needs no interleaving-order reasoning: whichever
    tile a boundary edge belongs to can draw it at its own moment in
    depth_order without a later-painted neighbor ever overpainting it.

    Returns (dst_y, dst_x) in tile_screen_origin(...)'s frame, matching
    seam_edge_indices' shape (no src_y/src_x -- a solid outline color is
    painted, nothing sampled).

    The two sides of each pair share their apex column (half_w - 1 or
    half_w), same as skirt_quad_indices -- a flat outline color is a
    harmless no-op when a column is painted from both sides, not the
    double-shading hazard seam_edge_indices' own apex split exists to
    avoid, so this doesn't need that split either."""
    if side not in ("left", "right", "up_left", "up_right"):
        raise ValueError(
            f"side must be one of left/right/up_left/up_right, got {side!r}"
        )
    half_w, _half_h = half_dims(tile_px)
    tops, bottoms, used = _diamond_column_edges(tile_px)
    cols = np.arange(2 * half_w)
    in_side = (cols <= half_w) if side in ("left", "up_left") else (cols >= half_w)
    edge_x = cols[used & in_side]
    edge_row = bottoms if side in ("left", "right") else tops
    return edge_row[edge_x].astype(np.int64), edge_x.astype(np.int64)


def corner_rise_px(elevations: np.ndarray, proj: IsoProjection, rule: str = "average") -> np.ndarray:
    """Phase 6 (Sloped)'s per-corner height field: a (h+1, w+1) int64 array
    of canvas-pixel rise, corner_rise[cy, cx] blending the up-to-4 real
    tiles whose own grid footprint touches grid vertex (cx, cy) -- tiles
    (cx-1, cy-1)/(cx, cy-1)/(cx-1, cy)/(cx, cy) ["NW"/"NE"/"SW"/"SE" from
    the corner's own point of view], clipped at the map edge (a border
    vertex has as few as 1 touching tile, a corner one has exactly 1, an
    interior vertex has all 4).

    rule picks how those 1-4 touching elevations combine into one corner
    value -- "max" (today's default: what the game itself was MEASURED to
    do, off in-game captures, replacing an earlier preliminary read that
    guessed "average"), "min", or "average". A constant, not a structural
    choice: switching it never touches sloped_quad_indices or anything
    downstream. "average" is computed as
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
    calls seeing IDENTICAL corner values at a shared vertex.

    That identity is NECESSARY but not SUFFICIENT for adjacent tiles to
    meet exactly, and the difference is worth stating because this
    docstring used to claim otherwise: shared corner values held throughout
    while an east-west ramp still leaked 6600 gap pixels, because the old
    warp derived each column's shift from its own MEAN rise, which the
    neighbour never computes. Abutment additionally requires both tiles
    deriving the same integer cut from the same shared-EDGE interpolation
    of these values -- see _sloped_column_runs."""
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
    which is what lets render.py's slope shading take a closed-form
    gradient of a bilinear height patch, and what unit_rise_px converts
    into axis-aligned tile fractions (fx = 1 - fq, fy = fp) for its own
    two-triangle rise formula.

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


def _round_div(num, den):
    """Round-half-up integer division. Exact for any sign of `num`, for
    `den > 0`, and works elementwise on arrays as well as on scalars.

    Round-half-up rather than numpy's round-half-to-even because both sides
    of a shared tile edge must land on the SAME integer row: banker's
    rounding breaks ties by the parity of the result, which is not a
    property the two neighbours compute in common. Floats are avoided here
    for the same reason -- two mathematically equal float expressions can
    differ by a ULP and land either side of a .5 boundary."""
    return (2 * num + den) // (2 * den)


@lru_cache(maxsize=8)
def _diamond_column_runs(tile_px: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(cols, tops, lens, col_starts, order) -- the corner-INDEPENDENT half
    of the sloped column resample, so it is computed once per tile_px
    rather than once per corner shape.

    INDEXING CONVENTION, which the whole resample depends on: `cols` lists
    the columns that contain diamond pixels at all, and every OTHER array
    here is indexed by ABSOLUTE column `c`, i.e. has length 2*half_w with
    zero-filled entries at the two unused edge columns (0 and 2*half_w - 1;
    see _diamond_column_edges on why those are empty). This matches
    _diamond_column_edges' own tops/bottoms/used convention rather than
    introducing a second, position-in-`cols` one -- mixing the two is
    exactly where an off-by-one would hide.

    `order` indexes diamond_indices(tile_px)' four arrays, sorted by column
    and then by row, so `order[col_starts[c] : col_starts[c] + lens[c]]`
    is column c's pixels top to bottom. That slice is what the resample
    gathers through, and it is contiguous because a diamond column is."""
    tops, bottoms, used = _diamond_column_edges(tile_px)
    half_w, _half_h = half_dims(tile_px)
    cols = np.flatnonzero(used).astype(np.int64)
    lens = np.where(used, bottoms - tops + 1, 0).astype(np.int64)
    col_starts = np.zeros(2 * half_w, dtype=np.int64)
    col_starts[1:] = np.cumsum(lens)[:-1]
    dst_y, dst_x, _src_y, _src_x = diamond_indices(tile_px)
    order = np.lexsort((dst_y, dst_x))
    return cols, tops.astype(np.int64), lens, col_starts, order


@lru_cache(maxsize=8)
def _identity_uv_idx(tile_px: int) -> np.ndarray:
    """The identity permutation over diamond_indices(tile_px)' pixels, as a
    read-only cached array -- what sloped_quad_indices' equal-corner branch
    hands back so that `shade[uv_idx] is shade`-equivalent (a no-op gather)
    without allocating per call. Read-only because it is module-global
    shared state, unlike the fresh arrays the sloped path builds."""
    idx = np.arange(diamond_indices(tile_px)[0].size, dtype=np.int64)
    idx.flags.writeable = False
    return idx


def _sloped_column_runs(
    tile_px: int, nw: int, ne: int, sw: int, se: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(cols, run_start, run_len, raw_len) -- where each screen column of a
    sloped tile starts and how many rows long it is. Absolute-column
    indexed, per _diamond_column_runs' convention above.

    THE POINT, and the reason the seams existed: each cut is evaluated on
    the tile EDGE it is shared with, parametrized by column index alone, by
    integer-only arithmetic. Two tiles meeting at an edge therefore derive
    the identical cut row from the identical inputs, so they abut exactly.
    The old rigid per-column shift instead moved a whole column by that
    column's own MEAN rise -- a quantity the neighbour never computes, so
    the two disagreed and left a gap or an overlap depending on the sign.

        local column   upper cut edge   lower cut edge   k
        c < half_w     NW -> NE         NW -> SW         c
        c >= half_w    NE -> SE         SW -> SE         c - half_w

    The fraction is k/(half_w - 1), not the geometrically exact
    (2k+1)/(2*half_w), so that a cut lands EXACTLY on the corner value at
    k = 0 and k = half_w - 1. That is what makes the two apex columns agree
    with the DIAGONAL neighbour by construction rather than by rounding
    luck. half_w >= 2 always, so the denominator is never 0.

    `raw_len` is returned pre-clamp on purpose: the `max(_, 1)` clamp is
    reachable at elev_step_pct=200 (a one-level slope at 45 degrees on
    screen) and must be an observable, testable path rather than a silent
    one. The clamp can only ever LENGTHEN a run, so it can only ever cause
    overlap, never a hole -- which is why gaps are impossible here for any
    corner input, legal or otherwise. That is strictly stronger than the
    rigid-translation invariant it replaces, which only ruled out holes
    WITHIN a tile."""
    cols, tops, lens, _col_starts, _order = _diamond_column_runs(tile_px)
    half_w, _half_h = half_dims(tile_px)
    den = 2 * (half_w - 1)

    left = cols < half_w
    k = np.where(left, cols, cols - half_w)
    a_up = np.where(left, nw, ne)
    b_up = np.where(left, ne, se)
    a_lo = np.where(left, nw, sw)
    b_lo = np.where(left, sw, se)
    r_up = a_up + _round_div(2 * k * (b_up - a_up), den)
    r_lo = a_lo + _round_div(2 * k * (b_lo - a_lo), den)

    run_start = np.zeros(2 * half_w, dtype=np.int64)
    run_len = np.zeros(2 * half_w, dtype=np.int64)
    raw_len = np.zeros(2 * half_w, dtype=np.int64)
    run_start[cols] = tops[cols] - r_up
    raw_len[cols] = lens[cols] - (r_lo - r_up)
    run_len[cols] = np.maximum(raw_len[cols], 1)
    return cols, run_start, run_len, raw_len


def unit_rise_px(corner_rise: np.ndarray, x: int, y: int, fx: float, fy: float) -> int:
    """The pixel rise of Sloped's own painted surface at one point inside
    tile (x, y) -- (fx, fy) in [0, 1] are the point's axis-aligned tile
    fractions (fx = mapx - x, fy = mapy - y), the same convention
    tile_uv_fractions' docstring names (fx = 1 - fq, fy = fp).

    NOT bilinear -- that was this project's own superseded framing for this
    surface (every spec source that used to call this "bilinear unit-height
    interpolation" predates the seam resample). _sloped_column_runs derives
    each screen column's cut by interpolating corner values along the tile
    EDGE it shares with a neighbour -- the left half of the tile
    (column < half_w) only ever sees nw/ne/sw, the right half only ever
    sees ne/sw/se -- which is a two-triangle piecewise-planar surface split
    along the screen-vertical NE-SW diagonal (fx + fy = 1), not a smooth
    4-corner blend:

        fx + fy <= 1:  rise = nw + fx*(ne - nw) + fy*(sw - nw)
        fx + fy >  1:  rise = se - (1 - fx)*(se - sw) - (1 - fy)*(se - ne)

    Measured against the shipped renderer (recovering the per-pixel rise
    sloped_quad_indices actually paints, over every painted pixel of all 81
    corner configurations at tile_px=64, elev_step=8): this two-triangle
    model disagrees with the renderer by at most 0.88px (integer-rounding
    residual). A bilinear fit over the same 4 corners disagrees by up to
    7.99px -- a full elevation level -- because it is not the surface the
    renderer draws.

    Rounding contract: this and `_round_div` are the only two places this
    module rounds a rise, and both round half-up, exactly once, from an
    EXACT rational -- `fx`/`fy` arrive as Python floats with an exact
    binary value, converted via `fractions.Fraction` rather than scaled and
    truncated, so the same float bit pattern always produces the same
    pixel. That matters here even though no neighbour independently
    recomputes this value (unlike `_sloped_column_runs`' shared-edge
    agreement): the SAME unit and corners must still yield the SAME pixel
    in every render and in every pick, or paint and hit-test drift apart
    silently. Callers must call this function rather than re-deriving the
    formula, so that guarantee actually holds."""
    nw = int(corner_rise[y, x])
    ne = int(corner_rise[y, x + 1])
    sw = int(corner_rise[y + 1, x])
    se = int(corner_rise[y + 1, x + 1])
    p = Fraction(fx)
    q = Fraction(fy)
    if p + q <= 1:
        rise = nw + p * (ne - nw) + q * (sw - nw)
    else:
        rise = se - (1 - p) * (se - sw) - (1 - q) * (se - ne)
    return int(_round_div(rise.numerator, rise.denominator))


@lru_cache(maxsize=1024)
def sloped_quad_indices(
    tile_px: int, d_nw: int, d_ne: int, d_sw: int, d_se: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sloped mode's counterpart to diamond_indices -- the same diamond
    warped so that each screen column is cut where the tile's own 4 corner
    pixel-rises say it should be (corner_rise_px()'s output at this tile's
    4 grid corners: (x, y), (x+1, y), (x, y+1), (x+1, y+1) for
    d_nw/d_ne/d_sw/d_se respectively) instead of diamond_indices' single
    uniform per-tile shift.

    Returns FIVE arrays, (dst_y, dst_x, src_y, src_x, uv_idx), not
    diamond_indices' four. A sloped column is a VARIABLE-length run, so the
    output is no longer a permutation of the diamond's own pixels and no
    longer positionally aligned with anything derived from
    tile_uv_fractions(). `uv_idx` is that alignment, restored explicitly:
    for output pixel i, uv_idx[i] is the diamond_indices/tile_uv_fractions
    entry it resamples, so a per-pixel quantity computed over the diamond
    (render._slope_shade's shading factor is the only one today) is
    gathered as `shade[uv_idx]`. src_y/src_x are already gathered that way
    here, so callers only need uv_idx for their own parallel arrays.

    Normalizes against min(d_nw, d_ne, d_sw, d_se) before computing
    anything, so the cache key collapses every "all four corners equal"
    case (any absolute elevation) to the SAME normalized (0, 0, 0, 0) key.
    The caller MUST subtract that same minimum from tile_screen_origin()'s
    own base_y before painting -- this function folds the remainder into
    dst_y, not into a returned offset, the same "derived pixel quantity
    folded into base_y, not returned separately" pattern render.py's
    skirt/shadow callers already use. See render.py's _render_tile_sloped.

    dst_y is NOT bounded below by 0 (an older version of this docstring
    claimed it was, and that was measured false even then): the bound is
    0 <= r_up, r_lo <= max(corners), so rows lie in
    [-max(corners), 2*half_h - 1]. What keeps a negative row safe on canvas
    is _clipped_paint's masking plus corner_headroom_px' sizing, never a
    min-0 property -- see tests/test_sloped_geometry.py's
    test_run_rows_stay_within_derived_bounds.

    Delegates to diamond_indices(tile_px) verbatim when all four corners
    are equal (always (0, 0, 0, 0) after normalization) -- not an
    approximation of that case, the EXACT SAME arrays, plus
    _identity_uv_idx's no-op gather as the fifth. Floating-point warp math
    never runs. That's what makes render_terrain_sloped's flat-map output
    assertable byte-identical to render_terrain_iso's, rather than merely
    close. NOTE THE ALIASING ASYMMETRY between the two branches: the
    equal-corner path hands back module-global CACHED arrays (mutating one
    corrupts every later tile, which is why _identity_uv_idx is read-only),
    while the sloped path below allocates all five fresh.

    Where the cuts come from, and why the old tile-to-tile seams are gone:
    _sloped_column_runs evaluates every cut on the tile EDGE it is shared
    with, by integer-only arithmetic, so two neighbours derive the identical
    cut row from identical inputs and abut exactly. The predecessor shifted
    each column rigidly by that column's own MEAN rise -- a quantity the
    neighbour never computes -- which left a 1px diamond lattice of gaps and
    overlaps across every sloped area. See that function's docstring for the
    cut table and tests/test_sloped_geometry.py's
    test_adjacent_tiles_abut_exactly for the proof that replaced the old
    rigid-translation argument.

    The source resample within a run is a uniform nearest-neighbour pick,
    integer only: pixel i of a run of run_len takes diamond row
    ((2*i + 1) * n) // (2 * run_len) of that column's n original rows. Call
    it what it is -- a LINEARIZATION, not just NN aliasing: `rise` along a
    column is quadratic in v whenever the twist term (ne - nw - se + sw) is
    non-zero, so this is exact at both endpoints and at run_len == n (a
    near-flat tile degrades to the identity gather), and sub-pixel in
    between for real slopes. It is swappable without touching the contract,
    since every invariant is stated on dst_y/run_len, never on the pick.

    maxsize=1024, not skirt_quad_indices'/shadow_quad_indices' 256: this
    key is a 4-tuple under an averaging rule, not those functions' 3-tuple,
    so the reachable key space is larger for the same real maps -- see
    tests/test_sloped_geometry.py's corpus-tier cardinality measurement,
    which this constant should be revisited against if it ever undersizes
    in practice (a bounded lru_cache degrades to slower re-computation on a
    miss, never incorrect output, so undersizing is a perf regression, not
    a correctness one).

    Sized from measurement 2026-08-20, not by feel, because the 5-tuple's
    per-entry payload is 7.5x the old one's (every array freshly allocated,
    where the predecessor had three of four aliasing diamond_indices' own
    cache): 61 KB per entry at tile_px=64, 246 KB at 128. The realistic
    working set is ONE open map's corner keys across its live mip levels --
    measured at most 171 distinct corner keys in any single corpus file,
    times 5 mip levels = 855 entries, about 55 MB with every level
    resident. 1024 covers that with headroom while halving the theoretical
    worst case the old 2048 allowed. (329 keys is the count across the
    WHOLE corpus at once; that is not a working set, since only one map is
    open at a time.)"""
    d_min = min(d_nw, d_ne, d_sw, d_se)
    nw, ne, sw, se = d_nw - d_min, d_ne - d_min, d_sw - d_min, d_se - d_min
    dia_dst_y, dia_dst_x, dia_src_y, dia_src_x = diamond_indices(tile_px)
    if nw == 0 and ne == 0 and sw == 0 and se == 0:
        return dia_dst_y, dia_dst_x, dia_src_y, dia_src_x, _identity_uv_idx(tile_px)

    cols, run_start, run_len, _raw_len = _sloped_column_runs(tile_px, nw, ne, sw, se)
    _cols, _tops, lens, col_starts, order = _diamond_column_runs(tile_px)
    # Per emitted column (position within `cols`, not absolute column):
    # its run length, its original diamond length, where its run starts on
    # canvas, and where its pixels start in `order`.
    lengths = run_len[cols]
    n_rows = lens[cols]
    ends = np.cumsum(lengths)
    which = np.repeat(np.arange(cols.size, dtype=np.int64), lengths)
    row_in_run = np.arange(int(ends[-1]), dtype=np.int64) - np.repeat(ends - lengths, lengths)

    dst_x = cols[which]
    dst_y = run_start[cols][which] + row_in_run
    # The min() is belt-and-braces, not a live guard: pick(run_len - 1) is
    # n - ceil(n / (2*run_len)), which is <= n-1 for every run_len >= 1, at
    # any length ratio. Note run_len > n is the ORDINARY lengthening case
    # (an east-rising column stretches, ~25% of a tile at tile_px=64), not
    # the max(raw_len, 1) clamp -- those are different things.
    pick = np.minimum(((2 * row_in_run + 1) * n_rows[which]) // (2 * lengths[which]), n_rows[which] - 1)
    uv_idx = order[col_starts[cols][which] + pick]
    return dst_y, dst_x, dia_src_y[uv_idx], dia_src_x[uv_idx], uv_idx


@lru_cache(maxsize=256)
def sloped_tile_outline(
    tile_px: int, d_nw: int, d_ne: int, d_sw: int, d_se: int
) -> tuple[tuple[int, int], ...]:
    """The closed screen outline of one sloped tile, as (x, y) points in the
    same tile-local space sloped_quad_indices' dst_x/dst_y use -- so a caller
    places it at exactly the base_x/base_y it would paint that tile at,
    -d_min normalization included. Track C4's hover/highlight outline.

    A sloped tile is NOT a 4-point diamond and cannot reuse
    unit_pick.diamond_points: each screen column is cut independently by
    _sloped_column_runs, so the silhouette is a pair of warped staircases
    (an east-rising tile is visibly taller on its east side than its west).
    Traced as the exact pixel boundary -- top edge left to right, then the
    bottom edge back -- rather than as a smoothed hull through column
    centres, so the outline bounds precisely the pixels the tile claims and
    a highlight can never suggest coverage the pick plane disagrees with.

    Columns 0 and 2*half_w - 1 hold no diamond pixels at all (see
    _diamond_column_edges), so `cols` is contiguous and the two staircases
    join without a gap. Cached on the same normalized corner key
    sloped_quad_indices uses, for the same reason: a real map has few
    distinct corner shapes, and a brush highlight rebuilds this per tile."""
    d_min = min(d_nw, d_ne, d_sw, d_se)
    nw, ne, sw, se = d_nw - d_min, d_ne - d_min, d_sw - d_min, d_se - d_min
    cols, run_start, run_len, _raw_len = _sloped_column_runs(tile_px, nw, ne, sw, se)

    top: list[tuple[int, int]] = []
    bottom: list[tuple[int, int]] = []
    for c in cols.tolist():
        y0 = int(run_start[c])
        y1 = y0 + int(run_len[c])
        top.append((c, y0))
        top.append((c + 1, y0))
        bottom.append((c, y1))
        bottom.append((c + 1, y1))
    # Reversed so the bottom is walked right-to-left, closing the ring: the
    # top ends at the rightmost column's right edge and the reversed bottom
    # starts there, leaving one vertical edge at each end of the tile.
    return tuple(top + bottom[::-1])


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
    render_cache.py's IsoChunkCache-backed compositing (composite_rect_iso() via
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
        # arithmetic, see diamond_membership) and this continuous solve
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
                if not bool(diamond_membership(local_x, local_y, proj.half_w, proj.half_h)):
                    continue
                d = y - x
                if best is None or d > best_d:
                    best, best_d = (x, y), d
    return best
