"""Footprint-aware unit hit-testing and selection geometry -- phase 3's
P3-c.

Qt-free on purpose, so it can be tested in the default tier without a `gui`
mark: unit_polygons() returns plain coordinate tuples and the viewer turns
them into QPolygonF. Import direction is one-way (render -> unit_filter,
unit_pick -> render), so there is no cycle.

Why an index rather than a scan
-------------------------------
This project's example scenarios reach ~10,871 units in one file. Neither
existing per-unit structure can answer "which unit is under this pixel":

- render._units_by_tile() is keyed on footprint tiles too, but it is a paint
  structure, not an identity one: its values are (unit, color) pairs with no
  `order`, no player_id, and no stable address, so it cannot disambiguate
  stacked units or survive a selection across a rebuild. It is also rebuilt
  on a different schedule -- it is filter-scoped like this index, but lives
  inside the chunk caches.
- render._flat_unit_draws() is pixel-space, drops unit identity entirely,
  and is a construction-time snapshot held inside a chunk cache.

UnitIndex is therefore a third structure, keyed on every tile in a unit's
FOOTPRINT. That is what makes "click a building anywhere on its 4x4 slab"
work without a symmetric-radius search around its anchor tile -- deliberately
not added; the footprint expansion already answers the question exactly.

The two asymmetries this module exists to respect
-------------------------------------------------
Both are load-bearing, and both make the obvious reuse wrong:

1. render._unit_iso_footprint() paints every footprint tile at the unit's
   OWN tile's elevation, so a multi-tile building is a flat slab at one
   height. iso_geometry.screen_to_tile() is therefore NOT reusable for unit
   picking: it gates each candidate on `int(elevations[y, x]) == e`, which
   is exactly the wrong test for a unit diamond sitting over a tile at a
   different terrain height. pick_unit() reuses its *structure* and swaps
   that gate.
2. For the same reason a unit's highlight is not a union of per-tile terrain
   polygons -- every diamond sits at the unit's own elevation, not each
   footprint tile's own.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from descape import iso_geometry, render
from descape.unit_filter import UnitFilter


@dataclass(frozen=True)
class UnitEntry:
    """One pickable unit, with everything a pick or an inspector needs and
    nothing that would go stale.

    own_x/own_y are int(unit.x)/int(unit.y) -- the tile whose elevation the
    whole footprint is drawn at (see this module's asymmetry 1), not a
    footprint corner.
    """

    player_id: int
    unit: object  # AoE2ScenarioParser Unit; duck-typed, see render.py
    own_x: int
    own_y: int
    order: int


@dataclass
class UnitIndex:
    """Footprint-expanded unit index -- built once per (scenario, filter),
    reused by every pick.

    Stores TILE COORDINATES ONLY, never elevations. That is deliberate: a
    terrain or elevation edit therefore never invalidates this index, and
    only a filter change or (phase 3.5) a unit mutation does. Elevation is
    read live from the caller's own elevations array at pick time, which is
    the same array MapView keeps in sync -- so a pick can never act on a
    stale height snapshot.
    """

    entries: list[UnitEntry] = field(default_factory=list)
    # Every tile in each unit's footprint, not just its own tile.
    by_tile: dict[tuple[int, int], list[int]] = field(default_factory=dict)
    # (player_id, reference_id) -> entry. THE stable address of a unit: a
    # list index or object identity would not survive phase 3.5's add/remove.
    by_key: dict[tuple[int, int], UnitEntry] = field(default_factory=dict)

    def entry_for_key(self, key: tuple[int, int]) -> UnitEntry | None:
        return self.by_key.get(key)


def unit_key(player_id: int, unit) -> tuple[int, int]:
    """The identity phase 3 selects by, resolved late through
    UnitIndex.by_key rather than held as an object reference."""
    return (player_id, unit.reference_id)


def build_index(scenario, unit_filter: UnitFilter = UnitFilter()) -> UnitIndex:
    """Builds the index in EXACTLY render._flat_unit_draws()/_units_by_tile()'s
    own player-then-unit-list iteration order, so `order` is directly
    comparable against paint order.

    Filtered units are absent entirely rather than flagged, so a hidden unit
    is not pickable -- the same object drives what's drawn and what's
    selectable, and the two cannot disagree.

    Off-map units (render.unit_tile_bounds() returns None) are dropped,
    exactly as _flat_unit_draws() already drops them. `order` is therefore a
    dense rank over the SURVIVING units, not a position in
    unit_manager.units. That is fine because order is only ever compared
    relatively, and dropping a prefix-consistent subset preserves relative
    order -- but it does mean order is not a unit address. by_key is.
    """
    mm = scenario.map_manager
    tile_w, tile_h = mm.map_width, mm.map_height
    index = UnitIndex()

    for player_id, units in enumerate(scenario.unit_manager.units):
        for unit in units:
            if not unit_filter.matches(player_id, unit):
                continue
            bounds = render.unit_tile_bounds(unit, tile_w, tile_h)
            if bounds is None:
                continue
            tile_x0, tile_x1, tile_y0, tile_y1 = bounds
            entry = UnitEntry(
                player_id=player_id,
                unit=unit,
                own_x=int(unit.x),
                own_y=int(unit.y),
                order=len(index.entries),
            )
            index.entries.append(entry)
            index.by_key[unit_key(player_id, unit)] = entry
            for ty in range(tile_y0, tile_y1):
                for tx in range(tile_x0, tile_x1):
                    index.by_tile.setdefault((tx, ty), []).append(entry.order)
    return index


def _flat_key(entry: UnitEntry) -> int:
    """Flat's paint order is composite_rect_flat()'s own: it selects hits
    with np.nonzero (ascending indices, i.e. overlay_units' order) and blits
    opaque rects in that order, so last-drawn wins outright."""
    return entry.order


def _stepped_key(order: int, x: int, y: int) -> tuple[int, int, int]:
    """Stepped's paint order, reproduced exactly.

    depth_order() lexsorts (d = y - x, x), and render._units_by_tile()
    buckets a unit into EVERY footprint tile, so each of its diamonds paints
    at the position of the tile it covers -- not at the unit's own tile. The
    key is therefore a function of the COVERING tile (x, y), which for a
    multi-tile building differs per diamond.

    That is why this takes coordinates rather than reading own_x/own_y off
    the entry: those were here because a footprint extended past its own
    tile and same-d units could therefore overlap. Now the covering tile IS
    the paint position, and `order` is what breaks a tie WITHIN one tile's
    bucket -- matching Flat's last-drawn-wins, which the renderer's
    per-tile bucketing made Stepped agree with.
    """
    return (y - x, x, order)


def _pick_unit_flat(index: UnitIndex, sx: int, sy: int, tile_px: int, tile_w: int, tile_h: int) -> UnitEntry | None:
    tx, ty = sx // tile_px, sy // tile_px
    if not (0 <= tx < tile_w and 0 <= ty < tile_h):
        return None
    orders = index.by_tile.get((tx, ty))
    if not orders:
        return None
    return max((index.entries[o] for o in orders), key=_flat_key)


def _pick_unit_stepped(
    index: UnitIndex,
    sx: int,
    sy: int,
    elevations: np.ndarray,
    proj: iso_geometry.IsoProjection,
) -> tuple[UnitEntry, tuple[int, int]] | None:
    """iso_geometry.screen_to_tile()'s structure with the elevation gate
    swapped for a unit-footprint gate (see this module's asymmetry 1).

    Returns (entry, (x, y)) -- the winner AND the footprint tile it was
    covering when it won, because that tile is its paint position (see
    _stepped_key) and pick_unit() needs it for the terrain compare. The
    winner's own tile is the wrong thing to compare there once a unit paints
    one diamond per footprint tile.

    Terrain occlusion is applied by the caller, pick_unit(), and the ORDER
    matters: losers are discarded here, before any terrain test, never
    after. Applying the terrain test per candidate and taking the max over
    the survivors is a different -- also defensible, but untested -- rule.
    """
    h, w = elevations.shape
    best: tuple[UnitEntry, tuple[int, int]] | None = None
    best_key: tuple[int, int, int] | None = None

    for e in range(proj.min_elev, proj.max_elev + 1):
        u = sx - proj.origin_x - proj.half_w
        v = (sy - proj.origin_y - proj.half_h) + e * proj.elev_step
        cx = (u / proj.half_w - v / proj.half_h) / 2
        cy = (u / proj.half_w + v / proj.half_h) / 2
        for x in (int(np.floor(cx)), int(np.floor(cx)) + 1):
            for y in (int(np.floor(cy)), int(np.floor(cy)) + 1):
                if not (0 <= x < w and 0 <= y < h):
                    continue
                orders = index.by_tile.get((x, y))
                if not orders:
                    continue
                origin_sx, origin_sy = iso_geometry.tile_screen_origin(x, y, e, proj)
                local_x, local_y = sx - origin_sx, sy - origin_sy
                if not bool(iso_geometry.diamond_membership(local_x, local_y, proj.half_w, proj.half_h)):
                    continue
                for order in orders:
                    entry = index.entries[order]
                    # The swapped gate: this candidate diamond is only real
                    # if the unit's OWN tile sits at elevation e, since that
                    # is the single height its whole footprint is drawn at.
                    if int(elevations[entry.own_y, entry.own_x]) != e:
                        continue
                    key = _stepped_key(order, x, y)
                    if best_key is None or key > best_key:
                        best, best_key = (entry, (x, y)), key
    return best


def _pick_unit_sloped(
    index: UnitIndex,
    sx: int,
    sy: int,
    corner_rise: np.ndarray,
    proj: iso_geometry.IsoProjection,
) -> tuple[UnitEntry, tuple[int, int]] | None:
    """_pick_unit_stepped()'s shape, keyed on the SCREEN LATTICE instead of
    on integer elevation levels -- Track C5's Step 3.

    Returns (entry, (x, y)) with the same meaning as the stepped branch:
    the winner and the footprint tile it was covering when it won, since
    that tile is its paint position (see _stepped_key).

    Why a different parametrization. Stepped can enumerate candidates by
    elevation because every unit sits at an integer level; a Sloped unit
    sits at iso_geometry.unit_rise_px(), a per-unit PIXEL value with no
    small enumerable set. So this enumerates the two lattice coordinates
    directly instead:

      sx = origin_x + (x + y) * half_w + local_x,  local_x in [0, 2*half_w)
      sy = origin_y + (y - x) * half_h - rise + local_y, likewise for y

    The first confines s = x + y to exactly two values (a half-open window
    of width 2*half_w spans two multiples of half_w), and the second
    confines d = y - x to a window whose width is set by the map's own rise
    range. s and d must share parity for (x, y) to be integral, and the two
    s candidates have opposite parity, so each d yields exactly one tile --
    which is why the parity test below is exact arithmetic, not a filter
    that could drop a real candidate.

    **The d bound is derived, and generous on purpose.** Under
    SLOPE_CORNER_RULE = "max" a corner can sit a level above its own tile,
    and elev_step scales with the persisted elev_step_pct stop (this
    machine holds 50; the Appearance slider reaches 200), so an
    under-enumerating loop would return None on a real unit -- which reads
    as "nothing there" rather than as a bug. The bound therefore comes from
    corner_rise's own min/max rather than from proj.max_elev, and carries
    a spare row at each end; over-enumeration costs a dict lookup that
    misses, since every candidate is still confirmed by
    diamond_membership() against that unit's OWN rise.

    No ID plane and no new cache, deliberately: Track C4's pick plane stays
    terrain-only (render.composite_ids_rect_sloped's own note), which is
    what keeps this branch clear of that plane's cost re-measurement.
    """
    h1, w1 = corner_rise.shape
    w, h = w1 - 1, h1 - 1
    half_w, half_h = proj.half_w, proj.half_h
    rise_lo, rise_hi = int(corner_rise.min()), int(corner_rise.max())

    q = (sx - proj.origin_x) // half_w
    v = sy - proj.origin_y
    d_lo = (v + rise_lo - 2 * half_h) // half_h - 1
    d_hi = (v + rise_hi) // half_h + 1

    best: tuple[UnitEntry, tuple[int, int]] | None = None
    best_key: tuple[int, int, int] | None = None

    for d in range(d_lo, d_hi + 1):
        for s in (q - 1, q):
            if (s + d) % 2:
                continue
            x, y = (s - d) // 2, (s + d) // 2
            if not (0 <= x < w and 0 <= y < h):
                continue
            orders = index.by_tile.get((x, y))
            if not orders:
                continue
            origin_sx = proj.origin_x + (x + y) * half_w
            row = proj.origin_y + (y - x) * half_h
            for order in orders:
                entry = index.entries[order]
                # Per unit, not per tile: the whole footprint sits at ONE
                # height (the flat pad _paint_tile_and_units_sloped paints),
                # but two units covering this same tile can sit at two
                # different heights, so membership cannot be hoisted out.
                rise = unit_rise_px_for(entry, corner_rise)
                local_x, local_y = sx - origin_sx, sy - (row - rise)
                if not bool(iso_geometry.diamond_membership(local_x, local_y, half_w, half_h)):
                    continue
                key = _stepped_key(order, x, y)
                if best_key is None or key > best_key:
                    best, best_key = (entry, (x, y)), key
    return best


def unit_rise_px_for(entry: UnitEntry, corner_rise: np.ndarray) -> int:
    """The canvas-pixel rise one unit's whole footprint sits at in Sloped --
    the single place picking and highlighting resolve it, so neither can
    drift from render._paint_tile_and_units_sloped's own call.

    Reads the unit's OWN tile's corners at its OWN sub-tile fractions, the
    asymmetry-1 rule expressed in pixels (see this module's header): a
    multi-tile building is still a flat slab at one height, so every
    footprint diamond gets this same value rather than its own tile's."""
    unit = entry.unit
    return iso_geometry.unit_rise_px(
        corner_rise, entry.own_x, entry.own_y, unit.x - entry.own_x, unit.y - entry.own_y
    )


def pick_unit(
    index: UnitIndex,
    style: str,
    sx: int,
    sy: int,
    tile_px: int,
    tile_w: int,
    tile_h: int,
    elevations: np.ndarray | None = None,
    proj: iso_geometry.IsoProjection | None = None,
    corner_rise: np.ndarray | None = None,
    terrain_tile: tuple[int, int] | None = None,
) -> UnitEntry | None:
    """The topmost VISIBLE unit at canvas pixel (sx, sy), or None.

    Sloped needs corner_rise (its height field, in place of `elevations`)
    and terrain_tile, and returns None without them. terrain_tile is the
    already-resolved tile under (sx, sy) -- Sloped's terrain has no analytic
    inverse, so it is looked up through render_cache.SlopedChunkCache.pick_tile()
    by the caller and passed in, rather than handing this module a cache
    reference. That keeps unit_pick Qt-free and cache-free, as it has always
    been. Pass None for it exactly as Stepped's screen_to_tile() returning
    None means "no terrain here", i.e. the unit is unoccluded.

    Terrain occlusion, in this order (the ordering is the spec, not an
    implementation detail):

    1. Take the max-key unit among all candidates covering the pixel.
       Unit-vs-unit occlusion needs nothing more -- diamonds are painted
       opaquely, so last-painted wins.
    2. THEN resolve the terrain tile and discard the winner if the terrain's
       own key is strictly greater than the key of the winner's COVERING
       footprint tile -- the tile whose diamond the pixel actually fell in,
       which is where that diamond paints. (Comparing against the unit's own
       tile would be wrong now that a multi-tile building paints one diamond
       per footprint tile, each at its own depth.) Equal means the same
       tile, where the compositor paints terrain first and then that tile's
       units, so the unit wins.

    Residual, documented rather than hidden: screen_to_tile() returns None
    on skirt-face pixels (phase 2's own accepted behavior). There, the unit
    is treated as unoccluded. Sloped paints no skirts (see
    iso_geometry.corner_rise_px), so it has no such hole -- its own
    terrain_tile is None only genuinely off-map.
    """
    if style == "flat":
        return _pick_unit_flat(index, sx, sy, tile_px, tile_w, tile_h)
    if style == "sloped":
        if corner_rise is None or proj is None:
            return None
        found = _pick_unit_sloped(index, sx, sy, corner_rise, proj)
        terrain = terrain_tile
    elif style == "stepped":
        if elevations is None or proj is None:
            return None
        found = _pick_unit_stepped(index, sx, sy, elevations, proj)
        terrain = iso_geometry.screen_to_tile(sx, sy, elevations, proj)
    else:
        raise ValueError(f"unknown terrain style {style!r}")

    if found is None:
        return None
    winner, (cover_x, cover_y) = found

    if terrain is None:
        return winner
    tx, ty = terrain
    if (ty - tx, tx) > (cover_y - cover_x, cover_x):
        return None
    return winner


def unit_polygons(
    entry: UnitEntry,
    style: str,
    tile_px: int,
    tile_w: int,
    tile_h: int,
    elevations: np.ndarray | None = None,
    proj: iso_geometry.IsoProjection | None = None,
    corner_rise: np.ndarray | None = None,
) -> list[list[tuple[float, float]]] | None:
    """The unit's on-screen highlight, as a list of polygons, each a list of
    (x, y) points. Plain tuples rather than QPolygonF so this module stays
    Qt-free; the viewer converts.

    - Flat: one axis-aligned rect, matching render._draw_unit() exactly.
    - Stepped: one diamond per footprint tile, ALL at the unit's own tile's
      elevation, matching render._draw_unit_iso(). This is asymmetry 2 --
      using each footprint tile's own terrain elevation here would look
      right on flat ground and drift apart on a slope.
    - Sloped: the same body with the height expression swapped for
      unit_rise_px_for() -- one diamond per footprint tile, all at the
      unit's own PIXEL rise, matching render._draw_unit_sloped(). Needs
      corner_rise; returns None without it.

    Sloped's diamonds are plain diamond_points(), NOT the warped quad its
    terrain tile paints, because the MARKER is a plain diamond too (see
    render._draw_unit_sloped). The highlight therefore matches what is
    drawn, which is this function's contract -- that the marker itself does
    not conform to the warped tile footprint is a separate, logged shape
    defect, and fixing it here would make the outline disagree with the
    pixels.
    """
    bounds = render.unit_tile_bounds(entry.unit, tile_w, tile_h)
    if bounds is None:
        return None
    tile_x0, tile_x1, tile_y0, tile_y1 = bounds

    if style == "flat":
        x0, y0 = tile_x0 * tile_px, tile_y0 * tile_px
        x1, y1 = tile_x1 * tile_px, tile_y1 * tile_px
        return [[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]]

    if style == "sloped":
        if corner_rise is None or proj is None:
            return None
        rise = unit_rise_px_for(entry, corner_rise)
        elevation = 0
    elif style == "stepped":
        if elevations is None or proj is None:
            return None
        rise = 0
        elevation = int(elevations[entry.own_y, entry.own_x])
    else:
        raise ValueError(f"unknown terrain style {style!r}")

    half_w, half_h = proj.half_w, proj.half_h
    polygons = []
    for ty in range(tile_y0, tile_y1):
        for tx in range(tile_x0, tile_x1):
            ox, oy = iso_geometry.tile_screen_origin(tx, ty, elevation, proj)
            polygons.append(diamond_points(ox, oy - rise, half_w, half_h))
    return polygons


def units_in_rect(index: UnitIndex, tx0: int, ty0: int, tx1: int, ty1: int) -> list[UnitEntry]:
    """Every entry with at least one footprint tile inside the half-open
    tile rectangle [tx0, tx1) x [ty0, ty1) -- phase 3.5b's b2.1, the
    marquee's own query.

    Built on by_tile rather than a scan over index.entries, per this
    module's own "why an index rather than a scan" rationale: cost is
    proportional to the rectangle's tile area, not the file's total unit
    count. Qt-free and style-agnostic on purpose -- turning an arbitrary
    ON-SCREEN marquee rectangle into a TILE rectangle is real per-style
    geometry (see MapView._pick_tile's own docstring on why Stepped/Sloped
    have no closed-form inverse for an arbitrary point, let alone a
    rectangle), and that conversion belongs in the Qt view layer, not here.

    Already filter-respecting for free: by_tile only ever holds entries
    build_index() let through, so a filtered-out unit (a hidden tree, say)
    was never inserted and cannot be returned.

    Dedups by entry.order, since by_tile keys on every FOOTPRINT tile --
    a multi-tile building covering more than one tile inside the rectangle
    must still appear exactly once.
    """
    seen: set[int] = set()
    result: list[UnitEntry] = []
    for ty in range(ty0, ty1):
        for tx in range(tx0, tx1):
            for order in index.by_tile.get((tx, ty), ()):
                if order not in seen:
                    seen.add(order)
                    result.append(index.entries[order])
    return result


def diamond_points(ox: int, oy: int, half_w: int, half_h: int) -> list[tuple[float, float]]:
    """The 4-point diamond for a tile box whose top-left is (ox, oy).

    Factored out (rather than duplicated into MapView._tile_polygon) so tile
    highlights and unit highlights can never disagree about diamond shape.
    """
    return [
        (ox + half_w, oy),
        (ox + 2 * half_w, oy + half_h),
        (ox + half_w, oy + 2 * half_h),
        (ox, oy + half_h),
    ]
