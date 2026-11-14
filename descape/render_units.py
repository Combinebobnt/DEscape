"""The render's unit geometry: spans, anchors, tile bounds, stored rotation, per-generation memos, wall overrides, colour and marks."""

from __future__ import annotations

import copy
import math
import weakref

import numpy as np

from descape import (
    game_dataset,
    games,
    iso_geometry,
    unit_rotation,
    unit_sprites,
)
from descape.render_paint import _clipped_paint_rgba, _clipped_paint_solid, _record_ink
from descape.scenario_io import LoadedScenario
from descape.terrain_palette import (
    BUILDING_TILE_OFFSETS,
    BUILDING_TILE_SPANS,
    RESOURCE_COLORS,
    TREE_COLOR,
    TREE_UNIT_IDS,
    tile_span,
)
from descape.unit_filter import UnitFilter

NON_BUILDING_SPAN = (1, 1)


def player_art_for(scenario, player_id: int) -> tuple[int | None, int, int | None] | None:
    """`player_id`'s (art_civ, age, tech_civ) building art (GH #48, civ_art), or
    None for today's Gaia-table art: a duck-typed fake scenario carries no
    player_art, and a file shown with the other game's data (GH #67) indexes
    civs the active art tables do not have."""
    art = getattr(scenario, "player_art", None)
    if art is None or not 0 <= player_id < len(art):
        return None
    if games.for_scenario(scenario).id != game_dataset.active():
        return None
    return art[player_id]


def _span_start(coord: float, span: int) -> int:
    """Lowest tile of a `span`-wide footprint anchored on `coord`.

    Two branches, because only one of them has measured data behind it.

    A span-1 axis anchors at int(coord) -- the unit's own tile, by definition,
    and what every unit did before spans existed. This branch carries all
    non-buildings, whose coordinates are arbitrary floats: 291,148 axis values
    across the example corpus have 15,370 distinct fractional parts.

    A span > 1 axis works in half-tiles, which is exact for buildings, whose
    coordinates are always exact multiples of 0.5 (measured: 27,320 axis
    values, fractional part 0.0 or 0.5, nothing else). Rounding half up rather
    than flooring is what keeps the own-tile invariant on legal off-parity
    placements. Deep-water fish (GH #178) and class-14 doodads (TASK-134) are
    not buildings: 21 of 1049 corpus fish and 24 of 205 doodads sit off the
    half-tile grid and are accepted as quantized, and the own-tile invariant
    holds for any float.

    Per axis, not per unit: a gate segment is (4, 1) and takes both branches.
    Applying the half-tile branch to a span-1 axis would move any unit whose
    fractional part exceeds 0.75 a whole tile (5,671 real corpus values do).
    """
    if span <= 1:
        return int(coord)
    return (round(coord * 2) - span + 1) // 2


def _sprite_axis_centre(coord: float, span: int) -> float:
    """Where a footprint actually paints on one axis, in continuous tile
    coords -- the argument iso_geometry.map_point_to_screen() takes.

    Two branches, matching _span_start's own split and for the same reason:
    only one of them has measured data behind it.

    A span-1 axis passes `coord` through untouched. Those are the
    non-buildings, whose coordinates are arbitrary floats (291,148 corpus axis
    values, 15,370 distinct fractional parts), and quantizing them to
    `int(coord) + 0.5` is exactly what used to make sub-tile position
    unreachable on screen: 88.0% of corpus units do sit at `.5`, but the ~9.2%
    tail was drawn up to half a tile from where its file says it stands, and
    the inspector's 0.01 spinboxes and the 0.1 arrow nudge both moved a unit
    the render never showed moving.

    A span > 1 axis keeps `_span_start(coord, span) + span / 2`, deliberately.
    Building coordinates are exact multiples of 0.5 (measured: 27,320 axis
    values, nothing else), and a building is drawn as a half-tile-quantized
    slab on purpose -- see _span_start. A class-14 doodad's footprint is
    quantized the same way, but its sprite is not: _resolve_unit_sprite keeps
    the stored point for it (_sprite_keeps_own_point, TASK-134), and for the
    visible static objects with a span (shipwrecks, rubble, TASK-277).

    Per axis, not per unit: a gate segment is (4, 1) and takes both branches.
    """
    if span <= 1:
        return coord
    return _span_start(coord, span) + span / 2


def unit_footprint_centre(unit) -> tuple[float, float]:
    """The continuous map point a unit's footprint is drawn centred on:
    `_sprite_axis_centre` per axis, the expression the sprite anchor uses.

    For anything outside this module that must sit on the drawn footprint
    (the range ring, GH #200), so the rounding stays in `_span_start` alone.
    The stored coordinate on a span-1 axis and for a building at `tile +
    span/2`; half a tile from it for a building stored on the other parity.
    Unclamped, so a building hanging off a map edge keeps its true centre."""
    span_x, span_y = tile_span(unit.unit_const, NON_BUILDING_SPAN)
    return _sprite_axis_centre(unit.x, span_x), _sprite_axis_centre(unit.y, span_y)


def unit_paint_offset(unit) -> tuple[float, float]:
    """(dx, dy) in CONTINUOUS TILE UNITS between where a unit's coloured mark
    paints now and the integer-tile position it painted at before Stage 1 of
    off-grid placement -- the single place that shift is derived, so the three
    painters, the three pick loops, the highlight outline and the dirty-bbox
    cannot drift apart.

    Zero on a span > 1 axis, by construction: `_sprite_axis_centre` keeps that
    axis half-tile-quantized, so the difference is exactly zero and every
    building's slab is byte-identical to before. `coord - int(coord) - 0.5` on
    a span-1 axis, which is 0.0 for the 88.0% of corpus units sitting at
    exactly `.5` -- so the centred case is byte-identical too, and only the
    ~9.2% off-centre tail actually moves.

    Per axis, not per unit: a gate segment is (4, 1), so its x stays quantized
    while its y follows the file.

    Bounded by |dx| < 0.5 and |dy| < 0.5. Every candidate-enumeration widening
    downstream (unit_pick's pick loops, _unit_screen_bbox_iso) is sized off
    that bound, so an under-enumerating loop -- which returns None and reads
    as "nothing there" rather than as a bug -- stays impossible rather than
    merely unlikely.
    """
    span_x, span_y = tile_span(unit.unit_const, NON_BUILDING_SPAN)
    dx = unit.x - int(unit.x) - 0.5 if span_x <= 1 else 0.0
    dy = unit.y - int(unit.y) - 0.5 if span_y <= 1 else 0.0
    return dx, dy


def span_anchor(tile_x: int, tile_y: int, span_x: int, span_y: int) -> tuple[float, float]:
    """The (x, y) that anchors a `(span_x, span_y)` footprint on `(tile_x,
    tile_y)` -- the inverse of `_span_start` per axis, used by the Cliff tool
    (Track B Stage 1) to place a new object under the clicked tile without
    re-deriving A4's per-suffix parity table by hand (that table has a
    counterexample: Short Marble Cliff 1 is 3x2, not every family's 3x3).

    `tile + span/2` solves `_span_start(coord, span) == tile` for every span,
    including span == 1 (`_span_start`'s own other branch): a half-integer
    result for an odd span, a whole one for an even span -- exactly the
    sub-tile parity every corpus cliff placement already carries.
    """
    return tile_x + span_x / 2, tile_y + span_y / 2


def snapped_anchor(
    tile: tuple[int, int],
    point: tuple[float, float] | None,
    span: tuple[int, int],
    map_w: int,
    map_h: int,
) -> tuple[float, float]:
    """Snapped placement's (x, y) for a `span` footprint picked at `tile`, with
    `point` the continuous cursor point or None (TASK-193, GH #206).

    Per axis, on the game's grid and centred on the cursor: an odd span sits
    at `tile + 0.5`; an even span at the whole number nearest the cursor,
    `floor(point + 0.5)` (a tie rounds up), or `tile + 1` with no point. An
    even anchor is clamped to `[span/2, size - span/2]`, so it stays whole and
    the whole footprint stays on the map, as a group move keeps it.
    """
    out = []
    for i, size in enumerate((map_w, map_h)):
        if span[i] % 2:
            out.append(tile[i] + 0.5)
            continue
        anchor = tile[i] + 1 if point is None else math.floor(point[i] + 0.5)
        half = span[i] // 2
        out.append(float(max(half, min(anchor, size - half))))
    return out[0], out[1]


def created_unit_point(unit_const: int, tile_x: int, tile_y: int) -> tuple[float, float]:
    """The (x, y) the game gives a unit a trigger creates on `(tile_x,
    tile_y)` (Create Object, Place Foundation), for its map ghost (GH #369).

    Per axis, buildings only: an even span centres on the tile's low corner,
    the whole number `tile`, so a 4x4 castle covers `tile-2..tile+1`; an odd
    span stays at `tile + 0.5`. Measured in game on 1.59: castle (Place
    Foundation), house, barracks and a (4, 1) gate each spawned inside a
    palisade ring round exactly that footprint. `tile + 0.5` for an even
    span would round half up in `_span_start` and draw it one tile high on
    both axes. Every non-building keeps `tile + 0.5`: unmeasured. The other
    even shapes (diagonal gates, 8x8, mixed 3x2, 2x1) and Return of Rome
    follow the per-axis rule unmeasured.
    """
    if unit_const not in BUILDING_TILE_SPANS:
        return tile_x + 0.5, tile_y + 0.5
    span_x, span_y = tile_span(unit_const, NON_BUILDING_SPAN)
    return (
        float(tile_x) if span_x % 2 == 0 else tile_x + 0.5,
        float(tile_y) if span_y % 2 == 0 else tile_y + 0.5,
    )


def unit_tile_bounds(unit, tile_w: int, tile_h: int) -> tuple[int, int, int, int] | None:
    """(tile_x0, tile_x1, tile_y0, tile_y1) -- the tile-space bounding box
    _draw_unit() would paint for this unit, half-open like Python ranges.
    None if the unit is off-map. Split out from _draw_unit() so
    refresh_units_over() can cheaply test overlap with a dirty-tile set
    without touching img.

    **Size comes from the .dat alone, never from position.** The engine and
    the in-game editor both allow a building to sit on the coordinate parity
    its size does not "expect" (46 of 158 Mills and 32 of 254 Castles in the
    example corpus do), so deriving the span from where a unit sits would
    render every off-grid House 3x3 -- exactly the bug this replaced. Position
    only picks the anchor; an off-grid building is drawn at its true size,
    quantized to the nearer tile.

    Which .dat field, though, is per-table: a building's span comes from its
    `clearance_size`, a cliff's from its `collision_size` (every cliff const
    reports the same useless flat clearance) -- see
    tools/gen_unit_render_data.py's "object_spans" docs. tile_span() is
    the merged lookup; the distinction lives in the generator, not here.

    **Invariant, relied on downstream: (int(unit.x), int(unit.y)) is always
    inside the returned bounds.** _unit_iso_footprint reads elevations[py, px]
    for the whole slab, unit_pick's stepped gate keys on own_x/own_y, and
    _unit_screen_bbox_iso indexes tile_x1 - 1, which is only safe while the
    clamped range is non-empty. Don't change the anchor rule without
    re-checking all three."""
    px, py = int(unit.x), int(unit.y)
    if not (0 <= px < tile_w and 0 <= py < tile_h):
        return None
    span_x, span_y = tile_span(unit.unit_const, NON_BUILDING_SPAN)
    x0, y0 = _span_start(unit.x, span_x), _span_start(unit.y, span_y)
    tile_x0, tile_x1 = max(0, x0), min(tile_w, x0 + span_x)
    tile_y0, tile_y1 = max(0, y0), min(tile_h, y0 + span_y)
    return tile_x0, tile_x1, tile_y0, tile_y1


def unit_at(unit, x: float, y: float):
    """A read-only stand-in for `unit` sitting at (x, y) -- the mid-drag move
    preview's whole trick, and the reason that feature needs no
    position-taking sibling of unit_tile_bounds/unit_polygons/the sprite
    anchor expression.

    A shallow copy of the real object rather than a hand-rolled duck type.
    Every geometry function here already derives everything from `unit.x`/
    `unit.y` plus .dat lookups keyed on `unit_const`, so a copy with two
    fields changed runs the SAME code the landed move will, leaving no second
    copy of the arithmetic to drift. A hand-rolled stand-in would instead have
    to enumerate every attribute those walks read (unit_filter.matches,
    _unit_color, stored_rotation, tile_span, unit_paint_offset, ...) and would
    go missing one.

    **Do not translate a computed box by a tile delta instead.**
    unit_tile_bounds CLAMPS to the map edges, so a translated clamped box is
    wrong for any unit dragged near an edge; recomputing from a stand-in at
    the destination is not.

    Never hand the result to a write path: UnitEditModel._locate() cannot find
    it in any player's unit list, so it raises rather than silently editing a
    copy nothing will ever save."""
    proxy = copy.copy(unit)
    proxy.x, proxy.y = x, y
    return proxy


def unit_mark_color(unit, player_color: tuple[int, int, int]) -> tuple[int, int, int]:
    """The colour a unit's coloured mark paints in -- _unit_color()'s public
    entry point, for the mid-drag preview's mark fallback. Routed through the
    same function the three painters use rather than re-deriving the
    tree/resource/player split, so a ghost can never preview a colour the
    landed unit will not have."""
    return _unit_color(unit, player_color)


def unit_occupied_tiles(unit, tile_w: int, tile_h: int) -> list[tuple[int, int]] | None:
    """Every on-map tile this unit's footprint actually paints, as absolute
    (tx, ty) pairs -- the plan-of-record Part 2 replacement for enumerating
    unit_tile_bounds()'s rect wholesale. Ordinary units/buildings get exactly
    that rect's tiles, in the same (ty outer, tx inner) order the old inline
    double loops used; the 36 BUILDING_TILE_OFFSETS consts (diagonal gates)
    get their real sparse 6-tile set instead. None if off-map, same contract
    as unit_tile_bounds().

    Stepped/Sloped-only by convention of most of its callers (_units_by_tile,
    sprite_draws_by_anchor, unit_pick.unit_polygons' non-flat branch) --
    Flat's _draw_unit()/_flat_unit_draws() and unit_pick.build_index() keep
    reading unit_tile_bounds()'s rect directly, by design (a deliberate
    Flat carve-out). wall_variant_rotation_overrides() is the one exception:
    it builds the connector tile set that all three sprite-rendering paths
    share, Flat included, because a wall's real shape can't depend on which
    render mode is drawing it."""
    return occupied_tiles_for(unit.unit_const, unit.x, unit.y, tile_w, tile_h)


def occupied_tiles_for(
    unit_const: int, x: float, y: float, tile_w: int, tile_h: int
) -> list[tuple[int, int]] | None:
    """unit_occupied_tiles() for a const and a position that no Unit object
    carries yet -- map mirroring's Stage 2 asks where an image WOULD sit
    before it exists. Same contract, same None-when-off-map answer."""
    px, py = int(x), int(y)
    if not (0 <= px < tile_w and 0 <= py < tile_h):
        return None
    span_x, span_y = tile_span(unit_const, NON_BUILDING_SPAN)
    x0, y0 = _span_start(x, span_x), _span_start(y, span_y)
    offsets = BUILDING_TILE_OFFSETS.get(unit_const)
    if offsets is None:
        return [
            (tx, ty)
            for ty in range(max(0, y0), min(tile_h, y0 + span_y))
            for tx in range(max(0, x0), min(tile_w, x0 + span_x))
        ]
    return [
        (x0 + ox, y0 + oy)
        for ox, oy in offsets
        if 0 <= x0 + ox < tile_w and 0 <= y0 + oy < tile_h
    ]


def unit_occupies_tile(unit, tx: int, ty: int) -> bool:
    """True unless (tx, ty) is a bbox tile a diagonal gate's real (sparse)
    footprint excludes -- the predicate form of unit_occupied_tiles(), for a
    caller that already has a candidate tile from a shared, still-full-rect
    structure (unit_pick's by_tile) and only needs to narrow it further.
    Only the 36 BUILDING_TILE_OFFSETS consts can ever answer False; every
    other unit occupies its whole unit_tile_bounds() rect."""
    offsets = BUILDING_TILE_OFFSETS.get(unit.unit_const)
    if offsets is None:
        return True
    span_x, span_y = tile_span(unit.unit_const, NON_BUILDING_SPAN)
    x0, y0 = _span_start(unit.x, span_x), _span_start(unit.y, span_y)
    return (tx - x0, ty - y0) in offsets


def stored_rotation(player_id: int, unit) -> float:
    """The rotation value a unit's sprite lookup should start from -- the
    single place the GAIA rule lives, for all four independent per-unit walks
    (_flat_icon_layer_sliced, sprite_draws_by_anchor_sliced,
    _flat_icon_for_unit, and wall_variant_rotation_overrides' own candidate
    scan). Any connectivity-derived override is applied by the caller, on top
    of this.

    **GAIA's `rotation` is often not an angle** -- AGENTS.md's hard rule: for ~65% of
    GAIA objects it is a graphic-VARIANT index (trees, doodads, cliffs), with
    integer values well outside [0, 2*pi). Feeding one to angle_index() gets a
    scrambled frame or an out-of-file index, so GAIA resolved at 0.0
    unconditionally until this function existed.

    The narrowing (2026-09-05 cliffs plan, Track A1; widened by the 2026-09-06
    Tier B plan): a GAIA const whose graphic is KNOWN to be variant-indexed
    keeps its stored value, because for those consts the field is meaningful
    and rotation_is_variant() routes it to variant_index() rather than
    angle_index(). Tier B made that known-set generated (unit_graphic_map.json's
    `rotation_is_variant` field, from the .dat's own unit.type != 70) rather
    than the two hand-kept frozensets Track A2 shipped with, which is what
    fixed trees drawing variant 0 in every map.

    GH #217 widened it to the ANGLE consts (unit_rotation.rotation_is_angle(),
    the predicate set_rotation() and Rotate use): a GAIA creatable's rotation
    is a real facing, and the game draws it, checked in the official editor
    and in a scenario DEscape wrote. Only a GAIA const that is neither keeps
    zeroing, the safe "we don't know what this integer means" answer.

    A1 and A2 are one change, not two: un-zeroing a cliff without cliffs in the
    variant set sends rotation 6.0 through angle_index() to index 24, past the
    24-frame file, and 246 corpus placements degrade to coloured marks."""
    const = unit.unit_const
    if player_id == 0 and not (unit_sprites.rotation_is_variant(const) or unit_rotation.rotation_is_angle(const)):
        return 0.0
    return float(unit.rotation)


# Batch D's D1b: one WeakKeyDictionary[scenario] -> (gen, value) per
# memoized function below, so neither can be invalidated by the other's
# writes. `scenario.unit_gen` is the sole invalidation key (see
# LoadedScenario.unit_gen's own docstring for exactly what bumps it).
_ANCHOR_TILES_MEMO: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_WALL_OVERRIDES_MEMO: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_OWN_TILE_INDEX_MEMO: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_GARRISON_HOSTS_MEMO: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _unit_gen_cached(memo: weakref.WeakKeyDictionary, scenario, compute):
    """Returns compute()'s result, cached in `memo` against `scenario`'s
    current unit_gen. getattr, not attribute access: both of this batch's
    memoized functions are called with duck-typed scenario stand-ins in
    tests (e.g. tests/test_wall_connectivity.py's), which carry no unit_gen
    -- those must always compute fresh and must never touch `memo` at all,
    which is exactly what returning before the dict lookup below gives."""
    gen = getattr(scenario, "unit_gen", None)
    if gen is None:
        return compute()
    cached = memo.get(scenario)
    if cached is not None and cached[0] == gen:
        return cached[1]
    value = compute()
    memo[scenario] = (gen, value)
    return value


def anchor_tiles(scenario) -> set[tuple[int, int]]:
    """Every unit's own `(int(u.x), int(u.y))` tile -- the set `_dirty_
    screen_bbox`'s with_sprites block pads sprite reach around (see that
    block's own comment for why the filter is deliberately ignored: a
    filtered-out unit's tile padding a bit further is safe over-inclusion).
    Pure function of scenario.unit_manager.units, memoized per D1b above."""

    def _compute() -> set[tuple[int, int]]:
        return {(int(u.x), int(u.y)) for units in scenario.unit_manager.units for u in units}

    return _unit_gen_cached(_ANCHOR_TILES_MEMO, scenario, _compute)


def unit_own_tile_index(scenario) -> dict[tuple[int, int], list[tuple[int, int, object]]]:
    """Own tile `(int(u.x), int(u.y))` -> every unit there as `(player_id,
    i, unit)`, i being its position in its player's list (the key
    _resolve_unit_sprite() and the wall overrides need). Over ALL units,
    ignoring any filter, like anchor_tiles() above and memoized the same way.
    Lets an elevation edit find the units whose placement reads a changed
    height without walking every unit."""

    def _compute() -> dict[tuple[int, int], list[tuple[int, int, object]]]:
        out: dict[tuple[int, int], list[tuple[int, int, object]]] = {}
        for player_id, units in enumerate(scenario.unit_manager.units):
            for i, u in enumerate(units):
                out.setdefault((int(u.x), int(u.y)), []).append((player_id, i, u))
        return out

    return _unit_gen_cached(_OWN_TILE_INDEX_MEMO, scenario, _compute)


def garrison_host_ids(scenario) -> frozenset[int]:
    """Every reference_id some unit is garrisoned in (GH #203): the hosts
    that draw a garrison flag. An occupant is UnitFilter.matches()' own rule,
    a garrisoned_in_id other than -1 and the unit's own reference_id. Over ALL
    units, blind to every filter, since the flag is the host's state, and
    memoized like anchor_tiles(). An id no live unit carries draws nothing."""

    def _compute() -> frozenset[int]:
        hosts = set()
        for units in scenario.unit_manager.units:
            for u in units:
                host = getattr(u, "garrisoned_in_id", -1)
                if host != -1 and host != getattr(u, "reference_id", None):
                    hosts.add(host)
        return frozenset(hosts)

    return _unit_gen_cached(_GARRISON_HOSTS_MEMO, scenario, _compute)


def wall_variant_rotation_overrides(scenario) -> dict[tuple[int, int], float]:
    """Memoized wrapper (Batch D's D1b) around _wall_variant_rotation_
    overrides_uncached() below -- signature unchanged, so
    tests/test_wall_connectivity.py's `"unit_filter" not in inspect.
    signature(wall_variant_rotation_overrides)` assertion still holds."""
    return _wall_overrides_and_verdict(scenario)[0]


def walls_file_is_radian(scenario) -> bool:
    """The file-wide verdict the overrides were drawn under
    (unit_sprites.file_is_radian() over every eligible wall), a by-product of
    the same memoized walk. render_cache's wall_verdict_flipped() compares it
    with the verdict its unit sources were built against (TASK-583)."""
    return _wall_overrides_and_verdict(scenario)[1]


def _wall_overrides_and_verdict(scenario) -> tuple[dict[tuple[int, int], float], bool]:
    return _unit_gen_cached(
        _WALL_OVERRIDES_MEMO, scenario, lambda: _wall_overrides_and_verdict_uncached(scenario)
    )


def _wall_variant_rotation_overrides_uncached(scenario) -> dict[tuple[int, int], float]:
    """The overrides alone, uncached: tools/scan_wall_rotation.py and tools/bench_sprite_memo.py read this."""
    return _wall_overrides_and_verdict_uncached(scenario)[0]


def _wall_overrides_and_verdict_uncached(scenario) -> tuple[dict[tuple[int, int], float], bool]:
    """Per-(player_id, index-in-that-player's-unit-list) derived rotation for
    a wall/gate whose real shape isn't recoverable from its own stored
    `rotation` -- see tools/scan_wall_rotation.py, which measures this
    function against examples/ as the regression guard.

    Built over ALL units, ignoring unit_filter -- a wall's real shape does not
    depend on which players are currently shown, and hiding player 2 must not
    reshape player 1's wall. That also makes this a pure function of scenario
    unit data, which is what lets sprite_draws_by_anchor_sliced(),
    _flat_icon_layer_sliced() and _flat_icon_for_unit() share one
    implementation despite each keeping its own independent per-unit walk.

    Keyed by (player_id, i) where i is the unit's own position in
    scenario.unit_manager.units[player_id] -- NOT _flat_icon_layer_sliced()'s
    `row` counter, which only advances past units unit_filter keeps. Keying on
    `row` would silently misassign every wall past the first filtered-out
    unit whenever a filter is active.

    **Precondition 3 (real index vs. radian-encoded) is decided per FILE, not
    per unit.** unit_sprites.rotation_variant_eligible() only covers
    preconditions 1-2 (variant graphic, angle_count == 5); a literal-looking
    0.0 rotation is otherwise ambiguous (index 0 under both stored
    conventions), and measured on examples/ (tools/scan_wall_rotation.py),
    45.6% (241/529) of the 0.0-rotation walls in a file that also carries a
    real radian-encoded wall sit on a shape-changing neighbour mask -- far
    from a rounding error. So a literal integer is only trusted in a file
    that never uses the radian convention at all; inside one that does, every
    wall draws its 8-bit neighbour derivation, diagonal runs included.

    **A lone wall or a run end draws as the tower in every file**: AoE2:DE
    draws a wall with at most one neighbour as a tower whatever its stored
    index (a wall moved out of a run by an older editor keeps its run index).
    In an integer file that is the only override, and it is added only where
    the stored index is not already the tower, so a game-saved file keeps an
    empty dict. The per-wall rule is unit_sprites.wall_drawn_shape(); this is
    draw-only, and no write path reads it.

    Returns (overrides, file_is_radian): the verdict rides along so a cache
    can tell when an edit flipped it without a second walk."""
    mm = scenario.map_manager
    tile_w, tile_h = mm.map_width, mm.map_height

    connector_consts = unit_sprites.wall_connector_consts()
    connector_tiles: set[tuple[int, int]] = set()
    for units in scenario.unit_manager.units:
        for unit in units:
            if unit.unit_const in connector_consts:
                occupied = unit_occupied_tiles(unit, tile_w, tile_h)
                if occupied is not None:
                    connector_tiles.update(occupied)

    candidates: list[tuple[int, int, object, float]] = []
    for player_id, units in enumerate(scenario.unit_manager.units):
        for i, unit in enumerate(units):
            if not unit_sprites.rotation_variant_eligible(unit.unit_const):
                continue
            # Same seam the three drawing walks use, so the per-file
            # radian/integer classification below sees exactly the values they
            # would resolve. Every eligible const is variant-indexed by
            # definition, so a GAIA-owned wall now contributes its real stored
            # index here instead of a blanket 0.0 -- which is the point: it was
            # the zeroing, not the wall itself, that made it look literal.
            rotation = stored_rotation(player_id, unit)
            candidates.append((player_id, i, unit, rotation))

    # rotation_variant_eligible() guarantees angle_count == 5 for every
    # candidate, so 5 is safe to hardcode here rather than re-deriving it from
    # graphic_map() -- this module has no other reason to reach into that
    # table directly.
    file_is_radian = unit_sprites.file_is_radian([rotation for *_, rotation in candidates], 5)

    overrides: dict[tuple[int, int], float] = {}
    for player_id, i, unit, rotation in candidates:
        bounds = unit_tile_bounds(unit, tile_w, tile_h)
        if bounds is None:
            continue
        tx, ty = bounds[0], bounds[2]
        mask8 = unit_sprites.neighbour_mask(tx, ty, connector_tiles, diagonals=True)
        # Through the module attribute: tools/release_render_diff.py patches it.
        drawn = unit_sprites.wall_drawn_shape(mask8, unit_sprites.variant_index(rotation, 5), file_is_radian)
        if drawn is not None:
            overrides[(player_id, i)] = float(drawn)
    return overrides, file_is_radian


def _unit_color(unit, player_color: tuple[int, int, int]) -> tuple[int, int, int]:
    """The color rule shared by both Flat's _draw_unit() and Stepped's
    _draw_unit_iso() (Phase 5 reuses this rather than re-deriving it):
    trees are always dark green, non-building resource/decoration objects
    get their real minimap color, everything else (including buildings) is
    colored by owning player -- all regardless of owner, matching AoE2's own
    minimap for the first two categories."""
    is_building = unit.unit_const in BUILDING_TILE_SPANS
    if unit.unit_const in TREE_UNIT_IDS:
        return TREE_COLOR
    if not is_building and unit.unit_const in RESOURCE_COLORS:
        return RESOURCE_COLORS[unit.unit_const]
    return player_color


def _draw_unit(
    img: np.ndarray,
    unit,
    player_color,
    tile_w: int,
    tile_h: int,
    tile_px: int,
    icon: unit_sprites.SpriteDraw | None = None,
) -> None:
    """Draws one unit's mark into img, in place -- a colored dot, sized to
    its real footprint and colored per _unit_color(), a 1-tile dot for
    everything else. Shared by overlay_units() (every unit, full map) and
    refresh_units_over() (only units overlapping a dirty-tile set, after an
    edit) so the two paths can never draw a unit differently.

    icon (P3-g7), if given, REPLACES that mark with a footprint-fitted .sld
    icon centred in the same rect -- see composite_rect_flat()'s docstring for
    why replace rather than overlay.

    The rect is shifted by unit_paint_offset() (off-grid placement, Stage 1), so a
    span-1 unit's mark sits where its file says it stands rather than on its
    tile. Flat is the trivial projection: one continuous tile unit is exactly
    tile_px pixels. The shift is 0.0 for every building and for every unit at
    an exact `.5`, so this is byte-identical for all but the off-centre tail
    -- and it is clipped to the canvas rather than to the tile grid, since a
    shifted mark on an edge tile can now reach past the map."""
    bounds = unit_tile_bounds(unit, tile_w, tile_h)
    if bounds is None:
        return
    tile_x0, tile_x1, tile_y0, tile_y1 = bounds
    color = _unit_color(unit, player_color)

    # Tile-space bounds scaled to pixel space -- render_terrain() renders at
    # tile_px pixels per tile, not one pixel per tile. tile_px here must
    # match whatever render_terrain() used for this same scenario -- true as
    # long as both derive it via tile_pixels_for_map() rather than a
    # passed-in value, since it's a pure function of the map's own
    # (unchanging) dimensions.
    dx, dy = unit_paint_offset(unit)
    off_x, off_y = round(dx * tile_px), round(dy * tile_px)
    y0, y1 = tile_y0 * tile_px + off_y, tile_y1 * tile_px + off_y
    x0, x1 = tile_x0 * tile_px + off_x, tile_x1 * tile_px + off_x
    if icon is not None:
        # Centred on the UNCLIPPED rect: _clipped_paint_rgba does its own
        # bounds handling, and clipping first would slide the icon sideways.
        ih, iw = icon.rgba.shape[:2]
        _clipped_paint_rgba(img, y0 + (y1 - y0 - ih) // 2, x0 + (x1 - x0 - iw) // 2, icon.rgba)
        return
    h, w = img.shape[:2]
    y0, y1 = max(0, min(y0, h)), max(0, min(y1, h))
    x0, x1 = max(0, min(x0, w)), max(0, min(x1, w))
    if y0 < y1 and x0 < x1:
        img[y0:y1, x0:x1] = color


def _unit_iso_footprint(
    unit, tile_w: int, tile_h: int, elevations: np.ndarray
) -> tuple[int, int, int, int, int] | None:
    """(tile_x0, tile_x1, tile_y0, tile_y1, elevation) for one unit's iso
    footprint -- the same tile-space bounds unit_tile_bounds() computes for
    Flat mode, plus the single elevation every footprint tile is drawn at:
    elevations[py, px], the
    unit's OWN floored tile's elevation -- not each footprint tile's own
    individual terrain elevation. That's Phase 5's whole plan for
    multi-tile buildings ("center tile's elevation for the whole
    footprint") -- a flat slab, deliberately not per-corner-interpolated
    (real sloped ramps are Phase 6, out of scope here). None if the unit is
    off-map.

    elevations[py, px] is safe because unit_tile_bounds() guarantees the
    unit's own tile lies inside the bounds it returns -- see its docstring.
    An even-span building has no centre tile at all, so "its own tile" is the
    right way to say this, not "its center"."""
    bounds = unit_tile_bounds(unit, tile_w, tile_h)
    if bounds is None:
        return None
    px, py = int(unit.x), int(unit.y)
    return (*bounds, int(elevations[py, px]))


def _draw_unit_iso(
    img: np.ndarray,
    unit,
    color: tuple[int, int, int],
    tx: int,
    ty: int,
    tile_px: int,
    proj: iso_geometry.IsoProjection,
    elevations: np.ndarray,
    offset: tuple[int, int] = (0, 0),
    ink: np.ndarray | None = None,
    ink_rise: int = 0,
) -> None:
    """Stepped mode's counterpart to _draw_unit() (Phase 5): paints ONE of a
    unit's footprint diamonds -- the one covering tile (tx, ty) -- filled,
    fitting naturally into the same diamond shape terrain tiles paint in
    rather than the scaled-up rectangle Flat mode's _draw_unit() uses. A
    multi-tile building reaches this function once per footprint tile,
    because _units_by_tile() buckets it into every one of them, so each
    diamond lands at its own tile's moment in the depth walk instead of the
    whole slab landing at the unit's own tile's moment (which is what let
    later terrain tiles paint back over half of it).

    The diamond sits at the unit's OWN tile's elevation, not (tx, ty)'s, so
    a multi-tile building still renders as one flat slab at a single height
    -- see _unit_iso_footprint and unit_pick's "asymmetry 1". Reading that
    elevation directly rather than via _unit_iso_footprint() means this
    function has no off-map guard of its own; _units_by_tile() has already
    dropped anything unit_tile_bounds() rejects, which is why the index
    below is safe. offset/clipping mirror _render_tile_iso()'s own
    scratch-canvas contract, for refresh_region_iso()'s incremental redraw.

    ink, when given, is the two-pass ink plane: the diamond's pixels are
    recorded there at ink_rise (_unit_mark_rise_iso)."""
    dst_y, dst_x, _src_y, _src_x = iso_geometry.diamond_indices(tile_px)
    off_x, off_y = offset
    base_y, base_x = _unit_mark_base_iso(unit, tx, ty, proj, elevations)
    extent = iso_geometry.index_extent(iso_geometry.diamond_indices, tile_px)
    _clipped_paint_solid(img, base_y - off_y, base_x - off_x, dst_y, dst_x, color, extent=extent)
    if ink is not None:
        _record_ink(ink, base_y - off_y, base_x - off_x, dst_y, dst_x, ink_rise)


def _unit_mark_rise_iso(unit, proj: iso_geometry.IsoProjection, elevations) -> int:
    """A Stepped mark's base rise: its unit's own tile's height, the one
    _unit_mark_base_iso places it at."""
    return int(elevations[int(unit.y), int(unit.x)]) * proj.elev_step


def _unit_mark_base_iso(unit, tx: int, ty: int, proj: iso_geometry.IsoProjection, elevations) -> tuple[int, int]:
    """(base_y, base_x) of a Stepped mark's diamond on footprint tile (tx, ty),
    shared by _draw_unit_iso and the native UnitPack."""
    elevation = int(elevations[int(unit.y), int(unit.x)])
    # The diamond is positioned by its CENTRE (off-grid placement, Stage 1), so a
    # span-1 unit's mark follows its sub-tile position instead of snapping to
    # (tx, ty). map_point_to_screen(tx + 0.5, ty + 0.5, elevation * elev_step)
    # is tile_screen_origin(tx, ty, elevation) + (half_w, half_h) exactly, so
    # a zero offset reproduces the old expression byte for byte.
    dx, dy = unit_paint_offset(unit)
    cx, cy = iso_geometry.map_point_to_screen(
        tx + 0.5 + dx, ty + 0.5 + dy, elevation * proj.elev_step, proj
    )
    return cy - proj.half_h, cx - proj.half_w


def _unit_screen_bbox_iso(
    unit,
    tile_w: int,
    tile_h: int,
    proj: iso_geometry.IsoProjection,
    elevations: np.ndarray,
    extra_top_px: int = 0,
) -> tuple[int, int, int, int] | None:
    """(sx0, sy0, sx1, sy1) canvas-pixel bbox a Stepped-mode unit's footprint
    would occupy -- the closed-form counterpart to actually drawing it
    (_draw_unit_iso() must still visit every footprint tile to paint its
    diamond, since each one's screen position differs, but the bbox extremes
    don't need that loop): screen_x = origin_x + (x+y)*half_w is monotonic
    increasing in both x and y independently, and screen_y = origin_y +
    (y-x)*half_h - elevation*elev_step is monotonic increasing in y and
    decreasing in x, with the SAME elevation for every footprint tile (see
    _unit_iso_footprint) -- so the bbox's four extremes are exactly at the
    footprint tile-range's own corners. Used by refresh_region_iso() to
    cheaply test a unit's overlap against a candidate region without an
    O(footprint size) scan, even across this project's ~11,000-unit real
    files. None if the unit is off-map.

    extra_top_px raises the TOP edge only (Track C5's Step 2), and is 0 for
    every Stepped caller so their bboxes stay byte-identical. Sloped needs
    it because the sentence above -- "a building's screen bbox only ever
    depends on proj + elevations, never on corner_rise", the grounds on
    which SlopedChunkCache reuses this function verbatim -- stops being
    true once a unit sits at unit_rise_px() rather than at
    elevation * elev_step. Under SLOPE_CORNER_RULE = "max" a corner is
    never BELOW its own tile's elevation and can be above it, so the
    interpolated rise only ever moves a unit UP the screen: sy1 is
    unaffected, and only sy0 needs the headroom. Under-covering here is a
    MISSED bystander paint (a stale pixel), not a no-op, so the caller
    passes a proven bound rather than a guess -- see
    SlopedChunkCache._refresh_source_caches()."""
    footprint = _unit_iso_footprint(unit, tile_w, tile_h, elevations)
    if footprint is None:
        return None
    tile_x0, tile_x1, tile_y0, tile_y1, elevation = footprint
    half_w, half_h = proj.half_w, proj.half_h
    sx0, _ = iso_geometry.tile_screen_origin(tile_x0, tile_y0, elevation, proj)
    sx1, _ = iso_geometry.tile_screen_origin(tile_x1 - 1, tile_y1 - 1, elevation, proj)
    _, sy0 = iso_geometry.tile_screen_origin(tile_x1 - 1, tile_y0, elevation, proj)
    _, sy1 = iso_geometry.tile_screen_origin(tile_x0, tile_y1 - 1, elevation, proj)
    # Off-grid placement, Stage 1: a span-1 mark is shifted off its tile by up to
    # half a tile per axis, so the extremes above are no longer the whole box.
    # Widened by the WHOLE half-tile rather than by this unit's own offset --
    # under-covering here is a MISSED bystander paint (a stale pixel), not a
    # no-op, and the monotonicity argument above only survives on a bound that
    # doesn't depend on which corner the shift happens to point at.
    dx, dy = unit_paint_offset(unit)
    if dx or dy:
        sx0, sx1 = sx0 - half_w, sx1 + half_w
        sy0, sy1 = sy0 - half_h, sy1 + half_h
    return sx0, sy0 - extra_top_px, sx1 + 2 * half_w, sy1 + 2 * half_h


def _units_by_tile(
    scenario: LoadedScenario, unit_filter: UnitFilter = UnitFilter()
) -> dict[tuple[int, int], list[tuple]]:
    """Every unit, grouped by EVERY tile of its footprint -- the key Stepped
    mode's compositor (_paint_tile_and_units_iso) uses to draw each of a
    unit's diamonds at exactly the point in depth_order's loop where that
    diamond belongs. A multi-tile building therefore appears in as many
    buckets as it has footprint tiles, and paints one diamond per bucket.

    Own-tile-only bucketing was the earlier shape, and it was a bug: the
    unit painted its whole footprint slab at its own tile's moment, and
    every footprint tile later in depth_order then painted its own terrain
    diamond back over it -- a span-4 building kept about half its footprint.

    Preserves overlay_units()'s own per-player/per-unit-list stacking order
    within each tile's bucket (built by iterating players/units in that same
    order), so units sharing a tile still paint in the same relative order
    Flat mode would.

    **Off-map units are dropped here, and that is load-bearing downstream.**
    This is now the only off-map guard in the Stepped draw path:
    _draw_unit_iso() indexes elevations[int(unit.y), int(unit.x)] directly
    rather than re-deriving bounds, so it relies on this function having
    already skipped anything unit_tile_bounds() rejects. Buckets over the
    CLAMPED bounds that function returns.

    unit_filter (phase 3) drops non-matching units outright rather than
    marking them -- so a hidden unit is absent from every downstream
    consumer at once, including phase 3's pick index, and cannot be
    selected through a tile it no longer paints on. A default UnitFilter()
    keeps every unit in its original relative order, which is what makes
    the filter's arrival a byte-identical no-op for every existing
    caller."""
    mm = scenario.map_manager
    tile_w, tile_h = mm.map_width, mm.map_height
    buckets: dict[tuple[int, int], list] = {}
    for player_id, units in enumerate(scenario.unit_manager.units):
        player_color = scenario.player_colors[player_id]
        for unit in units:
            if not unit_filter.matches(player_id, unit):
                continue
            tiles = unit_occupied_tiles(unit, tile_w, tile_h)
            if tiles is None:
                continue
            entry = (unit, _unit_color(unit, player_color))
            for tx, ty in tiles:
                buckets.setdefault((tx, ty), []).append(entry)
    return buckets
