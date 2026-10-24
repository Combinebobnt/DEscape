"""Qt-free geometry for View > Range Rings -- GH #49's base range ring.

Split the way grid_overlay already is: this module decides WHERE the ring
goes, map_view's QGraphicsPathItem decides how it looks. Nothing here imports
PyQt5.

Nothing here draws an ellipse. The ring is a true circle in TILE space, and
isometric projection is what renders it as the 2:1 ellipse the issue's
screenshot shows, so every sample goes through iso_geometry's own forward
map rather than through a hand-written ellipse. That also keeps the ring on
the same projection the sprites and the footprint outlines use.

Distance is measured on the ground plane, the rule descape/ruler.py already
states for the Ruler ("AoE2 resolves unit range on the ground plane, so a
unit standing on a hill is not further away"). Only the height the ring is
DRAWN at follows the building, and that is one value for the whole ring,
passed in by the caller.

The always-on layer (GH #153 / #155) rings every unit in the index, not just
the selection, under two independent toggles. All Buildings uses the same
radius rule as the selection ring, limited to unit_kind.building_consts() so
mobile siege drops out. Map Revealers rings each class-30 revealer at its
plain line of sight from its stored point, no span term: revealers are 0.1
collision point objects. Invisible Objects carry LOS 0 in the .dat, so they
draw none.

Trigger Changes (GH #49 slice 2) adds a dashed ring per selected building for
each trigger that changes its maximum range, from trigger_range's per-trigger
results: same footprint-edge rule, values at or below 0 and values equal to
the base range dropped (they would sit on the solid ring), deduplicated and
capped at MAX_TRIGGER_RINGS in trigger order. A building whose base range is
0 still gets a dashed ring when a trigger gives it one. Buildings only, the
selection ring's BUILDING_TILE_SPANS gate; the always-on layers get none.
"""

from __future__ import annotations

import math

from descape import iso_geometry, terrain_palette, trigger_range, unit_kind
from descape.unit_stats_table import line_of_sight, unit_stats

# At most this many dashed trigger rings per building; the Inspector lists every value.
MAX_TRIGGER_RINGS = 6

# Samples per ring, rounded UP to a multiple of eight. Eight, not four:
# sampling from theta 0 in 90-degree steps hits the tile-space axis extremes,
# but the SCREEN extremes of the projected ellipse sit at 45 + k*90 degrees,
# and the ring's bounding box is only the ellipse's when both sets are
# sampled.
SEGMENT_MULTIPLE = 8
MIN_SEGMENTS = 64
SEGMENTS_PER_TILE = 16


def ring_radius_tiles(range_tiles: float, span: tuple[int, int]) -> float:
    """The ring's radius in tiles, measured from the FOOTPRINT EDGE: a 4x4
    Castle with range 8 draws at 10 tiles.

    The one named place that rule lives, so the in-game pass can flip it in a
    single edit. The alternative is plain `range_tiles`, measured from the
    building's centre; only a side-by-side against the game's own circle
    settles which one AoE2 draws. This shape matches the game resolving reach
    from a unit's collision radius, and degenerates to `range_tiles` for a
    1x1."""
    return float(range_tiles) + max(span) / 2.0


def ring_radius_for_const(unit_const: int) -> float | None:
    """The ring radius for a placed `unit_const`, or None when it draws no
    ring: not a building, or a building with no range.

    BUILDING_TILE_SPANS is the codebase's own is-a-building membership test
    (render._unit_color reads it that way), and is read live here rather than
    snapshotted, since tests inject synthetic consts into it.

    The ring is the building's REACH, unit_stats' `max_range` (the .dat's
    `type_50.max_range`), not `range`, the panel figure the Inspector's Range
    row shows. The two differ on 2 of 491 buildings in the Sept 2026 .dat:
    Poenari Castle (445) displays 0 but shoots Castle arrows to 8 (GH #111),
    and hidden const 2276 displays 6 against a reach of 7."""
    if not is_ring_building(unit_const):
        return None
    range_tiles = base_max_range(unit_const)
    if range_tiles <= 0.0:
        return None
    return ring_radius_tiles(range_tiles, terrain_palette.BUILDING_TILE_SPANS[unit_const])


def is_ring_building(unit_const: int) -> bool:
    """The selection ring's building gate (BUILDING_TILE_SPANS), read live."""
    return unit_const in terrain_palette.BUILDING_TILE_SPANS


def base_max_range(unit_const: int) -> float:
    """The .dat reach a trigger change starts from, 0.0 when absent."""
    return float(unit_stats(unit_const).get("max_range", 0.0))


def building_trigger_ranges(
    index: trigger_range.RangeEffectIndex,
    *,
    player_id: int,
    unit_const: int,
    reference_id: int,
    own_tile: tuple[int, int],
) -> tuple[trigger_range.TriggerRange, ...]:
    """trigger_range.trigger_ranges() for a placed building, () for anything
    outside the BUILDING_TILE_SPANS gate. The rings and the Inspector line share it."""
    if not is_ring_building(unit_const):
        return ()
    return trigger_range.trigger_ranges(
        index,
        base_max_range(unit_const),
        player_id=player_id,
        unit_const=unit_const,
        reference_id=reference_id,
        own_tile=own_tile,
    )


def trigger_ring_radii(
    index: trigger_range.RangeEffectIndex,
    *,
    player_id: int,
    unit_const: int,
    reference_id: int,
    own_tile: tuple[int, int],
) -> tuple[float, ...]:
    """The dashed rings' radii in tiles for one placed building, in trigger
    order, or () for a non-building or no drawable trigger result."""
    ranges = building_trigger_ranges(
        index, player_id=player_id, unit_const=unit_const, reference_id=reference_id, own_tile=own_tile
    )
    if not ranges:
        return ()
    span = terrain_palette.BUILDING_TILE_SPANS[unit_const]
    base = base_max_range(unit_const)
    values: list[float] = []
    for result in ranges:
        if result.value <= 0.0 or result.value == base or result.value in values:
            continue
        values.append(result.value)
        if len(values) == MAX_TRIGGER_RINGS:
            break
    return tuple(ring_radius_tiles(value, span) for value in values)


def revealer_radius_for_const(unit_const: int) -> float | None:
    """A map revealer's ring radius, or None for any other const.

    The one named place the revealer rule lives, like ring_radius_tiles(), so
    an in-game comparison can flip it in a single edit: plain LOS from the
    stored point. Membership is a rule, not a list: a "revealer" invisible
    category with LOS > 0, which drops the LOS-0 placeholders."""
    if unit_kind.invisible_category(unit_const) != "revealer":
        return None
    return line_of_sight(unit_const)


def persistent_ring_radius(unit_const: int, *, revealers: bool, buildings: bool) -> float | None:
    """The always-on layer's radius for `unit_const` under its two toggles,
    or None when neither draws it."""
    if revealers:
        radius = revealer_radius_for_const(unit_const)
        if radius is not None:
            return radius
    if buildings and unit_const in unit_kind.building_consts():
        return ring_radius_for_const(unit_const)
    return None


def selection_ring_radius(unit_const: int, *, buildings: bool, revealers: bool) -> float | None:
    """The selection ring's radius for a selected `unit_const` under Selected
    Buildings and Selected Map Revealers, or None when neither draws it.
    Buildings keep ring_radius_for_const()'s rule, mobile siege included."""
    if buildings:
        radius = ring_radius_for_const(unit_const)
        if radius is not None:
            return radius
    if revealers:
        return revealer_radius_for_const(unit_const)
    return None


def ring_segments(radius_tiles: float) -> int:
    """How many samples a ring of this radius takes, a multiple of
    SEGMENT_MULTIPLE. A 10-tile ring is 160, cheap for one QPainterPath."""
    wanted = max(MIN_SEGMENTS, math.ceil(SEGMENTS_PER_TILE * max(0.0, float(radius_tiles))))
    return math.ceil(wanted / SEGMENT_MULTIPLE) * SEGMENT_MULTIPLE


def ring_points(
    cx: float,
    cy: float,
    radius_tiles: float,
    style: str,
    tile_px: int | None = None,
    proj: iso_geometry.IsoProjection | None = None,
    rise_px: int = 0,
) -> list[tuple[float, float]]:
    """The ring around continuous map point (cx, cy) as a polygon in scene
    space, plain tuples so this module stays Qt-free (unit_pick.unit_polygons'
    own convention). Implicitly closed: the last point joins the first, and
    map_view's _add_closed_polygons() is what closes the subpath.

    Flat's scene space is (mx * tile_px, my * tile_px) (grid_axes()' own flat
    branch), so the ring is a TRUE circle there and Flat + Isometric's view
    transform shears it into the right ellipse for free. Stepped and Sloped
    project each sample through map_point_to_screen() at the caller's single
    `rise_px`."""
    segments = ring_segments(radius_tiles)
    samples = (
        (
            cx + radius_tiles * math.cos(2.0 * math.pi * i / segments),
            cy + radius_tiles * math.sin(2.0 * math.pi * i / segments),
        )
        for i in range(segments)
    )
    if style == "flat":
        if tile_px is None:
            raise ValueError("flat needs tile_px")
        return [(mx * tile_px, my * tile_px) for mx, my in samples]
    if style in ("stepped", "sloped"):
        if proj is None:
            raise ValueError(f"{style} needs proj")
        points = []
        for mx, my in samples:
            sx, sy = iso_geometry.map_point_to_screen(mx, my, int(rise_px), proj)
            points.append((float(sx), float(sy)))
        return points
    raise ValueError(f"unknown terrain style {style!r}")
