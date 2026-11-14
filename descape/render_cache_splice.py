"""The chunk caches' unit and elevation splices: UnitSplice, eligibility and refusal, component walks, splice plans."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from descape import (
    iso_geometry,
    render_bbox,
    render_sprites,
    render_units,
    unit_sprites,
)
from descape.scenario_io import LoadedScenario
from descape.unit_filter import UnitFilter


@dataclass(frozen=True)
class UnitSplice:
    """One single-unit edit, carrying everything invalidate_units()'s splice
    path (Batch D's D4) needs to update units_by_tile/building_bboxes/
    sprites without a full source rebuild. The viewer (D5) builds one of
    these per edited unit from whatever begin_unit_edit/commit_unit_edit
    already captured; this class only names the shape.

    player_id/index are the unit's position in
    scenario.unit_manager.units[player_id] AT CALL TIME for an add or a
    move. A removal's index is never read: every cache drops a removed
    unit by identity (_drop_from_tiles, skip_ids, Flat's _row_of), so a
    batch removal that shifts later units' indices is safe too.

    old_own_tile/new_own_tile are (int(x), int(y)) pre-/post-edit -- the
    key building_bboxes uses -- and must be captured by the caller before/
    after the mutation, since by the time invalidate_units() runs the unit
    object already holds its NEW x/y and the old one is gone from it.

    old_tiles/new_tiles are render_units.unit_occupied_tiles()'s own ORDERED
    result, pre- and post-edit. Order matters: unit_sprites.
    sprite_anchor_tile() picks a specific tile out of this list, and it
    must see exactly the list a full rebuild would have used. Every
    by_anchor key a unit writes, column slices' included (TASK-221), is one
    of these tiles, so clearing old_tiles clears all of them.

    Empty old_tiles/old_own_tile=None means an add (Place, no pre-edit
    state); empty new_tiles/new_own_tile=None means a removal (single
    Delete, no post-edit state). Identical old and new tiles mean a
    re-anchor: the unit stayed put but its tile's elevation changed (see
    _elevation_splices() and _reanchor_units()).

    old_player_id is the pre-edit owner for a reassign (Convert), None when
    the owner didn't change. player_id/index are then the DESTINATION list
    and the unit's appended position in it. A reassign reorders the source
    list, which is safe for Stepped/Sloped because nothing spliced there
    stores another unit's list index: their layers are tile-keyed, removal
    is by `is` identity, and the wall-override memo is keyed on the model's
    unit generation. FlatChunkCache, whose rows are player-ordered, reads
    old_player_id for its rows; Stepped/Sloped read it only through
    is_reassign and is_move, which let a Convert skip the wall const guard
    and a moved wall take the membership component splice."""

    player_id: int
    index: int
    unit: object
    old_own_tile: tuple[int, int] | None
    new_own_tile: tuple[int, int] | None
    old_tiles: tuple[tuple[int, int], ...]
    new_tiles: tuple[tuple[int, int], ...]
    old_player_id: int | None = None

    @property
    def changed_tiles(self) -> frozenset[tuple[int, int]]:
        return frozenset(self.old_tiles) | frozenset(self.new_tiles)

    @property
    def is_reassign(self) -> bool:
        """A pure Convert: the owner changed and nothing else did. Such an edit
        moves no unit, so no wall's neighbour mask changes (see _splice_eligible)."""
        return (
            self.old_player_id is not None
            and self.old_tiles == self.new_tiles
            and self.old_own_tile == self.new_own_tile
        )

    @property
    def is_move(self) -> bool:
        """Present before and after, same owner: Move, Nudge, Set field, a
        gate orientation swap, or a wall whose shape index a move rewrote in
        place (old and new tiles equal). For a wall/connector that changes
        its tiles, set_position() keeps list order, and two fields-only
        writes of a rotation-variant const's rotation exist, both literal
        set_wall_variant() indices that keep the file's radian verdict (in
        an integer file one can add or drop the written wall's own lone-wall
        key, and that wall is in the edit). A move's (GH #266,
        viewer._apply_wall_reshapes()) goes wholesale instead when it would
        flip the verdict (wall_run.MovePlan.radian_flip), and any other
        batch that flips it (an undo restoring a radian value) is caught by
        wall_verdict_flipped() and taken wholesale (TASK-583). The Selection
        transform's (viewer._write_lift_units()) writes only in an
        integer-encoded file (region_transform._rotation_write()), where
        every wall already stores a literal. What changes is the neighbour
        mask of the walls beside its old and new tiles
        (_connector_neighbours()), and the rewritten walls' own shapes, each
        of which carries a splice of its own."""
        return self.old_own_tile is not None and self.new_own_tile is not None and self.old_player_id is None


def _const_splice_eligible(unit) -> bool:
    """The const half of _splice_eligible(), shared with FlatChunkCache's row
    splice: a wall/connector or rotation-variant const's shape is a function
    of its neighbours, so re-resolving the edited unit alone is not enough
    when it moves. Stepped/Sloped skip it for a pure reassign; Flat does not."""
    return not (
        unit_sprites.rotation_variant_eligible(unit.unit_const)
        or unit.unit_const in unit_sprites.wall_connector_consts()
    )


class _ReachFallback:
    """REACH_FALLBACK's type: a named singleton so a failed identity check
    reads as the sentinel in a traceback, not as a bare object()."""

    def __repr__(self) -> str:
        return "REACH_FALLBACK"


# sprite_extent_before()/sprite_extent_after()'s "size this edit with today's
# reach-padded bbox instead" answer. Compare by identity.
REACH_FALLBACK = _ReachFallback()


def _union_bbox(a: tuple[int, int, int, int] | None, b: tuple[int, int, int, int] | None):
    """The union of two half-open bboxes, either of which may be None."""
    if a is None:
        return b
    if b is None:
        return a
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _splice_eligible(units_by_tile: dict, splice: UnitSplice) -> bool:
    """Whether invalidate_units()'s splice path may handle `splice` without
    a full source rebuild. False is the documented fallback (the caller's
    job is then to call self._refresh_source_caches() instead), for two
    reasons neither of which this module can safely work around:

    - A wall/connector const's override is a function of its NEIGHBOURS
      (render_units.wall_variant_rotation_overrides' own docstring): moving one
      reshapes up to eight other units this splice never touches, and the
      override dict's own key shifts on any removal. A moved one is refused
      here and spliced by _membership_component_splices(), which re-derives
      those neighbours too (_batch_splice_refusal()'s `const:move`).
      A pure reassign (UnitSplice.is_reassign) skips it: the neighbour mask
      is built over every player's connectors, and stored_rotation() is
      owner-free for every rotation-variant const, so no wall's shape or the
      file's radian/integer verdict can change. Only the edited unit's own
      override key moves, and the splice passes the real, freshly memoized
      dict (_splice_overrides()).
    - Any tile this edit touches (old or new footprint -- own-tile
      included, since a non-wall/-gate unit's own tile is always a member
      of its own unit_occupied_tiles(), per unit_tile_bounds' own
      invariant) that already, or would, carry more than this one unit.
      building_bboxes/by_anchor union several units' data at one key with
      no way to subtract just one back out, and corpus files do have >1
      unit sharing an own-tile (5 of 16 example files, 20 tiles total --
      e.g. a decoration placed on a building)."""
    unit = splice.unit
    if not splice.is_reassign and not _const_splice_eligible(unit):
        return False
    # The all(not any(...)) one-liner ruff suggests here double-negates.
    for tile in splice.changed_tiles:  # noqa: SIM110
        if any(u is not unit for u, _ in units_by_tile.get(tile, ())):
            return False
    return True


# Past this many splices a unit-edit batch takes the wholesale path instead. A backstop:
# the batch callers' own cost guard (viewer._SPLICE_COST_RATIO) binds first below ~16k units.
UNIT_SPLICE_MAX_UNITS = 1000


def _batch_splice_eligible(units_by_tile: dict, changed: list[UnitSplice]) -> bool:
    """_splice_eligible() for a whole batch, which the per-splice form rejects
    whenever one splice removes the unit another replaces: Draw's stroke end
    removes the tree on a changed tile and rolls a new one onto it.

    A pre-edit occupant of a touched tile is allowed if the batch removes it
    (new_own_tile None), since its slots are dropped before anything lands.
    After the edit a tile may hold at most one batch unit. A non-batch
    occupant still rejects, as it does per splice. A single-splice batch gets
    exactly _splice_eligible()'s answer. Callers must apply removals first
    (_removals_first)."""
    return _batch_splice_refusal(units_by_tile, changed) is None


def _const_refusal(changed: list[UnitSplice]) -> str | None:
    """_batch_splice_refusal()'s const clause over the whole batch. For a
    wall/connector/rotation-variant unit that is not a pure reassign:
    `const:add` or `const:remove` (either shifts override keys, so wholesale
    sources; either can also flip the file's radian verdict, which the caller's
    repaint learns from wall_verdict_flipped()), `const:other` (an owner change
    that also moved), else `const:move` if one moved, else None."""
    moved = False
    for s in changed:
        if s.is_reassign or _const_splice_eligible(s.unit):
            continue
        if s.is_move:
            moved = True
        elif s.old_own_tile is None:
            return "const:add"
        elif s.new_own_tile is None:
            return "const:remove"
        else:
            return "const:other"
    return "const:move" if moved else None


def _batch_splice_refusal(units_by_tile: dict, changed: list[UnitSplice], cap: int | None = None) -> str | None:
    """Which clause of _batch_splice_eligible() refuses `changed`, or None if
    none does: `cap`, then the const clause over the whole batch
    (_const_refusal()), then the first of `occupant:foreign` (a unit outside
    the batch), `occupant:batch` (a batch unit that stays, e.g. one a group
    Move carries off the tile another lands on) and `claimed` (two batch
    units landing on one tile). So an occupant or claimed answer means the
    batch holds no wall/connector const but a reassigned one. `const:move`
    and those three go on to a component splice (_splice_plan_or_refusal());
    `cap` and the other const answers are final. `cap` None reads
    UNIT_SPLICE_MAX_UNITS at call time."""
    if len(changed) > (UNIT_SPLICE_MAX_UNITS if cap is None else cap):
        return "cap"
    const = _const_refusal(changed)
    if const is not None:
        return const
    removed = {id(s.unit) for s in changed if s.new_own_tile is None}
    batch = {id(s.unit) for s in changed}
    claimed: dict[tuple[int, int], int] = {}
    for s in changed:
        unit = s.unit
        for tile in s.changed_tiles:
            for u, _ in units_by_tile.get(tile, ()):
                if u is not unit and id(u) not in removed:
                    return "occupant:batch" if id(u) in batch else "occupant:foreign"
        for tile in s.new_tiles:
            if claimed.setdefault(tile, id(unit)) != id(unit):
                return "claimed"
    return None


# Past this many units in a unit-edit batch's shared-tile component (Convert or
# not) the batch takes the wholesale path instead. Set 2026-09-29 from old-allies'
# brush-9 Draw rows: 335-345-unit components splice in 36-55 ms against 100-139 ms
# wholesale. Matches _ELEV_SPLICE_MAX_UNITS' measured re-anchor cost (~13us/unit).
_COMPONENT_SPLICE_MAX_UNITS = 1000

# Past both caps above: invalidate_units()' visible-only splice (cap_levels), Stepped/Sloped, sprites
# on. min(3000, half the batch where it stops winning), measured 2026-10-14 by bench_stroke_end --visible-cap.
VISIBLE_SPLICE_MAX_UNITS = 3000


def _locate(own_index: dict, unit) -> tuple[int, int] | None:
    """unit's current (player_id, i), from the own-tile index; None if absent."""
    for player_id, i, u in own_index.get((int(unit.x), int(unit.y)), ()):
        if u is unit:
            return player_id, i
    return None


def _shared_tile_component(
    scenario: LoadedScenario, units_by_tile: dict, unit_filter: UnitFilter, own_index: dict, seeds, cap: int
) -> list[tuple] | None:
    """The connected component of units sharing any footprint or own tile with
    a seed, transitively, sorted into the wholesale walk's order (player-major,
    then list index). None past `cap` members, or if a unit can't be located.

    Seeds are (player_id, i, unit, own tile, occupied tiles) tuples and are
    always kept. A tile's neighbours are the units in units_by_tile (visible
    footprints) and in own_index (every own-tile key). A non-seed the filter
    hides or that is off-map holds no key and is skipped. Every key on the
    component's tiles then belongs to component units alone, so re-deriving
    them all in the returned order reproduces a fresh build exactly (see
    _reanchor_units' two phases, and _splice_units_by_tile, which drops and
    re-appends each unit in turn)."""
    members = {id(seed[2]): seed for seed in seeds}
    if len(members) > cap:
        return None
    tiles = [t for seed in seeds for t in (*seed[4], seed[3]) if t is not None]
    found, _why = _walk_component(scenario, units_by_tile, unit_filter, own_index, tiles, set(members), cap - len(members))
    if found is None:
        return None
    return sorted([*members.values(), *found], key=lambda m: (m[0], m[1]))


def _walk_component(
    scenario: LoadedScenario, units_by_tile: dict, unit_filter: UnitFilter, own_index: dict, tiles,
    known: set[int], room: int,
) -> tuple[list[tuple] | None, str | None]:
    """The shared-tile walk under both component splices: every unit reachable
    from `tiles` through a shared footprint or own tile, transitively, as
    unsorted (player_id, i, unit, own tile, occupied tiles) tuples. Or None and
    why: `cap` past `room` units found, `none` if one can't be located.

    `known` holds the ids of units the caller already has (seeds, a unit
    edit's batch). They are never located, returned or walked from, which is
    what lets a removed batch unit (absent from the post-edit own_index) sit
    in units_by_tile. A tile's neighbours are the units in units_by_tile
    (visible footprints) and in own_index (every own-tile key). A unit the
    filter hides or that is off-map holds no key and is skipped."""
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height
    known = set(known)
    found: list[tuple] = []
    queue = list(tiles)
    seen: set[tuple[int, int]] = set()
    while queue:
        tile = queue.pop()
        if tile in seen:
            continue
        seen.add(tile)
        neighbours = list(own_index.get(tile, ()))
        for u, _color in units_by_tile.get(tile, ()):
            if id(u) not in known:
                where = _locate(own_index, u)
                if where is None:
                    return None, "none"
                neighbours.append((*where, u))
        for player_id, i, u in neighbours:
            if id(u) in known or not unit_filter.matches(player_id, u):
                continue
            occupied = render_units.unit_occupied_tiles(u, w, h)
            if occupied is None:
                continue
            own = (int(u.x), int(u.y))
            known.add(id(u))
            found.append((player_id, i, u, own, tuple(occupied)))
            if len(found) > room:
                return None, "cap"
            queue.extend(occupied)
            queue.append(own)
    return found, None


def _reassign_component_splices(
    scenario: LoadedScenario, units_by_tile: dict, unit_filter: UnitFilter, changed: list[UnitSplice],
    cap: int | None = None,
) -> tuple[list[UnitSplice] | None, str | None]:
    """A Convert batch that _batch_splice_eligible() refused for a shared tile,
    widened to its _shared_tile_component(), or None and why (`cap`, `none`)
    when that can't be done.

    A reassign moves no unit, so the component argument holds. Batch units
    seed it at their new tiles and are always kept, so a filter flip is
    covered. Each unit is spliced at its CURRENT (player_id, i) as a
    re-anchor, batch units keeping old_player_id.

    None unless every splice is a pure reassign, or past `cap` component
    units (None: _COMPONENT_SPLICE_MAX_UNITS, read at call time)."""
    if cap is None:
        cap = _COMPONENT_SPLICE_MAX_UNITS
    if not all(s.is_reassign for s in changed):
        return None, "none"
    if len(changed) > cap:
        return None, "cap"
    own_index = render_units.unit_own_tile_index(scenario)
    batch = {id(s.unit): s for s in changed}
    seeds = []
    for s in changed:
        where = _locate(own_index, s.unit)
        if where is None:
            return None, "none"
        seeds.append((*where, s.unit, s.new_own_tile, s.new_tiles))
    tiles = [t for seed in seeds for t in (*seed[4], seed[3]) if t is not None]
    found, why = _walk_component(
        scenario, units_by_tile, unit_filter, own_index, tiles, set(batch), cap - len(seeds)
    )
    if found is None:
        return None, why

    out: list[UnitSplice] = []
    for player_id, i, unit, own, tiles in sorted([*seeds, *found], key=lambda m: (m[0], m[1])):
        s = batch.get(id(unit))
        old_player_id = s.old_player_id if s is not None else None
        out.append(UnitSplice(player_id, i, unit, own, own, tiles, tiles, old_player_id=old_player_id))
    return out, None


def _connector_neighbours(
    scenario: LoadedScenario, unit_filter: UnitFilter, own_index: dict, changed: list[UnitSplice], known: set[int]
) -> list[tuple]:
    """Every rotation-variant unit outside `known` whose shape a batch
    connector's move can change, as (player_id, i, unit, own tile, occupied
    tiles). render_units._wall_variant_rotation_overrides_uncached() reads a
    wall's 8-bit mask over every connector's tiles, at its bounds' low
    corner, so a moved connector reshapes only the walls whose low corner is
    one of the eight tiles around one of its old or new tiles. Those tiles
    are dilated by NEIGHBOUR_OFFSETS (the tiles themselves kept too, which
    errs wide), looked up in the filter-free own_index, and every 1x1
    rotation-variant unit there is kept. A hidden or off-map one holds no
    key and is skipped. Over-inclusion is harmless, a re-anchor being
    idempotent; under-inclusion would leave a wall stale."""
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height
    connectors = unit_sprites.wall_connector_consts()
    dilated: set[tuple[int, int]] = set()
    for s in changed:
        if s.unit.unit_const not in connectors:
            continue
        for x, y in (*s.old_tiles, *s.new_tiles):
            dilated.add((x, y))
            dilated.update((x + dx, y + dy) for dx, dy, _bit in unit_sprites.NEIGHBOUR_OFFSETS)
    out: list[tuple] = []
    seen = set(known)
    for tile in dilated:
        for player_id, i, unit in own_index.get(tile, ()):
            if id(unit) in seen or not unit_sprites.rotation_variant_eligible(unit.unit_const):
                continue
            seen.add(id(unit))
            bounds = render_units.unit_tile_bounds(unit, w, h)
            if bounds is None or (bounds[0], bounds[2]) not in dilated or not unit_filter.matches(player_id, unit):
                continue
            occupied = render_units.unit_occupied_tiles(unit, w, h)
            if occupied is not None:
                out.append((player_id, i, unit, (int(unit.x), int(unit.y)), tuple(occupied)))
    return out


def _membership_component_splices(
    scenario: LoadedScenario, units_by_tile: dict, unit_filter: UnitFilter, changed: list[UnitSplice],
    cap: int | None = None,
) -> tuple[list[UnitSplice] | None, str | None]:
    """A unit-edit batch that is not a pure Convert (Draw's stroke end, its
    undo/redo, a group Move) and that _batch_splice_eligible() refused for a
    shared tile or a moved wall/connector (`const:move`), widened to every
    unit sharing a footprint or own tile with it, transitively. Or None and
    why: `cap`, `none` (a unit that can't be located) or `const` (an added
    or removed wall/connector, which _batch_splice_refusal() already sends
    wholesale).

    Batch splices are kept as they are and never re-located: a removed unit
    is not in the post-edit own_index. The walk (_walk_component()) starts
    from every batch splice's old and new tiles and both own tiles. Old tiles
    find the pre-edit co-occupants, since units_by_tile is still pre-edit
    here; new tiles find the post-edit ones, the same units, since only batch
    units moved. So a non-batch unit met on the way did not move, and its
    own_index (player_id, i) is right: it becomes a re-anchor splice there.

    Order: removals first, then every other splice by (player_id, index), the
    wholesale walk's order. Batch splices carry their post-edit index, which
    is what a fresh walk sees. _splice_units_by_tile()'s drop-and-append and
    _reanchor_units()' clear-then-re-add then leave every bucket and shared
    key on the component's tiles in fresh-build order.

    A moved wall/connector reshapes walls beside it that share no tile with
    it: _connector_neighbours() adds them, stationary, as re-anchor splices,
    and their tiles seed the walk too, so their co-occupants join. Every
    rotation-variant unit in the plan then resolves with the real overrides
    (_splice_overrides()), memoized on the edit's own unit_gen bump. None
    past `cap` units in total, neighbours included (None:
    _COMPONENT_SPLICE_MAX_UNITS, read at call time)."""
    if cap is None:
        cap = _COMPONENT_SPLICE_MAX_UNITS
    if any(not s.is_reassign and not s.is_move and not _const_splice_eligible(s.unit) for s in changed):
        return None, "const"
    if len(changed) > cap:
        return None, "cap"
    own_index = render_units.unit_own_tile_index(scenario)
    known = {id(s.unit) for s in changed}
    neighbours = _connector_neighbours(scenario, unit_filter, own_index, changed, known)
    room = cap - len(changed) - len(neighbours)
    if room < 0:
        return None, "cap"
    tiles: list[tuple[int, int]] = []
    for s in changed:
        tiles.extend(s.old_tiles)
        tiles.extend(s.new_tiles)
        tiles.extend(t for t in (s.old_own_tile, s.new_own_tile) if t is not None)
    for _p, _i, unit, own, occ in neighbours:
        tiles.extend(occ)
        tiles.append(own)
        known.add(id(unit))
    found, why = _walk_component(scenario, units_by_tile, unit_filter, own_index, tiles, known, room)
    if found is None:
        return None, why
    reanchors = [UnitSplice(p, i, unit, own, own, occ, occ) for p, i, unit, own, occ in (*neighbours, *found)]
    staying = [s for s in changed if s.new_own_tile is not None]
    walk_order = sorted([*staying, *reanchors], key=lambda s: (s.player_id, s.index))
    return [s for s in changed if s.new_own_tile is None] + walk_order, None


def _splice_plan(
    scenario: LoadedScenario, units_by_tile: dict, unit_filter: UnitFilter, changed: list[UnitSplice]
) -> list[UnitSplice] | None:
    """The splices a Stepped/Sloped unit edit applies, or None for the
    wholesale path: `changed` itself if _batch_splice_eligible() accepts it,
    else its shared-tile component (_reassign_component_splices() for a
    Convert, _membership_component_splices() for everything else).
    Module-level so can_splice(), invalidate_units() and tools/ share it."""
    return _splice_plan_or_refusal(scenario, units_by_tile, unit_filter, changed)[0]


def _splice_plan_or_refusal(
    scenario: LoadedScenario, units_by_tile: dict, unit_filter: UnitFilter, changed: list[UnitSplice],
    cap: int | None = None, component_cap: int | None = None,
) -> tuple[list[UnitSplice] | None, str | None]:
    """_splice_plan()'s answer paired with why it is None: the refusing
    _batch_splice_refusal() clause, plus `/component:<why>` when the component
    fallback was tried and failed too. Perf Trace's `splice_refused=<reason>`
    token (_trace_splice_refusal()): `cap`, `const:add`, `const:remove`,
    `const:other`, or `<clause>/component:<why>` where the clause is
    `const:move`, `occupant:*` or `claimed`. `cap` and `component_cap` are
    the two caps' overrides (None: the module constants), which only
    invalidate_units()' visible-only splice passes."""
    reason = _batch_splice_refusal(units_by_tile, changed, cap)
    if reason is None:
        return changed, None
    if reason == "cap" or (reason.startswith("const:") and reason != "const:move"):
        return None, reason
    component = _reassign_component_splices if all(s.is_reassign for s in changed) else _membership_component_splices
    plan, why = component(scenario, units_by_tile, unit_filter, changed, component_cap)
    return plan, None if plan is not None else f"{reason}/component:{why}"


def _splice_overrides(scenario: LoadedScenario, changed: list[UnitSplice]) -> dict[tuple[int, int], float]:
    """The wall overrides a unit-edit splice must resolve with: the real dict
    when any spliced unit reads it (a reassigned or moved wall, or one a
    moved connector reshapes), else {} to skip the lookup. Memoized on
    scenario.unit_gen, which reassign() and set_position() bump."""
    if any(unit_sprites.rotation_variant_eligible(s.unit.unit_const) for s in changed):
        return render_units.wall_variant_rotation_overrides(scenario)
    return {}


def _verdict_can_move(changed: list[UnitSplice]) -> bool:
    """Whether `changed` could flip the file's wall radian verdict: some splice
    adds, removes or rewrites a rotation-variant wall. A pure reassign cannot,
    since stored_rotation() is owner-free for those consts (_splice_eligible())."""
    return any(not s.is_reassign and unit_sprites.rotation_variant_eligible(s.unit.unit_const) for s in changed)


def _removals_first(changed: list[UnitSplice]) -> list[UnitSplice]:
    """`changed` with its removals moved to the front, order otherwise kept.
    A removal clears every slot on its old tiles, so one applied after an add
    onto the same tile would delete that add's contribution."""
    return [s for s in changed if s.new_own_tile is None] + [s for s in changed if s.new_own_tile is not None]


def _write_dirty_elevations(
    scenario: LoadedScenario, elevations: np.ndarray, proj: iso_geometry.IsoProjection, dirty_indices,
    flatten: bool = False,
) -> bool | None:
    """refresh_wholesale()'s elevation write: each dirty tile's elevation (0
    under `flatten`, Flat + Isometric) into `elevations` in place, vectorized.
    None, writing nothing, if any lies outside proj's range (the caller then
    re-renders, as dirty_screen_bbox_iso() declining does); else whether any
    tile's elevation moved."""
    mm = scenario.map_manager
    w = mm.map_width
    idx = np.unique(np.fromiter(dirty_indices, dtype=np.int64, count=len(dirty_indices)))
    xs, ys = idx % w, idx // w
    if flatten:
        new = np.zeros(idx.size, dtype=elevations.dtype)
    else:
        terrain = mm.terrain
        new = np.fromiter((terrain[i].elevation for i in idx.tolist()), dtype=np.int64, count=idx.size)
        if ((new < proj.min_elev) | (new > proj.max_elev)).any():
            return None
    moved = bool((elevations[ys, xs] != new).any())
    elevations[ys, xs] = new
    return moved


# Past this many re-anchored units an elevation patch takes the wholesale path
# instead. Measured on June Event (14,106 units, sprites on): _reanchor_units
# costs ~1ms + 13us/unit and the bystander-grid patch 0.75ms at 132 keys, 1.8ms
# at 1000 (a full grid build is ~5.9ms), so ~16ms at 1000 against a 90-320ms
# wholesale rebuild. A real brush-9 stroke step re-anchors <=132.
_ELEV_SPLICE_MAX_UNITS = 1000

# Sloped's own cap, aimed at whole-stroke undo (1.1k-7.7k members). 2026-10-08
# probe, headroom held: break-even 7.3k-10k+ members, so min(3000, 7258 / 2).
_SLOPED_ELEV_SPLICE_MAX_UNITS = 3000


def _elevation_splices(
    scenario: LoadedScenario, units_by_tile: dict, unit_filter: UnitFilter, tiles, cap: int | None = None
) -> list[UnitSplice] | None:
    """One re-anchor UnitSplice per member of the _shared_tile_component()
    seeded by every unit whose OWN tile is in `tiles`, in walk order, or None
    (the caller then falls back to its wholesale path for the whole edit,
    never a partial one).

    Units the filter hides, or off-map ones, contribute nothing to either
    layer before or after, so they seed nothing. An elevation edit moves no
    unit, so the component argument holds: a co-occupant whose own tile did
    not change re-derives to an identical draw, and re-adding it restores the
    shared key _reanchor_units' clear phase emptied. Unlike _splice_eligible()
    there is no const check: a wall's neighbour-derived override stays valid
    as long as the caller passes the real one.

    None past `cap` members in total (default _ELEV_SPLICE_MAX_UNITS, read
    at call time), where the wholesale rebuild is the cheaper of the two.
    2026-10-07 undo probe: an uncapped in-op splice of 3860-4563 members
    costs 41-58ms against a 44-63ms wholesale -2/-1 build + pack, so a
    separate in-op cap was not taken."""
    if cap is None:
        cap = _ELEV_SPLICE_MAX_UNITS
    members = _elevation_component(scenario, units_by_tile, unit_filter, tiles, cap)
    if members is None:
        return None
    return [UnitSplice(p, i, unit, own, own, occ, occ) for p, i, unit, own, occ in members]


def _elevation_component(scenario: LoadedScenario, units_by_tile: dict, unit_filter: UnitFilter, tiles, cap: int):
    """_elevation_splices()' _shared_tile_component(), seeded by every visible
    unit whose own tile is in `tiles`, or None past `cap` members."""
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height
    own_index = render_units.unit_own_tile_index(scenario)
    seeds = []
    for own in tiles:
        for player_id, i, unit in own_index.get(own, ()):
            if not unit_filter.matches(player_id, unit):
                continue
            occupied = render_units.unit_occupied_tiles(unit, w, h)
            if occupied is not None:
                seeds.append((player_id, i, unit, own, tuple(occupied)))
    return _shared_tile_component(scenario, units_by_tile, unit_filter, own_index, seeds, cap)


def _splice_tiles(splices: list[UnitSplice]) -> set[tuple[int, int]]:
    """Every tile a splice batch can change a unit-pack row or farm entry on:
    both footprints (marks, sprite slots and farm tiles all key inside one)
    plus both own tiles."""
    tiles: set[tuple[int, int]] = set()
    for s in splices:
        tiles.update(s.old_tiles)
        tiles.update(s.new_tiles)
        tiles.update(t for t in (s.old_own_tile, s.new_own_tile) if t is not None)
    return tiles


def _refreshed_pack(
    pack, splices, units_by_tile, old_sprites, old_version, old_heights, sprites, version, heights, fold=None
):
    """pack refreshed over the batch's tiles, or None to force a full rebuild
    when it didn't describe the pre-edit sources (a refresh can't repair a
    pack that was already stale). fold rides with Sloped's heights: it is
    rebound whenever they are, so the `is` match on heights covers it."""
    if pack is None or not pack.matches(units_by_tile, old_sprites, old_version, old_heights):
        return None
    pack.refresh(_splice_tiles(splices), units_by_tile, sprites, version, heights, fold)
    return pack


def _dilate(tiles, w: int, h: int) -> set[tuple[int, int]]:
    """`tiles` plus their 8 neighbours, clipped to the map."""
    return {
        (x + dx, y + dy)
        for x, y in tiles
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if 0 <= x + dx < w and 0 <= y + dy < h
    }


def _corner_changed_tiles(old_cr: np.ndarray, new_cr: np.ndarray) -> set[tuple[int, int]]:
    """Every tile with any of its 4 corners changed between two (h+1, w+1)
    corner_rise arrays."""
    c = old_cr != new_cr
    ys, xs = np.nonzero(c[:-1, :-1] | c[:-1, 1:] | c[1:, :-1] | c[1:, 1:])
    return set(zip(xs.tolist(), ys.tolist(), strict=True))


def _fold_changed_tiles(old_fold: np.ndarray, new_fold: np.ndarray) -> set[tuple[int, int]]:
    """Every tile whose Sloped fold (iso_geometry.tile_folds()) differs. A
    fold reads the second-highest tile at each corner, which corner_rise
    does not carry, so it can flip with all four corners unchanged."""
    ys, xs = np.nonzero(old_fold != new_fold)
    return set(zip(xs.tolist(), ys.tolist(), strict=True))


def _drop_from_tiles(units_by_tile: dict, splice: UnitSplice) -> None:
    """Removes splice.unit from every bucket it used to occupy, deleting a
    bucket that empties (an empty list and a missing key are not the same
    thing to render_units._units_by_tile()'s consumers)."""
    for tile in splice.old_tiles:
        bucket = units_by_tile.get(tile)
        if not bucket:
            continue
        remaining = [e for e in bucket if e[0] is not splice.unit]
        if remaining:
            units_by_tile[tile] = remaining
        else:
            del units_by_tile[tile]


def _splice_units_by_tile(
    units_by_tile: dict, scenario: LoadedScenario, unit_filter: UnitFilter, splice: UnitSplice
) -> None:
    """The O(footprint) alternative to render_units._units_by_tile()'s O(all
    units) rebuild -- mutates units_by_tile in place. Valid only under
    _batch_splice_eligible()'s guard, applied removals-first: by the time a
    splice runs, every tile it touches holds at most `splice.unit` itself or
    another unit this batch removes, so removing it from its old buckets and
    appending it to its new ones can't clobber another unit's entry or
    disturb paint order within a shared bucket (there is none). Or under
    _reassign_component_splices() or _membership_component_splices(), whose
    list names every unit in each touched bucket: removals first, which only
    drop, then every other unit in the wholesale order, so each
    drop-and-append leaves every bucket in that order.

    A filtered-out unit still has its OLD buckets cleared before the early
    return. The one const-changing edit, gate-orientation cycling, now splices
    as a connector move (_membership_component_splices()), and matches()
    reads the const through GH #65's const gates, so a sibling the filter
    treats differently would flip it here: returning first would leave a
    stale entry behind."""
    _drop_from_tiles(units_by_tile, splice)
    if not unit_filter.matches(splice.player_id, splice.unit):
        return
    if splice.new_tiles:
        player_color = scenario.player_colors[splice.player_id]
        entry = (splice.unit, render_units._unit_color(splice.unit, player_color))
        for tile in splice.new_tiles:
            units_by_tile.setdefault(tile, []).append(entry)


def _splice_building_and_sprites(
    building_bboxes: dict,
    sprites: render_sprites.SpriteLayer | None,
    scenario: LoadedScenario,
    proj: iso_geometry.IsoProjection,
    elevations: np.ndarray,
    unit_filter: UnitFilter,
    corner_rise: np.ndarray | None,
    extra_top_px: int,
    splice: UnitSplice,
    with_farms: bool = True,
    tree_scale: float = 1.0,
    hero_glow: bool = False,
    wall_alpha: float = 1.0,
    fold: np.ndarray | None = None,
) -> render_sprites.SpriteLayer | None:
    """_reanchor_units() for a single splice, with overrides {}. {} is only
    right for a splice of a unit that isn't rotation-variant-eligible, whose
    resolve never consults the dict (render_units._wall_variant_rotation_
    overrides_uncached's own candidate filter). A wall, reassigned or moved (the
    membership component splice), must go through _reanchor_units() with
    _splice_overrides() instead, as both invalidate_units() do."""
    return _reanchor_units(
        building_bboxes, sprites, scenario, proj, elevations, unit_filter, corner_rise,
        extra_top_px, [splice], {}, with_farms, tree_scale, hero_glow, wall_alpha, fold=fold,
    )


def _reanchor_units(
    building_bboxes: dict,
    sprites: render_sprites.SpriteLayer | None,
    scenario: LoadedScenario,
    proj: iso_geometry.IsoProjection,
    elevations: np.ndarray,
    unit_filter: UnitFilter,
    corner_rise: np.ndarray | None,
    extra_top_px: int,
    splices: list[UnitSplice],
    overrides: dict[tuple[int, int], float],
    with_farms: bool = True,
    tree_scale: float = 1.0,
    hero_glow: bool = False,
    wall_alpha: float = 1.0,
    old_bboxes: dict | None = None,
    fold: np.ndarray | None = None,
) -> render_sprites.SpriteLayer | None:
    """Applies every splice in `splices` to one level's building_bboxes --
    keyed by own-tile -- and, if `sprites` is not None, its SpriteLayer.
    Mutates building_bboxes IN PLACE (callers rely on that for their own
    id()-based staleness proxies, e.g. tests/test_patch_unit_sources.py) and
    returns the (possibly new, since
    SpriteLayer is frozen) sprite layer for the caller to reassign -- `None`
    in, `None` back out, unchanged, when sprites are disabled. The
    SpriteLayer's four dicts are copied ONCE per call, not once per splice:
    a batch of N units pays one whole-dict copy, which is the point of the
    batch form.

    A splice with old_tiles == new_tiles and old_own_tile == new_own_tile
    is a RE-ANCHOR: the unit did not move, but something its placement
    reads did (an elevation edit under it). Same delete-then-recompute
    either way.

    Only valid under a guard that makes every touched key (the unit's
    own-tile for building_bboxes, its per-piece slot tiles, all inside its
    footprint, for by_anchor/bboxes/farm_by_tile) belong to the batch's own
    units alone -- _batch_splice_eligible() for unit edits,
    _reassign_component_splices() for a Convert on shared tiles,
    _membership_component_splices() for any other unit edit on shared tiles
    (removals, then the rest in walk order), _elevation_splices() (the same
    component) for elevation edits -- so each key is a plain
    delete-then-recompute rather than a subtraction from a union with unknown
    other contributors.

    Two phases, so several splices may share a key: every splice's old keys
    are cleared first, then every contribution is re-added in list order
    through render_sprites._accumulate_contribution(), the wholesale walk's own merge.
    A list in the walk's order (player-major, then list index) therefore
    rebuilds a shared key exactly: by_anchor's concatenation order, both bbox
    unions and farm_by_tile's later-unit-wins. The building part at each key
    is the union over every splice whose new own tile it is, like
    render_bbox._building_bboxes_iso(). _removals_first() is redundant under this
    order but harmless.

    An add (no old tiles) clears its NEW tiles in phase 1, which the guard
    above already makes safe. A level built from the live lists between a
    model-only add and its splice (the Unit Brush's window, GH #170: a zoom
    onto a stale level, or a pending flush seeded from the live own-tile
    index) already draws the unit, and the append alone would draw it twice.
    The same clear-then-recompute is what makes a removal or a reassign
    idempotent there.

    overrides is passed to render_sprites._resolve_unit_sprite() as-is. {} is fine
    only when no splice is a wall. A reassign, an elevation re-anchor and a
    membership component with a moved connector (its reshaped neighbours
    included) all admit walls and must pass the real dict (_splice_overrides()).

    The building part goes through render_bbox._building_bbox_for(), the same
    predicate render_bbox._building_bboxes_iso() uses, and only for a unit the
    filter keeps: a span test alone would drop an off-centre 1x1 mark's
    bbox, which the wholesale walk keeps.

    with_farms and tree_scale must be the OWNING CACHE's current View >
    Layers values, not the defaults: this re-resolves the edited unit, so a
    hardcoded True would quietly re-admit a farm's terrain override on the
    next move/rotate after the layer was turned off, and a hardcoded 1.0
    would re-inflate a moved tree to full size under Small Trees. hero_glow
    too: a hardcoded False would drop a moved hero's ring. wall_alpha
    likewise, or a moved wall would draw opaque.

    old_bboxes, if given, receives every building_bboxes key this call
    rewrites mapped to its value before (None if absent), for
    render_bbox.patch_bystander_grid(). Recorded at the write, so the set is exact."""
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height
    if sprites is not None:
        by_anchor = dict(sprites.by_anchor)
        bboxes = dict(sprites.bboxes)
        skip_ids = set(sprites.skip_ids)
        farm_by_tile = dict(sprites.farm_by_tile)

    # own-tile and the sprite keys are independent (sprite_anchor_tile() picks the
    # footprint tile LAST in depth order; slots and column slices pick others), so
    # every key either side could write to is collected and reconciled once at the end.
    touched_keys: set[tuple[int, int]] = set()

    # Phase 1: clear. Every OLD footprint tile, not just sprite_anchor_tile():
    # a composite's pieces carry their own depth slots across its footprint.
    for splice in splices:
        touched_keys.update(key for key in (splice.old_own_tile, splice.new_own_tile) if key is not None)
        if sprites is None:
            continue
        # An add clears its new tiles: a level rebuilt from the live lists before its splice already draws it.
        cleared = splice.new_tiles if splice.old_own_tile is None else splice.old_tiles
        for tile in cleared:
            by_anchor.pop(tile, None)
            bboxes.pop(tile, None)
            farm_by_tile.pop(tile, None)
        touched_keys.update(cleared)
        skip_ids.discard(id(splice.unit))

    # Phase 2: re-add in list order, the building part unioned per own tile.
    building_parts: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for splice in splices:
        unit = splice.unit
        if splice.new_own_tile is not None and unit_filter.matches(splice.player_id, unit):
            bbox = render_bbox._building_bbox_for(unit, w, h, proj, elevations, extra_top_px)
            if bbox is not None:
                key = splice.new_own_tile
                building_parts[key] = _union_bbox(building_parts.get(key), bbox)
        if sprites is not None and splice.new_tiles:
            contribution = render_sprites._resolve_unit_sprite(
                scenario, proj, elevations, unit_filter, corner_rise, overrides, with_farms,
                splice.player_id, splice.index, unit, tree_scale, hero_glow, wall_alpha, fold=fold,
            )
            if contribution is not None:
                render_sprites._accumulate_contribution(contribution, by_anchor, bboxes, skip_ids, farm_by_tile)
                touched_keys.update(contribution.by_anchor)

    # Phase 3: render_sprites.merge_sprite_bboxes()'s per-key union, over the touched keys only.
    for key in touched_keys:
        if old_bboxes is not None:
            old_bboxes[key] = building_bboxes.get(key)
        merged = _union_bbox(building_parts.get(key), bboxes.get(key) if sprites is not None else None)
        if merged is None:
            building_bboxes.pop(key, None)
        else:
            building_bboxes[key] = merged

    if sprites is None:
        return None
    return replace(sprites, by_anchor=by_anchor, bboxes=bboxes, skip_ids=frozenset(skip_ids), farm_by_tile=farm_by_tile)
