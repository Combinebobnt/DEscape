"""Qt-free LRU caches of composited canvas chunks, one per terrain style.

Split out of render.py, which had grown past 3700 lines. The dependency runs
one way only: this module imports render, render.py never imports it back --
every ChunkCache mention still in render.py is a comment or a docstring, not
one is code, so the cluster is an acyclic leaf. A re-export shim would have
manufactured a cycle that only survived by deferring every dereference to
call time, which is why the split landed as one commit rather than two.

Back-references reach render through ATTRIBUTE ACCESS (render.foo(...)),
never `from descape.render import foo`. That is load-bearing, not style:
tests/test_mip_geometry.py monkeypatches render._building_bboxes_iso, which
IsoChunkCache and SlopedChunkCache call, and a from-import binds at import
time and would turn that patch into a silent no-op. All 13 back-references
route through `render.` so a future patch target cannot break either.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field, replace

import numpy as np

from descape import composite_backend, iso_geometry, native_composite, perf_trace, render, settings, unit_sprites
from descape.grid_overlay import DEFAULT_GRID, GridBake
from descape.scenario_io import LoadedScenario
from descape.unit_filter import UnitFilter
from descape.view_layers import DEFAULT_LAYERS, LayerState

# LayerState fields that are baked into a SpriteLayer's own contents when it
# is BUILT, rather than read off self.layers at paint time. Flipping one needs
# the unit sources rebuilding, not just the chunks evicting -- see
# _ChunkCacheBase.set_layers(). A new build-time row belongs here; a paint-time
# one deliberately does not.
_BUILD_TIME_LAYER_FIELDS = ("farm_overlay", "small_trees", "hero_glow")

# Default chunk size for IsoChunkCache -- measured, not guessed: benched
# CHUNK_PX in {256, 512, 1024} via tools/verify_iso_chunks.py on this
# project's largest real map (480x480, tile_px=32). 1024 wins cold
# full-canvas assembly by only ~8% over 512 (4.7s vs 5.1s) while being a
# much coarser invalidation granularity for a moving viewport; 256 loses on
# both cold assembly (~6s) and warm single-tile patch cost (~44ms vs ~28ms).
DEFAULT_CHUNK_PX = 512

# Unit-pack tiles derived per pack_warm_job() step. Measured 2026-09-27 at
# 0.9-1.5us per occupied tile (Dos Pilas the densest), so a slice is ~1-1.5ms dev.
PACK_WARM_TILES = 1024


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

    old_tiles/new_tiles are render.unit_occupied_tiles()'s own ORDERED
    result, pre- and post-edit. Order matters: unit_sprites.
    sprite_anchor_tile() picks a specific tile out of this list, and it
    must see exactly the list a full rebuild would have used.

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
        gate orientation swap. For a wall/connector that changes its tiles
        only. set_position() keeps rotation and list order, and no fields-only
        edit writes a rotation-variant const's rotation (none is ANGLE or
        cyclable), so the file's radian verdict and every override key hold.
        What changes is the neighbour mask of the walls beside its old and
        new tiles (_connector_neighbours())."""
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
      (render.wall_variant_rotation_overrides' own docstring): moving one
      reshapes up to four other units this splice never touches, and the
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
    `const:add` or `const:remove` (either can flip the file's radian verdict
    and shift override keys, so wholesale), `const:other` (an owner change
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


def _batch_splice_refusal(units_by_tile: dict, changed: list[UnitSplice]) -> str | None:
    """Which clause of _batch_splice_eligible() refuses `changed`, or None if
    none does: `cap`, then the const clause over the whole batch
    (_const_refusal()), then the first of `occupant:foreign` (a unit outside
    the batch), `occupant:batch` (a batch unit that stays, e.g. one a group
    Move carries off the tile another lands on) and `claimed` (two batch
    units landing on one tile). So an occupant or claimed answer means the
    batch holds no wall/connector const but a reassigned one. `const:move`
    and those three go on to a component splice (_splice_plan_or_refusal());
    `cap` and the other const answers are final."""
    if len(changed) > UNIT_SPLICE_MAX_UNITS:
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
            occupied = render.unit_occupied_tiles(u, w, h)
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
    scenario: LoadedScenario, units_by_tile: dict, unit_filter: UnitFilter, changed: list[UnitSplice]
) -> tuple[list[UnitSplice] | None, str | None]:
    """A Convert batch that _batch_splice_eligible() refused for a shared tile,
    widened to its _shared_tile_component(), or None and why (`cap`, `none`)
    when that can't be done.

    A reassign moves no unit, so the component argument holds. Batch units
    seed it at their new tiles and are always kept, so a filter flip is
    covered. Each unit is spliced at its CURRENT (player_id, i) as a
    re-anchor, batch units keeping old_player_id.

    None unless every splice is a pure reassign, or past
    _COMPONENT_SPLICE_MAX_UNITS component units."""
    if not all(s.is_reassign for s in changed):
        return None, "none"
    if len(changed) > _COMPONENT_SPLICE_MAX_UNITS:
        return None, "cap"
    own_index = render.unit_own_tile_index(scenario)
    batch = {id(s.unit): s for s in changed}
    seeds = []
    for s in changed:
        where = _locate(own_index, s.unit)
        if where is None:
            return None, "none"
        seeds.append((*where, s.unit, s.new_own_tile, s.new_tiles))
    tiles = [t for seed in seeds for t in (*seed[4], seed[3]) if t is not None]
    found, why = _walk_component(
        scenario, units_by_tile, unit_filter, own_index, tiles, set(batch), _COMPONENT_SPLICE_MAX_UNITS - len(seeds)
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
    tiles). render._wall_variant_rotation_overrides_uncached() reads a
    wall's 4-bit orthogonal mask over every connector's tiles, at its bounds'
    low corner, so a moved connector reshapes only the walls whose low corner
    is orthogonally beside one of its old or new tiles. Those tiles are
    dilated by NEIGHBOUR_OFFSETS[:4] (the tiles themselves kept too, which
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
            dilated.update((x + dx, y + dy) for dx, dy, _bit in unit_sprites.NEIGHBOUR_OFFSETS[:4])
    out: list[tuple] = []
    seen = set(known)
    for tile in dilated:
        for player_id, i, unit in own_index.get(tile, ()):
            if id(unit) in seen or not unit_sprites.rotation_variant_eligible(unit.unit_const):
                continue
            seen.add(id(unit))
            bounds = render.unit_tile_bounds(unit, w, h)
            if bounds is None or (bounds[0], bounds[2]) not in dilated or not unit_filter.matches(player_id, unit):
                continue
            occupied = render.unit_occupied_tiles(unit, w, h)
            if occupied is not None:
                out.append((player_id, i, unit, (int(unit.x), int(unit.y)), tuple(occupied)))
    return out


def _membership_component_splices(
    scenario: LoadedScenario, units_by_tile: dict, unit_filter: UnitFilter, changed: list[UnitSplice]
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
    past _COMPONENT_SPLICE_MAX_UNITS units in total, neighbours included."""
    if any(not s.is_reassign and not s.is_move and not _const_splice_eligible(s.unit) for s in changed):
        return None, "const"
    if len(changed) > _COMPONENT_SPLICE_MAX_UNITS:
        return None, "cap"
    own_index = render.unit_own_tile_index(scenario)
    known = {id(s.unit) for s in changed}
    neighbours = _connector_neighbours(scenario, unit_filter, own_index, changed, known)
    room = _COMPONENT_SPLICE_MAX_UNITS - len(changed) - len(neighbours)
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
    scenario: LoadedScenario, units_by_tile: dict, unit_filter: UnitFilter, changed: list[UnitSplice]
) -> tuple[list[UnitSplice] | None, str | None]:
    """_splice_plan()'s answer paired with why it is None: the refusing
    _batch_splice_refusal() clause, plus `/component:<why>` when the component
    fallback was tried and failed too. Perf Trace's `splice_refused=<reason>`
    token (_trace_splice_refusal()): `cap`, `const:add`, `const:remove`,
    `const:other`, or `<clause>/component:<why>` where the clause is
    `const:move`, `occupant:*` or `claimed`."""
    reason = _batch_splice_refusal(units_by_tile, changed)
    if reason is None:
        return changed, None
    if reason == "cap" or (reason.startswith("const:") and reason != "const:move"):
        return None, reason
    component = _reassign_component_splices if all(s.is_reassign for s in changed) else _membership_component_splices
    plan, why = component(scenario, units_by_tile, unit_filter, changed)
    return plan, None if plan is not None else f"{reason}/component:{why}"


def _splice_overrides(scenario: LoadedScenario, changed: list[UnitSplice]) -> dict[tuple[int, int], float]:
    """The wall overrides a unit-edit splice must resolve with: the real dict
    when any spliced unit reads it (a reassigned or moved wall, or one a
    moved connector reshapes), else {} to skip the lookup. Memoized on
    scenario.unit_gen, which reassign() and set_position() bump."""
    if any(unit_sprites.rotation_variant_eligible(s.unit.unit_const) for s in changed):
        return render.wall_variant_rotation_overrides(scenario)
    return {}


def _removals_first(changed: list[UnitSplice]) -> list[UnitSplice]:
    """`changed` with its removals moved to the front, order otherwise kept.
    A removal clears every slot on its old tiles, so one applied after an add
    onto the same tile would delete that add's contribution."""
    return [s for s in changed if s.new_own_tile is None] + [s for s in changed if s.new_own_tile is not None]


# Past this many re-anchored units an elevation patch takes the wholesale path
# instead. Measured on June Event (14,106 units, sprites on): _reanchor_units
# costs ~1ms + 13us/unit and the bystander-grid patch 0.75ms at 132 keys, 1.8ms
# at 1000 (a full grid build is ~5.9ms), so ~16ms at 1000 against a 90-320ms
# wholesale rebuild. A real brush-9 stroke step re-anchors <=132.
_ELEV_SPLICE_MAX_UNITS = 1000


def _elevation_splices(
    scenario: LoadedScenario, units_by_tile: dict, unit_filter: UnitFilter, tiles
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

    None past _ELEV_SPLICE_MAX_UNITS members in total, where the wholesale
    rebuild is the cheaper of the two."""
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height
    own_index = render.unit_own_tile_index(scenario)
    seeds = []
    for own in tiles:
        for player_id, i, unit in own_index.get(own, ()):
            if not unit_filter.matches(player_id, unit):
                continue
            occupied = render.unit_occupied_tiles(unit, w, h)
            if occupied is not None:
                seeds.append((player_id, i, unit, own, tuple(occupied)))
    members = _shared_tile_component(scenario, units_by_tile, unit_filter, own_index, seeds, _ELEV_SPLICE_MAX_UNITS)
    if members is None:
        return None
    return [UnitSplice(p, i, unit, own, own, occ, occ) for p, i, unit, own, occ in members]


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


def _refreshed_pack(pack, splices, units_by_tile, old_sprites, old_version, old_heights, sprites, version, heights):
    """pack refreshed over the batch's tiles, or None to force a full rebuild
    when it didn't describe the pre-edit sources (a refresh can't repair a
    pack that was already stale)."""
    if pack is None or not pack.matches(units_by_tile, old_sprites, old_version, old_heights):
        return None
    pack.refresh(_splice_tiles(splices), units_by_tile, sprites, version, heights)
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


def _drop_from_tiles(units_by_tile: dict, splice: UnitSplice) -> None:
    """Removes splice.unit from every bucket it used to occupy, deleting a
    bucket that empties (an empty list and a missing key are not the same
    thing to render._units_by_tile()'s consumers)."""
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
    """The O(footprint) alternative to render._units_by_tile()'s O(all
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
        entry = (splice.unit, render._unit_color(splice.unit, player_color))
        for tile in splice.new_tiles:
            units_by_tile.setdefault(tile, []).append(entry)


def _splice_building_and_sprites(
    building_bboxes: dict,
    sprites: render.SpriteLayer | None,
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
) -> render.SpriteLayer | None:
    """_reanchor_units() for a single splice, with overrides {}. {} is only
    right for a splice of a unit that isn't rotation-variant-eligible, whose
    resolve never consults the dict (render._wall_variant_rotation_overrides_
    uncached's own candidate filter). A wall, reassigned or moved (the
    membership component splice), must go through _reanchor_units() with
    _splice_overrides() instead, as both invalidate_units() do."""
    return _reanchor_units(
        building_bboxes, sprites, scenario, proj, elevations, unit_filter, corner_rise,
        extra_top_px, [splice], {}, with_farms, tree_scale, hero_glow,
    )


def _reanchor_units(
    building_bboxes: dict,
    sprites: render.SpriteLayer | None,
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
    old_bboxes: dict | None = None,
) -> render.SpriteLayer | None:
    """Applies every splice in `splices` to one cache's (or, for Iso, one
    level's) building_bboxes -- keyed by own-tile -- and, if `sprites` is
    not None, its SpriteLayer. Mutates building_bboxes IN PLACE (callers
    rely on that for their own id()-based staleness proxies, e.g.
    SlopedChunkCache._set_building_bboxes / tests/
    test_patch_unit_sources.py) and returns the (possibly new, since
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
    through render._accumulate_contribution(), the wholesale walk's own merge.
    A list in the walk's order (player-major, then list index) therefore
    rebuilds a shared key exactly: by_anchor's concatenation order, both bbox
    unions and farm_by_tile's later-unit-wins. The building part at each key
    is the union over every splice whose new own tile it is, like
    render._building_bboxes_iso(). _removals_first() is redundant under this
    order but harmless.

    overrides is passed to render._resolve_unit_sprite() as-is. {} is fine
    only when no splice is a wall. A reassign, an elevation re-anchor and a
    membership component with a moved connector (its reshaped neighbours
    included) all admit walls and must pass the real dict (_splice_overrides()).

    The building part goes through render._building_bbox_for(), the same
    predicate render._building_bboxes_iso() uses, and only for a unit the
    filter keeps: a span test alone would drop an off-centre 1x1 mark's
    bbox, which the wholesale walk keeps.

    with_farms and tree_scale must be the OWNING CACHE's current View >
    Layers values, not the defaults: this re-resolves the edited unit, so a
    hardcoded True would quietly re-admit a farm's terrain override on the
    next move/rotate after the layer was turned off, and a hardcoded 1.0
    would re-inflate a moved tree to full size under Small Trees. hero_glow
    too: a hardcoded False would drop a moved hero's ring.

    old_bboxes, if given, receives every building_bboxes key this call
    rewrites mapped to its value before (None if absent), for
    render.patch_bystander_grid(). Recorded at the write, so the set is exact."""
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height
    if sprites is not None:
        by_anchor = dict(sprites.by_anchor)
        bboxes = dict(sprites.bboxes)
        skip_ids = set(sprites.skip_ids)
        farm_by_tile = dict(sprites.farm_by_tile)

    # own-tile and sprite-anchor-tile are independent keys (sprite_anchor_tile()
    # picks the footprint tile LAST in depth order), so every key either side
    # could write to is collected and reconciled once at the end.
    touched_keys: set[tuple[int, int]] = set()

    # Phase 1: clear. Every OLD footprint tile, not just sprite_anchor_tile():
    # a composite's pieces carry their own depth slots across its footprint.
    for splice in splices:
        touched_keys.update(key for key in (splice.old_own_tile, splice.new_own_tile) if key is not None)
        if sprites is None:
            continue
        for tile in splice.old_tiles:
            by_anchor.pop(tile, None)
            bboxes.pop(tile, None)
            farm_by_tile.pop(tile, None)
        touched_keys.update(splice.old_tiles)
        skip_ids.discard(id(splice.unit))

    # Phase 2: re-add in list order, the building part unioned per own tile.
    building_parts: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for splice in splices:
        unit = splice.unit
        if splice.new_own_tile is not None and unit_filter.matches(splice.player_id, unit):
            bbox = render._building_bbox_for(unit, w, h, proj, elevations, extra_top_px)
            if bbox is not None:
                key = splice.new_own_tile
                building_parts[key] = _union_bbox(building_parts.get(key), bbox)
        if sprites is not None and splice.new_tiles:
            contribution = render._resolve_unit_sprite(
                scenario, proj, elevations, unit_filter, corner_rise, overrides, with_farms,
                splice.player_id, splice.index, unit, tree_scale, hero_glow,
            )
            if contribution is not None:
                render._accumulate_contribution(contribution, by_anchor, bboxes, skip_ids, farm_by_tile)
                touched_keys.update(contribution.by_anchor)

    # Phase 3: render.merge_sprite_bboxes()'s per-key union, over the touched keys only.
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


@dataclass
class LevelWarmJob:
    """One mip level's warm, handed to level_warm.LevelWarmer: a resumable
    walk plus the closure that installs its result on THIS cache. The walk
    builds the level's sprite layer (level_warm_job) or derives its unit
    pack's pending tiles in place (pack_warm_job, whose install is a no-op).

    The split is what keeps Qt out of this module. The driver owns pacing
    (a QTimer and a wall-clock budget) and knows nothing about what a level
    is; the cache owns the walk and the install semantics -- including the
    validity predicate, which differs per style and is the whole correctness
    argument for installing a layer built across several event-loop turns.

    install() returns True if it landed, False if the job was revalidated
    away (the source mutated under the warm, or a real paint already built
    the level). A False is a normal outcome, not an error."""

    gen: object
    install: object
    # Perf Trace's label for this job's steps in a level-warm tick.
    kind: str = "walk"


@dataclass
class ChunkJob:
    """One chunk's composite with every Python step already done on the GUI
    thread, so run() is a single nogil kernel call safe on a worker thread
    (Batch F T2, margin_warm). The cache installs the result only through
    install_chunk(), which rejects it if anything mutated since `epoch`."""

    key: tuple[int, int, int]
    epoch: int
    kernel: object
    scratch: np.ndarray
    args: tuple
    # Owners of memory args reach only by address (the unit pack's sprites).
    holds: tuple = ()

    def run(self) -> bool:
        """Paints scratch. False: the kernel declined and painted nothing."""
        return self.kernel(self.scratch, *self.args) is not False


class _ChunkCacheBase:
    """Grid/LRU bookkeeping shared by IsoChunkCache (Stepped, Phase B-B) and
    FlatChunkCache (Flat, Phase B-E) -- extracted because get_chunk/
    render_rect/patch/invalidate_region are pure chunk-grid arithmetic with
    zero mode-specific content, proven correct by tools/verify_iso_chunks.py's
    own byte-identity checks well before this split existed. This is NOT a
    weakening of composite_rect_iso()/composite_rect_flat()'s own "stay an
    independent implementation, never re-expressed in terms of the chunk
    path" rule -- that rule is about the COMPOSITOR (only ever reached here
    through the subclass's _composite_rect() hook, still two genuinely
    separate functions); this class owns LRU/grid bookkeeping only, the same
    thing any other chunk cache would.

    A subclass must, in its own __init__ (kept fully subclass-owned, not
    called from here, so each cache's own constructor signature/docstring
    stays exactly as-is): set self.chunk_px, call self._init_mip_levels(...)
    (Phase B-D-a; the level table must exist before _init_max_chunks() can
    read canvas_dims()), call self._refresh_source_caches() once, then call
    self._init_max_chunks(chunk_px, max_chunks). It must also implement:
      - canvas_dims(mip=0) -> (width, height) in LEVEL canvas pixels
      - _composite_rect(mip, x0, y0, x1, y1) -> (h, w, 3) uint8 array
      - _refresh_source_caches() -> None
      - a `style` class attribute (one of terrain_style.TERRAIN_STYLES:
        "flat" / "stepped" / "sloped") -- checked at the
        MapView.set_source() boundary (Phase B-E) to catch a cache wired to
        the wrong terrain style at construction time, rather than only once
        an edit exposes the mismatch later.

    Coordinate-space convention (Phase B-D-a, not stated explicitly
    elsewhere): render_rect()/get_chunk()/canvas_dims() all take
    LEVEL pixels/indices -- a caller past this class (Phase B-D-c's paint())
    is expected to already know which level it's asking for. patch()/
    invalidate_region() instead take REFERENCE canvas pixels, because their
    only real callers (ViewerWindow._apply_dirty, via dirty_screen_bbox_iso
    and a locally-recomputed reference tile_px) have no notion of levels at
    all -- Track B-D-a/b deliberately never touch viewer.py. _bbox_to_level()
    is the one conversion point between the two spaces.

    mip is part of every cache key (Phase B-B); Phase B-D-a makes get_chunk()/
    render_rect() actually use it for the pixel math (previously always 0),
    and patch()/invalidate_region() fan out across every RESIDENT level
    (not every enumerated one -- see _init_mip_levels' docstring) instead of
    hardcoding chunk 0."""

    style: str = ""
    # P3-g. Per-INSTANCE state with a class-level default, so every subclass
    # has the attribute without each __init__ having to set it. A class default
    # is safe here only because it is an immutable bool that set_sprites_enabled
    # rebinds on the instance; never give a mutable one this treatment.
    sprites_enabled: bool = False
    # View > Grid's baked spec, read by every _composite_rect beside
    # self.layers. Same class-default reasoning: frozen, rebound per instance.
    grid: GridBake = DEFAULT_GRID
    # Bumped by every mutating entry point, so a ChunkJob composited before
    # one is never installed after it (install_chunk). Same immutable-default
    # reasoning as above.
    _mutation_epoch: int = 0
    # Whether _unit_pack_of() exists: Stepped and Sloped keep a native unit pack.
    _has_unit_pack: bool = False
    # Flat int64 terrain id per tile (y * w + x) for the native composite, or
    # None (Flat, numpy backend). The default is None, so rebinding is safe.
    _terrain_ids: np.ndarray | None = None

    def _init_mip_levels(self, tile_px_by_level: dict[int, int]) -> None:
        """Enumerates the level set ONCE, at construction -- never lazily.
        Level 0 must be present and must equal self.tile_px (D2: scene
        space is pinned to the reference level, permanently).

        Phase B-D-a passes a literal {0: self.tile_px} (no other levels
        exist yet); Phase B-D-b replaces that call with a real per-level
        set from iso_geometry.mip_projections_for()/mip_tile_px_candidates().
        Enumerating eagerly (geometry only -- tile_px/proj are cheap) while
        leaving PIXELS and per-level source state lazy is what phase b's
        non-tautology byte-identity test depends on: the level set must
        already be fixed before that test's ground-truth call gets its
        tile_pixels_for_map() monkeypatched.

        Level index L means tile_px = reference_tile_px * 2**L: POSITIVE L
        is FINER (mip-up), NEGATIVE L is COARSER (mip-down) -- the only
        reading consistent with mip_for_scale()'s
        clamp(ceil(log2(scale)), ...) rule, since zooming in raises scale
        and must raise the selected level."""
        assert tile_px_by_level.get(0) == self.tile_px, (
            f"level 0 must be the reference tile_px ({self.tile_px}), got {tile_px_by_level.get(0)!r}"
        )
        self._mip_tile_px: dict[int, int] = dict(sorted(tile_px_by_level.items()))

    def mip_levels(self) -> list[int]:
        """Every enumerated level index, ascending. Always contains 0.
        Length 1 is a normal case, not a degenerate one -- e.g. a reference
        tile_px of 16 at elev_step_pct=10 has no exact neighbor at all
        (measured, see iso_geometry.mip_projections_for's own docstring)."""
        return list(self._mip_tile_px)

    def mip_tile_px(self, mip: int = 0) -> int:
        return self._mip_tile_px[mip]

    def mip_scale(self, mip: int = 0) -> float:
        """Scene-space scale factor for this level: reference_tile_px /
        level_tile_px == 2**-mip. Both operands are always powers of two in
        [MIP_MIN_TILE_PIXELS, MIP_MAX_TILE_PIXELS], so this division is
        exactly representable in binary float -- mip_scale(0) == 1.0 is an
        EXACT comparison, which is what Phase B-D-c's "keep the point
        overload at S == 1.0" safety branch relies on."""
        return self._mip_tile_px[0] / self._mip_tile_px[mip]

    def mip_for_scale(self, scale: float) -> int:
        """The coarsest level whose own scene-to-device magnification stays
        <= 1 (never magnify past what's actually resident): clamp(ceil(
        log2(scale)), min(mip_levels()), max(mip_levels())). Residual
        magnification lands in (0.5, 1.0] whenever the derived level is
        actually available -- exactly 1.0 at a power-of-two scale (no
        resampling at all), which is what keeps a unit device transform on
        the reference level (see mip_scale()'s own docstring). scale <= 0
        is guarded by returning the coarsest available level rather than
        raising -- a defensive floor for a degenerate transform, not a
        case Phase B-D-c's real callers are expected to hit.

        2026-08-28: ceil, not the naively-symmetric floor(log2(scale)) + 1
        -- they agree everywhere except exact powers of two, where ceil
        picks the coarser (better) of the two and keeps scale=1.0 selecting
        level 0 exactly. That exactness is load-bearing: testkit.
        qt_capture.scene_rect_to_array's whole 1:1 byte-identity contract
        depends on a unit device transform always landing on the reference
        level, and floor(...) + 1 would break it."""
        levels = self.mip_levels()
        if scale <= 0:
            return levels[0]
        raw = math.ceil(math.log2(scale))
        return max(levels[0], min(levels[-1], raw))

    def _bbox_to_level(self, mip: int, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """A REFERENCE-canvas-pixel bbox converted to level `mip` pixels.
        All-integer and superset-safe in BOTH directions: floor the low
        edge, ceil the high edge, so the level rect is never a strict
        subset of the true footprint (a subset would leave stale pixels
        behind). Exact because canvas dims scale by exactly
        level_tile_px/reference_tile_px -- proven per level by
        is_exact_mip (Phase B-D-b), asserted on canvas_dims() itself
        (the value the blit actually trusts) at construction.

        Identity whenever mip's tile_px equals the reference's -- true for
        every mip in Phase B-D-a, since the level set is still {0}."""
        t_ref, t_lvl = self._mip_tile_px[0], self._mip_tile_px[mip]
        if t_lvl == t_ref:
            return bbox
        px0, py0, px1, py1 = bbox
        return (
            (px0 * t_lvl) // t_ref,
            (py0 * t_lvl) // t_ref,
            -((-px1 * t_lvl) // t_ref),
            -((-py1 * t_lvl) // t_ref),
        )

    def _level_bbox_to_reference(self, mip: int, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """The inverse of _bbox_to_level(): a level-`mip`-pixel bbox in
        REFERENCE canvas pixels, floor low and ceil high by t_ref/t_lvl, so
        _bbox_to_level() of the result is always a superset of `bbox`."""
        t_ref, t_lvl = self._mip_tile_px[0], self._mip_tile_px[mip]
        if t_lvl == t_ref:
            return bbox
        lx0, ly0, lx1, ly1 = bbox
        return (
            (lx0 * t_ref) // t_lvl,
            (ly0 * t_ref) // t_lvl,
            -((-lx1 * t_ref) // t_lvl),
            -((-ly1 * t_ref) // t_lvl),
        )

    def resident_levels(self) -> list[int]:
        """Every mip holding at least one cached chunk, ascending: the levels
        patch()/invalidate_region() fan out over when given no `levels`."""
        return sorted({key[0] for key in self._cache})

    def _select_levels(self, levels) -> list[int]:
        """resident_levels(), narrowed to `levels` when that is not None."""
        resident = self.resident_levels()
        if levels is None:
            return resident
        wanted = set(levels)
        return [mip for mip in resident if mip in wanted]

    def _layer_resolve_args(self) -> tuple[bool, float, bool]:
        """(with_farms, tree_scale, hero_glow) for render._resolve_unit_sprite(),
        from this cache's View > Layers state. The one source for every
        _reanchor_units() call and sprite_extent_after(), so they cannot differ."""
        return self.layers.farm_overlay, self.layers.tree_scale, self.layers.hero_glow

    def _unit_level_state(self, mip: int):
        """(proj, sprites, building_bboxes, corner_rise, extra_top_px) for level
        `mip`, as currently installed: never rebuilt by this call. Stepped and
        Sloped implement it; Flat has no per-unit sprite extent and raises."""
        raise NotImplementedError(f"{type(self).__name__} has no unit-edit sprite extent")

    def _extent_fallback(self, changed: list[UnitSplice], mip: int | None) -> bool:
        """Whether a unit edit must be sized with today's reach bbox rather
        than sprite_extent_before()/after(): a moved or added wall, connector
        or rotation-variant const (its neighbours' art changes too), or a
        visible level with no resident chunks, or a level whose unit state was
        never built. A pure reassign of one is exempt: it moves nothing, so no
        neighbour mask changes (see _splice_eligible). Also a level with
        deferred elevation work: its layer is behind, and flushing here would
        read units the caller already moved against pre-edit buckets."""
        if any(not s.is_reassign and not _const_splice_eligible(s.unit) for s in changed):
            return True
        if mip is None or mip not in self.resident_levels() or self._defers_patch(mip):
            return True
        _proj, sprites, building_bboxes, _rise, _top = self._unit_level_state(mip)
        if building_bboxes is None:
            return True
        return self.with_units and self.sprites_enabled and sprites is None

    def sprite_extent_before(self, changed: list[UnitSplice], mip: int | None):
        """What `changed`'s units paint at level `mip` before the edit, as a
        REFERENCE-canvas bbox: None (nothing), or REACH_FALLBACK (see
        _extent_fallback()). Must run before invalidate_units(changed), which
        replaces the layer this reads.

        Read from the level's own layer: its sprite bboxes on every old
        footprint tile (a contribution's keys all lie inside its unit's
        footprint) plus the building bbox at the old own tile, which is the
        mark with the sprite already merged in. A tile shared with another
        unit over-covers, which is safe.

        A STALE Stepped level (gen behind _source_gen) is read as is, with no
        fallback. Every composite at a level, paint or margin-warm job, goes
        through _level(), which rebuilds a stale layer first; so a resident
        chunk at a stale level never shows a unit state newer than its layer.
        A unit changed since the layer was built was repainted at that level
        by its own edit, whose bbox covered its old and new extents, and
        evicting leaves nothing resident that shows the newer state. What
        the stale layer holds at these tiles is then either exactly what is
        on screen or over-coverage."""
        if self._extent_fallback(changed, mip):
            return REACH_FALLBACK
        _proj, sprites, building_bboxes, _rise, _top = self._unit_level_state(mip)
        box = None
        for s in changed:
            if sprites is not None:
                for tile in s.old_tiles:
                    box = _union_bbox(box, sprites.bboxes.get(tile))
            if s.old_own_tile is not None:
                box = _union_bbox(box, building_bboxes.get(s.old_own_tile))
        return None if box is None else self._level_bbox_to_reference(mip, box)

    def sprite_extent_after(self, changed: list[UnitSplice], mip: int | None):
        """sprite_extent_before()'s counterpart after invalidate_units(changed):
        what the edited units paint at level `mip` now. Resolved per unit
        with _reanchor_units()' own arguments rather than read from the layer,
        since a Stepped level the wholesale fallback left stale has no
        post-edit layer until its next composite. overrides are
        _splice_overrides()': the real dict when a reassigned wall reads it,
        else {}. A moved one takes REACH_FALLBACK even though its sources now
        splice: its reshaped neighbours are not in `changed`."""
        if self._extent_fallback(changed, mip):
            return REACH_FALLBACK
        proj, _sprites, _bb, corner_rise, extra_top = self._unit_level_state(mip)
        scenario = self.scenario
        mm = scenario.map_manager
        with_farms, tree_scale, hero_glow = self._layer_resolve_args()
        overrides = _splice_overrides(scenario, changed)
        box = None
        for s in changed:
            if s.new_own_tile is None or not self.with_units:
                continue
            if self.sprites_enabled:
                contribution = render._resolve_unit_sprite(
                    scenario, proj, self.elevations, self.unit_filter, corner_rise, overrides, with_farms,
                    s.player_id, s.index, s.unit, tree_scale, hero_glow,
                )
                if contribution is not None:
                    for bbox in contribution.bboxes.values():
                        box = _union_bbox(box, bbox)
            if self.unit_filter.matches(s.player_id, s.unit):
                box = _union_bbox(
                    box, render._building_bbox_for(s.unit, mm.map_width, mm.map_height, proj, self.elevations, extra_top)
                )
        return None if box is None else self._level_bbox_to_reference(mip, box)

    def _init_max_chunks(self, chunk_px: int, max_chunks: int | None) -> None:
        """Whole-canvas-at-chunk_px default, not some smaller fixed
        constant: the viewer always fitInView()s the full map on open, so
        the default/steady-state working set IS every chunk. A smaller cap
        would silently thrash (evict chunks the very next full repaint
        needs again) instead of ever reaching a warm, blit-only steady
        state -- the exact failure mode a chunk cache exists to avoid. See
        each subclass's own docstring for the measured memory cost of this
        default.

        Phase B-D-a: max_chunks explicitly passed keeps capping by COUNT
        with no byte bound at all (preserves every existing caller that
        passes one, e.g. tools/verify_iso_chunks.py's max_chunks=2/100000
        eviction tests, with no test edits); max_chunks=None (the real
        app's default) switches to a BYTE budget sized at exactly today's
        admitted set (canvas_dims() area * 3), tracked incrementally in
        self._cache_bytes rather than recomputed per eviction check. A
        byte budget is not itself the mechanism that stops N resident mip
        levels multiplying memory -- with chunk_px constant in LEVEL
        pixels, a byte budget is approximately a count budget too; what
        actually bounds total memory is the single shared global LRU
        below, unchanged, evicting across every level's chunks under one
        shared bound. The byte form matters for exactness on the ragged
        last chunk at each level, and because "N levels redistribute
        memory, they don't multiply it" is a claim about bytes."""
        self.chunk_px = chunk_px
        self._cache: OrderedDict[tuple[int, int, int], np.ndarray] = OrderedDict()
        self._cache_bytes = 0
        if max_chunks is None:
            canvas_w, canvas_h = self.canvas_dims()
            self.max_chunks: int | None = None
            self.max_bytes: int | None = canvas_w * canvas_h * 3
        else:
            self.max_chunks = max_chunks
            self.max_bytes = None

    def canvas_dims(self, mip: int = 0) -> tuple[int, int]:
        raise NotImplementedError

    def _composite_rect(self, mip: int, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        raise NotImplementedError

    def _refresh_source_caches(self, elevation_changed: set | None = None, splice_levels=None) -> None:
        """splice_levels: patch()'s rebuild_levels, the levels allowed to do
        per-level source work now. Only IsoChunkCache's elevation splice reads it."""
        raise NotImplementedError

    def _defers_patch(self, mip: int) -> bool:
        """Whether a resident level must evict patch()'s chunks rather than
        composite them: its unit layer has deferred work (IsoChunkCache)."""
        return False

    def can_splice(self, changed: list[UnitSplice] | None) -> bool:
        """Whether invalidate_units(changed) would splice rather than rebuild,
        for a caller whose repaint shape depends on it (a fallback's eager
        patch() pays the level rebuild in the handler). Base: never."""
        return False

    def _unit_splice_plan(self, changed: list[UnitSplice] | None) -> list[UnitSplice] | None:
        """_splice_plan() over this cache's own sources (Stepped/Sloped), or
        None for the wholesale path. Not memoized: the viewer never asks
        can_splice() before a Stepped/Sloped invalidate_units()."""
        if not self.with_units or changed is None:
            return None
        return _splice_plan(self.scenario, self.units_by_tile, self.unit_filter, changed)

    def _trace_splice_refusal(self, changed: list[UnitSplice] | None, skipped: bool = False) -> None:
        """Records why invalidate_units(changed) is about to go wholesale, as a
        zero-length `splice_refused=<reason>` phase on the open op or stroke end.
        Recomputed only while Perf Trace is on, and before the sources change.
        `skipped`: the caller passed splice_levels=False, so the reason is `skipped`."""
        if changed is None or not self.with_units or not perf_trace.is_enabled():
            return
        reason = "skipped" if skipped else _splice_plan_or_refusal(
            self.scenario, self.units_by_tile, self.unit_filter, changed
        )[1]
        with perf_trace.phase(f"splice_refused={reason}"):
            pass

    def _update_units_by_tile_in_place(self, changed: list[UnitSplice]) -> bool:
        """units_by_tile brought up to date for a membership-only batch without
        render._units_by_tile()'s walk over every unit, and _units_version
        bumped (a unit pack's identity check can't see an in-place mutation).
        False, touching nothing, unless units are on and every splice is a pure
        add (old_own_tile None) or a pure removal (new_own_tile None), no unit is
        on both sides, and each player's adds are exactly the last k units of
        its list, in `changed` order (FlatChunkCache._flat_splice_eligible()'s
        tail check). The caller then takes the wholesale rebuild.

        Why this equals a fresh build: buckets are ordered player-major, then
        by list index. Dropping removals by identity (_drop_from_tiles) keeps
        that order. A tail add has the highest index for its player, so its
        fresh-build slot is right after its player's run: after the last entry
        whose owner is <= its player. The owner of an existing entry is read as
        `unit.player`, which UnitManager keeps equal to the list a unit sits in
        (add, reassign and restore all resync it). A filtered-out or off-map
        add gets no entry, as in _units_by_tile()."""
        if not self.with_units or not changed:
            return False
        adds: dict[int, list[UnitSplice]] = {}
        removals: list[UnitSplice] = []
        for s in changed:
            if s.old_own_tile is None:
                adds.setdefault(s.player_id, []).append(s)
            elif s.new_own_tile is None:
                removals.append(s)
            else:
                return False
        if {id(s.unit) for s in removals} & {id(s.unit) for group in adds.values() for s in group}:
            return False
        units = self.scenario.unit_manager.units
        for player_id, group in adds.items():
            start = len(units[player_id]) - len(group)
            for k, s in enumerate(group):
                if s.index != start + k or units[player_id][s.index] is not s.unit:
                    return False
        self._units_version += 1
        by_tile = self.units_by_tile
        for s in removals:
            _drop_from_tiles(by_tile, s)
        for player_id in sorted(adds):
            color = self.scenario.player_colors[player_id]
            for s in adds[player_id]:
                if not s.new_tiles or not self.unit_filter.matches(player_id, s.unit):
                    continue
                entry = (s.unit, render._unit_color(s.unit, color))
                for tile in s.new_tiles:
                    bucket = by_tile.setdefault(tile, [])
                    pos = len(bucket)
                    while pos and bucket[pos - 1][0].player > player_id:
                        pos -= 1
                    bucket.insert(pos, entry)
        return True

    def invalidate_units(self, changed: list[UnitSplice] | None = None, splice_levels: bool = True) -> None:
        """Forces this cache's unit-derived structures to rebuild on the next
        composite -- the explicit hook for a unit-editing mutation (place/
        move/delete/reassign), which changes unit.x/y/player without going
        through set_unit_filter()'s own unchanged-filter guard.

        changed (Batch D's D4): None (the default, and every pre-D4 caller's
        exact behaviour) means "unknown" and takes this base wholesale path
        unconditionally. A non-None list is a per-cache splice opportunity
        for IsoChunkCache/SlopedChunkCache, each scoped to the single-unit
        tool paths that can build one -- see UnitSplice's own docstring and
        each override's. This base implementation ignores `changed`
        entirely and always does the safe, wholesale thing: it is the
        documented fallback every override's own ineligibility guard falls
        back to, not just the None-caller's path.

        The unconditional body is exactly _refresh_source_caches()'s own
        unconditional (elevation_changed=None) rebuild, which is already
        correct for Iso/Sloped -- see each class's own docstring.
        FlatChunkCache overrides this: its _refresh_source_caches() is a
        no-op once unit_draws exists, so it needs the extra del/clear step
        this base version doesn't have to do.

        splice_levels (Stepped/Sloped only): False skips the level splice and
        tries the in-place units_by_tile update instead. Ignored here and on Flat.
        """
        self._mutation_epoch += 1
        self._refresh_source_caches()

    def level_warm_job(self, mip: int) -> LevelWarmJob | None:
        """This level's sprite layer as a resumable warm, or None if there is
        nothing worth warming -- the entry point level_warm.LevelWarmer
        drives (2026-09-04 plan, Step 2).

        The base implementation returns None, which is SlopedChunkCache's
        real answer rather than a stub: Sloped enumerates exactly one mip
        level (_init_mip_levels({0: tile_px})), so it can never hit a
        not-yet-visited one -- there is no first-zoom stall there to remove.
        IsoChunkCache and FlatChunkCache override this."""
        return None

    def is_level_resident(self, mip: int) -> bool:
        """Whether mip's sprite/icon layer is already built -- the
        precondition margin_warm.MarginWarmer (2026-09-07 plan, Step A3.3)
        must check before ever calling get_chunk() on a level: get_chunk()
        on a not-yet-visited level synchronously builds the WHOLE level
        inside that call -- 0.6-4.3s (level_warm.py's own docstring), the
        exact freeze this whole subsystem exists to remove, now inside a
        margin-warm timer callback with no user action to blame it on.

        True is SlopedChunkCache's real answer, not a permissive base-class
        default -- checked, not assumed: _refresh_source_caches() there
        builds units_by_tile, corner_rise, building_bboxes and sprites
        eagerly, at construction and on every refresh ("no generation-
        counter laziness ... this cache has only one (mip 0) level" -- see
        that method's own docstring), so _composite_rect() has nothing
        lazy left to trigger. IsoChunkCache and FlatChunkCache override
        this with their own real staleness predicates."""
        return True

    def _unit_pack_of(self, mip: int, create: bool):
        """Level `mip`'s native unit pack: with create, the one a composite
        would use (built if missing or stale); without, the installed one only
        if it still matches its sources, else None. Only called when
        _has_unit_pack is True."""
        raise NotImplementedError

    def pack_warm_job(self, mip: int, after_level_warm: bool = False) -> LevelWarmJob | None:
        """The level's pending unit-pack tiles, derived a slice per step
        (maintainer plan 2026-09-27) so a first-touch chunk finds them done.
        Separate from level_warm_job(), whose None is pinned against
        is_level_resident().

        None when there is nothing to derive: no pack on this cache (numpy,
        units off, Flat); a level that isn't resident, unless
        after_level_warm says a level warm queued ahead of this job will make
        it so; or a resident level whose pack is current with nothing
        pending, which keeps the per-stroke-step re-arm from starting a timer.

        Byte-identical by construction: the walk calls the pack's own ensure()
        earlier than a composite would. Between slices it stops quietly once
        the level leaves residency or its pack is replaced or goes stale."""
        if not self._has_unit_pack or composite_backend.native is None or not self.with_units:
            return None
        # A deferring level's pack is refreshed by its flush; reading it here would force one mid-stroke.
        # after_level_warm: the flush job queued ahead of this one runs it first.
        if self._defers_patch(mip) and not after_level_warm:
            return None
        if self.is_level_resident(mip):
            pack = self._unit_pack_of(mip, create=False)
            if pack is not None and pack.pending().size == 0:
                return None
        elif not after_level_warm:
            return None
        return LevelWarmJob(gen=self._pack_warm_walk(mip), install=lambda _payload: True, kind="pack")

    def _pack_warm_walk(self, mip: int):
        if not self.is_level_resident(mip):
            return
        pack = self._unit_pack_of(mip, create=True)
        pending = pack.pending()
        for i in range(0, pending.size, PACK_WARM_TILES):
            yield
            if (
                not self.is_level_resident(mip)
                or self._defers_patch(mip)
                or self._unit_pack_of(mip, create=False) is not pack
            ):
                return
            pack.ensure(pending[i : i + PACK_WARM_TILES])

    def has_chunk(self, mip: int, cx: int, cy: int) -> bool:
        """Whether chunk (mip, cx, cy) is already cached -- a plain
        membership read, no LRU touch (unlike get_chunk()'s move_to_end()).
        margin_warm.ring_chunks() uses this to drop already-resident chunks
        from a margin ring before it's ever queued, so
        MarginWarmer.is_active stays honest about remaining work instead of
        counting a chunk that needs no warming at all."""
        return (mip, cx, cy) in self._cache

    def chunk_index_range(self, mip: int, x0: int, y0: int, x1: int, y1: int) -> tuple[int, int, int, int]:
        """(cx0, cy0, cx1, cy1) -- the inclusive chunk-grid index range that
        LEVEL `mip` pixel rect [x0, x1) x [y0, y1) covers, at this cache's
        chunk_px. `mip` is accepted (unused here) purely so a call site that
        already has a level's mip alongside its rect can pass both without a
        second lookup -- chunk_px is the same at every level, so mip plays
        no part in the arithmetic.

        Written three times already, inline, as this same `x0 // chunk_px` /
        `(x1 - 1) // chunk_px` pair: render_rect(), patch(), and
        invalidate_region(). New code (2026-09-07 plan's A2/A3) calls this
        instead of a fourth copy. Those three existing sites are
        deliberately NOT rewired to it: render_rect() is the hottest
        byte-identity-guarded path in this cache, and patch()/
        invalidate_region() interleave this arithmetic with a
        reference-to-level conversion this function doesn't do -- rewiring
        either buys nothing here at a real risk of regressing code this
        module's own byte-identity tests already cover. The duplication is
        noted, not chased."""
        cx0, cy0 = x0 // self.chunk_px, y0 // self.chunk_px
        cx1, cy1 = (x1 - 1) // self.chunk_px, (y1 - 1) // self.chunk_px
        return cx0, cy0, cx1, cy1

    def _evict(self) -> None:
        """Evicts least-recently-used chunks while EITHER configured bound
        (see _init_max_chunks) is exceeded. A chunk is at most chunk_px**2
        * 3 bytes and is clipped to canvas bounds at the high edge, so a
        just-inserted chunk can never itself exceed max_bytes on a
        default-constructed cache -- this can't evict down to empty."""
        while self._cache and (
            (self.max_chunks is not None and len(self._cache) > self.max_chunks)
            or (self.max_bytes is not None and self._cache_bytes > self.max_bytes)
        ):
            _key, victim = self._cache.popitem(last=False)
            self._cache_bytes -= victim.nbytes

    def get_chunk(self, mip: int, cx: int, cy: int) -> np.ndarray:
        """Returns chunk (mip, cx, cy)'s composited pixels, from cache if
        present (moved to most-recently-used), else composited fresh via
        self._composite_rect() and inserted, evicting past either bound
        (see _evict). Clipped to canvas bounds at the high edge -- a chunk
        straddling the canvas edge is smaller than chunk_px x chunk_px,
        same "ragged last chunk" shape any tile-based grid has. cx/cy are
        LEVEL chunk-grid indices, sized against that level's own
        canvas_dims(mip)."""
        key = (mip, cx, cy)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        chunk = self._composite_rect(mip, *self._chunk_rect(mip, cx, cy))
        self._insert(key, chunk)
        return chunk

    def _chunk_rect(self, mip: int, cx: int, cy: int) -> tuple[int, int, int, int]:
        canvas_w, canvas_h = self.canvas_dims(mip)
        x0, y0 = cx * self.chunk_px, cy * self.chunk_px
        return x0, y0, min(x0 + self.chunk_px, canvas_w), min(y0 + self.chunk_px, canvas_h)

    def _insert(self, key: tuple[int, int, int], chunk: np.ndarray) -> None:
        self._cache[key] = chunk
        self._cache_bytes += chunk.nbytes
        self._cache.move_to_end(key)
        self._evict()

    def _prepare_rect(self, mip: int, x0: int, y0: int, x1: int, y1: int) -> tuple | None:
        """render.prepare_rect_native() for this cache's sources, or None when
        the rect has no worker-safe path. Base (Flat): none."""
        return None

    def prepare_chunk_job(self, mip: int, cx: int, cy: int) -> ChunkJob | None:
        """get_chunk() split for margin_warm's worker threads: everything but
        the kernel runs here, on the caller's (GUI) thread. None means call
        get_chunk() instead: the chunk is cached, or has no worker-safe path
        (numpy backend, Flat, a texture the kernel can't crop)."""
        key = (mip, cx, cy)
        if key in self._cache:
            return None
        prepared = self._prepare_rect(mip, *self._chunk_rect(mip, cx, cy))
        if prepared is None:
            return None
        kernel, scratch, args, holds = prepared
        return ChunkJob(key, self._mutation_epoch, kernel, scratch, args, holds)

    def install_chunk(self, job: ChunkJob, painted: bool) -> bool:
        """Caches a finished job's pixels as get_chunk() would have. Refused
        (False) if the kernel declined, if any mutation ran since the job was
        prepared (patch() only fixes chunks already cached, never one in
        flight), or if a paint already built the chunk."""
        if not painted or job.epoch != self._mutation_epoch or job.key in self._cache:
            return False
        self._insert(job.key, job.scratch)
        return True

    def render_rect(self, x0: int, y0: int, x1: int, y1: int, mip: int = 0) -> np.ndarray:
        """Assembles pixels for [x0, x1) x [y0, y1) -- LEVEL `mip` pixels,
        clipped to that level's own canvas bounds -- from chunks, fetching/
        compositing each via get_chunk() as needed. The stitched result
        must be byte-identical to the corresponding crop of an independent
        full render at that level regardless of chunk request order or
        what was already cached -- see tools/verify_iso_chunks.py (Stepped)
        / tests/test_flat_chunks.py (Flat) at mip=0, tests/
        test_mip_geometry.py (Phase B-D-b) at mip != 0."""
        canvas_w, canvas_h = self.canvas_dims(mip)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(canvas_w, x1), min(canvas_h, y1)
        if x1 <= x0 or y1 <= y0:
            return np.zeros((0, 0, 3), dtype=np.uint8)

        out = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
        cx0, cy0 = x0 // self.chunk_px, y0 // self.chunk_px
        cx1, cy1 = (x1 - 1) // self.chunk_px, (y1 - 1) // self.chunk_px
        for cy in range(cy0, cy1 + 1):
            for cx in range(cx0, cx1 + 1):
                chunk = self.get_chunk(mip, cx, cy)
                chunk_x0, chunk_y0 = cx * self.chunk_px, cy * self.chunk_px
                ox0, oy0 = max(x0, chunk_x0), max(y0, chunk_y0)
                ox1, oy1 = min(x1, chunk_x0 + chunk.shape[1]), min(y1, chunk_y0 + chunk.shape[0])
                if ox1 <= ox0 or oy1 <= oy0:
                    continue
                out[oy0 - y0 : oy1 - y0, ox0 - x0 : ox1 - x0] = chunk[
                    oy0 - chunk_y0 : oy1 - chunk_y0, ox0 - chunk_x0 : ox1 - chunk_x0
                ]
        return out

    def patch(
        self, bbox: tuple[int, int, int, int], elevation_changed: set | None = None, levels=None, rebuild_levels=None
    ) -> None:
        """"Patch, don't drop": for every chunk CURRENTLY cached, at every
        RESIDENT mip level (only those in `levels`, when given: a unit edit
        patches its visible level alone), that bbox (REFERENCE canvas pixels) overlaps
        once converted to that level, recomposites just the intersected
        sub-rect via self._composite_rect() and writes it into the
        existing chunk array in place -- the cached chunk stays valid
        immediately, without paying a full chunk recomposite (or leaving a
        stale one on screen until the next get_chunk() eviction). Chunks
        NOT currently cached need no action: get_chunk() always composites
        fresh against the current state, so there's nothing stale to fix
        for those.

        Iterates RESIDENT levels ({key[0] for key in self._cache}), not
        every ENUMERATED level (mip_levels()) -- a level holding zero
        cached chunks is never entered, which is what keeps a single edit's
        cost from scaling with the total level count rather than with how
        many levels are actually warm.

        Looks up the overlapping chunk-grid range directly (same index math
        as invalidate_region()) rather than scanning every cached entry --
        a real cost difference once max_chunks is in the hundreds and only
        a handful of chunks overlap a single edit's bbox.

        bbox must already reflect the edit; this method does not mutate any
        underlying source state itself (elevations, terrain) -- callers
        that need to (dirty_screen_bbox_iso() for Stepped) do so before
        calling this. Refreshes this cache's own per-edit derived state
        first (_refresh_terrain_ids(bbox), then self._refresh_source_caches()) -- see each subclass's own
        docstring for what that means to it.

        elevation_changed (draw-perf plan Step 3): the subset of this edit's
        dirty tiles whose elevation actually moved (see
        render._dirty_screen_bbox()'s own docstring for where this comes
        from) -- passed straight through to _refresh_source_caches(). None
        (the default) means "unknown", which every _refresh_source_caches()
        override treats as "assume the worst and rebuild everything", i.e.
        today's behavior -- every call site that doesn't yet compute this
        set keeps its exact prior cost and correctness. The degenerate-bbox
        check runs BEFORE this call now (it used to run after, paying a full
        rebuild for a bbox with nothing to patch).

        rebuild_levels: None (every level may) or the levels allowed to do
        per-level source work (rebuild or splice) inside this call. A resident
        level outside it that the refresh left stale (not is_level_resident())
        evicts bbox's chunks instead of patching them, so a Stepped elevation
        step that bumps the gen rebuilds the visible level only, not every
        resident one; the stale level rebuilds when next composited or warmed.
        A current level outside it defers its elevation splice instead
        (IsoChunkCache._splice_levels), and any level with deferred work
        (_defers_patch) evicts too, so no cached chunk shows an unflushed layer."""
        px0, py0, px1, py1 = bbox
        # Before the early return too: the caller may have written elevations in place.
        self._mutation_epoch += 1
        if px1 <= px0 or py1 <= py0:
            return
        self._refresh_terrain_ids(bbox)
        self._refresh_source_caches(elevation_changed, splice_levels=rebuild_levels)
        for mip in self._select_levels(levels):
            lx0, ly0, lx1, ly1 = self._bbox_to_level(mip, bbox)
            if lx1 <= lx0 or ly1 <= ly0:
                continue
            cx0, cy0 = lx0 // self.chunk_px, ly0 // self.chunk_px
            cx1, cy1 = (lx1 - 1) // self.chunk_px, (ly1 - 1) // self.chunk_px
            evict = (
                rebuild_levels is not None and mip not in rebuild_levels and not self.is_level_resident(mip)
            ) or self._defers_patch(mip)
            for cy in range(cy0, cy1 + 1):
                for cx in range(cx0, cx1 + 1):
                    chunk = self._cache.get((mip, cx, cy))
                    if chunk is None:
                        continue
                    if evict:
                        self._cache_bytes -= chunk.nbytes
                        del self._cache[(mip, cx, cy)]
                        continue
                    chunk_x0, chunk_y0 = cx * self.chunk_px, cy * self.chunk_px
                    chunk_x1, chunk_y1 = chunk_x0 + chunk.shape[1], chunk_y0 + chunk.shape[0]
                    ix0, iy0 = max(lx0, chunk_x0), max(ly0, chunk_y0)
                    ix1, iy1 = min(lx1, chunk_x1), min(ly1, chunk_y1)
                    if ix1 <= ix0 or iy1 <= iy0:
                        continue
                    patched = self._composite_rect(mip, ix0, iy0, ix1, iy1)
                    chunk[iy0 - chunk_y0 : iy1 - chunk_y0, ix0 - chunk_x0 : ix1 - chunk_x0] = patched

    def patch_area(self, bbox: tuple[int, int, int, int], levels=None) -> int:
        """Level pixels patch(bbox, levels=levels) would recomposite: bbox
        clipped to each cached chunk, summed over the resident levels. Mirrors
        patch()'s loop without compositing, so a caller can price the eager patch first."""
        px0, py0, px1, py1 = bbox
        if px1 <= px0 or py1 <= py0:
            return 0
        area = 0
        for mip in self._select_levels(levels):
            lx0, ly0, lx1, ly1 = self._bbox_to_level(mip, bbox)
            if lx1 <= lx0 or ly1 <= ly0:
                continue
            cx0, cy0, cx1, cy1 = self.chunk_index_range(mip, lx0, ly0, lx1, ly1)
            for cy in range(cy0, cy1 + 1):
                for cx in range(cx0, cx1 + 1):
                    chunk = self._cache.get((mip, cx, cy))
                    if chunk is None:
                        continue
                    chunk_x0, chunk_y0 = cx * self.chunk_px, cy * self.chunk_px
                    w = min(lx1, chunk_x0 + chunk.shape[1]) - max(lx0, chunk_x0)
                    h = min(ly1, chunk_y0 + chunk.shape[0]) - max(ly0, chunk_y0)
                    if w > 0 and h > 0:
                        area += w * h
        return area

    def patch_rects(self, rects) -> None:
        """patch() for each rect in rects -- Phase B-E's Flat edits patch
        per-tile rects rather than one union bbox (a union over a scattered
        undo set can span the whole map, turning patch() into a full
        recomposite; see ViewerWindow._apply_dirty's Flat branch).

        No elevation_changed parameter: this is Flat's only patch_rects()
        caller today, and FlatChunkCache._refresh_source_caches() is a
        documented no-op regardless of what's passed to it -- there is
        nothing here for that set to gate."""
        for rect in rects:
            self.patch(rect)

    def invalidate_region(self, bbox: tuple[int, int, int, int], levels=None, terrain_changed: bool = True) -> None:
        """Evicts every cached chunk, at every RESIDENT mip level (only those
        in `levels`, when given), whose
        grid cell (once bbox is converted to that level) intersects it --
        forces a full recomposite from get_chunk() next time that chunk is
        requested, rather than trusting whatever's cached. Distinct from
        patch(): patch() keeps a cached chunk valid immediately at the cost
        of only the touched sub-rect; this drops it outright. Exists as its
        own primitive -- separate from patch() -- so a correctness bug
        elsewhere can't be masked by patch() quietly papering over it;
        tools/verify_iso_chunks.py's "invalidate_region round-trips" check
        calls this directly, forces a real recomposite via get_chunk(), and
        compares against a fresh full render.

        Per-level conversion fixes a real latent bug the old mip-agnostic
        chunk-index match had: a finer level's canvas is LARGER, so its
        chunk grid extends further -- a reference-derived index range used
        unconverted would under-evict a finer level's chunks near the
        canvas high edge, leaving stale pixels there. Coarser levels were
        merely over-evicted (harmless); this fixes both directions the same
        way, by computing each resident level's own true chunk-index
        range instead of reusing the reference's.

        terrain_changed=False skips the terrain-id mirror re-read, for a
        caller whose change is not terrain (units, layers, grid, sprites):
        the whole-canvas re-read costs 2.9 ms at 240x240 and 11 ms at 480x480."""
        _px0, _py0, _px1, _py1 = bbox
        self._mutation_epoch += 1
        if terrain_changed:
            self._refresh_terrain_ids(bbox)
        ranges: dict[int, tuple[int, int, int, int]] = {}
        for mip in self._select_levels(levels):
            lx0, ly0, lx1, ly1 = self._bbox_to_level(mip, bbox)
            if lx1 <= lx0 or ly1 <= ly0:
                continue
            ranges[mip] = (
                lx0 // self.chunk_px,
                ly0 // self.chunk_px,
                (lx1 - 1) // self.chunk_px,
                (ly1 - 1) // self.chunk_px,
            )
        for key in list(self._cache):
            mip, cx, cy = key
            rng = ranges.get(mip)
            if rng is None:
                continue
            cx0, cy0, cx1, cy1 = rng
            if cx0 <= cx <= cx1 and cy0 <= cy <= cy1:
                self._cache_bytes -= self._cache[key].nbytes
                del self._cache[key]

    def _init_terrain_ids(self) -> None:
        """Builds the terrain-id mirror on the native backend (2.9 ms at 240x240,
        11 ms at 480x480). A full re-render builds a fresh cache, so a fresh mirror."""
        if composite_backend.native is None:
            return
        mm = self.scenario.map_manager
        self._terrain_ids = native_composite._terrain_ids(self.scenario, np.arange(mm.map_width * mm.map_height))

    def _refresh_terrain_ids(self, bbox: tuple[int, int, int, int]) -> None:
        """Re-reads the mirror over bbox (REFERENCE canvas pixels) after a
        terrain edit. Every in-app terrain write reaches the cache through
        patch() or invalidate_region() with a bbox that covers each dirty
        tile's footprint, and tiles_in_screen_rect's swept test returns a
        superset of the tiles painting into it, so every dirty tile is re-read.
        A whole-canvas bbox reads every tile directly, which beats the rect
        enumeration at that size."""
        if self._terrain_ids is None:
            return
        px0, py0, px1, py1 = bbox
        if px1 <= px0 or py1 <= py0:
            return
        mm = self.scenario.map_manager
        w, h = mm.map_width, mm.map_height
        canvas_w, canvas_h = self.canvas_dims(0)
        if px0 <= 0 and py0 <= 0 and px1 >= canvas_w and py1 >= canvas_h:
            idx = np.arange(w * h)
        else:
            tiles = iso_geometry.tiles_in_screen_rect(px0, py0, px1, py1, w, h, self.proj)
            idx = tiles[:, 1] * w + tiles[:, 0]
        if idx.size:
            self._terrain_ids[idx] = native_composite._terrain_ids(self.scenario, idx)

    def _refresh_unit_sources(self) -> None:
        """Rebuilds whatever per-cache structure holds units, after the unit
        filter changed. Overridden by FlatChunkCache, whose
        _refresh_source_caches() is a deliberate no-op once unit_draws
        exists."""
        self._mutation_epoch += 1
        self._refresh_source_caches()

    def set_unit_filter(self, unit_filter: UnitFilter) -> None:
        """Swaps the unit filter and makes it actually visible -- phase 3's
        P3-a.

        Three steps, and skipping the LAST one is the trap this method
        exists to close: units are composited into cached chunk PIXELS, so
        storing the filter and rebuilding the source structures alone leaves
        every already-composited chunk showing the old set of units. The
        filter would appear to do nothing until the user happened to scroll
        somewhere uncached, which reads as "the toggle is broken" rather
        than "the cache is stale".

        A no-op guard on an unchanged filter is deliberate rather than
        missing: UnitFilter is frozen and compares by value, and evicting
        the whole canvas is the single most expensive thing this class can
        be asked to do."""
        if unit_filter == self.unit_filter:
            return
        self.unit_filter = unit_filter
        self._refresh_unit_sources()
        self.invalidate_region((0, 0, *self.canvas_dims(0)), terrain_changed=False)

    def set_layers(self, layers: LayerState) -> None:
        """Swaps the View > Layers state and makes it actually visible.

        Same three-step shape as set_unit_filter() above, including its
        no-op guard (LayerState is frozen and compares by value, and
        evicting the whole canvas is the most expensive thing this class
        can be asked to do), and the same final step is the trap: both
        layers are baked into cached chunk PIXELS, so storing the state
        alone leaves every already-composited chunk showing the old render.

        The two kinds of field cost different things, so they are diffed
        rather than both paying for the worse case:

        - A PAINT-TIME field (terrain_textures) is read straight off
          self.layers by _composite_rect, so eviction alone is the whole
          update.
        - A BUILD-TIME field (_BUILD_TIME_LAYER_FIELDS: farm_overlay,
          small_trees, hero_glow) is baked into the SpriteLayer's own contents when it
          is BUILT, so it needs _refresh_unit_sources() first. Skipping that
          would leave IsoChunkCache._level()'s `gen != self._source_gen`
          short-circuit holding and the level serving the pre-flip layer --
          the under-repaint class set_sprites_enabled() documents, not
          cosmetic staleness.

        FlatChunkCache inherits this unchanged and correctly: it reaches
        neither build-time field (no farm path at all, and icon_for()
        contain-fits every unit into its own footprint), so a flip there
        rebuilds and repaints to the same pixels rather than pretending to
        apply a layer it has no path for."""
        if layers == self.layers:
            return
        rebuild = any(
            getattr(layers, f) != getattr(self.layers, f) for f in _BUILD_TIME_LAYER_FIELDS
        )
        self.layers = layers
        if rebuild:
            self._refresh_unit_sources()
        self.invalidate_region((0, 0, *self.canvas_dims(0)), terrain_changed=False)

    def set_grid(self, grid: GridBake) -> None:
        """Swaps the baked View > Grid spec and makes it visible: set_layers()'
        shape for a paint-time field, so eviction alone is the whole update.
        Stored rather than passed per call so patch()/patch_rects() reach it
        too; a per-call argument would punch grid-free rects into every edit."""
        if grid == self.grid:
            return
        self.grid = grid
        self.invalidate_region((0, 0, *self.canvas_dims(0)), terrain_changed=False)

    def set_sprites_enabled(self, enabled: bool) -> None:
        """Stores whether this cache composites real .sld sprites -- P3-g's
        toggle. The base implementation ONLY stores the value; every concrete
        subclass now overrides it to actually rebuild and evict (IsoChunkCache
        for Stepped, SlopedChunkCache from P3-g6, FlatChunkCache from P3-g7),
        so this body is reached only by a subclass that has yet to grow a
        sprite path. Storing the value here regardless is what lets
        ViewerWindow carry the toggle's state across an Elevation View switch
        without special-casing which style is live.

        Deliberately on the BASE class rather than on IsoChunkCache: it is the
        hook each style's sprite path fills in by overriding."""
        self.sprites_enabled = enabled


# P3-g3: whether IsoChunkCache composites real .sld sprites instead of
# coloured marks. OFF for now, deliberately -- P3-g4 is the measurement gate
# that decided an opt-in toggle over default-on, and defaulting off here means
# every existing byte-identity test keeps its meaning untouched while the new
# sprite tests opt in explicitly. The headless full-render path has its own
# with_sprites argument instead, so a test can render ground truth without
# touching global state.
#
# P3-g: this is the CONSTRUCTION DEFAULT for IsoChunkCache's `sprites`
# parameter, not a live switch. Nothing reads it at call time anymore -- the
# live flag is per-cache (_ChunkCacheBase.sprites_enabled), because a
# per-window toggle cannot be a process global: module state outlives a
# window's close(), so a GUI toggle would leak across windows and into other
# tests. Defined HERE, above IsoChunkCache, rather than beside SpriteLayer
# below, purely because a default argument is evaluated when the class body
# executes -- from below the class it would raise NameError at import.
SPRITES_ENABLED = False


@dataclass
class _IsoLevel:
    """One mip level's own state (Phase B-D). tile_px/proj are enumerated
    at construction (see _ChunkCacheBase._init_mip_levels) and are pure
    geometry -- cheap, and IsoChunkCache.canvas_dims() reads .proj
    DIRECTLY, never through _level(), so sizing the cache can never trigger
    a bbox build. building_bboxes is the expensive, PROJECTION-DEPENDENT
    part (see _building_bboxes_iso/_unit_screen_bbox_iso): built lazily on
    first composite at this level, and rebuilt lazily whenever `gen` falls
    behind the cache's own _source_gen -- see IsoChunkCache._level()."""

    tile_px: int
    proj: iso_geometry.IsoProjection
    building_bboxes: dict | None = None
    # A5: building_bboxes' own chunk-bucketed index, built beside it in
    # _assemble_level so the two can never disagree about which dict they mean.
    bystander_grid: render.BystanderGrid | None = None
    gen: int = -1
    # P3-g3. Per-level for the same reason building_bboxes is: a sprite is
    # scaled by 2*half_w/NATIVE_TILE_W, so it is projection-dependent, and it
    # is built in the same lazy step rather than a second one.
    sprites: render.SpriteLayer | None = None
    # The per-unit contributions `sprites` was merged from, so the next
    # wholesale rebuild re-resolves only changed units. None with sprites off.
    memo: render.SpriteMemo | None = None
    # Batch F N2's native view of units_by_tile + sprites at this level.
    unit_pack: native_composite.UnitPack | None = None
    # Elevation tiles whose re-anchor this current level has deferred (an
    # off-screen level during a Stepped stroke); IsoChunkCache._flush_pending().
    pending_elev: set[tuple[int, int]] = field(default_factory=set)


class IsoChunkCache(_ChunkCacheBase):
    """Qt-free LRU cache of composited Stepped-mode canvas chunks, keyed by
    (mip, chunk_x, chunk_y) -- Phase B-B of Track B.
    chunk_x/chunk_y are chunk-GRID indices: canvas pixel
    (chunk_x*chunk_px, chunk_y*chunk_px) is that chunk's own origin. mip is
    real as of Phase B-D-a (previously always 0); Phase B-D-b (this
    version) enumerates the REAL per-level projection set via
    iso_geometry.mip_projections_for(), keyed off settings.get_elev_step_pct()
    -- the same function every real proj-construction site in this module
    already calls, so the cache reading it too reproduces exactly what
    built `proj`. Neither B-D-a nor B-D-b changes a single rendered pixel
    reachable from the running app: nothing outside this class and its
    tests ever asks for mip != 0 until Phase B-D-c wires MapCanvasItem.
    paint() to a real LOD signal.

    Each chunk is composited independently via composite_rect_iso() -- the
    SAME function render_terrain_iso_with_proj()'s full loop and
    refresh_region_iso()'s per-edit patch both reduce to -- so a chunk's
    pixels never depend on which OTHER chunks happen to be cached or in
    what order they were requested (this class's own load-bearing
    correctness bar, see tools/verify_iso_chunks.py). Grid/LRU mechanics
    (get_chunk/render_rect/patch/invalidate_region) live in _ChunkCacheBase,
    shared with Phase B-E's FlatChunkCache -- this class supplies only the
    Stepped-specific pieces: canvas_dims(), _composite_rect(), and
    _refresh_source_caches().

    Wired into viewer.py's Stepped mode as of Phase B-C, via MapCanvasItem;
    also exercised standalone by tools/verify_iso_chunks.py and its own
    bench.

    max_chunks defaults to covering the WHOLE canvas at chunk_px (see
    _ChunkCacheBase._init_max_chunks()'s own docstring for why).

    NOT free memory-wise: a fully-warmed cache holds roughly as many total
    pixel bytes as the old single canvas did (this project's biggest real
    map, 480x480 at chunk_px=512, needs 480 chunks), and render_rect() (see
    below) additionally allocates a fresh full-canvas-sized stitched output
    array on every full-viewport call -- so a full-viewport composite
    transiently needs the persistent cache AND that scratch buffer at once.
    Measured on that map: peak RSS for an open+warm+edit workflow went
    743MB -> 1127MB versus the pre-B-C single-buffer approach -- a real,
    accepted cost of avoiding thrashing at the default zoom, not a memory
    win. Pass an explicit
    smaller max_chunks (as tools/verify_iso_chunks.py's eviction tests do)
    to exercise real LRU eviction instead."""

    style = "stepped"
    _has_unit_pack = True

    def __init__(
        self,
        scenario: LoadedScenario,
        elevations: np.ndarray,
        proj: iso_geometry.IsoProjection,
        tile_px: int,
        chunk_px: int = DEFAULT_CHUNK_PX,
        max_chunks: int | None = None,
        with_units: bool = True,
        unit_filter: UnitFilter = UnitFilter(),
        sprites: bool = SPRITES_ENABLED,
        layers: LayerState = DEFAULT_LAYERS,
    ):
        self.scenario = scenario
        self.elevations = elevations
        self.proj = proj
        self.tile_px = tile_px
        self.with_units = with_units
        self.unit_filter = unit_filter
        # Must be set before _refresh_source_caches() below, since _level()
        # reads it to decide whether to build a level's sprite layer at all.
        self.sprites_enabled = sprites
        # Same ordering constraint: _level() reads the build-time layer
        # fields (_BUILD_TIME_LAYER_FIELDS) to build a level's sprite layer.
        self.layers = layers
        # Phase B-D-b: the REAL per-level projection set. settings.
        # get_elev_step_pct() is read here rather than threaded through as
        # a parameter because it's exactly the same function every real
        # proj-construction site in this module already calls to build
        # `proj` itself -- so this reproduces what built `proj`, and
        # mip_projections_for's own identity-level self-check (an assert)
        # fails loudly if the two ever disagreed. Must happen before
        # _init_max_chunks(), which reads canvas_dims() -> self._levels[0].proj.
        mm = scenario.map_manager
        projs = iso_geometry.mip_projections_for(mm.map_width, mm.map_height, proj, settings.get_elev_step_pct())
        self._levels: dict[int, _IsoLevel] = {level: _IsoLevel(tile_px=p.tile_px, proj=p) for level, p in projs.items()}
        self._init_mip_levels({level: lvl.tile_px for level, lvl in self._levels.items()})
        # canvas_dims()'s own exactness assert: is_exact_mip() proves every
        # IsoProjection field scales exactly, but canvas_dims() adds the
        # skirt-headroom term (_canvas_pixel_dims) on top of proj.canvas_h --
        # the value the blit actually trusts -- so assert it here too,
        # against the real function rather than duplicating its formula
        # into iso_geometry (a second place that formula could drift).
        ref_w, ref_h = render._canvas_pixel_dims(proj)
        for lvl in self._levels.values():
            lw, lh = render._canvas_pixel_dims(lvl.proj)
            assert lw * proj.tile_px == ref_w * lvl.tile_px and lh * proj.tile_px == ref_h * lvl.tile_px, (
                f"level tile_px={lvl.tile_px}'s canvas_dims (skirt headroom included) isn't an "
                f"exact mip of the reference's -- {(lw, lh)} vs reference {(ref_w, ref_h)}"
            )
        self._source_gen = 0
        # Bumped by every splice that leaves _source_gen alone, so a warm
        # started before one can tell its half-walked layer is stale.
        self._splice_epoch = 0
        # Bumped whenever units_by_tile is mutated in place, which a unit
        # pack's identity check can't see.
        self._units_version = 0
        self._init_terrain_ids()
        self._refresh_source_caches()
        self._init_max_chunks(chunk_px, max_chunks)

    def _refresh_source_caches(self, elevation_changed: set | None = None, splice_levels=None) -> None:
        """Rebuilds the SHARED, tile-space units_by_tile and bumps the
        source generation counter -- must run whenever elevations could
        have changed underneath this cache (construction, and every
        patch()). Deliberately does NOT rebuild any level's
        building_bboxes here: those are PROJECTION-dependent (a building's
        screen bbox depends on its center tile's elevation via
        _unit_screen_bbox_iso), so under mips they are per-level, and
        rebuilding every ENUMERATED level here would make a single edit
        pay a whole-map walk per level -- for levels that may hold no
        cached chunks at all. Each level's bboxes are instead rebuilt
        lazily, in _level(), the first time that level is actually
        composited after this bump -- see _level()'s own docstring.

        elevation_changed (draw-perf plan Step 3, 3a/3b): None means
        "unknown" -- construction, or a unit-filter/sprite-toggle refresh
        via _refresh_unit_sources() -- and gets today's unconditional
        wholesale behavior (units_by_tile rebuilds, gen always bumps),
        since THOSE changed and every level's bboxes/sprites must go stale.

        A patch() caller instead passes the elevation-changed subset of its
        own edit (empty for a terrain-paint-only edit). units_by_tile is
        NEVER rebuilt then (3a): no terrain or elevation edit can move a
        unit, so this dict cannot go stale from a patch().

        The unit layers are re-anchored in place rather than rebuilt (the
        2026-09-24 anchor-local splice). The wholesale level rebuild cost
        21-91ms per stroke step with sprites on, not the ~15-20ms the gen
        gate was accepted on. Stepped reads a unit's OWN tile only
        (_resolve_unit_sprite's elevations[uy, ux], _unit_iso_footprint's own
        floored tile), so the affected units are exactly those whose own
        tile is in elevation_changed -- radius 0, unlike Sloped's 1 -- plus
        any unit sharing a tile with one, which re-derives unchanged
        (_elevation_splices()' component). Each
        already-current level gets one _reanchor_units() call with the real
        wall overrides (an elevation edit leaves them valid) and a patched
        bystander_grid. A stale level is left stale and rebuilds fully in
        _level(). No gen bump, so _splice_epoch is bumped instead for
        level_warm_job()'s install predicate. A component past
        _ELEV_SPLICE_MAX_UNITS bumps the gen, the wholesale fallback.

        splice_levels (patch()'s rebuild_levels, 2026-09-28 stepped-defer
        plan): a current level outside it adds elevation_changed to its
        pending_elev instead of splicing, flushed on its next _level()."""
        if elevation_changed is None:
            self.units_by_tile = render._units_by_tile(self.scenario, self.unit_filter) if self.with_units else {}
            self._source_gen += 1
            return
        if not self.with_units or not elevation_changed:
            return
        splices = _elevation_splices(self.scenario, self.units_by_tile, self.unit_filter, elevation_changed)
        if splices is None:
            self._source_gen += 1
            return
        if not splices:
            return
        overrides = render.wall_variant_rotation_overrides(self.scenario)
        self._splice_levels(splices, overrides, self._units_version, splice_levels, elevation_changed)

    def _splice_levels(
        self, splices: list[UnitSplice], overrides: dict, old_version: int, levels=None, pending=None
    ) -> None:
        """_splice_level() on every already-current level. Shared by the
        elevation patch and invalidate_units(); bumps _splice_epoch since
        neither moves _source_gen. old_version is _units_version before the
        caller's own units_by_tile splice.

        levels (elevation patch only): a current level outside it unions
        `pending`, the edit's elevation-changed tiles, into its pending_elev
        instead. Re-anchor is delete-then-recompute from live state, so one
        flush over a stroke's union equals splicing each step in turn."""
        for mip, lvl in self._levels.items():
            if lvl.gen != self._source_gen:
                continue
            if levels is not None and mip not in levels:
                lvl.pending_elev |= pending
                continue
            self._splice_level(lvl, splices, overrides, old_version)
        self._splice_epoch += 1

    def _splice_level(self, lvl: _IsoLevel, splices: list[UnitSplice], overrides: dict, old_version: int) -> None:
        """_reanchor_units() on one current level, then its grid (patched over
        the keys the splice rewrote, not rebuilt) and unit pack. Reads the
        level directly: this runs inside _flush_pending()."""
        old_sprites = lvl.sprites
        old_bboxes: dict = {}
        lvl.sprites = _reanchor_units(
            lvl.building_bboxes, lvl.sprites, self.scenario, lvl.proj, self.elevations,
            self.unit_filter, None, 0, splices, overrides, *self._layer_resolve_args(),
            old_bboxes=old_bboxes,
        )
        lvl.bystander_grid = (
            render.build_bystander_grid(lvl.building_bboxes, self.chunk_px)
            if lvl.bystander_grid is None
            else render.patch_bystander_grid(lvl.bystander_grid, old_bboxes, lvl.building_bboxes)
        )
        lvl.unit_pack = _refreshed_pack(
            lvl.unit_pack, splices, self.units_by_tile, old_sprites, old_version, self.elevations,
            lvl.sprites, self._units_version, self.elevations,
        )

    def _defers_patch(self, mip: int) -> bool:
        # A stale level's pending set is moot: its rebuild clears it.
        lvl = self._levels[mip]
        return lvl.gen == self._source_gen and bool(lvl.pending_elev)

    def _flush_pending(self, mip: int) -> None:
        """Applies a current level's deferred elevation re-anchors: one
        _elevation_splices() over the union, then _splice_level(). A union
        past _ELEV_SPLICE_MAX_UNITS marks the level stale instead, so
        _level() rebuilds it wholesale.

        Only valid while units_by_tile agrees with the scenario, which is why
        _level() is the one caller: between a unit edit and its
        invalidate_units() the unit has moved but its buckets have not, and
        a flush there would re-add it at its new tile on top of the old one.
        invalidate_units() needs no flush of its own: its splice reaches
        pending levels too (they are current), and the later flush then
        reads consistent state."""
        lvl = self._levels[mip]
        with perf_trace.level("flush", mip) as ev:
            # Cleared first so nothing reached from the splice below can re-enter.
            tiles, lvl.pending_elev = lvl.pending_elev, set()
            if ev is not None:
                ev.tiles = len(tiles)
            splices = _elevation_splices(self.scenario, self.units_by_tile, self.unit_filter, tiles)
            if splices is None:
                if ev is not None:
                    ev.kind = "flush>rebuild"
                lvl.gen = -1
                return
            if not splices:
                return
            overrides = render.wall_variant_rotation_overrides(self.scenario)
            self._splice_level(lvl, splices, overrides, self._units_version)
            self._splice_epoch += 1

    def _level(self, mip: int) -> _IsoLevel:
        """The mip level's own state, rebuilding its building_bboxes if
        stale (gen != self._source_gen). Called only from _composite_rect,
        so a level is never rebuilt just because it's resident -- only
        when actually composited. Combined with patch()'s "iterate resident
        levels only" (_ChunkCacheBase.patch), an edit at a fixed zoom (one
        level resident) pays exactly one _building_bboxes_iso rebuild,
        identical to pre-mip behavior; immediately after a mip switch (two
        levels resident), the first edit pays two -- units_by_tile itself
        is still built once per edit regardless of level count, so the
        real cost is bounded by resident level count, not enumerated level
        count. A current level with deferred elevation work is flushed first
        (_flush_pending), which may itself mark it stale."""
        lvl = self._levels[mip]
        if lvl.gen == self._source_gen and lvl.pending_elev:
            self._flush_pending(mip)
        if lvl.gen != self._source_gen:
            with perf_trace.level("build", mip) as ev:
                # P3-g3: resolving and decoding sprites is far too expensive to do
                # per chunk, so it rides this same per-level lazy rebuild.
                sprites = memo = None
                if self.with_units and self.sprites_enabled:
                    sprites, memo = render._drain(self._sprite_walk(lvl))
                if ev is not None:
                    ev.mark("walk")
                bboxes, grid = render._drain(self._assemble_level(lvl, sprites))
                self._commit_level(mip, sprites, memo, bboxes, grid, self._source_gen)
                if ev is not None:
                    ev.mark("install")
        return lvl

    def _sprite_walk(self, lvl: _IsoLevel):
        """The level's memo-backed sprite walk, shared by _level() and
        level_warm_job() so the two can't pass different arguments."""
        return render.sprite_draws_by_anchor_sliced(
            self.scenario, lvl.proj, self.elevations, self.unit_filter,
            with_farms=self.layers.farm_overlay,
            tree_scale=self.layers.tree_scale,
            hero_glow=self.layers.hero_glow,
            memo=lvl.memo or render.SpriteMemo(),
        )

    def _assemble_level(self, lvl: _IsoLevel, sprites: render.SpriteLayer | None):
        """A level's building_bboxes and bystander grid from an already-built
        sprite layer, as a resumable generator (at most render.ASSEMBLY_SLICE
        units or keys per step) returning (bboxes, grid). Installs nothing:
        _commit_level() does that.

        One assembly for both paths: _level() drains it, the warm jobs step it
        across ticks (2026-09-29 warm-tick plan). Two assemblies would be the
        divergence class Flat's row-count assert exists to catch, and Stepped
        has no equivalent tripwire.

        merge_sprite_bboxes is what makes composite_rect_iso pull a sprite's
        anchor tile in as a bystander -- without it a sprite clips at its
        owning chunk's edge. The grid is built from the MERGED dict."""
        mm = self.scenario.map_manager
        # Its own step: the walk's last one ends here, not inside the first slice.
        yield
        bboxes = {}
        if self.with_units:
            bboxes = yield from render._building_bboxes_iso_sliced(
                self.scenario, mm.map_width, mm.map_height, lvl.proj, self.elevations, unit_filter=self.unit_filter
            )
        if sprites is not None:
            bboxes = yield from render.merge_sprite_bboxes_sliced(bboxes, sprites)
        grid = yield from render.build_bystander_grid_sliced(bboxes, self.chunk_px)
        return bboxes, grid

    def _commit_level(
        self, mip: int, sprites: render.SpriteLayer | None, memo: render.SpriteMemo | None,
        bboxes: dict, grid: render.BystanderGrid, gen: int,
    ) -> None:
        """Installs an assembled level: field assignment only, so a warm's
        install step costs nothing. The warm's install predicate runs first."""
        lvl = self._levels[mip]
        lvl.sprites = sprites
        lvl.memo = memo
        lvl.unit_pack = None
        # A fresh build already reflects every elevation.
        lvl.pending_elev = set()
        lvl.building_bboxes = bboxes
        lvl.bystander_grid = grid
        lvl.gen = gen

    def _rebuild_walk(self, lvl: _IsoLevel):
        """The sliced level rebuild both warm jobs run: the sprite walk (sprites
        on), then _assemble_level(). Returns (sprites, memo, bboxes, grid)."""
        sprites = memo = None
        if self.with_units and self.sprites_enabled:
            sprites, memo = yield from self._sprite_walk(lvl)
        bboxes, grid = yield from self._assemble_level(lvl, sprites)
        return sprites, memo, bboxes, grid

    def _rebuild_install(self, mip: int, built: tuple, start_gen: int, start_epoch: int) -> bool:
        """A warm job's install: the three-check predicate (see level_warm_job),
        then _commit_level()."""
        if (
            self._source_gen != start_gen
            or self._splice_epoch != start_epoch
            or self._levels[mip].gen == self._source_gen
        ):
            return False
        self._commit_level(mip, *built, start_gen)
        return True

    def can_splice(self, changed: list[UnitSplice] | None) -> bool:
        """Whether invalidate_units(changed) would take its splice path."""
        return self._unit_splice_plan(changed) is not None

    def invalidate_units(self, changed: list[UnitSplice] | None = None, splice_levels: bool = True) -> None:
        """Batch D's D4 splice: `changed` (a list of UnitSplice, one per
        single-unit-tool-path edit) lets a Move/Nudge/Rotate/Set field/Place/
        single Delete update units_by_tile and every ALREADY-BUILT level's
        building_bboxes/sprites in place, instead of paying
        _refresh_source_caches()'s full _units_by_tile() rebuild plus every
        resident level's next-composite _building_bboxes_iso/
        sprite_draws_by_anchor walk.

        None (unknown -- every pre-D4 caller, plus set_unit_filter()/
        set_sprites_enabled() via _refresh_unit_sources()) keeps today's
        exact wholesale path via the base class. A `changed` that
        _splice_plan() can't handle (see _batch_splice_eligible() and the two
        component splices) falls back to that same wholesale path --
        this method never partially applies a batch: if any entry in
        `changed` isn't splice-safe, ALL of it is covered for free by one
        _refresh_source_caches() call, since that derives fresh from live
        scenario state regardless of which edits produced it. A batch on
        shared tiles (a Convert: _reassign_component_splices(); a Draw stroke
        end, its undo or a group Move: _membership_component_splices())
        splices its whole component instead of `changed`: that list is built
        before _units_version moves and before units_by_tile is touched,
        since its search reads the pre-edit buckets. A wholesale fallback
        says why on the Perf Trace line (_trace_splice_refusal()).

        **Deliberately does NOT bump self._source_gen.** _level()'s
        gen-gated laziness is the whole reason this splice is worth writing
        (see this class's own docstring): bumping the gen would mark every
        level -- including the one this method just spliced -- stale again,
        forcing the exact per-level _building_bboxes_iso/
        sprite_draws_by_anchor rebuild this method exists to avoid. Only
        levels with `lvl.gen == self._source_gen` (already built and
        current) are spliced; an already-stale level is left alone and
        rebuilds fully fresh, correctly, the next time _level() reaches it.
        Not bumping the gen is also why _splice_levels() bumps
        _splice_epoch, which level_warm_job()'s install predicate checks:
        without it a warm started before this splice would install a layer
        describing the pre-edit units. The caller (_after_unit_mutation, D5)
        still cancels warms first, which saves the wasted walk.

        The whole batch goes through ONE _reanchor_units() call per level,
        so a Convert of N units copies each SpriteLayer dict once, not N
        times. overrides come from _splice_overrides() over the plan: a
        reassigned wall, a moved one and the walls a moved connector reshapes
        (_connector_neighbours()) all resolve with the real dict. An added or
        removed wall/connector still goes wholesale (`const:add`/`const:remove`).

        A refused batch, or any batch with splice_levels=False (the caller knows
        a later elevation patch will bump the gen anyway, so a level splice would
        be wasted), first tries _update_units_by_tile_in_place(): units_by_tile
        does not depend on elevation, so a membership-only batch patches it in
        place and bumps the gen, leaving every level stale exactly as the
        wholesale path does. A zero-length `units_in_place` phase marks it."""
        self._mutation_epoch += 1
        plan = self._unit_splice_plan(changed) if splice_levels else None
        if plan is None:
            self._trace_splice_refusal(changed, skipped=not splice_levels)
            if changed is not None and self._update_units_by_tile_in_place(changed):
                with perf_trace.phase("units_in_place"):
                    pass
                self._source_gen += 1
                return
            self._refresh_source_caches()
            return
        changed = _removals_first(plan)
        old_version = self._units_version
        self._units_version += 1
        for s in changed:
            _splice_units_by_tile(self.units_by_tile, self.scenario, self.unit_filter, s)
        self._splice_levels(changed, _splice_overrides(self.scenario, changed), old_version)

    def level_warm_job(self, mip: int) -> LevelWarmJob | None:
        """Stepped's resumable warm for one level -- see _ChunkCacheBase's
        own docstring for the shape.

        None when there is nothing expensive left to warm: with sprites (or
        units) off, a level costs only _building_bboxes_iso, measured at
        3.9ms against sprite_draws_by_anchor's 676ms, so ticking for it would
        be pure overhead. None too when the level is already current, which
        is the common case for the level the opening paint just built.

        **The install predicate is three checks, and `lvl.gen ==
        self._source_gen` is the one that isn't obvious.** The first two are
        staleness (the source moved under a warm that started before it, so
        the layer describes a scenario that no longer exists): a gen bump,
        or a splice (_splice_epoch), which moves units or elevations without
        one. The third is the opposite case: a real paint got there first
        and built an equivalent level, and overwriting it would throw away a
        layer already wired into the chunk cache to install a fresh copy of
        the same thing.

        A current level with deferred elevation work gets a flush job instead,
        sprites on or off (_flush_warm_walk), so no chunk warm or paint pays
        the flush (2026-09-29 warm-tick plan)."""
        lvl = self._levels[mip]
        if lvl.gen == self._source_gen and lvl.pending_elev:
            return self._flush_warm_job(mip)
        if not (self.with_units and self.sprites_enabled):
            return None
        start_gen = self._source_gen
        start_epoch = self._splice_epoch
        if lvl.gen == start_gen:
            return None
        # The walk and the assembly both step here; install only commits.
        return LevelWarmJob(
            gen=self._rebuild_walk(lvl),
            install=lambda built: self._rebuild_install(mip, built, start_gen, start_epoch),
        )

    def _flush_warm_job(self, mip: int) -> LevelWarmJob:
        """level_warm_job() for a current level with pending tiles: the walk's
        payload is None when the flush left the level current, else
        (rebuild, start_gen, start_epoch)."""

        def install(payload) -> bool:
            return payload is None or self._rebuild_install(mip, *payload)

        return LevelWarmJob(gen=self._flush_warm_walk(mip), install=install, kind="flush")

    def _flush_warm_walk(self, mip: int):
        """_flush_pending() as one step, then, if an over-cap union marked the
        level stale, level_warm_job()'s sliced rebuild. Re-checks on its first
        step, since a paint may have flushed the level after start(). Safe
        between ticks: every mutation cancels warms before it runs and re-arms
        after invalidate_units(), so units_by_tile agrees with the scenario."""
        lvl = self._levels[mip]
        if lvl.gen == self._source_gen and lvl.pending_elev:
            self._flush_pending(mip)
        if lvl.gen == self._source_gen:
            return None
        # Captured after the flush: an under-cap one bumps _splice_epoch.
        start_gen, start_epoch = self._source_gen, self._splice_epoch
        yield
        built = yield from self._rebuild_walk(lvl)
        return built, start_gen, start_epoch

    def is_level_resident(self, mip: int) -> bool:
        """Mirrors level_warm_job()'s own staleness check (`lvl.gen ==
        self._source_gen`) rather than level_warm_job()'s None-or-not
        answer: the two predicates ask different questions (this one is a
        pure "is get_chunk() safe to call right now", level_warm_job() also
        factors in "is there anything worth WARMING" -- with sprites off a
        level can be resident yet still return a level_warm_job() of None,
        since a cheap _building_bboxes_iso-only rebuild isn't worth
        ticking for). tests/test_margin_warm.py pins the two together after
        a completed warm so a future change to one can't silently diverge
        from the other.

        A current level with deferred elevation work is not resident either:
        get_chunk() would flush it, and an over-cap flush rebuilds it
        wholesale inside that call. level_warm_job() flushes it instead
        (2026-09-29 warm-tick plan)."""
        lvl = self._levels[mip]
        return lvl.gen == self._source_gen and not lvl.pending_elev

    def set_sprites_enabled(self, enabled: bool) -> None:
        """Turns real .sld sprites on or off on a LIVE cache -- P3-g's toggle.

        Shaped like set_unit_filter() above, guard included, plus one extra
        step. The no-op guard matters for the same reason it does there:
        evicting the whole canvas is the most expensive thing this class can
        be asked to do.

        _refresh_unit_sources() is what bumps _source_gen, and skipping it
        would leave _level()'s `gen != self._source_gen` short-circuit
        holding: the level would keep serving the pre-flip sprite layer, with
        merge_sprite_bboxes' contribution still folded into
        building_bboxes. That is the under-repaint class of bug, not cosmetic
        staleness. invalidate_region() is the set_unit_filter() trap
        unchanged -- units are baked into chunk PIXELS, so without it the
        toggle appears to do nothing until the user scrolls somewhere
        uncached.

        The extra step is warming the resident levels eagerly (step 5). It
        looks redundant next to the lazy rebuild _level() already does, and
        it is not: turning sprites ON pays a 0.5-4.1s cold .sld decode
        (P3-g4's measurement, the reason the default is off), and the caller
        holds a wait cursor around THIS call. Left lazy, the cursor lifts
        before the real work starts and the window instead freezes for those
        seconds inside the next Qt paint, with a normal cursor and no
        indication anything is happening -- the exact UX P3-g4 cited.

        Step ordering is load-bearing: the resident set must be captured
        before invalidate_region() empties the cache and takes it away."""
        if enabled == self.sprites_enabled:
            return
        self.sprites_enabled = enabled
        resident = sorted({key[0] for key in self._cache}) or [0]
        self._refresh_unit_sources()
        for mip in resident:
            self._level(mip)
        self.invalidate_region((0, 0, *self.canvas_dims(0)), terrain_changed=False)

    def set_layers(self, layers: LayerState) -> None:
        """The base method plus set_sprites_enabled()'s eager resident-level
        warm, and only on a _BUILD_TIME_LAYER_FIELDS flip, which is the only
        kind that rebuilds a sprite layer at all.

        Same reason and same load-bearing step order as there: the caller
        holds a wait cursor around this call, so leaving the rebuild to
        _level()'s laziness would lift the cursor and then freeze the window
        inside the next Qt paint instead; and the resident set has to be
        captured BEFORE invalidate_region() empties the cache and takes it
        away.

        Cheaper than set_sprites_enabled() in practice, on both halves.
        Small Trees does miss _scaled_cache for every tree (its key carries
        the factor), but the .sld decode behind that sits in its own
        _native_cache, keyed without the factor, so the flip pays a re-tint
        and a re-resize rather than that method's cold decode. Flipping the
        farm
        layer re-walks the units but decodes no new .sld, so it does not pay
        that method's 0.5-4.1s cold-decode case."""
        if layers == self.layers:
            return
        if all(
            getattr(layers, f) == getattr(self.layers, f) for f in _BUILD_TIME_LAYER_FIELDS
        ):
            super().set_layers(layers)
            return
        self.layers = layers
        resident = sorted({key[0] for key in self._cache}) or [0]
        self._refresh_unit_sources()
        for mip in resident:
            self._level(mip)
        self.invalidate_region((0, 0, *self.canvas_dims(0)), terrain_changed=False)

    def _unit_level_state(self, mip: int):
        """The level as installed, read straight off self._levels, never via
        _level(): a stale level stays stale (see sprite_extent_before())."""
        lvl = self._levels[mip]
        return lvl.proj, lvl.sprites, lvl.building_bboxes, None, 0

    @property
    def building_bboxes(self) -> dict:
        """Level 0's building_bboxes, forcing a rebuild first if stale.
        Debugging/introspection accessor only -- composite_rect_iso() is
        always called with a specific level's own bboxes via _level(mip),
        never through this property."""
        return self._level(0).building_bboxes

    def canvas_dims(self, mip: int = 0) -> tuple[int, int]:
        """(width, height) in LEVEL `mip` canvas pixels, including skirt
        headroom -- see _canvas_pixel_dims(). Indexes self._levels[mip]
        DIRECTLY (never through _level()), so sizing the cache can never
        trigger a building_bboxes build."""
        return render._canvas_pixel_dims(self._levels[mip].proj)

    def _current_pack(self, lvl: _IsoLevel) -> native_composite.UnitPack | None:
        pack = lvl.unit_pack
        if pack is not None and pack.matches(self.units_by_tile, lvl.sprites, self._units_version, self.elevations):
            return pack
        return None

    def _unit_pack_of(self, mip: int, create: bool):
        lvl = self._levels[mip]
        return self._native_units(lvl, mip) if create else self._current_pack(lvl)

    def _native_units(self, lvl: _IsoLevel, mip: int):
        """The level's unit pack for composite_rect_iso's N2 path (None on
        numpy), rebuilt whenever it no longer describes the level's sources."""
        if composite_backend.native is None:
            return None
        if not self.with_units:
            return native_composite.NO_UNITS
        pack = self._current_pack(lvl)
        if pack is None:
            mm = self.scenario.map_manager
            with perf_trace.level("pack", mip):
                pack = lvl.unit_pack = native_composite.UnitPack(
                    False, mm.map_width, mm.map_height, lvl.proj, self.units_by_tile, lvl.sprites,
                    self._units_version, self.elevations,
                )
        return pack

    def _composite_rect(self, mip: int, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        lvl = self._level(mip)
        # Resolved first so a pack build stays its own level event, outside the phase.
        unit_pack = self._native_units(lvl, mip)
        with perf_trace.repaint_phase("composite"):
            return render.composite_rect_iso(
                self.scenario,
                x0,
                y0,
                x1,
                y1,
                self.elevations,
                lvl.proj,
                lvl.tile_px,
                self.units_by_tile,
                lvl.building_bboxes,
                self.with_units,
                sprites=lvl.sprites,
                bystander_grid=lvl.bystander_grid,
                layers=self.layers,
                grid=self.grid,
                unit_pack=unit_pack,
                terrain_ids=self._terrain_ids,
            )

    def _prepare_rect(self, mip: int, x0: int, y0: int, x1: int, y1: int) -> tuple | None:
        lvl = self._level(mip)
        return render.prepare_rect_native(
            False, self.scenario, x0, y0, x1, y1, self.elevations, lvl.proj, lvl.tile_px, lvl.building_bboxes,
            self.with_units, lvl.bystander_grid, self.layers, self.grid, self._native_units(lvl, mip),
            terrain_ids=self._terrain_ids,
        )


class FlatChunkCache(_ChunkCacheBase):
    """Flat mode's counterpart to IsoChunkCache -- Phase B-E of Track B. Same
    grid/LRU mechanics (_ChunkCacheBase), composited via composite_rect_flat()
    instead of composite_rect_iso()
    (see that function's own docstring for the correctness argument behind
    never needing refresh_units_over()'s fixed-point dirty-tile expansion
    here).

    max_chunks defaults to covering the WHOLE canvas at chunk_px, same as
    IsoChunkCache and for the same fitInView()-on-open reason -- unchanged
    here even though Flat's canvas is the LARGER of the two (675 MiB vs
    340 MiB at 480x480): capping this to something smaller would thrash
    at the app's own default zoom exactly
    the way IsoChunkCache's own docstring already argues against. Phase
    B-D's mip levels are the real fix for the fit-to-view memory cost, not
    a smaller cap here.

    _refresh_source_caches() is a documented no-op after construction:
    unlike Stepped's building bboxes (which move with their center tile's
    elevation, see IsoChunkCache._refresh_source_caches()), Flat's
    unit_draws (_flat_unit_draws()) derive only from unit.x/unit.y,
    unit_const, the owning player index, and map dimensions -- none of
    which any terrain or elevation edit touches. That makes patch() here
    genuinely cheaper than Stepped's per-edit re-anchor of the units on
    changed tiles, not just an equivalent no-op restated. A unit
    edit goes through invalidate_units() instead, which splices the edited
    rows or, failing that, forces a rebuild. Don't
    "fix" this no-op into an unconditional rebuild instead, since that
    would silently reintroduce the per-edit cost this class exists to
    avoid paying for edits that were never about units at all. The same
    argument covers P3-g7's per-level icon layers, which derive from the same
    unit data and so go stale through the same funnel.

    Real .sld icons as of P3-g7 (`sprites=`/set_sprites_enabled()), replacing
    the coloured rect per unit -- see _level_icons() and
    render.composite_rect_flat()."""

    style = "flat"

    def __init__(
        self,
        scenario: LoadedScenario,
        tile_px: int,
        chunk_px: int = DEFAULT_CHUNK_PX,
        max_chunks: int | None = None,
        with_units: bool = True,
        unit_filter: UnitFilter = UnitFilter(),
        sprites: bool = SPRITES_ENABLED,
        layers: LayerState = DEFAULT_LAYERS,
    ):
        self.scenario = scenario
        self.tile_px = tile_px
        self.with_units = with_units
        self.unit_filter = unit_filter
        # Must be set before _refresh_source_caches() below, the same ordering
        # trap IsoChunkCache.__init__ and SlopedChunkCache.__init__ document.
        self.sprites_enabled = sprites
        # Only layers.terrain_textures means anything here -- Flat has no
        # farm path. Stored anyway so ViewerWindow can carry one state
        # across a style switch without special-casing which cache is live.
        self.layers = layers
        # Phase B-D-b: the real candidate ladder, UNFILTERED -- "unfiltered"
        # here means the MIP ladder, nothing to do with unit_filter. Flat has no
        # projection (canvas_dims() is a bare multiply, exact at every
        # tile_px, no elev_step term to break exactness), so every power of
        # two mip_tile_px_candidates() finds is a real exact level, unlike
        # Stepped's mip_projections_for() which must filter through a real
        # exactness check. Must precede _init_max_chunks(), which reads
        # canvas_dims().
        self._init_mip_levels(iso_geometry.mip_tile_px_candidates(tile_px))
        # Per-level unit_draws (Phase B-D: "Flat's real
        # per-level work is unit_draws"). Level 0's draws live in
        # self.unit_draws (unchanged attribute, still read directly by
        # tests/test_flat_chunks.py); other levels are built lazily here.
        self._level_draws: dict[int, tuple[np.ndarray, np.ndarray] | None] = {}
        # P3-g7's per-level icon layers. Level 0 lives in here TOO, unlike
        # _level_draws -- see _level_icons().
        self._level_icon_layers: dict[int, dict[int, unit_sprites.SpriteDraw] | None] = {}
        # Bumped by invalidate_units(), read only by level_warm_job(). Flat
        # needs a counter where the rest of this class does not, and the
        # reason is specific to the warm: every OTHER reader of the icon
        # layers goes through _level_icons(), which finds an absent dict entry
        # and rebuilds -- "absent" is a complete staleness test for a
        # synchronous caller. A warm spans event-loop turns, so it has to
        # distinguish "absent because nobody built it yet" from "absent
        # because invalidate_units() cleared it WHILE I was walking", and
        # those two states are byte-identical. The row-count assert doesn't
        # cover the gap either: a unit MOVED (not added or removed) leaves the
        # count intact while shifting the icons' meaning. A row splice leaves
        # the layer present but re-keyed, which the counter covers too.
        self._unit_gen = 0
        # _refresh_source_caches() also sets self._row_uid/_row_player, one
        # entry per row of every level's draws (identical across levels).
        self._refresh_source_caches()
        self._init_max_chunks(chunk_px, max_chunks)

    def _refresh_source_caches(self, elevation_changed: set | None = None, splice_levels=None) -> None:
        """See this class's own docstring: a documented no-op once
        unit_draws already exists (nothing a terrain/elevation edit touches
        can make it stale), except at construction, where it must actually
        build unit_draws the first time. elevation_changed unused -- Flat
        has no elevation term at all, see this module's docstring."""
        if not hasattr(self, "unit_draws"):
            if self.with_units:
                bboxes, colors, self._row_uid, self._row_player = render._flat_unit_rows(
                    self.scenario, self.tile_px, self.unit_filter
                )
                self.unit_draws = (bboxes, colors)
            else:
                self.unit_draws = None
                self._row_uid = self._row_player = None

    def can_splice(self, changed: list[UnitSplice] | None) -> bool:
        """Whether invalidate_units(changed) would take its row splice."""
        return self.with_units and bool(changed) and self._flat_splice_eligible(changed)

    def sprite_extent_before(self, changed: list[UnitSplice], mip: int | None):
        """Flat patches per-tile rects (its icons stay inside their footprint),
        so a unit edit here never asks for a sprite extent: raise, whatever mip."""
        raise NotImplementedError("FlatChunkCache has no unit-edit sprite extent")

    sprite_extent_after = sprite_extent_before

    def invalidate_units(self, changed: list[UnitSplice] | None = None, splice_levels: bool = True) -> None:
        """Brings unit_draws, every other level's draws and every icon layer
        up to date after a unit edit. Phase 3.5b's unit-editing UI calls this
        via ViewerWindow._after_unit_mutation(); it overrides the base class
        because _refresh_source_caches()'s no-op would otherwise miss those
        edits.

        `changed` (a list of UnitSplice, in mutation order) takes the row
        splice when _flat_splice_eligible() allows it: Move/Nudge/Rotate/Set
        field rewrite their row in place, Delete and a reassign's source side
        drop their row, Add and a reassign's destination side insert one at
        the end of the owner's block (add() and reassign() both append). Only
        resident levels hold rows, and a non-resident level builds fresh from
        live scenario state in the same order. None, an empty list, or an
        ineligible batch takes the wholesale path below, which drops
        everything and rebuilds level 0's draws now and the rest lazily.
        splice_levels is the Stepped/Sloped keyword, ignored here."""
        self._mutation_epoch += 1
        if self.can_splice(changed):
            self._splice_rows(changed)
            return
        if hasattr(self, "unit_draws"):
            del self.unit_draws
        self._level_draws.clear()
        # In the same statement group as the draws, not a step later: icons are
        # keyed by ROW INDEX into the draws, so a unit mutation or filter change
        # that dropped one without the other would leave a stale icon painting
        # on a shifted row -- i.e. on the wrong unit.
        self._level_icon_layers.clear()
        # In the same group for the same reason: an in-flight warm's icons are
        # keyed to the row order this call just invalidated.
        self._unit_gen += 1
        self._refresh_source_caches()

    @staticmethod
    def _is_row_move(splice: UnitSplice) -> bool:
        """Same owner, present before and after: the row stays where it is."""
        return (
            splice.old_own_tile is not None
            and splice.new_own_tile is not None
            and splice.old_player_id in (None, splice.player_id)
        )

    def _row_of(self, unit) -> int | None:
        hits = np.flatnonzero(self._row_uid == id(unit))
        return int(hits[0]) if hits.size else None

    def _wants_row(self, player_id: int, unit) -> bool:
        """Whether a fresh _flat_unit_rows() walk would give `unit` a row."""
        mm = self.scenario.map_manager
        return self.unit_filter.matches(player_id, unit) and (
            render.unit_tile_bounds(unit, mm.map_width, mm.map_height) is not None
        )

    def _flat_splice_eligible(self, changed: list[UnitSplice]) -> bool:
        """Whether _splice_rows() reproduces a fresh walk for `changed`. No
        tile-sharing test, unlike _splice_eligible(): Flat paints overlapping
        rects in row order and the splice keeps that order exactly. What it
        can't do is a moved or added neighbour-dependent const (a pure
        reassign of one moves nothing, so no neighbour's icon changes), a unit
        edited twice in one
        batch (its row would be looked up against the wrong state), or a Move
        that enters or leaves the map (its row would have to appear or vanish
        mid-block, not at the block's end). Nor an insertion that isn't at its
        owner's list tail: undo's restore() puts a removed unit back mid-list,
        so each player's inserted units must be exactly the last ones in
        units[p], in `changed` order."""
        seen: set[int] = set()
        inserted: dict[int, list] = {}
        for s in changed:
            if (not s.is_reassign and not _const_splice_eligible(s.unit)) or id(s.unit) in seen:
                return False
            seen.add(id(s.unit))
            if self._is_row_move(s):
                if (self._row_of(s.unit) is not None) != self._wants_row(s.player_id, s.unit):
                    return False
            elif s.new_own_tile is not None:
                inserted.setdefault(s.player_id, []).append(s.unit)
        units = self.scenario.unit_manager.units
        for player_id, added in inserted.items():
            tail = units[player_id][-len(added):]
            if len(tail) != len(added) or any(a is not b for a, b in zip(added, tail, strict=True)):
                return False
        return True

    def _splice_rows(self, changed: list[UnitSplice]) -> None:
        """The row splice itself, applied to every resident level's draws and
        icon layer plus the row_uid/row_player side table. Overrides are
        _splice_overrides()': the real dict when a reassigned wall's icon
        reads it, else {} (_flat_splice_eligible() refuses a moved one)."""
        scenario = self.scenario
        mm = scenario.map_manager
        w, h = mm.map_width, mm.map_height
        overrides = _splice_overrides(scenario, changed)
        draws = {0: self.unit_draws, **self._level_draws}
        icon_layers = {mip: icons for mip, icons in self._level_icon_layers.items() if icons is not None}
        moves = [s for s in changed if self._is_row_move(s)]
        removals = [s for s in changed if not self._is_row_move(s) and s.old_own_tile is not None]
        insertions = [s for s in changed if not self._is_row_move(s) and s.new_own_tile is not None]

        # 1. Moves rewrite their own row in place; indices don't shift.
        for s in moves:
            row = self._row_of(s.unit)
            if row is None:
                continue
            bounds = render.unit_tile_bounds(s.unit, w, h)
            color = render._unit_color(s.unit, scenario.player_colors[s.player_id])
            for mip, (bboxes, colors) in draws.items():
                bboxes[row] = render._flat_unit_bbox(bounds, s.unit, self._mip_tile_px[mip])
                colors[row] = color
            team_index = scenario.team_indices[s.player_id]
            for mip, icons in icon_layers.items():
                icon = render._flat_unit_icon(
                    s.unit, s.player_id, s.index, overrides, team_index, bounds, self._mip_tile_px[mip]
                )
                if icon is None:
                    icons.pop(row, None)
                else:
                    icons[row] = icon

        # 2. Removals, looked up before anything shifts.
        removed = np.array(sorted({r for s in removals if (r := self._row_of(s.unit)) is not None}), dtype=np.int64)
        if removed.size:
            keep = np.ones(len(self._row_uid), dtype=bool)
            keep[removed] = False
            self._row_uid, self._row_player = self._row_uid[keep], self._row_player[keep]
            draws = {mip: (bboxes[keep], colors[keep]) for mip, (bboxes, colors) in draws.items()}
            for mip, icons in icon_layers.items():
                keys = np.fromiter(icons.keys(), dtype=np.int64, count=len(icons))
                survivors = keep[keys]
                new_keys = keys[survivors] - np.searchsorted(removed, keys[survivors])
                values = [v for v, alive in zip(icons.values(), survivors.tolist(), strict=True) if alive]
                icon_layers[mip] = dict(zip(new_keys.tolist(), values, strict=True))

        # 3. Insertions at the end of the owner's block, in `changed` order.
        added = [
            (int(np.searchsorted(self._row_player, s.player_id, side="right")), s, render.unit_tile_bounds(s.unit, w, h))
            for s in insertions
            if self._wants_row(s.player_id, s.unit)
        ]
        if added:
            positions = np.array([pos for pos, _s, _b in added], dtype=np.int64)
            order = np.argsort(positions, kind="stable")
            final = np.empty(len(added), dtype=np.int64)
            final[order] = positions[order] + np.arange(len(added))
            grown = len(self._row_uid) + len(added)
            is_new = np.zeros(grown, dtype=bool)
            is_new[final] = True

            def grow(old: np.ndarray, new_values) -> np.ndarray:
                out = np.empty((grown, *old.shape[1:]), dtype=old.dtype)
                out[~is_new] = old
                out[final] = new_values
                return out

            self._row_uid = grow(self._row_uid, [id(s.unit) for _p, s, _b in added])
            self._row_player = grow(self._row_player, [s.player_id for _p, s, _b in added])
            colors_new = [render._unit_color(s.unit, scenario.player_colors[s.player_id]) for _p, s, _b in added]
            draws = {
                mip: (
                    grow(bboxes, [render._flat_unit_bbox(b, s.unit, self._mip_tile_px[mip]) for _p, s, b in added]),
                    grow(colors, colors_new),
                )
                for mip, (bboxes, colors) in draws.items()
            }
            sorted_positions = positions[order]
            for mip, icons in icon_layers.items():
                keys = np.fromiter(icons.keys(), dtype=np.int64, count=len(icons))
                new_keys = keys + np.searchsorted(sorted_positions, keys, side="right")
                rekeyed = dict(zip(new_keys.tolist(), icons.values(), strict=True))
                for row, (_p, s, bounds) in zip(final.tolist(), added, strict=True):
                    icon = render._flat_unit_icon(
                        s.unit, s.player_id, s.index, overrides, scenario.team_indices[s.player_id],
                        bounds, self._mip_tile_px[mip],
                    )
                    if icon is not None:
                        rekeyed[row] = icon
                icon_layers[mip] = rekeyed

        self.unit_draws = draws.pop(0)
        self._level_draws.update(draws)
        self._level_icon_layers.update(icon_layers)
        # An in-flight warm walked the pre-splice rows; this makes its install refuse.
        self._unit_gen += 1
        rows = len(self._row_uid)
        assert all(len(bboxes) == rows for bboxes, _colors in (self.unit_draws, *self._level_draws.values())), (
            "a Flat row splice left a level's draws out of step with the row table"
        )

    def _refresh_unit_sources(self) -> None:
        """See _ChunkCacheBase._refresh_unit_sources(): the base version
        calls _refresh_source_caches(), which this class defines as a no-op
        once unit_draws exists, so a filter change has to go through
        invalidate_units() to force the real rebuild (level 0's draws AND
        every other level's)."""
        self.invalidate_units()

    def _level_unit_draws(self, mip: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Per-level _flat_unit_draws() -- Flat's counterpart to Stepped's
        per-level building_bboxes. No generation counter needed here,
        unlike IsoChunkCache._level(): unit_draws depends only on
        unit.x/unit.y, unit_const, owning player index and map dimensions,
        none of which any terrain/elevation edit touches -- the same
        property that makes _refresh_source_caches() a no-op after
        construction. A unit edit is what makes them stale, and
        invalidate_units() either splices the edited rows into every entry
        here or clears this dict --
        which is also why set_unit_filter() routes through invalidate_units()
        rather than rebuilding self.unit_draws alone: these per-level draws
        must honour the new filter too, or hidden units would keep painting
        at every zoom level except the one that happened to be resident."""
        if not self.with_units:
            return None
        if mip == 0:
            return self.unit_draws
        draws = self._level_draws.get(mip)
        if draws is None:
            with perf_trace.level("flat-draws", mip):
                draws = render._flat_unit_draws(self.scenario, self._mip_tile_px[mip], self.unit_filter)
            self._level_draws[mip] = draws
        return draws

    def _level_icons(self, mip: int) -> dict[int, unit_sprites.SpriteDraw] | None:
        """Per-level _flat_icon_layer() -- P3-g7's counterpart to
        _level_unit_draws() above, built lazily for the same reason and going
        stale through the same invalidate_units() funnel. None when this cache
        is not drawing sprites at all.

        **Level 0 is stored in the dict like every other level**, deliberately
        NOT mirroring _level_unit_draws()' `mip == 0` special case. That case
        exists only because self.unit_draws is a public attribute read
        externally (tests/test_flat_chunks.py); icons have no such reader, so a
        self.icons twin would buy nothing and add a second lifetime to keep in
        lockstep with unit_draws -- precisely the row-index desync this layer's
        whole keying scheme depends on avoiding. One dict, one lifetime.

        The row-count assert is the other half of that: _flat_icon_layer() and
        _flat_unit_draws() are two separate walks that MUST agree row for row,
        and this turns a future divergence into a build-time failure instead of
        icons silently painting on the wrong units. It compares WALK LENGTHS,
        not the maximum key -- a skipped unit that failed to advance `row`
        shifts every later icon onto its neighbour while staying in range, and
        a max-key check would sail past exactly that."""
        if not (self.with_units and self.sprites_enabled):
            return None
        if mip not in self._level_icon_layers:
            with perf_trace.level("flat-icons", mip):
                icons = render._flat_icon_layer(self.scenario, self._mip_tile_px[mip], self.unit_filter)
            self._install_icon_layer(mip, icons)
        return self._level_icon_layers[mip]

    def _install_icon_layer(self, mip: int, icons_and_rows: tuple[dict[int, unit_sprites.SpriteDraw], int]) -> None:
        """Installs an already-built (icons, row_count) pair, row-count assert
        included -- the shared tail of _level_icons() above and
        level_warm_job()'s install closure, so the warm path can't grow its
        own copy of the assert-then-store step (or quietly skip the assert)."""
        icons, rows = icons_and_rows
        draws = self._level_unit_draws(mip)
        assert rows == (0 if draws is None else len(draws[0])), (
            f"icon layer at mip {mip} walked {rows} units but _flat_unit_draws() "
            f"produced {0 if draws is None else len(draws[0])} rows -- the two walks have diverged"
        )
        self._level_icon_layers[mip] = icons

    def level_warm_job(self, mip: int) -> LevelWarmJob | None:
        """Flat's resumable warm for one level -- IsoChunkCache.
        level_warm_job()'s counterpart, same shape, different validity
        predicate (see self._unit_gen's own comment in __init__ for why
        "absent from _level_icon_layers" cannot be the whole test here).

        _flat_unit_draws (measured 14.1ms) is not sliced and runs whole
        inside the install, same as Stepped's _building_bboxes_iso: only the
        icon walk (~727ms cold) is worth resuming."""
        if not (self.with_units and self.sprites_enabled):
            return None
        if mip in self._level_icon_layers:
            return None
        start_gen = self._unit_gen
        gen = render._flat_icon_layer_sliced(self.scenario, self._mip_tile_px[mip], self.unit_filter)

        def install(icons_and_rows: tuple[dict[int, unit_sprites.SpriteDraw], int]) -> bool:
            if self._unit_gen != start_gen or mip in self._level_icon_layers:
                return False
            self._install_icon_layer(mip, icons_and_rows)
            return True

        return LevelWarmJob(gen=gen, install=install, kind="icons")

    def is_level_resident(self, mip: int) -> bool:
        """Flat's own predicate -- "absent from _level_icon_layers" IS a
        complete staleness test here (unlike level_warm_job(), which needs
        self._unit_gen to distinguish "never built" from "invalidated mid-
        warm", see __init__'s own comment on that counter): a chunk warm
        only ever calls this synchronously, with no warm in flight to have
        invalidated anything between the check and the get_chunk() call it
        gates."""
        return mip in self._level_icon_layers

    def set_sprites_enabled(self, enabled: bool) -> None:
        """Turns real .sld icons on or off on a LIVE cache (P3-g7).

        IsoChunkCache.set_sprites_enabled()'s shape, mip loop included --
        Flat has a real ladder, so that class is the precedent here rather
        than g6's single-level SlopedChunkCache, despite it being the more
        recent one. See it for why each step is load-bearing: the resident
        set must be captured BEFORE invalidate_region() empties the cache,
        and the resident levels are warmed eagerly so the 0.5-4.1s cold .sld
        decode happens under the caller's wait cursor rather than inside the
        next Qt paint with a normal cursor."""
        if enabled == self.sprites_enabled:
            return
        self.sprites_enabled = enabled
        resident = sorted({key[0] for key in self._cache}) or [0]
        self._refresh_unit_sources()
        for mip in resident:
            self._level_icons(mip)
        self.invalidate_region((0, 0, *self.canvas_dims(0)), terrain_changed=False)

    def canvas_dims(self, mip: int = 0) -> tuple[int, int]:
        """Unchanged by P3-g7, and worth saying so rather than leaving the
        next reader to check: an icon is contain-fitted into its own footprint
        rect and so can never leave it, while Flat's canvas is exact by
        construction. g6's canvas_dims()-widening fix for Sloped, and its
        MapView/MapCanvasItem.refresh_canvas_dims() follow-up, have no
        analogue here."""
        mm = self.scenario.map_manager
        tile_px = self._mip_tile_px[mip]
        return mm.map_width * tile_px, mm.map_height * tile_px

    def _composite_rect(self, mip: int, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        # Resolved first so their level builds stay level events, outside the phase.
        unit_draws = self._level_unit_draws(mip)
        icons = self._level_icons(mip)
        with perf_trace.repaint_phase("composite"):
            return render.composite_rect_flat(
                self.scenario,
                x0,
                y0,
                x1,
                y1,
                self._mip_tile_px[mip],
                unit_draws=unit_draws,
                with_units=self.with_units,
                icons=icons,
                layers=self.layers,
                grid=self.grid,
            )


MAX_PICK_PLANES = 4
"""How many Sloped chunk pick planes (SlopedChunkCache._pick_plane) stay
memoized at once. Sized for the access pattern, not by feel: a cursor is in
one chunk at a time and a drag along a chunk seam alternates between two, so
2 already removes the thrash case; 4 covers a diagonal drag through a
four-chunk corner without letting the memo grow into a second full-canvas
cache. At DEFAULT_CHUNK_PX=512 that is 512*512*4 = 1 MB per plane, 4 MB
capped -- negligible against the colour cache's whole-canvas budget."""

PICK_PLANE_PATCH_MAX_FRACTION = 0.35
"""How much of one resident pick plane an edit's bbox may cover before
SlopedChunkCache.patch() drops that plane outright instead of rewriting the
intersected sub-rect in place (Batch B step B8). At or below the fraction:
patch. Above it: drop, and let _pick_plane() rebuild on demand.

Patching costs its share of a full rebuild EAGERLY, always; dropping costs a
full rebuild LAZILY, and only if that plane is read again.

Measured, not chosen. tools/bench_pick_plane_patch.py section 1 on
F7_2_Dos Pilas (480x480, tile_px=32, chunk_px=512) times each sub-rect
against a full-plane rebuild taken right beside it (8.4-8.8ms):

    square sub-rect   10% area -> 20% cost   25% -> 32%   40% -> 54%
                      50% -> 62%   75% -> 83%   90% -> 98%   100% -> 98%
    full-width band   10% area -> 32% cost   25% -> 42%   50% -> 62%
                      75% -> 84%   90% -> 92%

There is no crossover below 100% to sit on: cost stays under the full
rebuild all the way up, because the only fixed term is the per-call
candidate walk plus the np.full, worth about 10% of a plane for a square
sub-rect and about 24% for a band (a band spans every candidate column, so
it pays more border per pixel). So the value comes from a budget instead:
one edit's eager pick-plane work should stay inside the single full rebuild
dropping already costs, since the cursor's own plane is read on the very
next mouseMoveEvent and for that one plane dropping is not really lazy. A
drag along a chunk seam alternates between TWO resident planes (see
MAX_PICK_PLANES above). Interpolating the rows either side of 0.35 puts one
patched plane at 47% (square) to 51% (band) of a full rebuild, so two of
them cost 0.94 to 1.01 of one and the seam case breaks even at worst, while the common mid-plane small-brush edit at ~0.3 of a
plane is a clear win. A wide brush spreads 0.6 to 0.8 over three or four
planes, lands above the fraction, and keeps exactly today's drop behaviour
rather than paying ~3x eagerly for planes that may never be read."""


class SlopedChunkCache(_ChunkCacheBase):
    """Qt-free LRU cache of composited Sloped-mode canvas chunks -- Phase 6's
    Track C3 counterpart to IsoChunkCache/
    FlatChunkCache. Same grid/LRU mechanics (_ChunkCacheBase), composited
    via composite_rect_sloped() instead of composite_rect_iso()/
    composite_rect_flat() -- a chunk's pixels never depend on which OTHER
    chunks happen to be cached or in what order they were requested, the
    same correctness bar tools/verify_iso_chunks.py/tests/
    test_flat_chunks.py hold their own compositors to (see tests/
    test_sloped_chunks.py for this class's own version of those checks).

    Single mip level (_init_mip_levels({0: tile_px})) -- deliberately, not
    an oversight: this is the same "level set is still {0}" intermediate
    state IsoChunkCache/FlatChunkCache themselves were in before Track
    B-D's real per-level ladder landed (see _ChunkCacheBase._init_mip_
    levels' own docstring), and it lands during that same track's active
    development. A genuine Sloped mip ladder is a legitimate follow-up --
    this class's bilinear warp would need its own per-level corner_rise_px
    array, mirroring IsoChunkCache's per-level building_bboxes -- kept out
    of Track C's own approved scope rather than built inline here.

    proj must carry corner_headroom_steps=1 (see sloped_elevations_and_
    proj()) -- built by the caller and passed in, matching IsoChunkCache's
    own convention of taking proj as a constructor parameter rather than
    building it itself."""

    style = "sloped"
    _has_unit_pack = True

    def __init__(
        self,
        scenario: LoadedScenario,
        elevations: np.ndarray,
        corner_rise: np.ndarray,
        proj: iso_geometry.IsoProjection,
        tile_px: int,
        chunk_px: int = DEFAULT_CHUNK_PX,
        max_chunks: int | None = None,
        with_units: bool = True,
        unit_filter: UnitFilter = UnitFilter(),
        sprites: bool = SPRITES_ENABLED,
        layers: LayerState = DEFAULT_LAYERS,
    ):
        self.scenario = scenario
        self.elevations = elevations
        # Superseded immediately by _refresh_source_caches()' own re-derivation
        # below, which owns this array from here on (see its docstring). Kept
        # as a parameter because every caller already holds the value and the
        # constructor stays parallel to elevations/proj, but a caller cannot
        # make it disagree with elevations by passing a mismatched pair.
        self.corner_rise = corner_rise
        self.proj = proj
        self.tile_px = tile_px
        self.with_units = with_units
        self.unit_filter = unit_filter
        # Must be set before _refresh_source_caches() below, since it reads
        # this to decide whether to build a sprite layer at all -- same
        # ordering trap IsoChunkCache.__init__ documents for itself.
        self.sprites_enabled = sprites
        # Same ordering constraint: _refresh_source_caches() reads the
        # build-time layer fields (_BUILD_TIME_LAYER_FIELDS) when it builds
        # the sprite layer.
        self.layers = layers
        self._pick_planes: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
        # IsoChunkCache's per-level memo, for this cache's one level.
        self._sprite_memo: render.SpriteMemo | None = None
        # IsoChunkCache's per-level unit pack and units_by_tile version.
        self._unit_pack: native_composite.UnitPack | None = None
        self._units_version = 0
        self._init_mip_levels({0: tile_px})
        # Set here, not left to _init_max_chunks below: _set_building_bboxes()
        # reads it and _refresh_source_caches() runs first.
        self.chunk_px = chunk_px
        self._init_terrain_ids()
        self._refresh_source_caches()
        self._init_max_chunks(chunk_px, max_chunks)

    def _refresh_source_caches(self, elevation_changed: set | None = None, splice_levels=None) -> None:
        """Rebuilds units_by_tile and building_bboxes -- unlike
        IsoChunkCache, no generation-counter laziness: this cache has only
        one (mip 0) level, so there is no "rebuild only when actually
        composited at THIS level" saving to make (see IsoChunkCache.
        _refresh_source_caches()'s own docstring for why that laziness
        exists there and would buy nothing here).

        building_bboxes reuses _building_bboxes_iso()/_unit_screen_bbox_
        iso() rather than a Sloped-specific reimplementation, but as of
        Track C5's Step 2 it must pass extra_top_px: Sloped's own proj is
        geometrically identical to Stepped's for the same map (see
        IsoProjection.corner_headroom_px's own comment), yet a unit there
        no longer sits at elevation * elev_step -- it sits at
        iso_geometry.unit_rise_px(), which under SLOPE_CORNER_RULE = "max"
        is never lower and can be higher. The "bystander" widening this
        feeds is an accepted coarse superset in the OVER direction only
        (composite_rect_iso()'s own docstring: "a union bbox flagging a
        tile as a bystander slightly more often than the tightest possible
        test would is a no-op extra paint, never a missed one"); under-
        covering is a missed paint, i.e. a stale pixel, so the headroom
        below is derived rather than assumed.

        `headroom` is an exact global bound, not the plan's "one
        elev_step": for every tile, how far its highest own corner sits
        above its own elevation, maxed over the map. One elev_step is what
        that evaluates to under this project's +-1-elevation-neighbour
        invariant, but a LOADED file is not required to satisfy that
        invariant, and this costs only four vectorized array maxes. Zero on
        a flat map, so flat-map byte-identity is untouched. self._headroom
        records the value the current building_bboxes were built with.

        Sloped honours unit_filter exactly as Flat and Stepped do -- and,
        since C5, so does unit picking (unit_pick.pick_unit's sloped
        branch reads this same cache's corner_rise through
        MapView._sloped_cache).

        corner_rise is rebuilt here too (Track C4's Step 3), which is what
        makes Sloped editable at all: before this it was received once at
        construction and never refreshed, so an elevation edit repainted
        tiles against the pre-edit height field. It belongs in THIS method
        specifically, not in the patch() override above it, because this is
        the one hook whose contract is already "run whenever elevations
        could have changed underneath this cache" -- putting it in patch()
        would skip invalidate_region() and any future caller.

        Rebuilt whole rather than per-dirty-ring, and the cost is measured
        rather than assumed: 4.6ms at 480x480, this project's largest map,
        against a 25ms end-to-end sloped patch there (2026-08-21). So it is
        real but not dominant, and a ring-only update -- which the C4 plan
        offers as the alternative -- stays available as a later optimization
        rather than being needed now. It also makes the redundant
        rebuild at construction a non-issue, and buys a real property for it:
        corner_rise can no longer disagree with self.elevations, however the
        constructor was called.

        self.sprites (Track P3-g6): built here, on the same "whenever
        elevations could have changed" cadence as corner_rise itself, since
        a sprite's anchor depends on corner_rise via unit_rise_px() -- an
        edit that leaves this stale would repaint sprites at their pre-edit
        height. with_farms defaults True (Track C6): a farm now drapes over
        its own footprint tiles' warped terrain, matching Stepped, whenever
        sprites are on; the sprites-off case still falls back to the plain
        mark (see render.py's own farm handling for that residual).
        merge_sprite_bboxes() runs AFTER the headroom-widened building
        bboxes above, never replacing them -- both operations only ever
        grow a bbox (F3), so the order between them doesn't affect the
        result, but merging into the already-final dict avoids a second
        allocation.

        elevation_changed (draw-perf plan Step 3): None means "unknown" --
        construction, or a unit-filter/sprite-toggle refresh via
        _refresh_unit_sources() -- and gets the unconditional wholesale
        rebuild below (units_by_tile included), same as before this
        parameter existed.

        A patch() caller instead passes the elevation-changed subset of its
        own edit. 3a applies here too: units_by_tile never rebuilds on a
        patch(), since no terrain or elevation edit can move a unit. An
        empty elevation_changed (a terrain-paint-only edit) skips
        everything below, since none of it depends on anything terrain
        paint touches. A non-empty one always rebuilds corner_rise whole
        (it depends on EVERY changed tile, not just ones under units) and
        recomputes the headroom, then re-anchors only the units that read a
        changed corner rather than rebuilding every unit (the 2026-09-24
        anchor-local splice; the wholesale rebuild was 194-208ms per stroke
        step on June Event with sprites on).

        Sloped reads a unit's own tile's FOUR corners (unit_rise_px), and
        under SLOPE_CORNER_RULE each corner blends the up-to-4 tiles
        touching it, so a changed tile moves the corners of every tile in
        the 3x3 around it: the affected units are those whose own tile is in
        elevation_changed dilated by one -- radius 1, unlike Stepped's 0 --
        plus any unit sharing a tile with one (_elevation_splices()'
        component, re-derived unchanged).
        They are re-anchored against the NEW corner_rise, so sprites and
        corner_rise cannot drift apart. The headroom feeds every building's
        bbox, not just the re-anchored ones, so a headroom change (in
        practice only a flat <-> non-flat transition) takes the wholesale
        path, as does a component past _ELEV_SPLICE_MAX_UNITS."""
        if elevation_changed is not None:
            if not elevation_changed:
                return
            old_corner_rise, old_sprites = self.corner_rise, self.sprites
            self.corner_rise = iso_geometry.corner_rise_px(self.elevations, self.proj, rule=render.SLOPE_CORNER_RULE)
            if not self.with_units:
                return
            headroom = render._unit_rise_headroom_px(self.corner_rise, self.elevations, self.proj)
            mm = self.scenario.map_manager
            splices = (
                _elevation_splices(
                    self.scenario, self.units_by_tile, self.unit_filter,
                    _dilate(elevation_changed, mm.map_width, mm.map_height),
                )
                if headroom == self._headroom
                else None
            )
            if splices is None:
                self._rebuild_unit_layers(headroom)
                return
            if splices:
                old_bboxes: dict = {}
                self.sprites = _reanchor_units(
                    self.building_bboxes, self.sprites, self.scenario, self.proj, self.elevations,
                    self.unit_filter, self.corner_rise, headroom, splices,
                    render.wall_variant_rotation_overrides(self.scenario), *self._layer_resolve_args(),
                    old_bboxes=old_bboxes,
                )
                self._patch_bystander_grid(old_bboxes)
            # An empty batch still rebinds the pack to the new corner_rise.
            self._unit_pack = _refreshed_pack(
                self._unit_pack, splices, self.units_by_tile, old_sprites, self._units_version, old_corner_rise,
                self.sprites, self._units_version, self.corner_rise,
            )
            return

        self.units_by_tile = render._units_by_tile(self.scenario, self.unit_filter) if self.with_units else {}
        self.corner_rise = iso_geometry.corner_rise_px(self.elevations, self.proj, rule=render.SLOPE_CORNER_RULE)
        self._rebuild_unit_layers(render._unit_rise_headroom_px(self.corner_rise, self.elevations, self.proj))

    def _rebuild_unit_layers(self, headroom: int) -> None:
        """The wholesale building_bboxes + sprites rebuild against the current
        corner_rise, recording the headroom it used."""
        with perf_trace.level("sloped-units", 0):
            self._rebuild_unit_layers_timed(headroom)

    def _rebuild_unit_layers_timed(self, headroom: int) -> None:
        self._headroom = headroom
        self._unit_pack = None
        mm = self.scenario.map_manager
        building_bboxes = (
            render._building_bboxes_iso(
                self.scenario, mm.map_width, mm.map_height, self.proj, self.elevations, headroom,
                unit_filter=self.unit_filter,
            )
            if self.with_units
            else {}
        )
        memo, self.sprites, self._sprite_memo = self._sprite_memo, None, None
        if self.with_units and self.sprites_enabled:
            self.sprites, self._sprite_memo = render._drain(render.sprite_draws_by_anchor_sliced(
                self.scenario, self.proj, self.elevations, self.unit_filter,
                corner_rise=self.corner_rise, with_farms=self.layers.farm_overlay,
                tree_scale=self.layers.tree_scale,
                hero_glow=self.layers.hero_glow,
                memo=memo or render.SpriteMemo(),
            ))
        self._set_building_bboxes(
            render.merge_sprite_bboxes(building_bboxes, self.sprites)
            if self.sprites is not None
            else building_bboxes
        )

    def _set_building_bboxes(self, bboxes: dict) -> None:
        """The only place this class assigns building_bboxes, so its chunk-
        bucketed index can never be left describing an older dict. A new
        dict gets a full grid build here; one spliced in place by
        _reanchor_units() gets its grid patched in _patch_bystander_grid().

        Assigns the dict itself rather than a copy, since the patch-unit-
        sources test uses id(cache.building_bboxes) as its staleness proxy."""
        self.building_bboxes = bboxes
        self.bystander_grid = render.build_bystander_grid(bboxes, self.chunk_px)

    def _patch_bystander_grid(self, old_bboxes: dict) -> None:
        """The grid brought up to date after _reanchor_units() rewrote the keys
        in old_bboxes of building_bboxes in place."""
        self.bystander_grid = render.patch_bystander_grid(self.bystander_grid, old_bboxes, self.building_bboxes)

    def can_splice(self, changed: list[UnitSplice] | None) -> bool:
        """Whether invalidate_units(changed) would take its splice path."""
        return self._unit_splice_plan(changed) is not None

    def invalidate_units(self, changed: list[UnitSplice] | None = None, splice_levels: bool = True) -> None:
        """Batch D's D4 splice, SlopedChunkCache's counterpart to
        IsoChunkCache.invalidate_units() -- see that method's own docstring
        for the shared contract (None/ineligible both mean "fall back to
        the wholesale _refresh_source_caches() path, covering the whole
        `changed` batch for free").

        Simpler than Iso's version in one real way: a single mip-0 level
        (no per-level gen-gated laziness to preserve), so there is nothing
        to leave alone -- every splice always touches this cache's one
        building_bboxes/sprites pair, spliced in place with its grid patched
        (_patch_bystander_grid()), so the id()-based staleness proxy tests
        still see the same invariant (dict mutated in place, id() unchanged)
        that a `patch()` would leave it in.

        corner_rise is untouched, same as _refresh_source_caches()'s own
        elevation_changed=empty-set branch: no unit edit moves elevation,
        so the spliced bboxes use self._headroom, the value every other
        entry in building_bboxes was built with. `_pick_planes` is
        terrain-only and stays untouched for the same reason patch() leaves
        it alone here. The batch goes through one _reanchor_units() call,
        one SpriteLayer copy for the whole Convert stroke, with
        _splice_overrides() for the same reason as Iso's, and a batch on
        shared tiles splices its whole component, built first, as Iso's does.

        A refused batch, or splice_levels=False, tries Iso's in-place
        units_by_tile update first; on success the unit layers rebuild eagerly
        as the wholesale path does, minus the units_by_tile walk."""
        self._mutation_epoch += 1
        plan = self._unit_splice_plan(changed) if splice_levels else None
        if plan is None:
            self._trace_splice_refusal(changed, skipped=not splice_levels)
            if changed is not None and self._update_units_by_tile_in_place(changed):
                with perf_trace.phase("units_in_place"):
                    pass
                self.corner_rise = iso_geometry.corner_rise_px(self.elevations, self.proj, rule=render.SLOPE_CORNER_RULE)
                self._rebuild_unit_layers(render._unit_rise_headroom_px(self.corner_rise, self.elevations, self.proj))
                return
            self._refresh_source_caches()
            return
        changed = _removals_first(plan)
        old_version, old_sprites = self._units_version, self.sprites
        self._units_version += 1
        for s in changed:
            _splice_units_by_tile(self.units_by_tile, self.scenario, self.unit_filter, s)
        proj, _sprites, _bb, corner_rise, extra_top = self._unit_level_state(0)
        old_bboxes: dict = {}
        self.sprites = _reanchor_units(
            self.building_bboxes, self.sprites, self.scenario, proj, self.elevations,
            self.unit_filter, corner_rise, extra_top, changed, _splice_overrides(self.scenario, changed),
            *self._layer_resolve_args(), old_bboxes=old_bboxes,
        )
        self._patch_bystander_grid(old_bboxes)
        self._unit_pack = _refreshed_pack(
            self._unit_pack, changed, self.units_by_tile, old_sprites, old_version, self.corner_rise,
            self.sprites, self._units_version, self.corner_rise,
        )

    def _unit_level_state(self, mip: int):
        """This cache's one level. Always current: every source refresh here
        is eager, so there is no stale layer to worry about."""
        assert mip == 0, f"SlopedChunkCache has only mip level 0, got {mip}"
        return self.proj, self.sprites, self.building_bboxes, self.corner_rise, self._headroom

    def canvas_dims(self, mip: int = 0) -> tuple[int, int]:
        """(width, height) in canvas pixels -- proj.canvas_w/canvas_h alone,
        NOT render_terrain_sloped_with_proj()'s own padded allocation
        (that function adds Stepped's skirt_headroom formula on top,
        purely so a flat map's output is byte-identical to render_terrain_
        iso's -- see its own docstring). Sloped paints no skirts (see
        corner_rise_px's own docstring), so real content never needs that
        extra padding; get_chunk()'s existing high-edge clip already
        handles this cache's canvas being smaller than that function's.
        A pixel that rounds just past this tight boundary at the very top
        edge (corner_headroom_px's own float-rounding safety margin, see
        IsoProjection's comment) is silently dropped by _clipped_paint,
        the same accepted tradeoff that function's own docstring documents
        for its scratch-canvas call site -- not new here, not a correctness
        gap this class introduces.

        When sprites_enabled (Track P3-g6), returns render._canvas_pixel_
        dims(self.proj) instead -- the same skirt-padded height Stepped's
        own buffer uses -- rather than the tight bound above. Sloped's tight
        canvas fits terrain alone; a sprite reaches further than that (see
        _sprite_reach_px), and unlike Stepped, Sloped has no skirt margin to
        absorb the difference. This does not eliminate sprite-reach
        clipping -- Stepped doesn't either, see that function's own accepted
        upward gap -- it brings Sloped to the same tolerance Stepped
        already has, instead of a new, Sloped-only regression. set_sprites_
        enabled()'s invalidate_region() call is what makes the newly-
        visible strip actually get painted once this answer changes."""
        assert mip == 0, f"SlopedChunkCache has only mip level 0, got {mip}"
        if self.sprites_enabled:
            return render._canvas_pixel_dims(self.proj)
        return self.proj.canvas_w, self.proj.canvas_h

    def _current_pack(self) -> native_composite.UnitPack | None:
        pack = self._unit_pack
        if pack is not None and pack.matches(self.units_by_tile, self.sprites, self._units_version, self.corner_rise):
            return pack
        return None

    def _unit_pack_of(self, mip: int, create: bool):
        assert mip == 0, f"SlopedChunkCache has only mip level 0, got {mip}"
        return self._native_units() if create else self._current_pack()

    def _native_units(self):
        """IsoChunkCache._native_units() for this cache's one level."""
        if composite_backend.native is None:
            return None
        if not self.with_units:
            return native_composite.NO_UNITS
        pack = self._current_pack()
        if pack is None:
            mm = self.scenario.map_manager
            with perf_trace.level("pack", 0):
                pack = self._unit_pack = native_composite.UnitPack(
                    True, mm.map_width, mm.map_height, self.proj, self.units_by_tile, self.sprites,
                    self._units_version, self.corner_rise,
                )
        return pack

    def _composite_rect(self, mip: int, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        assert mip == 0, f"SlopedChunkCache has only mip level 0, got {mip}"
        # Resolved first so a pack build stays its own level event, outside the phase.
        unit_pack = self._native_units()
        with perf_trace.repaint_phase("composite"):
            return render.composite_rect_sloped(
                self.scenario,
                x0,
                y0,
                x1,
                y1,
                self.corner_rise,
                self.proj,
                self.tile_px,
                self.units_by_tile,
                self.building_bboxes,
                self.with_units,
                sprites=self.sprites,
                bystander_grid=self.bystander_grid,
                layers=self.layers,
                grid=self.grid,
                unit_pack=unit_pack,
                terrain_ids=self._terrain_ids,
            )

    def _prepare_rect(self, mip: int, x0: int, y0: int, x1: int, y1: int) -> tuple | None:
        assert mip == 0, f"SlopedChunkCache has only mip level 0, got {mip}"
        return render.prepare_rect_native(
            True, self.scenario, x0, y0, x1, y1, self.corner_rise, self.proj, self.tile_px, self.building_bboxes,
            self.with_units, self.bystander_grid, self.layers, self.grid, self._native_units(),
            terrain_ids=self._terrain_ids,
        )

    def _pick_plane(self, cx: int, cy: int) -> np.ndarray:
        """Chunk (cx, cy)'s int32 tile-id plane, built on demand and
        memoized -- deliberately held OUTSIDE self._cache and outside
        self._cache_bytes, per the C plan's decision 7.

        The reason is memory, and it was decided against a measured cost:
        Track B-C already accepted a 743MB -> 1127MB RSS regression, and
        keeping an int32 plane resident alongside every cached RGB chunk
        would add ~33% on top of that (4 bytes/px against 3) for no
        correctness gain -- a pick plane is only ever needed for the ONE
        chunk under the cursor, where an RGB chunk is needed for every
        visible one. So this is a tiny separate LRU, not a second channel
        on the main one. Moving the mouse within a chunk is free; crossing
        a chunk boundary costs one ID-only recomposite, which the Gate 2
        benchmark (tools/bench_sloped_patch.py) measures separately from
        the colour path precisely because it is much cheaper.

        Same high-edge clip as get_chunk(), so a plane straddling the
        canvas edge is the same ragged shape its colour chunk is."""
        key = (cx, cy)
        cached = self._pick_planes.get(key)
        if cached is not None:
            self._pick_planes.move_to_end(key)
            return cached

        canvas_w, canvas_h = self.canvas_dims()
        x0, y0 = cx * self.chunk_px, cy * self.chunk_px
        x1, y1 = min(x0 + self.chunk_px, canvas_w), min(y0 + self.chunk_px, canvas_h)
        plane = render.composite_ids_rect_sloped(
            self.scenario, x0, y0, x1, y1, self.corner_rise, self.proj, self.tile_px
        )
        self._pick_planes[key] = plane
        self._pick_planes.move_to_end(key)
        while len(self._pick_planes) > MAX_PICK_PLANES:
            self._pick_planes.popitem(last=False)
        return plane

    def pick_tile(self, sx: int, sy: int) -> tuple[int, int] | None:
        """The (tile_x, tile_y) whose sloped surface covers reference-canvas
        pixel (sx, sy), or None for a background pixel or one off-canvas --
        Track C4's answer to "what did the user just click on", and the
        Sloped counterpart to iso_geometry.screen_to_tile().

        Reference-canvas pixels, which is also scene space: mip decision D2
        pins scene space to the reference level permanently, and this cache
        has only that one level anyway, so viewer.py can pass a scene
        position through unconverted.

        Worth stating because it is the opposite of what the missing-feature
        history suggests: this is STRICTLY BETTER than Stepped's
        screen_to_tile(), which returns None on any skirt-face pixel (a
        documented accepted residual -- there is no exact analytic inverse
        for a skirt). Sloped paints no skirts at all, so it has no such
        hole: every pixel inside the ground outline resolves to exactly one
        tile, which tests/test_sloped_pick.py asserts exhaustively."""
        canvas_w, canvas_h = self.canvas_dims()
        if not (0 <= sx < canvas_w and 0 <= sy < canvas_h):
            return None
        cx, cy = sx // self.chunk_px, sy // self.chunk_px
        plane = self._pick_plane(cx, cy)
        tile_id = int(plane[sy - cy * self.chunk_px, sx - cx * self.chunk_px])
        if tile_id == render.PICK_ID_NONE:
            return None
        map_w = self.scenario.map_manager.map_width
        return tile_id % map_w, tile_id // map_w

    def patch(
        self, bbox: tuple[int, int, int, int], elevation_changed: set | None = None, levels=None
    ) -> None:
        """"Patch, don't drop" for the pick planes too (Batch B step B8),
        the same contract base patch() already holds for colour chunks: a
        resident plane the bbox only partly covers gets that sub-rect
        recomposited and slice-assigned back in place, so the drag's very
        next mouseMoveEvent reads a valid plane instead of paying a full
        512-squared rebuild. Only the planes the bbox covers more than
        PICK_PLANE_PATCH_MAX_FRACTION of are still dropped; see that
        constant for the measurement behind the split. Non-resident planes
        need no action at all, for base patch()'s own reason: _pick_plane()
        always builds fresh against current state.

        Three things about the ordering below are load-bearing:

        An empty elevation_changed returns without touching any plane. A
        plane's ids depend only on corner_rise, proj, tile_px and the map
        dims, never on terrain ids or units, and _refresh_source_caches()
        provably leaves corner_rise alone on that value, so a
        terrain-paint-only edit cannot move a single id.
        elevation_changed=None still means "unknown" and takes the full
        path.

        The rewrite happens AFTER super().patch(), not before it like the
        old drop did, because corner_rise, the plane's one mutable input, is
        rebuilt inside it via _refresh_source_caches(). Rasterizing first
        would paint pre-edit geometry into a plane that then looks current.
        If super().patch() raises, every plane is dropped and the exception
        re-raised: a half-updated corner_rise with warm planes is
        the one state that must not survive. Each survivor is re-get()'d
        afterwards too, since the LRU may have evicted it in between, and
        deliberately WITHOUT move_to_end(): a patch is not a use, and
        promoting it would evict a plane the cursor is about to read.

        Under-coverage would be a silent wrong click target rather than a
        crash (pick_tile feeds unit_pick._pick_unit_sloped, where a 1x1
        unit's membership test is the terrain tile), so it is worth stating
        why it cannot happen. _render_tile_sloped_ids inherits its extent
        from _render_tile_sloped by construction (same corner_rise lookups,
        same d_min, same tile_screen_origin, same sloped_quad_indices entry,
        same _clipped_paint), so a tile's id footprint matches its colour
        footprint exactly; and the bbox is dirty_screen_bbox_sloped's,
        which the colour path already trusts and which is computed
        with_units plus the sprite term, i.e. a strict superset of the
        terrain-only extent. Over-covering is free:
        composite_ids_rect_sloped is a rect-keyed core that starts from a
        fresh PICK_ID_NONE fill, so a rewritten sub-rect is self-clearing
        and needs no ring dilation.

        invalidate_region()/set_unit_filter() keep the wholesale clear:
        neither fires per patched tile, so there's nothing to save by
        narrowing them, and unlike patch() their bbox/call is not on the
        hot per-move path."""
        px0, py0, px1, py1 = bbox
        if px1 <= px0 or py1 <= py0:
            return
        if elevation_changed is not None and not elevation_changed:
            super().patch(bbox, elevation_changed, levels)
            return

        targets = []
        for (cx, cy), plane in list(self._pick_planes.items()):
            plane_x0, plane_y0 = cx * self.chunk_px, cy * self.chunk_px
            # plane.shape, never chunk_px: _pick_plane() clips an edge plane
            # to canvas_dims(), exactly as base patch() uses chunk.shape.
            plane_x1, plane_y1 = plane_x0 + plane.shape[1], plane_y0 + plane.shape[0]
            ix0, iy0 = max(px0, plane_x0), max(py0, plane_y0)
            ix1, iy1 = min(px1, plane_x1), min(py1, plane_y1)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            covered = (ix1 - ix0) * (iy1 - iy0) / (plane.shape[1] * plane.shape[0])
            if covered > PICK_PLANE_PATCH_MAX_FRACTION:
                self._pick_planes.pop((cx, cy), None)
            else:
                targets.append((cx, cy, ix0, iy0, ix1, iy1))

        try:
            super().patch(bbox, elevation_changed, levels)
        except BaseException:
            self._pick_planes.clear()
            raise

        for cx, cy, ix0, iy0, ix1, iy1 in targets:
            plane = self._pick_planes.get((cx, cy))
            if plane is None:
                continue
            plane_x0, plane_y0 = cx * self.chunk_px, cy * self.chunk_px
            sub = render.composite_ids_rect_sloped(
                self.scenario, ix0, iy0, ix1, iy1, self.corner_rise, self.proj, self.tile_px
            )
            plane[iy0 - plane_y0 : iy1 - plane_y0, ix0 - plane_x0 : ix1 - plane_x0] = sub

    def invalidate_region(self, bbox: tuple[int, int, int, int], levels=None, terrain_changed: bool = True) -> None:
        self._pick_planes.clear()
        super().invalidate_region(bbox, levels, terrain_changed)

    def set_unit_filter(self, unit_filter: UnitFilter) -> None:
        # Defensive, not load-bearing: the pick plane is terrain-only (see
        # _render_tile_sloped_ids), so no filter change can alter an id, and
        # Track C5 left it that way rather than folding units in. Kept so the
        # memo can never outlive a source-state change if that ever varies.
        self._pick_planes.clear()
        super().set_unit_filter(unit_filter)

    def set_sprites_enabled(self, enabled: bool) -> None:
        """Turns real .sld sprites on or off on a live cache -- Track P3-g6,
        IsoChunkCache.set_sprites_enabled()'s shape minus the mip loop (this
        cache has only one level).

        No per-level warm-eager step either: that step exists there to keep
        a wait cursor over the cold .sld decode instead of freezing the next
        Qt paint with no indication anything is happening. _refresh_source_
        caches() below already pays that decode eagerly and synchronously
        (it isn't gated behind a lazy _level() rebuild the way IsoChunkCache's
        is), so there is nothing left to warm.

        invalidate_region()'s bbox is read from canvas_dims() AFTER
        self.sprites_enabled flips -- load-bearing when turning sprites ON at
        a low elev_step_pct stop, where canvas_dims() itself grows (the Step
        0 fix): invalidating the OLD, smaller bbox would leave the newly-
        visible bottom strip never composited at all, not just stale."""
        if enabled == self.sprites_enabled:
            return
        self.sprites_enabled = enabled
        self._refresh_source_caches()
        self.invalidate_region((0, 0, *self.canvas_dims(0)), terrain_changed=False)
