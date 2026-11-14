"""_LeveledChunkCache, the Stepped and Sloped caches' per-elevation-level base, with its level types and native-prime switches."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from descape import (
    composite_backend,
    iso_geometry,
    native_composite,
    perf_trace,
    render_bbox,
    render_paint,
    render_sprites,
    render_units,
)
from descape.render_cache_base import _ChunkCacheBase
from descape.render_cache_common import _BUILD_TIME_LAYER_FIELDS, LevelWarmJob
from descape.render_cache_splice import (
    VISIBLE_SPLICE_MAX_UNITS,
    UnitSplice,
    _elevation_splices,
    _reanchor_units,
    _refreshed_pack,
    _removals_first,
    _splice_overrides,
    _splice_plan_or_refusal,
    _splice_units_by_tile,
)
from descape.view_layers import LayerState, render_key

# Whether a level warm chains native_prime_job() (TASK-031.85) and a re-arm re-queues an
# unfinished one (TASK-031.82 D6); tools/bench_gui_zoom.py's --no-margin-fix turns it off.
WARM_NATIVE_INPUTS = True
# The finest level native_prime_job() primes: 128 px assembly is one atom over the 3 ms limit
# (TASK-031.85 census); TASK-031.87's sliced assembly raises it.
NATIVE_PRIME_MAX_TILE_PX = 96


@dataclass
class _IsoLevel:
    """One mip level's own state (Phase B-D). tile_px/proj are enumerated
    at construction (see _ChunkCacheBase._init_mip_levels) and are pure
    geometry -- cheap, and IsoChunkCache.canvas_dims() reads .proj
    DIRECTLY, never through _level(), so sizing the cache can never trigger
    a bbox build. building_bboxes is the expensive, PROJECTION-DEPENDENT
    part (see _building_bboxes_iso/_unit_screen_bbox_iso): built lazily on
    first composite at this level, and rebuilt lazily whenever `gen` falls
    behind the cache's own _source_gen -- see _LeveledChunkCache._level()."""

    tile_px: int
    proj: iso_geometry.IsoProjection
    building_bboxes: dict | None = None
    # A5: building_bboxes' own chunk-bucketed index, built beside it in
    # _assemble_level so the two can never disagree about which dict they mean.
    bystander_grid: render_bbox.BystanderGrid | None = None
    gen: int = -1
    # P3-g3. Per-level for the same reason building_bboxes is: a sprite is
    # scaled by 2*half_w/NATIVE_TILE_W, so it is projection-dependent, and it
    # is built in the same lazy step rather than a second one.
    sprites: render_sprites.SpriteLayer | None = None
    # The per-unit contributions `sprites` was merged from, so the next
    # wholesale rebuild re-resolves only changed units. None with sprites off.
    memo: render_sprites.SpriteMemo | None = None
    # Batch F N2's native view of units_by_tile + sprites at this level.
    unit_pack: native_composite.UnitPack | None = None
    # Elevation tiles whose re-anchor this current level has deferred (an
    # off-screen level during a Stepped stroke); _LeveledChunkCache._flush_pending().
    pending_elev: set[tuple[int, int]] = field(default_factory=set)
    # The extra_top_px building_bboxes were built with (Sloped's headroom; always 0 on Stepped).
    extra_top: int = 0
    # The heights array sprites and unit_pack were last synced to (_level_heights()).
    heights: np.ndarray | None = None


@dataclass
class _SlopedLevel(_IsoLevel):
    """A Sloped level (TASK-031.60): _IsoLevel plus the level's own
    corner_rise and headroom, derived at its own proj from the elevations
    whenever level 0's corner_rise is rebound (SlopedChunkCache._level_rise()).
    Geometry, not unit layers: deriving them never builds the level."""

    corner_rise: np.ndarray | None = None
    # The live extra_top target at this level; extra_top is what the bboxes were built with.
    headroom: int = 0
    # The level-0 corner_rise these two were derived alongside.
    rise_source: np.ndarray | None = None


class _LeveledChunkCache(_ChunkCacheBase):
    """The lazy per-level unit layers Stepped and Sloped share (TASK-031.59):
    one _IsoLevel per mip, built in _level() the first time the level is
    composited after its gen falls behind _source_gen, and spliced in place by
    elevation and unit edits on the levels already current.

    A subclass supplies the projections (_init_levels()), canvas_dims(),
    _composite_rect()/_prepare_rect() and _refresh_source_caches(), plus the
    hooks below, which are the only per-level values the styles disagree on.
    Their defaults are Stepped's: no corner_rise for the sprite walk and the
    re-anchor, self.elevations as the pack heights, no extra bbox top. _sloped
    is the native kernels' style flag."""

    _sloped = False
    # The per-level record; Sloped's adds the level's own corner_rise and headroom.
    _level_type: type[_IsoLevel] = _IsoLevel

    def _init_levels(self, projs: dict[int, iso_geometry.IsoProjection]) -> None:
        """One unbuilt _level_type record per projection, the base mip
        machinery, and the counters the lazy build and the splices read."""
        self._levels: dict[int, _IsoLevel] = {
            level: self._level_type(tile_px=p.tile_px, proj=p) for level, p in projs.items()
        }
        self._init_mip_levels({level: lvl.tile_px for level, lvl in self._levels.items()})
        self._source_gen = 0
        # mip -> the _source_gen a native prime walk ran to its end under (TASK-031.82 D6).
        self._native_primed: dict[int, int] = {}
        # Bumped by every splice that leaves _source_gen alone, so a warm
        # started before one can tell its half-walked layer is stale.
        self._splice_epoch = 0
        # Bumped whenever units_by_tile is mutated in place, which a unit
        # pack's identity check can't see.
        self._units_version = 0

    def _level_corner_rise(self, lvl: _IsoLevel) -> np.ndarray | None:
        """The corner_rise the level's sprite walk and re-anchor read."""
        return None

    def _level_heights(self, lvl: _IsoLevel) -> np.ndarray:
        """The heights array the level's sprites and unit pack are built against."""
        return self.elevations

    def _level_extra_top(self, lvl: _IsoLevel) -> int:
        """The extra_top_px the level's building_bboxes are built with."""
        return 0

    def _level_synced(self, lvl: _IsoLevel) -> bool:
        """Whether the level's layers were synced to the live heights and
        extra_top. Always true on Stepped, whose elevations array is written in place."""
        return lvl.heights is self._level_heights(lvl) and lvl.extra_top == self._level_extra_top(lvl)

    def _refresh_all_units(self) -> None:
        """The wholesale half of _refresh_source_caches(): units_by_tile
        rebuilt and every level marked stale, to rebuild lazily in _level()."""
        self.units_by_tile = render_units._units_by_tile(self.scenario, self.unit_filter) if self.with_units else {}
        self._source_gen += 1

    def _refresh_level_units(self, seeds: set, splice_levels=None) -> None:
        """The elevation half of _refresh_source_caches(): the unit layers are
        re-anchored in place rather than rebuilt (the 2026-09-24 anchor-local
        splice). The wholesale level rebuild cost 21-91ms per stroke step with
        sprites on on Stepped, not the ~15-20ms the gen gate was accepted on.

        `seeds` are the own tiles whose units the edit can move:
        elevation_changed on Stepped, which reads a unit's own tile only
        (_resolve_unit_sprite's elevations[uy, ux], _unit_iso_footprint's own
        floored tile), plus elevation_seed_tiles()' corner-changed and
        fold-changed tiles on Sloped. The affected units are exactly those,
        plus any unit sharing a tile with one, which re-derives unchanged
        (_elevation_splices()' component). Each already-current level gets one _reanchor_units()
        call with the real wall overrides (an elevation edit leaves them
        valid) and a patched bystander_grid. A stale level is left stale and
        rebuilds fully in _level(). No gen bump, so _splice_epoch is bumped
        instead for level_warm_job()'s install predicate. A component past
        elevation_splice_cap() bumps the gen, the wholesale fallback.

        An empty component returns early only while every current level is
        _level_synced(): a Sloped edit replaces corner_rise and may move the
        headroom with no unit in reach, and the levels still need their pack
        rebound and their bboxes widened (_splice_level() with no splices).

        splice_levels (patch()'s rebuild_levels, 2026-09-28 stepped-defer
        plan): a current level outside it adds `seeds` to its pending_elev
        instead of splicing, flushed on its next _level()."""
        cap = self.elevation_splice_cap()
        splices = _elevation_splices(self.scenario, self.units_by_tile, self.unit_filter, seeds, cap)
        if splices is None:
            self._trace_elevation_refusal(seeds, cap)
            self._source_gen += 1
            return
        if not splices and all(self._level_synced(lvl) for lvl in self._levels.values() if lvl.gen == self._source_gen):
            return
        overrides = render_units.wall_variant_rotation_overrides(self.scenario)
        self._splice_levels(splices, overrides, self._units_version, splice_levels, seeds)

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
        level directly: this runs inside _flush_pending().

        A level whose extra_top no longer matches _level_extra_top() (a
        Sloped headroom change) rebuilds its bboxes and grid whole instead of
        patching the grid, empty `splices` included: the headroom widens every
        building's bbox, while sprites never read it and the pack never reads
        a bbox. The pack is refreshed from the heights it was synced to onto
        the live ones, so an empty batch still rebinds it."""
        old_sprites = lvl.sprites
        extra_top = self._level_extra_top(lvl)
        old_bboxes: dict = {}
        if splices:
            lvl.sprites = _reanchor_units(
                lvl.building_bboxes, lvl.sprites, self.scenario, lvl.proj, self.elevations,
                self.unit_filter, self._level_corner_rise(lvl), extra_top, splices, overrides,
                *self._layer_resolve_args(), old_bboxes=old_bboxes, fold=self._level_fold(),
            )
        if extra_top != lvl.extra_top:
            mip = next(m for m, other in self._levels.items() if other is lvl)
            # Per level: each Sloped level checks its own headroom (TASK-031.60).
            with perf_trace.phase(f"elev_headroom={lvl.extra_top}->{extra_top}@{mip}"):
                pass
            with perf_trace.level("bboxes", mip):
                lvl.building_bboxes, lvl.bystander_grid = render_paint._drain(self._assemble_level(lvl, lvl.sprites))
                lvl.extra_top = extra_top
        elif splices:
            lvl.bystander_grid = (
                render_bbox.build_bystander_grid(lvl.building_bboxes, self.chunk_px)
                if lvl.bystander_grid is None
                else render_bbox.patch_bystander_grid(lvl.bystander_grid, old_bboxes, lvl.building_bboxes)
            )
        heights = self._level_heights(lvl)
        lvl.unit_pack = _refreshed_pack(
            lvl.unit_pack, splices, self.units_by_tile, old_sprites, old_version, lvl.heights,
            lvl.sprites, self._units_version, heights, self._level_fold(),
        )
        lvl.heights = heights

    def _defers_patch(self, mip: int) -> bool:
        # A stale level's pending set is moot: its rebuild clears it.
        lvl = self._levels[mip]
        return lvl.gen == self._source_gen and bool(lvl.pending_elev)

    def _layer_current(self, mip: int) -> bool:
        lvl = self._levels.get(mip)
        return lvl is not None and lvl.gen == self._source_gen and not lvl.pending_elev and self._level_synced(lvl)

    def _flush_pending(self, mip: int) -> None:
        """Applies a current level's deferred elevation re-anchors: one
        _elevation_splices() over the union, then _splice_level(). A union
        past elevation_splice_cap() marks the level stale instead, so
        _level() rebuilds it wholesale. An empty union still splices when the
        level is not _level_synced(), as _refresh_level_units() does.

        Only valid while units_by_tile agrees with the scenario, which is why
        _level() is the one caller: between a unit edit and its
        invalidate_units() the unit has moved but its buckets have not, and
        a flush there would re-add it at its new tile on top of the old one.
        invalidate_units() needs no flush of its own: its splice reaches
        pending levels too (they are current), and the later flush then
        reads consistent state.

        The Eraser and Convert break that precondition for up to a flush
        interval: they mutate the model live and splice at their flush
        (viewer._erase_units(), _convert_one()). A paint inside that window
        still flushes here. Measured (TASK-326): an erased unit that shared a
        pending tile cannot be located, so _elevation_splices() answers None
        and the level rebuilds wholesale inside this call; a converted one
        re-derives under its new owner. The flush's invalidate_units() then
        leaves the level equal to a fresh cache's (pinned by
        tests/test_stepped_defer_offscreen.py), so the cost is one in-paint
        build, and the repaint is the Eraser's stale-level mark's (TASK-318).
        The Unit Brush (GH #170) has the same window: the live own-tile index
        seeds an add on a pending tile, so this flush already draws it, and
        its add splice clears its new tiles before re-adding (_reanchor_units())."""
        lvl = self._levels[mip]
        with perf_trace.level("flush", mip) as ev:
            # Cleared first so nothing reached from the splice below can re-enter.
            tiles, lvl.pending_elev = lvl.pending_elev, set()
            if ev is not None:
                ev.tiles = len(tiles)
            splices = _elevation_splices(
                self.scenario, self.units_by_tile, self.unit_filter, tiles, self.elevation_splice_cap()
            )
            if splices is None:
                if ev is not None:
                    ev.kind = "flush>rebuild"
                lvl.gen = -1
                return
            if not splices and self._level_synced(lvl):
                return
            overrides = render_units.wall_variant_rotation_overrides(self.scenario)
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
            heights, extra_top = self._level_heights(lvl), self._level_extra_top(lvl)
            with perf_trace.level("build", mip) as ev:
                # P3-g3: resolving and decoding sprites is far too expensive to do
                # per chunk, so it rides this same per-level lazy rebuild.
                sprites = memo = None
                if self.with_units and self.sprites_enabled:
                    sprites, memo = render_paint._drain(self._sprite_walk(lvl))
                if ev is not None:
                    ev.mark("walk")
                bboxes, grid = render_paint._drain(self._assemble_level(lvl, sprites))
                self._commit_level(mip, sprites, memo, bboxes, grid, self._source_gen, heights, extra_top)
                if ev is not None:
                    ev.mark("install")
        return lvl

    def _sprite_walk(self, lvl: _IsoLevel):
        """The level's memo-backed sprite walk, shared by _level() and
        level_warm_job() so the two can't pass different arguments."""
        return render_sprites.sprite_draws_by_anchor_sliced(
            self.scenario, lvl.proj, self.elevations, self.unit_filter,
            corner_rise=self._level_corner_rise(lvl),
            fold=self._level_fold(),
            with_farms=self.layers.farm_overlay,
            tree_scale=self.layers.tree_scale,
            hero_glow=self.layers.hero_glow,
            wall_alpha=self.layers.wall_alpha,
            memo=lvl.memo or render_sprites.SpriteMemo(),
        )

    def _assemble_level(self, lvl: _IsoLevel, sprites: render_sprites.SpriteLayer | None):
        """A level's building_bboxes and bystander grid from an already-built
        sprite layer, as a resumable generator (at most render_paint.ASSEMBLY_SLICE
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
            bboxes = yield from render_bbox._building_bboxes_iso_sliced(
                self.scenario, mm.map_width, mm.map_height, lvl.proj, self.elevations, self._level_extra_top(lvl),
                unit_filter=self.unit_filter,
            )
        if sprites is not None:
            bboxes = yield from render_sprites.merge_sprite_bboxes_sliced(bboxes, sprites)
        grid = yield from render_bbox.build_bystander_grid_sliced(bboxes, self.chunk_px)
        return bboxes, grid

    def _commit_level(
        self, mip: int, sprites: render_sprites.SpriteLayer | None, memo: render_sprites.SpriteMemo | None,
        bboxes: dict, grid: render_bbox.BystanderGrid, gen: int, heights: np.ndarray, extra_top: int,
    ) -> None:
        """Installs an assembled level: field assignment only, so a warm's
        install step costs nothing. The warm's install predicate runs first.
        heights and extra_top are the hooks' values captured at walk start."""
        lvl = self._levels[mip]
        lvl.sprites = sprites
        lvl.memo = memo
        lvl.unit_pack = None
        # A fresh build already reflects every elevation.
        lvl.pending_elev = set()
        lvl.building_bboxes = bboxes
        lvl.bystander_grid = grid
        lvl.heights = heights
        lvl.extra_top = extra_top
        lvl.gen = gen

    def _rebuild_walk(self, lvl: _IsoLevel):
        """The sliced level rebuild both warm jobs run: the sprite walk (sprites
        on), then _assemble_level(). Returns (sprites, memo, bboxes, grid)."""
        sprites = memo = None
        if self.with_units and self.sprites_enabled:
            sprites, memo = yield from self._sprite_walk(lvl)
        bboxes, grid = yield from self._assemble_level(lvl, sprites)
        return sprites, memo, bboxes, grid

    def _rebuild_install(
        self, mip: int, built: tuple, start_gen: int, start_epoch: int, start_unit_gen, heights: np.ndarray,
        extra_top: int,
    ) -> bool:
        """A warm job's install: the five-check predicate (see level_warm_job),
        then _commit_level(). heights and extra_top were captured with start_gen."""
        lvl = self._levels[mip]
        if (
            self._source_gen != start_gen
            or self._splice_epoch != start_epoch
            or self._model_unit_gen() != start_unit_gen
            or lvl.gen == self._source_gen
            or self._level_heights(lvl) is not heights
        ):
            return False
        self._commit_level(mip, *built, start_gen, heights, extra_top)
        return True

    def can_splice(self, changed: list[UnitSplice] | None) -> bool:
        """Whether invalidate_units(changed) would take its splice path."""
        return self._unit_splice_plan(changed) is not None

    def invalidate_units(
        self, changed: list[UnitSplice] | None = None, splice_levels: bool = True, cap_levels=None
    ) -> None:
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

        The one exception is a batch past the splice caps with `cap_levels`
        given and sprites on (_splice_visible()): it bumps the gen once and
        restamps only the cap_levels it spliced, since splicing every level
        would cost more than rebuilding the ones off screen when reached.

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
        wholesale path does. A zero-length `units_in_place` phase marks it.

        A garrison flag flip on a host outside `changed` is the caller's to
        add (viewer._flag_flip_splices()); every path here stamps the live
        hosts as reflected (_record_flag_hosts()).

        A batch that flips the file's wall radian verdict (wall_verdict_flipped(),
        TASK-583) is taken as None: every wall's drawn shape moved, so no
        splice, component or in-place update can cover it."""
        self._mutation_epoch += 1
        self._record_flag_hosts()
        changed = self._note_wall_verdict(changed)
        plan = self._unit_splice_plan(changed) if splice_levels else None
        if plan is None:
            if splice_levels and cap_levels and self._splice_visible(changed, cap_levels):
                return
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

    def _splice_visible(self, changed: list[UnitSplice] | None, cap_levels) -> bool:
        """invalidate_units()' path for a batch past both splice caps (TASK-031.61
        c2x): the same plan under VISIBLE_SPLICE_MAX_UNITS, spliced into
        units_by_tile and into only the `cap_levels` levels that are current.
        One _source_gen bump marks every other level stale, as the wholesale
        path does, and also refuses an in-flight warm through its existing
        predicate; each spliced level is then restamped current. A zero-length
        `splice_visible=<splices>` phase marks it.

        False, touching nothing, with sprites off, with no cap_level current, or
        when the plan is refused or holds more than VISIBLE_SPLICE_MAX_UNITS
        splices. The sprites-off refusal dates from a ~4 ms level build; the
        TASK-031.26 zoom pass measured that rebuild at 11.5-37.3 ms on old-allies
        and June, and whether the splice beats it there is unmeasured. The
        targets are read before anything moves. A target with pending_elev
        keeps it: its later flush reads the consistent state this leaves."""
        if changed is None or not (self.with_units and self.sprites_enabled):
            return False
        targets = [
            self._levels[m] for m in cap_levels if m in self._levels and self._levels[m].gen == self._source_gen
        ]
        if not targets:
            return False
        cap = VISIBLE_SPLICE_MAX_UNITS
        plan, _why = _splice_plan_or_refusal(
            self.scenario, self.units_by_tile, self.unit_filter, changed, cap=cap, component_cap=cap
        )
        if plan is None or len(plan) > cap:
            return False
        plan = _removals_first(plan)
        old_version = self._units_version
        for s in plan:
            _splice_units_by_tile(self.units_by_tile, self.scenario, self.unit_filter, s)
        self._units_version += 1
        self._source_gen += 1
        overrides = _splice_overrides(self.scenario, plan)
        for lvl in targets:
            self._splice_level(lvl, plan, overrides, old_version)
            lvl.gen = self._source_gen
        self._splice_epoch += 1
        with perf_trace.phase(f"splice_visible={len(plan)}"):
            pass
        return True

    def level_warm_job(self, mip: int) -> LevelWarmJob | None:
        """The resumable warm for one level -- see _ChunkCacheBase's own
        docstring for the shape.

        None with units off (a level then has nothing to assemble) or when
        the level is already current, which is the common case for the level
        the opening paint just built. Sprites off still warms: the level's
        assembly alone (_building_bboxes_iso, the bystander grid) measured
        13.5-31.2 ms dev on old-allies and 11.5-37.3 on June (TASK-031.26 zoom
        pass), built inside the first paint at that level, and with no warm the load warm never
        composites its chunks either (_queue_load_warm waits for residency).

        **The install predicate is five checks, and `lvl.gen ==
        self._source_gen` is the one that isn't obvious.** The first two are
        staleness (the source moved under a warm that started before it, so
        the layer describes a scenario that no longer exists): a gen bump,
        or a splice (_splice_epoch), which moves units or elevations without
        one. The third is staleness on the model side: scenario.unit_gen
        (_model_unit_gen()) moved, which catches a mutation the cache has not
        heard of yet. The Eraser and Convert mutate mid-drag and splice only
        at their flush, so neither counter above moves in between, and a
        `del` under the walk's enumerate() cursor skips a unit (TASK-326). The
        fourth is the opposite case: a real paint got there first
        and built an equivalent level, and overwriting it would throw away a
        layer already wired into the chunk cache to install a fresh copy of
        the same thing. The fifth is staleness again: _level_heights() is no
        longer the array captured at start. Sloped replaces corner_rise on
        every elevation patch, even one whose splice is empty and moves
        neither counter; on Stepped it never fires.

        A current level with deferred elevation work gets a flush job instead,
        sprites on or off (_flush_warm_walk), so no chunk warm or paint pays
        the flush (2026-09-29 warm-tick plan)."""
        lvl = self._levels[mip]
        if lvl.gen == self._source_gen and lvl.pending_elev:
            return self._flush_warm_job(mip)
        if not self.with_units:
            return None
        start_gen = self._source_gen
        start_epoch = self._splice_epoch
        start_unit_gen = self._model_unit_gen()
        if lvl.gen == start_gen:
            return None
        heights, extra_top = self._level_heights(lvl), self._level_extra_top(lvl)
        # The walk and the assembly both step here; install only commits.
        return LevelWarmJob(
            gen=self._rebuild_walk(lvl),
            install=lambda built: self._rebuild_install(
                mip, built, start_gen, start_epoch, start_unit_gen, heights, extra_top
            ),
        )

    def _flush_warm_job(self, mip: int) -> LevelWarmJob:
        """level_warm_job() for a current level with pending tiles: the walk's
        payload is None when the flush left the level current, else
        (rebuild, start_gen, start_epoch, start_unit_gen, heights, extra_top)."""

        def install(payload) -> bool:
            return payload is None or self._rebuild_install(mip, *payload)

        return LevelWarmJob(gen=self._flush_warm_walk(mip), install=install, kind="flush")

    def level_warmable(self, mip: int) -> bool:
        """level_warm_job(mip) is not None, without building the job."""
        lvl = self._levels[mip]
        if lvl.gen == self._source_gen and lvl.pending_elev:
            return True
        return self.with_units and lvl.gen != self._source_gen

    def _flush_warm_walk(self, mip: int):
        """_flush_pending() as one step, then, if an over-cap union marked the
        level stale, level_warm_job()'s sliced rebuild. Re-checks on its first
        step, since a paint may have flushed the level after start(). Safe
        between ticks: every mutation cancels warms before it runs and re-arms
        after invalidate_units(), so units_by_tile agrees with the scenario.
        That covers the Eraser and Convert too, which cancel at their live
        mutation and arm nothing until its flush (viewer._cancel_warms(),
        TASK-326); unit_gen in the payload is the install's second line."""
        lvl = self._levels[mip]
        if lvl.gen == self._source_gen and lvl.pending_elev:
            self._flush_pending(mip)
        if lvl.gen == self._source_gen:
            return None
        # Captured after the flush: an under-cap one bumps _splice_epoch.
        start_gen, start_epoch, start_unit_gen = self._source_gen, self._splice_epoch, self._model_unit_gen()
        heights, extra_top = self._level_heights(lvl), self._level_extra_top(lvl)
        yield
        built = yield from self._rebuild_walk(lvl)
        return built, start_gen, start_epoch, start_unit_gen, heights, extra_top

    def is_level_resident(self, mip: int) -> bool:
        """Mirrors level_warm_job()'s own staleness check (`lvl.gen ==
        self._source_gen`) rather than level_warm_job()'s None-or-not
        answer: the two predicates ask different questions (this one is a
        pure "is get_chunk() safe to call right now", level_warm_job() also
        factors in "is there anything worth WARMING" -- with units off a
        level can be stale yet still return a level_warm_job() of None,
        since a terrain-only rebuild has nothing to assemble).
        tests/test_margin_warm.py pins the two together after
        a completed warm so a future change to one can't silently diverge
        from the other.

        A current level with deferred elevation work is not resident either:
        get_chunk() would flush it, and an over-cap flush rebuilds it
        wholesale inside that call. level_warm_job() flushes it instead
        (2026-09-29 warm-tick plan)."""
        lvl = self._levels[mip]
        return lvl.gen == self._source_gen and not lvl.pending_elev

    def native_prime_job(self, mip: int, after_level_warm: bool = False) -> LevelWarmJob | None:
        """native_composite.prime_walk() for this level, as a job whose install
        is a no-op: pack_warm_job()'s shape. It only fills caches earlier
        than a first prepare would, so a chunk before or after it is
        byte-identical. Its one state is the finished-walk marker
        (native_primed()), which margin_warm's defer and the re-queue read.

        None with WARM_NATIVE_INPUTS off, on the numpy backend, above
        NATIVE_PRIME_MAX_TILE_PX, or unless after_level_warm says a level
        warm is queued ahead or the level is resident and not yet primed
        under its current _source_gen (TASK-031.82 D6: a walk a crossing
        cancelled is re-queued by the next re-arm). So a re-arm over
        resident, primed levels queues nothing and starts no timer. A
        restarted walk re-runs its finished atoms as cache hits.
        Between atoms the walk stops quietly once the level leaves residency."""
        if not WARM_NATIVE_INPUTS or composite_backend.native is None:
            return None
        if self._levels[mip].tile_px > NATIVE_PRIME_MAX_TILE_PX:
            return None
        if not after_level_warm and (not self.is_level_resident(mip) or self.native_primed(mip)):
            return None
        return LevelWarmJob(gen=self._native_prime_walk(mip), install=lambda _payload: True, kind="native")

    def native_primed(self, mip: int) -> bool:
        """Whether a prime walk ran to its end under the current _source_gen.
        Readiness only (D3's deferral, D6's re-queue): chunk output never reads it."""
        return self._native_primed.get(mip) == self._source_gen

    def _native_prime_walk(self, mip: int):
        lvl = self._levels[mip]
        start_gen = self._source_gen
        heights = (lambda: (self._level_corner_rise(lvl), self._level_fold())) if self._sloped else None
        walk = native_composite.prime_walk(
            lvl.tile_px, lvl.proj.elev_step, sloped=self._sloped, blend_keys=self._prime_blend_keys, heights=heights
        )
        # The walk's first step is empty; each later next() runs one atom, after the check.
        for _ in walk:
            yield
            if not self.is_level_resident(mip):
                walk.close()
                return
        # A paint can rebuild the level between atoms while residency holds.
        if self._source_gen == start_gen:
            self._native_primed[mip] = start_gen

    def _prime_blend_keys(self) -> list | None:
        """The KeyMap's keys for prime_walk(), None while blending is off."""
        key_map = self._blend_key_map
        if key_map is None or not key_map.keys or not render_paint.blend_effective(self.layers):
            return None
        return key_map.keys

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
        before invalidate_region() empties the cache and takes it away, and
        invalidate_region()'s bbox is read from canvas_dims() after the flip:
        Sloped's canvas grows with sprites on at a low elev_step_pct stop, and
        the old, smaller bbox would leave the new bottom strip never composited."""
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
        and a re-resize rather than that method's cold decode, and Translucent
        Walls the same for every wall and gate. Flipping the farm layer
        re-walks the units but decodes no new .sld, so it does not pay
        that method's 0.5-4.1s cold-decode case."""
        layers = render_key(layers)
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
        _level(): a stale level stays stale (see sprite_extent_before()). Both
        height terms come from the hooks, live: a stale level's own extra_top
        paired with the live corner_rise would under-cover after a refused,
        headroom-raising undo, while the live one over-covers, which is safe."""
        lvl = self._levels[mip]
        return lvl.proj, lvl.sprites, lvl.building_bboxes, self._level_corner_rise(lvl), self._level_extra_top(lvl)

    @property
    def building_bboxes(self) -> dict:
        """Level 0's building_bboxes, forcing a rebuild first if stale.
        Debugging/introspection accessor only -- composite_rect_iso() is
        always called with a specific level's own bboxes via _level(mip),
        never through this property."""
        return self._level(0).building_bboxes

    def _current_pack(self, lvl: _IsoLevel) -> native_composite.UnitPack | None:
        pack = lvl.unit_pack
        if pack is not None and pack.matches(
            self.units_by_tile, lvl.sprites, self._units_version, self._level_heights(lvl)
        ):
            return pack
        return None

    def _unit_pack_of(self, mip: int, create: bool):
        lvl = self._levels[mip]
        return self._native_units(lvl, mip) if create else self._current_pack(lvl)

    def _native_units(self, lvl: _IsoLevel, mip: int):
        """The level's unit pack for the native composite's N2 path (None on
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
                    self._sloped, mm.map_width, mm.map_height, lvl.proj, self.units_by_tile, lvl.sprites,
                    self._units_version, self._level_heights(lvl), self._level_fold(),
                )
        return pack

    def _derive_pack_over(self, mip: int, rects: list[tuple[int, int, int, int]]) -> None:
        # Resolves the level (a build or deferred flush) and pack the first prepare would.
        lvl = self._level(mip)
        pack = native_composite._units_of(self._native_units(lvl, mip))
        if pack is None or not self.with_units or not rects:
            return
        pending = pack.pending()
        if not pending.size:
            return
        x0, y0 = min(r[0] for r in rects), min(r[1] for r in rects)
        x1, y1 = max(r[2] for r in rects), max(r[3] for r in rects)
        # A serial composite derives these inside `composite`; the level and pack resolve above stay outside.
        with perf_trace.repaint_phase("prepare"):
            if pending.size <= (x1 - x0) * (y1 - y0) / lvl.tile_px**2:
                pack.derive_pending()
                return
            # Each rect's candidates are a subset of the bounding rect's: pack_warm_job()'s ensure(), called earlier.
            candidates = render_bbox._rect_candidates(
                self.scenario, x0, y0, x1, y1, lvl.proj, self.with_units, lvl.building_bboxes, lvl.bystander_grid
            )
            if candidates.shape[0]:
                pack.ensure(candidates[:, 1].astype(np.int64) * self.scenario.map_manager.map_width + candidates[:, 0])

    def _derive_pack_for_slices(self, mip: int, rects: list[tuple[int, int, int, int]]) -> None:
        """Derives every pending tile (pack_warm_job()'s own argument) when
        there are no more of them than the slices' summed tile area, else each
        slice's own candidates. Both derive at about 1 us per tile (TASK-031.67
        step 0), and the first form skips re-deriving the candidates, which the
        prepares do again; the second keeps a pack the level build just made
        from paying a whole-map derive (20-170 ms on the corpus) for a few slices."""
        lvl = self._level(mip)
        pack = native_composite._units_of(self._native_units(lvl, mip))
        if pack is None or not self.with_units or not rects:
            return
        pending = pack.pending()
        if not pending.size:
            return
        tiles = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in rects) / lvl.tile_px**2
        with perf_trace.repaint_phase("prepare"):
            if pending.size <= tiles:
                pack.derive_pending()
                return
            w = self.scenario.map_manager.map_width
            idx = []
            for x0, y0, x1, y1 in rects:
                candidates = render_bbox._rect_candidates(
                    self.scenario, x0, y0, x1, y1, lvl.proj, self.with_units, lvl.building_bboxes, lvl.bystander_grid
                )
                if candidates.shape[0]:
                    idx.append(candidates[:, 1].astype(np.int64) * w + candidates[:, 0])
            if idx:
                pack.ensure(np.concatenate(idx))
