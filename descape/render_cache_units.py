"""_ChunkCacheBase's units: sprite extents, splice plans and refusal traces, invalidate_units, flag hosts, the wall verdict."""

from __future__ import annotations

import numpy as np

from descape import (
    perf_trace,
    render_bbox,
    render_sprites,
    render_units,
)
from descape.render_cache_splice import (
    _ELEV_SPLICE_MAX_UNITS,
    REACH_FALLBACK,
    UnitSplice,
    _const_splice_eligible,
    _drop_from_tiles,
    _elevation_component,
    _splice_overrides,
    _splice_plan,
    _splice_plan_or_refusal,
    _union_bbox,
    _verdict_can_move,
)


class _ChunkCacheUnitsMixin:
    """_ChunkCacheBase's unit invalidation (render_cache.py's docstring has the split's rules)."""

    def _layer_resolve_args(self) -> tuple[bool, float, bool, float]:
        """(with_farms, tree_scale, hero_glow, wall_alpha) for
        render_sprites._resolve_unit_sprite(), from this cache's View > Layers state.
        The one source for every _reanchor_units() call and
        sprite_extent_after(), so they cannot differ."""
        layers = self.layers
        return layers.farm_overlay, layers.tree_scale, layers.hero_glow, layers.wall_alpha

    def _unit_level_state(self, mip: int):
        """(proj, sprites, building_bboxes, corner_rise, extra_top_px) for level
        `mip`, as currently installed: never rebuilt by this call. Stepped and
        Sloped implement it; Flat has no per-unit sprite extent and raises.
        The fold that goes with corner_rise is _level_fold()'s."""
        raise NotImplementedError(f"{type(self).__name__} has no unit-edit sprite extent")

    def _level_fold(self) -> np.ndarray | None:
        """The tile_folds() array that goes with the level's corner_rise: None
        off Sloped. Level-independent, since a fold reads elevations only."""
        return None

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
        on screen or over-coverage.

        That holds only for a caller that invalidates in the same handler as
        its mutation. One that mutates first and invalidates later (the
        Eraser's flush) must not trust a level that was stale, or had pending
        elevation work, at the mutation: a composite in between rebuilds it
        from the live scenario (TASK-318, viewer._erase_units())."""
        if self._extent_fallback(changed, mip):
            return REACH_FALLBACK
        return self._layer_extent(changed, mip, after=False)

    def _layer_extent(self, changed: list[UnitSplice], mip: int, after: bool):
        """`changed`'s extent read from level `mip`'s installed layer: sprite
        bboxes over each old (or, `after`, new) footprint tile plus the
        building bbox at that own tile. A reference-canvas bbox or None."""
        _proj, sprites, building_bboxes, _rise, _top = self._unit_level_state(mip)
        box = None
        for s in changed:
            tiles, own = (s.new_tiles, s.new_own_tile) if after else (s.old_tiles, s.old_own_tile)
            if sprites is not None:
                for tile in tiles:
                    box = _union_bbox(box, sprites.bboxes.get(tile))
            if own is not None:
                box = _union_bbox(box, building_bboxes.get(own))
        return None if box is None else self._level_bbox_to_reference(mip, box)

    def _layer_current(self, mip: int) -> bool:
        """Whether level `mip`'s installed layer matches the live units and
        heights, so sprite_extent_after() may read it. Base: never."""
        return False

    def sprite_extent_after(self, changed: list[UnitSplice], mip: int | None):
        """sprite_extent_before()'s counterpart after invalidate_units(changed):
        what the edited units paint at level `mip` now. Read from the layer
        as the old side is when the level is current (_layer_current(): a
        splice kept or restamped it, TASK-031.61 c2x), which _reanchor_units()
        already resolved; a tile shared with another unit over-covers, which
        is safe. Otherwise resolved per unit with _reanchor_units()' own
        arguments, since a level the wholesale fallback left stale has no
        post-edit layer until its next composite. overrides are
        _splice_overrides()': the real dict when a reassigned wall reads it,
        else {}. A moved one takes REACH_FALLBACK even though its sources now
        splice: its reshaped neighbours are not in `changed`."""
        if self._extent_fallback(changed, mip):
            return REACH_FALLBACK
        if self._layer_current(mip):
            return self._layer_extent(changed, mip, after=True)
        return self._resolved_extent(changed, mip)

    def _resolved_extent(self, changed: list[UnitSplice], mip: int):
        """sprite_extent_after()'s per-unit resolve, for a level whose layer is behind."""
        proj, _sprites, _bb, corner_rise, extra_top = self._unit_level_state(mip)
        scenario = self.scenario
        mm = scenario.map_manager
        with_farms, tree_scale, hero_glow, wall_alpha = self._layer_resolve_args()
        overrides = _splice_overrides(scenario, changed)
        box = None
        for s in changed:
            if s.new_own_tile is None or not self.with_units:
                continue
            if self.sprites_enabled:
                contribution = render_sprites._resolve_unit_sprite(
                    scenario, proj, self.elevations, self.unit_filter, corner_rise, overrides, with_farms,
                    s.player_id, s.index, s.unit, tree_scale, hero_glow, wall_alpha, fold=self._level_fold(),
                )
                if contribution is not None:
                    for bbox in contribution.bboxes.values():
                        box = _union_bbox(box, bbox)
            if self.unit_filter.matches(s.player_id, s.unit):
                box = _union_bbox(
                    box, render_bbox._building_bbox_for(s.unit, mm.map_width, mm.map_height, proj, self.elevations, extra_top)
                )
        return None if box is None else self._level_bbox_to_reference(mip, box)

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

    def _trace_splice_refusal(
        self, changed: list[UnitSplice] | None, skipped: bool = False, flipped: bool = False
    ) -> None:
        """Records why invalidate_units(changed) is about to go wholesale, as a
        zero-length `splice_refused=<reason>` phase on the open op or stroke end.
        Recomputed only while Perf Trace is on, and before the sources change.
        `skipped`: the caller passed splice_levels=False, so the reason is `skipped`.
        `flipped`: the batch flipped the wall radian verdict, `radian_flip`."""
        if changed is None or not self.with_units or not perf_trace.is_enabled():
            return
        if flipped:
            reason = "radian_flip"
        elif skipped:
            reason = "skipped"
        else:
            reason = _splice_plan_or_refusal(self.scenario, self.units_by_tile, self.unit_filter, changed)[1]
        with perf_trace.phase(f"splice_refused={reason}"):
            pass

    def _trace_elevation_refusal(self, seed_tiles: set, cap: int) -> None:
        """A patch()'s elevation splice going wholesale, as a zero-length
        `elev_splice_refused=<members>/<cap>` phase on the open op or stroke.
        <members> is the uncapped component seeded from `seed_tiles`, walked
        again only while Perf Trace is on (as _trace_splice_refusal() recomputes
        its plan); `?` when a unit can't be located."""
        if not perf_trace.is_enabled():
            return
        members = _elevation_component(self.scenario, self.units_by_tile, self.unit_filter, seed_tiles, 10**9)
        count = "?" if members is None else len(members)
        with perf_trace.phase(f"elev_splice_refused={count}/{cap}"):
            pass

    def elevation_splice_cap(self) -> int:
        """The component size past which this cache's elevation patch goes
        wholesale (ViewerWindow._elevation_bumps_anyway() reads it)."""
        return _ELEV_SPLICE_MAX_UNITS

    def elevation_seed_tiles(self, elevation_changed: set, new_elev: dict) -> set:
        """The own tiles whose units seed this cache's elevation splice for an
        edit about to change `elevation_changed` to `new_elev` ({tile: level}).
        Called before the edit reaches self.elevations. Own tile only here."""
        return elevation_changed

    def _update_units_by_tile_in_place(self, changed: list[UnitSplice]) -> bool:
        """units_by_tile brought up to date for a membership-only batch without
        render_units._units_by_tile()'s walk over every unit, and _units_version
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
                entry = (s.unit, render_units._unit_color(s.unit, color))
                for tile in s.new_tiles:
                    bucket = by_tile.setdefault(tile, [])
                    pos = len(bucket)
                    while pos and bucket[pos - 1][0].player > player_id:
                        pos -= 1
                    bucket.insert(pos, entry)
        return True

    def invalidate_units(
        self, changed: list[UnitSplice] | None = None, splice_levels: bool = True, cap_levels=None
    ) -> None:
        """Forces this cache's unit-derived structures to rebuild on the next
        composite -- the explicit hook for a unit-editing mutation (place/
        move/delete/reassign), which changes unit.x/y/player without going
        through set_unit_filter()'s own unchanged-filter guard.

        changed (Batch D's D4): None (the default, and every pre-D4 caller's
        exact behaviour) means "unknown" and takes this base wholesale path
        unconditionally. A non-None list is a splice opportunity for
        IsoChunkCache and SlopedChunkCache, through their shared
        _LeveledChunkCache.invalidate_units() override, scoped to the
        single-unit tool paths that can build one -- see UnitSplice's own
        docstring and that override's. This base implementation ignores `changed`
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
        cap_levels (Stepped/Sloped only): the levels a batch past the splice
        caps may still splice. Ignored here and on Flat.
        """
        self._mutation_epoch += 1
        self._record_flag_hosts()
        self._record_wall_verdict()
        self._refresh_source_caches()

    def _record_flag_hosts(self) -> None:
        """Stamps the live garrison hosts as the ones this cache's unit sources
        now reflect. Called at construction and by every unit-source refresh
        (invalidate_units(), _refresh_unit_sources()), never by an elevation
        patch, which a mutate-first tool (the Eraser) can run before its
        flush: stamping there would hide a flip from that flush."""
        self._flag_hosts = render_units.garrison_host_ids(self.scenario)

    def flag_flips(self) -> frozenset[int]:
        """The hosts whose garrison flag appeared or cleared since the last
        unit-source refresh (GH #203): the live garrison_host_ids() against
        what this cache's sources reflect. A pure read, for a scoped edit to
        add the hosts its own splices do not name (viewer._flag_flip_splices())."""
        return render_units.garrison_host_ids(self.scenario) ^ self._flag_hosts

    def _record_wall_verdict(self) -> None:
        """Stamps the file's wall radian verdict (render_units.walls_file_is_radian())
        as the one this cache's unit sources reflect. Only where the sources are
        rebuilt against a verdict that may have moved: construction,
        _refresh_unit_sources(), and invalidate_units() when it goes wholesale
        or its batch could move the verdict (_verdict_can_move()). Never on an
        elevation patch, for _record_flag_hosts()'s reason. Only with sprites
        on: the verdict picks wall frames, which marks never draw, and a
        sprites toggle refreshes the unit sources, so it stamps there."""
        if self._wall_shapes_drawn():
            self._wall_radian = render_units.walls_file_is_radian(self.scenario)

    def _wall_shapes_drawn(self) -> bool:
        return self.with_units and self.sprites_enabled

    def wall_verdict_flipped(self, changed: list[UnitSplice] | None) -> bool:
        """Whether the scoped edit `changed` flipped the file-wide wall radian
        verdict since the last stamp (TASK-583): every wall's drawn shape then
        changes, so the sources and the repaint must both go wholesale. A pure
        read. False with no verdict read unless units and sprites are on and
        some splice could move it; None is already wholesale, so False too."""
        if not self._wall_shapes_drawn() or not changed or not _verdict_can_move(changed):
            return False
        return render_units.walls_file_is_radian(self.scenario) != self._wall_radian

    def _note_wall_verdict(self, changed: list[UnitSplice] | None) -> list[UnitSplice] | None:
        """invalidate_units()' shared opening: `changed`, or None on a verdict
        flip (traced as `radian_flip`), with the verdict re-stamped whenever the
        call goes wholesale or its batch could have moved it."""
        if self.wall_verdict_flipped(changed):
            self._trace_splice_refusal(changed, flipped=True)
            changed = None
        if changed is None or _verdict_can_move(changed):
            self._record_wall_verdict()
        return changed

    def _model_unit_gen(self):
        """scenario.unit_gen, which every UnitEditModel mutator bumps, or None
        for a duck-typed test scenario without one (render_units._unit_gen_cached()'s
        convention). A warm captures it at start and refuses to install across
        a move (TASK-326)."""
        return getattr(self.scenario, "unit_gen", None)
