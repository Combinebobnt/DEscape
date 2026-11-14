"""ViewerWindow's unit edit path: _unit_edit, _after_unit_mutation, the unit index patch and the chunk-cache patch path."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from contextlib import contextmanager

import numpy as np

from descape import (
    perf_trace,
    render_cache_splice,
    unit_pick,
    unit_sprites,
)
from descape.edit_history import (
    CompositeDiffRecord,
    TileDiffRecord,
    UnitDiffRecord,
)
from descape.render_bbox import dirty_screen_bbox_iso, dirty_screen_bbox_sloped, dirty_screen_bboxes_iso_grouped
from descape.render_cache_splice import REACH_FALLBACK, UnitSplice
from descape.render_paint import flat_tile_rects, tile_pixels_for_map
from descape.render_units import unit_occupied_tiles, unit_own_tile_index
from descape.unit_model import (
    UnitEditModel,
    merged_field_record,
)
from descape.viewer_common import (
    _RotationDrag,
)

# Stepped/Sloped batch unit edits patch the visible level's bbox eagerly while that recomposites
# at most this multiple of the visible chunks' area, else evict the bbox's chunks for the next
# paint. Applies to the visible-level split (_repaint_unit_edit_split()): the tight bbox and,
# since 2026-09-30, the reach-padded one too. Measured with tools/bench_stroke_end.py
# --area-ratio: see _patch_area_exceeds_viewport().
_TIGHT_PATCH_AREA_RATIO = {"stepped": 0.3, "sloped": 1.0}


# Stepped only (TASK-031.61 C1). A batch past the ratio above patches one rect per visible-level
# chunk, sized from 16x16-tile blocks of the edit, while those rects recomposite at most this
# multiple of the visible chunks' area; past it, it evicts. Measured with tools/bench_stroke_end.py
# --rect-ratio: see _patch_area_exceeds_viewport().
_RECT_PATCH_AREA_RATIO = 0.25
_UNIT_RECT_BLOCK_TILES = 16


@dataclasses.dataclass
class _UnitEditBlock:
    """One block of a Stepped batch unit edit (ViewerWindow._unit_edit_blocks()):
    its splices and what they paint at the visible level before and after
    the edit, reference bboxes or None (cache.sprite_extent_before()/after())."""

    key: tuple[int, int] | None
    splices: list
    before: tuple[int, int, int, int] | None = None
    after: tuple[int, int, int, int] | None = None


class _ViewerUnitEditMixin:
    """ViewerWindow's unit edit and repaint path (viewer.py's docstring has the split's rules)."""

    # -- unit editing --------------------------------------------------------
    #
    # Every unit mutation (place/move/delete/reassign) goes through
    # _unit_edit(). Unlike _trigger_edit(), which commits even on a raised
    # structural_edit() because a failed structural edit still dirties every
    # blob, every UnitEditModel operation validates BEFORE it mutates (add()'s
    # player-range check, remove()'s garrison-reference guard, reassign()'s
    # range check, and set_position()/reassign()'s _locate() lookup all raise
    # first, before touching any field) -- so an exception here always means
    # nothing was touched, and aborting rather than committing a no-op record
    # is correct.

    @contextmanager
    def _unit_edit(
        self,
        model: UnitEditModel,
        label: str,
        players: Sequence[int],
        splices: list | None = None,
        fields_only: bool = False,
        drag: _RotationDrag | None = None,
    ):
        """Open one recorded unit edit, and close it whatever happens.

        `players` is the caller's declaration of which player list(s) the
        upcoming edit will touch -- one for place/move/delete, both source and
        destination for reassign -- passed straight through to
        UnitEditModel.begin_unit_edit(). Enter only once a real change is
        confirmed (e.g. the new value differs from the current one): unlike
        _trigger_edit, there is no failure mode here that still needs a
        record, so a spurious enter would record a genuine phantom undo step.

        splices (Batch D's D5): None (the default) keeps this method's exact
        pre-D5 behaviour -- a wholesale _after_unit_mutation(None). A caller
        doing a single-unit splice-eligible edit (Move/Nudge/Rotate/Set
        field/gate orientation/Place/single Delete) instead passes an
        initially-empty list and appends one render_cache_splice.UnitSplice to it
        per touched unit from INSIDE the `with` block, after that unit's own
        model.set_*()/add()/remove() call -- a splice needs the unit's
        post-edit state, which doesn't exist before then. This method only
        reads `splices` back once the block has exited normally, so every
        append must already have happened by that point (asserted below:
        entering with a non-None `splices` and leaving it empty means the
        caller forgot to record the very edit it declared, which would
        silently skip the cache/index refresh this whole mechanism exists to
        do). A gate orientation cycle that clears or reshapes walls
        (GH #339) goes wholesale instead: its removals shift list indices.
        So does a single Delete that reshapes the walls beside a removed
        wall or gate (GH #340).

        fields_only (Batch D's D6): passed straight through to
        UnitEditModel.begin_unit_edit(). True for every splice-eligible
        caller above except Place and single Delete -- add()/remove() are
        membership mutators UnitEditModel refuses to run inside a
        fields_only edit, so only the pure set_position/set_rotation/
        set_unit_const/set_status/set_capture_flag/set_wall_variant callers
        (Move/Nudge/Rotate/Set field/gate orientation, and a move's wall
        reshapes, GH #266) may pass it.

        drag (TASK-548): a compass spoke, a fields_only edit folded into the
        drag's record (_fold_drag_record) rather than pushed on its own, with
        no status line of its own: the drag logs one at its release.

        terrain_current: no caller of this method writes a tile, so the tail
        is told the terrain mirrors are current (_after_unit_mutation()),
        except inside an open Erase stroke, whose Terrain Overlay writes are
        not in the mirrors until it ends."""
        with perf_trace.op(label):
            model.begin_unit_edit(players, fields_only=fields_only)
            try:
                yield model
            except Exception:
                model.abort_unit_edit()
                # A raise can leave the pick index half-patched (a Move's in-place patch), and
                # Units entry reuses a non-None index: drop it so the next need rebuilds it.
                self._stack_cycle = None
                self.map_view.set_unit_index(None)
                raise
            if drag is None:
                model.commit_unit_edit(label, self.edit_history)
            else:
                self._fold_drag_record(drag, model.commit_unit_edit(label, self.edit_history, push=False))
            if splices is not None:
                assert splices, "a splice-tracking _unit_edit exited with no UnitSplice recorded"
            # An open Erase stroke's Terrain Overlay writes are not in the terrain mirrors yet.
            self._after_unit_mutation(splices, terrain_current=self._erase is None)
            self._update_title()
            self._update_edit_actions()
            if drag is None:
                self._log_status(label)

    def _fold_drag_record(self, drag: _RotationDrag, record: UnitDiffRecord) -> None:
        """A compass spoke's unpushed record: folded into the drag's own if that
        is still the newest one, else pushed for the drag to follow from here."""
        if drag.record is not None:
            merged = merged_field_record(drag.record, record)
            if self.edit_history.replace_top(drag.record, merged):
                drag.record = merged
                return
        self.edit_history.push_unit_record(record)
        drag.pushes += 1
        drag.record = record

    def _unit_footprint(self, unit) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
        """(own_tile, occupied_tiles) for `unit` at its CURRENT x/y/unit_const
        -- the snapshot a render_cache_splice.UnitSplice needs on each side of a
        single-unit edit (Batch D's D5), taken by calling this once before
        and once after the model mutation."""
        mm = self.scenario.map_manager
        own_tile = (int(unit.x), int(unit.y))
        tiles = tuple(unit_occupied_tiles(unit, mm.map_width, mm.map_height) or ())
        return own_tile, tiles

    def _unit_list_index(self, player_id: int, unit) -> int:
        """`unit`'s position in scenario.unit_manager.units[player_id] --
        render_sprites._resolve_unit_sprite()'s own `i`, which a render_cache.
        UnitSplice built outside UnitEditModel (Batch D's D5) has no other
        way to learn. A linear scan by identity: Unit isn't hashable and
        UnitEditModel._locate() is that model's own private lookup, not
        exposed for a caller building a splice from outside it. Safe only
        for a splice-eligible op, which never reorders the list before this
        runs -- see render_cache_splice.UnitSplice's own docstring."""
        for index, candidate in enumerate(self.scenario.unit_manager.units[player_id]):
            if candidate is unit:
                return index
        raise ValueError("unit not found in its own player's list")

    def _patch_unit_index_for_move(self, player_id: int, unit, old_bounds) -> None:
        """Batch D's D5: the unit-pick index counterpart of a splice-eligible
        Move/Nudge/Set-field edit. A no-op if the index doesn't exist yet
        (not in Units mode) -- the same guard _rebuild_unit_index()'s own
        callers rely on, since there is nothing here to keep in step with."""
        index = self.map_view._unit_index
        if index is not None:
            unit_pick.patch_index_for_move(self.scenario, index, player_id, unit, old_bounds)

    def _elevation_changed_tiles(self, record) -> set[tuple[int, int]]:
        """E: the (x, y) tiles whose elevation `record`'s tile child (or `record`
        itself) changes. Empty in Flat+Iso, whose patch flattens elevation away."""
        if self._terrain_style == "flat":
            return set()
        children = record.children if isinstance(record, CompositeDiffRecord) else [record]
        terrain = self.scenario.map_manager.terrain
        return {
            (terrain[i].x, terrain[i].y)
            for child in children
            if isinstance(child, TileDiffRecord)
            for i, old, new in child.changes
            if old[1] != new[1]
        }

    def _elevation_bumps_anyway(self, elevation_changed: set[tuple[int, int]]) -> bool:
        """Whether the elevation patch after a unit edit is certain to bump the
        cache's gen anyway (Stepped and Sloped alike), so a level splice of
        the unit edit before it would be wasted. Counts the seed set
        render_cache_splice._elevation_splices() starts from: the filter-matching,
        on-map units whose own tile is in the cache's elevation_seed_tiles(), a
        lower bound on its component, against its elevation_splice_cap(). The
        cache's elevations are still pre-edit here, so the new levels come
        from the map. Call it after the unit edit: the own-tile index is then
        the post-edit one the patch reads, memoized on unit_gen."""
        cache = self._cache
        if not elevation_changed or cache is None or not cache.with_units:
            return False
        mm = self.scenario.map_manager
        w, h = mm.map_width, mm.map_height
        new_elev = {t: mm.get_tile(*t).elevation for t in elevation_changed}
        tiles = cache.elevation_seed_tiles(elevation_changed, new_elev)
        own_index = unit_own_tile_index(self.scenario)
        cap = cache.elevation_splice_cap()
        seeds = 0
        for own in tiles:
            for player_id, _index, unit in own_index.get(own, ()):
                if cache.unit_filter.matches(player_id, unit) and unit_occupied_tiles(unit, w, h) is not None:
                    seeds += 1
                    if seeds > cap:
                        return True
        return False

    def _after_unit_mutation(
        self,
        changed: list[UnitSplice] | None = None,
        defer_index: bool = False,
        rebuild_index: bool = False,
        batch: bool = False,
        sources_refreshed: bool = False,
        terrain_bbox: tuple[int, int, int, int] | None = None,
        terrain_current: bool = False,
        terrain_tiles=None,
        pre_stale_levels=frozenset(),
    ) -> None:
        """Shared invalidation tail for every unit mutation, whether driven
        by a tool (_unit_edit's own commit, above) or by undo/redo
        (_move_history's "unit" branch).

        Units paint in every mode (_build_filters_button's own docstring:
        filtering "affects View and Terrain too"), and Ctrl+Z is global, so
        the cache-invalidate-and-repaint step below must NOT be mode-gated --
        only the inspector half of the selection reconciliation that follows
        is, mirroring _refresh_selection_after_filter()'s own mode gate.

        Two DISTINCT invalidate_region()s are both required, the same pair
        _on_filter_changed()'s own set_unit_filter() call bundles internally:
        self._cache.invalidate_units() only rebuilds the cache's SOURCE
        structures (units_by_tile, per-level sprite layers via the source-gen
        bump) -- it never touches self._cache's own dict of already-
        COMPOSITED chunk pixel arrays, so a chunk rendered before this edit
        stays cached with the pre-edit pixels. _ChunkCacheBase.
        invalidate_region() (called on the CACHE, not on self.map_view) is
        the separate primitive that actually evicts those, forcing a real
        recomposite next paint. self.map_view.invalidate_region() only
        schedules that repaint; without the cache-level call first it just
        re-blits the same stale pixels. Missing this step is exactly what
        made place/move/delete/reassign change the model correctly while the
        map kept showing the pre-edit unit -- caught after the fact, not by
        any of this slice's own tests, none of which asserted on pixels.

        changed (Batch D's D5): None (the default, and every pre-D5 caller's
        exact behaviour) means "unknown" and takes the wholesale branch
        below. A single-unit splice-eligible edit instead passes a non-empty
        list of render_cache_splice.UnitSplice -- see that class's own docstring --
        and _patch_unit_edit_cache() below scopes both the source splice
        (D4's invalidate_units(changed)) and the actual repaint to the bbox
        that edit's old/new footprints could have touched, instead of the
        whole canvas. The one exception is a `changed` that flips the file's
        wall radian verdict (cache.wall_verdict_flipped(), TASK-583: Delete,
        Cut or the Eraser taking the last radian wall, or an undo bringing one
        back), which changes every wall's drawn shape: it takes the wholesale
        branch, sources and whole-canvas repaint both.

        The unit-pick index is deliberately NOT touched here when `changed`
        is given: patch_index_for_move()/patch_index_for_add() each need
        that specific op's own pre-edit bounds/filter state, which only the
        call site still has by the time this method runs -- so every
        `changed`-passing caller has already patched (Move/Nudge/Set field/
        Place) or fully rebuilt (gate orientation's span change, single
        Delete) self.map_view._unit_index itself, before ever reaching here.

        defer_index: the Convert brush's mid-stroke repaint only. Repaints
        and re-arms the warm, but skips the index rebuild and selection
        reconciliation; the stroke's final call does both.

        rebuild_index: forces the full _rebuild_unit_index() even though
        `changed` is given. For a caller whose splices were never matched by
        an index patch, i.e. the Convert stroke's final call.

        batch: a _if_spliceable() caller (Draw's stroke end, membership
        undo/redo), whose bbox may be evicted rather than patched; see
        _patch_unit_edit_cache().

        sources_refreshed: the wholesale branch's caller already ran its own
        invalidate_units() before its tile patch (Paste Region, and undo/redo's
        sources-first branch), so it is skipped here; the whole-canvas
        eviction pair is kept.

        terrain_bbox: with `changed`, a terrain bbox whose sources are
        refreshed but which is not yet composited (TASK-031.61 A4, see
        _refresh_after_history_move()). _patch_unit_edit_cache() repaints it
        with the unit bbox on every exit.

        terrain_current: with `changed`, the caller's promise that the cache's
        terrain mirrors already match every tile, so the scoped repaint's
        evictions skip their re-read (TASK-031.61 C0, about 10 ms each on a
        near-canvas bbox). Only Draw's stroke end, a splice-carrying history
        move and _unit_edit() pass it. False (the default) keeps the re-read:
        an Erase stroke's Terrain Overlay writes reach the mirrors only at
        its next flush or its end, after that call's own unit step.

        terrain_tiles: with terrain_bbox, the tile indices that bbox was sized
        from (TASK-031.61 C1), so the scoped repaint can size the terrain per
        block of tiles too. None with a terrain_bbox keeps the union bbox.

        pre_stale_levels: with `changed`, the Eraser's marked levels
        (TASK-318, _erase_units()); passed to _patch_unit_edit_cache().
        """
        # A third invalidation, on the same terms as the two below: an
        # in-flight warm is walking the unit list this edit just changed.
        # Re-armed below, once the cache is invalidated (Batch B step B1).
        self._cancel_warms()
        # Scoped edits patch the index without _rebuild_unit_index(), which
        # otherwise resets this; a moved unit can leave or join a stack.
        self._stack_cycle = None
        if self._cache is not None:
            # TASK-583: a scoped edit that flips the file's wall radian verdict reshapes walls anywhere.
            if changed is not None and not self._cache.wall_verdict_flipped(changed):
                self._patch_unit_edit_cache(
                    changed, batch, terrain_bbox, terrain_current, terrain_tiles, pre_stale_levels=pre_stale_levels
                )
            else:
                if not sources_refreshed:
                    with perf_trace.phase("unit_sources"):
                        self._cache.invalidate_units(changed)
                canvas_w, canvas_h = self._cache.canvas_dims(0)
                self._cache.invalidate_region((0, 0, canvas_w, canvas_h), terrain_changed=False)
                self.map_view.invalidate_region((0, 0, canvas_w, canvas_h))
        # Above the mode gate, not at the literal tail: undo is global, so a
        # unit edit reverted from Terrain mode must re-arm the warm too.
        with perf_trace.phase("level_warm_start"):
            self._start_level_warm()
        # Before the defer_index return: each Convert flush patches its own batch.
        with perf_trace.phase("unit_refs"):
            self._patch_unit_ref_index(changed)
        if defer_index:
            return
        with perf_trace.phase("unit_refs"):
            self._on_unit_references_moved()
        # Unconditional: undo is global. A repopulate, since a reassign can give an inactive slot units.
        with perf_trace.phase("unit_stats"):
            self._repopulate_stats_players()
        # GH #144: marks only; the dialog coalesces its re-run behind singleShot(0).
        self._find_dialog_mark_stale()
        if self.mode != "units":
            # Nothing patched the index outside Units, and Units entry reuses it, so it stays current
            # or None: rebuilt for the footprint overlay, an armed picker or a kept selection, else dropped.
            if self._needs_unit_index():
                with perf_trace.phase("unit_index"):
                    self._rebuild_unit_index()
            else:
                self._stack_cycle = None
                self.map_view.set_unit_index(None)
            if self._selection:
                with perf_trace.phase("selection_view"):
                    self._reconcile_selection()
                self._drop_unit_index_if_unneeded()
            return
        if changed is None or rebuild_index:
            with perf_trace.phase("unit_index"):
                self._rebuild_unit_index()
        else:
            # The index was patched in place, so nothing derived from it --
            # the stacks, the footprint outlines -- was recomputed.
            with perf_trace.phase("unit_stacks"):
                self.map_view.refresh_after_index_patch()
        with perf_trace.phase("selection_view"):
            self._refresh_selection_view()
        # TASK-327: a just-placed unit turns the still ghost refused at once, as in the game.
        with perf_trace.phase("place_ghost"):
            self.map_view.refresh_place_ghost()

    def _patch_unit_edit_cache(
        self, changed: list[UnitSplice], batch: bool = False, terrain_bbox: tuple[int, int, int, int] | None = None,
        terrain_current: bool = False, terrain_tiles=None, pre_stale_levels=frozenset(),
    ) -> None:
        """The scoped branch of _after_unit_mutation() (Batch D's D5):
        splices self._cache's unit-derived sources (D4's own invalidate_
        units(changed)) and repaints only the bbox `changed`'s old/new
        footprints could have touched, rather than invalidating the whole
        canvas the way the wholesale branch does.

        invalidate_units(changed) may still fall back to a full source
        rebuild internally for one or more entries (an added or removed
        wall/connector const, or a shared-tile component past its cap -- see render_cache.
        _splice_plan()) -- that only changes how the SOURCE data got
        refreshed, not how much of the canvas needs recompositing, since
        nothing but this edit's own unit(s) actually changed. A batch that
        flips the file's wall radian verdict never reaches here:
        _after_unit_mutation() sends it wholesale (TASK-583). Stepped/Sloped
        pass the visible level as cap_levels, so a batch past both splice caps
        still splices that level alone (TASK-031.61 c2x) and the split finds it
        current instead of rebuilding it.

        Stepped/Sloped with sprites on size the VISIBLE level's repaint
        (map_view.viewport_chunk_target()'s mip) from the edited units' real
        extents: the tile term (terrain, marks' footprints, farm tiles) plus
        cache.sprite_extent_before(), read from that level's layer, and
        cache.sprite_extent_after(), read from the spliced layer when that
        level is current, else resolved per unit. Every other resident
        level evicts today's reach-padded bbox instead, so no non-visible level
        is resolved or recomposited here. Qt repaints only the visible level.

        Ordering constraint: on a CompositeDiffRecord undo the terrain
        child's _apply_dirty() runs before this method, and the pre side must
        be read after it and before invalidate_units(), which replaces the
        layer it reads. So it must not be hoisted out of this method.

        Today's reach-padded bbox is kept whenever the tight one can't be
        trusted (render_cache_splice.REACH_FALLBACK): a moved or placed wall,
        connector or rotation-variant const, whose neighbours' art changes too
        and which today's 700 px pad covers (a Convert of one moves nothing
        and is sized tight), or a visible level holding no resident chunks.
        With sprites off it is already exact. It goes through the same split
        (_repaint_unit_edit_split()): the visible level patches or evicts it,
        every other resident level evicts it, so a level a wholesale source
        fallback left stale rebuilds through LevelWarmer, not in the handler.
        Only with no viewport target at all does it patch every resident level,
        always eagerly: with no visible area there is nothing to price against.

        Flat has no elevation term and no dirty_screen_bbox_* counterpart --
        mirrors _apply_dirty_render's own Flat branch, patching
        render_paint.flat_tile_rects() of the touched tiles rather than a union
        bbox, widened to their 8 neighbours when the edit moves, adds or
        removes a wall or connector (a wall's drawn shape reads its 8-bit
        neighbour mask). No blend ring (a unit edit moves no blend key) and no
        evict rule.

        In the split, a `batch` whose bbox is large against the viewport
        (_patch_area_exceeds_viewport()) evicts that bbox's chunks instead of
        patching them: same correctness argument, since nothing outside the
        bbox changed, and chunks off the bbox stay resident for the next pan.
        The split prices the visible level alone (_TIGHT_PATCH_AREA_RATIO).
        Single-unit tools always patch.

        terrain_bbox (Stepped/Sloped, TASK-031.61 A4): a history move's
        terrain bbox, sources refreshed but not composited. The split repaints
        and evicts its union with the unit bboxes; the no-target patch covers
        it too; the whole-canvas fallback already does; the early returns
        composite it alone, as the tile patch would have.

        terrain_current (Stepped/Sloped, TASK-031.61 C0): passed to the split,
        whose evictions then skip the terrain mirror re-read. The whole-canvas
        fallback keeps its re-read, and the eager patches always re-read.

        Stepped `batch` edits (TASK-031.61 C1) read both extents per 16x16-tile
        block of `changed` (_unit_edit_blocks()), the batch extent being their
        union. The tight split may then patch one rect per chunk from those
        blocks instead of evicting (_unit_edit_chunk_rects()). terrain_tiles
        are the tiles behind terrain_bbox, sized block by block with the
        units; a terrain_bbox without them keeps the union bbox alone.

        pre_stale_levels (the Eraser's mark, TASK-318): a visible level in it
        skips the pre read, so the batch takes the reach branch with no
        blocks. Its units left the model while that level was stale, so a
        rebuild since may have dropped them from the layer the read uses.

        Stepped/Sloped first add _flag_flip_splices(): a garrison edit names
        only its occupants, while the host's flag (GH #203) appears or
        clears. Both extents and the source splice see the hosts."""
        # Convert's stroke end can drain an empty list; Flat would treat [] as wholesale.
        if not changed:
            self._composite_deferred_terrain(terrain_bbox)
            return
        pre = post = REACH_FALLBACK
        visible_mip = None
        blocks = None
        if self._render_style != "flat":
            # A new list, never the caller's: it also feeds the pick index, which must not see host re-anchors.
            changed = changed + self._flag_flip_splices(changed)
            target = self.map_view.viewport_chunk_target()
            visible_mip = None if target is None else target[0]
            if self._cache.sprites_enabled and visible_mip not in pre_stale_levels:
                with perf_trace.phase("sprite_extent"):
                    if (
                        batch and self._render_style == "stepped" and visible_mip is not None
                        and (terrain_bbox is None or terrain_tiles is not None)
                    ):
                        blocks = self._unit_edit_blocks(changed)
                        pre = self._block_sprite_extents(blocks, "before", visible_mip)
                    else:
                        pre = self._cache.sprite_extent_before(changed, visible_mip)
        with perf_trace.phase("unit_sources"):
            self._cache.invalidate_units(changed, cap_levels=None if visible_mip is None else (visible_mip,))
        old_tiles = {t for s in changed for t in s.old_tiles}
        new_tiles = {t for s in changed for t in s.new_tiles}
        touched = old_tiles | new_tiles
        if not touched:
            self._composite_deferred_terrain(terrain_bbox)
            return
        if self._render_style == "flat":
            mm = self.scenario.map_manager
            tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
            connectors = unit_sprites.wall_connector_consts()
            if any(not s.is_reassign and s.unit.unit_const in connectors for s in changed):
                # A wall's drawn shape reads its 8 neighbours, so theirs can change too.
                touched = render_cache_splice._dilate(touched, mm.map_width, mm.map_height)
            rects = flat_tile_rects(touched, tile_px)
            with perf_trace.phase("unit_patch"):
                self._cache.patch_rects(rects)
            for rect in rects:
                self.map_view.invalidate_region(rect)
            return
        if pre is not REACH_FALLBACK:
            with perf_trace.phase("sprite_extent"):
                if blocks is not None:
                    post = self._block_sprite_extents(blocks, "after", visible_mip)
                else:
                    post = self._cache.sprite_extent_after(changed, visible_mip)
        if post is not REACH_FALLBACK:
            with perf_trace.phase("unit_bbox"):
                bbox, elevation_changed = self._unit_edit_bbox(changed, pre, post)
            if bbox is not None:
                self._repaint_unit_edit_split(
                    changed, bbox, elevation_changed, visible_mip, batch, terrain_bbox=terrain_bbox,
                    terrain_current=terrain_current, blocks=blocks, terrain_tiles=terrain_tiles,
                )
                return
        with perf_trace.phase("unit_bbox"):
            bbox, elevation_changed = self._unit_edit_bbox(changed)
        if bbox is None:
            # Defensive, not expected (see dirty_screen_bbox_iso's own
            # docstring): a unit edit never moves a tile's elevation, so this
            # only fires if the terrain was already out of range beforehand.
            canvas_w, canvas_h = self._cache.canvas_dims(0)
            self._cache.invalidate_region((0, 0, canvas_w, canvas_h))
            self.map_view.invalidate_region((0, 0, canvas_w, canvas_h))
            return
        if visible_mip is not None:
            self._repaint_unit_edit_split(
                changed, bbox, elevation_changed, visible_mip, batch, reach=bbox, terrain_bbox=terrain_bbox,
                terrain_current=terrain_current,
            )
            return
        # No viewport target, so no visible area to price an eviction against: patch.
        bbox = render_cache_splice._union_bbox(bbox, terrain_bbox)
        with perf_trace.phase("unit_patch"):
            self._cache.patch(bbox, elevation_changed=elevation_changed)
        self.map_view.invalidate_region(bbox)

    def _flag_flip_splices(self, changed: list[UnitSplice]) -> list[UnitSplice]:
        """A re-anchor splice for each live host whose garrison flag (GH #203)
        appeared or cleared since the cache's last unit-source refresh and
        that `changed` does not already carry. A link, unload, Garrison Add
        or Paste, a delete or an Eraser stroke splices only the occupant, so
        without these the host keeps its old flag. Every caller that passes
        invalidate_units() a list goes through here, before that call, which
        stamps the flips as reflected. [] with sprites or units off, where no
        flag draws, and when nothing flipped: the unit walk runs only then."""
        cache = self._cache
        if cache is None or not (cache.with_units and cache.sprites_enabled):
            return []
        flips = cache.flag_flips()
        if not flips:
            return []
        known = {id(s.unit) for s in changed}
        splices: list[UnitSplice] = []
        for player_id, units in enumerate(self.scenario.unit_manager.units):
            for index, unit in enumerate(units):
                if unit.reference_id in flips and id(unit) not in known:
                    own, tiles = self._unit_footprint(unit)
                    if tiles:
                        splices.append(UnitSplice(player_id, index, unit, own, own, tiles, tiles))
        return splices

    def _composite_deferred_terrain(self, terrain_bbox: tuple[int, int, int, int] | None) -> None:
        """The tile patch _apply_dirty() deferred (A4), for a unit step that
        repaints nothing: patch_composite() as patch() would have run it."""
        if terrain_bbox is None:
            return
        target = self.map_view.viewport_chunk_target()
        with perf_trace.phase("patch"):
            self._cache.patch_composite(
                terrain_bbox, levels=self._visible_levels(target),
                rebuild_levels=None if target is None else (target[0],),
            )
        self.map_view.invalidate_region(terrain_bbox)

    def _visible_levels(self, target) -> tuple[int, ...] | None:
        """The levels a terrain patch composites in place (TASK-031.73), the
        rest evicting under its bbox: None with no viewport target (every
        level patches), else the level in view, plus the level a placeholder
        draws while one is pending, so its chunks under the brush stay."""
        if target is None:
            return None
        item = self.map_view._canvas_item
        source = None if item is None or item.pending_mip() is None else item.pending_source()
        return (target[0],) if source is None or source == target[0] else (target[0], source)

    def _repaint_unit_edit_split(
        self, changed: list[UnitSplice], bbox: tuple[int, int, int, int], elevation_changed: set,
        visible_mip: int, batch: bool, reach: tuple[int, int, int, int] | None = None,
        terrain_bbox: tuple[int, int, int, int] | None = None, terrain_current: bool = False,
        blocks: list[_UnitEditBlock] | None = None, terrain_tiles=None,
    ) -> None:
        """_patch_unit_edit_cache()'s visible-level repaint, tight or reach: the
        visible level patches or evicts `bbox` by the usual area rule, priced
        on that level alone since no other level is patched; every other
        resident level evicts today's reach-padded bbox, which costs nothing
        until it is shown. `reach` is that bbox when the caller already holds
        it (the reach branch, where it is `bbox`), else derived here. A
        deferred terrain_bbox (A4) is unioned into both.

        terrain_current (TASK-031.61 C0): both evictions skip the terrain
        mirror re-read (invalidate_region()'s terrain_changed=False), the
        caller having vouched that both mirrors match the tiles. The eager
        patch re-reads its bbox either way.

        blocks (Stepped, TASK-031.61 C1): given, a batch past the area rule
        does not evict the visible level. It patches one rect per chunk there
        (_unit_edit_chunk_rects(), which sizes terrain_tiles too), each
        inside `bbox`, unless those rects are past their own ratio or cannot
        be sized; then it evicts as before, having paid for the sizing
        (1-19 ms on the 2026-10-12 sweep). A visible level the batch left
        stale (a gen bump past VISIBLE_SPLICE_MAX_UNITS, or with sprites off)
        then rebuilds here, inside the first rect, not at the next paint; one
        under that bound was spliced current (TASK-031.61 c2x). The other
        levels' eviction and the region Qt repaints do not change."""
        others = [mip for mip in self._cache.resident_levels() if mip != visible_mip]
        if others and reach is None:
            with perf_trace.phase("unit_bbox"):
                reach, _ = self._unit_edit_bbox(changed)
            if reach is None:
                reach = (0, 0, *self._cache.canvas_dims(0))
        if terrain_bbox is not None:
            bbox = render_cache_splice._union_bbox(bbox, terrain_bbox)
            reach = render_cache_splice._union_bbox(reach, terrain_bbox) if reach is not None else None
        visible = (visible_mip,)
        evict = batch and self._patch_area_exceeds_viewport(bbox, levels=visible)
        rects = None
        if evict and blocks is not None and _RECT_PATCH_AREA_RATIO > 0:
            # Only here: a batch that patches its bbox never needs the block rects sized.
            with perf_trace.phase("unit_bbox"):
                rects = self._unit_edit_chunk_rects(blocks, terrain_tiles, visible_mip)
            if rects is not None and self._patch_area_exceeds_viewport(bbox, levels=visible, rects=rects):
                rects = None
        # Sizing stays outside the phase, as in the reach branch, so unit_patch times cache work only.
        with perf_trace.phase("unit_patch"):
            if others:
                self._cache.invalidate_region(reach, levels=others, terrain_changed=not terrain_current)
            if rects is not None:
                self._cache.patch_composite_rects(
                    rects, levels=visible, rebuild_levels=visible, terrain_current=terrain_current
                )
            elif evict:
                self._cache.invalidate_region(bbox, levels=visible, terrain_changed=not terrain_current)
            else:
                self._cache.patch(bbox, elevation_changed=elevation_changed, levels=visible)
        self.map_view.invalidate_region(bbox)

    def _unit_edit_blocks(self, changed: list[UnitSplice]) -> list[_UnitEditBlock]:
        """`changed` split by the _UNIT_RECT_BLOCK_TILES-square block of tiles
        its own tile is in (the new one, else the old), for the per-chunk
        rects (TASK-031.61 C1). Every splice is in exactly one block, so the
        blocks' extents union to the batch's."""
        size = _UNIT_RECT_BLOCK_TILES
        by_key: dict = {}
        for s in changed:
            own = s.new_own_tile if s.new_own_tile is not None else s.old_own_tile
            key = None if own is None else (own[0] // size, own[1] // size)
            block = by_key.get(key)
            if block is None:
                block = by_key[key] = _UnitEditBlock(key, [])
            block.splices.append(s)
        return list(by_key.values())

    def _block_sprite_extents(self, blocks: list[_UnitEditBlock], side: str, mip: int):
        """cache.sprite_extent_before()/after() (`side`) read per block and
        kept on it. Returns their union, which is the batch call's own answer,
        or REACH_FALLBACK as soon as one block gives it (the batch call would
        too: its only per-splice test is an any())."""
        extent = getattr(self._cache, f"sprite_extent_{side}")
        union = None
        for block in blocks:
            box = extent(block.splices, mip)
            if box is REACH_FALLBACK:
                return REACH_FALLBACK
            setattr(block, side, box)
            union = render_cache_splice._union_bbox(union, box)
        return union

    def _unit_edit_chunk_rects(
        self, blocks: list[_UnitEditBlock], terrain_tiles, visible_mip: int
    ) -> list[tuple[int, int, int, int]] | None:
        """One reference-canvas rect per visible-level chunk a Stepped batch
        unit edit changes (TASK-031.61 C1), for cache.patch_composite_rects().

        Each block's rect is _unit_tiles_bbox()'s answer for its splices, its
        own extents and the terrain_tiles (a history move's deferred tile
        indices, A4) in the same block of tiles: dirty_screen_bbox_iso() over
        that block's tiles. The tiles go in as they are, since that call
        already dilates its dirty set by one tile, which terrain blending
        needs and _apply_dirty() relies on. Every block is sized in one pass
        (dirty_screen_bboxes_iso_grouped()), the per-block calls being what a
        large undo spent 8-89 ms on; they still run when that pass declines
        (a tile whose elevation moved, which no caller hands over today).
        cache.chunk_rects() then keeps one rect per chunk, the bbox of the
        block rects clipped to it: block rects overlap, and patched one by
        one they cost more than evicting.

        None, for the caller to evict as before, when a bbox call declines
        (see _patch_unit_edit_cache()'s own defensive branch)."""
        size = _UNIT_RECT_BLOCK_TILES
        width = self.scenario.map_manager.map_width
        terrain_by_block: dict = {}
        if terrain_tiles is not None and len(terrain_tiles):
            idx = np.fromiter(terrain_tiles, dtype=np.int64, count=len(terrain_tiles))
            columns = width // size + 1
            keys = (idx // width // size) * columns + (idx % width) // size
            order = np.argsort(keys, kind="stable")
            idx, keys = idx[order], keys[order]
            starts = np.flatnonzero(np.diff(keys, prepend=keys[0] - 1))
            for key, tiles in zip(keys[starts].tolist(), np.split(idx, starts[1:]), strict=True):
                terrain_by_block[(key % columns, key // columns)] = tiles.tolist()
        rects = []
        # (splices, before, after, terrain tile indices) per rect that has a tile term.
        jobs: list = []
        for block in blocks:
            terrain = terrain_by_block.pop(block.key, ())
            if not terrain and not any(s.old_tiles or s.new_tiles for s in block.splices):
                # No tile on the map, so no tile term: its extents alone (None, in practice).
                rect = render_cache_splice._union_bbox(block.before, block.after)
                if rect is not None:
                    rects.append(rect)
                continue
            jobs.append((block.splices, block.before, block.after, terrain))
        jobs.extend(([], None, None, terrain) for terrain in terrain_by_block.values())
        groups = []
        for splices, _before, _after, terrain in jobs:
            touched = {t for s in splices for t in s.old_tiles}
            touched.update(t for s in splices for t in s.new_tiles)
            group = [y * width + x for x, y in touched]
            group.extend(terrain)
            groups.append(group)
        tile_boxes = dirty_screen_bboxes_iso_grouped(
            self.scenario, groups, self._iso_elevations, self._iso_proj,
            flatten_elevations=(self._terrain_style == "flat"),
        )
        for i, (splices, before, after, terrain) in enumerate(jobs):
            if tile_boxes is None:
                rect, _ = self._unit_tiles_bbox(splices, before, after, terrain)
            else:
                rect = tile_boxes[i] and self._with_sprite_extents(tile_boxes[i], before, after)
            if rect is None:
                return None
            rects.append(rect)
        return self._cache.chunk_rects(rects, visible_mip)

    def _patch_area_exceeds_viewport(self, bbox: tuple[int, int, int, int], levels, rects=None) -> bool:
        """Whether patch(bbox, levels=levels) would recomposite more than
        _TIGHT_PATCH_AREA_RATIO of the visible chunks' area. Past that,
        patching eagerly costs more than evicting the bbox's chunks and
        recompositing the visible ones whole at the next paint. Only the
        visible-level split (_repaint_unit_edit_split(), levels=(visible,))
        asks; with no viewport target this answers False.

        rects (Stepped, TASK-031.61 C1): the per-chunk rects that would be
        patched in place of an over-ratio bbox. Their summed patch_area() is
        priced instead, against _RECT_PATCH_AREA_RATIO: whether
        cache.patch_composite_rects(rects) would recomposite more than that.
        Past it the split evicts the bbox as before. That eviction is not
        free of the rects: they were sized to be priced.

        Measured 2026-09-27 on old-allies and Joan 1, sprites on
        (tools/bench_stroke_end.py --area-ratio): the split patches cheaper
        than it evicts at every area it reaches, 0.05-0.3 (by ~3 ms,
        crossover extrapolated past the splice cap, ~0.4-0.55), so Stepped
        takes 0.3, the top of the measured range. Sloped, whose chunk
        recomposite is the warp, recomposites cheaper by patching at every
        measured area, so its 1.0 is an extrapolated bound (its evict edge at
        0.22-0.3 is within noise).

        Re-measured 2026-10-10 for Sloped's mip ladder (TASK-031.60) at fit
        zoom, old-allies, Trees on, ratios 0/0.3/0.5/1.0/inf x3: level -2
        is visible and every priced area is 0.00-0.04, so all nonzero ratios
        take one path. Patching beat evicting on 29 of 32 rows (median
        4.5 ms). Of the other 3, one is noise (+2 ms) and two are over-cap
        gen bumps, where the patch pays the level rebuild in the handler
        instead of the next paint (+34/+51 ms). The crossover lies past
        what fit zoom reaches, so Sloped keeps 1.0.

        Measured 2026-10-11 for Stepped's per-chunk rects (TASK-031.61 C1),
        same view and map, terrain blending on (--strokes rows, GH #180's
        drag, --terrains 109,13, --rect-ratio 0 against inf x3 interleaved,
        2-12 rows at brush 9 and 8 at brush 5): the rects priced 0.20-0.43
        under union bboxes of 0.42-0.57, handler plus next paint compared.
        At 0.20-0.24 patching them beat evicting on all 5 rows, by 28-61 ms.
        Past that it depends on the batch. A stroke end that only removes
        trees still wins at 0.27-0.34 (by ~21 ms) and breaks even at
        0.36-0.39. A 10k-tree stroke end, its undo and an undo that carries
        terrain tiles are within noise at 0.30 (-6/-10/+16 ms) and lose
        11-94 ms from 0.34, their per-block bbox calls (26-89 ms) tipping
        it. So the rects take 0.25, the top of the range every row won in.
        An over-cap batch's level rebuild then runs in the handler, not the
        next paint; both are in the totals compared.

        2026-10-12: the blocks are sized in one array pass
        (render_bbox.dirty_screen_bboxes_iso_grouped()), 1-19 ms on those rows
        where the per-block calls took 8-89, with the same rects and so the
        same priced areas. That is what a batch past 0.25 now pays before it
        evicts. The ratio was not re-derived: the rect side of every row
        above is cheaper by the difference, so 0.25 is conservative.

        A retired second ratio (Stepped 0.05) priced a patch of every
        resident level, which also paid each stale level's rebuild in the
        patch (~100 ms vs ~47 ms with one level, old-allies). Since
        2026-09-30 no caller patches more than the visible level here."""
        target = self.map_view.viewport_chunk_target()
        if target is None:
            return False
        _mip, cx0, cy0, cx1, cy1 = target
        visible = (cx1 - cx0 + 1) * (cy1 - cy0 + 1) * self._cache.chunk_px**2
        if rects is not None:
            area = sum(self._cache.patch_area(rect, levels=levels) for rect in rects)
            return area > _RECT_PATCH_AREA_RATIO * visible
        return self._cache.patch_area(bbox, levels=levels) > _TIGHT_PATCH_AREA_RATIO[self._render_style] * visible

    def _unit_edit_bbox(
        self, changed: list[UnitSplice], pre=REACH_FALLBACK, post=REACH_FALLBACK
    ) -> tuple[tuple[int, int, int, int] | None, set]:
        """Stepped/Sloped: the reference-canvas bbox `changed`'s old and new
        footprints could have touched, and the elevation-changed set the
        dirty-bbox helper fills (always empty for a unit edit).

        With pre/post left at REACH_FALLBACK this is today's reach-padded
        bbox, sized from scenario state alone. Given the cache's
        sprite_extent_before()/after() (reference bboxes or None), the sprite
        term is those real extents instead: the tile term without sprite
        reach, unioned with both and clamped to the reference canvas. Either
        way, dirty_screen_bbox_*() writes the touched tiles' elevations back
        into the shared array, a no-op for a unit edit."""
        return self._unit_tiles_bbox(changed, pre, post)

    def _unit_tiles_bbox(
        self, changed: list[UnitSplice], pre, post, terrain_tiles=()
    ) -> tuple[tuple[int, int, int, int] | None, set]:
        """_unit_edit_bbox()'s body, which also sizes one block of a batch
        (_unit_edit_chunk_rects()): terrain_tiles are tile indices that go
        into the same dirty set as `changed`'s footprints, so one
        dirty_screen_bbox_*() call dilates and sweeps both."""
        tight = pre is not REACH_FALLBACK and post is not REACH_FALLBACK
        with_sprites = self._cache.sprites_enabled and not tight
        old_tiles = {t for s in changed for t in s.old_tiles}
        touched = old_tiles | {t for s in changed for t in s.new_tiles}
        width = self.scenario.map_manager.map_width
        dirty_indices = [y * width + x for x, y in touched]
        dirty_indices.extend(terrain_tiles)
        elevation_changed: set = set()
        if self._render_style == "stepped":
            bbox = dirty_screen_bbox_iso(
                self.scenario, dirty_indices, self._iso_elevations, self._iso_proj, with_units=True,
                with_sprites=with_sprites, elevation_changed=elevation_changed,
                flatten_elevations=(self._terrain_style == "flat"), extra_anchor_tiles=old_tiles,
                units_changed=True,
            )
        else:
            bbox = dirty_screen_bbox_sloped(
                self.scenario, dirty_indices, self._iso_elevations, self._iso_proj, with_units=True,
                with_sprites=with_sprites, elevation_changed=elevation_changed,
                extra_anchor_tiles=old_tiles, units_changed=True,
            )
        if tight and bbox is not None:
            bbox = self._with_sprite_extents(bbox, pre, post)
        return bbox, elevation_changed

    def _with_sprite_extents(self, bbox: tuple[int, int, int, int], pre, post) -> tuple[int, int, int, int]:
        """A tight tile-term bbox unioned with the cache's sprite extents
        (reference bboxes or None) and clamped to the reference canvas."""
        x0, y0, x1, y1 = bbox
        for extent in (pre, post):
            if extent is not None:
                x0, y0 = min(x0, extent[0]), min(y0, extent[1])
                x1, y1 = max(x1, extent[2]), max(y1, extent[3])
        canvas_w, canvas_h = self._cache.canvas_dims(0)
        return (max(0, x0), max(0, y0), min(canvas_w, x1), min(canvas_h, y1))
