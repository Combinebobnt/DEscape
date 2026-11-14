"""ViewerWindow's edit strokes: terrain strokes, shape commit, Convert, the Unit Brush and the Eraser."""

from __future__ import annotations

import random
import weakref
from contextlib import contextmanager, suppress

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QMessageBox,
)

from descape import (
    cliff_chain,
    debug_log,
    eraser,
    perf_trace,
    place_rotation,
    render_cache_splice,
    scatter,
    terrain_target,
    terrain_units,
    unit_brush,
    unit_kind,
    unit_pick,
    unit_rotation,
    unit_shape,
    unit_sprites,
    unit_variant,
    wall_run,
)
from descape.beach_edges import apply_beach_ring
from descape.edit_history import (
    CompositeDiffRecord,
    DiffRecord,
    TileDiffRecord,
)
from descape.elevation_tools import set_tiles_elevation
from descape.render_cache_splice import UnitSplice
from descape.terrain_palette import name_for_terrain_id
from descape.unit_filter import GAIA_PLAYER_ID, MAX_PLAYER_ID
from descape.viewer_common import (
    _STROKE_LABELS,
    _WALL_TOOLS,
    BRUSH_TOOLS,
    ELEVATION_LEVEL_MAX,
    TERRAIN_UNIT_CONFIRM_THRESHOLD,
    _gaia_only_refused_note,
    _unit_name,
)

# Stroke tools that write only tile.elevation, which Flat never draws: their live repaint is skipped there.
_FLAT_SKIP_REPAINT_TOOLS = ("elevation", "set_level")


def _shape_shore(tiles) -> list[tuple[int, int]]:
    """The shape's tiles with an 8-neighbour outside it. Ringing these equals ringing
    the whole shape: the extra interior tiles it reaches are the shape's own terrain,
    which neither ring mode converts (water mode skips water, land mode converts water only)."""
    core = set(tiles)
    return [
        (x, y)
        for x, y in core
        if any((x + dx, y + dy) not in core for dx in (-1, 0, 1) for dy in (-1, 0, 1))
    ]


class _ViewerStrokesMixin:
    """ViewerWindow's edit strokes (viewer.py's docstring has the split's rules)."""

    # -- Edit-tool strokes (terrain paint / elevation raise-lower / set level) --
    # One stroke (mouse-down through release, or a defensive close on leave --
    # see MapView) is exactly one descape.edit_history.EditHistory record and
    # one undo step, but repaints live as it's dragged. See edit_history.py's
    # module docstring for why nothing here writes to a tile directly.

    def on_edit_stroke_start(self) -> None:
        if self.scenario is None:
            return
        if self._current_tool == "convert":
            self._begin_convert_stroke()
            return
        if self._current_tool == "unit_brush":
            self._begin_unit_brush_stroke()
            return
        if self._current_tool == "eraser":
            self._begin_erase_stroke()
            return
        # Above begin_stroke(), like convert: a cliff stroke writes units,
        # not terrain, so falling through would open an EditHistory terrain
        # stroke that commits empty at release -- the plan's own third
        # Stage 1 invariant, now live.
        if self._current_tool == "cliff":
            # GH #247: right erases, as in the in-game editor; left places.
            if self.map_view.stroke_button() == Qt.RightButton:
                self._begin_cliff_erase()
            else:
                self._begin_cliff_stroke()
            return
        with perf_trace.phase("stroke_snapshot"):
            self.edit_history.begin_stroke(self.scenario.map_manager.terrain)
        with perf_trace.phase("stroke_start"):
            self._init_stroke_state()

    def _init_stroke_state(self) -> None:
        """on_edit_stroke_start's per-stroke bookkeeping, after the snapshot."""
        # index -> the tile state as of the last time we handed that index to
        # _apply_dirty. A dict, not a set of indices: elevation propagation
        # can change one tile SEVERAL times over a single drag, and a
        # membership-only set silently drops every change after the first --
        # see _apply_stroke_dirty and tests/test_stroke_elevation_sync.py.
        self._stroke_seen_state: dict[int, tuple[int, int, int]] = {}
        # Painted-tile dedupe for this stroke, keyed on the actual tile a
        # brush touched -- see _mutate_stroke_tile's own comment for why
        # this must be separate from MapView._stroke_touched (which is
        # keyed on the CURSOR tile, and is only a cheap early-out, not a
        # correctness guarantee once a brush is bigger than one tile).
        self._stroke_painted: set[tuple[int, int]] = set()
        # Every flat index the stroke's mutations reported writing, so the
        # release diffs only these (build_stroke_record's `touched`).
        self._stroke_written: set[int] = set()
        # Auto beach, snapshotted HERE rather than read per touch. The
        # checkbox, beach combo and width spin are all live widgets, so
        # reading them inside on_edit_stroke_tile would half-apply a
        # mid-stroke change across one undo record. A separate Water tool
        # would not have had this problem; the checkbox does.
        # The Target is snapshotted the same way and for the same reason.
        self._stroke_target = self._terrain_target() if self._current_tool == "draw" else terrain_target.TERRAIN
        self._stroke_auto_beach = (
            (
                self._stroke_target == terrain_target.TERRAIN
                and self.auto_beach_check.isEnabled()
                and self.auto_beach_check.isChecked(),
                self.beach_combo.currentData(),
                self.beach_width_spin.value(),
            )
            if self._current_tool == "draw"
            else (False, None, 0)
        )

    def on_edit_stroke_tile(self, x: int, y: int, modifiers) -> None:
        self.on_edit_stroke_tiles([(x, y)], modifiers)

    def on_edit_stroke_tiles(self, tiles, modifiers) -> None:
        """MapView's per-event stroke entry: `tiles` is the cursor path in
        order. Mutates tile by tile exactly as that many single-tile calls
        would (Elevate's propagation order depends on it), then works out
        the repaint set from the tiles written and runs _apply_dirty once
        for the whole event."""
        if self.scenario is None:
            return
        # On the open stroke, not the tool, matching on_edit_stroke_end.
        if self._erase is not None:
            with self._close_stroke_on_error(_STROKE_LABELS["eraser"]):
                self._erase_stroke_tiles(tiles)
            return
        if self._unit_brush is not None:
            with self._close_stroke_on_error(_STROKE_LABELS["unit_brush"]):
                self._unit_brush_stroke_tiles(tiles, modifiers)
            return
        if self._current_tool == "unit_brush":
            # A press that opened nothing (a right press, no object, a wall): the drag paints nothing.
            return
        if self._current_tool == "convert":
            for x, y in tiles:
                self._convert_stroke_tile(x, y)
            return
        if self._current_tool == "cliff":
            for x, y in tiles:
                self._cliff_stroke_tile(x, y)
            # One preview per event, not per tile.
            if self._cliff_stroke is not None:
                self._show_cliff_run(self._cliff_stroke, self._cliff_stroke_existing, self._cliff_ghost_parts)
            return
        # Closed early by a raise below: the rest of this drag writes nothing.
        if not self.edit_history.in_stroke:
            return
        label = terrain_target.stroke_label(
            self._current_tool, self._stroke_target, _STROKE_LABELS.get(self._current_tool, "Edit")
        )
        with self._close_stroke_on_error(label):
            touched: set[int] = set()
            for x, y in tiles:
                touched |= self._mutate_stroke_tile(x, y, modifiers)
            if touched:
                self._stroke_written |= touched
                self._apply_stroke_dirty(touched)

    @contextmanager
    def _close_stroke_on_error(self, label: str):
        """Wraps work done inside an open EditHistory stroke. On a raise (e.g.
        MapManager.get_tile() on a non-square map) the stroke is committed
        before the exception propagates, so _stroke_before can't stay set and
        wedge every later begin_stroke(). Committed, not aborted: tiles already
        written can't be un-painted (abort_stroke()'s docstring), and this
        keeps them on the undo stack and repainted instead of silently live."""
        try:
            yield
        except BaseException:
            if self._erase is not None:
                # The Eraser's own end closes its unit edit and tile stroke as one record.
                with suppress(Exception):
                    self._end_erase_stroke()
            if self._unit_brush is not None:
                # Committed, as the Eraser's: the adds so far are live.
                with suppress(Exception):
                    self._end_unit_brush_stroke()
            if self.edit_history.in_stroke:
                # Full scan: the raise may have cut a mutation off before it reported its writes.
                touched = self.edit_history.commit_stroke(label, self.scenario.map_manager.terrain)
                self._stroke_seen_state = {}
                self._stroke_painted = set()
                self._stroke_written = set()
                self._stroke_auto_beach = (False, None, 0)
                self._stroke_target = terrain_target.TERRAIN
                self._update_edit_actions()
                self._update_title()
                # The repaint can hit the same fault; the original exception is the one to report.
                with suppress(Exception):
                    self._apply_dirty(touched)
            raise

    def _mutate_stroke_tile(self, x: int, y: int, modifiers) -> set[int]:
        """One cursor tile's footprint edit. Returns every flat index it
        wrote (propagation and beach ring included), empty if none."""
        mm = self.scenario.map_manager
        # Every drag tool reaching this method today (terrain/elevation/
        # set_level) has supports_brush=True, so this is BRUSH_TOOLS in
        # practice -- checked explicitly rather than assumed, so a future
        # is_edit_tool tool with no brush still degrades to its old
        # single-tile behavior instead of silently expanding.
        if self._current_tool in BRUSH_TOOLS:
            footprint = self._brush_footprint(x, y)
        elif 0 <= x < mm.map_width and 0 <= y < mm.map_height:
            footprint = [(x, y)]
        else:
            footprint = []
        # Dedupe against tiles this stroke already painted -- NOT against
        # MapView._stroke_touched's cursor-tile set. With a brush bigger
        # than one tile, a single painted tile falls under many distinct
        # cursor tiles during a drag; without this, Elevate's accumulating
        # +/-1 (see below) would raise/lower the same tile once per cursor
        # tile that overlapped it, not once per stroke.
        footprint = [t for t in footprint if t not in self._stroke_painted]
        if not footprint:
            return set()
        with perf_trace.phase("stroke_mutate"):
            touched = self._mutate_footprint(mm, footprint, modifiers)
        if touched is None:
            return set()
        self._stroke_painted.update(footprint)
        return touched

    def _mutate_footprint(self, mm, footprint, modifiers) -> set[int] | None:
        """_mutate_stroke_tile's write for the current tool: every flat index
        written, or None for a tool with no stroke mutation."""
        if self._current_tool == "draw":
            terrain_id = self.terrain_panel.terrain_id()
            width = mm.map_width
            # The stroke's own Target, never the live combo. On Terrain this also clears
            # the layer, which the game and terrain blending draw over terrain_id.
            target = self._stroke_target
            for tx, ty in footprint:
                terrain_target.write_tile(mm.get_tile(tx, ty), target, terrain_id)
            touched = {ty * width + tx for tx, ty in footprint}
            # The shoreline, laid AFTER the core so a tile this touch just
            # painted can never be beached. Its ring indices must be in
            # `touched`, or they stay unpainted until a later event.
            # `footprint` is already the dedupe-filtered core, which is
            # correct: a tile skipped because an earlier touch painted it
            # already holds the stroke's terrain, and its ring was laid then.
            beach_on, beach_id, beach_width = self._stroke_auto_beach
            if beach_on:
                touched.update(apply_beach_ring(mm, footprint, terrain_id, beach_id, beach_width))
        elif self._current_tool == "elevation":
            delta = -1 if modifiers & Qt.ShiftModifier else 1
            # Clamp to the legal range Set Elevation's spinbox already
            # enforces (ELEVATION_LEVEL_MAX) -- Elevate had no clamp at all
            # before Phase 4: going below 0 wrote a negative int that
            # scenario_write.py's raw byte patch (`body[o+1] = tile.elevation`)
            # would crash on at save time, and going above the legal range
            # is exactly the case the fixed-range canvas sizing every Stepped
            # projection uses (iso_geometry.canvas_size_and_origin, see
            # iso_geometry.MAX_ELEVATION) now depends on never happening for
            # Stepped mode's incremental redraw to stay safe. Both real, not
            # hypothetical -- confirmed by reading scenario_write.py and by
            # descape.render.dirty_screen_bbox_iso's own guard below.
            #
            # Every target's current elevation is read here, before ANY tile
            # in this footprint is written -- set_tiles_elevation's
            # propagation pass can still change a not-yet-processed
            # footprint tile's elevation (that's the whole point of a
            # footprint-wide xys), and reading "current" late would apply
            # delta on top of that propagated value instead of the value
            # this cursor tile's drag actually found.
            targets = [
                (tx, ty, max(0, min(ELEVATION_LEVEL_MAX, mm.get_tile(tx, ty).elevation + delta)))
                for tx, ty in footprint
            ]
            touched = set_tiles_elevation(mm, targets)
        elif self._current_tool == "set_level":
            level = self.elevation_level_spin.value()
            touched = set_tiles_elevation(mm, [(tx, ty, level) for tx, ty in footprint])
        else:
            return None
        return touched

    def _apply_stroke_dirty(self, touched: set[int]) -> None:
        # Compared on STATE, not index membership: propagation re-changes one
        # tile several times per drag, and each change must reach _apply_dirty,
        # the only writer of MapView._iso_elevations (tests/test_stroke_elevation_sync.py).
        with perf_trace.phase("stroke_dirty"):
            new_dirty = self.edit_history.stroke_new_dirty(
                touched, self.scenario.map_manager.terrain, self._stroke_seen_state
            )
        if self._terrain_style == "flat" and self._current_tool in _FLAT_SKIP_REPAINT_TOOLS:
            # Neither Flat render path reads tile.elevation, so the repaint is byte-identical
            # (tests/test_flat_elevate_skip.py). Keeps _apply_dirty's warm cancel/re-arm.
            if new_dirty:
                with perf_trace.phase("warm_restart"):
                    self._cancel_warms()
                    self._start_level_warm()
            return
        self._apply_dirty(new_dirty)

    def on_edit_stroke_end(self) -> None:
        if self.scenario is None:
            return
        # The per-step overlay timer skips kept selections while MapView's stroke is active.
        if self._selection and self._current_tool in ("elevation", "set_level") and self._terrain_style != "flat":
            self.map_view.schedule_footprint_refresh()
        # First, and on the open stroke rather than the tool, so no tool state
        # can strand its unit edit and tile stroke open.
        if self._erase is not None:
            self._end_erase_stroke()
            return
        if self._unit_brush is not None:
            self._end_unit_brush_stroke()
            return
        if self._current_tool == "unit_brush":
            return
        if self._current_tool == "convert":
            self._end_convert_stroke()
            return
        if self._current_tool == "cliff":
            self._end_cliff_stroke()
            return
        # _close_stroke_on_error() already committed it; MapView still sends the release.
        if not self.edit_history.in_stroke:
            return
        base_label = _STROKE_LABELS.get(self._current_tool, "Edit")
        target = self._stroke_target
        label = terrain_target.stroke_label(self._current_tool, target, base_label)
        mm = self.scenario.map_manager
        if self._current_tool == "draw":
            # Not commit_stroke(): terrain_units needs the built-but-unpushed
            # record first, to decide whether it rides alone or inside a
            # CompositeDiffRecord with the trees/eye-candy it triggers.
            with perf_trace.phase("stroke_record"):
                tile_record = self.edit_history.build_stroke_record(label, mm.terrain, self._stroke_written)
            if target == terrain_target.TERRAIN:
                with perf_trace.phase("unit_plan"):
                    record, splices = self._apply_terrain_unit_plan(tile_record, mm, with_splices=True)
            else:
                # A layer-only stroke changes no terrain_id, so there is nothing to plant or remove.
                record, splices = tile_record, None
                changed = len(tile_record.changes) if tile_record is not None else 0
                self._log_status(
                    self._layer_edit_status(label, target, self.terrain_panel.terrain_id(), changed, None)
                )
            # Every tile in the record was already repainted live, step by step.
            self._push_terrain_unit_record(record, splices, live_painted=True)
        else:
            # The History dialog's _notify rides inside commit_stroke, so it is part of this phase.
            with perf_trace.phase("stroke_commit"):
                self.edit_history.commit_stroke(label, mm.terrain, self._stroke_written)
        self._stroke_seen_state = {}
        self._stroke_painted = set()
        self._stroke_written = set()
        self._stroke_auto_beach = (False, None, 0)
        self._stroke_target = terrain_target.TERRAIN
        with perf_trace.phase("edit_actions"):
            self._update_edit_actions()
            self._update_title()
        # The tool's own label whatever the Target: tools/bench_gui_drag.py parses it.
        perf_trace.flush(base_label.lower().replace(" ", "-"))

    def on_shape_commit(self, tiles, latch: unit_shape.ShapeLatch | None = None) -> None:
        """Draw Line / Draw Rectangle: one undo record per drag, applied at
        release with the exact tile set MapView's preview named. MapView
        computes that set (see its own _shape_tiles) so the committed shape
        can never disagree with the previewed one.

        Shaped like on_fill() rather than the stroke handlers, for the same
        reason: there is no live drag to give feedback during, so per-event
        stroke bookkeeping buys nothing here.
        Routed through _apply_terrain_unit_plan() too, so Trees/Eye candy
        follow a painted line exactly as they follow a Draw stroke. Wrapped
        in on_fill()'s busy/wait-cursor guard: a full-map filled rectangle
        is the same order of work as a full-map flood fill (~540ms in Flat),
        long enough that the window would otherwise look hung."""
        if self.scenario is None or self._busy or not tiles:
            return
        # Unit Line and Unit Shape (GH #365) hand back the latch their press
        # read, and commit from it alone: above the _WALL_TOOLS check, which
        # would send a unit pick to _commit_wall_run() and its refusal.
        if latch is not None:
            if latch.branch == unit_shape.WALL:
                self._commit_wall_run(tiles, latch.consts[0], owner=latch.owner)
            else:
                self._commit_unit_shape(tiles, latch)
            return
        # Place Unit's wall branch (GH #98) is the other non-terrain shape
        # commit, so this method's whole body below is the terrain branch.
        # Dispatched before the tree/eye-candy size prompt, which has nothing
        # to say about units.
        if self._current_tool in _WALL_TOOLS:
            self._commit_wall_run(tiles, self.units_panel.selected_object_const())
            return
        mm = self.scenario.map_manager
        terrain_id = self.terrain_panel.terrain_id()
        # Read once, before the busy guard: one drag is one Target.
        target = self._terrain_target()
        on_base = target == terrain_target.TERRAIN
        label = terrain_target.stroke_label(self._current_tool, target, _STROKE_LABELS.get(self._current_tool, "Edit"))

        # Same pre-write size guard as on_fill's -- sized before anything is
        # written, so Cancel leaves the map untouched. Draw is exempt there
        # because a stroke's count is bounded by brush size x drag length; a
        # filled rectangle has no such bound, so it needs the guard.
        # Off the Terrain target nothing is planted, so there is nothing to ask about.
        paints_units = on_base and (self.paint_trees_check.isChecked() or self.paint_eye_candy_check.isChecked())
        if paints_units and len(tiles) > TERRAIN_UNIT_CONFIRM_THRESHOLD:
            reply = QMessageBox.question(
                self,
                "Large edit",
                f"This covers {len(tiles)} tiles and can place a large number of "
                f"trees/eye candy units, which may take a while. Continue?",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if reply != QMessageBox.Yes:
                return

        # GH #88: read before the busy guard disables the window (and with it the box).
        beach_on = on_base and self.auto_beach_check.isEnabled() and self.auto_beach_check.isChecked()
        beach_id, beach_width = self.beach_combo.currentData(), self.beach_width_spin.value()
        self._busy = True
        self.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            self.edit_history.begin_stroke(mm.terrain)
            with self._close_stroke_on_error(label):
                for tx, ty in tiles:
                    terrain_target.write_tile(mm.get_tile(tx, ty), target, terrain_id)
                if beach_on:
                    apply_beach_ring(mm, _shape_shore(tiles), terrain_id, beach_id, beach_width)
            tile_record = self.edit_history.build_stroke_record(label, mm.terrain)
            if on_base:
                record, _splices = self._apply_terrain_unit_plan(tile_record, mm)
            else:
                # A layer-only edit changes no terrain_id: nothing to plant or remove.
                record = tile_record
            self._push_terrain_unit_record(record)
            if isinstance(record, TileDiffRecord):
                # No CompositeDiffRecord, so _push_terrain_unit_record skipped
                # the repaint -- and unlike a stroke, nothing repainted
                # incrementally on the way here. Same branch on_fill takes.
                self._apply_dirty(record.touched_indices())
        finally:
            QApplication.restoreOverrideCursor()
            self.setEnabled(True)
            self._busy = False
        self._update_edit_actions()
        self._update_title()
        # Logged even when nothing changed, same reasoning as on_fill's: a
        # silent no-op reads as a broken tool.
        if on_base:
            self._log_status(f"{label}: {len(tiles)} tiles set to {name_for_terrain_id(terrain_id)}")
        else:
            changed = len(tile_record.changes) if tile_record is not None else 0
            self._log_status(self._layer_edit_status(label, target, terrain_id, changed, len(tiles)))

    def _apply_terrain_unit_plan(
        self, tile_record: TileDiffRecord | None, mm, with_splices: bool = False
    ) -> tuple[DiffRecord | None, list[UnitSplice] | None]:
        """Given an unpushed TileDiffRecord from one terrain-paint gesture
        (Draw's stroke or Paint Can's one-shot fill), rolls Trees/Eye candy
        per descape/terrain_units.py's measured placement model and returns
        the record to push: `tile_record` unchanged if nothing was planned
        (both checkboxes off, no terrain actually changed, or this file's
        units are read-only), or a CompositeDiffRecord bundling it with a
        UnitDiffRecord so one Ctrl+Z undoes both -- exactly paste_region()'s
        own shape. `tile_record` may itself be None (a no-op stroke/fill);
        that passes straight through, matching build_stroke_record's own
        "nothing changed" convention.

        Paired with one render_cache_splice.UnitSplice per removed and added unit
        when `with_splices` asks for them, else None (the wholesale
        invalidation). Flat also needs the plan within UNIT_SPLICE_MAX_UNITS.
        Stepped/Sloped build them past it (TASK-031.61 A2): the cache refuses
        such a batch with `cap` and updates units_by_tile in place, and the
        repaint stays scoped to the stroke's bbox instead of the whole canvas.
        """
        if tile_record is None:
            return None, None
        trees = self.paint_trees_check.isChecked()
        doodads = self.paint_eye_candy_check.isChecked()
        if not trees and not doodads:
            return tile_record, None

        width = mm.map_width
        # Sub-phases split on_edit_stroke_end's `unit_plan` phase; null under any other caller.
        with perf_trace.sub_phase("unit_plan", "scan"):
            tiles_by_index = {i: (i % width, i // width) for i, _old, _new in tile_record.changes}
            enabled_consts: frozenset[int] = frozenset()
            if trees:
                enabled_consts |= terrain_units.TREE_CONSTS
            if doodads:
                enabled_consts |= terrain_units.DOODAD_CONSTS
            existing: dict[tuple[int, int], list] = {}
            for unit in self.scenario.unit_manager.units[GAIA_PLAYER_ID]:
                if unit.unit_const in enabled_consts:
                    existing.setdefault((int(unit.x), int(unit.y)), []).append(unit)
            # GH #183: never plant on a changed tile under a cliff's footprint.
            changed_tiles = set(tiles_by_index.values())
            blocked: set[tuple[int, int]] = set()
            for cliff in cliff_chain.existing_cliffs(self.scenario, width, mm.map_height):
                x0, x1, y0, y1 = cliff.bounds
                blocked.update(
                    (x, y) for y in range(y0, y1) for x in range(x0, x1) if (x, y) in changed_tiles
                )

        with perf_trace.sub_phase("unit_plan", "plan"):
            plan = terrain_units.plan_terrain_units(
                tile_record.changes,
                tiles_by_index,
                existing,
                trees=trees,
                doodads=doodads,
                rng=random.Random(),
                blocked=blocked,
            )
        if not plan:
            return tile_record, None

        model = self._ensure_unit_edits()
        if model is None:
            self._log_status("Trees/eye candy skipped -- units are read-only for this file", level=debug_log.WARNING)
            return tile_record, None

        gaia = self.scenario.unit_manager.units[GAIA_PLAYER_ID]
        splices: list[UnitSplice] | None = None
        # Stepped/Sloped past the cap too: the cache refuses with `cap` and updates units_by_tile in place.
        uncapped = self._render_style != "flat"
        with perf_trace.sub_phase("unit_plan", "splices"):
            if with_splices and (
                uncapped or len(plan.removes) + len(plan.adds) <= render_cache_splice.UNIT_SPLICE_MAX_UNITS
            ):
                # Footprints before remove_many(); the index is never read for a removal.
                position = {id(u): i for i, u in enumerate(gaia)} if plan.removes else {}
                splices = []
                for unit in plan.removes:
                    own, tiles = self._unit_footprint(unit)
                    splices.append(UnitSplice(GAIA_PLAYER_ID, position[id(unit)], unit, own, None, tiles, ()))
        with perf_trace.sub_phase("unit_plan", "begin"):
            model.begin_unit_edit([GAIA_PLAYER_ID])
        with perf_trace.sub_phase("unit_plan", "model"):
            try:
                if plan.removes:
                    model.remove_many(plan.removes)
                added = model.add_many(GAIA_PLAYER_ID, plan.adds) if plan.adds else []
            except Exception:
                model.abort_unit_edit()
                raise
        if splices is not None:
            with perf_trace.sub_phase("unit_plan", "splices"):
                # add_many() appends, so the new units are the list's last len(added).
                base = len(gaia) - len(added)
                for offset, unit in enumerate(added):
                    own, tiles = self._unit_footprint(unit)
                    splices.append(UnitSplice(GAIA_PLAYER_ID, base + offset, unit, None, own, (), tiles))
        with perf_trace.sub_phase("unit_plan", "commit"):
            unit_record = model.commit_unit_edit(tile_record.label, self.edit_history, push=False)
        return CompositeDiffRecord(tile_record.label, children=[tile_record, unit_record]), splices

    def _push_terrain_unit_record(
        self, record: DiffRecord | None, splices: list[UnitSplice] | None = None, live_painted: bool = False
    ) -> None:
        """Pushes what _apply_terrain_unit_plan() returned. A plain
        TileDiffRecord needs no extra repaint here -- Draw's own live
        per-tile _apply_dirty() during the drag (on_edit_stroke_tile) and
        Paint Can's own busy-guarded caller already cover the terrain --
        but a CompositeDiffRecord means units changed too, which needs the
        unit-cache invalidation/repaint _after_unit_mutation() does and
        which nothing upstream has done yet.

        live_painted: Draw's stroke end, whose tiles were all repainted
        during the drag, so only the units' own footprints need patching,
        via `splices`. Paint Can and the shapes pass False and keep the
        whole-record repaint plus the wholesale unit invalidation. Each of
        those live repaints also re-read both terrain mirrors, so the unit
        step is told the terrain is current (_after_unit_mutation()).
        """
        if record is None:
            return
        if isinstance(record, CompositeDiffRecord):
            self.edit_history.push_composite_record(record)
            if not live_painted:
                self._apply_dirty(record.touched_indices())
                splices = None
            # Draw is a Terrain tool; if it ever reaches here in Units mode, no index patch matched the splices.
            self._after_unit_mutation(
                self._if_spliceable(splices), rebuild_index=self.mode == "units", batch=True,
                terrain_current=live_painted,
            )
        else:
            self.edit_history.push_tile_record(record)

    # -- Convert brush (phase 3.5b's b2.5) --------------------------------
    #
    # Deliberately NOT built on edit_history.begin_stroke/commit_stroke like
    # the terrain strokes above: those diff TileState over a fixed-size
    # array, and a unit edit is a UnitEditModel record instead. The edit is
    # opened once at stroke start declaring ALL nine players (which ones a
    # drag will touch isn't known yet), each touched unit is reassigned
    # live, and the edit commits once at stroke end: one record per drag,
    # like every other brush tool.
    #
    # The repaint is coalesced by _unit_stroke_refresh_timer. The pick index is
    # deliberately left stale until stroke end: a full rebuild costs ~60ms on
    # the largest corpus file, and _convert_touched_tiles already guarantees
    # no tile is examined twice, so stale entries can't be reassigned twice.
    #
    # Each reassign queues a render_cache_splice.UnitSplice, so a flush splices only
    # the converted units instead of rebuilding every unit-derived structure.

    CONVERT_REFRESH_MS = 100

    def _begin_convert_stroke(self) -> None:
        # An open Unit Brush stroke shares the pending list: its own end commits and repaints it.
        if self._unit_brush is not None:
            self._end_unit_brush_stroke()
        # A stroke that never reached _end_convert_stroke (none known) must not
        # leave its edit open, or the next unit edit raises. Committed, as the
        # brush's: its reassigns are live, and its end repaints the unflushed ones.
        if self._convert_model is not None:
            self._end_convert_stroke()
        self._convert_done = set()
        self._convert_refused = set()
        self._convert_held = set()
        self._convert_hosts = None
        self._convert_touched_tiles = set()
        # Read once and held for the whole drag rather than re-read per
        # touched tile: the combo can't actually change while the mouse is
        # captured, and holding it fixed keeps every reassignment in one
        # drag consistent even if that ever stopped being true.
        self._convert_destination = self.units_panel.owner_id()
        model = self._ensure_unit_edits()
        if model is None:
            return
        model.begin_unit_edit(list(range(GAIA_PLAYER_ID, MAX_PLAYER_ID + 1)))
        self._convert_model = model

    def _convert_stroke_tile(self, x: int, y: int) -> None:
        model = self._convert_model
        if model is None or self.scenario is None:
            return
        index = self.map_view._unit_index
        if index is None:
            return
        footprint = self._brush_footprint(x, y)
        sources = {pid for pid, action in self.convert_source_actions.items() if action.isChecked()}
        destination = self._convert_destination
        destination_units = self.scenario.unit_manager.units[destination]
        converted = False
        for tx, ty in footprint:
            if (tx, ty) in self._convert_touched_tiles:
                continue
            self._convert_touched_tiles.add((tx, ty))
            for order in index.by_tile.get((tx, ty), ()):
                entry = index.entries[order]
                key = unit_pick.unit_key(entry.player_id, entry.unit)
                # A multi-tile unit appears in several touched tiles' buckets.
                if key in self._convert_done or key in self._convert_refused or key in self._convert_held:
                    continue
                if entry.player_id in sources and entry.player_id != destination:
                    # GH #32: never a GAIA-only object (or a host carrying one) to a player.
                    if self._owner_change_refused(model, entry, destination):
                        self._convert_refused.add(key)
                        # A refused host's visible occupants stay with it: no mixed-owner garrison.
                        self._convert_held.update(
                            unit_pick.unit_key(o.player_id, o.unit)
                            for o in self._garrison_occupant_entries(model, [entry])
                        )
                        continue
                    # TASK-474: an occupant reached before its refused host stays with it too.
                    if self._convert_host_refused(model, entry, sources, destination):
                        self._convert_held.add(key)
                        continue
                    self._convert_one(model, entry, key, destination, destination_units)
                    # by_tile lacks hidden occupants. They go with their host
                    # whatever their own owner, like the Inspector's Owner (GH #115).
                    for occupant in self._garrison_occupant_entries(model, [entry]):
                        occupant_key = unit_pick.unit_key(occupant.player_id, occupant.unit)
                        if occupant_key not in self._convert_done and occupant.player_id != destination:
                            self._convert_one(model, occupant, occupant_key, destination, destination_units)
                    converted = True
        # Throttle, not debounce: restarting a running timer on every tile
        # would hold the repaint off for as long as the drag keeps moving.
        if converted and not self._unit_stroke_refresh_timer.isActive():
            self._unit_stroke_refresh_timer.start(self.CONVERT_REFRESH_MS)

    def _convert_host_refused(self, model, entry, sources, destination: int) -> bool:
        """Whether `entry` sits (transitively) inside a host this stroke would
        refuse (GH #32) once it reached it, so the occupant waits for its host
        instead of converting first: an occupant's own tile can come earlier
        in the stroke or in by_tile than its host's (TASK-474)."""
        host_ref = getattr(entry.unit, "garrisoned_in_id", -1)
        if host_ref in (-1, entry.unit.reference_id):
            return False
        if self._convert_hosts is None:
            self._convert_hosts = {}
            for pid, units in enumerate(self.scenario.unit_manager.units):
                for unit in units:
                    self._convert_hosts.setdefault(unit.reference_id, (pid, unit))
        seen = {id(entry.unit)}
        while host_ref != -1:
            found = self._convert_hosts.get(host_ref)
            if found is None or id(found[1]) in seen:
                return False
            pid, host = found
            seen.add(id(host))
            # A host already converted carried its garrison; its key is the pre-stroke one.
            if unit_pick.unit_key(pid, host) in self._convert_done:
                return False
            if pid in sources and pid != destination:
                host_entry = unit_pick.UnitEntry(pid, host, int(host.x), int(host.y), -1)
                if self._owner_change_refused(model, host_entry, destination):
                    return True
            host_ref = getattr(host, "garrisoned_in_id", -1)
            if host_ref == host.reference_id:
                return False
        return False

    def _convert_one(self, model, entry, key, destination: int, destination_units) -> None:
        """One live reassign inside the open Convert edit, queued as a splice."""
        old_own, old_tiles = self._unit_footprint(entry.unit)
        # As _erase_units(): a warm in flight walks the lists reassign() edits (TASK-326).
        self._cancel_warms()
        model.reassign(entry.unit, destination)
        # reassign() appends, so the index is the tail, not an O(n) _unit_list_index() scan.
        self._unit_stroke_pending_splices.append(UnitSplice(
            destination, len(destination_units) - 1, entry.unit,
            old_own, old_own, old_tiles, old_tiles, old_player_id=entry.player_id,
        ))
        self._convert_done.add(key)

    def _flush_unit_stroke_refresh(self) -> None:
        """The mid-stroke repaint of Convert and the Unit Brush: cache and
        canvas only, index left stale (see this section's header)."""
        open_stroke = self._convert_model is not None or self._unit_brush is not None
        if open_stroke and self._unit_stroke_pending_splices:
            pending, self._unit_stroke_pending_splices = self._unit_stroke_pending_splices, []
            self._after_unit_mutation(changed=pending, defer_index=True)

    def _end_convert_stroke(self) -> None:
        self._unit_stroke_refresh_timer.stop()
        model = self._convert_model
        self._convert_model = None
        count = len(self._convert_done)
        refused_note = _gaia_only_refused_note(len(self._convert_refused))
        self._convert_done = set()
        self._convert_refused = set()
        self._convert_held = set()
        self._convert_hosts = None
        self._convert_touched_tiles = set()
        pending, self._unit_stroke_pending_splices = self._unit_stroke_pending_splices, []
        if model is None:
            return
        owner_text = "GAIA" if self._convert_destination == GAIA_PLAYER_ID else f"Player {self._convert_destination}"
        if not count:
            model.abort_unit_edit()
            if refused_note:
                self._log_status(f"Converted 0 units to {owner_text}{refused_note}")
            return
        label = "Convert unit" if count == 1 else f"Convert {count} units"
        model.commit_unit_edit(label, self.edit_history)
        # rebuild_index is load-bearing: the pick index was left stale all stroke.
        self._after_unit_mutation(changed=pending, rebuild_index=True)
        self._update_title()
        self._update_edit_actions()
        self._log_status(f"Converted {count} unit(s) to {owner_text}{refused_note}")

    # -- Unit Brush (GH #170) -----------------------------------------------
    #
    # Convert's stroke shape with Place Unit's placement. One UnitEditModel
    # edit opened at press over the owners the stroke's consts can go to, units
    # added live, the shared throttled flush above, one record and an index
    # rebuild at release. What each dab places is descape/unit_brush.py's.
    #
    # TASK-318's stale-level mark is not needed here: it guards a removal,
    # whose before-extent is read from a level's layer. An add's splice has no
    # old tiles, so sprite_extent_before() reads nothing for it, and a level
    # rebuilt from the live scenario inside the window only widens the after-extent.
    # Such a level already draws the add, so the add splice clears its new tiles
    # before drawing it (render_cache_splice._reanchor_units()), and it is drawn once.

    def _begin_unit_brush_stroke(self) -> None:
        # A stroke that never reached its end is committed, as the Eraser's is: its adds are live.
        if self._unit_brush is not None:
            self._end_unit_brush_stroke()
        # So is an open Convert stroke, which shares the pending list.
        if self._convert_model is not None:
            self._end_convert_stroke()
        # Left button only: the Eraser is one key away.
        if self.map_view.stroke_button() != Qt.LeftButton:
            return
        consts = self.units_panel.selected_object_choices()
        if not consts:
            self._log_status("Unit Brush: choose an object first")
            return
        for const in consts:
            reason = unit_brush.is_refused_const(const)
            if reason is not None:
                self._log_status(f"Unit Brush: can't paint {_unit_name(const)}: {reason}")
                return
        model = self._ensure_unit_edits()
        if model is None:
            return
        # Every widget read once, so a mid-drag change cannot half-apply across one record.
        stroke = unit_brush.BrushStroke(
            consts=tuple(consts),
            chosen_owner=self.units_panel.owner_id(),
            density=self.unit_brush_density_spin.value() / 100,
            rules=not self._restrictions_lifted(),
            randomize=self.unit_brush_random_check.isChecked(),
            pending=self._place_rotation,
            rng=random.Random(),
            off_grid=self.off_grid_check.isChecked(),
        )
        model.begin_unit_edit(stroke.owners())
        self._unit_brush = stroke
        self._unit_brush_model = model
        self._unit_brush_base = self._unit_brush_owned(stroke)

    def _unit_add_fn(self, model, splices: list):
        """An add for unit_brush.place_candidate(): adds one Spec through
        `model` and appends its splice to `splices`. The Unit Brush and Unit
        Shape (GH #365) share it."""
        units = self.scenario.unit_manager.units
        cancelled = False

        def add(spec: unit_brush.Spec):
            nonlocal cancelled
            if not cancelled:
                # As _convert_one(): a warm in flight walks the lists add() appends to (TASK-326).
                self._cancel_warms()
                cancelled = True
            unit = model.add(spec.player, spec.unit_const, spec.x, spec.y, rotation=spec.rotation)
            own, occupied = self._unit_footprint(unit)
            # add() appends, so the index is the tail, not an O(n) _unit_list_index() scan.
            splices.append(UnitSplice(spec.player, len(units[spec.player]) - 1, unit, None, own, (), occupied))
            return unit

        return add

    def _unit_brush_owned(self, stroke: unit_brush.BrushStroke) -> int:
        units = self.scenario.unit_manager.units
        return sum(len(units[player]) for player in stroke.owners())

    def _unit_brush_stroke_tiles(self, tiles, modifiers) -> None:
        """One MapView event's cursor path: every brush tile not yet rolled this drag, as one dab."""
        stroke, model = self._unit_brush, self._unit_brush_model
        if stroke is None or model is None or self.scenario is None:
            return
        if stroke.jitter is None:
            # on_edit_stroke_start() gets no modifiers, so a held Alt is read here, once.
            alt = modifiers is not None and bool(modifiers & Qt.AltModifier)
            stroke.jitter = scatter.MAX_JITTER if stroke.off_grid or alt else 0.0
        footprint = []
        for x, y in tiles:
            footprint.extend(t for t in self._brush_footprint(x, y) if t not in stroke.rolled)
        if not footprint:
            return
        add = self._unit_add_fn(model, self._unit_stroke_pending_splices)
        with model.batch_adds():
            placed = unit_brush.paint_dab(self.scenario, stroke, footprint, add)
        # Throttle, not debounce, as Convert's.
        if placed and not self._unit_stroke_refresh_timer.isActive():
            self._unit_stroke_refresh_timer.start(self.CONVERT_REFRESH_MS)

    def _end_unit_brush_stroke(self) -> None:
        self._unit_stroke_refresh_timer.stop()
        stroke, self._unit_brush = self._unit_brush, None
        model, self._unit_brush_model = self._unit_brush_model, None
        pending, self._unit_stroke_pending_splices = self._unit_stroke_pending_splices, []
        if stroke is None or model is None:
            return
        if model is not self.unit_edits:
            # A document swap mid-drag (_drop_unit_brush_stroke() is the usual path): not this history's.
            model.abort_unit_edit()
            return
        # Counted in the model, not the plan: a raise right after an add() leaves a unit the plan never saw.
        count = self._unit_brush_owned(stroke) - self._unit_brush_base
        if not count:
            model.abort_unit_edit()
            self._log_status(self._unit_brush_status(stroke))
            return
        label = "Paint unit" if count == 1 else f"Paint {count} units"
        model.commit_unit_edit(label, self.edit_history)
        # rebuild_index is load-bearing: the pick index was left stale all stroke.
        # An add the plan never saw may have no splice either: None rebuilds the cache whole.
        self._after_unit_mutation(changed=pending if count == len(stroke.placed) else None, rebuild_index=True)
        self._update_title()
        self._update_edit_actions()
        self._log_status(self._unit_brush_status(stroke, count))

    def _drop_unit_brush_stroke(self) -> None:
        """A load or close mid-drag: the open stroke's edit is the closed
        file's, so later moves paint nothing into the next document."""
        stroke, self._unit_brush = self._unit_brush, None
        model, self._unit_brush_model = self._unit_brush_model, None
        if stroke is None:
            return
        self._unit_stroke_refresh_timer.stop()
        self._unit_stroke_pending_splices = []
        if model is not None:
            model.abort_unit_edit()
        self._log_status("Unit Brush: the document changed during the drag, so nothing was recorded")

    def _commit_unit_shape(self, cells, latch: unit_shape.ShapeLatch) -> None:
        """Unit Line and Unit Shape's unit branch (GH #365): one candidate per
        lattice cell through the Unit Brush's tail, as one undo record, or
        none when nothing was placed. A one-shot commit has no later call to
        close a stuck edit, so the edit is closed in a finally."""
        if len(cells) > TERRAIN_UNIT_CONFIRM_THRESHOLD:
            reply = QMessageBox.question(
                self,
                "Large edit",
                f"This places up to {len(cells)} objects, which may take a while. Continue?",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if reply != QMessageBox.Yes:
                return
        model = self._ensure_unit_edits()
        if model is None:
            return
        owners = sorted({unit_kind.placement_owner(const, latch.owner) for const in latch.consts})
        units = self.scenario.unit_manager.units
        base = sum(len(units[player]) for player in owners)
        splices: list = []
        stroke = None
        count = 0
        model.begin_unit_edit(owners)
        try:
            add = self._unit_add_fn(model, splices)
            with model.batch_adds():
                stroke = unit_shape.place_cells(self.scenario, latch, cells, add, random.Random())
        finally:
            # Counted in the model, not the plan: a raise right after an add() leaves a unit the plan never saw.
            count = sum(len(units[player]) for player in owners) - base
            if not count:
                model.abort_unit_edit()
            else:
                if len(latch.consts) == 1:
                    name = _unit_name(latch.consts[0])
                    label = f"Place {name}" if count == 1 else f"Place {count} {name}"
                else:
                    label = f"Draw {dict(unit_shape.SHAPE_KINDS).get(latch.kind, 'Line')}"
                model.commit_unit_edit(label, self.edit_history)
                complete = stroke is not None and count == len(stroke.placed)
                self._after_unit_mutation(changed=splices if complete else None, rebuild_index=True)
                self._update_title()
                self._update_edit_actions()
        self._log_status(self._unit_brush_status(stroke, count, verb="Placed", label=latch.label))

    def _unit_brush_status(
        self, stroke: unit_brush.BrushStroke, count: int = 0, verb: str = "Painted", label: str | None = None
    ) -> str:
        """The release line: what was painted (`count`, the model's), for whom,
        and the rule refusals alone (density misses, self-stack and the map
        edge are not counted). A mixed pick is named by `label`, else the
        catalog's current pick."""
        if len(stroke.consts) == 1:
            name = _unit_name(stroke.consts[0])
        else:
            name = (self.units_panel.selected_object_label() if label is None else label) or "objects"
        owner = stroke.chosen_owner
        owner_text = "GAIA" if owner == GAIA_PLAYER_ID else f"Player {owner}"
        status = f"{verb} {count} {name} for {owner_text}"
        if not stroke.randomize and len(stroke.consts) == 1:
            const = stroke.consts[0]
            if unit_rotation.rotation_is_angle(const) or unit_variant.is_cyclable(const):
                rotation = place_rotation.rotation_for(stroke.pending, const, unit_kind.placement_owner(const, owner))
                status += f", {self._placement_rotation_text(const, rotation)}"
        if stroke.refused:
            reason = stroke.refusal_reason
            status += f", {stroke.refused} refused"
            if reason:
                status += f" ({reason[0].upper()}{reason[1:]}. Tick Free placement to place them anyway.)"
        if stroke.gaia_only:
            status += f" ({stroke.gaia_only} GAIA-only, so placed for GAIA)"
        hidden = sum(1 for player, unit in stroke.placed if not self._unit_filter.matches(player, unit))
        if hidden:
            status += f" ({hidden} hidden by a Filters toggle, so they won't be visible)"
        return status

    # -- Eraser brush (GH #184) ---------------------------------------------
    #
    # Convert's stroke shape: one UnitEditModel edit over all nine players
    # opened at press, units removed live, one record at release. Two
    # differences. The hit index is built fresh at press from the current
    # filter, because map_view._unit_index is None in Terrain mode and must
    # not be mutated mid-stroke in Units mode. And Terrain Overlay adds a
    # scoped EditHistory tile stroke, which rides with the unit record in one
    # CompositeDiffRecord when both changed.

    def _erase_overlay_ok(self) -> bool:
        """Whether Terrain Overlay applies: the same gate as every layered-terrain write."""
        return self._layer_write_ok()

    def _erase_targets(self) -> eraser.EraseTargets:
        """The ticked targets, with Overlay dropped where it can't apply."""
        ticked = {name: action.isChecked() for name, action in self.eraser_target_actions.items()}
        ticked["overlay"] = ticked["overlay"] and self._erase_overlay_ok()
        return eraser.EraseTargets(**ticked)

    def _tick_every_erase_target(self) -> None:
        for name in eraser.EVERYTHING_FIELDS:
            self.eraser_target_actions[name].setChecked(True)

    def _begin_erase_stroke(
        self, targets: eraser.EraseTargets | None = None, cursor_only: bool = False, cliff_ends: bool = False
    ) -> eraser.EraseStroke:
        """Opens an Eraser stroke: the menu's targets and the brush by default,
        or the Cliff tool's right-button erase (_begin_cliff_erase())."""
        # A stroke that never reached its end is committed, not dropped: its removals are live.
        if self._erase is not None:
            self._end_erase_stroke()
        # Snapshotted once: a mid-drag tick change must not half-apply across one undo record.
        if targets is None:
            targets = self._erase_targets()
        stroke = eraser.EraseStroke(
            targets=targets, overlay=targets.overlay, cursor_only=cursor_only, cliff_ends=cliff_ends
        )
        # Tile stroke first: if it raises, no unit edit is left open behind it.
        if stroke.overlay:
            # Scoped: each cleared tile is captured just before its write, so press costs O(1).
            self.edit_history.begin_stroke(self.scenario.map_manager.terrain, indices=())
        if targets.any_units:
            model = self._ensure_unit_edits()
            if model is not None:
                stroke.index = unit_pick.build_index(self.scenario, self._unit_filter)
                model.begin_unit_edit(list(range(GAIA_PLAYER_ID, MAX_PLAYER_ID + 1)))
                stroke.model = model
        self._erase = stroke
        return stroke

    def _begin_cliff_erase(self) -> None:
        """GH #247: a right press with the Cliff tool erases the visible cliffs
        under the cursor tile, through the Eraser's own stroke, and re-caps
        the run ends it leaves at release (_end_erase_stroke())."""
        view = self.map_view
        view.set_unit_ghost()
        view.set_cliff_footprint(None)
        self._cliff_hover_memo = None
        # Before any removal: the release judges the survivors on the map the press saw.
        cliffs = self._cliff_existing()
        stroke = self._begin_erase_stroke(
            targets=eraser.EraseTargets(units=False, buildings=False, cliffs=True), cursor_only=True, cliff_ends=True
        )
        stroke.cliffs = cliffs
        stroke.scenario_ref = weakref.ref(self.scenario)
        stroke.expected_gen = self.scenario.unit_gen

    def _erase_stroke_tiles(self, tiles) -> None:
        """One MapView event's cursor path: every brush tile not yet visited this drag."""
        stroke = self._erase
        if stroke is None or self.scenario is None:
            return
        mm = self.scenario.map_manager
        footprint = []
        for x, y in tiles:
            if stroke.cursor_only:
                covered = [(x, y)] if 0 <= x < mm.map_width and 0 <= y < mm.map_height else []
            else:
                covered = self._brush_footprint(x, y)
            for tile in covered:
                if tile not in stroke.touched_tiles:
                    stroke.touched_tiles.add(tile)
                    footprint.append(tile)
        if not footprint:
            return
        if stroke.model is not None:
            hits = eraser.collect_hits(stroke.targets, stroke.index, footprint, stroke.done)
            if hits:
                self._erase_units(stroke, hits)
        if stroke.overlay:
            terrain = mm.terrain
            for tx, ty in footprint:
                i = ty * mm.map_width + tx
                if terrain[i].layer != -1:
                    self.edit_history.stroke_capture(terrain, i)
                    terrain[i].layer = -1
                    stroke.layer_cleared += 1
                    stroke.pending_layer_tiles.append(i)

    def _erase_units(self, stroke: eraser.EraseStroke, hits) -> None:
        """Removes `hits` and their garrisons live, queueing one removal splice each.

        The model changes here and the cache only at the next flush or the
        release, so every resident level that is not is_level_resident() now
        is marked (TASK-318): a paint, a zoom or the release's tile patch can
        rebuild such a level from the live scenario inside that window, and
        its layer then lacks the erased units while their chunks still show
        them. A stale level rebuilds that way, and so does a current one with
        pending elevation work, whose _flush_pending() re-anchors from live
        state; hence that predicate and not a gen compare. The flush sizes
        the batch by reach on a marked visible level (_patch_unit_edit_cache())."""
        model = stroke.model
        # Delete's own pre-flight (GH #42): a host takes its occupants, and a
        # unit remove_many() would refuse (a -1 reference_id) is left in place.
        entries, removable, cascaded_ids = self._delete_preflight(model, hits)
        stroke.done.update(id(e.unit) for e in entries)
        stroke.refused += len(entries) - len(removable)
        if not removable:
            return
        # Footprints before remove_many(), as Draw's trees do. A removal's index is never read (-1).
        connectors = unit_sprites.wall_connector_consts()
        for entry in removable:
            own, tiles = self._unit_footprint(entry.unit)
            stroke.pending_splices.append(
                UnitSplice(entry.player_id, -1, entry.unit, own, None, tiles, ())
            )
            if entry.unit.unit_const in connectors:
                stroke.wall_vacated.update(tiles)
        cache = self._cache
        if cache is not None:
            stroke.pending_stale_levels.update(
                mip for mip in cache.resident_levels() if not cache.is_level_resident(mip)
            )
        # A warm armed by the last flush walks the lists remove_many() edits (TASK-326).
        self._cancel_warms()
        # Anything else that changed the units since (an undo, a redo, a swap) turns the end rewrites off.
        if stroke.expected_gen is not None and (
            stroke.scenario_ref() is not self.scenario or self.scenario.unit_gen != stroke.expected_gen
        ):
            stroke.expected_gen = None
        model.remove_many([e.unit for e in removable])
        if stroke.expected_gen is not None:
            stroke.expected_gen = self.scenario.unit_gen
        stroke.erased_ids.update(id(e.unit) for e in removable)
        stroke.erased += len(removable)
        stroke.cascaded += sum(1 for e in removable if id(e.unit) in cascaded_ids)
        stroke.reference_ids.extend(int(e.unit.reference_id) for e in removable)
        # Throttle, not debounce, as Convert's.
        if not self._erase_refresh_timer.isActive():
            self._erase_refresh_timer.start(self.CONVERT_REFRESH_MS)

    def _flush_erase_refresh(self) -> None:
        """The mid-stroke repaint: cache and canvas only, the pick index left
        for the stroke end, as _flush_unit_stroke_refresh(). Terrain Overlay's
        tiles cleared since the last flush follow the unit step, which
        repaints only what its erased units reached. Units first: the tile
        patch can rebuild a stale visible level, and the unit step has to
        read its erased sprites' extents from that level before it does.
        The levels _erase_units() marked go with the splices, so a level
        rebuilt since those erases sizes the batch by reach.

        Only an erased unit arms the flush (_erase_units()), so a tile
        cleared with no unit erased since still waits for the next flush or
        the release."""
        stroke = self._erase
        if stroke is None:
            return
        if stroke.pending_splices:
            pending, stroke.pending_splices = stroke.pending_splices, []
            stale, stroke.pending_stale_levels = stroke.pending_stale_levels, set()
            self._after_unit_mutation(
                changed=self._if_spliceable(pending), defer_index=True, batch=True, pre_stale_levels=stale
            )
        if stroke.pending_layer_tiles:
            cleared, stroke.pending_layer_tiles = stroke.pending_layer_tiles, []
            self._apply_dirty(cleared)

    def _end_erase_stroke(self) -> None:
        self._erase_refresh_timer.stop()
        stroke, self._erase = self._erase, None
        if stroke is None:
            return
        try:
            self._commit_erase_stroke(stroke)
        finally:
            if stroke.cliff_ends:
                # The Cliff hover ghost comes back at the cursor without a mouse move.
                self.map_view.refresh_cliff_ghost()

    def _drop_erase_stroke(self) -> None:
        """A load or close mid-drag: the open Erase stroke's edit is the closed
        file's. Dropped here so later moves erase nothing and splice nothing
        into the next document's cache; _commit_erase_stroke() keeps a backstop."""
        self._erase_refresh_timer.stop()
        stroke, self._erase = self._erase, None
        if stroke is None:
            return
        if stroke.model is not None:
            stroke.model.abort_unit_edit()
        self._log_status("Erase: the document changed during the drag, so nothing was recorded")

    def _rewrite_cliff_ends(self, stroke: eraser.EraseStroke) -> bool:
        """The Cliff erase's end rewrites, inside its still-open unit edit
        (docs/WRITE_RULES.md's Cliff drag exception, erase trigger). Returns whether any
        was attempted, i.e. whether the repaint has to be wholesale."""
        scenario = stroke.scenario_ref() if stroke.scenario_ref is not None else None
        if stroke.expected_gen is None or scenario is not self.scenario or scenario.unit_gen != stroke.expected_gen:
            stroke.ends_note = "the map's units changed during the drag, so no cliff ends were reshaped"
            return False
        # All of it, plan included: a raise must still reach commit_unit_edit(), the removals are live.
        try:
            existing, index = stroke.cliffs
            erased = {k for k, cliff in enumerate(existing) if id(cliff.unit) in stroke.erased_ids}
            rewrites, stroke.left_alone = cliff_chain.erase_rewrites(existing, index, erased)
            rewrites = [(existing[r.existing_index].unit, r.rotation) for r in rewrites]
            rewrites = [(unit, frame) for unit, frame in rewrites if unit is not None]
            for unit, frame in rewrites:
                stroke.model.set_cliff_frame(unit, frame)
                stroke.reshaped += 1
        except Exception as e:  # noqa: BLE001
            stroke.ends_note = f"reshaping the cliff ends stopped: {type(e).__name__}: {e}"
            # Unknown how far it got: repaint wholesale.
            return True
        return bool(rewrites)

    def _rewrite_wall_ends(self, stroke: eraser.EraseStroke) -> bool:
        """The walls left beside the walls and gates this stroke erased take
        the shape their neighbours now give them (GH #340), inside its
        still-open unit edit, read off the map as it is at release. Returns
        whether any was attempted, i.e. whether the repaint has to be
        wholesale."""
        rewrites = []
        # All of it, plan included: a raise must still reach commit_unit_edit(), the removals are live.
        try:
            mm = self.scenario.map_manager
            tiles, walls = wall_run.wall_scene(self.scenario, mm.map_width, mm.map_height)
            rewrites = wall_run.plan_removal_rewrites(stroke.wall_vacated, existing_tiles=tiles, existing_walls=walls)
            for _player_id, unit, variant in rewrites:
                stroke.model.set_wall_variant(unit, variant)
                stroke.walls_reshaped += 1
        except Exception as e:  # noqa: BLE001
            stroke.ends_note = f"reshaping the walls beside the erased ones stopped: {type(e).__name__}: {e}"
            return True
        return bool(rewrites)

    def _commit_erase_stroke(self, stroke: eraser.EraseStroke) -> None:
        pending, stroke.pending_splices = stroke.pending_splices, []
        stale, stroke.pending_stale_levels = stroke.pending_stale_levels, set()
        if stroke.model is not None and stroke.model is not self.unit_edits:
            # A document swap mid-drag dropped the model: its edit is the closed file's, not this history's.
            stroke.model.abort_unit_edit()
            stroke.model, stroke.erased, stroke.cascaded, stroke.refused, stroke.reference_ids = None, 0, 0, 0, []
            stroke.ends_note = "the document changed during the drag, so nothing was recorded"
        count = stroke.erased
        if count:
            noun = ("cliff", "cliffs") if stroke.cliff_ends else ("object", "objects")
            label = f"Erase {noun[0]}" if count == 1 else f"Erase {count} {noun[1]}"
        else:
            label = _STROKE_LABELS["eraser"]
        tile_record = None
        if stroke.overlay and self.edit_history.in_stroke:
            tile_record = self.edit_history.build_stroke_record(label, self.scenario.map_manager.terrain)
        unit_record = None
        wholesale = False
        if stroke.model is not None:
            if count:
                if stroke.cliff_ends:
                    wholesale = self._rewrite_cliff_ends(stroke)
                if stroke.wall_vacated:
                    wholesale = self._rewrite_wall_ends(stroke) or wholesale
                unit_record = stroke.model.commit_unit_edit(label, self.edit_history, push=tile_record is None)
            else:
                stroke.model.abort_unit_edit()
        if tile_record is not None:
            if unit_record is not None:
                self.edit_history.push_composite_record(
                    CompositeDiffRecord(label, children=[tile_record, unit_record])
                )
            else:
                self.edit_history.push_tile_record(tile_record)
        # Units first, as _flush_erase_refresh(): the tile patch can rebuild a stale visible level.
        if unit_record is not None:
            # rebuild_index is load-bearing: map_view's index was left stale all stroke.
            # A rewritten frame has no removal splice: wholesale, as undo takes it.
            self._after_unit_mutation(
                None if wholesale else self._if_spliceable(pending),
                rebuild_index=True, batch=True, pre_stale_levels=stale,
            )
        if tile_record is not None:
            # A cleared `layer` changes pixels only while terrain blending is effective.
            self._apply_dirty(tile_record.touched_indices())
        if tile_record is not None or unit_record is not None:
            self._update_title()
            self._update_edit_actions()
        self._log_status(self._erase_status(stroke, tile_record))

    def _erase_status(self, stroke: eraser.EraseStroke, tile_record) -> str:
        """The stroke's status line. Trigger references are warned, never
        refused, like Delete and Find's Delete; counted only when triggers
        are already parsed, since the parse can take seconds. A Cliff tool
        erase (`cliff_ends`) words it as cliffs, with its end rewrites."""
        parts = []
        if stroke.erased:
            carried = f" ({stroke.cascaded} garrisoned)" if stroke.cascaded else ""
            parts.append(f"Erased {stroke.erased} {'cliff' if stroke.cliff_ends else 'object'}(s){carried}")
        if stroke.reshaped:
            parts.append(f"{stroke.reshaped} reshaped to end caps")
        if stroke.walls_reshaped:
            parts.append(f"{stroke.walls_reshaped} adjacent wall(s) reshaped")
        if stroke.left_alone:
            parts.append(f"{stroke.left_alone} left as they were (a corner there needs a different-size piece)")
        cleared = len(tile_record.changes) if tile_record is not None else 0
        if cleared:
            parts.append(f"cleared layered terrain on {cleared} tile(s)")
        if not parts:
            hidden = ""
            # With a note (the swap backstop's) the filter is not why nothing was erased.
            hint = stroke.cliff_ends and not stroke.ends_note
            if hint and not self._unit_filter.show_gaia:
                hidden = " (Show GAIA is off)"
            elif hint and self._unit_filter.players is not None:
                hidden = " (a Filters toggle may be hiding it)"
            parts.append(f"Erased nothing{hidden}")
        notes = [stroke.ends_note] if stroke.ends_note else []
        counts = self._find_trigger_ref_counts() if stroke.reference_ids else None
        if counts:
            referenced = sum(1 for ref in stroke.reference_ids if counts.get(ref, 0) > 0)
            if referenced:
                notes.append(f"{referenced} referenced by triggers")
        if stroke.refused:
            notes.append(f"{stroke.refused} not erased (garrison reference)")
        return ", ".join(parts) + "".join(f"; {note}" for note in notes)
