"""ViewerWindow's undo history: the edit actions' state, undo, redo, history moves and the refresh after one."""

from __future__ import annotations

import contextlib

from descape import (
    debug_log,
    perf_trace,
    render_cache_splice,
)
from descape.edit_history import (
    CompositeDiffRecord,
    HistoryMoveFailed,
    UnitDiffRecord,
)
from descape.render_cache_splice import UnitSplice
from descape.text_edits import _MultiLineEdit
from descape.unit_filter import GAIA_PLAYER_ID

# Flat only. Batch splice ~0.3-0.4ms per spliced unit (eager patch), wholesale ~20us per map
# unit: splice only below 1/16 of the map's units. Measured by tools/bench_stroke_end.py.
_SPLICE_COST_RATIO = 16


class _ViewerHistoryMixin:
    """ViewerWindow's undo and redo (viewer.py's docstring has the split's rules)."""

    def _update_edit_actions(self) -> None:
        # The post-edit sweep: an edit under a still pointer must not leave the readout on the old tile (TASK-419).
        self._refresh_hover_text()
        self._refresh_paste_move_validity()
        if self._lift is not None and self._live_lift() is None:
            self._set_lift(None)
        box = self._focused_text_box()
        if box is not None:
            self.undo_action.setEnabled(box.document().isUndoAvailable())
            self.redo_action.setEnabled(box.document().isRedoAvailable())
            return
        can_undo = self.scenario is not None and self.edit_history.can_undo
        can_redo = self.scenario is not None and self.edit_history.can_redo
        self.undo_action.setEnabled(can_undo)
        self.redo_action.setEnabled(can_redo)

    def _focused_text_box(self) -> _MultiLineEdit | None:
        """The multi-line box Edit > Undo/Redo act on (GH #38), or None for
        EditHistory. A box deleted by a form rebuild reads as None."""
        box = self._text_undo_box
        if box is None:
            return None
        try:
            box.document()
        except RuntimeError:
            self._text_undo_box = None
            return None
        return box

    def _on_app_focus_changed(self, old, new) -> None:
        """GH #38: while one of this window's _MultiLineEdit boxes has focus,
        the Edit menu follows its document's own undo stack. Opening the menu
        leaves focusWidget() on the box (a popup focus-out, which the box also
        exempts from committing), so the menu stays live while it is open."""
        # GH #64: from a floating panel's box into this window is a window switch the box does not
        # commit, but it is the click inside one window that would have committed it when docked.
        # A WM may report it in two steps, through no focus widget at all.
        if isinstance(old, _MultiLineEdit) and self.panel_dock.isFloating() and self.panel_dock.isAncestorOf(old):
            self._left_floating_box = old
        if new is not None:
            left, self._left_floating_box = self._left_floating_box, None
            if left is not None and self.isAncestorOf(new):
                with contextlib.suppress(RuntimeError):
                    left.editingFinished.emit()
        # GH #137: before the early return, which every non-box move takes.
        self._update_clipboard_actions()
        box = new if isinstance(new, _MultiLineEdit) and self._owns_widget(new) else None
        if box is self._text_undo_box:
            return
        old_box = self._focused_text_box()
        if old_box is not None:
            document = old_box.document()
            document.undoAvailable.disconnect(self._on_text_undo_available)
            document.redoAvailable.disconnect(self._on_text_undo_available)
        self._text_undo_box = box
        if box is not None:
            box.document().undoAvailable.connect(self._on_text_undo_available)
            box.document().redoAvailable.connect(self._on_text_undo_available)
        self._update_edit_actions()

    def _on_text_undo_available(self, _available: bool) -> None:
        self._update_edit_actions()

    def undo(self) -> None:
        box = self._focused_text_box()
        if box is not None:
            box.undo()
            return
        if self.scenario is None:
            return
        self._move_history(self.edit_history.peek_undo(), self.edit_history.undo, "Undo")

    def redo(self) -> None:
        box = self._focused_text_box()
        if box is not None:
            box.redo()
            return
        if self.scenario is None:
            return
        self._move_history(self.edit_history.peek_redo(), self.edit_history.redo, "Redo")

    def _move_history(self, record, move, action: str, quiet: bool = False) -> None:
        try:
            with perf_trace.op(action):
                self._move_history_traced(record, move, action, quiet)
        finally:
            # The Cliff hover ghost reads the map's cliffs: re-resolve it at the cursor, after a stopped move too.
            self.map_view.refresh_cliff_ghost()

    def _move_history_traced(self, record, move, action: str, quiet: bool = False) -> None:
        """One undo or redo step: read what the record needs *before* the
        cursor moves, move it, then hand the rest to
        _refresh_after_history_move() (shared with the History window's jump,
        so the two cannot drift apart).

        Batch D's D6: a plain UnitDiffRecord for a fields_only edit (Move/
        Nudge/Rotate/Set field/gate orientation) exposes unit_field_entries,
        which lets the unit branch below build one render_cache_splice.UnitSplice
        per touched unit -- the same scoped _after_unit_mutation() path
        _unit_edit() itself takes -- instead of the wholesale invalidation
        every other unit record (place/delete/reassign/paste, or a
        CompositeDiffRecord, neither of which carries that attribute) still
        gets. The pre-mutation footprint has to be read before `move()` runs
        below: by the time the unit branch is reached the model has already
        been put back to the record's other side.

        A GAIA-only add/remove record (Draw's Trees/Eye candy, alone or in its
        CompositeDiffRecord), or a paste-shaped one (every snapshotted player
        a pure tail append or truncation: Paste Region, non-GAIA Place), gets
        the same scoped path through _membership_diff().

        E, the tiles a unit record's tile child changes the elevation of, goes
        to _refresh_after_history_move() as elev_changed, which then runs the
        unit sources before the tile patch (Stepped/Sloped).
        """
        if record is None:
            # Still logs, exactly as paste_region() and fill do on a genuine
            # no-op: silently doing nothing reads as a broken keybind rather
            # than as an empty history.
            self._update_edit_actions()
            self._log_status(f"Nothing to {action.lower()}")
            return
        field_entries = getattr(record, "unit_field_entries", None)
        with perf_trace.phase("history_diff"):
            old_footprints = (
                [(player_id, index, unit, *self._unit_footprint(unit)) for player_id, index, unit in field_entries]
                if field_entries
                else None
            )
            membership = None if field_entries else self._membership_diff(record, move == self.edit_history.undo)
        elev_changed = set()
        if "unit" in record.kinds():
            # Its own leaf phase, not inside history_diff: phase times sum flat, so nesting double-counts.
            with perf_trace.phase("elev_predict"):
                elev_changed = self._elevation_changed_tiles(record)
        collapse = self.trigger_panel.collapse_snapshot() if "trigger" in record.kinds() else None
        try:
            with perf_trace.phase("history_restore"):
                dirty = move(
                    self.scenario.map_manager.terrain,
                    self.trigger_edits,
                    self.option_edits,
                    self.unit_edits,
                    self.message_edits,
                )
        except HistoryMoveFailed as exc:
            # The precomputed splices assumed a finished move: refresh wholesale, then crash-report as before.
            self.trigger_panel.follow_collapse(collapse)
            self._refresh_after_history_move(exc.kinds, exc.touched)
            self._log_status(f"{action} stopped: {exc}", level=debug_log.WARNING)
            raise
        self.trigger_panel.follow_collapse(collapse)
        splices = None
        if old_footprints is not None:
            splices = []
            for player_id, index, unit, old_own, old_tiles in old_footprints:
                new_own, new_tiles = self._unit_footprint(unit)
                splices.append(UnitSplice(player_id, index, unit, old_own, new_own, old_tiles, new_tiles))
        elif membership is not None:
            splices, arriving = membership
            for player_id, index, unit in arriving:
                own, tiles = self._unit_footprint(unit)
                splices.append(UnitSplice(player_id, index, unit, None, own, (), tiles))
            # Stepped/Sloped uncapped (TASK-031.61 A3): past the cap the cache refuses with `cap` itself.
            splices = self._if_spliceable(
                splices, cap=None if self._render_style != "flat" else render_cache_splice.UNIT_SPLICE_MAX_UNITS
            )
        # `record.kinds()`, not `record.kind`: a CompositeDiffRecord
        # (phase 2.8's region paste) can carry more than one domain in a
        # single record, and each still needs the same refresh a plain
        # record of that domain would get.
        self._refresh_after_history_move(
            record.kinds(), dirty, splices, batch=membership is not None, elev_changed=elev_changed
        )
        # quiet: a region move undoes its own previous paste as an internal
        # step, and logging "Undo: Paste Region" there would read as the
        # user's own undo. Suppresses only this line, nothing else.
        if not quiet:
            self._log_status(f"{action}: {record.label}")

    def _if_spliceable(self, splices: list[UnitSplice] | None, cap: int | None = None) -> list[UnitSplice] | None:
        """`splices` for the scoped repaint (_patch_unit_edit_cache()), or
        None for the lazy whole-canvas one. Used by the batch callers (Draw's
        stroke end, membership undo/redo), not the single-unit tools, whose
        small eager patch is always the cheaper of the two. None past `cap`
        splices when one is given (Flat's membership undo/redo; Stepped/Sloped
        pass none since TASK-031.61 A3, leaving the cap to the cache).

        Stepped/Sloped: always `splices`. invalidate_units(splices) either
        splices every level, splices the visible one alone past the caps
        (TASK-031.61 c2x), or falls back to a source rebuild that the sprite
        memo keeps cheap, and the whole-canvas path pays that same rebuild. Whether the
        bbox is then patched eagerly or only evicted is
        _patch_unit_edit_cache()'s call, by area.

        Flat has no memo, so a batch its cache refuses would pay the icon
        layer rebuild inside the eager patch: None then, and past
        1/_SPLICE_COST_RATIO of the map's units."""
        if splices is None or self._cache is None or (cap is not None and len(splices) > cap):
            return None
        if self._render_style == "flat":
            total = sum(len(units) for units in self.scenario.unit_manager.units)
            if len(splices) * _SPLICE_COST_RATIO > total or not self._cache.can_splice(splices):
                return None
            return splices
        return splices

    def _membership_diff(self, record, undo: bool):
        """The undo/redo counterpart of Draw's stroke-end and Paste Region's
        splices: for a unit record that only adds or removes units, alone or as
        a CompositeDiffRecord child, returns (removal splices, [(player_id,
        index, unit), ...] still to add) with the removals' footprints read now,
        before `move()` restores anything. The arriving units' footprints must
        be read after it. None (the wholesale path) for any other shape.

        Two shapes. GAIA-only (Trees/Eye candy, GAIA Place/Delete/Scatter):
        _gaia_membership_diff(), with its caps and state/order checks. Every
        other whole-list snapshot (Paste Region, its undo and redo, non-GAIA
        Place and its undo): _tail_membership_diff(). Flat takes the second
        shape wholesale, as before."""
        children = record.children if isinstance(record, CompositeDiffRecord) else [record]
        unit_records = [c for c in children if isinstance(c, UnitDiffRecord)]
        if len(unit_records) != 1:
            return None
        current, target = unit_records[0].before, unit_records[0].after
        if undo:
            current, target = target, current
        players = getattr(current, "players", None)
        if players is None or set(players) != set(target.players):
            return None
        if set(players) == {GAIA_PLAYER_ID}:
            return self._gaia_membership_diff(players[GAIA_PLAYER_ID], target.players[GAIA_PLAYER_ID])
        if self._render_style == "flat":
            return None
        return self._tail_membership_diff(players, target.players)

    def _tail_membership_diff(self, now_players: dict, then_players: dict):
        """_membership_diff()'s paste shape: every snapshotted player's list is
        a pure tail append (`then == now + tail`) or a pure tail truncation, the
        units that stay keep their state, and no unit is on both sides (so a
        Convert of a list's last unit is refused). No cap: the cache picks
        splice, in-place or wholesale."""
        removals: list[UnitSplice] = []
        arriving: list[tuple[int, int, object]] = []
        for player_id, now in now_players.items():
            then = then_players[player_id]
            keep = min(len(now.units), len(then.units))
            if any(a is not b for a, b in zip(now.units[:keep], then.units[:keep], strict=True)):
                return None
            if now.states[:keep] != then.states[:keep]:
                return None
            for index in range(keep, len(now.units)):
                unit = now.units[index]
                own, tiles = self._unit_footprint(unit)
                removals.append(UnitSplice(player_id, index, unit, own, None, tiles, ()))
            arriving.extend((player_id, index, then.units[index]) for index in range(keep, len(then.units)))
        if {id(s.unit) for s in removals} & {id(unit) for _p, _i, unit in arriving}:
            return None
        return removals, arriving

    def _gaia_membership_diff(self, now, then):
        """_membership_diff()'s GAIA-only shape, over the GAIA list's two
        snapshots. None unless every unit on both sides keeps an identical
        state and its relative order, and on Flat the change is within
        UNIT_SPLICE_MAX_UNITS. Stepped/Sloped have no cap here (TASK-031.61
        A3): past the caps the cache splices the visible level alone up to
        VISIBLE_SPLICE_MAX_UNITS (TASK-031.61 c2x), else refuses with `cap` and
        updates units_by_tile in place or rebuilds it; the repaint stays scoped."""
        capped = self._render_style == "flat"
        if capped and abs(len(now.units) - len(then.units)) > render_cache_splice.UNIT_SPLICE_MAX_UNITS:
            return None
        now_states = {id(u): state for u, state in zip(now.units, now.states, strict=True)}
        then_ids: set[int] = set()
        arriving = []
        for index, (unit, state) in enumerate(zip(then.units, then.states, strict=True)):
            then_ids.add(id(unit))
            known = now_states.get(id(unit))
            if known is None:
                arriving.append((GAIA_PLAYER_ID, index, unit))
            elif known != state:
                return None
        leaving = [(i, u) for i, u in enumerate(now.units) if id(u) not in then_ids]
        if capped and len(leaving) + len(arriving) > render_cache_splice.UNIT_SPLICE_MAX_UNITS:
            return None
        # Flat's rows follow list order, and a splice cannot reorder the units that stay.
        staying_now = [id(u) for u in now.units if id(u) in then_ids]
        if staying_now != [id(u) for u in then.units if id(u) in now_states]:
            return None
        removals = []
        for index, unit in leaving:
            own, tiles = self._unit_footprint(unit)
            removals.append(UnitSplice(GAIA_PLAYER_ID, index, unit, own, None, tiles, ()))
        return removals, arriving

    def _refresh_after_history_move(
        self, kinds, dirty, splices=None, batch: bool = False, elev_changed: set | None = None
    ) -> None:
        """Everything the window has to put back in step once the history
        cursor has moved and the models have been restored -- shared by
        undo/redo (_move_history above) and by the History window's multi-step
        jump (_on_history_jump), so the two cannot drift apart.

        Branches on `kinds` rather than on a record: a trigger record needs
        the trigger panel rebuilt and yields no tile indices, a tile record
        needs the incremental repaint, either a trigger or an option record
        needs the Map Options form put back in step with the models, and a
        unit record needs the render caches invalidated and the
        selection/inspector reconciled (_after_unit_mutation()).

        _update_title() runs here rather than in _apply_dirty(), which
        early-returns on an empty index list: since a trigger undo produces no
        dirty tiles, the window's "*" marker would otherwise never update on
        one. Every other _apply_dirty() caller already updates the title
        itself.

        `splices` is Batch D's D6 scoped unit path: a list of UnitSplice for a
        fields_only record or a membership change (_membership_diff()), or None
        for the wholesale invalidation. A jump
        always passes None -- its span reads each record's old footprints
        immediately before that record moves, which a one-shot multi-record
        move cannot do. `batch` marks the membership case for
        _after_unit_mutation().

        `elev_changed` is E, the tiles the record's tile child changed the
        elevation of (a jump passes None). A unit record with a non-empty E on
        Stepped/Sloped (the paste composite, Mirror Map) runs its unit sources
        BEFORE the tile patch, as Paste Region does: the patch then reads
        units_by_tile that agrees with the units (_flush_pending), and a gen
        bump lands before its one level build rather than throwing it away.
        The level splice is skipped when the patch will bump the gen anyway
        (_elevation_bumps_anyway()). This skips _patch_unit_edit_cache() and
        its ordering constraint; _after_unit_mutation(changed=None) rebuilds
        the pick index in Units mode (and rebuilds or drops it elsewhere).
        Everything else keeps the tiles-first order below, where a tile patch
        that took a full branch (_apply_dirty() returns True) runs the unit
        step wholesale: a re-rendered cache was built from the restored units,
        so the splices would insert an add twice (TASK-031.61 A2b), and A5's
        in-place refresh rebuilds Sloped's unit layers from them too. The
        sources-first branch needs no such rule: its splices run on the
        pre-move sources, before any full branch.

        On the tiles-first branch, Stepped/Sloped splices that touch a tile
        composite the terrain once (TASK-031.61 A4): the tile patch runs only
        its source half (_apply_dirty()'s `deferred`) and the unit step
        repaints the union of the terrain bbox and its own. Patching the
        terrain first would recomposite chunks the unit step then evicts.
        `dirty` goes with the bbox as terrain_tiles (TASK-031.61 C1), so
        Stepped's per-chunk rects can size the terrain by block of tiles.

        terrain_current: that scoped unit step is told the terrain mirrors
        are current (_after_unit_mutation()). Every tile the move restored is
        in `dirty`, and _apply_dirty() re-read the mirrors over it on each
        branch that leaves the splices standing (patch(), A4's
        patch_sources(), or top-down Flat's patch_rects() and its eviction,
        which keeps terrain_changed); a full branch runs the unit step
        wholesale. Not inside an open Erase stroke, whose Terrain Overlay
        writes are not in the mirrors until it ends.
        """
        # Undo/redo rewrites what the armed field and its form show.
        self.disarm_unit_picker()
        sources_first = (
            "unit" in kinds and bool(elev_changed) and self._cache is not None and self._render_style != "flat"
        )
        if sources_first:
            self._cancel_warms()
            with perf_trace.phase("elev_predict"):
                bumps = self._elevation_bumps_anyway(elev_changed)
            with perf_trace.phase("unit_sources"):
                self._cache.invalidate_units(
                    splices + self._flag_flip_splices(splices) if splices else None, splice_levels=not bumps
                )
            self._apply_dirty(dirty)
            self._after_unit_mutation(sources_refreshed=True)
        else:
            # A4: Stepped/Sloped splices that touch a tile composite the terrain bbox with their own.
            deferred = (
                []
                if "unit" in kinds and self._cache is not None and self._render_style != "flat"
                and splices and any(s.old_tiles or s.new_tiles for s in splices)
                else None
            )
            full = self._apply_dirty(dirty, deferred=deferred)
        if "unit" in kinds and not sources_first:
            # A full branch rebuilt the cache (or Sloped's unit layers) from the restored units: no splicing.
            if splices is not None and not full:
                # _after_unit_mutation() deliberately leaves the pick index
                # alone whenever `changed` is given (its own docstring),
                # trusting the call site to have handled it already -- here
                # that's this method, standing in for whichever tool call
                # site originally built the record. A full rebuild rather
                # than a move/add-shaped patch: this path is undo/redo, not
                # the interactive drag D6 exists to make cheap, and it is
                # the exact rebuild the pre-D6 wholesale branch always paid
                # for anyway.
                if self.mode == "units":
                    with perf_trace.phase("unit_index"):
                        self._rebuild_unit_index()
                # _apply_dirty() above re-read the mirrors over `dirty`; an open Erase stroke's writes are not in them.
                self._after_unit_mutation(
                    splices, batch=batch, terrain_bbox=deferred[0] if deferred else None,
                    terrain_current=self._erase is None, terrain_tiles=dirty if deferred else None,
                )
            else:
                self._after_unit_mutation()
        if "trigger" in kinds and self.mode == "triggers":
            # Rebuilt wholesale rather than patched: the panel is a read-only
            # view over the parsed manager, and an undo can change trigger
            # membership, order and ids at once.
            self._show_triggers()
        if "trigger" in kinds:
            self.trigger_panel.refresh_variables()
            # Mode-free: the overlay persists into View, where the panel is not refreshed.
            self._rederive_trigger_overlay()
            # Mode-free too, so a trigger undo from any mode reaches Find (GH #144).
            self._find_triggers_bumped()
            self._invalidate_trigger_ranges()
        if kinds & {"options", "trigger"}:
            # Both kinds, not just "options": the exec-order row is shown in
            # the Map Options panel but recorded as a trigger edit, so undoing
            # one produces a "trigger" record while the user is looking at this
            # form. A no-op outside the mode.
            self._repopulate_map_options()
        if "options" in kinds:
            # A Players mode field, and a Diplomacy grid cell, both ride an
            # OptionsDiffRecord too (decision 1), so their own undo/redo
            # must refresh those panels the same way -- a no-op outside
            # their own mode, same as the call above.
            self._repopulate_players()
            self._repopulate_diplomacy()
            # Runs after the caller's move has applied the undo/redo to
            # option_edits, so the recompute reads the post-move pending values.
            self._after_player_color_change()
            # An undone Number of Players edit changes the stats combo's list, and Swap's gate.
            self._repopulate_stats_players()
            self._regate_swap()
            self._update_technical_info()
            self._refresh_colour_mood_preview()
        elif "trigger" in kinds:
            self._update_player_stats()
        if "messages" in kinds:
            self._repopulate_messages()
        self._update_title()
        self._update_edit_actions()
