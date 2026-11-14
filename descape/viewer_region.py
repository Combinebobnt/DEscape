"""ViewerWindow's region tools: region select, the clipboard and history dialogs, paste, move, lift and
transform, fill, the eyedropper and mirror."""

from __future__ import annotations

import dataclasses
import math
import time
from types import SimpleNamespace
from typing import NamedTuple

from AoE2ScenarioParser.exceptions.asp_exceptions import UnsupportedAttributeError
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QMessageBox,
)

from descape import (
    clipboard_history,
    debug_log,
    perf_trace,
    placement_rules,
    region_clipboard,
    region_transform,
    terrain_target,
    unit_kind,
    unit_pick,
    unit_rotation,
    unit_variant,
    wall_run,
)
from descape.clipboard_dialog import ClipboardHistoryDialog
from descape.edit_history import (
    CompositeDiffRecord,
    HistoryMoveFailed,
    TileDiffRecord,
)
from descape.elevation_tools import set_tiles_elevation
from descape.fill_tools import fill_indices
from descape.history_dialog import EditHistoryDialog
from descape.mirror_tools import (
    OVERLAPS_SKIP,
    compose,
    walls_are_radian,
)
from descape.render_cache_splice import UnitSplice
from descape.render_units import NON_BUILDING_SPAN
from descape.scatter_dialog import units_region
from descape.terrain_palette import name_for_terrain_id, tile_span
from descape.unit_filter import GAIA_PLAYER_ID, MAX_PLAYER_ID
from descape.unit_model import (
    capture_flag_of,
    span_low_corner,
)
from descape.viewer_common import (
    _STROKE_LABELS,
    ELEVATION_LEVEL_MAX,
    TERRAIN_UNIT_CONFIRM_THRESHOLD,
    _reshaping_clause,
    _unit_name,
    tool_applicable,
)


def clamp_paste_anchor(block, tx0: int, ty0: int, map_w: int, map_h: int) -> tuple[int, int]:
    """Paste Units' anchor (GH #192) shifted by whole tiles until every
    top-level unit's footprint is on the map, measured as clamp_group_delta()
    measures one. Unlike a move, a footprint overhanging an edge is pulled
    back in. Occupants ride with their host and do not limit the shift.
    Unchanged when no shift fits the group (it is larger than the map)."""
    lo_x = lo_y = -math.inf
    hi_x = hi_y = math.inf
    for unit in block.units:
        if unit.garrison_slot != -1:
            continue
        span_x, span_y = tile_span(unit.unit_const, NON_BUILDING_SPAN)
        placed = SimpleNamespace(unit_const=unit.unit_const, x=tx0 + unit.dx, y=ty0 + unit.dy)
        low_x, low_y = span_low_corner(placed)
        # span_low_corner() truncates a span-1 axis toward zero, so (-1, 0) would read as tile 0.
        if span_x <= 1:
            low_x = math.floor(placed.x)
        if span_y <= 1:
            low_y = math.floor(placed.y)
        lo_x, hi_x = max(lo_x, -low_x), min(hi_x, map_w - (low_x + span_x))
        lo_y, hi_y = max(lo_y, -low_y), min(hi_y, map_h - (low_y + span_y))
    if lo_x > hi_x or lo_y > hi_y:
        return tx0, ty0
    return tx0 + min(max(0, lo_x), hi_x), ty0 + min(max(0, lo_y), hi_y)


def _units_text(count: int, carried: int) -> str:
    """"1 unit" or "N units", plus how many more ride inside them as garrisons."""
    text = "1 unit" if count == 1 else f"{count} units"
    return f"{text} ({carried} garrisoned)" if carried else text


# The D4 reflections: a lift whose composed element is one of these is a flip, else a turn (TASK-256).
_REFLECTIONS = frozenset({"mx", "my", "d", "a"})

# Paste's status line when the file's units cannot be edited (Paste Region and Paste Units).
_PASTE_UNITS_SKIPPED = "Units were skipped -- units are read-only for this file"


class _PasteOutcome(NamedTuple):
    """_paste_block_at()'s result. `record` is None when nothing changed;
    `added` the (player_id, unit) pairs created, in paste order; `reshaped`
    how many existing walls the paste reshaped (TASK-585)."""

    record: object
    parts: list
    units_skipped: bool
    added: list
    reshaped: int


class _ViewerRegionMixin:
    """ViewerWindow's region select, clipboard and transforms (viewer.py's docstring has the split's rules)."""

    # -- Region select/copy/paste (phase 2.8) -----------------------------

    def on_region_selected(self, region: tuple[int, int, int, int] | None) -> None:
        """The one handler for every way the committed region changes: a
        completed Select drag or an Escape-clear (both reported by MapView
        via the callback it was constructed with), and this window's own
        Select All/Deselect actions, which call it directly. Keeps
        self._region and MapView's own rendering copy in sync from either
        direction -- see MapView.set_region()'s docstring for the split."""
        self._region = region
        # Cleared unconditionally: this is the single sync point for every
        # committed-region change, and a selection that is no longer the
        # paste's own is not movable. paste_region()/on_region_move() re-arm
        # AFTER calling this, which is why their arming comes last.
        self._set_paste_move(None)
        self.map_view.set_region(region)
        self._update_tool_enabled()
        if self._find_dialog is not None:
            self._find_dialog.region_changed(region)

    def _on_region_gesture(self, region: tuple[int, int, int, int] | None) -> None:
        """MapView's region callback (a Select drag, or its Escape-clear): a
        user's new region ends a lift (TASK-256), which a lift's own
        on_region_selected() call must not."""
        self.on_lift_finalize()
        self.on_region_selected(region)

    def select_all(self) -> None:
        if self.scenario is None:
            return
        self.on_lift_finalize()
        mm = self.scenario.map_manager
        region = (0, 0, mm.map_width, mm.map_height)
        self.on_region_selected(region)
        self._log_status(f"Selected {mm.map_width}x{mm.map_height} region (whole map)")

    def deselect(self) -> None:
        if self.scenario is None:
            return
        self.on_lift_finalize()
        self.on_region_selected(None)
        self._log_status("Deselected")

    def _units_select_active(self) -> bool:
        """Whether Select All, Deselect and Escape act on units (TASK-325)."""
        return self.mode == "units" and self._current_tool == "select"

    def select_all_units(self) -> None:
        """Replaces the unit selection with every unit the filters show,
        leaving the region alone (TASK-325)."""
        index = self.map_view._unit_index
        if self.scenario is None or index is None:
            return
        self.on_lift_finalize()
        mm = self.scenario.map_manager
        width, height = mm.map_width, mm.map_height
        # Same off-map skip as units_in_screen_rect: a moved-off entry isn't drawn.
        entries = [e for e in index.entries if 0 <= e.own_x < width and 0 <= e.own_y < height]
        self._selection = [unit_pick.unit_key(e.player_id, e.unit) for e in entries]
        self._stack_cycle = None
        self._refresh_selection_view()
        if not entries:
            self._log_status("Select All: no units shown")
        elif len(entries) == 1:
            unit = entries[0].unit
            self._log_status(f"Selected {_unit_name(unit.unit_const)} at ({unit.x:g}, {unit.y:g})")
        else:
            self._log_status(f"{len(entries)} units selected (all shown)")

    def deselect_units(self) -> bool:
        """Clears the unit selection; False when there was none (TASK-325)."""
        if not self._selection:
            return False
        self.on_lift_finalize()
        self._selection = []
        self._stack_cycle = None
        self._refresh_selection_view()
        self._log_status("Selection cleared")
        return True

    def _units_escape(self) -> bool:
        """MapView's Escape hook: True when it cleared a unit selection."""
        return self._units_select_active() and self.deselect_units()

    @property
    def _region_clipboard(self):
        """The active clipboard entry's block, or None.

        Permanent, not a migration shim: every reader of this name -- Paste
        Region, _update_tool_enabled()'s gate, tools/verify_copy_paste.py,
        and three test modules -- means exactly "the active clipboard block",
        so the property is semantically honest. Read-only on purpose: with no
        setter, any leftover assignment raises AttributeError loudly at
        construction rather than silently shadowing the history."""
        return self._clipboard_history.active_block

    def copy_region(self) -> None:
        with perf_trace.op("copy"):
            self._copy_region()

    def _copy_region(self) -> None:
        """Copy Region: snapshots the committed selection's terrain,
        elevation and units, and pushes it onto the clipboard history as the
        new active entry. No undo record -- nothing is mutated."""
        if self.scenario is None or self._region is None:
            return
        tx0, ty0, tx1, ty1 = self._region
        with perf_trace.phase("region_copy"):
            block = region_clipboard.copy_region(
                self.scenario.map_manager, self.scenario.unit_manager, tx0, ty0, tx1, ty1
            )
        self._push_clipboard(block)
        w, h = tx1 - tx0, ty1 - ty0
        self._log_status(
            f"Copied {w}x{h} region ({len(block.units)} units) "
            f"(entry {len(self._clipboard_history.entries)} of {clipboard_history.MAX_ENTRIES})"
        )

    def _push_clipboard(self, block) -> None:
        """Makes `block` the new active clipboard entry: the shared tail of
        Copy Region and Copy Units."""
        block = self._with_file_game(block)
        with perf_trace.phase("thumbnail"):
            thumbnail = clipboard_history.thumbnail_rgb(block)
        self._clipboard_history.push(block, thumbnail, clipboard_history.default_label(block))
        # Every history mutation clears the move state, with no exceptions to
        # remember: a move replays its OWN snapshot, so leaving it armed would
        # still be correct, but dragging after the user has shifted clipboard
        # intent would move the PREVIOUSLY pasted content.
        self._set_paste_move(None)
        self._update_tool_enabled()
        self._refresh_clipboard_dialog()

    def copy_units(self, entries=None, quiet: bool = False) -> bool:
        """Copy Units (GH #192): pushes the selected units, and what they
        hold, as a clipboard entry without tiles. `entries` defaults to the
        Units-mode selection. No undo record: nothing is mutated. False when
        there was nothing to copy."""
        if self.scenario is None:
            return False
        if entries is None:
            entries = self._selected_unit_entries()
        with perf_trace.op("copy"):
            with perf_trace.phase("region_copy"):
                block = region_clipboard.copy_units(self.scenario.unit_manager, [e.unit for e in entries])
            if block is None:
                return False
            self._push_clipboard(block)
        if not quiet:
            carried = sum(1 for u in block.units if u.garrison_slot != -1)
            self._log_status(
                f"Copied {_units_text(len(block.units) - carried, carried)} "
                f"(entry {len(self._clipboard_history.entries)} of {clipboard_history.MAX_ENTRIES})"
            )
        return True

    def _clipboard_rows(self):
        return [
            (e.entry_id, e.label, e.block.width, e.block.height, len(e.block.units), e.thumbnail)
            for e in self._clipboard_history.entries
        ]

    def _show_clipboard_history(self) -> None:
        if self._clipboard_dialog is None:
            self._clipboard_dialog = ClipboardHistoryDialog(
                self,
                on_activate=self._on_clipboard_activate,
                on_delete=self._on_clipboard_delete,
                on_clear=self._on_clipboard_clear,
                on_rename=self._on_clipboard_rename,
            )
        self._refresh_clipboard_dialog()
        self._clipboard_dialog.show()
        self._clipboard_dialog.raise_()
        self._clipboard_dialog.activateWindow()

    def _refresh_clipboard_dialog(self) -> None:
        if self._clipboard_dialog is None:
            return
        self._clipboard_dialog.set_entries(
            self._clipboard_rows(), self._clipboard_history.active_id
        )

    def _history_rows(self):
        """One row per DiffRecord, oldest first. id(record) as the row id:
        the History window restores its selection by id because _push()'s
        redo truncation and overflow trim both shift every later index."""
        return [
            (id(record), record.label, ", ".join(sorted(record.kinds())))
            for record in self.edit_history.records
        ]

    def _show_history_dialog(self) -> None:
        if self._history_dialog is None:
            self._history_dialog = EditHistoryDialog(self, on_jump=self._on_history_jump)
        # show() before the refresh, not after: the refresh below deliberately
        # skips a hidden window, and Close only hides this one.
        self._history_dialog.show()
        self._history_dialog.raise_()
        self._history_dialog.activateWindow()
        self._refresh_history_dialog()

    def _refresh_history_dialog(self) -> None:
        """EditHistory.on_change's target -- fires on every history mutation,
        including during load and close, so it no-ops until the window exists
        and again once the user has closed it (close() only hides a QDialog,
        and rebuilding a few hundred hidden rows per stroke is pure waste)."""
        if self._history_dialog is None or not self._history_dialog.isVisible():
            return
        self._history_dialog.set_rows(
            self._history_rows(), self.edit_history.cursor, self.edit_history.saved_at_cursor
        )

    def _on_history_jump(self, target: int) -> None:
        with perf_trace.op("history"):
            self._history_jump(target)
        # As _move_history(): the Cliff hover ghost reads the map's cliffs.
        self.map_view.refresh_cliff_ghost()

    def _history_jump(self, target: int) -> None:
        """Undo or redo the whole span between the cursor and `target` as one
        gesture, then run undo/redo's own refresh tail.

        Refused rather than raised on a bad span: this is reached from a
        double-click inside a non-modal dialog, and a RuntimeError out of
        require_target() (a record whose model is gone) or a ValueError out of
        a stale row would otherwise surface as a QMessageBox behind the
        dialog. A record whose own move raises (HistoryMoveFailed) stops the
        jump there: what moved is refreshed wholesale and the stop is logged."""
        if self.scenario is None:
            self._log_status("Jump: no scenario is open")
            return
        if self._busy or self._stroke_in_progress():
            self._log_status("Jump: an edit is still in progress")
            return
        start = self.edit_history.cursor
        collapse = self.trigger_panel.collapse_snapshot()
        try:
            kinds = self.edit_history.span_kinds(target)
            dirty = self.edit_history.jump_to(
                target,
                self.scenario.map_manager.terrain,
                self.trigger_edits,
                self.option_edits,
                self.unit_edits,
                self.message_edits,
            )
        except HistoryMoveFailed as exc:
            if "trigger" in exc.kinds:
                self.trigger_panel.follow_collapse(collapse)
            self._refresh_after_history_move(exc.kinds, exc.touched)
            self._log_status(f"Jump stopped after {exc.done} of {exc.total} steps: {exc}", level=debug_log.WARNING)
            return
        except (RuntimeError, ValueError) as exc:
            self._log_status(f"Jump refused: {exc}", level=debug_log.WARNING)
            return
        if target == start:
            self._log_status("Already at that history entry")
            return
        if "trigger" in kinds:
            self.trigger_panel.follow_collapse(collapse)
        self._refresh_after_history_move(kinds, dirty)
        steps = abs(target - start)
        direction = "back" if target < start else "forward"
        label = self.edit_history.records[target - 1].label if target > 0 else "Opened file"
        plural = "" if steps == 1 else "s"
        self._log_status(f"Jumped {direction} {steps} step{plural} to: {label}")

    def _on_clipboard_activate(self, entry_id: int) -> None:
        if not self._clipboard_history.set_active(entry_id):
            return
        self._set_paste_move(None)
        self._update_tool_enabled()
        self._refresh_clipboard_dialog()
        entry = self._clipboard_history.active
        self._log_status(f"Clipboard entry active: {entry.label}")

    def _on_clipboard_delete(self, entry_id: int) -> None:
        if not self._clipboard_history.remove(entry_id):
            return
        self._set_paste_move(None)
        self._update_tool_enabled()
        self._refresh_clipboard_dialog()
        self._log_status("Deleted clipboard entry")

    def _on_clipboard_clear(self) -> None:
        if not self._clipboard_history.entries:
            return
        self._clipboard_history.clear()
        self._set_paste_move(None)
        self._update_tool_enabled()
        self._refresh_clipboard_dialog()
        self._log_status("Cleared the clipboard history")

    def _on_clipboard_rename(self, entry_id: int, label: str) -> None:
        if not self._clipboard_history.rename(entry_id, label):
            return
        self._set_paste_move(None)
        self._refresh_clipboard_dialog()
        self._log_status(f"Renamed clipboard entry to {label}")

    def paste_region(self) -> None:
        """Paste Region: writes the clipboard's checked categories anchored
        on the hovered tile (falling back to the current region's own
        top-left with no hover), as ONE undo step regardless of how many
        categories are checked -- see edit_history.CompositeDiffRecord.
        Terrain+elevation ride a single TileDiffRecord (they share
        TileState); units ride a separate UnitDiffRecord built with
        push=False so it doesn't land on the stack until it's known how many
        other children there are."""
        block = self._clipboard_history.active_block
        if self.scenario is None or block is None:
            return
        if refusal := self._cross_game_refusal(block):
            self._log_status(f"Paste refused: {refusal}", level=debug_log.WARNING)
            return
        # A coalesced move still pending would leave _hover_tile one tile behind the cursor.
        self.map_view.flush_pending_move()
        if not block.has_tiles:
            self._paste_units(block)
            return
        if self._hover_tile is not None:
            tx0, ty0 = self._hover_tile
        elif self._region is not None:
            tx0, ty0 = self._region[0], self._region[1]
        else:
            self._log_status("Paste Region: no target tile -- hover the map or select a region first")
            return

        do_terrain = self.paste_terrain_check.isChecked()
        # Defensive re-check of the same gate the checkbox's own setEnabled()
        # already encodes (_update_tool_enabled): the checkbox could be
        # stale mid-call the same way the old tool-scoped gating's own
        # comment documented. set_tiles_elevation raises ValueError via
        # MapManager.get_tile on a non-square map. _paste_block_at() closes its
        # stroke on a raise, so it can't wedge later edits, but the paste would
        # still fail halfway.
        do_elevation = self.paste_elevation_check.isChecked() and self.scenario.map_is_square
        do_units = self.paste_units_check.isChecked()
        record, parts, units_skipped, _added, reshaped = self._paste_block_at(
            block, tx0, ty0, do_terrain, do_elevation, do_units
        )
        self.on_region_selected(self._pasted_region(block, tx0, ty0))
        # Arming comes LAST, after on_region_selected(), which clears the
        # move state unconditionally as the single sync point for every
        # committed-region change. A paste that pushed nothing (no category
        # checked, or a destination already identical) arms nothing: there is
        # no record to replace and so nothing to move.
        if record is not None:
            self._set_paste_move(
                region_clipboard.PasteMove(
                    block, (tx0, ty0), do_terrain, do_elevation, do_units, record
                )
            )

        label = ", ".join(parts) if parts else "nothing (no category checked)"
        if record is None and parts:
            self._log_status(f"Pasted {label} at ({tx0}, {ty0}) (no change)")
        else:
            self._log_status(f"Pasted {label} at ({tx0}, {ty0}){_reshaping_clause(reshaped)}")
        if units_skipped:
            self._log_status(_PASTE_UNITS_SKIPPED, level=debug_log.WARNING)

    def _paste_units(self, block) -> None:
        """Paste Units (GH #192): a clipboard entry without tiles, centred on
        the hovered tile and pulled inside the map, as one undo step.

        The three paste-filter checkboxes are not read: they split a mixed
        block into categories and this one has a single category. The tile
        region and the paste-move state are left alone, and in Units mode the
        pasted top-level units become the selection."""
        mm = self.scenario.map_manager
        target = self._hover_tile
        if target is None and self._region is not None:
            target = self._region[0], self._region[1]
        # Flat reports the tile under the cursor even past the map's edge.
        if target is None or not (0 <= target[0] < mm.map_width and 0 <= target[1] < mm.map_height):
            self._log_status("Paste Units: no target tile, hover the map first")
            return
        centred = region_clipboard.centred_anchor(block, *target)
        tx0, ty0 = clamp_paste_anchor(block, *centred, mm.map_width, mm.map_height)
        _record, _parts, units_skipped, added, reshaped = self._paste_block_at(block, tx0, ty0, False, False, True)
        if units_skipped:
            self._log_status(_PASTE_UNITS_SKIPPED, level=debug_log.WARNING)
            return
        top_level = [(player_id, unit) for player_id, unit in added if unit.garrisoned_in_id == -1]
        hidden = 0
        if self.mode == "units" and top_level:
            self._selection = [unit_pick.unit_key(player_id, unit) for player_id, unit in top_level]
            # Drops a pasted unit the live filter hides, as it would a clicked one.
            self._refresh_selection_view()
            hidden = len(top_level) - len(self._selection)
        # Where the group's centre landed: the hovered tile unless the clamp shifted it.
        centre = tx0 + (block.width - 1) // 2, ty0 + (block.height - 1) // 2
        status = f"Pasted {_units_text(len(top_level), len(added) - len(top_level))} at ({centre[0]}, {centre[1]})"
        status += _reshaping_clause(reshaped)
        if len(added) < len(block.units):
            status += f", {len(block.units) - len(added)} off the map skipped"
        if hidden:
            # on_unit_place()'s clause: a paste the live filter hides looks like nothing happened.
            what = "it, so it" if len(top_level) == 1 else f"{hidden} of them, so they"
            status += f" (a Filters toggle is hiding {what} won't be visible)"
        self._log_status(status)

    def _set_paste_move(self, state) -> None:
        """The ONE assignment path for _paste_move, so MapView's movable flag
        is pushed from the same place and can never disagree with it. Rotate
        and Flip re-gate here too: a paste arms after on_region_selected()'s
        own _update_tool_enabled(), so nothing else would see it (GH #197)."""
        self._paste_move = state
        self.map_view.set_region_movable(state is not None)
        self._update_rotate_enabled()

    def _refresh_paste_move_validity(self) -> None:
        """One identity check, not a list of event hooks: the state is valid
        only while the paste's own record is still the next thing undo would
        take. That single condition already covers every way it should lapse --
        any other edit pushes a record on top, an undo or redo moves the cursor
        off it, and loading or closing a file resets the history entirely.

        Called from _update_edit_actions(), which is already the
        post-mutation and post-undo/redo sweep point. The two things this
        cannot see (a new selection, and a clipboard change) clear the state
        explicitly at their own call sites."""
        if self._paste_move is None:
            return
        if self.edit_history.peek_undo() is not self._paste_move.record:
            self._set_paste_move(None)

    def on_region_move(self, dx: int, dy: int) -> None:
        """Commit a region drag: undo the paste that armed this, then re-paste
        the same snapshot at the new anchor.

        Because the undo rewinds the cursor and the re-paste truncates at it
        (EditHistory._push opens with del records[cursor:]), the history ends
        with ONE record covering the paste and every move of it -- no pop, no
        in-place record mutation, no new coalescing API."""
        if self.scenario is None:
            return
        self._refresh_paste_move_validity()
        if self._paste_move is None:
            self._log_status("Move Region: the pasted region is no longer movable")
            return
        # Captured here, as this step's last act: _update_edit_actions() below
        # runs the validity sweep, which clears self._paste_move the instant
        # the undo moves peek_undo() off the paste record. Everything after
        # this line must read the local.
        state = self._paste_move

        self._move_history(self.edit_history.peek_undo(), self.edit_history.undo, "Undo", quiet=True)

        # The old location is restored BEFORE the new paste is written, so a
        # short drag (where the two footprints overlap heavily) reads restored
        # terrain when set_tiles_elevation propagates its skirt.
        tx0, ty0 = state.anchor[0] + dx, state.anchor[1] + dy
        block = state.block
        record, _parts, _units_skipped, _added, reshaped = self._paste_block_at(
            block, tx0, ty0, state.do_terrain, state.do_elevation, state.do_units
        )
        if record is None:
            # The destination already matched the block byte for byte. Leave
            # it un-armed rather than re-arming on nothing to move to. This
            # strands a redo entry: records[cursor - 1] still holds the old
            # paste, so Ctrl+Z here undoes whatever preceded that paste and
            # Ctrl+Y brings it back. Coherent, and accepted as-is.
            self._log_status(f"Moved pasted region to ({tx0}, {ty0}) (no change)")
            return
        self.on_region_selected(self._pasted_region(block, tx0, ty0))
        # A turned block carries its turn state along; the pivot moves with it.
        pivot2 = None if state.pivot2 is None else (state.pivot2[0] + 2 * dx, state.pivot2[1] + 2 * dy)
        self._set_paste_move(
            region_clipboard.PasteMove(
                block, (tx0, ty0), state.do_terrain, state.do_elevation, state.do_units, record,
                source=state.source, element=state.element, pivot2=pivot2,
            )
        )
        self._log_status(f"Moved pasted region to ({tx0}, {ty0}){_reshaping_clause(reshaped)}")

    def on_region_transform(self, step: str) -> None:
        """Turn or flip the just-pasted region as a block (GH #197): undo the
        paste that armed this, then re-paste the block transformed, which
        leaves one undo record for the paste and every turn and move of it, as
        on_region_move() does. `step` is a region_transform direction.

        Every turn re-derives from the block as first pasted and the element
        accumulated since, about a pivot fixed at the first turn, so four
        quarter turns give back the very first paste. The paste's own category
        flags are replayed, not the checkboxes. The clipboard is untouched.
        The Rotate and Flip actions call this, and so does the wheel's rotate
        binding (GH #245, MapView.set_wheel_paste_hooks), one call per notch.
        A handle (TASK-255) would call this too."""
        verb = "Rotate" if step in (region_transform.ROTATE_CW, region_transform.ROTATE_CCW) else "Flip"
        if self.scenario is None or self._stroke_in_progress() or self.map_view.region_drag_live():
            return
        self._refresh_paste_move_validity()
        if self._paste_move is None:
            self._log_status(f"{verb}: paste a region first, then turn it")
            return
        # A local, as in on_region_move(): the undo below lapses self._paste_move.
        state = self._paste_move
        element = compose(step, state.element)
        source = state.source or state.block
        pivot2 = state.pivot2 or region_transform.pivot2_of(state.anchor, state.block)
        units = [unit for player_units in self.scenario.unit_manager.units for unit in player_units]
        file_radian = walls_are_radian([*units, *source.units])
        result = region_transform.transform_block(source, element, file_radian)
        block = result.block
        tx0, ty0 = region_transform.anchor_for(pivot2, block)
        mm = self.scenario.map_manager
        # Checked before any undo, so a refusal leaves the paste as it was and still armed.
        if not region_transform.lands_on_map(block, (tx0, ty0), mm.map_width, mm.map_height):
            self._log_status(f"{verb}: the turned block would land off the map, move it inward first")
            return

        self._move_history(self.edit_history.peek_undo(), self.edit_history.undo, "Undo", quiet=True)
        record, _parts, _units_skipped, _added, reshaped = self._paste_block_at(
            block, tx0, ty0, state.do_terrain, state.do_elevation, state.do_units
        )
        if record is None:
            # Nothing landed (say a units-only paste turned wholly off the map):
            # bring the paste back rather than let a turn make it vanish.
            self._move_history(self.edit_history.peek_redo(), self.edit_history.redo, "Redo", quiet=True)
            self._set_paste_move(state)
            self._log_status(f"{verb}: nothing would land there, left as it was")
            return
        self.on_region_selected(self._pasted_region(block, tx0, ty0))
        self._set_paste_move(
            region_clipboard.PasteMove(
                block, (tx0, ty0), state.do_terrain, state.do_elevation, state.do_units, record,
                source=source, element=element, pivot2=pivot2,
            )
        )
        status = {
            region_transform.ROTATE_CW: "Rotated pasted region ↻",
            region_transform.ROTATE_CCW: "Rotated pasted region ↺",
            region_transform.FLIP_WEST_NORTH: "Flipped pasted region West ↔ North",
            region_transform.FLIP_WEST_SOUTH: "Flipped pasted region West ↔ South",
        }[step]
        status += _reshaping_clause(reshaped)
        if result.kept_orientation:
            status += f"; {result.kept_orientation} objects kept their orientation (cliffs, bridges)"
        self._log_status(status)

    # -- Flip and Rotate Group in place (TASK-256) ---------------------------

    def on_selection_transform(self, step: str) -> None:
        """Edit > Flip Selection's pair and Rotate Selection's Rotate Group
        pair. `step` is a region_transform direction, applied to the first of
        _transform_target()'s targets: an armed paste (GH #197's re-paste,
        on_region_transform()), a Units-mode selection as one group about its
        shared centre, or a committed region in place. The two in-place
        targets keep every unit's identity, and a run of steps on one of them
        is one undo record (_lift_step()). While Place Unit's or the Unit
        Brush's preview is showing they leave everything alone: Rotate turns
        the preview there."""
        verb = "Flip" if step in (region_transform.FLIP_WEST_NORTH, region_transform.FLIP_WEST_SOUTH) else "Rotate Group"
        if self.scenario is None or self._stroke_in_progress() or self.map_view.region_drag_live():
            return
        if self._place_ghost_active():
            self._log_status(f"{verb}: nothing changed while placing (Rotate turns the placement preview)")
            return
        target = self._transform_target()
        if target == "paste":
            self.on_region_transform(step)
        elif target == "units":
            state = self._live_lift("units") or self._units_lift(verb)
            if state is not None:
                self._lift_step(state, step, verb)
        elif target == "region":
            self._transform_region(step, verb)
        else:
            self._log_status(f"{verb}: select units (Units mode) or a region first")

    def _transform_target(self) -> str | None:
        """What Flip and Rotate Group act on, first match wins: "paste" (an
        armed paste), "units" (a Units-mode selection), "region" (a committed
        region), else None, and nothing in Triggers mode. A pure read. Place
        Unit's preview, which they leave alone, is checked when they run, so
        enablement does not follow the hover (_update_rotate_enabled())."""
        if self.scenario is None or self.mode == "triggers":
            return None
        if self._paste_armed():
            return "paste"
        if self.mode == "units" and self._selection:
            return "units"
        return "region" if self._region is not None else None

    def _region_rotate_target(self) -> bool:
        """Whether the Rotate entries turn the committed region in place
        (TASK-256): one is selected and no paste is armed (that is
        _region_turn_target()'s), outside Triggers mode, and Units mode has no
        selection or Place target, which keep today's Rotate. A pure read; the
        wheel's rotate binding never asks it, so a notch never turns terrain."""
        if self.scenario is None or self._region is None or self.mode == "triggers" or self._paste_armed():
            return False
        return not (self.mode == "units" and (self._selection or self._place_rotate_armed()))

    def _region_write_refusal(self, verb: str) -> str | None:
        """Why the committed region cannot be turned in place, or None. Its
        terrain and elevation are written together, and elevation writes need
        a square map, so a non-square one refuses the whole turn rather than
        tear terrain from elevation. Paste has no callable gate of its own."""
        reason = self.game_data_lock_reason()
        if reason is not None:
            return f"{verb}: {reason}"
        if not self.scenario.terrain_write_supported:
            return f"{verb}: this file's terrain cannot be edited, so its region cannot turn"
        if not self.scenario.map_is_square:
            return f"{verb}: a region turns with its elevation, which needs a square map"
        return None

    def _transform_region(self, step: str, verb: str) -> None:
        """The committed region flipped or turned in place: its terrain,
        elevation and the units Copy Region would capture, which keep their
        identity. A non-square quarter turn writes the turned rectangle about
        the same centre and leaves the tiles it no longer covers as they were."""
        state = self._live_lift("region") or self._region_lift(verb)
        if state is not None:
            self._lift_step(state, step, verb)

    def _region_lift(self, verb: str):
        """A new lift for the committed region, not stored yet, or None
        (logged). Its members are the units whose anchor tile is in it (Copy
        Region's rule) plus what they hold; `tiles` is its tiles-only block."""
        refusal = self._region_write_refusal(verb)
        if refusal is not None:
            self._log_status(refusal, level=debug_log.WARNING)
            return None
        mm = self.scenario.map_manager
        tx0, ty0, tx1, ty1 = rect = self._region
        captured = [
            unit_pick.UnitEntry(player_id, unit, int(unit.x), int(unit.y), -1)
            for player_id, units in enumerate(self.scenario.unit_manager.units)
            for unit in units
            if tx0 <= int(unit.x) < tx1 and ty0 <= int(unit.y) < ty1
        ]
        rows = list(captured)
        if captured:
            model = self._ensure_unit_edits()
            if model is None:
                return None
            rows += self._garrison_occupant_entries(model, captured)
        members = tuple((entry.player_id, entry.unit) for entry in rows)
        tiles = dataclasses.replace(region_clipboard.copy_region(mm, self.scenario.unit_manager, *rect), units=())
        base = self.edit_history.peek_undo()
        return region_clipboard.Lift(
            "region", (), members, self._first_units(members), rect, self._walls_radian(), base, base,
            tiles=tiles, region=rect,
        )

    def _write_lift_region(self, model, state, plan, label: str):
        """The region half of _lift_step(): the turned tiles at the image
        rectangle through the paste's own tile writer, and the units through
        the same setters as a unit group, as one record. Not through
        _unit_edit(), which pushes its own record (begin/commit with
        push=False instead, then one CompositeDiffRecord, as paste does), and
        refreshed in paste's order: unit sources, then the tile patch.
        The units are written before the tiles, so a setter raising leaves
        no tile written (TASK-490). Returns the pushed record, or the lift's
        base when nothing changed."""
        unit_record = None
        if plan.images:
            model.begin_unit_edit(sorted({player_id for player_id, _unit in state.members}), fields_only=True)
            self._write_lift_units_or_restore(model, state, plan, label)
            unit_record = model.commit_unit_edit(label, self.edit_history, push=False)
        tile_record, _parts = self._write_block_tiles(plan.block, plan.rect[0], plan.rect[1], True, True, label=label)
        # Tiles first in the record, as before: a composite undoes its children in reverse.
        children = [record for record in (tile_record, unit_record) if record is not None]
        if not children:
            return state.base
        if len(children) == 1:
            pushed = children[0]
            if tile_record is not None:
                self.edit_history.push_tile_record(pushed)
            else:
                self.edit_history.push_unit_record(pushed)
        else:
            pushed = CompositeDiffRecord(label, children=children)
            self.edit_history.push_composite_record(pushed)
        dirty = pushed.touched_indices()
        if unit_record is None:
            self._apply_dirty(dirty)
        else:
            # Gate footprints change, so no splices: wholesale sources, then the tile patch.
            self._cancel_warms()
            if self._cache is not None:
                self._cache.invalidate_units()
            self._apply_dirty(dirty)
            self._after_unit_mutation(sources_refreshed=True)
        self._update_edit_actions()
        self._update_title()
        return pushed

    def _set_lift(self, state) -> None:
        """The ONE assignment path for _lift. Rotate and Flip re-gate here, as
        in _set_paste_move()."""
        self._lift = state
        self._update_rotate_enabled()

    def _live_lift(self, kind: str | None = None):
        """The lift while it still holds, else None: its record is the next
        thing undo takes (so any other edit, an undo or a load lapses it) and
        its target is the one it lifted, the same selection keys in Units mode
        or the same region outside Triggers. A pure read."""
        state = self._lift
        if state is None or (kind is not None and state.kind != kind):
            return None
        if self.edit_history.peek_undo() is not state.record:
            return None
        if state.kind == "units":
            held = self.mode == "units" and tuple(self._selection) == state.keys
        else:
            held = self.mode != "triggers" and self._region == state.region
        return state if held else None

    def on_lift_finalize(self) -> bool:
        """Ends the lift, keeping what it wrote: the document already holds it.
        True when one was live. Called from the gestures that pick a new
        selection (a click, a marquee, Deselect, a Select drag) and a mode
        switch, never from the refreshes a lift's own steps run."""
        live = self._live_lift() is not None
        if self._lift is not None:
            self._set_lift(None)
        return live

    def _apply_lift_key(self) -> bool:
        """MapView's Enter: finalizes a live lift, with a status line."""
        state = self._live_lift()
        if state is None:
            return False
        self.on_lift_finalize()
        if state.element == "id":
            self._log_status("Applied (no change)")
        else:
            self._log_status(f"Applied {'flip' if state.element in _REFLECTIONS else 'turn'}")
        return True

    def on_lift_revert(self) -> bool:
        """MapView's Escape while a lift is live: back to before its first
        step, keeping the selection, and a turned region's rectangle back to
        its first one, which the history does not track. True when it acted,
        so the next Escape does what it does without a lift."""
        state = self._live_lift()
        if state is None:
            return False
        if state.record is not state.base:
            self._move_history(self.edit_history.peek_undo(), self.edit_history.undo, "Undo", quiet=True)
        if state.kind == "region" and self._region != state.rect:
            self.on_region_selected(state.rect)
        self._set_lift(None)
        self._log_status("Reverted to before the first flip or turn")
        return True

    def _units_lift(self, verb: str):
        """A new lift for the Units-mode selection, not stored yet, or None
        (logged). Its members are the selected units plus what they hold,
        which rides along as in a move (_garrison_occupant_entries()); its
        rect is the footprint box of the members no other member holds, so an
        occupant stored away from its host cannot shift the pivot."""
        index = self.map_view._unit_index
        entries = [] if index is None else [e for e in (index.entry_for_key(k) for k in self._selection) if e is not None]
        if not entries:
            self._log_status(f"{verb}: no selected unit is shown")
            return None
        model = self._ensure_unit_edits()
        if model is None:
            return None
        rows = entries + self._garrison_occupant_entries(model, entries)
        members = tuple((entry.player_id, entry.unit) for entry in rows)
        first_units = self._first_units(members)
        mm = self.scenario.map_manager
        top = [unit for (_player_id, unit), first in zip(members, first_units, strict=True) if first.holder == -1]
        rect = units_region(top, mm.map_width, mm.map_height)
        if rect is None:
            self._log_status(f"{verb}: the selected units are off the map")
            return None
        base = self.edit_history.peek_undo()
        return region_clipboard.Lift(
            "units", tuple(self._selection), members, first_units, rect, self._walls_radian(), base, base
        )

    @staticmethod
    def _first_units(members) -> tuple:
        """region_transform.FirstUnit per (player_id, unit) member, keyed by
        its index. A holder resolves only to a reference_id exactly one member
        has, as copy_region() resolves garrison slots."""
        counts: dict[int, int] = {}
        for _player_id, unit in members:
            counts[unit.reference_id] = counts.get(unit.reference_id, 0) + 1
        slot = {unit.reference_id: i for i, (_player_id, unit) in enumerate(members) if counts[unit.reference_id] == 1}
        firsts = []
        for i, (_player_id, unit) in enumerate(members):
            inside = unit.garrisoned_in_id
            holder = -1 if inside in (-1, unit.reference_id) else slot.get(inside, -1)
            firsts.append(
                region_transform.FirstUnit(i, unit.unit_const, float(unit.x), float(unit.y), float(unit.rotation), holder)
            )
        return tuple(firsts)

    def _walls_radian(self) -> bool:
        return walls_are_radian([unit for units in self.scenario.unit_manager.units for unit in units])

    def _lift_step(self, state, step: str, verb: str) -> None:
        """One flip or turn of a lift (a live one, or a new one not stored
        yet). The composed element is planned from the first state and every
        refusal runs before any history move, so a refused step changes
        nothing and a refused first step creates no lift. Accepted, the lift's
        record is undone quietly (the document is then back at its first
        state), the plan is written as one new record, or none for the
        identity, and the lift is stored last, after any selection update. A
        setter raising here is a bug in the plan, so it propagates, but only
        once the step's unit writes are restored and the quietly undone record
        is redone: the document and history are as before the step (TASK-490)."""
        first_step = state is not self._lift
        element = compose(step, state.element)
        plan = region_transform.plan_in_place(
            state.first_units, element, state.rect, state.file_radian, tiles=state.tiles
        )
        refusal = self._lift_refusal(state, plan, verb)
        if refusal is not None:
            self._log_status(refusal, level=debug_log.WARNING)
            return
        model = self._ensure_unit_edits() if state.members else None
        if state.members and model is None:
            return
        undone = state.record is not state.base
        if undone:
            self._move_history(self.edit_history.peek_undo(), self.edit_history.undo, "Undo", quiet=True)
        label = f"{'Flip' if element in _REFLECTIONS else 'Turn'} {self._lift_noun(state)}"
        try:
            if element == "id":
                record = state.base
            elif state.kind == "units":
                players = sorted({player_id for player_id, _unit in state.members})
                with self._unit_edit(model, label, players, fields_only=True):
                    self._write_lift_units_or_restore(model, state, plan, label)
                record = self.edit_history.peek_undo()
            else:
                record = self._write_lift_region(model, state, plan, label)
        except Exception:
            # Nothing was pushed: put the lift's own record back on top, as before the step.
            if undone and self.edit_history.peek_redo() is state.record:
                self._move_history(state.record, self.edit_history.redo, "Redo", quiet=True)
                # The quiet undo lapsed the lift (_update_edit_actions()): it holds again now.
                self._set_lift(state)
            raise
        region = None
        if state.kind == "region":
            region = plan.rect
            self.on_region_selected(region)
        self._set_lift(dataclasses.replace(state, element=element, record=record, region=region))
        self._log_status(self._lift_status(state, step, plan, first_step))

    def _lift_noun(self, state) -> str:
        if state.kind == "region":
            return "region"
        selected = sum(1 for first in state.first_units if first.holder == -1)
        return _units_text(selected, len(state.first_units) - selected)

    def _lift_status(self, state, step: str, plan, first_step: bool) -> str:
        """"Flipped 12 units West ↔ North" or "Turned region ↻", with the
        objects that kept their orientation, and on a lift's first step how to
        end it."""
        status = {
            region_transform.ROTATE_CW: f"Turned {self._lift_noun(state)} ↻",
            region_transform.ROTATE_CCW: f"Turned {self._lift_noun(state)} ↺",
            region_transform.FLIP_WEST_NORTH: f"Flipped {self._lift_noun(state)} West ↔ North",
            region_transform.FLIP_WEST_SOUTH: f"Flipped {self._lift_noun(state)} West ↔ South",
        }[step]
        if plan.kept_orientation:
            status += f"; {plan.kept_orientation} objects kept their orientation (cliffs, bridges)"
        if first_step:
            status += " (Enter to apply, Esc to undo)"
        return status

    def _lift_refusal(self, state, plan, verb: str) -> str | None:
        """Why a planned step cannot be written, or None, read before any
        history move. Any image footprint off the map refuses, Free placement
        or not. Otherwise placement is all or nothing, as a group move's is
        (_group_move_refusal()), with the members never blocking each other,
        but checked per image const, since a turned gate's footprint is its
        new sibling's: the full rule for a unit group, overlap alone for a
        region, whose terrain turns with its units. Garrisoned units skip it,
        as move_refusal() does. A lift's own earlier step only ever moved the
        members, which the grid leaves out, so reading the document now is
        reading the first state. A region's turned rectangle must fit whole
        too: a clipped one would lose tiles."""
        mm = self.scenario.map_manager
        x0, y0, x1, y1 = plan.rect
        fits = state.kind != "region" or (x0 >= 0 and y0 >= 0 and x1 <= mm.map_width and y1 <= mm.map_height)
        if not fits or not all(
            region_transform.image_on_map(image.unit_const, image.x, image.y, mm.map_width, mm.map_height)
            for image in plan.images
        ):
            return f"{verb}: part of the selection would land off the map, move it inward first"
        if not plan.images or self._restrictions_lifted():
            return None
        grid = placement_rules.BlockerGrid(self.scenario, exclude=[unit for _player_id, unit in state.members])
        reasons = []
        checked = 0
        for image in plan.images:
            _player_id, unit = state.members[image.key]
            if unit.garrisoned_in_id != -1:
                continue
            checked += 1
            if state.kind == "units":
                reason = placement_rules.refusal(self.scenario, image.unit_const, image.x, image.y, grid=grid)
            else:
                found = grid.blocker(image.unit_const, image.x, image.y)
                reason = None if found is None else (
                    f"blocked by {_unit_name(found[1].unit_const)} "
                    f"({'GAIA' if found[0] == GAIA_PLAYER_ID else f'Player {found[0]}'}) "
                    f"at ({found[1].x:g}, {found[1].y:g})"
                )
            if reason is not None:
                reasons.append(reason)
        if not reasons:
            return None
        return (
            f"{verb}: {len(reasons)} of {checked} units can't go there (first: {reasons[0]}). "
            "Tick Free placement (Units mode, Place Unit or Move) to place them anyway."
        )

    @staticmethod
    def _write_lift_units(model, state, plan) -> None:
        """Each image onto its member, through the guarded setters only and in
        this order: a gate's orientation sibling (set_unit_const()), the
        position, then a facing (set_rotation()) or a wall's shape index
        (set_wall_variant()). Everything else about the unit stays."""
        for image in plan.images:
            _player_id, unit = state.members[image.key]
            if image.unit_const != unit.unit_const:
                model.set_unit_const(unit, image.unit_const)
            model.set_position(unit, image.x, image.y, unit.z)
            if image.rotation_write is not None:
                kind, value = image.rotation_write
                if kind == region_transform.WRITE_ANGLE:
                    model.set_rotation(unit, value)
                else:
                    model.set_wall_variant(unit, int(value))

    def _write_lift_units_or_restore(self, model, state, plan, label: str) -> None:
        """_write_lift_units() inside an open unit edit. On a raise the edit is
        closed unrecorded and its before state restored, since
        abort_unit_edit() keeps what the setters already wrote (TASK-490)."""
        try:
            self._write_lift_units(model, state, plan)
        except Exception:
            model.restore(model.commit_unit_edit(label, self.edit_history, push=False).before)
            raise

    def _pasted_region(self, block, tx0: int, ty0: int) -> tuple[int, int, int, int]:
        """Where the selection goes after a block lands at (tx0, ty0), so the
        user sees the result and a second paste is idempotent. Clamped to the
        map like _clipped_bounds() -- an off-map anchor or an oversized block
        would otherwise hand MapView a region reaching past the elevation
        grid's edge and crash in _tile_polygon."""
        mm = self.scenario.map_manager
        return (
            max(0, tx0),
            max(0, ty0),
            min(mm.map_width, tx0 + block.width),
            min(mm.map_height, ty0 + block.height),
        )

    def _paste_block_at(
        self, block, tx0: int, ty0: int, do_terrain: bool, do_elevation: bool, do_units: bool
    ):
        with perf_trace.op("paste"):
            return self._paste_block_at_traced(block, tx0, ty0, do_terrain, do_elevation, do_units)

    def _paste_block_at_traced(
        self, block, tx0: int, ty0: int, do_terrain: bool, do_elevation: bool, do_units: bool
    ):
        """The write half of Paste Region, with the anchor and the three
        category flags passed in rather than read from _hover_tile and the
        checkboxes -- so Paste Region and a region MOVE share exactly one
        paste write path.

        Returns a _PasteOutcome; its record is None when nothing changed,
        which is what tells the caller there is no record to replace and
        therefore nothing to move.

        Walls (TASK-585): each pasted wall takes the shape its new neighbours
        give it, and the existing walls beside a tile a pasted wall or gate
        newly covers re-derive (wall_run.scenario_paste_plan()), inside the
        unit edit, so undo, the paste move and the paste turn need nothing
        more. Nothing at the destination is removed.

        One undo step regardless of how many categories are checked: terrain
        and elevation ride a single TileDiffRecord (they share TileState),
        units ride a separate UnitDiffRecord built with push=False so it
        doesn't land on the stack until it's known how many other children
        there are -- see edit_history.CompositeDiffRecord.

        A block without tiles (GH #192's copied units) has only its unit
        half, whatever the caller's flags say, and its record is labelled
        Paste Units."""
        mm = self.scenario.map_manager
        children: list = []
        parts: list[str] = []
        units_skipped = False
        tile_record = None
        unit_splices: list[UnitSplice] | None = None
        added_units: list[tuple[int, object]] = []
        rewrites: list = []
        if not block.has_tiles:
            # paste_terrain() and elevation_targets() index the tile tuples and would raise.
            do_terrain = do_elevation = False
        unit_label = "Paste Region" if block.has_tiles else "Paste Units"

        if do_terrain or do_elevation:
            tile_record, tile_parts = self._write_block_tiles(block, tx0, ty0, do_terrain, do_elevation)
            parts.extend(tile_parts)
            if tile_record is not None:
                children.append(tile_record)

        if do_units and block.units:
            unit_edits = self._ensure_unit_edits()
            if unit_edits is None:
                units_skipped = True
            else:
                targets = region_clipboard.unit_paste_targets(block, tx0, ty0, mm.map_width, mm.map_height)
                if targets:
                    # TASK-585: before the edit opens, so every rewritten wall's owner is snapshotted.
                    wall_plan = wall_run.scenario_paste_plan(self.scenario, mm.map_width, mm.map_height, targets)
                    rewrites = wall_plan.rewrites if wall_plan else []
                    owners = {int(t.unit.player) for t in targets} | {int(p) for p, _u, _v in rewrites}
                    unit_edits.begin_unit_edit(owners)
                    new_ids: dict[int, int] = {}
                    by_slot: dict[int, object] = {}
                    units = self.scenario.unit_manager.units
                    unit_splices = []
                    with perf_trace.phase("paste_units"), unit_edits.batch_adds():
                        for t in targets:
                            u = t.unit
                            added = unit_edits.add(
                                player=u.player,
                                unit_const=u.unit_const,
                                x=t.x,
                                y=t.y,
                                z=u.z,
                                rotation=u.rotation,  # verbatim from the block, which may be a turned one
                                status=u.status,
                                initial_animation_frame=u.initial_animation_frame,
                                garrisoned_in_id=new_ids.get(t.holder_slot, -1),
                                caption_string_id=u.caption_string_id,
                                caption_string=u.caption_string,
                                capture_flag=u.capture_flag,
                            )
                            new_ids[t.slot] = added.reference_id
                            by_slot[t.slot] = added
                            # add() appends, so the index is the tail (never _unit_list_index's scan).
                            player_id = int(u.player)
                            own, tiles = self._unit_footprint(added)
                            unit_splices.append(
                                UnitSplice(player_id, len(units[player_id]) - 1, added, None, own, (), tiles)
                            )
                            added_units.append((player_id, added))
                    if wall_plan:
                        wall_run.apply_rewrites(unit_edits, rewrites)
                        for slot, index in wall_plan.shapes.items():
                            unit_edits.set_wall_variant(by_slot[slot], index)
                    with perf_trace.phase("unit_commit"):
                        unit_record = unit_edits.commit_unit_edit(unit_label, self.edit_history, push=False)
                    children.append(unit_record)
                parts.append("units")

        pushed = None
        dirty: list[int] = []
        if len(children) == 1:
            pushed = children[0]
            if isinstance(pushed, TileDiffRecord):
                self.edit_history.push_tile_record(pushed)
            else:
                self.edit_history.push_unit_record(pushed)
            dirty = pushed.touched_indices()
        elif len(children) > 1:
            pushed = CompositeDiffRecord("Paste Region", children=children)
            self.edit_history.push_composite_record(pushed)
            dirty = pushed.touched_indices()

        if children:
            if unit_splices is None:
                self._apply_dirty(dirty)
            elif self._render_style == "flat":
                # Flat keeps its wholesale refresh: no row splice, no cost-gate bypass.
                self._cancel_warms()
                if self._cache is not None:
                    with perf_trace.phase("unit_sources"):
                        self._cache.invalidate_units()
                self._apply_dirty(dirty)
                self._after_unit_mutation(sources_refreshed=True)
            elif tile_record is not None:
                # Sources before the tile patch, which then reads units_by_tile that agrees with the
                # units (_flush_pending), so any gen bump lands before its one level build, not after.
                self._cancel_warms()
                if self._cache is not None:
                    with perf_trace.phase("elev_predict"):
                        bumps = self._elevation_bumps_anyway(self._elevation_changed_tiles(tile_record))
                    with perf_trace.phase("unit_sources"):
                        self._cache.invalidate_units(
                            unit_splices + self._flag_flip_splices(unit_splices), splice_levels=not bumps
                        )
                self._apply_dirty(dirty)
                self._after_unit_mutation(sources_refreshed=True)
            else:
                # Units only: Draw release's shape. The index is rebuilt, or the pasted units can't be picked.
                # Reshaped walls carry no splice and can flip the radian verdict: wholesale (TASK-585).
                self._after_unit_mutation(
                    None if rewrites else self._if_spliceable(unit_splices),
                    batch=True,
                    rebuild_index=self.mode == "units",
                )
        self._update_edit_actions()
        self._update_title()
        return _PasteOutcome(pushed, parts, units_skipped, added_units, len(rewrites))

    def _write_block_tiles(
        self, block, tx0: int, ty0: int, do_terrain: bool, do_elevation: bool, label: str = "Paste Region"
    ):
        """The tile half of a paste, shared with a Select region's in-place
        flip or turn (TASK-256): `block`'s terrain and layers, then its
        elevation through set_tiles_elevation() and its skirt, at (tx0, ty0),
        clipped to the map. Returns (unpushed tile record or None, parts)."""
        mm = self.scenario.map_manager
        parts: list[str] = []
        with perf_trace.phase("paste_tiles"):
            # Scoped snapshot: the block rectangle up front, each skirt tile just before its write.
            history = self.edit_history
            rect = region_clipboard.clipped_indices(block, tx0, ty0, mm.map_width, mm.map_height)
            history.begin_stroke(mm.terrain, indices=rect)
            with self._close_stroke_on_error(label):
                if do_terrain:
                    # A file with no layer field would draw a pasted layer and never save it.
                    region_clipboard.paste_terrain(mm, block, tx0, ty0, carry_layer=self.scenario.terrain_has_layer)
                    parts.append("terrain")
                if do_elevation:
                    targets = region_clipboard.elevation_targets(block, tx0, ty0, mm.map_width, mm.map_height)
                    set_tiles_elevation(mm, targets, before_write=lambda i: history.stroke_capture(mm.terrain, i))
                    parts.append("elevation")
            tile_record = self.edit_history.build_stroke_record(label, mm.terrain)
        return tile_record, parts

    def on_fill(self, x: int, y: int, modifiers) -> None:
        with perf_trace.op("fill"):
            self._on_fill(x, y, modifiers)

    def _on_fill(self, x: int, y: int, modifiers) -> None:
        """Paint Can: one flood fill per left click -- MapView routes
        CLICK_TOOLS here directly (see mousePressEvent), never through the
        stroke handlers above. Shaped like paste_region() just above, not like
        on_edit_stroke_tile(): begin_stroke/build_stroke_record once at the
        end (via _apply_terrain_unit_plan()) rather than
        begin_stroke/stroke_new_dirty/commit_stroke, since there's no
        drag to give live feedback during. `modifiers` is accepted only for
        signature symmetry with on_edit_stroke_tile/on_click_edit and is
        deliberately ignored -- a fill has no Shift/right-button inverse.

        Wraps the whole operation in the same wait-cursor/setEnabled(False)/
        self._busy pattern load_scenario()/refresh_map()/_render_current()
        already use: tools/bench_fill_latency.py measured a full-map fill on
        a 480x480 map at ~540ms (Flat) to ~2.4s (Stepped, before
        STEPPED_FULL_RERENDER_THRESHOLD's fallback) end to end -- long
        enough that the window would otherwise sit frozen with no visible
        indication anything is happening, and that a second click dispatched
        via processEvents() mid-fill could reenter this method while
        self.scenario is still being mutated. Applied unconditionally rather
        than only above some measured size, matching every other guarded
        method in this class -- the guard is cheap for the common small-fill
        case too."""
        if self.scenario is None or self._busy:
            return
        mm = self.scenario.map_manager
        if not (0 <= x < mm.map_width and 0 <= y < mm.map_height):
            return
        terrain_id = self.terrain_panel.terrain_id()
        name = name_for_terrain_id(terrain_id)
        target = self._terrain_target()
        on_base = target == terrain_target.TERRAIN
        clicked = mm.terrain[y * mm.map_width + x]
        same_base = clicked.terrain_id == terrain_id
        # Exactly the tiles this click changes. Empty is a no-op, logged (a
        # silent one reads as a broken keybind) before any snapshot or dialog.
        region = fill_indices(mm, x, y, target, terrain_id)
        if not region:
            if on_base:
                reason = f"already {name}"
            elif target == terrain_target.CLEAR:
                reason = "no layered terrain here"
            elif same_base:
                reason = f"{name} is the base terrain here and it has no layered terrain"
            else:
                reason = f"the layered terrain is already {name}"
            self._log_status(f"Fill at ({x}, {y}): {reason} (no change)")
            return
        # A fill of a terrain over itself changes no terrain_id on either
        # target: it only clears the region's layered terrain.
        clears_layers = target == terrain_target.CLEAR or same_base
        # Only a base-terrain change plants or removes trees and eye candy.
        paints_units = on_base and not same_base

        # Large-operation guards, sized BEFORE anything is written, so Cancel
        # leaves the map untouched. Paint Can only -- Draw's counts are
        # bounded by brush size x drag length (see
        # TERRAIN_UNIT_CONFIRM_THRESHOLD's own comment). The Layer and Clear
        # layer targets never ask: the click names one patch and plants nothing.
        question = None
        if on_base and len(region) > TERRAIN_UNIT_CONFIRM_THRESHOLD:
            if same_base:
                question = (
                    f"This area is already {name}. Filling it clears the layered terrain on "
                    f"{len(region)} tiles. Continue?"
                )
            elif self.paint_trees_check.isChecked() or self.paint_eye_candy_check.isChecked():
                # Skipped when neither box is on: a terrain-only fill has no per-tile unit cost.
                question = (
                    f"This fill covers {len(region)} tiles and can place a large number of "
                    f"trees/eye candy units, which may take a while. Continue?"
                )
        if question is not None:
            # Its own phase, so the user's think time isn't read as untimed fill work.
            with perf_trace.phase("confirm_dialog"):
                reply = QMessageBox.question(
                    self, "Large fill", question, QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel
                )
            if reply != QMessageBox.Yes:
                return

        # A Layer fill of the region's own base only clears, so History says so.
        label_target = terrain_target.CLEAR if clears_layers and not on_base else target
        label = terrain_target.stroke_label("fill", label_target, _STROKE_LABELS["fill"])
        self._busy = True
        self.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            t0 = time.perf_counter()
            self.edit_history.begin_stroke(mm.terrain)
            terrain = mm.terrain
            with self._close_stroke_on_error(label):
                for i in region:
                    terrain_target.write_tile(terrain[i], target, terrain_id)
            tile_record = self.edit_history.build_stroke_record(label, mm.terrain)
            if paints_units:
                record, _splices = self._apply_terrain_unit_plan(tile_record, mm)
            else:
                record = tile_record
            self._push_terrain_unit_record(record)
            dirty = record.touched_indices() if record is not None else []
            if isinstance(record, TileDiffRecord):
                # No CompositeDiffRecord (no tree/doodad change) -- the
                # terrain repaint _push_terrain_unit_record() skips still
                # has to happen once, here, since Paint Can (unlike Draw)
                # never repaints incrementally during the fill itself.
                self._apply_dirty(dirty)
            elapsed = time.perf_counter() - t0
        finally:
            QApplication.restoreOverrideCursor()
            self.setEnabled(True)
            self._busy = False
        self._update_edit_actions()
        self._update_title()
        if clears_layers:
            line = f"Cleared layered terrain on {len(dirty)} tiles from ({x}, {y})"
        elif on_base:
            line = f"Filled {len(dirty)} tiles with {name} from ({x}, {y})"
        else:
            line = f"Filled {len(dirty)} tiles with layer {name} from ({x}, {y})"
        line += f" applied in {elapsed:.2f}s"
        # A layer-only edit changes no pixel while layered terrain is not drawn: say so.
        note = self._layer_visibility_note() if (clears_layers or not on_base) else ""
        self._log_status(f"{line} ({note})" if note else line)

    def on_click_edit(self, x: int, y: int, modifiers) -> None:
        """MapView's CLICK_TOOLS destination -- dispatches to whichever
        one-shot tool is actually active, since Paint Can (on_fill) used to
        be the only member and was wired here directly. Eyedropper is the
        second."""
        if self._current_tool == "eyedropper":
            self.pick_tile_value(x, y, modifiers)
        elif self._current_tool == "create_objects":
            self.stamp_create_objects(x, y)
        else:
            self.on_fill(x, y, modifiers)

    def pick_tile_value(self, x: int, y: int, modifiers) -> None:
        """Eyedropper: reads a tile's terrain into the sidebar picker and its
        elevation into the Level param, no undo record since nothing is mutated.
        A pick that took either part switches back to the previous tool when
        Switch back after a pick is on (TASK-287); Ctrl in `modifiers` keeps the tool.

        Indexes mm.terrain directly rather than calling mm.get_tile(), which
        raises on any non-square map (fill_tools.py documents this) -- a
        pick must work on every write_ok map, squareness included or not."""
        if self.scenario is None:
            return
        mm = self.scenario.map_manager
        if not (0 <= x < mm.map_width and 0 <= y < mm.map_height):
            return
        tile = mm.terrain[y * mm.map_width + x]
        # On the Layer target a layered tile gives its layered terrain, the one a Layer stroke writes.
        picks_layer = self._terrain_target() == terrain_target.LAYER and tile.layer != -1
        terrain_id = tile.layer if picks_layer else tile.terrain_id
        elevation = tile.elevation

        terrain_took = self.terrain_panel.set_terrain(terrain_id)
        if terrain_took:
            terrain_part = name_for_terrain_id(terrain_id)
        else:
            terrain_part = f"{name_for_terrain_id(terrain_id)} (not in the terrain picker, unchanged)"
        if picks_layer:
            terrain_part = f"layer {terrain_part} (over {name_for_terrain_id(tile.terrain_id)})"

        elevation_took = 0 <= elevation <= ELEVATION_LEVEL_MAX
        if elevation_took:
            self.elevation_level_spin.setValue(elevation)
            elevation_part = f"elevation {elevation}"
        else:
            elevation_part = f"elevation {elevation} (out of range, unchanged)"

        self._log_status(f"Picked {terrain_part}, {elevation_part} from ({x}, {y})")
        if terrain_took or elevation_took:
            self._eyedropper_switch_back(modifiers)

    def _eyedropper_switch_back(self, modifiers) -> None:
        """TASK-287: after a pick that applied something, back to the tool the
        Eyedropper was entered from, when the box is on, Ctrl is not held, and
        that tool is still enabled and applicable in this mode."""
        tool = self._eyedropper_return_tool
        if not self.eyedropper_return_check.isChecked() or modifiers & Qt.ControlModifier or tool is None:
            return
        action = getattr(self, f"{tool}_action", None)
        if action is None or not action.isEnabled() or not tool_applicable(tool, self.mode):
            return
        action.setChecked(True)

    def pick_unit_value(self, entry, tile, modifiers) -> None:
        """Eyedropper in Units mode (MapView's on_unit_eyedrop): the clicked
        unit's type becomes the Units catalog's pending object. A plain click
        also copies its owner into the Owner combo and its facing (an ANGLE
        const) or variant (a cyclable one) into Place Unit's pending rotation,
        so the next placement clones it; Shift copies the type alone.

        Reads the unit and writes UI state only, so there is no undo record.
        What a later click stores still comes from place_rotation.rotation_for()
        alone, a GAIA unit's facing included (GH #217). Walls, gates, cliffs
        and single-frame consts leave the rotation alone. A pick the catalog
        took switches back as pick_tile_value()'s does (TASK-287)."""
        if self.scenario is None:
            return
        unit = entry.unit
        const = unit.unit_const
        x, y = tile
        name = _unit_name(const)
        catalog = self.units_panel.catalog_view
        had_filter = bool(catalog.filter_edit.text())
        box = catalog.show_hidden_checkbox
        had_hidden = box is None or box.isChecked()
        had_class = catalog.class_combo.currentData()
        had_owner = self.units_panel.owner_id()
        if not self.units_panel.pick_object(const):
            self._log_status(f"Picked {name} from ({x}, {y}) (not in the catalog, unchanged)")
            return
        # What pick_object() changed to show the row, as rotate_place_ghost's gate turn says.
        revealed = []
        if had_filter and not catalog.filter_edit.text():
            revealed.append("cleared the catalog filter")
        if had_class is not None and catalog.class_combo.currentData() is None:
            revealed.append("reset the class filter")
        if not had_hidden and box.isChecked():
            revealed.append(f'ticked "{box.text()}"')
        # A plain click names the owner it copies anyway; Shift says so only here.
        if modifiers & Qt.ShiftModifier and had_owner != self.units_panel.owner_id():
            revealed.append("set Owner to GAIA")
        note = f" ({' and '.join(revealed)} to show it)" if revealed else ""
        if modifiers & Qt.ShiftModifier:
            self._log_status(f"Picked {name} from ({x}, {y}){note}")
            self._eyedropper_switch_back(modifiers)
            return
        # GH #32: a bad file's player-owned GAIA-only object is picked for GAIA, where it places.
        owner = unit_kind.placement_owner(const, entry.player_id)
        self.units_panel.select_owner(owner)
        parts = [name, "GAIA" if owner == GAIA_PLAYER_ID else f"Player {owner}"]
        if unit_rotation.rotation_is_angle(const):
            self._place_rotation = dataclasses.replace(self._place_rotation, facing=unit.rotation)
            parts.append(self._placement_rotation_text(const, unit.rotation))
        elif unit_variant.is_cyclable(const):
            variant = unit_variant.variant_of(
                unit.rotation, unit_rotation.angle_count_for(const), unit_variant.variant_count_for(const)
            )
            self._place_rotation = dataclasses.replace(self._place_rotation, variant=variant, variant_const=const)
            parts.append(self._placement_rotation_text(const, unit.rotation))
        self.map_view.refresh_place_ghost()
        self._log_status(f"Picked {', '.join(parts)} from ({x}, {y}){note}")
        self._eyedropper_switch_back(modifiers)

    def on_mirror(self, plan, unit_plan=None) -> list[int] | None:
        """Map mirroring's apply path (mirror_tools.plan_mirror -> here),
        called by MirrorDialog for both Preview and Apply. Copies on_fill()'s
        busy-guard/single-history-record/incremental-repaint shape near-
        verbatim -- see that method's own docstring for why (no live drag
        to give feedback during).

        Returns None -- refusing to apply, no record pushed -- when
        plan.elevation_violations is non-empty: that check exists precisely
        because an illegal ±1 jump crashes AoE2:DE at load time
        (scenario_write.py), and per the plan there is no safe auto-repair
        (any repair would break the symmetry the mirror was asked to
        produce). Otherwise returns the list of dirty tile indices, same
        shape as on_fill() -- empty if the map was already symmetric under
        the chosen mode/slice, which is a genuine no-op, not a refusal.

        `plan.changes` is already fully computed with no possibility of
        raising, so the mutator handed to the history here is a bare
        assignment loop, per plan_mirror's own "everything computed before
        any mutation" contract.

        `unit_plan` (Stage 2) is mirror_tools.plan_mirror_units()' output or
        None. When present, its removals and images ride a UnitDiffRecord
        built with push=False, and the two records go on the stack as one
        CompositeDiffRecord -- Paste Region's own shape, so one Ctrl+Z undoes
        terrain, elevation and units together. A blocked unit plan refuses
        the whole operation, terrain included, for the same reason an
        elevation seam violation does: a partial mirror is not what was
        asked for. Under the skip and place overlap policies (TASK-250) the
        straddlers no longer block, but a garrison block still does, and the
        elevation check below takes no policy at all."""
        if self.scenario is None or self._busy:
            return None
        if unit_plan is not None and unit_plan.blocked:
            self._log_status(self._mirror_unit_refusal(unit_plan), level=debug_log.WARNING)
            return None
        if plan.elevation_violations:
            width = self.scenario.map_manager.map_width
            sample = ", ".join(
                f"({i % width}, {i // width})-({j % width}, {j // width})"
                for i, j in plan.elevation_violations[:5]
            )
            self._log_status(
                f"Mirror refused: {len(plan.elevation_violations)} elevation seam "
                f"violation(s) would exceed the +/-1 limit (e.g. {sample}) -- pick a "
                f"different mode, or flatten the source region's edges first"
            )
            return None

        mm = self.scenario.map_manager
        changes = plan.changes

        self._busy = True
        self.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            children: list = []
            # The unit half goes FIRST, before a single tile is touched: it is
            # the half that can still raise (remove_many refuses a garrisoned
            # unit), and it rolls itself back through abort_unit_edit(). Doing
            # it after the tile loop would leave the map mirrored with an
            # unpushed record and no undo.
            unit_record = self._apply_mirror_units(unit_plan)
            # begin_stroke/build_stroke_record rather than apply(): the tile
            # record has to stay unpushed until it is known whether a unit
            # record rides with it.
            self.edit_history.begin_stroke(mm.terrain)
            for idx, (terrain_id, elevation, layer) in changes:
                tile = mm.terrain[idx]
                tile.terrain_id, tile.elevation, tile.layer = terrain_id, elevation, layer
            tile_record = self.edit_history.build_stroke_record("Mirror Map", mm.terrain)
            if tile_record is not None:
                children.append(tile_record)
            if unit_record is not None:
                children.append(unit_record)

            if len(children) == 1:
                record = children[0]
                if isinstance(record, TileDiffRecord):
                    self.edit_history.push_tile_record(record)
                else:
                    self.edit_history.push_unit_record(record)
                dirty = record.touched_indices()
            elif len(children) > 1:
                composite = CompositeDiffRecord("Mirror Map", children=children)
                self.edit_history.push_composite_record(composite)
                dirty = composite.touched_indices()
            else:
                dirty = []
            self._apply_dirty(dirty)
            if unit_record is not None:
                self._after_unit_mutation()
        finally:
            QApplication.restoreOverrideCursor()
            self.setEnabled(True)
            self._busy = False
        self._update_edit_actions()
        self._update_title()
        overlaps = self._mirror_overlap_note(unit_plan)
        if dirty:
            units = "" if unit_plan is None else (
                f", {len(unit_plan.removals)} unit(s) replaced by {len(unit_plan.images)}"
            )
            self._log_status(f"Mirrored {len(dirty)} tiles{units}{overlaps}")
        else:
            self._log_status(f"Mirror: already symmetric under this mode (no change){overlaps}")
        return dirty

    @staticmethod
    def _mirror_overlap_note(unit_plan) -> str:
        """The skip or place report: how many copies landed on a source unit,
        and the first few tiles they land on (the copy's, not the source's)."""
        if unit_plan is None or not unit_plan.overlapping:
            return ""
        sample = ", ".join(f"({int(image.x)}, {int(image.y)})" for image in unit_plan.overlapping[:5])
        fate = "skipped" if unit_plan.overlaps == OVERLAPS_SKIP else "placed overlapping"
        return f"; {len(unit_plan.overlapping)} overlapping cop(ies) {fate} (e.g. {sample})"

    def _mirror_unit_refusal(self, unit_plan) -> str:
        """Why a unit plan blocks. Names the first few offenders by tile, the
        same shape on_mirror's elevation-seam message uses."""
        reasons = []
        if unit_plan.overlap_blocked:
            # The unit in the way and the tile it spills onto, not the unit whose copy moved (TASK-043).
            sample = "; ".join(
                f"{_unit_name(u.unit_const)} at ({int(u.x)}, {int(u.y)}), tile ({tx}, {ty})"
                for u, (tx, ty) in unit_plan.overlap_blockers[:5]
            )
            reasons.append(
                f"{len(unit_plan.overlap_blockers)} unit(s) straddle the symmetry axis or a slice "
                f"boundary, so copies would land on them (e.g. {sample})"
            )
        if unit_plan.garrisoned_blockers:
            sample = ", ".join(f"({int(u.x)}, {int(u.y)})" for u in unit_plan.garrisoned_blockers[:5])
            reasons.append(
                f"{len(unit_plan.garrisoned_blockers)} unit(s) outside the source slice hold a "
                f"garrison and cannot be replaced (e.g. {sample})"
            )
        return (
            "Mirror refused: " + "; ".join(reasons)
            + " -- pick a different mode, or move the offending objects off the seam"
        )

    def _apply_mirror_units(self, unit_plan):
        """Removes the destination slices' units and adds the source's images,
        as one unpushed UnitDiffRecord. None when there is nothing to do.

        All nine players are declared up front, following the Convert brush:
        ownership rotation can move a unit to any of them, and the snapshot
        has to cover every list the edit touches.
        """
        if unit_plan is None or not (unit_plan.removals or unit_plan.images):
            return None
        model = self._ensure_unit_edits()
        if model is None:
            return None
        model.begin_unit_edit(range(GAIA_PLAYER_ID, MAX_PLAYER_ID + 1))
        try:
            if unit_plan.removals:
                model.remove_many(unit_plan.removals)
            for image in unit_plan.images:
                source = image.source
                try:
                    caption_id, caption = source.caption_string_id, source.caption_string
                except (UnsupportedAttributeError, AttributeError):
                    caption_id, caption = -1, ""  # pre-1.55: the garrison clipboard's defensive read
                model.add(
                    player=image.player,
                    unit_const=image.unit_const,
                    x=image.x,
                    y=image.y,
                    z=source.z,
                    rotation=image.rotation,
                    status=source.status,
                    initial_animation_frame=source.initial_animation_frame,
                    garrisoned_in_id=-1,  # a copy never inherits a garrison link
                    caption_string_id=caption_id,
                    caption_string=caption,
                    capture_flag=capture_flag_of(source),
                )
        except Exception:
            model.abort_unit_edit()
            raise
        return model.commit_unit_edit("Mirror Map", self.edit_history, push=False)
