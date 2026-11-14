"""ViewerWindow's unit selection and movement: the unit index, click, stack and marquee select, drag
and group move, a drop into a garrison, the drag turn, ghosts and nudge."""

from __future__ import annotations

import math

from PyQt5.QtCore import Qt
from PyQt5.QtGui import (
    QKeySequence,
)
from PyQt5.QtWidgets import (
    QAction,
)

from descape import (
    debug_log,
    garrison,
    gate_orientation,
    move_turn,
    perf_trace,
    placement_rules,
    settings,
    unit_pick,
    unit_rotation,
    unit_variant,
    wall_run,
)
from descape.render_cache_splice import UnitSplice
from descape.render_sprites import unit_sprite_draws_at
from descape.render_units import unit_at, unit_mark_color, unit_tile_bounds, wall_variant_rotation_overrides
from descape.unit_model import (
    reanchored,
)
from descape.viewer_common import (
    OFF_GRID_FALLBACK_MESSAGE,
    _garrison_referrers,
    _ghost_entry,
    _team_colors,
    _unit_name,
    clamp_group_delta,
)

# D2's arrow-key nudge amounts, in tiles: a fine nudge, and Shift's whole-tile
# step (matching the tile-centre grid a click/place snaps to).
_UNIT_NUDGE_STEP = 0.1
_UNIT_NUDGE_STEP_SHIFT = 1.0

# GH #75: a group drag ghosts every member up to this many, then the grabbed one only.
GROUP_GHOST_CAP = 200


class _ViewerUnitMoveMixin:
    """ViewerWindow's unit selection, drag and nudge (viewer.py's docstring has the split's rules)."""

    def _needs_unit_index(self) -> bool:
        """Units mode needs the pick index, and so do View > Footprint
        Outlines, View > Layers > Fish Outlines (TASK-499) and the always-on
        Range Rings layers (GH #153 / #155), which are reachable from every
        mode. Without those terms the overlays
        silently draw nothing outside Units mode, since the index is built on
        demand rather than at load. A kept unit selection needs it too, so its
        keys reconcile after an edit made in another mode (TASK-359)."""
        return (
            self.mode == "units"
            or settings.get_footprint_outlines()
            or self._layers.fish_outlines
            or settings.get_range_rings_revealers()
            or settings.get_range_rings_buildings()
            or self._unit_picker is not None
            or bool(self._selection)
        )

    def _drop_unit_index_if_unneeded(self) -> None:
        """After a reconcile outside Units: the selection may have been all
        that needed the index, and the index is current or None."""
        if self.map_view._unit_index is not None and not self._needs_unit_index():
            self._stack_cycle = None
            self.map_view.set_unit_index(None)

    def _rebuild_unit_index(self) -> None:
        """Rebuilds the pick index for the current scenario + filter and hands
        it to MapView.

        Built on demand (the first entry to Units mode, or a filter change
        while already in it) rather than at load: an ~11k-unit file costs a
        real walk, and opening a map for terrain work must not pay for it.

        The index is current or None: it may survive outside Units mode, and
        every unit edit or filter change there rebuilds it for whatever still
        needs it (the footprint overlay, an armed picker) or drops it. So
        _switch_mode() reuses a non-None index on entry instead of calling this.
        """
        # A new index can reorder or resize any stack the cycle points into.
        self._stack_cycle = None
        if self.scenario is None:
            self.map_view.set_team_colors(None)
            self.map_view.set_unit_index(None)
            return
        # Before set_unit_index(), whose own highlight refresh then draws with these.
        self.map_view.set_team_colors(_team_colors(self.scenario))
        self.map_view.set_unit_index(unit_pick.build_index(self.scenario, self._unit_filter))

    def on_click_select(self, pos, modifiers) -> tuple[int, int] | None:
        """MapView's sixth injected callable: a Select-tool press in Units mode.

        Selection is a COLLECTION from day one (self._selection), even though
        phase 3 only ever held 0 or 1: future marquee/bulk items would
        otherwise force an immediate rewrite of a scalar _selected_unit. And
        identity is the (player_id, reference_id) key resolved late through
        the index, never a list position or an object reference -- phase
        3.5's add/remove invalidates both of those.

        b2.2 wires the modifiers this always received but ignored (settled
        fact 3): Ctrl toggles the clicked unit's own membership; Shift adds
        it without ever removing. Neither is "range" selection -- units have
        no ordering for that to mean anything against -- so both degrade to
        a plain set operation over the existing selection. A modified click
        on empty ground leaves the selection alone (Ctrl/Shift mean "adjust
        the set", not "replace it"); a plain click on empty ground clears it,
        same as before this slice.

        A plain click on a stack (unit_pick.stack_groups()) selects exactly
        what pick_unit returns; each repeat click on the same stack selects
        the next unit down, wrapping. Ctrl/Shift are set operations, not
        stack navigation, and reset the cycle.

        Returns the key the click acted on (None on empty ground), which
        MapView uses to tell a press on a unit from one on empty ground, and
        which is kept as _marquee_press_key for the marquee that may follow.
        Selecting only: dragging is the Move tool's (on_move_press()), so a
        plain click on a member of a group selects that unit at press.
        """
        found = self._press_pick(pos)
        self._marquee_press_key = None
        self.on_lift_finalize()  # TASK-256: any click ends a flip or turn
        if found is None:
            self._stack_cycle = None
            if not (modifiers & (Qt.ControlModifier | Qt.ShiftModifier)):
                self._selection = []
                self._refresh_selection_view()
                self._log_status("Selection cleared")
            return None
        entry, cover_tile = found
        key = unit_pick.unit_key(entry.player_id, entry.unit)
        stack_note = ""
        if modifiers & Qt.ControlModifier:
            self._stack_cycle = None
            if key in self._selection:
                self._selection = [k for k in self._selection if k != key]
            else:
                self._selection = [*self._selection, key]
        elif modifiers & Qt.ShiftModifier:
            self._stack_cycle = None
            if key not in self._selection:
                self._selection = [*self._selection, key]
        else:
            entry, stack_note = self._cycle_stack(entry, cover_tile)
            key = unit_pick.unit_key(entry.player_id, entry.unit)
            self._selection = [key]
        self._refresh_selection_view()
        if len(self._selection) == 1:
            self._log_status(
                f"Selected {_unit_name(entry.unit.unit_const)} at ({entry.unit.x:g}, {entry.unit.y:g}){stack_note}"
            )
        else:
            self._log_status(f"{len(self._selection)} units selected")
        self._marquee_press_key = key
        return key

    def _press_pick(self, pos):
        """The one unit pick a Select or Move press makes, as (entry, cover
        tile) or None, resetting the per-press drag bookkeeping with it."""
        self._ghost_cap_logged = None
        self._reset_drag_rule_memos()
        found = self.map_view.pick_unit_cover_at(pos)
        self._drag_grab_tile = None if found is None else found[1]
        return found

    def _reset_drag_rule_memos(self, grid: bool = True) -> None:
        """Drops the drag previews' rule memos (TASK-327). The grid holds only
        units, so a terrain edit keeps it (`grid=False`)."""
        self._drag_rule_memo = None
        self._group_flags_memo = None
        self._drag_stand_in_memo.clear()
        if grid:
            self._group_grid_memo = None

    def on_move_press(self, pos, modifiers) -> tuple[int, int] | None:
        """The Move tool's press in Units mode (TASK-012, GH #47): the key the
        drag will move, or None to drag nothing. Modifiers never change the
        selection here; Shift and Alt only matter at release.

        A selected unit drags as it is (with its group when the selection has
        2+). A press on a stack whose selected member is lower down drags that
        member, so cycling to a unit with Select and dragging it with Move
        works. Anything else selects just the pressed unit first. The key
        returned is always in the selection, which _is_group_key() and
        _drag_entries() rely on. Empty ground leaves the selection alone.
        """
        self._drag_turn = move_turn.MoveTurn()
        found = self._press_pick(pos)
        self.on_lift_finalize()  # TASK-256: any click ends a flip or turn
        if found is None:
            return None
        entry, cover_tile = found
        key = unit_pick.unit_key(entry.player_id, entry.unit)
        if key in self._selection:
            return key
        for member in self.map_view.stack_group_at(cover_tile) or ():
            member_key = unit_pick.unit_key(member.player_id, member.unit)
            if member_key in self._selection:
                return member_key
        self._stack_cycle = None
        self._selection = [key]
        self._refresh_selection_view()
        self._log_status(f"Selected {_unit_name(entry.unit.unit_const)} at ({entry.unit.x:g}, {entry.unit.y:g})")
        return key

    def _on_pan_click(self, pos) -> None:
        """MapView's click under Pan in Units mode: the first one on a unit
        this session logs which tools pick and drag units, with their live keys."""
        if self._pan_hint_logged or self.map_view.pick_unit_at(pos) is None:
            return
        self._pan_hint_logged = True

        def named(label: str, action: QAction) -> str:
            key = action.shortcut().toString(QKeySequence.NativeText)
            return f"{label} ({key})" if key else label

        self._log_status(
            f"Pan moves the view. {named('Select', self.select_action)} picks units, "
            f"{named('Move', self.move_action)} drags them."
        )

    def _is_group_key(self, key) -> bool:
        """Whether a drag on `key` moves the whole selection (GH #75)."""
        return key is not None and len(self._selection) >= 2 and key in self._selection

    def _cycle_stack(self, picked, cover_tile: tuple[int, int]):
        """(entry to select, status suffix) for a plain click that picked
        `picked` while covering `cover_tile`, advancing _stack_cycle."""
        members = self.map_view.stack_group_at(cover_tile)
        if not members:
            self._stack_cycle = None
            return picked, ""
        cycle = self._stack_cycle
        continuing = cycle is not None and cycle[0] == cover_tile and self._selection == [cycle[2]]
        chosen, idx = unit_pick.stack_cycle_step(members, picked, cycle[1] if continuing else None)
        self._stack_cycle = (cover_tile, idx, unit_pick.unit_key(chosen.player_id, chosen.unit))
        if idx == -1:
            return chosen, ""
        return chosen, f" ({idx + 1} of {len(members)} stacked here)"

    def on_marquee_select(self, keys: list[tuple[int, int]], modifiers) -> None:
        """MapView's eleventh injected callable -- a completed marquee drag
        in Units mode (b2.3). `keys` are the (player_id, reference_id) keys
        of every unit MapView._units_in_scene_rect() found under the dragged
        rectangle (by footprint tile in Flat, by on-screen anchor in the
        isometric styles), already filter-respecting.

        Same modifier convention as on_click_select's own toggle/add (b2.2),
        applied to the whole covered set at once: Ctrl toggles each key's
        membership (a second marquee over an already-selected group
        deselects exactly that overlap); Shift adds without removing; plain
        replaces the selection outright.

        The press that started the marquee was already a click (TASK-252):
        under Ctrl or Shift the unit it acted on (_marquee_press_key) is
        dropped from `keys`, so the rectangle cannot toggle or add it a second
        time. A marquee left with no keys changes nothing, so whatever the
        press did stands: a press on a unit that drifts into an empty
        rectangle is that click, and a press on empty ground already cleared
        (plain) or kept (Ctrl/Shift) the selection.
        """
        press_key, self._marquee_press_key = self._marquee_press_key, None
        if modifiers & (Qt.ControlModifier | Qt.ShiftModifier):
            keys = [k for k in keys if k != press_key]
        if not keys:
            return
        self.on_lift_finalize()
        if modifiers & Qt.ControlModifier:
            existing = set(self._selection)
            toggled = existing.symmetric_difference(keys)
            self._selection = list(toggled)
        elif modifiers & Qt.ShiftModifier:
            merged = list(self._selection)
            for key in keys:
                if key not in merged:
                    merged.append(key)
            self._selection = merged
        else:
            self._selection = list(keys)
        self._refresh_selection_view()
        if not self._selection:
            self._log_status("Selection cleared")
        elif len(self._selection) == 1:
            index = self.map_view._unit_index
            entry = index.entry_for_key(self._selection[0]) if index is not None else None
            if entry is not None:
                self._log_status(
                    f"Selected {_unit_name(entry.unit.unit_const)} at ({entry.unit.x:g}, {entry.unit.y:g})"
                )
        else:
            self._log_status(f"{len(self._selection)} units selected")

    def on_select_whole_stack(self) -> None:
        """Edit > Select Whole Stack (GH #75): adds every member of every stack
        holding a selected unit, keeping order and dropping duplicates.

        Found by membership, not by tile: a stack is keyed by its hidden
        unit's own tile (unit_pick.stack_scan), which need not be the selected
        unit's tile."""
        if not self._selection:
            self._log_status("Select Whole Stack: nothing is selected")
            return
        selected = set(self._selection)
        widened = list(self._selection)
        seen = set(widened)
        stacked = False
        for members in self.map_view._stack_groups.values():
            keys = [unit_pick.unit_key(m.player_id, m.unit) for m in members]
            if selected.isdisjoint(keys):
                continue
            stacked = True
            for key in keys:
                if key not in seen:
                    seen.add(key)
                    widened.append(key)
        if not stacked:
            self._log_status("Select Whole Stack: no selected unit is stacked")
            return
        self.on_lift_finalize()
        self._stack_cycle = None
        self._selection = widened
        self._refresh_selection_view()
        self._log_status(f"{len(self._selection)} units selected (stack)")

    def _entry_for_reference(self, ref_id: int):
        """The live unit carrying `ref_id` as a UnitEntry, synthesized with
        order -1 when the filter keeps it out of the pick index, or None."""
        ref = self._unit_reference_index().get(ref_id)
        if ref is None:
            return None
        index = self.map_view._unit_index
        entry = index.entry_for_key((ref.player_id, ref.reference_id)) if index is not None else None
        if entry is not None:
            return entry
        unit = next((u for u in self.scenario.unit_manager.units[ref.player_id] if u.reference_id == ref_id), None)
        return None if unit is None else unit_pick.UnitEntry(ref.player_id, unit, int(unit.x), int(unit.y), -1)

    def _inside_dragged(self, entry, dragged: set[int]) -> bool:
        """Whether `entry` is one of the dragged units (by id()) or inside one."""
        unit, seen = entry.unit, set()
        while True:
            if id(unit) in dragged:
                return True
            link = unit.garrisoned_in_id
            if link == -1 or link in seen:
                return False
            seen.add(link)
            holder = self._entry_for_reference(link)
            if holder is None:
                return False
            unit = holder.unit

    def _drag_entries(self, key, entry) -> list:
        """What a drag of `key` carries: the whole selection for a group key."""
        if not self._is_group_key(key):
            return [entry]
        index = self.map_view._unit_index
        return [e for e in (index.entry_for_key(k) for k in self._selection) if e is not None]

    def _garrison_host_at(self, pos, dragged=frozenset()):
        """The host entry under `pos` that can_hold anything, or None: the one
        hit rule a drop (GH #115) and Place Unit (TASK-549) share. The hit skips
        `dragged` (unit id()s) and their occupants, falling through a stack,
        and a visible occupant resolves to its host."""
        found = self.map_view.pick_unit_cover_at(pos)
        if found is None:
            return None
        hit, cover = found
        candidates = [hit] + [m for m in (self.map_view.stack_group_at(cover) or ()) if m.unit is not hit.unit]
        target = next((c for c in candidates if not self._inside_dragged(c, dragged)), None)
        seen: set[int] = set()
        while target is not None and target.unit.garrisoned_in_id != -1 and target.unit.reference_id not in seen:
            seen.add(target.unit.reference_id)
            target = self._entry_for_reference(target.unit.garrisoned_in_id)
        if target is None or self._inside_dragged(target, dragged):
            return None
        return target if garrison.can_hold(target.unit.unit_const) else None

    def _place_garrison_host(self, pos, const: int):
        """The host Place Unit would put `const` inside at `pos` (TASK-549), or
        None for a plain place. The cheap gates come first, since this runs on
        every hover move: the tool is Place Unit, Free placement is off and
        `const` can go inside something at all."""
        if self._current_tool != "place_unit" or self._restrictions_lifted() or not garrison.can_garrison(const):
            return None
        return self._garrison_host_at(pos)

    def _place_garrison_refusal(self, host_entry, const: int) -> str | None:
        """garrison.refusal() for one new `const` inside `host_entry`'s unit,
        cached per host, const and unit edit like the drop preview's. A new
        unit holds nothing and is inside nothing, so no structural check."""
        host = host_entry.unit
        key = (self._doc_id, unit_pick.unit_key(host_entry.player_id, host), const, self.scenario.unit_gen)
        cached = self._place_garrison_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        name = _unit_name(host.unit_const)
        message = garrison.refusal(
            host.unit_const, name, len(self._garrison_occupants(host)), [const], name_of=_unit_name
        )
        self._place_garrison_cache = (key, message)
        return message

    def _garrison_drop(self, entries, pos, modifiers, preview_key=None):
        """(host_entry, refusal or None) when releasing `entries` at `pos`
        garrisons them, else None for a plain move (GH #115). The preview and
        the release both ask this, so the outline cannot disagree with the drop.

        Shift always moves. The hit skips the dragged units and their
        occupants, falling through a stack, and a visible occupant resolves
        to its host. `preview_key` caches the refusal for one drag and host."""
        if modifiers is not None and modifiers & Qt.ShiftModifier:
            return None
        target = self._garrison_host_at(pos, {id(e.unit) for e in entries})
        if target is None:
            return None
        host = target.unit
        if not any(garrison.accepts(host.unit_const, e.unit.unit_const) for e in entries):
            return None
        cache_key = (preview_key, unit_pick.unit_key(target.player_id, host), self.scenario.unit_gen)
        cached = self._drop_preview_cache
        if preview_key is not None and cached is not None and cached[0] == cache_key:
            return target, cached[1]
        message = self._garrison_refusal_for(host, entries)
        self._drop_preview_cache = (cache_key, message) if preview_key is not None else None
        return target, message

    def on_unit_move(self, key: tuple[int, int], pos, modifiers) -> None:
        """MapView's eighth injected callable -- a press-drag-release past
        UNIT_DRAG_THRESHOLD_PX on a picked unit (b1.5). `key` is the
        (player_id, reference_id) MapView picked at press time; resolved
        late through the live index here, same as every other selection-key
        use, since add/remove/reassign elsewhere in the same session can
        invalidate it before the release arrives.

        A drop on a host that can take the dragged units garrisons them
        instead (GH #115, _garrison_drop()); Shift keeps it a move.

        A move placement_rules refuses changes nothing and adds no undo
        record (GH #176), unless Free placement is ticked. A group is all or
        nothing (_group_move_refusal()).

        With "Moving a gate updates walls" ticked, a gate that changes tiles
        also clears the walls it lands on and fills the line it left, in the
        same record (GH #194, _move_plan()). Those walls never block the
        move; anything else under the gate still does.

        A wall that changes tiles takes the shape of its new place, and the
        walls beside the tile it left and the one it reached reshape, in the
        same record (GH #266, _move_plan()). That stays a fields-only splice
        edit unless the plan is `wholesale`.

        A turn the drag holds (GH #280, _turn_unit_drag()) lands in the same
        record: its `rotation` or const goes through the setter Rotate uses,
        taken from the drag's stand-in, so the drop is the ghost. A gate turned
        lands by its new span, and "Moving a gate updates walls" reads that
        span for the tiles it reaches and its stored one for those it left.
        Back on its own point the drop is a turn in place: nothing's position
        is written, and it reads as a Rotate. A garrison drop drops the turn.
        """
        try:
            self._commit_unit_drag(key, pos, modifiers)
        finally:
            self._drag_turn = move_turn.MoveTurn()

    def _commit_unit_drag(self, key, pos, modifiers) -> None:
        """on_unit_move's body, whichever way it ends."""
        if self.scenario is None:
            return
        index = self.map_view._unit_index
        entry = index.entry_for_key(key) if index is not None else None
        if entry is None:
            return
        entries = self._drag_entries(key, entry)
        group = self._is_group_key(key)
        drop = self._garrison_drop(entries, pos, modifiers)
        if drop is not None:
            host_entry, message = drop
            if message is not None:
                self._log_status(message)
                return
            turned = any(self._drag_stand_in(e, group) is not e.unit for e in entries)
            if self._garrison_units(host_entry, entries) and turned:
                self._log_status("Move: the turn was dropped, since a garrisoned unit keeps its facing")
            return
        if group:
            self._move_group(entry, pos, modifiers)
            return
        unit = entry.unit
        stand_in = self._drag_stand_in(entry, False)
        point = self._placement_point(pos, modifiers, unit_const=stand_in.unit_const)
        if point is None:
            return
        model = self._ensure_unit_edits()
        if model is None:
            return
        x, y = point
        turning = stand_in is not unit
        # Effectively never fires off-grid (two float coordinates rarely
        # land on the same bits twice), and still correct for the snapped
        # path, which is the only reason it is still here. The stand-in's
        # point differs from the unit's only by a turned gate's re-anchor.
        in_place = (x, y) == (stand_in.x, stand_in.y)
        if in_place and not turning:
            return
        new_const = stand_in.unit_const
        # The garrison rides along, onto the host's new point (GH #42).
        occupants = [] if in_place else self._garrison_occupant_entries(model, [entry])
        # Planned before the refusal, and only read by it: a refused move changes nothing.
        gate_plan = self._move_plan(
            model, [(e, x, y) for e in (entry, *occupants)], gate_walls=True, consts={id(unit): new_const}
        )
        # A facing never changes a footprint, so a unit already refused where it stands can still turn.
        if not self._restrictions_lifted() and (not in_place or new_const != unit.unit_const):
            # GH #176. The mover never blocks itself; its occupants are garrisoned, so they never block.
            # GH #194: nor do the walls a moved gate replaces, as on_unit_place's gate_walls.
            exclude = [unit, *self._gate_move_replaced(gate_plan)]
            reason = placement_rules.move_refusal(self.scenario, unit, x, y, exclude=exclude, unit_const=new_const)
            if reason is not None:
                self._log_status(f"Move: {reason}. Tick Free placement to move it anyway.")
                return
        mm = self.scenario.map_manager
        clauses = self._gate_move_clauses(gate_plan, [unit])
        moved = f"Moved {_unit_name(unit.unit_const)} to ({x:g}, {y:g})"
        if turning:
            moved += f", {self._turn_clause(unit, stand_in)}"
        label = "Rotate unit" if in_place else "Move and rotate unit" if turning else "Move unit"

        def report() -> None:
            done = self._rotate_status(*self._turn_classes([entry]), 0, 0) if in_place else moved
            self._log_status(done + clauses)

        owners = {entry.player_id} | {e.player_id for e in occupants}
        if gate_plan:
            owners |= wall_run.move_plan_players(gate_plan)
        if gate_plan is not None and gate_plan.wholesale:
            # GH #194: removals and fills shift list indices, so no splice; one wholesale record.
            with self._unit_edit(model, label, sorted(owners)):
                # First: the only call that can refuse, so a refusal leaves nothing moved.
                model.remove_many([wall for _player_id, wall in gate_plan.removals])
                if turning:
                    self._apply_drag_turn(model, stand_in, unit)
                if not in_place:
                    model.set_position(unit, x, y, unit.z)
                for occupant in occupants:
                    model.set_position(occupant.unit, x, y, occupant.unit.z)
                wall_run.apply_move_plan(model, gate_plan)
            report()
            return
        old_bounds = unit_tile_bounds(unit, mm.map_width, mm.map_height)
        old_own, old_tiles = self._unit_footprint(unit)
        idx = self._unit_list_index(entry.player_id, unit)
        splices: list[UnitSplice] = []
        with self._unit_edit(model, label, sorted(owners), splices, fields_only=True):
            # After the captures: a splice's old tiles are the stored footprint.
            swapped = turning and self._apply_drag_turn(model, stand_in, unit)
            if not in_place:
                model.set_position(unit, x, y, unit.z)
            new_own, new_tiles = self._unit_footprint(unit)
            splices.append(UnitSplice(entry.player_id, idx, unit, old_own, new_own, old_tiles, new_tiles))
            # Not scoped to a span change, so a swapped const rebuilds below instead.
            if not swapped and not in_place:
                self._patch_unit_index_for_move(entry.player_id, unit, old_bounds)
            for occupant in occupants:
                self._move_occupant_to(model, occupant, x, y, splices)
            if gate_plan:
                self._apply_wall_reshapes(model, gate_plan, splices)
            if swapped:
                self._rebuild_unit_index()
        report()

    def _drag_stand_in(self, entry, group: bool):
        """`entry`'s unit as the drag's turn leaves it at its source, for the
        preview and the drop to read, or the real unit itself when the turn
        changes nothing for it: that member takes no re-anchor and no write.
        Never a write target (render_units.unit_at()).

        `group` is Rotate's map-edge rule. A group member turns on its own spot,
        so a gate that would hang off the map there keeps its const: the clamp
        doesn't pull an overhanging footprint in. A single drag doesn't ask,
        since its fit is the landing's and the resolver answers that.

        Memoized per unit and input: Esc or a refused drop bumps no unit_gen,
        so the key has to name the unit and `group`, not just the turn."""
        turn = self._drag_turn
        unit = entry.unit
        if not turn:
            return unit
        memo_key = (turn, self.scenario.unit_gen, id(unit), group)
        memo = self._drag_stand_in_memo
        if memo_key in memo:
            return memo[memo_key]
        const, rotation = move_turn.turned(turn, unit.unit_const, unit.rotation)
        swapped = const != unit.unit_const
        if (const, rotation) == (unit.unit_const, unit.rotation) or (
            group and swapped and not self._gate_cycle_fits(unit, turn.gate)
        ):
            stand_in = unit
        else:
            stand_in = unit_at(unit, *(reanchored(unit, const) if swapped else (unit.x, unit.y)))
            stand_in.unit_const, stand_in.rotation = const, rotation
        memo[memo_key] = stand_in
        return stand_in

    def _apply_drag_turn(self, model, stand_in, unit) -> bool:
        """Writes the turn `stand_in` holds through `unit`'s one setter, both
        values taken from it, so the preview and the drop share one answer.
        Inside an open _unit_edit, after the footprint captures and before
        set_position(). Returns whether the const changed."""
        if stand_in.unit_const != unit.unit_const:
            model.set_unit_const(unit, stand_in.unit_const)
            return True
        if unit_rotation.rotation_is_angle(unit.unit_const):
            model.set_rotation(unit, stand_in.rotation)
        else:
            model.set_variant(unit, stand_in.rotation)
        return False

    def _turn_clause(self, unit, stand_in) -> str:
        """What a turn makes of `unit`: "facing 6/16", "variant 5/42", or a gate's sibling."""
        if stand_in.unit_const != unit.unit_const:
            return f"now {_unit_name(stand_in.unit_const)}"
        return self._placement_rotation_text(unit.unit_const, stand_in.rotation)

    @staticmethod
    def _turn_classes(entries) -> tuple[list, list, list]:
        """(rotatable, gates, variants) among `entries`, as on_unit_rotate picks them."""
        return (
            [e for e in entries if unit_rotation.rotation_is_angle(e.unit.unit_const)],
            [e for e in entries if gate_orientation.is_gate(e.unit.unit_const)],
            [e for e in entries if unit_variant.is_cyclable(e.unit.unit_const)],
        )

    def _turn_skips(self, entries) -> tuple[int, int]:
        """(gates that won't fit, members that can't turn) of a group under the
        drag's turn, Rotate's two skip counts."""
        rotatable, gates, variants = self._turn_classes(entries)
        steps = self._drag_turn.gate
        blocked = sum(
            gate_orientation.cycle_const(e.unit.unit_const, steps) != e.unit.unit_const
            and not self._gate_cycle_fits(e.unit, steps)
            for e in gates
        )
        return blocked, len(entries) - len(rotatable) - len(gates) - len(variants)

    def _turn_unit_drag(self, turn: move_turn.MoveTurn) -> None:
        """A Rotate input on a live Move drag (GH #280). Nothing is written and
        no cache touched until the release: the turn goes on the ghost and into
        the status line, and on_unit_move lands it with the move, so Esc or a
        refused drop leaves the unit as it stands. Turned in place by Rotate's
        rules, per const (move_turn.turned()); only the group member's map-edge
        rule depends on the drag (_drag_stand_in())."""
        self._drag_turn = turn
        self.map_view.refresh_unit_drag_ghost()
        key = self.map_view._unit_drag_key
        index = self.map_view._unit_index
        entry = index.entry_for_key(key) if index is not None else None
        if entry is None:
            return
        if self._is_group_key(key):
            entries = self._drag_entries(key, entry)
            turning = sum(self._drag_stand_in(e, True) is not e.unit for e in entries)
            blocked, skipped = self._turn_skips(entries)
            status = f"Move: turning {turning} of {len(entries)} units"
            if blocked:
                status += f"; {blocked} gate(s) skipped (would hang off the map edge)"
            if skipped:
                status += f"; {skipped} skipped (no facing or variant to change)"
            self._log_status(status)
            return
        const = entry.unit.unit_const
        stand_in = self._drag_stand_in(entry, False)
        if gate_orientation.is_gate(const):
            self._log_status(f"Move: {_unit_name(stand_in.unit_const)}")
        elif unit_variant.is_rotation_editable(const):
            self._log_status(f"Move: {_unit_name(const)} {self._placement_rotation_text(const, stand_in.rotation)}")
        else:
            self._log_status(f"Move: {_unit_name(const)} has no facing or variant to rotate")

    def _group_delta(self, anchor, pos, modifiers, commit: bool):
        """(entries, dx, dy, clamped) for a group drag grabbing `anchor`, or
        None. Snapped moves by whole tiles from the press's cover tile, so
        every member keeps its parity; off-grid moves the anchor onto the point,
        as a single off-grid move does. Both measure each member as the
        drag's turn leaves it (_drag_stand_in()), a gate's new span included."""
        index = self.map_view._unit_index
        if index is None:
            return None
        entries = [e for e in (index.entry_for_key(k) for k in self._selection) if e is not None]
        if not entries:
            return None
        pos = self.map_view._clamped_to_map_rect(pos)
        point, fell_back = self._resolve_placement_point(pos, modifiers)
        if commit and fell_back:
            self._log_status(OFF_GRID_FALLBACK_MESSAGE)
        off_grid = self._off_grid_requested(modifiers)
        if point is None:
            # Off the diamond in Stepped/Sloped: the ground-plane point, which the clamp then pulls in.
            ground = self.map_view.ground_map_point(pos)
            if ground is None:
                return None
            point = ground if off_grid else (math.floor(ground[0]) + 0.5, math.floor(ground[1]) + 0.5)
        unit = self._drag_stand_in(anchor, True)
        if off_grid:
            dx, dy = point[0] - unit.x, point[1] - unit.y
        else:
            grab = self._drag_grab_tile or (int(unit.x), int(unit.y))
            dx, dy = math.floor(point[0]) - grab[0], math.floor(point[1]) - grab[1]
        mm = self.scenario.map_manager
        dx, dy, clamped = clamp_group_delta(
            [self._drag_stand_in(e, True) for e in entries], dx, dy, mm.map_width, mm.map_height, whole_tiles=not off_grid
        )
        return entries, dx, dy, clamped

    def _move_group(self, anchor, pos, modifiers) -> None:
        """on_unit_move for a key in a 2+ selection: every member moves by the
        same clamped delta, in one undo record (GH #75).

        A turn the drag holds lands on each member at its own spot, then the
        delta (GH #280). With a zero delta that is a turn in place: only the
        turned members are written, nobody's position is, occupants stay out
        of it, and only a member whose const changed is judged by the rules."""
        resolved = self._group_delta(anchor, pos, modifiers, commit=True)
        if resolved is None:
            return
        entries, dx, dy, clamped = resolved
        stand_ins = {id(e.unit): s for e in entries if (s := self._drag_stand_in(e, True)) is not e.unit}
        in_place = (dx, dy) == (0, 0)
        if in_place and not stand_ins:
            if clamped:
                self._log_status(f"{len(entries)} units already at the map edge; nothing moved")
            return
        model = self._ensure_unit_edits()
        if model is None:
            return
        selected, judged = entries, entries
        blocked = skipped = 0
        if in_place:
            blocked, skipped = self._turn_skips(selected)
            entries = [e for e in entries if id(e.unit) in stand_ins]
            judged = [e for e in entries if stand_ins[id(e.unit)].unit_const != e.unit.unit_const]
            label = "Rotate unit" if len(entries) == 1 else f"Rotate {len(entries)} units"
        elif stand_ins:
            label = f"Move and rotate {len(entries)} units"
        else:
            label = f"Move {len(entries)} units"
        # One op: _move_units' own same-label op is transparent inside it.
        with perf_trace.op(label):
            # GH #194: one plan for the refusal and the move (TASK-352); only its removals unblock.
            planned = self._plan_unit_move(model, entries, dx, dy, stand_ins, occupants=not in_place)
            if not self._restrictions_lifted():
                replaced = self._gate_move_replaced(planned[1])
                refused = self._group_move_refusal(selected, dx, dy, replaced, stand_ins, judged)
                if refused is not None:
                    self._log_status(refused, level=debug_log.WARNING)
                    return
            gate_plan = self._move_units(model, entries, dx, dy, label, planned, stand_ins=stand_ins)
        clauses = self._gate_move_clauses(gate_plan, [e.unit for e in entries])
        if in_place:
            self._log_status(self._rotate_status(*self._turn_classes(entries), blocked, skipped) + clauses)
            return
        status = f"Moved {len(entries)} units by ({dx:g}, {dy:g})"
        if stand_ins:
            status += f", turning {len(stand_ins)}"
        if clamped:
            status += ", stopped at the map edge"
        self._log_status(status + clauses)

    def _group_move_refusal(
        self, entries, dx: float, dy: float, replaced=(), stand_ins=None, judged=None
    ) -> str | None:
        """Why placement_rules refuses moving every entry by (dx, dy), as a
        status line, or None (GH #176). All or nothing, so a formation never
        tears. The members never block each other: they keep their spacing.
        `replaced` is the walls a moved gate's plan removes (GH #194), which
        do not block either.

        `stand_ins`, {id(unit): stand-in}, are the members a drag turns, read
        for their position and const (GH #280), and `judged` narrows who is
        asked; `entries` stays the whole selection, for the shared grid.

        Counted with move_refused() through the preview's own grid; only the
        first refused member's message is built (TASK-327), since a terrain
        message costs a language-file parse."""
        grid = self._group_grid(entries)
        judged = entries if judged is None else judged
        turned = stand_ins or {}
        laid = self._group_laid(entries, dx, dy, lambda e: turned.get(id(e.unit), e.unit))

        def landing(unit):
            source = turned.get(id(unit), unit)
            return source.x + dx, source.y + dy

        def landing_const(unit) -> int:
            return turned.get(id(unit), unit).unit_const

        refused = [
            entry.unit
            for entry in judged
            if placement_rules.move_refused(
                self.scenario,
                entry.unit,
                *landing(entry.unit),
                exclude=replaced,
                grid=grid,
                unit_const=landing_const(entry.unit),
                laid=laid,
            )
        ]
        if not refused:
            return None
        first = refused[0]
        reason = placement_rules.move_refusal(
            self.scenario,
            first,
            *landing(first),
            exclude=replaced,
            grid=grid,
            unit_const=landing_const(first),
            laid=laid,
        )
        return (
            f"Move: {len(refused)} of {len(judged)} units can't go there (first: {reason}). "
            "Tick Free placement to move them anyway."
        )

    def _group_laid(self, entries, dx: float, dy: float, stand_in_of) -> dict | None:
        """The laid map a group move by (dx, dy) is judged on (TASK-483): a
        bridge in the group lays its deck where it lands, not where it stood
        (placement_rules.moved_laid_terrain()). None, the per-edit memo, when
        no member lays a foundation. `stand_in_of(entry)` is the unit or turn
        whose point the member moves from."""
        if not any(e.unit.unit_const in placement_rules.FOUNDATION_TERRAIN for e in entries):
            return None
        moved = {}
        for entry in entries:
            source = stand_in_of(entry)
            moved[id(entry.unit)] = (source.x + dx, source.y + dy)
        return placement_rules.moved_laid_terrain(self.scenario, moved)

    def _move_occupant_to(self, model, entry, x: float, y: float, splices: list) -> None:
        """One garrisoned unit onto (x, y), inside its host's own open
        _unit_edit. `z` stays the occupant's own: the host's move doesn't
        change the ground under it, and the two are already co-located.
        Scatter's move mode also writes each host through here (GH #105)."""
        unit = entry.unit
        mm = self.scenario.map_manager
        old_bounds = unit_tile_bounds(unit, mm.map_width, mm.map_height)
        old_own, old_tiles = self._unit_footprint(unit)
        idx = self._unit_list_index(entry.player_id, unit)
        model.set_position(unit, x, y, unit.z)
        new_own, new_tiles = self._unit_footprint(unit)
        splices.append(UnitSplice(entry.player_id, idx, unit, old_own, new_own, old_tiles, new_tiles))
        self._patch_unit_index_for_move(entry.player_id, unit, old_bounds)

    def _garrison_occupant_entries(self, model, entries) -> list:
        """Every unit garrisoned inside one of `entries`, as UnitEntry rows,
        deduped against `entries` itself (GH #42).

        Occupants are hidden by default now, so they cannot be selected
        alongside their host: a move or a delete that left them behind would
        strand them at the host's old point, or dangle their
        garrisoned_in_id. Every such op expands its selection through here
        first.

        A hidden occupant has no index entry to return, so one is
        synthesized with `order=-1`: these rows are only ever read for
        player_id/unit (the move loops and the delete pre-flight), never
        appended to a UnitIndex, and the real entry is reused when Show
        Garrisoned is on. The walk is transitive -- an occupant may itself
        hold a garrison -- and identity-keyed, since Unit isn't hashable.
        """
        model.warm_garrison_map()
        seen = {id(e.unit) for e in entries}
        pending = [e.unit for e in entries]
        occupants = []
        while pending:
            for occupant in _garrison_referrers(model, pending.pop()):
                if id(occupant) in seen:
                    continue
                seen.add(id(occupant))
                occupants.append(occupant)
                pending.append(occupant)
        if not occupants:
            return []
        owners = {id(u): pid for pid, units in enumerate(self.scenario.unit_manager.units) for u in units}
        index = self.map_view._unit_index
        rows = []
        for unit in occupants:
            player_id = owners[id(unit)]
            entry = index.entry_for_key(unit_pick.unit_key(player_id, unit)) if index is not None else None
            rows.append(entry or unit_pick.UnitEntry(player_id, unit, int(unit.x), int(unit.y), -1))
        return rows

    def _plan_unit_move(
        self, model, entries, dx: float, dy: float, stand_ins=None, occupants: bool = True
    ) -> tuple[list, wall_run.MovePlan | None]:
        """(entries plus their garrison occupants, _move_plan() for moving
        them all by (dx, dy)): what _move_units() moves and applies.
        `stand_ins` is _move_units()'s; `occupants` False is a turn in place,
        which carries nobody."""
        if occupants:
            with perf_trace.phase("move_garrison"):
                entries = list(entries) + self._garrison_occupant_entries(model, entries)
        turned = stand_ins or {}
        sources = [(e, turned.get(id(e.unit), e.unit)) for e in entries]
        return entries, self._move_plan(
            model,
            [(e, source.x + dx, source.y + dy) for e, source in sources],
            gate_walls=True,
            consts={id(e.unit): source.unit_const for e, source in sources if source is not e.unit},
        )

    def _move_units(
        self, model, entries, dx: float, dy: float, label: str, planned=None, dz: float = 0.0, stand_ins=None
    ) -> wall_run.MovePlan | None:
        """Moves every entry by (dx, dy) in ONE undo record, splicing each
        unit's footprint. `rotation` passes through verbatim, except a wall's
        shape index the move re-derives (GH #266) and a turn in `stand_ins`,
        and so does `z` unless a group Z step (GH #154) passes a non-zero `dz`.

        A host's garrison rides along (GH #42): its occupants sit at its own
        point and cannot be selected while hidden, so the same delta applies
        to them in the same undo record.

        `stand_ins`, {id(unit): stand-in}, are the members a Move drag turned
        (GH #280; never a hidden occupant, an arrow nudge or Scatter). Each
        takes its turn first, then the delta. With a zero delta they take the
        turn only, and no position is rewritten.

        Returns _move_plan()'s answer for the caller's status line (GH #194,
        GH #266), applied here: a `wholesale` one as one wholesale record,
        any other truthy one inside the splice edit (_apply_wall_reshapes()).
        `planned` is _plan_unit_move()'s answer for these same arguments, when
        the caller already has it (TASK-352); None plans here."""
        turned = stand_ins or {}
        # A turn in place is the only call with a zero delta and a member to skip.
        moves = bool(dx or dy or dz)
        # Opened here so the garrison walk lands in the op; _unit_edit's same-label op is transparent.
        with perf_trace.op(label):
            if planned is None:
                planned = self._plan_unit_move(model, entries, dx, dy, turned)
            entries, gate_plan = planned
            players = sorted({e.player_id for e in entries} | (wall_run.move_plan_players(gate_plan) if gate_plan else set()))
            if gate_plan is not None and gate_plan.wholesale:
                # GH #194: removals and fills shift list indices, so no splices; see on_unit_move.
                with self._unit_edit(model, label, players):
                    model.remove_many([wall for _player_id, wall in gate_plan.removals])
                    for entry in entries:
                        unit = entry.unit
                        stand_in = turned.get(id(unit))
                        if stand_in is not None:
                            self._apply_drag_turn(model, stand_in, unit)
                        if stand_in is None or moves:
                            # Guarded: unit.z + 0.0 would turn a stored -0.0 into 0.0.
                            model.set_position(unit, unit.x + dx, unit.y + dy, unit.z + dz if dz else unit.z)
                    wall_run.apply_move_plan(model, gate_plan)
                return gate_plan
            mm = self.scenario.map_manager
            splices: list[UnitSplice] = []
            rebuild = False
            with self._unit_edit(model, label, players, splices, fields_only=True):
                for entry in entries:
                    unit = entry.unit
                    old_bounds = unit_tile_bounds(unit, mm.map_width, mm.map_height)
                    old_own, old_tiles = self._unit_footprint(unit)
                    with perf_trace.phase("move_list_index"):
                        idx = self._unit_list_index(entry.player_id, unit)
                    # After the captures: a splice's old tiles are the stored footprint.
                    stand_in = turned.get(id(unit))
                    swapped = stand_in is not None and self._apply_drag_turn(model, stand_in, unit)
                    rebuild = rebuild or swapped
                    position = stand_in is None or moves
                    if position:
                        model.set_position(unit, unit.x + dx, unit.y + dy, unit.z + dz if dz else unit.z)
                    new_own, new_tiles = self._unit_footprint(unit)
                    splices.append(UnitSplice(entry.player_id, idx, unit, old_own, new_own, old_tiles, new_tiles))
                    # Not scoped to a span change, so a swapped const rebuilds below instead.
                    if position and not swapped:
                        with perf_trace.phase("move_pick_patch"):
                            self._patch_unit_index_for_move(entry.player_id, unit, old_bounds)
                if gate_plan:
                    self._apply_wall_reshapes(model, gate_plan, splices)
                if rebuild:
                    self._rebuild_unit_index()
        return gate_plan

    def on_unit_drag_preview(self, key: tuple[int, int], pos, modifiers) -> None:
        """MapView's fourteenth injected callable -- one mouse-move of an
        in-progress unit drag, past UNIT_DRAG_THRESHOLD_PX (the mid-drag move
        preview). `key` is resolved late through the live index, same as
        on_unit_move's.

        Resolving lives here rather than in MapView because MapView holds no
        scenario: it has no player_colors, no team_indices and no sprites
        toggle. The toggle in particular must be read from
        self._sprites_enabled -- the WINDOW's copy -- and never off the live
        cache, which is a derived mirror rebuilt from scratch on every style
        switch and file open.

        Every failure to resolve clears the ghost rather than leaving a stale
        one at the last destination that DID resolve: a preview frozen a few
        tiles behind the cursor is a worse lie than no preview.

        The host a release here would garrison into is outlined (GH #115),
        set or cleared once up front so no ghost exit path leaves it stale.

        A ghost the release would refuse is drawn refused (TASK-327), asked
        in the release's own branch order (_single_drag_refused(),
        _group_refused_ids()), so the two cannot disagree."""
        if self.scenario is None:
            self.map_view.set_unit_drop_target(None)
            self.map_view.set_unit_ghost()
            return
        index = self.map_view._unit_index
        entry = index.entry_for_key(key) if index is not None else None
        if entry is None:
            self.map_view.set_unit_drop_target(None)
            self.map_view.set_unit_ghost()
            return
        # A group release clamps to the map rect (map_view), so the outline must test the same point.
        drop_pos = self.map_view._clamped_to_map_rect(pos) if self._is_group_key(key) else pos
        drop = self._garrison_drop(self._drag_entries(key, entry), drop_pos, modifiers, preview_key=key)
        self.map_view.set_unit_drop_target(drop[0] if drop is not None and drop[1] is None else None)
        if self._is_group_key(key):
            self._preview_group(key, entry, pos, modifiers, drop)
            return
        # The same resolver on_unit_move commits through, so the preview
        # cannot lie about where the unit lands -- including under the
        # Off-grid checkbox and its held-Alt one-off. Its fallback status
        # line is deliberately dropped here; see _resolve_placement_point.
        stand_in = self._drag_stand_in(entry, False)
        point, _fell_back = self._resolve_placement_point(pos, modifiers, unit_const=stand_in.unit_const)
        if point is None:
            self.map_view.set_unit_ghost()
            return
        ghost = unit_at(stand_in, *point)
        refused = self._single_drag_refused(key, entry, stand_in, point, drop)
        draws = self._drag_ghost_sprite_draws(entry, ghost)
        if draws:
            self.map_view.set_unit_ghost(draws=draws, refused=refused)
            return
        polygons, color = self._ghost_mark(entry.player_id, ghost)
        self.map_view.set_unit_ghost(polygons=polygons, color=color, refused=refused)

    def _single_drag_refused(self, key, entry, stand_in, point: tuple[float, float], drop) -> bool:
        """Whether on_unit_move would refuse dropping `entry` at `point`, in
        its order: a host that takes it never refuses and one that does not
        always does, Free placement or not; a drop back on its own point with
        its const unchanged has no footprint to judge; Free placement lifts
        the rule. `stand_in` is the drag's turn (_drag_stand_in()). The rule's
        answer alone is memoized, per tile crossing on a snapped drag."""
        if drop is not None:
            return drop[1] is not None
        unit = entry.unit
        const = stand_in.unit_const
        own_point = point == (stand_in.x, stand_in.y) and const == unit.unit_const
        if own_point or self._restrictions_lifted() or self._moves_gate_walls([entry]):
            return False
        memo_key = (key, point, const, self.scenario.unit_gen)
        memo = self._drag_rule_memo
        if memo is not None and memo[0] == memo_key:
            return memo[1]
        refused = placement_rules.move_refused(self.scenario, unit, *point, exclude=[unit], unit_const=const)
        self._drag_rule_memo = (memo_key, refused)
        return refused

    def _moves_gate_walls(self, entries) -> bool:
        """Whether a drop of `entries` may replace walls under a moved gate
        (GH #194). The release then excludes those walls through the unit
        model's gate plan, which a preview does not build, so it is drawn
        untinted rather than refused over walls the drop would replace."""
        return self.gate_walls_check.isChecked() and any(gate_orientation.is_gate(e.unit.unit_const) for e in entries)

    def _ghost_mark(self, player_id: int, ghost, ghost_entry=None):
        """(polygons, color) of the coloured-mark ghost for `player_id`'s `ghost`."""
        if ghost_entry is None:
            ghost_entry = _ghost_entry(player_id, ghost)
        return (
            self.map_view._unit_polygons_for(ghost_entry),
            unit_mark_color(ghost, self.scenario.player_colors[player_id]),
        )

    def _preview_group(self, key, anchor, pos, modifiers, drop=None) -> None:
        """One ghost per group member at the same clamped delta the commit
        uses. Past GROUP_GHOST_CAP members only the anchor is ghosted.

        Only the members the release would refuse are drawn refused (TASK-327),
        though a refused group moves nothing. Past the cap the anchor's ghost
        stands for the members checked (_group_refused_ids())."""
        resolved = self._group_delta(anchor, pos, modifiers, commit=False)
        if resolved is None:
            self.map_view.set_unit_ghost()
            return
        entries, dx, dy, _clamped = resolved
        members = entries
        capped = len(entries) > GROUP_GHOST_CAP
        if capped:
            members = [anchor]
            # Once per drag: this runs on every mouse-move.
            if self._ghost_cap_logged != key:
                self._ghost_cap_logged = key
                self._log_status(
                    f"Moving {len(entries)} units "
                    "(preview shows the grabbed one only; placement is checked in full on release)"
                )
        refused_ids = self._group_refused_ids(entries, anchor, dx, dy, drop)
        mm = self.scenario.map_manager
        draws: list = []
        marks: list = []
        refused_draws: list = []
        refused_marks: list = []
        for entry in members:
            stand_in = self._drag_stand_in(entry, True)
            ghost = unit_at(stand_in, stand_in.x + dx, stand_in.y + dy)
            # A member nudged off-map stays off-map, and has nothing to draw.
            if unit_tile_bounds(ghost, mm.map_width, mm.map_height) is None:
                continue
            refused = bool(refused_ids) if capped else id(entry.unit) in refused_ids
            sprite = self._drag_ghost_sprite_draws(entry, ghost)
            if sprite:
                (refused_draws if refused else draws).extend(sprite)
            else:
                (refused_marks if refused else marks).append(self._ghost_mark(entry.player_id, ghost))
        self.map_view.set_unit_ghosts(draws=draws, marks=marks, refused_draws=refused_draws, refused_marks=refused_marks)

    def _group_refused_ids(self, entries, anchor, dx: float, dy: float, drop) -> set[int]:
        """id(unit) of every member _move_group() would refuse at (dx, dy), in
        on_unit_move's branch order: a host's answer first (every member when
        it refuses, Free placement or not; none when it takes them), then a
        zero delta and Free placement (none). With a turn, a zero delta still
        judges the members whose const changed, as _move_group() does. Up to
        GROUP_GHOST_CAP members are checked, the grabbed one first; the release
        checks all. Empty for a drag that may replace walls (_moves_gate_walls()).

        The answer is memoized per delta and turn through one BlockerGrid per
        drag (_group_grid()), so a snapped drag asks once per tile crossing."""
        if drop is not None:
            return {id(e.unit) for e in entries} if drop[1] is not None else set()
        turn = self._drag_turn
        in_place = (dx, dy) == (0, 0)
        if (in_place and not turn) or self._restrictions_lifted():
            return set()
        gen, box = self.scenario.unit_gen, self.gate_walls_check.isChecked()
        memo = self._group_flags_memo
        if memo is not None and memo[0] is self._selection and memo[1:6] == (dx, dy, gen, box, turn):
            return memo[6]
        ids: set[int] = set()
        # Inside the memo: the scan for a gate runs once per delta, not per mouse move.
        if not self._moves_gate_walls(entries):
            pairs = [(e, self._drag_stand_in(e, True)) for e in entries]
            if in_place:
                pairs = [(e, stand_in) for e, stand_in in pairs if stand_in.unit_const != e.unit.unit_const]
            if len(pairs) > GROUP_GHOST_CAP:
                # Stable, so the grabbed member leads and the rest keep their order.
                pairs = sorted(pairs, key=lambda pair: pair[0].unit is not anchor.unit)[:GROUP_GHOST_CAP]
            grid = self._group_grid(entries)
            # The whole group lands, capped or not, so every member's landing lays its deck.
            laid = self._group_laid(entries, dx, dy, lambda e: self._drag_stand_in(e, True))
            ids = {
                id(e.unit)
                for e, stand_in in pairs
                if placement_rules.move_refused(
                    self.scenario,
                    e.unit,
                    stand_in.x + dx,
                    stand_in.y + dy,
                    grid=grid,
                    unit_const=stand_in.unit_const,
                    laid=laid,
                )
            }
        self._group_flags_memo = (self._selection, dx, dy, gen, box, turn, ids)
        return ids

    def _group_grid(self, entries) -> placement_rules.BlockerGrid:
        """The group drag's BlockerGrid, built once per selection and unit
        edit with every selected unit excluded, so members never block each
        other (GH #176). The preview and the drop share it (TASK-327)."""
        gen = self.scenario.unit_gen
        memo = self._group_grid_memo
        if memo is not None and memo[0] is self._selection and memo[1] == gen:
            return memo[2]
        grid = placement_rules.BlockerGrid(self.scenario, exclude=[e.unit for e in entries])
        self._group_grid_memo = (self._selection, gen, grid)
        return grid

    def _ghost_sprite_draws(self, player_id: int, ghost, rotation_override: float | None = None) -> list:
        """`ghost`'s real sprite pieces for owner `player_id`, or [] to fall
        back to the coloured mark. The drag passes its wall-connectivity
        `rotation_override`; Place Unit's hover ghost passes None.

        **Flat is always the mark**, never a sprite. Flat does draw sprites
        now, as footprint-fitted icons, but a Flat ghost with an icon is out
        of scope here (GH #53 Part B left it as the mark)."""
        if not self._ghost_sprites_on():
            return []
        view = self.map_view
        cache = view._sloped_cache()
        return unit_sprite_draws_at(
            self.scenario,
            view._iso_proj,
            view._iso_elevations,
            None if cache is None else cache.corner_rise,
            player_id,
            ghost,
            rotation_override=rotation_override,
            # The window's live value, not the default: a ghost left on 1.0
            # would drag a tree at full size and snap it small on drop.
            tree_scale=self._layers.tree_scale,
            hero_glow=self._layers.hero_glow,
            wall_alpha=self._layers.wall_alpha,
            fold=None if cache is None else cache.fold,
        )

    def _ghost_sprites_on(self) -> bool:
        """Whether a ghost can draw sprites at all, or is always the mark."""
        view = self.map_view
        if not self._sprites_enabled or view._terrain_style == "flat":
            return False
        return view._iso_proj is not None and view._iso_elevations is not None

    def _drag_ghost_sprite_draws(self, entry, ghost) -> list:
        """_ghost_sprite_draws for a dragged entry. The override is resolved only
        when a sprite can draw, so a Flat drag never pays its walk."""
        if not self._ghost_sprites_on():
            return []
        return self._ghost_sprite_draws(entry.player_id, ghost, self._ghost_rotation_override(entry))

    def _ghost_rotation_override(self, entry) -> float | None:
        """This unit's wall-connectivity-derived rotation, resolved once per
        drag rather than per mouse-move.

        Two walks sit behind it that must not run per frame: the memo behind
        wall_variant_rotation_overrides() rebuilds over every unit in the file
        whenever the unit generation bumps, and _unit_list_index() is a linear
        scan by identity. The cache is keyed on the overrides dict OBJECT, not
        on a generation number -- the memo hands back a new dict whenever it
        rebuilds, so an `is` test against the one this value was derived from
        is exactly "still the same answer", with nothing to keep in sync.

        Boundary worth knowing during an in-app pass: this is the override the
        unit's SOURCE neighbours give it. A wall dropped somewhere with
        different neighbours has its index re-derived by the move's plan on
        release (GH #266, _move_plan()), so the ghost can show the wall shape
        it has now rather than the one it lands as. Recomputing connectivity
        per move is the whole-file walk this preview exists to avoid."""
        overrides = wall_variant_rotation_overrides(self.scenario)
        key = unit_pick.unit_key(entry.player_id, entry.unit)
        cached = self._ghost_rotation
        if cached is None or cached[0] is not overrides:
            cached = (overrides, {})
            self._ghost_rotation = cached
        memo = cached[1]
        # Keyed per unit so a group drag (GH #75) scans each member once, not once per frame.
        if key not in memo:
            index = self._unit_list_index(entry.player_id, entry.unit)
            memo[key] = overrides.get((entry.player_id, index))
        return memo[key]

    def on_unit_nudge(self, dx: int, dy: int, modifiers) -> bool:
        """MapView's ninth injected callable -- an arrow key in Units mode
        (b1.5's second half, generalized to the whole selection by b2.4).

        **The return value is load-bearing, not incidental.** True means the
        nudge happened and owns the key; False means it refused, and MapView
        falls through to a view_pan_* pan on the same press (see its
        keyPressEvent). Every early return below is a case where panning is
        the right answer, "nothing is selected" most of all.
        The step stops at the map edge as a group drag does (GH #75,
        TASK-468): if any member would leave, nothing moves, and that refusal
        still returns True, so a held arrow never turns into a pan at the edge.
        One record per press, matching every other discrete-keypress edit in
        this app (e.g. the ]/[ tool-value step) -- not merged across
        repeats, unlike a typed-digit spinbox edit. That holds for a group
        nudge too: every selected unit moves in the SAME undo step, not one
        step per unit, since begin_unit_edit(players) already captures every
        touched player's whole list up front (unit_model.py's own
        PlayerListSnapshot)."""
        if self.scenario is None or not self._selection:
            return False
        index = self.map_view._unit_index
        if index is None:
            return False
        entries = [e for e in (index.entry_for_key(k) for k in self._selection) if e is not None]
        if not entries:
            return False
        model = self._ensure_unit_edits()
        if model is None:
            return False
        whole_tiles = bool(modifiers & Qt.ShiftModifier)
        step = _UNIT_NUDGE_STEP_SHIFT if whole_tiles else _UNIT_NUDGE_STEP
        mm = self.scenario.map_manager
        _cx, _cy, clamped = clamp_group_delta(
            [e.unit for e in entries], dx * step, dy * step, mm.map_width, mm.map_height, whole_tiles=whole_tiles
        )
        if clamped:
            # Whole steps only: a step that only partly fits moves nothing.
            if len(entries) == 1:
                self._log_status(f"{_unit_name(entries[0].unit.unit_const)} already at the map edge; nothing moved")
            else:
                self._log_status(f"{len(entries)} units already at the map edge; nothing moved")
            return True
        label = "Nudge unit" if len(entries) == 1 else f"Nudge {len(entries)} units"
        clauses = self._gate_move_clauses(
            self._move_units(model, entries, dx * step, dy * step, label), [e.unit for e in entries]
        )
        if len(entries) == 1:
            unit = entries[0].unit
            self._log_status(f"Moved {_unit_name(unit.unit_const)} to ({unit.x:g}, {unit.y:g}){clauses}")
        else:
            self._log_status(f"Moved {len(entries)} units{clauses}")
        return True
