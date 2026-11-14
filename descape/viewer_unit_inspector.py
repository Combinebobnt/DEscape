"""ViewerWindow's unit inspector: unit field edits, the rotation drag, selection reconcile and the garrison block."""

from __future__ import annotations

import math

from AoE2ScenarioParser.exceptions.asp_exceptions import UnsupportedAttributeError
from PyQt5.QtWidgets import (
    QDialog,
)

from descape import (
    debug_log,
    garrison,
    gate_orientation,
    object_catalog,
    unit_fields,
    unit_kind,
    unit_pick,
    unit_rotation,
    unit_variant,
    wall_run,
)
from descape.constant_picker import CatalogBrowseDialog
from descape.render_cache_splice import UnitSplice
from descape.render_units import unit_tile_bounds
from descape.unit_filter import GAIA_PLAYER_ID, UnitFilter
from descape.unit_model import (
    capture_flag_of,
)
from descape.viewer_common import (
    _gaia_only_refused_note,
    _garrison_referrers,
    _RotationDrag,
    _unit_name,
    clamp_group_delta,
    tool_applicable,
)


def clamp_level_delta(values, delta: float, lo: float, hi: float) -> tuple[float, bool]:
    """(delta, clamped): a whole-step Z shift pulled into the range that keeps
    every value already inside lo..hi there (GH #154). A value outside does
    not limit it, as an off-map member does not limit clamp_group_delta()."""
    inside = [value for value in values if lo <= value <= hi]
    if not inside:
        return delta, False
    low = min(0, math.ceil(lo - min(inside)))
    high = max(0, math.floor(hi - max(inside)))
    clamped = float(min(max(delta, low), high))
    return clamped, clamped != delta


# TASK-359: every mode keeps the unit selection; these draw it and let Scatter read it.
# Elsewhere it is dormant: kept, not drawn (Triggers' Pick from map borrows the layer).
_UNIT_SELECTION_LIVE_MODES = frozenset({"units", "terrain"})


class _ViewerUnitInspectorMixin:
    """ViewerWindow's unit inspector and garrison block (viewer.py's docstring has the split's rules)."""

    def _on_unit_field_changed(self, spec: unit_fields.UnitFieldSpec, value) -> None:
        """The single funnel every unit inspector editor reports through --
        mirrors set_trigger_field()'s role for the trigger form.

        Applies to every selected unit (GH #71): X/Y/Z and Owner set all of
        them to the typed value, and Rotation sets every ANGLE member to one
        facing, skipping the rest, or with no ANGLE member every cyclable one
        to one variant (GH #123), or with only gates every gate to one
        orientation (GH #61). Unit State and Capture Flag (GH #186) set every
        member to the picked raw value. A one-unit selection is a group of one, so
        both cases share this path, and each edit is one undo record. A group's
        X/Y/Z step, and a step on a "(mixed)" Rotation, go through
        _on_unit_field_stepped instead (GH #154, TASK-548).

        UnitsPanel itself suppresses this during a programmatic populate (its
        own `_populating` guard, mirroring PlayersPanel._changed()'s), so
        this only ever fires for a genuine user edit -- but a genuinely
        no-op edit (typing the same X back) must still not record one,
        hence the per-unit equality checks below.
        """
        if not self._selection:
            return
        model = self._ensure_unit_edits()
        if model is None:
            self._update_unit_inspector_from_selection()
            return
        index = self.map_view._unit_index
        if index is None:
            return
        entries = [e for e in (index.entry_for_key(k) for k in self._selection) if e is not None]
        if not entries:
            return
        single = len(entries) == 1

        if spec.field_id == "player":
            new_player = int(value)

            def reselect(old_to_new) -> None:
                # Inside the block: _unit_edit's exit reconciles the selection
                # and would drop the stale (old_player, ref) keys. A refused
                # GAIA-only entry (GH #32) keeps its own key.
                self._selection = [old_to_new.get(key, key) for key in self._selection]

            changed, carried, _refused, gaia_only = self._reassign_units(entries, new_player, on_applied=reselect)
            owner = "GAIA" if new_player == GAIA_PLAYER_ID else f"Player {new_player}"
            if not changed and not carried:
                if gaia_only:
                    # The Owner field already shows the refused player: put it back.
                    self._update_unit_inspector_from_selection()
                    self._log_status(f"Not reassigned to {owner}{_gaia_only_refused_note(gaia_only)}", level=debug_log.WARNING)
                return
            suffix = f" ({carried} garrisoned)" if carried else ""
            suffix += _gaia_only_refused_note(gaia_only)
            if single:
                self._log_status(f"Reassigned unit to {owner}{suffix}")
            else:
                self._log_status(f"Reassigned {changed} units to {owner}{suffix}")
            return

        if spec.field_id == "rotation":
            # The panel shows the same mode for this selection (GH #123): a
            # facing if any member is ANGLE, else a variant number.
            mode = unit_variant.rotation_field_mode(e.unit.unit_const for e in entries)
            if mode is None:
                return
            if mode == "variant":
                self._set_variant_field(model, entries, int(value), single)
                return
            if mode == "gate":
                self._set_gate_field(model, entries, int(value), single)
                return
            # Facing mode: VARIANT/INERT members are skipped, their rotation
            # passed through verbatim (AGENTS.md hard rule); set_rotation raises.
            angle_entries = [e for e in entries if unit_rotation.rotation_is_angle(e.unit.unit_const)]
            skipped = len(entries) - len(angle_entries)
            if not angle_entries:
                return
            # `value` is a whole facing (GH #61) on the panel's scale: the
            # shared direction count, or the largest when members differ.
            scale = unit_rotation.facing_scale(e.unit.unit_const for e in angle_entries)
            changes = []
            for entry in angle_entries:
                count = unit_rotation.angle_count_for(entry.unit.unit_const)
                facing = unit_rotation.snap_facing(int(value), scale, count)
                # Already showing this facing: no write, so an off-grid stored
                # value is only snapped by a real edit.
                if unit_rotation.rotation_to_facing(entry.unit.rotation, count) == facing:
                    continue
                rotation = unit_rotation.facing_to_rotation(facing, count)
                if rotation != entry.unit.rotation:
                    changes.append((entry, rotation))
            if not changes:
                return
            # A typed value stands alone even if a drag was left open.
            drag = self._rotation_drag if self.units_panel.dial_committing else None
            # In a drag one spoke may turn only the trebuchets, so count the group.
            turning = len(changes) if drag is None else len(angle_entries)
            label = "Set unit Rotation" if single else f"Set rotation on {turning} units"
            players = sorted({e.player_id for e, _ in changes})
            splices: list[UnitSplice] = []
            with self._unit_edit(model, label, players, splices, fields_only=True, drag=drag):
                self._write_facings(model, changes, splices)
            if drag is not None:
                drag.skipped = skipped
            elif skipped:
                self._log_status(f"{label} ({skipped} skipped: not rotatable)", level=debug_log.WARNING)
            return

        if spec.kind == unit_fields.CHOICE:
            # Unit State / Capture Flag (GH #186): exactly the selected units,
            # with no garrison carry, unlike a typed coordinate below.
            capture = spec.field_id == "capture_flag"
            if capture and not model.supports_capture_flag:
                # The panel disables the row before 1.59; set_capture_flag() would raise.
                return
            raw = int(value)
            changing = [e for e in entries if (capture_flag_of(e.unit) if capture else e.unit.status) != raw]
            if not changing:
                return
            label = f"Set {spec.label}" if single else f"Set {spec.label} on {len(changing)} units"
            setter = model.set_capture_flag if capture else model.set_status
            players = sorted({e.player_id for e in changing})
            splices = []
            with self._unit_edit(model, label, players, splices, fields_only=True):
                for entry in changing:
                    unit = entry.unit
                    # Nothing moves: one footprint on both sides, as Rotation's splice above.
                    own, tiles = self._unit_footprint(unit)
                    idx = self._unit_list_index(entry.player_id, unit)
                    setter(unit, raw)
                    splices.append(UnitSplice(entry.player_id, idx, unit, own, own, tiles, tiles))
            return

        axis = spec.field_id

        def axis_value(unit):
            return getattr(unit, "z", 0.0) if axis == "z" else getattr(unit, axis)

        changing = [e for e in entries if axis_value(e.unit) != value]
        if not changing:
            return
        label = f"Set unit {spec.label}" if single else f"Set {spec.label} on {len(changing)} units"
        # A typed coordinate carries the host's garrison with it (GH #42),
        # counted out of the label: the user edited the units they selected.
        # An occupant already on the typed value has nothing to write.
        changing += [e for e in self._garrison_occupant_entries(model, changing) if axis_value(e.unit) != value]
        mm = self.scenario.map_manager
        # GH #266: a typed X or Y carries walls too (no gate part, as GH #194 kept this out of the box).
        plan = None
        if axis in ("x", "y"):
            plan = self._move_plan(
                model,
                [(e, value if axis == "x" else e.unit.x, value if axis == "y" else e.unit.y) for e in changing],
                gate_walls=False,
            )
        players = sorted({e.player_id for e in changing} | (wall_run.move_plan_players(plan) if plan else set()))
        wholesale = plan is not None and plan.wholesale
        splices = []
        with self._unit_edit(model, label, players, None if wholesale else splices, fields_only=not wholesale):
            for entry in changing:
                unit = entry.unit
                x, y, z = unit.x, unit.y, getattr(unit, "z", 0.0)
                if axis == "x":
                    x = value
                elif axis == "y":
                    y = value
                elif axis == "z":
                    z = value
                old_bounds = unit_tile_bounds(unit, mm.map_width, mm.map_height)
                old_own, old_tiles = self._unit_footprint(unit)
                idx = self._unit_list_index(entry.player_id, unit)
                model.set_position(unit, x, y, z)
                new_own, new_tiles = self._unit_footprint(unit)
                splices.append(UnitSplice(entry.player_id, idx, unit, old_own, new_own, old_tiles, new_tiles))
                self._patch_unit_index_for_move(entry.player_id, unit, old_bounds)
            if wholesale:
                wall_run.apply_move_plan(model, plan)
            elif plan:
                self._apply_wall_reshapes(model, plan, splices)
        # A reshape follows _unit_edit's bare label with a fuller line, as Rotation's skips do.
        clauses = self._gate_move_clauses(plan, [e.unit for e in changing])
        if clauses:
            self._log_status(label + clauses)

    def _on_unit_field_stepped(self, spec: unit_fields.UnitFieldSpec, delta: float) -> None:
        """A group's X, Y or Z step (GH #154): every selected unit moves by
        `delta` and the formation holds, where a typed value replaces. A step
        on a "(mixed)" Rotation turns every member the same way
        (_step_rotation_field, TASK-548).

        A nudge, not a typed set: it goes through _move_units, so a host's
        garrison rides along (GH #42) and "Moving a gate updates walls" (GH
        #194) applies, with no placement-rule refusal, like the arrow-key
        nudge. A step that would push a member off the map (or Z past its
        limit) moves by as many whole steps as fit, as a group drag does
        (GH #75). One undo record per step."""
        if not self._selection:
            return
        model = self._ensure_unit_edits()
        if model is None:
            self._update_unit_inspector_from_selection()
            return
        index = self.map_view._unit_index
        if index is None:
            return
        entries = [e for e in (index.entry_for_key(k) for k in self._selection) if e is not None]
        if not entries:
            return
        if spec.field_id == "rotation":
            self._step_rotation_field(model, entries, int(delta))
            return
        mm = self.scenario.map_manager
        units = [e.unit for e in entries]
        dx = dy = dz = 0.0
        if spec.field_id == "z":
            lo, hi = unit_fields.coordinate_bounds("z", mm.map_width, mm.map_height)
            dz, clamped = clamp_level_delta([unit.z for unit in units], delta, lo, hi)
            moved, limit = dz, "at the Z limit"
        else:
            step = (delta, 0.0) if spec.field_id == "x" else (0.0, delta)
            dx, dy, clamped = clamp_group_delta(units, *step, mm.map_width, mm.map_height, whole_tiles=True)
            moved, limit = (dx if spec.field_id == "x" else dy), "at the map edge"
        count = len(entries)
        if moved == 0:
            if clamped:
                self._log_status(f"{count} units already {limit}; nothing moved")
            return
        gate_plan = self._move_units(model, entries, dx, dy, f"Shift {spec.label} on {count} units", dz=dz)
        status = f"Moved {count} units by {moved:+g} on {spec.label}"
        if clamped:
            status += f", stopped {limit}"
        self._log_status(status + self._gate_move_clauses(gate_plan, units))

    def _on_rotation_drag(self, active: bool, hidden: bool = False) -> bool:
        """The compass's press and release (TASK-548): each spoke between them
        is a finished edit that _unit_edit folds into the drag's record, so a
        drag is one undo step (TASK-023) with nothing held open across mouse
        events. Any other edit during it stands alone; the next spoke then
        finds replace_top() refusing and pushes a fresh record for the drag to
        follow.

        A press returns False when the edit is refused, which
        _ensure_unit_edits() has already explained. `hidden` is the dial hiding
        before its release: that runs inside _apply_conditional_fields, where a
        history move would nest, so the record simply stands.

        Back on the start with one record, the drag leaves no record. A real
        undo, not a value write, so an off-grid start returns verbatim. A redo
        tail from before the drag stays gone: its first spoke was an edit.
        """
        if active:
            if self._ensure_unit_edits() is None:
                return False
            self._rotation_drag = _RotationDrag(self.units_panel.unit_rotation_dial.value)
            return True
        drag, self._rotation_drag = self._rotation_drag, None
        if drag is None or drag.record is None or drag.record is not self.edit_history.peek_undo() or hidden:
            return True
        dial_value = self.units_panel.unit_rotation_dial.value
        if drag.start is not None and dial_value == drag.start and drag.pushes == 1:
            self._move_history(drag.record, self.edit_history.undo, "Undo", quiet=True)
            self.edit_history.drop_redo()
            self._update_edit_actions()
            return True
        suffix = f" ({drag.skipped} skipped: not rotatable)" if drag.skipped else ""
        self._log_status(drag.record.label + suffix)
        return True

    def _step_rotation_field(self, model, entries, steps: int) -> None:
        """A step over a "(mixed)" Rotation (TASK-548): every member turns and
        keeps its offset from the others, in whichever mode the field is in,
        where a typed value aligns them. The writers and their scopes are
        Rotate's and Cycle Variant's, and an off-grid facing moves along as
        Rotate moves it. One undo record per notch.

        A member whose turn lands on its own frame is dropped, and with none
        left there is no record: Ctrl+wheel and PageUp arrive as +-10, a whole
        cycle for a 2-direction unit or a tree with 2, 5 or 10 variants."""
        mode = unit_variant.rotation_field_mode(e.unit.unit_const for e in entries)
        blocked = 0
        if mode == "facing":
            writable = [e for e in entries if unit_rotation.rotation_is_angle(e.unit.unit_const)]
            scale = unit_rotation.facing_scale(e.unit.unit_const for e in writable)
            lead = next(e.unit for e in writable if unit_rotation.angle_count_for(e.unit.unit_const) == scale)
            lead_facing = unit_rotation.rotation_to_facing(lead.rotation, scale)
            changes = []
            for entry in writable:
                count = unit_rotation.angle_count_for(entry.unit.unit_const)
                turn = unit_rotation.member_steps(lead_facing, scale, count, steps)
                if turn % count:
                    changes.append((entry, unit_rotation.rotate_step(entry.unit.rotation, count, turn)))
            turned = [entry for entry, _rotation in changes]

            def write(splices) -> None:
                self._write_facings(model, changes, splices)
        elif mode == "variant":
            writable = [e for e in entries if unit_variant.is_cyclable(e.unit.unit_const)]
            turned = [e for e in writable if steps % unit_variant.variant_count_for(e.unit.unit_const)]

            def next_variant(unit) -> float:
                angle_count = unit_rotation.angle_count_for(unit.unit_const)
                return unit_variant.cycle_step(unit.rotation, angle_count, unit_variant.variant_count_for(unit.unit_const), steps)

            def write(splices) -> None:
                self._write_variants(model, turned, next_variant, splices)
        elif mode == "gate":
            writable = entries
            changing = [e for e in writable if steps % len(gate_orientation.orientation_siblings(e.unit.unit_const))]
            turned = [e for e in changing if self._gate_cycle_fits(e.unit, steps)]
            blocked = len(changing) - len(turned)

            def write(splices) -> None:
                self._write_gate_consts(model, turned, lambda _unit: steps, splices)
        else:
            return
        if turned:
            units = f"{len(turned)} unit{'' if len(turned) == 1 else 's'}"
            label = f"Turn {units}"
            status = f"Turned {units} by {steps:+d}"
            players = {e.player_id for e in turned}
            gate_plan = self._gate_cycle_plan(model, turned, lambda _unit: steps) if mode == "gate" else None
            if gate_plan:
                # GH #339, as on_unit_rotate: removals first, one wholesale record.
                with self._unit_edit(model, label, sorted(players | wall_run.gate_move_players(gate_plan))):
                    model.remove_many([wall for _player_id, wall in gate_plan.removals])
                    write(None)
                    wall_run.apply_gate_move_plan(model, gate_plan)
                status += self._gate_move_clauses(gate_plan)
            else:
                splices: list[UnitSplice] = []
                with self._unit_edit(model, label, sorted(players), splices, fields_only=True):
                    write(splices)
            skipped = len(entries) - len(writable)
            self._log_status(
                f"{status} ({skipped} skipped: not rotatable)" if skipped else status,
                level=debug_log.WARNING if skipped else debug_log.INFO,
            )
        if blocked:
            self._log_status(f"Set Rotation: {blocked} gate(s) skipped (would hang off the map edge)", level=debug_log.WARNING)
            # A blocked notch writes nothing, so nothing else repaints the field.
            self._update_unit_inspector_from_selection()

    def _set_variant_field(self, model, entries, variant: int, single: bool) -> None:
        """The Rotation field's variant mode: every cyclable member takes
        `variant` via set_variant(), the Cycle Variant exception (GH #123).
        Walls, cliffs and INERT members are skipped, never written."""
        cyclable = [e for e in entries if unit_variant.is_cyclable(e.unit.unit_const)]
        skipped = len(entries) - len(cyclable)
        changes = []
        for entry in cyclable:
            const = entry.unit.unit_const
            count = unit_variant.variant_count_for(const)
            # The panel's range already stops variant >= count; defensive.
            if variant >= count:
                continue
            if unit_variant.variant_of(entry.unit.rotation, unit_rotation.angle_count_for(const), count) == variant:
                continue
            changes.append(entry)
        if not changes:
            return
        label = "Set unit variant" if single else f"Set variant on {len(changes)} units"
        splices: list[UnitSplice] = []
        with self._unit_edit(model, label, sorted({e.player_id for e in changes}), splices, fields_only=True):
            self._write_variants(model, changes, lambda _unit: float(variant), splices)
        if skipped:
            self._log_status(f"{label} ({skipped} skipped: not rotatable)", level=debug_log.WARNING)

    def _set_gate_field(self, model, entries, index: int, single: bool) -> None:
        """The Rotation field's gate mode (GH #61): every member, all gates, takes
        orientation `index` through Rotate's own writer, so typing n is n presses from 0."""

        def steps_for(unit) -> int:
            siblings = gate_orientation.orientation_siblings(unit.unit_const)
            return (index - siblings.index(unit.unit_const)) % len(siblings)

        changing = [e for e in entries if steps_for(e.unit)]
        turned = [e for e in changing if self._gate_cycle_fits(e.unit, steps_for(e.unit))]
        blocked = len(changing) - len(turned)
        if turned:
            label = "Set unit Rotation" if single else f"Set rotation on {len(turned)} units"
            players = {e.player_id for e in turned}
            gate_plan = self._gate_cycle_plan(model, turned, steps_for)
            if gate_plan:
                # GH #339, as on_unit_rotate: removals first, one wholesale record.
                with self._unit_edit(model, label, sorted(players | wall_run.gate_move_players(gate_plan))):
                    model.remove_many([wall for _player_id, wall in gate_plan.removals])
                    self._write_gate_consts(model, turned, steps_for, None)
                    wall_run.apply_gate_move_plan(model, gate_plan)
                self._log_status(label + self._gate_move_clauses(gate_plan))
            else:
                splices: list[UnitSplice] = []
                with self._unit_edit(model, label, sorted(players), splices, fields_only=True):
                    self._write_gate_consts(model, turned, steps_for, splices)
        if blocked:
            self._log_status(f"Set Rotation: {blocked} gate(s) skipped (would hang off the map edge)", level=debug_log.WARNING)
            # The field still shows the typed index; put it back to what the gates hold.
            self._update_unit_inspector_from_selection()

    def _update_unit_inspector_from_selection(self) -> None:
        """Puts the inspector back in step with the model after an edit that
        was refused (_ensure_unit_edits() returned None) -- the unit-editor
        counterpart to _repopulate_map_options(). The panel's live values are
        what _on_unit_field_changed's equality guard compares against, so
        leaving a refused edit on screen would make the next genuine change
        to that field look like a no-op."""
        if self.mode != "units" or not self._selection:
            return
        self._refresh_selection_view()

    def _selection_live(self) -> bool:
        """Whether this mode draws the unit selection and lets an action read it (TASK-359)."""
        return self.mode in _UNIT_SELECTION_LIVE_MODES

    def _reconcile_selection(self) -> list:
        """The single reconciliation point for self._selection (b2.2):
        drops any key that no longer resolves against the live unit index
        and, in a live mode, pushes the surviving entries to MapView's N-way
        highlight. Returns those entries.

        Every path that used to read/write self._selection[0] directly and
        silently discard the rest (on_click_select, on_unit_nudge,
        on_unit_delete, _after_unit_mutation, _refresh_selection_after_filter)
        goes through this now, so a dead key dropped by one path can't
        linger and desync from another -- this had to land before the
        marquee, not after.

        With no index the keys stay as they are: it is rebuilt before any
        reader needs them. A dormant mode prunes keys only and never writes
        MapView's selection layer, which Pick from map borrows in Triggers;
        _switch_mode() clears that layer once on entry (TASK-359)."""
        index = self.map_view._unit_index
        if index is None:
            return []
        entries = [e for e in (index.entry_for_key(key) for key in self._selection) if e is not None]
        keys = [unit_pick.unit_key(e.player_id, e.unit) for e in entries]
        changed = keys != self._selection
        self._selection = keys
        if self._selection_live():
            self.map_view.set_unit_selection(entries)
        if changed and self.mode != "units":
            self._update_tool_enabled()
        return entries

    def _refresh_selection_view(self) -> None:
        """_reconcile_selection(), then the Units-mode half: the garrison
        picker, the tool gates and the inspector. One entry shows its
        fields (D5), 2+ show the group fields (GH #71), and zero shows
        unit_inspector_empty's text."""
        # Before the highlight push, so the trigger rings draw with it (GH #49 slice 2).
        trigger_effects = self._push_trigger_range_effects()
        entries = self._reconcile_selection()
        # Garrison pick belongs to the one selected host (GH #115).
        if self._garrison_picker is not None and self._selection != [self._garrison_picker]:
            self._disarm_garrison_picker(restore_tool=True)
        # Rotate is the first action gated on there BEING a selection, so this
        # reconciliation point is now also where that gate is re-evaluated.
        # Not a recursion risk despite _update_tool_enabled()'s own
        # pan_action.setChecked(True) branch: that re-enters _on_tool_selected,
        # which reaches _update_tool_enabled() again (terminating, per its own
        # comment) but never comes back through here or _after_unit_mutation.
        self._update_tool_enabled()
        if len(entries) == 1:
            self.units_panel.show_unit(entries[0])
            self._apply_garrison_block(entries[0])
            self._show_trigger_range_line(trigger_effects)
            return
        if entries:
            self.units_panel.show_group(entries)
            return
        self.units_panel.show_selection_count(0)

    # -- Inspector garrison block (GH #42) ------------------------------

    def _garrison_occupants(self, unit) -> list:
        """(player_id, occupant) for every unit garrisoned inside `unit`.

        Deliberately does NOT build the edit model: this runs on every
        selection change, and _ensure_unit_edits()' construction gate scans
        the whole file and can refuse with a dialog. An already-built model
        answers the membership question off its O(1) reverse-map; without
        one this is a walk of the nine lists, the same cost as the pick
        index's own build.
        """
        if unit.reference_id == -1:
            return []  # see _garrison_referrers: -1 is "inside nothing"
        model = self.unit_edits
        inside = {id(u) for u in _garrison_referrers(model, unit)} if model is not None else None
        rows = []
        for player_id, units in enumerate(self.scenario.unit_manager.units):
            for candidate in units:
                if candidate is unit:
                    continue
                if inside is not None:
                    if id(candidate) in inside:
                        rows.append((player_id, candidate))
                elif getattr(candidate, "garrisoned_in_id", -1) == unit.reference_id:
                    rows.append((player_id, candidate))
        return rows

    def _apply_garrison_block(self, entry) -> None:
        """Shows the Inspector's Garrison list for a host, or hides it.

        Shown for a const the game lets hold something, and also for one
        that already holds something it shouldn't -- a file is displayed as
        it is, never corrected, and an occupant hidden by the default filter
        would otherwise be unreachable.
        """
        unit = entry.unit
        occupants = self._garrison_occupants(unit)
        if not occupants and not garrison.can_hold(unit.unit_const):
            self.units_panel.hide_garrison()
            return
        rows = [
            (
                _unit_name(occupant.unit_const),
                "GAIA" if player_id == GAIA_PLAYER_ID else f"Player {player_id}",
                occupant.reference_id,
            )
            for player_id, occupant in occupants
        ]
        wrong_type = sum(1 for _pid, o in occupants if not garrison.accepts(unit.unit_const, o.unit_const))
        cut = self._pending_garrison_cut()
        can_paste = (bool(self._garrison_clipboard) or cut is not None) and garrison.can_hold(unit.unit_const)
        # GH #204: the marks show only on the host the rows were cut from.
        marked = cut.occupants if cut is not None and cut.host == unit.reference_id else ()
        gaia_only = unit_kind.gaia_only_consts()
        self.units_panel.show_garrison(
            rows,
            garrison.capacity(unit.unit_const),
            wrong_type,
            can_paste=can_paste,
            cut=marked,
            gaia_only=[o.reference_id for _pid, o in occupants if o.unit_const in gaia_only],
        )

    def _selected_host_entry(self):
        """The one selected unit's entry, or None -- the garrison block only
        ever exists for a single selection."""
        index = self.map_view._unit_index
        if self.scenario is None or index is None or len(self._selection) != 1:
            return None
        return index.entry_for_key(self._selection[0])

    def _on_garrison_add(self) -> None:
        """The Garrison block's Add... button: pick an object the host
        admits, then create it inside. The picker is restricted to
        garrison.eligible_consts(), so a refusal is normally unreachable
        from the UI -- _garrison_add_const() still checks, since it is also
        the path a test (and any later caller) drives."""
        entry = self._selected_host_entry()
        if entry is None:
            return
        eligible = garrison.eligible_consts(entry.unit.unit_const)
        catalog = [row for row in object_catalog.objects() if row.id in eligible]
        if not catalog:
            self._log_status(f"Garrison: nothing can go inside {_unit_name(entry.unit.unit_const)}")
            return
        dialog = CatalogBrowseDialog(catalog, parent=self)
        dialog.setWindowTitle(f"Garrison {_unit_name(entry.unit.unit_const)}")
        if dialog.exec_() != QDialog.Accepted:
            return
        unit_const = dialog.selected_id()
        if unit_const is not None:
            self._garrison_add_const(entry, unit_const)

    def _garrison_add_const(self, entry, unit_const: int) -> None:
        """Add...'s write half, without the modal picker.

        The new unit takes the host's own player, point and z, rotation 0
        and the host's reference_id. add() sets the link at construction;
        set_garrisoned_in() is the one writer for an existing unit. Not a
        fields_only edit.
        """
        host = entry.unit
        message = garrison.refusal(
            host.unit_const,
            _unit_name(host.unit_const),
            len(self._garrison_occupants(host)),
            [unit_const],
            name_of=_unit_name,
        )
        if message is not None:
            self._log_status(message)
            return
        model = self._ensure_unit_edits()
        if model is None:
            return
        player = entry.player_id
        splices: list[UnitSplice] = []
        with self._unit_edit(model, "Garrison unit", [player], splices):
            unit = model.add(
                player,
                unit_const,
                host.x,
                host.y,
                getattr(host, "z", 0.0),
                garrisoned_in_id=host.reference_id,
            )
            new_own, new_tiles = self._unit_footprint(unit)
            idx = self._unit_list_index(player, unit)
            splices.append(UnitSplice(player, idx, unit, None, new_own, (), new_tiles))
            index = self.map_view._unit_index
            if index is not None:
                # A no-op while Show Garrisoned is off, which is the point:
                # matches() is the single gate, so drawn and pickable stay
                # in step without this path knowing about the filter.
                unit_pick.patch_index_for_add(self.scenario, index, player, unit, self._unit_filter)
        self._log_status(f"Garrisoned {_unit_name(unit_const)} inside {_unit_name(host.unit_const)}")

    def _garrison_link_refusal(self, host, units) -> str | None:
        """set_garrisoned_in()'s structural scope as a status line, so a UI
        path never reaches the model's raise. Model-free, for the drag preview."""
        host_name = _unit_name(host.unit_const)
        by_ref: dict = {}
        holders: set[int] = set()
        for player_units in self.scenario.unit_manager.units:
            for candidate in player_units:
                by_ref.setdefault(candidate.reference_id, candidate)
                if candidate.garrisoned_in_id != candidate.reference_id:
                    holders.add(candidate.garrisoned_in_id)
        chain: set[int] = set()
        link = host.reference_id
        while link != -1 and link not in chain:
            chain.add(link)
            holder = by_ref.get(link)
            link = -1 if holder is None else holder.garrisoned_in_id
        for unit in units:
            if unit is host:
                return f"Garrison: {host_name} cannot go inside itself"
            if unit.reference_id in chain:
                return f"Garrison: {host_name} is inside {_unit_name(unit.unit_const)}"
            # -1 is "inside nothing", never a holder (see _garrison_referrers).
            if unit.reference_id != -1 and unit.reference_id in holders:
                return f"Garrison: {_unit_name(unit.unit_const)} holds a garrison of its own; unload it first"
        return None

    def _garrison_refusal_for(self, host, entries) -> str | None:
        """Every check _garrison_units() makes, as one status line or None:
        the structural scope, then garrison.refusal() over the whole batch."""
        incoming = [e for e in entries if e.unit.garrisoned_in_id != host.reference_id]
        if not incoming:
            return f"Garrison: already inside {_unit_name(host.unit_const)}"
        return self._garrison_link_refusal(host, [e.unit for e in incoming]) or garrison.refusal(
            host.unit_const,
            _unit_name(host.unit_const),
            len(self._garrison_occupants(host)),
            [e.unit.unit_const for e in incoming],
            name_of=_unit_name,
        )

    def _garrison_units(self, host_entry, entries, label: str = "Garrison units", verb: str = "Garrisoned") -> bool:
        """Puts existing units inside `host_entry`'s unit, at its point, in one
        undo record (GH #115). False, with a status line, when refused.
        `label` and `verb` reword the record and the status line for Paste's move."""
        host = host_entry.unit
        model = self._ensure_unit_edits()
        if model is None:
            return False
        model.warm_garrison_map()
        message = self._garrison_refusal_for(host, entries)
        if message is not None:
            self._log_status(message)
            return False
        incoming = [e for e in entries if e.unit.garrisoned_in_id != host.reference_id]
        players = sorted({e.player_id for e in incoming})
        splices: list[UnitSplice] = []
        with self._unit_edit(model, label, players, splices):
            for entry in incoming:
                unit = entry.unit
                old_own, old_tiles = self._unit_footprint(unit)
                idx = self._unit_list_index(entry.player_id, unit)
                model.set_garrisoned_in(unit, host.reference_id)
                model.set_position(unit, host.x, host.y, getattr(host, "z", 0.0))
                new_own, new_tiles = self._unit_footprint(unit)
                splices.append(UnitSplice(entry.player_id, idx, unit, old_own, new_own, old_tiles, new_tiles))
            # A garrisoned unit leaves the index under the default filter, and no remove patch exists.
            self._rebuild_unit_index()
        self._log_status(f"{verb} {len(incoming)} unit(s) inside {_unit_name(host.unit_const)}")
        return True

    def _unload_units(self, host_entry, rows) -> bool:
        """Takes `rows` ((player_id, unit) pairs inside `host_entry`'s unit) out,
        onto garrison.unload_tile() beside its footprint, in one undo record."""
        host = host_entry.unit
        rows = [(pid, u) for pid, u in rows if u.garrisoned_in_id != -1]
        if not rows:
            return False
        mm = self.scenario.map_manager
        bounds = unit_tile_bounds(host, mm.map_width, mm.map_height)
        tile = None if bounds is None else garrison.unload_tile(bounds, mm.map_width, mm.map_height)
        if tile is None:
            self._log_status(f"Unload: no tile beside {_unit_name(host.unit_const)} is on the map")
            return False
        model = self._ensure_unit_edits()
        if model is None:
            return False
        x, y = tile[0] + 0.5, tile[1] + 0.5
        index = self.map_view._unit_index
        players = sorted({pid for pid, _u in rows})
        splices: list[UnitSplice] = []
        rebuild = False
        with self._unit_edit(model, "Unload units", players, splices):
            for player_id, unit in rows:
                old_bounds = unit_tile_bounds(unit, mm.map_width, mm.map_height)
                old_own, old_tiles = self._unit_footprint(unit)
                idx = self._unit_list_index(player_id, unit)
                indexed = index is not None and index.entry_for_key(unit_pick.unit_key(player_id, unit)) is not None
                model.set_garrisoned_in(unit, -1)
                model.set_position(unit, x, y, unit.z)
                new_own, new_tiles = self._unit_footprint(unit)
                splices.append(UnitSplice(player_id, idx, unit, old_own, new_own, old_tiles, new_tiles))
                if index is None:
                    continue
                # Show Garrisoned on: already indexed, so a move, never a second entry.
                if indexed:
                    unit_pick.patch_index_for_move(self.scenario, index, player_id, unit, old_bounds)
                else:
                    rebuild = True
            # An append would rank the unit after every later player's; Unload is rare, so rebuild.
            if rebuild:
                self._rebuild_unit_index()
        self._log_status(f"Unloaded {len(rows)} unit(s) from {_unit_name(host.unit_const)}")
        return True

    def _on_garrison_delete(self, reference_ids) -> None:
        """The Garrison block's Delete button: the selected occupants go in
        one undo record, along with anything garrisoned inside them."""
        entry = self._selected_host_entry()
        if entry is None:
            return
        wanted = set(reference_ids)
        targets = [(pid, u) for pid, u in self._garrison_occupants(entry.unit) if u.reference_id in wanted]
        if not targets:
            return
        model = self._ensure_unit_edits()
        if model is None:
            return
        rows = [unit_pick.UnitEntry(pid, u, int(u.x), int(u.y), -1) for pid, u in targets]
        rows += self._garrison_occupant_entries(model, rows)
        players = sorted({row.player_id for row in rows})
        label = "Remove from garrison" if len(rows) == 1 else f"Remove {len(rows)} from garrison"
        with self._unit_edit(model, label, players):
            model.remove_many([row.unit for row in rows])
        self._log_status(f"Removed {len(rows)} unit(s) from {_unit_name(entry.unit.unit_const)}'s garrison")

    def _on_garrison_navigate(self, reference_id: int) -> None:
        """Double-click on an occupant row. Selecting one needs it in the
        pick index, and the default filter keeps it out -- so this says what
        to turn on rather than silently doing nothing."""
        entry = self._selected_host_entry()
        if entry is None:
            return
        if not self._unit_filter.show_garrisoned:
            self._log_status("Turn on Filters > Show Garrisoned Units to select a unit inside another one")
            return
        target = next(
            ((pid, u) for pid, u in self._garrison_occupants(entry.unit) if u.reference_id == reference_id),
            None,
        )
        if target is None:
            return
        player_id, occupant = target
        key = unit_pick.unit_key(player_id, occupant)
        index = self.map_view._unit_index
        if index is None or index.entry_for_key(key) is None:
            return
        self._selection = [key]
        self._refresh_selection_view()
        self._log_status(f"Selected {_unit_name(occupant.unit_const)} (garrisoned)")

    # -- GH #115: Pick from map, Copy/Paste, Unload, per-row Owner ---------

    def _selected_garrison_rows(self, entry, reference_ids) -> list:
        """(player_id, occupant) for the block's selected rows, in list order."""
        wanted = set(reference_ids)
        return [(pid, u) for pid, u in self._garrison_occupants(entry.unit) if u.reference_id in wanted]

    def _on_garrison_pick(self, checked: bool) -> None:
        """Pick from map: each click garrisons the picked unit in the selected
        host, through _garrison_units(), until Esc, right-click or disarm. Its
        own armed state on the trigger picker's press path; arming keeps the
        selection, so the block stays and the highlight shows the host. Arming
        forces Pan; Esc, right-click or the button returns to the tool it was
        armed from (TASK-252), since Pan no longer selects in Units mode."""
        if not checked:
            self._disarm_garrison_picker(restore_tool=True)
            return
        entry = self._selected_host_entry()
        if entry is None or self.scenario is None:
            self.units_panel.set_garrison_pick_armed(False)
            return
        self.disarm_unit_picker()
        self._garrison_return_tool = self._current_tool
        self.pan_action.setChecked(True)
        self._garrison_picker = unit_pick.unit_key(entry.player_id, entry.unit)
        self.map_view.set_unit_picker(True)
        self.units_panel.set_garrison_pick_armed(True)
        self._focus_map_view()
        hint = f"Click units to put inside {_unit_name(entry.unit.unit_const)}; Esc to finish"
        if self._unit_filter != UnitFilter():
            hint += " (units hidden by Filters can't be picked)"
        self._log_status(hint)

    def _disarm_garrison_picker(self, restore_tool: bool = False) -> None:
        """Ends the Garrison pick. `restore_tool` only from a user cancel (Esc,
        right-click, the button, the host or selection going): the tool it was
        armed from comes back if Pan is still the tool and that one still applies.
        A mode switch, load, undo/redo or another picker arming never restores."""
        if self._garrison_picker is None:
            return
        self._garrison_picker = None
        tool, self._garrison_return_tool = self._garrison_return_tool, None
        if self._unit_picker is None:
            self.map_view.set_unit_picker(False)
        self.units_panel.set_garrison_pick_armed(False)
        if not restore_tool or tool is None or tool == "pan" or not self.pan_action.isChecked():
            return
        action = getattr(self, f"{tool}_action", None)
        if action is not None and tool_applicable(tool, self.mode) and action.isEnabled():
            action.setChecked(True)

    def _on_unit_picker_cancel(self) -> None:
        """MapView's Esc or right-click while a pick is armed."""
        self._disarm_garrison_picker(restore_tool=True)
        self.disarm_unit_picker()

    def _on_map_unit_picked(self, key: tuple[int, int]) -> None:
        """MapView's pick, to whichever picker is armed."""
        if self._garrison_picker is not None:
            self._on_garrison_picked(key)
        else:
            self._on_unit_picked(key)

    def _on_garrison_picked(self, key: tuple[int, int]) -> None:
        index = self.map_view._unit_index
        host = index.entry_for_key(self._garrison_picker) if index is not None else None
        if host is None:
            self._disarm_garrison_picker(restore_tool=True)
            self._log_status("Garrison: the host is gone; Pick from map is off")
            return
        picked = index.entry_for_key(key)
        if picked is None:
            return
        if picked.unit is host.unit:
            self._log_status("Garrison: that is the host itself")
            return
        if picked.unit.garrisoned_in_id == host.unit.reference_id:
            self._log_status(f"Garrison: already inside {_unit_name(host.unit.unit_const)}")
            return
        self._garrison_units(host, [picked])

    @staticmethod
    def _garrison_clips(rows) -> tuple[garrison.GarrisonClip, ...]:
        """A GarrisonClip per (player_id, occupant) row, for Copy and Duplicate."""
        clips = []
        for player_id, unit in rows:
            try:
                caption_id, caption = unit.caption_string_id, unit.caption_string
            except (UnsupportedAttributeError, AttributeError):
                caption_id, caption = -1, ""  # region_clipboard.copy_region()'s defensive read
            clips.append(garrison.GarrisonClip(
                player=player_id,
                unit_const=unit.unit_const,
                rotation=unit.rotation,
                status=unit.status,
                initial_animation_frame=unit.initial_animation_frame,
                caption_string_id=caption_id,
                caption_string=caption,
                capture_flag=capture_flag_of(unit),
            ))
        return tuple(clips)

    def _on_garrison_duplicate(self, reference_ids) -> None:
        """Duplicate (TASK-549): copies of the selected rows inside the same
        host, through Paste's own checks and writes, in one record. The
        garrison clipboard and a pending Cut are left as they are."""
        entry = self._selected_host_entry()
        if entry is None:
            return
        rows = self._selected_garrison_rows(entry, reference_ids)
        if rows:
            self._add_garrison_copies(entry, self._garrison_clips(rows), "Duplicate in garrison", "Duplicated")

    def _on_garrison_copy(self, reference_ids) -> None:
        entry = self._selected_host_entry()
        if entry is None:
            return
        rows = self._selected_garrison_rows(entry, reference_ids)
        if not rows:
            return
        clips = self._garrison_clips(rows)
        self._garrison_clipboard = clips
        # GH #204: a Copy replaces a pending Cut, whose rows stay where they are.
        self._garrison_cut = None
        self.units_panel.set_garrison_cut(())
        self.units_panel.set_garrison_can_paste(garrison.can_hold(entry.unit.unit_const))
        self._log_status(f"Copied {len(clips)} garrisoned unit(s)")

    def _on_garrison_cut(self, reference_ids) -> None:
        """Marks the selected rows for a move (GH #204). Nothing is edited
        until Paste: the rows stay inside this host, and no record is made."""
        entry = self._selected_host_entry()
        if entry is None:
            return
        rows = self._selected_garrison_rows(entry, reference_ids)
        if not rows:
            return
        occupants = tuple(unit.reference_id for _pid, unit in rows)
        self._garrison_cut = garrison.GarrisonCut(self._doc_id, entry.unit.reference_id, occupants)
        self._garrison_clipboard = ()
        self.units_panel.set_garrison_cut(occupants)
        self.units_panel.set_garrison_can_paste(garrison.can_hold(entry.unit.unit_const))
        self._log_status(f"Cut {len(occupants)} garrisoned unit(s); Paste inside another host moves them")

    def _pending_garrison_cut(self):
        """The Cut waiting for a Paste, or None once its document is gone."""
        cut = self._garrison_cut
        return cut if cut is not None and cut.doc_id == self._doc_id else None

    def _paste_garrison_cut(self, host_entry, cut) -> None:
        """Paste after a Cut (GH #204): the cut rows still inside the host they
        were cut from move into `host_entry`'s unit through _garrison_units(),
        so its checks and its one record apply. A refusal keeps the cut."""
        source = self._entry_for_reference(cut.host)
        rows = [] if source is None else self._selected_garrison_rows(source, cut.occupants)
        if rows:
            entries = [unit_pick.UnitEntry(pid, u, int(u.x), int(u.y), -1) for pid, u in rows]
            if not self._garrison_units(host_entry, entries, label="Move into garrison", verb="Moved"):
                return
        else:
            self._log_status("Paste: the cut units are no longer inside the host they were cut from")
        self._garrison_cut = None
        self.units_panel.set_garrison_can_paste(False)

    def _on_garrison_paste(self) -> None:
        """New copies of the clipboard inside the selected host, each with its
        own owner, all or nothing through garrison.refusal(). One record.
        After a Cut it moves the cut rows instead (_paste_garrison_cut())."""
        entry = self._selected_host_entry()
        cut = self._pending_garrison_cut()
        if entry is not None and cut is not None:
            self._paste_garrison_cut(entry, cut)
            return
        clips = self._garrison_clipboard
        if entry is None or not clips:
            return
        self._add_garrison_copies(entry, clips, "Paste into garrison", "Pasted")

    def _add_garrison_copies(self, entry, clips, label: str, verb: str) -> bool:
        """New units from `clips` inside `entry`'s unit, each with its own
        owner, all or nothing through garrison.refusal(), in one record named
        `label`. False, with a status line, when refused. Paste and Duplicate."""
        host = entry.unit
        message = garrison.refusal(
            host.unit_const,
            _unit_name(host.unit_const),
            len(self._garrison_occupants(host)),
            [c.unit_const for c in clips],
            name_of=_unit_name,
        )
        if message is not None:
            self._log_status(message)
            return False
        model = self._ensure_unit_edits()
        if model is None:
            return False
        players = sorted({c.player for c in clips})
        splices: list[UnitSplice] = []
        with self._unit_edit(model, label, players, splices):
            for clip in clips:
                unit = model.add(
                    clip.player,
                    clip.unit_const,
                    host.x,
                    host.y,
                    getattr(host, "z", 0.0),
                    rotation=clip.rotation,
                    status=clip.status,
                    initial_animation_frame=clip.initial_animation_frame,
                    garrisoned_in_id=host.reference_id,
                    caption_string_id=clip.caption_string_id,
                    caption_string=clip.caption_string,
                    capture_flag=clip.capture_flag,
                )
                new_own, new_tiles = self._unit_footprint(unit)
                idx = self._unit_list_index(clip.player, unit)
                splices.append(UnitSplice(clip.player, idx, unit, None, new_own, (), new_tiles))
                index = self.map_view._unit_index
                if index is not None:
                    unit_pick.patch_index_for_add(self.scenario, index, clip.player, unit, self._unit_filter)
        self._log_status(f"{verb} {len(clips)} unit(s) inside {_unit_name(host.unit_const)}")
        return True

    def _on_garrison_unload(self, reference_ids) -> None:
        entry = self._selected_host_entry()
        if entry is not None:
            self._unload_units(entry, self._selected_garrison_rows(entry, reference_ids))

    def _on_garrison_owner(self, reference_ids, player_id: int) -> None:
        """The per-row Owner: reassign() on the selected rows only. The host
        and its other occupants keep theirs; the link is untouched."""
        entry = self._selected_host_entry()
        if entry is None:
            return
        changing = [(pid, u) for pid, u in self._selected_garrison_rows(entry, reference_ids) if pid != player_id]
        # GH #32: a GAIA-only occupant stays with its owner.
        gaia_only = sum(1 for _pid, u in changing if unit_kind.owner_refused(u.unit_const, player_id))
        changing = [(pid, u) for pid, u in changing if not unit_kind.owner_refused(u.unit_const, player_id)]
        owner = "GAIA" if player_id == GAIA_PLAYER_ID else f"Player {player_id}"
        if not changing:
            if gaia_only:
                self._log_status(f"Not reassigned to {owner}{_gaia_only_refused_note(gaia_only)}", level=debug_log.WARNING)
            return
        model = self._ensure_unit_edits()
        if model is None:
            return
        players = sorted({pid for pid, _u in changing} | {player_id})
        label = "Reassign garrisoned unit" if len(changing) == 1 else f"Reassign {len(changing)} garrisoned units"
        with self._unit_edit(model, label, players):
            for _pid, unit in changing:
                model.reassign(unit, player_id)
        self._log_status(f"Reassigned {len(changing)} garrisoned unit(s) to {owner}{_gaia_only_refused_note(gaia_only)}")
