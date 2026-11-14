"""ViewerWindow's unit operations: rotate, cycle variant and gate orientation, delete and cut, reassign and replace."""

from __future__ import annotations

from collections import Counter

from PyQt5.QtWidgets import (
    QMessageBox,
)

from descape import (
    debug_log,
    find_objects,
    find_triggers,
    gate_orientation,
    move_turn,
    region_transform,
    unit_kind,
    unit_pick,
    unit_references,
    unit_rotation,
    unit_variant,
    wall_run,
)
from descape.find_dialog import confirm_with_choices
from descape.render_cache_splice import UnitSplice
from descape.render_units import NON_BUILDING_SPAN
from descape.terrain_palette import tile_span
from descape.unit_filter import GAIA_PLAYER_ID
from descape.unit_model import (
    reanchored,
    replaced_position,
    span_low_corner,
)
from descape.viewer_common import (
    _reshaping_clause,
    _unit_name,
)

# _report_delete()'s past tense for each verb that removes units.
_REMOVE_VERB_PAST = {"Delete": "Deleted", "Cut": "Cut"}


# GH #32: Find and Replace's reason for a player's unit and a GAIA-only new type.
GAIA_ONLY_REPLACE_REASON = "GAIA-only type; only GAIA can own it"


class _ViewerUnitOpsMixin:
    """ViewerWindow's rotate, variant, delete, reassign and replace (viewer.py's docstring has the split's rules)."""

    def on_unit_rotate(self, steps: int, gate_steps: int | None = None, variant_steps=None) -> None:
        """Turns every selected unit whose `rotation` is genuinely an angle by
        `steps` whole stored frames, cycles every selected gate through
        `gate_steps` of its own four orientations, and steps every cyclable
        tree or scenery object through its graphic variants. Phase 3.5b's b3,
        variants since GH #123.

        Modelled on on_unit_nudge: the whole selection moves in ONE undo
        record, since begin_unit_edit(players) already captures every touched
        player's list up front. Positive steps rotate clockwise on screen (see
        unit_rotation.rotate_step and gate_orientation.cycle_const).

        **Each unit rotates about its own centre.** Turning the selection as
        one group about a shared pivot is Rotate Group's
        (on_selection_transform(), TASK-256), a group transform rather than a
        per-unit field edit.

        `gate_steps` defaults to `steps` and exists because a gate step is 45
        degrees while a rotation step is one stored frame: the coarse path
        passes a quarter turn in each unit, so a mixed selection turns the same
        visible amount instead of the gates silently doing a full 4-step
        identity cycle.

        A gate whose new footprint would hang off the map is skipped and
        counted: (4, 1) to (1, 4) grows three tiles on the other axis, and
        this is the layer that knows the map's dimensions. The model stays
        dimension-free.

        A turned gate never stands on walls (GH #339, Place's GH #159 rule):
        every press removes the walls under its new footprint and re-derives
        the walls beside its old and new tiles, in one wholesale record. A
        cycle that touches no wall keeps the fields_only splice record.

        `variant_steps` maps a member's own variant_count to its steps and
        defaults to `steps`, so Rotate ↻ agrees with Next Variant. Variants go
        through set_variant(), the Cycle Variant exception, never set_rotation().
        Per member is fine here, unlike facings: variants have no alignment to
        drift out of.

        Walls, cliffs, single-variant objects and INERT consts (every other
        single-frame graphic) are skipped rather than refused: a marquee over
        a village will always include some, and failing the whole action for
        them would make Rotate unusable exactly where it is most wanted.

        Mode-gated explicitly, unlike nudge/delete: those arrive through
        MapView's injected callables and so are Units-mode-only for free,
        while this is a QAction whose shortcut is live in every mode.

        GH #172: while Place Unit's (or, GH #355, the Unit Brush's) ghost is
        showing, Rotate turns the ghost instead (rotate_place_ghost), and the
        selection is left alone.

        GH #197: with a just-pasted region armed and no Units-mode selection
        or Place target, Rotate turns that whole region a quarter turn
        instead (on_region_transform), in any mode but Triggers. That re-pastes
        new units rather than editing these.

        TASK-256: with no paste armed, the same holds for a committed Select
        region, which turns a quarter turn in place (_transform_region()),
        editing the units on it and keeping their identity.

        GH #280: during a live Move drag, Rotate turns the drag instead
        (_turn_unit_drag()) and writes nothing until the release. Past the
        threshold only: a held press that never moved is not a move.

        GH #355: while a Unit Brush stroke is open, Rotate changes nothing.
        The stroke read its facing at the press, and the selection must not
        turn inside the stroke's open edit.
        """
        if self.map_view.unit_drag_live():
            self._turn_unit_drag(move_turn.fine(self._drag_turn, steps))
            return
        if self._unit_brush_stroke_refuses_rotate():
            return
        if self._place_ghost_active():
            self.rotate_place_ghost(steps=steps)
            return
        if self._region_turn_target():
            self.on_region_transform(region_transform.ROTATE_CW if steps > 0 else region_transform.ROTATE_CCW)
            return
        if self._region_rotate_target():
            if not (self._stroke_in_progress() or self.map_view.region_drag_live()):
                self._transform_region(region_transform.ROTATE_CW if steps > 0 else region_transform.ROTATE_CCW, "Rotate")
            return
        if self.scenario is None or self.mode != "units":
            return
        if not self._selection:
            if self._place_rotate_armed():
                self._log_status("Rotate: nothing to rotate (hover the map to turn the preview)")
            return
        index = self.map_view._unit_index
        if index is None:
            return
        gate_steps = steps if gate_steps is None else gate_steps
        if variant_steps is None:
            def variant_steps(_count: int) -> int:
                return steps
        entries = [e for e in (index.entry_for_key(k) for k in self._selection) if e is not None]
        rotatable = [e for e in entries if unit_rotation.rotation_is_angle(e.unit.unit_const)]
        gates = [e for e in entries if gate_orientation.is_gate(e.unit.unit_const)]
        turned_gates = [e for e in gates if self._gate_cycle_fits(e.unit, gate_steps)]
        variants = [e for e in entries if unit_variant.is_cyclable(e.unit.unit_const)]
        blocked = len(gates) - len(turned_gates)
        skipped = len(entries) - len(rotatable) - len(gates) - len(variants)
        if not rotatable and not turned_gates and not variants:
            reasons = []
            if blocked:
                reasons.append(f"{blocked} gate(s) would hang off the map edge")
            if skipped:
                reasons.append(f"{skipped} selected unit(s) have no facing or variant to change")
            if reasons:
                self._log_status(f"Rotate: {'; '.join(reasons)}")
            return
        model = self._ensure_unit_edits()
        if model is None:
            return
        touched = rotatable + turned_gates + variants
        label = "Rotate unit" if len(touched) == 1 else f"Rotate {len(touched)} units"
        players = {e.player_id for e in touched}

        def next_variant(unit) -> float:
            angle_count = unit_rotation.angle_count_for(unit.unit_const)
            count = unit_variant.variant_count_for(unit.unit_const)
            return unit_variant.cycle_step(unit.rotation, angle_count, count, variant_steps(count))

        def write(splices) -> None:
            facings = []
            for entry in rotatable:
                angle_count = unit_rotation.angle_count_for(entry.unit.unit_const)
                facings.append((entry, unit_rotation.rotate_step(entry.unit.rotation, angle_count, steps)))
            self._write_facings(model, facings, splices)
            self._write_variants(model, variants, next_variant, splices)
            self._write_gate_consts(model, turned_gates, lambda _unit: gate_steps, splices)

        gate_plan = self._gate_cycle_plan(model, turned_gates, lambda _unit: gate_steps)
        if gate_plan:
            # GH #339: removals shift list indices, so no splice; one wholesale record.
            with self._unit_edit(model, label, sorted(players | wall_run.gate_move_players(gate_plan))):
                # First: the only call that can refuse, so a refusal leaves nothing turned.
                model.remove_many([wall for _player_id, wall in gate_plan.removals])
                write(None)
                wall_run.apply_gate_move_plan(model, gate_plan)
        else:
            splices: list[UnitSplice] = []
            with self._unit_edit(model, label, sorted(players), splices, fields_only=True):
                write(splices)
        clauses = self._gate_move_clauses(gate_plan)
        self._log_status(self._rotate_status(rotatable, turned_gates, variants, blocked, skipped, clauses))

    def _write_facings(self, model, changes, splices: list[UnitSplice] | None) -> None:
        """set_rotation() on each (entry, rotation) pair, recording its splice.
        Shared by Rotate and the Rotation field, typed or stepped. Caller holds _unit_edit.

        Splice-eligible (Batch D's D5): rotation never moves the footprint, so
        old/new own_tile and old/new tiles are identical. The splice still
        exists so the cache re-resolves this unit's now-different sprite frame,
        and the pick index is left untouched (nothing pickable moved).
        `splices` None: a wholesale edit (a gate cycle that clears walls,
        GH #339), whose list indices have shifted; nothing is recorded.
        """
        for entry, rotation in changes:
            if splices is None:
                model.set_rotation(entry.unit, rotation)
                continue
            unit = entry.unit
            old_own, old_tiles = self._unit_footprint(unit)
            idx = self._unit_list_index(entry.player_id, unit)
            model.set_rotation(unit, rotation)
            new_own, new_tiles = self._unit_footprint(unit)
            splices.append(UnitSplice(entry.player_id, idx, unit, old_own, new_own, old_tiles, new_tiles))

    def _gate_cycle_plan(self, model, entries, steps_for) -> wall_run.GateMovePlan | None:
        """_gate_change_plan() for cycling each gate entry `steps_for(unit)`
        orientations along, at the position set_unit_const() re-anchors it to
        (reanchored()). Planned before the edit opens, so a truthy plan can
        take the wholesale path (GH #339)."""
        changes = []
        for entry in entries:
            new_const = gate_orientation.cycle_const(entry.unit.unit_const, steps_for(entry.unit))
            changes.append((entry.unit, new_const, *reanchored(entry.unit, new_const)))
        return self._gate_change_plan(model, changes) if changes else None

    def _write_gate_consts(self, model, entries, steps_for, splices: list[UnitSplice] | None) -> None:
        """set_unit_const() to each gate's sibling `steps_for(unit)` along, recording
        its splice. Shared by Rotate and the Rotation field (GH #61). Caller holds _unit_edit.

        set_unit_const() changes span, which unit_pick.patch_index_for_move()
        is explicitly not scoped to (see its own docstring). One full rebuild
        covers the whole batch, including any facings or variants the caller
        wrote first, which never needed one on their own.

        `splices` None: the wholesale edit a gate cycle takes when it clears
        walls (GH #339); nothing is recorded and the wholesale refresh
        rebuilds the index instead.
        """
        for entry in entries:
            unit = entry.unit
            new_const = gate_orientation.cycle_const(unit.unit_const, steps_for(unit))
            if splices is None:
                model.set_unit_const(unit, new_const)
                continue
            old_own, old_tiles = self._unit_footprint(unit)
            idx = self._unit_list_index(entry.player_id, unit)
            model.set_unit_const(unit, new_const)
            new_own, new_tiles = self._unit_footprint(unit)
            splices.append(UnitSplice(entry.player_id, idx, unit, old_own, new_own, old_tiles, new_tiles))
        if entries and splices is not None:
            self._rebuild_unit_index()

    def _write_variants(self, model, entries, value_for, splices: list[UnitSplice] | None) -> None:
        """set_variant(unit, value_for(unit)) on each entry, recording its
        splice. Shared by Cycle Variant, Rotate and the Rotation field, so
        all three write a variant the same way. Caller holds _unit_edit.
        `splices` None: a wholesale edit (GH #339), nothing recorded."""
        for entry in entries:
            unit = entry.unit
            if splices is None:
                model.set_variant(unit, value_for(unit))
                continue
            old_own, old_tiles = self._unit_footprint(unit)
            idx = self._unit_list_index(entry.player_id, unit)
            model.set_variant(unit, value_for(unit))
            new_own, new_tiles = self._unit_footprint(unit)
            splices.append(UnitSplice(entry.player_id, idx, unit, old_own, new_own, old_tiles, new_tiles))

    def _gate_cycle_fits(self, unit, gate_steps: int) -> bool:
        """Whether this gate's next footprint still fits on the map.

        Measured from span_low_corner(), the same low corner
        UnitEditModel.set_unit_const() re-anchors from, so the two sides
        cannot disagree about which tiles the swap would claim. Both ends are
        checked: a span-4 axis puts the low corner two tiles below the unit's
        own, which can already be negative near the map's origin.
        """
        new_const = gate_orientation.cycle_const(unit.unit_const, gate_steps)
        low_x, low_y = span_low_corner(unit)
        span_x, span_y = tile_span(new_const, NON_BUILDING_SPAN)
        mm = self.scenario.map_manager
        return (
            low_x >= 0
            and low_y >= 0
            and low_x + span_x <= mm.map_width
            and low_y + span_y <= mm.map_height
        )

    def _rotate_status(self, rotatable, gates, variants, blocked: int, skipped: int, clauses: str = "") -> str:
        """on_unit_rotate's report line: what turned, what cycled, what was
        left alone. Names the single unit when exactly one thing was touched,
        the way every other unit action's status does. `clauses` is a gate
        cycle's wall tail (_gate_move_clauses(), GH #339), before the skips."""
        parts = []
        alone = len(rotatable) + len(gates) + len(variants) == 1
        if rotatable:
            if alone:
                unit = rotatable[0].unit
                count = unit_rotation.angle_count_for(unit.unit_const)
                facing = unit_rotation.rotation_to_facing(unit.rotation, count)
                parts.append(f"Rotated {_unit_name(unit.unit_const)} to facing {facing}/{count}")
            else:
                parts.append(f"Rotated {len(rotatable)} unit{'' if len(rotatable) == 1 else 's'}")
        if gates:
            if alone:
                unit = gates[0].unit
                parts.append(f"Cycled {_unit_name(unit.unit_const)} to ({unit.x:g}, {unit.y:g})")
            else:
                parts.append(f"cycled {len(gates)} gate{'' if len(gates) == 1 else 's'}")
        if variants:
            if alone:
                parts.append(self._variant_done("Cycled", variants[0].unit))
            else:
                owner = "object's" if len(variants) == 1 else "objects'"
                parts.append(f"cycled {len(variants)} {owner} variant")
        done = ", ".join(parts)
        done = done[:1].upper() + done[1:] + clauses
        if blocked:
            done += f"; {blocked} gate(s) skipped (would hang off the map edge)"
        if skipped:
            done += f"; {skipped} skipped (no facing or variant to change)"
        return done

    @staticmethod
    def _variant_done(verb: str, unit) -> str:
        """"Cycled Tree (Oak) to variant 5/42", read through variant_of()."""
        const = unit.unit_const
        count = unit_variant.variant_count_for(const)
        variant = unit_variant.variant_of(unit.rotation, unit_rotation.angle_count_for(const), count)
        return f"{verb} {_unit_name(const)} to variant {variant}/{count}"

    def on_unit_variant(self, steps: int = 0, randomize: bool = False) -> None:
        """Steps (or randomizes) the graphic variant of every selected cyclable
        unit, in one undo record. Modelled on on_unit_rotate(): non-cyclable
        units are skipped and counted rather than refused, and the QAction's
        shortcut is live in every mode, so the mode gate is explicit.
        """
        if self.scenario is None or self.mode != "units" or not self._selection:
            return
        index = self.map_view._unit_index
        if index is None:
            return
        entries = [e for e in (index.entry_for_key(k) for k in self._selection) if e is not None]
        cyclable = [e for e in entries if unit_variant.is_cyclable(e.unit.unit_const)]
        skipped = len(entries) - len(cyclable)
        if not cyclable:
            if skipped:
                self._log_status(f"Cycle Variant: {skipped} selected unit(s) have no graphic variants")
            return
        model = self._ensure_unit_edits()
        if model is None:
            return
        verb = "Randomize" if randomize else "Cycle"
        label = f"{verb} unit variant" if len(cyclable) == 1 else f"{verb} {len(cyclable)} units' variant"

        def new_variant(unit) -> float:
            angle_count = unit_rotation.angle_count_for(unit.unit_const)
            variant_count = unit_variant.variant_count_for(unit.unit_const)
            if randomize:
                return unit_variant.random_variant(unit.rotation, angle_count, variant_count, self._variant_rng)
            return unit_variant.cycle_step(unit.rotation, angle_count, variant_count, steps)

        splices: list[UnitSplice] = []
        with self._unit_edit(model, label, sorted({e.player_id for e in cyclable}), splices, fields_only=True):
            self._write_variants(model, cyclable, new_variant, splices)
        if len(cyclable) == 1:
            done = self._variant_done("Randomized" if randomize else "Cycled", cyclable[0].unit)
        else:
            done = f"{'Randomized' if randomize else 'Cycled'} {len(cyclable)} units' variant"
        if skipped:
            done += f"; {skipped} skipped (this object has no graphic variants)"
        self._log_status(done)

    def on_unit_rotate_coarse(self, direction: int) -> None:
        """Rotate by the closest whole number of frames to a quarter turn.

        The coarse step is its own action rather than a Shift modifier on the
        fine one: a QAction's shortcut IS the key combination, so unlike
        on_unit_nudge (which gets a live `modifiers` from MapView's key event)
        there is no modifier for this path to read. Four keybind rows, two
        entry points.

        Gates count their own quarter turn: two of their four 45-degree
        orientation steps, regardless of what frame count the selection's
        first rotatable unit has. A tree or scenery object steps a quarter of
        its own variant count (GH #123): oak 42 -> 11, pine 27 -> 7.
        """
        if self.map_view.unit_drag_live():  # GH #280, as on_unit_rotate
            entry = self._first_rotatable_entry()
            count = None if entry is None else unit_rotation.angle_count_for(entry.unit.unit_const)
            frames = 1 if count is None else unit_rotation.quarter_turn_steps(count)
            self._turn_unit_drag(move_turn.coarse(self._drag_turn, direction, frames))
            return
        if self._unit_brush_stroke_refuses_rotate():
            return
        if self._place_ghost_active():
            self.rotate_place_ghost(coarse_direction=direction)
            return
        # GH #197: a region has one step size, so the coarse pair turns it a quarter turn too.
        if self._region_turn_target():
            self.on_region_transform(region_transform.ROTATE_CW if direction > 0 else region_transform.ROTATE_CCW)
            return
        if self._region_rotate_target():
            self.on_unit_rotate(direction)  # TASK-256: the committed region's quarter turn
            return
        entry = self._first_rotatable_entry()
        if entry is None:
            steps = direction  # no facing to take a step size from
        else:
            steps = direction * unit_rotation.quarter_turn_steps(unit_rotation.angle_count_for(entry.unit.unit_const))
        self.on_unit_rotate(
            steps,
            gate_steps=direction * gate_orientation.QUARTER_TURN_STEPS,
            variant_steps=lambda count: direction * unit_rotation.quarter_turn_steps(count),
        )

    def _unit_brush_stroke_refuses_rotate(self) -> bool:
        """GH #355: with a Unit Brush stroke open, logs why Rotate does nothing and says True."""
        if self._unit_brush is None:
            return False
        self._log_status("Unit Brush: turn the preview between strokes")
        return True

    def _first_rotatable_entry(self):
        """The first selected entry whose rotation is an angle, or None.

        A mixed selection can span several angle_counts, and a quarter turn is
        a different number of frames in each. This picks the step size from
        the first rotatable unit and applies it to all of them, so one press
        stays one undo record; a selection of mixed graphics turns by the same
        frame count rather than the same visual angle. Rotating each by its
        own quarter turn is the alternative, and it is worse: two adjacent
        units would silently drift apart in facing on repeated presses.
        """
        index = self.map_view._unit_index
        if index is None or self.mode != "units":
            return None
        for key in self._selection:
            entry = index.entry_for_key(key)
            if entry is not None and unit_rotation.rotation_is_angle(entry.unit.unit_const):
                return entry
        return None

    def on_unit_delete(self, modifiers) -> None:
        """MapView's tenth injected callable -- the Delete key in Units mode
        (b1.6).

        No confirm dialog (revised from D4's original design): every other
        edit tool in this app (Draw, Elevate, Delete Unit's own reassign/
        move/place siblings) commits straight to the undo-backed
        EditHistory with no "are you sure", and Ctrl+Z is the same one
        keystroke away regardless. remove()'s own garrison-reference guard
        stays a hard refusal, unrelated to confirmation -- a dangling
        reference is legal on disk and the in-game editor permits it, so
        this must not additionally hard-block on one.

        Deferred from D4's full design: naming which triggers reference this
        unit's reference_id needs a per-trigger vocabulary scan
        (trigger_fields.field_specs() over every condition/effect, matched
        against the Unit/Unit[] presentations) that this slice doesn't
        build.

        b2.4 generalizes this to the whole selection, as one undo record --
        but unlike nudge/move, a group delete can be PARTIALLY refused (some
        selected units garrison-referenced, others not), and `_unit_edit`
        must never be entered for zero real changes (its own docstring: a
        spurious enter records a phantom undo step). So the referencing
        check runs as a pre-flight over the whole group, via
        UnitEditModel.referencing() -- the same guard remove() itself
        applies, factored out so both call sites can't drift apart -- and
        only the units it clears go into one commit.
        """
        if self.scenario is None or not self._selection:
            return
        index = self.map_view._unit_index
        if index is None:
            return
        entries = [e for e in (index.entry_for_key(k) for k in self._selection) if e is not None]
        if not entries:
            return
        model = self._ensure_unit_edits()
        if model is None:
            return
        entries, removable, cascaded_ids = self._delete_preflight(model, entries)
        if not removable:
            self._report_delete(entries, removable, cascaded_ids)
            return
        reshaped = self._remove_units(model, removable, "Delete")
        self._report_delete(entries, removable, cascaded_ids, reshaped=reshaped)

    def cut_units(self) -> None:
        """Cut Units (GH #192): Copy Units, then Delete's exact rules (the
        garrison cascade and the refusal, _delete_preflight()) as one undo
        record. Nothing is copied unless something is about to be removed,
        and a refused unit is neither copied nor removed."""
        entries = self._selected_unit_entries()
        if self.scenario is None or not entries:
            return
        model = self._ensure_unit_edits()
        if model is None:
            self._log_status("Cut Units: units are read-only for this file", level=debug_log.WARNING)
            return
        entries, removable, cascaded_ids = self._delete_preflight(model, entries)
        if not removable:
            self._report_delete(entries, removable, cascaded_ids, "Cut")
            return
        # Without the cascade: the copy's own garrison closure finds those again.
        self.copy_units([e for e in removable if id(e.unit) not in cascaded_ids], quiet=True)
        reshaped = self._remove_units(model, removable, "Cut")
        self._report_delete(entries, removable, cascaded_ids, "Cut", reshaped=reshaped)

    def _replace_wall_plan(self, changes) -> wall_run.ReplacePlan | None:
        """wall_run.scenario_replace_rewrites() for a Replace into a non-gate
        const (TASK-584), `changes` [(player_id, unit, new_const, x, y)]."""
        mm = self.scenario.map_manager
        return wall_run.scenario_replace_rewrites(self.scenario, mm.map_width, mm.map_height, changes)

    def _removal_rewrites(self, units) -> list:
        """wall_run.scenario_removal_rewrites() for removing `units` (GH #340):
        the walls left beside a removed wall or gate re-derive. [] with no
        wall or gate among them, before any scene walk."""
        mm = self.scenario.map_manager
        return wall_run.scenario_removal_rewrites(self.scenario, mm.map_width, mm.map_height, units)

    def _remove_units(self, model, removable, verb: str) -> int:
        """Removes `removable` (a non-empty _delete_preflight() result) as one
        undo record labelled with `verb`: Delete's and Cut's shared write.
        Returns how many walls beside a removed wall or gate were reshaped
        (GH #340); any reshape makes the record wholesale."""
        rewrites = self._removal_rewrites([e.unit for e in removable])
        players = sorted({e.player_id for e in removable} | {player_id for player_id, _u, _v in rewrites})
        label = f"{verb} unit" if len(removable) == 1 else f"{verb} {len(removable)} units"
        if rewrites:
            # The reshaped neighbours have no removal splice: wholesale, for any count.
            with self._unit_edit(model, label, players):
                model.remove_many([e.unit for e in removable])
                wall_run.apply_rewrites(model, rewrites)
        elif len(removable) == 1:
            # Splice-eligible (Batch D's D5's "single Delete", with no wall to reshape): a batch of
            # 2+ removals keeps the wholesale path below instead, since each
            # unit's own list index (render_cache_splice.UnitSplice.index, needed
            # for wall_variant_rotation_overrides' key) shifts under a
            # LATER removal from the same player's list -- see UnitSplice's
            # own docstring. That 2+ path passes no splices, so no per-unit
            # index has to survive a later removal and it takes remove_many().
            entry = removable[0]
            unit = entry.unit
            old_own, old_tiles = self._unit_footprint(unit)
            idx = self._unit_list_index(entry.player_id, unit)
            splices: list[UnitSplice] = []
            with self._unit_edit(model, label, players, splices):
                model.remove(unit)
                splices.append(UnitSplice(entry.player_id, idx, unit, old_own, None, old_tiles, ()))
                # No patch_index_for_remove() exists (see unit_pick.py's own
                # D4 note) -- a full rebuild is the only path for a removal.
                self._rebuild_unit_index()
        else:
            with self._unit_edit(model, label, players):
                model.remove_many([e.unit for e in removable])
        return len(rewrites)

    def _delete_preflight(self, model, entries) -> tuple[list, list, set[int]]:
        """(entries plus their garrison cascade, the removable ones, the
        cascaded units' ids): on_unit_delete's and Find's shared pre-flight.

        Deleting a host deletes what it holds, in the same undo step (GH
        #42): occupants are hidden by default, so they can't be selected
        alongside it, and leaving them would dangle their link. The refusal
        therefore only survives for a unit whose holder is NOT in the batch,
        a dangling reference or an unselected host."""
        # One garrison reverse-map walk for the whole pre-flight, instead of
        # leaving the first referencing() call to find it cold.
        model.warm_garrison_map()
        occupants = self._garrison_occupant_entries(model, entries)
        cascaded_ids = {id(e.unit) for e in occupants}
        entries = list(entries) + occupants
        batch = {id(e.unit) for e in entries}
        # model.referencing(), not _garrison_referrers(): this decides what
        # remove_many() will accept, and the model keeps the -1 bucket. So a
        # unit whose own reference_id is -1 is still refused here, exactly as
        # it was before the cascade, rather than raising out of the model.
        removable = [e for e in entries if all(id(u) in batch for u in model.referencing(e.unit))]
        return entries, removable, cascaded_ids

    def _report_delete(
        self, entries, removable, cascaded_ids, verb: str = "Delete", reshaped: int = 0
    ) -> tuple[int, int, int]:
        """The status line and refusal warning for a delete that already ran
        (or was wholly refused); returns (removed, cascaded, blocked). `verb`
        is "Delete", or "Cut" for Cut Units, which words the same report.
        `reshaped` counts the walls left beside a removed wall or gate that
        took a new shape (GH #340), in _gate_move_clauses()' wording."""
        done = _REMOVE_VERB_PAST[verb]
        blocked = len(entries) - len(removable)
        if not removable:
            self._log_status(f"{verb} failed: referenced by a garrison", level=debug_log.WARNING)
            self._warn(
                f"Cannot {verb.lower()} unit" if len(entries) == 1 else f"Cannot {verb.lower()} units",
                "Referenced by another unit's garrison." if len(entries) == 1
                else "All selected units are referenced by another unit's garrison.",
            )
            return 0, 0, blocked
        # Only the cascade that actually ran, not every occupant found: a
        # host whose own removal was refused takes its occupants with it.
        cascaded = sum(1 for e in removable if id(e.unit) in cascaded_ids)
        carried = f" ({cascaded} garrisoned)" if cascaded else ""
        if reshaped:
            carried += f", reshaping {reshaped} adjacent"
        if blocked:
            self._log_status(f"{done} {len(removable)}{carried}; {blocked} refused (garrison reference)", level=debug_log.WARNING)
            self._warn(
                f"Some units not {done.lower()}",
                f"{blocked} unit(s) are referenced by another unit's garrison and were not {done.lower()}.",
            )
        elif len(removable) - cascaded == 1:
            name = next(_unit_name(e.unit.unit_const) for e in removable if id(e.unit) not in cascaded_ids)
            self._log_status(f"{done} {name}{carried}")
        else:
            self._log_status(f"{done} {len(removable) - cascaded} units{carried}")
        return len(removable), cascaded, blocked

    def _delete_units(self, entries) -> tuple[int, int, int]:
        """Deletes duck-typed (player_id, unit) entries and their garrisons in
        one undo record, always via remove_many(): Find's Delete, which reaches
        filtered, garrisoned and off-map units the pick index lacks. Returns
        (removed, cascaded, blocked)."""
        if self.scenario is None or not entries:
            return 0, 0, 0
        model = self._ensure_unit_edits()
        if model is None:
            return 0, 0, 0
        entries, removable, cascaded_ids = self._delete_preflight(model, entries)
        rewrites = []
        if removable:
            rewrites = self._removal_rewrites([e.unit for e in removable])
            players = {e.player_id for e in removable} | {player_id for player_id, _u, _v in rewrites}
            label = "Delete unit" if len(removable) == 1 else f"Delete {len(removable)} units"
            with self._unit_edit(model, label, sorted(players)):
                model.remove_many([e.unit for e in removable])
                # GH #340: the walls left beside a removed wall or gate re-derive.
                wall_run.apply_rewrites(model, rewrites)
        return self._report_delete(entries, removable, cascaded_ids, reshaped=len(rewrites))

    def _reassign_units(
        self, entries, new_player: int, on_applied=None, skip_orphan_occupants: bool = False
    ) -> tuple[int, int, int, int]:
        """Moves entries to `new_player` in one undo record, carrying every
        host's garrison (GH #115).

        `on_applied(old_to_new_keys)` runs inside the _unit_edit block, where
        a key rewrite has to happen: leaving the block drops keys that no
        longer resolve. `skip_orphan_occupants` (Find) refuses an occupant
        entry whose host is not in the batch rather than creating a
        mixed-owner garrison, which GH #115 Step 0 has not cleared in-game;
        the Inspector keeps its per-unit behaviour.

        GH #32: an entry whose own move or carried garrison would give a
        GAIA-only object to a player is left out first, and so is an entry
        garrisoned inside one, uncounted (TASK-474). Returns (changed,
        carried, refused, gaia_only_refused)."""
        model = self._ensure_unit_edits()
        if model is None:
            return 0, 0, 0, 0
        refused = 0
        kept = [e for e in entries if not self._owner_change_refused(model, e, new_player)]
        gaia_only_refused = len(entries) - len(kept)
        kept_ids = {id(e.unit) for e in kept}
        refused_hosts = [e for e in entries if id(e.unit) not in kept_ids]
        entries = kept
        if skip_orphan_occupants:
            batch_refs = {e.unit.reference_id for e in entries}
            kept = []
            for e in entries:
                host = getattr(e.unit, "garrisoned_in_id", -1)
                if host not in (-1, e.unit.reference_id) and host not in batch_refs:
                    refused += 1
                    continue
                kept.append(e)
            entries = kept
        if refused_hosts:
            # TASK-474: an occupant selected with its refused host stays with it, no mixed-owner garrison.
            held = {id(o.unit) for o in self._garrison_occupant_entries(model, refused_hosts)}
            entries = [e for e in entries if id(e.unit) not in held]
        batch_ids = {id(e.unit) for e in entries}
        changing = [e for e in entries if e.player_id != new_player]
        # A host's garrison goes with it (GH #115), whatever each occupant's own owner.
        carried = [
            e
            for e in self._garrison_occupant_entries(model, entries)
            if e.player_id != new_player and id(e.unit) not in batch_ids
        ]
        if not changing and not carried:
            return 0, 0, refused, gaia_only_refused
        label = "Reassign unit" if len(entries) == 1 else f"Reassign {len(changing)} units"
        players = sorted({e.player_id for e in changing + carried} | {new_player})
        with self._unit_edit(model, label, players):
            old_keys = [unit_pick.unit_key(e.player_id, e.unit) for e in changing + carried]
            for entry in changing + carried:
                model.reassign(entry.unit, new_player)
            if on_applied is not None:
                on_applied({old: (new_player, old[1]) for old in old_keys})
        return len(changing), len(carried), refused, gaia_only_refused

    def _owner_change_refused(self, model, entry, new_player: int) -> bool:
        """GH #32: moving `entry` to `new_player` would give a GAIA-only object
        to a player, itself or one of the garrison it carries."""
        if new_player == GAIA_PLAYER_ID:
            return False
        if entry.player_id != new_player and unit_kind.owner_refused(entry.unit.unit_const, new_player):
            return True
        return any(
            occupant.player_id != new_player and unit_kind.owner_refused(occupant.unit.unit_const, new_player)
            for occupant in self._garrison_occupant_entries(model, [entry])
        )

    def _replace_preflight(self, model, entries, new_const: int) -> tuple[list, dict]:
        """(the entries a Replace to `new_const` will change, {key: reason} for
        the rest), by UnitEditModel.replace_batch_refusals(): dropping one row
        can change what its host or occupant may become, either way. A
        duplicated reference_id, and a player's unit for a GAIA-only
        `new_const` (GH #32), are refused outright."""
        mm = self.scenario.map_manager
        ref_counts = Counter(int(u.reference_id) for _p, u in unit_references.all_units(self.scenario))
        refused: dict[tuple[int, int], str] = {}
        pool = []
        for e in entries:
            if ref_counts[int(e.unit.reference_id)] > 1:
                refused[unit_pick.unit_key(e.player_id, e.unit)] = "reference id placed more than once"
            elif unit_kind.owner_refused(new_const, e.player_id):
                # GH #32: a player's unit never becomes a GAIA-only object.
                refused[unit_pick.unit_key(e.player_id, e.unit)] = GAIA_ONLY_REPLACE_REASON
            else:
                pool.append(e)
        reasons = model.replace_batch_refusals([e.unit for e in pool], new_const, mm.map_width, mm.map_height)
        candidates = []
        for e, reason in zip(pool, reasons, strict=True):
            if reason is None:
                candidates.append(e)
            else:
                refused[unit_pick.unit_key(e.player_id, e.unit)] = reason
        return candidates, refused

    def _replace_units(self, entries, new_const: int, follow_writes=(), trig_model=None) -> tuple[int, dict]:
        """Find's Replace (GH #144): every entry becomes `new_const` via
        UnitEditModel.replace_type(), in one wholesale undo record. Returns
        (replaced, {key: refusal reason}).

        The batch pre-flight (replace_refusal() with `planned`) runs to a
        fixpoint first, so the edit cannot raise midway. fields_only=False on
        purpose: a fields-only record undoes through per-unit splices that read
        only the current const, so redoing wall->tree would leave neighbouring
        walls drawn with a stale shape. A moved host's garrison moves to its
        new point in the same record, where _move_units keeps it.

        Into a gate const (GH #339), the walls under each new gate's
        footprint go first and the walls beside its old and new tiles
        re-derive last (_gate_change_plan()), in the same record. A row the
        pre-flight refused is never removed, and no candidate is: two
        candidates that become overlapping gates stay so.

        Into any other const (TASK-584), the walls beside a tile that starts
        or stops holding a wall or gate re-derive, a unit turned into a wall
        from anything else takes the shape its new neighbours give it, and a
        wall kept a wall re-derives only when a tile beside it changed
        (_replace_wall_plan()). Nothing is removed, so a unit replaced into a
        wall on a wall's tile stays stacked. A refused row keeps its const and
        counts as a neighbour."""
        if self.scenario is None or not entries:
            return 0, {}
        model = self._ensure_unit_edits()
        if model is None:
            return 0, {}
        mm = self.scenario.map_manager
        candidates, refused = self._replace_preflight(model, entries, new_const)
        gate_plan = None
        wall_plan = None
        if candidates:
            movers = [e for e in candidates if replaced_position(e.unit, new_const) != (e.unit.x, e.unit.y)]
            riders = {id(e.unit): self._garrison_occupant_entries(model, [e]) for e in movers}
            players = {e.player_id for e in candidates}
            players |= {r.player_id for rows in riders.values() for r in rows}
            if gate_orientation.is_gate(new_const):
                changes = [(e.unit, new_const, *replaced_position(e.unit, new_const)) for e in candidates]
                gate_plan = self._gate_change_plan(model, changes, keep_ids=frozenset(id(e.unit) for e in entries))
                if gate_plan:
                    players |= wall_run.gate_move_players(gate_plan)
            else:
                wall_plan = self._replace_wall_plan(
                    [(e.player_id, e.unit, new_const, *replaced_position(e.unit, new_const)) for e in candidates]
                )
                if wall_plan:
                    players |= {player_id for player_id, _unit, _variant in wall_plan.rewrites}
            label = "Replace unit" if len(candidates) == 1 else f"Replace {len(candidates)} units"
            follow_writes = list(follow_writes) if trig_model is not None else []
            edit = (
                self._composite_edit(
                    label, model, sorted(players), trig_model, sorted({w.trigger_index for w in follow_writes})
                )
                if follow_writes
                else self._unit_edit(model, label, sorted(players))
            )
            with edit:
                if gate_plan:
                    # First: the only call that can refuse, so a refusal leaves nothing replaced.
                    model.remove_many([wall for _player_id, wall in gate_plan.removals])
                planned = {e.unit.reference_id: new_const for e in candidates}
                for e in candidates:
                    model.replace_type(e.unit, new_const, map_w=mm.map_width, map_h=mm.map_height, planned=planned)
                for e in movers:
                    host = e.unit
                    for rider in riders[id(host)]:
                        model.set_position(rider.unit, host.x, host.y, getattr(host, "z", 0.0))
                if gate_plan:
                    wall_run.apply_gate_move_plan(model, gate_plan)
                if wall_plan:
                    wall_run.apply_replace_plan(model, wall_plan)
                if follow_writes:
                    find_triggers.write_fields(trig_model.manager(), follow_writes)
        status = f"Replaced {len(candidates)}" + self._gate_move_clauses(gate_plan)
        if wall_plan:
            status += _reshaping_clause(len(wall_plan.rewrites))
        if refused:
            reasons = Counter(refused.values())
            detail = ", ".join(f"{n} {reason}" for reason, n in reasons.most_common())
            status += f"; {len(refused)} refused ({detail})"
        self._log_status(status)
        return len(candidates), refused

    def _select_unit_keys(self, keys) -> tuple[int, int]:
        """Find's Select on Map: switches to Units mode and selects every key
        the pick index resolves. Filtered, garrisoned and off-map units have no
        entry; they are counted, not revealed (no filter flips in v1).
        Returns (selected, hidden)."""
        keys = list(keys)
        if self.scenario is None or not keys:
            return 0, 0
        # The pick index, and so selection, only exists in Units mode.
        self.mode_combo.setCurrentText("Units")
        index = self.map_view._unit_index
        resolved = [k for k in keys if index is not None and index.entry_for_key(k) is not None]
        self._selection = resolved
        self._refresh_selection_view()
        hidden = len(keys) - len(resolved)
        status = f"Selected {len(resolved)} of {len(keys)}"
        if hidden:
            status += f"; {hidden} hidden by filters / garrisoned / off-map"
        self._log_status(status)
        return len(resolved), hidden

    def _centre_on_unit_key(self, key) -> bool:
        """Centres the view on one unit, resolved over every placed unit
        (never the pick index): _navigate_to_finding's centre step."""
        if self.scenario is None:
            return False
        resolved = find_objects.resolve_keys(self.scenario, [key])
        if not resolved:
            self._log_status("That object is no longer on the map; re-run Find")
            return False
        unit = resolved[0][1]
        mm = self.scenario.map_manager
        x = max(0, min(mm.map_width - 1, int(unit.x)))
        y = max(0, min(mm.map_height - 1, int(unit.y)))
        self.map_view.center_on_tile(x, y)
        return True

    def _default_warn(self, title: str, text: str) -> None:
        QMessageBox.warning(self, title, text)

    def _default_confirm(self, title: str, text: str, choices=(), actions=()) -> dict | None:
        """None when cancelled, else {choice key: checked} (empty with no choices)."""
        return confirm_with_choices(self, title, text, choices, actions)
