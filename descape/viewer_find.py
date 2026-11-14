"""ViewerWindow's analysis tools: Analysis, Find and Replace, trigger range rings and rulers."""

from __future__ import annotations

import time

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QLineEdit,
)

from descape import (
    debug_log,
    find_objects,
    find_triggers,
    library_compat,
    map_analysis,
    range_overlay,
    ruler,
    trigger_range,
    unit_pick,
)
from descape.analysis_dialog import AnalysisDialog
from descape.edit_history import (
    RulerDiffRecord,
)
from descape.find_dialog import FindCallbacks, FindDialog
from descape.scenario_io import (
    parse_triggers,
    unsupported_version_sentence,
)
from descape.text_edits import _MultiLineEdit
from descape.trigger_model import (
    TriggerEditModel,
)
from descape.trigger_panel import TriggerPanel
from descape.unit_filter import GAIA_PLAYER_ID
from descape.viewer_common import (
    _gaia_only_refused_note,
    _unit_name,
)


class _ViewerFindMixin:
    """ViewerWindow's Analysis, Find and Replace and rulers (viewer.py's docstring has the split's rules)."""

    def _show_analysis(self) -> None:
        """Tools > Map Analysis: a read-only pass, then the modeless results
        dialog. Busy-guarded like load_scenario(): the trigger parse inside
        can take seconds on a large file."""
        if self.scenario is None or self._busy:
            return
        self._busy = True
        self.setEnabled(False)
        triggers_known = self.scenario.trigger_read_supported
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            started = time.perf_counter()
            report = map_analysis.analyze(self.scenario)
            elapsed = time.perf_counter() - started
        finally:
            QApplication.restoreOverrideCursor()
            self.setEnabled(True)
            self._busy = False
        self._log_status(f"Map Analysis: {report.headline} ({elapsed:.2f}s)")
        # The analysis just parsed the triggers; the next mode switch can't notice the flip.
        self._update_info()
        if self.scenario.trigger_read_supported != triggers_known:
            self._sync_trigger_ranges()
        if self._analysis_dialog is None:
            self._analysis_dialog = AnalysisDialog(
                self,
                on_navigate=self._navigate_to_finding,
                on_select=self._focus_analysis_marker,
                on_closed=self.map_view.clear_analysis_markers,
                on_refresh=self._show_analysis,
            )
        mm = self.scenario.map_manager
        self.map_view.set_analysis_markers(map_analysis.marker_anchors(report, mm.map_width, mm.map_height))
        # After the markers: a push drops the focus ring, which a row kept across a re-run sets again.
        self._analysis_dialog.set_report(report, self.scenario.path.name)
        self._analysis_dialog.show()
        self._analysis_dialog.raise_()
        self._analysis_dialog.activateWindow()

    def _show_find_dialog(self) -> None:
        """Edit > Find and Replace… (GH #144): the modeless dialog, reused."""
        if self.scenario is None:
            return
        if self._find_dialog is None:
            callbacks = FindCallbacks(
                find_objects=self._find_objects,
                select_keys=self._select_unit_keys,
                centre_on_key=self._centre_on_unit_key,
                delete_keys=self._find_delete_keys,
                replace_preflight=self._find_replace_preflight,
                replace_keys=self._find_replace_keys,
                reassign_keys=self._find_reassign_keys,
                region=lambda: self._region,
                warn=self._find_warn,
                confirm=self._find_confirm,
                trigger_status=self._find_trigger_status,
                find_triggers=self._find_trigger_hits,
                text_replace=self._find_text_replace,
                type_replace=self._find_type_replace,
                set_enabled=self._find_set_enabled,
                delete_triggers=self._find_delete_triggers,
                reveal=self._reveal_trigger_entry,
                follow_preview=self._find_follow_preview,
                trigger_refs_for=self._find_trigger_refs_for,
            )
            self._find_dialog = FindDialog(self, callbacks)
            self._find_dialog.refresh_player_labels(self._player_labels, tuple(self.scenario.player_colors))
        self._find_dialog.region_changed(self._region)
        self._find_dialog.show()
        self._find_dialog.raise_()
        self._find_dialog.activateWindow()

    def _close_find_dialog(self) -> None:
        """Its rows describe the document they were found in, and so do the cached counts."""
        self._trigger_ref_cache = None
        if self._find_dialog is not None:
            self._find_dialog.close()
            self._find_dialog.deleteLater()
            self._find_dialog = None

    # Late-bound, so a test that stubs _warn/_confirm after the dialog exists still reaches the stub.
    def _find_warn(self, title: str, text: str) -> None:
        self._warn(title, text)

    def _find_confirm(self, title: str, text: str, choices=(), actions=()) -> dict | None:
        return self._confirm(title, text, choices, actions)

    def _find_dialog_mark_stale(self) -> None:
        """Unit edits: object rows and trigger placed-object hits change, the counts don't."""
        if self._find_dialog is not None:
            self._find_dialog.mark_stale("units")

    def _find_objects(self, criteria) -> list:
        if self.scenario is None:
            return []
        mm = self.scenario.map_manager
        counts = self._find_trigger_ref_counts()
        return find_objects.find(self.scenario, criteria, mm.map_width, mm.map_height, trigger_ref_counts=counts)

    def _find_trigger_ref_counts(self) -> dict | None:
        """Cached behind _trigger_gen; None (a blank column) while triggers are unparsed."""
        if self.scenario is None:
            return None
        cache = self._trigger_ref_cache
        if cache is not None and cache[0] == self._trigger_gen and cache[1] is not None:
            return cache[1]
        counts = find_triggers.trigger_ref_counts(self.scenario)
        self._trigger_ref_cache = (self._trigger_gen, counts)
        return counts

    def _find_triggers_bumped(self) -> None:
        """Every trigger edit and trigger undo/redo: counts change, both tabs go stale."""
        self._trigger_gen += 1
        if self._find_dialog is not None:
            self._find_dialog.mark_stale("triggers")

    # -- trigger range rings (GH #49 slice 2) ------------------------------

    def _trigger_range_effects(self):
        """The file's range-changing effects (a tuple), or trigger_range.UNREAD
        / UNREADABLE. Never forces the parse: until Triggers mode, Find or Map
        Analysis has parsed, this is UNREAD (find_triggers.trigger_ref_counts()'
        rule). Cached behind _trigger_gen; a sentinel never is, so a parse
        since the last ask is picked up."""
        scenario = self.scenario
        if scenario is None:
            return trigger_range.UNREAD
        version = scenario.scenario_version
        if scenario.trigger_read_supported is False or not (
            version and library_compat.vocabulary_is_available(version)
        ):
            return trigger_range.UNREADABLE
        if scenario._trigger_manager is None:
            return trigger_range.UNREAD
        cache = self._trigger_range_cache
        if cache is not None and cache[0] == self._trigger_gen:
            return cache[1]
        manager = parse_triggers(scenario)
        if manager is None:
            return trigger_range.UNREADABLE
        effects = trigger_range.collect_range_effects(manager, library_compat.load_vocabulary(version))
        self._trigger_range_cache = (self._trigger_gen, effects)
        return effects

    def _push_trigger_range_effects(self):
        """MapView's half: the effects (or None for a sentinel). Returns what it read."""
        effects = self._trigger_range_effects()
        self.map_view.set_trigger_range_effects(effects if isinstance(effects, tuple) else None)
        return effects

    def _show_trigger_range_line(self, effects) -> None:
        """The Inspector's half, for exactly one selected unit; any other
        count hides the line. Ignores both View toggles: it never parses."""
        index = self.map_view._unit_index
        entries = [] if index is None else [e for e in map(index.entry_for_key, self._selection) if e is not None]
        if len(entries) != 1:
            self.units_panel.show_trigger_ranges("")
            return
        entry = entries[0]
        unit_const = entry.unit.unit_const
        ringable = range_overlay.is_ring_building(unit_const)
        ranges: tuple = ()
        unresolved = 0
        state = effects
        range_index = self.map_view._trigger_range_index
        if isinstance(effects, tuple):
            state = None
            if ringable and range_index is not None:
                unit = {
                    "player_id": entry.player_id,
                    "unit_const": unit_const,
                    "reference_id": entry.unit.reference_id,
                    "own_tile": (entry.own_x, entry.own_y),
                }
                ranges = range_overlay.building_trigger_ranges(range_index, **unit)
                unresolved = trigger_range.unresolved_count(range_index, **unit)
        self.units_panel.show_trigger_ranges(
            *trigger_range.describe(ranges, unresolved, state=state, ringable=ringable)
        )

    def _sync_trigger_ranges(self) -> None:
        """Push the effects to MapView, then the Inspector line for the viewer's own selection."""
        self._show_trigger_range_line(self._push_trigger_range_effects())

    def _invalidate_trigger_ranges(self) -> None:
        """After a trigger edit or undo, behind _find_triggers_bumped() (the
        cache keys on its counter). Re-walks only where a drawn selection shows
        the result: a kept selection in Triggers mode is not drawn, and the
        switch back to Units pushes again (Roll The Dice holds 18,135 triggers).
        Safe because both selection producers push first."""
        self.map_view.set_trigger_range_effects(None)
        if (self._selection_live() and self._selection) or (
            self._unit_picker is not None and self.map_view._unit_select_keys
        ):
            self._sync_trigger_ranges()

    def _find_trigger_status(self) -> tuple[str | None, bool]:
        """(why triggers can't be read, or None; whether they can be written).
        Parses on first call, which Find does on the Triggers tab's first show."""
        if self.scenario is None:
            return TriggerPanel._NO_DOCUMENT, False
        triggers_known = self.scenario.trigger_read_supported
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            manager = parse_triggers(self.scenario)
        finally:
            QApplication.restoreOverrideCursor()
        if self.scenario.trigger_read_supported != triggers_known:
            # A first parse can light up a live selection's trigger rings (GH #49 slice 2).
            self._sync_trigger_ranges()
        if manager is None:
            key = TriggerPanel._unsupported_key(self.scenario)
            sentence = unsupported_version_sentence(self.scenario.scenario_version)
            return TriggerPanel._UNSUPPORTED[key].format(sentence=sentence), False
        # Only meaningful once parse_triggers() has run (it is set there).
        return None, bool(self.scenario.trigger_write_supported)

    def _find_trigger_hits(self, criteria) -> list:
        if self.scenario is None:
            return []
        return find_triggers.find_triggers(self.scenario, criteria)

    def _commit_focused_field(self) -> None:
        """Commit the focused field's typed value as leaving it would: a shortcut, a window close
        or another window's op leaves no field, and a multi-line box ignores a menu or window switch."""
        # A locked document refuses the edit, but the field would count it as committed.
        if self.game_data_lock_reason():
            return
        # A floating panel is a window of its own, with its own focus widget (GH #64).
        # An inactive float keeps its focus widget across a mode change, so skip one on a hidden page.
        windows = (self, self.panel_dock) if self.panel_dock.isFloating() else (self,)
        for window in windows:
            widget = window.focusWidget()
            if not self._owns_widget(widget) or (window is not self and not widget.isVisible()):
                continue
            if isinstance(widget, QAbstractSpinBox):
                widget.interpretText()
                widget.editingFinished.emit()
            elif isinstance(widget, _MultiLineEdit) or (isinstance(widget, QLineEdit) and widget.hasAcceptableInput()):
                widget.editingFinished.emit()

    def _find_trigger_op_model(self) -> TriggerEditModel | None:
        if self._find_op_refused():
            return None
        self._commit_focused_field()
        if parse_triggers(self.scenario) is None or not self.scenario.trigger_write_supported:
            self._log_status("Find and Replace: this file's triggers are read-only", level=debug_log.WARNING)
            return None
        return self._ensure_trigger_edits()

    def _apply_trigger_writes(self, model, plan, label: str) -> None:
        """One _trigger_edit declaring only the triggers that change. An empty
        plan opens nothing: commit_trigger_edit() pushes even for no change."""
        if not plan.writes:
            return
        with self._trigger_edit(model, label, content_touched=plan.touched, refresh="panel") as m:
            find_triggers.write_fields(m.manager(), plan.writes)

    def _find_text_replace(self, hits, text: str, regex: bool, case_sensitive: bool, replacement: str) -> dict:
        """Find's Replace text: returns {hit location: reason} for refused and stale hits."""
        model = self._find_trigger_op_model()
        if model is None:
            return {}
        valid, stale = find_triggers.revalidate(self.scenario, hits)
        pattern = find_triggers.compile_pattern(text, regex, case_sensitive)
        if pattern is None:
            return {}
        try:
            plan = find_triggers.plan_text_replace(self.scenario, valid, pattern, replacement, regex)
        except find_triggers.FindPatternError as e:
            self._warn("Replace text", str(e))
            return {}
        n = len(plan.writes)
        self._apply_trigger_writes(model, plan, f"Replace text in {n} field{'s' if n != 1 else ''}")
        return self._trigger_op_report("Replaced text in", plan, stale)

    def _find_type_replace(self, hits, old_consts, new_const: int) -> dict:
        model = self._find_trigger_op_model()
        if model is None:
            return {}
        valid, stale = find_triggers.revalidate(self.scenario, hits)
        plan = find_triggers.plan_type_replace(self.scenario, valid, frozenset(old_consts), new_const)
        n = len(plan.writes)
        self._apply_trigger_writes(model, plan, f"Replace type in {n} field{'s' if n != 1 else ''}")
        return self._trigger_op_report("Replaced type in", plan, stale)

    def _trigger_op_report(self, verb: str, plan, stale) -> dict:
        status = f"{verb} {len(plan.writes)} fields across {len(plan.touched)} triggers"
        if plan.refusals:
            status += f"; {len(plan.refusals)} refused"
        if stale:
            status += f"; {len(stale)} results changed since Find; re-run Find"
        self._log_status(status)
        notes = {h.location: reason for h, reason in plan.refusals}
        notes.update({h.location: "changed since Find; re-run Find" for h in stale})
        return notes

    def _parse_triggers_syncing(self):
        """parse_triggers(), plus the trigger-ring sync a first parse owes a live selection (GH #49)."""
        triggers_known = self.scenario.trigger_read_supported
        manager = parse_triggers(self.scenario)
        if self.scenario.trigger_read_supported != triggers_known:
            self._sync_trigger_ranges()
        return manager

    def _find_valid_trigger_indices(self, parents) -> list[int]:
        """`parents` is {trigger_index: (name, child hits)}. A trigger still
        holds its name and at least one valid child (or matched on its own)."""
        manager = self._parse_triggers_syncing()
        valid_children, _stale = find_triggers.revalidate(
            self.scenario, [h for _name, children in parents.values() for h in children]
        )
        still = {h.trigger_index for h in valid_children}
        result = [
            index
            for index, (name, _children) in parents.items()
            if index in still
            and manager is not None
            and 0 <= index < len(manager.triggers)
            and (manager.triggers[index].name or "") == name
        ]
        if len(result) < len(parents):
            self._log_status(f"{len(parents) - len(result)} results changed since Find; re-run Find")
        return result

    def _find_set_enabled(self, parents, value: bool) -> int:
        model = self._find_trigger_op_model()
        if model is None:
            return 0
        indices = self._find_valid_trigger_indices(parents)
        triggers = model.manager().triggers
        changing = [i for i in indices if bool(triggers[i].enabled) != value]
        if not changing:
            return 0
        label = f"{'Enable' if value else 'Disable'} {len(changing)} trigger{'s' if len(changing) != 1 else ''}"
        with self._trigger_edit(model, label, content_touched=changing, refresh="panel") as m:
            for i in changing:
                m.manager().triggers[i].enabled = int(value)
        return len(changing)

    def _find_delete_triggers(self, parents) -> int:
        model = self._find_trigger_op_model()
        if model is None:
            return 0
        indices = self._find_valid_trigger_indices(parents)
        if indices:
            self.trigger_structural_edit("delete", indices)
        return len(indices)

    def _reveal_trigger_entry(self, trigger_index: int, kind: str, entry_index: int) -> None:
        """Find's Jump: Triggers mode, the panel's filter and tag facet cleared
        (selected_trigger_indices() skips hidden rows), then the row and entry."""
        if self.scenario is None:
            return
        self.mode_combo.setCurrentText("Triggers")
        panel = self.trigger_panel
        if panel.filter_edit.text():
            panel.filter_edit.clear()
        if panel.tag_combo.currentData() is not None:
            panel.tag_combo.setCurrentIndex(0)
        panel.reveal_trigger(trigger_index)
        if kind in ("condition", "effect") and entry_index >= 0:
            panel.select_entry(kind, entry_index)

    def _find_follow_preview(self, keys, new_const: int):
        """(FollowPlan or None, note) for an object Replace's confirm. Both models
        are ensured here, before the confirm, so the bracket never meets None."""
        if self._find_op_refused():
            return None, ""
        model = self._ensure_unit_edits()
        if model is None:
            return None, ""
        status, writable = self._find_trigger_status()
        if status is not None:
            return None, "Trigger type filters are not followed: this file's triggers can't be read."
        if not writable:
            return None, "Trigger type filters are not followed: triggers are read-only for this file."
        self._commit_focused_field()
        if self._ensure_trigger_edits() is None:
            return None, "Trigger type filters are not followed: triggers are read-only for this file."
        candidates, _refused = self._replace_preflight(model, self._find_entries(keys), new_const)
        replaced = {int(e.unit.reference_id): (int(e.unit.unit_const), new_const) for e in candidates}
        return find_triggers.plan_follow_replace(self.scenario, replaced), ""

    def _find_trigger_refs_for(self, keys) -> int | None:
        """How many trigger fields reference these objects (parsing if needed), or None if unreadable."""
        if self.scenario is None or self._parse_triggers_syncing() is None:
            return None
        counts = self._find_trigger_ref_counts() or {}
        return sum(counts.get(ref, 0) for _player, ref in keys)

    def _find_entries(self, keys) -> list:
        """Duck-typed (player_id, unit) entries for keys, over every placed unit."""
        return [
            unit_pick.UnitEntry(player_id, unit, int(unit.x), int(unit.y), -1)
            for player_id, unit in find_objects.resolve_keys(self.scenario, keys)
        ]

    def _find_op_refused(self) -> bool:
        """A Find op is refused mid-stroke or while busy, as a History jump is."""
        if self.scenario is None or self._busy or self._stroke_in_progress():
            self._log_status("Find and Replace: finish the current edit first")
            return True
        return False

    def _find_delete_keys(self, keys) -> tuple[int, int, int]:
        if self._find_op_refused():
            return 0, 0, 0
        return self._delete_units(self._find_entries(keys))

    def _find_unresolved(self, keys, entries) -> dict:
        found = {unit_pick.unit_key(e.player_id, e.unit) for e in entries}
        return {key: "no longer on the map; re-run Find" for key in keys if key not in found}

    def _find_replace_preflight(self, keys, new_const: int) -> dict:
        if self._find_op_refused():
            return dict.fromkeys(keys, "busy")
        model = self._ensure_unit_edits()
        if model is None:
            return dict.fromkeys(keys, "units are read-only for this file")
        entries = self._find_entries(keys)
        _candidates, refused = self._replace_preflight(model, entries, new_const)
        return {**self._find_unresolved(keys, entries), **refused}

    def _find_replace_keys(self, keys, new_const: int, follow_writes=()) -> tuple[int, dict]:
        """`follow_writes` are the chosen tiers of _find_follow_preview()'s plan."""
        if self._find_op_refused():
            return 0, dict.fromkeys(keys, "busy")
        entries = self._find_entries(keys)
        trig_model = self.trigger_edits if follow_writes else None
        replaced, refused = self._replace_units(entries, new_const, follow_writes, trig_model)
        return replaced, {**self._find_unresolved(keys, entries), **refused}

    def _find_reassign_keys(self, keys, new_player: int) -> dict:
        if self._find_op_refused():
            return {}
        remapped: dict = {}
        entries = self._find_entries(keys)
        changed, carried, refused, gaia_only = self._reassign_units(
            entries, new_player, on_applied=remapped.update, skip_orphan_occupants=True
        )
        owner = "GAIA" if new_player == GAIA_PLAYER_ID else f"Player {new_player}"
        status = f"Reassigned {changed} units to {owner}"
        if carried:
            status += f" ({carried} garrisoned)"
        if refused:
            status += f"; {refused} refused (garrisoned in a host not being reassigned)"
        status += _gaia_only_refused_note(gaia_only)
        self._log_status(status)
        return remapped

    def _close_analysis_dialog(self) -> None:
        """Its findings describe the document they were run on; a stale row
        must not navigate inside a different map."""
        if self._analysis_dialog is not None:
            self._analysis_dialog.close()
            self._analysis_dialog.deleteLater()
            self._analysis_dialog = None
        self.map_view.clear_analysis_markers()

    def _focus_analysis_marker(self, finding: map_analysis.Finding | None) -> None:
        """The dialog's current row: ring its marker, clamped like the anchors."""
        if finding is None or self.scenario is None:
            self.map_view.set_analysis_marker_focus(None)
            return
        mm = self.scenario.map_manager
        self.map_view.set_analysis_marker_focus(map_analysis.marker_tile(finding, mm.map_width, mm.map_height))

    def _navigate_to_finding(self, finding: map_analysis.Finding) -> None:
        if self.scenario is None:
            return
        tile = finding.tile
        if finding.unit_key is not None:
            # The pick index, and so selection, only exists in Units mode.
            self.mode_combo.setCurrentText("Units")
            index = self.map_view._unit_index
            entry = index.entry_for_key(finding.unit_key) if index is not None else None
            if entry is not None:
                self._selection = [finding.unit_key]
                self._refresh_selection_view()
                tile = (int(entry.unit.x), int(entry.unit.y))
                self._log_status(
                    f"Selected {_unit_name(entry.unit.unit_const)} at ({entry.unit.x:g}, {entry.unit.y:g})"
                )
        if tile is None:
            return
        mm = self.scenario.map_manager
        x = max(0, min(mm.map_width - 1, tile[0]))
        y = max(0, min(mm.map_height - 1, tile[1]))
        self.map_view.center_on_tile(x, y)

    def on_ruler_measured(self, measurement) -> None:
        """MapView's completed-measurement callback. Logs the same string the
        on-map label shows, plus both endpoints, so the reading survives being
        scrolled off screen and can be compared against an earlier one; the
        label alone only ever shows the most recent measurement."""
        a_x, a_y = measurement.a
        b_x, b_y = measurement.b
        self._log_status(
            f"Ruler: {ruler.format_measurement(measurement)} "
            f"from ({a_x}, {a_y}) to ({b_x}, {b_y})"
        )
        # GH #108: once per session, on the first ruler that actually stays on the map.
        if measurement.a != measurement.b and not self._ruler_hint_logged:
            self._ruler_hint_logged = True
            self._log_status("Right-click a ruler's end to remove it, or use Clear rulers")

    def _on_pinned_rulers_changed(self, count: int) -> None:
        """MapView's pinned-ruler count hook: re-gates Clear rulers."""
        self.clear_rulers_button.setEnabled(
            count > 0 and self._current_tool == "ruler" and self.ruler_action.isEnabled()
        )

    def _on_pinned_rulers_edited(self, label: str, added: tuple, removed: tuple) -> None:
        """GH #108: a user ruler add, remove or Clear rulers becomes a view-only
        undo step. With file edits to redo it is drawn but not recorded, since a
        push would throw that redo away; ruler-only redo is discarded as usual."""
        if self.scenario is None:
            return
        if self.edit_history.redo_has_file_edits:
            if not self._ruler_unrecorded_noted:
                self._ruler_unrecorded_noted = True
                self._log_status("Ruler changes are not added to Undo while there are undone edits to redo")
            return
        self._ruler_unrecorded_noted = False
        self.edit_history.push_ruler_record(RulerDiffRecord(label, added, removed, self.map_view.apply_ruler_delta))
        self._update_edit_actions()
        self._update_title()

    def on_ruler_changed(self, measurement) -> None:
        """MapView's live-update callback: fires on every drag frame, not
        just a completed one, so this is what keeps the status bar in sync
        while the on-map label is off screen. `measurement` is the pending one
        while a measurement is in progress, otherwise the newest pinned ruler,
        and None once none is left (Clear rulers, File > Close/New)."""
        self.ruler_status_label.setText(
            f"  {ruler.format_measurement(measurement)}  " if measurement is not None else ""
        )
