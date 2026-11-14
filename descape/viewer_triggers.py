"""ViewerWindow's trigger editing: trigger, entry and tag edits, the trigger and entry clipboards,
structural edits, sections and variables."""

from __future__ import annotations

import copy
import functools
from collections.abc import Sequence
from contextlib import contextmanager, suppress

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
)

from descape import (
    debug_log,
    library_compat,
    perf_trace,
    trigger_clipboard,
    trigger_fields,
    trigger_organize,
)
from descape.edit_history import (
    CompositeDiffRecord,
)
from descape.scenario_io import (
    parse_triggers,
)
from descape.trigger_model import (
    EXEC_MODE_DISPLAY,
    EXEC_MODE_LEGACY,
    EXEC_MODE_UNKNOWN,
    TriggerEditModel,
    TriggerEditsUnavailableError,
    block_room,
    display_order_moved_to_slot,
    display_order_with_block_inserted,
    display_order_with_copies_inserted,
    exec_order_value,
    id_order_refusal,
    id_order_to_edge,
    move_entries,
    moved_display_order_block,
    moved_id_order,
    moved_section_display_order,
    moved_to_section_edge,
    paste_anchor,
    remove_entries,
    renumber_refusal,
    renumber_to_display_order,
    renumber_to_order,
    resolve_exec_mode,
    section_edge_refusal,
    would_move_entries,
)
from descape.unit_model import (
    UnitEditModel,
)
from descape.viewer_common import (
    _entries_of,
)

# TASK-053.02: said once per document, on the first display-order move in a legacy file.
LEGACY_REORDER_MESSAGE = (
    "This file runs triggers in ID order (legacy), so moving them in the list does not change "
    "which runs first. Renumber IDs to Display Order, on the trigger list's right-click menu, "
    "makes them run in the listed order, and moving them in File order (trigger ID) view changes "
    "their IDs; Map Options' Trigger execution order switches the file to display order instead."
)
# GH #2: the status line after Renumber IDs to Display Order, one per execution mode.
LEGACY_RENUMBER_MESSAGE = "This file runs triggers in ID order (legacy), so they now run in the listed order."
DISPLAY_RENUMBER_MESSAGE = "This file runs triggers in display order, so which trigger runs first is unchanged."
UNKNOWN_RENUMBER_MESSAGE = (
    "This file stores no execution order setting; if the game runs it by ID, "
    "which trigger runs first may have changed."
)
# TASK-569: after a File order view ID move, once per document and mode; a legacy file says nothing.
DISPLAY_ID_MOVE_MESSAGE = (
    "This file runs triggers in display order, so changing IDs does not change which runs first."
)
UNKNOWN_ID_MOVE_MESSAGE = UNKNOWN_RENUMBER_MESSAGE


class _ViewerTriggersMixin:
    """ViewerWindow's trigger editing (viewer.py's docstring has the split's rules)."""

    # -- trigger editing ----------------------------------------------------
    #
    # Every trigger mutation in the app goes through _trigger_edit(). The model
    # contract behind it (trigger_model.py) fails silently in every direction:
    # an edit that skips mark_dirty() splices its pre-edit bytes back on save,
    # and a mutation that sets model dirtiness without pushing a record closes
    # the document with no save prompt. Routing every call site through one
    # helper is what keeps those from being per-call-site mistakes.

    def _ensure_trigger_edits(self) -> TriggerEditModel | None:
        """The document's TriggerEditModel, built on first use.

        Construction is deferred to the first real edit, not done at panel
        populate: that is what keeps a browse-only session's save
        byte-identical. A file whose triggers parsed can still refuse to be
        edited (the alignment gate), so this can return None after a
        successful parse.
        """
        if self._refuse_locked_edit():
            return None
        if self.trigger_edits is not None:
            return self.trigger_edits
        if self.scenario is None:
            return None
        try:
            self.trigger_edits = TriggerEditModel(self.scenario)
        except TriggerEditsUnavailableError as e:
            self._log_status(f"Triggers are read-only for this file: {e}", level=debug_log.WARNING)
            self._warn("Triggers are read-only", str(e))
            return None
        return self.trigger_edits

    @contextmanager
    def _composite_edit(
        self,
        label: str,
        unit_model: UnitEditModel | None = None,
        players: Sequence[int] = (),
        trig_model: TriggerEditModel | None = None,
        content_touched: Sequence[int] = (),
    ):
        """One undo step across units, triggers and option fields (GH #144's
        Replace that retargets trigger type filters, GH #185's Swap Players):
        each model half present commits with push=False, and the
        OptionsDiffRecords the caller appends to the yielded list join them in
        one CompositeDiffRecord.

        Takes models, never ensures them: the caller ran the _ensure_*() it
        needs before its confirm. Every plan is resolved before the bracket
        opens, so a raise inside is a bug; every half still commits as one
        composite so whatever mutated stays undoable (never abort only one
        half), and the error goes through _warn, not a box of its own."""
        records: list = []
        with perf_trace.op(label):
            if unit_model is not None:
                unit_model.begin_unit_edit(players, fields_only=False)
            if trig_model is not None:
                try:
                    trig_model.begin_trigger_edit(content_touched)
                except Exception:
                    if unit_model is not None:
                        unit_model.abort_unit_edit()
                    raise
            error: Exception | None = None
            try:
                yield records
            except Exception as e:  # noqa: BLE001 -- reported below, never swallowed
                error = e
            children = []
            if unit_model is not None:
                children.append(unit_model.commit_unit_edit(label, self.edit_history, push=False))
            if trig_model is not None:
                children.append(trig_model.commit_trigger_edit(label, self.edit_history, push=False))
            children.extend(records)
            if children:
                self.edit_history.push_composite_record(CompositeDiffRecord(label, children))
            if unit_model is not None:
                self._after_unit_mutation()
            if trig_model is not None:
                self._after_trigger_edit(label, None, "panel")
            if error is not None:
                self._log_status(f"{label} failed: {type(error).__name__}: {error}", level=debug_log.ERROR)
                self._warn(
                    f"{label} failed",
                    f"{error}\n\nWhat was changed before the failure is one undo step; Ctrl+Z reverts it.",
                )

    @contextmanager
    def _trigger_edit(
        self,
        model: TriggerEditModel,
        label: str,
        content_touched: Sequence[int] = (),
        refresh: str = "",
        refresh_arg=None,
    ):
        """Open one recorded trigger edit, and close it whatever happens.

        `content_touched` names every trigger this edit may rewrite in place,
        in pre-edit numbering. Signature diffing cannot see that class of
        change, so an undeclared one is spliced away on save.

        `refresh` is how much of the panel the edit invalidates: "" for a field
        edit, which patches its own row; "entries" for a condition or effect
        added or removed; "order" for a display-order-only move, which patches
        the moved row's position and nothing else (`refresh_arg` carries the
        (trigger_index, delta) that move needs); "variables" for a variable
        added or removed, which no tree in the panel shows at all; "panel" for
        anything that changes the trigger list itself.

        Enter only once a real value change is confirmed. commit_trigger_edit()
        pushes unconditionally by design, so entering on a focus-out, on a
        programmatic widget populate, or on merely opening the picker would
        record a phantom undo step and dirty the trigger.
        """
        # Only the tiers that can move sections; a field edit's rename has _retitle_section().
        collapse = self.trigger_panel.collapse_snapshot() if refresh in ("panel", "order") else None
        model.begin_trigger_edit(content_touched)
        error: Exception | None = None
        try:
            yield model
        except Exception as e:  # noqa: BLE001 -- reported below, never swallowed
            error = e
        finally:
            # Commits even on failure: a raised structural_edit() is not a
            # no-op, it marks every blob dirty, so that state still needs a
            # record to undo.
            model.commit_trigger_edit(label, self.edit_history)
            if error is None:
                self.trigger_panel.follow_collapse(collapse)
            self._after_trigger_edit(label, error, refresh, refresh_arg)

    def _after_trigger_edit(
        self, label: str, error: Exception | None, refresh: str = "", refresh_arg=None
    ) -> None:
        if refresh == "variables":
            # The non-modal variables dialog can remain open in any mode.
            self.trigger_panel.refresh_variables()
        elif refresh and self.mode == "triggers":
            if refresh == "panel":
                # Rebuilt rather than patched, for the same reason
                # _move_history rebuilds on an undo: a structural edit can
                # change membership, order and ids at once, and there is no
                # partial update that is cheaper to get right than a
                # repopulate.
                self._show_triggers()
            elif refresh == "order" and error is None:
                # The one tier that is neither a full repopulate nor a no-op:
                # a display-order move changes nothing but one row's position,
                # and Move Up/Down is a repeated click -- "panel"'s full
                # rebuild would scroll a 590-trigger list to the top on every
                # press, which is the exact slowness DEscape exists to escape.
                #
                # Only on success. move_row() blindly moves one row by its delta without
                # reading the model, unlike show_scenario() which re-reads it
                # regardless -- so on a raised structural_edit(), where the
                # display order was already rolled back to its pre-edit value
                # (structural_edit()'s except branch), patching the tree here
                # would show the move as if it succeeded while the model (and
                # the next save) disagree. Falls back to the truthful full
                # rebuild instead, same as every other failed structural edit.
                self.trigger_panel.move_row(*refresh_arg)
            elif refresh == "order":
                self._show_triggers()
            else:
                self.trigger_panel.refresh_entries()
        # Every tier, field edits included: a typed area_x2 must move the outline.
        self._rederive_trigger_overlay()
        # GH #144: counts and Find results follow every trigger edit, panel-side ones included.
        self._find_triggers_bumped()
        self._invalidate_trigger_ranges()
        # A player field edit changes page 0's trigger counts; a memoized re-read, 0-3ms.
        self._update_player_stats()
        self._update_title()
        self._update_edit_actions()
        if error is None:
            self._log_status(label)
            return
        # A failed structural edit leaves the document correct and saveable but
        # without its minimal-diff guarantee, so it must not read as a clean
        # no-op.
        self._log_status(f"{label} failed: {type(error).__name__}: {error}", level=debug_log.ERROR)
        self._warn(
            "Trigger edit failed",
            f"{error}\n\nThe document is still correct and can be saved, but this file's "
            f"triggers will now be rewritten in full rather than spliced.",
        )

    def set_trigger_field(self, index: int, spec: trigger_fields.FieldSpec, value) -> None:
        """Write one of a trigger's own fields. The single funnel every trigger
        field widget reports through, so the contract lives in one place."""
        model = self._ensure_trigger_edits()
        if model is None:
            return
        with self._trigger_edit(model, f"Set trigger {spec.label}", content_touched=[index]) as m:
            setattr(m.manager().triggers[index], spec.attribute, value)

    def _tag_carriers(self, model: TriggerEditModel, tag: str) -> list[int]:
        return [
            index
            for index, trigger in enumerate(model.manager().triggers)
            if tag in trigger_organize.tag_chain(trigger.name or "")
        ]

    def _rename_carriers(self, model: TriggerEditModel, label: str, renamed: dict[int, str]) -> None:
        """Assign every precomputed name in one recorded edit. Each carrier is
        declared in content_touched, so exactly those blobs re-serialize."""
        with self._trigger_edit(model, label, content_touched=list(renamed), refresh="panel") as m:
            triggers = m.manager().triggers
            for index, name in renamed.items():
                triggers[index].name = name

    def rename_trigger_tag(self, old: str, new: str) -> None:
        """Rename tag `old` to `new` wherever it sits in each trigger's tag
        chain, as one undo step. Onto an existing tag this merges the two
        facets; the panel asks first. A no-op (empty, unchanged, stale tag)
        records nothing."""
        new = new.strip()
        if not new or new == old:
            return
        model = self._ensure_trigger_edits()
        if model is None:
            return
        carriers = self._tag_carriers(model, old)
        if not carriers:
            return
        triggers = model.manager().triggers
        renamed = {index: trigger_organize.retag_in_chain(triggers[index].name, old, new) for index in carriers}
        # Validated before the edit opens: commit_trigger_edit() pushes
        # unconditionally, so a refusal inside it would still record a step.
        for index, name in renamed.items():
            original = triggers[index].name
            wanted = [new if tag == old else tag for tag in trigger_organize.tag_chain(original)]
            if trigger_organize.tag_chain(name) != wanted:
                segments, _ = trigger_organize.chain_segments(original)
                # The segment `new` actually breaks: on a mixed chain that need not be the first `old`.
                segment = next(
                    (
                        text
                        for text, tag in segments
                        if tag == old
                        and trigger_organize.parse_tag(trigger_organize.retag_name(text, new)) != new
                    ),
                    None,
                )
                reason = "it would not read back as that tag"
                if segment is not None:
                    prefix, _, suffix = trigger_organize.split_tag(segment)
                    opener, closer = prefix.strip(), suffix.lstrip()[0]
                    if closer in new:
                        reason = f'it contains "{closer}", which ends a {opener}{closer} tag'
                QMessageBox.warning(
                    self,
                    "Cannot rename tag",
                    f'Tag "{old}" cannot be renamed to "{new}": {reason} in "{original}".',
                )
                return
        follow = self.mode == "triggers" and self.trigger_panel.current_tag() == old
        count = len(carriers)
        label = f'Rename tag "{old}" to "{new}" ({count} trigger{"s" if count != 1 else ""})'
        self._rename_carriers(model, label, renamed)
        if follow:
            # After the rebuild: the combo's restore-by-data fell back to "All
            # tags" because `old` no longer exists.
            self.trigger_panel.set_tag_filter(new)

    def remove_trigger_tag(self, tag: str) -> None:
        """Strip tag `tag` from every trigger carrying it, wherever it sits in
        the tag chain, as one undo step. A stale tag records nothing."""
        model = self._ensure_trigger_edits()
        if model is None:
            return
        carriers = self._tag_carriers(model, tag)
        if not carriers:
            return
        triggers = model.manager().triggers
        renamed = {index: trigger_organize.strip_tag_from_chain(triggers[index].name, tag) for index in carriers}
        count = len(carriers)
        label = f'Remove tag "{tag}" ({count} trigger{"s" if count != 1 else ""})'
        self._rename_carriers(model, label, renamed)

    def add_trigger_tag(self, indices: Sequence[int], tag: str) -> None:
        """Prefix `[tag] ` onto each trigger in `indices` whose leading tag
        chain lacks it, as one undo step (GH #101). On a tagged name it chains
        in front, the "[P1][D1] ..." shape real files use."""
        tag = tag.strip()
        model = self._ensure_trigger_edits() if tag else None
        if model is None:
            return
        triggers = model.manager().triggers
        targets = [
            i
            for i in dict.fromkeys(indices)
            if 0 <= i < len(triggers) and tag not in trigger_organize.tag_chain(triggers[i].name or "")
        ]
        if not targets:
            self._log_status(f'Every selected trigger already carries tag "{tag}"')
            return
        renamed = {i: f"[{tag}] {triggers[i].name}" if triggers[i].name else f"[{tag}]" for i in targets}
        # Validated before the edit opens, as rename_trigger_tag() does.
        if any(trigger_organize.parse_tag(name) != tag for name in renamed.values()):
            QMessageBox.warning(
                self, "Cannot add tag", f'Tag "{tag}" cannot be added: it contains "]", which ends a [tag].'
            )
            return
        count = len(targets)
        self._rename_carriers(model, f'Add tag "{tag}" ({count} trigger{"s" if count != 1 else ""})', renamed)

    def remove_trigger_tag_from(self, indices: Sequence[int], tag: str) -> None:
        """Strip `tag` from the triggers in `indices` that carry it, as one
        undo step (GH #101). remove_trigger_tag() is the every-trigger form."""
        model = self._ensure_trigger_edits()
        if model is None:
            return
        triggers = model.manager().triggers
        carriers = [
            i
            for i in dict.fromkeys(indices)
            if 0 <= i < len(triggers) and tag in trigger_organize.tag_chain(triggers[i].name or "")
        ]
        if not carriers:
            return
        renamed = {i: trigger_organize.strip_tag_from_chain(triggers[i].name, tag) for i in carriers}
        count = len(carriers)
        label = f'Remove tag "{tag}" from {count} selected trigger{"s" if count != 1 else ""}'
        self._rename_carriers(model, label, renamed)

    def set_entry_field(
        self, index: int, kind: str, entry_index: int, spec: trigger_fields.FieldSpec, value
    ) -> None:
        """Write one field of one condition or effect: set_entry_fields()'s
        N = 1 call."""
        self.set_entry_fields(index, [(kind, entry_index)], spec, value)

    def set_entry_fields(
        self, index: int, refs: Sequence[tuple[str, int]], spec: trigger_fields.FieldSpec, value
    ) -> None:
        """Write one field across conditions and effects of one trigger, as
        one undo record (GH #60). The second and last funnel the panel reports
        edits through.

        Only entries not already holding `value` are written, and that set is
        resolved before the bracket opens: entering it on a no-op would record
        a phantom step, and a redundant setattr still runs Effect.quantity's
        armour/attack split.
        """
        model = self._ensure_trigger_edits()
        if model is None:
            return
        targets = self._entries_needing(model, index, refs, {spec.attribute: value})
        if not targets:
            return
        if len(targets) == 1:
            label = f"Set {targets[0][0]} {spec.label}"
        else:
            label = f"Set {spec.label} on {len(targets)} entries"
        reports = self._write_entry_fields(model, index, targets, {spec.attribute: value}, label)
        if reports:
            # One status line: N effects switched the same way say the same thing.
            self._log_status(reports[0] if len(set(reports)) == 1 else f"{label} - adjusted {len(reports)} effects' quantity fields")

    def set_entry_field_group(
        self, index: int, ref: tuple[str, int], values: dict, label: str, defer_refresh: bool = False
    ) -> None:
        """Write a coordinate group (GH #138's Set and Reset) on one entry as
        one undo record, through set_entry_fields()' own funnel body. A no-op
        records nothing. The form is repopulated afterwards, since a field
        edit only patches labels; `defer_refresh` when a form button fired it."""
        model = self._ensure_trigger_edits()
        if model is None:
            return
        targets = self._entries_needing(model, index, [ref], values)
        if not targets:
            return
        reports = self._write_entry_fields(model, index, targets, values, label)
        if reports:
            self._log_status(reports[0])
        if self.mode != "triggers":
            return
        # GH #166: Set Location/Area and Reset are how a missing field gets fixed, so the rows follow now.
        self.trigger_panel.refresh_trigger(index)
        refresh = functools.partial(self.trigger_panel.repopulate_form_if_current, (tuple(ref),))
        if defer_refresh:
            # The clear unparents the clicked button, inside its own signal.
            QTimer.singleShot(0, refresh)
        else:
            refresh()

    @staticmethod
    def _entries_needing(model: TriggerEditModel, index: int, refs, values: dict) -> list[tuple[str, int]]:
        """The refs whose entry does not already hold every value, resolved
        before any bracket opens: entering one on a no-op would record a phantom
        step, and a redundant setattr still runs Effect.quantity's armour/attack split."""
        manager = model.manager()
        if not 0 <= index < len(manager.triggers):
            return []
        trigger = manager.triggers[index]
        targets = []
        for kind, entry_index in refs:
            siblings = _entries_of(trigger, kind)
            if not 0 <= entry_index < len(siblings):
                continue
            try:
                differs = any(getattr(siblings[entry_index], a) != v for a, v in values.items())
            except Exception:  # noqa: BLE001 -- a version-gated read; write as before
                differs = True
            if differs:
                targets.append((kind, entry_index))
        return targets

    def _write_entry_fields(self, model: TriggerEditModel, index: int, targets, values: dict, label: str) -> list[str]:
        """Assign `values` on each target entry inside ONE recorded bracket,
        with an effect's quantity-cluster fixup in the same record. Returns
        the fixup status lines."""
        # Pre-set, because _trigger_edit() reports a raised body rather than
        # re-raising it: on failure the block below never assigns.
        reports = []
        with self._trigger_edit(model, label, content_touched=[index]) as m:
            trigger = m.manager().triggers[index]
            for kind, entry_index in targets:
                entry = _entries_of(trigger, kind)[entry_index]
                before = trigger_fields.live_quantity_slot(entry) if kind == "effect" else ""
                for attribute, value in values.items():
                    setattr(entry, attribute, value)
                if kind != "effect":
                    continue
                # Same undo record: an object_attributes switch can re-point
                # serialization at a cluster slot that was never populated.
                definitions, type_attribute = self._trigger_vocabulary(kind)
                definition = (definitions or {}).get(getattr(entry, type_attribute, None))
                writes = trigger_fields.cluster_fixup(
                    entry,
                    before,
                    trigger_fields.live_quantity_slot(entry),
                    definition.attributes if definition is not None else (),
                )
                for name, new in writes:
                    setattr(entry, name, new)
                if writes:
                    reports.append(self._cluster_fixup_report(label, writes))
        return reports

    @staticmethod
    def _cluster_fixup_report(label: str, writes) -> str:
        """The status line for an edit that rewrote other cluster fields. Said
        out loud: a silent field rewrite is worse than a noisy one."""
        if not writes:
            return ""
        parts = [
            f"cleared {name.replace('_', ' ')}"
            if value is None
            else f"set {name.replace('_', ' ')} to {value:g}"
            for name, value in writes
        ]
        return f"{label} - {', '.join(parts)}"

    def copy_triggers(self) -> None:
        """Put the selected triggers on the trigger clipboard (GH #27). Reads
        only: no model is built and nothing is recorded."""
        if self.scenario is None:
            return
        manager = parse_triggers(self.scenario)
        # GH #134: a collapsed section header copies its whole section.
        indices = self.trigger_panel.selected_block_indices()
        if manager is None or not indices:
            return
        self._trigger_clipboard = self._with_file_game(
            trigger_clipboard.copy_block(manager, indices, doc_id=self._doc_id, scenario_version=self.scenario.scenario_version)
        )
        self._log_status(f"Copied {self._trigger_clipboard.label} to the clipboard.")
        self._sync_trigger_clipboard_state()

    def cut_triggers(self) -> None:
        """GH #137: the selected triggers onto the trigger clipboard, then
        removed, as one undo record. See trigger_structural_edit()'s "cut".
        A collapsed section header cuts its whole section (GH #134)."""
        indices = self.trigger_panel.selected_block_indices()
        if indices:
            self.trigger_structural_edit("cut", indices)

    def _focused_editor(self) -> QLineEdit | QPlainTextEdit | None:
        """The focused text editor in this window that Edit > Cut/Copy/Paste
        defer to in Triggers mode (GH #137), or None. A spin box or an editable
        combo answers with its line edit."""
        widget = QApplication.focusWidget()
        if not self._owns_widget(widget):
            return None
        if isinstance(widget, (QLineEdit, QPlainTextEdit)):
            return widget
        if isinstance(widget, QAbstractSpinBox) or (isinstance(widget, QComboBox) and widget.isEditable()):
            # findChild, not lineEdit(): QAbstractSpinBox.lineEdit() is protected.
            return widget.findChild(QLineEdit)
        return None

    def _update_clipboard_actions(self) -> None:
        """Edit > Cut/Copy/Paste's texts, tooltips and enabled states. Part of
        _update_tool_enabled(), and re-run on every focus move, since in
        Triggers mode a focused text editor takes the three over, and in
        Units mode the Units panel's own focus takes Cut and Copy (GH #192,
        _units_clipboard_focus())."""
        if self.mode != "triggers":
            # Phase 2.8's region pair, gated on the committed region and
            # clipboard alone: a region survives a tool switch.
            has_map = self.scenario is not None
            # GH #192: in Units mode a unit selection takes Cut and Copy from the region.
            units_selected = has_map and self.mode == "units" and bool(self._selection)
            focus = self._units_clipboard_focus() if units_selected else "units"
            if self.mode == "units":
                self.cut_action.setText("Cu&t Units")
                self.cut_action.setToolTip("Cut the selected units and what they garrison, as one undo step")
                self.cut_action.setEnabled(units_selected and focus == "units")
            else:
                self.cut_action.setText("Cu&t Triggers")
                self.cut_action.setToolTip("Cut the selected triggers (Triggers mode only)")
                self.cut_action.setEnabled(False)
            if units_selected and focus == "label":
                self.copy_action.setText("&Copy")
                self.copy_action.setToolTip("Copy the selected text")
                self.copy_action.setEnabled(True)
            elif units_selected:
                self.copy_action.setText("&Copy Units")
                self.copy_action.setToolTip("Copy the selected units and what they garrison")
                # The Garrison list has its own Copy button and clipboard.
                self.copy_action.setEnabled(focus != "garrison")
            else:
                self.copy_action.setText("&Copy Region")
                self.copy_action.setToolTip("Copy the selected region's terrain, elevation and units")
                self.copy_action.setEnabled(has_map and self._region is not None)
            block = self._region_clipboard
            if block is not None and not block.has_tiles:
                self.paste_action.setText("&Paste Units")
                self.paste_action.setToolTip("Paste the copied units centred on the hovered tile, as one undo step")
            else:
                self.paste_action.setText("&Paste Region")
                self.paste_action.setToolTip("Paste the copied region at the hovered tile, as one undo step")
            self.paste_action.setEnabled(has_map and block is not None)
            return
        editor = self._focused_editor()
        if editor is not None:
            writable = not editor.isReadOnly()
            for action, text, tip, enabled in (
                (self.cut_action, "Cu&t", "Cut the selected text", writable),
                (self.copy_action, "&Copy", "Copy the selected text", True),
                (self.paste_action, "&Paste", "Paste into the focused text field", writable),
            ):
                action.setText(text)
                action.setToolTip(tip)
                action.setEnabled(enabled)
            return
        if self._entry_tree_focused():
            self._update_entry_clipboard_actions()
            return
        # The right-click menus' gate: an open type picker holds the current trigger.
        editable = self.trigger_panel._can_edit_now()
        selected = bool(self.trigger_panel.selected_trigger_indices())
        self.cut_action.setText("Cu&t Triggers")
        self.copy_action.setText("&Copy Triggers")
        self.paste_action.setText("&Paste Triggers")
        self.cut_action.setEnabled(editable and selected)
        self.copy_action.setEnabled(editable and selected)
        # GH #3: enabled on any held block; a refused press logs its reason.
        self.paste_action.setEnabled(editable and self._trigger_clipboard is not None)
        reason = self._trigger_paste_refusal() if self._trigger_clipboard is not None else ""
        self.paste_action.setToolTip(reason or "Paste the copied triggers below the selected one, or below a collapsed section's end")
        self.cut_action.setToolTip("Cut the selected triggers (a collapsed section cuts whole), to paste elsewhere with their links kept")
        self.copy_action.setToolTip("Copy the selected triggers (a collapsed section copies whole), to paste here or into another scenario")

    def _entry_tree_focused(self) -> bool:
        """GH #137: the condition/effect tree focused means entry verbs for
        Edit > Cut/Copy/Paste; anything else in Triggers mode, trigger verbs."""
        return self.mode == "triggers" and self.trigger_panel.entry_tree.hasFocus()

    def _update_entry_clipboard_actions(self) -> None:
        panel = self.trigger_panel
        editable = panel._can_edit_now()
        single = panel.current_trigger_index() is not None and len(panel.selected_trigger_indices()) <= 1
        entries = single and bool(panel.selected_entry_refs())
        self.cut_action.setText("Cu&t Conditions/Effects")
        self.copy_action.setText("&Copy Conditions/Effects")
        self.paste_action.setText("&Paste Conditions/Effects")
        self.cut_action.setEnabled(editable and entries)
        self.copy_action.setEnabled(editable and entries)
        self.paste_action.setEnabled(editable and single and self._entry_clipboard is not None)
        reason = self._paste_refusal(self._entry_clipboard) if self._entry_clipboard is not None else ""
        self.paste_action.setToolTip(reason or "Paste the copied conditions and effects into this trigger")
        self.cut_action.setToolTip("Cut the selected conditions and effects, to paste into any trigger")
        self.copy_action.setToolTip("Copy the selected conditions and effects, to paste into any trigger")

    def _units_clipboard_focus(self) -> str:
        """Who answers Cut and Copy while units are selected in Units mode
        (GH #192): "garrison" with the Inspector's Garrison list focused (it
        has its own Copy and Paste buttons and clipboard), "label" with a
        selectable label holding selected text focused (Copy copies that
        text), else "units", the map selection. A spin box never gets here:
        it keeps the keys through Qt's own shortcut override."""
        if self.units_panel.garrison_tree.hasFocus():
            return "garrison"
        widget = QApplication.focusWidget()
        if isinstance(widget, QLabel) and self._owns_widget(widget) and widget.hasSelectedText():
            return "label"
        return "units"

    def _dispatch_cut(self, _checked=False) -> None:
        if self.mode == "units":
            if self._selection and self._units_clipboard_focus() == "units":
                self.cut_units()
            return
        if self.mode != "triggers":
            return
        if (editor := self._focused_editor()) is not None:
            editor.cut()
        elif self._entry_tree_focused():
            self.cut_entries()
        else:
            self.cut_triggers()

    def _dispatch_copy(self, _checked=False) -> None:
        if self.mode == "units" and self._selection:
            # GH #192: a unit selection wins over the region; with none, Copy Region as before.
            focus = self._units_clipboard_focus()
            if focus == "label":
                QApplication.clipboard().setText(QApplication.focusWidget().selectedText())
            elif focus == "units":
                self.copy_units()
        elif self.mode != "triggers":
            self.copy_region()
        elif (editor := self._focused_editor()) is not None:
            editor.copy()
        elif self._entry_tree_focused():
            self.copy_entries_to_clipboard()
        else:
            self.copy_triggers()

    def _dispatch_paste(self, _checked=False) -> None:
        if self.mode != "triggers":
            self.paste_region()
        elif (editor := self._focused_editor()) is not None:
            editor.paste()
        elif self._entry_tree_focused():
            self.paste_entries(self.trigger_panel.current_entry_ref())
        else:
            self.paste_triggers()

    def _dispatch_select_all(self, _checked=False) -> None:
        """Ctrl+A: every trigger in Triggers mode (GH #3), every shown unit
        under Units Select (TASK-325), the whole map as a region elsewhere."""
        if self.mode == "triggers":
            self.trigger_panel.select_all()
        elif self._units_select_active():
            self.select_all_units()
        else:
            self.select_all()

    def _dispatch_deselect(self, _checked=False) -> None:
        """Ctrl+Shift+A: under Units Select the unit selection first, then the
        region (TASK-325), the same rule as Escape."""
        if self.mode == "triggers":
            self.trigger_panel.clear_trigger_selection()
        elif self._units_select_active():
            if not self.deselect_units():
                self.deselect()
        else:
            self.deselect()

    def paste_triggers(self) -> None:
        """Paste the trigger clipboard below the current trigger, or below the
        whole section when it is a collapsed header (GH #134)."""
        current = self.trigger_panel.current_paste_anchor()
        self.trigger_structural_edit("paste", [] if current is None else [current])

    def _trigger_paste_refusal(self) -> str:
        return self._paste_refusal(self._trigger_clipboard)

    def _paste_refusal(self, block) -> str:
        """Why `block`, a TriggerBlock or an EntryBlock (GH #137), cannot be
        pasted here, or "" when it can. A block from another scenario version
        can carry fields, and effect types, this one does not store
        (cross-version paste is not supported)."""
        if block is None or self.scenario is None:
            return "Nothing to paste."
        if refusal := self._cross_game_refusal(block):
            return refusal
        here = self.scenario.scenario_version
        if block.scenario_version != here:
            return f"Copied from a {block.scenario_version} scenario; this one is {here}."
        if block.source_doc_id != self._doc_id and not library_compat.vocabulary_is_available(here):
            return f"No trigger vocabulary for scenario {here}, so unit references cannot be cleared."
        return ""

    def _exec_mode(self) -> str:
        """Which axis this file runs its triggers by, a pending Map Options
        change included (resolve_exec_mode(), as the status line reads it)."""
        if self.scenario is None:
            return EXEC_MODE_UNKNOWN
        model = self.trigger_edits
        pending = model.exec_order if model is not None and model.exec_order_supported else None
        return resolve_exec_mode(exec_order_value(self.scenario), pending)[0]

    def _legacy_exec_order(self) -> bool:
        """Whether this file runs its triggers in ID order (_exec_mode())."""
        return self._exec_mode() == EXEC_MODE_LEGACY

    def _warn_legacy_reorder(self, before_order: list[int]) -> None:
        """LEGACY_REORDER_MESSAGE, once per document, after a display-order
        move that changed the order in a file that runs in ID order."""
        model = self.trigger_edits
        if model is None or list(model.manager().trigger_display_order) == before_order:
            return
        if self._legacy_reorder_warned == self._doc_id or not self._legacy_exec_order():
            return
        self._legacy_reorder_warned = self._doc_id
        self._log_status(LEGACY_REORDER_MESSAGE)

    def _sync_trigger_clipboard_state(self) -> None:
        reason = self._trigger_paste_refusal() if self._trigger_clipboard is not None else ""
        self.trigger_panel.set_clipboard_state(self._trigger_clipboard is not None, reason)
        self._update_tool_enabled()

    def _sync_entry_clipboard_state(self) -> None:
        block = self._entry_clipboard
        reason = self._paste_refusal(block) if block is not None else ""
        self.trigger_panel.set_entry_clipboard_state(block is not None, reason)
        self._update_clipboard_actions()

    def copy_entries_to_clipboard(self) -> None:
        """Put the selected conditions and effects on the entry clipboard
        (GH #137). Reads only, like copy_triggers(): no model, no record."""
        panel = self.trigger_panel
        index = panel.current_trigger_index()
        refs = panel.selected_entry_refs()
        if self.scenario is None or index is None or len(panel.selected_trigger_indices()) > 1 or not refs:
            return
        manager = parse_triggers(self.scenario)
        if manager is None or not 0 <= index < len(manager.triggers):
            return
        self._entry_clipboard = self._with_file_game(
            trigger_clipboard.copy_entries(manager, index, refs, doc_id=self._doc_id, scenario_version=self.scenario.scenario_version)
        )
        self._log_status(f"Copied {self._entry_clipboard.label} to the clipboard.")
        self._sync_entry_clipboard_state()

    def cut_entries(self) -> None:
        panel = self.trigger_panel
        index = panel.current_trigger_index()
        refs = panel.selected_entry_refs()
        if index is not None and len(panel.selected_trigger_indices()) <= 1 and refs:
            self.entry_structural_edit("cut", index, refs[0][0], refs, -1)

    def paste_entries(self, anchor: tuple[str, int] | None = None) -> None:
        """The entry clipboard into the current trigger, below `anchor` when
        it is a condition or effect of that kind (see paste_entries())."""
        panel = self.trigger_panel
        index = panel.current_trigger_index()
        if index is not None and len(panel.selected_trigger_indices()) <= 1:
            self.entry_structural_edit("paste", index, "effect", [], -1, anchor=anchor)

    def _unique_trigger_name(self, manager) -> str:
        """"New trigger", or the first free "New trigger N".

        Duplicate names are legal in-game, but phase 4c's folders and labels
        will have to key on names -- trigger_id does not survive a reorder --
        so a new trigger is not handed a name the document already uses.
        """
        taken = {(t.name or "") for t in manager.triggers}
        if "New trigger" not in taken:
            return "New trigger"
        n = 2
        while f"New trigger {n}" in taken:
            n += 1
        return f"New trigger {n}"

    def trigger_structural_edit(self, op: str, indices: Sequence[int], arg=None) -> None:
        """Add, copy, delete, or reorder triggers. The third funnel, and the
        only one that goes through TriggerEditModel.structural_edit().

        `indices` are list indices, the panel's selection in display order;
        "new" ignores them. However many there are, the op is one
        _trigger_edit bracket and so one undo record: TriggerDiffRecord is a
        whole-model before/after snapshot, so N mutations coalesce.

        Copy (the Duplicate button) makes N independent duplicates in place:
        a copied trigger's (de)activate references still point at the
        originals, because copy_trigger() deepcopies. (Copy + Paste is the
        other verb, and moves a linked block; see trigger_clipboard.py.) Copy
        costs N renumbers, each source re-resolved by identity before its own
        call. GH #134: each contiguous display run of sources gets its copies
        as one block right after the run, or after the end of the section the
        run ends in when it holds a divider, and a copied divider is named by
        trigger_organize.copied_name() so it stays one
        (display_order_with_copies_inserted()).

        Paste appends the clipboard block (import_triggers(index=-1), which
        renumbers nothing) and then inserts it into display order directly
        below `indices[0]`, or at the end with no anchor. A block holding a
        divider lands below the last trigger of `indices[0]`'s section
        instead (trigger_model.paste_anchor()), so it never splits it. Caveat: under
        legacy_exec_order == 1 the id order is the execution order, so a pasted
        block always executes last on a legacy file however it is displayed.
        A block copied from another document (GH #3) also clears unit
        references and names variables, in this same record; see
        trigger_clipboard.py.

        Cut (GH #137) is Delete with a cut block taken first, so the record
        reads "Cut". The block reaches the clipboard only if the triggers
        really went: _trigger_edit reports a raised body rather than
        re-raising it. Pasting it back re-points the links other triggers
        held into it, inside the paste's own record.

        No `content_touched` on any of these. The library renumbers trigger_id
        across the whole list and remaps (de)activate-trigger references to
        match -- remove_triggers() even resets a reference to a deleted trigger
        to -1 -- but every one of those is the id-remap class, which the
        reference-signature diff detects on its own. trigger_id itself is not a
        serialized field (verified against versions/DE/*/structure.json), so a
        trigger that only moved keeps its blob. A pure display-order move is
        even further from that: it renumbers nothing at all.

        Copy is the one op that additionally has to repair display order.
        copy_trigger()'s append_after_source path ends at reorder_triggers(),
        whose triggers-setter resets trigger_display_order to identity -- so
        without display_order_with_copies_inserted() this would silently
        flatten a custom order (6 of 14 parseable corpus files carry one), and
        on a legacy-exec-order file silently rewrite what order the whole
        scenario executes in. Confirmed by direct measurement against
        atilla_1_scn_resaved before this fix existed.

        move_up/move_down never call reorder_triggers()/move_triggers() at
        all -- reordering here means permuting trigger_display_order only,
        ids are never renumbered. The panel only offers these while sorted
        by display order and unfiltered (_trigger_move_reasons(), which its
        buttons, menus and shortcuts all go through), so index always names a
        trigger currently at the display slot the verb acts on. `arg` is how
        many slots to move (None: one; GH #211's Shift+click passes
        MOVE_FAR_STEPS), capped by the room the block's leading member has
        (trigger_model.block_room()), so a split selection keeps its spacing
        and a block with no room records nothing. One row is patched in
        place; several repopulate. "move_top"/"move_bottom" (GH #211) gather
        `indices` at the top or the bottom of their own section
        (_move_to_section_edge()): one permutation, so one record.

        "renumber" (GH #2) renumbers the whole list: every trigger
        takes the ID of its display position (renumber_to_display_order()), so
        a legacy file runs in the listed order and File order view matches
        Display order. `indices` is only the selection to keep; it may be
        empty. renumber_refusal() is checked before the bracket opens, so a
        refusal records nothing, and one status line then says what changed
        for this file's execution mode.

        "move_ids" (GH #2, TASK-569) is File order view's Move Up/Down and
        Move to Top/Bottom: `indices` move to earlier or later IDs, `arg` a
        signed step count or "top"/"bottom" (_move_trigger_ids()). Every
        trigger keeps its display slot.

        Section management carries its payload in `arg`: "new_section" takes
        the title and lands a divider at the end of `indices[0]`'s section (or
        of display order), "move_to_section" takes the target section's
        header index (None for the leading section). Both are display-order
        edits like move_up/move_down, so on a legacy_exec_order == 0 file a
        move to another section also changes when the trigger runs.
        "section_up"/"section_down" (GH #133) swap `indices`' one section,
        whole, with its neighbour: one permutation, so one record.
        """
        if op == "new_section":
            self._new_section(indices, arg)
            return
        if op == "move_to_section":
            self._move_to_section(indices, arg)
            return
        if op in ("section_up", "section_down"):
            self._move_section(indices, -1 if op == "section_up" else 1)
            return
        if op in ("move_top", "move_bottom"):
            self._move_to_section_edge(indices, op == "move_top")
            return
        if op == "move_ids":
            self._move_trigger_ids(indices, arg)
            return
        model = self._ensure_trigger_edits()
        if model is None:
            return
        manager = model.manager()
        indices = list(dict.fromkeys(indices))
        if not all(0 <= i < len(manager.triggers) for i in indices):
            return
        if op not in ("new", "paste", "renumber") and not indices:
            return
        if op == "renumber":
            reason = renumber_refusal(manager)
            if reason:
                self._log_status(f"Cannot renumber trigger IDs: {reason}", level=debug_log.WARNING)
                return
        # move_up/move_down's signed display slots; 0 for every other op.
        move = 0
        if op in ("move_up", "move_down"):
            direction = -1 if op == "move_up" else 1
            steps = arg or 1
            # An index missing from the order stays uncapped: mutate raises and the edit rolls back.
            with suppress(IndexError):
                steps = min(steps, block_room(list(manager.trigger_display_order), indices, direction))
            if steps <= 0:
                # No room: nothing would change, and a commit always pushes a record.
                return
            move = direction * steps
        block = self._trigger_clipboard
        if op == "paste":
            # GH #3: Paste stays enabled on a held block, so the refusal is said here.
            # Refused before _trigger_edit opens, so no empty record is pushed.
            reason = self._trigger_paste_refusal()
            if reason:
                if block is not None:
                    self._log_status(f"Cannot paste these triggers: {reason}", level=debug_log.WARNING)
                return
        paste_results: list = []
        # Paste's anchor: the block lands directly below it in display order.
        anchor = indices[0] if op == "paste" and indices else None
        before_len = len(manager.triggers)
        sources = [manager.triggers[i] for i in indices]
        copies: list = []
        # GH #137: Cut is Delete plus a block taken first, inbound links included.
        cut_block = (
            trigger_clipboard.copy_block(
                manager, indices, doc_id=self._doc_id, scenario_version=self.scenario.scenario_version, cut=True
            )
            if op == "cut"
            else None
        )

        def mutate(m):
            if op == "new":
                return m.add_trigger(self._unique_trigger_name(m))
            if op == "copy":
                before_triggers = list(m.triggers)
                before_order = list(m.trigger_display_order)
                copy_of = {}
                for source in sources:
                    # Re-resolved by identity: each copy renumbers the list.
                    index = next(i for i, t in enumerate(m.triggers) if t is source)
                    duplicate = m.copy_trigger(index)
                    duplicate.name = trigger_organize.copied_name(source.name or "")
                    copy_of[id(source)] = duplicate
                    copies.append(duplicate)
                # Once, after every copy: each copy_trigger() flattened the order.
                m.trigger_display_order = display_order_with_copies_inserted(
                    before_triggers, before_order, m.triggers, copy_of
                )
                return None
            if op in ("move_up", "move_down"):
                m.trigger_display_order = moved_display_order_block(
                    list(m.trigger_display_order), indices, move
                )
                return None
            if op == "paste":
                # Captured before the import, which flattens display order;
                # pushed back through the setter (restore()'s Trap 7).
                before_order = list(m.trigger_display_order)
                before_names = [t.name or "" for t in m.triggers]
                result = trigger_clipboard.paste_into(m, block, doc_id=self._doc_id)
                paste_results.append(result)
                new_indices = list(range(before_len, before_len + len(result.triggers)))
                has_divider = any(trigger_organize.is_divider(t.name or "") for t in result.triggers)
                landing = paste_anchor(before_names, before_order, anchor, has_divider)
                m.trigger_display_order = display_order_with_block_inserted(before_order, new_indices, landing)
                return None
            if op == "renumber":
                return renumber_to_display_order(m)
            return m.remove_triggers(sorted(indices))

        count = len(indices)
        noun = f"trigger {indices[0]}" if count == 1 else f"{count} triggers"
        if op == "new":
            label, refresh, refresh_arg = "New trigger", "panel", None
        elif op == "renumber":
            label, refresh, refresh_arg = "Renumber trigger IDs to display order", "panel", None
        elif op == "paste":
            label, refresh, refresh_arg = f"Paste {block.label}", "panel", None
        elif op == "copy":
            label, refresh, refresh_arg = f"Copy {noun}", "panel", None
        elif op in ("move_up", "move_down"):
            label = f"Move {noun} {'up' if move < 0 else 'down'}"
            if abs(move) > 1:
                label += f" {abs(move)} places"
            # The one-row fast path stays single-row; N rows repopulate.
            refresh, refresh_arg = ("order", (indices[0], move)) if count == 1 else ("panel", None)
        elif op == "cut":
            label, refresh, refresh_arg = f"Cut {noun}", "panel", None
        else:
            label, refresh, refresh_arg = f"Delete {noun}", "panel", None

        display_before = list(manager.trigger_display_order)
        # Stays empty on a raise: _trigger_edit reports a raised body rather than re-raising it.
        succeeded: list[bool] = []
        with self._trigger_edit(model, label, refresh=refresh, refresh_arg=refresh_arg) as m:
            m.structural_edit(mutate)
            succeeded.append(True)
        if move:
            self._warn_legacy_reorder(display_before)

        live = model.manager().triggers
        if op == "renumber":
            if succeeded:
                self._log_status({
                    EXEC_MODE_LEGACY: LEGACY_RENUMBER_MESSAGE,
                    EXEC_MODE_DISPLAY: DISPLAY_RENUMBER_MESSAGE,
                }.get(self._exec_mode(), UNKNOWN_RENUMBER_MESSAGE))
            self.trigger_panel.select_triggers(self._indices_of(live, sources))
            return
        if cut_block is not None and len(live) == before_len - count:
            # Only once the removal really happened: _trigger_edit reports a raise, it does not re-raise.
            self._trigger_clipboard = cut_block
            if cut_block.inbound:
                n = len(cut_block.inbound)
                self._log_status(f"{n} trigger link{'s' if n != 1 else ''} into the cut triggers will reconnect on Paste.")
            self._sync_trigger_clipboard_state()
        if paste_results and paste_results[0].cross_document:
            # "panel" repopulates the trees only; named variables also stale the dialog.
            self.trigger_panel.refresh_variables()
            self._log_status(trigger_clipboard.paste_report(paste_results[0]))
        elif paste_results and (report := trigger_clipboard.relink_report(paste_results[0])):
            self._log_status(report)
        if paste_results and block.from_cut and self._legacy_exec_order():
            self._log_status(
                "This file runs triggers in ID order (legacy), so the pasted triggers run last wherever they are listed."
            )
        if op == "new":
            select = [len(live) - 1]
        elif op == "paste":
            # Valid because the import appends without renumbering.
            select = list(range(before_len, len(live))) or indices
        elif op == "copy":
            select = [i for i, t in enumerate(live) if any(t is c for c in copies)] or [indices[0]]
        elif op in ("delete", "cut"):
            select = [min(*indices, len(live) - 1)]
        else:
            # Ids never change on a move.
            select = indices
        if abs(move) > 1:
            # GH #211: a far move may leave the view, or land in a collapsed section.
            self.trigger_panel.show_moved_triggers(select, move)
            return
        self.trigger_panel.select_triggers(select)

    def _land_on(self, indices: Sequence[int]) -> None:
        """Select `indices` after a section edit, expanding the first one's
        section so a collapsed target does not swallow the result."""
        if indices:
            self.trigger_panel.reveal_trigger(indices[0])
        self.trigger_panel.select_triggers(indices)

    def _refuse_section_title(self, name: str, title: str) -> bool:
        """Warn and return True when `name` would not carry `title` back.
        Called before the edit opens, so a refusal records nothing."""
        reason = trigger_organize.divider_title_error(name, title)
        if reason:
            QMessageBox.warning(self, "Cannot use that section title", f'"{title}": {reason}.')
        return bool(reason)

    def _new_section(self, indices: Sequence[int], title) -> None:
        """Add a divider titled `title`, formatted like the file's own, at the
        end of `indices[0]`'s section: an empty section right after it. With
        no anchor it goes at the end of display order."""
        title = (title or "").strip()
        model = self._ensure_trigger_edits()
        if model is None or not title:
            return
        manager = model.manager()
        names = [t.name or "" for t in manager.triggers]
        order = list(manager.trigger_display_order)
        name = trigger_organize.format_divider(title, trigger_organize.divider_format(names))
        if self._refuse_section_title(name, title):
            return
        anchor = next((i for i in indices if 0 <= i < len(names)), None)
        target = len(order)
        if anchor is not None:
            header = trigger_organize.section_header_of(names, order, anchor)
            target = trigger_organize.section_end_slot(names, order, header)

        def mutate(m):
            m.add_trigger(name)
            # The getter appends the new index (Trap 7); set it back through
            # the setter at the target slot.
            new_index = len(m.triggers) - 1
            live = [i for i in m.trigger_display_order if i != new_index]
            live.insert(target, new_index)
            m.trigger_display_order = live

        with self._trigger_edit(model, f'New section "{title}"', refresh="panel") as m:
            m.structural_edit(mutate)
        self._land_on([len(model.manager().triggers) - 1])

    def _move_to_section(self, indices: Sequence[int], header_index) -> None:
        """Move the triggers at `indices`, as one block in display order, to
        the end of the section headed by `header_index` (None: the leading
        section). A pure display-order permutation, like Move Up/Down."""
        model = self._ensure_trigger_edits()
        if model is None:
            return
        manager = model.manager()
        names = [t.name or "" for t in manager.triggers]
        indices = list(dict.fromkeys(indices))
        if not indices or not all(0 <= i < len(names) for i in indices):
            return
        # Moving a header is "Move a whole section", a separate item.
        if any(trigger_organize.is_divider(names[i]) for i in indices):
            return
        order = list(manager.trigger_display_order)
        try:
            target = trigger_organize.section_end_slot(names, order, header_index)
        except ValueError:
            return
        moved = display_order_moved_to_slot(order, indices, target)
        if moved == order:
            return
        section = (
            "before the first section"
            if header_index is None
            else f'"{trigger_organize.section_label(names[header_index].strip())}"'
        )
        count = len(indices)
        noun = f"trigger {indices[0]}" if count == 1 else f"{count} triggers"

        def mutate(m):
            m.trigger_display_order = moved

        where = section if header_index is None else f"to {section}"
        with self._trigger_edit(model, f"Move {noun} {where}", refresh="panel") as m:
            m.structural_edit(mutate)
        self._warn_legacy_reorder(order)
        self._land_on(sorted(indices, key=moved.index))

    def _move_to_section_edge(self, indices: Sequence[int], top: bool) -> None:
        """Move the triggers at `indices`, gathered into one block in display
        order, to the top or the bottom of their own section (GH #211): just
        below its divider, or past its last member. A file with no divider is
        one section. A pure display-order permutation, like Move Up/Down, so
        "panel" refresh and one record. A refusal (a selected header, a
        selection spanning sections, already there) is said in the status
        bar and records nothing."""
        model = self._ensure_trigger_edits()
        if model is None:
            return
        manager = model.manager()
        names = [t.name or "" for t in manager.triggers]
        indices = list(dict.fromkeys(indices))
        if not indices or not all(0 <= i < len(names) for i in indices):
            return
        order = list(manager.trigger_display_order)
        parts = trigger_organize.sections(names, order)
        where = "top" if top else "bottom"
        reason = section_edge_refusal(order, parts, indices, top)
        if reason:
            self._log_status(f"Cannot move to the {where}: {reason}", level=debug_log.WARNING)
            return
        moved = moved_to_section_edge(order, parts, indices, top)
        count = len(indices)
        noun = f"trigger {indices[0]}" if count == 1 else f"{count} triggers"
        sectioned = any(section.header_index is not None for section in parts)

        def mutate(m):
            m.trigger_display_order = moved

        label = f"Move {noun} to the {where}{' of the section' if sectioned else ''}"
        with self._trigger_edit(model, label, refresh="panel") as m:
            m.structural_edit(mutate)
        self._warn_legacy_reorder(order)
        # Ids never change on a move, so the pre-move indices still name the rows.
        self.trigger_panel.show_moved_triggers(sorted(indices, key=moved.index), -1 if top else 1)

    @staticmethod
    def _indices_of(live, triggers) -> list[int]:
        """The list indices of `triggers` in `live`, by object identity, in
        list order. By an id set: Select All then renumber on 18k triggers
        would make a nested scan ~3e8 steps."""
        wanted = {id(trigger) for trigger in triggers}
        return [i for i, t in enumerate(live) if id(t) in wanted]

    def _move_trigger_ids(self, indices: Sequence[int], arg) -> None:
        """File order view's Move Up/Down and Move to Top/Bottom (GH #2,
        TASK-569): the triggers at `indices` take earlier or later IDs, the
        ones they pass shift by one, links follow, and every trigger keeps its
        display slot (renumber_to_order()). `arg` is a signed step count, as
        Display order's moves capped by the block's room, or "top"/"bottom".

        One record per press, "panel" refresh. A refusal (id_order_refusal(),
        or no room) is said or skipped before the bracket, so records nothing.
        The status line says once per document and mode that a display-order
        or unflagged file's run order is unaffected or uncertain; a legacy
        file's move does what it shows, so it says nothing."""
        model = self._ensure_trigger_edits()
        if model is None:
            return
        manager = model.manager()
        count = len(manager.triggers)
        indices = list(dict.fromkeys(indices))
        if not indices or not all(0 <= i < count for i in indices):
            return
        n = len(indices)
        noun = f"trigger {indices[0]} ID" if n == 1 else f"{n} trigger IDs"
        if arg in ("top", "bottom"):
            id_order = id_order_to_edge(count, indices, arg == "top")
            if id_order == list(range(count)):
                # The panel tooltip's wording for the same state.
                self._log_status(f"Cannot move trigger IDs: already {'first' if arg == 'top' else 'last'}")
                return
            label = f"Move {noun} to the {'start' if arg == 'top' else 'end'}"
            delta = -1 if arg == "top" else 1
        elif isinstance(arg, int) and arg:
            direction = -1 if arg < 0 else 1
            steps = min(abs(arg), block_room(range(count), indices, direction))
            if steps <= 0:
                return
            delta = direction * steps
            id_order = moved_id_order(count, indices, delta)
            label = f"Move {noun} {'up' if delta < 0 else 'down'}"
            if steps > 1:
                label += f" {steps} places"
        else:
            return
        reason = id_order_refusal(manager, id_order)
        if reason:
            self._log_status(f"Cannot move trigger IDs: {reason}", level=debug_log.WARNING)
            return
        current = self.trigger_panel.current_trigger_index()
        lead = manager.triggers[current] if current in indices else None
        sources = [manager.triggers[i] for i in indices]
        succeeded: list[bool] = []
        with self._trigger_edit(model, label, refresh="panel") as m:
            m.structural_edit(lambda m: renumber_to_order(m, id_order))
            succeeded.append(True)
        live = model.manager().triggers
        if succeeded:
            mode = self._exec_mode()
            key = (self._doc_id, mode)
            if mode != EXEC_MODE_LEGACY and key not in self._id_move_warned:
                self._id_move_warned.add(key)
                self._log_status(DISPLAY_ID_MOVE_MESSAGE if mode == EXEC_MODE_DISPLAY else UNKNOWN_ID_MOVE_MESSAGE)
        select = self._indices_of(live, sources)
        if lead is not None:
            # The current row stays current: select_triggers() makes the first one current.
            first = self._indices_of(live, [lead])
            select = first + [i for i in select if i not in first]
        self.trigger_panel.show_moved_triggers(select, delta)

    def _move_section(self, indices: Sequence[int], delta: int) -> None:
        """Section Up/Down (GH #133): swap the one section holding every
        trigger at `indices` with its neighbour, as a block. A pure
        display-order permutation like Move Up/Down, so "panel" refresh and
        one record. A selection spanning sections, or one the move refuses
        (leading run, either end), records nothing."""
        model = self._ensure_trigger_edits()
        if model is None:
            return
        manager = model.manager()
        names = [t.name or "" for t in manager.triggers]
        indices = list(dict.fromkeys(indices))
        if not indices or not all(0 <= i < len(names) for i in indices):
            return
        order = list(manager.trigger_display_order)
        headers = {trigger_organize.section_header_of(names, order, i) for i in indices}
        if len(headers) != 1:
            return
        (header,) = headers
        try:
            moved = moved_section_display_order(order, names, indices[0], delta)
        except IndexError:
            return
        title = names[header].strip()

        def mutate(m):
            m.trigger_display_order = moved

        label = f'Move section "{title}" {"up" if delta < 0 else "down"}'
        with self._trigger_edit(model, label, refresh="panel") as m:
            m.structural_edit(mutate)
        self._warn_legacy_reorder(order)
        # Ids never change on a move, so the pre-move indices still name the rows.
        self.trigger_panel.select_triggers(indices)

    def entry_structural_edit(
        self,
        op: str,
        index: int,
        kind: str,
        entry_index: int | Sequence[tuple[str, int]],
        type_id: int,
        *,
        block: trigger_clipboard.EntryBlock | None = None,
        anchor: tuple[str, int] | None = None,
    ) -> None:
        """Add, copy, delete, retype, cut or paste conditions and effects.

        `new` and `retype` take one `entry_index`. `copy`, `delete` and `cut`
        also take a sequence of (kind, index) refs there, so a selection
        spanning both lists is one undo record (GH #60): delete goes through
        trigger_model.remove_entries(), copy appends in the order given.

        GH #137: `cut` is the entry clipboard filled, then `delete`, in one
        record; the block reaches the clipboard only if the rows really went.
        `paste` inserts `block` (the entry clipboard by default) below
        `anchor`, see trigger_clipboard.paste_entries(); `kind` and
        `entry_index` are unused. A paste runs inside structural_edit(), since
        a cross-document one can name variables, which only it notices.
        Refusals run before the pair opens, so they record nothing.

        Paste aside, deliberately not a structural_edit(): none of these changes
        the trigger list, so the alignment check has nothing to reconcile. They are content
        edits on one trigger, and `content_touched` is what marks its blob --
        without it the pre-edit bytes splice straight back over the new
        condition on save, with no error anywhere.

        `condition_order` / `effect_order` (GH #213), the order the game editor
        lists a trigger's entries in: the library's getter heals the array on
        a length change by dropping values past the end or appending. That is
        right for an append (`new` and `copy` land last in display; `retype`
        appends, pops and replaces at the same index, so nothing heals) and
        for an identity order, and for nothing else. On [0, 3, 1, 4, 2],
        removing list index 0 leaves [0, 3, 1, 2] and the survivors change
        places. So `delete` and `cut` go through remove_entries() and `paste`
        through trigger_clipboard.paste_entries(), which read the order first
        and set it again. Verified by write-and-reload, not assumed.
        """
        model = self._ensure_trigger_edits()
        if model is None:
            return
        manager = model.manager()
        if not 0 <= index < len(manager.triggers):
            return

        definitions, type_attribute = self._trigger_vocabulary(kind)
        if op == "retype" and definitions is None:
            # Unreachable from the UI, which gates the Type… button on the
            # panel's own vocabulary handle. Refuses rather than guessing.
            return
        if op == "paste":
            self._paste_entries_into(model, index, self._entry_clipboard if block is None else block, anchor)
            return

        if isinstance(entry_index, int):
            refs = [(kind, entry_index)]
        elif op in ("copy", "delete", "cut"):
            refs = [(str(k), int(i)) for k, i in entry_index]
        else:
            return
        if not refs:
            return
        label = f"{op.title()} {refs[0][0]}" if len(refs) == 1 else f"{op.title()} {len(refs)} entries"
        cut_block = (
            trigger_clipboard.copy_entries(
                manager, index, refs, doc_id=self._doc_id, scenario_version=self.scenario.scenario_version
            )
            if op == "cut"
            else None
        )
        before_count = len(manager.triggers[index].conditions) + len(manager.triggers[index].effects)

        # Pre-set, because _trigger_edit() reports a raised body rather than
        # re-raising it: on failure the block below never assigns.
        select: list[tuple[str, int]] = []
        report = ""
        with self._trigger_edit(model, label, content_touched=[index], refresh="entries") as m:
            trigger = m.manager().triggers[index]
            entries = trigger.conditions if kind == "condition" else trigger.effects
            if op == "retype":
                # Read before the retype replaces the entry: the dropped fields are the old type's.
                old_definition = definitions.get(getattr(entries[entry_index], type_attribute, None))
                dropped = trigger_fields.retype_entry(
                    trigger, kind, entry_index, type_id, definitions, type_attribute
                )
                select = [(kind, entry_index)]
                report = self._retype_report(kind, definitions[type_id], dropped, old_definition)
            elif op == "new":
                # The generic private entry point rather than the public
                # new_condition.<name>() wrappers: 71 vocabulary names across
                # the shipped versions have no wrapper method at all
                # ("enable/disable_object", "or"). See test_private_api_guard.
                if kind == "condition":
                    trigger._add_condition(type_id)
                else:
                    trigger._add_effect(type_id)
                select = [(kind, len(entries) - 1)]
            elif op == "copy":
                # deepcopy rather than a field-by-field copy, for restore()'s
                # own trap-5 reason: copying fields across would walk into
                # Effect.quantity's armour/attack bit-split.
                for ref_kind, ref_index in refs:
                    siblings = _entries_of(trigger, ref_kind)
                    siblings.append(copy.deepcopy(siblings[ref_index]))
                    select.append((ref_kind, len(siblings) - 1))
            else:
                # The first ref's list, at the lowest removed display slot, clamped.
                landing = remove_entries(trigger, refs)
                select = [] if landing is None else [landing]

        if cut_block is not None:
            # Only once the rows really went: _trigger_edit reports a raise, it does not re-raise.
            after = model.manager().triggers[index]
            if len(after.conditions) + len(after.effects) == before_count - len(refs):
                self._entry_clipboard = cut_block
                self._sync_entry_clipboard_state()
        # Outside the pair: the tree is rebuilt by _after_trigger_edit, so the
        # row to land on only exists once that has run.
        if self.mode == "triggers" and select:
            self.trigger_panel.select_entries(select)
        if report and select:
            self._log_status(report)

    def entry_move(self, index: int, refs: Sequence[tuple[str, int]], delta: int, to_edge: bool = False) -> None:
        """Move the conditions and effects at `refs` up (`delta` < 0) or down
        inside trigger `index`, as one undo record (GH #213): by `abs(delta)`
        display slots, stopping at the end of each list (GH #211's five
        places), or with `to_edge` to the top or bottom of each list.

        Permutes `condition_order` / `effect_order` only, through
        trigger_model.move_entries(), as the game editor does and as trigger
        Move Up/Down does with trigger_display_order: the lists stay as they
        are, so every (kind, index) ref keeps naming the same entry. A move
        that would change nothing opens no edit, since a commit always pushes.
        """
        model = self._ensure_trigger_edits()
        if model is None:
            return
        manager = model.manager()
        if not 0 <= index < len(manager.triggers):
            return
        refs = [(str(kind), int(entry_index)) for kind, entry_index in refs]
        if not would_move_entries(manager.triggers[index], refs, delta, to_edge):
            return
        what = refs[0][0] if len(refs) == 1 else f"{len(refs)} entries"
        way = "up" if delta < 0 else "down"
        if to_edge:
            label = f"Move {what} to the {'top' if delta < 0 else 'bottom'}"
        elif abs(delta) > 1:
            # "Up to": the distance is not capped up front, so a row near the end moves less.
            label = f"Move {what} {way} to {abs(delta)} places"
        else:
            label = f"Move {what} {way}"
        # Read before the rebuild, which leaves the entry tree scrolled to its top.
        scroll = self.trigger_panel.entry_tree.verticalScrollBar().value()
        with self._trigger_edit(model, label, content_touched=[index], refresh="entries") as m:
            move_entries(m.manager().triggers[index], refs, delta, to_edge)
        if self.mode == "triggers":
            self.trigger_panel.reselect_moved_entries(refs, delta, scroll)

    def _paste_entries_into(self, model, index: int, block, anchor) -> None:
        """entry_structural_edit()'s "paste" (GH #137)."""
        if block is None:
            return
        reason = self._paste_refusal(block)
        if reason:
            self._log_status(f"Cannot paste these conditions/effects: {reason}", level=debug_log.WARNING)
            return
        results: list = []

        def mutate(m):
            results.append(trigger_clipboard.paste_entries(m, index, block, anchor, doc_id=self._doc_id))

        with self._trigger_edit(model, f"Paste {block.label}", content_touched=[index], refresh="entries") as m:
            m.structural_edit(mutate, content_touched=[index])
        if not results:
            return
        if results[0].cross_document:
            self.trigger_panel.refresh_variables()
            self._log_status(trigger_clipboard.entry_paste_report(block, results[0]))
        elif report := trigger_clipboard.entry_relink_report(block, results[0]):
            self._log_status(report)
        if self.mode == "triggers":
            self.trigger_panel.select_entries(results[0].refs)

    # The game's Create Object effect id, the only template GH #59 accepts.
    _CREATE_OBJECT_EFFECT = 11

    def stamp_create_objects(self, x: int, y: int) -> None:
        """GH #59's Create Objects tool: one copy of the current Create Object
        effect per tile of the brush footprint at (x, y), as one undo record.

        A funnel of its own rather than an entry_structural_edit() op, since
        that signature has no room for a tile list. Every refusal runs before
        _trigger_edit() opens, so a refused click records nothing. Tiles that
        already hold the same object for the same player are skipped, so a
        second click on the same spot stacks nothing.
        """
        panel = self.trigger_panel
        if self.scenario is None or self.mode != "triggers" or not panel._editable:
            self._log_status("Create Objects: triggers are not editable in this file")
            return
        index = panel.current_trigger_index()
        if index is None:
            self._log_status("Create Objects: select a trigger first")
            return
        if len(panel.selected_trigger_indices()) > 1:
            self._log_status("Create Objects: select a single trigger, not several")
            return
        if panel.picker_showing():
            self._log_status("Create Objects: close the type picker first")
            return
        ref = panel.current_entry_ref()
        if ref is None or ref[0] != "effect":
            self._log_status("Create Objects: select a Create Object effect to stamp")
            return
        entry_index = ref[1]
        model = self._ensure_trigger_edits()
        if model is None:
            return
        triggers = model.manager().triggers
        if not 0 <= index < len(triggers):
            return
        effects = triggers[index].effects
        if not 0 <= entry_index < len(effects):
            return
        template = effects[entry_index]
        if template.effect_type != self._CREATE_OBJECT_EFFECT:
            self._log_status("Create Objects: select a Create Object effect to stamp")
            return

        footprint = self._brush_footprint(x, y)
        occupied = {
            (effect.location_x, effect.location_y)
            for effect in effects
            if effect.effect_type == self._CREATE_OBJECT_EFFECT
            and effect.object_list_unit_id == template.object_list_unit_id
            and effect.source_player == template.source_player
        }
        tiles = [tile for tile in footprint if tile not in occupied]
        skipped = len(footprint) - len(tiles)
        if not tiles:
            self._log_status(f"Create Objects: all {skipped} tiles already have this object")
            return

        # Pre-set, because _trigger_edit() reports a raised body rather than
        # re-raising it: on failure the block below never assigns.
        select = -1
        with self._trigger_edit(
            model, f"Create {len(tiles)} objects", content_touched=[index], refresh="entries"
        ) as m:
            trigger = m.manager().triggers[index]
            source = trigger.effects[entry_index]
            for tx, ty in tiles:
                # deepcopy for restore()'s trap-5 reason (entry_structural_edit's
                # "copy" op): a field-by-field copy walks into the quantity split.
                effect = copy.deepcopy(source)
                effect.location_x, effect.location_y = tx, ty
                trigger.effects.append(effect)
            select = len(trigger.effects) - 1

        if select < 0:
            return
        if self.mode == "triggers":
            panel.select_entry("effect", select)
        suffix = f" ({skipped} tiles already had one)" if skipped else ""
        self._log_status(f"Created {len(tiles)} Create Object effects{suffix}")

    def _trigger_vocabulary(self, kind: str):
        """(type id -> definition, type attribute) for the loaded scenario's
        version, or (None, attribute) when the library ships none for it.

        Read off the library's own per-version JSON rather than its
        module-level dicts, which are rewritten per load: load_vocabulary() is
        @cache'd, so asking per edit is free.
        """
        type_attribute = "condition_type" if kind == "condition" else "effect_type"
        loaded = self.scenario
        if loaded is None or not library_compat.vocabulary_is_available(
            loaded.scenario_version
        ):
            return (None, type_attribute)
        vocabulary = library_compat.load_vocabulary(loaded.scenario_version)
        entries = vocabulary.conditions if kind == "condition" else vocabulary.effects
        return (entries, type_attribute)

    @staticmethod
    def _retype_report(kind: str, new_definition, dropped: tuple[str, ...], old_definition) -> str:
        """The status line for one retype. A line rather than a modal: Delete
        has no confirmation either, and the whole retype is one undo step.
        Dropped fields read as the old type's form captioned them (GH #212)."""
        name = new_definition.name.replace("_", " ")
        if not dropped:
            return f"Changed {kind} to {name}."
        fields = ", ".join(trigger_fields.field_label(old_definition, field) for field in dropped)
        plural = "" if len(dropped) == 1 else "s"
        return f"Changed {kind} to {name}; {len(dropped)} field{plural} did not carry over ({fields})."

    # Variables are capped at 256 by the format, and add_variable() searches
    # 0..255 for the lowest free id and raises IndexError when there is none.
    _MAX_VARIABLES = 256

    def variable_structural_edit(self, op: str, variable_id: int, name: str) -> None:
        """Add, remove, or rename a trigger variable. The fifth funnel (4b.6c).

        Goes through structural_edit() like the trigger one does, for a
        different reason: no blob goes stale and no trigger is touched, but
        _variable_signature() is diffed in there and is the only thing that
        sets _variables_dirty. Mutating manager.variables outside it leaves the
        old variable block spliced back verbatim, old count included, and the
        edit vanishes on save with nothing raising.

        Removal is a plain list mutation because TriggerManager has no
        remove_variable(). Nothing renumbers: the surviving variables keep
        their ids, so the block simply writes one entry fewer and the removed
        id becomes a gap. Conditions and effects referencing that id are left
        pointing at a variable that no longer has a name, which is what the
        in-game editor does with them too.

        The one trap that follows from the gap: add_variable() takes the
        *lowest* free id, so removing a variable and adding another hands the
        new one the removed one's id, and with it every reference the document
        still holds to it.

        Rename mutates the existing Variable object's `name` in place --
        there is no library method that replaces it in the list, and
        replacing it here would just be add-then-remove with an extra id
        churn. `_variable_signature()` reads `.name` fresh on every diff, so
        structural_edit() still notices; TriggerSnapshot.variables is a deep
        copy rather than a list of references, so the pre-edit name is not
        overwritten by the very mutation it exists to undo.

        Both cases are pre-validated before the pair is opened rather than
        inside it. commit_trigger_edit() pushes unconditionally, and a raised
        structural_edit() marks every blob dirty and costs the document its
        minimal-diff guarantee, so an edit that cannot succeed must not be
        opened at all.
        """
        model = self._ensure_trigger_edits()
        if model is None:
            return
        manager = model.manager()

        if op == "add":
            name = (name or "").strip()
            if not name:
                return
            if len(manager.variables) >= self._MAX_VARIABLES:
                self._log_status(f"Add variable failed: all {self._MAX_VARIABLES} variable ids are in use", level=debug_log.WARNING)
                QMessageBox.warning(
                    self,
                    "No variable id available",
                    f"This scenario already uses all {self._MAX_VARIABLES} variable ids.",
                )
                return
            label = f"Add variable {name}"

            def mutate(m):
                return m.add_variable(name)

        elif op == "rename":
            name = (name or "").strip()
            target = next((v for v in manager.variables if v.variable_id == variable_id), None)
            if not name or target is None or target.name == name:
                return
            label = f"Rename variable {variable_id} to {name}"

            def mutate(m):
                for variable in m.variables:
                    if variable.variable_id == variable_id:
                        variable.name = name
                        return
                return

        else:
            if not any(v.variable_id == variable_id for v in manager.variables):
                return
            label = f"Remove variable {variable_id}"

            def mutate(m):
                for variable in list(m.variables):
                    if variable.variable_id == variable_id:
                        m.variables.remove(variable)
                        return
                return

        with self._trigger_edit(model, label, refresh="variables") as m:
            m.structural_edit(mutate)
