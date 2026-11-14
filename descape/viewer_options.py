"""ViewerWindow's scenario options: Map Options, Players, AI, Point of View, Disabled Objects, Swap Players, Diplomacy and Messages."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from contextlib import contextmanager

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import (
    QColor,
)
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QMessageBox,
)

from descape import (
    ai_scripts,
    content_roots,
    debug_log,
    diplomacy_fields,
    disables_fields,
    option_fields,
    perf_trace,
    player_fields,
    player_labels,
    player_swap,
)
from descape.disables_dialog import DisablesDialog
from descape.edit_history import (
    CompositeDiffRecord,
    MessagesDiffRecord,
    OptionsDiffRecord,
)
from descape.messages_fields import MESSAGE_FIELDS
from descape.messages_model import MessageEditsUnavailableError, MessagesEditModel
from descape.options_model import (
    MAP_STRING_FIELDS,
    OptionEditsUnavailableError,
    OptionsEditModel,
    allied_victory_write_supported,
    diplomacy_write_supported,
    disables_write_supported,
    map_strings_write_supported,
    options_write_supported,
    personality_custom_supported,
    personality_write_supported,
    player_count_write_supported,
    players_write_supported,
    script_name_write_supported,
)
from descape.players_panel import STORED_PERSONALITY, PersonalityOptions, personality_choice_label
from descape.scenario_io import (
    parse_triggers,
    refresh_player_render_context,
    xs_attachment,
)
from descape.swap_players_dialog import SwapPlayersDialog
from descape.trigger_model import (
    exec_order_write_supported,
)
from descape.unit_filter import GAIA_PLAYER_ID
from descape.unit_model import (
    UnitEditModel,
    UnitEditsUnavailableError,
)
from descape.viewer_common import (
    _SCRIPT_NAME_FIELD,
    _team_colors,
)

# The one Map Options row whose write path is TriggerEditModel rather than
# OptionsEditModel. Named here rather than matched inline, because the split is
# checked in four places (the edit fan-out, the editable-field gate, the
# pending-value merge, and the read-only note) and a typo in any one of them
# fails silently in the direction of dropping the edit.
_EXEC_ORDER_FIELD = "legacy_exec_order"
_EXEC_ORDER_SECTION = "Triggers"


def _all_diplomacy_cell_ids(loaded) -> frozenset[str]:
    """Every stance and allied-victory cell id the grid stores (the full 8x8
    including the diagonal, and all 8 allied-victory flags) -- what the
    editable-field and read-only-reason gates are computed against, not just
    the 7 opponent rows a panel actually shows for one selected player."""
    return frozenset(
        {
            **diplomacy_fields.stance_offsets(loaded),
            **diplomacy_fields.allied_victory_offsets(loaded),
        }
    )


def _diplomacy_field_label(cell_id: str) -> str:
    """A human label for an undo/status entry, from a raw cell id -- the
    Diplomacy-grid counterpart to an OptionFieldSpec's own `.label`, which
    these synthetic ids have no spec to carry."""
    parsed = diplomacy_fields.parse_cell_id(cell_id)
    if parsed[0] == "stance":
        _, row, col = parsed
        return f"P{row} stance toward P{col}"
    _, player = parsed
    return f"P{player} allied victory"


_MESSAGE_FIELD_LABELS = {spec.field_id: spec.label for spec in MESSAGE_FIELDS}


def _message_field_label(field_id: str) -> str:
    """A human label for an undo/status entry, from a raw Messages field id --
    the id-field counterpart ("hints_id") has no MessageFieldSpec of its own,
    so it borrows its text field's label with a suffix."""
    if field_id.endswith("_id"):
        base = field_id[: -len("_id")]
        return f"{_MESSAGE_FIELD_LABELS.get(base, base)} string ID"
    return _MESSAGE_FIELD_LABELS.get(field_id, field_id)


class _ViewerOptionsMixin:
    """ViewerWindow's options, players, diplomacy and messages pages (viewer.py's docstring has the split's rules)."""

    def _show_triggers(self) -> None:
        """Populate the Triggers panel. The one repopulate path, directly
        parallel to _show_map_options() and its stated "the one repopulate
        path" rationale: every caller that changes what the panel should show
        resolves the pending exec-order value here once, rather than
        repeating the resolution at each call site.

        set_exec_order() normalises self.trigger_edits.exec_order back to
        None when set to the file's own byte, so a flip-and-flip-back
        correctly stops showing "unsaved change" here too.
        """
        pending = (
            self.trigger_edits.exec_order
            if self.trigger_edits is not None and self.trigger_edits.exec_order_supported
            else None
        )
        # The repopulate rebuilds the form, and the Pick button with it.
        self.disarm_unit_picker()
        self.trigger_panel.show_scenario(self.scenario, pending)
        self._sync_trigger_clipboard_state()
        self._sync_entry_clipboard_state()
        # A repopulate can leave nothing current, which emits no entry change.
        self._on_trigger_focus_changed()

    def _show_map_options(self) -> None:
        """Populate the Map Options panel, parsing the Triggers section first.

        The parse is what makes the trigger execution-order row appear at all:
        option_fields.specs_for() reads a field's presence off the loaded
        sections, and "Triggers" is not among them until parse_triggers() has
        run. Same on-demand cost Triggers mode already pays, and not paid at
        load for the same reason.

        A file whose Triggers section cannot be parsed (the 1.54/trigger-3.9
        set) simply has no exec-order row, which is indistinguishable in the
        form from a scenario version that never stored the flag. Logged rather
        than left silent, since those are different situations to a user.

        Also the one repopulate path, so every caller that changes what the
        panel should show (mode entry, an undo, a redo) goes through the same
        pending-value and gate resolution rather than repeating it.
        """
        if self.scenario is None:
            self.map_options_panel.clear_document()
            return
        triggers_ok = parse_triggers(self.scenario) is not None
        self.map_options_panel.show_scenario(
            self.scenario,
            values=self._pending_option_values(),
            editable_fields=self._editable_option_fields(),
            read_only_reasons=self._map_options_read_only_reasons(),
            notes=self._map_options_notes(),
        )
        if not triggers_ok:
            self._log_status(
                "Map Options: this file's Triggers section can't be read, so the "
                "trigger execution-order setting is not listed."
            )

    def _pending_option_values(self) -> dict[str, int | str]:
        """Raw values that differ from what the file's retrievers hold.

        Neither write path mutates a retriever -- the scalars are byte patches
        and exec-order is one byte of a spliced tail -- so a repopulate that
        did not pass these would reset every row to the file's value while the
        models still held the edits. That is not only a wrong display: the
        panel's own equality guard compares against what it last showed, so
        re-setting a field to its pending value would then report an edit the
        model already has, and record a second undo step for it.
        """
        values: dict[str, int | str] = {}
        if self.option_edits is not None:
            values.update(self.option_edits.pending_values())
        if self.trigger_edits is not None and self.trigger_edits.exec_order_supported:
            values[_EXEC_ORDER_FIELD] = self.trigger_edits.exec_order
        return values

    def _editable_option_fields(self) -> frozenset[str]:
        """Which Map Options rows this file will accept edits for.

        Two gates, resolved separately, because neither says anything about the
        other's rows: options_model walks GlobalVictory/Diplomacy/Map/Options
        and explicitly skips the Triggers section, while trigger_write_supported
        covers only the Triggers section. A file with an untrustworthy Map
        anchor can still edit exec-order, and a file that failed the trigger
        alignment gate can still edit every scalar.
        """
        if self.game_data_lock_reason() is not None:
            return frozenset()  # GH #67
        specs = option_fields.specs_for(self.scenario)
        editable = set()
        if options_write_supported(self.scenario, specs):
            editable.update(
                spec.field_id
                for spec in specs
                if spec.field_id not in (_EXEC_ORDER_FIELD, _SCRIPT_NAME_FIELD, *MAP_STRING_FIELDS)
            )
            # Its own gate (the Files walk); listed only on a 1.40+ file.
            if any(spec.field_id == _SCRIPT_NAME_FIELD for spec in specs) and script_name_write_supported(
                self.scenario
            ):
                editable.add(_SCRIPT_NAME_FIELD)
            # Each its own gate too (the Map string walk); the model needs the scalar one to exist at all.
            listed = {spec.field_id for spec in specs}
            for field_id, name in MAP_STRING_FIELDS.items():
                if field_id in listed and map_strings_write_supported(self.scenario, name):
                    editable.add(field_id)
        if self._exec_order_writable():
            editable.add(_EXEC_ORDER_FIELD)
        return frozenset(editable)

    def _exec_order_writable(self) -> bool:
        # The model answered from the original section; the live one diverges after a commit (TASK-404).
        if self.trigger_edits is not None:
            return self.trigger_edits.exec_order_supported
        return exec_order_write_supported(self.scenario)

    _EXEC_ORDER_READ_ONLY = (
        "Read-only for this file: its Triggers section failed the alignment gate a "
        "save would splice through, so this flag cannot be written back."
    )
    _SCALARS_READ_ONLY = (
        "Read-only for this file: its map-option block failed verification, so "
        "patching this setting would land at an offset that cannot be trusted."
    )

    _SCRIPT_NAME_READ_ONLY = (
        "Read-only for this file: its script filename or Files section failed "
        "verification, so this setting cannot be written back."
    )

    _SCRIPT_NAME_EMBEDDED_READ_ONLY = (
        "Read-only for this file: it embeds its XS script's text, and a new "
        "name would leave that text under it."
    )

    _MAP_STRING_READ_ONLY = (
        "Read-only for this file: its Map section's leading strings failed "
        "verification, so this setting cannot be written back."
    )

    def _map_options_read_only_reasons(self) -> dict[str, str]:
        """Per-row tooltip text for every listed row that is not editable.

        A greyed row with no tooltip is the failure this exists to prevent, and
        the exec-order row is the live case: it carries no spec tooltip of its
        own, so before this a user hovering it learned nothing. Two texts, one
        per gate, for the same reason the gates are separate -- a row greyed
        because the Map anchor failed and a row greyed because the Triggers
        section failed are different situations to whoever has to act on it.

        Rows shown as-stored are deliberately absent here: their own label
        already carries a tooltip accounting for the raw number, which is the
        more specific explanation and must not be displaced.
        """
        editable = self._editable_option_fields()
        reasons = {}
        for spec in option_fields.specs_for(self.scenario):
            if spec.field_id in editable:
                continue
            if spec.section == _EXEC_ORDER_SECTION:
                reasons[spec.field_id] = self._EXEC_ORDER_READ_ONLY
            elif spec.field_id == _SCRIPT_NAME_FIELD and options_write_supported(
                self.scenario, option_fields.specs_for(self.scenario)
            ):
                # The gate above already forced the Files parse, so the embedded length is known.
                reasons[spec.field_id] = (
                    self._SCRIPT_NAME_EMBEDDED_READ_ONLY
                    if xs_attachment(self.scenario)[1]
                    else self._SCRIPT_NAME_READ_ONLY
                )
            elif spec.field_id in MAP_STRING_FIELDS and options_write_supported(
                self.scenario, option_fields.specs_for(self.scenario)
            ):
                reasons[spec.field_id] = self._MAP_STRING_READ_ONLY
            else:
                reasons[spec.field_id] = self._SCALARS_READ_ONLY
        return reasons

    def _map_options_notes(self) -> list[str]:
        """The status-line summary alongside those tooltips. Only the
        exec-order gate needs one: a greyed scalar is already covered by the
        panel's own read-only count, while a single greyed row among editable
        ones would otherwise go unremarked in the summary."""
        specs = option_fields.specs_for(self.scenario)
        listed = any(spec.field_id == _EXEC_ORDER_FIELD for spec in specs)
        if listed and not self._exec_order_writable():
            return [
                (
                    "Trigger execution order is read-only for this file: its Triggers "
                    "section failed the alignment gate that a save would splice through."
                )
            ]
        return []

    def set_option_field(self, spec, value: int | str) -> None:
        """MapOptionsPanel's one callback: the user set `spec` to raw `value`.

        Fans out to the two write paths the panel deliberately does not know
        apart: TriggerEditModel for exec-order, which lives inside the Triggers
        region and is emitted by that model's serialize() tail, and
        OptionsEditModel for every byte-patched scalar. Wiring exec-order to
        the options model instead would leave self.trigger_edits None on a save
        whose only change was that flag, and write_scenario() would emit the
        original bytes with no error at all.
        """
        if spec.section == _EXEC_ORDER_SECTION:
            self._set_exec_order(spec, value)
        else:
            self._set_option_scalar(spec, value)

    def _set_exec_order(self, spec, value: int) -> None:
        """The exec-order row, through the trigger model's own edit pair so it
        gets a TriggerDiffRecord -- which is also what makes the flag and
        trigger_display_order restore together on an undo."""
        model = self._ensure_trigger_edits()
        if model is None:
            self._repopulate_map_options()
            return
        if not model.exec_order_supported:
            self._log_status(
                "Trigger execution order cannot be written back for this file."
            )
            self._repopulate_map_options()
            return
        with self._trigger_edit(model, f"Set {spec.label}") as m:
            m.set_exec_order(value)

    @contextmanager
    def _option_edit(self, model: OptionsEditModel, field_id: str, label: str):
        """Shared undo/redo wiring for one OptionsEditModel field edit:
        reads `before` from the model, lets the caller mutate it, and on a
        successful exit pushes one OptionsDiffRecord and updates title/edit
        actions/status. Extracted out of _set_option_scalar() once a second
        call site needed it (Players mode, step 3c) -- that method said
        this was coming the moment it did.

        No begin/commit pair is needed, unlike the trigger side: the before
        value is readable from the model at any time, so there is no timing
        window to protect. If the caller's mutation raises, the code after
        `yield` never runs and no record is pushed -- the same "nothing
        recorded on a refused edit" behaviour _set_option_scalar() always
        had.
        """
        before = model.current_value(field_id)
        yield model
        after = model.current_value(field_id)
        self.edit_history.push_options_record(OptionsDiffRecord(label, field_id, before, after))
        self._update_title()
        self._update_edit_actions()
        self._log_status(label)
        self._refresh_colour_mood_preview()

    def _set_option_scalar(self, spec, value: int | str) -> None:
        """Every other Map Options row: one in-place byte patch, one
        OptionsDiffRecord, via the shared _option_edit() contextmanager."""
        model = self._ensure_option_edits()
        if model is None:
            self._repopulate_map_options()
            return
        try:
            with self._option_edit(model, spec.field_id, f"Set {spec.label}"):
                model.set_value(spec.field_id, value)
        except (KeyError, ValueError) as e:
            self._log_status(f"Map option {spec.field_id} not set: {e}", level=debug_log.WARNING)
            self._repopulate_map_options()
            return
        if spec.field_id == _SCRIPT_NAME_FIELD:
            self._update_technical_info()

    def _ensure_option_edits(self) -> OptionsEditModel | None:
        """The document's OptionsEditModel, built on first use.

        Deferred to the first real edit exactly as _ensure_trigger_edits() is,
        and for the same reason: a browse-only session must save
        byte-identically. Returns None for a file the model refuses, which the
        panel's gate should already have kept unreachable -- reported rather
        than swallowed, since silently ignoring an edit the user made is the
        worse failure.
        """
        if self._refuse_locked_edit():
            return None
        if self.option_edits is not None:
            return self.option_edits
        if self.scenario is None:
            return None
        try:
            self.option_edits = OptionsEditModel(self.scenario)
        except OptionEditsUnavailableError as e:
            self._log_status(f"Map options are read-only for this file: {e}", level=debug_log.WARNING)
            QMessageBox.warning(self, "Map options are read-only", str(e))
            return None
        return self.option_edits

    def _ensure_unit_edits(self) -> UnitEditModel | None:
        """The document's UnitEditModel, built on first use.

        Deferred to the first real unit edit exactly as _ensure_option_edits()
        is, and for the same reason: a browse-only session must save
        byte-identically, and the construction gate below scans every unit in
        the file (up to 10,871 on the largest corpus file).

        That one-time scan is a multi-hundred-ms freeze on a large file, so
        it gets a wait cursor. Only the first call reaches it; the cached
        returns above are free and push no cursor. No processEvents() to go
        with it, unlike every other wait-cursor site here: this runs from
        mid-stroke (_apply_terrain_unit_plan) and from inside other busy
        wraps (paste, fill), where a real event-loop turn would be a
        reentrancy hole, and the setEnabled(False)/_busy pair those sites
        use to close it cannot nest.
        """
        if self._refuse_locked_edit():
            return None
        if self.unit_edits is not None:
            return self.unit_edits
        if self.scenario is None:
            return None
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # Restored before the warning box below, not after it: a modal
            # dialog under a wait cursor reads as a still-frozen window.
            try:
                # An op, not a phase: a phase of the open op, else its own line (Nudge builds it first).
                with perf_trace.op("unit_model_build"):
                    self.unit_edits = UnitEditModel(self.scenario)
            finally:
                QApplication.restoreOverrideCursor()
        except UnitEditsUnavailableError as e:
            self._log_status(f"Units are read-only for this file: {e}", level=debug_log.WARNING)
            self._warn("Units are read-only", str(e))
            return None
        return self.unit_edits

    def _schedule_unit_model_prebuild(self) -> None:
        """Builds the UnitEditModel at idle after a load (TASK-031.61 A1), so
        the first unit edit doesn't pay its ~22 ms scan inside the stroke end.

        Paced like the load's warms (level_warm.IdleCall: backs off while a
        mouse button is held) and, like them, a no-op for a window not shown.
        Its own driver, not a step of the load warm: that warm never starts
        with Preload zoom levels off or nothing to warm, and _cancel_warms()
        drops it on every edit. Its 0 ms timer can fire in the same dispatch
        as a warm tick (unmeasured).
        Not built inside the load, which would add the scan to real load
        latency. Left unfrozen: it adds ~24 gc-tracked objects (+0.07 ms per
        full collection, old-allies), and re-freezing before the warm's own
        collect would freeze the first paint's garbage.

        The build assigns only if no model exists yet for the same document:
        a map resize attaches its own remapped model after load_scenario()
        returns, and an edit before idle builds synchronously
        (_ensure_unit_edits). A read-only file is only logged here; the first
        real edit still shows the modal."""
        if self.unit_edits is not None or self.scenario is None or not self.isVisible():
            return
        doc_id = self._doc_id
        self._unit_model_prebuild.start(lambda: self._prebuild_unit_model(doc_id))

    def _prebuild_unit_model(self, doc_id: str) -> None:
        if self.scenario is None or self.unit_edits is not None or doc_id != self._doc_id:
            return
        try:
            with perf_trace.op("unit_model_prebuild"):
                model = UnitEditModel(self.scenario)
        except UnitEditsUnavailableError as e:
            debug_log.warning(f"Unit model prebuild skipped, units are read-only for this file: {e}")
            return
        except Exception as e:  # noqa: BLE001
            # Never into the crash dialog from idle: the first real edit's sync build raises it in context.
            debug_log.error(f"Unit model prebuild failed: {type(e).__name__}: {e}")
            return
        self.unit_edits = model

    def _repopulate_map_options(self) -> None:
        """Put the form back in step with the models, after an edit that was
        refused or one applied from somewhere other than the panel itself (an
        undo). The panel's live values are what its equality guard compares
        against, so leaving a refused edit on screen would make the next
        genuine change to that field look like a no-op."""
        if self.mode == "map_options":
            self._show_map_options()

    def _show_players(self) -> None:
        """Populate the Players panel -- the single repopulate path for
        Players mode (mode entry, a new document, an edit, an undo/redo, or
        a refused edit all call only this, via _repopulate_players() for
        the latter three -- see PlayersPanel.show_scenario()'s own
        docstring for why one method covers all of them without resetting
        the player selection on every edit).

        Unlike _show_map_options(), no section needs parsing on demand
        first: every player_fields.py spec lives in DataHeader/Options/
        PlayerDataTwo/Map/Units, all of which load_map_and_units() already
        parses, so there is nothing here analogous to the Triggers
        on-demand parse.
        """
        if self.scenario is None:
            self.players_panel.clear_document()
            return
        self.players_panel.show_scenario(
            self.scenario,
            editable_fields=self._editable_player_fields(),
            read_only_reasons=self._players_read_only_reasons(),
            pending_values=self._pending_player_values(),
            player_count=self._current_player_count(),
            disables_editable=self._disables_editable(),
            disables_carrier_ok=self._options_carrier_ok(),
            personality=self._personality_options(),
            swap_editable=self._swap_editable(),
            active_players=self._current_active_players(),
        )

    # -- AI personality (GH #126) ------------------------------------------

    _PERSONALITY_GATE_REASON = (
        "Read-only for this file: its AI personality block failed verification, so "
        "rewriting it would land at an offset that cannot be trusted. Every other "
        "player setting here is unaffected."
    )
    _PERSONALITY_CUSTOM_REASON = (
        "Only Standard and None can be written to this file: its AI script "
        "library (after the triggers) could not be verified."
    )

    def _ai_scan(self) -> tuple[ai_scripts.AiChoice, ...]:
        """Built-ins plus every .ai in the install, profile and mods, scanned
        once (file names only; no script is read until one is picked)."""
        if self._ai_choices is None:
            self._ai_roots = content_roots.content_roots()
            self._ai_choices = tuple(ai_scripts.scan_ai_choices(self._ai_roots))
            self._ai_unresolvable = {}
        return self._ai_choices

    def _drop_ai_scan(self) -> None:
        """Forget the scan so _ai_scan() redoes it; nothing repopulates."""
        self._ai_choices = None
        self._ai_roots = None
        self._ai_unresolvable = {}

    def invalidate_ai_choices(self) -> None:
        """Rescan on the next Players-mode show (the install path or the game's mod list changed)."""
        self._drop_ai_scan()
        self._repopulate_players()

    def _personality_custom_ok(self) -> bool:
        """personality_custom_supported() for the open file, once per
        document: it forces the trigger parse (over a second on a large file)."""
        if self.scenario is None:
            return False
        cached = self._personality_custom_cache
        if cached is None or cached[0] is not self.scenario:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                cached = (self.scenario, personality_custom_supported(self.scenario))
            finally:
                QApplication.restoreOverrideCursor()
            self._personality_custom_cache = cached
        return cached[1]

    def _personality_options(self) -> PersonalityOptions:
        if self.scenario is None or "personality" not in self._editable_player_fields():
            return PersonalityOptions(choices=self._ai_scan() if self.scenario is not None else ai_scripts.BUILTINS)
        pending_keys = {}
        if self.option_edits is not None and self.option_edits.personality_supported:
            for pid in range(1, player_fields.NUM_PLAYERS + 1):
                pending_keys[pid] = self.option_edits.current_personality(pid).key
        return PersonalityOptions(
            choices=self._ai_scan(),
            custom_reason="" if self._personality_custom_ok() else self._PERSONALITY_CUSTOM_REASON,
            unresolvable=dict(self._ai_unresolvable),
            pending_keys=pending_keys,
        )

    def _set_personality(self, player_id: int, key: str) -> None:
        """The Personality combo's edit. Always repopulates (the AI type row
        follows it, and a refusal must snap the combo back), but on the next
        event-loop turn: this runs inside that combo's own signal, and the
        repopulate deletes it."""
        self._apply_personality(player_id, key)
        QTimer.singleShot(0, self._repopulate_players)

    def _apply_personality(self, player_id: int, key: str) -> None:
        """Resolve the picked choice (reading its script and includes now),
        then one OptionsDiffRecord whose value is the choice key, so
        undo/redo replays the cached bytes."""
        model = self._ensure_option_edits()
        if model is None or not model.personality_supported:
            self._log_status("AI personality cannot be written back for this file.", level=debug_log.WARNING)
            return
        if key == STORED_PERSONALITY:
            choice = model.original_personality(player_id)
            label = f"(stored) {choice.stored_name}"
        else:
            choice = next((c for c in self._ai_scan() if c.key == key), None)
            if choice is None:
                return
            label = personality_choice_label(choice)
            try:
                choice = ai_scripts.resolve(choice, self._ai_roots)
            except ai_scripts.AiResolveError as e:
                self._ai_unresolvable[key] = f"Cannot be used: {e}"
                self._log_status(f"P{player_id} Personality not set: {e}", level=debug_log.WARNING)
                return
        field_id = player_fields.personality_field_id(player_id)
        original = model.original_personality(player_id)
        target = original.key if ai_scripts.matches_stored(choice, original.stored_name, original.ai_type) else choice.key
        if target == model.current_value(field_id):
            return  # e.g. a same-named copy of what is already set
        try:
            with self._option_edit(model, field_id, f"Set P{player_id} Personality: {label}"):
                model.set_personality(player_id, choice)
        except (KeyError, ValueError) as e:
            self._log_status(f"P{player_id} Personality not set: {e}", level=debug_log.WARNING)

    def _editable_player_fields(self) -> frozenset[str]:
        """Which Players mode field ids (bare, not "player:<field>:<player>")
        this file will accept edits for -- the same set for every player,
        since players_write_supported() is a document-wide gate, not a
        per-player one.

        Two gates, resolved separately per the maintainer plan's decision
        1: options_write_supported() is the *carrier* OptionsEditModel's own
        gate -- construction never reaches the player block at all if this
        fails, so player edits inherit that failure regardless of whether
        the per-player block itself would have verified. players_write_
        supported() is player_fields.verify_player_block(), independent in
        the other direction: a PlayerDataTwo/Units-only failure must not
        greyed out Map Options or the Diplomacy grid.

        Number of Players is unioned in from its own third gate rather than
        sharing this one -- see _editable_player_count_field(). It survives
        a players_write_supported() failure, and can fail on its own while
        every per-player row here stays editable.
        """
        if self.game_data_lock_reason() is not None:
            return frozenset()  # GH #67
        if self.scenario is None:
            return frozenset()
        if not options_write_supported(self.scenario, option_fields.specs_for(self.scenario)):
            return frozenset()
        per_player: frozenset[str] = frozenset()
        if players_write_supported(self.scenario):
            per_player = frozenset(
                player_fields.parse_player_field_id(key)[0]
                for key in player_fields.write_targets(self.scenario)
            )
            # GH #126: its own gate on top, never folded into write_targets().
            if personality_write_supported(self.scenario):
                per_player |= {"personality"}
        return per_player | self._editable_player_count_field()

    def _editable_player_count_field(self) -> frozenset[str]:
        """Number of Players' own gate, kept out of
        _editable_player_fields()' shared body because it is genuinely a
        different gate: it still inherits the carrier model's
        options_write_supported(), but instead of players_write_supported()
        it needs player_count_write_supported() -- the eight `active` flags
        plus FileHeader.player_count's own offset. The two disagree on a
        real corpus file (a scenario version 1.37 one, whose header walk
        does not reconcile), which is exactly why folding them together
        would grey out eleven editable rows to protect one.
        """
        if self.scenario is None:
            return frozenset()
        if not options_write_supported(self.scenario, option_fields.specs_for(self.scenario)):
            return frozenset()
        if not player_count_write_supported(self.scenario):
            return frozenset()
        return frozenset({player_fields.PLAYER_COUNT_FIELD_ID})

    _PLAYER_CARRIER_GATE_REASON = (
        "Read-only for this file: its map-option block failed verification, "
        "which the per-player write path shares a model with."
    )
    _PLAYER_GATE_REASON = (
        "Read-only for this file: its per-player block failed verification, so "
        "patching this setting would land at an offset that cannot be trusted."
    )
    _PLAYER_COUNT_GATE_REASON = (
        "Read-only for this file: this setting is stored twice (each player's "
        "active flag, and a count in the file header), and this file's header "
        "layout could not be resolved, so the two copies could not be kept in "
        "step. Every other player setting here is unaffected."
    )

    def _players_read_only_reasons(self) -> dict[str, str]:
        """Per-row tooltip text for ordinary Tier-1 rows a gate refused.

        player_type is deliberately absent here: PlayersPanel resolves its
        own reason unconditionally, since that is a fact about the field,
        not about this file. personality gets its own gate's reason when
        only that gate failed (GH #126). tribe_name is an ordinary Tier-1 row as of
        step 3d, and civilization/architecture as of the civ/architecture
        maintainer plan's Step B, and all three get the same gate reason as
        any other Tier-1 field -- Step A's brief window where
        civilization/architecture needed a field-kind-specific reason
        (unwritable on a 1.56+ file for a reason unrelated to this file's
        own per-player block verifying) closed once Step B's resizing
        splice made every version writable through the same
        players_write_supported() gate.
        """
        if self.scenario is None:
            return {}
        editable = self._editable_player_fields()
        carrier_ok = options_write_supported(self.scenario, option_fields.specs_for(self.scenario))
        reasons = {}
        for spec in player_fields.specs_for(self.scenario):
            if spec.field_id in editable or spec.field_id in player_fields._NEVER_WRITABLE:
                continue
            if not carrier_ok:
                reasons[spec.field_id] = self._PLAYER_CARRIER_GATE_REASON
            elif spec.field_id == "personality" and players_write_supported(self.scenario):
                reasons[spec.field_id] = self._PERSONALITY_GATE_REASON
            else:
                reasons[spec.field_id] = self._PLAYER_GATE_REASON
        if player_fields.PLAYER_COUNT_FIELD_ID not in editable:
            reasons[player_fields.PLAYER_COUNT_FIELD_ID] = (
                self._PLAYER_CARRIER_GATE_REASON if not carrier_ok
                else self._PLAYER_COUNT_GATE_REASON
            )
        return reasons

    def _pending_player_values(self) -> dict[str, dict[int, int | str]]:
        """{field_id: {player_id: value}} for every pending Players mode
        edit, across GAIA and all 8 players at once -- PlayersPanel needs every
        player's pending value up front so switching the combo can show a
        different player's edit without a round trip back here. Same
        reasoning as _pending_option_values(): a byte-patch write never
        mutates the retriever it patches, so a repopulate that skipped this
        would reset the form to the file's values while the model still
        held the edits.
        """
        values: dict[str, dict[int, int]] = {}
        if self.option_edits is not None:
            for key, value in self.option_edits.pending_values().items():
                if key.startswith(player_fields.PERSONALITY_FIELD_PREFIX):
                    # Shown as the chosen name, and the AI type row follows it.
                    player_id = player_fields.parse_personality_field_id(key)
                    choice = self.option_edits.current_personality(player_id)
                    values.setdefault("personality", {})[player_id] = choice.stored_name
                    values.setdefault("player_type", {})[player_id] = choice.ai_type
                    continue
                if not key.startswith("player:"):
                    continue
                field_id, player_id = player_fields.parse_player_field_id(key)
                values.setdefault(field_id, {})[player_id] = value
        return values

    def _pending_options(self) -> dict[str, int | str]:
        """The carrier model's pending edits, or {} before it exists --
        what player_fields.defined_player_ids()/_count() take as their
        `pending` argument."""
        return {} if self.option_edits is None else self.option_edits.pending_values()

    def _current_player_count(self) -> int:
        """Number of Players as edited, falling back to what the file
        stores -- see player_fields.defined_player_count()."""
        if self.scenario is None:
            return 0
        return player_fields.defined_player_count(self.scenario, self._pending_options())

    def _current_active_players(self) -> list[int]:
        """The player numbers Diplomacy mode's grid should show, as edited
        -- see player_fields.defined_player_ids(), whose pending branch
        makes this a contiguous prefix."""
        if self.scenario is None:
            return []
        return player_fields.defined_player_ids(self.scenario, self._pending_options())

    def set_player_count(self, value: int) -> None:
        """PlayersPanel's second callback: the user set Number of Players.

        Rides the same OptionsEditModel and the same OptionsDiffRecord
        every other Players mode row does (decision 1), so undo/redo needs
        nothing of its own.

        The trailing _repopulate_diplomacy() is a no-op today -- this
        spinbox only exists in Players mode, and entering Diplomacy runs
        _show_diplomacy() from scratch, which is what actually resizes the
        grid off the pending count. It is here for the same reason
        _move_history() calls both: a second entry point for this edit
        would otherwise leave a live grid stale.

        _repopulate_players() re-marks the Players selector's rows past the
        new count (TASK-063); the selection itself stays where it was.
        """
        model = self._ensure_option_edits()
        if model is None:
            self._repopulate_players()
            return
        try:
            with self._option_edit(
                model, player_fields.PLAYER_COUNT_FIELD_ID, f"Set number of players to {value}"
            ):
                model.set_value(player_fields.PLAYER_COUNT_FIELD_ID, value)
        except (KeyError, ValueError) as e:
            self._log_status(f"Number of players not set: {e}", level=debug_log.WARNING)
            self._repopulate_players()
            return
        self._repopulate_players()
        self._repopulate_diplomacy()
        self._repopulate_stats_players()
        # Lowering the count drops the markers of the players it deactivated.
        self._refresh_camera_markers()
        self._regate_swap()  # Swap needs two players (GH #185)

    def set_player_field(self, spec, player_id: int, value: int | str) -> None:
        """PlayersPanel's one callback: the user set `spec` to raw `value`
        for `player_id`. Every Players mode row rides the same
        OptionsEditModel Map Options and Diplomacy do (decision 1), so this
        needs no equivalent of set_option_field()'s exec-order/scalar fan-out
        -- there is only ever one destination.
        """
        if spec.field_id == "personality":
            self._set_personality(player_id, value)
            return
        model = self._ensure_option_edits()
        if model is None:
            self._repopulate_players()
            return
        field_id = player_fields.player_field_id(spec.field_id, player_id)
        who = "GAIA" if player_id == GAIA_PLAYER_ID else f"P{player_id}"
        try:
            with self._option_edit(model, field_id, f"Set {who} {spec.label}"):
                model.set_value(field_id, value)
        except (KeyError, ValueError) as e:
            self._log_status(f"Player field {field_id} not set: {e}", level=debug_log.WARNING)
            self._repopulate_players()
            return
        # After the _option_edit block, never inside it: set_value() runs in
        # the yield, so a recompute from in there would read pre-edit pending
        # values and then wrongly conclude nothing changed.
        self._after_player_color_change()

    # -- Point of View and the camera markers (GH #22) ---------------------

    def _pov_specs(self):
        """The (x, y) Point of View spec pair for the open file, or None
        when it has neither (a pre-1.40 file). One lookup shared by every
        reader below, since specs_for() walks every spec's retriever."""
        if self.scenario is None:
            return None
        specs = {s.field_id: s for s in player_fields.specs_for(self.scenario)}
        x_spec = specs.get(player_fields.POV_X_FIELD)
        y_spec = specs.get(player_fields.POV_Y_FIELD)
        if x_spec is None or y_spec is None:
            return None
        return x_spec, y_spec

    def _player_view(self, player_id: int, specs=None) -> tuple[int, int] | None:
        """P{n}'s Point of View as currently edited, or None when the file
        has no such field or the view is unset (-1 on either axis).

        Pending edits lead, the file's own value follows -- the same
        precedence PlayersPanel shows on screen, so an unsaved Set View
        moves the marker immediately rather than at the next save."""
        specs = specs or self._pov_specs()
        if specs is None:
            return None
        pending = self._pending_player_values()
        values = []
        for spec in specs:
            value = pending.get(spec.field_id, {}).get(player_id)
            if value is None:
                value = player_fields.current_value(self.scenario, spec, player_id)
            values.append(value)
        x, y = values
        if not isinstance(x, int) or not isinstance(y, int) or x < 0 or y < 0:
            return None
        return x, y

    def set_player_view(self, player_id: int) -> None:
        """PlayersPanel's Set View: write the tile at the centre of the map
        view as P{n}'s starting camera."""
        tile = self.map_view.viewport_centre_tile()
        if tile is None:
            self._log_status("View centre is off the map -- Set View wrote nothing")
            return
        self._write_player_view(player_id, tile[0], tile[1], f"Set P{player_id} Point of View")

    def centre_player_view(self, player_id: int) -> None:
        """PlayersPanel's Set to Centre (GH #157): P{n}'s starting camera on
        the tile holding the map's centre point, each axis on its own."""
        if self.scenario is None:
            return
        mm = self.scenario.map_manager
        self._write_player_view(
            player_id, mm.map_width // 2, mm.map_height // 2, f"Centre P{player_id} Point of View"
        )

    def reset_player_view(self, player_id: int) -> None:
        """PlayersPanel's Reset View: back to (-1, -1), which is what every
        unset slot across the corpus stores. Whether the game's own Reset
        writes the same pair is the in-game pass's question."""
        unset = player_fields.POV_UNSET
        self._write_player_view(player_id, unset, unset, f"Reset P{player_id} Point of View")

    def go_to_player_view(self, player_id: int) -> None:
        """PlayersPanel's Go to View: scroll to P{n}'s starting camera,
        clamped to the map the same way _navigate_to_finding clamps."""
        view = self._player_view(player_id)
        if view is None or self.scenario is None:
            return
        mm = self.scenario.map_manager
        x = max(0, min(mm.map_width - 1, view[0]))
        y = max(0, min(mm.map_height - 1, view[1]))
        self.map_view.center_on_tile(x, y)
        self._log_status(f"Centred on P{player_id}'s point of view ({x}, {y})")

    def _write_player_view(self, player_id: int, x: int, y: int, label: str) -> None:
        """Both coordinates as ONE undo step -- the composite shape
        set_disabled_ids() uses, and for the same reason: this is a single
        user gesture, and two set_player_field() calls would cost two
        Ctrl+Z and leave a half-moved camera in between."""
        specs = self._pov_specs()
        model = self._ensure_option_edits()
        if specs is None or model is None:
            self._repopulate_players()
            return
        records = []
        for spec, value in zip(specs, (x, y), strict=True):
            field_id = player_fields.player_field_id(spec.field_id, player_id)
            try:
                before = model.current_value(field_id)
                model.set_value(field_id, value)
            except (KeyError, ValueError) as e:
                self._log_status(f"Player field {field_id} not set: {e}", level=debug_log.WARNING)
                continue
            if model.current_value(field_id) == before:
                continue
            records.append(OptionsDiffRecord(label, field_id, before, value))
        if not records:
            self._repopulate_players()
            return
        if len(records) == 1:
            self.edit_history.push_options_record(records[0])
        else:
            self.edit_history.push_composite_record(CompositeDiffRecord(label, records))
        self._update_title()
        self._update_edit_actions()
        self._log_status(label)
        # The edit came from a button, not from a widget, so the panel's own
        # _values/_pending_values never saw it: without this the spinboxes
        # would still show the pre-edit numbers and the buttons' enabled
        # states would be stale.
        self._repopulate_players()
        self._refresh_camera_markers()

    def _on_player_selected(self, player_id: int) -> None:
        """PlayersPanel's selection notice. Only the markers' emphasis
        follows it; which player the panel is showing is read back off the
        panel rather than copied here (see its current_player_id())."""
        self._refresh_camera_markers()

    def _refresh_camera_markers(self) -> None:
        """View > Player Cameras: one marker per active player whose Point
        of View is set and lands on the map, in that player's colour.

        Emphasis applies only in Players mode -- elsewhere there is no
        selected player for it to mean anything about."""
        specs = self._pov_specs()
        if self.scenario is None or specs is None:
            self.map_view.set_camera_markers([], None)
            return
        mm = self.scenario.map_manager
        markers = []
        for player_id in range(1, self._current_player_count() + 1):
            view = self._player_view(player_id, specs)
            if view is None:
                continue
            x, y = view
            if not (0 <= x < mm.map_width and 0 <= y < mm.map_height):
                continue
            markers.append((player_id, x, y, QColor(*self.scenario.player_colors[player_id])))
        emphasised = self.players_panel.current_player_id() if self.mode == "players" else None
        self.map_view.set_camera_markers(markers, emphasised)

    # -- Disabled Objects (GH #57) ----------------------------------------

    def _options_carrier_ok(self) -> bool:
        """Whether the carrier OptionsEditModel's own map-option gate passes
        -- what tells the Disabled Objects button's tooltip which of its two
        gates refused."""
        if self.scenario is None:
            return False
        return options_write_supported(self.scenario, option_fields.specs_for(self.scenario))

    def _disables_editable(self) -> bool:
        """Whether the Disabled Objects dialog can be opened and applied.

        Two gates, the same shape _editable_diplomacy_fields() uses and for
        the same reason: options_write_supported() is the *carrier*
        OptionsEditModel's own gate, so a disables edit inherits its failure
        regardless; disables_write_supported() is independent in the other
        direction, so an Options-region failure must not grey out a single
        Map Options row.
        """
        if self.game_data_lock_reason() is not None:
            return False  # GH #67
        if self.scenario is None:
            return False
        if not self._options_carrier_ok():
            return False
        return disables_write_supported(self.scenario)

    def _current_disabled_ids(self) -> dict[tuple[str, int], tuple[int, ...]]:
        """All 24 lists as the dialog should open on them: the model's
        pending value where there is one, the file's stored value otherwise.

        Read through the model rather than the file whenever one exists, for
        the same reason _pending_option_values() exists at all -- a splice
        write never mutates the retrievers it rebuilds from, so re-reading
        the file would show the original and a second OK would re-record an
        edit the model already has.
        """
        current: dict[tuple[str, int], tuple[int, ...]] = {}
        model = self.option_edits
        for field_id in disables_fields.all_field_ids():
            key = disables_fields.parse_disables_field_id(field_id)
            if model is not None and model.disables_supported:
                current[key] = tuple(model.current_value(field_id))
            else:
                current[key] = disables_fields.current_ids(self.scenario, *key)
        return current

    def _build_disables_dialog(self) -> DisablesDialog | None:
        """Constructs and wires the dialog without showing it.

        Split from _show_disables_dialog() so the GUI tests have something to
        drive: exec_() is modal and would hang an offscreen run outright
        (tests/test_keybinds.py's own docstring records the same trap for
        SettingsDialog).
        """
        if self.scenario is None or not self._disables_editable():
            return None
        return DisablesDialog(
            self._current_disabled_ids(),
            player_colors=self.scenario.player_colors,
            on_accept=self.set_disabled_ids,
            parent=self,
            labels=self._player_labels,
        )

    def _show_disables_dialog(self) -> None:
        dialog = self._build_disables_dialog()
        if dialog is None:
            return
        dialog.exec_()

    def set_disabled_ids(self, changes: Mapping[tuple[str, int], Sequence[int]]) -> None:
        """DisablesDialog's one callback: every list the user changed, in one
        call, applied as ONE undo step.

        One composite record rather than one per list, because the dialog is
        modal: the whole session was a single user gesture (open, edit,
        OK), and unwinding it should cost one Ctrl+Z, the same reasoning
        phase 2.8's region paste established.
        """
        if not changes:
            return
        model = self._ensure_option_edits()
        if model is None or not model.disables_supported:
            self._repopulate_players()
            return
        records = []
        for (category, player_id), ids in changes.items():
            field_id = disables_fields.disables_field_id(category, player_id)
            try:
                before = model.current_value(field_id)
                model.set_value(field_id, ids)
            except (KeyError, ValueError) as e:
                self._log_status(f"Disabled {category} for P{player_id} not set: {e}", level=debug_log.WARNING)
                continue
            if model.current_value(field_id) == before:
                continue
            records.append(
                OptionsDiffRecord(
                    f"Set P{player_id} disabled {category}", field_id, before, tuple(ids)
                )
            )
        if not records:
            self._repopulate_players()
            return
        if len(records) == 1:
            self.edit_history.push_options_record(records[0])
        else:
            self.edit_history.push_composite_record(
                CompositeDiffRecord("Set disabled objects", records)
            )
        self._update_title()
        self._update_edit_actions()
        self._log_status(
            f"Set disabled objects ({len(records)} list{'' if len(records) == 1 else 's'})"
        )

    # -- Swap Players (GH #185) ------------------------------------------

    def _swap_editable(self) -> bool:
        """Map > Swap Players… and the panel's button: a document is open, not
        data-locked (GH #67), with at least two defined players. The dialog's
        preview gives every finer reason (player_swap.plan_swap())."""
        if self.scenario is None or self.game_data_lock_reason() is not None:
            return False
        return self._current_player_count() >= 2

    def _regate_swap(self) -> None:
        """Gate both entry points by _swap_editable(): also after a Number of
        Players edit and its undo/redo, which change the count it reads."""
        editable = self._swap_editable()
        self.swap_players_action.setEnabled(editable)
        self.players_panel.set_swap_enabled(editable)

    def _swap_gates(self) -> player_swap.SwapGates:
        """What this file accepts writes to: player_swap.file_gates(), closed whole by the data lock (GH #67)."""
        if self.scenario is None or self.game_data_lock_reason() is not None:
            return player_swap.SwapGates()
        gates = player_swap.file_gates(self.scenario, personality_custom=False)
        if "personality" not in gates.player_fields:
            return gates  # the custom-AI check forces the cold trigger parse, for a value unused here
        return dataclasses.replace(gates, personality_custom=self._personality_custom_ok())

    def _plan_swap(self, a: int, b: int, opts: player_swap.SwapOptions):
        """plan_swap() on the live document: the file overlaid with pending
        option edits. Ensures no model; parses the triggers when their box is ticked."""
        model = self.option_edits
        personalities = None
        if model is not None and model.personality_supported:
            personalities = {pid: model.current_personality(pid) for pid in (a, b)}
        manager = None
        if opts.triggers:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                manager = self._parse_triggers_syncing()
            finally:
                QApplication.restoreOverrideCursor()
        return player_swap.plan_swap(
            self.scenario, a, b, opts, self._pending_options(), self._swap_gates(), manager, personalities
        )

    def _usable_swap_plan(self, a: int, b: int, opts: player_swap.SwapOptions) -> player_swap.SwapPlan | None:
        """_plan_swap(), or None after logging why there is nothing to apply."""
        plan = self._plan_swap(a, b, opts)
        if isinstance(plan, player_swap.SwapRefusal):
            self._log_status(plan.reason)
            return None
        if plan.is_empty:
            self._log_status(f"Nothing differs between P{a} and P{b}")
            return None
        return plan

    def swap_preview(self, a: int, b: int, opts: player_swap.SwapOptions) -> tuple[str, bool]:
        """The dialog's callback: (its preview line, whether Swap is allowed).
        Never ensures a model: _ensure_*() pop message boxes or scan every unit."""
        if self.scenario is None:
            return "No map open.", False
        plan = self._plan_swap(a, b, opts)
        if isinstance(plan, player_swap.SwapRefusal):
            return plan.reason, False
        return plan.summary(), True

    def swap_players(self, a: int, b: int, opts: player_swap.SwapOptions) -> bool:
        """Swap P`a` and P`b` as one undo step. Ensures only the models the plan
        writes, and re-plans when one was just built so its overlay is the one applied."""
        if self.scenario is None or self._refuse_locked_edit():
            return False
        plan = self._usable_swap_plan(a, b, opts)
        if plan is None:
            return False
        before = (self.option_edits, self.unit_edits, self.trigger_edits)
        wanted = plan.writes_to
        ensures = (self._ensure_option_edits, self._ensure_unit_edits, self._ensure_trigger_edits)
        models = tuple(ensure() if want else None for want, ensure in zip(wanted, ensures, strict=True))
        if any(want and model is None for want, model in zip(wanted, models, strict=True)):
            return False  # its _ensure_*() said why
        if any(want and old is None for want, old in zip(wanted, before, strict=True)):
            # A model built just now: plan again, so the overlay applied is its own.
            plan = self._usable_swap_plan(a, b, opts)
            if plan is None:
                return False
            if any(want and model is None for want, model in zip(plan.writes_to, models, strict=True)):
                self._log_status(f"{plan.label} not applied: the document changed while it was planned")
                return False
        option_model, unit_model, trig_model = models
        failed = False
        with self._composite_edit(
            plan.label,
            unit_model,
            [a, b] if unit_model is not None else (),
            trig_model,
            plan.touched_triggers if trig_model is not None else (),
        ) as records:
            try:
                player_swap.apply_swap(
                    plan, option_model, unit_model, trig_model.manager() if trig_model is not None else None, records
                )
            except Exception:
                failed = True  # _composite_edit() records what changed and reports it
                raise
            if unit_model is not None:
                # Inside the bracket: its close drops selection keys that no longer resolve.
                self._selection = player_swap.players_swapped(self._selection, a, b)
        self._after_player_color_change()
        self._repopulate_players()
        self._repopulate_diplomacy()
        self._update_player_stats()
        self._update_title()
        self._update_edit_actions()
        if not failed:
            self._log_status(f"{plan.label}. {plan.summary()}")
        return not failed

    def _build_swap_dialog(self, initial: int | None = None) -> SwapPlayersDialog | None:
        """The dialog, wired and not shown, A preset to `initial` (P1, or the
        first defined player, when that is GAIA or undefined). Split from
        _show_swap_dialog() for the GUI tests: exec_() would hang offscreen."""
        if not self._swap_editable():
            return None
        players = self._current_active_players()
        if initial not in players:
            initial = players[0]
        return SwapPlayersDialog(
            players,
            initial,
            lambda a: player_swap.swap_partners(self.scenario, a, self._pending_options()),
            self.swap_preview,
            parent=self,
            labels=self._player_labels,
            colors=self.scenario.player_colors,
        )

    def _show_swap_dialog(self, initial: int | None = None) -> None:
        dialog = self._build_swap_dialog(initial)
        if dialog is None:
            return
        if dialog.exec_() == QDialog.Accepted:
            self.swap_players(*dialog.choice())

    def _show_swap_dialog_from_panel(self) -> None:
        self._show_swap_dialog(self.players_panel.current_player_id())

    def _after_player_color_change(self) -> None:
        """Shared tail for every edit that COULD have changed a player's
        colour -- Players mode's own form and the undo/redo of one. The
        Players-mode counterpart of _after_unit_mutation(), which this
        deliberately does not reuse: that method also rebuilds the unit pick
        index and resets the stack cycle, neither of which a recolour
        touches.

        Recompute strictly BEFORE invalidating. invalidate_units() bumps
        _source_gen and FlatChunkCache overrides it with an eager rebuild, so
        re-deriving afterwards would bake the stale tuples straight back in.

        Gating the invalidate on an actual difference is what makes this safe
        to call unconditionally: this runs for every player field (gold,
        wood, lock_personality...) and the undo/redo branch fires for Map
        Options and Diplomacy edits too, so an ungated call would evict the
        whole canvas on a gold-amount edit. The recompute itself is free (8
        ints to two 9-tuples).

        All four live player_colors/team_indices readers are unit-drawing
        paths, so terrain needs nothing here.

        The recolour work itself is _apply_player_color_change(); this
        wrapper exists only to hang the camera-marker refresh off a path
        that returns early inside that method on every edit that did NOT
        change a colour -- which includes a typed Point of View value and
        the undo of one, both of which move a marker.
        """
        self._apply_player_color_change()
        # After the recolour, never before it: the markers draw in
        # scenario.player_colors, which _apply_player_color_change() is what
        # re-derives.
        self._refresh_camera_markers()
        # Here, not in _apply_player_color_change(): its early return skips tribe name edits.
        self._refresh_player_labels()

    def _apply_player_color_change(self) -> None:
        if self.scenario is None or self.option_edits is None:
            return
        # Colours, team indices and GH #48's building art (civ, architecture, age).
        if not refresh_player_render_context(self.scenario, self._pending_player_values()):
            return
        # GH #372: a civilization or architecture edit changes the owners' portraits.
        self.units_panel.refresh_portrait_civs()
        self.trigger_panel.refresh_portraits()
        # A live selection coloured by owner follows the recolour.
        self.map_view.set_team_colors(_team_colors(self.scenario))
        self.map_view.refresh_unit_highlight()
        # Here, not in set_team_colors(): _rebuild_unit_index() pushes colours too.
        self.map_view.refresh_footprint_owner_colors()
        self.map_view.refresh_ring_layer_owner_colors()
        # GH #167: a trigger ghost's mark colour and building art; past the gate, so a gold edit resolves nothing.
        self.map_view.refresh_trigger_ghosts()
        # An in-flight warm is walking unit data this recolour just changed;
        # re-armed below, same as _after_unit_mutation()'s own third
        # invalidation.
        self._cancel_warms()
        if self._cache is not None:
            self._cache.invalidate_units()
            canvas_w, canvas_h = self._cache.canvas_dims(0)
            self._cache.invalidate_region((0, 0, canvas_w, canvas_h), terrain_changed=False)
            self.map_view.invalidate_region((0, 0, canvas_w, canvas_h))
        self._start_level_warm()

    def _refresh_player_labels(self) -> None:
        """GH #130: push every player selector's label and swatch, from the
        tribe names and colours as edited. Unconditional and not mode-gated,
        so a selector in another mode is never stale; a no-op when nothing changed."""
        if self.scenario is None:
            labels, colors = player_labels.DEFAULT_LABELS, None
        else:
            pending = self._pending_player_values().get(player_labels.TRIBE_NAME_FIELD, {})
            labels = player_labels.labels_for(player_labels.tribe_names(self.scenario, pending))
            # A copy: refresh_player_colors() may reassign the tuple this caches.
            colors = tuple(self.scenario.player_colors)
        if (labels, colors) == self._player_labels_pushed:
            return
        self._player_labels_pushed = (labels, colors)
        self._player_labels = labels
        self.units_panel.refresh_player_labels(labels, colors)
        self.players_panel.refresh_player_labels(labels, colors)
        self.diplomacy_panel.refresh_player_labels(labels, colors)
        self.trigger_panel.refresh_player_labels(labels, colors)
        if self._find_dialog is not None:
            self._find_dialog.refresh_player_labels(labels, colors)
        # Text only: Fusion draws a checkable action's icon in place of its checkmark.
        for actions in (self.convert_source_actions, self.player_actions):
            for pid, action in actions.items():
                action.setText(labels.text[pid])
                action.setToolTip(labels.full[pid])
        self._relabel_stats_players()

    def _repopulate_players(self) -> None:
        """Put the form back in step with the models, after an edit that was
        refused or one applied from somewhere other than the panel itself
        (an undo) -- same reasoning as _repopulate_map_options()."""
        if self.mode == "players":
            self._show_players()

    def _show_diplomacy(self) -> None:
        """Populate the Diplomacy panel -- the single repopulate path for
        Diplomacy mode (mode entry, a new document, an edit, an undo/redo,
        or a refused edit all call only this, via _repopulate_diplomacy()
        for the latter three -- see PlayersPanel.show_scenario()'s own
        docstring for why one method covers all of them without resetting
        the player selection on every edit; DiplomacyPanel.show_scenario()
        follows the same rule).

        Like Players mode and unlike Map Options, no section needs parsing
        on demand first: the Diplomacy section is already fully parsed by
        load_map_and_units().
        """
        if self.scenario is None:
            self.diplomacy_panel.clear_document()
            return
        specs = self._diplomacy_option_specs()
        spec_ids = {spec.field_id for spec in specs}
        editable = self._editable_diplomacy_fields() | (
            self._editable_option_fields() & spec_ids
        )
        reasons = dict(self._diplomacy_read_only_reasons())
        reasons.update(
            {k: v for k, v in self._map_options_read_only_reasons().items() if k in spec_ids}
        )
        values = dict(self._pending_diplomacy_values())
        values.update(
            {k: v for k, v in self._pending_option_values().items() if k in spec_ids}
        )
        self.diplomacy_panel.show_scenario(
            self.scenario,
            option_specs=specs,
            editable_fields=editable,
            read_only_reasons=reasons,
            pending_values=values,
            active_players=self._current_active_players(),
        )

    def _diplomacy_option_specs(self) -> tuple[option_fields.OptionFieldSpec, ...]:
        """The Teams group's specs (step 5): option_fields.specs_for()
        covers every panel's rows together, so this filters to the ones
        DiplomacyPanel renders -- the same split MapOptionsPanel makes for
        its own panel."""
        if self.scenario is None:
            return ()
        return tuple(s for s in option_fields.specs_for(self.scenario) if s.panel == "diplomacy")

    def _editable_diplomacy_fields(self) -> frozenset[str]:
        """Which Diplomacy cell ids (stance:row:col, allied_victory:player)
        this file will accept edits for.

        Same two-gate shape as _editable_player_fields(), and for the same
        reason: options_write_supported() is the *carrier* OptionsEditModel's
        own gate -- construction never adds the grid's cells at all if this
        fails, so grid edits inherit that failure regardless of whether the
        grid itself would have verified. diplomacy_write_supported() is
        independent in the other direction: a Diplomacy-section-only failure
        must not grey out Map Options or Players mode.
        """
        if self.game_data_lock_reason() is not None:
            return frozenset()  # GH #67
        if self.scenario is None:
            return frozenset()
        if not options_write_supported(self.scenario, option_fields.specs_for(self.scenario)):
            return frozenset()
        if not diplomacy_write_supported(self.scenario):
            return frozenset()
        cells = _all_diplomacy_cell_ids(self.scenario)
        if not allied_victory_write_supported(self.scenario):
            cells -= frozenset(diplomacy_fields.allied_victory_offsets(self.scenario))
        return cells

    _DIPLOMACY_CARRIER_GATE_REASON = (
        "Read-only for this file: its map-option block failed verification, "
        "which the Diplomacy grid's write path shares a model with."
    )
    _DIPLOMACY_GATE_REASON = (
        "Read-only for this file: its Diplomacy section failed verification, so "
        "patching this cell would land at an offset that cannot be trusted."
    )
    _ALLIED_VICTORY_MIRROR_REASON = (
        "Read-only for this file: the flag's second stored copy (in the Units "
        "section) could not be located and verified, and writing only one copy "
        "would leave the two disagreeing."
    )

    def _diplomacy_read_only_reasons(self) -> dict[str, str]:
        """Per-cell tooltip text for every cell that is not editable -- same
        reasoning as _players_read_only_reasons()."""
        if self.scenario is None:
            return {}
        editable = self._editable_diplomacy_fields()
        carrier_ok = options_write_supported(self.scenario, option_fields.specs_for(self.scenario))
        mirror_only = (
            carrier_ok
            and diplomacy_write_supported(self.scenario)
            and not allied_victory_write_supported(self.scenario)
        )
        reasons = {}
        for cell_id in _all_diplomacy_cell_ids(self.scenario):
            if cell_id in editable:
                continue
            if not carrier_ok:
                reasons[cell_id] = self._DIPLOMACY_CARRIER_GATE_REASON
            elif mirror_only and cell_id.startswith("allied_victory:"):
                reasons[cell_id] = self._ALLIED_VICTORY_MIRROR_REASON
            else:
                reasons[cell_id] = self._DIPLOMACY_GATE_REASON
        return reasons

    def _pending_diplomacy_values(self) -> dict[str, int]:
        """Raw values that differ from what the file's retrievers hold --
        same reasoning as _pending_option_values(): a byte-patch write never
        mutates the retriever it patches, so a repopulate that skipped this
        would reset every cell to the file's value while the model still
        held the edits."""
        values: dict[str, int] = {}
        if self.option_edits is not None:
            values = {
                key: value
                for key, value in self.option_edits.pending_values().items()
                if key.startswith(("stance:", "allied_victory:"))
            }
        return values

    def set_diplomacy_field(self, cell_id: str, value: int) -> None:
        """DiplomacyPanel's one callback: the user set the stance or
        allied-victory cell `cell_id` to raw `value`. Every Diplomacy row
        rides the same OptionsEditModel Map Options and Players do (decision
        1), so this needs no exec-order-style fan-out -- there is only ever
        one destination.
        """
        model = self._ensure_option_edits()
        if model is None:
            self._repopulate_diplomacy()
            return
        try:
            with self._option_edit(model, cell_id, f"Set {_diplomacy_field_label(cell_id)}"):
                model.set_value(cell_id, value)
        except (KeyError, ValueError) as e:
            self._log_status(f"Diplomacy field {cell_id} not set: {e}", level=debug_log.WARNING)
            self._repopulate_diplomacy()

    def _repopulate_diplomacy(self) -> None:
        """Put the form back in step with the models, after an edit that was
        refused or one applied from somewhere other than the panel itself
        (an undo) -- same reasoning as _repopulate_map_options()."""
        if self.mode == "diplomacy":
            self._show_diplomacy()

    _MESSAGES_READ_ONLY_REASON = (
        "Read-only for this file: its Messages section failed load-time "
        "verification, so a splice would land at an offset that cannot be "
        "trusted."
    )

    def _repopulate_messages(self) -> None:
        """Populate the Messages panel -- the single repopulate path for
        Messages mode (mode entry, a new document, an edit, an undo/redo, or
        a refused edit all call only this). A no-op outside the mode, same
        guard as _repopulate_diplomacy().

        Passes pending_values() the same way _pending_option_values() does:
        the splice write path never mutates the parsed retrievers, so a
        repopulate that skipped this would reset the form to the file's text
        while the model still held the edits.
        """
        if self.mode != "messages":
            return
        if self.scenario is None:
            self.messages_panel.clear_document()
            return
        editable = self.scenario.messages_write_supported and self.game_data_lock_reason() is None
        self.messages_panel.show_scenario(
            self.scenario,
            values=self._pending_message_values(),
            editable=editable,
            read_only_reasons={} if editable else {"*": self._MESSAGES_READ_ONLY_REASON},
        )

    def _pending_message_values(self) -> dict[str, str | int]:
        """Raw values that differ from what the file's retrievers hold --
        same reasoning as _pending_option_values(): the splice write path
        never mutates a retriever, so a repopulate that skipped this would
        reset every field to the file's value while the model still held the
        edits."""
        if self.message_edits is None:
            return {}
        return self.message_edits.pending_values()

    def set_message_field(self, field_id: str, value: str | int) -> None:
        """MessagesPanel's one callback: the user set `field_id` (a text
        field id, or an "<id>_id" string-table id) to raw `value`."""
        model = self._ensure_message_edits()
        if model is None:
            self._repopulate_messages()
            return
        try:
            with self._messages_edit(model, field_id, f"Set {_message_field_label(field_id)}"):
                model.set_value(field_id, value)
        except (KeyError, ValueError) as e:
            self._log_status(f"Message field {field_id} not set: {e}", level=debug_log.WARNING)
            self._repopulate_messages()

    @contextmanager
    def _messages_edit(self, model: MessagesEditModel, field_id: str, label: str):
        """Shared undo/redo wiring for one MessagesEditModel field edit, the
        Messages-mode counterpart to _option_edit()."""
        before = model.current_value(field_id)
        yield model
        after = model.current_value(field_id)
        self.edit_history.push_messages_record(MessagesDiffRecord(label, field_id, before, after))
        self._update_title()
        self._update_edit_actions()
        self._log_status(label)

    def _ensure_message_edits(self) -> MessagesEditModel | None:
        """The document's MessagesEditModel, built on first use.

        Deferred to the first real edit exactly as _ensure_option_edits() is,
        and for the same reason: a browse-only session must save
        byte-identically. Returns None for a file the model refuses, which
        the panel's gate should already have kept unreachable -- reported
        rather than swallowed, since silently ignoring an edit the user made
        is the worse failure.
        """
        if self._refuse_locked_edit():
            return None
        if self.message_edits is not None:
            return self.message_edits
        if self.scenario is None:
            return None
        try:
            self.message_edits = MessagesEditModel(self.scenario)
        except MessageEditsUnavailableError as e:
            self._log_status(f"Messages are read-only for this file: {e}", level=debug_log.WARNING)
            QMessageBox.warning(self, "Messages are read-only", str(e))
            return None
        return self.message_edits
